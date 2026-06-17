"""
精子优选流程 性能基准测试
===========================
测试项目:
  1. Detection (YOLOv8n TensorRT FP32) - 预处理 + 推理 + 后处理
  2. Tracking (JPDAF) - predict + correct
  3. Kinematics (VSL/ALH 计算)
  4. 整体 FPS / 每帧总时间

用法:
  python test_pipeline_benchmark.py                          # 使用默认测试视频
  python test_pipeline_benchmark.py --video path/to/video.mp4
  python test_pipeline_benchmark.py --camera 0               # 使用摄像头
  python test_pipeline_benchmark.py --frames 300             # 测试300帧
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque

# 复用项目中的跟踪器
from jpdaf_tracker import JPDAFilter


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT 引擎加载 & 推理 (从 detection_thread.py 提取, 去除 Qt 依赖)
# ──────────────────────────────────────────────────────────────────────────────

class TRTDetector:
    """轻量级 TensorRT YOLOv8n 检测器, 用于基准测试"""

    def __init__(self, engine_path="yolo_weights/best.engine",
                 conf_threshold=0.25, iou_threshold=0.45, max_detections=150):
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.stream = None
        self.original_shape = None

    # ── 引擎初始化 ──────────────────────────────────────────────────────────

    def initialize(self):
        """加载 TensorRT 引擎并分配缓冲区"""
        print(f"[Detector] 加载引擎: {self.engine_path}")
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"找不到引擎文件: {self.engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError("无法加载 TensorRT 引擎")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._allocate_buffers()
        self._warmup()

        # 打印引擎信息
        inp = self.inputs[0]
        print(f"[Detector] 输入: shape={inp['shape']}, dtype={inp['dtype']}")
        print(f"[Detector] 输出: shape={self.outputs[0]['shape']}, dtype={self.outputs[0]['dtype']}")

        # 检查精度
        if inp['dtype'] == np.float16:
            print("[Detector] 精度: FP16")
        else:
            print("[Detector] 精度: FP32")

    def _allocate_buffers(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))

            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            host_mem = cuda.pagelocked_empty(size, dtype)

            entry = {
                'name': name, 'shape': shape, 'dtype': dtype,
                'host': host_mem, 'device': device_mem, 'size': size
            }
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    def _warmup(self, n=3):
        """预热引擎"""
        dummy = np.random.randn(*self.inputs[0]['shape']).astype(self.inputs[0]['dtype'])
        for _ in range(n):
            self._infer(dummy)
        print(f"[Detector] 预热完成 ({n} 次推理)")

    # ── 预处理 ───────────────────────────────────────────────────────────────

    def preprocess(self, img):
        """BGR→RGB, resize, normalize, CHW, batch"""
        self.original_shape = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, _, h, w = self.inputs[0]['shape']
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        if self.inputs[0]['dtype'] != img.dtype:
            img = img.astype(self.inputs[0]['dtype'])
        return img

    # ── 推理 ─────────────────────────────────────────────────────────────────

    def _infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.flatten())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

    # ── 后处理 ───────────────────────────────────────────────────────────────

    def postprocess(self, output):
        if output.ndim == 3:
            output = output[0]
        predictions = output.T  # [8400, 84]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = class_scores > self.conf_threshold
        if not mask.any():
            return np.array([]), np.array([]), np.array([])

        boxes = boxes[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]

        boxes_xyxy = self._xywh2xyxy(boxes)
        sx = self.original_shape[1] / 640
        sy = self.original_shape[0] / 640
        boxes_xyxy[:, [0, 2]] *= sx
        boxes_xyxy[:, [1, 3]] *= sy

        indices = self._nms(boxes_xyxy, class_scores, self.iou_threshold)
        return boxes_xyxy[indices], class_scores[indices], class_ids[indices]

    @staticmethod
    def _xywh2xyxy(boxes):
        out = np.zeros_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def detect(self, img):
        """完整检测流程: 预处理 → 推理 → 后处理"""
        input_data = self.preprocess(img)
        output = self._infer(input_data)
        boxes, scores, class_ids = self.postprocess(output)

        if len(boxes) > self.max_detections:
            idx = np.argsort(scores)[::-1][:self.max_detections]
            boxes, scores, class_ids = boxes[idx], scores[idx], class_ids[idx]

        return boxes, scores, class_ids

    def cleanup(self):
        """释放 GPU 资源"""
        try:
            import pycuda.driver as cuda
            cuda.Context.pop()
        except:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 运动学参数计算 (从 jpdaf_tracker.py 提取)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_kinematics_for_tracks(tracker, pixel_to_micron=0.7):
    """
    对 tracker 中所有活跃轨迹计算运动学参数 (VSL, ALH, grade)
    返回计算耗时 (秒)
    """
    t0 = time.perf_counter()

    for track in tracker.tracks:
        if not track.isActive:
            continue
        if len(track.trajectory) < 5:
            continue

        recent_pts = list(track.trajectory)[-30:]
        recent_times = list(track.trajectory_times)[-30:]

        # ── VSL ──
        if len(recent_pts) >= 2:
            start = np.array(recent_pts[0])
            end = np.array(recent_pts[-1])
            dist_px = np.linalg.norm(end - start)
            dist_um = dist_px * pixel_to_micron
            dt = recent_times[-1] - recent_times[0]
            track.vsl = dist_um / dt if dt > 0 else 0.0

        # ── ALH ──
        if len(recent_pts) >= 3:
            pts = np.array(recent_pts)
            x, y = pts[:, 0], pts[:, 1]
            n = len(x)
            sx, sy = np.sum(x), np.sum(y)
            sxx, sxy = np.sum(x * x), np.sum(x * y)
            denom = n * sxx - sx * sx
            if abs(denom) > 1e-10:
                a = (n * sxy - sx * sy) / denom
                b = (sy - a * sx) / n
                dists = np.abs(a * pts[:, 0] - pts[:, 1] + b) / np.sqrt(a * a + 1)
                track.alh = np.mean(dists) * pixel_to_micron * 2
            else:
                xm = np.mean(x)
                track.alh = np.mean(np.abs(pts[:, 0] - xm)) * pixel_to_micron * 2

        # ── 平滑 & 分级 ──
        alpha = 0.2
        if track.vsl_smooth == 0.0:
            track.vsl_smooth = track.vsl
            track.alh_smooth = track.alh
        else:
            track.vsl_smooth = alpha * track.vsl + (1 - alpha) * track.vsl_smooth
            track.alh_smooth = alpha * track.alh + (1 - alpha) * track.alh_smooth

        track.update_grade()

    return time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# 主测试流程
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    print("=" * 70)
    print("精子优选流程 性能基准测试")
    print("=" * 70)

    # ── 1. 加载引擎 ──
    detector = TRTDetector(
        engine_path=args.engine,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=150,
    )
    detector.initialize()

    # ── 2. 打开视频源 ──
    if args.camera is not None:
        cap = cv2.VideoCapture(int(args.camera))
        source_desc = f"摄像头 {args.camera}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = args.video

    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频源: {source_desc}")
        return

    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[视频] {source_desc}")
    print(f"[视频] 分辨率: {w}x{h}, FPS: {video_fps:.1f}, 总帧数: {total_frames_video}")

    # ── 3. 初始化跟踪器 ──
    tracker = JPDAFilter(
        process_noise=20.0,
        measure_noise=2.0,
        detect_prob=0.7,
        gate_prob=0.95,
    )

    # ── 4. 统计变量 ──
    N = args.frames
    detect_times = []      # 预处理 + 推理 + 后处理
    preprocess_times = []
    infer_times = []
    postprocess_times = []
    track_times = []       # JPDAF predict + correct
    kinematic_times = []   # VSL / ALH / grade
    total_times = []       # 每帧总时间
    detections_per_frame = []

    print(f"\n开始测试, 目标帧数: {N}")
    print("-" * 70)

    frame_idx = 0
    while frame_idx < N:
        ret, frame = cap.read()
        if not ret:
            if args.camera is not None:
                continue  # 摄像头模式下继续等待
            print(f"[视频] 已读完所有帧 ({frame_idx} 帧)")
            break

        frame_idx += 1
        t_total_start = time.perf_counter()

        # ── Detection ──
        # 预处理
        t0 = time.perf_counter()
        input_data = detector.preprocess(frame)
        t_pre = time.perf_counter() - t0

        # 推理
        t0 = time.perf_counter()
        output = detector._infer(input_data)
        t_inf = time.perf_counter() - t0

        # 后处理
        t0 = time.perf_counter()
        boxes, scores, class_ids = detector.postprocess(output)
        if len(boxes) > detector.max_detections:
            idx = np.argsort(scores)[::-1][:detector.max_detections]
            boxes, scores, class_ids = boxes[idx], scores[idx], class_ids[idx]
        t_post = time.perf_counter() - t0

        # ── Tracking ──
        t0 = time.perf_counter()
        detection_points = []
        for box in boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            detection_points.append((cx, cy))

        tracker.predict()
        tracker.correct(detection_points)
        active_tracks = tracker.get_active_tracks()
        t_track = time.perf_counter() - t0

        # ── Kinematics ──
        t0 = time.perf_counter()
        # 手动调用每个活跃轨迹的 calculate_motion_parameters
        for track in active_tracks:
            if len(track.trajectory) >= 5:
                track.calculate_motion_parameters()
        t_kin = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        # 记录
        preprocess_times.append(t_pre * 1000)
        infer_times.append(t_inf * 1000)
        postprocess_times.append(t_post * 1000)
        detect_times.append((t_pre + t_inf + t_post) * 1000)
        track_times.append(t_track * 1000)
        kinematic_times.append(t_kin * 1000)
        total_times.append(t_total * 1000)
        detections_per_frame.append(len(boxes))

        if frame_idx % 50 == 0:
            avg_total = np.mean(total_times[-50:])
            print(f"  帧 {frame_idx:4d}/{N} | 总时间: {avg_total:6.2f} ms | "
                  f"检测数: {len(boxes):3d} | 活跃轨迹: {len(active_tracks):3d}")

    cap.release()

    # ── 5. 汇总报告 ──
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"实际测试帧数:   {len(total_times)}")
    print(f"视频分辨率:     {w}x{h}")
    print(f"每帧平均检测数: {np.mean(detections_per_frame):.1f}")
    print()

    def stats(name, data):
        arr = np.array(data)
        print(f"  {name:<28s}  mean={np.mean(arr):7.2f} ms  "
              f"std={np.std(arr):6.2f} ms  "
              f"min={np.min(arr):7.2f} ms  "
              f"max={np.max(arr):7.2f} ms  "
              f"median={np.median(arr):7.2f} ms")

    print("─── Detection (YOLOv8n TensorRT) ───")
    stats("预处理 (Preprocess)", preprocess_times)
    stats("推理   (Inference)", infer_times)
    stats("后处理 (Postprocess)", postprocess_times)
    stats("检测总计", detect_times)
    print()
    print("─── Tracking (JPDAF) ───")
    stats("跟踪 (Predict+Correct)", track_times)
    print()
    print("─── Kinematics (VSL/ALH/Grade) ───")
    stats("运动学参数计算", kinematic_times)
    print()
    print("─── 整体 ───")
    stats("每帧总时间", total_times)

    total_ms = np.mean(total_times)
    fps = 1000.0 / total_ms if total_ms > 0 else 0
    print(f"\n  整体 FPS:  {fps:.2f} fps  (每帧 {total_ms:.2f} ms)")
    print()

    # 时间占比
    avg_det = np.mean(detect_times)
    avg_track = np.mean(track_times)
    avg_kin = np.mean(kinematic_times)
    avg_total = np.mean(total_times)
    print("─── 时间占比 ───")
    print(f"  Detection:  {avg_det:7.2f} ms  ({avg_det/avg_total*100:5.1f}%)")
    print(f"  Tracking:   {avg_track:7.2f} ms  ({avg_track/avg_total*100:5.1f}%)")
    print(f"  Kinematics: {avg_kin:7.2f} ms  ({avg_kin/avg_total*100:5.1f}%)")
    other = avg_total - avg_det - avg_track - avg_kin
    print(f"  Other:      {other:7.2f} ms  ({other/avg_total*100:5.1f}%)")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="精子优选流程 性能基准测试")
    parser.add_argument("--engine", type=str, default="yolo_weights/best.engine",
                        help="TensorRT 引擎文件路径")
    parser.add_argument("--video", type=str, default="test-videoes/using2_latest.mp4",
                        help="测试视频路径")
    parser.add_argument("--camera", type=str, default=None,
                        help="摄像头ID (如 0), 优先于 --video")
    parser.add_argument("--frames", type=int, default=200,
                        help="测试帧数 (默认200)")
    args = parser.parse_args()

    run_benchmark(args)
