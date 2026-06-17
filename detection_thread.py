import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import datetime
from pathlib import Path
from collections import deque
import threading
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, Qt
from PyQt6.QtGui import QImage, QPixmap
import colorsys

# 导入跟踪器
from jpdaf_tracker import JPDAFilter

# 导入精子注册表
from sperm_registry import SpermRegistry

# 添加分级参数定义
class GradeParameters:
    """精子分级参数"""
    # Grade 1
    GRADE_1_VSL_LOW_LIMIT = 11.0
    GRADE_1_ALH_HIGH_LIMIT = 0.6
    GRADE_1_HEAD_LENGTH_LOW_LIMIT = 4.25
    GRADE_1_HEAD_LENGTH_UP_LIMIT = 5.5
    GRADE_1_HEAD_WIDTH_LOW_LIMIT = 2.9
    GRADE_1_HEAD_WIDTH_UP_LIMIT = 3.68
    GRADE_1_HEAD_RATIO_LOW_LIMIT = 1.5
    GRADE_1_HEAD_RATIO_UP_LIMIT = 1.8
    GRADE_1_HEAD_REGULARITY_UP_LIMIT = 0.08
    GRADE_1_NECK_BEND_ANGLE_UP_LIMIT = 20.0
    GRADE_1_NECK_HEAD_ANGLE_UP_LIMIT = 11.0
    GRADE_1_NECK_WIDTH_UP_LIMIT = 0.5
    
    # Grade 2
    GRADE_2_VSL_LOW_LIMIT = 10.0
    GRADE_2_ALH_HIGH_LIMIT = 0.8
    GRADE_2_HEAD_RATIO_LOW_LIMIT = 1.45
    GRADE_2_HEAD_RATIO_UP_LIMIT = 2.0
    GRADE_2_HEAD_LENGTH_LOW_LIMIT = 4.0
    GRADE_2_HEAD_LENGTH_UP_LIMIT = 5.6
    GRADE_2_HEAD_WIDTH_LOW_LIMIT = 2.7
    GRADE_2_HEAD_WIDTH_UP_LIMIT = 3.85
    GRADE_2_HEAD_REGULARITY_UP_LIMIT = 0.1
    GRADE_2_NECK_BEND_ANGLE_UP_LIMIT = 25.0
    GRADE_2_NECK_HEAD_ANGLE_UP_LIMIT = 15.0
    GRADE_2_NECK_WIDTH_UP_LIMIT = 0.6
    
    # Grade 3
    GRADE_3_VSL_LOW_LIMIT = 7.5
    GRADE_3_ALH_HIGH_LIMIT = 1.0
    GRADE_3_HEAD_RATIO_LOW_LIMIT = 1.3
    GRADE_3_HEAD_RATIO_UP_LIMIT = 2.1
    GRADE_3_HEAD_REGULARITY_UP_LIMIT = 0.125
    GRADE_3_NECK_BEND_ANGLE_UP_LIMIT = 40.0
    GRADE_3_NECK_HEAD_ANGLE_UP_LIMIT = 20.0
    GRADE_3_NECK_WIDTH_UP_LIMIT = 0.8
    
    # Grade 4
    GRADE_4_VSL_LOW_LIMIT = 5.0
    GRADE_4_HEAD_RATIO_LOW_LIMIT = 1.2
    GRADE_4_HEAD_RATIO_UP_LIMIT = 2.2
    GRADE_4_HEAD_REGULARITY_UP_LIMIT = 0.15
    GRADE_4_NECK_BEND_ANGLE_UP_LIMIT = 50.0
    GRADE_4_NECK_HEAD_ANGLE_UP_LIMIT = 30.0
    GRADE_4_NECK_WIDTH_UP_LIMIT = 1.0
    
    # Grade 5
    GRADE_5_VSL_LOW_LIMIT = 2.0

class DetectionResult:
    """检测结果数据类"""
    def __init__(self, frame, boxes, scores, class_ids, tracks, best_track_id=None, top_candidates=None):
        self.frame = frame
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.tracks = tracks
        self.timestamp = time.time()
        self.detection_time = 0
        self.best_track_id = best_track_id
        self.top_candidates = top_candidates or []  # VSL最快的8个精子ID列表
        self.show_tracking = False

class DetectionThread(QThread):
    """检测和跟踪线程"""
    
    # 定义信号
    frame_ready = pyqtSignal(QPixmap)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float, float)
    detection_result = pyqtSignal(object)
    stats_updated = pyqtSignal(dict)
    pipeline_ready = pyqtSignal()  # 管线就绪信号（转发自segment_thread）
    
    def __init__(self, engine_path="yolo_weights/best.engine", parent=None):
        super().__init__(parent)
        
        self.engine_path = engine_path
        self.running = False
        self.paused = True
        self.mutex = QMutex()
        
        # 添加轨迹显示控制
        self.show_tracking = False
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=100)
        self.buffer_lock = threading.Lock()
        
        # 性能统计
        self.original_fps = 30.0
        self.detection_fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        
        # 检测参数
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.class_names = ['sperm']
        self.max_detections = 150
        
        # 跟踪器
        self.tracker = None
        self.track_colors = self._generate_colors(100)
        
        # 最优精子跟踪
        self.best_track_id = None
        self.top_candidates = []  # VSL最快的8个精子
        self.vsl_high_threshold = GradeParameters.GRADE_1_VSL_LOW_LIMIT
        self.vsl_threshold = GradeParameters.GRADE_5_VSL_LOW_LIMIT
        self.edge_margin = 50
        self.pixel_to_micron = 0.7  # 像素到微米的转换系数
        
        # 图像尺寸
        self.frame_width = 0
        self.frame_height = 0
        
        # TensorRT引擎
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        self.logger = None
        
        # CUDA上下文
        self.cuda_context = None
        
        # 精子注册表 (SQLite 持久化)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = os.path.join("SpermDatabase", f"sperm_experiment_{timestamp}.db")
        os.makedirs("SpermDatabase", exist_ok=True)
        self.registry = SpermRegistry(db_path=db_path)
        print(f"[DetectionThread] 注册表已创建: {db_path}")

        # 当前输出标签
        self.output_label = None
        self.fullscreen_label = None
        
        # 分割线程引用
        self.segment_thread = None
        self._pipeline_signaled = False  # 无分割线程时用于首帧通知
        
        # 统计信息
        self.total_detections = 0
        self.frame_count = 0
        
        # 分级参数
        self.grade_params = GradeParameters()
        
        # 连接信号
        self.frame_ready.connect(self._update_display)
        self.error_occurred.connect(self._handle_error)
    
    def set_pixel_to_micron(self, value):
        """设置像素到微米的转换系数"""
        self.pixel_to_micron = value
        # 同时更新跟踪器中的转换系数
        if self.tracker:
            for track in self.tracker.tracks:
                track.pixel_to_micron = value
    
    def set_show_tracking(self, show):
        """设置是否显示跟踪轨迹"""
        self.show_tracking = show
        print(f"检测线程: 轨迹显示已{'启用' if show else '禁用'}")
    
    def update_fps(self, fps):
        """更新原始视频FPS"""
        self.original_fps = fps
        print(f"检测线程: 原始视频FPS已更新为 {fps}")
    
    def set_segment_thread(self, segment_thread):
        """设置分割线程"""
        self.segment_thread = segment_thread
        if segment_thread and self.tracker:
            segment_thread.set_tracker(self.tracker)
        if segment_thread:
            segment_thread.registry = self.registry
            # 转发管线就绪信号
            segment_thread.pipeline_ready.connect(self.pipeline_ready)
    
    def _dims_to_tuple(self, dims):
        """将TensorRT Dims对象转换为Python元组"""
        try:
            dims_list = list(dims)
            return tuple(dims_list)
        except:
            dims_str = str(dims)
            dims_str = dims_str.strip('()[]{}')
            try:
                return tuple(int(x) for x in dims_str.split(',') if x.strip())
            except:
                print(f"警告: 无法解析维度 {dims}，使用默认形状 (1, 3, 640, 640)")
                return (1, 3, 640, 640)
    
    def initialize_engine(self):
        """初始化TensorRT引擎"""
        try:
            print("正在初始化TensorRT引擎...")
            
            # 创建CUDA上下文
            cuda.init()
            self.cuda_context = cuda.Device(0).make_context()
            
            # 检查引擎文件
            if not os.path.exists(self.engine_path):
                raise FileNotFoundError(f"找不到引擎文件: {self.engine_path}")
            
            # TensorRT logger
            self.logger = trt.Logger(trt.Logger.WARNING)
            
            # 加载引擎
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("无法加载TensorRT引擎")
            
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("无法创建执行上下文")
            
            # 创建CUDA流
            self.stream = cuda.Stream()
            
            # 准备输入输出
            self._allocate_buffers()
            
            print("TensorRT引擎初始化成功")
            
            # 初始化跟踪器，设置像素到微米的转换系数
            self.tracker = JPDAFilter(
                process_noise=20.0,
                measure_noise=2.0,
                detect_prob=0.7,
                gate_prob=0.95
            )
            
            # 如果分割线程已存在，传递跟踪器和注册表
            if self.segment_thread:
                self.segment_thread.set_tracker(self.tracker)
                self.segment_thread.registry = self.registry
            
            # 预热
            self._warmup()
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"初始化引擎失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self._cleanup_cuda()
            return False
    
    def _allocate_buffers(self):
        """分配GPU内存缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        print(f"引擎IO张量数量: {self.engine.num_io_tensors}")
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            print(f"处理张量 {i}: {tensor_name}, 模式: {tensor_mode}")
            
            try:
                tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                tensor_shape_dims = self.engine.get_tensor_shape(tensor_name)
                tensor_shape = self._dims_to_tuple(tensor_shape_dims)
                
                print(f"张量形状: {tensor_shape}, 数据类型: {tensor_dtype}")
                
                # 计算大小
                size = int(np.prod(tensor_shape))
                
                # 分配设备内存
                device_mem = cuda.mem_alloc(size * np.dtype(tensor_dtype).itemsize)
                self.bindings.append(int(device_mem))
                
                # 分配主机内存
                host_mem = cuda.pagelocked_empty(size, tensor_dtype)
                
                if tensor_mode == trt.TensorIOMode.INPUT:
                    self.inputs.append({
                        'name': tensor_name, 
                        'shape': tensor_shape, 
                        'dtype': tensor_dtype, 
                        'host': host_mem, 
                        'device': device_mem,
                        'size': size
                    })
                else:
                    self.outputs.append({
                        'name': tensor_name, 
                        'shape': tensor_shape, 
                        'dtype': tensor_dtype, 
                        'host': host_mem, 
                        'device': device_mem,
                        'size': size
                    })
            except Exception as e:
                print(f"处理张量 {tensor_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
    
    def _warmup(self):
        """预热模型"""
        print("正在预热模型...")
        try:
            if self.cuda_context:
                self.cuda_context.push()
            
            if self.inputs:
                dummy_shape = self.inputs[0]['shape']
                dummy_dtype = self.inputs[0]['dtype']
                print(f"创建预热输入, 形状: {dummy_shape}, 类型: {dummy_dtype}")
                dummy_input = np.random.randn(*dummy_shape).astype(dummy_dtype)
                
                for i in range(5):
                    start_time = time.time()
                    _ = self._infer(dummy_input)
                    warm_time = (time.time() - start_time) * 1000
                    print(f"预热第{i+1}次: {warm_time:.1f}ms")
                
                print("模型预热完成！")
            else:
                print("无法预热模型: 没有输入张量")
            
        except Exception as e:
            print(f"预热过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cuda_context:
                self.cuda_context.pop()
    
    def add_frame(self, frame):
        """添加新帧到缓冲区"""
        if frame is None:
            return
        
        # 更新帧尺寸
        if len(frame.shape) >= 2:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        with self.buffer_lock:
            self.frame_buffer.append(frame.copy())
    
    def set_output_label(self, label):
        """设置输出标签"""
        self.mutex.lock()
        self.output_label = label
        self.mutex.unlock()
    
    def set_fullscreen_label(self, label):
        """设置全屏标签"""
        self.mutex.lock()
        self.fullscreen_label = label
        self.mutex.unlock()

    def get_best_sperm_position(self):
        """获取最优精子的当前位置（线程安全，边缘时返回 None）"""
        self.mutex.lock()
        best_id = self.best_track_id
        self.mutex.unlock()

        if best_id is None or self.tracker is None:
            return None

        for track in self.tracker.get_active_tracks():
            if track.id == best_id and len(track.trajectory) > 0:
                x, y = track.trajectory[-1]
                if self._is_near_edge(x, y):
                    return None
                return (x, y)
        return None
    
    def start_detection(self):
        """开始检测"""
        self.mutex.lock()
        self.paused = False
        self.mutex.unlock()
        self._pipeline_signaled = False  # 重置管线就绪标志

        if not self.isRunning():
            self.running = True
            self.start()
    
    def pause_detection(self):
        """暂停检测"""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
    
    def stop_detection(self):
        """停止检测"""
        self.running = False
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
        with self.buffer_lock:
            self.frame_buffer.clear()
        
        # 重置最优精子ID和候选列表
        self.best_track_id = None
        self.top_candidates = []
    
    def _is_near_edge(self, x, y):
        """检查位置是否接近边缘"""
        if self.frame_width == 0 or self.frame_height == 0:
            return False
        
        near_left = x < self.edge_margin
        near_right = x > (self.frame_width - self.edge_margin)
        near_top = y < self.edge_margin
        near_bottom = y > (self.frame_height - self.edge_margin)
        
        return near_left or near_right or near_top or near_bottom
    
    def _find_top_candidates(self, active_tracks):
        """找到VSL最快的8个精子"""
        if not active_tracks:
            return []
        
        # 筛选出不在边缘的精子
        valid_tracks = []
        for track in active_tracks:
            # import pdb; pdb.set_trace()
            if track.vsl < 90.0:
                if len(track.trajectory) > 0:
                    x, y = track.trajectory[-1]
                    if not self._is_near_edge(x, y):
                        # 设置像素到微米的转换系数
                        track.pixel_to_micron = self.pixel_to_micron
                        valid_tracks.append(track)
        
        # 按VSL排序，选择最快的8个
        valid_tracks.sort(key=lambda t: getattr(t, 'vsl', 0), reverse=True)
        top_8 = valid_tracks[:8]
        
        # 返回ID列表
        return [track.id for track in top_8]
    
    def _find_best_sperm_simple(self, active_tracks):
        """简单的最优精子选择（基于VSL）"""
        if not active_tracks:
            return None
        
        # 首先检查当前最优精子是否仍然存在
        current_best_track = None
        if self.best_track_id is not None:
            for track in active_tracks:
                if track.id == self.best_track_id:
                    current_best_track = track
                    break
        
        # 检查当前最优精子是否需要更新
        need_update = False
        if current_best_track:
            # 检查速度是否仍然满足要求
            if not (hasattr(current_best_track, 'vsl') and current_best_track.vsl > self.vsl_threshold and current_best_track.vsl < self.vsl_high_threshold):
                print(f"最优精子 ID={self.best_track_id} 速度降至 {getattr(current_best_track, 'vsl', 0):.2f}，低于阈值")
                need_update = True
            # 检查是否接近边缘
            elif len(current_best_track.trajectory) > 0:
                x, y = current_best_track.trajectory[-1]
                if self._is_near_edge(x, y):
                    print(f"最优精子 ID={self.best_track_id} 接近边缘 (x={x:.0f}, y={y:.0f})，需要更新")
                    need_update = True
        else:
            # 当前最优精子已经消失
            need_update = True
        
        # 如果不需要更新，保持当前最优精子
        if not need_update and current_best_track:
            return self.best_track_id
        
        # 需要选择新的最优精子
        best_track = None
        best_vsl = self.vsl_threshold  # 只选择速度大于阈值的精子
        
        # 筛选满足条件的精子
        for track in active_tracks:
            # 跳过当前即将离开的最优精子
            if track.id == self.best_track_id and need_update:
                continue
            
            # 检查速度是否满足要求
            if not (hasattr(track, 'vsl') and track.vsl > self.vsl_threshold):
                continue
            
            # 检查是否不在边缘
            if len(track.trajectory) > 0:
                x, y = track.trajectory[-1]
                if self._is_near_edge(x, y):
                    continue
            
            # 选择速度最快的
            if track.vsl > best_vsl:
                best_vsl = track.vsl
                best_track = track
        
        # 更新最优精子
        if best_track:
            self.best_track_id = best_track.id
            # 强制设置最优精子为Grade 1
            best_track.grade = 1
            print(f"选择新的最优精子: ID={self.best_track_id}, VSL={best_vsl:.2f}")
            return self.best_track_id
        else:
            # 没有找到合适的精子
            print(f"当前视野中没有满足条件的精子（速度>{self.vsl_threshold}且不在边缘）")
            self.best_track_id = None
            return None
    
    def update_best_sperm_from_segmentation(self, track_id, grade):
        """从分割线程更新最优精子信息"""
        # 这个方法将被分割线程调用，用于更新最优精子的选择
        if grade == 1:
            # 如果是Grade 1精子，可能需要更新最优选择
            self.best_track_id = track_id
            print(f"分割线程更新最优精子: ID={track_id}, Grade={grade}")
    
    def run(self):
        """线程主函数"""
        if self.cuda_context:
            self.cuda_context.push()
        
        try:
            while self.running:
                # 检查是否暂停
                self.mutex.lock()
                paused = self.paused
                self.mutex.unlock()
                
                if paused:
                    self.msleep(50)
                    continue
                
                # 从缓冲区获取帧
                frame = None
                with self.buffer_lock:
                    if self.frame_buffer:
                        frame = self.frame_buffer.popleft()
                    buffer_size = len(self.frame_buffer)
                
                if frame is None:
                    self.msleep(1)
                    continue
                
                # 处理帧
                try:
                    detection_start = time.time()
                    
                    # 预处理
                    input_data = self._preprocess(frame)
                    
                    # 推理
                    output = self._infer(input_data)
                    
                    # 后处理
                    boxes, scores, class_ids = self._postprocess(output)
                    
                    # 限制检测数量
                    if len(boxes) > self.max_detections:
                        indices = np.argsort(scores)[::-1][:self.max_detections]
                        boxes = boxes[indices]
                        scores = scores[indices]
                        class_ids = class_ids[indices]
                    
                    # 更新跟踪器
                    active_tracks = []
                    if self.tracker and len(boxes) > 0:
                        detection_points = []
                        for box in boxes:
                            x_center = (box[0] + box[2]) / 2
                            y_center = (box[1] + box[3]) / 2
                            detection_points.append((x_center, y_center))
                        
                        self.tracker.predict()
                        self.tracker.correct(detection_points)
                        active_tracks = self.tracker.get_active_tracks()
                    elif self.tracker:
                        self.tracker.predict()
                        self.tracker.correct([])
                        active_tracks = self.tracker.get_active_tracks()
                    
                    # 找到VSL最快的8个精子
                    # self.top_candidates = self._find_top_candidates(active_tracks)

                    # 推送运动学数据到注册表 (含检测框匹配)
                    for track in active_tracks:
                        if len(track.trajectory) >= 5:
                            tx, ty = track.trajectory[-1]
                            # 匹配最近的检测框
                            best_box = None
                            best_dist = float('inf')
                            for box in boxes:
                                cx = (box[0] + box[2]) / 2
                                cy = (box[1] + box[3]) / 2
                                dist = (tx - cx)**2 + (ty - cy)**2
                                if dist < best_dist:
                                    best_dist = dist
                                    best_box = box
                            bbox = tuple(best_box) if best_box is not None else None
                            self.registry.update_kinematics(
                                track_id=track.id,
                                vsl=getattr(track, 'vsl', 0.0),
                                alh=getattr(track, 'alh', 0.0),
                                kinematic_grade=getattr(track, 'grade', 6),
                                pos_x=tx, pos_y=ty,
                                bbox=bbox
                            )

                    # 标记消失的轨迹为非活跃
                    active_ids = {t.id for t in active_tracks}
                    inactive_ids = [tid for tid in list(self.registry._records.keys()) if tid not in active_ids]
                    if inactive_ids:
                        self.registry.mark_inactive(inactive_ids)

                    # 使用注册表调度选择 K=8 个精子
                    frame_shape = (self.frame_height, self.frame_width)
                    self.top_candidates = self.registry.select_for_segmentation(
                        K=8, frame_shape=frame_shape, current_frame=self.frame_count
                    )

                    # 最优精子由 SegmentThread 在分割后设置（避免双线程调用 get_best_sperm 导致 hold_count 加速）
                    # self.best_track_id 由 SegmentThread.update → detection_thread.best_track_id

                    # 定期刷写数据库
                    if self.frame_count % 10 == 0:
                        self.registry.flush_to_db()
                    
                    # 检测时间
                    detection_time = (time.time() - detection_start) * 1000  # ms
                    self.detection_times.append(detection_time)
                    
                    # 创建检测结果对象
                    detection_result = DetectionResult(
                        frame=frame.copy(),
                        boxes=boxes,
                        scores=scores,
                        class_ids=class_ids,
                        tracks=active_tracks,
                        best_track_id=self.best_track_id,
                        top_candidates=self.top_candidates
                    )
                    detection_result.detection_time = detection_time
                    detection_result.show_tracking = self.show_tracking
                    
                    # 发送检测结果给分割线程
                    if self.segment_thread:
                        self.segment_thread.add_detection_result(detection_result)
                    else:
                        # 没有分割线程，直接显示检测结果
                        result_frame = self._draw_best_sperm_with_tracking(frame, active_tracks, self.best_track_id)
                        self._send_frame(result_frame)
                        # 首帧产出后通知管线就绪
                        if not self._pipeline_signaled:
                            self._pipeline_signaled = True
                            self.pipeline_ready.emit()
                    
                    # 更新统计
                    self.frame_count += 1
                    self.total_detections = len(boxes)
                    
                    # 计算FPS
                    if len(self.detection_times) > 0:
                        avg_time = np.mean(self.detection_times)
                        self.detection_fps = 1000.0 / avg_time if avg_time > 0 else 0.0
                    
                    # 统计满足速度要求的精子数量
                    qualified_count = sum(1 for track in active_tracks 
                                        if hasattr(track, 'vsl') and track.vsl > self.vsl_threshold)
                    
                    # 统计不在边缘的合格精子数量
                    non_edge_qualified_count = 0
                    for track in active_tracks:
                        if hasattr(track, 'vsl') and track.vsl > self.vsl_threshold:
                            if len(track.trajectory) > 0:
                                x, y = track.trajectory[-1]
                                if not self._is_near_edge(x, y):
                                    non_edge_qualified_count += 1
                    
                    # 发送统计信息
                    registry_stats = self.registry.get_stats()
                    stats = {
                        'detection_fps': self.detection_fps,
                        'detection_buffer': buffer_size,
                        'detection_buffer_max': self.frame_buffer.maxlen,
                        'detection_count': len(boxes),
                        'detection_time': detection_time,
                        'active_tracks': len(active_tracks),
                        'qualified_tracks': qualified_count,
                        'non_edge_qualified': non_edge_qualified_count,
                        'best_track_id': self.best_track_id,
                        'top_candidates': len(self.top_candidates),
                        'vsl_threshold': self.vsl_threshold,
                        'edge_margin': self.edge_margin,
                        'candidate_pool_size': registry_stats['candidate_pool_size'],
                        'scheduled_sperm': len(self.top_candidates),
                        'composite_score_top': registry_stats['composite_score_top'],
                        'avg_measurement_count': registry_stats['avg_measurement_count'],
                        'db_path': registry_stats['db_path'],
                    }
                    self.stats_updated.emit(stats)
                    
                    # 发送检测结果信号
                    self.detection_result.emit(detection_result)
                    
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self._send_frame(frame)
        
        finally:
            if self.cuda_context:
                self.cuda_context.pop()
    
    def _draw_best_sperm_with_tracking(self, img, tracks, best_track_id):
        """绘制最优精子的检测框，并根据show_tracking显示轨迹和椭圆"""
        img_result = img.copy()
        
        # 如果没有最优精子，返回原图
        if best_track_id is None:
            return img_result
        
        # Grade 1的颜色（绿色）
        grade1_color = (0, 255, 0)
        
        # 找到最优精子
        best_track = None
        for track in tracks:
            if track.id == best_track_id:
                best_track = track
                break
        
        if best_track is None or len(best_track.trajectory) == 0:
            return img_result
        
        center_x, center_y = best_track.trajectory[-1]
        
        # 绘制检测框
        box_size = 64
        half_size = box_size // 2
        x1 = int(center_x - half_size)
        y1 = int(center_y - half_size)
        x2 = int(center_x + half_size)
        y2 = int(center_y + half_size)
        
        # 确保框在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_result.shape[1], x2)
        y2 = min(img_result.shape[0], y2)
        
        # 绘制边界框
        cv2.rectangle(img_result, (x1, y1), (x2, y2), grade1_color, 2)
        
        # 如果开启了轨迹显示
        if self.show_tracking:
            # 绘制轨迹
            if len(best_track.trajectory) > 1:
                trajectory_points = np.array(best_track.trajectory, dtype=np.int32)
                # 绘制轨迹线，颜色逐渐变淡
                for i in range(1, len(trajectory_points)):
                    # 计算透明度（越早的点越淡）
                    alpha = i / len(trajectory_points)
                    color_intensity = int(255 * alpha)
                    track_color = (0, color_intensity, 0)  # 绿色渐变
                    
                    cv2.line(img_result, 
                            tuple(trajectory_points[i-1]), 
                            tuple(trajectory_points[i]), 
                            track_color, 2)
            
            # 绘制门限椭圆（只画边框）
            if hasattr(best_track, 'gatingEllipse') and best_track.gatingEllipse is not None:
                try:
                    ellipse = best_track.gatingEllipse
                    # ellipse格式: ((center_x, center_y), (major_axis, minor_axis), angle)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    axes = (int(ellipse[1][0]/2), int(ellipse[1][1]/2))  # 半轴长度
                    angle = ellipse[2]
                    
                    # 绘制椭圆边框（黄色）
                    cv2.ellipse(img_result, center, axes, angle, 0, 360, (0, 255, 255), 2)
                except Exception as e:
                    print(f"绘制椭圆时出错: {e}")
            
            # 显示精子信息
            if hasattr(best_track, 'vsl'):
                info_text = f"ID:{best_track.id} VSL:{best_track.vsl:.1f} Grade:{getattr(best_track, 'grade', 'N/A')}"
                cv2.putText(img_result, info_text, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, grade1_color, 1)
        
        return img_result
    
    def _preprocess(self, img):
        """预处理图片"""
        if img is None:
            raise ValueError("无法处理空图像")
        
        self.original_shape = img.shape[:2]
        
        # 转换颜色空间 BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小到输入尺寸
        input_height, input_width = 640, 640
        if self.inputs and len(self.inputs[0]['shape']) >= 4:
            _, _, input_height, input_width = self.inputs[0]['shape']
        
        img_resized = cv2.resize(img, (input_width, input_height))
        
        # 归一化到[0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 转换为CHW格式
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # 添加batch维度
        img_batch = np.expand_dims(img_chw, axis=0)
        
        # 确保连续内存并转换为正确的数据类型
        img_batch = np.ascontiguousarray(img_batch)
        
        if self.inputs and self.inputs[0]['dtype'] != img_batch.dtype:
            img_batch = img_batch.astype(self.inputs[0]['dtype'])
            
        return img_batch
    
    def _infer(self, input_data):
        """执行推理"""
        try:
            if not self.inputs or not self.outputs:
                raise RuntimeError("未初始化的输入/输出缓冲区")
            
            expected_shape = self.inputs[0]['shape']
            
            if input_data.shape != expected_shape:
                if np.prod(input_data.shape) == np.prod(expected_shape):
                    input_data = input_data.reshape(expected_shape)
                else:
                    raise ValueError(f"输入数据形状无法调整: {input_data.shape} -> {expected_shape}")
            
            expected_dtype = self.inputs[0]['dtype']
            if input_data.dtype != expected_dtype:
                input_data = input_data.astype(expected_dtype)
            
            flat_input = input_data.flatten()
            
            if flat_input.size != self.inputs[0]['size']:
                raise ValueError(f"展平后的输入大小不匹配: 得到 {flat_input.size}, 期望 {self.inputs[0]['size']}")
            
            np.copyto(self.inputs[0]['host'], flat_input)
            
            cuda.memcpy_htod_async(
                self.inputs[0]['device'], 
                self.inputs[0]['host'], 
                self.stream
            )
            
            for inp in self.inputs:
                self.context.set_tensor_address(inp['name'], int(inp['device']))
            
            for out in self.outputs:
                self.context.set_tensor_address(out['name'], int(out['device']))
            
            status = self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            if not status:
                raise RuntimeError("推理执行失败")
            
            for output in self.outputs:
                cuda.memcpy_dtoh_async(
                    output['host'], 
                    output['device'], 
                    self.stream
                )
            
            self.stream.synchronize()
            
            output_data = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
            return output_data
            
        except Exception as e:
            print(f"推理过程出错: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _postprocess(self, output):
        """后处理"""
        # YOLOv8输出格式处理
        if output.ndim == 3:
            output = output[0]
        
        predictions = output.T
        
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
        
        scale_x = self.original_shape[1] / 640
        scale_y = self.original_shape[0] / 640
        
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
        
        indices = self._nms(boxes_xyxy, class_scores, self.iou_threshold)
        
        return boxes_xyxy[indices], class_scores[indices], class_ids[indices]
    
    def _xywh2xyxy(self, boxes):
        """转换边界框格式"""
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return boxes_xyxy
    
    def _nms(self, boxes, scores, iou_threshold):
        """非极大值抑制"""
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
    
    def _generate_colors(self, num_colors):
        """生成颜色"""
        colors = []
        for i in range(num_colors):
            h = i / num_colors
            s = 0.8
            v = 0.8
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgb = tuple(int(x * 255) for x in rgb)
            colors.append(rgb)
        return colors
    
    def _send_frame(self, frame):
        """发送处理后的帧"""
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, 
                          QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.frame_ready.emit(pixmap)
        except Exception as e:
            print(f"发送帧时出错: {e}")
    
    def _update_display(self, pixmap):
        """更新显示"""
        if self.fullscreen_label and self.fullscreen_label.isVisible():
            self._update_label(self.fullscreen_label, pixmap)
        elif self.output_label:
            self._update_label(self.output_label, pixmap)
    
    def _update_label(self, label, pixmap):
        """更新标签显示"""
        if label and not pixmap.isNull():
            try:
                label_size = label.size()
                
                if label_size.width() > 10 and label_size.height() > 10:
                    scaled_pixmap = pixmap.scaled(
                        label_size,
                        aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                        transformMode=Qt.TransformationMode.SmoothTransformation
                    )
                    label.setPixmap(scaled_pixmap)
                else:
                    label.setPixmap(pixmap)
            except Exception as e:
                print(f"更新显示时出错: {e}")
    
    def _handle_error(self, error_msg):
        """处理错误"""
        print(f"检测线程错误: {error_msg}")
    
    def set_conf_threshold(self, value):
        """设置置信度阈值"""
        self.conf_threshold = value
        print(f"置信度阈值已更新为: {value}")
    
    def set_iou_threshold(self, value):
        """设置IoU阈值"""
        self.iou_threshold = value
        print(f"IoU阈值已更新为: {value}")
    
    def set_vsl_threshold(self, threshold):
        """设置VSL速度阈值"""
        self.vsl_threshold = threshold
        print(f"VSL速度阈值已更新为: {threshold}")
    
    def set_edge_margin(self, margin):
        """设置边缘距离阈值"""
        self.edge_margin = margin
        print(f"边缘距离阈值已更新为: {margin} 像素")
    
    def get_buffer_status(self):
        """获取缓冲区状态"""
        with self.buffer_lock:
            return len(self.frame_buffer), self.frame_buffer.maxlen
    
    def _cleanup_cuda(self):
        """清理CUDA资源"""
        try:
            if self.cuda_context:
                self.cuda_context.pop()
                self.cuda_context = None
        except:
            pass
    
    def __del__(self):
        """析构函数，清理资源"""
        self._cleanup_cuda()