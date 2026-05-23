import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import yaml
from collections import deque
import threading
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, Qt
from PyQt6.QtGui import QImage, QPixmap
import albumentations as A
from albumentations.core.composition import Compose
import colorsys

# 从detection_thread导入分级参数
from detection_thread import GradeParameters

class SpermMorphology:
    """精子形态学数据"""
    def __init__(self):
        self.head_length = 0.0
        self.head_width = 0.0
        self.head_ratio = 0.0
        self.head_area = 0.0
        self.vsl = 0.0
        self.alh = 0.0
        self.grade = -1

class SegmentThread(QThread):
    """分割线程"""
    
    # 定义信号
    segmented_frame = pyqtSignal(np.ndarray)
    frame_ready = pyqtSignal(QPixmap)
    error_occurred = pyqtSignal(str)
    stats_updated = pyqtSignal(dict)
    best_sperm_updated = pyqtSignal(int, int)  # track_id, grade
    
    def __init__(self, engine_path="seg_weights/ADSCNet_sperm_ROINAHead_250707/model_fp32.engine", 
                 config_path="seg_weights/ADSCNet_sperm_ROINAHead_250707/config.yml", 
                 parent=None):
        super().__init__(parent)
        
        self.engine_path = engine_path
        self.config_path = config_path
        self.running = False
        self.paused = True
        self.mutex = QMutex()

        # Add show tracking control
        self.show_tracking = False
        
        # 检测结果缓冲区
        self.detection_buffer = deque(maxlen=50)
        self.buffer_lock = threading.Lock()
        
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
        
        # 分割参数
        self.num_classes = 4
        self.input_size = 64
        self.roi_size = 64
        
        # 透明度
        self.overlay_alpha = 0.5
        
        # 跟踪器引用
        self.tracker = None
        
        # 输出标签
        self.output_label = None
        
        # 性能统计
        self.segment_fps = 0.0
        self.segment_times = deque(maxlen=30)
        self.total_segmentations = 0
        
        # 像素到微米的转换系数
        self.pixel_to_micron = 0.1
        
        # 分级参数
        self.grade_params = GradeParameters()
        
        # 精子形态学数据缓存
        self.sperm_morphology_cache = {}  # {track_id: SpermMorphology}
        
        # 检测线程引用
        self.detection_thread = None
        
        # 最优精子ID
        self.best_sperm_id = None
        
        # 边缘距离阈值
        self.edge_margin = 50
        
        # 加载配置
        self.load_config()
        
        # 设置预处理变换
        self.transform = Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(),
        ])
        
        # 生成分割颜色
        self.seg_colors = self._get_segmentation_colors(self.num_classes)
        
        # Grade颜色定义
        self.grade_colors = {
            1: (0, 255, 0),      # Grade 1: 绿色
            2: (0, 255, 255),    # Grade 2: 黄色
            3: (255, 128, 0),    # Grade 3: 橙色
            4: (255, 0, 0),      # Grade 4: 红色
            5: (128, 0, 128),    # Grade 5: 紫色
            -1: (128, 128, 128)  # 未分级: 灰色
        }
        
        # 连接信号
        self.frame_ready.connect(self._update_display)
    
    def set_tracker(self, tracker):
        """设置跟踪器引用"""
        self.tracker = tracker
    
    def set_detection_thread(self, detection_thread):
        """设置检测线程引用"""
        self.detection_thread = detection_thread
    
    def set_output_label(self, label):
        """设置输出标签"""
        self.mutex.lock()
        self.output_label = label
        self.mutex.unlock()
    
    def set_show_tracking(self, show):
        """设置是否显示跟踪轨迹"""
        self.show_tracking = show
        print(f"分割线程: 轨迹显示已{'启用' if show else '禁用'}")
    
    def set_pixel_to_micron(self, value):
        """设置像素到微米的转换系数"""
        self.pixel_to_micron = value
    
    def _get_segmentation_colors(self, num_classes):
        """生成分割颜色"""
        seg_colors = []
        for i in range(num_classes):
            h = (i * 0.618033988749895) % 1.0
            s = 0.7
            v = 0.95
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgb = tuple(int(x * 255) for x in rgb)
            seg_colors.append(rgb)
        return seg_colors
    
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                    self.num_classes = config.get('num_classes', 4)
                    print(f"加载配置: num_classes={self.num_classes}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
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
                print(f"警告: 无法解析维度 {dims}，使用默认形状 (1, 3, 64, 64)")
                return (1, 3, 64, 64)
    
    def initialize_engine(self):
        """初始化TensorRT引擎"""
        try:
            print("正在初始化分割TensorRT引擎...")
            
            # 创建CUDA上下文
            self.cuda_context = cuda.Device(0).make_context()
            
            # 检查引擎文件
            if not os.path.exists(self.engine_path):
                raise FileNotFoundError(f"找不到引擎文件: {self.engine_path}")
            
            # TensorRT logger
            self.logger = trt.Logger(trt.Logger.WARNING)
            
            # 加载引擎
            with open(self.engine_path, 'rb') as f:
                self.runtime = trt.Runtime(self.logger)
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                raise RuntimeError("无法加载分割TensorRT引擎")
            
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("无法创建执行上下文")
            
            # 获取输入张量信息
            self.input_tensor_names = []
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.input_tensor_names.append(tensor_name)
            
            if not self.input_tensor_names:
                raise RuntimeError("引擎没有输入张量")
            
            self.input_tensor_name = self.input_tensor_names[0]
            
            # 获取输入形状
            engine_input_dims = self.engine.get_tensor_shape(self.input_tensor_name)
            self.engine_input_shape = self._dims_to_tuple(engine_input_dims)
            
            # 设置输入形状
            if len(self.engine_input_shape) >= 4:
                self.input_shape = (1, *self.engine_input_shape[1:])
            else:
                self.input_shape = (1, 3, self.input_size, self.input_size)
            
            # 设置执行上下文的输入形状
            try:
                self.context.set_input_shape(self.input_tensor_name, self.input_shape)
                print(f"已设置输入形状: {self.input_shape}")
            except Exception as e:
                print(f"设置输入形状失败: {e}")
            
            # 创建CUDA流
            self.stream = cuda.Stream()
            
            # 分配缓冲区
            self._allocate_buffers()
            
            print("分割TensorRT引擎初始化成功")
            
            # 预热
            self._warmup()
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"初始化分割引擎失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self._cleanup_cuda()
            return False
    
    def _allocate_buffers(self):
        """分配GPU内存缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_shapes = []
        self.output_tensor_names = []
        
        print("开始分配内存缓冲区...")
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            print(f"处理张量 {i}: {tensor_name}, 模式: {tensor_mode}")
            
            if tensor_mode == trt.TensorIOMode.INPUT:
                shape = self.input_shape
                print(f"输入张量形状: {shape}")
                
                try:
                    self.context.set_input_shape(tensor_name, shape)
                except Exception as e:
                    print(f"设置输入形状失败: {e}")
            else:
                try:
                    output_dims = self.context.get_tensor_shape(tensor_name)
                    shape = self._dims_to_tuple(output_dims)
                    
                    if len(shape) >= 4 and shape[0] != self.input_shape[0] and shape[0] != -1:
                        shape = (self.input_shape[0], *shape[1:])
                    
                except Exception as e:
                    print(f"获取输出形状失败: {e}，使用默认形状")
                    shape = (1, self.num_classes, self.input_size, self.input_size)
                
                self.output_tensor_names.append(tensor_name)
            
            try:
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            except Exception as e:
                print(f"获取数据类型失败: {e}，使用默认类型 float32")
                dtype = np.float32
            
            try:
                size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                device_mem = cuda.mem_alloc(size)
                host_mem = cuda.pagelocked_empty(shape, dtype)
                
                if tensor_mode == trt.TensorIOMode.INPUT:
                    self.inputs.append({
                        'host': host_mem, 
                        'device': device_mem, 
                        'shape': shape,
                        'name': tensor_name
                    })
                else:
                    self.outputs.append({
                        'host': host_mem, 
                        'device': device_mem, 
                        'shape': shape,
                        'name': tensor_name
                    })
                    self.output_shapes.append(shape)
            except Exception as e:
                print(f"分配内存失败: {e}")
                raise
        
        print(f"内存缓冲区分配完成。输入数量: {len(self.inputs)}, 输出数量: {len(self.outputs)}")
    
    def _warmup(self):
        """预热模型"""
        print("正在预热分割模型...")
        try:
            if self.cuda_context:
                self.cuda_context.push()
            
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            for i in range(5):
                start_time = time.time()
                _ = self._infer(dummy_input)
                warm_time = (time.time() - start_time) * 1000
                print(f"分割预热第{i+1}次: {warm_time:.1f}ms")
            
            print("分割模型预热完成！")
            
        except Exception as e:
            print(f"分割预热过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cuda_context:
                self.cuda_context.pop()
    
    def add_detection_result(self, detection_result):
        """添加检测结果到缓冲区"""
        with self.buffer_lock:
            self.detection_buffer.append(detection_result)
    
    def start_segmentation(self):
        """开始分割"""
        self.mutex.lock()
        self.paused = False
        self.mutex.unlock()
        
        if not self.isRunning():
            self.running = True
            self.start()
    
    def pause_segmentation(self):
        """暂停分割"""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
    
    def stop_segmentation(self):
        """停止分割"""
        self.running = False
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
        with self.buffer_lock:
            self.detection_buffer.clear()
    
    def _is_near_edge(self, x, y, frame_shape):
        """检查位置是否接近边缘"""
        height, width = frame_shape[:2]
        near_left = x < self.edge_margin
        near_right = x > (width - self.edge_margin)
        near_top = y < self.edge_margin
        near_bottom = y > (height - self.edge_margin)
        
        return near_left or near_right or near_top or near_bottom
    
    def _should_update_best_sperm(self, best_track, frame_shape):
        """判断是否需要更新最优精子"""
        if best_track is None:
            return True
        
        if len(best_track.trajectory) == 0:
            return True
        
        # 检查精子是否还在视野内
        x, y = best_track.trajectory[-1]
        if self._is_near_edge(x, y, frame_shape):
            return True
        
        return False
    
    def _calculate_head_morphology(self, masks):
        """计算精子头部形态学参数"""
        # 根据类别定义合并头部掩码
        # 类别1（顶体）+ 类别2（细胞核）= 正常头部
        # 类别3（non-measurable head）= 异常头部
        
        head_mask = np.zeros_like(masks[0], dtype=np.uint8)
        
        # 检查是否有类别1和2
        has_acrosome = np.any(masks[0] > 0.5) if len(masks) > 0 else False
        has_nucleus = np.any(masks[1] > 0.5) if len(masks) > 1 else False
        has_nonmeasurable = np.any(masks[2] > 0.5) if len(masks) > 2 else False
        
        if has_acrosome and has_nucleus:
            # 正常头部：合并顶体和细胞核
            head_mask = np.logical_or(masks[0] > 0.5, masks[1] > 0.5).astype(np.uint8) * 255
        elif has_nonmeasurable:
            # 异常头部：使用non-measurable head
            head_mask = (masks[2] > 0.5).astype(np.uint8) * 255
        else:
            # 没有检测到头部
            return 0.0, 0.0, 0.0, 0.0
        
        # 找到轮廓
        contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0, 0.0, 0.0, 0.0
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算面积
        area_pixels = cv2.contourArea(largest_contour)
        head_area = area_pixels * self.pixel_to_micron * self.pixel_to_micron
        
        # 拟合椭圆
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            # ellipse = ((center_x, center_y), (width, height), angle)
            width_pixels = ellipse[1][0]
            height_pixels = ellipse[1][1]
            
            # 长度是较大的值，宽度是较小的值
            head_length_pixels = max(width_pixels, height_pixels)
            head_width_pixels = min(width_pixels, height_pixels)
            
            # 转换为微米
            head_length = head_length_pixels * self.pixel_to_micron
            head_width = head_width_pixels * self.pixel_to_micron
            
            # 计算长宽比
            head_ratio = head_length / head_width if head_width > 0 else 0.0
        else:
            # 轮廓点太少，使用边界框
            x, y, w, h = cv2.boundingRect(largest_contour)
            head_length = max(w, h) * self.pixel_to_micron
            head_width = min(w, h) * self.pixel_to_micron
            head_ratio = head_length / head_width if head_width > 0 else 0.0
        
        return head_length, head_width, head_ratio, head_area
    
    def _grade_sperm(self, morphology):
        """根据形态学参数对精子进行分级"""
        # Grade 1
        if (self.grade_params.GRADE_1_VSL_LOW_LIMIT <= morphology.vsl <= 11.0 and
            0.0 < morphology.alh <= self.grade_params.GRADE_1_ALH_HIGH_LIMIT and
            self.grade_params.GRADE_1_HEAD_LENGTH_LOW_LIMIT <= morphology.head_length <= self.grade_params.GRADE_1_HEAD_LENGTH_UP_LIMIT and
            self.grade_params.GRADE_1_HEAD_WIDTH_LOW_LIMIT <= morphology.head_width <= self.grade_params.GRADE_1_HEAD_WIDTH_UP_LIMIT and
            self.grade_params.GRADE_1_HEAD_RATIO_LOW_LIMIT <= morphology.head_ratio <= self.grade_params.GRADE_1_HEAD_RATIO_UP_LIMIT):
            return 1
        
        # Grade 2
        elif (self.grade_params.GRADE_2_VSL_LOW_LIMIT <= morphology.vsl <= 11.0 and
              0.0 < morphology.alh <= self.grade_params.GRADE_2_ALH_HIGH_LIMIT and
              self.grade_params.GRADE_2_HEAD_LENGTH_LOW_LIMIT <= morphology.head_length <= self.grade_params.GRADE_2_HEAD_LENGTH_UP_LIMIT and
              self.grade_params.GRADE_2_HEAD_WIDTH_LOW_LIMIT <= morphology.head_width <= self.grade_params.GRADE_2_HEAD_WIDTH_UP_LIMIT and
              self.grade_params.GRADE_2_HEAD_RATIO_LOW_LIMIT <= morphology.head_ratio <= self.grade_params.GRADE_2_HEAD_RATIO_UP_LIMIT):
            return 2
        
        # Grade 3
        elif (self.grade_params.GRADE_3_VSL_LOW_LIMIT <= morphology.vsl <= 11.0 and
              0.0 < morphology.alh <= self.grade_params.GRADE_3_ALH_HIGH_LIMIT and
              self.grade_params.GRADE_3_HEAD_RATIO_LOW_LIMIT <= morphology.head_ratio <= self.grade_params.GRADE_3_HEAD_RATIO_UP_LIMIT):
            return 3
        
        # Grade 4
        elif (self.grade_params.GRADE_4_VSL_LOW_LIMIT <= morphology.vsl <= 11.0 and
              self.grade_params.GRADE_4_HEAD_RATIO_LOW_LIMIT <= morphology.head_ratio <= self.grade_params.GRADE_4_HEAD_RATIO_UP_LIMIT):
            return 4
        
        # Grade 5
        elif self.grade_params.GRADE_5_VSL_LOW_LIMIT <= morphology.vsl <= 11.0:
            return 5
        
        else:
            return -1
    
    def _find_best_grade1_sperm(self, candidates_morphology):
        """从候选精子中找到最优的Grade 1精子"""
        grade1_sperms = []
        
        for track_id, morphology in candidates_morphology.items():
            if morphology.grade == 1:
                grade1_sperms.append((track_id, morphology))
        
        if not grade1_sperms:
            # 没有Grade 1精子，选择等级最高的
            best_grade = 6
            best_track_id = None
            best_vsl = 0.0
            
            for track_id, morphology in candidates_morphology.items():
                if morphology.grade != -1:
                    if morphology.grade < best_grade:
                        best_grade = morphology.grade
                        best_track_id = track_id
                        best_vsl = morphology.vsl
                    elif morphology.grade == best_grade and morphology.vsl > best_vsl:
                        best_track_id = track_id
                        best_vsl = morphology.vsl
            
            return best_track_id
        
        # 有Grade 1精子，选择VSL最快的
        best_track_id = None
        best_vsl = 0.0
        
        for track_id, morphology in grade1_sperms:
            if morphology.vsl > best_vsl:
                best_vsl = morphology.vsl
                best_track_id = track_id
        
        return best_track_id
    
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
                
                # 从缓冲区获取检测结果
                detection_result = None
                with self.buffer_lock:
                    if self.detection_buffer:
                        detection_result = self.detection_buffer.popleft()
                    buffer_size = len(self.detection_buffer)
                
                if detection_result is None:
                    self.msleep(10)
                    continue
                
                # 处理分割
                try:
                    segment_start = time.time()
                    
                    if self.show_tracking:
                        # 录制模式：显示所有信息
                        result_frame = self._process_all_sperms_segmentation(detection_result)
                    else:
                        # 正常模式：只显示最优精子
                        result_frame = self._process_best_sperm_only(detection_result)
                    
                    # 分割时间
                    segment_time = (time.time() - segment_start) * 1000  # ms
                    self.segment_times.append(segment_time)
                    
                    # 计算FPS
                    if len(self.segment_times) > 0:
                        avg_time = np.mean(self.segment_times)
                        self.segment_fps = 1000.0 / avg_time if avg_time > 0 else 0.0
                    
                    # 发送结果帧
                    self.segmented_frame.emit(result_frame)
                    self._send_frame(result_frame)
                    
                    # 更新统计
                    self.total_segmentations += 1
                    
                    # 发送统计信息
                    stats = {
                        'segment_fps': self.segment_fps,
                        'segment_buffer': buffer_size,
                        'segment_buffer_max': self.detection_buffer.maxlen,
                        'segment_time': segment_time,
                        'total_time': segment_time + getattr(detection_result, 'detection_time', 0)
                    }
                    self.stats_updated.emit(stats)
                    
                except Exception as e:
                    print(f"分割处理出错: {e}")
                    import traceback
                    traceback.print_exc()
                    # 发送原始帧
                    if detection_result:
                        self._send_frame(detection_result.frame)
        
        finally:
            if self.cuda_context:
                self.cuda_context.pop()
    
    def _process_best_sperm_only(self, detection_result):
        """正常模式：只处理和显示最优精子"""
        frame = detection_result.frame
        tracks = detection_result.tracks
        top_candidates = detection_result.top_candidates
        
        result_frame = frame.copy()
        
        # 如果没有候选精子，直接返回原图
        if not top_candidates or not tracks:
            return result_frame
        
        # 获取当前最优精子
        best_track = None
        if self.best_sperm_id:
            for track in tracks:
                if track.id == self.best_sperm_id:
                    best_track = track
                    break
        
        # 判断是否需要更新最优精子
        if self._should_update_best_sperm(best_track, frame.shape):
            # 需要重新选择最优精子
            candidates_morphology = {}
            
            # 找到所有候选精子
            candidate_tracks = []
            for track in tracks:
                if track.id in top_candidates:
                    candidate_tracks.append(track)
            
            # 对候选精子进行分析
            for track in candidate_tracks:
                if len(track.trajectory) == 0:
                    continue
                
                center_x, center_y = track.trajectory[-1]
                
                # 跳过边缘精子
                if self._is_near_edge(center_x, center_y, frame.shape):
                    continue
                
                # 提取ROI
                roi, roi_box = self._extract_roi(frame, center_x, center_y)
                
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    # 预处理ROI
                    roi_tensor = self._preprocess(roi)
                    
                    # 运行分割
                    try:
                        seg_output = self._infer(roi_tensor)
                        
                        # 后处理
                        masks, probs = self._postprocess(seg_output, roi_size=(self.roi_size, self.roi_size))
                        
                        # 计算形态学参数
                        head_length, head_width, head_ratio, head_area = self._calculate_head_morphology(masks)
                        
                        # 创建形态学数据对象
                        morphology = SpermMorphology()
                        morphology.head_length = head_length
                        morphology.head_width = head_width
                        morphology.head_ratio = head_ratio
                        morphology.head_area = head_area
                        morphology.vsl = getattr(track, 'vsl', 0.0)
                        morphology.alh = getattr(track, 'alh', 0.0)
                        
                        # 进行分级
                        morphology.grade = self._grade_sperm(morphology)
                        
                        # 保存到缓存
                        candidates_morphology[track.id] = morphology
                        self.sperm_morphology_cache[track.id] = morphology
                        
                        # 更新跟踪器中的分级
                        track.grade = morphology.grade
                        
                    except Exception as e:
                        print(f"处理精子 ID={track.id} 的分割失败: {e}")
            
            # 找到最优的精子
            if candidates_morphology:
                self.best_sperm_id = self._find_best_grade1_sperm(candidates_morphology)
                
                # 更新best_track
                for track in tracks:
                    if track.id == self.best_sperm_id:
                        best_track = track
                        break
                
                # 通知检测线程
                if self.best_sperm_id and self.detection_thread:
                    self.detection_thread.best_track_id = self.best_sperm_id
                    if self.best_sperm_id in candidates_morphology:
                        best_grade = candidates_morphology[self.best_sperm_id].grade
                        best_sperm_area = candidates_morphology[self.best_sperm_id].head_area
                        best_sperm_ratio = candidates_morphology[self.best_sperm_id].head_ratio
                        best_sperm_vsl = candidates_morphology[self.best_sperm_id].vsl
                        self.best_sperm_updated.emit(self.best_sperm_id, best_grade)
                        print(f"选择新的最优精子: ID={self.best_sperm_id}, Grade={best_grade}, VSL={best_sperm_vsl}, Head Area={best_sperm_area}, Head Ratio={best_sperm_ratio}")
        
        # 绘制最优精子
        if best_track and len(best_track.trajectory) > 0:
            center_x, center_y = best_track.trajectory[-1]
            
            # Grade 1的颜色（绿色）
            grade1_color = self.grade_colors[1]
            
            # 绘制64x64检测框
            box_size = 64
            half_size = box_size // 2
            x1 = int(center_x - half_size)
            y1 = int(center_y - half_size)
            x2 = int(center_x + half_size)
            y2 = int(center_y + half_size)
            
            # 确保框在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(result_frame.shape[1], x2)
            y2 = min(result_frame.shape[0], y2)
            
            # 绘制检测框（不显示任何文字）
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), grade1_color, 2)
            
            # 提取ROI进行分割可视化
            roi, roi_box = self._extract_roi(frame, center_x, center_y)
            
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                roi_tensor = self._preprocess(roi)
                
                try:
                    seg_output = self._infer(roi_tensor)
                    masks, probs = self._postprocess(seg_output, roi_size=(self.roi_size, self.roi_size))
                    
                    # 应用分割叠加
                    result_frame = self._overlay_segmentation(result_frame, masks, roi_box, 
                                                            self.seg_colors, self.overlay_alpha)
                except Exception as e:
                    print(f"可视化分割失败: {e}")
        
        return result_frame
    
    def _process_all_sperms_segmentation(self, detection_result):
        """录制模式：处理和显示所有精子的完整信息"""
        frame = detection_result.frame
        tracks = detection_result.tracks
        
        result_frame = frame.copy()
        
        if not tracks:
            return result_frame
        
        # 处理所有活跃的精子
        for track in tracks:
            if len(track.trajectory) == 0:
                continue
            
            center_x, center_y = track.trajectory[-1]
            
            # 获取或计算精子的分级
            if track.id in self.sperm_morphology_cache:
                morphology = self.sperm_morphology_cache[track.id]
                grade = morphology.grade
            else:
                # 需要进行分割和分析
                roi, roi_box = self._extract_roi(frame, center_x, center_y)
                
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    roi_tensor = self._preprocess(roi)
                    
                    try:
                        seg_output = self._infer(roi_tensor)
                        masks, probs = self._postprocess(seg_output, roi_size=(self.roi_size, self.roi_size))
                        
                        # 计算形态学参数
                        head_length, head_width, head_ratio, head_area = self._calculate_head_morphology(masks)
                        
                        # 创建形态学数据对象
                        morphology = SpermMorphology()
                        morphology.head_length = head_length
                        morphology.head_width = head_width
                        morphology.head_ratio = head_ratio
                        morphology.head_area = head_area
                        morphology.vsl = getattr(track, 'vsl', 0.0)
                        morphology.alh = getattr(track, 'alh', 0.0)
                        
                        # 进行分级
                        morphology.grade = self._grade_sperm(morphology)
                        grade = morphology.grade
                        
                        # 保存到缓存
                        self.sperm_morphology_cache[track.id] = morphology
                        
                        # 更新跟踪器中的分级
                        track.grade = grade
                        
                    except Exception as e:
                        print(f"处理精子 ID={track.id} 的分割失败: {e}")
                        grade = -1
                else:
                    grade = -1
            
            # 获取颜色
            color = self.grade_colors.get(grade, self.grade_colors[-1])
            
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
            x2 = min(result_frame.shape[1], x2)
            y2 = min(result_frame.shape[0], y2)
            
            # 绘制边界框，最优精子用更粗的线
            thickness = 3 if track.id == self.best_sperm_id else 2
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            # 显示精子信息
            vsl = getattr(track, 'vsl', 0.0)
            info_text = f"ID:{track.id} G{grade} VSL:{vsl:.1f}"
            cv2.putText(result_frame, info_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 绘制轨迹
            if len(track.trajectory) > 1:
                trajectory_points = np.array(track.trajectory, dtype=np.int32)
                for i in range(1, len(trajectory_points)):
                    alpha = i / len(trajectory_points)
                    color_intensity = int(255 * alpha)
                    # 使用等级对应的颜色
                    if grade == 1:
                        track_color = (0, color_intensity, 0)  # 绿色渐变
                    elif grade == 2:
                        track_color = (0, color_intensity, color_intensity)  # 黄色渐变
                    elif grade == 3:
                        track_color = (color_intensity, int(color_intensity * 0.5), 0)  # 橙色渐变
                    elif grade == 4:
                        track_color = (color_intensity, 0, 0)  # 红色渐变
                    else:
                        track_color = (int(color_intensity * 0.5), int(color_intensity * 0.5), int(color_intensity * 0.5))  # 灰色渐变
                    
                    cv2.line(result_frame, 
                            tuple(trajectory_points[i-1]), 
                            tuple(trajectory_points[i]), 
                            track_color, 2)
            
            # 绘制门限椭圆
            if hasattr(track, 'gatingEllipse') and track.gatingEllipse is not None:
                try:
                    ellipse = track.gatingEllipse
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    axes = (int(ellipse[1][0]/2), int(ellipse[1][1]/2))
                    angle = ellipse[2]
                    cv2.ellipse(result_frame, center, axes, angle, 0, 360, (0, 255, 255), 2)
                except Exception as e:
                    print(f"绘制椭圆时出错: {e}")
            
            # 绘制分割mask
            roi, roi_box = self._extract_roi(frame, center_x, center_y)
            
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                roi_tensor = self._preprocess(roi)
                
                try:
                    seg_output = self._infer(roi_tensor)
                    masks, probs = self._postprocess(seg_output, roi_size=(self.roi_size, self.roi_size))
                    
                    # 应用分割叠加
                    result_frame = self._overlay_segmentation(result_frame, masks, roi_box, 
                                                            self.seg_colors, self.overlay_alpha)
                except Exception as e:
                    print(f"可视化分割失败: {e}")
        
        return result_frame
    
    def _extract_roi(self, frame, center_x, center_y):
        """提取ROI"""
        h, w = frame.shape[:2]
        
        half_size = self.roi_size // 2
        x1 = int(max(0, center_x - half_size))
        y1 = int(max(0, center_y - half_size))
        x2 = int(min(w, center_x + half_size))
        y2 = int(min(h, center_y + half_size))
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.shape[0] != self.roi_size or roi.shape[1] != self.roi_size:
            padded_roi = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
            roi_h, roi_w = roi.shape[:2]
            padded_roi[:roi_h, :roi_w, :] = roi
            roi = padded_roi
        
        return roi, (x1, y1, x2, y2)
    
    def _preprocess(self, img):
        """预处理ROI"""
        if img is None:
            raise ValueError("Cannot process empty image")
        
        dummy_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        augmented = self.transform(image=img, mask=dummy_mask)
        img_transformed = augmented['image']
        
        img_normalized = img_transformed.astype('float32') / 255
        img_chw = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_chw, axis=0)
        img_batch = np.ascontiguousarray(img_batch)
        
        return img_batch
    
    def _infer(self, input_data):
        """执行推理"""
        try:
            if input_data.shape != self.input_shape:
                if np.prod(input_data.shape) == np.prod(self.input_shape):
                    input_data = input_data.reshape(self.input_shape)
                else:
                    raise ValueError(f"输入形状不匹配: 得到 {input_data.shape}, 期望 {self.input_shape}")
            
            np.copyto(self.inputs[0]['host'], input_data)
            
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
                raise RuntimeError("执行推理失败")
            
            for output in self.outputs:
                cuda.memcpy_dtoh_async(
                    output['host'],
                    output['device'],
                    self.stream
                )
            
            self.stream.synchronize()
            
            outputs = []
            for i, output in enumerate(self.outputs):
                output_data = output['host'].copy()
                outputs.append(output_data)
            
            return outputs[0] if len(outputs) == 1 else outputs
            
        except Exception as e:
            print(f"推理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            dummy_output = np.zeros((1, self.num_classes, self.input_size, self.input_size), dtype=np.float32)
            return dummy_output
    
    def _postprocess(self, output, roi_size=(64, 64)):
        """后处理分割输出"""
        output_prob = 1 / (1 + np.exp(-output))
        
        output_binary = output_prob.copy()
        output_binary[output_binary >= 0.5] = 1
        output_binary[output_binary < 0.5] = 0
        
        segmentation_masks = []
        for c in range(self.num_classes):
            mask_resized = cv2.resize(
                output_binary[0, c],
                (roi_size[1], roi_size[0])
            )
            segmentation_masks.append(mask_resized)
        
        return segmentation_masks, output_prob[0]
    
    def _overlay_segmentation(self, frame, masks, roi_box, seg_colors, alpha=0.5):
        """叠加分割结果"""
        x1, y1, x2, y2 = roi_box
        roi_h, roi_w = y2 - y1, x2 - x1
        
        overlay = frame.copy()
        roi = overlay[y1:y2, x1:x2]
        
        for i, mask in enumerate(masks):
            if mask.shape[0] != roi_h or mask.shape[1] != roi_w:
                mask = cv2.resize(mask, (roi_w, roi_h))
            
            binary_mask = mask > 0.5
            
            if np.any(binary_mask):
                color_array = np.array(seg_colors[i], dtype=np.uint8)
                for c in range(3):
                    roi[:, :, c] = np.where(
                        binary_mask,
                        roi[:, :, c] * (1 - alpha) + color_array[c] * alpha,
                        roi[:, :, c]
                    )
        
        return overlay
    
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
        if self.output_label and not pixmap.isNull():
            try:
                label_size = self.output_label.size()
                
                if label_size.width() > 10 and label_size.height() > 10:
                    scaled_pixmap = pixmap.scaled(
                        label_size,
                        aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                        transformMode=Qt.TransformationMode.SmoothTransformation
                    )
                    self.output_label.setPixmap(scaled_pixmap)
                else:
                    self.output_label.setPixmap(pixmap)
            except Exception as e:
                print(f"更新显示时出错: {e}")
    
    def get_buffer_status(self):
        """获取缓冲区状态"""
        with self.buffer_lock:
            return len(self.detection_buffer), self.detection_buffer.maxlen
    
    def _cleanup_cuda(self):
        """清理CUDA资源"""
        try:
            if self.cuda_context:
                self.cuda_context.pop()
                self.cuda_context = None
        except:
            pass
    
    def __del__(self):
        """析构函数"""
        self._cleanup_cuda()