import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QSize, Qt, QCoreApplication
from PyQt6.QtGui import QImage, QPixmap, QGuiApplication
from PyQt6.QtWidgets import QLabel

class VideoTestThread(QThread):
    """视频测试线程，用于读取视频并显示在QLabel上"""
    
    # 定义信号
    frame_ready = pyqtSignal(QPixmap)
    error_occurred = pyqtSignal(str)
    new_frame = pyqtSignal(np.ndarray)  # 新帧信号
    fps_changed = pyqtSignal(float)      # FPS变化信号
    
    def __init__(self, output_label, parent=None):
        """初始化视频线程"""
        super().__init__(parent)
        self.output_label = output_label
        self.running = False
        self.paused = False  # 初始化为不暂停，让视频自动播放
        self.mutex = QMutex()
        # self.video_path = "test-videoes/XY-test1.mp4"  # 指定视频路径
        # self.video_path = r"test-videoes/pvp1.mp4"  # 指定视频路径
        # self.video_path = r"test-videoes/pvp2.mp4"  # 指定视频路径

        # self.video_path = r"E:\\MMRL-LAB-NAS\\Han-Workspace\\20250815-SwimUpUsingLatest\\using1.mp4"  # 指定视频路径
        # self.video_path = r"E:\\MMRL-LAB-NAS\\Han-Workspace\\20250815-SwimUpUsingLatest\\using2.mp4"  # 指定视频路径
        # self.video_path = r"E:\\MMRL-LAB-NAS\\Han-Workspace\\20250815-SwimUpUsingLatest\\using3.mp4"  # 指定视频路径
        # self.video_path = r"E:\\MMRL-LAB-NAS\\Lab电脑实验数据传输\\Han-Workspace\\20250815-SwimUpUsingLatest\\using4.mp4"
        # self.video_path = r"E:\\MMRL-LAB-NAS\\Lab电脑实验数据传输\\Han-Workspace\\20250815-SwimUpUsingLatest\short4usin\\using3.mp4"
        # self.video_path = r"E:\\MMRL-LAB-NAS\\Han-Workspace\\20250815-SwimUpUsingLatest\\using4.mp4"  # 指定视频路径

        # self.video_path = r"test-videoes\\using2_1.mp4"  # 指定视频路径
        # self.video_path = r"test-videoes\\using2_2.mp4"  # 指定视频路径
        # self.video_path = r"test-videoes\\using2_2_using.mp4"  # 指定视频路径
        self.video_path = r"test-videoes\\using2_using_latest.mp4"  # 指定视频路径

        self.current_frame = None
        self.cap = None
        self.last_label_size = QSize(0, 0)
        self.current_frame_position = 0
        
        # 添加检测线程引用
        self.detection_thread = None
        self.detection_enabled = False  # 标记检测是否启用
        self._pipeline_active = False  # 管线是否已就绪（首帧产出后为True）
        
        # 连接信号
        self.frame_ready.connect(self.update_frame)
        self.error_occurred.connect(self.handle_error)
        
        # 检查视频文件是否存在
        if not os.path.exists(self.video_path):
            print(f"警告: 找不到视频文件 {self.video_path}")
            # 尝试其他路径
            alternative_paths = [
                "test-videoes/swimUpUsing_5min.mp4",
                "../test-videoes/20241105_115725389.mp4",
                "./test-videoes/20241105_115725389.mp4"
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    self.video_path = path
                    print(f"找到视频文件: {self.video_path}")
                    break
            else:
                print("未找到指定视频文件，将创建测试视频")
                self.video_path = self.create_test_video()
    
    def set_detection_thread(self, detection_thread):
        """设置检测线程"""
        self.detection_thread = detection_thread
    
    def enable_detection(self, enabled, pipeline_active=False):
        """启用或禁用检测"""
        self.mutex.lock()
        self.detection_enabled = enabled
        self._pipeline_active = pipeline_active
        self.mutex.unlock()
    
    def create_test_video(self):
        """创建一个简单的测试视频"""
        test_video_path = "test_video.mp4"
        
        # 检查是否已经存在测试视频
        if os.path.exists(test_video_path):
            return test_video_path
        
        # 创建测试视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
        
        # 生成100帧测试画面
        for i in range(100):
            # 创建一个渐变的灰度图像
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            gray_value = int((i / 100) * 255)
            frame[:, :] = (gray_value, gray_value, gray_value)
            
            # 添加一些移动的圆形模拟精子
            for j in range(5):
                x = int(320 + 200 * np.cos(i * 0.1 + j))
                y = int(240 + 150 * np.sin(i * 0.1 + j))
                cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)
            
            # 添加帧号文本
            cv2.putText(frame, f"Frame: {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"创建测试视频: {test_video_path}")
        return test_video_path
    
    # 在 VideoTestThread 的 run 方法中，确保每一帧都发送，不跳帧
    def run(self):
        """线程主函数"""
        self.running = True
        
        # 打开视频文件
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error_occurred.emit(f"无法打开视频文件: {self.video_path}")
                return
            
            # 获取视频属性
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            delay = int(1000 / fps) if fps > 0 else 33  # 默认30fps
            
            print(f"视频信息: FPS={fps}, 总帧数={total_frames}")
            
            # 发送FPS信息
            self.fps_changed.emit(fps)
            
            # 帧计数器
            frame_count = 0
            
            while self.running:
                # 检查是否暂停
                self.mutex.lock()
                paused = self.paused
                detection_enabled = self.detection_enabled
                pipeline_active = self._pipeline_active
                self.mutex.unlock()
                
                if paused:
                    self.msleep(50)  # 降低CPU使用率
                    continue
                
                # 读取视频帧
                ret, frame = self.cap.read()
                if not ret:
                    # 视频结束，循环播放
                    print("视频播放结束，重新开始")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_position = 0
                    frame_count = 0
                    continue
                
                frame_count += 1
                
                # 更新当前帧位置
                self.current_frame_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # 保存当前帧（保持原始BGR格式）
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.mutex.unlock()
                
                # 每一帧都发送，不跳帧
                # 根据检测状态决定如何处理帧
                if self.detection_thread and detection_enabled:
                    # 发送到检测线程
                    self.new_frame.emit(frame)
                    # 管线未就绪时，同时显示原始帧（防止画面空白闪烁）
                    if not pipeline_active:
                        self.send_frame(frame)
                else:
                    # 没有启用检测，直接显示原始帧
                    self.send_frame(frame)
                
                # 控制帧率
                self.msleep(delay)
            
            # 释放资源
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
        except Exception as e:
            self.error_occurred.emit(f"视频处理出错: {str(e)}")
    
    def send_frame(self, frame):
        """将OpenCV帧转换为QPixmap并发送信号"""
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.frame_ready.emit(pixmap)
        except Exception as e:
            print(f"发送帧时出错: {e}")
    
    def stop(self):
        """停止线程"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        
        # 确保资源释放
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def pause(self):
        """暂停视频播放"""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        print("视频已暂停")
    
    def resume(self):
        """恢复视频播放"""
        self.mutex.lock()
        self.paused = False
        self.mutex.unlock()
        print("视频已恢复")
    
    def change_output_label(self, new_label):
        """切换输出标签"""
        self.mutex.lock()
        self.output_label = new_label
        self.mutex.unlock()
        
        # 如果有当前帧，立即在新标签上显示
        if self.current_frame is not None and new_label is not None:
            self.send_frame(self.current_frame)
    
    def update_frame(self, pixmap):
        """更新视频帧"""
        if self.output_label and not pixmap.isNull():
            try:
                # 获取标签大小
                label_size = self.output_label.size()
                
                # 只有当标签大小有效时才进行缩放
                if label_size.width() > 10 and label_size.height() > 10:
                    # 根据标签大小缩放图像，保持长宽比
                    scaled_pixmap = pixmap.scaled(
                        label_size,
                        aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                        transformMode=Qt.TransformationMode.SmoothTransformation
                    )
                    
                    # 更新标签
                    self.output_label.setPixmap(scaled_pixmap)
                    
                    # 保存当前标签大小，用于优化
                    self.last_label_size = label_size
                else:
                    # 标签大小无效，使用原始大小
                    self.output_label.setPixmap(pixmap)
            except Exception as e:
                print(f"更新帧时出错: {e}")
    
    def handle_error(self, error_msg):
        """处理错误"""
        print(f"视频线程错误: {error_msg}")