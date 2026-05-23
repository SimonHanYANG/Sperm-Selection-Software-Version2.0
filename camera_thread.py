from PyQt6.QtCore import QThread, pyqtSignal
import pypylon.pylon as pylon
from pypylon import genicam
import numpy as np
import logging
import cv2
import time
from collections import deque

class CameraThread(QThread):
    new_image_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    fps_signal = pyqtSignal(float)  # 相机FPS信号
    
    def __init__(self, output_label=None):
        super().__init__()
        self.output_label = output_label
        self.running = False
        self.camera = None
        self.frame_rate = 30.0
        self.logger = logging.getLogger(__name__)
        self.current_frame = None
        
        # FPS计算
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()
        
    def set_output_label(self, label):
        """设置输出标签"""
        self.output_label = label
    
    def change_output_label(self, label):
        """改变输出标签"""
        self.output_label = label
        
        # 如果有当前帧，则立即显示在新标签上
        if self.current_frame is not None and self.output_label is not None:
            self.display_frame(self.current_frame)
    
    def run(self):
        self.running = True
        try:
            # 获取传输层工厂
            tl_factory = pylon.TlFactory.GetInstance()
            
            # 查找所有可用设备
            devices = tl_factory.EnumerateDevices()
            
            if not devices:
                self.error_signal.emit("No cameras found.")
                return
            
            # 创建和连接相机
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            print(f"Using device: {self.camera.GetDeviceInfo().GetModelName()}")
            
            # 打开相机
            self.camera.Open()
            
            # 配置相机设置
            self._configure_camera()
            
            # 开始抓取
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("Started grabbing images")
            
            while self.running and self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # 记录时间
                    current_time = time.time()
                    
                    # 复制数组以避免数据竞争
                    img_array = grab_result.Array.copy()
                    
                    # 转换为彩色图像以适应界面显示
                    if len(img_array.shape) == 2:  # 灰度图像
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    
                    # 保存当前帧
                    self.current_frame = img_array
                    
                    # 发送到处理线程
                    self.new_image_signal.emit(img_array)
                    
                    # 显示在标签上（如果没有检测线程处理）
                    self.display_frame(img_array)
                    
                    # 计算FPS
                    frame_time = current_time - self.last_time
                    self.frame_times.append(frame_time)
                    if len(self.frame_times) > 0:
                        avg_time = np.mean(self.frame_times)
                        self.fps = 1.0 / avg_time if avg_time > 0 else 0.0
                        self.fps_signal.emit(self.fps)
                    
                    self.last_time = current_time
                
                grab_result.Release()
                
        except genicam.GenericException as e:
            error_msg = f"GenICam exception: {e}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
        except Exception as e:
            error_msg = f"Camera error: {e}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
        finally:
            self._cleanup()
    
    def display_frame(self, frame):
        """在标签上显示帧"""
        if self.output_label is None:
            return
            
        try:
            # 将 NumPy 数组转换为 Qt 图像
            h, w, c = frame.shape
            bytes_per_line = c * w
            
            from PyQt6.QtGui import QImage, QPixmap
            from PyQt6.QtCore import Qt
            
            # 创建 QImage (BGR -> RGB)
            if c == 3:  # BGR
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:  # 灰度
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            # 创建适合标签的 QPixmap
            pixmap = QPixmap.fromImage(qt_image)
            
            # 根据标签尺寸调整图像大小（保持宽高比）
            label_size = self.output_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, 
                                          Qt.TransformationMode.SmoothTransformation)
            
            # 设置 pixmap
            self.output_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"Error displaying frame: {e}")
    
    def _configure_camera(self):
        """配置相机参数"""
        try:
            # 设置为连续采集
            if genicam.IsAvailable(self.camera.TriggerMode):
                self.camera.TriggerMode.SetValue("Off")
            
            # 心跳超时（适用于GigE相机）
            if (self.camera.GetDeviceInfo().GetDeviceClass() == "BaslerGigE" and 
                genicam.IsAvailable(self.camera.GevHeartbeatTimeout)):
                self.camera.GevHeartbeatTimeout.SetValue(1000)
            
            # 帧率设置
            if genicam.IsAvailable(self.camera.AcquisitionFrameRateEnable):
                self.camera.AcquisitionFrameRateEnable.SetValue(True)
                
                if genicam.IsAvailable(self.camera.AcquisitionFrameRateAbs):
                    self.camera.AcquisitionFrameRateAbs.SetValue(self.frame_rate)
                elif genicam.IsAvailable(self.camera.AcquisitionFrameRate):
                    self.camera.AcquisitionFrameRate.SetValue(self.frame_rate)
            
            # 像素格式
            if genicam.IsAvailable(self.camera.PixelFormat):
                self.camera.PixelFormat.SetValue("Mono8")
            
            # ROI设置 (1600x1200)
            if (genicam.IsAvailable(self.camera.Width) and 
                genicam.IsAvailable(self.camera.Height)):
                self.camera.OffsetX.SetValue(0)
                self.camera.OffsetY.SetValue(0)
                self.camera.Width.SetValue(1600)
                self.camera.Height.SetValue(1200)
                
        except genicam.GenericException as e:
            error_msg = f"Failed to configure camera: {e}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
            raise
    
    def _cleanup(self):
        """清理相机资源"""
        try:
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                    print("Stopped grabbing images")
                
                if self.camera.IsOpen():
                    print(f"Closing camera {self.camera.GetDeviceInfo().GetModelName()}")
                    self.camera.Close()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        """停止相机线程"""
        self.running = False
        self.wait(3000)  # 等待最多3秒以便安全关闭
    
    def get_fps(self):
        """获取当前FPS"""
        return self.fps