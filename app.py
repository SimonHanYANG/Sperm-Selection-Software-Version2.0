import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QMessageBox, 
                             QSizePolicy, QVBoxLayout, QHBoxLayout, QWidget,
                             QSplitter, QGridLayout)
from PyQt6 import uic
from PyQt6.QtCore import QDir, Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QGuiApplication, QFont
from video_thread import VideoTestThread
from detection_thread import DetectionThread
from segment_thread import SegmentThread
from camera_thread import CameraThread

class FullScreenVideoLabel(QLabel):
    """用于全屏显示视频的自定义标签"""
    closed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        screen = QGuiApplication.primaryScreen()
        if screen:
            self.screen_size = screen.size()
            self.setGeometry(0, 0, self.screen_size.width(), self.screen_size.height())
        
    def mouseDoubleClickEvent(self, event):
        """双击退出全屏"""
        self.exit_fullscreen()
        
    def keyPressEvent(self, event):
        """按ESC退出全屏"""
        if event.key() == Qt.Key.Key_Escape:
            self.exit_fullscreen()
            
    def exit_fullscreen(self):
        """退出全屏"""
        self.closed.emit()
        self.close()

class StatsLabel(QLabel):
    """用于显示统计信息的标签"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: #00FF00;
                padding: 8px;
                border-radius: 5px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        self.setWordWrap(True)
        self.setMinimumHeight(90)
        self.setMaximumHeight(120)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置当前工作目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        QDir.setCurrent(current_dir)
        
        # 资源文件夹路径
        self.resource_path = os.path.join(current_dir, "resource-files")
        
        # 全屏视频标签
        self.fullscreen_label = None
        
        # 录制状态
        self.is_recording = False
        
        # 统计信息
        self.stats = {
            'video_fps': 0.0,
            'camera_fps': 0.0,
            'detection_fps': 0.0,
            'detection_buffer': 0,
            'detection_buffer_max': 100,
            'segment_fps': 0.0,
            'segment_buffer': 0,
            'segment_buffer_max': 50,
            'detection_count': 0,
            'detection_time': 0.0,
            'segment_time': 0.0,
            'total_time': 0.0,
            'active_tracks': 0
        }
        
        # 相机线程
        self.camera_thread = None
        self.camera_enabled = False
        
        # 视频线程
        self.video_thread = None
        self.video_enabled = True
        
        # 检测线程
        self.detection_thread = None
        self.detection_enabled = False
        
        # 分割线程
        self.segment_thread = None
        self.segment_enabled = False
        
        # 记录切换前的检测和分割状态
        self.was_detecting = False
        self.was_segmenting = False
        
        # 统计更新定时器
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats_display)
        self.stats_timer.start(100)  # 每100ms更新一次
        
        # 加载UI文件
        try:
            uic.loadUi("interfacedesign.ui", self)
            print("UI界面加载成功!")
            
            # 设置窗口属性
            self.setup_window()
            
            # 配置视频标签
            self.configure_video_label()
            
            # 手动设置图标
            self.manually_set_icons()
            
            # 添加统计显示标签 (创建但不添加到布局)
            self.add_stats_display()

            # 重新组织实时分析标签页的布局
            self.reorganize_realtime_analysis_layout()
            
            # 设置视频标签双击事件
            if hasattr(self, 'label_4'):
                self.label_4.mouseDoubleClickEvent = self.video_label_double_clicked
            
            # 启动视频线程
            self.start_video_thread()
            
            # 初始化检测线程
            self.initialize_detection_thread()
            
            # 初始化分割线程
            self.initialize_segment_thread()
            
            # 预初始化相机线程
            self.initialize_camera_thread()
            
        except Exception as e:
            print(f"加载UI界面失败: {e}")
            sys.exit(1)
        
        self.setup_connections()
    
    def on_clear_clicked(self):
        """清除按钮被点击 - 切换相机/视频模式"""
        print("清除按钮被点击 - 开始切换模式")
        
        # 记录当前检测和分割的状态
        self.was_detecting = self.detection_enabled
        self.was_segmenting = self.segment_enabled
        
        # 停止检测和分割线程
        self._stop_and_cleanup_detection_segmentation()
        
        # 等待线程完全停止后再切换
        QTimer.singleShot(500, self._perform_mode_switch)
    
    def _stop_and_cleanup_detection_segmentation(self):
        """停止并清理检测和分割线程"""
        print("正在停止并清理检测和分割线程...")
        
        # 停止分割线程
        if self.segment_thread:
            if self.segment_thread.isRunning():
                self.segment_thread.stop_segmentation()
                self.segment_thread.wait()
            self.segment_thread.deleteLater()
            self.segment_thread = None
            self.segment_enabled = False
            print("分割线程已停止并清理")
        
        # 停止检测线程
        if self.detection_thread:
            if self.detection_thread.isRunning():
                self.detection_thread.stop_detection()
                self.detection_thread.wait()
            
            # 断开所有连接
            try:
                if self.video_thread:
                    self.video_thread.new_frame.disconnect(self.detection_thread.add_frame)
                    self.video_thread.fps_changed.disconnect(self.detection_thread.update_fps)
            except TypeError:
                pass
            
            try:
                if self.camera_thread:
                    self.camera_thread.new_image_signal.disconnect(self.detection_thread.add_frame)
            except TypeError:
                pass
                
            self.detection_thread.deleteLater()
            self.detection_thread = None
            self.detection_enabled = False
            print("检测线程已停止并清理")
        
        # 通知视频线程禁用检测
        if self.video_thread:
            self.video_thread.enable_detection(False, pipeline_active=False)
            self.video_thread.set_detection_thread(None)
    
    def _perform_mode_switch(self):
        """执行模式切换"""
        if self.camera_enabled:
            # 切换到视频模式
            self._switch_to_video_mode()
        else:
            # 切换到相机模式
            self._switch_to_camera_mode()
    
    def _switch_to_video_mode(self):
        """切换到视频模式"""
        print("正在切换到视频模式...")
        
        # 停止相机线程
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_enabled = False
        
        # 重启视频线程
        if self.video_thread is None or not self.video_thread.isRunning():
            self.start_video_thread()
        else:
            # 确保视频线程恢复播放
            self.video_thread.resume()
        
        self.video_enabled = True
        
        # 重新初始化检测和分割线程
        QTimer.singleShot(100, self._reinitialize_detection_and_segmentation)
        
        QMessageBox.information(self, "模式切换", "已切换到视频模式")
        print("已成功切换到视频模式")
    
    def _switch_to_camera_mode(self):
        """切换到相机模式"""
        print("正在切换到相机模式...")
        
        # 确保相机线程已初始化
        if self.camera_thread is None:
            self.initialize_camera_thread()
            
        if self.camera_thread is None:
            QMessageBox.warning(self, "错误", "无法初始化相机")
            return
        
        # 停止视频线程
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_enabled = False
        
        # 启动相机线程
        if not self.camera_thread.isRunning():
            if hasattr(self, 'label_4'):
                self.camera_thread.set_output_label(self.label_4)
            
            self.camera_thread.start()
            self.camera_enabled = True
        
        # 重新初始化检测和分割线程
        QTimer.singleShot(100, self._reinitialize_detection_and_segmentation)
        
        QMessageBox.information(self, "模式切换", "已切换到相机模式")
        print("已成功切换到相机模式")
    
    def _reinitialize_detection_and_segmentation(self):
        """重新初始化检测和分割线程"""
        print("正在重新初始化检测和分割线程...")
        
        # 重新初始化检测线程
        self.initialize_detection_thread()
        
        # 重新初始化分割线程
        self.initialize_segment_thread()
        
        # 根据输入源建立连接
        if self.detection_thread:
            if self.camera_enabled and self.camera_thread:
                # 相机模式
                self.camera_thread.new_image_signal.connect(self.detection_thread.add_frame)
                print("已连接相机到检测线程")
            elif self.video_enabled and self.video_thread:
                # 视频模式
                self.video_thread.set_detection_thread(self.detection_thread)
                self.video_thread.new_frame.connect(self.detection_thread.add_frame)
                self.video_thread.fps_changed.connect(self.detection_thread.update_fps)
                print("已连接视频到检测线程")
        
        # 如果切换前正在运行，则自动恢复
        if self.was_detecting or self.was_segmenting:
            QTimer.singleShot(500, self._auto_resume_detection_segmentation)
    
    def _auto_resume_detection_segmentation(self):
        """自动恢复检测和分割"""
        print("自动恢复检测和分割...")
        
        if self.was_detecting and self.detection_thread:
            self.detection_thread.start_detection()
            self.detection_enabled = True

            # 通知视频线程启用检测（管线未就绪，同时显示原始帧）
            if self.video_enabled and self.video_thread:
                self.video_thread.enable_detection(True, pipeline_active=False)
            
            print("已恢复检测")
        
        if self.was_segmenting and self.segment_thread:
            self.segment_thread.start_segmentation()
            self.segment_enabled = True
            print("已恢复分割")
    
    def initialize_detection_thread(self):
        """初始化检测线程"""
        try:
            # 创建检测线程
            self.detection_thread = DetectionThread()
            
            # 初始化TensorRT引擎
            if self.detection_thread.initialize_engine():
                print("检测线程初始化成功")
                
                # 设置输出标签
                if hasattr(self, 'label_4'):
                    self.detection_thread.set_output_label(self.label_4)
                
                # 设置是否显示轨迹 (默认不显示)
                self.detection_thread.set_show_tracking(self.is_recording)
                
                # 连接信号
                self.detection_thread.stats_updated.connect(self.update_detection_stats)
                self.detection_thread.pipeline_ready.connect(self._on_pipeline_ready)

                # 连接视频线程和检测线程
                if self.video_thread and self.video_enabled:
                    self.video_thread.set_detection_thread(self.detection_thread)
                    self.video_thread.new_frame.connect(self.detection_thread.add_frame)
                    self.video_thread.fps_changed.connect(self.detection_thread.update_fps)
            else:
                print("检测线程初始化失败")
                self.detection_thread = None
                
        except Exception as e:
            print(f"创建检测线程时出错: {e}")
            self.detection_thread = None
    
    def initialize_segment_thread(self):
        """初始化分割线程"""
        try:
            # 创建分割线程
            self.segment_thread = SegmentThread()
            
            # 初始化TensorRT引擎
            if self.segment_thread.initialize_engine():
                print("分割线程初始化成功")
                
                # 设置输出标签
                if hasattr(self, 'label_4'):
                    self.segment_thread.set_output_label(self.label_4)
                
                # 设置是否显示轨迹 (默认不显示)
                self.segment_thread.set_show_tracking(self.is_recording)
                
                # 设置像素到微米的转换系数
                self.segment_thread.set_pixel_to_micron(0.7)  # 根据实际情况调整
                
                # 连接信号
                self.segment_thread.stats_updated.connect(self.update_segment_stats)
                
                # 连接检测线程和分割线程
                if self.detection_thread:
                    self.detection_thread.set_segment_thread(self.segment_thread)
                    self.segment_thread.set_detection_thread(self.detection_thread)
                    if hasattr(self.detection_thread, 'tracker') and self.detection_thread.tracker:
                        self.segment_thread.set_tracker(self.detection_thread.tracker)
                
            else:
                print("分割线程初始化失败")
                self.segment_thread = None
                
        except Exception as e:
            print(f"创建分割线程时出错: {e}")
            self.segment_thread = None

    def add_stats_display(self):
        """添加统计信息显示"""
        # 创建统计标签，但不添加到任何布局中
        self.stats_label = StatsLabel()
        # 设置自适应长度
        self.stats_label.setMinimumHeight(100)
        self.stats_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    # 修改 update_stats_display 方法
    def update_stats_display(self):
        """更新统计信息显示"""
        if hasattr(self, 'stats_label'):
            # 获取分级统计
            grade_stats = self._get_grade_statistics()

            # 更紧凑的统计信息格式
            stats_text = f"""FPS: 视频 {self.stats.get('video_fps', 0):.1f} | 相机 {self.stats.get('camera_fps', 0):.1f} | 检测 {self.stats.get('detection_fps', 0):.1f} | 分割 {self.stats.get('segment_fps', 0):.1f}

缓冲: 检测 {self.stats.get('detection_buffer', 0)}/{self.stats.get('detection_buffer_max', 100)} | 分割 {self.stats.get('segment_buffer', 0)}/{self.stats.get('segment_buffer_max', 50)}

检测数: {self.stats.get('detection_count', 0)} | 跟踪数: {self.stats.get('active_tracks', 0)}

分级: G1:{grade_stats[1]} G2:{grade_stats[2]} G3:{grade_stats[3]} G4:{grade_stats[4]} G5:{grade_stats[5]} G6:{grade_stats[6]}

候选池: {self.stats.get('candidate_pool_size', 0)} | 本轮调度: {self.stats.get('scheduled_sperm', 0)} | 平均测量: {self.stats.get('avg_measurement_count', 0):.1f} | 最高复合分: {self.stats.get('composite_score_top', 0):.2f}

时间(ms): 检测 {self.stats.get('detection_time', 0):.1f} | 分割 {self.stats.get('segment_time', 0):.1f} | 总计 {self.stats.get('total_time', 0):.1f}

数据库: {self.stats.get('db_path', '')}"""

            self.stats_label.setText(stats_text.strip())

    def _get_grade_statistics(self):
        """获取分级统计信息"""
        grade_stats = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, -1: 0}

        # 从检测线程获取跟踪器
        if self.detection_thread and hasattr(self.detection_thread, 'tracker') and self.detection_thread.tracker:
            tracks = self.detection_thread.tracker.get_active_tracks()
            for track in tracks:
                if hasattr(track, 'grade'):
                    grade = track.grade
                    if grade in grade_stats:
                        grade_stats[grade] += 1

        return grade_stats

    def initialize_camera_thread(self):
        """初始化相机线程"""
        try:
            if hasattr(self, 'label_4'):
                self.camera_thread = CameraThread(self.label_4)
                
                # 连接信号
                self.camera_thread.error_signal.connect(self.handle_camera_error)
                self.camera_thread.fps_signal.connect(self.update_camera_fps)
                
                print("相机线程已初始化")
        except Exception as e:
            print(f"初始化相机线程失败: {e}")
            self.camera_thread = None
    
    def update_camera_fps(self, fps):
        """更新相机FPS"""
        self.stats['camera_fps'] = fps
    
    def handle_camera_error(self, error_message):
        """处理相机错误"""
        QMessageBox.warning(self, "相机错误", error_message)
        # 相机错误时自动切换回视频模式
        if self.camera_enabled:
            self._switch_to_video_mode()
    
    def update_detection_stats(self, stats):
        """更新检测统计信息"""
        for key, value in stats.items():
            self.stats[key] = value
    
    def update_segment_stats(self, stats):
        """更新分割统计信息"""
        for key, value in stats.items():
            self.stats[key] = value
    
    def reorganize_realtime_analysis_layout(self):
        """重新组织实时分析标签页的布局"""
        # 查找实时分析标签页
        if hasattr(self, 'tab_2'):
            # 创建主布局
            main_layout = QHBoxLayout()
            
            # 创建左侧视频区域
            left_widget = QWidget()
            left_layout = QVBoxLayout()
            
            # 添加logo
            if hasattr(self, 'label_22'):
                self.label_22.setMaximumHeight(50)
                self.label_22.setMinimumHeight(50)
                self.label_22.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                self.label_22.setScaledContents(False)
                self.label_22.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                
                if not self.label_22.pixmap().isNull():
                    original_pixmap = self.label_22.pixmap()
                    scaled_pixmap = original_pixmap.scaled(
                        original_pixmap.width() * 50 / original_pixmap.height(), 
                        50, 
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.label_22.setPixmap(scaled_pixmap)
                
                left_layout.addWidget(self.label_22)
            
            # 添加视频标签
            if hasattr(self, 'label_4'):
                self.label_4.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                left_layout.addWidget(self.label_4)
            
            left_widget.setLayout(left_layout)
            
            # 创建右侧控制区域
            right_widget = QWidget()
            right_widget.setMaximumWidth(400)
            right_widget.setMinimumWidth(350)
            right_layout = QVBoxLayout()
            
            # 按钮区域
            button_grid = QGridLayout()
            buttons = [
                ('pushButton', 0, 0),      # 开始
                ('pushButton_3', 0, 1),    # 停止
                ('pushButton_4', 1, 0),    # 录制
                ('pushButton_2', 1, 1),    # 清除
            ]
            
            for btn_name, row, col in buttons:
                if hasattr(self, btn_name):
                    btn = getattr(self, btn_name)
                    btn.setMinimumHeight(50)
                    button_grid.addWidget(btn, row, col)
            
            right_layout.addLayout(button_grid)
            
            # 选择状态区域
            if hasattr(self, 'label_52'):
                self.label_52.setMinimumHeight(250)
                self.label_52.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                right_layout.addWidget(self.label_52)
                
                self.reorganize_status_labels()
            
            # 参数区域
            if hasattr(self, 'label_51'):
                self.label_51.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                right_layout.addWidget(self.label_51)
                
                self.reorganize_parameter_controls()
            
            right_widget.setLayout(right_layout)
            
            # 使用分割器
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(left_widget)
            splitter.addWidget(right_widget)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)
            
            main_layout.addWidget(splitter)
            
            # 设置标签页的布局
            self.tab_2.setLayout(main_layout)
    
    def reorganize_status_labels(self):
        """重新组织状态标签的布局"""
        if hasattr(self, 'label_52'):
            status_layout = QVBoxLayout()
            status_layout.setContentsMargins(10, 10, 10, 10)
            
            # 状态项
            status_items = [
                ('label_11', 'label_8', 'pushButton_7'),
                ('label_12', 'label_9', 'pushButton_8'),
                ('label_13', 'label_41', 'pushButton_9'),
            ]
            
            for icon_label, text_label, button in status_items:
                h_layout = QHBoxLayout()
                
                if hasattr(self, icon_label):
                    icon = getattr(self, icon_label)
                    icon.setMaximumSize(60, 60)
                    icon.setScaledContents(True)
                    h_layout.addWidget(icon)
                
                if hasattr(self, text_label):
                    text = getattr(self, text_label)
                    text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                    h_layout.addWidget(text)
                
                if hasattr(self, button):
                    btn = getattr(self, button)
                    btn.setMaximumWidth(80)
                    h_layout.addWidget(btn)
                
                status_layout.addLayout(h_layout)
            
            # 清除label_52原有内容并设置新布局
            self.label_52.setText("")
            status_widget = QWidget()
            status_widget.setLayout(status_layout)
            
            # 将widget放入label_52
            label_layout = QVBoxLayout()
            label_layout.addWidget(status_widget)
            self.label_52.setLayout(label_layout)
    
    def reorganize_parameter_controls(self):
        """重新组织参数控件的布局"""
        if hasattr(self, 'label_51'):
            param_layout = QVBoxLayout()
            param_layout.setContentsMargins(10, 10, 10, 10)
            
            # 参数项
            param_items = [
                ('label_5', 'comboBox'),
                ('label_6', 'comboBox_2'),
                ('label_10', 'comboBox_3'),
            ]
            
            # 添加参数控件
            for label_name, combo_name in param_items:
                if hasattr(self, label_name) and hasattr(self, combo_name):
                    label = getattr(self, label_name)
                    combo = getattr(self, combo_name)
                    
                    param_layout.addWidget(label)
                    param_layout.addWidget(combo)
                    param_layout.addSpacing(5)  # 减小间距
            
            # 在最后一个下拉框后添加统计信息标签
            if hasattr(self, 'stats_label'):
                param_layout.addSpacing(10)
                # 移除之前可能已存在的父级关系
                if self.stats_label.parent():
                    self.stats_label.setParent(None)
                param_layout.addWidget(self.stats_label)
            
            # 添加设置按钮，确保它在统计标签下方
            if hasattr(self, 'pushButton_6'):
                param_layout.addSpacing(10)
                param_layout.addStretch()  # 弹性空间
                self.pushButton_6.setMinimumHeight(40)
                param_layout.addWidget(self.pushButton_6)
            
            # 清除label_51原有内容并设置新布局
            self.label_51.setText("")
            param_widget = QWidget()
            param_widget.setLayout(param_layout)
            
            # 将widget放入label_51
            label_layout = QVBoxLayout()
            label_layout.addWidget(param_widget)
            self.label_51.setLayout(label_layout)
    
    def setup_window(self):
        """设置窗口属性"""
        # 确保窗口有标准的最小化、最大化和关闭按钮
        self.setWindowFlags(Qt.WindowType.Window)
        
        # 设置最小窗口大小
        self.setMinimumSize(1200, 800)
        
        # 获取屏幕尺寸
        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_rect = screen.availableGeometry()
            window_width = self.width()
            window_height = self.height()
            
            # 如果窗口大小超过屏幕大小的90%，则最大化窗口
            if window_width > screen_rect.width() * 0.9 or window_height > screen_rect.height() * 0.9:
                self.showMaximized()
            else:
                # 居中显示窗口
                x = (screen_rect.width() - window_width) // 2
                y = (screen_rect.height() - window_height) // 2
                self.move(x, y)
    
    def configure_video_label(self):
        """配置视频标签"""
        if hasattr(self, 'label_4'):
            # 设置大小策略为自适应扩展
            self.label_4.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.label_4.setMinimumSize(640, 480)
            self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label_4.setStyleSheet("background-color: black;")
            self.label_4.setScaledContents(False)
    
    def video_label_double_clicked(self, event):
        """处理视频标签的双击事件"""
        if not self.fullscreen_label:
            self.enter_fullscreen()
    
    def enter_fullscreen(self):
        """进入全屏模式"""
        if self.fullscreen_label:
            return
            
        # 创建新的全屏标签
        self.fullscreen_label = FullScreenVideoLabel()
        
        # 连接关闭信号
        self.fullscreen_label.closed.connect(self.exit_fullscreen)
        
        # 更新输出
        if self.segment_enabled and self.segment_thread:
            # 分割线程输出到全屏
            self.segment_thread.set_output_label(self.fullscreen_label)
        elif self.detection_enabled and self.detection_thread:
            # 检测线程输出到全屏
            self.detection_thread.set_fullscreen_label(self.fullscreen_label)
        elif self.camera_enabled and self.camera_thread:
            # 相机输出到全屏
            self.camera_thread.change_output_label(self.fullscreen_label)
        else:
            # 视频输出到全屏
            if self.video_thread:
                self.video_thread.change_output_label(self.fullscreen_label)
        
        # 显示全屏
        self.fullscreen_label.showFullScreen()
        
        print("进入全屏模式")
    
    def exit_fullscreen(self):
        """退出全屏模式"""
        if not self.fullscreen_label:
            return
            
        print("退出全屏模式")
        
        # 恢复输出到原始标签
        if hasattr(self, 'label_4'):
            if self.segment_enabled and self.segment_thread:
                self.segment_thread.set_output_label(self.label_4)
            elif self.detection_enabled and self.detection_thread:
                self.detection_thread.set_fullscreen_label(None)
            elif self.camera_enabled and self.camera_thread:
                self.camera_thread.change_output_label(self.label_4)
            else:
                if self.video_thread:
                    self.video_thread.change_output_label(self.label_4)
        
        # 清理全屏标签
        self.fullscreen_label = None
    
    def start_video_thread(self):
        """启动视频测试线程"""
        if hasattr(self, 'label_4'):
            try:
                self.video_thread = VideoTestThread(self.label_4)
                
                # 连接FPS信号
                self.video_thread.fps_changed.connect(self.update_video_fps)
                
                self.video_thread.start()
                self.video_enabled = True
                print("视频线程已启动")
            except Exception as e:
                print(f"启动视频线程失败: {e}")
                QMessageBox.critical(self, "错误", f"无法启动视频线程: {str(e)}")
    
    def update_video_fps(self, fps):
        """更新视频FPS"""
        self.stats['video_fps'] = fps
    
    def get_resource_path(self, filename):
        """获取资源文件的完整路径"""
        full_path = os.path.join(self.resource_path, filename)
        if os.path.exists(full_path):
            return full_path
        else:
            print(f"警告: 文件不存在: {full_path}")
            return None
    
    def manually_set_icons(self):
        """手动设置图标"""
        # 设置窗口图标
        ico_path = self.get_resource_path("SpermICO.ico")
        if ico_path:
            self.setWindowIcon(QIcon(ico_path))
        
        # 设置TabWidget的图标
        if hasattr(self, 'tabWidget'):
            tab_icons = {
                0: "canshu 1.png",
                1: "fenxi 1.png",
                2: "zice1.png",
                3: "bangzhu 1.png",
            }
            
            for index, icon_name in tab_icons.items():
                if index < self.tabWidget.count():
                    icon_path = self.get_resource_path(icon_name)
                    if icon_path:
                        self.tabWidget.setTabIcon(index, QIcon(icon_path))
        
        # 设置按钮图标
        button_icons = {
            'pushButton': "kaishi 1.png",
            'pushButton_2': "qingling 1.png",
            'pushButton_3': "tingzhi 1.png",
            'pushButton_4': "shipin 1.png",
            'pushButton_10': "Read.png",
            'pushButton_11': "front.png",
            'pushButton_12': "next.png",
            'pushButton_13': "qingling 2.png",
        }
        
        for btn_name, icon_name in button_icons.items():
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                icon_path = self.get_resource_path(icon_name)
                if icon_path:
                    btn.setIcon(QIcon(icon_path))
        
        # 设置标签图片
        label_images = {
            # 'label_21': "logo1.png",
            # 'label_22': "logo1.png",
            'label_23': "logo2.png",
            'label_24': "rjmc.png",
            'label_25': "bbh.png",
            'label_26': "gsmc.png",
            'label_27': "lxfs.png",
            'label_11': "lv.png",
            'label_12': "qianlv.png",
            'label_13': "huang.png",
            'label_42': "logo1.png",
        }
        
        for label_name, image_name in label_images.items():
            if hasattr(self, label_name):
                label = getattr(self, label_name)
                image_path = self.get_resource_path(image_name)
                if image_path:
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        label.setPixmap(pixmap)
                        label.setScaledContents(True)
        
        # 设置工具栏动作图标
        action_icons = {
            'actionhelp': "bangzhu 1.png",
            'actiontest': "zice1.png",
        }
        
        for action_name, icon_name in action_icons.items():
            if hasattr(self, action_name):
                action = getattr(self, action_name)
                icon_path = self.get_resource_path(icon_name)
                if icon_path:
                    action.setIcon(QIcon(icon_path))
    
    def setup_connections(self):
        """设置信号和槽连接"""
        # 连接主要按钮的点击事件
        buttons = [
            ('pushButton', self.on_start_clicked),
            ('pushButton_2', self.on_clear_clicked),
            ('pushButton_3', self.on_stop_clicked),
            ('pushButton_4', self.on_record_clicked),
            ('pushButton_5', self.on_create_record_clicked),
            ('pushButton_6', self.on_setting_clicked),
            ('pushButton_7', self.on_show_level1_clicked),
            ('pushButton_8', self.on_show_level2_clicked),
            ('pushButton_9', self.on_show_level3_clicked),
            ('pushButton_10', self.on_read_clicked),
            ('pushButton_11', self.on_prev_frame_clicked),
            ('pushButton_12', self.on_next_frame_clicked),
            ('pushButton_13', self.on_clear_test_clicked),
        ]
        
        for btn_name, handler in buttons:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                btn.clicked.connect(handler)
        
        # 连接工具栏动作
        if hasattr(self, 'actionhelp'):
            self.actionhelp.triggered.connect(self.on_help_triggered)
        if hasattr(self, 'actiontest'):
            self.actiontest.triggered.connect(self.on_test_triggered)
    
    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        
        # 窗口大小变化时，视频帧会自动重新调整大小
        if self.video_thread and hasattr(self, 'label_4'):
            pass
    
    def closeEvent(self, event):
        """处理窗口关闭事件"""
        # 停止定时器
        self.stats_timer.stop()
        
        # 停止相机线程
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        # 停止分割线程
        if self.segment_thread and self.segment_thread.isRunning():
            self.segment_thread.stop_segmentation()
            self.segment_thread.wait()
        
        # 停止检测线程
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop_detection()
            self.detection_thread.wait()
        
        # 停止视频线程
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        # 关闭全屏显示
        if self.fullscreen_label:
            self.fullscreen_label.close()
        
        event.accept()
    
    # 按钮点击事件处理函数
    def on_start_clicked(self):
        """开始按钮被点击"""
        print("开始按钮被点击")
        
        if self.detection_thread:
            # 启动检测
            self.detection_thread.start_detection()
            self.detection_enabled = True
            
            # 通知视频线程启用检测（管线未就绪，同时显示原始帧）
            if self.video_enabled and self.video_thread:
                self.video_thread.enable_detection(True, pipeline_active=False)
        else:
            print("检测线程未初始化")
            
        if self.segment_thread:
            # 启动分割
            self.segment_thread.start_segmentation()
            self.segment_enabled = True
        else:
            print("分割线程未初始化")
    
    def on_stop_clicked(self):
        """停止按钮被点击"""
        print("停止按钮被点击")
        
        if self.detection_thread:
            self.detection_thread.pause_detection()
            self.detection_enabled = False
            
            # 通知视频线程禁用检测
            if self.video_enabled and self.video_thread:
                self.video_thread.enable_detection(False, pipeline_active=False)
        
        if self.segment_thread:
            self.segment_thread.pause_segmentation()
            self.segment_enabled = False

    def _on_pipeline_ready(self):
        """管线已就绪，停止视频线程的原始帧直接显示"""
        if self.video_thread:
            self.video_thread.enable_detection(True, pipeline_active=True)

    def on_record_clicked(self):
        """录制按钮被点击"""
        print("录制按钮被点击")
        # 切换录制状态
        self.is_recording = not self.is_recording
        
        # 更新检测和分割线程的轨迹显示状态
        if self.detection_thread:
            self.detection_thread.set_show_tracking(self.is_recording)
        
        if self.segment_thread:
            self.segment_thread.set_show_tracking(self.is_recording)
        
        # 提示用户
        status = "开启" if self.is_recording else "关闭"
        QMessageBox.information(self, "录制状态", f"已{status}轨迹显示")
        
    def on_create_record_clicked(self):
        print("建档按钮被点击")
    
    def on_setting_clicked(self):
        print("设置按钮被点击")
    
    def on_show_level1_clicked(self):
        print("显示第一级精子按钮被点击")
    
    def on_show_level2_clicked(self):
        print("显示第二级精子按钮被点击")
    
    def on_show_level3_clicked(self):
        print("显示第三级精子按钮被点击")
    
    def on_read_clicked(self):
        print("读取按钮被点击")
    
    def on_prev_frame_clicked(self):
        print("上一帧按钮被点击")
    
    def on_next_frame_clicked(self):
        print("下一帧按钮被点击")
    
    def on_clear_test_clicked(self):
        print("清空测试按钮被点击")
    
    def on_help_triggered(self):
        print("帮助动作被触发")
        self.tabWidget.setCurrentIndex(3)
    
    def on_test_triggered(self):
        print("功能自测动作被触发")
        self.tabWidget.setCurrentIndex(2)