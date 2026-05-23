import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from app import MainWindow

if __name__ == "__main__":
    # 设置高DPI支持（PyQt6）
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    
    # PyQt6 默认支持高DPI，不需要额外设置
    
    window = MainWindow()
    window.show()
    
    # 打开程序后直接切换到"实时分析"界面
    if hasattr(window, 'tabWidget'):
        window.tabWidget.setCurrentIndex(1)
    
    sys.exit(app.exec())