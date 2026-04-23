"""
Mini-map stitcher — PySide6 + OpenCV (SIFT / FLANN) + YOLO Detection
使用 ultralytics 库加载 ONNX 模型（完全按照用户的代码方式）
Windows only.

Requirements:
    pip install PySide6 opencv-contrib-python numpy ultralytics
"""

import sys
import traceback
print("Importing modules...")
from PySide6.QtWidgets import QApplication, QMessageBox
from ui.main_window import MainWindow


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Starting main...")
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        print("Creating MainWindow...")
        win = MainWindow()
        win.show()
        print("Executing app...")
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        print(f"Startup error: {e}")
        traceback.print_exc()
        if not QApplication.instance():
            app = QApplication(sys.argv)
        QMessageBox.critical(None, "Startup Error", f"An error occurred during startup:\n{e}\n\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
