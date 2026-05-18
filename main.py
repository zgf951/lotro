import sys
import logging
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox,
                               QFileDialog, QMessageBox, QGroupBox, QSplitter,
                               QCheckBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QGuiApplication
import numpy as np
import cv2

from window_manager import WindowManager
from capture import ScreenCapture
from improved_stitcher import ImprovedMiniMapStitcher as MiniMapStitcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegionSelector(QWidget):
    """全屏区域选择器，支持鼠标拖拽选取区域"""
    region_selected = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.start_pos = None
        self.current_pos = None
        self.selected_rect = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def show_selector(self):
        """显示选择器"""
        screens = QGuiApplication.screens()
        if screens:
            total_geometry = screens[0].availableGeometry()
            for screen in screens[1:]:
                total_geometry = total_geometry.united(screen.availableGeometry())
            self.setGeometry(total_geometry)
        self.show()
        self.activateWindow()
        self.raise_()

    def paintEvent(self, event):
        """绘制选择框"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self.start_pos and self.current_pos:
            rect = QRect(self.start_pos, self.current_pos).normalized()

            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(rect, QColor(0, 0, 0, 0))
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            pen = QPen(QColor(0, 150, 255), 2)
            painter.setPen(pen)
            painter.drawRect(rect)

            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawLine(rect.topLeft(), rect.topRight())
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())

            info_text = f"位置: ({rect.x()}, {rect.y()})  大小: {rect.width()} x {rect.height()}"
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)

            text_rect = rect.adjusted(0, rect.height() + 5, 0, 30)
            if text_rect.bottom() > self.height():
                text_rect.moveTop(rect.top() - 30)
            if text_rect.right() > self.width():
                text_rect.moveRight(rect.right())

            painter.fillRect(text_rect, QColor(0, 0, 0, 180))
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, info_text)

        hint_text = "拖拽鼠标选择区域，按 ESC 取消"
        painter.setPen(Qt.GlobalColor.white)
        font = QFont()
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(10, 10, -10, -10), 
                        Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, 
                        hint_text)

        painter.end()

    def mousePressEvent(self, event):
        """鼠标按下"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.position().toPoint()
            self.current_pos = event.position().toPoint()
            self.update()

    def mouseMoveEvent(self, event):
        """鼠标移动"""
        if self.start_pos:
            self.current_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        """鼠标释放"""
        if event.button() == Qt.MouseButton.LeftButton and self.start_pos and self.current_pos:
            rect = QRect(self.start_pos, self.current_pos).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.selected_rect = rect
                self.region_selected.emit(rect.x(), rect.y(), rect.width(), rect.height())
            self.hide()
            self.start_pos = None
            self.current_pos = None

    def keyPressEvent(self, event):
        """按键事件"""
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.start_pos = None
            self.current_pos = None


class CaptureWorker(QThread):
    """捕获工作线程"""
    frame_captured = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, hwnd: int, capture_region: tuple, capture_interval_ms: int = 200):
        super().__init__()
        self.hwnd = hwnd
        self.capture_region = capture_region
        self.capture_interval_ms = capture_interval_ms
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                x, y, w, h = self.capture_region
                frame = ScreenCapture.capture_window(self.hwnd, x, y, w, h)

                if frame is not None:
                    self.frame_captured.emit(frame)
                else:
                    self.error.emit("捕获失败")

                self.msleep(self.capture_interval_ms)

            except Exception as e:
                self.error.emit(str(e))
                self.msleep(500)

    def stop(self):
        self._running = False
        self.wait()


class MapViewWidget(QLabel):
    """地图显示组件"""

    def __init__(self):
        super().__init__()
        self.map_image: np.ndarray = None
        self.scale = 1.0
        self.offset = np.array([0, 0], dtype=np.float32)

        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #1a1a2e;")
        self.setText("请开始拼接...")
        self.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        self.setFont(font)

    def set_map(self, img: np.ndarray):
        """设置地图图像"""
        self.map_image = img
        self.update()

    def wheelEvent(self, event):
        """滚轮缩放"""
        if self.map_image is not None:
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 0.9
            self.scale = max(0.1, min(5.0, self.scale * factor))
            self.update()

    def paintEvent(self, event):
        """绘制地图"""
        super().paintEvent(event)

        if self.map_image is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 转换图像格式
        img_rgb = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        bytes_per_line = 3 * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 计算缩放后的尺寸
        scaled_w = int(w * self.scale)
        scaled_h = int(h * self.scale)

        # 居中显示
        x = (self.width() - scaled_w) // 2
        y = (self.height() - scaled_h) // 2

        painter.drawPixmap(x, y, QPixmap.fromImage(q_img).scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        painter.end()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.stitcher = MiniMapStitcher()
        self.capture_worker: CaptureWorker = None
        self.capture_region = (0, 0, 200, 200)
        self.is_stitching = False
        self.capture_interval_ms = 200

        self.window_manager = WindowManager()

        self.region_selector = RegionSelector()
        self.region_selector.region_selected.connect(self.on_region_selected)

        self.setup_ui()
        self.refresh_windows()

        self.window_combo.currentIndexChanged.connect(self.on_window_changed)

    def setup_ui(self):
        """设置UI"""
        self.setWindowTitle("网页/游戏地图拼接工具 (改进版)")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        # 右侧地图显示
        self.map_view = MapViewWidget()
        main_layout.addWidget(self.map_view, 3)

    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 窗口选择
        window_group = QGroupBox("窗口选择")
        window_layout = QVBoxLayout(window_group)

        self.window_combo = QComboBox()
        window_layout.addWidget(self.window_combo)

        refresh_btn = QPushButton("刷新窗口列表")
        refresh_btn.clicked.connect(self.refresh_windows)
        window_layout.addWidget(refresh_btn)

        layout.addWidget(window_group)

        # 区域选择
        region_group = QGroupBox("捕获区域")
        region_layout = QVBoxLayout(region_group)

        self.region_label = QLabel("未选择区域")
        self.region_label.setStyleSheet("padding: 10px; background-color: #2d2d44; border-radius: 5px;")
        self.region_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        region_layout.addWidget(self.region_label)

        select_region_btn = QPushButton("🖱️ 选择区域")
        select_region_btn.clicked.connect(self.select_region)
        select_region_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        region_layout.addWidget(select_region_btn)

        preview_btn = QPushButton("预览区域")
        preview_btn.clicked.connect(self.preview_region)
        region_layout.addWidget(preview_btn)

        layout.addWidget(region_group)

        # 控制按钮
        control_group = QGroupBox("拼接控制")
        control_layout = QVBoxLayout(control_group)

        # 捕获间隔
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("捕获间隔(ms):"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(50, 2000)
        self.interval_spin.setValue(self.capture_interval_ms)
        self.interval_spin.setSingleStep(50)
        interval_layout.addWidget(self.interval_spin)
        control_layout.addLayout(interval_layout)

        self.start_btn = QPushButton("开始拼接")
        self.start_btn.clicked.connect(self.toggle_stitching)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        control_layout.addWidget(self.start_btn)

        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_stitching)
        control_layout.addWidget(reset_btn)

        layout.addWidget(control_group)

        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)

        save_btn = QPushButton("保存地图")
        save_btn.clicked.connect(self.save_map)
        file_layout.addWidget(save_btn)

        load_btn = QPushButton("加载地图")
        load_btn.clicked.connect(self.load_map)
        file_layout.addWidget(load_btn)

        layout.addWidget(file_group)

        layout.addStretch()

        return panel

    def refresh_windows(self):
        """刷新窗口列表"""
        self.window_combo.clear()
        windows = WindowManager.list_windows()

        for hwnd, title in windows:
            display_title = f"{title} (hwnd: {hwnd})"
            self.window_combo.addItem(display_title, hwnd)

    def on_window_changed(self, index: int):
        """窗口选择变化时自动绑定/解绑"""
        if index < 0:
            self.window_manager.unbind_window()
            return

        hwnd = self.window_combo.itemData(index)
        if hwnd is not None:
            self.window_manager.bind_window(hwnd)

    def get_bound_hwnd(self) -> int:
        """获取当前绑定的窗口句柄"""
        return self.window_manager.bound_hwnd

    def select_region(self):
        """打开区域选择器"""
        hwnd = self.get_bound_hwnd()
        if hwnd is None:
            QMessageBox.warning(self, "警告", "请先选择一个窗口")
            return
        self.region_selector.show_selector()

    def on_region_selected(self, screen_x: int, screen_y: int, w: int, h: int):
        """区域选择完成回调，将屏幕坐标转换为窗口相对坐标"""
        hwnd = self.get_bound_hwnd()
        if hwnd is None:
            return

        client_pos = self.window_manager.screen_to_client(screen_x, screen_y)
        if client_pos is None:
            QMessageBox.warning(self, "警告", "坐标转换失败")
            return

        rel_x, rel_y = client_pos
        self.capture_region = (rel_x, rel_y, w, h)
        self.region_label.setText(
            f"窗口相对坐标\n"
            f"位置: ({rel_x}, {rel_y})\n"
            f"大小: {w} x {h}"
        )

    def preview_region(self):
        """预览捕获区域"""
        hwnd = self.get_bound_hwnd()
        if hwnd is None:
            QMessageBox.warning(self, "警告", "请先选择一个窗口")
            return

        x, y, w, h = self.capture_region

        frame = ScreenCapture.capture_window(hwnd, x, y, w, h)

        if frame is not None:
            self.map_view.set_map(frame)
        else:
            QMessageBox.warning(self, "警告", "预览失败")

    def toggle_stitching(self):
        """切换拼接状态"""
        if not self.is_stitching:
            self.start_stitching()
        else:
            self.stop_stitching()

    def start_stitching(self):
        """开始拼接"""
        hwnd = self.get_bound_hwnd()
        if hwnd is None:
            QMessageBox.warning(self, "警告", "请先选择一个窗口")
            return

        self.capture_interval_ms = int(self.interval_spin.value())
        self.is_stitching = True
        self.start_btn.setText("停止拼接")
        self.start_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")

        self.capture_worker = CaptureWorker(hwnd, self.capture_region, self.capture_interval_ms)
        self.capture_worker.frame_captured.connect(self.on_frame_captured)
        self.capture_worker.error.connect(self.on_capture_error)
        self.capture_worker.start()

    def stop_stitching(self):
        """停止拼接"""
        if self.capture_worker:
            self.capture_worker.stop()
            self.capture_worker = None

        self.is_stitching = False
        self.start_btn.setText("开始拼接")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")

    def on_frame_captured(self, frame: np.ndarray):
        """接收到新帧"""
        self.stitcher.add_frame(frame)

        if self.stitcher.stitched_map is not None:
            self.map_view.set_map(self.stitcher.stitched_map)

    def on_capture_error(self, error_msg: str):
        """捕获错误"""
        logger.error(error_msg)

    def reset_stitching(self):
        """重置拼接"""
        self.stop_stitching()
        self.stitcher.reset()
        self.map_view.map_image = None
        self.map_view.setText("请开始拼接...")
        self.map_view.update()

    def save_map(self):
        """保存地图"""
        if self.stitcher.stitched_map is None:
            QMessageBox.warning(self, "警告", "没有可保存的地图")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存地图", "", "PNG文件 (*.png);;JPEG文件 (*.jpg)"
        )

        if filepath:
            self.stitcher.save_map(filepath)
            QMessageBox.information(self, "成功", "地图保存成功")

    def load_map(self):
        """加载地图"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载地图", "", "图像文件 (*.png *.jpg *.jpeg)"
        )

        if filepath:
            if self.stitcher.load_map(filepath):
                self.map_view.set_map(self.stitcher.stitched_map)
                QMessageBox.information(self, "成功", "地图加载成功")
            else:
                QMessageBox.warning(self, "错误", "地图加载失败")

    def closeEvent(self, event):
        """关闭事件"""
        self.stop_stitching()
        self.window_manager.unbind_window()
        event.accept()


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
