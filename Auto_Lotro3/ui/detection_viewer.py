"""独立 YOLO 检测显示窗口"""
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap


class DetectionViewerWindow(QMainWindow):
    """独立 YOLO 检测显示窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO 检测实时显示")
        self.setMinimumSize(640, 540)
        self.resize(800, 600)
        
        # 窗口状态存储
        self._window_state = {
            'x': 150,
            'y': 150,
            'width': 800,
            'height': 600
        }
        self._load_window_state()
        
        # 数据
        self._current_frame = None
        self._detections = []
        self._fps = 0.0
        self._fps_history = []  # FPS 历史记录用于平滑
        self._frame_count = 0
        self._last_fps_update = 0
        self._last_detect_img = None  # 缓存最后一帧检测图像
        
        self._build_ui()
        
        # 定时更新显示 - 提高更新频率到 60 FPS
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(16)  # 60 FPS 更新（更流畅）
        self._update_timer.timeout.connect(self._update_display)
        # 定时器先启动，但窗口默认隐藏
        self._update_timer.start()
    
    def _build_ui(self):
        """构建 UI"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # ── 信息栏 ──────────────────────────────────────────
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)
        
        self._lbl_info = QLabel("等待帧...")
        self._lbl_info.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self._lbl_info)
        
        info_layout.addStretch()
        
        self._lbl_fps = QLabel("FPS: 0.0")
        self._lbl_fps.setStyleSheet("color: #00aa00; font-weight: bold; font-family: Consolas;")
        info_layout.addWidget(self._lbl_fps)
        
        self._lbl_count = QLabel("检测数：0")
        self._lbl_count.setStyleSheet("color: #0066cc; font-weight: bold; font-family: Consolas;")
        info_layout.addWidget(self._lbl_count)
        
        layout.addWidget(info_frame)
        
        # ── 检测图像显示 ──────────────────────────────────────────
        self._lbl_image = QLabel()
        self._lbl_image.setMinimumSize(640, 480)
        self._lbl_image.setAlignment(Qt.AlignCenter)
        self._lbl_image.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._lbl_image, 1)
        
        # ── 控制按钮 ──────────────────────────────────────────
        btn_layout = QHBoxLayout()
        
        self._btn_close = QPushButton("关闭")
        self._btn_close.setMinimumHeight(32)
        self._btn_close.clicked.connect(self._on_close)
        btn_layout.addWidget(self._btn_close)
        
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
    
    def update_frame(self, frame, detections, fps=0.0):
        """更新帧和检测结果 - 优化：只复制必要的数据"""
        # 只在帧变化时更新，避免重复处理
        if frame is not None:
            self._current_frame = frame
            self._last_detect_img = frame  # 缓存检测图像
        
        self._detections = detections
        
        # 使用滑动平均计算 FPS（更稳定和准确）
        if fps > 0:
            self._fps_history.append(fps)
            # 保留最近 10 帧的 FPS
            if len(self._fps_history) > 10:
                self._fps_history.pop(0)
            # 计算平均 FPS
            self._fps = sum(self._fps_history) / len(self._fps_history)
        
        self._frame_count += 1
    
    def _update_display(self):
        """更新显示 - 优化渲染性能"""
        if self._current_frame is None:
            self._lbl_image.clear()
            self._lbl_info.setText("等待帧...")
            return
        
        # 使用缓存的检测图像，避免重复 copy
        display_frame = self._last_detect_img if self._last_detect_img is not None else self._current_frame
        h, w = display_frame.shape[:2]
        
        # 只在有检测框时才复制和绘制
        if self._detections:
            display_frame = display_frame.copy()
            # 绘制所有检测框
            for det in self._detections:
                x1, y1, x2, y2, conf, cls_id, cx, cy = det[:8]
                # 确保坐标在范围内
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                # 绘制矩形框
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签
                label = f"C{cls_id} {conf:.2f}"
                cv2.putText(display_frame, label, (x1, max(20, y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 绘制中心点
                cv2.circle(display_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # 转换为 QImage 并显示 - 优化：避免不必要的缩放
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 只在尺寸变化时才缩放
        label_size = self._lbl_image.size()
        if label_size.width() > 0 and label_size.height() > 0:
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.FastTransformation  # 使用快速缩放算法
            )
            self._lbl_image.setPixmap(scaled_pixmap)
        else:
            self._lbl_image.setPixmap(pixmap)

        # 更新信息栏
        self._lbl_info.setText(f"帧尺寸：{w}×{h}")
        self._lbl_count.setText(f"检测数：{len(self._detections)}")
        self._lbl_fps.setText(f"FPS: {self._fps:.1f}")

    def showEvent(self, event):
        """窗口显示事件 - 重新启动定时器"""
        # 窗口显示时重新启动定时器，确保画面更新
        if not self._update_timer.isActive():
            self._update_timer.start()
        super().showEvent(event)

    def _on_close(self):
        """关闭窗口"""
        self._save_window_state()
        # 关闭时停止定时器以节省资源
        self._update_timer.stop()
        self.hide()

    def _save_window_state(self):
        """保存窗口状态"""
        self._window_state = {
            'x': self.x(),
            'y': self.y(),
            'width': self.width(),
            'height': self.height()
        }
        import os
        import json
        state_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.detection_viewer_state.json'
        )
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self._window_state, f, indent=2)

    def _load_window_state(self):
        """加载窗口状态"""
        import os
        import json
        state_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.detection_viewer_state.json'
        )
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self._window_state = state
                self.setGeometry(
                    state.get('x', 150),
                    state.get('y', 150),
                    state.get('width', 800),
                    state.get('height', 600)
                )
            except:
                pass

    def closeEvent(self, event):
        """关闭事件"""
        self._save_window_state()
        self._update_timer.stop()
        self.hide()
        event.ignore()