from PySide6.QtWidgets import QWidget, QLabel
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QPainter, QPen, QColor, QCursor, QPixmap, QImage
import numpy as np
import cv2
import ctypes
import ctypes.wintypes

# Windows API handles
user32 = ctypes.windll.user32


def _get_cursor_pos():
    """获取鼠标位置（原始像素坐标，不受DPI缩放影响）"""
    pt = ctypes.wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


# ── Red-border overlay for window highlight ────────────────────────────────────

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._rect = QRect()

    def set_rect(self, x, y, w, h):
        self._rect = QRect(0, 0, w, h)
        self.setGeometry(x, y, w, h)
        self.show()
        self.update()

    def hide_overlay(self):
        self.hide()

    def paintEvent(self, _event):
        if self._rect.isNull():
            return
        p = QPainter(self)
        p.setPen(QPen(QColor(255, 0, 0), 3))
        p.drawRect(2, 2, self._rect.width() - 4, self._rect.height() - 4)


# ── Full-screen region selector ────────────────────────────────────────────────

class RegionSelector(QWidget):
    """Press-drag to select; Enter to confirm; Esc to cancel."""
    region_selected = Signal(int, int, int, int)
    cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._start = self._end = None
        self._dragging = False

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._start and self._end:
                x1 = min(self._start.x(), self._end.x())
                y1 = min(self._start.y(), self._end.y())
                x2 = max(self._start.x(), self._end.x())
                y2 = max(self._start.y(), self._end.y())
                w, h = x2 - x1, y2 - y1
                if w > 4 and h > 4:
                    self.hide()
                    self.region_selected.emit(x1, y1, w, h)
                    return
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.cancelled.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 使用Windows API直接获取鼠标位置
            x, y = _get_cursor_pos()
            from PySide6.QtCore import QPoint
            self._start = self._end = QPoint(x, y)
            self._dragging = True

    def mouseMoveEvent(self, event):
        if self._dragging:
            # 使用Windows API直接获取鼠标位置
            x, y = _get_cursor_pos()
            from PySide6.QtCore import QPoint
            self._end = QPoint(x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 使用Windows API直接获取鼠标位置
            x, y = _get_cursor_pos()
            from PySide6.QtCore import QPoint
            self._end = QPoint(x, y)
            self._dragging = False
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 80))
        if self._start and self._end:
            x1 = min(self._start.x(), self._end.x())
            y1 = min(self._start.y(), self._end.y())
            x2 = max(self._start.x(), self._end.x())
            y2 = max(self._start.y(), self._end.y())
            sel = QRect(x1, y1, x2 - x1, y2 - y1)
            p.fillRect(sel, QColor(255, 255, 255, 30))
            p.setPen(QPen(QColor(255, 0, 0), 2))
            p.drawRect(sel)


# ── Canvas ─────────────────────────────────────────────────────────────────────

_CANVAS_SIZE = 4000
_DISPLAY_SIZE = 400


class CanvasWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(_DISPLAY_SIZE, _DISPLAY_SIZE)
        self.setStyleSheet("background:#1a1a2e; border:1px solid #555;")
        self._ext_canvas = None
        self._own_canvas = np.zeros((_CANVAS_SIZE, _CANVAS_SIZE, 3), dtype=np.uint8)
        self._vx = 0
        self._vy = 0
        self._draw_traj = False
        self._trajectory = None
        self._detections = None
        self._draw_detect = False
        self._route_pts = None  # 规划路线点（静态，青色）
        self._player_marker = None  # 玩家当前位置 (x, y)
        self._refresh()

    def set_external_canvas(self, arr: np.ndarray):
        self._ext_canvas = arr
        cx = arr.shape[1] // 2
        cy = arr.shape[0] // 2
        self._vx = max(0, cx - _DISPLAY_SIZE // 2)
        self._vy = max(0, cy - _DISPLAY_SIZE // 2)

    def set_route(self, points):
        """设置规划路线（静态青色线），并将视口居中到路线中点"""
        self._route_pts = points
        if points:
            mid = points[len(points) // 2]
            src = self._ext_canvas if self._ext_canvas is not None else self._own_canvas
            max_vx = max(0, src.shape[1] - _DISPLAY_SIZE)
            max_vy = max(0, src.shape[0] - _DISPLAY_SIZE)
            self._vx = max(0, min(max_vx, int(mid[0]) - _DISPLAY_SIZE // 2))
            self._vy = max(0, min(max_vy, int(mid[1]) - _DISPLAY_SIZE // 2))
        self._refresh()

    def set_player_marker(self, x: float, y: float):
        """更新玩家位置标记，视口跟随玩家居中"""
        self._player_marker = (x, y)
        src = self._ext_canvas if self._ext_canvas is not None else self._own_canvas
        max_vx = max(0, src.shape[1] - _DISPLAY_SIZE)
        max_vy = max(0, src.shape[0] - _DISPLAY_SIZE)
        self._vx = max(0, min(max_vx, int(x) - _DISPLAY_SIZE // 2))
        self._vy = max(0, min(max_vy, int(y) - _DISPLAY_SIZE // 2))
        self._refresh()

    def clear_player_marker(self):
        """清除玩家位置标记"""
        self._player_marker = None
        self._refresh()

    def set_trajectory_source(self, traj_list, enabled: bool):
        self._trajectory = traj_list
        self._draw_traj = enabled

    def set_draw_trajectory(self, enabled: bool):
        self._draw_traj = enabled

    def set_detections(self, detections, enabled: bool):
        self._detections = detections
        self._draw_detect = enabled

    def set_draw_detections(self, enabled: bool):
        self._draw_detect = enabled

    def reset(self):
        self._ext_canvas = None
        self._own_canvas[:] = 0
        self._trajectory = None
        self._detections = None
        self._route_pts = None
        self._player_marker = None
        self._draw_traj = False
        self._draw_detect = False
        self._vx = 0
        self._vy = 0
        self._refresh()

    def get_canvas(self) -> np.ndarray:
        if self._ext_canvas is not None:
            return self._ext_canvas.copy()
        return self._own_canvas.copy()

    def place_first_frame(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        cx = _CANVAS_SIZE // 2
        cy = _CANVAS_SIZE // 2
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = min(_CANVAS_SIZE, x1 + w)
        y2 = min(_CANVAS_SIZE, y1 + h)
        self._own_canvas[y1:y2, x1:x2] = img_bgr[:y2 - y1, :x2 - x1]
        self._vx = max(0, cx - _DISPLAY_SIZE // 2)
        self._vy = max(0, cy - _DISPLAY_SIZE // 2)

        # 强制清除绘制标志，确保不会绘制任何轨迹或检测框
        self._draw_traj = False
        self._draw_detect = False

        self._refresh()

    def refresh_from_external(self):
        self._refresh()

    def _refresh(self):
        src = self._ext_canvas if self._ext_canvas is not None else self._own_canvas
        h_s, w_s = src.shape[:2]
        x1 = self._vx;
        y1 = self._vy
        x2 = min(w_s, x1 + _DISPLAY_SIZE)
        y2 = min(h_s, y1 + _DISPLAY_SIZE)
        crop = src[y1:y2, x1:x2].copy()
        if crop.shape[0] < _DISPLAY_SIZE or crop.shape[1] < _DISPLAY_SIZE:
            pad = np.zeros((_DISPLAY_SIZE, _DISPLAY_SIZE, 3), dtype=np.uint8)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad

        # 叠加轨迹
        if self._draw_traj and self._trajectory and len(self._trajectory) >= 2:
            pts = self._trajectory

            def to_crop(cx, cy):
                return (int(cx - self._vx), int(cy - self._vy))

            # 只绘制在裁剪区域内的轨迹
            crop_h, crop_w = crop.shape[:2]
            for i in range(1, len(pts)):
                p1 = to_crop(*pts[i - 1])
                p2 = to_crop(*pts[i])
                # 检查点是否在裁剪区域内
                if (0 <= p1[0] < crop_w and 0 <= p1[1] < crop_h and
                        0 <= p2[0] < crop_w and 0 <= p2[1] < crop_h):
                    cv2.line(crop, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

            if len(pts) >= 2:
                prev = to_crop(*pts[-2])
                curr = to_crop(*pts[-1])
                # 检查点是否在裁剪区域内
                if (0 <= prev[0] < crop_w and 0 <= prev[1] < crop_h and
                        0 <= curr[0] < crop_w and 0 <= curr[1] < crop_h):
                    cv2.arrowedLine(crop, prev, curr, (0, 0, 255),
                                    2, cv2.LINE_AA, tipLength=0.4)

        # 叠加检测结果
        if self._draw_detect and self._detections:
            # 检测框坐标是相对于小地图区域的，直接绘制在当前帧上
            # 简化处理：直接在裁剪区域上绘制检测框
            for det in self._detections:
                # 兼容 8 个元素和 9 个元素的格式
                if len(det) >= 8:
                    x1_d, y1_d, x2_d, y2_d, conf, class_id, cx, cy = det[:8]
                    cls_name = det[8] if len(det) >= 9 else f"C{class_id}"
                else:
                    continue

                # 转换到裁剪区域坐标
                # 由于裁剪区域显示的是小地图的当前视图，
                # 我们需要将检测框坐标映射到裁剪区域
                # 这里简化处理，直接使用检测框坐标
                x1_crop = x1_d
                y1_crop = y1_d
                x2_crop = x2_d
                y2_crop = y2_d

                if not (x2_crop < 0 or x1_crop >= _DISPLAY_SIZE or
                        y2_crop < 0 or y1_crop >= _DISPLAY_SIZE):
                    x1_crop = max(0, x1_crop)
                    y1_crop = max(0, y1_crop)
                    x2_crop = min(_DISPLAY_SIZE, x2_crop)
                    y2_crop = min(_DISPLAY_SIZE, y2_crop)

                    cv2.rectangle(crop, (x1_crop, y1_crop), (x2_crop, y2_crop),
                                  (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(crop, label, (x1_crop, max(15, y1_crop - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # ── 叠加规划路线（青色，canvas2 专用） ───────────────────────────
        if self._route_pts and len(self._route_pts) >= 2:
            crop_h, crop_w = crop.shape[:2]
            # 每隔几个点采样，避免密集点时性能问题
            step = max(1, len(self._route_pts) // 500)
            sampled = self._route_pts[::step]
            for i in range(1, len(sampled)):
                p1 = (int(sampled[i - 1][0]) - self._vx,
                      int(sampled[i - 1][1]) - self._vy)
                p2 = (int(sampled[i][0]) - self._vx,
                      int(sampled[i][1]) - self._vy)
                # 只绘制至少一个端点在视口内（或附近）的线段
                margin = 10
                if ((-margin < p1[0] < crop_w + margin or -margin < p2[0] < crop_w + margin) and
                        (-margin < p1[1] < crop_h + margin or -margin < p2[1] < crop_h + margin)):
                    cv2.line(crop, p1, p2, (0, 220, 220), 2, cv2.LINE_AA)

        # ── 叠加玩家位置标记（绿色圆点，canvas2 专用） ───────────────────
        if self._player_marker is not None:
            px = int(self._player_marker[0]) - self._vx
            py = int(self._player_marker[1]) - self._vy
            if 4 <= px < _DISPLAY_SIZE - 4 and 4 <= py < _DISPLAY_SIZE - 4:
                cv2.circle(crop, (px, py), 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(crop, (px, py), 5, (0, 220, 60), -1, cv2.LINE_AA)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))