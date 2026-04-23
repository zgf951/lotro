"""
轮廓标定工具 v2
===============
流程：
  1. 点击"框选截图"→ 拖拽选择游戏小地图区域 → 自动放大3倍加载到画布
  2. 点击"提取轮廓"：
       - 手动模式：先点中心点，再点箭尖点
       - 自动模式：HSV 自动提取（需 HSV 范围匹配）
  3. 确认无误后"保存模板"
  4. 勾选"自动截图 + 实时匹配"→ 实时输出角度
"""

import sys
import os
import cv2
import numpy as np
import json
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QFileDialog,
    QMessageBox, QDoubleSpinBox, QTextEdit,
    QCheckBox, QRadioButton,
)
from PySide6.QtCore import Qt, QTimer, Signal, QRect, QPointF
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont, QKeyEvent,
)

# 根据实际项目结构调整以下导入
from utils.window_manager import WindowManager
from ui.widgets import OverlayWindow
from core.contour_matcher import ContourMatcher, bearing_to_direction


# ═══════════════════════════════════════════════════════════════════════
# 全屏框选窗口
# ═══════════════════════════════════════════════════════════════════════

class RegionSelector(QWidget):
    """全屏透明遮罩，左键拖拽框选屏幕区域。"""

    region_selected = Signal(int, int, int, int)   # x, y, w, h
    cancelled       = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._start = None
        self._end   = None
        self._selecting = False

    def start(self):
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.show()
        self.raise_()
        self.activateWindow()

    # ── 绘制 ──────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))
        if self._start and self._end:
            rect = self._rect()
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, QColor(0, 0, 0, 0))
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(0, 200, 255), 2))
            painter.drawRect(rect)
            painter.setPen(QColor(255, 255, 0))
            font = painter.font(); font.setPointSize(11); painter.setFont(font)
            painter.drawText(rect.x(), max(rect.y() - 6, 14),
                             f"{rect.width()} × {rect.height()}")
        painter.end()

    def _rect(self):
        x1, y1 = self._start.x(), self._start.y()
        x2, y2 = self._end.x(),   self._end.y()
        return QRect(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))

    # ── 鼠标事件 ──────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position().toPoint()
            self._end   = self._start
            self._selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._selecting:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._selecting:
            self._end = event.position().toPoint()
            self._selecting = False
            rect = self._rect()
            self.hide()
            if rect.width() > 4 and rect.height() > 4:
                self.region_selected.emit(
                    rect.x(), rect.y(), rect.width(), rect.height())
            else:
                self.cancelled.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._selecting = False
            self.hide()
            self.cancelled.emit()


# ═══════════════════════════════════════════════════════════════════════
# 画布
# ═══════════════════════════════════════════════════════════════════════

class ContourCanvas(QWidget):
    """显示图像、轮廓、中心点、箭尖点；支持点击选点。"""

    point_selected = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setMaximumSize(800, 800)
        self._image         = None
        self._contours      = []
        self._center        = None
        self._tip           = None
        self._angle_line    = None
        self._scale         = 1.0
        self._offset        = QPointF(0, 0)
        self._is_selecting  = False
        self._selection_mode = None   # 'center' | 'tip'

    # ── 设置方法 ──────────────────────────────────────────────────────

    def set_image(self, image):
        self._image = image
        self._contours = []; self._center = None; self._tip = None
        self._is_selecting = False; self._selection_mode = None
        self.update()

    def set_selection_mode(self, mode):
        self._is_selecting = True; self._selection_mode = mode
        self.setCursor(Qt.CursorShape.CrossCursor); self.update()

    def cancel_selection(self):
        self._is_selecting = False; self._selection_mode = None
        self.setCursor(Qt.CursorShape.ArrowCursor); self.update()

    def set_contours(self, c): self._contours = c; self.update()
    def set_center(self, c):   self._center   = c; self.update()
    def set_tip(self, t):      self._tip      = t; self.update()
    def set_angle_line(self, l): self._angle_line = l; self.update()

    def clear(self):
        self._image = None; self._contours = []
        self._center = None; self._tip = None; self._angle_line = None
        self.update()

    # ── 绘制 ──────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._image is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        img = self._image
        h, w = img.shape[:2]
        self._scale = min(self.width() / w, self.height() / h)
        nw, nh = int(w * self._scale), int(h * self._scale)
        ox = (self.width()  - nw) // 2
        oy = (self.height() - nh) // 2
        self._offset = QPointF(ox, oy)

        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            nw, nh, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        p.drawPixmap(ox, oy, pix)

        def to_canvas(x, y):
            return QPointF(ox + x * self._scale, oy + y * self._scale)

        # 轮廓（绿）
        if self._contours:
            p.setPen(QPen(QColor(0, 255, 0), 2))
            for cnt in self._contours:
                pts = [to_canvas(pt[0], pt[1]) for pt in cnt.reshape(-1, 2)]
                if pts: p.drawPolygon(pts)

        # 中心点（绿实心）
        if self._center:
            cp = to_canvas(*self._center)
            p.setBrush(QColor(0, 255, 0))
            p.setPen(QPen(QColor(0, 180, 0), 2))
            p.drawEllipse(cp, 7, 7)

        # 箭尖（红实心）
        if self._tip:
            tp = to_canvas(*self._tip)
            p.setBrush(QColor(255, 0, 0))
            p.setPen(QPen(QColor(200, 0, 0), 2))
            p.drawEllipse(tp, 7, 7)

        # 连线（黄）
        if self._center and self._tip:
            p.setPen(QPen(QColor(255, 220, 0), 2))
            p.drawLine(to_canvas(*self._center), to_canvas(*self._tip))

        # 角度指示线（青色虚线）
        if self._angle_line:
            s, e = self._angle_line
            p.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine))
            p.drawLine(to_canvas(*s), to_canvas(*e))

        # 选点提示
        if self._is_selecting:
            text = ("请点击三角形中心" if self._selection_mode == 'center'
                    else "请点击箭头尖端")
            p.setPen(QColor(255, 255, 0))
            font = p.font(); font.setPointSize(12); font.setBold(True)
            p.setFont(font)
            p.drawText(10, 26, text)

        p.end()

    # ── 鼠标 ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton
                and self._image is not None and self._is_selecting):
            x = (event.position().x() - self._offset.x()) / self._scale
            y = (event.position().y() - self._offset.y()) / self._scale
            if self._selection_mode == 'center':
                self._center = (int(x), int(y))
            elif self._selection_mode == 'tip':
                self._tip = (int(x), int(y))
            self.point_selected.emit(x, y)
            self.update()


# ═══════════════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════════════

class ContourCalibrator(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("轮廓标定工具 v2")
        self.setMinimumSize(1100, 750)

        self._current_frame   = None   # 当前显示/匹配用的帧（已放大 3×）
        self._capture_region  = None   # (x, y, w, h) 客户区相对坐标
        self._matcher         = ContourMatcher()
        self._is_auto_extract = False
        self._dxgi_capture    = None   # DXGI 窗口捕获器

        # 从父窗口获取窗口管理器
        self._window_mgr = None
        if parent and hasattr(parent, '_window_mgr'):
            self._window_mgr = parent._window_mgr

        # 区域选择器
        self._region_sel = RegionSelector()
        self._region_sel.region_selected.connect(self._on_region_selected)
        self._region_sel.cancelled.connect(
            lambda: self._log("[截图] 已取消"))

        # 实时捕获定时器（每 100ms 截一次）
        self._cap_timer = QTimer(self)
        self._cap_timer.setInterval(100)
        self._cap_timer.timeout.connect(self._auto_capture)

        # 匹配定时器（每 100ms 匹配一次，通常与截图同频）
        self._match_timer = QTimer(self)
        self._match_timer.setInterval(100)
        self._match_timer.timeout.connect(self._match_contour)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        main = QHBoxLayout(root)

        # 左侧
        left_grp = QGroupBox("图像画布")
        left_lay = QVBoxLayout(left_grp)

        self._canvas = ContourCanvas()
        self._canvas.point_selected.connect(self._on_point_selected)
        left_lay.addWidget(self._canvas)

        row1 = QHBoxLayout()
        self._btn_snap = QPushButton("📸 框选截图（×3放大）")
        self._btn_snap.setMinimumHeight(32)
        self._btn_snap.clicked.connect(self._snapshot)
        row1.addWidget(self._btn_snap)

        self._chk_live = QCheckBox("🔄 实时截图")
        self._chk_live.setToolTip("勾选后持续从已框选区域截图，配合实时匹配使用")
        self._chk_live.stateChanged.connect(self._on_live_toggled)
        row1.addWidget(self._chk_live)
        left_lay.addLayout(row1)

        # 提取组
        ext_grp = QGroupBox("轮廓提取")
        ext_lay = QVBoxLayout(ext_grp)

        row2 = QHBoxLayout()
        self._radio_manual = QRadioButton("手动点击")
        self._radio_auto   = QRadioButton("自动提取（HSV）")
        self._radio_manual.setChecked(True)
        self._radio_manual.toggled.connect(self._on_mode_changed)
        row2.addWidget(self._radio_manual); row2.addWidget(self._radio_auto)
        ext_lay.addLayout(row2)

        self._lbl_hint = QLabel('点击"提取轮廓"→ 在画布上依次点中心、箭尖')
        self._lbl_hint.setWordWrap(True)
        ext_lay.addWidget(self._lbl_hint)

        row3 = QHBoxLayout()
        self._btn_extract = QPushButton("🔍 提取轮廓")
        self._btn_extract.setMinimumHeight(32)
        self._btn_extract.clicked.connect(self._extract_contour)
        self._btn_extract.setEnabled(False)
        row3.addWidget(self._btn_extract)

        self._btn_clear = QPushButton("🗑️ 清除")
        self._btn_clear.setMinimumHeight(32)
        self._btn_clear.clicked.connect(self._clear_canvas)
        row3.addWidget(self._btn_clear)
        ext_lay.addLayout(row3)
        left_lay.addWidget(ext_grp)
        main.addWidget(left_grp, stretch=2)

        # 右侧
        right = QVBoxLayout()

        # 模板管理
        tmpl_grp = QGroupBox("模板管理")
        tmpl_lay = QVBoxLayout(tmpl_grp)

        row4 = QHBoxLayout()
        self._btn_save = QPushButton("💾 保存模板")
        self._btn_save.setMinimumHeight(32)
        self._btn_save.clicked.connect(self._save_template)
        self._btn_save.setEnabled(False)
        row4.addWidget(self._btn_save)

        self._btn_load = QPushButton("📂 加载模板")
        self._btn_load.setMinimumHeight(32)
        self._btn_load.clicked.connect(self._load_template)
        row4.addWidget(self._btn_load)
        tmpl_lay.addLayout(row4)

        self._lbl_tmpl = QLabel("模板：未加载")
        self._lbl_tmpl.setWordWrap(True)
        tmpl_lay.addWidget(self._lbl_tmpl)

        std_row = QHBoxLayout()
        std_row.addWidget(QLabel("标准方向:"))
        self._spin_bearing = QDoubleSpinBox()
        self._spin_bearing.setRange(0, 360); self._spin_bearing.setDecimals(1)
        self._spin_bearing.setSuffix("°")
        std_row.addWidget(self._spin_bearing)
        self._btn_north = QPushButton("设为朝北(0°)")
        self._btn_north.clicked.connect(lambda: self._spin_bearing.setValue(0))
        std_row.addWidget(self._btn_north)
        std_row.addStretch()
        tmpl_lay.addLayout(std_row)
        right.addWidget(tmpl_grp)

        # 实时匹配
        match_grp = QGroupBox("实时匹配")
        match_lay = QVBoxLayout(match_grp)

        self._chk_auto_match = QCheckBox("开启实时匹配")
        self._chk_auto_match.toggled.connect(self._on_match_toggled)
        match_lay.addWidget(self._chk_auto_match)
        right.addWidget(match_grp)

        # 结果
        res_grp = QGroupBox("匹配结果")
        res_lay = QVBoxLayout(res_grp)

        self._lbl_angle = QLabel("角度：--")
        self._lbl_angle.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self._lbl_angle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_lay.addWidget(self._lbl_angle)

        self._lbl_dir = QLabel("方向：--")
        self._lbl_dir.setFont(QFont("Arial", 14))
        self._lbl_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_lay.addWidget(self._lbl_dir)

        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumHeight(200)
        self._log_box.setStyleSheet(
            "background:#0d1117;color:#c9d1d9;font-family:Consolas;")
        res_lay.addWidget(self._log_box)
        right.addWidget(res_grp)
        right.addStretch()
        main.addLayout(right, stretch=1)
        
        # UI 构建完成后，记录窗口管理器状态
        if self._window_mgr:
            self._log(f"[窗口] 已继承父窗口的窗口管理器，句柄：{self._window_mgr.hwnd}")
        else:
            self._log("[窗口] 未检测到父窗口管理器，将使用屏幕坐标模式")

    # ── 日志 ──────────────────────────────────────────────────────────

    def _log(self, msg):
        print(msg)
        self._log_box.append(msg)
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())

    # ── 截图 ──────────────────────────────────────────────────────────

    def _snapshot(self):
        """隐藏主窗口 → 弹出全屏遮罩 → 拖拽框选 → 截图放大3×加载画布"""
        self._log("[截图] 请在屏幕上拖拽框选包含三角形的区域（Esc取消）")
        self.hide()
        QTimer.singleShot(150, self._region_sel.start)

    def _on_region_selected(self, x, y, w, h):
        """框选完成：将屏幕坐标转换为客户区坐标，然后截图放大 3 倍"""
        self.show(); self.raise_()
        
        # 如果有窗口绑定，将屏幕坐标转换为客户区坐标
        if self._window_mgr and self._window_mgr.is_valid():
            # 使用窗口管理器转换坐标
            rel_x, rel_y = self._window_mgr.screen_to_client(x, y)
            self._capture_region = (rel_x, rel_y, w, h)
            self._log(f"[截图] 屏幕坐标：({x},{y}) → 客户区坐标：({rel_x},{rel_y})  {w}×{h}px")
        else:
            # 没有窗口绑定，使用屏幕坐标
            self._capture_region = (x, y, w, h)
            self._log(f"[截图] 未绑定窗口，使用屏幕坐标：({x},{y})  {w}×{h}px")
        
        img = self._grab_region(*self._capture_region)
        if img is not None:
            self._load_frame(img, w, h)

    def _grab_region(self, x, y, w, h):
        """使用 DXGI 截取窗口客户区指定区域，返回 BGR ndarray 或 None。"""
        # 优先使用 DXGI 捕获窗口客户区
        if self._window_mgr and self._window_mgr.is_valid():
            try:
                if self._dxgi_capture is None:
                    from utils.dxgi_capture import DxgiWindowCapture
                    self._dxgi_capture = DxgiWindowCapture(self._window_mgr.hwnd)
                    self._log(f"[DXGI] 捕获器已创建，句柄：{self._window_mgr.hwnd}")
                
                # 如果窗口句柄变化，重新初始化
                if self._dxgi_capture.hwnd != self._window_mgr.hwnd:
                    self._dxgi_capture.hwnd = self._window_mgr.hwnd
                    self._log(f"[DXGI] 窗口句柄已更新")
                
                img = self._dxgi_capture.capture(x, y, w, h)
                if img is None:
                    self._log(f"[DXGI] 捕获失败：({x}, {y}, {w}, {h})")
                    # 回退到 MSS
                    return self._grab_with_mss(x, y, w, h, use_screen_coords=True)
                
                self._log(f"[DXGI] ✓ 捕获成功：{img.shape}, 类型：{img.dtype}")
                return img
            except Exception as e:
                self._log(f"[DXGI] ✗ 失败：{e}")
                # 回退到 MSS
                return self._grab_with_mss(x, y, w, h, use_screen_coords=True)
        else:
            # 没有窗口绑定，使用 MSS 截取屏幕坐标
            return self._grab_with_mss(x, y, w, h, use_screen_coords=False)
    
    def _grab_with_mss(self, x, y, w, h, use_screen_coords=False):
        """使用 MSS 截取屏幕区域（回退方案）"""
        try:
            import mss
            if use_screen_coords:
                # 窗口绑定模式下，x,y 是客户区坐标，需要转换为屏幕坐标
                client_left, client_top = self._window_mgr.client_to_screen(0, 0)
                screen_x = client_left + x
                screen_y = client_top + y
                self._log(f"[MSS] 客户区 ({x},{y}) → 屏幕 ({screen_x},{screen_y})")
            else:
                # 纯屏幕坐标模式
                screen_x, screen_y = x, y
            
            with mss.mss() as sct:
                raw = sct.grab({"left": screen_x, "top": screen_y, "width": w, "height": h})
            return cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        except Exception as e:
            self._log(f"[MSS] ✗ 失败：{e}")
            return None

    def _load_frame(self, img, orig_w, orig_h):
        """放大3倍后加载到画布"""
        big = cv2.resize(img, None, fx=3, fy=3,
                         interpolation=cv2.INTER_NEAREST)
        self._current_frame = big
        self._canvas.set_image(big)
        self._btn_extract.setEnabled(True)
        self._log(f"[截图] ✓ 原始 {orig_w}×{orig_h}px → 画布 "
                  f"{big.shape[1]}×{big.shape[0]}px（×3）")

    def _auto_capture(self):
        """实时截图定时器回调：刷新当前帧（不重置画布上的标注）"""
        if not self._capture_region:
            return
        
        # 使用 DXGI 自动获取窗口位置并截图
        rel_x, rel_y, w, h = self._capture_region
        img = self._grab_region(rel_x, rel_y, w, h)
        
        if img is not None:
            # 原始图像用于匹配
            self._current_frame = img
            # 放大图像用于显示
            big = cv2.resize(img, None, fx=3, fy=3,
                             interpolation=cv2.INTER_NEAREST)
            self._current_frame_big = big
            # 只更新底层图像，不重置轮廓/标注
            self._canvas._image = big
            self._canvas.update()

    def _on_live_toggled(self, state):
        if state:
            if not self._capture_region:
                self._chk_live.setChecked(False)
                self._log("[截图] ✗ 请先框选截图区域")
                return
            self._cap_timer.start()
            self._log("[截图] 实时截图已启动")
        else:
            self._cap_timer.stop()
            self._log("[截图] 实时截图已停止")

    # ── 提取轮廓 ──────────────────────────────────────────────────────

    def _on_mode_changed(self):
        if self._radio_manual.isChecked():
            self._lbl_hint.setText('点击"提取轮廓"→ 在画布上依次点中心、箭尖')
            self._is_auto_extract = False
        else:
            self._lbl_hint.setText('点击"提取轮廓"→ 自动 HSV 颜色提取')
            self._is_auto_extract = True

    def _extract_contour(self):
        if self._current_frame is None:
            self._log("[提取] 请先截图"); return
        if self._is_auto_extract:
            self._auto_extract()
        else:
            self._canvas.set_selection_mode('center')
            self._log("[选点] 请点击三角形的中心点")

    def _on_point_selected(self, x, y):
        if not self._canvas._is_selecting:
            return
        if self._canvas._selection_mode == 'center':
            self._log(f"[选点] 中心点：({x:.0f}, {y:.0f})  → 请点击箭头尖端")
            self._canvas.set_selection_mode('tip')
        elif self._canvas._selection_mode == 'tip':
            self._log(f"[选点] 箭尖点：({x:.0f}, {y:.0f})")
            self._canvas.cancel_selection()
            self._finish_manual()

    def _finish_manual(self):
        center = self._canvas._center
        tip    = self._canvas._tip
        if not center or not tip:
            self._log("[提取] ✗ 选点未完成"); return

        dx = tip[0] - center[0]
        dy = center[1] - tip[1]   # 翻转 Y
        math_a  = math.degrees(math.atan2(dy, dx))
        bearing = (90.0 - math_a + 360.0) % 360.0

        # 构造一个简单三角形轮廓以便保存
        cx, cy = center; tx, ty = tip
        length = math.hypot(tx-cx, ty-cy)
        if length > 0:
            perp_x = -(ty-cy)/length * 10
            perp_y =  (tx-cx)/length * 10
            contour = np.array([
                [[tx, ty]],
                [[int(cx-perp_x), int(cy-perp_y)]],
                [[int(cx+perp_x), int(cy+perp_y)]],
            ], dtype=np.int32)
            self._canvas.set_contours([contour])

        self._spin_bearing.setValue(bearing)
        self._draw_north_line(center)
        self._btn_save.setEnabled(True)
        self._log(f"[提取] ✓  {bearing:.1f}°  {bearing_to_direction(bearing)}")

    def _auto_extract(self):
        result = self._matcher.extract_contour(self._current_frame)
        if "error" in result:
            self._log(f"[提取] ✗ {result['error']}"); return

        self._canvas.set_contours([result["contour"]])
        self._canvas.set_center(result["center"])
        self._canvas.set_tip(result["tip"])
        self._spin_bearing.setValue(result["bearing"])
        self._draw_north_line(result["center"])
        self._btn_save.setEnabled(True)
        self._log(f"[提取] ✓  {result['bearing']:.1f}°  "
                  f"{bearing_to_direction(result['bearing'])}")

    def _draw_north_line(self, center):
        cx, cy = center
        end_x = cx
        end_y = cy - 50   # 屏幕向上 = 北
        self._canvas.set_angle_line(((cx, cy), (end_x, end_y)))

    # ── 模板保存/加载 ─────────────────────────────────────────────────

    def _save_template(self):
        cnts   = self._canvas._contours
        center = self._canvas._center
        tip    = self._canvas._tip
        if not cnts or center is None or tip is None:
            self._log("[保存] ✗ 请先提取轮廓"); return

        # 先提取当前帧的角度（用于后续差值计算）
        if self._current_frame is not None:
            extract_result = self._matcher.extract_contour(self._current_frame)
            if "error" not in extract_result:
                computed_bearing = extract_result["bearing"]
                self._log(f"[保存] 计算角度：{computed_bearing:.1f}°")
            else:
                computed_bearing = None
                self._log(f"[保存] 无法提取计算角度：{extract_result['error']}")
        else:
            computed_bearing = None
            self._log("[保存] 警告：当前帧为空，无法保存计算角度")

        bearing = self._spin_bearing.value()
        path    = self._matcher.save_template(
            cnts[0], center, tip, bearing, computed_bearing)
        self._log(f"[保存] ✓ {os.path.basename(path)}  标准方向 {bearing:.1f}°")
        self._lbl_tmpl.setText(
            f"模板：{os.path.basename(path)}\n标准方向：{bearing:.1f}°")

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载模板", "", "JSON (*.json)")
        if not path: return
        if self._matcher.load_template(path):
            info = self._matcher.get_template_info()
            self._log(f"[加载] ✓ {os.path.basename(path)}")
            self._lbl_tmpl.setText(
                f"模板：{os.path.basename(path)}\n"
                f"标准方向：{info['standard_bearing']:.1f}°")
        else:
            QMessageBox.critical(self, "错误", "加载模板失败")

    # ── 实时匹配 ──────────────────────────────────────────────────────

    def _on_match_toggled(self, checked):
        if checked:
            if not self._capture_region:
                self._chk_auto_match.setChecked(False)
                self._log("[匹配] ✗ 请先框选截图区域"); return
            # 同时启动实时截图
            self._chk_live.setChecked(True)
            self._match_timer.start()
            self._log("[匹配] 实时匹配已启动")
        else:
            self._match_timer.stop()
            self._log("[匹配] 实时匹配已停止")

    def _match_contour(self):
        # 检查模板是否加载
        if self._matcher._template_contour is None:
            self._lbl_angle.setText("角度：--")
            self._lbl_dir.setText("✗ 模板未加载")
            return
        if self._current_frame is None:
            self._lbl_angle.setText("角度：--")
            self._lbl_dir.setText("✗ 图像未加载")
            return
            
        result = self._matcher.match(self._current_frame)
        if "error" in result:
            self._lbl_angle.setText("角度：--")
            self._lbl_dir.setText(f"✗ {result['error']}")
            return

        bearing = result["final_bearing"]
        self._lbl_angle.setText(f"{bearing:.1f}°")
        self._lbl_dir.setText(bearing_to_direction(bearing))
        # 同步更新画布上的轮廓（显示实时检测结果）
        cnt = result.get("current_contour")
        if cnt is not None:
            self._canvas.set_contours([cnt])

    # ── 清除 ──────────────────────────────────────────────────────────

    def _clear_canvas(self):
        self._canvas.clear()
        self._current_frame = None
        self._btn_extract.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._log("[清除] 画布已清空")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ContourCalibrator()
    win.show()
    sys.exit(app.exec())