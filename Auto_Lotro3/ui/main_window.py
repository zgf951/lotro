import sys
import os
import ctypes
import ctypes.wintypes
import numpy as np
import cv2
import time
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QSpinBox, QGroupBox, QCheckBox, QDoubleSpinBox,
    QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QObject, QRect, QTimer
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QCursor, QFont, QKeyEvent,
)
from utils.win32_utils import get_window_at_cursor, get_window_info, get_window_rect, get_window_client_rect, \
    screen_to_client, bitblt_capture
from utils.window_manager import WindowManager
from ui.widgets import OverlayWindow, RegionSelector, CanvasWidget
from core.stitcher import MiniMap, StitchWorker
from ui.detection_viewer import DetectionViewerWindow
from core.combat import CombatConfig
from core.combat_worker import CombatWorker
from core.trajectory_path import TrajectoryManager, TrajectoryPath
from core.map_data_saver import MapDataSaver
from ui.contour_calibrator import ContourCalibrator


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("迷你地图拼接工具 + YOLO 检测")
        self.setMinimumSize(560, 900)

        # 设置主窗口句柄（用于窗口管理器跳过自己的窗口）
        WindowManager.set_main_window_handle(int(self.winId()))

        # state
        self._hwnd = None
        self._last_hwnd = None
        self._last_cls = ""
        self._last_title = ""
        self._last_pid = ""
        self._map_region = None
        self._raw_region = None
        self._first_frame = None
        self._worker = None
        self._minimap = None
        self._capturing = False
        self._offset_x = 0
        self._offset_y = 0

        # 窗口管理器
        self._window_mgr = WindowManager()

        # YOLO（直接存储模型对象，不是检测器）
        self._yolo_model = None
        self._yolo_loaded = False
        self._yolo_detecting = False
        self._current_detections = []
        self._current_frame = None
        self._detection_fps = 0.0

        # 独立检测显示窗口
        self._detection_viewer = None

        # 自动战斗相关
        self._combat_worker = None
        self._combat_config = CombatConfig()

        # 轨迹寻路相关
        self._traj_manager = TrajectoryManager(grid_size=5)
        self._trajectory_loaded = False
        self._trajectory_name = ""
        self._loaded_canvas = None  # 预加载的地图图片
        self._canvas_offset = [0, 0]  # 地图裁剪偏移（[c0, r0]）

        # 轮廓标定工具
        self._contour_calibrator = None

        # canvas2 定时器（轨迹地图刷新）
        self._canvas2_timer = QTimer(self)
        self._canvas2_timer.setInterval(200)
        self._canvas2_timer.timeout.connect(self._tick_canvas2)

        # helpers
        self._overlay = OverlayWindow()
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(40)
        self._hover_timer.timeout.connect(self._poll_hover)

        self._region_sel = RegionSelector()
        self._region_sel.region_selected.connect(self._on_region_selected)
        self._region_sel.cancelled.connect(self._on_region_cancelled)

        self._canvas_timer = QTimer(self)
        self._canvas_timer.setInterval(200)
        self._canvas_timer.timeout.connect(self._tick_canvas)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(8)
        vbox.setContentsMargins(12, 12, 12, 12)

        # title
        lbl = QLabel("迷你地图拼接工具 + YOLO 检测")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont();
        f.setPointSize(13);
        f.setBold(True);
        lbl.setFont(f)
        vbox.addWidget(lbl)

        # canvas
        row = QHBoxLayout()
        row.addStretch()
        self._canvas = CanvasWidget()
        row.addWidget(self._canvas)
        row.addStretch()
        vbox.addLayout(row)

        # ── YOLO 模型加载组 ─────────────────────────────────────────────
        yolo_group = QGroupBox("YOLO 模型配置")
        yolo_layout = QHBoxLayout(yolo_group)

        self._btn_load_model = QPushButton("加载 YOLO 模型")
        self._btn_load_model.setMinimumHeight(28)
        self._btn_load_model.clicked.connect(self._load_yolo_model)
        yolo_layout.addWidget(self._btn_load_model)

        self._lbl_model_status = QLabel("未加载")
        self._lbl_model_status.setStyleSheet("color: #ff6b6b;")
        yolo_layout.addWidget(self._lbl_model_status)

        yolo_layout.addWidget(QLabel("置信度阈值:"))
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.0, 1.0)
        self._spin_conf.setValue(0.5)
        self._spin_conf.setSingleStep(0.05)
        yolo_layout.addWidget(self._spin_conf)

        yolo_layout.addStretch()
        vbox.addWidget(yolo_group)

        # ── 轮廓标定工具组 ──────────────────────────────────────────────
        contour_group = QGroupBox("轮廓标定工具")
        contour_layout = QHBoxLayout(contour_group)
        
        self._btn_open_calibrator = QPushButton("🎯 轮廓标定")
        self._btn_open_calibrator.setMinimumHeight(28)
        self._btn_open_calibrator.clicked.connect(self._open_contour_calibrator)
        contour_layout.addWidget(self._btn_open_calibrator)
        
        contour_layout.addStretch()
        vbox.addWidget(contour_group)

        # ── 检测控制组 ──────────────────────────────────────────────────
        detect_group = QGroupBox("检测控制")
        detect_layout = QVBoxLayout(detect_group)

        # 第一行：开始/停止按钮
        btn_row1 = QHBoxLayout()
        self._btn_start_detect = QPushButton("开始检测")
        self._btn_start_detect.setMinimumHeight(28)
        self._btn_start_detect.clicked.connect(self._start_detection)
        self._btn_start_detect.setEnabled(False)
        btn_row1.addWidget(self._btn_start_detect)

        self._btn_stop_detect = QPushButton("关闭检测")
        self._btn_stop_detect.setMinimumHeight(28)
        self._btn_stop_detect.clicked.connect(self._stop_detection)
        self._btn_stop_detect.setEnabled(False)
        btn_row1.addWidget(self._btn_stop_detect)

        detect_layout.addLayout(btn_row1)

        # 第二行：显示检测按钮
        btn_row2 = QHBoxLayout()
        self._btn_show_detect = QPushButton("🔍 显示检测")
        self._btn_show_detect.setMinimumHeight(32)
        self._btn_show_detect.clicked.connect(self._show_detection_viewer)
        btn_row2.addWidget(self._btn_show_detect)
        btn_row2.addStretch()
        detect_layout.addLayout(btn_row2)

        detect_layout.addStretch()
        vbox.addWidget(detect_group)

        # ── 自动战斗控制组 ──────────────────────────────────────────────
        combat_group = QGroupBox("自动战斗控制（测试版）")
        combat_layout = QVBoxLayout(combat_group)

        # 第一行：开始/停止按钮
        combat_btn_row = QHBoxLayout()
        self._btn_load_combat_map = QPushButton("📂 加载地图")
        self._btn_load_combat_map.setMinimumHeight(28)
        self._btn_load_combat_map.clicked.connect(self._load_map_data)
        combat_btn_row.addWidget(self._btn_load_combat_map)

        self._btn_start_combat = QPushButton("⚔️ 开始打怪")
        self._btn_start_combat.setMinimumHeight(28)
        self._btn_start_combat.clicked.connect(self._start_combat)
        self._btn_start_combat.setEnabled(False)
        combat_btn_row.addWidget(self._btn_start_combat)

        self._btn_stop_combat = QPushButton("⏹️ 停止")
        self._btn_stop_combat.setMinimumHeight(28)
        self._btn_stop_combat.clicked.connect(self._stop_combat)
        self._btn_stop_combat.setEnabled(False)
        combat_btn_row.addWidget(self._btn_stop_combat)

        combat_layout.addLayout(combat_btn_row)

        # 第二行：状态显示
        combat_status_row = QHBoxLayout()
        combat_layout.addWidget(QLabel("状态:"))
        self._lbl_combat_status = QLabel("空闲")
        self._lbl_combat_status.setStyleSheet("color: #868e96; font-weight: bold;")
        combat_layout.addWidget(self._lbl_combat_status)
        combat_layout.addStretch()

        # 第三行：目标显示
        combat_target_row = QHBoxLayout()
        combat_layout.addWidget(QLabel("目标:"))
        self._lbl_combat_target = QLabel("无")
        self._lbl_combat_target.setStyleSheet("color: #868e96;")
        combat_layout.addWidget(self._lbl_combat_target)
        combat_layout.addStretch()

        # 第四行：技能键位设置
        combat_skill_row = QHBoxLayout()
        combat_layout.addWidget(QLabel("技能键位:"))
        self._edit_skill_keys = QSpinBox()
        self._edit_skill_keys.setRange(1, 9)
        self._edit_skill_keys.setValue(5)
        self._edit_skill_keys.setPrefix("数量：")
        combat_skill_row.addWidget(self._edit_skill_keys)
        combat_layout.addLayout(combat_skill_row)

        # 第五行：攻击范围设置
        combat_range_row = QHBoxLayout()
        combat_layout.addWidget(QLabel("攻击范围:"))
        self._spin_attack_range = QSpinBox()
        self._spin_attack_range.setRange(10, 200)
        self._spin_attack_range.setValue(50)
        self._spin_attack_range.setSuffix(" 像素")
        combat_range_row.addWidget(self._spin_attack_range)
        combat_layout.addLayout(combat_range_row)

        combat_layout.addStretch()
        vbox.addWidget(combat_group)

        # ── 轨迹寻路组（独立功能） ──────────────────────────────────────
        traj_path_group = QGroupBox("轨迹寻路（独立功能）")
        traj_path_layout = QVBoxLayout(traj_path_group)

        # 地图预览画布（canvas2）
        canvas2_row = QHBoxLayout()
        canvas2_row.addStretch()
        self._canvas2 = CanvasWidget()
        canvas2_row.addWidget(self._canvas2)
        canvas2_row.addStretch()
        traj_path_layout.addLayout(canvas2_row)

        # 说明文字
        traj_path_layout.addWidget(QLabel("加载地图后显示路线；绑定窗口后可直接开始跟随"))

        # 第一行：轨迹名称
        name_row = QHBoxLayout()
        traj_path_layout.addWidget(QLabel("轨迹名称:"))
        self._edit_traj_name = QTextEdit()
        self._edit_traj_name.setMaximumHeight(30)
        self._edit_traj_name.setPlaceholderText("例如：route_001")
        name_row.addWidget(self._edit_traj_name)
        traj_path_layout.addLayout(name_row)

        # 第二行：记录按钮
        record_btn_row = QHBoxLayout()
        self._btn_record_traj = QPushButton("🔴 开始记录")
        self._btn_record_traj.setMinimumHeight(28)
        self._btn_record_traj.clicked.connect(self._toggle_recording)
        self._btn_record_traj.setEnabled(False)
        record_btn_row.addWidget(self._btn_record_traj)

        self._btn_save_traj = QPushButton("💾 保存轨迹")
        self._btn_save_traj.setMinimumHeight(28)
        self._btn_save_traj.clicked.connect(self._save_trajectory)
        self._btn_save_traj.setEnabled(False)
        record_btn_row.addWidget(self._btn_save_traj)

        traj_path_layout.addLayout(record_btn_row)

        # 第三行：播放按钮
        play_btn_row = QHBoxLayout()
        self._btn_play_traj = QPushButton("▶️ 开始跟随")
        self._btn_play_traj.setMinimumHeight(28)
        self._btn_play_traj.clicked.connect(self._start_trajectory_following)
        self._btn_play_traj.setEnabled(False)
        play_btn_row.addWidget(self._btn_play_traj)

        self._btn_stop_traj = QPushButton("⏹️ 停止")
        self._btn_stop_traj.setMinimumHeight(28)
        self._btn_stop_traj.clicked.connect(self._stop_trajectory_following)
        self._btn_stop_traj.setEnabled(False)
        play_btn_row.addWidget(self._btn_stop_traj)

        traj_path_layout.addLayout(play_btn_row)

        # 第四行：状态显示
        traj_status_row = QHBoxLayout()
        traj_path_layout.addWidget(QLabel("状态:"))
        self._lbl_traj_status = QLabel("空闲")
        self._lbl_traj_status.setStyleSheet("color: #868e96; font-weight: bold;")
        traj_path_layout.addWidget(self._lbl_traj_status)
        traj_path_layout.addStretch()

        # 第五行：路径点显示
        traj_points_row = QHBoxLayout()
        traj_path_layout.addWidget(QLabel("记录点数:"))
        self._lbl_traj_points = QLabel("0")
        self._lbl_traj_points.setStyleSheet("color: #868e96;")
        traj_path_layout.addWidget(self._lbl_traj_points)
        traj_path_layout.addStretch()

        # 第六行：进度显示
        traj_progress_row = QHBoxLayout()
        traj_path_layout.addWidget(QLabel("跟随进度:"))
        self._lbl_traj_progress = QLabel("0/0")
        self._lbl_traj_progress.setStyleSheet("color: #868e96;")
        traj_path_layout.addWidget(self._lbl_traj_progress)
        traj_path_layout.addStretch()

        traj_path_layout.addStretch()
        vbox.addWidget(traj_path_group)

        # ── 偏移修正组 ──────────────────────────────────────────────────
        offset_group = QGroupBox("截图区域偏移修正")
        offset_layout = QHBoxLayout(offset_group)

        offset_layout.addWidget(QLabel("X 偏移:"))
        self._spin_ox = QSpinBox()
        self._spin_ox.setRange(-2000, 2000)
        self._spin_ox.setValue(0)
        offset_layout.addWidget(self._spin_ox)

        offset_layout.addWidget(QLabel("Y 偏移:"))
        self._spin_oy = QSpinBox()
        self._spin_oy.setRange(-2000, 2000)
        self._spin_oy.setValue(0)
        offset_layout.addWidget(self._spin_oy)

        self._btn_apply_offset = QPushButton("应用修正")
        self._btn_apply_offset.clicked.connect(self._apply_offset)
        offset_layout.addWidget(self._btn_apply_offset)

        vbox.addWidget(offset_group)

        # ── 轨迹绘制 ────────────────────────────────────────────────────
        traj_row = QHBoxLayout()
        self._chk_traj = QCheckBox("绘制轨迹（红线标注移动路径）")
        self._chk_traj.setChecked(False)
        self._chk_traj.toggled.connect(self._on_traj_toggled)
        traj_row.addWidget(self._chk_traj)
        traj_row.addStretch()
        vbox.addLayout(traj_row)

        # buttons
        btns = QHBoxLayout()
        self._btn_cap = QPushButton("捕获窗口")
        self._btn_bind = QPushButton("绑定窗口")
        self._btn_snap = QPushButton("截取初始地图")
        self._btn_start = QPushButton("开始拼图")
        self._btn_stop = QPushButton("结束拼图")
        for b in (self._btn_cap, self._btn_bind,
                  self._btn_snap, self._btn_start, self._btn_stop):
            b.setMinimumHeight(32)
            btns.addWidget(b)
        vbox.addLayout(btns)

        # log
        vbox.addWidget(QLabel("日志："))
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumHeight(170)
        self._log_box.setStyleSheet(
            "background:#0d1117; color:#c9d1d9; font-family:Consolas;")
        vbox.addWidget(self._log_box)

        # connect
        self._btn_cap.clicked.connect(self._capture_click)
        self._btn_bind.clicked.connect(self._do_bind)
        self._btn_snap.clicked.connect(self._do_snapshot)
        self._btn_start.clicked.connect(self._do_start)
        self._btn_stop.clicked.connect(self._do_stop)

        self._refresh_btns()

    def _log(self, msg: str):
        print(msg)  # Print to console for debugging
        self._log_box.append(msg)
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())

    def _refresh_btns(self):
        ok_hwnd = self._window_mgr.is_valid()
        ok_region = self._map_region is not None
        stitching = self._worker is not None
        yolo_loaded = self._yolo_loaded
        self._btn_bind.setEnabled(self._last_hwnd is not None)
        self._btn_snap.setEnabled(ok_hwnd)
        self._btn_start.setEnabled(ok_hwnd and ok_region and not stitching)
        self._btn_stop.setEnabled(stitching)

        # 战斗按钮：需要 YOLO 模型已加载，且要么拼图在运行，要么已加载地图轨迹
        self._btn_start_combat.setEnabled(yolo_loaded and ok_hwnd and (stitching or self._trajectory_loaded) and (
                    self._combat_worker is None or not self._combat_worker.is_running()))

        # 轨迹寻路按钮
        if ok_hwnd:
            self._btn_record_traj.setEnabled(True)
            self._btn_save_traj.setEnabled(True)
            # 只要窗口已绑定且轨迹已加载，就可以开始跟随（不需要先拼图）
            self._btn_play_traj.setEnabled(self._trajectory_loaded)
        else:
            self._btn_record_traj.setEnabled(False)
            self._btn_save_traj.setEnabled(False)
            self._btn_play_traj.setEnabled(False)

    def _on_traj_toggled(self, checked: bool):
        self._canvas.set_draw_trajectory(checked)

    def _show_detection_viewer(self):
        """显示独立检测查看器"""
        if self._detection_viewer is None:
            self._detection_viewer = DetectionViewerWindow(self)

        # 无论是否有数据，都先显示窗口
        self._detection_viewer.show()
        self._detection_viewer.raise_()
        self._detection_viewer.activateWindow()

        # 更新最新数据（在窗口显示后更新，确保能触发刷新）
        if self._current_frame is not None:
            self._detection_viewer.update_frame(
                self._current_frame,
                self._current_detections,
                self._detection_fps
            )
            
    def _open_contour_calibrator(self):
        """打开轮廓标定工具"""
        if self._contour_calibrator is None:
            self._contour_calibrator = ContourCalibrator(self)
            
        self._contour_calibrator.show()
        self._contour_calibrator.raise_()
        self._contour_calibrator.activateWindow()
        self._log("[标定] 轮廓标定工具已打开")

    # ── YOLO 相关 ────────────────────────────────────────────────────────

    def _load_yolo_model(self):
        """加载 YOLO 模型"""
        model_path = r"G:\Auto_Lotro\best.onnx"

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", f"模型文件不存在:\n{model_path}")
            self._log(f"[YOLO] ✗ 模型文件不存在")
            return

        try:
            self._log(f"[YOLO] 正在加载模型...")
            self._log(f"[YOLO] 模型路径：{model_path}")
            # 直接加载模型，完全按照用户的方式
            self._yolo_model = YOLO(model_path)
            self._yolo_loaded = True
            self._lbl_model_status.setText("✓ 已加载")
            self._lbl_model_status.setStyleSheet("color: #51cf66;")
            self._btn_start_detect.setEnabled(True)
            self._log(f"[YOLO] ✓ 模型加载成功")

            # 如果 Worker 已经存在（先启动拼图后加载模型的情况），更新模型
            if self._worker and self._yolo_model:
                self._worker.set_yolo_model(self._yolo_model)
                self._log("[YOLO] 已更新模型到 Worker")

                # 如果检测已启用，立即开始检测
                if self._yolo_detecting:
                    self._worker.set_yolo_detecting(True)
                    self._log("[YOLO] 检测已启动")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败:\n{e}")
            self._log(f"[YOLO] ✗ 加载异常：{e}")

    def _start_detection(self):
        """开始检测"""
        if not self._yolo_loaded:
            self._log("[YOLO] 请先加载模型")
            return

        self._yolo_detecting = True
        self._btn_start_detect.setEnabled(False)
        self._btn_stop_detect.setEnabled(True)

        # 输出绑定窗口的句柄
        if self._window_mgr.is_valid():
            self._log(f"[YOLO] 检测已启动，阈值：{self._spin_conf.value():.2f}")
            self._log(f"[YOLO] 绑定窗口句柄：{self._window_mgr.hwnd}")
        else:
            self._log("[YOLO] 警告：窗口未绑定！")

        # 通知 Worker 线程开始检测
        if self._worker:
            self._worker.set_yolo_detecting(True)
            self._log("[YOLO] 已通知 Worker 开始检测")
        else:
            self._log("[YOLO] 提示：拼图未启动，检测将在拼图启动后生效")

    def _stop_detection(self):
        """关闭检测"""
        self._yolo_detecting = False
        self._btn_start_detect.setEnabled(True)
        self._btn_stop_detect.setEnabled(False)
        self._canvas.set_detections(None, False)
        self._current_detections = []
        self._current_frame = None
        self._detection_fps = 0.0
        self._log("[YOLO] 检测已关闭")

        # 通知检测窗口清空显示
        if self._detection_viewer and self._detection_viewer.isVisible():
            self._detection_viewer.update_frame(None, [], 0.0)

        # 通知 Worker 线程停止检测
        if self._worker:
            self._worker.set_yolo_detecting(False)
            self._log("[YOLO] 已通知 Worker 停止检测")

    # ── 自动战斗相关 ────────────────────────────────────────────────────────

    def _update_combat_config(self):
        """更新战斗配置"""
        skill_count = self._edit_skill_keys.value()
        skill_keys = [str(i) for i in range(1, skill_count + 1)]
        self._combat_config.skill_keys = skill_keys
        self._combat_config.attack_range = self._spin_attack_range.value()

    def _load_map_data(self):
        """加载拼图地图数据（包含 JSON 和图片）"""
        save_dir = r"G:\map"
        os.makedirs(save_dir, exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载地图数据", save_dir, "JSON 文件 (*.json)"
        )

        if not file_path:
            return

        try:
            # 1. 加载 JSON 数据
            with open(file_path, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)

            # 2. 尝试加载对应的图片
            img_path = file_path.replace(".json", ".jpg")
            if not os.path.exists(img_path):
                img_path = file_path.replace(".json", ".png")

            if os.path.exists(img_path):
                self._loaded_canvas = cv2.imread(img_path)
                self._log(f"[地图] ✓ 已加载地图图片：{os.path.basename(img_path)}")
            else:
                self._log(f"[地图] ✗ 未找到对应的地图图片：{os.path.basename(img_path)}")
                self._loaded_canvas = None

            # 3. 提取元数据
            metadata = data.get("metadata", {})
            self._map_region = metadata.get("map_region")
            self._canvas_offset = metadata.get("canvas_offset", [0, 0])

            # 4. 加载到战斗核心
            if self._combat_worker is None:
                self._combat_worker = CombatWorker()
                self._combat_worker.skill_signal.connect(self._on_skill_pressed)
                self._combat_worker.click_signal.connect(self._on_monster_clicked)
                self._combat_worker.status_signal.connect(self._on_combat_status_updated)
                self._combat_worker.target_signal.connect(self._on_combat_target_updated)
                self._combat_worker.log_signal.connect(self._log)

            if self._combat_worker._combat_core.load_map_json(file_path):
                self._trajectory_loaded = True
                self._log(f"[战斗] ✓ 已同步寻路轨迹")

            # 5. 加载到轨迹管理器（用于非战斗寻路）
            traj_data = data.get("trajectory", {})
            points = traj_data.get("points", [])
            if points:
                from core.trajectory_path import TrajectoryPath
                trajectory = TrajectoryPath(points=[(p[0], p[1]) for p in points])
                # 使用文件名作为轨迹名称
                self._trajectory_name = os.path.splitext(os.path.basename(file_path))[0]
                self._traj_manager._saved_trajectories[self._trajectory_name] = trajectory
                self._edit_traj_name.setText(self._trajectory_name)
                self._log(f"[轨迹] ✓ 已加载独立寻路轨迹：{self._trajectory_name}")

            self._refresh_btns()
            self._log(f"[地图] ✓ 完整地图数据加载成功")

            # 在 canvas2 上显示地图和路线
            self._init_canvas2()

        except Exception as e:
            self._log(f"[地图] ✗ 加载失败：{e}")
            import traceback
            self._log(traceback.format_exc())

    def _init_canvas2(self):
        """加载地图后在 canvas2 上显示地图和路线"""
        if self._loaded_canvas is None:
            return
        self._canvas2.reset()
        self._canvas2.set_external_canvas(self._loaded_canvas)
        # 绘制路线（轨迹点已是图片相对坐标）
        if self._trajectory_name:
            traj = self._traj_manager.load_trajectory(self._trajectory_name)
            if traj and traj.points:
                self._canvas2.set_route(traj.points)
        self._log("[画布2] 地图和路线已显示")

    def _tick_canvas2(self):
        """定时刷新 canvas2 的玩家位置标记"""
        if self._minimap is None or self._loaded_canvas is None:
            return
        player_pos = self._minimap.get_player_position()
        if player_pos:
            # 原始画布坐标 → 图片相对坐标（减去裁剪时记录的偏移）
            c0 = int(self._canvas_offset[0])
            r0 = int(self._canvas_offset[1])
            self._canvas2.set_player_marker(player_pos[0] - c0,
                                            player_pos[1] - r0)

    def _start_combat(self):
        """启动自动战斗"""
        if not self._yolo_loaded:
            self._log("[战斗] 请先加载 YOLO 模型")
            return

        if not self._window_mgr.is_valid():
            self._log("[战斗] 请先绑定窗口")
            return

        # --- 独立运行支持（寻路打怪） ---
        if self._worker is None:
            if hasattr(self, '_loaded_canvas') and self._loaded_canvas is not None and self._map_region:
                self._log("[战斗] 检测到拼图未启动，正在启动仅匹配模式进行寻路打怪...")

                # 初始化 MiniMap 并设置已加载的画布
                self._minimap = MiniMap()
                offset = getattr(self, '_canvas_offset', (0, 0))
                self._minimap.set_canvas(self._loaded_canvas, offset)

                # 创建 StitchWorker
                self._worker = StitchWorker(self._window_mgr, self._map_region, self._minimap, self._yolo_model)
                self._worker.set_match_only(True)

                # 连接信号
                self._worker.log_signal.connect(self._log)
                self._worker.detections_signal.connect(self._on_detections_received)
                self._worker.frame_signal.connect(self._on_frame_received)
                self._worker.finished.connect(self._on_worker_done)

                # 启动 Worker
                self._worker.start()
                self._canvas_timer.start(100)
            else:
                self._log("[战斗] ✗ 无法启动：请先开始拼图或加载包含地图图片的 JSON")
                return

        # 更新配置
        self._update_combat_config()

        # 创建战斗 Worker
        if self._combat_worker is None:
            self._combat_worker = CombatWorker()
            self._combat_worker.skill_signal.connect(self._on_skill_pressed)
            self._combat_worker.click_signal.connect(self._on_monster_clicked)
            self._combat_worker.status_signal.connect(self._on_combat_status_updated)
            self._combat_worker.target_signal.connect(self._on_combat_target_updated)
            self._combat_worker.log_signal.connect(self._log)

        # 设置配置和玩家位置
        self._combat_worker.set_config(self._combat_config)

        # 设置玩家位置
        if self._minimap:
            pos = self._minimap.get_player_position()
            if pos:
                self._combat_worker.set_player_position(pos[0], pos[1])
            else:
                self._combat_worker.set_player_position(640, 360)
        else:
            self._combat_worker.set_player_position(640, 360)

        # 启动战斗
        self._combat_worker.start()

        # 强制开启 YOLO 检测（如果是自动打怪，必须开启检测）
        self._yolo_detecting = True
        if self._worker:
            self._worker.set_yolo_detecting(True)
            self._log("[战斗] 已强制开启 YOLO 检测")

        # 更新 UI 状态
        self._btn_start_combat.setEnabled(False)
        self._btn_stop_combat.setEnabled(True)
        self._btn_start_detect.setEnabled(False)
        self._btn_stop_detect.setEnabled(False)

        self._log("[战斗] 自动战斗已启动")

    def _stop_combat(self):
        """停止自动战斗"""
        if self._combat_worker:
            self._combat_worker.stop()

        # 更新 UI 状态
        self._btn_start_combat.setEnabled(True)
        self._btn_stop_combat.setEnabled(False)
        self._btn_start_detect.setEnabled(True)

        # 重置状态显示
        self._lbl_combat_status.setText("空闲")
        self._lbl_combat_target.setText("无")

        self._log("[战斗] 自动战斗已停止")

    def _on_skill_pressed(self, skill_key: str):
        """技能按键按下"""
        from core.combat_worker import KeySimulator
        KeySimulator.press_key(skill_key)
        self._log(f"[战斗] 释放技能：{skill_key}")

    def _on_monster_clicked(self, x: float, y: float):
        """点击怪物"""
        from core.combat_worker import KeySimulator
        # 将检测坐标转换为屏幕坐标
        if self._window_mgr.is_valid():
            # 获取客户区左上角在屏幕上的绝对坐标
            client_left, client_top = self._window_mgr.client_to_screen(0, 0)

            # 加上偏移得到屏幕坐标
            screen_x = int(client_left + x)
            screen_y = int(client_top + y)

            KeySimulator.click_at(screen_x, screen_y)
            self._log(f"[战斗] 点击怪物位置：({screen_x}, {screen_y})")

    def _on_combat_status_updated(self, status: str):
        """战斗状态更新"""
        self._lbl_combat_status.setText(status)

        # 根据状态改变颜色
        if status == "空闲":
            self._lbl_combat_status.setStyleSheet("color: #868e96; font-weight: bold;")
        elif status == "搜索中":
            self._lbl_combat_status.setStyleSheet("color: #fcc419; font-weight: bold;")
        elif status == "巡逻中":
            self._lbl_combat_status.setStyleSheet("color: #ae3ec9; font-weight: bold;")
        elif status == "锁定目标":
            self._lbl_combat_status.setStyleSheet("color: #20c997; font-weight: bold;")
        elif status == "移动中":
            self._lbl_combat_status.setStyleSheet("color: #339af0; font-weight: bold;")
        elif status == "战斗中":
            self._lbl_combat_status.setStyleSheet("color: #ff6b6b; font-weight: bold;")

    def _on_combat_target_updated(self, target: str):
        """战斗目标更新"""
        self._lbl_combat_target.setText(target)

    # ── 轨迹寻路相关 ────────────────────────────────────────────────────────

    def _toggle_recording(self):
        """切换记录状态"""
        if self._traj_manager.is_recording():
            # 停止记录
            self._traj_manager.stop_recording()
            self._btn_record_traj.setText("🔴 开始记录")
            self._btn_save_traj.setEnabled(True)
            self._lbl_traj_status.setText("记录停止")
            self._lbl_traj_status.setStyleSheet("color: #fcc419; font-weight: bold;")
            self._log(f"[轨迹] 记录停止，共记录 {self._traj_manager.recorded_point_count} 个点")
        else:
            # 开始记录
            self._traj_manager.start_recording()
            self._btn_record_traj.setText("⏹️ 停止记录")
            self._btn_save_traj.setEnabled(False)
            self._lbl_traj_status.setText("记录中...")
            self._lbl_traj_status.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            self._log("[轨迹] 开始记录移动路径")

    def _save_trajectory(self):
        """保存当前轨迹到内存并导出到文件"""
        trajectory_name = self._edit_traj_name.toPlainText().strip()
        if not trajectory_name:
            trajectory_name = f"manual_route_{datetime.now().strftime('%H%M%S')}"

        if self._traj_manager.save_current_trajectory(trajectory_name):
            trajectory = self._traj_manager.load_trajectory(trajectory_name)

            # 同时保存到 G:\map 目录
            save_dir = r"G:\map"
            os.makedirs(save_dir, exist_ok=True)

            # 使用 MapDataSaver 格式保存
            json_path = os.path.join(save_dir, f"{trajectory_name}.json")

            data = {
                "version": "1.0",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "name": trajectory_name,
                "trajectory": {
                    "points": [[float(p[0]), float(p[1])] for p in trajectory.points],
                    "point_count": len(trajectory.points)
                }
            }

            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(data, f, ensure_ascii=False, indent=2)

                self._log(f"[轨迹] ✓ 已保存到内存和文件：{os.path.basename(json_path)}")
                self._btn_play_traj.setEnabled(True)
                self._lbl_traj_status.setText("已保存")
                self._lbl_traj_status.setStyleSheet("color: #20c997; font-weight: bold;")
            except Exception as e:
                self._log(f"[轨迹] ✗ 保存文件失败：{e}")
        else:
            self._log("[轨迹] ✗ 保存失败：没有记录的轨迹")

    def _start_trajectory_following(self):
        """开始跟随轨迹（不依赖拼图是否已启动）"""
        trajectory_name = self._edit_traj_name.toPlainText().strip()
        if not trajectory_name:
            self._log("[轨迹] 请输入轨迹名称")
            return

        trajectory = self._traj_manager.load_trajectory(trajectory_name)
        if not trajectory:
            self._log(f"[轨迹] 未找到轨迹：{trajectory_name}")
            return

        # 若拼图 Worker 未运行，自动启动仅匹配模式（无需截图/开始拼图）
        if self._worker is None:
            if self._loaded_canvas is None or not self._map_region:
                self._log("[轨迹] ✗ 请先加载包含地图图片的 JSON 文件")
                return

            self._log("[轨迹] 启动仅匹配模式（无需截图/拼图）...")
            self._minimap = MiniMap()
            self._minimap.set_canvas(self._loaded_canvas, self._canvas_offset)

            self._worker = StitchWorker(
                self._window_mgr, self._map_region, self._minimap, self._yolo_model)
            self._worker.set_match_only(True)
            self._worker.log_signal.connect(self._log)
            self._worker.detections_signal.connect(self._on_detections_received)
            self._worker.frame_signal.connect(self._on_frame_received)
            self._worker.finished.connect(self._on_worker_done)
            self._worker.start()
            self._log("[轨迹] 仅匹配 Worker 已启动")

        # 开始跟随
        self._traj_manager.start_following(trajectory)
        self._lbl_traj_status.setText("跟随中")
        self._lbl_traj_status.setStyleSheet("color: #339af0; font-weight: bold;")
        self._btn_play_traj.setEnabled(False)
        self._btn_stop_traj.setEnabled(True)
        self._log(f"[轨迹] ▶️ 开始跟随：{trajectory_name}")

        # 启动跟随更新定时器
        if not hasattr(self, "_traj_follow_timer") or self._traj_follow_timer is None:
            self._traj_follow_timer = QTimer(self)
            self._traj_follow_timer.setInterval(200)
            self._traj_follow_timer.timeout.connect(self._update_trajectory_following)
        self._traj_follow_timer.start()

        # 启动 canvas2 刷新定时器
        self._canvas2_timer.start()

    def _stop_trajectory_following(self):
        """停止跟随轨迹"""
        self._log("[轨迹] ⏹️ 停止跟随")

        if hasattr(self, "_traj_follow_timer") and self._traj_follow_timer:
            self._traj_follow_timer.stop()
            self._traj_follow_timer = None

        self._canvas2_timer.stop()
        self._traj_manager.stop_following()
        self._canvas2.clear_player_marker()

        self._lbl_traj_status.setText("已停止")
        self._lbl_traj_status.setStyleSheet("color: #868e96; font-weight: bold;")
        self._lbl_traj_progress.setText("0/0")
        self._btn_play_traj.setEnabled(True)
        self._btn_stop_traj.setEnabled(False)

    def _update_trajectory_following(self):
        """更新轨迹跟随（坐标系：轨迹点 = 图片相对坐标）"""
        if not self._traj_manager.is_following():
            self._stop_trajectory_following()
            return

        if not self._minimap:
            self._log("[轨迹] 警告：定位系统未启动")
            return

        player_pos = self._minimap.get_player_position()
        if not player_pos:
            return  # 匹配失败时静默跳过，不刷屏日志

        # 原始画布坐标 → 图片相对坐标（轨迹点的坐标系）
        c0 = int(self._canvas_offset[0])
        r0 = int(self._canvas_offset[1])
        traj_x = player_pos[0] - c0
        traj_y = player_pos[1] - r0

        player_angle = self._minimap.get_player_angle()
        move_cmd = self._traj_manager.update(traj_x, traj_y, player_angle)

        if move_cmd:
            from core.combat_worker import KeySimulator
            KeySimulator.press_key(move_cmd)

        current, total = self._traj_manager.get_progress()
        self._lbl_traj_progress.setText(f"{current}/{total}")

        # ── 偏移修正 ──────────────────────────────────────────────────────────

    def _apply_offset(self):
        self._offset_x = self._spin_ox.value()
        self._offset_y = self._spin_oy.value()
        if self._raw_region is not None:
            rx, ry, rw, rh = self._raw_region
            self._map_region = (rx + self._offset_x,
                                ry + self._offset_y, rw, rh)
            self._log(f"[偏移] 已修正  x={self._map_region[0]} y={self._map_region[1]}")
            if self._window_mgr.is_valid():
                try:
                    img = bitblt_capture(self._window_mgr.hwnd, *self._map_region)
                    self._first_frame = img
                    self._canvas.reset()
                    self._canvas.place_first_frame(img)
                    self._log(f"[偏移] 首帧已重新截取")
                except Exception as e:
                    self._log(f"[偏移] 重新截取失败：{e}")
        else:
            self._log("[偏移] 请先截取初始地图")
        self._refresh_btns()

    # ── 1. Capture ────────────────────────────────────────────────────────

    def _capture_click(self):
        """改进的窗口捕获"""
        self._capturing = True
        self._btn_cap.setText("将鼠标移到目标窗口，按 F1...")
        self._hover_timer.start()
        self._log("[捕获] 将鼠标移到游戏窗口，然后按 F1")

    def keyPressEvent(self, event: QKeyEvent):
        if self._capturing and event.key() == Qt.Key.Key_F1:
            self._capturing = False
            self._hover_timer.stop()
            self._overlay.hide_overlay()
            self._btn_cap.setText("捕获窗口")

            # 直接通过鼠标位置绑定
            if self._window_mgr.bind_by_cursor():
                self._log(f"[捕获] ✓ 成功捕获窗口")
                self._log(f"[捕获]   标题：{self._window_mgr.title}")
                self._log(f"[捕获]   句柄：{self._window_mgr.hwnd}")

                # 保存信息供后续使用
                self._last_hwnd = self._window_mgr.hwnd
                self._last_pid = str(self._window_mgr.pid)
                self._last_title = self._window_mgr.title
                self._last_cls = self._window_mgr.class_name
            else:
                self._log("[捕获] ✗ 未找到有效窗口")

            self._refresh_btns()
        else:
            super().keyPressEvent(event)

    def _poll_hover(self):
        hwnd, _px, _py = get_window_at_cursor()
        if hwnd:
            l, t, r, b = get_window_rect(hwnd)
            self._overlay.set_rect(l, t, r - l, b - t)

    # ── 2. Bind ───────────────────────────────────────────────────────────

    def _do_bind(self):
        """改进的绑定方法"""
        # 方法 1: 使用上次捕获的窗口信息
        if hasattr(self, '_last_hwnd') and self._last_hwnd:
            if self._window_mgr.bind_by_hwnd(self._last_hwnd):
                self._hwnd = self._window_mgr.hwnd
                self._log(f"[绑定] ✓ 成功绑定窗口")
                self._log(f"[绑定]   句柄：{self._window_mgr.hwnd}")
                self._log(f"[绑定]   标题：{self._window_mgr.title}")
                self._log(f"[绑定]   类名：{self._window_mgr.class_name}")
                self._log(f"[绑定]   进程：{self._window_mgr.pid}")

                # 输出窗口信息
                info = self._window_mgr.get_info()
                self._log(
                    f"[绑定]   窗口：{info['window_rect'][2] - info['window_rect'][0]}x{info['window_rect'][3] - info['window_rect'][1]}")
                self._log(
                    f"[绑定]   客户区：{info['client_rect'][2] - info['client_rect'][0]}x{info['client_rect'][3] - info['client_rect'][1]}")

                self._refresh_btns()
                return

        # 方法 2: 尝试通过 PID 绑定（更可靠）
        if hasattr(self, '_last_pid') and self._last_pid:
            self._log(f"[绑定] 尝试通过 PID {self._last_pid} 查找窗口...")
            if self._window_mgr.bind_by_pid(int(self._last_pid)):
                self._hwnd = self._window_mgr.hwnd
                self._log(f"[绑定] ✓ PID 查找成功")
                self._log(f"[绑定]   找到窗口：{self._window_mgr.title}")
                self._refresh_btns()
                return

        self._log("[绑定] ✗ 无法绑定窗口，请重新捕获")

    # ── 3. Snapshot ───────────────────────────────────────────────────────

    def _do_snapshot(self):
        if not self._window_mgr.is_valid():
            self._log("[截图] 请先绑定目标窗口")
            return
        self._log("[截图] 请框选小地图区域，Enter 确认，Esc 取消")
        self._map_region = None
        self._raw_region = None
        self._first_frame = None
        self._canvas.reset()
        self._region_sel.showFullScreen()
        self._region_sel.activateWindow()
        self._region_sel.raise_()

    def _on_region_selected(self, sx, sy, sw, sh):
        if not self._hwnd:
            self._log("[截图] 窗口未绑定，取消")
            return

        # 验证窗口是否有效
        if not self._window_mgr.is_valid():
            self._log("[截图] 窗口未绑定或已失效")
            return

        # 获取窗口位置信息用于调试
        info = self._window_mgr.get_info()
        l, t, r, b = info['window_rect']
        client_l, client_t, client_r, client_b = info['client_rect']

        # 将屏幕坐标直接转换为客户区坐标（使用窗口管理器）
        rx, ry = self._window_mgr.screen_to_client(sx, sy)

        self._log(f"[截图] 屏幕坐标：sx={sx} sy={sy}")
        self._log(f"[截图] 窗口矩形：left={l} top={t} right={r} bottom={b}")
        self._log(f"[截图] 客户区屏幕坐标：left={client_l} top={client_t}")
        self._log(f"[截图] 转换后客户区坐标：rx={rx} ry={ry}")

        # 应用偏移修正
        rx += self._offset_x
        ry += self._offset_y

        self._raw_region = (rx - self._offset_x, ry - self._offset_y, sw, sh)
        self._map_region = (rx, ry, sw, sh)

        self._log(f"[截图] 最终区域 x={rx} y={ry} w={sw} h={sh}")

        try:
            img = bitblt_capture(self._window_mgr.hwnd, rx, ry, sw, sh)
        except Exception as e:
            self._log(f"[截图] 捕获失败：{e}")
            return

        self._first_frame = img
        # 确保在放置第一帧之前，画布是完全干净的状态
        self._canvas.reset()
        self._canvas.place_first_frame(img)

        self._log(f"[截图] 首帧已放置到画布，尺寸 {sw}×{sh}")
        self._refresh_btns()

    def _on_region_cancelled(self):
        """取消截取初始地图时清除坐标区域"""
        self._log("[截图] 已取消，清除坐标区域")

        # 清除坐标区域相关状态
        self._map_region = None
        self._raw_region = None
        self._first_frame = None

        # 重置画布
        self._canvas.reset()

        # 重置按钮状态
        self._refresh_btns()

    # ── 4. Start stitch ───────────────────────────────────────────────────

    def _do_start(self):
        if self._worker:
            self._log("[拼图] 已在运行中")
            return

        # 检查窗口管理器是否有效
        if not (self._window_mgr.is_valid() and self._map_region and self._first_frame is not None):
            self._log("[拼图] 请先绑定窗口并截取初始地图")
            return

        # 验证窗口是否仍然有效
        if not self._window_mgr.is_valid():
            self._log("[拼图] 窗口已失效，请重新绑定")
            return

        self._minimap = MiniMap()
        self._minimap.update(self._first_frame)

        self._canvas.set_external_canvas(self._minimap.canvas)
        self._canvas.set_trajectory_source(
            self._minimap.trajectory,
            self._chk_traj.isChecked()
        )

        # 使用窗口管理器创建 StitchWorker
        # 始终传入模型对象，让 Worker 根据 _yolo_detecting 标志控制检测
        yolo_model_for_worker = self._yolo_model  # 始终传入模型

        self._worker = StitchWorker(
            self._window_mgr, self._map_region, self._minimap, yolo_model_for_worker)

        # 设置置信度
        if yolo_model_for_worker is not None:
            self._worker.set_yolo_conf(self._spin_conf.value())
            # 传递当前的检测状态
            self._worker.set_yolo_detecting(self._yolo_detecting)
            if self._yolo_detecting:
                self._log("[拼图] 已启动 (包含 YOLO 检测)")
            else:
                self._log("[拼图] 已启动 (YOLO 检测已加载但未启用)")
        else:
            self._log("[拼图] 已启动 (纯拼图模式，未加载 YOLO)")

        self._worker.log_signal.connect(self._log)
        self._worker.detections_signal.connect(self._on_detections_received)
        self._worker.frame_signal.connect(self._on_frame_received)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

        self._canvas_timer.start()
        self._refresh_btns()

    def _on_frame_received(self, frame, detections, fps):
        """接收帧和检测结果"""
        self._current_frame = frame
        self._current_detections = detections
        self._detection_fps = fps

        # 更新独立检测查看器（如果已打开）
        if self._detection_viewer and self._detection_viewer.isVisible():
            self._detection_viewer.update_frame(frame, detections, fps)

    def _on_detections_received(self, detections):
        """接收检测结果并显示"""
        self._current_detections = detections

        # 在主窗口画布上显示（如果需要）
        if self._yolo_detecting:
            self._canvas.set_detections(detections, True)

        # 如果正在拼图，追加世界坐标后传递给战斗 worker
        if self._minimap:
            player_pos = self._minimap.get_player_position()
            if player_pos:
                # ── 坐标转换说明 ──────────────────────────────────────────────
                # YOLO 在全窗口图像上检测，det[6]/det[7] 是客户区屏幕像素坐标。
                # 点击怪物、攻击范围判断均需要屏幕坐标，不能修改 [6][7]。
                # 寻路 (A*) 使用画布世界坐标。
                # 因此扩展 tuple 为 11 元素：
                #   [0-8]  原始字段保持不变（cx/cy 仍为屏幕像素）
                #   [9]    world_x —— 画布世界坐标 X
                #   [10]   world_y —— 画布世界坐标 Y
                # ─────────────────────────────────────────────────────────────
                enriched_detections = []
                if self._map_region:
                    mx, my, mw, mh = self._map_region
                    for det in detections:
                        rel_x = det[6] - mx
                        rel_y = det[7] - my
                        wx, wy = self._minimap.pixel_to_world(rel_x, rel_y)
                        enriched_detections.append(tuple(det) + (wx, wy))
                else:
                    for det in detections:
                        enriched_detections.append(tuple(det) + (det[6], det[7]))

                if self._combat_worker and self._combat_worker.is_running():
                    self._combat_worker.set_player_position(player_pos[0], player_pos[1])
                    player_angle = self._minimap.get_player_angle()
                    if player_angle is not None:
                        self._combat_worker.set_player_angle(player_angle)
                    self._combat_worker.update_detections(enriched_detections)
        else:
            if self._combat_worker and self._combat_worker.is_running():
                self._combat_worker.update_detections(detections)

    def _tick_canvas(self):
        if self._minimap is not None:
            self._canvas.refresh_from_external()

            # 更新轨迹记录
            if self._traj_manager.is_recording():
                player_pos = self._minimap.get_player_position()
                if player_pos:
                    self._traj_manager.add_point(player_pos[0], player_pos[1])
                    self._lbl_traj_points.setText(str(self._traj_manager.recorded_point_count))

    def _on_worker_done(self):
        self._canvas_timer.stop()
        self._worker = None
        self._refresh_btns()

    # ── 5. Stop stitch ────────────────────────────────────────────────────

    def _do_stop(self):
        """结束拼图并保存数据"""
        self._log("[拼图] 正在停止...")
        if self._worker:
            self._worker.stop()
        self._canvas_timer.stop()

        # 确保保存目录存在
        save_dir = r"G:\map"
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                self._log(f"[保存] 创建目录：{save_dir}")
        except Exception as e:
            self._log(f"[保存错误] 无法创建目录 {save_dir}: {e}")
            # 如果 G 盘不可写，尝试备选路径
            save_dir = os.path.join(os.path.expanduser("~"), "Documents", "Auto_Lotro_Maps")
            os.makedirs(save_dir, exist_ok=True)
            self._log(f"[保存] 使用备选目录：{save_dir}")

        if self._minimap is not None:
            full_canvas = self._minimap.canvas
            trajectory = self._minimap.trajectory.copy() if self._minimap.trajectory else []
        else:
            full_canvas = self._canvas.get_canvas()
            trajectory = []

        # 1. 裁剪有效区域并计算偏移
        gray = cv2.cvtColor(full_canvas, cv2.COLOR_BGR2GRAY)
        rows = np.any(gray > 0, axis=1)
        cols = np.any(gray > 0, axis=0)

        c0, r0 = 0, 0
        if rows.any() and cols.any():
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            out = full_canvas[r0:r1 + 1, c0:c1 + 1]

            # 2. 重要：修正轨迹点偏移
            # 轨迹点原本是相对于 4000x4000 画布的坐标
            # 裁剪后，需要减去裁剪左上角的偏移量 (c0, r0)
            if trajectory:
                trajectory = [(x - c0, y - r0) for x, y in trajectory]
                self._log(f"[保存] 修正轨迹偏移：({c0}, {r0})")
        else:
            out = full_canvas

        # 3. 使用 MapDataSaver 保存数据
        metadata = {
            "window_title": self._window_mgr.title if self._window_mgr.is_valid() else "",
            "map_region": self._map_region,
            "saved_by": "Auto_Lotro3",
            "canvas_offset": [int(c0), int(r0)]
        }

        try:
            self._log("[保存] 正在写入文件...")
            # 如果输入了轨迹名称，使用该名称作为文件名，否则使用默认时间戳格式
            base_name = self._edit_traj_name.toPlainText().strip() or None

            image_path, json_path = MapDataSaver.save_map_data(
                save_dir=save_dir,
                canvas=out,
                trajectory=trajectory,
                metadata=metadata,
                base_name=base_name
            )
            self._log(f"[保存] ✓ 已保存图片：{os.path.basename(image_path)}")
            self._log(f"[保存] ✓ 已保存轨迹数据：{os.path.basename(json_path)}")

            if trajectory:
                self._log(f"[保存] 轨迹点数：{len(trajectory)}")
        except Exception as e:
            self._log(f"[保存错误] {e}")
            import traceback
            self._log(f"[保存错误] {traceback.format_exc()}")

        # 清除检测框和轨迹状态
        self._canvas.reset()

        # 清除相关状态
        self._minimap = None
        self._map_region = None
        self._raw_region = None
        self._first_frame = None
        self._current_detections = []
        self._current_frame = None

        # 重置轨迹勾选框
        self._chk_traj.setChecked(False)

        self._refresh_btns()
        self._log("[保存] 画布已重置")

    # ── cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
        self._overlay.hide()
        super().closeEvent(event)