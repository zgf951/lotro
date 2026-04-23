import numpy as np
import cv2
import time
import threading
from typing import Optional, Tuple
from PySide6.QtCore import QObject, Signal
from utils.win32_utils import bitblt_capture, get_window_rect
from utils.window_manager import WindowManager
import lotro_arrow_v5 as arrow_v5


# ── MiniMap 拼图核心 ───────────────────────────────────────────────────────────

class MiniMap:
    def __init__(self, canvas_h=4000, canvas_w=4000):
        self.canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        # 增加特征点数量并调整 SIFT 参数以提高匹配精度
        self.sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=15, sigma=1.6)
        # 使用 FLANN 匹配器以提高匹配速度和精度
        index_params = dict(algorithm=1, trees=8)
        search_params = dict(checks=150)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.canvas_cx = canvas_w // 2
        self.canvas_cy = canvas_h // 2
        self.first_frame = True

        self._M_global = None
        self._prev_kp = None
        self._prev_des = None
        self.trajectory = []

        # 玩家朝向 (游戏角度，北=0，顺时针)
        self._facing_angle = 0.0
        self._facing_ready = False

    def set_canvas(self, canvas: np.ndarray, offset: Tuple[int, int] = (0, 0)):
        """设置预载入的画布，用于基于已有地图的寻路"""
        self.canvas = canvas.copy()
        # 记录偏移，用于坐标对齐
        self._canvas_offset = offset
        # 标记不再是首帧，但需要重新计算特征点
        self.first_frame = False

        # 计算整张画布的特征点，作为匹配基准
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        self._base_kp, self._base_des = self.sift.detectAndCompute(gray, None)

    def update_match_only(self, img: np.ndarray) -> Optional[Tuple[float, float]]:
        """仅进行匹配，不更新画布。返回玩家在画布上的世界坐标"""
        if self._base_des is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 4:
            return None

        # 与基准地图进行匹配
        try:
            matches = self.matcher.knnMatch(des, self._base_des, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            if len(good) < 8:
                return None

            src = np.float32([kp[m.queryIdx].pt for m in good])
            dst = np.float32([self._base_kp[m.trainIdx].pt for m in good])

            M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
            if M is not None:
                # 计算输入图中心在基准图上的位置
                h, w = img.shape[:2]
                cx_f, cy_f = w / 2.0, h / 2.0
                cx = M[0, 0] * cx_f + M[0, 1] * cy_f + M[0, 2]
                cy = M[1, 0] * cx_f + M[1, 1] * cy_f + M[1, 2]

                # 加上画布偏移得到世界坐标
                return cx + self._canvas_offset[0], cy + self._canvas_offset[1]
        except:
            pass
        return None

    def update(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        h, w = img.shape[:2]

        if self.first_frame or des is None or len(kp) < 4:
            tx = self.canvas_cx - w / 2
            ty = self.canvas_cy - h / 2
            self._M_global = np.array([[1, 0, tx],
                                       [0, 1, ty]], dtype=np.float64)
            self._paste_with_M(img, self._M_global)
            cx, cy = self._frame_center_on_canvas(w, h)
            self.trajectory = [(cx, cy)]
            self._prev_kp = kp
            self._prev_des = des
            self.first_frame = False
            # 第一帧也返回 True，表示初始化成功
            return True

        if self._prev_des is None:
            return False

        M_inc = self._estimate_affine(
            self._prev_kp, self._prev_des, kp, des)

        if M_inc is not None:
            M_inc_inv = self._invert_affine(M_inc)
            self._M_global = self._compose_affine(self._M_global, M_inc_inv)

        self._paste_with_M(img, self._M_global)
        cx, cy = self._frame_center_on_canvas(w, h)
        self.trajectory.append((cx, cy))
        self._prev_kp = kp
        self._prev_des = des
        return M_inc is not None

    def get_player_position(self) -> Optional[Tuple[float, float]]:
        """获取玩家当前在画布上的坐标"""
        if not self.trajectory:
            return None
        return self.trajectory[-1]

    def get_player_angle(self) -> Optional[float]:
        """获取玩家当前朝向 (游戏角度)"""
        if self._facing_ready:
            return self._facing_angle
        return None

    def pixel_to_world(self, x: float, y: float) -> Tuple[float, float]:
        """将相对于当前帧（拼图输入图）的像素坐标转换为画布世界坐标"""
        if self._M_global is None:
            return x, y

        M = self._M_global
        world_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        world_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return world_x, world_y

    def _frame_center_on_canvas(self, fw, fh):
        cx_f = fw / 2.0
        cy_f = fh / 2.0
        M = self._M_global
        cx = M[0, 0] * cx_f + M[0, 1] * cy_f + M[0, 2]
        cy = M[1, 0] * cx_f + M[1, 1] * cy_f + M[1, 2]
        return int(round(cx)), int(round(cy))

    def _paste_with_M(self, img, M):
        ch, cw = self.canvas.shape[:2]
        warped = cv2.warpAffine(img, M, (cw, ch),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        canvas_empty = np.all(self.canvas == 0, axis=2)
        warped_valid = np.any(warped != 0, axis=2)
        write_mask = canvas_empty & warped_valid
        self.canvas[write_mask] = warped[write_mask]

    def _estimate_affine(self, kp1, des1, kp2, des2):
        try:
            # 使用 FLANN 匹配器的 knnMatch 方法
            matches = self.matcher.knnMatch(des1, des2, k=2)
        except Exception:
            return None
        # 调整匹配阈值以提高匹配质量
        good = [m for m, n in matches if m.distance < 0.65 * n.distance]
        if len(good) < 8:
            return None
        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        # 调整 RANSAC 参数以提高匹配精度
        M, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=2000, confidence=0.999)
        if M is None or (inliers is not None and inliers.sum() < 10):
            return None
        return M

    @staticmethod
    def _invert_affine(M):
        M3 = np.vstack([M, [0, 0, 1]])
        M3_inv = np.linalg.inv(M3)
        return M3_inv[:2]

    @staticmethod
    def _compose_affine(M1, M2):
        M1_3 = np.vstack([M1, [0, 0, 1]])
        M2_3 = np.vstack([M2, [0, 0, 1]])
        return (M1_3 @ M2_3)[:2]


# ── Stitching worker（完全按照用户的方式）───────────────────────────────────

class StitchWorker(QObject):
    log_signal = Signal(str)
    detections_signal = Signal(list)
    frame_signal = Signal(object, list, float)  # frame, detections, fps
    finished = Signal()

    _CAPTURE_FPS = 4

    def __init__(self, window_manager: WindowManager, region, minimap: MiniMap, yolo_model=None):
        super().__init__()
        self._window_mgr = window_manager  # 使用窗口管理器
        self._region = region  # 小地图区域
        self._minimap = minimap
        self._yolo_model = yolo_model  # 直接是 YOLO 模型对象
        self._yolo_conf = 0.5
        self._running = False
        self._yolo_detecting = False
        self._match_only = False  # 新增：是否仅进行匹配

        # 获取客户区区域用于检测（更准确，排除标题栏和边框）
        if window_manager.is_valid():
            info = window_manager.get_info()
            client_l, client_t, client_r, client_b = info['client_rect']
            client_w = client_r - client_l
            client_h = client_b - client_t
            self._detect_region = (0, 0, client_w, client_h)  # 客户区区域
            # 输出初始化信息
            self.log_signal.emit(f"[拼图] 初始化完成，窗口句柄：{window_manager.hwnd}")
            self.log_signal.emit(f"[拼图] 小地图区域：{region}")
            self.log_signal.emit(f"[拼图] 检测区域：{self._detect_region} (客户区)")
        else:
            self.log_signal.emit("[拼图] 警告：窗口管理器未绑定有效窗口")
            self._detect_region = (0, 0, 800, 600)  # 默认值

    def set_yolo_conf(self, conf: float):
        """设置 YOLO 置信度"""
        self._yolo_conf = conf

    def set_yolo_detecting(self, detecting: bool):
        """设置是否启用检测"""
        self._yolo_detecting = detecting

    def set_yolo_model(self, model):
        """动态设置 YOLO 模型（支持运行时加载）"""
        self._yolo_model = model
        self.log_signal.emit("[YOLO] 模型已更新")

    def set_match_only(self, match_only: bool):
        """设置是否仅进行匹配"""
        self._match_only = match_only
        if match_only:
            self.log_signal.emit("[拼图] 已切换到仅匹配模式（用于已知地图寻路）")

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False

    def _run(self):
        # 拼图使用小地图区域
        map_x, map_y, map_w, map_h = self._region
        # 检测使用整个窗口区域
        detect_x, detect_y, detect_w, detect_h = self._detect_region
        interval = 1.0 / self._CAPTURE_FPS
        prev_img = None
        last_detection_log = 0
        consecutive_failures = 0  # 连续失败计数器
        detection_logged = False  # 是否已输出检测信息
        frame_count = 0  # 帧计数器，用于控制检测频率

        self.log_signal.emit("[拼图] 线程启动")

        self._arrow_detector = True
        arrow_v5.reset_smoother()
        self._last_arrow_lost_log = 0.0
        self.log_signal.emit("[拼图] 箭头检测器已启动")

        while self._running:
            t0 = time.perf_counter()

            # 检查窗口是否仍然有效
            if not self._window_mgr.is_valid():
                self.log_signal.emit("[拼图] 窗口已失效，停止拼图")
                break

            # 检查窗口是否最小化，如果最小化则还原
            if self._window_mgr.is_minimized():
                self.log_signal.emit("[拼图] 检测到窗口最小化，正在还原...")
                self._window_mgr.bring_to_front()
                time.sleep(0.5)  # 等待窗口还原
                continue

            try:
                # 拼图使用小地图截图
                curr = bitblt_capture(self._window_mgr.hwnd, map_x, map_y, map_w, map_h)
                consecutive_failures = 0  # 重置失败计数

                # ── 箭头检测（更新朝向） ──────────────────────────────────────────
                # 直接用拼图已截好的 curr 做检测，坐标系与用户框选区域完全一致
                # build_display 已是动态尺寸，不依赖 MAP_W/MAP_H 常量
                res = None
                arrow_frame = curr  # 与拼图共享同一帧，无需重复截图
                if self._arrow_detector:
                    try:
                        res = arrow_v5.detect(arrow_frame)
                        if res:
                            raw_angle = float(res["bearing"])
                            # 大角度跳变保护：若与上一帧差超过 90°，丢弃本帧（抗误检）
                            if self._minimap._facing_ready:
                                prev = float(self._minimap._facing_angle)
                                jump = abs(arrow_v5.angle_delta(raw_angle, prev))
                                if jump > 90.0:
                                    raw_angle = prev
                            game_angle = arrow_v5.smooth_filter(raw_angle)
                            compass_dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                            compass = compass_dirs[int((game_angle + 22.5) // 45) % 8]

                            # 更新 MiniMap 的朝向
                            self._minimap._facing_angle = game_angle
                            self._minimap._facing_ready = True
                            if frame_count % int(self._CAPTURE_FPS * 2) == 0:
                                self.log_signal.emit(f"[朝向] {game_angle:.1f}° {compass}")
                        else:
                            now = time.time()
                            if now - self._last_arrow_lost_log > 2.0:
                                self.log_signal.emit("[朝向] 未检测到箭头，保持上一朝向")
                                self._last_arrow_lost_log = now

                    except Exception as e:
                        self.log_signal.emit(f"[朝向] 检测异常：{e}")
                        arrow_frame = curr  # fallback

                # ── 调试窗口 ─────────────────────────────────────────────────
                try:
                    if res:
                        disp = arrow_v5.build_display(
                            arrow_frame, res, self._minimap._facing_angle)
                    else:
                        disp = arrow_v5.build_display_empty(arrow_frame)
                    cv2.imshow("LOTRO 方位检测", disp)
                    cv2.waitKey(1)
                except Exception:
                    pass

            except Exception as e:
                consecutive_failures += 1
                error_msg = f"[错误] 捕获失败：{e}"

                # 如果连续失败 3 次，尝试还原窗口
                if consecutive_failures >= 3:
                    self.log_signal.emit(f"[拼图] 连续捕获失败 {consecutive_failures} 次，尝试还原窗口")
                    self._window_mgr.bring_to_front()
                    time.sleep(0.5)  # 等待窗口还原
                    consecutive_failures = 0

                self.log_signal.emit(error_msg)
                time.sleep(interval)
                continue

            # ── 箭头检测（更新朝向） ──────────────────────────────────────────
            # (此段逻辑已上移至截图后)

            # 检测（使用整个窗口区域，每帧检测）
            detections = []
            detect_img = None
            if self._yolo_model is not None and self._yolo_detecting:
                # 每帧都检测
                # 只在第一次检测时输出信息
                if not detection_logged:
                    self.log_signal.emit(f"[检测] 已启动，区域：{detect_w}x{detect_h}")
                    self.log_signal.emit("[检测] 模式：每帧检测（最高流畅度）")
                    detection_logged = True

                try:
                    # 使用 bitblt_capture 捕获整个窗口图像用于检测（支持后台窗口）
                    try:
                        detect_img = bitblt_capture(self._window_mgr.hwnd, 0, 0, detect_w, detect_h)
                    except Exception as e:
                        # DXGI 偶尔失败，跳过本次检测
                        detect_img = None

                    # 检查截图是否有效
                    if detect_img is not None and detect_img.size > 0:
                        # YOLO 检测
                        results = self._yolo_model.predict(
                            source=detect_img,
                            save=False,
                            show=False,
                            imgsz=1280,
                            conf=self._yolo_conf,
                            verbose=False
                        )

                        # 处理检测结果
                        for r in results:
                            if hasattr(r, 'boxes') and r.boxes is not None:
                                for box in r.boxes:
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cx = (x1 + x2) // 2
                                    cy = (y1 + y2) // 2
                                    cls_name = f"类{cls_id}"
                                    if hasattr(self._yolo_model, 'names') and self._yolo_model.names is not None:
                                        try:
                                            cls_name = self._yolo_model.names[cls_id]
                                        except:
                                            cls_name = f"类{cls_id}"
                                    detections.append((x1, y1, x2, y2, conf, cls_id, cx, cy, cls_name))

                        self.detections_signal.emit(detections)

                        # 输出检测结果到日志
                        current_time = time.time()
                        if detections and (current_time - last_detection_log > 0.5):
                            detection_info = []
                            for det in detections:
                                if len(det) >= 8:
                                    x1_d, y1_d, x2_d, y2_d, conf, class_id, cx, cy = det[:8]
                                    cls_name_d = det[8] if len(det) >= 9 else f"类{class_id}"
                                    detection_info.append(f"{cls_name_d}({cx},{cy},{conf:.2f})")

                            self.log_signal.emit(f"[检测] 发现 {len(detections)} 个目标：{' | '.join(detection_info)}")
                            last_detection_log = current_time
                        elif not detections and (current_time - last_detection_log > 2.0):
                            self.log_signal.emit("[检测] 未发现目标")
                            last_detection_log = current_time
                    else:
                        detect_img = None

                except Exception as e:
                    self.log_signal.emit(f"[检测错误] {e}")
                    import traceback
                    self.log_signal.emit(f"[检测错误] {traceback.format_exc()}")
            else:
                # 即使不进行 YOLO 检测，也要发送空信号，以便战斗核心能够更新巡逻逻辑
                self.detections_signal.emit([])

            # 递增帧计数器（保持变量但不使用）
            frame_count += 1

            # 计算 FPS
            current_fps = 1.0 / max(0.001, time.perf_counter() - t0)

            # 每帧都发送帧信号（用于独立显示窗口），但检测数据只在检测帧更新
            if detect_img is not None:
                self.frame_signal.emit(detect_img, detections, current_fps)

            # 跳过无明显变化的帧
            if prev_img is not None and cv2.absdiff(curr, prev_img).mean() < 1.0:
                time.sleep(max(0, interval - (time.perf_counter() - t0)))
                continue

            if self._match_only:
                # 仅匹配模式：获取位置但不更新画布
                pos = self._minimap.update_match_only(curr)
                if pos:
                    # 将位置存入轨迹以供 get_player_position 使用
                    self._minimap.trajectory = [pos]
                else:
                    self.log_signal.emit("[警告] 匹配失败，请确保在加载的地图区域内")
            else:
                # 标准拼图模式
                ok = self._minimap.update(curr)
                if not ok:
                    self.log_signal.emit("[警告] 本帧匹配失败，跳过")

            prev_img = curr
            time.sleep(max(0, interval - (time.perf_counter() - t0)))

        # 关闭调试窗口
        try:
            cv2.destroyWindow("LOTRO 方位检测")
        except Exception:
            pass

        self.log_signal.emit("[拼图] 线程结束")
        self.finished.emit()