"""
LOTRO 小地图方位检测  v7 (RadarDirectionDetector 风格)
======================================================
算法（完全参考 RadarDirectionDetector）：
  1. 放大图像（INTER_NEAREST，保留像素颜色）
  2. HSV 提取红/橙色 → 二值掩码
  3. 形态学去噪
  4. 找最大轮廓
  5. 计算重心
  6. 找轮廓上距重心最远的点 = 箭尖
  7. atan2 → 罗盘方位（0=北，顺时针）
"""

import cv2
import numpy as np
import win32gui
import math
import time
import ctypes
import ctypes.wintypes
import os
from typing import Optional
from utils.dxgi_capture import DxgiWindowCapture

# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

WINDOW_NAME = "The Lord of the Rings Online™"

MAP_LEFT   = 979
MAP_TOP    = 82
MAP_RIGHT  = 1075
MAP_BOTTOM = 159

MAP_W = MAP_RIGHT  - MAP_LEFT
MAP_H = MAP_BOTTOM - MAP_TOP

DETECT_SCALE = 4

HSV_LOWER1 = np.array([  0, 120, 120])
HSV_UPPER1 = np.array([ 15, 255, 255])
HSV_LOWER2 = np.array([165, 120, 120])
HSV_UPPER2 = np.array([180, 255, 255])

DISPLAY_SCALE = 5
SMOOTH_K = 0.25

# ─────────────────────────────────────────────────────────────────────
_smooth_angle: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════
# 1. 截图
# ═══════════════════════════════════════════════════════════════════════

def get_client_origin():
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    if not hwnd:
        raise RuntimeError("找不到游戏窗口: {!r}".format(WINDOW_NAME))

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y


# 全局 DXGI 捕获器（仅独立运行时使用，与 hwnd 绑定）
_dxgi_capturer: "DxgiWindowCapture | None" = None

def _get_dxgi_capturer() -> DxgiWindowCapture:
    """惰性初始化全局 DXGI 捕获器（独立运行用）"""
    global _dxgi_capturer
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    if not hwnd:
        raise RuntimeError("找不到游戏窗口: {!r}".format(WINDOW_NAME))
    if _dxgi_capturer is None or _dxgi_capturer.hwnd != hwnd:
        _dxgi_capturer = DxgiWindowCapture(hwnd)
    return _dxgi_capturer


def capture_minimap(origin=None) -> np.ndarray:
    """使用 DXGI 截取小地图区域（独立运行时使用，origin 参数保留但不再需要）"""
    cap = _get_dxgi_capturer()
    img = cap.capture(MAP_LEFT, MAP_TOP, MAP_W, MAP_H)
    if img is None:
        raise RuntimeError("DXGI 截图失败")
    return img


# ═══════════════════════════════════════════════════════════════════════
# 2. 核心检测
# ═══════════════════════════════════════════════════════════════════════

def detect(frame):
    """
    输入原始小地图帧，返回 dict 或 None。
    返回值键：
      bearing  - 罗盘方位（0=北，90=东，顺时针）
      center   - 重心坐标（放大后坐标系）
      tip      - 箭尖坐标（放大后坐标系）
      mask     - 二值掩码（放大后）
      contour  - 最大轮廓（放大后坐标系）
      img_up   - 放大后的原图
    """
    # ① 放大
    img = cv2.resize(frame, None,
                     fx=DETECT_SCALE, fy=DETECT_SCALE,
                     interpolation=cv2.INTER_NEAREST)

    # ② HSV 提取
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, HSV_LOWER1, HSV_UPPER1)
    mask2 = cv2.inRange(hsv, HSV_LOWER2, HSV_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # ③ 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ④ 找最大轮廓，重心必须在图像中心附近（排除边缘噪点）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_cx = img.shape[1] / 2.0
    img_cy = img.shape[0] / 2.0
    max_offset = min(img.shape[0], img.shape[1]) * 0.35

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = None
    for c in contours_sorted:
        if cv2.contourArea(c) < 20:
            break
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        ccx = M["m10"] / M["m00"]
        ccy = M["m01"] / M["m00"]
        if math.hypot(ccx - img_cx, ccy - img_cy) <= max_offset:
            cnt = c
            break

    if cnt is None:
        return None

    # ⑤ 重心
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # ⑥ 箭尖 = approxPolyDP 简化多边形中内角最小的顶点
    #
    # 为什么不能用"最远点"：
    #   LOTRO 三角形箭头的底边两个角 比 顶点更远离重心（等腰三角形几何性质），
    #   "最远点"会找到底角，方向误差约 180°。
    #   最小内角的顶点才是真正的尖端。
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    # 至少需要 3 个顶点构成三角形；顶点太少时回退到原始轮廓
    if approx is None or len(approx) < 3:
        approx = cnt

    pts = approx.reshape(-1, 2).astype(float)
    n   = len(pts)
    tip = (cx, cy)
    min_angle = float('inf')
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-9:
            continue
        cos_a = np.dot(v1, v2) / denom
        angle = math.acos(float(np.clip(cos_a, -1.0, 1.0)))
        if angle < min_angle:
            min_angle = angle
            tip = (int(round(p_curr[0])), int(round(p_curr[1])))

    # ⑦ 角度计算
    dx_raw   = tip[0] - cx
    dy_raw   = cy - tip[1]          # 翻转 Y
    math_ang = math.degrees(math.atan2(dy_raw, dx_raw))
    bearing  = (90.0 - math_ang + 360.0) % 360.0

    return {
        "bearing":  bearing,
        "center":   (cx, cy),
        "tip":      tip,
        "mask":     mask,
        "contour":  cnt,
        "img_up":   img,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. 平滑与工具函数（供 stitcher.py 调用）
# ═══════════════════════════════════════════════════════════════════════

def smooth_filter(bearing: float) -> float:
    """低通滤波，减少角度抖动"""
    global _smooth_angle
    if _smooth_angle is None:
        _smooth_angle = bearing
        return bearing
    delta = (bearing - _smooth_angle + 540.0) % 360.0 - 180.0
    _smooth_angle = (_smooth_angle + delta * SMOOTH_K + 360.0) % 360.0
    return _smooth_angle


def reset_smoother():
    """重置平滑器（stitcher 启动时调用）"""
    global _smooth_angle
    _smooth_angle = None


def angle_delta(a: float, b: float) -> float:
    """计算两个罗盘角度之间的有符号差值，结果在 (-180, 180]"""
    return (a - b + 540.0) % 360.0 - 180.0


# ═══════════════════════════════════════════════════════════════════════
# 4. 显示（独立运行时使用）
# ═══════════════════════════════════════════════════════════════════════

# ASCII labels only – cv2.putText cannot render CJK characters
_DIR_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

def bearing_to_label(b):
    return _DIR_LABELS[int((b + 22.5) / 45.0) % 8]


def _draw_dir_arrow(img, cx, cy, bearing_deg, length, color, thickness=2):
    rad = math.radians(bearing_deg - 90.0)
    ex  = int(cx + length * math.cos(rad))
    ey  = int(cy + length * math.sin(rad))
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color,
                    thickness, cv2.LINE_AA, tipLength=0.3)


def _annotate(img, result, bearing):
    cv2.drawContours(img, [result["contour"]], -1, (255, 220, 0), 1, cv2.LINE_AA)
    cv2.circle(img, result["center"], 5, (0, 220,   0), -1, cv2.LINE_AA)
    cv2.circle(img, result["tip"],    5, (0,  60, 255), -1, cv2.LINE_AA)
    cv2.line(img, result["center"], result["tip"], (0, 220, 0), 1, cv2.LINE_AA)


def build_display(frame, result, bearing):
    """
    左栏：掩码（DS 空间标注后 resize）
    右栏：原图（VS 空间标注）+ 北箭头(蓝) + 当前方位箭头(黄)

    尺寸全部从 frame.shape 动态推算，兼容任意输入分辨率。
    标注流程：
      左栏 → 在 DS(4x) 图上画好再 resize 到 VS(5x)  — 与原版一致，无坐标错位
      右栏 → resize 到 VS(5x) 后用换算坐标画标注
    """
    DS  = DETECT_SCALE      # 4
    VS  = DISPLAY_SCALE     # 5
    PAD = 4

    fh, fw = frame.shape[:2]   # 原始帧尺寸
    target_h = fh * VS          # 显示目标高（VS 空间）
    target_w = fw * VS          # 显示目标宽（VS 空间）

    # ── 左栏：在 DS 空间标注再 resize ─────────────────────────────────
    # result 坐标已在 DS(4x) 空间，直接标注后整体 resize，不会有坐标偏移
    clean_mask = np.zeros_like(result["mask"])
    cv2.drawContours(clean_mask, [result["contour"]], -1, 255, -1)
    left_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    _annotate(left_bgr, result, bearing)          # 在 DS 空间画点/线
    left_bgr = cv2.resize(left_bgr, (target_w, target_h),
                          interpolation=cv2.INTER_NEAREST)  # 整体缩放到 VS

    # ── 右栏：resize 到 VS 空间后标注 ────────────────────────────────
    right = cv2.resize(frame, (target_w, target_h),
                       interpolation=cv2.INTER_LINEAR)

    # DS → VS 坐标换算（ratio = VS/DS）
    ratio = VS / DS
    r_result = {
        "contour": (result["contour"].astype(np.float32) * ratio).astype(np.int32),
        "center":  (int(result["center"][0] * ratio),
                    int(result["center"][1] * ratio)),
        "tip":     (int(result["tip"][0] * ratio),
                    int(result["tip"][1] * ratio)),
    }
    _annotate(right, r_result, bearing)

    # 方位箭头从检测到的重心出发（DS→VS 坐标换算），与三角形中心完全重合
    rh, rw = right.shape[:2]
    acx = int(result["center"][0] * ratio)
    acy = int(result["center"][1] * ratio)
    # 限制在图像边界内（防止重心检测异常时越界）
    acx = max(10, min(rw - 10, acx))
    acy = max(10, min(rh - 10, acy))
    alen = min(rw, rh) // 2 - 6
    _draw_dir_arrow(right, acx, acy, 0.0,     alen, (255, 140,   0), 2)  # 蓝=北
    _draw_dir_arrow(right, acx, acy, bearing, alen, (  0, 230, 230), 2)  # 黄=当前

    # ASCII 文字（cv2 不支持 CJK）
    label = "{:.1f} {}  tip=({},{}) ctr=({},{})".format(
        bearing, bearing_to_label(bearing),
        result["tip"][0], result["tip"][1],
        result["center"][0], result["center"][1]
    )
    cv2.putText(right, label, (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 80), 1, cv2.LINE_AA)

    gap = np.zeros((target_h, PAD, 3), dtype=np.uint8)
    return np.hstack([left_bgr, gap, right])


def build_display_empty(frame):
    VS  = DISPLAY_SCALE
    PAD = 4
    fh, fw = frame.shape[:2]
    h, w   = fh * VS, fw * VS
    left = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(left, "no detection", (4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    right = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    gap   = np.zeros((h, PAD, 3), dtype=np.uint8)
    return np.hstack([left, gap, right])


# ═══════════════════════════════════════════════════════════════════════
# 5. 主程序（独立运行）
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("LOTRO 方位检测 v7 启动 ── Esc 退出")
    print("小地图: ({},{})→({},{})  {}×{}px".format(
        MAP_LEFT, MAP_TOP, MAP_RIGHT, MAP_BOTTOM, MAP_W, MAP_H))

    print("{:<10} {}".format("角度", "方向"))
    print("-" * 25)

    while True:
        frame  = capture_minimap()
        result = detect(frame)

        if result is not None:
            bearing = smooth_filter(result["bearing"])
            print("► {:6.1f}°  {}".format(bearing, bearing_to_label(bearing)))
            disp = build_display(frame, result, bearing)
        else:
            disp = build_display_empty(frame)

        cv2.imshow("LOTRO 方位检测", disp)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.02)

    print("\n已退出。")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()