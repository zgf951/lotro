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

显示：
  左栏：掩码 + 轮廓(青) + 重心(绿) + 箭尖(红)
  右栏：原图放大 + 同样标注 + 蓝箭(北) + 黄箭(当前方位)
  终端：滚动输出角度

依赖：pip install opencv-python numpy pywin32 mss
"""

import cv2
import numpy as np
import win32gui
import mss
import math
import time
import ctypes
import os
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

WINDOW_NAME = "The Lord of the Rings Online™"

MAP_LEFT   = 979
MAP_TOP    = 82
MAP_RIGHT  = 1075
MAP_BOTTOM = 159

MAP_W = MAP_RIGHT  - MAP_LEFT   # 96 px
MAP_H = MAP_BOTTOM - MAP_TOP    # 77 px

# 检测前先放大（与 RadarDirectionDetector 相同，改善小图像精度）
DETECT_SCALE = 4

# HSV 箭头颜色范围
# 实测箭尖：BGR=(75,123,209) → HSV=(11,164,209)
HSV_LOWER1 = np.array([  0, 120, 120])
HSV_UPPER1 = np.array([ 15, 255, 255])   # 橙红（H≤15，不含黄色 H>20）
HSV_LOWER2 = np.array([165, 120, 120])
HSV_UPPER2 = np.array([180, 255, 255])   # 深红

# 显示放大倍数（独立于检测放大）
DISPLAY_SCALE = 5

# 低通滤波
SMOOTH_K = 0.25

# ─────────────────────────────────────────────────────────────────────
_smooth_angle = None   # type: Optional[float]


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


def capture_minimap(origin):
    mon = {"left":   origin[0] + MAP_LEFT,
           "top":    origin[1] + MAP_TOP,
           "width":  MAP_W,
           "height": MAP_H}
    with mss.mss() as sct:
        raw = np.array(sct.grab(mon))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


# ═══════════════════════════════════════════════════════════════════════
# 2. 核心检测（RadarDirectionDetector 逻辑）
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

    # ④ 找最大轮廓，且重心必须在图像中心附近（排除边缘噪点）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_cx = img.shape[1] / 2.0
    img_cy = img.shape[0] / 2.0
    # 允许偏离中心的最大距离：图像短边的 35%
    max_offset = min(img.shape[0], img.shape[1]) * 0.35

    # 按面积从大到小遍历，取第一个重心在中心附近的轮廓
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

    # ⑥ 最远点 = 箭尖
    max_dist, tip = 0, (cx, cy)
    for p in cnt:
        x, y = p[0]
        d = (x - cx) ** 2 + (y - cy) ** 2
        if d > max_dist:
            max_dist = d
            tip = (int(x), int(y))

    # ⑦ 角度计算
    # RadarDirectionDetector 原式：dy = cy - tip[1]（Y 轴翻转）
    # atan2(dy, dx) → 数学角（0=东，逆时针）
    # 转罗盘方位：bearing = (90 - math_angle + 360) % 360
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
# 3. 平滑滤波
# ═══════════════════════════════════════════════════════════════════════

def smooth_filter(bearing):
    global _smooth_angle
    if _smooth_angle is None:
        _smooth_angle = bearing
        return bearing
    delta = (bearing - _smooth_angle + 540.0) % 360.0 - 180.0
    _smooth_angle = (_smooth_angle + delta * SMOOTH_K + 360.0) % 360.0
    return _smooth_angle


# ═══════════════════════════════════════════════════════════════════════
# 4. 显示
# ═══════════════════════════════════════════════════════════════════════

_DIR_LABELS = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]

def bearing_to_label(b):
    return _DIR_LABELS[int((b + 22.5) / 45.0) % 8]


def _draw_dir_arrow(img, cx, cy, bearing_deg, length, color, thickness=2):
    """从 (cx,cy) 按罗盘方位画带箭头线。"""
    rad = math.radians(bearing_deg - 90.0)
    ex  = int(cx + length * math.cos(rad))
    ey  = int(cy + length * math.sin(rad))
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color,
                    thickness, cv2.LINE_AA, tipLength=0.3)


def _annotate(img, result, bearing):
    """在图上叠加：轮廓(青) + 重心(绿) + 箭尖(红) + 连线(绿)。"""
    cv2.drawContours(img, [result["contour"]], -1, (255, 220, 0), 1, cv2.LINE_AA)
    cv2.circle(img, result["center"], 5, (0, 220,   0), -1, cv2.LINE_AA)
    cv2.circle(img, result["tip"],    5, (0,  60, 255), -1, cv2.LINE_AA)
    cv2.line(img, result["center"], result["tip"], (0, 220, 0), 1, cv2.LINE_AA)


def build_display(frame, result, bearing):
    """
    左栏：掩码（DETECT_SCALE 放大）+ 标注
    右栏：原图（DISPLAY_SCALE 放大）+ 方位箭头 + 标注（坐标换算回 DISPLAY_SCALE）
    """
    DS = DETECT_SCALE
    VS = DISPLAY_SCALE
    PAD = 4

    # ── 左栏：只显示最大轮廓（过滤掉周围噪点）────────────────────────
    clean_mask = np.zeros_like(result["mask"])
    cv2.drawContours(clean_mask, [result["contour"]], -1, 255, -1)  # 填充最大轮廓
    left_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    _annotate(left_bgr, result, bearing)

    # 统一缩放到 DISPLAY_SCALE（左右栏等高等宽）
    target_h = MAP_H * VS
    target_w = MAP_W * VS
    left_bgr = cv2.resize(left_bgr, (target_w, target_h),
                          interpolation=cv2.INTER_NEAREST)

    # ── 右栏：原图放大 ────────────────────────────────────────────────
    right = cv2.resize(frame, (target_w, target_h),
                       interpolation=cv2.INTER_LINEAR)

    # 把检测坐标从 DETECT_SCALE 换算到 DISPLAY_SCALE
    ratio = VS / DS
    r_result = {
        "contour": (result["contour"].astype(np.float32) * ratio).astype(np.int32),
        "center":  (int(result["center"][0] * ratio),
                    int(result["center"][1] * ratio)),
        "tip":     (int(result["tip"][0] * ratio),
                    int(result["tip"][1] * ratio)),
    }
    _annotate(right, r_result, bearing)

    # 方位箭头（从图像中心出发）
    rh, rw = right.shape[:2]
    acx, acy = rw // 2, rh // 2
    alen = min(rw, rh) // 2 - 6
    _draw_dir_arrow(right, acx, acy, 0.0,     alen, (255, 140,   0), 2)  # 蓝=北
    _draw_dir_arrow(right, acx, acy, bearing, alen, (  0, 230, 230), 2)  # 黄=当前

    # 文字
    sc_color = (80, 220, 80)
    cv2.putText(right, "{:.1f}  {}".format(bearing, bearing_to_label(bearing)),
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc_color, 1, cv2.LINE_AA)

    gap = np.zeros((target_h, PAD, 3), dtype=np.uint8)
    return np.hstack([left_bgr, gap, right])


def build_display_empty(frame):
    VS = DISPLAY_SCALE; PAD = 4
    h, w = MAP_H * VS, MAP_W * VS
    left = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(left, "no detection", (4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    right = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    gap   = np.zeros((h, PAD, 3), dtype=np.uint8)
    return np.hstack([left, gap, right])


# ═══════════════════════════════════════════════════════════════════════
# 5. 主程序
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("LOTRO 方位检测 v7 启动 ── Esc 退出")
    print("小地图: ({},{})→({},{})  {}×{}px".format(
        MAP_LEFT, MAP_TOP, MAP_RIGHT, MAP_BOTTOM, MAP_W, MAP_H))

    origin = get_client_origin()
    print("客户区原点: {}\n".format(origin))
    print("{:<10} {}".format("角度", "方向"))
    print("-" * 25)

    last_bearing = 0.0
    last_result  = None

    while True:
        frame  = capture_minimap(origin)
        result = detect(frame)

        if result is not None:
            bearing      = smooth_filter(result["bearing"])
            last_bearing = bearing
            last_result  = result
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