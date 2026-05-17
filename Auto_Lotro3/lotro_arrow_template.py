"""
LOTRO 小地图箭头检测 - 模板匹配版
====================================
这个版本使用模板匹配，对于形状固定的箭头更准确。

使用前需要：
1. 先用一张清晰的箭头截图训练/生成模板
2. 替换原有的 lotro_arrow_v5
"""

import cv2
import numpy as np
import math
import time
from typing import Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

DETECT_SCALE = 4
DISPLAY_SCALE = 5
SMOOTH_K = 0.25

# HSV 参数（用于提取箭头区域）
HSV_LOWER1 = np.array([0, 120, 120])
HSV_UPPER1 = np.array([15, 255, 255])
HSV_LOWER2 = np.array([165, 120, 120])
HSV_UPPER2 = np.array([180, 255, 255])

# ═══════════════════════════════════════════════════════════════════════
# 全局状态
# ═══════════════════════════════════════════════════════════════════════

_templates: Optional[list] = None  # 旋转后的模板列表
_smooth_angle: Optional[float] = None
_templates_initialized = False


# ═══════════════════════════════════════════════════════════════════════
# 1. 模板生成
# ═══════════════════════════════════════════════════════════════════════

def _extract_arrow_mask(img: np.ndarray) -> np.ndarray:
    """从图像中提取箭头的二值掩码"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, HSV_LOWER1, HSV_UPPER1)
    mask2 = cv2.inRange(hsv, HSV_LOWER2, HSV_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _create_base_template(mask: np.ndarray) -> np.ndarray:
    """从掩码中创建居中的正方形模板"""
    # 找到箭头区域
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("Mask is empty")
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # 扩展边界
    padding = max(w, h) // 2
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(mask.shape[1] - x, w + 2 * padding)
    h = min(mask.shape[0] - y, h + 2 * padding)
    
    # 裁剪
    roi = mask[y:y+h, x:x+w]
    
    # 转换为正方形
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    sx = (size - w) // 2
    sy = (size - h) // 2
    square[sy:sy+h, sx:sx+w] = roi
    
    return square


def _rotate_template(template: np.ndarray, angle: float) -> np.ndarray:
    """旋转模板（angle: 罗盘角度，0=北）"""
    h, w = template.shape
    cx, cy = w // 2, h // 2
    
    # 罗盘角度 → OpenCV 旋转角度（顺时针为正）
    # 罗盘 0° → 模板保持向上
    cv_angle = -angle
    
    M = cv2.getRotationMatrix2D((cx, cy), cv_angle, 1.0)
    rotated = cv2.warpAffine(template, M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
    return rotated


def initialize_templates_from_image(frame: np.ndarray):
    """
    从一张包含箭头的截图初始化模板
    
    第一次使用前必须调用这个函数！
    """
    global _templates, _templates_initialized
    
    # 放大图像
    img = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                    interpolation=cv2.INTER_NEAREST)
    
    # 提取掩码
    mask = _extract_arrow_mask(img)
    
    # 创建基础模板
    base_template = _create_base_template(mask)
    
    # 生成 0-360° 的模板（每 3° 一个）
    _templates = []
    for angle in range(0, 360, 3):
        rotated = _rotate_template(base_template, angle)
        _templates.append((angle, rotated))
    
    _templates_initialized = True
    print(f"[模板匹配] 已初始化 {len(_templates)} 个模板")


def initialize_templates_from_file(filepath: str):
    """从图片文件初始化模板"""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Cannot read image: {filepath}")
    initialize_templates_from_image(img)


# ═══════════════════════════════════════════════════════════════════════
# 2. 核心检测
# ═══════════════════════════════════════════════════════════════════════

def detect(frame: np.ndarray) -> Optional[dict]:
    """
    输入原始小地图帧，返回 dict 或 None
    """
    if not _templates_initialized or _templates is None:
        raise RuntimeError("Templates not initialized! Call initialize_templates_from_image first.")
    
    # 放大
    img = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                    interpolation=cv2.INTER_NEAREST)
    
    # 提取掩码
    mask = _extract_arrow_mask(img)
    
    # 检查是否有足够的像素
    if cv2.countNonZero(mask) < 20:
        return None
    
    # 找轮廓和重心
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    img_cx = img.shape[1] / 2.0
    img_cy = img.shape[0] / 2.0
    max_offset = min(img.shape[0], img.shape[1]) * 0.35
    
    best_cnt = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 20:
            break
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        ccx = M["m10"] / M["m00"]
        ccy = M["m01"] / M["m00"]
        if math.hypot(ccx - img_cx, ccy - img_cy) <= max_offset:
            best_cnt = c
            break
    
    if best_cnt is None:
        return None
    
    # 计算重心
    M = cv2.moments(best_cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # 模板匹配
    best_angle = 0
    best_score = -1.0
    
    for angle, template in _templates:
        try:
            result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_angle = angle
        except:
            continue
    
    if best_score < 0.3:  # 置信度太低
        return None
    
    # 亚像素细化（在最佳角度附近搜索）
    base_angle = best_angle
    base_template = _templates[base_angle // 3][1] if _templates else None
    
    # 计算箭尖位置（从角度推算）
    length = min(img.shape[0], img.shape[1]) // 4
    rad = math.radians(best_angle - 90)
    tip_x = int(cx + length * math.cos(rad))
    tip_y = int(cy + length * math.sin(rad))
    
    return {
        "bearing": float(best_angle),
        "center": (cx, cy),
        "tip": (tip_x, tip_y),
        "mask": mask,
        "contour": best_cnt,
        "img_up": img,
        "confidence": best_score
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. 平滑与工具函数（与原版保持一致，方便替换）
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
    """重置平滑器"""
    global _smooth_angle
    _smooth_angle = None


def angle_delta(a: float, b: float) -> float:
    """计算两个罗盘角度之间的有符号差值"""
    return (a - b + 540.0) % 360.0 - 180.0


# ═══════════════════════════════════════════════════════════════════════
# 4. 显示（与原版保持一致）
# ═══════════════════════════════════════════════════════════════════════

_DIR_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def bearing_to_label(b):
    return _DIR_LABELS[int((b + 22.5) / 45.0) % 8]


def _draw_dir_arrow(img, cx, cy, bearing_deg, length, color, thickness=2):
    rad = math.radians(bearing_deg - 90.0)
    ex = int(cx + length * math.cos(rad))
    ey = int(cy + length * math.sin(rad))
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color,
                    thickness, cv2.LINE_AA, tipLength=0.3)


def _annotate(img, result, bearing):
    cv2.drawContours(img, [result["contour"]], -1, (255, 220, 0), 1, cv2.LINE_AA)
    cv2.circle(img, result["center"], 5, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.circle(img, result["tip"], 5, (0, 60, 255), -1, cv2.LINE_AA)
    cv2.line(img, result["center"], result["tip"], (0, 220, 0), 1, cv2.LINE_AA)


def build_display(frame, result, bearing):
    """构建显示图像（与原版保持一致）"""
    DS = DETECT_SCALE
    VS = DISPLAY_SCALE
    PAD = 4

    fh, fw = frame.shape[:2]
    target_h = fh * VS
    target_w = fw * VS

    # 左栏
    clean_mask = np.zeros_like(result["mask"])
    cv2.drawContours(clean_mask, [result["contour"]], -1, 255, -1)
    left_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    _annotate(left_bgr, result, bearing)
    left_bgr = cv2.resize(left_bgr, (target_w, target_h),
                          interpolation=cv2.INTER_NEAREST)

    # 右栏
    right = cv2.resize(frame, (target_w, target_h),
                       interpolation=cv2.INTER_LINEAR)
    ratio = VS / DS
    r_result = {
        "contour": (result["contour"].astype(np.float32) * ratio).astype(np.int32),
        "center": (int(result["center"][0] * ratio), int(result["center"][1] * ratio)),
        "tip": (int(result["tip"][0] * ratio), int(result["tip"][1] * ratio)),
    }
    _annotate(right, r_result, bearing)

    rh, rw = right.shape[:2]
    acx = int(result["center"][0] * ratio)
    acy = int(result["center"][1] * ratio)
    acx = max(10, min(rw - 10, acx))
    acy = max(10, min(rh - 10, acy))
    alen = min(rw, rh) // 2 - 6
    _draw_dir_arrow(right, acx, acy, 0.0, alen, (255, 140, 0), 2)
    _draw_dir_arrow(right, acx, acy, bearing, alen, (0, 230, 230), 2)

    label = "{:.1f} {}".format(bearing, bearing_to_label(bearing))
    cv2.putText(right, label, (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 80), 1, cv2.LINE_AA)

    gap = np.zeros((target_h, PAD, 3), dtype=np.uint8)
    return np.hstack([left_bgr, gap, right])


def build_display_empty(frame):
    VS = DISPLAY_SCALE
    PAD = 4
    fh, fw = frame.shape[:2]
    h, w = fh * VS, fw * VS
    left = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(left, "no detection", (4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    right = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    gap = np.zeros((h, PAD, 3), dtype=np.uint8)
    return np.hstack([left, gap, right])
