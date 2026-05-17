"""
LOTRO 箭头检测 - YOLO 关键点版
==================================
这个版本使用 YOLOv8 进行关键点检测，
直接检测箭尖和重心，是最准确的方案。

需要:
1. ultralytics 库 (`pip install ultralytics`)
2. 标注好的数据集
3. 训练好的模型 (.pt 文件)
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple, Dict

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("WARNING: ultralytics not installed. YOLO mode not available.")

# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

DETECT_SCALE = 4
DISPLAY_SCALE = 5
SMOOTH_K = 0.25

# ═══════════════════════════════════════════════════════════════════════
# 全局状态
# ═══════════════════════════════════════════════════════════════════════

_model: Optional[YOLO] = None
_smooth_angle: Optional[float] = None
_model_loaded = False


# ═══════════════════════════════════════════════════════════════════════
# 1. 模型加载
# ═══════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    """
    加载 YOLO 模型
    
    Args:
        model_path: 模型文件路径 (.pt 或 .onnx)
    """
    global _model, _model_loaded
    
    if not HAS_ULTRALYTICS:
        raise ImportError("ultralytics not installed!")
    
    _model = YOLO(model_path)
    _model_loaded = True
    print(f"[YOLO] 已加载模型: {model_path}")


# ═══════════════════════════════════════════════════════════════════════
# 2. 核心检测
# ═══════════════════════════════════════════════════════════════════════

def detect(frame: np.ndarray) -> Optional[dict]:
    """
    检测箭头
    
    关键点定义:
    - 点0: 箭尖
    - 点1: 重心
    """
    if not _model_loaded or _model is None:
        raise RuntimeError("Model not loaded! Call load_model first.")
    
    # 推理
    results = _model.predict(
        source=frame,
        conf=0.3,
        verbose=False
    )
    
    if not results:
        return None
    
    for result in results:
        # 检查是否有关键点
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            kpts = result.keypoints
            
            # 提取关键点坐标
            # 假设第一个检测结果是箭头
            if len(kpts) > 0:
                kpt_data = kpts[0].xy  # shape: (1, num_kpts, 2)
                
                if len(kpt_data.shape) >= 3 and kpt_data.shape[1] >= 2:
                    # 获取箭尖和重心
                    tip = (int(kpt_data[0][0][0]), int(kpt_data[0][0][1]))
                    center = (int(kpt_data[0][1][0]), int(kpt_data[0][1][1]))
                    
                    # 计算角度
                    dx = tip[0] - center[0]
                    dy = center[1] - tip[1]  # Y轴翻转
                    math_ang = math.degrees(math.atan2(dy, dx))
                    bearing = (90.0 - math_ang + 360.0) % 360.0
                    
                    # 放大坐标（与现有代码保持一致）
                    h, w = frame.shape[:2]
                    img_up = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                                      interpolation=cv2.INTER_NEAREST)
                    
                    center_up = (int(center[0] * DETECT_SCALE), int(center[1] * DETECT_SCALE))
                    tip_up = (int(tip[0] * DETECT_SCALE), int(tip[1] * DETECT_SCALE))
                    
                    # 生成一个占位 mask
                    mask = np.zeros((h * DETECT_SCALE, w * DETECT_SCALE), dtype=np.uint8)
                    cv2.circle(mask, center_up, 10, 255, -1)
                    
                    # 生成一个占位 contour
                    contour = np.array([[[center_up[0] - 5, center_up[1] - 5]],
                                      [[center_up[0] + 5, center_up[1] - 5]],
                                      [[center_up[0] + 5, center_up[1] + 5]],
                                      [[center_up[0] - 5, center_up[1] + 5]]],
                                     dtype=np.int32)
                    
                    confidence = float(result.boxes.conf[0]) if hasattr(result, 'boxes') and result.boxes is not None else 1.0
                    
                    return {
                        "bearing": bearing,
                        "center": center_up,
                        "tip": tip_up,
                        "mask": mask,
                        "contour": contour,
                        "img_up": img_up,
                        "confidence": confidence
                    }
    
    return None


# ═══════════════════════════════════════════════════════════════════════
# 3. 平滑与工具函数（与原版保持一致）
# ═══════════════════════════════════════════════════════════════════════

def smooth_filter(bearing: float) -> float:
    """低通滤波"""
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
    """计算角度差"""
    return (a - b + 540.0) % 360.0 - 180.0


# ═══════════════════════════════════════════════════════════════════════
# 4. 显示函数（与原版保持一致）
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
    if "contour" in result:
        cv2.drawContours(img, [result["contour"]], -1, (255, 220, 0), 1, cv2.LINE_AA)
    cv2.circle(img, result["center"], 5, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.circle(img, result["tip"], 5, (0, 60, 255), -1, cv2.LINE_AA)
    cv2.line(img, result["center"], result["tip"], (0, 220, 0), 1, cv2.LINE_AA)


def build_display(frame, result, bearing):
    """构建显示图像"""
    DS = DETECT_SCALE
    VS = DISPLAY_SCALE
    PAD = 4

    fh, fw = frame.shape[:2]
    target_h = fh * VS
    target_w = fw * VS

    # 左栏
    mask = result.get("mask", np.zeros((fh * DS, fw * DS), dtype=np.uint8))
    left_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _annotate(left_bgr, result, bearing)
    left_bgr = cv2.resize(left_bgr, (target_w, target_h),
                          interpolation=cv2.INTER_NEAREST)

    # 右栏
    right = cv2.resize(frame, (target_w, target_h),
                       interpolation=cv2.INTER_LINEAR)
    ratio = VS / DS
    r_result = {
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


# ═══════════════════════════════════════════════════════════════════════
# 5. 数据集标注辅助工具
# ═══════════════════════════════════════════════════════════════════════

def create_label_template():
    """
    输出 YOLO 关键点标注格式说明
    
    数据集目录结构:
        dataset/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    
    标签格式 (每一行):
        class_id x0 y0 v0 x1 y1 v1 ...
        
        其中:
        - class_id: 类别ID (这里总是0)
        - x, y: 关键点坐标 (归一化到 0-1)
        - v: 可见性 (0=不可见, 1=可见)
    
    我们的两个关键点:
        点0: 箭尖
        点1: 重心
    """
    print("=" * 60)
    print("YOLO 关键点检测 - 数据集标注说明")
    print("=" * 60)
    print()
    print("目录结构:")
    print("  dataset/")
    print("    ├── images/")
    print("    │   ├── train/")
    print("    │   └── val/")
    print("    └── labels/")
    print("        ├── train/")
    print("        └── val/")
    print()
    print("标注格式 (每一行一个标注):")
    print("  0 x0 y0 1 x1 y1 1")
    print()
    print("  其中:")
    print("  - 0: 类别ID (箭头)")
    print("  - x0, y0: 箭尖坐标 (归一化到0-1)")
    print("  - x1, y1: 重心坐标 (归一化到0-1)")
    print("  - 1: 表示可见")
    print()
    print("使用 LabelImg 或 Roboflow 进行标注")
    print("=" * 60)


def train_model(data_yaml: str, epochs: int = 100, imgsz: int = 640):
    """
    训练模型
    
    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        imgsz: 图像尺寸
    """
    if not _model_loaded:
        # 加载预训练模型
        _model = YOLO('yolov8n-pose.pt')
    
    print(f"[YOLO] 开始训练: {data_yaml}")
    results = _model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16
    )
    print("[YOLO] 训练完成")
    return results


if __name__ == "__main__":
    create_label_template()
