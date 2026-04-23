"""
轮廓匹配核心模块 v2
=====================
实时角度检测策略（放弃 matchShapes，改用直接 HSV 轮廓法）：
  1. 对放大后的小地图做 HSV 颜色过滤，提取红/橙色三角形掩码
  2. 找最大轮廓，取重心 → 最远点 = 箭尖
  3. atan2(dx, -dy) → 罗盘方位（0=北，顺时针）

模板的作用：
  · 保存"当前朝北时"箭头的中心和箭尖坐标
  · 可以与实时检测结果对比，用于校验（可选）
"""

import cv2
import numpy as np
import json
import os
import math
from datetime import datetime
from typing import Dict, Tuple, Optional


def bearing_to_direction(bearing: float) -> str:
    """罗盘方位 → 中文方向"""
    labels = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return labels[int((bearing + 22.5) / 45.0) % 8]


# ── HSV 箭头颜色范围（适中版：平衡灵敏度和准确性）────────────────
# 实测箭尖：BGR=(75,123,209) → HSV=(11,164,209)
# 使用适中范围，配合模板中心定位
_HSV_LOWER1 = np.array([  0, 100, 120])
_HSV_UPPER1 = np.array([ 20, 255, 255])   # 橙红
_HSV_LOWER2 = np.array([160, 100, 120])
_HSV_UPPER2 = np.array([180, 255, 255])   # 深红

# 图像中心允许偏离比例（用于无模板时的自动提取）
_CENTER_RATIO = 0.45
# 模板中心搜索半径（像素，用于有模板时的精确定位）
_TEMPLATE_SEARCH_RADIUS = 30


class ContourMatcher:
    """轮廓匹配器（v2：直接 HSV 轮廓法）"""

    def __init__(self):
        self._template_contour = None
        self._template_center  = None
        self._template_tip     = None
        self._template_bearing = None
        self._template_path    = None

    # ── 颜色掩码 ────────────────────────────────────────────────────────

    def _color_mask(self, frame: np.ndarray, center: Tuple[int, int] = None) -> np.ndarray:
        """提取红/橙色箭头的二值掩码（uint8）。
        
        Args:
            frame: BGR 图像
            center: 可选的中心点坐标 (x, y)，如果提供则使用该点为中心
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1   = cv2.inRange(hsv, _HSV_LOWER1, _HSV_UPPER1)
        m2   = cv2.inRange(hsv, _HSV_LOWER2, _HSV_UPPER2)
        mask = cv2.bitwise_or(m1, m2)
        
        # 添加圆形中心区域屏蔽
        h, w = mask.shape
        if center is not None:
            # 使用模板中心
            cx, cy = center
            radius = _TEMPLATE_SEARCH_RADIUS
        else:
            # 使用图像中心
            cx, cy = w // 2, h // 2
            radius = min(w, h) * _CENTER_RATIO
        
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (int(cx), int(cy)), int(radius), 255, -1)
        mask = cv2.bitwise_and(mask, mask, mask=circle_mask)
        
        # 调试：输出掩码统计信息
        total_pixels = mask.size
        mask_pixels = cv2.countNonZero(mask)
        mask_ratio = mask_pixels / total_pixels * 100
        print(f"[HSV 调试] 图像：{frame.shape}, 中心：({cx}, {cy}), 半径：{radius}, 掩码像素：{mask_pixels}/{total_pixels} ({mask_ratio:.2f}%)")
        
        k    = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # ── 轮廓提取（对外接口，供标定工具"自动提取"按钮使用）─────────────

    def extract_contour(self, frame: np.ndarray, template_center: Tuple[int, int] = None) -> Dict:
        """
        从图像中提取箭头轮廓（HSV 颜色法）。
        返回 {"contour", "center", "tip", "bearing"} 或 {"error": ...}
        
        Args:
            frame: BGR 图像
            template_center: 可选的模板中心坐标，如果提供则优先使用该点为中心
        """
        if frame is None or frame.size == 0:
            return {"error": "图像为空"}

        try:
            mask = self._color_mask(frame, template_center)

            if mask.sum() < 255 * 5:
                return {"error": "未检测到红色/橙色像素（HSV 范围不匹配）"}

            h, w = mask.shape
            img_cx, img_cy = w / 2.0, h / 2.0
            max_offset = min(w, h) * _CENTER_RATIO

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return {"error": "未找到轮廓"}

            # 按面积从大到小，取重心在中心附近的轮廓
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
                # 如果提供了模板中心，使用模板中心；否则使用图像中心
                ref_cx, ref_cy = template_center if template_center else (img_cx, img_cy)
                search_radius = _TEMPLATE_SEARCH_RADIUS if template_center else max_offset
                if math.hypot(ccx - ref_cx, ccy - ref_cy) <= search_radius:
                    cnt = c
                    break

            if cnt is None:
                return {"error": "中心区域未找到有效轮廓"}

            # 重心
            M   = cv2.moments(cnt)
            cx  = int(M["m10"] / M["m00"])
            cy  = int(M["m01"] / M["m00"])

            # 最远点 = 箭尖
            max_dist, tip = 0, (cx, cy)
            for p in cnt:
                px, py = int(p[0][0]), int(p[0][1])
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d > max_dist:
                    max_dist = d
                    tip = (px, py)

            dx      = tip[0] - cx
            dy      = cy - tip[1]                          # 翻转 Y
            math_a  = math.degrees(math.atan2(dy, dx))
            bearing = (90.0 - math_a + 360.0) % 360.0

            return {
                "contour": cnt,
                "center":  (cx, cy),
                "tip":     tip,
                "bearing": bearing,
            }

        except Exception as e:
            return {"error": f"提取失败：{e}"}

    # ── 模板保存 / 加载 ─────────────────────────────────────────────────

    def save_template(self, contour: np.ndarray,
                      center: Tuple[int, int],
                      tip:    Tuple[int, int],
                      bearing: float,
                      template_dir: str = r"G:\map\templates") -> str:
        os.makedirs(template_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(template_dir, f"arrow_template_{ts}.json")

        data = {
            "contour":          contour.reshape(-1, 2).tolist(),
            "center":           list(center),
            "tip":              list(tip),
            "standard_bearing": bearing,
            "timestamp":        ts,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._template_contour = contour
        self._template_center  = center
        self._template_tip     = tip
        self._template_bearing = bearing
        self._template_path    = filepath
        return filepath

    def load_template(self, template_path: str) -> bool:
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._template_contour = np.array(
                data["contour"], dtype=np.int32).reshape(-1, 1, 2)
            self._template_center  = tuple(data["center"])
            self._template_tip     = tuple(data["tip"])
            self._template_bearing = data["standard_bearing"]
            self._template_path    = template_path
            return True
        except Exception as e:
            print(f"加载模板失败：{e}")
            return False

    def get_template_info(self) -> Dict:
        return {
            "path":             self._template_path,
            "center":           self._template_center,
            "tip":              self._template_tip,
            "standard_bearing": self._template_bearing,
        }

    # ── 实时匹配（直接 HSV 轮廓法，不用 matchShapes）──────────────────

    def match(self, frame: np.ndarray, rotation_step: int = 5) -> Dict:
        """
        实时检测当前帧的箭头方位。
        
        策略：
          1. 如果有模板，使用模板中心定位 → HSV 提取 → 计算角度
          2. 如果没有模板，使用图像中心定位 → HSV 提取 → 计算角度

        Args:
            frame:          当前小地图帧（BGR，建议已放大 3×）
            rotation_step:  保留参数，不再使用

        Returns:
            {"final_bearing", "match_score", "rotated_template",
             "current_contour"} 或 {"error": ...}
        """
        if frame is None or frame.size == 0:
            return {"error": "图像为空"}

        # 如果有模板，使用模板中心进行精确定位
        template_center = self._template_center
        result = self.extract_contour(frame, template_center)
        if "error" in result:
            return result

        bearing = result["bearing"]
        cnt     = result["contour"]

        return {
            "final_bearing":    bearing,
            "match_score":      1.0,        # HSV 法无需匹配分，固定为 1.0
            "rotated_template": cnt,         # 兼容旧接口，返回当前轮廓
            "current_contour":  cnt,
        }