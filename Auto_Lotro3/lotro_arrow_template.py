"""
基于模板匹配的箭头检测器
对于形状固定的箭头，这是最准确的方法！
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ArrowResult:
    bearing: float
    center: Tuple[int, int]
    confidence: float


class TemplateArrowDetector:
    """
    基于模板匹配的箭头检测器
    适合形状固定的 LOTRO 小地图箭头
    """
    
    def __init__(self):
        # 生成 360 度旋转的模板
        self.templates = self._generate_templates()
        
        # 历史记录
        self.history: List[float] = []
        self.history_maxlen = 8
        
        # HSV 参数（用于先筛选）
        self.hsv_low1 = np.array([0, 120, 120])
        self.hsv_high1 = np.array([15, 255, 255])
        self.hsv_low2 = np.array([165, 120, 120])
        self.hsv_high2 = np.array([180, 255, 255])
    
    def _generate_templates(self) -> List[Tuple[float, np.ndarray]]:
        """
        生成 360 度旋转的箭头模板
        每个角度一个模板
        """
        templates = []
        
        # 创建一个基础箭头模板（朝向 0° = 上）
        # 基于你提供的箭头形状
        base = self._create_base_template()
        
        # 每 5 度生成一个模板（0-360）
        for angle in range(0, 360, 5):
            rotated = self._rotate_template(base, angle)
            templates.append((angle, rotated))
        
        return templates
    
    def _create_base_template(self) -> np.ndarray:
        """创建基础箭头模板（朝北）"""
        size = 40
        template = np.zeros((size, size), dtype=np.uint8)
        cx, cy = size // 2, size // 2
        
        # 三角形箭头（朝向正上方）
        # 顶点坐标
        pts = np.array([
            [cx, 5],           # 箭尖（上）
            [cx - 12, cy + 12],  # 左下
            [cx + 12, cy + 12],  # 右下
        ], dtype=np.int32)
        
        # 填充三角形
        cv2.fillPoly(template, [pts], 255)
        
        return template
    
    def _rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
        """旋转模板"""
        h, w = template.shape[:2]
        cx, cy = w // 2, h // 2
        
        # 旋转矩阵（注意：OpenCV 顺时针旋转为正，但我们要数学坐标系旋转）
        # 我们的 angle 是罗盘角度：0=北，90=东
        # 需要转换为 OpenCV 旋转角度
        # 罗盘 0° -> 图像旋转 0°（保持朝上）
        # 罗盘 90° -> 图像旋转 -90°（朝右）
        cv_angle = -angle
        
        M = cv2.getRotationMatrix2D((cx, cy), cv_angle, 1.0)
        rotated = cv2.warpAffine(template, M, (w, h), 
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        return rotated
    
    def detect(self, frame: np.ndarray) -> Optional[ArrowResult]:
        """
        检测箭头朝向
        
        Args:
            frame: 小地图截图 (BGR)
            
        Returns:
            ArrowResult or None
        """
        h, w = frame.shape[:2]
        scale = 2
        img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        # ========== 步骤 1: 颜色掩码筛选 ==========
        mask = self._color_mask(img)
        
        # 如果掩码为空，尝试返回历史
        if cv2.countNonZero(mask) < 30:
            return self._get_history_result()
        
        # ========== 步骤 2: 模板匹配 ==========
        best_angle, best_score = self._match_templates(mask)
        
        if best_score < 0.5:
            return self._get_history_result()
        
        # ========== 步骤 3: 计算重心 ==========
        center_orig = self._find_center_from_mask(mask, scale, (h, w))
        
        # ========== 步骤 4: 平滑处理 ==========
        final_angle = self._smooth_angle(best_angle)
        
        return ArrowResult(
            bearing=final_angle,
            center=center_orig,
            confidence=best_score
        )
    
    def _color_mask(self, img: np.ndarray) -> np.ndarray:
        """提取红色区域"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, self.hsv_low1, self.hsv_high1)
        mask2 = cv2.inRange(hsv, self.hsv_low2, self.hsv_high2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _match_templates(self, mask: np.ndarray) -> Tuple[float, float]:
        """在所有模板中找最佳匹配"""
        h, w = mask.shape[:2]
        
        # 在中心区域搜索（因为箭头总是在中间）
        search_margin = int(min(h, w) * 0.2)
        
        best_score = -1
        best_angle = 0
        
        for angle, template in self.templates:
            # 模板匹配
            result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
            
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_angle = angle
        
        # 亚像素细化（在最佳角度附近搜索）
        best_angle, best_score = self._refine_angle(mask, best_angle)
        
        return best_angle, best_score
    
    def _refine_angle(self, mask: np.ndarray, base_angle: float) -> Tuple[float, float]:
        """细化角度（在 base_angle ±10° 范围内搜索）"""
        best_angle = base_angle
        best_score = -1
        
        base_template = self._create_base_template()
        
        # 以 1 度为步长搜索
        for angle in np.arange(base_angle - 10, base_angle + 10, 1.0):
            template = self._rotate_template(base_template, angle)
            
            result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_angle = angle
        
        return best_angle, best_score
    
    def _find_center_from_mask(self, mask: np.ndarray, scale: float, 
                              orig_shape: Tuple[int, int]) -> Tuple[int, int]:
        """从掩码计算重心"""
        h_orig, w_orig = orig_shape
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"] / scale)
                cy = int(M["m01"] / M["m00"] / scale)
                return (cx, cy)
        
        return (w_orig // 2, h_orig // 2)
    
    def _smooth_angle(self, angle: float) -> float:
        """时间平滑"""
        self.history.append(angle)
        if len(self.history) > self.history_maxlen:
            self.history.pop(0)
        
        if len(self.history) >= 3:
            # 向量平均
            angles_rad = np.radians(self.history)
            x = np.mean(np.cos(angles_rad))
            y = np.mean(np.sin(angles_rad))
            avg_angle = np.degrees(math.atan2(y, x)) % 360.0
            return avg_angle
        
        return angle
    
    def _get_history_result(self) -> Optional[ArrowResult]:
        """返回历史结果（如果有）"""
        if self.history:
            return ArrowResult(
                bearing=self.history[-1],
                center=(0, 0),
                confidence=0.5
            )
        return None


# ============================================
# 可视化调试工具
# ============================================

def visualize_templates():
    """显示模板"""
    detector = TemplateArrowDetector()
    
    # 选几个角度的模板显示
    angles_to_show = [0, 90, 180, 270, 45, 135, 225, 315]
    templates = dict(detector.templates)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, angle in enumerate(angles_to_show):
        # 找最接近的模板
        nearest = min(templates.keys(), key=lambda a: abs(a - angle))
        ax = axes[i]
        ax.imshow(templates[nearest], cmap='gray')
        ax.set_title(f"{nearest}°")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('arrow_templates.png')
    print("模板图已保存为 arrow_templates.png")


def test_on_image(image_path: str):
    """测试"""
    detector = TemplateArrowDetector()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载: {image_path}")
        return
    
    result = detector.detect(img)
    
    display = img.copy()
    
    if result:
        # 画方向线（从中心出发）
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        line_len = 30
        
        # 转换角度为坐标
        # 0° 北 -> (cx, cy-line_len)
        # 90° 东 -> (cx+line_len, cy)
        rad = math.radians(result.bearing)
        dx = math.sin(rad) * line_len
        dy = -math.cos(rad) * line_len
        
        end_x = int(cx + dx)
        end_y = int(cy + dy)
        
        cv2.arrowedLine(display, (cx, cy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(display, f"{result.bearing:.1f}°, conf:{result.confidence:.2f}",
                   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No detection", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("Template Matching Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--show-templates':
            visualize_templates()
        else:
            test_on_image(sys.argv[1])
    else:
        print("使用:")
        print("  python lotro_arrow_template.py <image_path>  # 测试图片")
        print("  python lotro_arrow_template.py --show-templates  # 显示模板")
