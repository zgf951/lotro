import cv2
import numpy as np
import math
from typing import Optional, Tuple, List


class ArrowDetectorFromTemplate:
    def __init__(self, template_mask: np.ndarray):
        """
        使用实际箭头掩码初始化检测器
        
        Args:
            template_mask: 二进制掩码，箭头区域为白色(255)，背景为黑色(0)
        """
        # 提取箭头的ROI（只保留箭头部分，去除多余背景）
        self.base_template = self._extract_arrow_roi(template_mask)
        
        # 生成旋转模板
        self.templates = self._generate_rotated_templates()
        
        # 历史记录
        self.history: List[float] = []
        self.history_maxlen = 8
        
        # 箭头中心坐标（在模板中的位置）
        h, w = self.base_template.shape
        self.template_center = (w // 2, h // 2)
        
    def _extract_arrow_roi(self, mask: np.ndarray) -> np.ndarray:
        """从掩码中提取箭头的ROI"""
        # 找到非零区域
        coords = cv2.findNonZero(mask)
        if coords is None:
            return mask
            
        # 获取边界框
        x, y, w, h = cv2.boundingRect(coords)
        
        # 扩展一点边界
        padding = max(w, h) // 4
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(mask.shape[1] - x, w + 2 * padding)
        h = min(mask.shape[0] - y, h + 2 * padding)
        
        # 裁剪
        roi = mask[y:y+h, x:x+w]
        
        # 使它成为正方形（居中）
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        sx = (size - w) // 2
        sy = (size - h) // 2
        square[sy:sy+h, sx:sx+w] = roi
        
        return square
    
    def _generate_rotated_templates(self) -> List[Tuple[float, np.ndarray]]:
        """生成所有角度的旋转模板"""
        templates = []
        
        # 每3度一个模板（为了速度）
        for angle in range(0, 360, 3):
            rotated = self._rotate_template(self.base_template, angle)
            templates.append((angle, rotated))
            
        return templates
    
    def _rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
        """旋转模板"""
        h, w = template.shape
        cx, cy = w // 2, h // 2
        
        # OpenCV 旋转是顺时针，但是我们的罗盘角度需要转换
        # 我们想要 template_angle=0 是朝北
        cv_angle = -angle
        
        M = cv2.getRotationMatrix2D((cx, cy), cv_angle, 1.0)
        rotated = cv2.warpAffine(template, M, (w, h), 
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        return rotated
    
    def detect(self, frame: np.ndarray) -> Optional[dict]:
        """
        检测箭头朝向
        
        Args:
            frame: 小地图截图（BGR格式）
            
        Returns:
            {'bearing': 角度, 'confidence': 置信度}
        """
        # 1. 颜色掩码提取红色区域
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 红/橙色范围
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if cv2.countNonZero(mask) < 20:
            return self._get_history_result()
        
        # 2. 模板匹配
        best_angle, best_score = self._match_templates(mask)
        
        if best_score < 0.4:
            return self._get_history_result()
        
        # 3. 平滑
        final_angle = self._smooth_angle(best_angle)
        
        return {
            'bearing': final_angle,
            'confidence': best_score
        }
    
    def _match_templates(self, mask: np.ndarray) -> Tuple[float, float]:
        """匹配所有模板"""
        best_score = -1.0
        best_angle = 0.0
        
        # 尝试不同的缩放（因为箭头大小可能变化）
        for scale in [0.8, 1.0, 1.2]:
            scaled_mask = cv2.resize(mask, None, fx=scale, fy=scale, 
                                    interpolation=cv2.INTER_NEAREST)
            
            for angle, template in self.templates:
                result = cv2.matchTemplate(scaled_mask, template, cv2.TM_CCOEFF_NORMED)
                
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_angle = angle
        
        # 细化搜索
        best_angle, best_score = self._refine_angle(mask, best_angle)
        
        return best_angle, best_score
    
    def _refine_angle(self, mask: np.ndarray, base_angle: float) -> Tuple[float, float]:
        """在最佳角度附近细化搜索"""
        best_score = -1.0
        best_angle = base_angle
        
        for angle in np.arange(base_angle - 10, base_angle + 10, 1.0):
            template = self._rotate_template(self.base_template, angle)
            result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_angle = angle
        
        return best_angle, best_score
    
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
            avg_angle = np.degrees(math.atan2(y, x)) % 360
            return avg_angle
        
        return angle
    
    def _get_history_result(self) -> Optional[dict]:
        if self.history:
            return {
                'bearing': self.history[-1],
                'confidence': 0.5
            }
        return None


def create_template_from_image(image: np.ndarray, 
                              arrow_bbox: Optional[Tuple[int, int, int, int]] = None) -> ArrowDetectorFromTemplate:
    """
    从截图创建检测器
    
    Args:
        image: 包含箭头的截图
        arrow_bbox: 箭头的边界框 (x, y, w, h)，如果为None则自动检测
        
    Returns:
        检测器实例
    """
    # 颜色分割
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 如果提供了bbox，只保留该区域
    if arrow_bbox is not None:
        x, y, w, h = arrow_bbox
        mask_roi = np.zeros_like(mask)
        mask_roi[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
        mask = mask_roi
    
    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return ArrowDetectorFromTemplate(mask)


if __name__ == "__main__":
    print("=" * 50)
    print("箭头模板匹配检测器")
    print("=" * 50)
    print()
    print("使用步骤:")
    print("1. 截取一张包含清晰箭头的小地图")
    print("2. 用 create_template_from_image() 创建检测器")
    print("3. 用 detector.detect(frame) 检测朝向")
    print()
    print("示例代码:")
    print("""
    # 从你的截图创建检测器
    template_img = cv2.imread("your_arrow_screenshot.png")
    detector = create_template_from_image(template_img)
    
    # 检测
    frame = cv2.imread("minimap_to_detect.png")
    result = detector.detect(frame)
    if result:
        print(f"朝向: {result['bearing']:.1f}°, 置信度: {result['confidence']:.2f}")
    """)
