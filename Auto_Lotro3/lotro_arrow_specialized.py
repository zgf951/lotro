"""
针对 LOTRO 小地图箭头的专用检测器
基于实际截图优化
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple


class LOTROArrowDetector:
    """
    专门针对 LOTRO 小地图箭头的检测器
    经过实际截图优化
    """
    
    def __init__(self):
        # 针对这个箭头优化的 HSV 参数
        self.hsv_ranges = [
            # 主要范围（红/橙）
            (np.array([0, 150, 150]), np.array([12, 255, 255])),
            (np.array([170, 150, 150]), np.array([180, 255, 255])),
        ]
        
        # 形态学核
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 历史角度（用于平滑）
        self.history_angles = []
        self.max_history = 15
        
        # 最后有效检测
        self.last_valid = None
        self.frames_since_last = 0
    
    def detect(self, frame: np.ndarray) -> Optional[dict]:
        """
        检测箭头并返回朝向
        
        Args:
            frame: 小地图区域截图 (BGR 格式)
            
        Returns:
            dict with 'bearing', 'center', 'tip' or None
        """
        h, w = frame.shape[:2]
        
        # 放大（保持像素风格）
        scale = 4
        img = cv2.resize(frame, None, fx=scale, fy=scale, 
                        interpolation=cv2.INTER_NEAREST)
        
        # ========== 步骤 1: 颜色分割 ==========
        mask = self._extract_arrow_mask(img)
        
        if mask is None:
            self.frames_since_last += 1
            if self.frames_since_last < 10 and self.last_valid:
                return self.last_valid
            return None
        
        # ========== 步骤 2: 轮廓处理 ==========
        arrow_contour = self._find_arrow_contour(mask, img.shape)
        
        if arrow_contour is None:
            self.frames_since_last += 1
            if self.frames_since_last < 10 and self.last_valid:
                return self.last_valid
            return None
        
        # ========== 步骤 3: 计算朝向 ==========
        result = self._calculate_direction(arrow_contour, img.shape, scale, mask)
        
        if result:
            # 保存并平滑
            result = self._smooth_result(result)
            self.last_valid = result
            self.frames_since_last = 0
            return result
        
        return None
    
    def _extract_arrow_mask(self, img: np.ndarray) -> Optional[np.ndarray]:
        """提取箭头的二值掩码"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 合并多个 HSV 范围
        mask_total = np.zeros(img.shape[:2], dtype=np.uint8)
        
        for low, high in self.hsv_ranges:
            mask = cv2.inRange(hsv, low, high)
            mask_total = cv2.bitwise_or(mask_total, mask)
        
        # 形态学处理
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, self.kernel_close)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, self.kernel_open)
        
        # 检查是否有足够的像素
        if cv2.countNonZero(mask_total) < 50:
            return None
        
        return mask_total
    
    def _find_arrow_contour(self, mask: np.ndarray, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """找到箭头的轮廓"""
        h, w = img_shape[:2]
        img_cx, img_cy = w // 2, h // 2
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 筛选轮廓
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            
            # 检查重心位置（应该在中心附近）
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            offset = math.hypot(cx - img_cx, cy - img_cy)
            max_offset = min(h, w) * 0.35
            
            if offset > max_offset:
                continue
            
            # 检查形状
            shape_score = self._score_arrow_shape(cnt)
            
            candidates.append((-shape_score, offset, cnt))
        
        if not candidates:
            return None
        
        # 选择最好的
        candidates.sort()
        return candidates[0][2]
    
    def _score_arrow_shape(self, cnt: np.ndarray) -> float:
        """为箭头形状打分 (0-1)"""
        # 1. 凸包检查
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(cnt)
        convexity = cnt_area / hull_area if hull_area > 0 else 0
        
        # 2. 三角形近似
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.1 * peri, True)
        
        vertex_score = 0.0
        if 3 <= len(approx) <= 5:
            vertex_score = 1.0 - (abs(len(approx) - 3) / 3)
        
        # 3. 长宽比
        rect = cv2.minAreaRect(cnt)
        (x, y), (bw, bh), angle = rect
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        
        aspect_score = 0.0
        if 1.2 <= aspect_ratio <= 3.0:
            aspect_score = 1.0 - abs(aspect_ratio - 2.0) / 2.0
        
        # 综合评分
        score = convexity * 0.3 + vertex_score * 0.4 + aspect_score * 0.3
        return max(0.0, min(1.0, score))
    
    def _calculate_direction(self, cnt: np.ndarray, img_shape: Tuple[int, int], 
                            scale: float, mask: np.ndarray) -> Optional[dict]:
        """计算箭头朝向"""
        h, w = img_shape[:2]
        
        # 计算重心
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center = (center_x, center_y)
        
        # ========== 关键改进：多种方法找箭尖 ==========
        
        # 方法 1: PCA 主方向（最可靠）
        tip_pca = self._find_tip_pca(cnt, center)
        
        # 方法 2: 距离变换找最远点
        tip_dist = self._find_tip_distance_transform(mask, center)
        
        # 方法 3: 最小内角
        tip_angle = self._find_tip_min_angle(cnt, center)
        
        # 方法 4: 轮廓凸包最远点
        tip_hull = self._find_tip_hull(cnt, center)
        
        # 投票选择最佳结果
        candidates = [tip_pca, tip_dist, tip_angle, tip_hull]
        tip = self._vote_best_tip(candidates, center)
        
        # 计算角度
        dx = tip[0] - center[0]
        dy = center[1] - tip[1]  # Y轴翻转
        math_angle = math.degrees(math.atan2(dy, dx))
        bearing = (90.0 - math_angle + 360.0) % 360.0
        
        # 转换回原始坐标
        center_orig = (int(center[0] / scale), int(center[1] / scale))
        tip_orig = (int(tip[0] / scale), int(tip[1] / scale))
        
        return {
            'bearing': bearing,
            'center': center_orig,
            'tip': tip_orig,
            'center_scaled': center,
            'tip_scaled': tip,
            'mask': mask,
            'contour': cnt
        }
    
    def _find_tip_pca(self, cnt: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int]:
        """用 PCA 找箭尖"""
        pts = cnt.reshape(-1, 2).astype(np.float32)
        
        if len(pts) < 3:
            return center
        
        # PCA
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        main_dir = eigenvectors[0]
        
        # 投影
        projections = []
        for pt in pts:
            proj = np.dot(pt - mean[0], main_dir)
            projections.append((proj, tuple(pt)))
        
        # 找两个极端
        projections.sort()
        tip1 = projections[0][1]
        tip2 = projections[-1][1]
        
        # 选择离重心更远的
        d1 = math.hypot(tip1[0]-center[0], tip1[1]-center[1])
        d2 = math.hypot(tip2[0]-center[0], tip2[1]-center[1])
        
        return tip1 if d1 > d2 else tip2
    
    def _find_tip_distance_transform(self, mask: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int]:
        """用距离变换找箭尖"""
        # 距离变换
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # 找最亮的点
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        
        if max_val > 0:
            return max_loc
        
        return center
    
    def _find_tip_min_angle(self, cnt: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int]:
        """用最小内角找箭尖"""
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
        
        if approx is None or len(approx) < 3:
            approx = cnt
        
        pts = approx.reshape(-1, 2).astype(float)
        n = len(pts)
        
        tip = center
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
        
        return tip
    
    def _find_tip_hull(self, cnt: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int]:
        """用凸包找最远点"""
        hull = cv2.convexHull(cnt)
        
        max_dist = 0
        tip = center
        
        for pt in hull:
            pt_tuple = (int(pt[0][0]), int(pt[0][1]))
            dist = math.hypot(pt_tuple[0]-center[0], pt_tuple[1]-center[1])
            if dist > max_dist:
                max_dist = dist
                tip = pt_tuple
        
        return tip
    
    def _vote_best_tip(self, candidates: list, center: Tuple[int, int]) -> Tuple[int, int]:
        """投票选择最佳箭尖"""
        # 计算两两一致性
        consensus_scores = {}
        
        for i, tip1 in enumerate(candidates):
            score = 0
            for j, tip2 in enumerate(candidates):
                if i == j:
                    continue
                dist = math.hypot(tip1[0]-tip2[0], tip1[1]-tip2[1])
                if dist < 15:  # 距离小于 15 像素认为一致
                    score += 1
            consensus_scores[tip1] = score
        
        # 选择一致性最高的
        best_tip = max(candidates, key=lambda t: consensus_scores.get(t, 0))
        
        return best_tip
    
    def _smooth_result(self, result: dict) -> dict:
        """平滑结果"""
        angle = result['bearing']
        
        self.history_angles.append(angle)
        if len(self.history_angles) > self.max_history:
            self.history_angles.pop(0)
        
        if len(self.history_angles) >= 5:
            # 转换为单位向量平均（处理角度循环问题）
            angles_rad = np.radians(self.history_angles)
            x = np.mean(np.cos(angles_rad))
            y = np.mean(np.sin(angles_rad))
            
            avg_angle = np.degrees(math.atan2(y, x)) % 360.0
            result['bearing'] = avg_angle
        
        return result


# ============================================
# 测试代码
# ============================================

def test_on_image(image_path: str):
    """测试检测器"""
    detector = LOTROArrowDetector()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图片: {image_path}")
        return
    
    # 假设整个图就是小地图区域
    result = detector.detect(img)
    
    # 显示
    display = img.copy()
    
    if result:
        # 画结果
        cv2.circle(display, result['center'], 3, (0, 255, 0), -1)
        cv2.circle(display, result['tip'], 3, (0, 0, 255), -1)
        cv2.line(display, result['center'], result['tip'], (0, 255, 255), 2)
        
        # 画角度
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        dir_idx = int((result['bearing'] + 22.5) // 45) % 8
        label = f"{result['bearing']:.1f}° {dirs[dir_idx]}"
        
        cv2.putText(display, label, (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No detection", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("Detection Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_on_image(sys.argv[1])
    else:
        print("使用: python lotro_arrow_specialized.py <image_path>")
