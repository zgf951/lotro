"""
箭头检测改进方案
================
方案 A：传统方法优化（无需重新训练）
方案 B：YOLO 关键点检测（推荐，需要训练）
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# ============================================================================
# 方案 A：传统方法优化
# ============================================================================

@dataclass
class ArrowDetectionResult:
    bearing: float          # 罗盘方位(0-360)
    center: Tuple[int, int] # 重心坐标
    tip: Tuple[int, int]    # 箭尖坐标
    mask: np.ndarray        # 二值掩码
    confidence: float = 1.0 # 检测置信度


class ImprovedArrowDetector:
    """
    改进的箭头检测器 - 传统方法优化版
    """
    
    def __init__(self):
        # 多组 HSV 参数（适应不同光照）
        self.hsv_params = [
            # 主参数（红/橙色）
            (np.array([0, 100, 100]), np.array([15, 255, 255])),
            (np.array([165, 100, 100]), np.array([180, 255, 255])),
            # 备选参数（更宽容）
            (np.array([0, 80, 80]), np.array([20, 255, 255])),
            (np.array([160, 80, 80]), np.array([180, 255, 255])),
        ]
        
        # 箭头形状模板（三角形）
        self.arrow_templates = self._create_arrow_templates()
        
        # 历史轨迹（用于时间滤波）
        self.history: List[float] = []
        self.history_maxlen = 10
        
        # 最后成功的检测
        self.last_valid_result: Optional[ArrowDetectionResult] = None
    
    def _create_arrow_templates(self) -> List[np.ndarray]:
        """创建箭头形状模板"""
        templates = []
        
        # 不同大小的三角形模板
        for size in [20, 25, 30]:
            template = np.zeros((size*2, size*2), dtype=np.uint8)
            # 等腰三角形
            pts = np.array([
                [size, 0],           # 箭尖
                [0, size*2 - 1],     # 左下角
                [size*2 - 1, size*2 - 1],  # 右下角
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
            templates.append(template)
        
        return templates
    
    def detect(self, frame: np.ndarray) -> Optional[ArrowDetectionResult]:
        """
        改进的箭头检测
        """
        h, w = frame.shape[:2]
        
        # 1. 多尺度放大
        best_result = None
        best_score = 0
        
        for scale in [3, 4, 5]:
            img = cv2.resize(frame, None, fx=scale, fy=scale, 
                           interpolation=cv2.INTER_NEAREST)
            
            # 2. 多组 HSV 参数检测
            for hsv_low, hsv_high in self.hsv_params:
                result = self._detect_with_hsv(img, hsv_low, hsv_high, scale)
                if result and result.confidence > best_score:
                    best_score = result.confidence
                    best_result = result
        
        # 3. 时间滤波
        if best_result:
            return self._temporal_filter(best_result)
        elif self.last_valid_result:
            # 如果检测失败但有历史记录，返回最后一次的结果（降低置信度）
            self.last_valid_result.confidence *= 0.9
            if self.last_valid_result.confidence > 0.3:
                return self.last_valid_result
        
        return None
    
    def _detect_with_hsv(self, img: np.ndarray, 
                        hsv_low: np.ndarray, 
                        hsv_high: np.ndarray,
                        scale: float) -> Optional[ArrowDetectionResult]:
        """使用指定 HSV 参数检测"""
        h, w = img.shape[:2]
        
        # HSV 颜色提取
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        
        # 形态学处理（更大的核）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 筛选轮廓
        img_cx, img_cy = w / 2.0, h / 2.0
        max_offset = min(h, w) * 0.4
        
        best_cnt = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            
            # 检查重心位置
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            offset = math.hypot(cx - img_cx, cy - img_cy)
            if offset > max_offset:
                continue
            
            # 计算形状匹配分数
            shape_score = self._match_arrow_shape(cnt, (h, w))
            
            # 计算总分
            score = shape_score * (1.0 - offset / max_offset * 0.3)
            
            if score > best_score:
                best_score = score
                best_cnt = cnt
        
        if best_cnt is None or best_score < 0.5:
            return None
        
        # 计算箭尖（使用改进方法）
        tip, center = self._find_arrow_tip_enhanced(best_cnt)
        
        # 计算朝向
        dx = tip[0] - center[0]
        dy = center[1] - tip[1]  # Y 轴翻转
        math_ang = math.degrees(math.atan2(dy, dx))
        bearing = (90.0 - math_ang + 360.0) % 360.0
        
        return ArrowDetectionResult(
            bearing=bearing,
            center=(int(center[0]/scale), int(center[1]/scale)),
            tip=(int(tip[0]/scale), int(tip[1]/scale)),
            mask=mask,
            confidence=best_score
        )
    
    def _match_arrow_shape(self, cnt: np.ndarray, img_size: Tuple[int, int]) -> float:
        """匹配箭头形状，返回相似度分数 0-1"""
        h, w = img_size
        
        # 1. 凸包检查（箭头应该近似凸多边形）
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > 0:
            convexity = cnt_area / hull_area
        else:
            convexity = 0
        
        # 2. 长宽比检查（箭头应该比较长）
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        (x, y), (bw, bh), angle = rect
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        
        # 3. 三角形近似检查
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
        
        # 综合评分
        score = 0.0
        
        # 凸性分数
        score += convexity * 0.3
        
        # 长宽比分数（1.5-3 比较合理）
        if 1.5 <= aspect_ratio <= 4.0:
            score += 0.3 * (1.0 - abs(aspect_ratio - 2.5) / 2.5)
        
        # 顶点数量分数（3-6 个顶点）
        if 3 <= len(approx) <= 6:
            score += 0.4
        
        return min(score, 1.0)
    
    def _find_arrow_tip_enhanced(self, cnt: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        改进的箭尖检测方法
        结合多种策略提高鲁棒性
        """
        # 计算重心
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)
        
        # 策略 1: PCA 主方向
        tip_pca = self._find_tip_by_pca(cnt, center)
        
        # 策略 2: 最小内角（原始方法）
        tip_angle = self._find_tip_by_min_angle(cnt, center)
        
        # 策略 3: 最远点（适用于某些情况）
        tip_farthest = self._find_farthest_point(cnt, center)
        
        # 投票选择最终结果
        candidates = [tip_pca, tip_angle, tip_farthest]
        
        # 计算两两距离，找出最一致的结果
        best_tip = None
        best_consensus = 0
        
        for candidate in candidates:
            consensus = 0
            for other in candidates:
                dist = math.hypot(candidate[0]-other[0], candidate[1]-other[1])
                if dist < 20:  # 距离小于 20 像素认为一致
                    consensus += 1
            if consensus > best_consensus:
                best_consensus = consensus
                best_tip = candidate
        
        if best_tip is None:
            best_tip = tip_pca  # 回退到 PCA 方法
        
        return best_tip, center
    
    def _find_tip_by_pca(self, cnt: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int]:
        """使用 PCA 主方向找箭尖"""
        # 提取轮廓点
        pts = cnt.reshape(-1, 2).astype(np.float32)
        
        if len(pts) < 3:
            return center
        
        # PCA
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        
        # 主方向
        main_dir = eigenvectors[0]
        
        # 沿着主方向投影，找到最远点
        projections = []
        for pt in pts:
            proj = np.dot(pt - mean[0], main_dir)
            projections.append((proj, tuple(pt)))
        
        # 找到两个极端点
        projections.sort()
        tip1 = projections[0][1]
        tip2 = projections[-1][1]
        
        # 选择离重心更远的那个作为箭尖
        dist1 = math.hypot(tip1[0]-center[0], tip1[1]-center[1])
        dist2 = math.hypot(tip2[0]-center[0], tip2[1]-center[1])
        
        return tip1 if dist1 > dist2 else tip2
    
    def _find_tip_by_min_angle(self, cnt: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int]:
        """原始方法：最小内角"""
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
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
    
    def _find_farthest_point(self, cnt: np.ndarray, center: Tuple[int, int]) -> Tuple[int, int]:
        """找离重心最远的点"""
        pts = cnt.reshape(-1, 2)
        
        max_dist = 0
        tip = center
        
        for pt in pts:
            dist = math.hypot(pt[0]-center[0], pt[1]-center[1])
            if dist > max_dist:
                max_dist = dist
                tip = (int(pt[0]), int(pt[1]))
        
        return tip
    
    def _temporal_filter(self, result: ArrowDetectionResult) -> ArrowDetectionResult:
        """时间滤波，减少抖动"""
        self.history.append(result.bearing)
        if len(self.history) > self.history_maxlen:
            self.history.pop(0)
        
        # 加权平均（越新权重越大）
        if len(self.history) >= 3:
            weights = np.linspace(0.5, 1.0, len(self.history))
            weights = weights / weights.sum()
            
            # 处理角度的循环性
            angles = np.array(self.history)
            # 转换为单位向量
            x = np.cos(np.radians(angles))
            y = np.sin(np.radians(angles))
            
            x_avg = np.sum(x * weights)
            y_avg = np.sum(y * weights)
            
            avg_bearing = np.degrees(math.atan2(y_avg, x_avg)) % 360.0
            result.bearing = avg_bearing
        
        self.last_valid_result = result
        return result


# ============================================================================
# 方案 B：YOLO 关键点检测（推荐方案）
# ============================================================================

class YOLOSegmentArrowDetector:
    """
    基于 YOLO 实例分割/关键点检测的箭头检测器
    需要训练模型后使用
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: YOLO 模型路径 (.pt 或 .onnx)
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_loaded = True
        except ImportError:
            print("警告: ultralytics 未安装，YOLO 模式不可用")
            self.model_loaded = False
        except Exception as e:
            print(f"警告: 加载 YOLO 模型失败: {e}")
            self.model_loaded = False
    
    def detect(self, frame: np.ndarray) -> Optional[ArrowDetectionResult]:
        """
        使用 YOLO 检测箭头
        
        预期模型输出:
        - 关键点检测: 检测箭尖(0)和重心(1)
        - 或实例分割: 检测箭头掩码
        """
        if not self.model_loaded:
            return None
        
        # 推理
        results = self.model.predict(
            source=frame,
            conf=0.3,
            verbose=False
        )
        
        for result in results:
            # 处理关键点检测结果
            if result.keypoints is not None and len(result.keypoints) > 0:
                return self._process_keypoints(result, frame.shape)
            
            # 处理实例分割结果
            elif result.masks is not None and len(result.masks) > 0:
                return self._process_segmentation(result, frame.shape)
        
        return None
    
    def _process_keypoints(self, result, img_shape: Tuple[int, int]) -> Optional[ArrowDetectionResult]:
        """处理关键点检测结果"""
        h, w = img_shape[:2]
        
        for kpts in result.keypoints:
            if len(kpts) >= 2:
                # 假设:
                # keypoint 0 = 箭尖
                # keypoint 1 = 重心
                tip_x, tip_y = int(kpts[0][0]), int(kpts[0][1])
                center_x, center_y = int(kpts[1][0]), int(kpts[1][1])
                
                # 计算朝向
                dx = tip_x - center_x
                dy = center_y - tip_y
                math_ang = math.degrees(math.atan2(dy, dx))
                bearing = (90.0 - math_ang + 360.0) % 360.0
                
                return ArrowDetectionResult(
                    bearing=bearing,
                    center=(center_x, center_y),
                    tip=(tip_x, tip_y),
                    mask=np.zeros((h, w), dtype=np.uint8),
                    confidence=float(result.boxes.conf[0]) if result.boxes else 1.0
                )
        
        return None
    
    def _process_segmentation(self, result, img_shape: Tuple[int, int]) -> Optional[ArrowDetectionResult]:
        """处理实例分割结果"""
        h, w = img_shape[:2]
        
        for i, mask in enumerate(result.masks.data):
            # 从掩码提取轮廓
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            
            # 找轮廓
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 使用改进的方法从掩码找箭尖
                cnt = max(contours, key=cv2.contourArea)
                
                # 使用传统方法处理精确的掩码
                temp_detector = ImprovedArrowDetector()
                result_obj = temp_detector._find_arrow_tip_enhanced(cnt)
                
                if result_obj:
                    tip, center = result_obj
                    
                    # 计算朝向
                    dx = tip[0] - center[0]
                    dy = center[1] - tip[1]
                    math_ang = math.degrees(math.atan2(dy, dx))
                    bearing = (90.0 - math_ang + 360.0) % 360.0
                    
                    return ArrowDetectionResult(
                        bearing=bearing,
                        center=center,
                        tip=tip,
                        mask=mask_np,
                        confidence=float(result.boxes.conf[i]) if result.boxes else 1.0
                    )
        
        return None


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """演示代码"""
    import sys
    
    # 尝试加载摄像头或图片
    if len(sys.argv) > 1:
        # 从文件加载
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print(f"无法加载图片: {sys.argv[1]}")
            return
    else:
        print("使用摄像头测试（按 q 退出）")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
    
    # 使用改进检测器
    detector = ImprovedArrowDetector()
    
    while True:
        if len(sys.argv) > 1:
            # 图片模式
            pass
        else:
            # 摄像头模式
            ret, frame = cap.read()
            if not ret:
                break
        
        # 检测
        result = detector.detect(frame)
        
        # 显示
        display = frame.copy()
        
        if result:
            # 画重心
            cv2.circle(display, result.center, 5, (0, 255, 0), -1)
            
            # 画箭尖
            cv2.circle(display, result.tip, 5, (0, 0, 255), -1)
            
            # 画方向线
            cv2.line(display, result.center, result.tip, (255, 255, 0), 2)
            
            # 显示角度
            cv2.putText(display, f"{result.bearing:.1f}°, conf:{result.confidence:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No arrow detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Arrow Detection", display)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
