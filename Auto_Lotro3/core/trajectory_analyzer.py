"""
轨迹图像分析模块
从保存的轨迹图片中提取轨迹线，转换为可通行的路径数据
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryData:
    """轨迹数据结构"""
    path_points: List[Tuple[int, int]]  # 路径点列表（像素坐标）
    start_point: Optional[Tuple[int, int]]  # 起点
    end_point: Optional[Tuple[int, int]]  # 终点
    width: int  # 图片宽度
    height: int  # 图片高度


class TrajectoryAnalyzer:
    """
    轨迹分析器
    从轨迹图片中提取轨迹线
    """
    
    def __init__(self):
        # 红色轨迹线的颜色范围（BGR 格式）
        # 红色在 HSV 中有两个范围：0-10 和 170-180
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        
        # 起点和终点颜色（BGR 格式）
        self.green_color = np.array([0, 255, 0])  # 起点
        self.blue_color = np.array([255, 0, 0])   # 终点
        
    def load_trajectory(self, image_path: str) -> Optional[TrajectoryData]:
        """
        从轨迹图片中提取路径信息
        :param image_path: 轨迹图片路径（例如：r"G:\map\002_trajectory.jpg"）
        :return: TrajectoryData 对象，如果失败返回 None
        """
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"[轨迹分析] 无法读取图片：{image_path}")
                return None
            
            height, width = img.shape[:2]
            print(f"[轨迹分析] 加载图片：{width}x{height}")
            
            # 1. 检测轨迹线
            trajectory_mask = self._extract_trajectory(img)
            
            # 2. 提取轨迹线路径
            path_points = self._extract_path_points(trajectory_mask)
            
            # 3. 检测起点（绿色）
            start_point = self._find_color_point(img, self.green_color)
            
            # 4. 检测终点（蓝色）
            end_point = self._find_color_point(img, self.blue_color)
            
            if not path_points:
                print("[轨迹分析] 警告：未找到有效轨迹")
                return None
            
            print(f"[轨迹分析] 提取成功：{len(path_points)} 个路径点")
            if start_point:
                print(f"[轨迹分析] 起点：{start_point}")
            if end_point:
                print(f"[轨迹分析] 终点：{end_point}")
            
            return TrajectoryData(
                path_points=path_points,
                start_point=start_point,
                end_point=end_point,
                width=width,
                height=height
            )
            
        except Exception as e:
            print(f"[轨迹分析] 错误：{e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_trajectory(self, img: np.ndarray) -> np.ndarray:
        """提取轨迹线（基于颜色识别）"""
        # 转换为 HSV 颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 阈值分割红色（红色在 HSV 中有两个范围）
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        
        # 合并两个 mask
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作：闭运算，填充小空洞
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _extract_path_points(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """从 mask 中提取路径点"""
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 找到最大的轮廓（主轨迹）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 如果轮廓太小，返回空
        if cv2.contourArea(largest_contour) < 100:
            return []
        
        # 提取轮廓点
        path_points = []
        for point in largest_contour:
            x, y = point[0]
            path_points.append((int(x), int(y)))
        
        # 简化路径（移除冗余点）
        path_points = self._simplify_path(path_points, epsilon=2.0)
        
        return path_points
    
    def _find_color_point(self, img: np.ndarray, color: np.ndarray) -> Optional[Tuple[int, int]]:
        """查找特定颜色的点（起点或终点）"""
        # 创建颜色掩码
        lower_color = np.maximum(color - 30, 0)
        upper_color = np.minimum(color + 30, 255)
        
        mask = cv2.inRange(img, lower_color, upper_color)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 10:
            return None
        
        # 计算质心
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)
    
    def _simplify_path(self, path: List[Tuple[int, int]], epsilon: float = 2.0) -> List[Tuple[int, int]]:
        """
        简化路径（Douglas-Peucker 算法）
        :param path: 原始路径点
        :param epsilon: 简化阈值
        :return: 简化后的路径
        """
        if len(path) < 3:
            return path
        
        path_array = np.array(path, dtype=np.float32)
        simplified = cv2.approxPolyDP(path_array, epsilon, False)
        
        # 正确处理 numpy 数组
        # simplified 是一个 shape 为 (N, 1, 2) 的数组
        return [(int(point[0][0]), int(point[0][1])) for point in simplified]
    
    def path_to_grid(self, path_points: List[Tuple[int, int]], 
                     grid_size: int = 5) -> List[Tuple[int, int]]:
        """
        将像素路径转换为网格路径
        :param path_points: 像素坐标路径
        :param grid_size: 网格大小（像素）
        :return: 网格坐标路径
        """
        grid_path = []
        for x, y in path_points:
            grid_x = x // grid_size
            grid_y = y // grid_size
            grid_path.append((grid_x, grid_y))
        return grid_path


class TrajectoryPathfinder:
    """
    基于轨迹的寻路器
    使用提取的轨迹作为参考路径
    """
    
    def __init__(self, trajectory_analyzer: TrajectoryAnalyzer):
        self._analyzer = trajectory_analyzer
        self._current_trajectory: Optional[TrajectoryData] = None
        self._grid_path: List[Tuple[int, int]] = []
        self._path_index = 0
    
    def load_trajectory_from_image(self, image_path: str, grid_size: int = 5) -> bool:
        """
        从图片加载轨迹
        :param image_path: 轨迹图片路径
        :param grid_size: 网格大小
        :return: 是否成功
        """
        trajectory = self._analyzer.load_trajectory(image_path)
        
        if trajectory is None:
            return False
        
        self._current_trajectory = trajectory
        self._grid_path = self._analyzer.path_to_grid(trajectory.path_points, grid_size)
        self._path_index = 0
        
        print(f"[轨迹寻路] 加载成功：{len(self._grid_path)} 个网格点")
        return True
    
    def get_next_point(self, current_x: int, current_y: int) -> Optional[Tuple[int, int]]:
        """
        获取下一个目标点
        :param current_x: 当前 X 坐标（网格）
        :param current_y: 当前 Y 坐标（网格）
        :return: 下一个目标点坐标
        """
        if not self._grid_path or self._path_index >= len(self._grid_path):
            return None
        
        # 找到距离当前点最近的路径点
        min_dist = float('inf')
        nearest_index = self._path_index
        
        for i in range(self._path_index, len(self._grid_path)):
            px, py = self._grid_path[i]
            dist = (px - current_x) ** 2 + (py - current_y) ** 2
            
            if dist < min_dist:
                min_dist = dist
                nearest_index = i
        
        # 更新路径索引
        self._path_index = nearest_index
        
        # 返回下一个点
        if self._path_index < len(self._grid_path) - 1:
            self._path_index += 1
            return self._grid_path[self._path_index]
        
        return None
    
    def is_path_complete(self) -> bool:
        """路径是否走完"""
        return self._path_index >= len(self._grid_path) - 1
    
    def reset(self):
        """重置路径"""
        self._path_index = 0
    
    def get_trajectory_data(self) -> Optional[TrajectoryData]:
        """获取轨迹数据"""
        return self._current_trajectory


def example_usage():
    """使用示例"""
    # 创建分析器
    analyzer = TrajectoryAnalyzer()
    
    # 加载轨迹图片
    trajectory = analyzer.load_trajectory(r"G:\map\002_trajectory.jpg")
    
    if trajectory:
        print(f"路径点数：{len(trajectory.path_points)}")
        print(f"起点：{trajectory.start_point}")
        print(f"终点：{trajectory.end_point}")
        
        # 转换为网格路径
        grid_path = analyzer.path_to_grid(trajectory.path_points, grid_size=5)
        print(f"网格路径点数：{len(grid_path)}")
        
        # 创建寻路器
        pathfinder = TrajectoryPathfinder(analyzer)
        pathfinder.load_trajectory_from_image(r"G:\map\002_trajectory.jpg")
        
        # 获取下一个点
        next_point = pathfinder.get_next_point(0, 0)
        if next_point:
            print(f"下一个目标点：{next_point}")


if __name__ == "__main__":
    example_usage()
