"""
路径平滑模块
对 A*生成的路径进行优化，移除冗余点并平滑轨迹
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .grid_map import GridMap


@dataclass
class SmoothPathResult:
    """路径平滑结果"""
    original_path: List[Tuple[int, int]]      # 原始路径
    smoothed_path: List[Tuple[int, int]]      # 平滑后的路径
    points_removed: int                        # 移除的点数
    reduction_ratio: float                     # 简化比例
    
    def __str__(self) -> str:
        return (f"SmoothPathResult(original={len(self.original_path)} points, "
                f"smoothed={len(self.smoothed_path)} points, "
                f"removed={self.points_removed}, "
                f"reduction={self.reduction_ratio:.2%})")


class PathSmoother:
    """
    路径平滑器
    提供多种路径平滑和优化算法
    """
    
    def __init__(self, grid_map: GridMap):
        """
        初始化路径平滑器
        
        Args:
            grid_map: 网格地图
        """
        self.grid_map = grid_map
    
    def smooth_path(self, path: List[Tuple[int, int]], 
                   method: str = "shortcut") -> SmoothPathResult:
        """
        平滑路径
        
        Args:
            path: 原始路径
            method: 平滑方法 ("shortcut", "gradient", "bezier")
            
        Returns:
            SmoothPathResult: 平滑结果
        """
        if len(path) <= 2:
            return SmoothPathResult(
                original_path=path,
                smoothed_path=path.copy(),
                points_removed=0,
                reduction_ratio=0.0
            )
        
        if method == "shortcut":
            smoothed = self._shortcut_smoothing(path)
        elif method == "gradient":
            smoothed = self._gradient_smoothing(path)
        elif method == "bezier":
            smoothed = self._bezier_smoothing(path)
        else:
            raise ValueError(f"未知的平滑方法：{method}")
        
        points_removed = len(path) - len(smoothed)
        reduction_ratio = points_removed / len(path) if len(path) > 0 else 0.0
        
        return SmoothPathResult(
            original_path=path,
            smoothed_path=smoothed,
            points_removed=points_removed,
            reduction_ratio=reduction_ratio
        )
    
    def _shortcut_smoothing(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        捷径平滑算法
        通过视线检查移除路径中的冗余点
        
        Args:
            path: 原始路径
            
        Returns:
            List[Tuple[int, int]]: 平滑后的路径
        """
        if len(path) <= 2:
            return path.copy()
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # 尝试直接连接到最远的可达点
            furthest_idx = current_idx + 1
            
            for test_idx in range(current_idx + 2, len(path)):
                # 检查从 current_idx 到 test_idx 是否有直线通路
                if self._has_line_of_sight(
                    path[current_idx][0], path[current_idx][1],
                    path[test_idx][0], path[test_idx][1]
                ):
                    furthest_idx = test_idx
                else:
                    break
            
            # 添加最远的可达点
            if furthest_idx > current_idx + 1:
                smoothed.append(path[furthest_idx])
                current_idx = furthest_idx
            else:
                current_idx += 1
                if current_idx < len(path):
                    smoothed.append(path[current_idx])
        
        return smoothed
    
    def _gradient_smoothing(self, path: List[Tuple[int, int]], 
                           iterations: int = 10,
                           alpha: float = 0.5,
                           beta: float = 0.1) -> List[Tuple[int, int]]:
        """
        梯度下降平滑算法
        通过优化路径点位置来平滑路径
        
        Args:
            path: 原始路径
            iterations: 迭代次数
            alpha: 数据保真度权重（保持接近原始路径）
            beta: 平滑度权重
            
        Returns:
            List[Tuple[int, int]]: 平滑后的路径
        """
        if len(path) <= 2:
            return path.copy()
        
        # 转换为浮点坐标进行优化
        smoothed = [(float(x), float(y)) for x, y in path]
        
        for _ in range(iterations):
            new_smoothed = smoothed.copy()
            
            # 固定起点和终点
            for i in range(1, len(smoothed) - 1):
                # 数据保真度项（拉向原始点）
                data_term_x = alpha * (path[i][0] - smoothed[i][0])
                data_term_y = alpha * (path[i][1] - smoothed[i][1])
                
                # 平滑项（拉向相邻点的中点）
                smooth_term_x = beta * (
                    (smoothed[i-1][0] + smoothed[i+1][0]) / 2 - smoothed[i][0]
                )
                smooth_term_y = beta * (
                    (smoothed[i-1][1] + smoothed[i+1][1]) / 2 - smoothed[i][1]
                )
                
                # 更新位置
                new_x = smoothed[i][0] + data_term_x + smooth_term_x
                new_y = smoothed[i][1] + data_term_y + smooth_term_y
                
                # 检查新位置是否可行
                grid_x, grid_y = int(round(new_x)), int(round(new_y))
                if self.grid_map.is_valid(grid_x, grid_y) and \
                   not self.grid_map.is_obstacle(grid_x, grid_y):
                    new_smoothed[i] = (new_x, new_y)
            
            smoothed = new_smoothed
        
        # 转换回整数坐标
        return [(int(round(x)), int(round(y))) for x, y in smoothed]
    
    def _bezier_smoothing(self, path: List[Tuple[int, int]], 
                         num_points: int = 50) -> List[Tuple[int, int]]:
        """
        贝塞尔曲线平滑
        使用贝塞尔曲线生成平滑路径
        
        Args:
            path: 原始路径
            num_points: 生成的点数
            
        Returns:
            List[Tuple[int, int]]: 平滑后的路径
        """
        if len(path) <= 2:
            return path.copy()
        
        # 简化路径（先使用捷径算法）
        simplified = self._shortcut_smoothing(path)
        
        if len(simplified) < 3:
            return simplified
        
        # 生成贝塞尔曲线点
        smoothed = []
        
        # 分段生成贝塞尔曲线
        for i in range(len(simplified) - 2):
            p0 = simplified[i]
            p1 = simplified[i + 1]
            p2 = simplified[i + 2]
            
            # 生成二次贝塞尔曲线点
            for t in np.linspace(0, 1, num_points // (len(simplified) - 2)):
                # 二次贝塞尔曲线公式
                x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
                y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
                smoothed.append((int(round(x)), int(round(y))))
        
        # 添加终点
        smoothed.append(simplified[-1])
        
        # 去重（移除连续的重复点）
        unique_smoothed = []
        for point in smoothed:
            if not unique_smoothed or unique_smoothed[-1] != point:
                unique_smoothed.append(point)
        
        return unique_smoothed
    
    def _has_line_of_sight(self, x0: int, y0: int, 
                          x1: int, y1: int) -> bool:
        """
        检查两点之间是否有直线通路（视线检查）
        使用 Bresenham 直线算法
        
        Args:
            x0, y0: 起点坐标
            x1, y1: 终点坐标
            
        Returns:
            bool: 是否有直线通路
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = (dx if dx > dy else -dy) // 2
        
        x, y = x0, y0
        
        while True:
            # 检查当前点
            if (x, y) != (x0, y0) and (x, y) != (x1, y1):
                if self.grid_map.is_obstacle(x, y):
                    return False
            
            if x == x1 and y == y1:
                break
            
            e2 = err
            if e2 > -dx:
                err -= dy
                x += sx
            if e2 < dy:
                err += dx
                y += sy
        
        return True
    
    def remove_redundant_points(self, path: List[Tuple[int, int]], 
                               tolerance: float = 0.5) -> List[Tuple[int, int]]:
        """
        移除共线的冗余点
        
        Args:
            path: 原始路径
            tolerance: 共线判断容差
            
        Returns:
            List[Tuple[int, int]]: 简化后的路径
        """
        if len(path) <= 2:
            return path.copy()
        
        simplified = [path[0], path[1]]
        
        for i in range(2, len(path)):
            # 检查最后三个点是否接近共线
            p1 = np.array(simplified[-2], dtype=float)
            p2 = np.array(simplified[-1], dtype=float)
            p3 = np.array(path[i], dtype=float)
            
            # 计算向量
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 计算叉积（判断是否共线）
            cross = np.cross(v1, v2)
            
            # 如果接近共线，跳过中间点
            if abs(cross) <= tolerance:
                simplified[-1] = path[i]
            else:
                simplified.append(path[i])
        
        return simplified
    
    def interpolate_path(self, path: List[Tuple[int, int]], 
                        step_size: float = 0.1) -> List[Tuple[float, float]]:
        """
        对路径进行插值，生成更密集的路径点
        
        Args:
            path: 原始路径
            step_size: 插值步长
            
        Returns:
            List[Tuple[float, float]]: 插值后的路径
        """
        if len(path) <= 1:
            return [(float(x), float(y)) for x, y in path]
        
        interpolated = []
        
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            
            # 计算两点间的距离
            dx = x1 - x0
            dy = y1 - y0
            distance = np.sqrt(dx * dx + dy * dy)
            
            # 计算需要的插值点数
            num_points = max(1, int(distance / step_size))
            
            # 生成插值点
            for j in range(num_points):
                t = j / num_points
                x = x0 + dx * t
                y = y0 + dy * t
                interpolated.append((x, y))
        
        # 添加终点
        interpolated.append((float(path[-1][0]), float(path[-1][1])))
        
        return interpolated
    
    def calculate_path_metrics(self, path: List[Tuple[int, int]]) -> dict:
        """
        计算路径的度量指标
        
        Args:
            path: 路径
            
        Returns:
            dict: 路径指标
        """
        if len(path) <= 1:
            return {
                'length': 0.0,
                'turns': 0,
                'straightness': 1.0,
                'efficiency': 1.0
            }
        
        # 计算路径长度
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx * dx + dy * dy)
        
        # 计算转弯次数
        turns = 0
        for i in range(1, len(path) - 1):
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            
            # 如果方向改变，计为转弯
            if v1 != v2:
                turns += 1
        
        # 计算直线性（起点到终点的直线距离 / 实际路径长度）
        start_to_end = np.sqrt(
            (path[-1][0] - path[0][0])**2 + 
            (path[-1][1] - path[0][1])**2
        )
        straightness = start_to_end / length if length > 0 else 1.0
        
        # 计算效率（理论最优距离 / 实际路径长度）
        # 这里使用曼哈顿距离作为理论最优的近似
        optimal = abs(path[-1][0] - path[0][0]) + abs(path[-1][1] - path[0][1])
        efficiency = optimal / length if length > 0 else 1.0
        
        return {
            'length': length,
            'turns': turns,
            'straightness': straightness,
            'efficiency': efficiency,
            'num_points': len(path)
        }
