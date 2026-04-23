"""
A*寻路算法核心模块
实现启发式搜索、路径规划和优化
"""

import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import time

from .grid_map import GridMap


class HeuristicType(Enum):
    """启发式函数类型"""
    MANHATTAN = "manhattan"      # 曼哈顿距离
    EUCLIDEAN = "euclidean"      # 欧几里得距离
    CHEBYSHEV = "chebyshev"      # 切比雪夫距离
    OCTILE = "octile"           # 八方向距离


@dataclass
class PathNode:
    """路径节点"""
    x: int                      # X 坐标
    y: int                      # Y 坐标
    g_cost: float = 0.0        # 从起点到当前节点的实际代价
    h_cost: float = 0.0        # 从当前节点到终点的启发式估计代价
    parent: Optional['PathNode'] = None  # 父节点
    
    @property
    def f_cost(self) -> float:
        """总代价 = g_cost + h_cost"""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: 'PathNode') -> bool:
        """用于优先队列排序"""
        return self.f_cost < other.f_cost
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PathNode):
            return False
        return self.x == other.x and self.y == other.y


@dataclass
class PathResult:
    """路径规划结果"""
    success: bool                           # 是否成功找到路径
    path: List[Tuple[int, int]] = field(default_factory=list)  # 路径点列表
    path_length: float = 0.0               # 路径长度
    nodes_expanded: int = 0                # 扩展的节点数
    search_time: float = 0.0               # 搜索时间（秒）
    message: str = ""                      # 结果消息
    
    def __str__(self) -> str:
        if self.success:
            return (f"PathResult(SUCCESS, length={self.path_length:.2f}, "
                    f"nodes={self.nodes_expanded}, time={self.search_time:.4f}s)")
        else:
            return f"PathResult(FAILED: {self.message})"


class AStarPathfinder:
    """
    A*寻路算法实现
    支持多种启发式函数和优化策略
    """
    
    def __init__(self, grid_map: GridMap, 
                 heuristic: HeuristicType = HeuristicType.OCTILE,
                 allow_diagonal: bool = True):
        """
        初始化 A*寻路器
        
        Args:
            grid_map: 网格地图
            heuristic: 启发式函数类型
            allow_diagonal: 是否允许对角线移动
        """
        self.grid_map = grid_map
        self.heuristic_type = heuristic
        self.allow_diagonal = allow_diagonal
        
        # 对角线移动的代价系数
        self.diagonal_cost = np.sqrt(2) if allow_diagonal else 1.0
        
        # 搜索统计
        self.nodes_expanded = 0
        self.nodes_generated = 0
    
    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        计算启发式距离
        
        Args:
            x1, y1: 起点坐标
            x2, y2: 终点坐标
            
        Returns:
            float: 启发式估计值
        """
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        if self.heuristic_type == HeuristicType.MANHATTAN:
            return dx + dy
        
        elif self.heuristic_type == HeuristicType.EUCLIDEAN:
            return np.sqrt(dx * dx + dy * dy)
        
        elif self.heuristic_type == HeuristicType.CHEBYSHEV:
            return max(dx, dy)
        
        elif self.heuristic_type == HeuristicType.OCTILE:
            # 八方向距离（最优的混合启发式）
            if self.allow_diagonal:
                return max(dx, dy) + (self.diagonal_cost - 1) * min(dx, dy)
            else:
                return dx + dy
        
        return dx + dy  # 默认回退到曼哈顿距离
    
    def _get_move_cost(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        计算从一个节点移动到另一个节点的代价
        
        Args:
            x1, y1: 起始节点
            x2, y2: 目标节点
            
        Returns:
            float: 移动代价
        """
        # 基础代价
        if x1 != x2 and y1 != y2:  # 对角线移动
            base_cost = self.diagonal_cost
        else:  # 正交移动
            base_cost = 1.0
        
        # 地形的平均代价
        terrain_cost = (self.grid_map.get_cost(x1, y1) + 
                       self.grid_map.get_cost(x2, y2)) / 2
        
        return base_cost * terrain_cost
    
    def find_path(self, start: Tuple[int, int], 
                  goal: Tuple[int, int],
                  timeout: float = 1.0) -> PathResult:
        """
        寻找从起点到终点的最优路径
        
        Args:
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
            timeout: 超时时间（秒）
            
        Returns:
            PathResult: 路径规划结果
        """
        start_time = time.time()
        
        # 验证起点和终点
        if not self.grid_map.is_valid(start[0], start[1]):
            return PathResult(
                success=False, 
                message=f"起点 {start} 超出地图范围"
            )
        
        if not self.grid_map.is_valid(goal[0], goal[1]):
            return PathResult(
                success=False, 
                message=f"终点 {goal} 超出地图范围"
            )
        
        if self.grid_map.is_obstacle(start[0], start[1]):
            return PathResult(
                success=False, 
                message=f"起点 {start} 是障碍物"
            )
        
        if self.grid_map.is_obstacle(goal[0], goal[1]):
            return PathResult(
                success=False, 
                message=f"终点 {goal} 是障碍物"
            )
        
        # 起点和终点相同
        if start == goal:
            return PathResult(
                success=True,
                path=[start],
                path_length=0.0,
                nodes_expanded=0,
                search_time=0.0,
                message="起点和终点相同"
            )
        
        # 初始化搜索
        start_node = PathNode(x=start[0], y=start[1])
        start_node.h_cost = self._heuristic(start[0], start[1], goal[0], goal[1])
        
        # 开放列表（优先队列）
        open_set = []
        heapq.heappush(open_set, start_node)
        
        # 记录到达每个节点的最优 g_cost
        g_scores: Dict[Tuple[int, int], float] = {(start[0], start[1]): 0.0}
        
        # 已访问的节点
        closed_set: Set[Tuple[int, int]] = set()
        
        # 用于重建路径的父节点映射
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # 重置统计
        self.nodes_expanded = 0
        self.nodes_generated = 1
        
        while open_set:
            # 检查超时
            if time.time() - start_time > timeout:
                return PathResult(
                    success=False,
                    message=f"搜索超时（{timeout}秒）"
                )
            
            # 获取 f_cost 最小的节点
            current = heapq.heappop(open_set)
            
            # 到达终点
            if (current.x, current.y) == goal:
                path = self._reconstruct_path(came_from, goal)
                path_length = self._calculate_path_length(path)
                
                return PathResult(
                    success=True,
                    path=path,
                    path_length=path_length,
                    nodes_expanded=self.nodes_expanded,
                    search_time=time.time() - start_time,
                    message="找到最优路径"
                )
            
            # 跳过已处理的节点
            if (current.x, current.y) in closed_set:
                continue
            
            # 标记为已处理
            closed_set.add((current.x, current.y))
            self.nodes_expanded += 1
            
            # 扩展邻居节点
            neighbors = self.grid_map.get_neighbors(
                current.x, current.y, 
                self.allow_diagonal
            )
            
            for nx, ny in neighbors:
                if (nx, ny) in closed_set:
                    continue
                
                # 计算新的 g_cost
                move_cost = self._get_move_cost(current.x, current.y, nx, ny)
                tentative_g = current.g_cost + move_cost
                
                # 如果找到更好的路径，更新
                if (nx, ny) not in g_scores or tentative_g < g_scores[(nx, ny)]:
                    g_scores[(nx, ny)] = tentative_g
                    came_from[(nx, ny)] = (current.x, current.y)
                    
                    # 创建新节点
                    neighbor_node = PathNode(
                        x=nx, 
                        y=ny,
                        g_cost=tentative_g,
                        h_cost=self._heuristic(nx, ny, goal[0], goal[1]),
                        parent=current
                    )
                    
                    heapq.heappush(open_set, neighbor_node)
                    self.nodes_generated += 1
        
        # 无法到达终点
        return PathResult(
            success=False,
            message="无法找到可行路径"
        )
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                         goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        重建路径
        
        Args:
            came_from: 父节点映射
            goal: 终点坐标
            
        Returns:
            List[Tuple[int, int]]: 路径点列表
        """
        path = [goal]
        current = goal
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def _calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """
        计算路径长度
        
        Args:
            path: 路径点列表
            
        Returns:
            float: 路径总长度
        """
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            
            if dx > 0 and dy > 0:  # 对角线移动
                total_length += self.diagonal_cost
            else:  # 正交移动
                total_length += 1.0
        
        return total_length
    
    def find_path_bi_directional(self, start: Tuple[int, int], 
                                  goal: Tuple[int, int],
                                  timeout: float = 1.0) -> PathResult:
        """
        双向 A*搜索（优化版本）
        
        Args:
            start: 起点坐标
            goal: 终点坐标
            timeout: 超时时间
            
        Returns:
            PathResult: 路径规划结果
        """
        start_time = time.time()
        
        # 验证起点和终点
        if not self.grid_map.is_valid(start[0], start[1]) or \
           not self.grid_map.is_valid(goal[0], goal[1]):
            return PathResult(
                success=False, 
                message="起点或终点超出地图范围"
            )
        
        if self.grid_map.is_obstacle(start[0], start[1]) or \
           self.grid_map.is_obstacle(goal[0], goal[1]):
            return PathResult(
                success=False, 
                message="起点或终点是障碍物"
            )
        
        # 起点和终点相同
        if start == goal:
            return PathResult(
                success=True,
                path=[start],
                path_length=0.0,
                nodes_expanded=0,
                search_time=0.0
            )
        
        # 初始化前向搜索
        start_node = PathNode(x=start[0], y=start[1])
        start_node.h_cost = self._heuristic(start[0], start[1], goal[0], goal[1])
        
        forward_open = []
        heapq.heappush(forward_open, start_node)
        forward_g = {(start[0], start[1]): 0.0}
        forward_closed: Set[Tuple[int, int]] = set()
        forward_parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # 初始化后向搜索
        goal_node = PathNode(x=goal[0], y=goal[1])
        goal_node.h_cost = self._heuristic(goal[0], goal[1], start[0], start[1])
        
        backward_open = []
        heapq.heappush(backward_open, goal_node)
        backward_g = {(goal[0], goal[1]): 0.0}
        backward_closed: Set[Tuple[int, int]] = set()
        backward_parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # 最佳相遇点
        best_meeting_point = None
        best_path_cost = float('inf')
        
        self.nodes_expanded = 0
        
        while forward_open and backward_open:
            # 检查超时
            if time.time() - start_time > timeout:
                return PathResult(
                    success=False,
                    message="搜索超时"
                )
            
            # 扩展前向搜索
            if forward_open:
                current = heapq.heappop(forward_open)
                
                if (current.x, current.y) not in forward_closed:
                    forward_closed.add((current.x, current.y))
                    self.nodes_expanded += 1
                    
                    # 检查是否在后向搜索的封闭列表中
                    if (current.x, current.y) in backward_closed:
                        cost = forward_g[(current.x, current.y)] + \
                              backward_g[(current.x, current.y)]
                        if cost < best_path_cost:
                            best_path_cost = cost
                            best_meeting_point = (current.x, current.y)
                    
                    # 扩展邻居
                    for nx, ny in self.grid_map.get_neighbors(
                        current.x, current.y, self.allow_diagonal
                    ):
                        if nx not in forward_closed:
                            move_cost = self._get_move_cost(
                                current.x, current.y, nx, ny
                            )
                            tentative_g = forward_g[(current.x, current.y)] + move_cost
                            
                            if (nx, ny) not in forward_g or tentative_g < forward_g[(nx, ny)]:
                                forward_g[(nx, ny)] = tentative_g
                                forward_parent[(nx, ny)] = (current.x, current.y)
                                
                                neighbor = PathNode(
                                    x=nx, y=ny,
                                    g_cost=tentative_g,
                                    h_cost=self._heuristic(nx, ny, goal[0], goal[1])
                                )
                                heapq.heappush(forward_open, neighbor)
            
            # 扩展后向搜索
            if backward_open:
                current = heapq.heappop(backward_open)
                
                if (current.x, current.y) not in backward_closed:
                    backward_closed.add((current.x, current.y))
                    self.nodes_expanded += 1
                    
                    # 检查是否在前向搜索的封闭列表中
                    if (current.x, current.y) in forward_closed:
                        cost = forward_g[(current.x, current.y)] + \
                              backward_g[(current.x, current.y)]
                        if cost < best_path_cost:
                            best_path_cost = cost
                            best_meeting_point = (current.x, current.y)
                    
                    # 扩展邻居
                    for nx, ny in self.grid_map.get_neighbors(
                        current.x, current.y, self.allow_diagonal
                    ):
                        if nx not in backward_closed:
                            move_cost = self._get_move_cost(
                                current.x, current.y, nx, ny
                            )
                            tentative_g = backward_g[(current.x, current.y)] + move_cost
                            
                            if (nx, ny) not in backward_g or tentative_g < backward_g[(nx, ny)]:
                                backward_g[(nx, ny)] = tentative_g
                                backward_parent[(nx, ny)] = (current.x, current.y)
                                
                                neighbor = PathNode(
                                    x=nx, y=ny,
                                    g_cost=tentative_g,
                                    h_cost=self._heuristic(nx, ny, start[0], start[1])
                                )
                                heapq.heappush(backward_open, neighbor)
            
            # 如果找到路径且代价小于剩余估计，可以提前终止
            if best_meeting_point:
                min_f = min(
                    forward_open[0].f_cost if forward_open else float('inf'),
                    backward_open[0].f_cost if backward_open else float('inf')
                )
                if min_f >= best_path_cost:
                    break
        
        if best_meeting_point:
            # 重建路径
            forward_path = []
            current = best_meeting_point
            while current in forward_parent:
                forward_path.append(current)
                current = forward_parent[current]
            forward_path.append(current)
            forward_path.reverse()
            
            backward_path = []
            current = best_meeting_point
            while current in backward_parent:
                current = backward_parent[current]
                backward_path.append(current)
            
            path = forward_path + backward_path
            path_length = self._calculate_path_length(path)
            
            return PathResult(
                success=True,
                path=path,
                path_length=path_length,
                nodes_expanded=self.nodes_expanded,
                search_time=time.time() - start_time,
                message="双向搜索找到路径"
            )
        
        return PathResult(
            success=False,
            message="无法找到可行路径"
        )
