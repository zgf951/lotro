"""
网格地图表示模块
支持障碍物、已探索区域和动态更新
"""

import numpy as np
from typing import Tuple, List, Optional, Set
from dataclasses import dataclass
from enum import IntEnum


class CellType(IntEnum):
    """网格单元类型"""
    EMPTY = 0           # 空白区域
    OBSTACLE = 1        # 障碍物
    EXPLORED = 2        # 已探索区域
    UNKNOWN = 3         # 未知区域


@dataclass
class GridConfig:
    """网格配置"""
    width: int = 100           # 地图宽度
    height: int = 100          # 地图高度
    resolution: float = 1.0    # 分辨率（米/像素）
    origin_x: float = 0.0      # 原点 X 坐标
    origin_y: float = 0.0      # 原点 Y 坐标


class GridMap:
    """
    网格地图类
    支持障碍物标记、已探索区域跟踪和代价查询
    """
    
    def __init__(self, config: Optional[GridConfig] = None):
        """
        初始化网格地图
        
        Args:
            config: 网格配置，默认使用 100x100 的地图
        """
        self.config = config or GridConfig()
        self.width = self.config.width
        self.height = self.config.height
        
        # 网格数据：存储单元类型
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 已探索区域标记
        self.explored_cells: Set[Tuple[int, int]] = set()
        
        # 障碍物集合（用于快速查询）
        self.obstacles: Set[Tuple[int, int]] = set()
        
        # 动态障碍物（可移动的障碍物）
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()
        
        # 代价网格（用于路径规划）
        self.cost_grid = np.ones((self.height, self.width), dtype=np.float32)
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        将世界坐标转换为网格坐标
        
        Args:
            x: 世界坐标 X
            y: 世界坐标 Y
            
        Returns:
            (grid_x, grid_y): 网格坐标
        """
        grid_x = int((x - self.config.origin_x) / self.config.resolution)
        grid_y = int((y - self.config.origin_y) / self.config.resolution)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        将网格坐标转换为世界坐标
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
            
        Returns:
            (world_x, world_y): 世界坐标
        """
        world_x = (grid_x + 0.5) * self.config.resolution + self.config.origin_x
        world_y = (grid_y + 0.5) * self.config.resolution + self.config.origin_y
        return (world_x, world_y)
    
    def is_valid(self, grid_x: int, grid_y: int) -> bool:
        """
        检查网格坐标是否在有效范围内
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
            
        Returns:
            bool: 是否在有效范围内
        """
        return (0 <= grid_x < self.width and 
                0 <= grid_y < self.height)
    
    def is_obstacle(self, grid_x: int, grid_y: int) -> bool:
        """
        检查指定位置是否为障碍物
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
            
        Returns:
            bool: 是否为障碍物
        """
        if not self.is_valid(grid_x, grid_y):
            return True
        return (grid_x, grid_y) in self.obstacles or \
               (grid_x, grid_y) in self.dynamic_obstacles
    
    def is_explored(self, grid_x: int, grid_y: int) -> bool:
        """
        检查指定位置是否已探索
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
            
        Returns:
            bool: 是否已探索
        """
        return (grid_x, grid_y) in self.explored_cells
    
    def set_obstacle(self, grid_x: int, grid_y: int, is_obstacle: bool = True):
        """
        设置障碍物
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
            is_obstacle: 是否设置为障碍物
        """
        if not self.is_valid(grid_x, grid_y):
            return
        
        if is_obstacle:
            self.obstacles.add((grid_x, grid_y))
            self.grid[grid_y, grid_x] = CellType.OBSTACLE
            self.cost_grid[grid_y, grid_x] = np.inf  # 无限代价
        else:
            self.obstacles.discard((grid_x, grid_y))
            self.grid[grid_y, grid_x] = CellType.EMPTY
            self.cost_grid[grid_y, grid_x] = 1.0
    
    def set_dynamic_obstacle(self, grid_x: int, grid_y: int, is_obstacle: bool = True):
        """
        设置动态障碍物
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
            is_obstacle: 是否设置为障碍物
        """
        if not self.is_valid(grid_x, grid_y):
            return
        
        if is_obstacle:
            self.dynamic_obstacles.add((grid_x, grid_y))
            self.cost_grid[grid_y, grid_x] = np.inf
        else:
            self.dynamic_obstacles.discard((grid_x, grid_y))
            if (grid_x, grid_y) not in self.obstacles:
                self.cost_grid[grid_y, grid_x] = 1.0
    
    def mark_explored(self, grid_x: int, grid_y: int):
        """
        标记为已探索区域
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
        """
        if not self.is_valid(grid_x, grid_y):
            return
        
        self.explored_cells.add((grid_x, grid_y))
        if self.grid[grid_y, grid_x] == CellType.EMPTY:
            self.grid[grid_y, grid_x] = CellType.EXPLORED
    
    def mark_explored_region(self, center_x: int, center_y: int, radius: int):
        """
        标记圆形区域为已探索
        
        Args:
            center_x: 中心 X 坐标
            center_y: 中心 Y 坐标
            radius: 半径
        """
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    nx, ny = center_x + dx, center_y + dy
                    if self.is_valid(nx, ny):
                        self.mark_explored(nx, ny)
    
    def get_cost(self, grid_x: int, grid_y: int) -> float:
        """
        获取指定位置的移动代价
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
            
        Returns:
            float: 移动代价，不可达返回无穷大
        """
        if not self.is_valid(grid_x, grid_y):
            return np.inf
        return self.cost_grid[grid_y, grid_x]
    
    def get_neighbors(self, grid_x: int, grid_y: int, 
                     allow_diagonal: bool = True) -> List[Tuple[int, int]]:
        """
        获取相邻的可通行网格
        
        Args:
            grid_x: 当前 X 坐标
            grid_y: 当前 Y 坐标
            allow_diagonal: 是否允许对角线移动
            
        Returns:
            List[Tuple[int, int]]: 相邻网格坐标列表
        """
        neighbors = []
        
        # 4 方向邻居
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # 8 方向邻居（包括对角线）
        if allow_diagonal:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        
        for dx, dy in directions:
            nx, ny = grid_x + dx, grid_y + dy
            if self.is_valid(nx, ny) and not self.is_obstacle(nx, ny):
                # 检查对角线移动时是否会被障碍物卡住
                if allow_diagonal and dx != 0 and dy != 0:
                    # 检查相邻的两个正交方向是否都是障碍物
                    if self.is_obstacle(grid_x + dx, grid_y) or \
                       self.is_obstacle(grid_x, grid_y + dy):
                        continue
                neighbors.append((nx, ny))
        
        return neighbors
    
    def clear_dynamic_obstacles(self):
        """清除所有动态障碍物"""
        self.dynamic_obstacles.clear()
        # 重新计算代价网格
        self._update_cost_grid()
    
    def _update_cost_grid(self):
        """更新代价网格"""
        self.cost_grid = np.ones((self.height, self.width), dtype=np.float32)
        for ox, oy in self.obstacles:
            self.cost_grid[oy, ox] = np.inf
    
    def add_border_obstacles(self, margin: int = 1):
        """
        在地图边界添加障碍物
        
        Args:
            margin: 边距（像素）
        """
        for x in range(self.width):
            for y in range(margin):
                self.set_obstacle(x, y)
                self.set_obstacle(x, self.height - 1 - y)
        
        for y in range(self.height):
            for x in range(margin):
                self.set_obstacle(x, y)
                self.set_obstacle(self.width - 1 - x, y)
    
    def load_from_array(self, data: np.ndarray):
        """
        从 numpy 数组加载地图
        
        Args:
            data: 2D numpy 数组，0 表示空地，1 表示障碍物
        """
        height, width = data.shape
        self.width = width
        self.height = height
        self.grid = data.astype(np.uint8)
        
        # 更新障碍物集合
        self.obstacles = set()
        for y in range(height):
            for x in range(width):
                if data[y, x] == CellType.OBSTACLE:
                    self.obstacles.add((x, y))
        
        self._update_cost_grid()
    
    def get_grid_data(self) -> np.ndarray:
        """
        获取网格数据
        
        Returns:
            np.ndarray: 网格数据数组
        """
        return self.grid.copy()
    
    def get_explored_cells(self) -> Set[Tuple[int, int]]:
        """
        获取所有已探索的单元格
        
        Returns:
            Set[Tuple[int, int]]: 已探索单元格集合
        """
        return self.explored_cells.copy()
    
    def set_walkable(self, grid_x: int, grid_y: int, is_walkable: bool = True):
        """
        设置网格是否可通行
        
        Args:
            grid_x: 网格 X 坐标
            grid_y: 网格 Y 坐标
            is_walkable: 是否可通行
        """
        self.set_obstacle(grid_x, grid_y, not is_walkable)

    def fill(self, cell_type: CellType):
        """
        使用指定类型填充整个地图
        
        Args:
            cell_type: 单元格类型
        """
        if cell_type == CellType.OBSTACLE:
            self.grid.fill(CellType.OBSTACLE)
            self.cost_grid.fill(np.inf)
            # 更新障碍物集合
            self.obstacles = set()
            for y in range(self.height):
                for x in range(self.width):
                    self.obstacles.add((x, y))
        else:
            self.grid.fill(cell_type)
            self.cost_grid.fill(1.0)
            self.obstacles.clear()
            self.dynamic_obstacles.clear()

    def reset(self):
        """重置地图"""
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.explored_cells.clear()
        self.obstacles.clear()
        self.dynamic_obstacles.clear()
        self._update_cost_grid()
    
    def __str__(self) -> str:
        """打印地图简略信息"""
        return (f"GridMap({self.width}x{self.height}, "
                f"obstacles={len(self.obstacles)}, "
                f"explored={len(self.explored_cells)})")
