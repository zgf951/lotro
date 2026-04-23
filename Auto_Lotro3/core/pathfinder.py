"""
寻路模块 - 分层架构设计

寻路模块 (PathFinder) 
 ├── 地图数据层 (MapData) 
 │   ├── 加载轨迹图片 
 │   ├── 构建可通行地图 
 │   └── 地图更新接口 
 ├── 路径规划层 (PathPlanner) 
 │   ├── A*算法 
 │   ├── 基于轨迹的寻路 
 │   └── 多种寻路策略 
 ├── 移动控制层 (Mover) 
 │   ├── WASD 控制 
 │   ├── 路径跟随 
 │   └── 状态反馈 
 └── 对外接口层 (API) 
     ├── 寻路请求 
     ├── 状态查询 
     └── 可视化调试
"""

import os
import cv2
import numpy as np
import time
import json
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum, auto

from astar_pathfinder import AStarPathfinder, GridMap, GridConfig, PathSmoother, HeuristicType, CellType
from core.trajectory_analyzer import TrajectoryAnalyzer, TrajectoryData


# ============================================================================
# 地图数据层 (MapData)
# ============================================================================

class MapDataType(Enum):
    """地图数据类型"""
    TRAJECTORY = auto()  # 轨迹地图
    EXPLORED = auto()    # 已探索区域
    CUSTOM = auto()      # 自定义地图


@dataclass
class MapData:
    """
    地图数据类
    存储和管理地图信息
    """
    map_type: MapDataType
    grid_map: GridMap
    trajectory_data: Optional[TrajectoryData] = None
    explored_area: Optional[np.ndarray] = None
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    grid_size: int = 5
    last_update: float = 0.0
    
    @classmethod
    def from_trajectory_image(cls, image_path: str, grid_size: int = 5) -> Optional['MapData']:
        """从轨迹图片创建地图数据"""
        try:
            # 加载轨迹
            analyzer = TrajectoryAnalyzer()
            trajectory = analyzer.load_trajectory(image_path)
            
            if not trajectory:
                return None
            
            return cls._create_from_trajectory(trajectory, grid_size)
            
        except Exception as e:
            print(f"[MapData] 创建失败：{e}")
            return None
    
    @classmethod
    def from_trajectory_json(cls, json_path: str, grid_size: int = 5) -> Optional['MapData']:
        """从 JSON 文件创建地图数据"""
        try:
            if not os.path.exists(json_path):
                print(f"[MapData] JSON 文件不存在：{json_path}")
                return None
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            traj_info = data.get("trajectory", {})
            points = traj_info.get("points", [])
            if not points:
                print(f"[MapData] JSON 中无轨迹点：{json_path}")
                return None
                
            canvas_info = data.get("canvas", {})
            width = canvas_info.get("width", 0)
            height = canvas_info.get("height", 0)
            
            # 转换为像素坐标路径
            path_points = [(int(p[0]), int(p[1])) for p in points]
            start_point = tuple(traj_info.get("start_point")) if traj_info.get("start_point") else None
            end_point = tuple(traj_info.get("end_point")) if traj_info.get("end_point") else None
            
            trajectory = TrajectoryData(
                path_points=path_points,
                start_point=start_point,
                end_point=end_point,
                width=width,
                height=height
            )
            
            return cls._create_from_trajectory(trajectory, grid_size)
            
        except Exception as e:
            print(f"[MapData] JSON 加载失败：{e}")
            return None
            
    @classmethod
    def _create_from_trajectory(cls, trajectory: TrajectoryData, grid_size: int) -> 'MapData':
        """内部方法：从 TrajectoryData 创建 MapData"""
        # 创建网格地图
        width = trajectory.width
        height = trajectory.height
        grid_width = width // grid_size
        grid_height = height // grid_size
        
        # 配置网格地图：设置合适的分辨率和原点
        config = GridConfig(
            width=grid_width,
            height=grid_height,
            resolution=float(grid_size),
            origin_x=0.0,
            origin_y=0.0
        )
        grid_map = GridMap(config)
        
        # 初始化为全障碍物（只允许在记录过的轨迹上移动）
        grid_map.fill(CellType.OBSTACLE)
        
        # 标记轨迹路径为可通行区域（周围一圈也都设为可通行）
        for x, y in trajectory.path_points:
            # 转换为网格坐标
            gx, gy = grid_map.world_to_grid(x, y)
            
            # 标记中心及周边网格为可通行（稍微扩宽一点路径）
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = gx + dx, gy + dy
                    if grid_map.is_valid(nx, ny):
                        grid_map.set_walkable(nx, ny, True)
        
        return cls(
            map_type=MapDataType.TRAJECTORY,
            grid_map=grid_map,
            trajectory_data=trajectory,
            width=width,
            height=height,
            grid_size=grid_size,
            last_update=time.time()
        )
    
    def update_map(self, new_data: Dict):
        """更新地图数据"""
        if 'obstacles' in new_data:
            self.obstacles = new_data['obstacles']
            for obs in self.obstacles:
                self.grid_map.set_obstacle(obs[0], obs[1], True)
        
        if 'walkable' in new_data:
            for pos, walkable in new_data['walkable'].items():
                self.grid_map.set_walkable(pos[0], pos[1], walkable)
        
        self.last_update = time.time()


# ============================================================================
# 路径规划层 (PathPlanner)
# ============================================================================

class PathStrategy(Enum):
    """寻路策略"""
    TRAJECTORY = auto()  # 基于轨迹
    ASTAR = auto()       # A*算法
    HYBRID = auto()      # 混合策略


@dataclass
class PathResult:
    """路径规划结果"""
    success: bool
    path: List[Tuple[int, int]]
    strategy: PathStrategy
    length: int
    message: str = ""


class PathPlanner:
    """
    路径规划器
    支持多种寻路策略
    """
    
    def __init__(self, map_data: MapData):
        self._map_data = map_data
        self._strategy = PathStrategy.HYBRID
        self._astar: Optional[AStarPathfinder] = None
        
        # 初始化 A*
        if map_data.grid_map:
            self._astar = AStarPathfinder(
                map_data.grid_map,
                heuristic=HeuristicType.OCTILE,
                allow_diagonal=True
            )
    
    def set_strategy(self, strategy: PathStrategy):
        """设置寻路策略"""
        self._strategy = strategy
    
    # 距离轨迹超过此阈值（世界坐标单位）时认为"偏离轨迹"
    _OFF_TRACK_THRESHOLD = 50.0

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> PathResult:
        """
        规划路径
        :param start: 起点坐标（世界坐标）
        :param goal: 终点坐标（世界坐标）
        :return: 路径规划结果
        """
        if self._strategy == PathStrategy.TRAJECTORY:
            return self._find_path_by_trajectory(start, goal)
        elif self._strategy == PathStrategy.ASTAR:
            return self._find_path_by_astar(start, goal)
        else:  # HYBRID
            result = self._find_path_by_trajectory(start, goal)
            if result.success:
                return result
            return self._find_path_by_astar(start, goal)

    @staticmethod
    def _nearest_traj_idx(pts: List[Tuple], pos: Tuple) -> int:
        """找到轨迹点列表中距 pos 最近的点的下标"""
        min_sq = float('inf')
        idx = 0
        for i, p in enumerate(pts):
            sq = (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2
            if sq < min_sq:
                min_sq = sq
                idx = i
        return idx

    def _find_path_by_trajectory(self, start: Tuple, goal: Tuple) -> PathResult:
        """
        基于轨迹的路径规划（支持偏轨恢复）

        逻辑：
          1. 在轨迹上分别找到距 start 最近的点（start_idx）
             和距 goal 最近的点（goal_idx）。
          2. 如果玩家偏离轨迹超过阈值，把 trajectory[start_idx] 作为
             第一个路点，让 Mover 先直线走回轨迹。
          3. 沿轨迹从 start_idx 走到 goal_idx（自动处理正向/反向）。
        """
        import math
        if not self._map_data.trajectory_data:
            return PathResult(
                success=False, path=[], strategy=PathStrategy.TRAJECTORY,
                length=0, message="无轨迹数据"
            )

        pts = self._map_data.trajectory_data.path_points
        if not pts:
            return PathResult(
                success=False, path=[], strategy=PathStrategy.TRAJECTORY,
                length=0, message="轨迹点为空"
            )

        start_idx = self._nearest_traj_idx(pts, start)
        goal_idx  = self._nearest_traj_idx(pts, goal)

        # 计算玩家偏离最近轨迹点的距离
        sp = pts[start_idx]
        off_track = math.hypot(sp[0] - start[0], sp[1] - start[1])

        path: List[Tuple] = []

        # 偏离轨迹时：先插入一个"回归点"（最近轨迹点），
        # Mover 会直线走向它，把玩家带回轨迹
        if off_track > self._OFF_TRACK_THRESHOLD:
            path.append(pts[start_idx])

        # 沿轨迹切片（正向或反向均支持）
        if goal_idx >= start_idx:
            segment = pts[start_idx: goal_idx + 1]
        else:
            segment = list(reversed(pts[goal_idx: start_idx + 1]))

        # 去重：若第一个路点和回归点重合则跳过
        if path and segment and segment[0] == path[-1]:
            segment = segment[1:]
        path.extend(segment)

        if not path:
            return PathResult(
                success=False, path=[], strategy=PathStrategy.TRAJECTORY,
                length=0, message="生成路径为空"
            )

        return PathResult(
            success=True,
            path=path,
            strategy=PathStrategy.TRAJECTORY,
            length=len(path),
            message=(f"轨迹寻路 start_idx={start_idx} goal_idx={goal_idx} "
                     f"偏离={off_track:.1f}")
        )
    
    def _find_path_by_astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> PathResult:
        """A*算法路径规划"""
        if not self._astar:
            return PathResult(
                success=False,
                path=[],
                strategy=PathStrategy.ASTAR,
                length=0,
                message="A*未初始化"
            )
        
        # 转换为网格坐标
        start_node = self._map_data.grid_map.world_to_grid(start[0], start[1])
        goal_node = self._map_data.grid_map.world_to_grid(goal[0], goal[1])
        
        # A*寻路
        result = self._astar.find_path(start_node, goal_node)
        
        if result.path:
            # 路径平滑
            smoother = PathSmoother(self._map_data.grid_map)
            smooth_result = smoother.smooth_path(result.path, method='shortcut')
            
            final_path = smooth_result.smoothed_path if smooth_result.smoothed_path else result.path
            
            # 关键：将网格坐标转换为世界坐标
            world_path = []
            for gx, gy in final_path:
                wx, wy = self._map_data.grid_map.grid_to_world(gx, gy)
                world_path.append((wx, wy))
            
            return PathResult(
                success=True,
                path=world_path,
                strategy=PathStrategy.ASTAR,
                length=len(world_path),
                message=f"A*寻路成功，长度：{len(world_path)}"
            )
        else:
            return PathResult(
                success=False,
                path=[],
                strategy=PathStrategy.ASTAR,
                length=0,
                message="A*未找到路径"
            )


# ============================================================================
# 移动控制层 (Mover)
# ============================================================================

class MoveState(Enum):
    """移动状态"""
    IDLE = auto()
    MOVING = auto()
    REACHED = auto()
    FAILED = auto()


@dataclass
class MoveCommand:
    """移动指令"""
    direction: str  # W, A, S, D
    duration: float = 0.2  # 持续时间（秒）


class Mover:
    """
    移动控制器
    负责 WASD 控制和路径跟随
    """
    
    def __init__(self):
        self._state = MoveState.IDLE
        self._current_path: List[Tuple[int, int]] = []
        self._path_index: int = 0          # 修复：原写法 `= int = 0` 会污染 built-in int
        self._player_pos: Optional[Tuple[float, float]] = None
        self._target_pos: Optional[Tuple[float, float]] = None
        self._move_threshold = 10.0
        self._grid_size = 5
    
    def set_path(self, path: List[Tuple[int, int]]):
        """设置跟随路径"""
        self._current_path = path
        self._path_index = 0
        self._state = MoveState.MOVING
    
    def update(self, player_pos: Tuple[float, float], player_angle: Optional[float] = None) -> Optional[MoveCommand]:
        """
        更新移动状态（迭代实现，避免密集路点时的递归栈溢出）
        :param player_pos: 玩家当前位置（世界坐标）
        :param player_angle: 玩家当前朝向（游戏角度）
        :return: 移动指令
        """
        self._player_pos = player_pos

        # 迭代跳过已到达的路点，避免递归深度问题
        while self._current_path and self._path_index < len(self._current_path):
            next_point = self._current_path[self._path_index]
            dx = next_point[0] - player_pos[0]
            dy = next_point[1] - player_pos[1]
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < self._move_threshold:
                # 已到达当前路点，前进到下一个
                self._path_index += 1
            else:
                # 还未到达，生成移动指令
                return MoveCommand(direction=self._calc_direction(dx, dy, player_angle))

        # 所有路点都已到达
        self._state = MoveState.REACHED
        return None
    
    def _calc_direction(self, dx: float, dy: float, player_angle: Optional[float] = None) -> str:
        """计算移动方向"""
        # 如果有玩家角度，使用角度控制
        if player_angle is not None:
            import math
            
            # 计算目标方向（游戏角度，北=0，顺时针）
            # Map Coords: Y increases Downwards.
            # dx = target.x - player.x
            # dy = target.y - player.y
            # Screen angle (East=0, CCW): atan2(-dy, dx)
            # Game angle (North=0, CW): (90 - screen_angle) % 360
            
            screen_angle = math.degrees(math.atan2(-dy, dx))
            target_angle = (90.0 - screen_angle) % 360.0
            
            # 计算角度差 (-180 到 180)
            diff = (target_angle - player_angle + 180) % 360 - 180
            
            # diff > 0：目标在当前朝向的顺时针方向（右侧） → 右转 → D
            # diff < 0：目标在当前朝向的逆时针方向（左侧） → 左转 → A
            if abs(diff) < 20:
                return "W"
            elif diff > 0:
                return "D"
            else:
                return "A"

        # 否则使用旧的坐标轴逻辑（仅适用于特定朝向假设）
        if abs(dx) > abs(dy):
            return "D" if dx > 0 else "A"
        else:
            return "S" if dy > 0 else "W"

    def get_state(self) -> MoveState:
        """获取移动状态"""
        return self._state

    def reset(self):
        """重置移动状态"""
        self._state = MoveState.IDLE
        self._current_path = []
        self._path_index = 0


# ============================================================================
# 对外接口层 (API)
# ============================================================================

class PathFinderAPI:
    """
    寻路模块对外接口
    提供统一的寻路服务
    """

    def __init__(self):
        self._map_data: Optional[MapData] = None
        self._planner: Optional[PathPlanner] = None
        self._mover: Optional[Mover] = None
        self._is_running = False
        self._debug_info: Dict = {}

    def load_trajectory(self, image_path: str, grid_size: int = 5) -> bool:
        """
        加载轨迹地图
        :param image_path: 轨迹图片路径
        :param grid_size: 网格大小
        :return: 是否成功
        """
        self._map_data = MapData.from_trajectory_image(image_path, grid_size)
        return self._post_load_map()

    def load_trajectory_json(self, json_path: str, grid_size: int = 5) -> bool:
        """
        从 JSON 文件加载轨迹地图
        :param json_path: JSON 文件路径
        :param grid_size: 网格大小
        :return: 是否成功
        """
        self._map_data = MapData.from_trajectory_json(json_path, grid_size)
        return self._post_load_map()

    def _post_load_map(self) -> bool:
        """加载地图后的后续处理"""
        if self._map_data:
            self._planner = PathPlanner(self._map_data)
            self._mover = Mover()
            self._debug_info['map_loaded'] = True
            self._debug_info['trajectory_points'] = len(self._map_data.trajectory_data.path_points) if self._map_data.trajectory_data else 0
            return True
        else:
            self._debug_info['map_loaded'] = False
            return False

    def start_pathfinding(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> bool:
        """
        开始寻路
        :param start_pos: 起点
        :param goal_pos: 终点
        :return: 是否成功
        """
        if not self._planner or not self._mover:
            self._debug_info['error'] = "寻路模块未初始化"
            return False

        # 规划路径
        result = self._planner.find_path(start_pos, goal_pos)

        if result.success:
            self._mover.set_path(result.path)
            self._is_running = True
            self._debug_info['path_length'] = result.length
            self._debug_info['strategy'] = result.strategy.name
            return True
        else:
            self._debug_info['error'] = result.message
            return False

    def update(self, player_pos: Tuple[float, float], player_angle: Optional[float] = None) -> Optional[MoveCommand]:
        """
        更新寻路状态
        :param player_pos: 玩家当前位置
        :param player_angle: 玩家当前朝向
        :return: 移动指令
        """
        if not self._is_running or not self._mover:
            return None

        command = self._mover.update(player_pos, player_angle)

        if self._mover.get_state() == MoveState.REACHED:
            self._is_running = False

        self._debug_info['player_pos'] = player_pos
        self._debug_info['path_index'] = self._mover._path_index
        self._debug_info['total_points'] = len(self._mover._current_path)

        return command

    def stop(self):
        """停止寻路"""
        self._is_running = False
        if self._mover:
            self._mover.reset()

    def get_status(self) -> Dict:
        """获取状态信息"""
        return {
            'is_running': self._is_running,
            'map_loaded': self._map_data is not None,
            'mover_state': self._mover.get_state().name if self._mover else 'IDLE',
            'debug': self._debug_info
        }

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        return self._debug_info.copy()


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    # 创建 API 实例
    pathfinder = PathFinderAPI()

    # 加载轨迹地图
    if pathfinder.load_trajectory(r"G:\map\001_trajectory.jpg"):
        print("✓ 地图加载成功")

        # 开始寻路
        start_pos = (0, 0)
        goal_pos = (500, 500)

        if pathfinder.start_pathfinding(start_pos, goal_pos):
            print("✓ 寻路启动成功")

            # 循环更新
            while True:
                player_pos = (100, 100)  # 从游戏获取
                command = pathfinder.update(player_pos)

                if command:
                    print(f"移动：{command.direction}")

                status = pathfinder.get_status()
                if not status['is_running']:
                    break
        else:
            print("✗ 寻路失败")
    else:
        print("✗ 地图加载失败")


if __name__ == "__main__":
    example_usage()