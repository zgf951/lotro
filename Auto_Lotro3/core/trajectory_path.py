"""
轨迹寻路模块
直接使用拼图系统实时记录的轨迹数据，而不是从图片识别
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TrajectoryPath:
    """轨迹路径数据"""
    points: List[Tuple[float, float]]  # 玩家位置序列（世界坐标）
    start_point: Optional[Tuple[float, float]] = None
    end_point: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.points:
            self.start_point = self.points[0]
            self.end_point = self.points[-1]
    
    @property
    def length(self) -> int:
        """路径点数量"""
        return len(self.points)
    
    def get_grid_path(self, grid_size: int = 5) -> List[Tuple[int, int]]:
        """转换为网格路径"""
        return [(int(x // grid_size), int(y // grid_size)) for x, y in self.points]


class TrajectoryRecorder:
    """
    轨迹记录器
    记录玩家在拼图过程中的移动路径
    """
    
    def __init__(self):
        self._trajectory: List[Tuple[float, float]] = []
        self._is_recording = False
    
    def start_recording(self):
        """开始记录"""
        self._trajectory = []
        self._is_recording = True
    
    def stop_recording(self):
        """停止记录"""
        self._is_recording = False
    
    def add_point(self, world_x: float, world_y: float):
        """
        添加一个轨迹点
        :param world_x: 玩家世界坐标 X
        :param world_y: 玩家世界坐标 Y
        """
        if self._is_recording:
            self._trajectory.append((world_x, world_y))
    
    def get_trajectory(self) -> Optional[TrajectoryPath]:
        """获取当前记录的轨迹"""
        if not self._trajectory:
            return None
        
        return TrajectoryPath(points=self._trajectory.copy())
    
    def clear(self):
        """清空轨迹"""
        self._trajectory = []
        self._is_recording = False
    
    @property
    def is_recording(self) -> bool:
        """是否正在记录"""
        return self._is_recording
    
    @property
    def point_count(self) -> int:
        """轨迹点数量"""
        return len(self._trajectory)


class TrajectoryFollower:
    """
    轨迹跟随器
    沿着记录的轨迹路径移动
    """
    
    def __init__(self, grid_size: int = 5):
        self._grid_size = grid_size
        self._trajectory: Optional[TrajectoryPath] = None
        self._grid_path: List[Tuple[int, int]] = []
        self._current_index = 0
        self._is_following = False
    
    def set_trajectory(self, trajectory: TrajectoryPath):
        """设置要跟随的轨迹"""
        self._trajectory = trajectory
        self._grid_path = trajectory.get_grid_path(self._grid_size)
        self._current_index = 0
        self._is_following = False
    
    def start_following(self):
        """开始跟随"""
        if self._grid_path:
            self._is_following = True
            self._current_index = 0
    
    def stop_following(self):
        """停止跟随"""
        self._is_following = False
    
    def update(self, player_x: float, player_y: float, player_angle: Optional[float] = None) -> Optional[str]:
        """
        更新跟随状态
        :param player_x: 玩家当前世界坐标 X
        :param player_y: 玩家当前世界坐标 Y
        :param player_angle: 玩家当前朝向 (游戏角度)
        :return: 移动方向 (W/A/S/D) 或 None
        """
        if not self._is_following or not self._grid_path:
            return None
        
        # 检查是否已到达终点
        if self._current_index >= len(self._grid_path):
            self._is_following = False
            return None
        
        # 获取下一个目标点
        next_grid = self._grid_path[self._current_index]
        next_x = next_grid[0] * self._grid_size
        next_y = next_grid[1] * self._grid_size
        
        # 计算距离
        dx = next_x - player_x
        dy = next_y - player_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # 如果到达当前目标点，移动到下一个点
        # 提高容差，避免在目标点附近打转
        if distance < self._grid_size * 2:
            self._current_index += 1
            if self._current_index >= len(self._grid_path):
                self._is_following = False
                return None
            
            # 递归调用以获取下一个点的方向
            return self.update(player_x, player_y, player_angle)
        
        # 计算移动方向
        return self._calc_direction(dx, dy, player_angle)

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
            
            # diff > 0：目标在顺时针方向（右侧） → 右转 → D
            # diff < 0：目标在逆时针方向（左侧） → 左转 → A
            if abs(diff) < 20:
                return "W"
            elif diff > 0:
                return "D"
            else:
                return "A"

        # 否则使用旧的坐标轴逻辑
        if abs(dx) > abs(dy):
            return "D" if dx > 0 else "A"
        else:
            return "S" if dy > 0 else "W"

    def get_progress(self) -> Tuple[int, int]:
        """获取当前进度 (当前索引，总点数)"""
        return (self._current_index, len(self._grid_path))

    def reset(self):
        """重置跟随状态"""
        self._current_index = 0
        self._is_following = False


class TrajectoryManager:
    """
    轨迹管理器
    统一管理轨迹记录和跟随
    """

    def __init__(self, grid_size: int = 5):
        self._recorder = TrajectoryRecorder()
        self._follower = TrajectoryFollower(grid_size)
        self._saved_trajectories: Dict[str, TrajectoryPath] = {}
        self._grid_size = grid_size

    def start_recording(self):
        """开始记录轨迹"""
        self._recorder.start_recording()

    def stop_recording(self):
        """停止记录轨迹"""
        self._recorder.stop_recording()

    def add_point(self, world_x: float, world_y: float):
        """添加轨迹点"""
        self._recorder.add_point(world_x, world_y)

    def save_current_trajectory(self, name: str) -> bool:
        """
        保存当前轨迹
        :param name: 轨迹名称
        :return: 是否成功
        """
        trajectory = self._recorder.get_trajectory()
        if trajectory:
            self._saved_trajectories[name] = trajectory
            return True
        return False

    def load_trajectory(self, name: str) -> Optional[TrajectoryPath]:
        """加载已保存的轨迹"""
        return self._saved_trajectories.get(name)

    def start_following(self, trajectory: TrajectoryPath):
        """开始跟随轨迹"""
        self._follower.set_trajectory(trajectory)
        self._follower.start_following()

    def stop_following(self):
        """停止跟随"""
        self._follower.stop_following()

    def update(self, player_x: float, player_y: float, player_angle: Optional[float] = None) -> Optional[str]:
        """更新跟随状态"""
        return self._follower.update(player_x, player_y, player_angle)

    def get_progress(self) -> Tuple[int, int]:
        """获取跟随进度"""
        return self._follower.get_progress()

    def is_following(self) -> bool:
        """是否正在跟随"""
        return self._follower._is_following

    def is_recording(self) -> bool:
        """是否正在记录"""
        return self._recorder.is_recording

    @property
    def recorded_point_count(self) -> int:
        """当前记录的点数"""
        return self._recorder.point_count