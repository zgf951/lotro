"""
自动战斗模块 - 核心逻辑
状态机：IDLE -> SEARCH -> MOVE -> FIGHT -> IDLE
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import time
import numpy as np
from core.pathfinder import PathFinderAPI, MoveCommand




class CombatState(Enum):
    """战斗状态枚举"""
    IDLE = auto()      # 空闲状态
    SEARCH = auto()    # 搜索怪物
    PATROL = auto()    # 巡逻中
    CLICK = auto()     # 点击怪物（锁定目标）
    MOVE = auto()      # 移动向怪物
    FIGHT = auto()     # 战斗中


@dataclass
class MonsterInfo:
    """怪物信息"""
    x1: float          # bounding box 左上 x（屏幕像素）
    y1: float          # bounding box 左上 y（屏幕像素）
    x2: float          # bounding box 右下 x（屏幕像素）
    y2: float          # bounding box 右下 y（屏幕像素）
    conf: float        # 置信度
    cls_id: int        # 类别 ID
    cx: float          # 中心点 x（屏幕像素，用于点击/攻击范围判断）
    cy: float          # 中心点 y（屏幕像素，用于点击/攻击范围判断）
    world_x: float = 0.0   # 中心点 x（画布世界坐标，用于 A* 寻路）
    world_y: float = 0.0   # 中心点 y（画布世界坐标，用于 A* 寻路）
    distance: float = 0.0  # 距玩家的距离（屏幕像素）


@dataclass
class CombatConfig:
    """战斗配置"""
    skill_keys: List[str] = field(default_factory=lambda: ['1', '2', '3', '4', '5'])
    attack_range: float = 50.0        # 攻击范围（像素）
    search_interval: float = 0.5      # 搜索间隔（秒）
    skill_delay: float = 0.3          # 技能释放间隔（秒）
    move_threshold: float = 10.0      # 移动阈值（像素）
    max_search_time: float = 10.0     # 最大搜索时间（秒）
    max_fight_time: float = 30.0      # 最大战斗时间（秒）
    ignored_classes: List[str] = field(default_factory=lambda: ['tank'])  # 忽略的类别名称


class CombatCore:
    """
    战斗核心类
    负责状态机管理、目标选择、移动和攻击决策
    """
    
    def __init__(self, config: Optional[CombatConfig] = None):
        self._state = CombatState.IDLE
        self._config = config or CombatConfig()
        self._current_target: Optional[MonsterInfo] = None
        self._last_search_time = 0.0
        self._last_skill_time = 0.0
        self._last_click_time = 0.0
        self._state_enter_time = 0.0
        self._skill_index = 0
        self._player_pos: Optional[Tuple[float, float]] = None
        self._enabled = False
        self._click_interval = 0.2  # 点击间隔（秒）
        
        # 寻路和巡逻相关
        self._pathfinder = PathFinderAPI()
        self._map_loaded = False
        self._move_command: Optional[MoveCommand] = None
        self._patrol_index = 0       # 当前巡逻到的轨迹点索引
        self._patrol_enabled = True  # 是否启用自动巡逻
        self._last_reset_vision_time = 0.0
        self._reset_vision_interval = 5.0  # 每 5 秒重置一次视觉以保持朝向
        self._need_zoom_out = False        # 是否需要拉远视角
        self._player_angle: float = 0.0    # 玩家朝向（游戏角度）
        
    @property
    def state(self) -> CombatState:
        """当前战斗状态"""
        return self._state
    
    @property
    def is_enabled(self) -> bool:
        """是否启用战斗"""
        return self._enabled
    
    @property
    def current_target(self) -> Optional[MonsterInfo]:
        """当前目标"""
        return self._current_target
    
    @property
    def config(self) -> CombatConfig:
        """战斗配置"""
        return self._config
    
    def set_config(self, config: CombatConfig):
        """更新战斗配置"""
        self._config = config
    
    def set_enabled(self, enabled: bool):
        """启用/禁用战斗"""
        self._enabled = enabled
        if not enabled:
            self._reset()
    
    def set_player_position(self, x: float, y: float):
        """设置玩家位置（基于拼图坐标）"""
        self._player_pos = (x, y)
    
    def set_player_angle(self, angle: float):
        """设置玩家朝向（游戏角度）"""
        self._player_angle = angle
    
    def load_map_json(self, json_path: str) -> bool:
        """加载寻路地图"""
        self._map_loaded = self._pathfinder.load_trajectory_json(json_path)
        if self._map_loaded:
            self._log_message = f"寻路地图加载成功：{json_path}"
            # 加载地图后，将巡逻点重置
            self._patrol_index = 0
        return self._map_loaded

    def start(self):
        """启动战斗"""
        if not self._enabled:
            return
        self._state = CombatState.SEARCH
        self._state_enter_time = time.perf_counter()
        self._last_search_time = 0.0
        
        # 如果加载了地图，初始化巡逻点为离玩家最近的点
        if self._map_loaded and self._player_pos:
            self._init_patrol_index()

    def _init_patrol_index(self):
        """将巡逻点初始化为离玩家最近的轨迹点"""
        if not self._map_loaded or not self._player_pos:
            return
            
        trajectory = self._pathfinder._map_data.trajectory_data.path_points
        if not trajectory:
            return
            
        min_dist = float('inf')
        nearest_idx = 0
        for i, point in enumerate(trajectory):
            dist = (point[0] - self._player_pos[0])**2 + (point[1] - self._player_pos[1])**2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        self._patrol_index = nearest_idx
        self._log_message = f"巡逻初始化：从轨迹点 {nearest_idx} 开始"
    
    def stop(self):
        """停止战斗"""
        self._reset()
    
    def _reset(self):
        """重置战斗状态"""
        self._state = CombatState.IDLE
        self._current_target = None
        self._last_search_time = 0.0
        self._last_skill_time = 0.0
        self._last_click_time = 0.0
        self._skill_index = 0
        self._state_enter_time = 0.0
    
    def update(self, detections: List[Tuple]) -> Optional[str]:
        """
        更新战斗状态
        :param detections: YOLO 检测结果列表 [(x1, y1, x2, y2, conf, cls_id, cx, cy), ...]
        :return: 需要按下的技能键，如果没有则返回 None
        """
        if not self._enabled:
            return None
        
        current_time = time.perf_counter()
        
        # 1. 优先拉远视角（针对 MOB）
        if self._need_zoom_out:
            self._need_zoom_out = False
            return "SCROLL"
            
        # 2. 定期重置视觉以保持朝向（巡逻或移动中）
        if self._state in [CombatState.PATROL, CombatState.MOVE]:
            if current_time - self._last_reset_vision_time > self._reset_vision_interval:
                self._last_reset_vision_time = current_time
                return "RESET_VISION"
        
        # 3. 状态机更新
        if self._state == CombatState.IDLE:
            return self._update_idle(current_time)
        elif self._state == CombatState.SEARCH:
            return self._update_search(detections, current_time)
        elif self._state == CombatState.PATROL:
            return self._update_patrol_state(detections, current_time)
        elif self._state == CombatState.CLICK:
            return self._update_click(detections, current_time)
        elif self._state == CombatState.MOVE:
            return self._update_move(current_time, detections)
        elif self._state == CombatState.FIGHT:
            return self._update_fight(detections, current_time)
        
        return None
    
    def _update_idle(self, current_time: float) -> Optional[str]:
        """空闲状态更新"""
        return None
    
    def _update_click(self, detections: List[Tuple], current_time: float) -> Optional[str]:
        """点击状态更新 - 点击最近的怪物锁定目标"""
        if not self._current_target:
            self._state = CombatState.SEARCH
            return None
        
        # 检查点击间隔
        if current_time - self._last_click_time < self._click_interval:
            return None
        
        self._last_click_time = current_time
        
        # 记录日志（cx/cy 此时为屏幕像素坐标）
        self._log_message = (f"点击怪物屏幕位置 "
                             f"({self._current_target.cx:.0f}, {self._current_target.cy:.0f})")
        
        # 切换到移动状态
        self._state = CombatState.MOVE
        self._state_enter_time = current_time
        
        # 返回特殊标记表示需要点击
        return "CLICK"
    
    def _update_search(self, detections: List[Tuple], current_time: float) -> Optional[str]:
        """搜索状态更新"""
        # 1. 检查是否检测到怪物
        monsters = self._process_detections(detections) if detections else []
        
        if monsters:
            # 选择最近的怪物
            monsters.sort(key=lambda m: m.distance)
            self._current_target = monsters[0]
            
            # 切换到锁定目标状态
            self._state = CombatState.CLICK
            self._state_enter_time = current_time
            self._last_click_time = 0.0
            self._need_zoom_out = True  # 发现怪物，准备拉远视角
            return None
            
        # 2. 如果没发现怪物，且加载了地图，切换到巡逻状态
        if self._map_loaded and self._patrol_enabled and self._player_pos:
            self._state = CombatState.PATROL
            self._state_enter_time = current_time
            return None
            
        # 3. 检查搜索超时
        if current_time - self._state_enter_time > self._config.max_search_time:
            self._state = CombatState.IDLE
            self._current_target = None
            return None
            
        return None

    def _update_patrol_state(self, detections: List[Tuple], current_time: float) -> Optional[str]:
        """巡逻状态更新"""
        # 1. 在巡逻过程中也要检查怪物
        monsters = self._process_detections(detections) if detections else []
        if monsters:
            # 发现怪物，停止巡逻寻路，切换到搜索（进而进入战斗）
            self._pathfinder.stop()
            self._state = CombatState.SEARCH
            self._state_enter_time = current_time
            return None
            
        # 2. 执行巡逻移动逻辑
        return self._update_patrol(current_time)

    # 玩家偏离轨迹超过此距离（世界坐标）时触发回归逻辑
    _OFF_TRACK_THRESHOLD = 60.0
    # 到达巡逻点的容差（世界坐标）
    _PATROL_REACH_THRESHOLD = 30.0
    # 巡逻点步进（跳过密集点）
    _PATROL_STEP = 10

    def _find_nearest_traj_idx(self, trajectory: list, pos: tuple) -> int:
        """在轨迹列表中找距 pos 最近的点的下标"""
        min_sq, idx = float('inf'), 0
        for i, p in enumerate(trajectory):
            sq = (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2
            if sq < min_sq:
                min_sq, idx = sq, i
        return idx

    def _update_patrol(self, current_time: float) -> Optional[str]:
        """
        巡逻状态更新：沿加载的轨迹循环移动，偏轨时自动回归。

        每帧流程：
          1. 找玩家最近的轨迹点（nearest_idx）
          2. 若玩家已超过 _patrol_index，推进索引（防止倒退）
          3. 若玩家偏离轨迹 > OFF_TRACK_THRESHOLD，目标临时改为 nearest 点
          4. 否则目标为 _patrol_index 对应的轨迹点
          5. 到达目标后步进 _patrol_index，循环
        """
        import math

        if not self._map_loaded or not self._player_pos:
            return None

        trajectory = self._pathfinder._map_data.trajectory_data.path_points
        if not trajectory:
            return None

        traj_len = len(trajectory)

        # ── 1. 每帧重定位：找最近轨迹点 ──────────────────────────────────
        nearest_idx = self._find_nearest_traj_idx(trajectory, self._player_pos)
        nearest_pt  = trajectory[nearest_idx]
        off_track   = math.hypot(nearest_pt[0] - self._player_pos[0],
                                  nearest_pt[1] - self._player_pos[1])

        # ── 2. 推进 patrol_index（不允许倒退） ───────────────────────────
        # 如果玩家在轨迹上的位置已超过当前 patrol_index，跟进
        if nearest_idx > self._patrol_index:
            self._patrol_index = nearest_idx

        # 循环边界
        if self._patrol_index >= traj_len:
            self._patrol_index = 0
            self._pathfinder.stop()
            self._log_message = "巡逻完成，从头循环"
            return None

        # ── 3. 确定本帧寻路目标 ───────────────────────────────────────────
        if off_track > self._OFF_TRACK_THRESHOLD:
            # 偏轨：先走回最近轨迹点
            target_point = nearest_pt
            if off_track > self._OFF_TRACK_THRESHOLD * 2:
                self._log_message = f"偏离轨迹 {off_track:.0f}px，正在返回最近点..."
        else:
            target_point = trajectory[self._patrol_index]

        # ── 4. 检查是否已到达目标点 ──────────────────────────────────────
        dx = target_point[0] - self._player_pos[0]
        dy = target_point[1] - self._player_pos[1]
        dist_to_target = math.hypot(dx, dy)

        if dist_to_target < self._PATROL_REACH_THRESHOLD:
            # 到达后步进
            self._patrol_index = min(
                self._patrol_index + self._PATROL_STEP, traj_len - 1
            )
            if self._patrol_index >= traj_len - 1:
                self._patrol_index = 0
                self._log_message = "巡逻完成，从头循环"
            self._pathfinder.stop()   # 让下一帧重新规划到新目标
            return None

        # ── 5. 启动/续用寻路 ─────────────────────────────────────────────
        if not self._pathfinder.get_status()['is_running']:
            ok = self._pathfinder.start_pathfinding(self._player_pos, target_point)
            if not ok:
                # 目标点可能在障碍物内，跳过几个点
                self._patrol_index = min(
                    self._patrol_index + 5, traj_len - 1
                )
                self._log_message = f"寻路失败，跳至轨迹点 {self._patrol_index}"
                return None

        cmd = self._pathfinder.update(self._player_pos, self._player_angle)
        if cmd:
            return cmd.direction

        return None
    
    def _process_detections(self, detections: List[Tuple]) -> List[MonsterInfo]:
        """处理检测结果，计算每个怪物的距离

        检测 tuple 格式（11 元素，由 main_window 填充）：
          [0-3]  x1, y1, x2, y2  —— bounding box（屏幕像素）
          [4]    conf
          [5]    cls_id
          [6]    cx              —— 中心 X（屏幕像素）
          [7]    cy              —— 中心 Y（屏幕像素）
          [8]    cls_name
          [9]    world_x         —— 画布世界坐标 X（可选，不存在时退化为 cx）
          [10]   world_y         —— 画布世界坐标 Y（可选，不存在时退化为 cy）

        distance 始终以屏幕像素度量（攻击范围配置亦为屏幕像素）。
        """
        monsters = []
        for det in detections:
            if len(det) < 8:
                continue

            # 检查类别名称并过滤
            cls_name = det[8] if len(det) >= 9 else ""
            if cls_name in self._config.ignored_classes:
                continue

            x1, y1, x2, y2, conf, cls_id, cx_screen, cy_screen = det[:8]

            # 世界坐标：优先使用扩展字段，否则退化为屏幕坐标
            world_x = float(det[9])  if len(det) >= 10 else float(cx_screen)
            world_y = float(det[10]) if len(det) >= 11 else float(cy_screen)

            monster = MonsterInfo(
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=conf, cls_id=cls_id,
                cx=cx_screen, cy=cy_screen,
                world_x=world_x, world_y=world_y,
            )

            # distance 用屏幕像素计算，与 attack_range 保持同一单位
            # 玩家始终在屏幕中央附近，用 (640, 360) 作为参考中心
            # （即使没有精确玩家位置，误差也在可接受范围内）
            monster.distance = ((cx_screen - 640) ** 2 +
                                (cy_screen - 360) ** 2) ** 0.5
            monsters.append(monster)
        return monsters
    
    def _update_move(self, current_time: float, detections: Optional[List[Tuple]] = None) -> Optional[str]:
        """移动状态更新"""
        if not self._current_target:
            self._state = CombatState.SEARCH
            return None

        # 用新检测结果刷新目标的屏幕坐标与世界坐标
        # 追踪容差用屏幕像素，与检测精度一致
        if detections and self._current_target:
            for det in detections:
                if len(det) < 8:
                    continue
                cx_new, cy_new = det[6], det[7]
                screen_dist = ((cx_new - self._current_target.cx) ** 2 +
                               (cy_new - self._current_target.cy) ** 2) ** 0.5
                if screen_dist < 80:   # 80 屏幕像素容差（宽松一些，应对抖动）
                    self._current_target.cx = cx_new
                    self._current_target.cy = cy_new
                    # 同步世界坐标
                    self._current_target.world_x = float(det[9])  if len(det) >= 10 else cx_new
                    self._current_target.world_y = float(det[10]) if len(det) >= 11 else cy_new
                    # 刷新屏幕距离
                    self._current_target.distance = ((cx_new - 640) ** 2 +
                                                     (cy_new - 360) ** 2) ** 0.5
                    break

        # 进入攻击范围（屏幕像素距离 vs 配置的屏幕像素范围）
        if self._current_target.distance <= self._config.attack_range:
            self._state = CombatState.FIGHT
            self._state_enter_time = current_time
            self._skill_index = 0
            self._pathfinder.stop()
            return None

        # 超过最大移动时间，放弃目标
        if current_time - self._state_enter_time > self._config.max_search_time:
            self._state = CombatState.IDLE
            self._current_target = None
            self._pathfinder.stop()
            return None

        # 使用 A* 寻路（世界坐标）
        if self._map_loaded and self._player_pos:
            if not self._pathfinder.get_status()['is_running']:
                # 目标的世界坐标作为寻路终点
                goal_pos = (self._current_target.world_x, self._current_target.world_y)
                if self._pathfinder.start_pathfinding(self._player_pos, goal_pos):
                    self._log_message = f"寻路移动到世界坐标：{goal_pos}"
                else:
                    self._log_message = "寻路失败，等待下一帧重试"

            cmd = self._pathfinder.update(self._player_pos, self._player_angle)
            if cmd:
                return cmd.direction

        return None
    
    def _update_fight(self, detections: List[Tuple], current_time: float) -> Optional[str]:
        """战斗状态更新"""
        # 超过最大战斗时间，放弃
        if current_time - self._state_enter_time > self._config.max_fight_time:
            self._log_message = "战斗超时，返回搜索状态"
            self._state = CombatState.IDLE
            self._current_target = None
            return None

        # 检查目标是否还在视野内，同时刷新其屏幕坐标
        # 追踪容差用屏幕像素
        target_exists = False
        if self._current_target:
            for det in detections:
                if len(det) < 8:
                    continue
                cx_new, cy_new = det[6], det[7]
                screen_dist = ((cx_new - self._current_target.cx) ** 2 +
                               (cy_new - self._current_target.cy) ** 2) ** 0.5
                if screen_dist < 80:   # 80 屏幕像素容差
                    target_exists = True
                    self._current_target.cx = cx_new
                    self._current_target.cy = cy_new
                    self._current_target.world_x = float(det[9])  if len(det) >= 10 else cx_new
                    self._current_target.world_y = float(det[10]) if len(det) >= 11 else cy_new
                    self._current_target.distance = ((cx_new - 640) ** 2 +
                                                     (cy_new - 360) ** 2) ** 0.5
                    break

        if not target_exists:
            self._log_message = "目标消失，重新搜索"
            self._current_target = None
            self._state = CombatState.SEARCH
            self._state_enter_time = current_time
            return None

        # 技能释放
        if current_time - self._last_skill_time >= self._config.skill_delay:
            self._last_skill_time = current_time
            skill_key = self._config.skill_keys[self._skill_index % len(self._config.skill_keys)]
            self._skill_index += 1
            return skill_key

        return None
    
    def get_status_text(self) -> str:
        """获取状态文本"""
        state_map = {
            CombatState.IDLE: "空闲",
            CombatState.SEARCH: "搜索中",
            CombatState.PATROL: "巡逻中",
            CombatState.CLICK: "锁定目标",
            CombatState.MOVE: "移动中",
            CombatState.FIGHT: "战斗中"
        }
        return state_map.get(self._state, "未知")
    
    def get_target_text(self) -> str:
        """获取目标文本"""
        if self._current_target:
            return f"怪物 (距离：{self._current_target.distance:.1f})"
        return "无"
    
    def get_and_clear_log(self) -> Optional[str]:
        """获取并清除日志消息"""
        if hasattr(self, '_log_message') and self._log_message:
            msg = self._log_message
            self._log_message = None
            return msg
        return None
