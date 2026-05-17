# LOTRO 项目 Code Wiki

## 1. 项目概览

LOTRO 是一个针对《指环王Online》(The Lord of the Rings Online)游戏的自动化辅助工具，主要提供以下功能：

- **迷你地图拼接**：通过计算机视觉技术实时拼接游戏小地图，构建完整的游戏世界地图
- **YOLO 目标检测**：使用深度学习模型检测游戏中的怪物和其他目标
- **自动战斗系统**：基于检测结果自动进行战斗操作
- **轨迹寻路**：记录和跟随预设的移动路径
- **玩家朝向检测**：通过箭头检测确定玩家在游戏中的朝向

**典型应用场景**：自动练级、资源采集、地图探索

---

## 2. 目录结构

项目采用模块化设计，将不同功能组件分离到独立的目录中，便于维护和扩展。

```
Auto_Lotro3/
├── astar_pathfinder/          # A* 寻路算法实现
│   ├── __init__.py
│   ├── astar.py              # A* 路径搜索核心
│   ├── grid_map.py           # 网格地图表示
│   └── path_smoother.py      # 路径平滑优化
│
├── core/                      # 核心功能模块
│   ├── stitcher.py           # 迷你地图拼接（MniMap、StitchWorker）
│   ├── combat.py             # 自动战斗核心逻辑
│   ├── combat_worker.py      # 战斗工作线程、按键模拟
│   ├── pathfinder.py         # 寻路模块 API（分层架构）
│   ├── trajectory_path.py    # 轨迹记录与跟随
│   ├── trajectory_analyzer.py # 轨迹分析与处理
│   ├── map_data_saver.py     # 地图数据持久化
│   └── contour_matcher.py     # 轮廓匹配工具
│
├── ui/                        # 图形用户界面
│   ├── main_window.py        # 主窗口（PySide6）
│   ├── widgets.py            # UI 组件（画布、区域选择器等）
│   ├── detection_viewer.py   # 独立检测结果显示窗口
│   └── contour_calibrator.py # 轮廓标定工具
│
├── utils/                     # 工具类和辅助函数
│   ├── window_manager.py     # 游戏窗口管理
│   ├── win32_utils.py        # Windows API 封装
│   └── dxgi_capture.py       # DXGI 屏幕捕获（dxcam）
│
├── calibrate_color.py         # 颜色标定工具
├── calibrate_minimap.py       # 小地图标定工具
├── lotro_arrow_v5.py         # 箭头检测/方位角识别
├── main.py                   # 主入口文件
└── test.py                   # 测试文件
```

### 核心目录职责

| 目录/文件 | 职责 | 主要内容 |
|-----------|------|----------|
| `core/` | 核心功能实现 | 地图拼接、战斗逻辑、路径规划 |
| `ui/` | 用户界面 | 主窗口、检测查看器、轮廓标定工具 |
| `utils/` | 工具函数 | 窗口管理、屏幕捕获、系统操作 |
| `astar_pathfinder/` | 寻路算法 | A* 路径搜索、网格地图处理 |

---

## 3. 系统架构与主流程

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         应用层 (Application)                         │
├─────────────────────────────────────────────────────────────────────┤
│  MainWindow (ui/main_window.py)                                     │
│  ├── 控制面板：YOLO模型加载、战斗控制、轨迹管理                      │
│  ├── CanvasWidget：地图显示、轨迹绘制                                │
│  └── 信号槽通信：worker线程与UI交互                                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      工作线程层 (Worker Threads)                      │
├─────────────────────────────────────────────────────────────────────┤
│  StitchWorker              │  CombatWorker                          │
│  ├── 屏幕截图捕获           │  ├── 战斗状态机管理                      │
│  ├── 地图拼接（SIFT/FLANN） │  ├── 目标选择与追踪                      │
│  ├── YOLO 目标检测          │  ├── 寻路请求                           │
│  └── 箭头朝向检测           │  └── 技能释放                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       核心算法层 (Core Algorithms)                  │
├─────────────────────────────────────────────────────────────────────┤
│  MiniMap (拼接)      │  CombatCore (战斗)   │  PathFinderAPI (寻路) │
│  ├── SIFT 特征提取   │  ├── 状态机         │  ├── 轨迹寻路         │
│  ├── FLANN 匹配      │  ├── 目标处理       │  ├── A* 算法         │
│  └── 仿射变换拼接    │  └── 技能调度       │  └── Mover 移动控制   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      系统接口层 (System Interfaces)                  │
├─────────────────────────────────────────────────────────────────────┤
│  WindowManager        │  DxgiWindowCapture     │  KeySimulator     │
│  ├── 窗口绑定         │  ├── DXGI 截图        │  ├── 按键模拟      │
│  ├── 坐标转换         │  ├── 区域裁剪         │  └── 鼠标点击      │
│  └── 前台管理         │  └── BGR 转换         │                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 主要业务流程

#### 流程 1：迷你地图拼接

```
用户操作                后台处理                    结果输出
   │                      │                          │
   ▼                      ▼                          ▼
┌──────────┐        ┌──────────────┐          ┌──────────┐
│ 捕获窗口  │──────▶│ bitblt_capture│          │          │
└──────────┘        └──────────────┘          │          │
   │                      │                   │  拼接    │
   ▼                      ▼                   │  地图    │
┌──────────┐        ┌──────────────┐          │  显示    │
│ 框选区域  │──────▶│ SIFT 特征提取 │          │          │
└──────────┘        └──────────────┘          │          │
   │                      │                   │          │
   ▼                      ▼                   │          │
┌──────────┐        ┌──────────────┐          │          │
│ 开始拼图  │──────▶│ FLANN 特征匹配│──────────┼────────▶│
└──────────┘        └──────────────┘          │          │
   │                      │                   │          │
   ▼                      ▼                   │          │
┌──────────┐        ┌──────────────┐          │          │
│ 结束保存  │◀──────│ 仿射变换拼接  │          └──────────┘
└──────────┘        └──────────────┘
```

#### 流程 2：自动战斗

```
检测输入                    战斗状态机                        操作输出
   │                           │                               │
   ▼                           ▼                               ▼
┌──────────┐            ┌──────────────┐              ┌──────────────┐
│ YOLO     │──────────▶ │    IDLE      │              │              │
│ 目标检测  │            │  (空闲)      │              │              │
└──────────┘            └──────────────┘              │              │
                               │                       │   按键/点击   │
                               ▼                       │              │
                        ┌──────────────┐               │              │
                        │   SEARCH     │◀─────┐        │              │
                        │  (搜索怪物)  │      │        │              │
                        └──────────────┘      │        │              │
                               │              │        │              │
                    发现怪物   ▼              │        │              │
                        ┌──────────────┐     │        │              │
                        │   CLICK      │─────┘        │              │
                        │  (锁定目标)  │              │              │
                        └──────────────┘              │              │
                               │                       │              │
                               ▼                       │              │
                        ┌──────────────┐              │              │
           未达范围     │    MOVE      │     到达范围   │              │
        ┌─────────────▶│  (移动中)    │──────────────┼──────▶ 点击目标
        │              └──────────────┘              │              │
        │                     │                      │              │
        │                     ▼                      │              │
        │              ┌──────────────┐              │              │
        │              │   FIGHT      │──────────────┼──────▶ 技能释放
        │              │  (战斗中)    │              │              │
        │              └──────────────┘              │              │
        │                     │                      │              │
        │        目标消失     ▼                      │              │
        └────────────────────IDLE◀───────────────────┘              │
                               │                                   │
                               ▼                                   │
                        ┌──────────────┐                          │
                        │   PATROL     │◀──────────────────────────┘
                        │  (巡逻)       │  无怪物且已加载地图
                        └──────────────┘
```

#### 流程 3：轨迹寻路

```
用户操作              记录模式                    跟随模式
   │                     │                          │
   ▼                     ▼                          ▼
┌──────────┐        ┌──────────────┐          ┌──────────────┐
│ 输入名称  │        │ StitchWorker │          │ 加载轨迹文件  │
└──────────┘        └──────────────┘          └──────────────┘
   │                     │                          │
   ▼                     ▼                          ▼
┌──────────┐        ┌──────────────┐          ┌──────────────┐
│ 开始记录  │──────▶│ MiniMap.update│          │ 轨迹分割     │
└──────────┘        │ 获取玩家位置  │          │ 路径规划     │
   │                └──────────────┘          └──────────────┘
   ▼                     │                          │
┌──────────┐        ┌──────────────┐          ┌──────────────┐
│ 游戏中   │◀──────│ Trajectory   │          │ Mover 控制   │
│ 移动     │        │ Manager      │          │ WASD 按键    │
└──────────┘        └──────────────┘          └──────────────┘
   │                     │                          │
   ▼                     ▼                          ▼
┌──────────┐        ┌──────────────┐          ┌──────────────┐
│ 停止记录  │        │ 位置点记录   │          │ 路径跟随     │
└──────────┘        │ 入列表       │          │ 循环移动     │
   │                └──────────────┘          └──────────────┘
   ▼                     │                          │
┌──────────┐        ┌──────────────┐          ┌──────────────┐
│ 保存轨迹  │──────▶│ JSON + 图片  │          │ 到达终点     │
└──────────┘        │ 文件输出     │          │ 循环/停止    │
                    └──────────────┘          └──────────────┘
```

---

## 4. 核心模块详解

### 4.1 MiniMap 类

**文件**：[core/stitcher.py#L14](file:///workspace/Auto_Lotro3/core/stitcher.py#L14)

**功能**：负责小地图的实时拼接和玩家位置追踪。

**核心属性**：
```python
self.canvas              # 4000x4000 拼接画布
self.sift               # SIFT 特征提取器
self.matcher            # FLANN 匹配器
self._M_global          # 全局仿射变换矩阵
self.trajectory          # 玩家移动轨迹点列表
self._facing_angle      # 玩家朝向（游戏角度）
```

**主要方法**：

| 方法 | 签名 | 功能 |
|------|------|------|
| `__init__` | `(canvas_h=4000, canvas_w=4000)` | 初始化拼接器 |
| `update` | `(img: np.ndarray) -> bool` | 添加新帧并更新画布 |
| `update_match_only` | `(img) -> Optional[Tuple[float, float]]` | 仅匹配不更新画布 |
| `get_player_position` | `() -> Optional[Tuple[float, float]]` | 获取画布坐标 |
| `get_player_angle` | `() -> Optional[float]` | 获取朝向角度 |
| `pixel_to_world` | `(x, y) -> Tuple[float, float]` | 像素→世界坐标转换 |
| `set_canvas` | `(canvas, offset)` | 预加载画布（用于寻路） |

**拼接算法流程**：
```
输入帧 → SIFT特征提取 → FLANN匹配 → 筛选good matches
                                    ↓
                             仿射变换估计
                                    ↓
                         计算帧间位移 M_inc
                                    ↓
                         更新全局变换 M_global
                                    ↓
                         透视变换拼接到画布
```

---

### 4.2 StitchWorker 类

**文件**：[core/stitcher.py#L193](file:///workspace/Auto_Lotro3/core/stitcher.py#L193)

**功能**：在后台线程中执行屏幕捕获、地图拼接、目标检测和朝向识别。

**信号定义**：
```python
log_signal = Signal(str)           # 日志输出
detections_signal = Signal(list)   # YOLO检测结果
frame_signal = Signal(object, list, float)  # 帧数据+检测+FPS
finished = Signal()                # 线程结束
```

**核心逻辑**：
```python
while self._running:
    # 1. 捕获小地图区域截图
    curr = bitblt_capture(hwnd, region)
    
    # 2. 箭头检测（更新朝向）
    res = arrow_v5.detect(curr)
    if res:
        self._minimap._facing_angle = arrow_v5.smooth_filter(res["bearing"])
    
    # 3. YOLO 目标检测（可选）
    if self._yolo_model and self._yolo_detecting:
        results = self._yolo_model.predict(source=detect_img, ...)
        detections = process_results(results)
        self.detections_signal.emit(detections)
    
    # 4. 地图拼接
    if self._match_only:
        pos = self._minimap.update_match_only(curr)  # 仅定位
    else:
        ok = self._minimap.update(curr)  # 拼接+定位
```

---

### 4.3 CombatCore 类

**文件**：[core/combat.py#L55](file:///workspace/Auto_Lotro3/core/combat.py#L55)

**功能**：战斗状态机核心，处理目标检测、路径规划和技能调度。

**状态枚举**：
```python
class CombatState(Enum):
    IDLE    = auto()   # 空闲
    SEARCH  = auto()   # 搜索怪物
    PATROL  = auto()   # 巡逻
    CLICK   = auto()   # 点击目标
    MOVE    = auto()   # 移动
    FIGHT   = auto()   # 战斗
```

**配置类**：
```python
@dataclass
class CombatConfig:
    skill_keys: List[str]     # 技能按键列表，默认 ['1','2','3','4','5']
    attack_range: float       # 攻击范围，默认 50 像素
    search_interval: float    # 搜索间隔，默认 0.5 秒
    skill_delay: float        # 技能释放间隔，默认 0.3 秒
    max_search_time: float    # 最大搜索时间，默认 10 秒
    max_fight_time: float     # 最大战斗时间，默认 30 秒
    ignored_classes: List[str]  # 忽略的类别，默认 ['tank']
```

**状态转换图**：
```
                    ┌─────────┐
                    │  IDLE   │
                    └────┬────┘
                         │ start()
                         ▼
              ┌──────────────────┐
              │      SEARCH      │
              └────────┬─────────┘
                       │ 发现怪物
                       ▼
              ┌──────────────────┐     未达范围
              │      CLICK       │─────────────────┐
              └──────────────────┘                  │
                       │                           │
                       ▼                           ▼
              ┌──────────────────┐      ┌──────────────────┐
              │       MOVE        │      │      PATROL      │ ◀─┐
              └────────┬─────────┘      └──────────────────┘   │ 巡逻时
                       │                                    │ 发现怪物
                       │ 到达范围                          │
                       ▼                                    │
              ┌──────────────────┐                          │
              │      FIGHT       │──────────────────────────┘
              └────────┬─────────┘      目标消失或超时
                       │
                       ▼
                    IDLE
```

---

### 4.4 PathFinderAPI 类

**文件**：[core/pathfinder.py#L483](file:///workspace/Auto_Lotro3/core/pathfinder.py#L483)

**功能**：提供统一的寻路服务接口，支持轨迹寻路和 A* 算法。

**分层架构**：

```
PathFinderAPI (对外接口层)
      │
      ├── PathPlanner (路径规划层)
      │       │
      │       ├── A* 算法 (AStarPathfinder)
      │       └── 轨迹跟随 (基于预存路径)
      │
      └── Mover (移动控制层)
              │
              ├── WASD 方向计算
              └── 路径跟随
```

**核心方法**：

| 方法 | 功能 |
|------|------|
| `load_trajectory_json(json_path)` | 从 JSON 加载轨迹地图 |
| `start_pathfinding(start, goal)` | 启动路径规划 |
| `update(player_pos, angle)` | 更新寻路状态，返回移动指令 |
| `stop()` | 停止寻路 |
| `get_status()` | 获取运行状态 |

---

### 4.5 TrajectoryManager 类

**文件**：[core/trajectory_path.py#L198](file:///workspace/Auto_Lotro3/core/trajectory_path.py#L198)

**功能**：管理轨迹的记录、保存和跟随。

**组成组件**：
```python
TrajectoryManager
├── TrajectoryRecorder   # 轨迹记录器
│       ├── start_recording()
│       ├── stop_recording()
│       └── add_point(x, y)
│
└── TrajectoryFollower   # 轨迹跟随器
        ├── set_trajectory()
        ├── start_following()
        └── update(player_pos) -> direction
```

**使用示例**：
```python
manager = TrajectoryManager(grid_size=5)

# 记录模式
manager.start_recording()
while moving:
    manager.add_point(player_x, player_y)
manager.stop_recording()
manager.save_current_trajectory("route_001")

# 跟随模式
trajectory = manager.load_trajectory("route_001")
manager.start_following(trajectory)
while manager.is_following():
    cmd = manager.update(player_x, player_y, player_angle)
    if cmd:
        KeySimulator.press_key(cmd)
```

---

### 4.6 AStarPathfinder 类

**文件**：[astar_pathfinder/astar.py#L69](file:///workspace/Auto_Lotro3/astar_pathfinder/astar.py#L69)

**功能**：A* 寻路算法实现，支持多种启发式函数。

**启发式类型**：
```python
class HeuristicType(Enum):
    MANHATTAN = "manhattan"   # 曼哈顿距离
    EUCLIDEAN = "euclidean"   # 欧几里得距离
    CHEBYSHEV = "chebyshev"   # 切比雪夫距离
    OCTILE    = "octile"      # 八方向距离（默认）
```

**算法流程**：
```python
def find_path(start, goal, timeout=1.0):
    # 1. 初始化
    open_set = [(start, h(start, goal))]  # 优先队列
    g_scores = {start: 0}                  # 实际代价
    came_from = {}                         # 父节点映射
    
    while open_set:
        # 2. 取最优节点
        current = pop_min_f_cost(open_set)
        
        # 3. 到达终点
        if current == goal:
            return reconstruct_path(came_from, goal)
        
        # 4. 扩展邻居
        for neighbor in get_neighbors(current):
            tentative_g = g_scores[current] + move_cost(current, neighbor)
            if tentative_g < g_scores.get(neighbor, inf):
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + h(neighbor, goal)
                push(open_set, (neighbor, f))
    
    return None  # 无路径
```

---

### 4.7 GridMap 类

**文件**：[astar_pathfinder/grid_map.py#L30](file:///workspace/Auto_Lotro3/astar_pathfinder/grid_map.py#L30)

**功能**：网格地图表示，支持障碍物标记和代价查询。

**单元类型**：
```python
class CellType(IntEnum):
    EMPTY     = 0    # 空白区域
    OBSTACLE  = 1    # 障碍物
    EXPLORED  = 2    # 已探索区域
    UNKNOWN   = 3    # 未知区域
```

**核心方法**：

| 方法 | 功能 |
|------|------|
| `world_to_grid(x, y)` | 世界坐标→网格坐标 |
| `grid_to_world(gx, gy)` | 网格坐标→世界坐标 |
| `is_obstacle(x, y)` | 检查是否为障碍物 |
| `get_neighbors(x, y)` | 获取相邻可通行格子 |
| `get_cost(x, y)` | 获取移动代价 |

---

### 4.8 WindowManager 类

**文件**：[utils/window_manager.py#L14](file:///workspace/Auto_Lotro3/utils/window_manager.py#L14)

**功能**：管理游戏窗口的绑定和操作。

**绑定方式**：
```python
# 方式1：通过鼠标位置
manager.bind_by_cursor()

# 方式2：通过窗口句柄
manager.bind_by_hwnd(hwnd)

# 方式3：通过进程ID
manager.bind_by_pid(pid)
```

**核心方法**：

| 方法 | 功能 |
|------|------|
| `is_valid()` | 检查窗口是否有效 |
| `is_minimized()` | 检查是否最小化 |
| `bring_to_front()` | 将窗口置于前台 |
| `client_to_screen(x, y)` | 客户区坐标→屏幕坐标 |
| `screen_to_client(x, y)` | 屏幕坐标→客户区坐标 |
| `get_info()` | 获取窗口详细信息 |

---

### 4.9 KeySimulator 类

**文件**：[core/combat_worker.py#L16](file:///workspace/Auto_Lotro3/core/combat_worker.py#L16)

**功能**：模拟键盘和鼠标操作（基于 pydirectinput）。

**主要方法**：
```python
class KeySimulator:
    @staticmethod
    def press_key(key: str):
        """模拟按键"""
        # 支持: '1'-'9', 'a'-'z', 'f1'-'f8', 'space', 'shift', 'ctrl', 'alt'
        pydirectinput.press(key)
    
    @staticmethod
    def click_at(x: float, y: float):
        """鼠标移动并点击"""
        pydirectinput.moveTo(int(x), int(y))
        pydirectinput.click()
    
    @staticmethod
    def scroll_down(clicks: int = 1):
        """向下滚动（拉远视角）"""
        for _ in range(clicks):
            pydirectinput.scroll(-120)
```

---

### 4.10 lotro_arrow_v5 模块

**文件**：[lotro_arrow_v5.py](file:///workspace/Auto_Lotro3/lotro_arrow_v5.py)

**功能**：检测小地图中的箭头，计算玩家朝向。

**算法流程**：
```
输入帧 → 放大4倍(INTER_NEAREST) → HSV颜色提取(红/橙色)
                                   ↓
                           形态学闭运算去噪
                                   ↓
                           找最大轮廓(重心在中心附近)
                                   ↓
                           approxPolyDP 找顶点
                                   ↓
                           计算最小内角顶点=箭尖
                                   ↓
                           atan2 → 罗盘方位(北=0,顺时针)
                                   ↓
                           平滑滤波(smooth_filter)
```

**方位计算**：
```python
# 罗盘方位 = 0-360，北=0，东=90
bearing = (90.0 - math_ang + 360.0) % 360.0

# 方位标签
compass_dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
compass = compass_dirs[int((bearing + 22.5) // 45) % 8]
```

---

## 5. 依赖关系图

```
依赖层级（从上到下）

┌────────────────────────────────────────────────────────────────┐
│                         用户接口层                              │
│   main.py ──▶ PySide6 GUI (ui/main_window.py)                  │
│                   ├── CanvasWidget (ui/widgets.py)              │
│                   ├── DetectionViewer (ui/detection_viewer.py)  │
│                   └── ContourCalibrator (ui/contour_calibrator) │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                        业务逻辑层                               │
│   StitchWorker (core/stitcher.py)                               │
│       ├── MiniMap ──▶ SIFT/FLANN (opencv-contrib)              │
│       ├── arrow_v5 ──▶ HSV检测                                  │
│       └── YOLO模型 (ultralytics)                               │
│                                                                │
│   CombatWorker (core/combat_worker.py)                         │
│       ├── CombatCore ──▶ CombatStateMachine                     │
│       ├── PathFinderAPI ──▶ AStar + Trajectory                  │
│       └── KeySimulator ──▶ pydirectinput                        │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                         算法层                                  │
│   astar_pathfinder/                                            │
│       ├── AStarPathfinder ──▶ heapq, numpy                     │
│       ├── GridMap ──▶ numpy                                    │
│       └── PathSmoother ──▶ numpy                              │
│                                                                │
│   core/                                                         │
│       ├── TrajectoryManager ──▶ numpy                          │
│       ├── MapDataSaver ──▶ json, cv2                          │
│       └── TrajectoryAnalyzer                                   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                        系统接口层                               │
│   utils/                                                        │
│       ├── window_manager.py ──▶ win32gui, win32con             │
│       ├── win32_utils.py ──▶ win32gui, ctypes                 │
│       └── dxgi_capture.py ──▶ dxcam, cv2                       │
└────────────────────────────────────────────────────────────────┘
```

### 依赖包说明

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| `PySide6` | 最新版 | GUI 框架 |
| `opencv-contrib-python` | 最新版 | SIFT/FLANN/HSV 图像处理 |
| `numpy` | 最新版 | 数值计算 |
| `ultralytics` | 最新版 | YOLO 模型推理 |
| `dxcam` | 最新版 | DXGI 屏幕捕获 |
| `pydirectinput` | 最新版 | DirectX 输入模拟 |
| `Pillow` | 最新版 | 图像处理辅助 |

---

## 6. 关键数据类型

### 6.1 检测结果格式

```python
# StitchWorker 发送的检测 tuple (11 元素)
detection = (
    x1, y1, x2, y2,     # bounding box 坐标（屏幕像素）
    conf,                # 置信度
    cls_id,              # 类别ID
    cx, cy,              # 中心点（屏幕像素）
    cls_name,            # 类别名称
    world_x,             # 世界坐标X（画布）
    world_y              # 世界坐标Y（画布）
)
```

### 6.2 轨迹数据格式

```python
# JSON 保存格式
{
    "version": "1.0",
    "timestamp": "20240101_120000",
    "canvas": {
        "width": 2000,
        "height": 1500
    },
    "trajectory": {
        "points": [[x1, y1], [x2, y2], ...],
        "point_count": 500,
        "start_point": [x1, y1],
        "end_point": [xn, yn]
    },
    "metadata": {
        "window_title": "The Lord of the Rings Online™",
        "map_region": [x, y, w, h],
        "canvas_offset": [c0, r0]
    }
}
```

### 6.3 战斗指令

```python
# CombatCore.update() 返回值
result = "W" / "A" / "S" / "D"  # 移动指令
result = "1" / "2" / ...         # 技能按键
result = "CLICK"                 # 点击目标
result = "SCROLL"                # 滚轮滚动
result = "RESET_VISION"          # 重置视角
result = None                    # 无指令
```

---

## 7. 配置说明

### 7.1 YOLO 模型配置

**默认路径**：`G:\Auto_Lotro\best.onnx`

**配置位置**：[ui/main_window.py#L475](file:///workspace/Auto_Lotro3/ui/main_window.py#L475)

```python
def _load_yolo_model(self):
    model_path = r"G:\Auto_Lotro\best.onnx"  # 可修改
    self._yolo_model = YOLO(model_path)
```

### 7.2 地图保存目录

**默认路径**：`G:\map`

**配置位置**：[ui/main_window.py#L561](file:///workspace/Auto_Lotro3/ui/main_window.py#L561)

```python
save_dir = r"G:\map"
```

### 7.3 战斗配置

通过 UI 界面调整：
- **技能键位数量**：1-9 个（默认 5 个）
- **攻击范围**：10-200 像素（默认 50）
- **置信度阈值**：0.0-1.0（默认 0.5）

### 7.4 小地图参数

**文件**：[lotro_arrow_v5.py](file:///workspace/Auto_Lotro3/lotro_arrow_v5.py#L29)

```python
# 小地图区域（像素坐标，需要根据实际调整）
MAP_LEFT   = 979
MAP_TOP    = 82
MAP_RIGHT  = 1075
MAP_BOTTOM = 159
MAP_W = 96   # 宽度
MAP_H = 77   # 高度

# HSV 颜色阈值
HSV_LOWER1 = np.array([  0, 120, 120])  # 红色低范围
HSV_UPPER1 = np.array([ 15, 255, 255])  # 红色高范围
HSV_LOWER2 = np.array([165, 120, 120])  # 红色（另一端）
HSV_UPPER2 = np.array([180, 255, 255])  # 红色（另一端）

# 平滑参数
SMOOTH_K = 0.25  # 低通滤波系数
```

---

## 8. 运行方式

### 8.1 环境准备

```bash
# 1. 安装 Python 3.9+
python --version

# 2. 安装依赖
pip install PySide6 opencv-contrib-python numpy ultralytics dxcam pydirectinput Pillow

# 3. 准备 YOLO 模型
#    将模型文件放置在 G:\Auto_Lotro\best.onnx
```

### 8.2 启动应用

```bash
cd /path/to/Auto_Lotro3
python main.py
```

### 8.3 操作流程

#### 地图拼接流程

```
1. 点击「捕获窗口」按钮
2. 将鼠标移到游戏窗口，按 F1
3. 点击「绑定窗口」按钮
4. 点击「截取初始地图」按钮
5. 框选小地图区域，Enter 确认
6. （可选）调整偏移修正参数
7. 点击「开始拼图」按钮
8. 移动角色进行探索
9. 点击「结束拼图」按钮保存
```

#### 自动战斗流程

```
1. 加载 YOLO 模型
2. 绑定游戏窗口
3. 点击「📂 加载地图」加载寻路地图（可选）
4. 配置技能键位数量和攻击范围
5. 点击「⚔️ 开始打怪」按钮
```

#### 轨迹寻路流程

```
1. 绑定游戏窗口
2. 在「轨迹名称」输入名称
3. 点击「🔴 开始记录」按钮
4. 在游戏中移动完成路径
5. 点击「⏹️ 停止记录」按钮
6. 点击「💾 保存轨迹」按钮
7. 后续可点击「▶️ 开始跟随」自动移动
```

---

## 9. 信号槽通信

### 9.1 Worker → UI

```python
# StitchWorker 信号
worker.log_signal(str)              # 日志消息
worker.detections_signal(list)      # 检测结果
worker.frame_signal(frame, list, float)  # 帧+FPS
worker.finished()                   # 完成

# CombatWorker 信号
worker.skill_signal(str)           # 技能按键
worker.click_signal(float, float)  # 点击坐标
worker.status_signal(str)          # 状态更新
worker.target_signal(str)          # 目标更新
worker.log_signal(str)             # 日志
```

### 9.2 UI → Worker

```python
# 通过方法调用
worker.set_yolo_model(model)
worker.set_yolo_detecting(True/False)
worker.set_yolo_conf(0.5)
worker.set_match_only(True/False)
worker.start()
worker.stop()
```

---

## 10. 常见问题与排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 窗口捕获失败 | 游戏窗口未激活 | 确保游戏窗口在前台，按 F1 时鼠标在游戏窗口上 |
| 拼图失败 | 小地图区域不正确 | 重新框选小地图区域 |
| 箭头检测失败 | HSV 阈值不匹配 | 调整 `HSV_LOWER1/UPPER1` 参数 |
| 检测不到怪物 | YOLO 模型未加载 | 检查模型路径，调整置信度阈值 |
| 战斗不释放技能 | 技能键位配置错误 | 调整技能键位数量 |
| 轨迹跟随失败 | 轨迹点不足 | 记录更多轨迹点 |
| 寻路失败 | 目标点在障碍物内 | 调整目标点位置 |

---

## 11. 性能优化建议

### 11.1 截图优化

- 默认捕获帧率：4 FPS
- 窗口最小化时自动跳过
- 连续失败 3 次后尝试还原窗口

### 11.2 YOLO 推理优化

- 输入尺寸：`imgsz=1280`
- 置信度阈值：适当提高减少误检
- 必要时启用隔帧检测

### 11.3 内存管理

- 地图拼接会占用较多内存
- 大地图建议适当降低画布分辨率
- 不使用时及时停止拼图线程

---

## 12. 文件索引

| 文件路径 | 主要类/函数 | 功能描述 |
|----------|------------|----------|
| [main.py](file:///workspace/Auto_Lotro3/main.py) | `main()` | 应用入口 |
| [lotro_arrow_v5.py](file:///workspace/Auto_Lotro3/lotro_arrow_v5.py) | `detect()`, `smooth_filter()` | 箭头检测 |
| [core/stitcher.py](file:///workspace/Auto_Lotro3/core/stitcher.py) | `MiniMap`, `StitchWorker` | 地图拼接 |
| [core/combat.py](file:///workspace/Auto_Lotro3/core/combat.py) | `CombatCore`, `CombatState` | 战斗逻辑 |
| [core/combat_worker.py](file:///workspace/Auto_Lotro3/core/combat_worker.py) | `CombatWorker`, `KeySimulator` | 战斗工作线程 |
| [core/pathfinder.py](file:///workspace/Auto_Lotro3/core/pathfinder.py) | `PathFinderAPI`, `PathPlanner` | 寻路模块 |
| [core/trajectory_path.py](file:///workspace/Auto_Lotro3/core/trajectory_path.py) | `TrajectoryManager` | 轨迹管理 |
| [astar_pathfinder/astar.py](file:///workspace/Auto_Lotro3/astar_pathfinder/astar.py) | `AStarPathfinder` | A* 算法 |
| [astar_pathfinder/grid_map.py](file:///workspace/Auto_Lotro3/astar_pathfinder/grid_map.py) | `GridMap` | 网格地图 |
| [ui/main_window.py](file:///workspace/Auto_Lotro3/ui/main_window.py) | `MainWindow` | 主窗口 |
| [ui/widgets.py](file:///workspace/Auto_Lotro3/ui/widgets.py) | `CanvasWidget`, `RegionSelector` | UI 组件 |
| [utils/window_manager.py](file:///workspace/Auto_Lotro3/utils/window_manager.py) | `WindowManager` | 窗口管理 |
| [utils/dxgi_capture.py](file:///workspace/Auto_Lotro3/utils/dxgi_capture.py) | `DxgiWindowCapture` | DXGI 截图 |
| [core/map_data_saver.py](file:///workspace/Auto_Lotro3/core/map_data_saver.py) | `MapDataSaver` | 数据持久化 |

---

*文档生成时间：2026-05-17*
