"""
自动战斗工作线程
在后台运行战斗逻辑，避免阻塞 UI
"""

from PySide6.QtCore import QObject, QThread, Signal, QTimer
from PySide6.QtGui import QKeySequence, QKeyEvent
from PySide6.QtWidgets import QApplication
import time
import pydirectinput
from typing import Optional, Tuple, List

from core.combat import CombatCore, CombatConfig, CombatState


class KeySimulator:
    """键盘模拟器 - 使用 pydirectinput 支持 DirectX 输入"""
    
    @staticmethod
    def press_key(key: str):
        """
        模拟按键按下
        :param key: 按键字符，如 '1', '2', 'a', 's', 'd', 'w', 'numpad0'
        """
        try:
            # 使用 pydirectinput 模拟按键（支持 DirectX）
            key = key.strip().lower()
            
            # 数字键
            if key.isdigit() and 1 <= int(key) <= 9:
                pydirectinput.press(key)
            # 字母键
            elif len(key) == 1 and key.isalpha():
                pydirectinput.press(key)
            # 特殊键
            elif key in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']:
                pydirectinput.press(key)
            elif key == 'space':
                pydirectinput.press(' ')
            elif key == 'shift':
                pydirectinput.press('shift')
            elif key == 'ctrl':
                pydirectinput.press('ctrl')
            elif key == 'alt':
                pydirectinput.press('alt')
            elif key == 'numpad0' or key == 'nub0' or key == 'num0':
                pydirectinput.press('numpad0')
            else:
                # 尝试直接按下
                pydirectinput.press(key)
                
        except Exception as e:
            print(f"[按键模拟错误] {e}")
    
    @staticmethod
    def scroll_down(clicks: int = 1):
        """
        向下滚动滚轮（拉远视角）
        :param clicks: 滚动次数
        """
        try:
            # pydirectinput.scroll 负值向下滚动
            # 每次滚动 -120 是一个标准步长
            for _ in range(clicks):
                pydirectinput.scroll(-120)
        except Exception as e:
            print(f"[滚轮滚动错误] {e}")

    @staticmethod
    def click_at(x: float, y: float):
        """
        在指定位置点击鼠标 - 使用 pydirectinput
        :param x: 屏幕 X 坐标
        :param y: 屏幕 Y 坐标
        """
        try:
            # 使用 pydirectinput 移动鼠标并点击
            pydirectinput.moveTo(int(x), int(y))
            pydirectinput.click()
        except Exception as e:
            print(f"[鼠标点击错误] {e}")


class CombatWorker(QObject):
    """
    战斗工作线程
    运行战斗核心逻辑，发送技能释放信号
    """
    
    # 信号定义
    skill_signal = Signal(str)              # 技能释放信号 (按键)
    click_signal = Signal(float, float)     # 点击信号 (目标 x, y)
    status_signal = Signal(str)             # 状态更新信号
    target_signal = Signal(str)             # 目标更新信号
    log_signal = Signal(str)                # 日志信号
    move_signal = Signal(float, float)      # 移动信号 (目标 x, y)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._combat_core = CombatCore()
        self._running = False
        self._thread: Optional[QThread] = None
        self._update_timer: Optional[QTimer] = None
        self._update_interval = 100  # 100ms 更新一次
        self._current_move_key: Optional[str] = None  # 当前按下的移动键
        
    def set_config(self, config: CombatConfig):
        """设置战斗配置"""
        self._combat_core.set_config(config)
    
    def set_player_position(self, x: float, y: float):
        """设置玩家位置"""
        self._combat_core.set_player_position(x, y)
    
    def set_player_angle(self, angle: float):
        """设置玩家朝向"""
        self._combat_core.set_player_angle(angle)
    
    def start(self):
        """启动战斗"""
        if self._running:
            return
        
        self._running = True
        self._combat_core.set_enabled(True)
        self._combat_core.start()
        
        # 创建定时器定期更新
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update)
        self._update_timer.start(self._update_interval)
        
        self.log_signal.emit("[战斗] 开始自动战斗")
    
    def stop(self):
        """停止战斗"""
        if not self._running:
            return
        
        self._running = False
        self._combat_core.set_enabled(False)
        self._combat_core.stop()
        
        # 停止所有移动按键
        if self._current_move_key:
            try:
                pydirectinput.keyUp(self._current_move_key.lower())
            except:
                pass
            self._current_move_key = None
        
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None
        
        self.log_signal.emit("[战斗] 停止自动战斗")
    
    def update_detections(self, detections: List[Tuple]):
        """更新检测结果"""
        if self._running and self._combat_core:
            # 在主线程中调用更新逻辑
            result = self._combat_core.update(detections)
            
            # 处理返回值
            if result:
                if result == "CLICK":
                    # 需要点击怪物
                    if self._combat_core.current_target:
                        target = self._combat_core.current_target
                        self.click_signal.emit(target.cx, target.cy)
                elif result == "SCROLL":
                    # 向下滚动滚轮
                    KeySimulator.scroll_down(3)
                    self.log_signal.emit("[战斗] 向下滚动滚轮 (拉远视角)")
                elif result == "RESET_VISION":
                    # 重置视觉朝向
                    KeySimulator.press_key("numpad0")
                    self.log_signal.emit("[战斗] 按下 Numpad 0 (重置视觉朝向)")
                elif result in ['W', 'A', 'S', 'D']:
                    # 移动指令：使用长按模式
                    move_key = result.lower()
                    if self._current_move_key != move_key:
                        # 如果换了按键，先松开之前的
                        if self._current_move_key:
                            pydirectinput.keyUp(self._current_move_key)
                        # 按下新的
                        pydirectinput.keyDown(move_key)
                        self._current_move_key = move_key
                else:
                    # 如果返回的是技能键，说明不需要移动了（或者移动由核心控制）
                    # 释放当前的移动键
                    if self._current_move_key:
                        pydirectinput.keyUp(self._current_move_key)
                        self._current_move_key = None
                    
                    # 释放技能
                    KeySimulator.press_key(result)
                    self.log_signal.emit(f"[战斗] 释放技能: {result}")
            else:
                # 如果没有返回指令，释放当前的移动键
                if self._current_move_key:
                    pydirectinput.keyUp(self._current_move_key)
                    self._current_move_key = None
            
            # 更新状态和目标显示
            self.status_signal.emit(self._combat_core.get_status_text())
            self.target_signal.emit(self._combat_core.get_target_text())
    
    def _update(self):
        """定期更新"""
        if not self._running:
            return
        
        # 定期更新状态和目标显示（即使没有新的检测结果）
        self.status_signal.emit(self._combat_core.get_status_text())
        self.target_signal.emit(self._combat_core.get_target_text())
    
    def get_current_state(self) -> CombatState:
        """获取当前战斗状态"""
        return self._combat_core.state
    
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running
