"""
窗口管理器 - 提供可靠的窗口绑定和管理功能
"""

import win32gui
import win32process
import win32con
import ctypes
from ctypes import wintypes
from typing import Optional, Tuple, Dict
import time


class WindowManager:
    """可靠的窗口管理器"""
    
    def __init__(self):
        self.hwnd = None
        self.pid = None
        self.class_name = None
        self.title = None
        self._last_check_time = 0
        self._valid_cache = False
        
    def bind_by_hwnd(self, hwnd: int) -> bool:
        """直接通过句柄绑定窗口"""
        if not hwnd or not win32gui.IsWindow(hwnd):
            return False
            
        # 验证窗口是否可见且可用
        if not win32gui.IsWindowVisible(hwnd):
            return False
            
        self.hwnd = hwnd
        _, self.pid = win32process.GetWindowThreadProcessId(hwnd)
        self.class_name = win32gui.GetClassName(hwnd)
        self.title = win32gui.GetWindowText(hwnd)
        self._valid_cache = True
        self._last_check_time = time.time()
        return True
    
    def bind_by_pid(self, pid: int, class_name: str = None, title_pattern: str = None) -> bool:
        """通过 PID 和可选条件绑定窗口"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                if window_pid == pid:
                    # 如果指定了类名，检查匹配
                    if class_name and win32gui.GetClassName(hwnd) != class_name:
                        return True
                    # 如果指定了标题模式，检查匹配
                    if title_pattern and title_pattern not in win32gui.GetWindowText(hwnd):
                        return True
                    windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            # 优先选择最前面的窗口
            for hwnd in windows:
                if win32gui.GetWindowText(hwnd):  # 优先选择有标题的
                    return self.bind_by_hwnd(hwnd)
            return self.bind_by_hwnd(windows[0])
        return False
    
    def bind_by_cursor(self) -> bool:
        """通过鼠标位置绑定窗口"""
        point = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        hwnd = ctypes.windll.user32.WindowFromPoint(point)
        
        if hwnd:
            # 跳过自己的窗口（需要外部设置）
            main_hwnd = self.get_main_window_handle()
            if main_hwnd and hwnd == int(main_hwnd):
                return False
            return self.bind_by_hwnd(hwnd)
        return False
    
    def is_valid(self) -> bool:
        """验证窗口是否仍然有效（包括最小化的窗口）"""
        if not self.hwnd:
            return False
        
        # 缓存验证结果 1 秒
        if time.time() - self._last_check_time < 1.0 and self._valid_cache:
            return True
            
        self._valid_cache = (
            win32gui.IsWindow(self.hwnd) and
            win32gui.IsWindowVisible(self.hwnd)
            # 注意：不检查 IsIconic，因为最小化的窗口仍然有效，可以还原
        )
        self._last_check_time = time.time()
        return self._valid_cache
    
    def is_minimized(self) -> bool:
        """检查窗口是否最小化"""
        if not self.hwnd:
            return True
        return win32gui.IsIconic(self.hwnd)
    
    def get_client_rect(self) -> Tuple[int, int, int, int]:
        """获取客户区矩形"""
        if not self.is_valid():
            return (0, 0, 0, 0)
        return win32gui.GetClientRect(self.hwnd)
    
    def get_window_rect(self) -> Tuple[int, int, int, int]:
        """获取窗口矩形"""
        if not self.is_valid():
            return (0, 0, 0, 0)
        return win32gui.GetWindowRect(self.hwnd)
    
    def client_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """客户区坐标转屏幕坐标"""
        if not self.is_valid():
            return (x, y)
        return win32gui.ClientToScreen(self.hwnd, (x, y))
    
    def screen_to_client(self, x: int, y: int) -> Tuple[int, int]:
        """屏幕坐标转客户区坐标"""
        if not self.is_valid():
            return (x, y)
        
        # 获取客户区在屏幕上的位置
        client_left, client_top = self.client_to_screen(0, 0)
        return (x - client_left, y - client_top)
    
    def bring_to_front(self) -> bool:
        """将窗口带到前台"""
        if not self.is_valid():
            return False
        
        try:
            # 如果窗口最小化，先还原
            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            
            # 尝试多种方式激活窗口
            win32gui.SetForegroundWindow(self.hwnd)
            win32gui.BringWindowToTop(self.hwnd)
            
            # 等待窗口激活
            time.sleep(0.1)
            return True
        except:
            return False
    
    def get_info(self) -> Dict:
        """获取窗口信息字典"""
        if not self.is_valid():
            return {}
        
        window_rect = self.get_window_rect()
        client_rect = self.get_client_rect()
        
        return {
            'hwnd': self.hwnd,
            'pid': self.pid,
            'class_name': self.class_name,
            'title': self.title,
            'window_rect': window_rect,
            'client_rect': client_rect,
            'is_valid': True,
            'is_visible': win32gui.IsWindowVisible(self.hwnd),
            'is_minimized': win32gui.IsIconic(self.hwnd)
        }
    
    @staticmethod
    def set_main_window_handle(hwnd):
        """设置主窗口句柄（用于跳过自己的窗口）"""
        WindowManager._main_window_hwnd = hwnd
    
    @staticmethod
    def get_main_window_handle():
        """获取主窗口句柄"""
        return getattr(WindowManager, '_main_window_hwnd', None)
    
    def clear(self):
        """清除绑定"""
        self.hwnd = None
        self.pid = None
        self.class_name = None
        self.title = None
        self._valid_cache = False
        self._last_check_time = 0
