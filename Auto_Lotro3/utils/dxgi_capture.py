"""
DXGI 窗口截图模块 - 使用 dxcam 库捕获 DirectX 渲染的窗口
适用于 LOTRO 等使用 DirectX 的游戏窗口

安装依赖:
    pip install dxcam opencv-contrib-python numpy

使用示例:
    from utils.dxgi_capture import DxgiWindowCapture
    
    capture = DxgiWindowCapture(hwnd)
    img = capture.capture(x, y, w, h)  # 截取客户区指定区域
"""

import numpy as np
import cv2
import ctypes
import ctypes.wintypes
import win32gui
import win32con
import win32ui
from pathlib import Path

# 尝试导入 dxcam
DXCAM_AVAILABLE = False
try:
    import dxcam
    DXCAM_AVAILABLE = True
    print(f"[DXGI] dxcam 加载成功，版本：{getattr(dxcam, '__version__', 'unknown')}")
except Exception as e:
    DXCAM_AVAILABLE = False
    print(f"[DXGI] dxcam 加载失败：{e}")
    print(f"[DXGI] 请确保已安装：pip install dxcam")

# Windows API
user32 = ctypes.windll.user32

# 启用 DPI 感知，解决 DPI 缩放问题
try:
    # 尝试设置进程的 DPI 感知
    user32.SetProcessDPIAware()
except:
    pass

# DPI 缩放辅助函数
def get_dpi_scale():
    """获取 DPI 缩放比例"""
    try:
        hdc = user32.GetDC(0)
        dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        dpi_y = ctypes.windll.gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
        user32.ReleaseDC(0, hdc)
        return dpi_x / 96.0, dpi_y / 96.0
    except:
        return 1.0, 1.0

def adjust_for_dpi(x, y, w, h):
    """根据 DPI 缩放调整坐标"""
    scale_x, scale_y = get_dpi_scale()
    return (int(x * scale_x), int(y * scale_y), 
            int(w * scale_x), int(h * scale_y))


class DxgiWindowCapture:
    """使用 DXGI 捕获窗口内容的类"""
    
    def __init__(self, hwnd=None, use_printwindow_fallback=True):
        """
        初始化 DXGI 捕获器
        
        Args:
            hwnd: 窗口句柄，如果为 None 则需要在 capture 前设置
            use_printwindow_fallback: 是否使用 PrintWindow 作为备用方案
        """
        if not DXCAM_AVAILABLE:
            raise ImportError("dxcam 库未安装，请运行：pip install dxcam")
        
        self._hwnd = hwnd
        self._dxcam = None
        self._last_width = 0
        self._last_height = 0
        self._monitor_rect = None  # 显示器坐标 (left, top, right, bottom)
        self._use_printwindow_fallback = use_printwindow_fallback
        self._debug_mode = False  # 调试模式，会保存调试图像
        
        if hwnd:
            self._init_dxcam(hwnd)
    
    def _init_dxcam(self, hwnd):
        """为特定窗口初始化 dxcam"""
        self._hwnd = hwnd
        
        # 获取窗口的显示器信息
        # 使用 GetWindowRect 获取窗口在屏幕上的位置
        window_rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(window_rect))
        self._window_screen_rect = (window_rect.left, window_rect.top, 
                                     window_rect.right, window_rect.bottom)
        
        # 获取客户区在屏幕上的坐标
        client_screen_rect = self._get_client_screen_rect()
        if client_screen_rect:
            self._client_screen_rect = client_screen_rect
            self._last_width = client_screen_rect[2] - client_screen_rect[0]
            self._last_height = client_screen_rect[3] - client_screen_rect[1]
        else:
            # 回退到旧方法
            client_rect = win32gui.GetClientRect(hwnd)
            self._last_width = client_rect[2] - client_rect[0]
            self._last_height = client_rect[3] - client_rect[1]
            self._client_screen_rect = None
        
        # 初始化 dxcam，指定显示器区域
        # dxcam 会捕获整个显示器，我们需要从中裁剪出窗口区域
        try:
            self._dxcam = dxcam.create(device_idx=0, max_buffer_len=1)
        except Exception as e:
            print(f"[DXGI] dxcam.create 失败：{e}，将使用备用方案")
            self._dxcam = None
        
        print(f"[DXGI] 初始化成功 - 窗口句柄：{hwnd}")
        print(f"[DXGI] 窗口屏幕位置：{self._window_screen_rect}")
        if self._client_screen_rect:
            print(f"[DXGI] 客户区屏幕位置：{self._client_screen_rect}")
        print(f"[DXGI] 客户区大小：{self._last_width}x{self._last_height}")
        
        # 获取 DPI 信息
        scale_x, scale_y = get_dpi_scale()
        print(f"[DXGI] DPI 缩放：x{scale_x:.2f}, y{scale_y:.2f}")
    
    @property
    def hwnd(self):
        return self._hwnd
    
    @hwnd.setter
    def hwnd(self, hwnd):
        if not hwnd or self._hwnd == hwnd:
            return
        if self._dxcam:
            # dxcam 新版本没有 destroy 方法，直接设为 None 即可
            self._dxcam = None
        self._init_dxcam(hwnd)
    
    def __del__(self):
        if self._dxcam:
            # dxcam 新版本没有 destroy 方法
            self._dxcam = None
    
    def _get_client_screen_rect(self):
        """获取客户区在屏幕上的坐标"""
        if not self._hwnd:
            return None
        
        # 获取客户区大小
        client_rect = win32gui.GetClientRect(self._hwnd)
        client_left, client_top = client_rect[0], client_rect[1]
        client_right, client_bottom = client_rect[2], client_rect[3]
        
        # 转换到屏幕坐标
        pt_left_top = ctypes.wintypes.POINT(client_left, client_top)
        pt_right_bottom = ctypes.wintypes.POINT(client_right, client_bottom)
        
        user32.ClientToScreen(self._hwnd, ctypes.byref(pt_left_top))
        user32.ClientToScreen(self._hwnd, ctypes.byref(pt_right_bottom))
        
        return (pt_left_top.x, pt_left_top.y, 
                pt_right_bottom.x, pt_right_bottom.y)
    
    def capture(self, x=0, y=0, w=None, h=None):
        """
        捕获窗口客户区的指定区域

        Args:
            x, y: 客户区坐标（相对坐标）
            w, h: 截取区域的宽度和高度，如果为 None 则截取整个客户区

        Returns:
            numpy.ndarray: BGR 格式的图像，如果失败返回 None
        """
        if not self._hwnd:
            print("[DXGI 错误] 未设置窗口句柄")
            return None

        # 尝试使用 dxcam
        if self._dxcam and DXCAM_AVAILABLE:
            try:
                result = self._capture_dxcam(x, y, w, h)
                if result is not None:
                    return result
                print("[DXGI] dxcam 捕获失败，尝试备用方案")
            except Exception as e:
                print(f"[DXGI] dxcam 捕获异常：{e}")
        
        # 使用备用方案：PrintWindow
        if self._use_printwindow_fallback:
            return self._capture_printwindow(x, y, w, h)
        
        return None
    
    def _capture_dxcam(self, x=0, y=0, w=None, h=None):
        """使用 dxcam 方式捕获"""
        # 每次调用都重新获取客户区屏幕坐标
        # 这样窗口移动后截图区域会自动跟随，不再捕捉到错误位置
        client_screen_rect = self._get_client_screen_rect()
        if not client_screen_rect:
            print("[DXGI 错误] 无法获取客户区屏幕坐标")
            return None
        self._client_screen_rect = client_screen_rect   # 同步缓存（供外部读取）

        screen_left, screen_top, screen_right, screen_bottom = client_screen_rect

        # 如果没有指定宽高，则截取整个客户区
        if w is None:
            w = screen_right - screen_left
        if h is None:
            h = screen_bottom - screen_top

        # 计算要截取的区域在屏幕上的坐标
        capture_left = screen_left + x
        capture_top = screen_top + y
        capture_width = w
        capture_height = h

        # 使用 dxcam 捕获整个显示器
        frame = self._dxcam.grab()

        if frame is None:
            print("[DXGI 错误] 捕获失败，返回空帧")
            return None

        # 从整个显示器图像中裁剪出窗口客户区
        # frame 是 RGB 格式
        frame_height, frame_width = frame.shape[:2]

        # 计算相对于显示器左上角的坐标
        # dxcam 捕获的是主显示器，所以直接使用屏幕坐标
        rel_left = capture_left
        rel_top = capture_top

        # 检查坐标是否在有效范围内
        if rel_left < 0 or rel_top < 0:
            print(f"[DXGI 错误] 坐标超出范围：left={rel_left}, top={rel_top}")
            return None

        if rel_left + capture_width > frame_width or rel_top + capture_height > frame_height:
            print(f"[DXGI 错误] 区域超出显示器范围：{rel_left}+{capture_width} > {frame_width} 或 {rel_top}+{capture_height} > {frame_height}")
            # 尝试裁剪到有效范围
            capture_width = min(capture_width, max(0, frame_width - rel_left))
            capture_height = min(capture_height, max(0, frame_height - rel_top))

            if capture_width <= 0 or capture_height <= 0:
                print(f"[DXGI 错误] 裁剪后区域无效：w={capture_width}, h={capture_height}")
                return None

        # 裁剪出目标区域
        try:
            crop = frame[rel_top:rel_top+capture_height, rel_left:rel_left+capture_width]

            if crop.size == 0:
                print(f"[DXGI 错误] 裁剪结果为空，shape={crop.shape}")
                return None

            # 转换为 BGR 格式（OpenCV 默认格式）
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            elif len(crop.shape) == 3 and crop.shape[2] == 4:
                crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)

            if self._debug_mode:
                cv2.imwrite("debug_dxcam_capture.png", crop)
                
            return crop

        except Exception as e:
            print(f"[DXGI 错误] 裁剪失败：{e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _capture_printwindow(self, x=0, y=0, w=None, h=None):
        """使用 PrintWindow API 捕获（备用方案）"""
        hwnd = self._hwnd
        
        try:
            # 获取客户区大小
            client_rect = win32gui.GetClientRect(hwnd)
            client_w = client_rect[2] - client_rect[0]
            client_h = client_rect[3] - client_rect[1]
            
            if w is None:
                w = client_w
            if h is None:
                h = client_h
            
            # 创建设备上下文
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # 创建位图
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, client_w, client_h)
            
            saveDC.SelectObject(saveBitMap)
            
            # 使用 PrintWindow 捕获
            result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
            
            # 转换为 numpy 数组
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
            
            # 裁剪到指定区域
            if x != 0 or y != 0 or w != client_w or h != client_h:
                img = img[y:y+h, x:x+w]
            
            # 转换格式：BGRA -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 释放资源
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            win32gui.DeleteObject(saveBitMap.GetHandle())
            
            if self._debug_mode:
                cv2.imwrite("debug_printwindow_capture.png", img)
            
            return img
            
        except Exception as e:
            print(f"[PrintWindow] 捕获失败：{e}")
            import traceback
            traceback.print_exc()
            return None

    def capture_full(self):
        """捕获整个窗口客户区"""
        return self.capture(0, 0)

    def destroy(self):
        """释放资源"""
        if self._dxcam:
            # dxcam 新版本没有 destroy 方法，直接设为 None
            self._dxcam = None


# 全局捕获器缓存
_capture_cache = {}


def dxgi_capture(hwnd, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    使用 DXGI 捕获窗口指定区域的便捷函数

    Args:
        hwnd: 窗口句柄
        x, y: 客户区坐标
        w, h: 截取区域大小

    Returns:
        numpy.ndarray: BGR 格式图像
    """
    global _capture_cache

    if hwnd not in _capture_cache:
        try:
            _capture_cache[hwnd] = DxgiWindowCapture(hwnd)
        except Exception as e:
            print(f"[DXGI] 创建捕获器失败：{e}")
            raise

    capture = _capture_cache[hwnd]

    # 如果窗口句柄变化了，重新初始化
    if capture.hwnd != hwnd:
        capture.hwnd = hwnd

    img = capture.capture(x, y, w, h)

    if img is None:
        raise RuntimeError(f"DXGI 捕获失败")

    return img


def cleanup_dxgi_cache():
    """清理 DXGI 捕获器缓存"""
    global _capture_cache
    for capture in _capture_cache.values():
        capture.destroy()
    _capture_cache.clear()