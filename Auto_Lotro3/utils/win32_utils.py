import ctypes
import ctypes.wintypes
import numpy as np
import cv2
import struct
import win32gui
import win32process

# ── Windows API handles ────────────────────────────────────────────────────────
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# ── DXGI Capture (for DirectX windows) ────────────────────────────────────────

try:
    from utils.dxgi_capture import dxgi_capture, cleanup_dxgi_cache, DxgiWindowCapture

    DXGI_AVAILABLE = True
except ImportError as e:
    DXGI_AVAILABLE = False
    print(f"警告：DXGI 捕获不可用：{e}")


    def dxgi_capture(hwnd, x, y, w, h):
        raise ImportError("DXGI 捕获不可用")


    def cleanup_dxgi_cache():
        pass


    class DxgiWindowCapture:
        def __init__(self, hwnd=None):
            raise ImportError("DXGI 捕获不可用")


# ── Windows helpers ────────────────────────────────────────────────────────────

def _cursor_pos():
    pt = ctypes.wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def get_window_at_cursor():
    px, py = _cursor_pos()
    hwnd = user32.WindowFromPoint(ctypes.wintypes.POINT(px, py))
    return hwnd, px, py


def get_window_info(hwnd):
    title_buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, title_buf, 256)
    cls_buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, cls_buf, 256)
    return hwnd, cls_buf.value, title_buf.value


def get_window_rect(hwnd):
    r = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def get_client_rect(hwnd):
    """获取窗口客户区矩形（不含标题栏和边框）"""
    r = ctypes.wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def client_to_screen(hwnd, x, y):
    """将客户区坐标转换为屏幕坐标"""
    pt = ctypes.wintypes.POINT(x, y)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y


def screen_to_client(hwnd, x, y):
    """将屏幕坐标转换为客户区坐标"""
    pt = ctypes.wintypes.POINT(x, y)
    user32.ScreenToClient(hwnd, ctypes.byref(pt))
    return pt.x, pt.y


def find_window_by_pid(pid):
    """根据进程 PID 查找窗口句柄"""
    hwnd_list = []

    def callback(hwnd, extra):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
        if found_pid == pid and win32gui.IsWindowVisible(hwnd):
            hwnd_list.append(hwnd)

    win32gui.EnumWindows(callback, None)
    return hwnd_list[0] if hwnd_list else None


def get_window_client_rect(hwnd):
    """获取窗口客户区的屏幕坐标"""
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return (left, top, right, bottom)


def mss_capture(hwnd, x: int, y: int, w: int, h: int) -> np.ndarray:
    """使用 DXGI 捕获窗口区域（保留函数名以维持向后兼容）"""
    return dxgi_capture(hwnd, x, y, w, h)


def bitblt_capture(hwnd, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    捕获窗口指定区域（仅使用 DXGI，适用于 DirectX 窗口）

    Args:
        hwnd: 窗口句柄
        x, y: 客户区坐标
        w, h: 截取区域大小

    Returns:
        numpy.ndarray: BGR 格式图像
    """
    # 使用 DXGI 捕获（支持 DirectX 窗口，支持后台）
    if DXGI_AVAILABLE:
        return dxgi_capture(hwnd, x, y, w, h)
    else:
        raise RuntimeError("DXGI 不可用，请安装 dxcam: pip install dxcam")