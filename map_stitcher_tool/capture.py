"""
屏幕捕获模块 - 支持 Win32 API 和 DXGI 两种方式捕获窗口
"""

import ctypes
import ctypes.wintypes
import numpy as np
import cv2
import win32gui
import win32ui
import win32con
import win32api

# 尝试导入 DXGI 捕获
try:
    from utils.dxgi_capture import DxgiWindowCapture, DXCAM_AVAILABLE as DXGI_AVAILABLE
except ImportError:
    DXGI_AVAILABLE = False
    DxgiWindowCapture = None


class ScreenCapture:
    """统一的屏幕捕获接口"""

    @staticmethod
    def capture_window(hwnd: int, x: int = 0, y: int = 0, w: int = None, h: int = None) -> np.ndarray:
        """
        捕获窗口指定区域

        Args:
            hwnd: 窗口句柄
            x, y: 客户区坐标（相对坐标）
            w, h: 截取区域大小，如果为 None 则截取整个客户区

        Returns:
            numpy.ndarray: BGR 格式的图像，如果失败返回 None
        """
        if DXGI_AVAILABLE:
            try:
                capture = DxgiWindowCapture(hwnd)
                img = capture.capture(x, y, w, h)
                return img
            except Exception as e:
                print(f"[ScreenCapture] DXGI 捕获失败，回退到 Win32: {e}")

        return ScreenCapture._win32_capture(hwnd, x, y, w, h)

    @staticmethod
    def _win32_capture(hwnd: int, x: int = 0, y: int = 0, w: int = None, h: int = None) -> np.ndarray:
        """使用 Win32 API 捕获窗口（备选方案）"""
        try:
            # 获取窗口客户区大小
            client_rect = win32gui.GetClientRect(hwnd)
            client_w = client_rect[2] - client_rect[0]
            client_h = client_rect[3] - client_rect[1]

            if w is None:
                w = client_w
            if h is None:
                h = client_h

            # 获取窗口设备上下文
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            # 创建位图
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
            saveDC.SelectObject(saveBitMap)

            # 捕获窗口内容到内存 DC
            result = saveDC.BitBlt(
                (x, y), (w, h),
                mfcDC,
                (x, y),
                win32con.SRCCOPY
            )

            # 转换为 numpy 数组
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img = img.reshape((h, w, 4))

            # 释放资源
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

            # 转换 BGRA 到 BGR
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return img

        except Exception as e:
            print(f"[ScreenCapture] Win32 捕获失败: {e}")
            return None
