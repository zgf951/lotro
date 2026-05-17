"""
截图测试工具
用于验证和调试截图坐标问题

使用方法：
    # 从项目根目录运行：
    python -m utils.capture_test [--hwnd HWND]

或：
    from utils.capture_test import test_capture
    test_capture(hwnd, x, y, w, h)
"""

import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dxgi_capture import DxgiWindowCapture, get_dpi_scale, adjust_for_dpi


def test_capture(hwnd, x=0, y=0, w=None, h=None, output_file="test_capture.png"):
    """
    测试截图功能
    
    Args:
        hwnd: 窗口句柄
        x, y, w, h: 要截取的区域（客户区坐标）
        output_file: 保存测试截图的文件名
    """
    print(f"{'='*60}")
    print(f"截图测试")
    print(f"{'='*60}")
    
    # 显示 DPI 信息
    scale_x, scale_y = get_dpi_scale()
    print(f"DPI 缩放：x{scale_x:.2f}, y{scale_y:.2f}")
    print()
    
    # 创建捕获器
    capture = DxgiWindowCapture(hwnd)
    capture._debug_mode = True
    
    print("测试 1: dxcam 方式...")
    img1 = capture._capture_dxcam(x, y, w, h)
    if img1 is not None:
        cv2.imwrite("test_dxcam.png", img1)
        print(f"  ✓ 成功！保存为 test_dxcam.png")
        print(f"  大小：{img1.shape}")
    else:
        print("  ✗ 失败")
    
    print()
    
    print("测试 2: PrintWindow 方式...")
    img2 = capture._capture_printwindow(x, y, w, h)
    if img2 is not None:
        cv2.imwrite("test_printwindow.png", img2)
        print(f"  ✓ 成功！保存为 test_printwindow.png")
        print(f"  大小：{img2.shape}")
    else:
        print("  ✗ 失败")
    
    print()
    
    print("测试 3: 通用方式（自动选择）...")
    img3 = capture.capture(x, y, w, h)
    if img3 is not None:
        cv2.imwrite(output_file, img3)
        print(f"  ✓ 成功！保存为 {output_file}")
        print(f"  大小：{img3.shape}")
        
        # 显示图像
        print("\n按任意键关闭预览...")
        cv2.imshow("Capture Test", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("  ✗ 失败")
    
    print()
    print("测试完成！")
    return img3


if __name__ == "__main__":
    import argparse
    import win32gui
    
    parser = argparse.ArgumentParser(description="截图测试工具")
    parser.add_argument("--hwnd", type=int, help="窗口句柄")
    parser.add_argument("--title", type=str, help="窗口标题（模糊匹配）")
    parser.add_argument("--x", type=int, default=0, help="x坐标")
    parser.add_argument("--y", type=int, default=0, help="y坐标")
    parser.add_argument("--w", type=int, default=None, help="宽度")
    parser.add_argument("--h", type=int, default=None, help="高度")
    args = parser.parse_args()
    
    hwnd = None
    
    if args.hwnd:
        hwnd = args.hwnd
    elif args.title:
        def callback(hwnd, windows):
            title = win32gui.GetWindowText(hwnd)
            if args.title.lower() in title.lower():
                windows.append((hwnd, title))
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if windows:
            print("找到匹配的窗口：")
            for h, t in windows[:10]:
                print(f"  HWND {h:08X}: {t}")
            hwnd = windows[0][0]
            print(f"\n选择了第一个窗口：HWND {hwnd:08X}")
    else:
        # 列出前 10 个窗口供选择
        print("没有指定窗口，列出窗口列表：")
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                windows.append((hwnd, win32gui.GetWindowText(hwnd)))
        windows = []
        win32gui.EnumWindows(callback, windows)
        for i, (h, t) in enumerate(windows[:10]):
            print(f"  [{i}] HWND {h:08X}: {t}")
        print("\n使用 --hwnd 或 --title 选择窗口")
        sys.exit(1)
    
    test_capture(hwnd, args.x, args.y, args.w, args.h)
