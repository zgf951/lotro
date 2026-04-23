"""
小地图坐标校准工具
==================
运行方法：
  1. 先打开游戏，进入游戏内场景（能看到小地图）
  2. 运行本脚本
  3. 在弹出的"全屏截图"窗口中，用鼠标框选小地图区域
  4. 松开鼠标后，脚本自动打印正确的 MAP_LEFT/TOP/RIGHT/BOTTOM 常量
  5. 把这四个值填回 lotro_arrow_fixed.py

操作：
  - 左键拖拽：框选区域
  - 按 R：重置选框
  - 按 Enter / 空格：确认并输出坐标
  - 按 Esc：退出
"""

import cv2
import numpy as np
import win32gui
import mss

WINDOW_NAME = "The Lord of the Rings Online™"

# ── 获取窗口客户区原点（屏幕绝对坐标）──────────────────────────────────────────
def get_client_origin():
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    if hwnd == 0:
        raise Exception(f"找不到游戏窗口：{WINDOW_NAME}")

    # GetWindowRect 含阴影/标题栏
    win_rect = win32gui.GetWindowRect(hwnd)

    # ClientToScreen 把客户区左上角转为屏幕坐标，可消除标题栏/边框偏移
    import ctypes
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    client_x, client_y = pt.x, pt.y

    print(f"[INFO] GetWindowRect  左上角: ({win_rect[0]}, {win_rect[1]})")
    print(f"[INFO] 客户区   左上角: ({client_x}, {client_y})")
    print(f"[INFO] 标题栏/边框偏移: dx={client_x - win_rect[0]}, dy={client_y - win_rect[1]}")
    return client_x, client_y


# ── 截取整个游戏客户区 ─────────────────────────────────────────────────────────
def capture_full_window(client_x, client_y):
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    # 获取客户区尺寸
    import ctypes
    class RECT(ctypes.Structure):
        _fields_ = [("left",ctypes.c_long),("top",ctypes.c_long),
                    ("right",ctypes.c_long),("bottom",ctypes.c_long)]
    r = RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top

    monitor = {"left": client_x, "top": client_y, "width": w, "height": h}
    with mss.mss() as sct:
        img = sct.grab(monitor)
    frame = np.array(img)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), w, h


# ── 交互式框选 ─────────────────────────────────────────────────────────────────
ix, iy, fx, fy = -1, -1, -1, -1
drawing = False

def mouse_cb(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y


def main():
    global ix, iy, fx, fy

    print("\n=== LOTRO 小地图坐标校准工具 ===")
    client_x, client_y = get_client_origin()
    frame_orig, W, H = capture_full_window(client_x, client_y)
    print(f"[INFO] 客户区尺寸: {W} × {H}")
    print("\n操作说明：")
    print("  左键拖拽框选小地图 → Enter/空格确认 → R重置 → Esc退出\n")

    cv2.namedWindow("校准 - 框选小地图区域")
    cv2.setMouseCallback("校准 - 框选小地图区域", mouse_cb)

    while True:
        frame = frame_orig.copy()

        # 画选框
        if ix != -1 and fx != -1:
            x1, y1 = min(ix, fx), min(iy, fy)
            x2, y2 = max(ix, fx), max(iy, fy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 显示当前坐标
            label = f"({x1},{y1}) - ({x2},{y2})  {x2-x1}x{y2-y1}px"
            cv2.putText(frame, label, (x1, max(y1-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        cv2.imshow("校准 - 框选小地图区域", frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # Esc
            print("[INFO] 已退出，未输出结果。")
            break

        if key in (13, 32):  # Enter 或 空格
            if ix == -1 or fx == -1 or abs(fx-ix) < 5 or abs(fy-iy) < 5:
                print("[WARN] 请先用鼠标框选小地图区域！")
                continue
            x1, y1 = min(ix, fx), min(iy, fy)
            x2, y2 = max(ix, fx), max(iy, fy)
            print("\n" + "="*50)
            print("✅ 校准完成！请将以下常量填入 lotro_arrow_fixed.py：")
            print("="*50)
            print(f"MAP_LEFT   = {x1}")
            print(f"MAP_TOP    = {y1}")
            print(f"MAP_RIGHT  = {x2}")
            print(f"MAP_BOTTOM = {y2}")
            print("="*50)
            print(f"区域大小: {x2-x1} × {y2-y1} 像素")

            # 同时验证：截取该区域并显示
            roi = frame_orig[y1:y2, x1:x2]
            roi_big = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("校准结果 - 放大预览", roi_big)
            print("\n[INFO] 已弹出放大预览窗口，确认是小地图后关闭即可。")
            cv2.waitKey(0)
            break

        if key == ord('r'):  # 重置
            ix = iy = fx = fy = -1
            print("[INFO] 选框已重置。")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()