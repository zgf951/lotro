"""
箭头颜色标定工具
================
用法：
  1. 运行本脚本（游戏保持运行）
  2. 弹出放大的小地图窗口
  3. 用鼠标左键点击箭头上的颜色（可多次点击采样）
  4. 按 S 键输出推荐的 HSV_RANGES 配置
  5. 按 R 键重置采样点
  6. 按 Esc 退出
"""

import cv2
import numpy as np
import win32gui
import mss
import ctypes

# ── 与主程序保持一致 ──────────────────────────────────────────────────
WINDOW_NAME = "The Lord of the Rings Online™"
MAP_LEFT   = 979
MAP_TOP    = 82
MAP_RIGHT  = 1075
MAP_BOTTOM = 159
MAP_W = MAP_RIGHT  - MAP_LEFT
MAP_H = MAP_BOTTOM - MAP_TOP
SCALE = 8   # 放大倍数，方便点击
# ─────────────────────────────────────────────────────────────────────

samples_bgr = []   # 采样的 BGR 值
samples_hsv = []   # 采样的 HSV 值

def get_client_origin():
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    if not hwnd:
        raise RuntimeError("找不到游戏窗口")
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT()
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y

def capture_minimap(origin):
    mon = {"left": origin[0]+MAP_LEFT, "top": origin[1]+MAP_TOP,
           "width": MAP_W, "height": MAP_H}
    with mss.mss() as sct:
        raw = np.array(sct.grab(mon))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

def mouse_cb(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    frame = param["frame"]
    # 缩放坐标还原到原始帧
    ox = x // SCALE
    oy = y // SCALE
    ox = min(max(ox, 0), MAP_W - 1)
    oy = min(max(oy, 0), MAP_H - 1)

    bgr = frame[oy, ox].tolist()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = hsv_img[oy, ox].tolist()

    samples_bgr.append(bgr)
    samples_hsv.append(hsv)

    print("采样点 ({:2d},{:2d})  BGR={:3d},{:3d},{:3d}  HSV={:3d},{:3d},{:3d}".format(
        ox, oy, bgr[0], bgr[1], bgr[2], hsv[0], hsv[1], hsv[2]))

def compute_ranges():
    if not samples_hsv:
        print("尚未采样，请先点击箭头")
        return
    arr = np.array(samples_hsv)
    h_vals = arr[:, 0]
    s_vals = arr[:, 1]
    v_vals = arr[:, 2]

    h_min, h_max = int(h_vals.min()), int(h_vals.max())
    s_min = max(100, int(s_vals.min()) - 20)   # S 下限至少 100，排除背景
    v_min = max(60,  int(v_vals.min()) - 20)   # V 下限至少 60，排除阴影

    # 红色跨越 0°，需要特殊处理
    PAD = 15
    print("\n" + "="*55)
    print("采样汇总（共 {} 个点）：".format(len(samples_hsv)))
    print("  H: {:3d} ~ {:3d}".format(h_min, h_max))
    print("  S: {:3d} ~ {:3d}".format(int(s_vals.min()), int(s_vals.max())))
    print("  V: {:3d} ~ {:3d}".format(int(v_vals.min()), int(v_vals.max())))

    if h_max - h_min > 90:
        print("\n!! H 范围跨度过大（>90°），建议重新只点击箭头核心颜色")

    # 判断是否跨越红色边界（H≈0/180）
    if h_min < 20 or h_max > 160:
        lo1 = (max(0, h_min - PAD), s_min, v_min)
        hi1 = (min(25, h_max + PAD), 255, 255)
        lo2 = (max(155, h_min - PAD), s_min, v_min)
        hi2 = (180, 255, 255)
        print("\n推荐写入 lotro_arrow_v5.py 的 HSV_RANGES：")
        print("HSV_RANGES = [")
        print("    (({:3d}, {:3d}, {:3d}), ({:3d}, {:3d}, {:3d})),   # 橙红".format(
            lo1[0],lo1[1],lo1[2], hi1[0],hi1[1],hi1[2]))
        print("    (({:3d}, {:3d}, {:3d}), ({:3d}, {:3d}, {:3d})),   # 深红".format(
            lo2[0],lo2[1],lo2[2], hi2[0],hi2[1],hi2[2]))
        print("]")
    else:
        lo = (max(0, h_min - PAD), s_min, v_min)
        hi = (min(179, h_max + PAD), 255, 255)
        print("\n推荐写入 lotro_arrow_v5.py 的 HSV_RANGES：")
        print("HSV_RANGES = [")
        print("    (({:3d}, {:3d}, {:3d}), ({:3d}, {:3d}, {:3d})),".format(
            lo[0],lo[1],lo[2], hi[0],hi[1],hi[2]))
        print("]")
    print("="*55 + "\n")

def main():
    origin = get_client_origin()
    print("操作说明：")
    print("  左键点击箭头颜色区域（多点采样更准确）")
    print("  S = 输出推荐 HSV_RANGES")
    print("  R = 重置所有采样点")
    print("  Esc = 退出\n")

    param = {"frame": None}
    cv2.namedWindow("颜色标定 - 点击箭头采样")
    cv2.setMouseCallback("颜色标定 - 点击箭头采样", mouse_cb, param)

    while True:
        frame = capture_minimap(origin)
        param["frame"] = frame

        # 放大显示，方便点击
        big = cv2.resize(frame, (MAP_W * SCALE, MAP_H * SCALE),
                         interpolation=cv2.INTER_NEAREST)

        # 绘制已采样的点
        for bgr, hsv in zip(samples_bgr, samples_hsv):
            # 找近似位置（只能近似，因为只记录了颜色）
            pass

        # 在角落显示实时 HSV
        mx, my = 0, 0  # 无法实时追踪，留空
        cv2.putText(big, "点击箭头采样  S=输出  R=重置  Esc=退出",
                    (4, MAP_H * SCALE - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 230), 1)
        cv2.putText(big, "已采样: {} 点".format(len(samples_hsv)),
                    (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)

        cv2.imshow("颜色标定 - 点击箭头采样", big)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:    # Esc
            break
        elif key == ord('s') or key == ord('S'):
            compute_ranges()
        elif key == ord('r') or key == ord('R'):
            samples_bgr.clear()
            samples_hsv.clear()
            print("采样点已重置")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()