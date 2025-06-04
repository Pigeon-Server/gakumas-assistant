import colorsys
from typing import Tuple

import cv2
import numpy as np


def rgb_to_hsv_range(rgb1, rgb2, expand_ratio=0.0):
    """
    输入两个 RGB 值（0-255），输出 OpenCV HSV 范围的 lower 和 upper，支持范围扩大。

    返回:
        lower: tuple(int, int, int)，OpenCV HSV下限（H:0-179, S,V:0-255）
        upper: tuple(int, int, int)，OpenCV HSV上限
    """
    def to_hsv_opencv(rgb):
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_opencv = int(h * 179)          # colorsys h是0~1，乘179映射OpenCV
        s_opencv = int(s * 255)
        v_opencv = int(v * 255)
        return (h_opencv, s_opencv, v_opencv)

    hsv1 = to_hsv_opencv(rgb1)
    hsv2 = to_hsv_opencv(rgb2)

    hsv_min = [min(a, b) for a, b in zip(hsv1, hsv2)]
    hsv_max = [max(a, b) for a, b in zip(hsv1, hsv2)]

    def expand_range(min_val, max_val, total):
        delta = (max_val - min_val) * expand_ratio / 2
        low = max(0, int(min_val - delta))
        high = min(total, int(max_val + delta))
        return low, high

    h_min, h_max = expand_range(hsv_min[0], hsv_max[0], 179)
    s_min, s_max = expand_range(hsv_min[1], hsv_max[1], 255)
    v_min, v_max = expand_range(hsv_min[2], hsv_max[2], 255)

    lower = (h_min, s_min, v_min)
    upper = (h_max, s_max, v_max)

    return lower, upper

def hsv_range_to_image_cv(lower, upper, height=50, width=300):
    """
    用 OpenCV HSV 范围的 lower 和 upper 生成一张条形图，表示色调范围。
    lower, upper: (H:0-179, S:0-255, V:0-255)
    """
    h_vals = np.linspace(lower[0], upper[0], width)
    s_val = (lower[1] + upper[1]) / 2
    v_val = (lower[2] + upper[2]) / 2

    img = np.zeros((height, width, 3), dtype=np.uint8)

    for i, h in enumerate(h_vals):
        # colorsys hsv 输入范围是 H:0-1, S/V:0-1，需要转换
        h_norm = h / 179
        s_norm = s_val / 255
        v_norm = v_val / 255
        r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
        img[:, i, 0] = int(b * 255)  # OpenCV 是 BGR 顺序
        img[:, i, 1] = int(g * 255)
        img[:, i, 2] = int(r * 255)

    cv2.imshow("HSV Range", img)
    cv2.waitKey(10)

def check_color_in_region(
        frame: np.array,
        region: Tuple[int, int, int, int],
        lower_color: Tuple[int, int, int],
        upper_color: Tuple[int, int, int],
        threshold=1
):
    """
    检查图像某区域是否存在指定 RGB 范围的颜色

    参数:
        frame         : 输入图像（BGR）
        region      : 区域 (x, y, w, h)，从图像中裁剪
        lower_color   : RGB 下限，如 (255, 0, 0)
        upper_color   : RGB 上限，如 (255, 100, 100)
        threshold   : 最小像素数量（默认 1），低于该值视为不存在

    返回:
        True 表示存在颜色，False 表示不存在
    """
    if frame.size == 0:
        return False
    x, y, w, h = map(int, region)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return False
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv_roi", hsv_roi)
    # 自动修正 RGB 顺序并转换为 HSV
    hsv_lower, hsv_upper = rgb_to_hsv_range(lower_color, upper_color,5)
    hsv_range_to_image_cv(hsv_lower, hsv_upper)
    mask = cv2.inRange(hsv_roi, hsv_lower, hsv_upper)
    cv2.imshow("mask", mask)
    cv2.waitKey(10)
    print(cv2.countNonZero(mask))
    return cv2.countNonZero(mask) >= threshold
