import os.path
from time import sleep

import cv2
import numpy as np

import matplotlib.pyplot as plt

from src.utils.opencv_tools import *
#
# white_target_upper = np.array([255,255,255])
# white_target_lower = np.array([183,185,185])
# white_lower, white_upper = rgb_to_hsv_range(white_target_upper, white_target_lower)
# hsv_range_to_image_cv(white_lower, white_upper)
#
# cyan_target_upper = np.array([98,222,229])
# cyan_target_lower = np.array([78,136,160])
# cyan_lower, cyan_upper = rgb_to_hsv_range(cyan_target_upper, cyan_target_lower, 3)
# hsv_range_to_image_cv(cyan_lower, cyan_upper)
#
# orange_target_upper = np.array([243,167,66])
# orange_target_lower = np.array([175,103,83])
# orange_lower, orange_upper = rgb_to_hsv_range(orange_target_upper, orange_target_lower, 3)
# hsv_range_to_image_cv(orange_lower, orange_upper)

def is_disabled(img):
    h, w = img.shape[:2]
    total_pixels = h * w  # 总像素数
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 颜色范围定义
    color_ranges = {
        'white': {
            'upper': np.array([179, 25, 255]),
            'lower': np.array([0, 0, 245]),
            'disabled_upper': np.array([106, 24, 193]),
            'disabled_lower': np.array([65, 0, 140])
        },
        'cyan': {
            'upper': np.array([104, 255, 255]),
            'lower': np.array([84, 196, 80]),
            'disabled_upper': np.array([104,178,182]),
            'disabled_lower': np.array([75,118,92])
        },
        'orange': {
            'upper': np.array([24, 255, 255]),
            'lower': np.array([0, 113, 210]),
            'disabled_upper': np.array([22, 178, 196]),
            'disabled_lower': np.array([0, 138, 176])
        }
    }

    def _check_color(color_range):
        # 提取颜色范围并创建蒙版
        mask = cv2.inRange(img_hsv, color_range['lower'], color_range['upper'])
        mask_disabled = cv2.inRange(img_hsv, color_range['disabled_lower'], color_range['disabled_upper'])

        # 如果颜色区域或禁用区域像素超过25%总像素
        if cv2.countNonZero(mask) > total_pixels * 0.25 or cv2.countNonZero(mask_disabled) > total_pixels * 0.25:
            # 判断禁用区域的像素是否超过50%
            if cv2.countNonZero(mask_disabled) > total_pixels * 0.50:
                return True
        return False

    # 按顺序检查每种颜色
    for color in color_ranges.values():
        if _check_color(color):
            return True

    return False



# 读取抠图后的按钮图像
base_path = os.path.join(os.getcwd(), "button_disabled_test")
for filename in os.listdir(base_path):
    if filename.endswith(".png"):
        image = cv2.imread(os.path.join(base_path, filename))
        print(filename)
        print(is_disabled(image))
        # sleep(3)
