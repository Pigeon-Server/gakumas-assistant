from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.constants import *
from src.utils.ocr_instance import get_ocr


@dataclass
class Button(Yolo_Box):
    text: str | None
    def __init__(self, element: Yolo_Box, no_text = False):
        super().__init__(element.x, element.y, element.w, element.h, element.label, element.frame)
        self.text = None if no_text else "".join([item.text for item in get_ocr(element.frame)])

    def is_disabled(self):
        h, w = self.frame.shape[:2]
        total_pixels = h * w  # 总像素数
        img_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

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

@dataclass
class ButtonList:
    buttons: List[Button]

    def __init__(self, yolo_results: Yolo_Results):
        self.buttons = [Button(el) for el in yolo_results.filter_by_label(base_labels.button)]

    def __bool__(self):
        return bool(self.buttons)

    def __len__(self):
        return len(self.buttons)

    def __iter__(self):
        return iter(self.buttons)

    @classmethod
    def from_list(cls, buttons: List[Button]):
        inst = cls.__new__(cls)
        inst.buttons = buttons
        return inst

    def get_button_by_text(self, text) -> Button | None:
        for button in self.buttons:
            if text in button.text:
                return button
        return None