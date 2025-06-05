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
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        brightness = np.mean(gray)
        contrast = np.std(gray)
        saturation = np.mean(hsv[:, :, 1])

        # 规则1：低对比度检测
        if contrast < 30:
            # 高亮度或高饱和度例外
            if brightness > 200 or (saturation > 100 and brightness > 150):
                return False
            return True

        # 规则2：低饱和度检测
        if saturation < 20:
            # 高对比度例外
            return contrast <= 60  # 65.89的case会返回False

        # 规则3：中等对比度+低亮度
        if contrast < 45 and brightness < 120:
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