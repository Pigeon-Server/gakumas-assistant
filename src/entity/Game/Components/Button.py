from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.core.inference.ocr_engine import OCRService
from src.utils.debug_tools import DebugTools
from src.utils.string_tools import string_match, MatchConfig

ocr_service = OCRService()


def _ocr_text(image: np.ndarray) -> str:
    """对按钮图像执行一次 OCR，并拼接结果。"""
    results = ocr_service.ocr(image)
    if not results:
        return ""
    return "".join(item.text for item in results)


def _extract_button_text(image: np.ndarray) -> str:
    """提取按钮文字；原图失败时，对白字按钮做轻量预处理后重试。"""
    if image is None or getattr(image, "size", 0) == 0:
        return ""

    text = _ocr_text(image)
    if text:
        return text

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        img_hsv,
        np.array([0, 0, 180]),
        np.array([180, 80, 255]),
    )
    if cv2.countNonZero(white_mask) > image.shape[0] * image.shape[1] * 0.01:
        white_mask = cv2.resize(white_mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        text = _ocr_text(cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR))
        if text:
            return text

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    return _ocr_text(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

@dataclass(eq=False)
class Button(Yolo_Box):
    text: str | None

    def __init__(self, element: Yolo_Box, no_text: bool = False):
        super().__init__(element.x, element.y, element.w, element.h, element.label, element.frame)
        self.text = None if no_text else _extract_button_text(element.frame)

    def __eq__(self, other):
        if isinstance(other, Button):
            return super().__eq__(other) and self.text == other.text
        return False

    def is_disabled(self):
        h, w = self.frame.shape[:2]
        total_pixels = h * w  # 总像素数
        img_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        # 颜色范围定义
        color_rules = {
            'white': {
                'upper': np.array([155, 30, 255]),
                'lower': np.array([0, 0, 120]),
                'disabled_upper': np.array([106, 24, 193]),
                'disabled_lower': np.array([65, 0, 140])
            },
            'cyan': {
                'upper': np.array([98,255,255]),
                'lower': np.array([86,61,0]),
                'disabled_upper': np.array([97,169,191]),
                'disabled_lower': np.array([86,109,101])
            },
            'orange': {
                'upper': np.array([17, 255, 255]),
                'lower': np.array([0, 113, 130]),
                'disabled_upper': np.array([22, 178, 196]),
                'disabled_lower': np.array([0, 138, 176])
            },
            'transparent-grey1': {
                'upper': np.array([122, 120, 180]),
                'lower': np.array([0, 0, 90]),
                'disabled_upper': np.array([68,90,120]),
                'disabled_lower': np.array([9,0,95])
            }
        }


        # 按顺序检查每种颜色
        for rule in color_rules.values():
            button_mask = cv2.inRange(img_hsv, rule['lower'], rule['upper'])

            # 小于 60% 判定不是这个颜色规则所属范围
            if cv2.countNonZero(button_mask) < total_pixels * 0.50:
                continue

            mask_disabled = cv2.inRange(img_hsv, rule['disabled_lower'], rule['disabled_upper'])

            if cv2.countNonZero(mask_disabled) > cv2.countNonZero(button_mask) * 0.50:
                return True
            return False

        return False

@dataclass
class ButtonList:
    buttons: List[Button]

    def __init__(self, yolo_results: Yolo_Results):
        self.buttons = [Button(el) for el in yolo_results.filter_by_label(BaseUILabels.BUTTON)]
        if self.buttons:
            for btn in self.buttons:
                DebugTools().add_box(
                    int(btn.x),
                    int(btn.y),
                    int(btn.w),
                    int(btn.h),
                    color=(255, 0, 0) if btn.is_disabled() else (0, 255, 0),
                    label=btn.text,
                    duration=3
                )

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

    def get_button_by_text(self, text, match_config: MatchConfig = None) -> Button | None:
        for button in self.buttons:
            if string_match(button.text, text, match_config):
                return button
        return None
