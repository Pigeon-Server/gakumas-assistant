from dataclasses import dataclass
from typing import List

from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.constants import *
from src.utils.ocr_instance import get_ocr


@dataclass
class Button(Yolo_Box):
    text: str | None
    def __init__(self, element: Yolo_Box, no_text = False):
        super().__init__(element.x, element.y, element.w, element.h, element.label, element.frame)
        self.text = None if no_text else "".join([item.text for item in get_ocr(element.frame)])

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