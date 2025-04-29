from dataclasses import dataclass

import numpy as np

from src.entity.Yolo_Box import Yolo_Box
from src.utils.number import median
from src.utils.ocr_instance import get_ocr


@dataclass
class Button(Yolo_Box):
    text: str
    def __init__(self, element: Yolo_Box):
        super().__init__(element.x, element.y, element.w, element.h, element.label, element.frame)
        self.text = "".join([item.text for item in get_ocr(element.frame)])