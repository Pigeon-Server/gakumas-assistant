from dataclasses import dataclass

from src.entity.Yolo_Box import Yolo_Box
from src.utils.number import median
from src.utils.ocr_instance import get_ocr


@dataclass
class Button:
    x: float
    y: float
    w: float
    h: float
    text: str
    def __init__(self, element: Yolo_Box):
        self.x = element.x
        self.y = element.y
        self.w = element.w
        self.h = element.h
        self.text = "".join([item.text for item in get_ocr(element.frame)])

    def get_COL(self):
        return int(median(self.x, self.w)), int(median(self.y, self.h))