from dataclasses import dataclass

from src.entity.Yolo import Yolo_Box
from src.utils.ocr_instance import get_ocr


@dataclass
class Button(Yolo_Box):
    text: str | None
    def __init__(self, element: Yolo_Box, no_text = False):
        super().__init__(element.x, element.y, element.w, element.h, element.label, element.frame)
        self.text = None if no_text else "".join([item.text for item in get_ocr(element.frame)])