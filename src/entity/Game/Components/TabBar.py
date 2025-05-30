from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src.entity.Yolo import Yolo_Box
from src.utils.ocr_instance import get_ocr, OCR_Result
from src.utils.yolo_tools import check_status_detection


@dataclass
class TabBarItem(Yolo_Box):
    text: str
    def __init__(self, x: float, y: float, w: float, h: float, text: str, body_element: Yolo_Box):
        self.text = text
        super().__init__(body_element.x+x, body_element.y+y, body_element.w+w, body_element.h+h, "TabBarItem", body_element.frame)
        self.frame = body_element.frame[y:y+h, x:x+w]

@dataclass
class TabBar(Yolo_Box):
    tab_items: List[TabBarItem]
    selected: TabBarItem = None
    def __init__(self, element: Yolo_Box):
        super().__init__(element.x, element.y, element.w, element.h, element.label, element.frame)
        self.tab_items = [TabBarItem(item.x, item.y, item.w, item.h, item.text, element) for item in get_ocr(element.frame)]
        for tab_item in self.tab_items:
            if check_status_detection(tab_item.frame):
                self.selected = tab_item
                break

