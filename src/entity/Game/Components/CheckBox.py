from dataclasses import dataclass

import numpy as np

from src.entity.Yolo import Yolo_Box
from src.utils.opencv_tools import check_status_detection


@dataclass
class CheckBox(Yolo_Box):
    checked: bool

    def __init__(self, element: Yolo_Box):
        super().__init__(element.x, element.y, element.w, element.h, "CheckBox", element.frame)
        self.checked = check_status_detection(element.frame)