from dataclasses import dataclass

import numpy as np

from src.utils.number import median


@dataclass
class Yolo_Box:
    x: float
    y: float
    w: float
    h: float
    label: str
    frame: np.ndarray

    def __init__(self, x: float, y: float, w: float, h: float, label: str, frame: np.ndarray):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.frame = frame

    def get_COL(self):
        return int(median(self.x, self.w)), int(median(self.y, self.h))