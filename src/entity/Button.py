from dataclasses import dataclass

from src.utils.number import median

@dataclass
class Button:
    x: float
    y: float
    w: float
    h: float
    text: str
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text

    def get_COL(self):
        return median(self.x, self.w), median(self.y, self.h)