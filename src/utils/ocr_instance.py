from dataclasses import dataclass

import numpy as np
import threading
from paddleocr import PaddleOCR
from src.utils.logger import logger


class PaddleOCRLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.ocr = PaddleOCR(
                    lang='japan',
                    # use_angle_cls=False,
                    show_log=False
                )
            return cls._instance


@dataclass
class OCR_Result:
    x: float
    y: float
    w: float
    h: float
    text: str
    confidence: float

    def __init__(self, x, y, w, h, text, confidence):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.confidence = confidence


def init_ocr():
    logger.info("loading PaddleOCR......")
    PaddleOCRLoader()


def get_ocr(img: np.array):
    """
        获取OCR接口实例
    """

    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        logger.warning(f"Empty images or dimensions are illegal: {img.shape if img is not None else 'None'}")
        return []

    def _quad_to_rect(box):
        """将四边形坐标转换为矩形框(x,y,width,height)"""
        # 提取所有x和y坐标
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        # 计算最小外接矩形
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y

        return [round(x), round(y), round(w), round(h)]

    # loader = PaddleOCRLoader()
    # result = loader.ocr.ocr(img, cls=False)

    loader = PaddleOCR(
        lang='japan',
        # use_angle_cls=False,
        show_log=False
    )
    result = loader.ocr(img, cls=False)

    if not result or not result[0]:
        return []

    temp = []
    for res in result[0]:
        temp.append(OCR_Result(
            *_quad_to_rect(res[0]),
            text=res[1][0],
            confidence=res[1][1]
        ))
    return temp
