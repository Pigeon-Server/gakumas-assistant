import re
from dataclasses import dataclass
from typing import List

import numpy as np
import threading
from paddleocr import PaddleOCR
from src.utils.logger import logger
from src.utils.number import median

@dataclass
class OCR_Result:
    x: float
    y: float
    w: float
    h: float
    cx: float
    cy: float
    text: str
    confidence: float

    def __init__(self, x, y, w, h, text, confidence):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = x + w / 2
        self.cy = y + h / 2
        self.text = text
        self.confidence = confidence

    def __eq__(self, other):
        if not isinstance(other, OCR_Result):
            return False
        return (self.x == other.x and self.y == other.y and self.w == other.w and
                self.h == other.h and self.text == other.text and self.confidence == other.confidence)

    def __hash__(self):
        return hash((self.x, self.y, self.w, self.h, self.text, self.confidence))

@dataclass
class OCR_ResultList:
    results: List[OCR_Result]

    def __init__(self, results: List[OCR_Result]):
        self.results = results

    def __bool__(self):
        return bool(self.results)

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def get_y_min(self) -> "OCR_Result":
        """返回Y轴最小的（最靠上）"""
        return min(self.results, key=lambda box: box.y)

    def get_y_max(self) -> "OCR_Result":
        """返回Y轴最小的（最靠下）"""
        return max(self.results, key=lambda box: box.h)

    def get_x_min(self) -> "OCR_Result":
        """返回X轴最小的（最靠左）"""
        return min(self.results, key=lambda box: box.x)

    def get_x_max(self) -> OCR_Result:
        """"返回X轴最大的（最靠右）"""
        return max(self.results, key=lambda box: box.w)

    @classmethod
    def _from(cls, results: List[OCR_Result]):
        inst = cls.__new__(cls)
        inst.results = results
        return inst


    def auto_merge_lines(self, cy_range: float = 3, width_gap: float = 10) -> 'OCR_ResultList':
        """
        自动合并行
        :param cy_range: 行之间的中心距离
        :param width_gap: 横向间距
        :return: 合并后的OCR结果列表
        """
        merged_results = []
        current_line = []
        prev_result = None

        for result in self.results:
            if prev_result:
                # 判断是否在同一行，且横向距离合适
                if abs(result.cy - prev_result.cy) <= cy_range and (result.x - prev_result.x - prev_result.w) <= width_gap:
                    # 如果在同一行，且横向间距符合条件，加入到当前行
                    current_line.append(result)
                else:
                    # 如果不在同一行，合并当前行
                    merged_text = ' '.join([r.text for r in current_line])
                    # 合并当前行的宽度
                    merged_width = current_line[-1].x + current_line[-1].w - current_line[0].x
                    merged_results.append(OCR_Result(
                        x=current_line[0].x,
                        y=current_line[0].y,
                        w=merged_width,
                        h=current_line[0].h,
                        text=merged_text,
                        confidence=None
                    ))
                    current_line = [result]  # 开始新的一行
            else:
                # 第一行直接加入
                current_line.append(result)

            prev_result = result

        # 合并最后一行
        if current_line:
            merged_text = ' '.join([r.text for r in current_line])
            merged_width = current_line[-1].x + current_line[-1].w - current_line[0].x
            merged_results.append(OCR_Result(
                x=current_line[0].x,
                y=current_line[0].y,
                w=merged_width,
                h=current_line[0].h,
                text=merged_text,
                confidence=None
            ))

        return self._from(merged_results)


    def calculate_confidence(self, current_line: List[OCR_Result]) -> float:
        """
        计算合并行的信心值，可以根据需求修改
        :param current_line: 当前行的OCR结果列表
        :return: 合并行的信心值
        """
        # 计算信心值：这里可以选择平均值或最大值，或其它
        return sum(r.confidence for r in current_line) / len(current_line)  # 取平均信心值

    def exclude(self, exclude_list: List[OCR_Result]) -> 'OCR_ResultList':
        """
        排除指定OCR结果并返回新的OCR_ResultList
        :param exclude_list: 要排除的OCR_Result列表
        :return:
        """
        # 返回不包含指定OCR结果的新列表
        new_results = [result for result in self.results if result not in exclude_list]
        return self._from(new_results)

    def search(self, query: str) -> 'OCR_ResultList':
        """
        使用正则表达式或者关键字查询OCR结果
        :param query: 查询关键字/正则表达式
        :return:
        """
        matched_results = []

        for result in self.results:
            try:
                if re.search(query, result.text):
                    matched_results.append(result)
            except re.error:
                if query in result.text:
                    matched_results.append(result)
        return self._from(matched_results)


class OCRService:
    ocr_engine: PaddleOCR
    def __init__(self):
        self.ocr_engine = PaddleOCR(
            lang='japan',
            # use_angle_cls=False,
            show_log=False
        )

    @classmethod
    def _quad_to_rect(cls, box):
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

    @classmethod
    def _map_result_to_ocr_result(cls, result):
        if not result or not result[0]:
            return []

        temp = []
        for res in result[0]:
            temp.append(OCR_Result(
                *cls._quad_to_rect(res[0]),
                text=res[1][0],
                confidence=res[1][1]
            ))
        return temp

    def ocr(self, img: np.ndarray):
        if img.size == 0:
            logger.warning(f"Empty images or dimensions are illegal: {img.shape if img is not None else 'None'}")
            return []
        result = self.ocr_engine.ocr(img, cls=False)
        result = self._map_result_to_ocr_result(result)
        return OCR_ResultList(result)



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
