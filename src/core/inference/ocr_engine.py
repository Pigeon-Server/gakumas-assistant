import threading
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src.entity.Base import SingletonMeta
from src.entity.Yolo import Yolo_Box
from src.constants.ocr.backend import OCRBackendType
from src.core.inference.ocr_backends import (
    RapidOCRBackend,
    create_ocr_backend,
)
from src.utils.dml_manager import DMLManager
from src.utils.logger import logger

from src.utils.opencv_tools import letterbox
from src.utils.performance_tools import timeit
from src.utils.string_tools import fullwidth_to_halfwidth, normalize_ocr_jp, MatchConfig, string_match


class OCRLoader(metaclass=SingletonMeta):
    _instance = None
    _infer_lock = threading.Lock()

    def __init__(self):
        self._backend = create_ocr_backend()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def requires_dml_lock(self) -> bool:
        return bool(getattr(self._backend, "requires_dml_lock", False))

    def _fallback_to_rapidocr(self, exc: Exception):
        if self.backend_name == OCRBackendType.RAPIDOCR:
            raise exc
        logger.warning(
            "OCR backend {} failed during inference, fallback to {}: {}",
            self.backend_name,
            OCRBackendType.RAPIDOCR,
            exc,
        )
        self._backend = RapidOCRBackend()
        return self._backend

    def __call__(self, *args, **kwargs):
        with self._infer_lock:
            try:
                return self._backend.infer(*args, **kwargs)
            except Exception as exc:
                fallback_backend = self._fallback_to_rapidocr(exc)
                return fallback_backend.infer(*args, **kwargs)

@dataclass
class OCR_Result(Yolo_Box):

    text: str
    confidence: float | None

    def __init__(self, x, y, w, h, text, confidence):
        # OCR results use (x1, y1, width, height) but Yolo_Box stores (x1, y1, x2, y2).
        # Pass x2=x+w and y2=y+h so that cx/cy are computed correctly as true centers.
        # Then restore w and h to width/height for merge_lines compatibility.
        super().__init__(x, y, x + w, y + h, None, None)
        self.w = w
        self.h = h
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

    def first(self) -> OCR_Result:
        return self.results[0]

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

    @staticmethod
    def calculate_confidence(current_line: List[OCR_Result]) -> float:
        """
        计算平均信心值
        :param current_line: 当前行的OCR结果列表
        :return: 平均信心值
        """
        valid_scores = [r.confidence for r in current_line if r.confidence is not None]
        if not valid_scores:
            return 0.0
        return sum(valid_scores) / len(valid_scores)  # 取平均信心值

    def exclude(self, exclude_list: List[OCR_Result]) -> 'OCR_ResultList':
        """
        排除指定OCR结果并返回新的OCR_ResultList
        :param exclude_list: 要排除的OCR_Result列表
        :return:
        """
        # 返回不包含指定OCR结果的新列表
        new_results = [result for result in self.results if result not in exclude_list]
        return self._from(new_results)

    def search(self, query: str | list[str], config: MatchConfig = None) -> 'OCR_ResultList':
        """
        使用 string_match 查询 OCR 结果
        :param query: 查询关键字或列表
        :param config: 匹配配置
        :return: OCR_ResultList
        """
        if config is None:
            config = MatchConfig()

        matched_results = []
        for result in self.results:
            match_result = string_match(result.text, query, config)
            if match_result:
                matched_results.append(result)

        return self._from(matched_results)



class OCRService:
    ocr_engine: OCRLoader

    def __init__(self):
        # Eagerly initialize the OCR model so it loads at app startup
        # (OCRLoader is a singleton, so cost is paid only once)
        self.ocr_engine = OCRLoader()

    def _get_ocr_engine(self) -> OCRLoader:
        return self.ocr_engine

    @classmethod
    def _map_result_to_ocr_result(cls, result, ratio: float, dw: float, dh:float) -> List[OCR_Result]:
        if result is None or result.boxes is None:
            return []

        temp = []
        for box, text, score in zip(result.boxes, result.txts, result.scores):
            x, y, w, h = cv2.boundingRect(box.astype(np.int32))
            x = (x - dw) / ratio
            y = (y - dh) / ratio
            w = w / ratio
            h = h / ratio
            temp.append(OCR_Result(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                text=normalize_ocr_jp(fullwidth_to_halfwidth(text)),
                confidence=None if score is None else float(score)
            ))
        return temp

    @timeit
    def ocr(self, img: np.ndarray):
        if img.size == 0:
            logger.warning(f"Empty images or dimensions are illegal: {img.shape if img is not None else 'None'}")
            return []
        img_letterbox, ratio, (dw, dh) = letterbox(img)
        # Keep OCR execution serialized through the existing DMLManager lock so a
        # mid-call fallback from Vision -> RapidOCR stays on the same safe path.
        with DMLManager.get_lock():
            result = self._get_ocr_engine()(img_letterbox, use_cls=False)
        result = self._map_result_to_ocr_result(result, ratio, dw, dh)
        return OCR_ResultList(result)
