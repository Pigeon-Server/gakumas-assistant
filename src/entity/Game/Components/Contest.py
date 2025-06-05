import re
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.logger import logger
from src.utils.ocr_instance import get_ocr, OCR_Result
from src.constants import *


@dataclass
class ContestItem(Yolo_Box):
    combat_power: int
    pt: int
    username: str

    def __init__(self, x: float, y: float, w: float, h: float, label: str, frame: np.ndarray):
        super().__init__(x, y, w, h, label, frame)
        ocr_result = get_ocr(frame)
        logger.debug(f"ocr_result: {ocr_result}")
        # [OCR_Result(x=514, y=0, w=89, h=24, text='+139p+', confidence=0.8393095135688782), OCR_Result(x=30, y=32, w=104, h=23, text='総合力合計', confidence=0.999838650226593), OCR_Result(x=26, y=71, w=165, h=32, text='106980', confidence=0.9857919216156006), OCR_Result(x=20, y=128, w=80, h=22, text='ふ一ちや', confidence=0.8578659892082214)]
        self._parse_ocr_results(ocr_result)

    def _parse_ocr_results(self, ocr_results: List[OCR_Result]):
        # 1. 找 “総合力合計” 作为 combat_power 的锚点
        power_anchor = next((r for r in ocr_results if "総合力合計" in r.text), None)
        if not power_anchor:
            raise ValueError("找不到[総合力合計]锚点")

        # 2. pt：最靠右上角的一个（y 最小，其次 x 最大）
        pt_result = min(ocr_results, key=lambda r: (r.y, -r.x))
        digits = re.findall(r'\d+', pt_result.text)
        pt = int(digits[0]) if digits else 0

        # 3. combat_power：在锚点下方最近的一条
        lower_results = [r for r in ocr_results if r.y > power_anchor.y]
        combat_power_result = min(lower_results, key=lambda r: r.y, default=None)
        combat_power = int(combat_power_result.text) if combat_power_result and combat_power_result.text.isdigit() else None

        # 4. username：最靠左下角（y 最大，其次 x 最小）
        username_result = max(ocr_results, key=lambda r: (r.y, -r.x))
        username = username_result.text

        self.pt = pt
        self.combat_power = combat_power
        self.username = username

class ContestList:
    contests: List[ContestItem] = []
    contest_area: np.ndarray
    _start_y: float
    _end_y: float
    _width: float

    def __init__(self, results: Yolo_Results, frame: np.array):
        _, self._width = frame.shape[:2]
        self._start_y = results.filter_by_label(base_labels.button).get_y_max_element().first().h
        self._end_y = results.filter_by_label(base_labels.back_btn).first().y
        self.contest_area = frame[self._start_y:self._end_y, 0:self._width]
        # cv2.imshow("contest_area", self.contest_area)
        self._get_contest_items()

    def __str__(self):
        return str(self.contests)

    def __len__(self):
        return len(self.contests)

    def __iter__(self):
        return iter(self.contests)

    def __bool__(self):
        return bool(self.contests)

    def _append_contest(self, x: float, y: float, w: float, h: float, frame: np.ndarray):
        # cv2.imshow(f"contest_{len(self.contests) + 1}", frame)
        self.contests.append(ContestItem(x, y, w, h, f"contest_{len(self.contests) + 1}",frame))

    def _get_contest_items(self):
        target_color = np.array([123,130,131])
        mask = cv2.inRange(self.contest_area, target_color-20, target_color+20)
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 依次提取每个区域
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # 筛选条件 宽度必须大于帧宽度的一半
            if w > self._width//2 and h > 10:
                roi = self.contest_area[y:y+h, x:x+w]
                self._append_contest(x, box_y := self._start_y+y, x+w, box_y+h, roi)
        # cv2.waitKey(0)

    def get_combat_power_min(self):
        return min(self.contests, key=lambda r: r.combat_power, default=None)

    def get_combat_power_max(self):
        return max(self.contests, key=lambda r: r.combat_power, default=None)

    def get_pt_min(self):
        return min(self.contests, key=lambda r: r.pt, default=None)

    def get_pt_max(self):
        return max(self.contests, key=lambda r: r.pt, default=None)