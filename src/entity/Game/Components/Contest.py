import re
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.core.inference.ocr_engine import OCRService, OCR_Result
from src.utils.debug_tools import DebugTools
from src.utils.i18n_tools import i18n_text
from src.utils.opencv_tools import gen_color_mask
from src.utils.string_tools import string_match, MatchConfig
from src.utils.logger import logger

ocr_service = OCRService()
debug_tools = DebugTools()

# OCR 常将 "力" 误识别为 "カ"（外形相似），此处用模糊匹配兼容两种情况
_ANCHOR_TEXT = "総合力合計"
_ANCHOR_MATCH_CONFIG = MatchConfig(use_fuzz=True, fuzz_threshold=70, use_contains=True)

@dataclass
class ContestItem(Yolo_Box):
    combat_power: int
    pt: int
    username: str

    def __init__(self, x: float, y: float, w: float, h: float, label: str, frame: np.ndarray):
        super().__init__(x, y, w, h, label, frame)
        ocr_result = ocr_service.ocr(frame)
        self._parse_ocr_results(ocr_result)

    def _parse_ocr_results(self, ocr_results: List[OCR_Result]):
        # 1. 找 “総合力合計” 作为 combat_power 的锚点
        #    OCR 常将 "力" 误识别为 "カ" (片假名)，使用模糊匹配兼容
        power_anchor = next(
            (r for r in ocr_results if string_match(r.text, _ANCHOR_TEXT, _ANCHOR_MATCH_CONFIG)),
            None,
        )
        if not power_anchor:
            raise ValueError(i18n_text("backend.game.combatPowerAnchorNotFound", fallback="找不到[総合力合計]锚点"))

        # 2. pt：最靠右上角的一个（y 最小，其次 x 最大）
        pt_result = min(ocr_results, key=lambda r: (r.y, -r.x))
        digits = re.findall(r'\d+', pt_result.text.replace("O", "0"))
        pt = int(digits[0]) if digits else 0

        lower_results = [r for r in ocr_results if r.y > power_anchor.y]
        combat_power_result = min(lower_results, key=lambda r: r.y, default=None)
        digits = re.findall(r'\d+', combat_power_result.text.replace("O", "0")) if combat_power_result else []
        combat_power = int(digits[0]) if digits else None

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
    is_exhausted: bool

    def __init__(self, results: Yolo_Results, frame: np.ndarray):
        height, width = frame.shape[:2]
        self.contests = []
        self.contest_area = frame[0:0, 0:width]
        self._start_y = 0
        self._end_y = float(height)
        self.is_exhausted = False

        self._start_y, self._end_y = self._get_contest_area_bounds(results, height)
        logger.debug(f"start_y={self._start_y}, end_y={self._end_y}")

        if self._end_y <= self._start_y:
            logger.debug("Invalid contest area bounds")
            return

        self.contest_area = frame[int(self._start_y):int(self._end_y), 0:width]
        if self.contest_area is None or self.contest_area.size == 0:
            logger.debug("No contest area")
            return
        contest_area_ocr = "".join([res.text for res in ocr_service.ocr(self.contest_area)])
        if string_match(contest_area_ocr, "消費しました", MatchConfig(fuzz_threshold=70)):
            logger.debug("Today's challenge opportunities have all been used up")
            self.is_exhausted = True
            return
        self._get_contest_items()

    @staticmethod
    def _get_contest_area_bounds(results: Yolo_Results, frame_height: int) -> tuple[int, int]:
        fallback_start = int(frame_height * 0.45)
        fallback_end = int(frame_height * 0.95)
        min_roi_height = int(frame_height * 0.12)

        back_buttons = results.filter_by_label(BaseUILabels.BACK_BTN)
        if back_buttons:
            end_y = int(back_buttons.get_y_min_element().first().y)
        else:
            end_y = fallback_end

        buttons = results.filter_by_label(BaseUILabels.BUTTON)
        candidate_bottoms: list[int] = []

        if buttons:
            # 底部弹窗按钮会污染边界推断，优先使用上半屏按钮。
            upper_buttons = [box for box in buttons if box.cy <= frame_height * 0.75]
            if upper_buttons:
                candidate_bottoms.extend(int(box.h) for box in upper_buttons)

            candidate_bottoms.extend(
                int(box.h)
                for box in buttons
                if frame_height * 0.28 <= box.h <= frame_height * 0.72
            )

        start_y = max(candidate_bottoms) if candidate_bottoms else fallback_start

        if end_y - start_y < min_roi_height:
            logger.debug(
                f"Contest bounds collapsed ({start_y}, {end_y}), fallback to ({fallback_start}, {fallback_end})"
            )
            start_y = fallback_start
            end_y = max(end_y, fallback_end)

        start_y = max(0, min(start_y, frame_height - 1))
        end_y = max(start_y + 1, min(end_y, frame_height))
        return start_y, end_y

    def __str__(self):
        return str(self.contests)

    def __len__(self):
        return len(self.contests)

    def __iter__(self):
        return iter(self.contests)

    def __bool__(self):
        return bool(self.contests)

    def _append_contest(self, x: float, y: float, w: float, h: float, frame: np.ndarray):
        self.contests.append(ContestItem(x, y, w, h, f"contest_{len(self.contests) + 1}",frame))

    def _get_contest_items(self):
        height, width, _ = self.contest_area.shape

        # 灰色
        lower1 = (0,0,75)
        upper1 = (179,75,140)
        # 白色
        lower2 = (0,0,235)
        upper2 = (179,15,255)

        mask1 = gen_color_mask(self.contest_area, lower1, upper1)
        mask2 = gen_color_mask(self.contest_area, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_item_height = max(80, int(height * 0.12))
        max_item_height = max(min_item_height + 1, int(height * 0.45))

        candidate_boxes: list[tuple[int, int, int, int, np.ndarray]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 仅保留宽且高度合理的候选区域，避免把整块背景或噪点当成对手卡。
            if w > width * 0.5 and min_item_height <= h <= max_item_height:
                candidate_boxes.append((x, y, w, h, cnt))
                continue
            debug_tools.add_box(x, box_y := int(self._start_y+y), x+w, box_y+h, color=(255,0,0))

        selected: list[tuple[int, int, int, int, np.ndarray]] = []
        for x, y, w, h, cnt in sorted(candidate_boxes, key=lambda item: item[1]):
            if selected and abs(y - selected[-1][1]) < max(20, h // 3):
                continue
            selected.append((x, y, w, h, cnt))
            if len(selected) >= 3:
                break

        for x, y, w, h, _ in selected:
            roi = self.contest_area[y:y + h, x:x + w]
            self._append_contest(x, box_y := self._start_y + y, x + w, box_y + h, roi)
            debug_tools.add_box(x, box_y := int(self._start_y + y), x + w, box_y + h, color=(127, 255, 0))

    def _get_valid_contests(self) -> List[ContestItem]:
        return [r for r in self.contests if r.combat_power is not None]

    def get_combat_power_min(self):
        return min(self._get_valid_contests(), key=lambda r: r.combat_power, default=None)

    def get_combat_power_max(self):
        return max(self._get_valid_contests(), key=lambda r: r.combat_power, default=None)

    def get_pt_min(self):
        return min(self._get_valid_contests(), key=lambda r: r.pt, default=None)

    def get_pt_max(self):
        return max(self._get_valid_contests(), key=lambda r: r.pt, default=None)