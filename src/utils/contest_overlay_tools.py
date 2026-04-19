import re
from dataclasses import dataclass

import numpy as np

from src.constants.game.text.contest_text import ContestText
from src.core.inference.ocr_engine import OCRService, OCR_Result, OCR_ResultList
from src.utils.debug_tools import DebugTools
from src.utils.string_tools import MatchConfig, normalize_ocr_jp, string_match

_SEASON_RANKING_MATCH = MatchConfig(fuzz_threshold=72, use_contains=True, normalize=True)
_GRADE_MATCH = MatchConfig(fuzz_threshold=85, use_contains=True, normalize=True)
_RANK_RE = re.compile(r"\d{2,4}")


def _get_ocr_service() -> OCRService:
    return OCRService()


def _get_debug_tools() -> DebugTools:
    return DebugTools()


@dataclass
class ContestSeasonOverlay:
    left: int
    top: int
    right: int
    bottom: int
    rank_text: str | None

    @property
    def center_x(self) -> int:
        return (self.left + self.right) // 2

    @property
    def center_y(self) -> int:
        return (self.top + self.bottom) // 2


@dataclass
class ContestGradeUpSplash:
    left: int
    top: int
    right: int
    bottom: int
    title_text: str

    @property
    def title_center_x(self) -> int:
        return (self.left + self.right) // 2

    @property
    def title_center_y(self) -> int:
        return (self.top + self.bottom) // 2


def _box_right(box: OCR_Result) -> int:
    return int(box.x + box.w)


def _box_bottom(box: OCR_Result) -> int:
    return int(box.y + box.h)


def _find_overlay_rank_box(
        ocr_results: list[OCR_Result],
        frame_width: int,
        title_box: OCR_Result,
) -> OCR_Result | None:
    rank_candidates: list[OCR_Result] = []
    frame_center_x = frame_width / 2
    for item in ocr_results:
        normalized_text = normalize_ocr_jp(item.text).replace("O", "0").replace("o", "0")
        if not _RANK_RE.fullmatch(normalized_text):
            continue
        if len(normalized_text) < 2:
            continue
        if item.y <= title_box.y:
            continue
        if abs(item.cx - frame_center_x) > frame_width * 0.18:
            continue
        rank_candidates.append(item)
    if not rank_candidates:
        return None
    return max(rank_candidates, key=lambda item: (item.h, len(item.text)))


def detect_contest_season_overlay(
        frame: np.ndarray | None,
        ocr_results: OCR_ResultList | list[OCR_Result] | None = None,
        add_debug_box: bool = False,
) -> ContestSeasonOverlay | None:
    """
    检测竞技场“赛季排行”覆盖层。

    该覆盖层没有标准按钮或 modal header，需要基于整帧 OCR 识别：
    - 标题“シーズンランキング”
    - 中央段位锚点“GRADE”
    - 标题下方居中的排名数字
    """
    if frame is None or frame.size == 0:
        return None

    if ocr_results is None:
        ocr_result_list = _get_ocr_service().ocr(frame)
        raw_results = list(ocr_result_list) if ocr_result_list else []
    elif isinstance(ocr_results, OCR_ResultList):
        raw_results = list(ocr_results)
    else:
        raw_results = list(ocr_results)

    if not raw_results:
        return None

    title_boxes = [
        item for item in raw_results
        if string_match(normalize_ocr_jp(item.text), ContestText.OVERLAY.SEASON_RANKING, _SEASON_RANKING_MATCH)
    ]
    grade_boxes = [
        item for item in raw_results
        if string_match(normalize_ocr_jp(item.text), ContestText.OVERLAY.GRADE, _GRADE_MATCH)
    ]
    if not title_boxes or not grade_boxes:
        return None

    title_box = max(title_boxes, key=lambda item: (item.w, -item.y))
    grade_box = max(grade_boxes, key=lambda item: (item.h, -item.y))
    rank_box = _find_overlay_rank_box(raw_results, frame.shape[1], title_box)
    if rank_box is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    left = max(0, min(int(title_box.x), int(grade_box.x), int(rank_box.x)) - int(frame_width * 0.05))
    top = max(0, min(int(title_box.y), int(grade_box.y), int(rank_box.y)) - int(frame_height * 0.04))
    right = min(
        frame_width,
        max(_box_right(title_box), _box_right(grade_box), _box_right(rank_box)) + int(frame_width * 0.05),
    )
    bottom = min(
        frame_height,
        max(_box_bottom(title_box), _box_bottom(grade_box), _box_bottom(rank_box)) + int(frame_height * 0.05),
    )
    overlay = ContestSeasonOverlay(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        rank_text=normalize_ocr_jp(rank_box.text),
    )

    if add_debug_box:
        _get_debug_tools().add_box(
            overlay.left,
            overlay.top,
            overlay.right,
            overlay.bottom,
            color=(0, 255, 255),
            label=f"竞技场赛季排行覆盖层 rank={overlay.rank_text}",
            duration=120,
        )

    return overlay


def detect_contest_grade_up_splash(
        frame: np.ndarray | None,
        ocr_results: OCR_ResultList | list[OCR_Result] | None = None,
        add_debug_box: bool = False,
) -> ContestGradeUpSplash | None:
    """
    检测竞技场「グレードUP」演出页。

    该页面没有可用按钮，YOLO 也没有稳定标签，因此依赖整帧 OCR：
    - 大标题「グレードUP」
    - 奖章下方的「GRADE」
    """
    if frame is None or frame.size == 0:
        return None

    if ocr_results is None:
        ocr_result_list = _get_ocr_service().ocr(frame)
        raw_results = list(ocr_result_list) if ocr_result_list else []
    elif isinstance(ocr_results, OCR_ResultList):
        raw_results = list(ocr_results)
    else:
        raw_results = list(ocr_results)

    if not raw_results:
        return None

    title_match = MatchConfig(fuzz_threshold=70, use_contains=True, normalize=True)
    title_boxes = [
        item for item in raw_results
        if string_match(normalize_ocr_jp(item.text), ContestText.OVERLAY.GRADE_UP, title_match)
    ]
    grade_boxes = [
        item for item in raw_results
        if string_match(normalize_ocr_jp(item.text), ContestText.OVERLAY.GRADE, _GRADE_MATCH)
    ]
    if not title_boxes or not grade_boxes:
        return None

    frame_height, frame_width = frame.shape[:2]
    title_box = max(title_boxes, key=lambda item: (item.w, item.h))
    grade_box = max(
        (
            item for item in grade_boxes
            if item.y > title_box.y and abs(item.cx - title_box.cx) <= frame_width * 0.2
        ),
        key=lambda item: (item.h, item.w),
        default=None,
    )
    if grade_box is None:
        return None

    splash = ContestGradeUpSplash(
        left=max(0, int(title_box.x) - int(frame_width * 0.03)),
        top=max(0, int(title_box.y) - int(frame_height * 0.03)),
        right=min(frame_width, _box_right(title_box) + int(frame_width * 0.03)),
        bottom=min(frame_height, _box_bottom(title_box) + int(frame_height * 0.03)),
        title_text=normalize_ocr_jp(title_box.text),
    )

    if add_debug_box:
        debug_tools = _get_debug_tools()
        debug_tools.add_box(
            splash.left,
            splash.top,
            splash.right,
            splash.bottom,
            color=(255, 0, 255),
            label=f"竞技场GradeUp title={splash.title_text}",
            duration=120,
        )
        debug_tools.add_box(
            int(grade_box.x),
            int(grade_box.y),
            _box_right(grade_box),
            _box_bottom(grade_box),
            color=(0, 255, 0),
            label="竞技场GradeUp徽章锚点",
            duration=120,
        )

    return splash
