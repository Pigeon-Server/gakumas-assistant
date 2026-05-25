from dataclasses import dataclass, field
import re
from typing import List, Optional

import cv2
import numpy as np

from src.entity.Yolo import Yolo_Box
from src.utils.string_tools import MatchConfig, string_match
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger
from src.core.inference.ocr_engine import OCRService

ocr_service = OCRService()
debug_tools = DebugTools()
_NEW_PREFIX_PATTERN = re.compile(r"^[A-Za-z]{3,5}")

@dataclass(eq=False)
class TabBarItem(Yolo_Box):
    text: str
    selection_frame: Optional[np.ndarray] = field(default=None, repr=False)

    def __init__(self, x: float, y: float, w: float, h: float, text: str, frame, selection_frame=None):
        self.text = text
        self.selection_frame = selection_frame
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        super().__init__(x, y, w, h, "TabBarItem", frame)


@dataclass(eq=False)
class TabBar(Yolo_Box):
    tab_items: List[TabBarItem]
    selected: TabBarItem = None

    def __init__(self, element: Yolo_Box):
        super().__init__(element.x, element.y, element.w, element.h, element.label, element.frame)
        self.tab_items = self._get_items()
        for tab_item in self.tab_items:
            if _is_selected_tab_frame(tab_item.selection_frame if tab_item.selection_frame is not None else tab_item.frame):
                self.selected = tab_item
                break

    def _get_items(self) -> List[TabBarItem]:
        if self.frame is None or self.frame.size == 0:
            return []
        height, width = self.frame.shape[:2]
        word_boxes = _extract_tab_word_boxes(self.frame)
        tab_groups = _group_word_boxes_into_tab_groups(word_boxes, width)
        logger.debug([_get_box_group_bounds(group) for group in tab_groups])
        tab_items = []
        for group in tab_groups:
            region_x1, region_y1, region_x2, region_y2 = _get_box_group_bounds(group)
            selected_x1, selected_y1, selected_x2, selected_y2 = _expand_tab_region(
                region_x1, region_y1, region_x2, region_y2, width, height
            )
            centered_boxes = _filter_centered_word_boxes(group, height)
            if not centered_boxes:
                logger.debug(f"Skip non-centered tab group: {group}")
                continue
            text_x1, text_y1, text_x2, text_y2 = _expand_tab_text_region(
                *_get_box_group_bounds(centered_boxes),
                width,
                height,
            )
            cropped = self.frame[text_y1:text_y2, text_x1:text_x2]
            selection_cropped = self.frame[selected_y1:selected_y2, selected_x1:selected_x2]
            ocr_results = ocr_service.ocr(cropped)
            text = _normalize_tab_text("".join(item.text for item in ocr_results))
            tab_items.append(TabBarItem(
                el_x := self.x + text_x1,
                el_y := self.y + text_y1,
                el_w := self.x + text_x2,
                el_h := self.y + text_y2,
                text,
                cropped,
                selection_cropped,
            ))
            debug_tools.add_box(el_x, el_y, el_w, el_h, label=text)
        logger.debug(tab_items)
        return tab_items

    def __iter__(self):
        return iter(self.tab_items)

    def __bool__(self):
        return bool(self.tab_items)

    def __len__(self):
        return len(self.tab_items)


def _extract_tab_word_boxes(frame: np.ndarray) -> List[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_accent = cv2.inRange(hsv, (0, 50, 0), (179, 255, 255))
    mask_gray = cv2.inRange(hsv, (0, 0, 0), (0, 0, 190))
    mask_combined = cv2.bitwise_or(mask_accent, mask_gray)

    processed = np.full(frame.shape, 255, dtype=np.uint8)
    processed[mask_combined > 0] = [0, 0, 0]
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_width = max(15, width // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.dilate(morphed, kernel, iterations=1)

    boxes = []
    padding = 5
    min_width = 20
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_width:
            continue
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        if _is_background_like_tab_box(x1, y1, x2, y2, width, height):
            continue
        boxes.append((x1, y1, x2, y2))
    return sorted(boxes, key=lambda box: box[0])


def _merge_word_boxes_into_tab_regions(
        word_boxes: List[tuple[int, int, int, int]],
        width: int,
        height: int,
) -> List[tuple[int, int, int, int]]:
    return [
        _get_box_group_bounds(group)
        for group in _group_word_boxes_into_tab_groups(word_boxes, width)
    ]


def _group_word_boxes_into_tab_groups(
        word_boxes: List[tuple[int, int, int, int]],
        width: int,
) -> List[List[tuple[int, int, int, int]]]:
    if not word_boxes:
        return []

    grouped_boxes: List[list] = []
    for box in word_boxes:
        x1, y1, x2, y2 = box
        for group in grouped_boxes:
            group_x1, _, group_x2, _, group_boxes = group
            overlap = max(0, min(group_x2, x2) - max(group_x1, x1))
            min_width = min(group_x2 - group_x1, x2 - x1)
            if overlap >= max(6, int(min_width * 0.2)):
                group[0] = min(group_x1, x1)
                group[1] = min(group[1], y1)
                group[2] = max(group_x2, x2)
                group[3] = max(group[3], y2)
                group_boxes.append(box)
                break
        else:
            grouped_boxes.append([x1, y1, x2, y2, [box]])

    min_group_width = max(30, width // 16)
    return [
        sorted(group_boxes, key=lambda box: box[0])
        for x1, _, x2, _, group_boxes in grouped_boxes
        if (x2 - x1) >= min_group_width
    ]


def _get_box_group_bounds(boxes: List[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return x1, y1, x2, y2


def _filter_centered_word_boxes(
        word_boxes: List[tuple[int, int, int, int]],
        height: int,
) -> List[tuple[int, int, int, int]]:
    center_y = height / 2
    max_offset = max(6, height // 6)
    return [
        box for box in word_boxes
        if abs(((box[1] + box[3]) / 2) - center_y) <= max_offset
    ]


def _is_background_like_tab_box(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
) -> bool:
    box_width = x2 - x1
    box_height = y2 - y1
    touches_border = x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1
    if not touches_border:
        return False
    if box_height >= height * 0.8:
        return True
    if box_width >= width * 0.15 and box_height <= height * 0.35:
        return True
    return False


def _expand_tab_region(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
) -> tuple[int, int, int, int]:
    box_width = x2 - x1
    box_height = y2 - y1
    horizontal_padding = max(4, int(box_width * 0.05))
    top_padding = min(y1, 2)
    bottom_padding = min(height - y2, max(6, int(box_height * 0.5)))
    return (
        max(0, x1 - horizontal_padding),
        max(0, y1 - top_padding),
        min(width, x2 + horizontal_padding),
        min(height, y2 + bottom_padding),
    )


def _expand_tab_text_region(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
) -> tuple[int, int, int, int]:
    box_width = x2 - x1
    box_height = y2 - y1
    horizontal_padding = max(4, int(box_width * 0.05))
    vertical_padding = max(2, int(box_height * 0.08))
    return (
        max(0, x1 - horizontal_padding),
        max(0, y1 - vertical_padding),
        min(width, x2 + horizontal_padding),
        min(height, y2 + vertical_padding),
    )


def _normalize_tab_text(text: str) -> str:
    text = text.replace(" ", "")
    match = _NEW_PREFIX_PATTERN.match(text)
    if match:
        prefix = match.group(0)
        if len(prefix) == 4 and string_match(prefix, "NEW", MatchConfig(fuzz_threshold=60)):
            stripped = text[4:]
            if stripped:
                return stripped
        if len(prefix) >= 3 and string_match(prefix[:3], "NEW", MatchConfig(fuzz_threshold=60)):
            stripped = text[3:]
            if stripped:
                return stripped
    return text


def _is_selected_tab_frame(frame: np.ndarray, threshold: float = 0.07) -> bool:
    if frame is None or frame.size == 0:
        return False
    height = frame.shape[0]
    lower_frame = frame[int(height * 0.35):, :]
    if lower_frame.size == 0:
        return False
    hsv = cv2.cvtColor(lower_frame, cv2.COLOR_BGR2HSV)
    warm_mask = cv2.inRange(hsv, (0, 80, 120), (25, 255, 255))
    pink_mask = cv2.inRange(hsv, (160, 80, 120), (179, 255, 255))
    highlight_pixels = cv2.countNonZero(cv2.bitwise_or(warm_mask, pink_mask))
    total_pixels = lower_frame.shape[0] * lower_frame.shape[1]
    return total_pixels > 0 and (highlight_pixels / total_pixels) >= threshold
