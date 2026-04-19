from __future__ import annotations

import re
import sys
from statistics import median
from typing import TYPE_CHECKING, Sequence

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.logger import logger

from .common import get_buttons, inertial_swipe

if TYPE_CHECKING:
    from src.main import AppProcessor

_PRESET_INDEX_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")


def _call_ui_attr(name: str, fallback, *args, **kwargs):
    ui_module = sys.modules.get("src.core.tasks.producer_challenge.ui")
    if ui_module is not None:
        candidate = getattr(ui_module, name, fallback)
        if candidate is not fallback:
            return candidate(*args, **kwargs)
    return fallback(*args, **kwargs)


def parse_preset_index(text: str | None) -> tuple[int, int] | None:
    normalized = str(text or "").replace(" ", "")
    if not normalized:
        return None
    match = _PRESET_INDEX_PATTERN.search(normalized)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def get_current_preset_index(app: "AppProcessor") -> tuple[int, int] | None:
    for button in get_buttons(app):
        if parsed := parse_preset_index(button.text):
            return parsed
    return None


def build_preset_swipe_paths(
    boxes: Sequence,
    *,
    frame_width: int,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    left = int(min(box.x for box in boxes))
    right = int(max(box.w for box in boxes))
    span = max(1, right - left)
    margin = max(40, int(span * 0.15))
    start_x = min(frame_width - 40, right - margin)
    end_x = max(40, left + margin)
    if start_x - end_x < 160:
        start_x = max(end_x + 160, int(frame_width * 0.75))
        end_x = int(frame_width * 0.25)

    rows: list[list] = []
    current_row: list = []
    row_anchor_cy: int | None = None
    for box in sorted(boxes, key=lambda item: item.cy):
        if row_anchor_cy is None or abs(box.cy - row_anchor_cy) <= 120:
            if not current_row:
                row_anchor_cy = box.cy
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            row_anchor_cy = box.cy
    if current_row:
        rows.append(current_row)

    return [
        (
            start_x,
            int(round(median([box.cy for box in row]))),
            end_x,
            int(round(median([box.cy for box in row]))),
        )
        for row in rows
    ]


def get_preset_swipe_paths(
    app: "AppProcessor",
    *,
    card_labels: Sequence[str],
) -> list[tuple[int, int, int, int]]:
    boxes = list(app.latest_results.filter_by_labels(list(card_labels)))
    if not boxes:
        boxes = list(app.latest_results.filter_by_label(BaseUILabels.BLANK_SLOT))
    if not boxes:
        raise TimeoutError("未识别到可切换编组的卡片区域")

    frame_width = app.latest_frame.shape[1]
    paths = build_preset_swipe_paths(boxes, frame_width=frame_width)
    if not paths:
        raise TimeoutError("未能计算编组横滑路径")
    return paths


def select_preset_by_horizontal_swipe(
    app: "AppProcessor",
    target_index: int,
    *,
    card_labels: Sequence[str],
    description: str,
    max_swipes: int | None = None,
) -> bool:
    current_info = _call_ui_attr(
        "get_current_preset_index",
        get_current_preset_index,
        app,
    )
    if current_info is None:
        raise TimeoutError(f"{description}页面未识别到编组编号")

    current_index, total = current_info
    if target_index < 1 or target_index > total:
        raise ValueError(
            f"{description}预设编号超出范围: {target_index} (1-{total})"
        )
    if current_index == target_index:
        logger.debug(f"{description}已在目标编组 {current_index}/{total}")
        return True

    left_swipe_increases = True
    stuck_attempts = 0
    swipe_limit = max_swipes or max(abs(target_index - current_index) * 2 + 4, 6)

    for attempt in range(1, swipe_limit + 1):
        paths = _call_ui_attr(
            "get_preset_swipe_paths",
            get_preset_swipe_paths,
            app,
            card_labels=card_labels,
        )
        start_x, start_y, end_x, end_y = paths[(attempt - 1) % len(paths)]
        should_increase = target_index > current_index
        swipe_left = should_increase if left_swipe_increases else not should_increase
        if swipe_left:
            _call_ui_attr(
                "inertial_swipe",
                inertial_swipe,
                app,
                start_x,
                start_y,
                end_x,
                end_y,
                duration=0.35,
                settle_timeout=4.5,
            )
        else:
            _call_ui_attr(
                "inertial_swipe",
                inertial_swipe,
                app,
                end_x,
                start_y,
                start_x,
                end_y,
                duration=0.35,
                settle_timeout=4.5,
            )

        updated_info = _call_ui_attr(
            "get_current_preset_index",
            get_current_preset_index,
            app,
        )
        if updated_info is None:
            raise TimeoutError(f"{description}页面横滑后未识别到新的编组编号")

        updated_index, updated_total = updated_info
        total = updated_total
        if updated_index == target_index:
            logger.debug(f"{description}切换到目标编组 {updated_index}/{total}")
            return True

        if updated_index != current_index:
            if swipe_left:
                left_swipe_increases = updated_index > current_index
            else:
                left_swipe_increases = updated_index < current_index
            logger.debug(
                f"{description}横滑后编组变化: {current_index}/{total} -> "
                f"{updated_index}/{total}, left_swipe_increases={left_swipe_increases}"
            )
            current_index = updated_index
            stuck_attempts = 0
            continue

        stuck_attempts += 1
        logger.debug(
            f"{description}横滑后编组编号未变化: {current_index}/{total}, "
            f"attempt={attempt}/{swipe_limit}, "
            f"path_index={(attempt - 1) % len(paths)}"
        )
        if stuck_attempts >= len(paths):
            left_swipe_increases = not left_swipe_increases
            stuck_attempts = 0

    raise TimeoutError(
        f"{description}未能切换到目标编组 {target_index}/{total}，"
        f"当前仍为 {current_index}/{total}"
    )
