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
    """动态调用 UI 模块中的函数，支持运行时替换实现。

    优先从 `src.core.tasks.producer_challenge.ui` 模块中按名称获取函数，
    如果不存在或模块未加载则回退到 fallback。

    Args:
        name: 要在 UI 模块中查找的函数名。
        fallback: 默认实现函数，当 UI 模块中找不到 name 时使用。
        *args: 位置参数，透传给目标函数。
        **kwargs: 关键字参数，透传给目标函数。

    Returns:
        目标函数的返回值，类型取决于被调用的函数。
    """
    ui_module = sys.modules.get("src.core.tasks.producer_challenge.ui")
    if ui_module is not None:
        candidate = getattr(ui_module, name, fallback)
        if candidate is not fallback:
            return candidate(*args, **kwargs)
    return fallback(*args, **kwargs)


def parse_preset_index(text: str | None) -> tuple[int, int] | None:
    """从文本中解析编组编号信息（当前编号/总数）。

    使用正则匹配 "数字/数字" 格式（如 "1/3"），去除空格后进行解析。

    Args:
        text: 待解析的文本，通常来自按钮的 OCR 识别结果。

    Returns:
        tuple[int, int] | None: (当前编组编号, 总编组数)，解析失败或文本为空时返回 None。
    """
    normalized = str(text or "").replace(" ", "")
    if not normalized:
        return None
    match = _PRESET_INDEX_PATTERN.search(normalized)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def get_current_preset_index(app: "AppProcessor") -> tuple[int, int] | None:
    """从当前画面按钮中获取编组编号信息。

    遍历画面中所有按钮，对每个按钮的 text 属性调用 parse_preset_index，
    返回第一个匹配到的 "数字/数字" 格式结果。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。

    Returns:
        tuple[int, int] | None: (当前编组编号, 总编组数)，未找到匹配按钮时返回 None。
    """
    for button in get_buttons(app):
        if parsed := parse_preset_index(button.text):
            return parsed
    return None


def build_preset_swipe_paths(
    boxes: Sequence,
    *,
    frame_width: int,
) -> list[tuple[int, int, int, int]]:
    """根据卡片检测框计算编组横滑路径。

    从检测框的坐标范围计算滑动的起始 X 和结束 X，然后根据 Y 坐标将检测框
    分组为多行，每行生成一条横滑路径（start_x, cy, end_x, cy）。

    Args:
        boxes: 卡片检测框序列，需具有 x, w, cy 属性。
        frame_width: 画面宽度（像素），用于边界约束。

    Returns:
        list[tuple[int, int, int, int]]: 每条路径为 (start_x, start_y, end_x, end_y) 元组。
    """
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
    """从当前画面获取编组横滑路径。

    优先使用指定 card_labels 检测卡片区域，未找到时回退到 BLANK_SLOT 标签。
    获取到检测框后委托给 build_preset_swipe_paths 计算路径。

    Args:
        app: 应用处理器实例，提供 latest_results 和 latest_frame。
        card_labels: 要检测的卡片标签名称列表（如支援卡/记忆卡标签）。

    Returns:
        list[tuple[int, int, int, int]]: 横滑路径列表。

    Raises:
        TimeoutError: 未检测到卡片区域或无法计算路径时抛出。
    """
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
    """通过横向滑动切换到目标编组。

    先 OCR 识别当前编组编号，然后循环横滑直到到达目标编号。每次滑动后重新
    识别编号以判断是否到达目标，同时自动校准 left_swipe_increases 方向。
    支持多行路径轮询，处理滑动卡住的情况。

    Args:
        app: 应用处理器实例，提供 latest_results 和 device 操作。
        target_index: 目标编组编号（1-based）。
        card_labels: 要检测的卡片标签名称列表，用于计算滑动路径。
        description: 日志和异常信息中的描述性前缀（如 "支援卡"、"记忆卡"）。
        max_swipes: 最大滑动次数，默认根据目标距离自动计算。

    Returns:
        bool: 已在目标编组返回 True（含初始就在目标的情况）。

    Raises:
        ValueError: target_index 超出范围时抛出。
        TimeoutError: 无法识别编组编号或超过最大滑动次数时抛出。
    """
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
