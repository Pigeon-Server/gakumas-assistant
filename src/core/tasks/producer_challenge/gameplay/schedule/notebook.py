from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCRService
from src.utils.logger import logger
from src.utils.string_tools import normalize_ocr_jp

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


_NOTEBOOK_OCR = OCRService()

# 特殊事件关键词（出现在宽幅横幅上，OCR 可读）
_NOTEBOOK_SPECIAL_KEYWORDS = ProduceText.NOTEBOOK_SPECIAL_KEYWORDS

# ── 图标背景色 HSV 色相映射（基于实测中位值） ──
# OpenCV HSV: H=0-180
# 实测值: ボーカル H≈164, ダンス H≈98, ビジュアル H≈21,
#         お出かけ H≈76-89, 活動 H≈17, 授業 H≈111
_NOTEBOOK_ICON_HUE_MAP: list[tuple[int, int, str]] = [
    (10, 19, ProduceText.BUSINESS),        # H≈17, 橙色
    (19, 40, ProduceText.VISUAL_LESSON),   # H≈21, 黄色
    (40, 95, ProduceText.OUTING),          # H≈76-89, 绿〜青绿
    (95, 108, ProduceText.DANCE_LESSON),   # H≈98, 青色
    (108, 130, ProduceText.CLASS),         # H≈111, 蓝紫
    (130, 180, ProduceText.VOCAL_LESSON),  # H≈164, 品红/粉红
    (0, 10, ProduceText.VOCAL_LESSON),     # H≈0-9, 红色端（品红绕回）
]

# 图标检测参数
_ICON_SAT_THRESHOLD = 80
_ICON_VAL_THRESHOLD = 100
_ICON_MIN_AREA = 2500
_ICON_MAX_WIDTH = 150
_ICON_MIN_WIDTH = 60
_ICON_MIN_HEIGHT = 50
_ICON_MAX_ASPECT = 2.0
_ICON_ROW_GAP = 100

# ── P手帳 图标 CLIP 识别 + 色相回退 ──
# 未识别的图标按色相缓存（每个色相类型仅缓存一张），
# 当后续 CLIP 学习到新类型时可重新识别。
_notebook_icon_clip_cache: dict[str, np.ndarray] = {}


def _classify_icon_hue(h: float) -> str:
    """根据 HSV 色相值分类图标对应的行动类型。"""
    for h_min, h_max, label in _NOTEBOOK_ICON_HUE_MAP:
        if h_min <= h < h_max:
            return label
    return "不明"


def _detect_notebook_icons(frame: np.ndarray) -> list[dict[str, Any]]:
    """检测 P手帳 画面中的彩色行动图标。"""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    panel_x1, panel_y1 = int(width * 0.04), int(height * 0.16)
    panel_x2, panel_y2 = int(width * 0.72), int(height * 0.90)
    panel_hsv = hsv[panel_y1:panel_y2, panel_x1:panel_x2]

    sat_mask = (
        (panel_hsv[:, :, 1] > _ICON_SAT_THRESHOLD)
        & (panel_hsv[:, :, 2] > _ICON_VAL_THRESHOLD)
    ).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        sat_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    icons: list[dict[str, Any]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _ICON_MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > _ICON_MAX_WIDTH or bw < _ICON_MIN_WIDTH or bh < _ICON_MIN_HEIGHT:
            continue
        if bh / bw > _ICON_MAX_ASPECT:
            continue

        roi_mask = sat_mask[y:y + bh, x:x + bw]
        roi_hue = panel_hsv[y:y + bh, x:x + bw, 0]
        sat_pixels = roi_mask > 0
        if np.count_nonzero(sat_pixels) < 50:
            continue
        median_h = float(np.median(roi_hue[sat_pixels]))

        ax, ay = x + panel_x1, y + panel_y1
        icon_image = frame[ay:ay + bh, ax:ax + bw].copy()

        action_type = _classify_icon_hue(median_h)
        icons.append(
            {
                "abs_x": ax,
                "abs_y": ay,
                "w": bw,
                "h": bh,
                "hue": median_h,
                "action_type": action_type,
                "icon_image": icon_image,
            }
        )

    icons.sort(key=lambda item: (item["abs_y"], item["abs_x"]))
    return icons


def _identify_notebook_icons_with_clip(
    app: "AppProcessor",
    icons: list[dict[str, Any]],
) -> None:
    """尝试用 CLIP 识别每个图标，未命中则保留色相分类并缓存图像。"""
    from . import _get_schedule_clip

    schedule_clip = _get_schedule_clip(app)

    for icon in icons:
        icon_img = icon.get("icon_image")
        if icon_img is None or icon_img.size == 0:
            icon["source"] = "hue"
            continue

        if schedule_clip is not None:
            try:
                matched = schedule_clip.retrieve(icon_img, similarity_threshold=0.90)
                if matched is not None:
                    icon["action_type"] = matched.action_id
                    icon["source"] = "clip"
                    icon.setdefault("metadata", {})["clip_action_id"] = matched.action_id
                    icon.setdefault("metadata", {})["param_kind"] = matched.param_kind
                    icon.setdefault("metadata", {})["rl_action_type"] = matched.rl_action_type
                    logger.debug(
                        "P手帳 CLIP: 命中 {} (H={:.0f})",
                        matched.action_id,
                        icon["hue"],
                    )
                    continue
            except Exception:  # noqa: BLE001
                pass

        icon["source"] = "hue"
        hue_label = icon["action_type"]
        if hue_label not in _notebook_icon_clip_cache:
            _notebook_icon_clip_cache[hue_label] = icon_img.copy()
            logger.debug(
                "P手帳: 缓存未识别图标 '{}' (H={:.0f}) 供后续 CLIP 学习",
                hue_label,
                icon["hue"],
            )


def _retry_cached_notebook_icons(app: "AppProcessor") -> dict[str, str]:
    """对缓存中的未识别图标重新尝试 CLIP 识别。"""
    from . import _get_schedule_clip

    schedule_clip = _get_schedule_clip(app)
    if schedule_clip is None or not _notebook_icon_clip_cache:
        return {}

    resolved: dict[str, str] = {}
    to_remove: list[str] = []

    for hue_label, icon_img in _notebook_icon_clip_cache.items():
        try:
            matched = schedule_clip.retrieve(icon_img, similarity_threshold=0.90)
            if matched is not None:
                resolved[hue_label] = matched.action_id
                to_remove.append(hue_label)
                logger.debug(
                    "P手帳 CLIP重试: '{}' → '{}'",
                    hue_label,
                    matched.action_id,
                )
        except Exception:  # noqa: BLE001
            pass

    for key in to_remove:
        del _notebook_icon_clip_cache[key]

    return resolved


def _group_icons_into_rows(icons: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """将图标按 Y 坐标分组为行（每行对应一个周）。"""
    if not icons:
        return []
    rows: list[list[dict[str, Any]]] = []
    current_row: list[dict[str, Any]] = [icons[0]]
    for icon in icons[1:]:
        if icon["abs_y"] - current_row[-1]["abs_y"] > _ICON_ROW_GAP:
            rows.append(current_row)
            current_row = [icon]
        else:
            current_row.append(icon)
    rows.append(current_row)
    return rows


def _detect_p_notebook_button(app: "AppProcessor"):
    """检测 P手帳 按钮（YOLO 标签 PC_P_MANUAL）。"""
    results = getattr(app, "latest_results", None)
    if results is None:
        return None
    manual_boxes = results.filter_by_label(ProducerLabels.PC_P_MANUAL)
    if not manual_boxes:
        return None
    button = manual_boxes.first()
    logger.debug(
        "P手帳: 检测到按钮 cx={}, cy={}",
        getattr(button, "cx", 0),
        getattr(button, "cy", 0),
    )
    return button


def _detect_p_notebook_close_button(app: "AppProcessor"):
    """检测 P手帳 面板关闭按钮（优先使用 YOLO 的通用关闭按钮标签）。"""
    results = getattr(app, "latest_results", None)
    if results is None:
        return None
    close_boxes = list(results.filter_by_label(BaseUILabels.CLOSE_BUTTON))
    if not close_boxes:
        return None
    frame = getattr(results, "frame", None)
    frame_height = int(frame.shape[0]) if frame is not None else 0
    if frame_height > 0:
        close_boxes = [
            box for box in close_boxes
            if int(getattr(box, "cy", 0)) >= int(frame_height * 0.65)
        ] or close_boxes
    button = max(close_boxes, key=lambda box: int(getattr(box, "cy", 0)))
    logger.debug(
        "P手帳: 检测到关闭按钮 cx={}, cy={}",
        getattr(button, "cx", 0),
        getattr(button, "cy", 0),
    )
    return button


def _open_p_notebook(app: "AppProcessor") -> bool:
    """打开p、手账并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    button = _detect_p_notebook_button(app)
    if button is None:
        logger.debug("P手帳: 未检测到按钮，跳过")
        return False
    app.device.click_element(button)
    logger.debug("P手帳: 已点击打开按钮")
    from time import sleep

    sleep(1.2)
    app.game_utils.wait_frame_stable(stable_count=2, timeout=3.0)
    return True


def _close_p_notebook(app: "AppProcessor", *, allow_fallback: bool = True) -> bool:
    """关闭p、手账并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        allow_fallback: 用于提供allow、fallback相关输入。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    from time import sleep

    close_button = _detect_p_notebook_close_button(app)
    if close_button is not None:
        sleep(0.3)
        app.device.click_element(close_button)
        logger.debug("P手帳: 点击检测到的关闭按钮")
        sleep(0.8)
        return True

    if not allow_fallback:
        logger.debug("P手帳: 未检测到关闭按钮，且已禁用坐标回退")
        return False

    from src.core.tasks.producer_challenge.shared.common import click_relative_point

    sleep(0.3)
    click_relative_point(app, x_ratio=0.50, y_ratio=0.944, label="p-notebook-close-x")
    logger.debug("P手帳: 未检测到关闭按钮，回退点击 × 区域")
    sleep(0.8)
    return True


def _read_notebook_schedule_page(app: "AppProcessor") -> list[dict[str, Any]]:
    """读取 P手帳 当前可见页面，结合 OCR + 颜色检测。"""
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return []

    height, width = frame.shape[:2]

    ocr_left = int(width * 0.04)
    ocr_right = int(width * 0.25)
    ocr_top = int(height * 0.18)
    ocr_bottom = int(height * 0.90)
    left_crop = frame[ocr_top:ocr_bottom, ocr_left:ocr_right]

    week_labels: list[dict[str, Any]] = []
    try:
        ocr_left_results = _NOTEBOOK_OCR.ocr(left_crop)
        merged_left = ocr_left_results.auto_merge_lines(
            cy_range=max(4, int(left_crop.shape[0] * 0.015)),
            width_gap=max(10, int(left_crop.shape[1] * 0.05)),
        )
        for line in merged_left:
            text = normalize_ocr_jp(getattr(line, "text", "")).strip()
            match = None
            if text:
                import re

                match = re.search(rf"(\d+)\s*[{ProduceText.WEEK}周]", text)
            if not match:
                cleaned = "".join(ch for ch in text if ch.isdigit())
                if cleaned and 1 <= int(cleaned) <= 20:
                    week_num = int(cleaned)
                else:
                    week_num = None
            else:
                week_num = int(match.group(1))
            if week_num is not None:
                cy_abs = int(getattr(line, "cy", 0)) + ocr_top
                week_labels.append({"week": week_num, "cy": cy_abs})
    except Exception:  # noqa: BLE001
        pass

    if week_labels:
        logger.debug(
            "P手帳 OCR: 读取到周数标签 {} (共{}个)",
            [item["week"] for item in week_labels],
            len(week_labels),
        )
    else:
        logger.debug("P手帳 OCR: 未读取到任何周数标签")

    center_left = int(width * 0.12)
    center_right = int(width * 0.78)
    center_crop = frame[ocr_top:ocr_bottom, center_left:center_right]

    special_events: list[dict[str, Any]] = []
    try:
        ocr_center_results = _NOTEBOOK_OCR.ocr(center_crop)
        merged_center = ocr_center_results.auto_merge_lines(
            cy_range=max(4, int(center_crop.shape[0] * 0.015)),
            width_gap=max(10, int(center_crop.shape[1] * 0.05)),
        )
        for line in merged_center:
            text = normalize_ocr_jp(getattr(line, "text", "")).strip()
            if len(text) < 2:
                continue
            if any(keyword in text for keyword in _NOTEBOOK_SPECIAL_KEYWORDS):
                cy_abs = int(getattr(line, "cy", 0)) + ocr_top
                special_events.append({"text": text, "cy": cy_abs})
    except Exception:  # noqa: BLE001
        pass

    icons = _detect_notebook_icons(frame)
    _identify_notebook_icons_with_clip(app, icons)
    icon_rows = _group_icons_into_rows(icons)

    logger.debug(
        "P手帳: 检测到 {} 个图标, 分为 {} 行 (周标签 {} 个, 特殊事件 {} 个)",
        len(icons),
        len(icon_rows),
        len(week_labels),
        len(special_events),
    )

    week_labels.sort(key=lambda item: item["cy"])
    entries: list[dict[str, Any]] = []
    used_icon_rows: set[int] = set()
    used_special: set[int] = set()

    special_weeks: dict[int, str] = {}
    for week_index, week_label in enumerate(week_labels):
        for special_index, special_event in enumerate(special_events):
            if special_index in used_special:
                continue
            if abs(special_event["cy"] - week_label["cy"]) < 200:
                special_weeks[week_index] = special_event["text"]
                used_special.add(special_index)
                break

    for week_index, week_label in enumerate(week_labels):
        week_num = week_label["week"]
        week_cy = week_label["cy"]

        if week_index in special_weeks:
            special_text = special_weeks[week_index]
            entries.append(
                {
                    "week": week_num,
                    "raw_text": f"{week_num}{ProduceText.WEEK}: {special_text}",
                    "actions": [],
                    "special_event": special_text,
                    "completed": False,
                    "is_action": True,
                    "is_week_label": True,
                }
            )
            continue

        actions: list[str] = []
        best_row_idx = -1
        best_row_dist = 999999
        for row_index, row in enumerate(icon_rows):
            if row_index in used_icon_rows:
                continue
            row_cy = int(np.mean([icon["abs_y"] for icon in row]))
            distance = abs(row_cy - week_cy)
            if distance < best_row_dist and distance < 200:
                best_row_dist = distance
                best_row_idx = row_index
        if best_row_idx >= 0:
            used_icon_rows.add(best_row_idx)
            for icon in icon_rows[best_row_idx]:
                actions.append(icon["action_type"])

        raw_text = (
            f"{week_num}{ProduceText.WEEK}: {' / '.join(actions)}"
            if actions
            else f"{week_num}{ProduceText.WEEK}"
        )
        entries.append(
            {
                "week": week_num,
                "raw_text": raw_text,
                "actions": actions,
                "special_event": None,
                "completed": False,
                "is_action": True,
                "is_week_label": True,
            }
        )

    for special_index, special_event in enumerate(special_events):
        if special_index not in used_special:
            entries.append(
                {
                    "week": 0,
                    "raw_text": special_event["text"],
                    "actions": [],
                    "special_event": special_event["text"],
                    "completed": False,
                    "is_action": True,
                    "is_week_label": False,
                }
            )

    orphan_rows = [index for index in range(len(icon_rows)) if index not in used_icon_rows]
    if orphan_rows and len(entries) >= 2:
        anchors: list[tuple[int, int]] = []
        for entry in entries:
            if entry.get("is_week_label") and entry.get("week", 0) > 0:
                for week_label in week_labels:
                    if week_label["week"] == entry["week"]:
                        anchors.append((week_label["cy"], entry["week"]))
                        break

        if len(anchors) >= 2:
            anchors.sort(key=lambda item: item[0])
            for row_index in orphan_rows:
                row = icon_rows[row_index]
                row_cy = int(np.mean([icon["abs_y"] for icon in row]))
                inferred_week = None

                if row_cy <= anchors[0][0]:
                    cy_diff = anchors[1][0] - anchors[0][0]
                    week_diff = anchors[0][1] - anchors[1][1]
                    if cy_diff > 0 and week_diff != 0:
                        steps = (anchors[0][0] - row_cy) / cy_diff
                        inferred_week = anchors[0][1] + round(steps * week_diff)
                elif row_cy >= anchors[-1][0]:
                    cy_diff = anchors[-1][0] - anchors[-2][0]
                    week_diff = anchors[-2][1] - anchors[-1][1]
                    if cy_diff > 0 and week_diff != 0:
                        steps = (row_cy - anchors[-1][0]) / cy_diff
                        inferred_week = anchors[-1][1] - round(steps * week_diff)
                else:
                    for anchor_index in range(len(anchors) - 1):
                        anchor_above, anchor_below = anchors[anchor_index], anchors[anchor_index + 1]
                        if anchor_above[0] <= row_cy <= anchor_below[0]:
                            span_cy = anchor_below[0] - anchor_above[0]
                            span_week = anchor_above[1] - anchor_below[1]
                            if span_cy > 0 and span_week > 0:
                                ratio = (row_cy - anchor_above[0]) / span_cy
                                inferred_week = anchor_above[1] - round(ratio * span_week)
                            break

                if inferred_week is not None and 1 <= inferred_week <= 20:
                    existing_weeks = {entry.get("week", 0) for entry in entries}
                    if inferred_week not in existing_weeks:
                        actions = [icon["action_type"] for icon in row]
                        entries.append(
                            {
                                "week": inferred_week,
                                "raw_text": f"{inferred_week}{ProduceText.WEEK}: {' / '.join(actions)}",
                                "actions": actions,
                                "special_event": None,
                                "completed": False,
                                "is_action": True,
                                "is_week_label": True,
                            }
                        )
                        logger.debug(
                            "P手帳: 推断未匹配图标行 → {}週 (cy={}, actions={})",
                            inferred_week,
                            row_cy,
                            actions,
                        )

    if orphan_rows:
        logger.debug(
            "P手帳: {} 个图标行未匹配到周标签 (已推断恢复)",
            len(orphan_rows),
        )

    entries.sort(key=lambda entry: entry.get("week", 0), reverse=True)

    debugger = getattr(app, "debug_tools", None)
    if debugger is not None and icons:
        for icon in icons:
            debugger.add_rect(
                icon["abs_x"],
                icon["abs_y"],
                icon["abs_x"] + icon["w"],
                icon["abs_y"] + icon["h"],
                color=(0, 255, 0),
                label=icon["action_type"][:6],
                duration=3.0,
            )

    return entries


def _notebook_scroll_and_check(
    app: "AppProcessor",
    start_y_ratio: float,
    end_y_ratio: float,
) -> bool:
    """P手帳 内执行一次滑动并检测是否发生了实际滚动。"""
    from src.core.tasks.producer_challenge.shared.common import get_frame_size

    width, height = get_frame_size(app)
    center_x = width // 2

    app.device.swipe(
        center_x,
        int(height * start_y_ratio),
        center_x,
        int(height * end_y_ratio),
        duration=0.4,
    )
    is_stable = app.game_utils.wait_frame_stable(
        threshold=0.985,
        stable_count=2,
        timeout=3.0,
    )
    if is_stable:
        return False
    app.game_utils.wait_frame_stable(threshold=0.985, stable_count=2, timeout=3.0)
    return True


def _scroll_notebook_up(app: "AppProcessor") -> bool:
    """P手帳 内向下拖动，内容向上卷，显示更高周数（未来）。"""
    return _notebook_scroll_and_check(app, 0.35, 0.65)


def _scroll_notebook_down(app: "AppProcessor") -> bool:
    """P手帳 内向上拖动，内容向下卷，显示更低周数（过去）。"""
    return _notebook_scroll_and_check(app, 0.65, 0.35)


def _scroll_notebook_to_bottom(app: "AppProcessor") -> None:
    """快速滑动到 P手帳 底部（最早的周）。"""
    for index in range(10):
        scrolled = _scroll_notebook_down(app)
        if not scrolled:
            logger.debug("P手帳: 已到达底部 (第{}次滑动)", index + 1)
            break


def read_p_notebook(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    max_scroll_pages: int = 5,
) -> list[dict[str, Any]]:
    """完整流程: 打开 P手帳 → 滑到底部 → 逐页向上读取 → 关闭。"""
    cache_key = f"p_notebook_week_{ctx.current_week}"
    cached = ctx.handler_state.get(cache_key)
    if cached is not None:
        logger.debug("P手帳: 第{}周已缓存，跳过读取", ctx.current_week)
        return cached

    if not _open_p_notebook(app):
        ctx.handler_state[cache_key] = []
        return []

    all_entries: dict[int, dict[str, Any]] = {}

    try:
        _scroll_notebook_to_bottom(app)

        page_entries = _read_notebook_schedule_page(app)
        for entry in page_entries:
            key = entry.get("week", 0)
            if key not in all_entries:
                all_entries[key] = entry
        logger.debug("P手帳: 底部首页读取 {} 周", len(page_entries))

        for page_index in range(max_scroll_pages):
            scrolled = _scroll_notebook_up(app)
            if not scrolled:
                logger.debug("P手帳: 已到顶，停止滚动")
                break
            new_entries = _read_notebook_schedule_page(app)
            added = 0
            for entry in new_entries:
                key = entry.get("week", 0)
                if key not in all_entries:
                    all_entries[key] = entry
                    added += 1
            logger.debug("P手帳: 向上滚动第{}页，新增 {} 周", page_index + 1, added)
    finally:
        _close_p_notebook(app)

    entries = sorted(all_entries.values(), key=lambda entry: entry.get("week", 0), reverse=True)
    logger.debug("P手帳: 最终 {} 个日程条目", len(entries))

    debugger = getattr(app, "debug_tools", None)
    if debugger is not None and entries:
        from src.core.tasks.producer_challenge.shared.common import get_frame_size

        width, _height = get_frame_size(app)
        parts = [entry["raw_text"][:16] for entry in entries[:5]]
        debugger.add_text(
            int(width * 0.02),
            20,
            f"P手帳: {' | '.join(parts)}",
            color=(100, 255, 200),
            font_size=12,
            duration=5.0,
        )

    ctx.handler_state[cache_key] = entries
    ctx.handler_state["p_notebook_schedule"] = entries
    ctx.handler_state["p_notebook_week"] = ctx.current_week

    ctx.record_operation(
        "read_p_notebook",
        target=f"week_{ctx.current_week}",
        details={
            "entry_count": len(entries),
            "scroll_pages": max_scroll_pages,
        },
    )
    return entries
