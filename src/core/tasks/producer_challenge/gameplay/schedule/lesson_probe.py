from __future__ import annotations

import re
from time import sleep
from typing import TYPE_CHECKING, List

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.shared.common import ocr_text
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger
from src.utils.string_tools import normalize_ocr_jp

from ..decision import detect_sp_badge, hydrate_lesson_candidates
from .types import ScheduleActionCandidate

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


# ── 授業課程選項 ──
_LESSON_PROBE_TAP_WAIT = 0.5      # 点击选项后等待 UI 动画
_LESSON_PROBE_INFER_WAIT = 0.3    # 等待 YOLO 再推理
_LESSON_STAMINA_COST_RE = re.compile(r"[-ー一](\d+)")  # 体力消耗匹配: "-4", "ー8"
_LESSON_INFO_OCR = OCRService()

# 授業效果描述中的属性关键词 → param_kind 映射
_LESSON_STAT_KEYWORDS: dict[str, str] = {
    ProduceText.VOCAL: "vocal",
    ProduceText.DANCE: "dance",
    ProduceText.VISUAL: "visual",
}

_lesson_debugger = DebugTools()


def _detect_lesson_stat_from_info(text: str) -> str:
    """从授業信息面板 OCR 文本中提取属性类型。"""
    for keyword, param_kind in _LESSON_STAT_KEYWORDS.items():
        if keyword in text:
            return param_kind
    return "unknown"


def _extract_lesson_stamina_cost(text: str) -> int | None:
    """从授業选项上方的 OCR 文本中提取体力消耗值。"""
    match = _LESSON_STAMINA_COST_RE.search(text or "")
    return int(match.group(1)) if match else None


def _read_lesson_info_panel(app: "AppProcessor") -> str:
    """读取当前授業选项信息面板（PC_ACTION_INFO）的 OCR 文本。"""
    results = getattr(app, "latest_results", None)
    if results is None:
        return ""
    info_boxes = results.filter_by_label(ProducerLabels.PC_ACTION_INFO)
    if not info_boxes:
        return ""
    info_box = info_boxes.first()
    frame = getattr(info_box, "frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return ""
    try:
        ocr_results = _LESSON_INFO_OCR.ocr(frame)
        merged = ocr_results.auto_merge_lines(
            cy_range=max(4, int(frame.shape[0] * 0.015)),
            width_gap=max(10, int(frame.shape[1] * 0.02)),
        )
        lines = [
            normalize_ocr_jp(getattr(line, "text", "")).strip()
            for line in merged
            if len(normalize_ocr_jp(getattr(line, "text", "")).strip()) >= 2
        ]
        text = "；".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.debug("lesson info: 面板 OCR 失败: {}", exc)
        text = ""

    if text:
        _lesson_debugger.add_box(
            int(getattr(info_box, "x", 0)),
            int(getattr(info_box, "y", 0)),
            int(getattr(info_box, "w", 0)),
            int(getattr(info_box, "h", 0)),
            label=f"LessonInfo: {text[:40]}",
            color=(100, 255, 100),
            alpha=0.15,
            duration=3.0,
            font_size=14,
        )
    return text


def _ocr_lesson_stamina_costs(
    app: "AppProcessor",
    action_boxes: list,
) -> list[int | None]:
    """OCR 读取每个授業选项上方的体力消耗标签。"""
    frame = getattr(app, "latest_frame", None)
    costs: list[int | None] = []
    if frame is None or getattr(frame, "size", 0) <= 0:
        return [None] * len(action_boxes)

    for box in action_boxes:
        bx = int(getattr(box, "x", 0))
        by = int(getattr(box, "y", 0))
        bw = int(getattr(box, "w", 0)) - bx if hasattr(box, "w") else 0
        cost_h = max(40, int(by * 0.03))
        cost_y = max(0, by - cost_h)
        cost_w = max(bw, 80)
        if cost_h <= 5 or cost_w <= 5:
            costs.append(None)
            continue
        try:
            crop = frame[cost_y:by, bx:bx + cost_w]
            if crop.size <= 0:
                costs.append(None)
                continue
            text = ocr_text(crop)
            cost = _extract_lesson_stamina_cost(text)
            costs.append(cost)
            if cost is not None:
                _lesson_debugger.add_box(
                    bx,
                    cost_y,
                    bx + cost_w,
                    cost_y + cost_h,
                    label=f"{ProduceText.STAMINA}-{cost}",
                    color=(255, 80, 80),
                    thickness=2,
                    duration=3.0,
                )
        except Exception:  # noqa: BLE001
            costs.append(None)
    return costs


def _probe_lesson_options(
    app: "AppProcessor",
    candidates: list[ScheduleActionCandidate],
) -> None:
    """逐个点击授業选项，从信息面板读取效果描述 + 属性类型。"""
    if not candidates:
        return

    logger.info("lesson: 授業探査開始 — {} 個選項", len(candidates))

    first_info = _read_lesson_info_panel(app)
    if first_info:
        stat_kind = _detect_lesson_stat_from_info(first_info)
        candidates[0].metadata["lesson_effect"] = first_info
        candidates[0].metadata["lesson_stat"] = stat_kind
        if stat_kind != "unknown":
            candidates[0].kind = stat_kind
        logger.debug(
            "lesson: 選項 #0 効果(預選): stat={}, text={}",
            stat_kind,
            first_info[:60],
        )

    for index, candidate in enumerate(candidates):
        if candidate.metadata.get("lesson_effect"):
            continue

        try:
            app.device.click_element(candidate.box)
            sleep(_LESSON_PROBE_TAP_WAIT)
            sleep(_LESSON_PROBE_INFER_WAIT)

            info_text = _read_lesson_info_panel(app)
            if info_text:
                stat_kind = _detect_lesson_stat_from_info(info_text)
                candidate.metadata["lesson_effect"] = info_text
                candidate.metadata["lesson_stat"] = stat_kind
                if stat_kind != "unknown":
                    candidate.kind = stat_kind
                logger.debug(
                    "lesson: 選項 #{} 効果: stat={}, text={}",
                    index,
                    stat_kind,
                    info_text[:60],
                )
            else:
                logger.debug("lesson: 選項 #{} 信息面板未検出", index)

            _lesson_debugger.add_box(
                int(getattr(candidate.box, "x", 0)),
                int(getattr(candidate.box, "y", 0)),
                int(getattr(candidate.box, "w", 0)),
                int(getattr(candidate.box, "h", 0)),
                color=(0, 200, 100),
                thickness=2,
                duration=3,
                label=f"Lesson#{index} {candidate.kind}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("lesson: 選項 #{} 探査異常: {}", index, exc)

    probed = [candidate for candidate in candidates if candidate.metadata.get("lesson_effect")]
    logger.info(
        "lesson: 授業探査完了 — {}/{} 個取得効果描述",
        len(probed),
        len(candidates),
    )


def _collect_lesson_option_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> List[ScheduleActionCandidate]:
    """采集授業課程選項候选。"""
    from . import _collect_schedule_action_boxes

    action_boxes = _collect_schedule_action_boxes(app)
    if not action_boxes:
        return []

    selected_index = (
        ctx.pending_schedule_index
        if position == GameplayPosition.SCHEDULE_LESSON_SELECTED
        else None
    )

    stamina_costs = _ocr_lesson_stamina_costs(app, action_boxes)

    candidates: list[ScheduleActionCandidate] = []
    for index, box in enumerate(action_boxes):
        title = ocr_text(box.frame) if getattr(box, "frame", None) is not None else ""
        cost = stamina_costs[index] if index < len(stamina_costs) else None
        candidates.append(
            ScheduleActionCandidate(
                index=index,
                title=title,
                kind="unknown",
                recommended=False,
                selected=selected_index == index,
                box=box,
                metadata={
                    "stamina_cost": cost,
                    "lesson_option": True,
                },
            )
        )

    if position != GameplayPosition.SCHEDULE_LESSON_SELECTED:
        _probe_lesson_options(app, candidates)

    for candidate in candidates:
        if candidate.box and detect_sp_badge(candidate.box):
            candidate.metadata["is_sp"] = True
            logger.info("SP badge detected visually on lesson candidate #{}", candidate.index)

    hydrate_lesson_candidates(candidates)

    logger.info(
        "lesson: 采集完成 — {} 個候选: {}",
        len(candidates),
        [(candidate.index, candidate.kind, candidate.metadata.get("stamina_cost")) for candidate in candidates],
    )
    return candidates
