from __future__ import annotations

import re
from statistics import median
from time import sleep
from typing import TYPE_CHECKING, Any, List

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.shared.common import (
    infer_param_kind,
    invoke_decision_strategy,
    ocr_text,
    resolve_candidate_index,
)
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig, normalize_ocr_jp, string_match

from ..decision import (
    build_decision_state,
    detect_sp_badge,
    hydrate_schedule_candidates,
    resolve_schedule_action_identity,
)
from .lesson_probe import (
    _collect_lesson_option_candidates,
)
from .notebook import (
    _close_p_notebook,
    _detect_p_notebook_close_button,
    _retry_cached_notebook_icons,
    read_p_notebook,
)
from .recovery import (
    _annotate_low_stamina_recovery_preference,
    _select_low_stamina_recovery_action,
)
from .types import ScheduleActionCandidate, ScheduleStepResult

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


_SCHEDULE_SCREEN_OCR = OCRService()
_PRESENT_SUPPORT_BONUS_RE = re.compile(r"\+\d+")
_SCHEDULE_LOOKUP_NOISE_TOKENS = ProduceText.SCHEDULE_LOOKUP_NOISE_TOKENS
_ACTION_INFO_OCR = OCRService()
_SCHEDULE_PENDING_CLICK_COUNT_KEY = "pending_schedule_click_count"
_SCHEDULE_ACTION_DECISION_POSITIONS = frozenset({
    GameplayPosition.SCHEDULE_IDLE,
    GameplayPosition.SCHEDULE_RECOMMEND,
})


def _detect_recommended_kind(app: "AppProcessor") -> str:
    """兼容旧测试入口，实际语义为周行动预览提示属性。"""
    return _detect_preview_hint_kind(app)


def _is_schedule_action_decision_position(position: str) -> bool:
    """判断当前 position 是否处于日程操作待决策状态。

    Args:
        position: 当前阶段下的细分画面位置标识。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    return position in _SCHEDULE_ACTION_DECISION_POSITIONS


def _schedule_notebook_mode(ctx: "ProduceContext") -> str:
    """处理日程`schedule_notebook_mode`。"""
    return str(getattr(ctx, "schedule_notebook_mode", "before_decision") or "before_decision").strip().lower()


def _has_decided_schedule_action_this_week(ctx: "ProduceContext") -> bool:
    """判断本周是否已完成日程操作决策。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    decided_week = ctx.handler_state.get("schedule_action_decided_week")
    try:
        return int(decided_week) == int(ctx.current_week)
    except (TypeError, ValueError):
        return False


def _mark_schedule_action_decided(ctx: "ProduceContext") -> None:
    """标记`mark_schedule_action_decided`。"""
    ctx.handler_state["schedule_action_decided_week"] = int(ctx.current_week)


def _get_pending_schedule_click_count(ctx: "ProduceContext") -> int:
    """读取当前待确认行程动作已经点击了几次。"""
    try:
        return max(0, int(ctx.handler_state.get(_SCHEDULE_PENDING_CLICK_COUNT_KEY) or 0))
    except (TypeError, ValueError):
        return 0


def _set_pending_schedule_target(ctx: "ProduceContext", candidate: "ScheduleActionCandidate", *, click_count: int = 0) -> None:
    """记录当前待确认的行程动作。"""
    ctx.pending_schedule_index = candidate.index
    ctx.pending_schedule_label = (
        candidate.title or candidate.kind or candidate.action_id or f"action_{candidate.index + 1}"
    )
    ctx.handler_state["pending_schedule_action_id"] = str(candidate.action_id or "").strip()
    ctx.handler_state[_SCHEDULE_PENDING_CLICK_COUNT_KEY] = max(0, int(click_count))


def _resolve_pending_schedule_target(
    candidates: list["ScheduleActionCandidate"],
    ctx: "ProduceContext",
) -> "ScheduleActionCandidate" | None:
    """根据上下文中的 pending 信息找回当前要确认的日程动作。"""
    pending_action_id = str(ctx.handler_state.get("pending_schedule_action_id") or "").strip()
    if pending_action_id:
        for candidate in candidates:
            if str(candidate.action_id or "").strip() == pending_action_id:
                return candidate

    pending_label = _normalize_schedule_text(ctx.pending_schedule_label)
    if pending_label:
        for candidate in candidates:
            if pending_label in {
                _normalize_schedule_text(candidate.title),
                _normalize_schedule_text(candidate.action_id),
            }:
                return candidate

    if ctx.pending_schedule_index is not None and 0 <= ctx.pending_schedule_index < len(candidates):
        return candidates[ctx.pending_schedule_index]

    return None


def _should_read_p_notebook_before_decision(
    ctx: "ProduceContext",
    *,
    position: str,
) -> bool:
    """判断决策前是否需要先读取 P 手账信息。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    if position != GameplayPosition.SCHEDULE_IDLE:
        return False
    if _schedule_notebook_mode(ctx) != "before_decision":
        return False
    if _has_decided_schedule_action_this_week(ctx):
        return False
    cache_key = f"p_notebook_week_{ctx.current_week}"
    return ctx.handler_state.get(cache_key) is None


def _ensure_p_notebook_closed_before_decision(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> bool:
    """处理ensure、p、手账、closed、before、decision并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    def _has_schedule_controls_visible() -> bool:
        """判断 has schedule controls visible 是否满足条件。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        results = getattr(app, "latest_results", None)
        if results is None:
            return False
        try:
            if len(list(results.filter_by_label(ProducerLabels.PC_ACTION))) > 0:
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            recommend_boxes = results.filter_by_label(ProducerLabels.PC_RECOMMEND_ACTION)
            return bool(recommend_boxes and len(recommend_boxes) > 0)
        except Exception:  # noqa: BLE001
            return False

    def _is_p_notebook_open_likely() -> bool:
        """判断 is p notebook open likely 是否满足条件。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        close_button = _detect_p_notebook_close_button(app)
        if close_button is None:
            return False
        # 常规 schedule 行动控件可见时，优先认为不是 P手帳 打开态，避免误拦截。
        if _has_schedule_controls_visible():
            return False
        return True

    if not _is_schedule_action_decision_position(position):
        return True
    if not _is_p_notebook_open_likely():
        return True

    logger.info("schedule: 自动决策前检测到 P手帳 可能打开，尝试关闭后再继续")
    closed = _close_p_notebook(app, allow_fallback=False)
    if not closed:
        logger.warning("schedule: 未检测到 P手帳 关闭按钮，跳过本轮自动决策")
        return False

    game_utils = getattr(app, "game_utils", None)
    wait_stable = getattr(game_utils, "wait_frame_stable", None)
    if callable(wait_stable):
        wait_stable(stable_count=2, timeout=2.0)
    if _is_p_notebook_open_likely():
        logger.warning("schedule: 关闭 P手帳 后仍疑似打开，跳过本轮自动决策")
        return False
    return True


def _normalize_schedule_text(text: str | None) -> str:
    """规范化`schedule_text`。"""
    return normalize_ocr_jp(str(text or "")).strip()


def _is_unknown_schedule_action_id(action_id: str | None) -> bool:
    """判断给定 action_id 是否属于未知日程操作。

    Args:
        action_id: 业务对象标识符，用于索引或匹配目标实体。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    normalized = str(action_id or "").strip()
    return not normalized or ":" in normalized or "unknown" in normalized


def _get_schedule_clip(app: "AppProcessor"):
    """获取周行动 CLIP 服务实例（若可用）。"""
    clip_manager = getattr(app, "clip_manager", None)
    if clip_manager is None:
        return None
    return getattr(clip_manager, "schedule_action_clip", None)


def _resolve_schedule_from_clip(
    app: "AppProcessor",
    box: Any,
) -> dict[str, Any] | None:
    """尝试使用 CLIP 记忆识别周行动图标。"""
    schedule_clip = _get_schedule_clip(app)
    if schedule_clip is None or box is None or getattr(box, "frame", None) is None:
        return None
    try:
        matched = schedule_clip.retrieve(box.frame)
    except Exception as exc:  # noqa: BLE001
        logger.debug("schedule CLIP: 识别失败，回退 OCR: {}", exc)
        return None
    if matched is None:
        return None
    logger.debug(
        "schedule CLIP: 命中 action_id={} kind={}",
        matched.action_id,
        matched.param_kind,
    )
    return {
        "action_id": matched.action_id,
        "param_kind": matched.param_kind,
        "rl_action_type": matched.rl_action_type,
    }


def _learn_schedule_clip(
    app: "AppProcessor",
    image: Any,
    action_id: str,
    *,
    param_kind: str = "",
    rl_action_type: str = "",
) -> None:
    """将已识别的周行动图标写入 CLIP 记忆库。"""
    if image is None or getattr(image, "size", 0) <= 0:
        return
    if not action_id or _is_unknown_schedule_action_id(action_id):
        return
    schedule_clip = _get_schedule_clip(app)
    if schedule_clip is None:
        return
    try:
        from src.core.services.clip.schedule_action import ScheduleActionIdentity

        payload = ScheduleActionIdentity(
            action_id=action_id,
            param_kind=param_kind or "",
            rl_action_type=rl_action_type or "",
        )
        schedule_clip.add_to_memory(image, payload, similarity_threshold=0.96)
    except Exception as exc:  # noqa: BLE001
        logger.debug("schedule CLIP: 学习失败 {}: {}", action_id, exc)


def _probe_action_info_panel(
    app: "AppProcessor",
    candidate: "ScheduleActionCandidate",
) -> str:
    """从 PC_ACTION_INFO 面板 OCR 读取当前候选的效果描述。"""
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
        ocr_results = _ACTION_INFO_OCR.ocr(frame)
        merged = ocr_results.auto_merge_lines(
            cy_range=max(4, int(frame.shape[0] * 0.015)),
            width_gap=max(10, int(frame.shape[1] * 0.02)),
        )
        lines = [
            normalize_ocr_jp(getattr(line, "text", "")).strip()
            for line in merged
            if len(normalize_ocr_jp(getattr(line, "text", "")).strip()) >= 2
        ]
        effect_lines = [
            line
            for line in lines
            if not any(
                token in line
                for token in (
                    candidate.title or "",
                    ProduceText.EXAM_CRITERIA,
                    ProduceText.PARAMETER_UP,
                )
            )
        ]
        text = "；".join(effect_lines) if effect_lines else "；".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.debug("schedule info: 面板 OCR 失败: {}", exc)
        text = ""

    debugger = getattr(app, "debug_tools", None)
    if debugger is not None and text:
        debugger.add_box(
            int(getattr(info_box, "x", 0)),
            int(getattr(info_box, "y", 0)),
            int(getattr(info_box, "w", 0)),
            int(getattr(info_box, "h", 0)),
            label=f"action_info: {text[:40]}",
            color=(50, 200, 255),
            alpha=0.15,
            duration=3.0,
            font_size=14,
        )

    if text:
        logger.debug(
            "schedule info: 候选[{}] '{}' 效果描述: {}",
            candidate.index,
            candidate.title,
            text[:80],
        )
    return text


def _schedule_title_resolution_score(text: str | None, *, index: int) -> float:
    """处理日程`schedule_title_resolution_score`。"""
    normalized = _normalize_schedule_text(text)
    if not normalized:
        return -100.0

    inferred_kind = infer_param_kind(normalized)
    resolution = resolve_schedule_action_identity(normalized, inferred_kind, index=index)
    action_id = str(getattr(resolution, "action_id", "") or "")
    metadata = dict(getattr(resolution, "metadata", {}) or {})

    score = min(len(normalized), 18) * 0.15
    if inferred_kind != "unknown":
        score += 6.0
    if not _is_unknown_schedule_action_id(action_id):
        score += 16.0
    if metadata.get("supported") is True:
        score += 2.0
    if metadata.get("rl_action_type"):
        score += 2.0
    if any(
        token and token in normalized
        for token in (
            ProduceText.OUTING,
            ProduceText.GO_OUT,
            ProduceText.CLASS,
            ProduceText.REST,
            ProduceText.BUSINESS,
            ProduceText.ACTIVITY,
            ProduceText.CONSULT,
            ProduceText.AUDITION,
            ProduceText.LESSON,
            ProduceText.SELF_LESSON,
            ProduceText.HARD_LESSON,
        )
    ):
        score += 4.0
    return score


def _choose_schedule_candidate_title(
    direct_title: str,
    lookup_texts: list[str],
    *,
    index: int,
) -> tuple[str, str]:
    """处理choose、日程、候选项、title并返回结果。

    Args:
        direct_title: 用于提供direct、title相关输入。
        lookup_texts: 用于提供lookup、texts相关输入。
        index: 用于提供index相关输入。

    Returns:
        tuple[str, str]: 返回值类型见注解。
    """
    normalized_direct = _normalize_schedule_text(direct_title)
    normalized_lookup = [
        text
        for text in (_normalize_schedule_text(value) for value in lookup_texts)
        if text
    ]
    if not normalized_direct and not normalized_lookup:
        return "", "direct"
    if not normalized_lookup:
        return normalized_direct, "direct"

    direct_score = _schedule_title_resolution_score(normalized_direct, index=index)
    best_lookup = max(
        normalized_lookup,
        key=lambda text: _schedule_title_resolution_score(text, index=index),
    )
    best_lookup_score = _schedule_title_resolution_score(best_lookup, index=index)

    if not normalized_direct:
        return best_lookup, "lookup"
    if best_lookup_score >= direct_score + 1.5:
        return best_lookup, "lookup"
    return normalized_direct, "direct"


def _collect_schedule_lookup_texts(app: "AppProcessor", action_boxes: list) -> list[list[str]]:
    """收集日程、lookup、texts并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        action_boxes: 用于提供操作、boxes相关输入。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0 or not action_boxes:
        return [[] for _ in action_boxes]

    height, _width = frame.shape[:2]
    merged_lines = _SCHEDULE_SCREEN_OCR.ocr(frame).auto_merge_lines(
        cy_range=max(6, int(height * 0.003)),
        width_gap=max(20, int(frame.shape[1] * 0.02)),
    )
    content_lines: list[tuple[object, str]] = []
    for line in merged_lines:
        text = _normalize_schedule_text(getattr(line, "text", ""))
        if not text or len(text) < 2:
            continue
        if not (height * 0.18 <= line.cy <= height * 0.82):
            continue
        if any(token in text for token in _SCHEDULE_LOOKUP_NOISE_TOKENS):
            continue
        content_lines.append((line, text))

    if not content_lines:
        return [[] for _ in action_boxes]

    lookup_texts: list[list[str]] = []
    for index, box in enumerate(action_boxes):
        box_width = max(1, int(getattr(box, "w", 0) - getattr(box, "x", 0)))
        box_height = max(1, int(getattr(box, "h", 0) - getattr(box, "y", 0)))
        top_boundary = max(0, int(getattr(box, "y", 0) - box_height * 0.35))
        bottom_boundary = min(height, int(getattr(box, "h", 0) + box_height * 0.45))
        if index > 0:
            top_boundary = max(top_boundary, int((action_boxes[index - 1].cy + box.cy) / 2))
        if index < len(action_boxes) - 1:
            bottom_boundary = min(bottom_boundary, int((box.cy + action_boxes[index + 1].cy) / 2))

        candidate_rows: list[tuple[float, str]] = []
        for line, text in content_lines:
            if not (top_boundary <= line.cy <= bottom_boundary):
                continue
            if not (
                int(getattr(box, "x", 0) - box_width * 0.25)
                <= line.cx
                <= int(getattr(box, "w", 0) + box_width * 0.25)
            ):
                continue
            vertical_gap = abs(float(line.cy) - float(box.cy))
            horizontal_gap = abs(float(line.cx) - float(box.cx))
            score = (
                vertical_gap * 2.0
                + horizontal_gap * 0.35
                - _schedule_title_resolution_score(text, index=index)
            )
            candidate_rows.append((score, text))

        candidate_rows.sort(key=lambda item: item[0])
        deduped: list[str] = []
        for _score, text in candidate_rows:
            if text not in deduped:
                deduped.append(text)
        lookup_texts.append(deduped)

    return lookup_texts


def _collect_schedule_action_boxes(app: "AppProcessor") -> list:
    """收集时间表动作候选框（含休む / PC_VACATION）。"""
    actions = list(app.latest_results.filter_by_label(ProducerLabels.PC_ACTION))
    if not actions:
        actions = list(app.latest_results.filter_by_label(ProducerLabels.UNIVERSAL_OPTIONS))
    vacation_boxes = list(app.latest_results.filter_by_label(ProducerLabels.PC_VACATION))
    all_boxes = actions + vacation_boxes
    if not all_boxes:
        return []

    box_heights = [
        max(1, int(getattr(item, "h", 0) - getattr(item, "y", 0)))
        for item in all_boxes
    ]
    row_threshold = max(8.0, min(48.0, float(median(box_heights)) * 0.45))
    sorted_by_cy = sorted(all_boxes, key=lambda item: float(getattr(item, "cy", 0)))

    rows: list[dict[str, Any]] = []
    for box in sorted_by_cy:
        cy = float(getattr(box, "cy", 0))
        matched_row: dict[str, Any] | None = None
        for row in rows:
            if abs(cy - float(row["anchor_cy"])) <= row_threshold:
                matched_row = row
                break
        if matched_row is None:
            rows.append({"anchor_cy": cy, "boxes": [box]})
            continue
        matched_row["boxes"].append(box)
        matched_row["anchor_cy"] = float(
            sum(float(getattr(item, "cy", 0)) for item in matched_row["boxes"])
            / len(matched_row["boxes"])
        )

    rows.sort(key=lambda row: float(row["anchor_cy"]))
    ordered: list[Any] = []
    for row in rows:
        ordered.extend(
            sorted(
                row["boxes"],
                key=lambda item: (
                    float(getattr(item, "cx", 0)),
                    float(getattr(item, "cy", 0)),
                ),
            )
        )
    return ordered


def _detect_preview_hint_kind(app: "AppProcessor") -> str:
    """检测周行动预览提示对应的属性类型。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        str: 处理后的文本结果。
    """
    preview_boxes = app.latest_results.filter_by_label(ProducerLabels.PC_RECOMMEND_ACTION)
    if not preview_boxes:
        return "unknown"
    return infer_param_kind(ocr_text(preview_boxes.first().frame))


def _looks_like_present_support_line(text: str) -> bool:
    """判断文本是否像“差入れ/支援”类选项描述行。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    normalized = normalize_ocr_jp(str(text or ""))
    return bool(
        string_match(
            normalized,
            ProduceText.PRESENT_SUPPORT,
            MatchConfig(fuzz_threshold=60, normalize=True),
        )
    ) or (
        ProduceText.PRESENT_SELECTION in normalized
        or ProduceText.PRESENT_SELECTION_SHORT in normalized
    )


def _collect_present_support_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
) -> List[ScheduleActionCandidate]:
    """收集present、支援卡、候选项并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return []

    height = frame.shape[0]
    ocr_results = list(_SCHEDULE_SCREEN_OCR.ocr(frame))
    if not ocr_results:
        return []

    content_lines = [
        item
        for item in ocr_results
        if height * 0.28 <= item.cy <= height * 0.78
    ]
    header_lines = [item for item in content_lines if _looks_like_present_support_line(item.text)]
    bonus_lines = [
        item
        for item in content_lines
        if _PRESENT_SUPPORT_BONUS_RE.search(normalize_ocr_jp(item.text))
    ]
    bonus_lines.sort(key=lambda item: item.cy)

    candidates: list[ScheduleActionCandidate] = []
    for index, bonus_line in enumerate(bonus_lines):
        prefix = None
        for line in reversed(header_lines):
            if line.cy <= bonus_line.cy and abs(bonus_line.cy - line.cy) <= height * 0.08:
                prefix = line
                break

        title_parts = []
        if prefix is not None:
            title_parts.append(normalize_ocr_jp(prefix.text))
        title_parts.append(normalize_ocr_jp(bonus_line.text))
        title = "".join(part for part in title_parts if part).strip()
        if not title:
            title = normalize_ocr_jp(bonus_line.text).strip()

        candidates.append(
            ScheduleActionCandidate(
                index=index,
                title=title,
                kind=infer_param_kind(title),
                recommended=False,
                selected=False,
                box=bonus_line,
                action_id=f"schedule_present_support_option_{index}",
                source="ocr_present_support",
                confidence=1.0,
                metadata={
                    "candidate_type": "present_support",
                    "effect_text": title,
                },
            )
        )

    if candidates:
        logger.debug(
            "schedule present support: 检测到 {} 个候选项: {}",
            len(candidates),
            [candidate.title for candidate in candidates],
        )
    return candidates


def collect_schedule_action_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> List[ScheduleActionCandidate]:
    """收集日程、操作、候选项并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    if position == GameplayPosition.SCHEDULE_PRESENT_SUPPORT:
        return _collect_present_support_candidates(app, ctx)

    if position in (
        GameplayPosition.SCHEDULE_LESSON_OPTIONS,
        GameplayPosition.SCHEDULE_LESSON_SELECTED,
    ):
        return _collect_lesson_option_candidates(app, ctx, position=position)

    action_boxes = _collect_schedule_action_boxes(app)
    preview_hint_kind = _detect_preview_hint_kind(app)
    selected_index = ctx.pending_schedule_index if position == "schedule_selected" else None
    pending_action_id = (
        str(ctx.handler_state.get("pending_schedule_action_id") or "").strip()
        if position == "schedule_selected"
        else ""
    )
    pending_label = (
        _normalize_schedule_text(ctx.pending_schedule_label)
        if position == "schedule_selected"
        else ""
    )
    lookup_text_groups = _collect_schedule_lookup_texts(app, action_boxes)

    candidates: list[ScheduleActionCandidate] = []
    for index, box in enumerate(action_boxes):
        if getattr(box, "label", "") == ProducerLabels.PC_VACATION:
            candidates.append(
                ScheduleActionCandidate(
                    index=index,
                    title=ProduceText.REST_ACTION,
                    kind="refresh",
                    recommended=False,
                    selected=selected_index == index,
                    box=box,
                    action_id="schedule_action_refresh",
                    source="yolo_vacation",
                    confidence=1.0,
                    metadata={
                        "is_vacation": True,
                        "rl_action_type": "refresh",
                        "schedule_family": "refresh",
                    },
                )
            )
            logger.info("schedule: 检测到休む按钮 (PC_VACATION)，index={}", index)
            continue

        clip_result = _resolve_schedule_from_clip(app, box)
        if clip_result is not None:
            action_id = clip_result["action_id"]
            param_kind = clip_result["param_kind"] or "unknown"
            rl_action_type = clip_result["rl_action_type"] or ""
            candidates.append(
                ScheduleActionCandidate(
                    index=index,
                    title=action_id,
                    kind=param_kind,
                    recommended=param_kind == preview_hint_kind and param_kind != "unknown",
                    selected=selected_index == index,
                    box=box,
                    action_id=action_id,
                    source="clip",
                    confidence=1.0,
                    metadata={
                        "clip_match": True,
                        "rl_action_type": rl_action_type,
                    },
                )
            )
            continue

        direct_title = ocr_text(box.frame)
        lookup_texts = list(lookup_text_groups[index]) if index < len(lookup_text_groups) else []
        title, title_source = _choose_schedule_candidate_title(
            direct_title,
            lookup_texts,
            index=index,
        )
        kind = infer_param_kind(title)
        candidates.append(
            ScheduleActionCandidate(
                index=index,
                title=title,
                kind=kind,
                recommended=kind == preview_hint_kind and kind != "unknown",
                selected=selected_index == index,
                box=box,
                metadata={
                    "ocr_title": _normalize_schedule_text(direct_title),
                    "lookup_texts": lookup_texts,
                    "title_source": title_source,
                },
            )
        )

    for candidate in candidates:
        if candidate.box and detect_sp_badge(candidate.box):
            candidate.metadata["is_sp"] = True
            logger.info("SP badge detected visually on candidate #{}", candidate.index)

    hydrate_schedule_candidates(candidates)

    if position == "schedule_selected":
        selected_candidate: ScheduleActionCandidate | None = None
        if pending_action_id:
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if str(getattr(candidate, "action_id", "") or "").strip() == pending_action_id
                ),
                None,
            )
        if selected_candidate is None and pending_label:
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if pending_label
                    in {
                        _normalize_schedule_text(getattr(candidate, "title", "")),
                        _normalize_schedule_text(getattr(candidate, "action_id", "")),
                    }
                ),
                None,
            )
        if selected_candidate is None and selected_index is not None and 0 <= selected_index < len(candidates):
            selected_candidate = candidates[selected_index]
        for candidate in candidates:
            candidate.selected = False
        if selected_candidate is not None:
            selected_candidate.selected = True
            ctx.pending_schedule_index = selected_candidate.index

    learned_new = False
    for candidate in candidates:
        if candidate.metadata.get("clip_match"):
            continue
        if _is_unknown_schedule_action_id(candidate.action_id):
            continue
        image = getattr(candidate.box, "frame", None) if candidate.box else None
        _learn_schedule_clip(
            app,
            image,
            candidate.action_id,
            param_kind=candidate.kind or "",
            rl_action_type=candidate.metadata.get("rl_action_type", ""),
        )
        learned_new = True

    if learned_new:
        _retry_cached_notebook_icons(app)

    if position == "schedule_selected":
        selected_candidate = next((candidate for candidate in candidates if candidate.selected), None)
        if selected_candidate is not None:
            effect_text = _probe_action_info_panel(app, selected_candidate)
            if effect_text:
                selected_candidate.metadata["effect_text"] = effect_text

    return candidates


def decide_schedule_action(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: List[ScheduleActionCandidate],
    *,
    position: str,
) -> int:
    """决策日程、操作并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        candidates: 候选项列表，供策略或规则选择目标动作。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        int: 计算得到的数值结果。
    """
    decision_state = build_decision_state(
        app,
        ctx,
        phase="schedule",
        position=position,
        candidates=candidates,
        reason="schedule_decision",
    )
    recovery_plan = None
    if position != GameplayPosition.SCHEDULE_SELECTED:
        recovery_plan = _select_low_stamina_recovery_action(decision_state)
        if recovery_plan is not None:
            _annotate_low_stamina_recovery_preference(
                decision_state,
                preferred_index=recovery_plan[0],
                reason=recovery_plan[1],
            )
    decision = invoke_decision_strategy(
        ctx.schedule_strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        resolved_index = resolve_candidate_index(decision, candidates)
        if recovery_plan is not None and resolved_index != recovery_plan[0]:
            logger.info(
                "schedule: 体力偏低，覆盖原决策 {} -> {} ({})",
                resolved_index,
                recovery_plan[0],
                recovery_plan[1],
            )
            from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper

            DecisionDumper.get_instance().update_last_resolved(
                resolved_index=recovery_plan[0],
                resolved_name=getattr(candidates[recovery_plan[0]], "title", "")
                if recovery_plan[0] < len(candidates)
                else "",
                fallback_used=True,
                fallback_reason=f"体力恢复覆盖({recovery_plan[1]})",
            )
            return recovery_plan[0]
        return resolved_index

    from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper

    fallback_index: int | None = None
    fallback_reason = ""

    if recovery_plan is not None:
        fallback_index = recovery_plan[0]
        fallback_reason = f"体力恢复({recovery_plan[1]})"

    if (
        fallback_index is None
        and ctx.pending_schedule_index is not None
        and 0 <= ctx.pending_schedule_index < len(candidates)
    ):
        fallback_index = ctx.pending_schedule_index
        fallback_reason = "pending 索引"

    if fallback_index is None:
        local_preference = dict(decision_state.get("local_preference", {}) or {})
        preferred_index = local_preference.get("index")
        if isinstance(preferred_index, int) and 0 <= preferred_index < len(candidates):
            fallback_index = preferred_index
            fallback_reason = str(local_preference.get("reason") or "本地偏好")

    if fallback_index is None:
        fallback_index = 0
        fallback_reason = "默认首选"

    resolved_name = ""
    if 0 <= fallback_index < len(candidates):
        resolved_name = getattr(candidates[fallback_index], "title", "")
    DecisionDumper.get_instance().update_last_resolved(
        resolved_index=fallback_index,
        resolved_name=resolved_name,
        fallback_used=True,
        fallback_reason=fallback_reason,
    )
    return fallback_index


def _confirm_vacation_modal(
    app: "AppProcessor",
    *,
    max_polls: int = 10,
    poll_interval: float = 0.5,
) -> bool:
    """处理confirm、vacation、弹窗并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        max_polls: 用于提供max、polls相关输入。
        poll_interval: 用于提供poll、interval相关输入。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    for attempt in range(max_polls):
        sleep(poll_interval)
        results = app.latest_results
        if results is None:
            logger.debug("schedule: 等待休み確認モーダル出现 ({}/{})", attempt + 1, max_polls)
            continue
        confirm_buttons = results.filter_by_label(ProducerLabels.CONFIRM_BUTTON)
        if confirm_buttons and len(confirm_buttons) > 0:
            app.device.click_element(confirm_buttons.first())
            logger.info("schedule: 休み確認モーダル — 点击确认按钮 (attempt={})", attempt + 1)
            return True
        logger.debug("schedule: 等待休み確認モーダル出现 ({}/{})", attempt + 1, max_polls)
    logger.warning("schedule: 休み確認モーダル未出现，超时退出")
    return False


def execute_schedule_step(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> ScheduleStepResult | None:
    """处理execute、日程、步骤并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        ScheduleStepResult | None: 返回值类型见注解。
    """
    candidates = collect_schedule_action_candidates(app, ctx, position=position)
    if not candidates:
        return None

    pending_target = _resolve_pending_schedule_target(candidates, ctx)
    pending_click_count = _get_pending_schedule_click_count(ctx)

    target_index: int
    target = pending_target
    if position == GameplayPosition.SCHEDULE_SELECTED:
        # 确认态应直接确认当前已选中的行动，避免再次触发策略决策。
        if target is None:
            selected_candidate = next((candidate for candidate in candidates if candidate.selected), None)
            if selected_candidate is not None:
                target = selected_candidate
                target_index = target.index
            elif (
                ctx.pending_schedule_index is not None
                and 0 <= ctx.pending_schedule_index < len(candidates)
            ):
                target_index = ctx.pending_schedule_index
                target = candidates[target_index]
                logger.debug(
                    "schedule: schedule_selected 未检测到 selected 标记，回退 pending 索引 {}",
                    target_index,
                )
            else:
                logger.warning("schedule: schedule_selected 缺少已选中候选，回退到策略决策")
                target_index = decide_schedule_action(app, ctx, candidates, position=position)
                target = candidates[target_index]
        else:
            target_index = target.index
        if target is not None:
            _set_pending_schedule_target(ctx, target, click_count=pending_click_count)
        if target is not None and target.box is not None and pending_click_count < 2:
            app.device.click_element(target.box)
            pending_click_count += 1
            ctx.handler_state[_SCHEDULE_PENDING_CLICK_COUNT_KEY] = pending_click_count
    else:
        if target is None:
            target_index = decide_schedule_action(app, ctx, candidates, position=position)
            target = candidates[target_index]
            _set_pending_schedule_target(ctx, target, click_count=0)
            pending_click_count = 0
        else:
            target_index = target.index
        if target.box is not None and target.metadata.get("is_vacation") is not True and pending_click_count < 2:
            app.device.click_element(target.box)
            pending_click_count += 1
            ctx.handler_state[_SCHEDULE_PENDING_CLICK_COUNT_KEY] = pending_click_count
        elif target.box is not None and target.metadata.get("is_vacation") is not True:
            logger.debug(
                "schedule: pending action 已完成双击，等待进入确认页 index={}",
                target.index,
            )
    if _is_schedule_action_decision_position(position):
        _mark_schedule_action_decided(ctx)

    logger.debug(
        "schedule step: position={}, target_index={}, title={!r}, kind={}, preview_hint_match={}",
        position,
        target_index,
        target.title,
        target.kind,
        target.recommended,
    )

    if target.metadata.get("is_vacation"):
        app.device.click_element(target.box)
        confirmed = _confirm_vacation_modal(app)
        ctx.record_operation(
            "confirm_vacation",
            target=ProduceText.REST_ACTION,
            details={
                "index": target.index,
                "modal_confirmed": confirmed,
            },
        )
        return ScheduleStepResult(status="confirmed", candidate=target)

    if position == GameplayPosition.SCHEDULE_PRESENT_SUPPORT:
        ctx.record_operation(
            "select_schedule_present_support",
            target=target.title or target.kind or target.action_id or f"option_{target.index + 1}",
            details={
                "index": target.index,
                "kind": target.kind,
                "action_id": target.action_id,
                "db_id": target.db_id,
                "source": target.source,
            },
        )
        return ScheduleStepResult(status="present_selected", candidate=target)

    if position == GameplayPosition.SCHEDULE_LESSON_SELECTED:
        ctx.record_operation(
            "confirm_lesson_option",
            target=target.title or target.kind or f"lesson_{target.index + 1}",
            details={
                "index": target.index,
                "kind": target.kind,
                "action_id": target.action_id,
                "stamina_cost": target.metadata.get("stamina_cost"),
            },
        )
        return ScheduleStepResult(status="confirmed", candidate=target)

    if position == GameplayPosition.SCHEDULE_LESSON_OPTIONS:
        ctx.pending_schedule_index = target.index
        ctx.pending_schedule_label = (
            target.title or target.kind or target.action_id or f"lesson_{target.index + 1}"
        )
        ctx.record_operation(
            "select_lesson_option",
            target=ctx.pending_schedule_label,
            details={
                "index": target.index,
                "kind": target.kind,
                "action_id": target.action_id,
                "stamina_cost": target.metadata.get("stamina_cost"),
                "lesson_effect": target.metadata.get("lesson_effect", "")[:80],
            },
        )
        return ScheduleStepResult(status="selected", candidate=target)

    if position == "schedule_selected":
        ctx.handler_state.pop("pending_schedule_action_id", None)
        ctx.record_operation(
            "confirm_schedule_action",
            target=target.title or target.kind or target.action_id or f"action_{target.index + 1}",
            details={
                "index": target.index,
                "kind": target.kind,
                "preview_hint_match": target.recommended,
                "action_id": target.action_id,
                "db_id": target.db_id,
            },
        )
        return ScheduleStepResult(status="confirmed", candidate=target)

    _set_pending_schedule_target(ctx, target, click_count=pending_click_count)
    ctx.record_operation(
        "select_schedule_action",
        target=ctx.pending_schedule_label,
        details={
            "index": target.index,
            "kind": target.kind,
            "recommended": target.recommended,
            "action_id": target.action_id,
            "db_id": target.db_id,
        },
    )
    return ScheduleStepResult(status="selected", candidate=target)


class ScheduleHandler:
    """日程行动选择的 gameplay handler 包装。"""

    phase_tag = "schedule"
    priority = 50

    _EVENT_POSITIONS = frozenset({
        "schedule_event_options",
        "schedule_event_dialogue",
    })

    def can_handle(self, app, ctx, phase, position):
        """判断当前画面是否应由该处理器接管。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        return phase == "schedule"

    def handle(self, app, ctx, phase, position):
        """执行处理器主逻辑并返回处理结果。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            返回执行结果对象，具体类型见函数注解。
        """
        from src.core.tasks.producer_challenge.gameplay.handler_base import HandlerResult

        if position in self._EVENT_POSITIONS:
            from src.core.tasks.producer_challenge.gameplay.dialogue import execute_dialogue_step

            result = execute_dialogue_step(app, ctx, position=position)
            if result is None:
                return HandlerResult.no_action(f"no dialogue action in schedule event ({position})")
            return HandlerResult.ok(f"schedule event {result.status}", sleep_after=0.6)

        if position == GameplayPosition.SCHEDULE_PRESENT_SUPPORT:
            result = execute_schedule_step(app, ctx, position=position)
            if result is None:
                return HandlerResult.no_action("no present support candidates found")
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "present_support_selection",
                "retry_limit": int(ctx.handler_state.get("present_support_unknown_retry_limit", 8) or 8),
                "retry_sleep": float(ctx.handler_state.get("present_support_unknown_retry_sleep", 0.8) or 0.8),
            }
            return HandlerResult.ok("schedule present support selected", sleep_after=0.8)

        if position == GameplayPosition.SCHEDULE_PRESENT_SUPPORT_SHOWCASE:
            from src.core.tasks.producer_challenge.shared.common import click_relative_point

            # 这里是活動支給奖励链里的展示页，点上方安全区域继续，避免误触底栏。
            click_relative_point(
                app,
                x_ratio=0.5,
                y_ratio=0.35,
                label="schedule-present-support-showcase",
            )
            ctx.record_operation(
                "advance_schedule_present_support_showcase",
                target=ProduceText.PRESENT_SUPPORT,
                position=position,
            )
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "present_support_showcase",
                "retry_limit": int(
                    ctx.handler_state.get("present_support_showcase_unknown_retry_limit", 6) or 6
                ),
                "retry_sleep": float(
                    ctx.handler_state.get("present_support_showcase_unknown_retry_sleep", 0.6) or 0.6
                ),
            }
            return HandlerResult.ok("schedule present support showcase advance", sleep_after=0.8)

        if position == GameplayPosition.SCHEDULE_LESSON_OPTIONS:
            result = execute_schedule_step(app, ctx, position=position)
            if result is None:
                return HandlerResult.no_action("no lesson option candidates found")
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "lesson_option_selection",
                "retry_limit": 6,
                "retry_sleep": 0.6,
            }
            return HandlerResult.ok(
                f"lesson option selected: {result.candidate.kind}",
                sleep_after=0.8,
            )

        if position == GameplayPosition.SCHEDULE_LESSON_SELECTED:
            result = execute_schedule_step(app, ctx, position=position)
            if result is None:
                return HandlerResult.no_action("no lesson selected candidate")
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "confirm_lesson_option",
                "retry_limit": 12,
                "retry_sleep": 0.7,
            }
            ctx.record_schedule_choice(
                result.candidate.title or result.candidate.kind or f"lesson_{result.candidate.index + 1}"
            )
            return HandlerResult.ok(
                f"lesson option confirmed: {result.candidate.kind}",
                sleep_after=0.8,
            )

        if position == GameplayPosition.SCHEDULE_IDLE:
            if not _ensure_p_notebook_closed_before_decision(app, ctx, position=position):
                return HandlerResult.no_action("p notebook not confirmed closed before decision")
            if _should_read_p_notebook_before_decision(ctx, position=position):
                cache_key = f"p_notebook_week_{ctx.current_week}"
                try:
                    notebook_entries = read_p_notebook(app, ctx, max_scroll_pages=2)
                    if notebook_entries:
                        logger.info(
                            "schedule: P手帳读取成功，第{}周，{}个日程条目",
                            ctx.current_week,
                            len(notebook_entries),
                        )
                        sleep(0.5)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("schedule: P手帳读取异常，跳过: {}", exc)
                    ctx.handler_state[cache_key] = []
                if not _ensure_p_notebook_closed_before_decision(app, ctx, position=position):
                    return HandlerResult.no_action("p notebook not closed after pre-decision read")
        elif _is_schedule_action_decision_position(position):
            if not _ensure_p_notebook_closed_before_decision(app, ctx, position=position):
                return HandlerResult.no_action("p notebook not confirmed closed before decision")

        result = execute_schedule_step(app, ctx, position=position)
        if result is None:
            no_action_key = "schedule_no_action_count"
            count = ctx.handler_state.get(no_action_key, 0) + 1
            ctx.handler_state[no_action_key] = count
            if count >= 2:
                from src.core.tasks.producer_challenge.shared.common import click_relative_point

                click_relative_point(app, x_ratio=0.5, y_ratio=0.35, label="schedule-idle-fallback-tap")
                logger.debug("schedule: 无候选行动，第{}次回退点击画面上方安全区域", count)
                return HandlerResult.ok("schedule idle fallback tap", sleep_after=0.8)
            return HandlerResult.no_action("no schedule actions found")

        ctx.handler_state.pop("schedule_no_action_count", None)

        if result.status == "confirmed":
            action_name = (
                result.candidate.title
                or result.candidate.kind
                or f"action_{result.candidate.index + 1}"
            )
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "confirm_schedule_action",
                "retry_limit": int(ctx.handler_state.get("schedule_confirm_unknown_retry_limit", 12) or 12),
                "retry_sleep": float(ctx.handler_state.get("schedule_confirm_unknown_retry_sleep", 0.7) or 0.7),
            }
            ctx.record_schedule_choice(action_name)
            ctx.handler_state.pop("schedule_action_decided_week", None)

        return HandlerResult.ok(f"schedule {result.status}", sleep_after=0.8)

    def __repr__(self):
        """处理repr并返回结果。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
        return f"<ScheduleHandler phase={self.phase_tag!r} priority={self.priority}>"
