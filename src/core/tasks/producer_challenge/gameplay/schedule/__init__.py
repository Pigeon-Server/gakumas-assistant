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
    first_matching_index,
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
    _detect_lesson_stat_from_info,
    _extract_lesson_stamina_cost,
    _collect_lesson_option_candidates,
    _lesson_debugger,
    _ocr_lesson_stamina_costs,
    _probe_lesson_options,
    _read_lesson_info_panel,
)
from .notebook import (
    _NOTEBOOK_ICON_HUE_MAP,
    _NOTEBOOK_SPECIAL_KEYWORDS,
    _classify_icon_hue,
    _close_p_notebook,
    _detect_notebook_icons,
    _detect_p_notebook_button,
    _group_icons_into_rows,
    _identify_notebook_icons_with_clip,
    _notebook_scroll_and_check,
    _open_p_notebook,
    _read_notebook_schedule_page,
    _retry_cached_notebook_icons,
    _scroll_notebook_down,
    _scroll_notebook_to_bottom,
    _scroll_notebook_up,
    read_p_notebook,
)
from .recovery import (
    _annotate_low_stamina_recovery_preference,
    _schedule_payload_family,
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


def _normalize_schedule_text(text: str | None) -> str:
    return normalize_ocr_jp(str(text or "")).strip()


def _is_unknown_schedule_action_id(action_id: str | None) -> bool:
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


def _detect_recommended_kind(app: "AppProcessor") -> str:
    recommend_boxes = app.latest_results.filter_by_label(ProducerLabels.PC_RECOMMEND_ACTION)
    if not recommend_boxes:
        return "unknown"
    return infer_param_kind(ocr_text(recommend_boxes.first().frame))


def _looks_like_present_support_line(text: str) -> bool:
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
    if position == GameplayPosition.SCHEDULE_PRESENT_SUPPORT:
        return _collect_present_support_candidates(app, ctx)

    if position in (
        GameplayPosition.SCHEDULE_LESSON_OPTIONS,
        GameplayPosition.SCHEDULE_LESSON_SELECTED,
    ):
        return _collect_lesson_option_candidates(app, ctx, position=position)

    action_boxes = _collect_schedule_action_boxes(app)
    recommended_kind = _detect_recommended_kind(app)
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
                    recommended=param_kind == recommended_kind and param_kind != "unknown",
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
                recommended=kind == recommended_kind and kind != "unknown",
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
        recommended_index = first_matching_index(candidates, kind=_detect_recommended_kind(app))
        if recommended_index is not None:
            fallback_index = recommended_index
            fallback_reason = "推荐类型"

    if fallback_index is None:
        for index, candidate in enumerate(candidates):
            if candidate.recommended:
                fallback_index = index
                fallback_reason = "推荐候选"
                break

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
    """等待并确认「休み確認」模态框。"""
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
    candidates = collect_schedule_action_candidates(app, ctx, position=position)
    if not candidates:
        return None

    target_index = decide_schedule_action(app, ctx, candidates, position=position)
    target = candidates[target_index]

    logger.debug(
        "schedule step: position={}, target_index={}, title={!r}, kind={}, recommended={}",
        position,
        target_index,
        target.title,
        target.kind,
        target.recommended,
    )

    app.device.click_element(target.box)

    if target.metadata.get("is_vacation"):
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
                "recommended": target.recommended,
                "action_id": target.action_id,
                "db_id": target.db_id,
            },
        )
        return ScheduleStepResult(status="confirmed", candidate=target)

    ctx.pending_schedule_index = target.index
    ctx.pending_schedule_label = (
        target.title or target.kind or target.action_id or f"action_{target.index + 1}"
    )
    ctx.handler_state["pending_schedule_action_id"] = str(target.action_id or "").strip()
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
        return phase == "schedule"

    def handle(self, app, ctx, phase, position):
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
            cache_key = f"p_notebook_week_{ctx.current_week}"
            if ctx.handler_state.get(cache_key) is None:
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

        return HandlerResult.ok(f"schedule {result.status}", sleep_after=0.8)

    def __repr__(self):
        return f"<ScheduleHandler phase={self.phase_tag!r} priority={self.priority}>"
