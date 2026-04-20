from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.utils.logger import logger

from .common import (
    first_matching_index,
    infer_param_kind,
    invoke_decision_strategy,
    ocr_text,
    resolve_candidate_index,
)
from .decision import build_decision_state, hydrate_schedule_candidates

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


@dataclass
class ScheduleActionCandidate:
    index: int
    title: str
    kind: str
    recommended: bool
    selected: bool
    box: Any = field(repr=False, default=None)
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleStepResult:
    status: str
    candidate: ScheduleActionCandidate


def _collect_schedule_action_boxes(app: "AppProcessor") -> list:
    """收集时间表动作候选框。

    优先使用 PC_ACTION 标签，若无则回退到 Universal Options。
    有些时间表画面（例如特殊事件周）使用 Options 而非 Action 标签。
    """
    actions = list(app.latest_results.filter_by_label(ProducerLabels.PC_ACTION))
    if not actions:
        actions = list(app.latest_results.filter_by_label(ProducerLabels.UNIVERSAL_OPTIONS))
    # 按垂直位置排序（选项通常纵向排列，cx 几乎相同）
    return sorted(actions, key=lambda item: item.cy)


def _detect_recommended_kind(app: "AppProcessor") -> str:
    recommend_boxes = app.latest_results.filter_by_label(ProducerLabels.PC_RECOMMEND_ACTION)
    if not recommend_boxes:
        return "unknown"
    return infer_param_kind(ocr_text(recommend_boxes.first().frame))


def collect_schedule_action_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> List[ScheduleActionCandidate]:
    action_boxes = _collect_schedule_action_boxes(app)
    recommended_kind = _detect_recommended_kind(app)
    selected_index = ctx.pending_schedule_index if position == "schedule_selected" else None

    candidates: list[ScheduleActionCandidate] = []
    for idx, box in enumerate(action_boxes):
        title = ocr_text(box.frame)
        kind = infer_param_kind(title)
        candidates.append(
            ScheduleActionCandidate(
                index=idx,
                title=title,
                kind=kind,
                recommended=kind == recommended_kind and kind != "unknown",
                selected=selected_index == idx,
                box=box,
            )
        )
    hydrate_schedule_candidates(candidates)
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
    decision = invoke_decision_strategy(
        ctx.schedule_strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        return resolve_candidate_index(decision, candidates)

    if ctx.pending_schedule_index is not None and 0 <= ctx.pending_schedule_index < len(candidates):
        return ctx.pending_schedule_index

    recommended_index = first_matching_index(candidates, kind=_detect_recommended_kind(app))
    if recommended_index is not None:
        return recommended_index

    for idx, candidate in enumerate(candidates):
        if candidate.recommended:
            return idx

    return 0


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
    if position == "schedule_selected":
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
    ctx.pending_schedule_label = target.title or target.kind or target.action_id or f"action_{target.index + 1}"
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


# ────────────────────────────────────────────────────────────
# Handler
# ────────────────────────────────────────────────────────────

class ScheduleHandler:
    """日程行动选择的 gameplay handler 包装。

    处理两类场景：
    1. 常规行程选择 — 委托给 execute_schedule_step()
    2. 行程事件对话（おでかけ等） — 选项交给 dialogue 逻辑，
       文本推进仅点击推进、绝不快进
    """

    phase_tag = "schedule"
    priority = 50

    # 行程事件相关位置集合
    _EVENT_POSITIONS = frozenset({
        "schedule_event_options",
        "schedule_event_dialogue",
    })

    def can_handle(self, app, ctx, phase, position):
        return phase == "schedule"

    def handle(self, app, ctx, phase, position):
        from src.core.tasks.producer_challenge.gameplay.handler_base import HandlerResult

        # ── 行程事件对话选项（おでかけ等の選択肢） ──
        if position == "schedule_event_options":
            from src.core.tasks.producer_challenge.gameplay.dialogue import (
                execute_dialogue_step,
            )
            result = execute_dialogue_step(app, ctx, position=position)
            if result is None:
                return HandlerResult.no_action("no dialogue options in schedule event")
            return HandlerResult.ok(f"schedule event {result.status}", sleep_after=0.6)

        # ── 行程事件对话文本推进（不快进） ──
        if position == "schedule_event_dialogue":
            from src.core.tasks.producer_challenge.gameplay.common import click_relative_point
            click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="schedule-event-advance")
            logger.debug("schedule: 行程事件对话推进（不快进）")
            return HandlerResult.ok("schedule event dialogue advance", sleep_after=0.6)

        # ── 常规行程选择 ──
        result = execute_schedule_step(app, ctx, position=position)
        if result is None:
            # 无候选行动（如活動支給の宝箱领取画面）——
            # 连续无候选时点击画面上方安全区域以推进（避免误触底栏按钮）
            no_action_key = "schedule_no_action_count"
            count = ctx.handler_state.get(no_action_key, 0) + 1
            ctx.handler_state[no_action_key] = count
            if count >= 2:
                from src.core.tasks.producer_challenge.gameplay.common import click_relative_point
                # 使用屏幕上方偏左位置（y=0.35），避免误触底栏的手牌库/P饮料按钮
                click_relative_point(app, x_ratio=0.5, y_ratio=0.35, label="schedule-idle-fallback-tap")
                logger.debug("schedule: 无候选行动，第{}次回退点击画面上方安全区域", count)
                return HandlerResult.ok("schedule idle fallback tap", sleep_after=0.8)
            return HandlerResult.no_action("no schedule actions found")

        # 找到候选项时重置无候选计数器
        ctx.handler_state.pop("schedule_no_action_count", None)

        if result.status == "confirmed":
            action_name = (
                result.candidate.title
                or result.candidate.kind
                or f"action_{result.candidate.index + 1}"
            )
            ctx.record_schedule_choice(action_name)

        return HandlerResult.ok(f"schedule {result.status}", sleep_after=0.8)

    def __repr__(self):
        return f"<ScheduleHandler phase={self.phase_tag!r} priority={self.priority}>"
