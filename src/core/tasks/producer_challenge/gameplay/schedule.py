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
    """周行程选择画面中的单个行程候选项。

    每个行程选项（如课程、对话、咨询等）对应一个该类的实例，
    包含 OCR 识别的标题、参数类型、是否推荐等信息，供决策策略选择使用。

    Attributes:
        index: 候选项在画面中的序号，从上到下依次为 0, 1, 2。
        title: 行程选项的 OCR 识别文本，如「レッスン」或「相談」。
        kind: 参数类型标签，如 vocal、dance、visual、unknown 等。
        recommended: 是否为系统推荐行程，通过比对 PC_RECOMMEND_ACTION 检测框判断。
        selected: 用户是否已点击选中该项，仅在 schedule_selected 位置时为 True。
        box: YOLO 检测框对象，用于计算点击坐标和可视化调试。
        action_id: 标准化动作标识，用于决策层与执行层之间的关联。
        db_id: 数据库中该行程实体的主键 ID，为空表示尚未完成识别。
        source: 候选项数据来源，如 ocr、db、fallback 等。
        confidence: 识别或匹配的置信度分数，数值越高越可靠。
        metadata: 扩展元数据字典，保存决策辅助字段。
    """
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
    """单次行程选择步骤的执行结果。

    Attributes:
        status: 步骤状态，"selected" 表示已选中待确认，"confirmed" 表示已确认完成。
        candidate: 本步骤选中的行程候选项对象，包含标题、类型、索引等信息。
    """
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
    """检测画面中系统推荐行程的参数类型。

    通过 YOLO 检测 PC_RECOMMEND_ACTION 标签的按钮，对其裁剪区域做 OCR，
    再根据 OCR 文本推断参数类型（vocal/dance/visual/unknown）。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        str: 推荐行程的参数类型标识，未检测到推荐按钮时返回 "unknown"。
    """
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
    """收集周行程选择画面中的所有行程候选项。

    通过 YOLO 检测 PC_ACTION 或 UNIVERSAL_OPTIONS 标签获取候选框，
    对每个框做 OCR 识别标题并推断参数类型，同时标记推荐项和已选中项。
    最后调用 hydrate_schedule_candidates 补充 action_id 和 db_id 等字段。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识，"schedule_selected" 表示已选中待确认。

    Returns:
        List[ScheduleActionCandidate]: 行程候选项列表，按垂直位置从上到下排序。
    """
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
    """决策本周应该选择哪个行程。

    决策优先级：
    1. 外部注入的 schedule_strategy 回调（如 RL 策略），返回策略选中的索引。
    2. ctx.pending_schedule_index，即上一轮已选中待确认的行程索引。
    3. 与系统推荐行程参数类型匹配的候选项。
    4. 标记为 recommended 的第一个候选项。
    5. 兜底选择索引 0（第一个行程）。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        candidates: 候选项列表，供策略或规则选择目标动作。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        int: 选中的候选项在 candidates 列表中的索引。
    """
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
    """执行单次行程选择步骤：收集候选项、做出决策、点击目标行程。

    根据 position 区分两种行为：
    - schedule_selected：已经是确认页，点击后记录为 confirmed，更新周计数。
    - 其他：点击后记录为 selected，保存 pending 索引等待确认页出现。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        ScheduleStepResult | None: 包含状态和选中候选项的结果对象；
        未检测到候选框时返回 None。
    """
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
# 处理器
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
        """当当前画面阶段为 schedule 时返回 True，表示由该处理器接管行程选择。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            bool: phase 等于 "schedule" 时返回 True。
        """
        return phase == "schedule"

    def handle(self, app, ctx, phase, position):
        """处理行程选择与行程事件对话的主逻辑。

        根据 position 区分三种场景：
        - schedule_event_options：行程事件中的对话选项，委托给 dialogue 逻辑处理。
        - schedule_event_dialogue：行程事件中的对话文本推进，仅点击推进、不快进。
        - 其他：常规行程选择，委托给 execute_schedule_step。

        当连续多帧无候选行动时（如活动补给宝箱画面），回退点击画面上方安全区域推进。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            HandlerResult: 包含操作状态、描述和等待时间的结果对象。
        """
        from src.core.tasks.producer_challenge.gameplay.handler_base import HandlerResult

        # ── 行程事件对话选项（如外出等）──
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
            # 无候选行动（如活动补给宝箱领取画面）——
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
        """返回处理器的字符串表示，包含阶段标签和优先级，便于日志输出和调试。

        Returns:
            str: 格式为 `<ScheduleHandler phase='schedule' priority=50>` 的字符串。
        """
        return f"<ScheduleHandler phase={self.phase_tag!r} priority={self.priority}>"
