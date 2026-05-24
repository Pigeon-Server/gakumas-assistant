"""弹窗 handler。

在 gameplay 中出现弹窗的场景:
  - 技能卡使用确认 (lesson内)
  - 強化确认
  - 支援事件「戻す」（撤回效果）
  - P饮料详情页
  - 各类信息提示框

默认行为: 点击确认/OK 关闭。handler 优先级较高，
因为弹窗覆盖在其他阶段之上，必须优先处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import TYPE_CHECKING, Any

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.game.text.produce_text import ProduceText
from src.core.inference.ocr_engine import OCRService
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayHandler,
    HandlerResult,
)
from src.core.tasks.producer_challenge.shared.common import (
    invoke_decision_strategy,
    resolve_candidate_index,
)
from src.core.tasks.producer_challenge.gameplay.decision import (
    build_followup_decision_state,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


_MODAL_SCREEN_OCR = OCRService()
_ZERO_VALUE_EFFECT_MODAL_RE = re.compile(
    rf"({'|'.join(map(re.escape, (*ProduceText.STATUS_VALUE_TOKENS, ProduceText.YARUKI)))})"
    rf"{ProduceText.ZERO_VALUE_NO_EFFECT_PREFIX}{ProduceText.NO_EFFECT_TRIGGERED}"
)
_BATTLE_BLOCKED_CARD_STATE_KEY = "battle_blocked_cards"
_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY = "battle_last_attempted_card"
_EXAM_RETRY_COUNT_RE = re.compile(ProduceText.REMAINING_COUNT_PATTERN)


@dataclass
class ModalActionCandidate:
    """定义 ModalActionCandidate 的结构化数据。

    Attributes:
        index: 候选项在当前列表中的序号（通常从上到下或从左到右）。
        title: 候选项主标题文本，通常来自 OCR 或预设文案。
        action_id: 标准化动作标识，用于在决策层与执行层之间关联同一操作。
        label: 用于界面展示或日志输出的短标签文本。
        selected: 是否为当前已选中项（True 表示已选中）。
        recommended: 是否为系统推荐项（True 表示推荐）。
        box: 候选项对应的检测框，用于点击、裁剪和可视化调试。
        db_id: 数据库中的实体 ID；为空通常表示当前候选项尚未完成实体识别。
        source: 候选项来源标记（如 OCR、DB、fallback），便于排查识别链路。
        confidence: 当前识别或匹配结果的置信度，数值越高代表结果越可靠。
        metadata: 扩展元数据，保存额外识别信息与决策辅助字段。
    """
    index: int
    title: str
    action_id: str
    label: str = ""
    selected: bool = False
    recommended: bool = False
    box: Any = field(repr=False, default=None)
    db_id: str = ""
    source: str = "modal"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _click_modal_action_direct(
    app: "AppProcessor",
    *,
    prefer_confirm: bool,
) -> bool:
    """直接通过 YOLO 检测结果点击弹窗按钮。

    PRODUCER 模型将按钮标记为 Universal Confirm button / Universal Cancel button，
    而非 ModalParser 期望的 Universal button，因此在 try_get_modal 失败时使用此方法。
    当无法解析结构化弹窗时，允许按需求优先点确认或取消。
    """
    results = app.latest_results
    confirm_boxes = list(results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
    cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
    close_boxes = list(results.filter_by_label(ProducerLabels.CLOSE_BUTTON))
    buttons = list(results.filter_by_label(ProducerLabels.BUTTON))

    preferred_boxes = confirm_boxes if prefer_confirm else cancel_boxes
    fallback_boxes = cancel_boxes if prefer_confirm else confirm_boxes
    if preferred_boxes:
        target = max(preferred_boxes, key=lambda b: b.cy)
        app.device.click_element(target)
        return True
    if close_boxes and not prefer_confirm:
        target = max(close_boxes, key=lambda b: b.cy)
        app.device.click_element(target)
        return True
    # 语义明确的回退按钮（如 Cancel/Close）优先于通用按钮，
    # 通用按钮可能是布局切换、设置图标等非确认/取消操作
    if fallback_boxes:
        target = max(fallback_boxes, key=lambda b: b.cy)
        app.device.click_element(target)
        return True
    if buttons:
        if len(buttons) >= 2:
            ordered_buttons = sorted(buttons, key=lambda item: (item.cx, item.cy))
            app.device.click_element(ordered_buttons[-1] if prefer_confirm else ordered_buttons[0])
            return True
        app.device.click_element(max(buttons, key=lambda b: b.cy))
        return True
    return False


def _read_modal_fallback_text(app: "AppProcessor") -> str:
    """处理read、弹窗、fallback、文本并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        str: 处理后的文本结果。
    """
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return ""
    return " ".join(item.text for item in _MODAL_SCREEN_OCR.ocr(frame))


def _is_invalid_skill_use_modal(text: str) -> bool:
    """判断当前弹窗是否为“技能无法使用”提示。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    normalized = str(text or "").replace("\n", " ")
    if not normalized:
        return False
    if (
        ProduceText.SKILL_CARD_USE_CONFIRM not in normalized
        and ProduceText.EXECUTE_CONFIRM not in normalized
    ):
        return False
    return (
        ProduceText.NO_EFFECT_TRIGGERED in normalized
        or bool(_ZERO_VALUE_EFFECT_MODAL_RE.search(normalized))
    )


def _is_connection_error_modal(text: str) -> bool:
    """判断当前弹窗是否为连接错误提示。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    normalized = str(text or "").replace("\n", " ")
    if not normalized:
        return False
    return (
        ModalText.TITLE.CONNECTION_ERROR in normalized
        or ButtonText.RETRY_NETWORK in normalized
        or ButtonText.TO_TITLE in normalized
    )


def _set_connection_error_retry(ctx: "ProduceContext") -> None:
    """设置`connection_error_retry`。"""
    ctx.handler_state["unknown_retry_override"] = {
        "reason": "connection_error_modal",
        "retry_limit": int(
            ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
        ),
        "retry_sleep": float(
            ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
        ),
    }


def _set_exam_retry_transition_retry(ctx: "ProduceContext") -> None:
    """设置`exam_retry_transition_retry`。"""
    # 再挑战时清空战斗状态，避免上一轮的出牌计数器残留导致无法出牌
    ctx.handler_state.pop("virtual_battle_state", None)
    ctx.handler_state["unknown_retry_override"] = {
        "reason": "exam_retry_confirm_modal",
        "retry_limit": int(
            ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
        ),
        "retry_sleep": float(
            ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
        ),
    }


def _is_exam_retry_confirm_modal(text: str) -> bool:
    """判断当前弹窗是否为考试重试确认弹窗。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    normalized = str(text or "").replace("\n", " ")
    if not normalized:
        return False
    return (
        ProduceText.EXAM_RESULT_RETRY_CONFIRM in normalized
        or (ButtonText.RETRY in normalized and ButtonText.PRODUCE_END in normalized)
    )


def _extract_exam_retry_count(text: str) -> int | None:
    """提取考试、retry、count并返回结果。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        int | None: 返回值类型见注解，语义由函数用途决定。
    """
    match = _EXAM_RETRY_COUNT_RE.search(str(text or ""))
    if not match:
        return None
    return int(match.group(1))


def _build_exam_retry_candidates(
    app: "AppProcessor",
    *,
    modal: Any = None,
    text: str = "",
) -> list[ModalActionCandidate]:
    """构建考试、重试、候选项列表并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        modal: 用于提供弹窗相关输入。
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    retry_box = getattr(modal, "cancel_button", None)
    end_box = getattr(modal, "confirm_button", None)
    results = getattr(app, "latest_results", None)
    if retry_box is None and results is not None and hasattr(results, "filter_by_label"):
        cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
        if cancel_boxes:
            retry_box = max(cancel_boxes, key=lambda item: item.cy)
    if end_box is None and results is not None and hasattr(results, "filter_by_label"):
        confirm_boxes = list(results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
        if confirm_boxes:
            end_box = max(confirm_boxes, key=lambda item: item.cy)

    remaining_retry_count = _extract_exam_retry_count(text)
    shared_metadata = {
        "modal_text": str(text or "").strip(),
        "remaining_retry_count": remaining_retry_count,
        "candidate_type": "exam_retry_confirm",
    }
    return [
        ModalActionCandidate(
            index=0,
            title=ButtonText.RETRY,
            label=ButtonText.RETRY,
            action_id="exam_retry",
            box=retry_box,
            metadata={
                **shared_metadata,
                "description": "重新挑战当前这场考试，保留本次培育流程。",
            },
        ),
        ModalActionCandidate(
            index=1,
            title=ButtonText.PRODUCE_END,
            label=ButtonText.PRODUCE_END,
            action_id="produce_end",
            box=end_box,
            metadata={
                **shared_metadata,
                "description": "放弃本次考试并直接结束这次培育，结果按失败处理。",
            },
        ),
    ]


def _decide_exam_retry_candidate(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: list[ModalActionCandidate],
) -> ModalActionCandidate:
    """决策考试、retry、候选项并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        ModalActionCandidate: 返回值类型见注解，语义由函数用途决定。
    """
    decision_state = build_followup_decision_state(
        ctx,
        phase="exam",
        position=GameplayPosition.EXAM_RETRY_CONFIRM_MODAL,
        candidates=candidates,
        reason="exam_retry_confirm_modal_decision",
    )
    llm_snapshot = dict(decision_state.get("llm_snapshot", {}) or {})
    offensive_counts = dict(llm_snapshot.get("offensive_counts", {}) or {})
    has_battle_context = bool(
        llm_snapshot.get("hand")
        or int(llm_snapshot.get("deck_count") or 0) > 0
        or int(llm_snapshot.get("score") or 0) > 0
        or int(llm_snapshot.get("target") or 0) > 0
        or any(int(value or 0) > 0 for value in offensive_counts.values())
    )
    if not has_battle_context:
        for candidate in candidates:
            if candidate.action_id == "exam_retry":
                logger.info("modal: 再挑战确认缺少上一帧考试快照，兜底选择重新挑战")
                return candidate
    strategy = ctx.modal_strategy or ctx.exam_strategy
    decision = invoke_decision_strategy(
        strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        return candidates[resolve_candidate_index(decision, candidates)]
    for candidate in candidates:
        if candidate.action_id == "exam_retry":
            logger.info("modal: 再挑战确认未拿到策略结果，兜底选择重新挑战")
            return candidate
    return candidates[0]


def _handle_exam_retry_confirm_modal(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
    modal: Any = None,
    text: str = "",
) -> HandlerResult | None:
    """处理handle、考试、retry、confirm、弹窗并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识。
        modal: 用于提供弹窗相关输入。
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        HandlerResult | None: 返回值类型见注解。
    """
    if not _is_exam_retry_confirm_modal(text):
        return None

    candidates = _build_exam_retry_candidates(app, modal=modal, text=text)
    target = _decide_exam_retry_candidate(app, ctx, candidates)

    # 当出现票券消耗确认弹窗（“キャンセル”/“決定”按钮）时，
    # “決定”（confirm）= 消耗票券再挑战；“キャンセル”（cancel）= 不再挑战。
    # → 若希望再挑战，必须点击 confirm。
    is_ticket_confirm = ProduceText.TICKET in str(text or "")
    if is_ticket_confirm:
        # 票券消耗确认：retry → 決定(confirm)，cancel → キャンセル(cancel)。
        prefer_confirm = target.action_id == "exam_retry"
    else:
        # 传统“再挑战/结束培育”弹窗：retry → cancel 侧，end → confirm 侧。
        prefer_confirm = target.action_id != "exam_retry"

    success = False
    if modal is not None:
        from src.core.tasks.producer_challenge.ui import click_modal_action_with_retry

        success = click_modal_action_with_retry(
            app,
            modal,
            prefer_confirm=prefer_confirm,
            retries=2,
            timeout=4.0,
            action_name="exam retry confirm modal",
        )
    if not success:
        success = _click_modal_action_direct(app, prefer_confirm=prefer_confirm)
    if not success:
        return HandlerResult.no_action("modal: exam retry confirm button missing")

    if target.action_id == "exam_retry":
        _set_exam_retry_transition_retry(ctx)
    elif target.action_id == "produce_end":
        # 选择结束培育 → 标记 pending，主循环继续用 PRODUCER 处理后续结果页面
        ctx.handler_state["produce_finishing_pending"] = True
        logger.info("modal: 已标记 produce_finishing_pending，继续 PRODUCER 处理结果链")
    ctx.record_operation(
        "handle_modal",
        target=target.action_id,
        details={
            "position": position,
            "label": target.title,
            "remaining_retry_count": target.metadata.get("remaining_retry_count"),
        },
    )
    logger.info(
        "modal: 考试失败后选择 [{}] (チケット確認={}, prefer_confirm={})",
        target.title,
        is_ticket_confirm,
        prefer_confirm,
    )
    return HandlerResult.ok(
        f"modal {ProduceText.EXAM_RESULT_RETRY_CONFIRM!r}: choose {target.action_id}",
        sleep_after=1.0,
    )


def _remember_blocked_battle_card(ctx: "ProduceContext") -> None:
    """记录`blocked_battle_card`到上下文状态。"""
    attempted = dict(ctx.handler_state.get(_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY, {}) or {})
    turn_marker = attempted.get("turn_marker")
    if not turn_marker:
        return
    keys = [
        str(value)
        for value in (
            attempted.get("action_id"),
            attempted.get("db_id"),
            attempted.get("title"),
        )
        if str(value or "").strip()
    ]
    if not keys:
        return
    blocked_state = dict(ctx.handler_state.get(_BATTLE_BLOCKED_CARD_STATE_KEY, {}) or {})
    if blocked_state.get("turn_marker") != turn_marker:
        blocked_state = {"turn_marker": turn_marker, "keys": []}
    blocked_keys = list(blocked_state.get("keys", []) or [])
    for key in keys:
        if key not in blocked_keys:
            blocked_keys.append(key)
    blocked_state["keys"] = blocked_keys
    ctx.handler_state[_BATTLE_BLOCKED_CARD_STATE_KEY] = blocked_state
    ctx.handler_state.pop(_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY, None)


def _cancel_invalid_skill_use_modal(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
    text: str,
) -> HandlerResult | None:
    """处理cancel、invalid、skill、use、弹窗并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识。
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        HandlerResult | None: 返回值类型见注解。
    """
    if not _is_invalid_skill_use_modal(text):
        return None
    _remember_blocked_battle_card(ctx)
    if not _click_modal_action_direct(app, prefer_confirm=False):
        return HandlerResult.no_action("modal: invalid skill-use warning detected but cancel button missing")
    logger.info("modal: 检测到无效技能卡确认，改为取消")
    ctx.record_operation(
        "handle_modal",
        target="cancel_invalid_skill_use",
        details={"position": position, "text": text[:120]},
    )
    time.sleep(0.5)
    return HandlerResult.ok("modal: cancel invalid skill use", sleep_after=0.5)


def _click_modal_confirm_direct(app: "AppProcessor") -> bool:
    """点击弹窗、confirm、direct并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    return _click_modal_action_direct(app, prefer_confirm=True)


class ModalHandler(GameplayHandler):
    """处理 producer gameplay 中的弹窗。"""

    phase_tag = "modal"
    priority = 90  # 弹窗覆盖其他阶段，优先处理

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
        return phase == "modal"

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
        from src.core.tasks.producer_challenge.ui import click_modal_action_with_retry

        # 累计连续 modal 计数，用于检测卡住情况
        stuck_key = "modal_stuck_count"
        stuck_count = ctx.handler_state.get(stuck_key, 0) + 1
        ctx.handler_state[stuck_key] = stuck_count

        results = app.latest_results
        fallback_text = _read_modal_fallback_text(app)

        # 如果连续3次以上弹窗未消失，尝试更激进的关闭策略
        if stuck_count >= 3:
            if _is_connection_error_modal(fallback_text) and _click_modal_action_direct(app, prefer_confirm=True):
                logger.info("modal: 检测到通信错误弹窗，继续点击重试")
                _set_connection_error_retry(ctx)
                ctx.record_operation(
                    "handle_modal",
                    target="retry_connection_error",
                    details={"stuck_count": stuck_count, "position": position},
                )
                time.sleep(0.8)
                return HandlerResult.ok("modal: retry connection error", sleep_after=1.0)
            retry_result = _handle_exam_retry_confirm_modal(
                app,
                ctx,
                position=position,
                text=fallback_text,
            )
            if retry_result is not None:
                return retry_result
            logger.warning(f"modal: 连续 {stuck_count} 次未消失，尝试备用关闭策略")
            # 仅对显式关闭类控件做兜底，避免在未知弹窗上误点取消导致退回标题/放弃流程
            close_boxes = list(results.filter_by_label(ProducerLabels.CLOSE_BUTTON))
            if close_boxes:
                app.device.click_element(close_boxes[0])
                logger.debug("modal: 尝试 Close Button")
                ctx.record_operation("handle_modal", target="close_button_fallback",
                                     details={"stuck_count": stuck_count})
                time.sleep(0.8)
                return HandlerResult.ok("modal: close button fallback", sleep_after=0.5)

        # ── P_DRINK_DETAIL 安全处理 ──
        # P饮料详情模态不应默认点击“使う”（会消耗饮料），
        # 正常丢弃流程由 p_drink.py 的 _execute_drink_discard_chain 内联处理。
        # 如果 ModalHandler 收到 P_DRINK_DETAIL，说明是异常情况，安全取消。
        if position == GameplayPosition.P_DRINK_DETAIL:
            cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
            if cancel_boxes:
                app.device.click_element(cancel_boxes[0])
                logger.info("modal: P_DRINK_DETAIL 安全取消（避免误用饮料）")
                ctx.record_operation(
                    "handle_modal",
                    target="p_drink_detail_safe_cancel",
                    details={"position": position},
                )
                time.sleep(0.5)
                return HandlerResult.ok("modal: p_drink_detail safe cancel", sleep_after=0.5)
            # 取消按钮未检测到时走通用逻辑（但优先取消而非确认）
            if _click_modal_action_direct(app, prefer_confirm=False):
                logger.info("modal: P_DRINK_DETAIL 回退取消")
                ctx.record_operation(
                    "handle_modal",
                    target="p_drink_detail_fallback_cancel",
                    details={"position": position},
                )
                time.sleep(0.5)
                return HandlerResult.ok("modal: p_drink_detail fallback cancel", sleep_after=0.5)

        modal = app.game_utils.try_get_modal(no_body=False)
        if modal is None:
            if _is_connection_error_modal(fallback_text):
                if _click_modal_confirm_direct(app):
                    logger.info("modal: 检测到通信错误弹窗，点击重试")
                    _set_connection_error_retry(ctx)
                    ctx.record_operation(
                        "handle_modal",
                        target="retry_connection_error",
                        details={"position": position},
                    )
                    time.sleep(0.5)
                    return HandlerResult.ok("modal: retry connection error", sleep_after=1.0)
            invalid_result = _cancel_invalid_skill_use_modal(
                app,
                ctx,
                position=position,
                text=fallback_text,
            )
            if invalid_result is not None:
                return invalid_result
            retry_result = _handle_exam_retry_confirm_modal(
                app,
                ctx,
                position=position,
                text=fallback_text,
            )
            if retry_result is not None:
                return retry_result
            # PRODUCER 模型的按钮标签与 ModalParser 期望的不一致，
            # 直接通过 YOLO 检测结果点击确认按钮
            if _click_modal_confirm_direct(app):
                logger.debug("modal: ModalParser 无法解析，直接点击确认按钮")
                ctx.record_operation("handle_modal", target="direct_confirm", details={"position": position})
                time.sleep(0.5)
                return HandlerResult.ok("modal: direct confirm", sleep_after=0.5)
            return HandlerResult.ok("modal already dismissed")

        title = modal.modal_title or ""
        body_text = str(getattr(modal, "modal_body_text", "") or "")
        combined_text = f"{title} {body_text}".strip()
        logger.debug(f"modal: {title!r} (position={position})")

        if _is_connection_error_modal(combined_text):
            success = click_modal_action_with_retry(
                app,
                modal,
                prefer_confirm=True,
                retries=2,
                timeout=4.0,
                action_name="connection error modal",
            )
            if not success:
                return HandlerResult.no_action(f"modal {title!r}: failed to retry connection error")
            _set_connection_error_retry(ctx)
            ctx.record_operation(
                "handle_modal",
                target="retry_connection_error",
                details={"position": position},
            )
            logger.info("modal: 检测到通信错误弹窗，点击重试")
            return HandlerResult.ok(f"modal {title!r}: retry connection error", sleep_after=1.0)

        invalid_result = None
        if _is_invalid_skill_use_modal(combined_text):
            _remember_blocked_battle_card(ctx)
            success = click_modal_action_with_retry(
                app,
                modal,
                prefer_confirm=False,
                retries=2,
                timeout=4.0,
                action_name="gameplay invalid skill modal",
            )
            if not success:
                return HandlerResult.no_action(f"modal {title!r}: failed to cancel invalid skill use")
            ctx.record_operation(
                "handle_modal",
                target="cancel_invalid_skill_use",
                details={"position": position, "text": combined_text[:120]},
            )
            logger.info("modal: 检测到无效技能卡确认，改为取消")
            return HandlerResult.ok(f"modal {title!r}: cancel invalid skill use", sleep_after=0.5)

        retry_result = _handle_exam_retry_confirm_modal(
            app,
            ctx,
            position=position,
            modal=modal,
            text=combined_text,
        )
        if retry_result is not None:
            return retry_result

        ctx.record_operation(
            "handle_modal",
            target=title,
            details={"position": position},
        )

        success = click_modal_action_with_retry(
            app,
            modal,
            prefer_confirm=True,
            retries=2,
            timeout=4.0,
            action_name="gameplay modal",
        )
        if not success:
            return HandlerResult.no_action(f"modal {title!r}: failed to dismiss")
        return HandlerResult.ok(f"modal {title!r}: dismissed", sleep_after=0.5)
