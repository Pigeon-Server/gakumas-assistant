from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import (
    P_DRINK_SELECTION_POSITIONS,
    SKILL_REWARD_SELECTION_POSITIONS,
    GameplayPosition,
)
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.shared.common import (
    invoke_decision_strategy,
    ocr_text,
    resolve_candidate_index,
)

from .gameplay_state import get_pipeline_position

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


def click_recommend_action(
    app: "AppProcessor",
    ctx: "ProduceContext | None" = None,
) -> str | None:
    """点击推荐行程按钮，执行日程表步骤。

    委托给 execute_schedule_step 处理推荐行程的点击逻辑，包括识别推荐按钮位置、
    点击并等待页面过渡。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。
        ctx: 培育上下文对象，包含策略配置和操作记录。为 None 时直接返回。

    Returns:
        str | None: 执行结果状态字符串（如 "clicked"），失败或 ctx 为 None 时返回 None。
    """
    if ctx is None:
        return None
    from src.core.tasks.producer_challenge.gameplay.schedule import (
        execute_schedule_step,
    )

    result = execute_schedule_step(app, ctx, position=get_pipeline_position(app))
    return result.status if result else None


def handle_skill_card_selection(
    app: "AppProcessor",
    ctx: "ProduceContext | None" = None,
) -> str | None:
    """处理授课（レッスン）阶段的技能卡选择。

    委托给 execute_lesson_step 处理授课阶段的技能卡选择逻辑，包括识别技能卡、
    根据策略选择目标卡牌并点击。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。
        ctx: 培育上下文对象，包含策略配置和操作记录。为 None 时直接返回。

    Returns:
        str | None: 执行结果状态字符串，失败或 ctx 为 None 时返回 None。
    """
    if ctx is None:
        return None
    from src.core.tasks.producer_challenge.gameplay.lesson import (
        execute_lesson_step,
    )

    result = execute_lesson_step(app, ctx, position=get_pipeline_position(app))
    return result.status if result else None


def _click_preferred_confirmation(app: "AppProcessor") -> bool:
    """点击确认按钮，优先使用 YOLO 检测到的 CONFIRM_BUTTON 标签。

    当画面中存在 CONFIRM_BUTTON 标签时点击该按钮；否则回退到检测所有 BUTTON 标签，
    点击 Y 坐标最大的按钮（通常确认按钮位于画面最下方）。

    Args:
        app: 应用处理器实例，提供 latest_results 和 device.click_element。

    Returns:
        bool: 成功找到并点击按钮返回 True，画面中无任何按钮返回 False。
    """
    confirm_boxes = app.latest_results.filter_by_label(
        ProducerLabels.CONFIRM_BUTTON
    )
    if confirm_boxes:
        app.device.click_element(confirm_boxes.first())
        return True

    buttons = app.latest_results.filter_by_label(BaseUILabels.BUTTON)
    if buttons:
        app.device.click_element(max(buttons, key=lambda button: button.cy))
        return True
    return False


def handle_p_drink_select(
    app: "AppProcessor",
    ctx: "ProduceContext | None" = None,
) -> str | None:
    """处理 P 饮料（Pドリンク）选择与确认。

    分两种情况：
    - P_DRINK_SELECTED 阶段：点击确认按钮，确认上次选择的 P 饮料，清除暂存状态。
    - P_DRINK_SELECT 阶段：检测画面中所有 P 饮料候选项，按 X 坐标排序后调用策略
      决策选择目标，点击后记录暂存状态（index + OCR 标签）供确认阶段使用。

    Args:
        app: 应用处理器实例，提供 latest_results 和 device.click_element。
        ctx: 培育上下文对象，包含 p_drink_strategy 策略和暂存状态。

    Returns:
        str | None: "confirmed" 表示确认成功，"selected" 表示选择成功，不在 P 饮料阶段或无候选时返回 None。
    """
    position = get_pipeline_position(app)
    if position not in P_DRINK_SELECTION_POSITIONS:
        return None

    if position == GameplayPosition.P_DRINK_SELECTED:
        if not _click_preferred_confirmation(app):
            return None
        if ctx is not None:
            ctx.record_operation(
                "confirm_p_drink",
                target=ctx.pending_p_drink_label or "p_drink",
                details={"index": ctx.pending_p_drink_index},
            )
            ctx.clear_p_drink_pending()
        sleep(1.0)
        return "confirmed"

    drinks = sorted(
        app.latest_results.filter_by_label(ProducerLabels.P_DRINK),
        key=lambda item: item.cx,
    )
    if not drinks:
        return None

    target_index = 0
    if ctx is not None:
        decision = invoke_decision_strategy(ctx.p_drink_strategy, app, ctx, drinks)
        target_index = resolve_candidate_index(
            decision,
            drinks,
            default_index=ctx.pending_p_drink_index or 0,
        )

    target = drinks[target_index]
    app.device.click_element(target)
    if ctx is not None:
        ctx.pending_p_drink_index = target_index
        ctx.pending_p_drink_label = (
            ocr_text(target.frame) or f"p_drink_{target_index + 1}"
        )
        ctx.record_operation(
            "select_p_drink",
            target=ctx.pending_p_drink_label,
            details={"index": target_index},
        )
    sleep(0.8)
    return "selected"


def handle_skill_reward_selection(
    app: "AppProcessor",
    ctx: "ProduceContext | None" = None,
) -> str | None:
    """处理技能奖励（スキル報酬）选择与确认。

    分两种情况：
    - SKILL_REWARD_SELECTED 阶段：点击确认按钮，确认上次选择的技能奖励，清除暂存状态。
    - SKILL_REWARD_SELECT 阶段：检测画面中所有技能奖励候选项（ACTIVE/MENTAL/TRAP/INFO 四种标签），
      按 X 坐标排序后调用策略决策选择目标，点击后记录暂存状态（index + OCR 标签）。

    Args:
        app: 应用处理器实例，提供 latest_results 和 device.click_element。
        ctx: 培育上下文对象，包含 skill_reward_strategy 策略和暂存状态。

    Returns:
        str | None: "confirmed" 表示确认成功，"selected" 表示选择成功，不在技能奖励阶段或无候选时返回 None。
    """
    position = get_pipeline_position(app)
    if position not in SKILL_REWARD_SELECTION_POSITIONS:
        return None

    if position == GameplayPosition.SKILL_REWARD_SELECTED:
        if not _click_preferred_confirmation(app):
            return None
        if ctx is not None:
            ctx.record_operation(
                "confirm_skill_reward",
                target=ctx.pending_skill_reward_label or "skill_reward",
                details={"index": ctx.pending_skill_reward_index},
            )
            ctx.clear_skill_reward_pending()
        sleep(1.0)
        return "confirmed"

    candidates = []
    for label in (
        ProducerLabels.SKILL_CARD_ACTIVE,
        ProducerLabels.SKILL_CARD_MENTAL,
        ProducerLabels.SKILL_CARD_TRAP,
        ProducerLabels.SKILL_CARD_INFO,
    ):
        candidates.extend(app.latest_results.filter_by_label(label))
    candidates = sorted(candidates, key=lambda item: item.cx)
    if not candidates:
        return None

    target_index = 0
    if ctx is not None:
        decision = invoke_decision_strategy(
            ctx.skill_reward_strategy,
            app,
            ctx,
            candidates,
        )
        target_index = resolve_candidate_index(
            decision,
            candidates,
            default_index=ctx.pending_skill_reward_index or 0,
        )

    target = candidates[target_index]
    app.device.click_element(target)
    if ctx is not None:
        ctx.pending_skill_reward_index = target_index
        ctx.pending_skill_reward_label = (
            ocr_text(target.frame) or f"skill_reward_{target_index + 1}"
        )
        ctx.record_operation(
            "select_skill_reward",
            target=ctx.pending_skill_reward_label,
            details={"index": target_index},
        )
    sleep(0.8)
    return "selected"
