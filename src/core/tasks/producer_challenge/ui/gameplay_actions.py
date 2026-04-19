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
    if ctx is None:
        return None
    from src.core.tasks.producer_challenge.gameplay.lesson import (
        execute_lesson_step,
    )

    result = execute_lesson_step(app, ctx, position=get_pipeline_position(app))
    return result.status if result else None


def _click_preferred_confirmation(app: "AppProcessor") -> bool:
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
