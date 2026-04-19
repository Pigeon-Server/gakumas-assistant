from __future__ import annotations

from time import sleep, time
from typing import TYPE_CHECKING

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Game.Components.Button import Button, ButtonList
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig

if TYPE_CHECKING:
    from src.entity.Game.Components.Modal import Modal
    from src.main import AppProcessor


def get_buttons(app: "AppProcessor") -> ButtonList:
    return ButtonList(app.latest_results)


def find_button(
    app: "AppProcessor",
    text: str,
    *,
    fuzz_threshold: float = 70,
    use_contains: bool = True,
) -> Button | None:
    return get_buttons(app).get_button_by_text(
        text,
        match_config=MatchConfig(
            fuzz_threshold=fuzz_threshold,
            use_contains=use_contains,
            normalize=True,
        ),
    )


def has_button(
    app: "AppProcessor",
    text: str,
    *,
    fuzz_threshold: float = 70,
    use_contains: bool = True,
) -> bool:
    return find_button(
        app,
        text,
        fuzz_threshold=fuzz_threshold,
        use_contains=use_contains,
    ) is not None


def wait_frame_stable(app: "AppProcessor", timeout: float = 4.0) -> None:
    app.game_utils.wait_frame_stable(
        threshold=0.985,
        stable_count=2,
        timeout=timeout,
    )


def inertial_swipe(
    app: "AppProcessor",
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    *,
    duration: float = 0.45,
    settle_timeout: float = 4.0,
    hold_end: float = 0.15,
    ease: str | None = "out_quad",
) -> None:
    """执行带惯性抑制的滑动。"""
    app.device.swipe(
        start_x,
        start_y,
        end_x,
        end_y,
        duration=duration,
        offset_y=0,
        hold_end=hold_end,
        ease=ease,
    )
    sleep(0.1)
    wait_frame_stable(app, timeout=settle_timeout)


def is_final_confirm_page(app: "AppProcessor") -> bool:
    if has_button(app, ButtonText.AUTO_SELECT, fuzz_threshold=75):
        return False
    if has_button(app, ButtonText.NEXT, fuzz_threshold=75):
        return False
    if has_button(app, ButtonText.RESET, fuzz_threshold=75):
        return False

    has_detail_button = has_button(
        app,
        ProduceText.FORMATION_DETAILS,
        fuzz_threshold=68,
    )
    has_start_button = has_button(
        app,
        ButtonText.PRODUCE_START,
        fuzz_threshold=65,
    )
    has_context = any(
        app.latest_results.exists_label(label)
        for label in (
            BaseUILabels.SUPPORT_CARD,
            BaseUILabels.MEMORY_CARD,
            BaseUILabels.SPECIAL_ITEMS,
        )
    )
    return bool(has_detail_button and has_start_button and has_context)


def wait_for_final_confirm_page(
    app: "AppProcessor",
    timeout: float = 15.0,
) -> bool:
    end_time = time() + timeout
    while time() < end_time:
        if is_final_confirm_page(app):
            wait_frame_stable(app, timeout=3.0)
            return True
        sleep(0.4)
    return False


def is_memory_selection_page(app: "AppProcessor") -> bool:
    if has_button(app, ButtonText.PRODUCE_START, fuzz_threshold=65):
        return False
    if not has_button(app, ButtonText.NEXT, fuzz_threshold=75):
        return False
    if not has_button(app, ButtonText.AUTO_SELECT, fuzz_threshold=75):
        return False
    if not has_button(app, ButtonText.RESET, fuzz_threshold=75):
        return False
    if not has_button(app, ProduceText.FORMATION_DETAILS, fuzz_threshold=68):
        return False
    return bool(app.latest_results.exists_label(BaseUILabels.MEMORY_CARD))


def wait_for_memory_selection_page(
    app: "AppProcessor",
    timeout: float = 12.0,
) -> bool:
    end_time = time() + timeout
    while time() < end_time:
        if is_memory_selection_page(app):
            wait_frame_stable(app, timeout=3.0)
            return True
        sleep(0.4)
    return False


def click_modal_action_with_retry(
    app: "AppProcessor",
    modal: "Modal | None" = None,
    *,
    prefer_confirm: bool = True,
    retries: int = 3,
    timeout: float = 5.0,
    action_name: str = "modal action",
) -> bool:
    current_modal = modal
    for attempt in range(1, retries + 1):
        if current_modal is None:
            current_modal = app.game_utils.try_get_modal(no_body=True)
        if current_modal is None:
            return True

        button = (
            current_modal.confirm_button
            if prefer_confirm
            else current_modal.cancel_button
        )
        if button is None:
            button = current_modal.cancel_button or current_modal.confirm_button
        if button is None:
            logger.warning(
                f"{action_name}: modal {current_modal.modal_title!r} has no actionable button"
            )
            return False

        if app.game_utils.click_modal_button_and_wait_transition(
            button,
            previous_modal_title=current_modal.modal_title,
            timeout=timeout,
            interval=0.2,
        ):
            wait_frame_stable(app, timeout=min(timeout, 3.0))
            return True

        logger.warning(
            f"{action_name}: modal {current_modal.modal_title!r} did not transition "
            f"after attempt {attempt}/{retries}"
        )
        sleep(0.5)
        current_modal = app.game_utils.try_get_modal(no_body=True)

    return False


def click_top_right_action(
    app: "AppProcessor",
    *,
    timeout: float = 6.0,
) -> bool:
    buttons = get_buttons(app)
    candidates = [
        button
        for button in buttons
        if button.cx >= 720 and button.cy <= 280
    ]
    candidates.sort(key=lambda button: (button.cy, -button.cx))
    if not candidates:
        return False
    return app.game_utils.click_element_and_wait_trigger(
        candidates[0],
        timeout=timeout,
    )
