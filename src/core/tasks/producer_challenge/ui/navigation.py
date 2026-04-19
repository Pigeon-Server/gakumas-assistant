from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import (
    GAMEPLAY_MODAL_POSITIONS,
    GameplayPosition,
)
from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels

from .common import (
    click_modal_action_with_retry,
    click_top_right_action,
    find_button,
)
from .gameplay_state import get_pipeline_position

if TYPE_CHECKING:
    from src.main import AppProcessor


def go_back_in_gameplay(app: "AppProcessor") -> bool:
    position = get_pipeline_position(app)
    if position in GAMEPLAY_MODAL_POSITIONS:
        modal = app.game_utils.try_get_modal(no_body=True)
        if modal is not None:
            prefer_confirm = position not in {
                GameplayPosition.P_DRINK_DETAIL,
                GameplayPosition.DETAIL_MODAL,
            }
            return click_modal_action_with_retry(
                app,
                modal,
                prefer_confirm=prefer_confirm,
                retries=2,
                timeout=4.0,
                action_name=f"go_back_in_gameplay[{position}]",
            )

    if close_buttons := app.latest_results.filter_by_label(BaseUILabels.CLOSE_BUTTON):
        return app.game_utils.click_element_and_wait_trigger(
            close_buttons.first(),
            timeout=3.0,
        )
    if back_buttons := app.latest_results.filter_by_label(BaseUILabels.BACK_BTN):
        return app.game_utils.click_element_and_wait_trigger(
            back_buttons.first(),
            timeout=3.0,
        )

    close_button = find_button(app, ButtonText.CLOSE, fuzz_threshold=60)
    if close_button is not None:
        return app.game_utils.click_element_and_wait_trigger(
            close_button,
            timeout=3.0,
        )

    cancel_button = find_button(app, ButtonText.CLOSE, fuzz_threshold=60) or find_button(
        app,
        ButtonText.CANCEL,
        fuzz_threshold=60,
    )
    if cancel_button is not None:
        return app.game_utils.click_element_and_wait_trigger(
            cancel_button,
            timeout=3.0,
        )

    return False


def go_home_from_gameplay(
    app: "AppProcessor",
    *,
    max_try: int = 4,
) -> bool:
    for _ in range(max_try):
        if app.latest_results.exists_label(BaseUILabels.TAB_HOME):
            return True

        if home_buttons := app.latest_results.filter_by_label(BaseUILabels.GO_HOME_BTN):
            if app.game_utils.click_element_and_wait_trigger(
                home_buttons.first(),
                timeout=3.0,
            ):
                sleep(1.0)
                continue

        text_button = (
            find_button(app, ButtonText.SAVE_AND_SUSPEND, fuzz_threshold=60)
            or find_button(app, ButtonText.HOME, fuzz_threshold=60)
            or find_button(app, ButtonText.RETIRE, fuzz_threshold=60)
        )
        if text_button is not None:
            if app.game_utils.click_element_and_wait_trigger(
                text_button,
                timeout=3.0,
            ):
                sleep(1.0)
                continue

        if go_back_in_gameplay(app):
            sleep(0.8)
            continue

        if click_top_right_action(app, timeout=2.0):
            sleep(1.0)
            continue

        break
    return app.latest_results.exists_label(BaseUILabels.TAB_HOME)
