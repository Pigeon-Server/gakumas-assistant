from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig, string_match

if TYPE_CHECKING:
    from src.entity.Game.Components.Modal import Modal
    from src.main import AppProcessor


def _dismiss_modal_until_closed(app: "AppProcessor", modal: "Modal", *, max_attempts: int = 3) -> bool:
    """
    重复点击模态框关闭按钮，直到模态框真正消失。

    :param app: 应用实例
    :param modal: 当前模态框
    :param max_attempts: 最多点击次数
    :return: True 表示模态框已关闭；False 表示多次尝试后仍未关闭
    """
    current_modal = modal
    for attempt in range(max_attempts):
        close_button = current_modal.cancel_button or current_modal.confirm_button
        if close_button is None:
            raise RuntimeError(f"Modal '{current_modal.modal_title}' close button not found.")

        closed = app.game_utils.click_modal_button_and_wait_transition(
            close_button,
            previous_modal_title=current_modal.modal_title,
            timeout=3,
            interval=0.2,
        )
        if closed:
            return True

        refreshed_modal = app.game_utils.try_get_modal(no_body=True)
        if refreshed_modal is None:
            logger.debug(f"Modal '{current_modal.modal_title}' disappeared after retry click.")
            return True

        if not string_match(refreshed_modal.modal_title, current_modal.modal_title, MatchConfig(fuzz_threshold=90)):
            logger.debug(
                f"Modal '{current_modal.modal_title}' transitioned to '{refreshed_modal.modal_title}'."
            )
            return True

        logger.warning(
            f"Modal '{current_modal.modal_title}' still open after close click. "
            f"Retrying close button... ({attempt + 1}/{max_attempts})"
        )
        current_modal = refreshed_modal

    return False


def action__claim_expenditure(app: "AppProcessor", max_attempts: int = 3) -> bool:
    """
    打开活动费弹窗；若误点进其他弹窗，则关闭后尝试下一个候选按钮。
    """
    from src.core.tasks.base_ui.goto_pages import goto__get_expenditure

    candidate_index = 0
    for attempt in range(max_attempts):
        goto__get_expenditure(app, candidate_index=candidate_index)
        modal = app.game_utils.wait_for_modal(None, no_body=True, timeout=4.0, interval=0.2)
        if not modal:
            continue

        if string_match(modal.modal_title, ModalText.TITLE.EXPENDITURE, MatchConfig(fuzz_threshold=90)):
            if not _dismiss_modal_until_closed(app, modal):
                raise TimeoutError(f"Modal '{modal.modal_title}' did not close after repeated clicks.")
            sleep(0.5)
            return True

        logger.warning(
            f"Unexpected modal '{modal.modal_title}' when opening expenditure. "
            f"Trying next candidate... ({attempt + 1}/{max_attempts})"
        )
        if not _dismiss_modal_until_closed(app, modal):
            raise TimeoutError(f"Unexpected modal '{modal.modal_title}' did not close after repeated clicks.")
        candidate_index += 1
        sleep(0.5)

    if app.latest_results.exists_label(BaseUILabels.TAB_HOME) and not app.latest_results.exists_label(BaseUILabels.MODAL_HEADER):
        logger.warning("There are no claimable expenses")
        return True

    raise TimeoutError("Timeout waiting for expenditure modal to appear.")
