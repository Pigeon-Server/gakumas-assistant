from copy import copy
from time import sleep
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.device.Android.app import Android_App
from src.entity.Game.Components.Button import ButtonList
from src.utils.logger import logger
from src.utils.opencv_tools import compute_ssim_score
from src.utils.string_tools import string_match, MatchConfig

if TYPE_CHECKING:
    from src.main import AppProcessor


def _has_close_button(results) -> bool:
    buttons = ButtonList(results)
    return buttons.get_button_by_text(ButtonText.CLOSE, MatchConfig(fuzz_threshold=75)) is not None


def _try_get_pass_reward_modal(app: "AppProcessor", no_body: bool = True):
    results = app.latest_results
    if results is None:
        return None

    has_header = results.exists_label(BaseUILabels.MODAL_HEADER)
    if not has_header and not _has_close_button(results):
        return None

    return app.game_utils.try_get_modal(no_body=no_body, require_header=has_header)


def claim_pass_rewards(app: "AppProcessor"):
    prev_page: Optional[np.ndarray] = None

    while True:
        app.game_utils.wait_frame_stable(0.9, 1)

        _process_modal(app)
        _collect_visible_rewards(app)
        _scroll_reward_list(app)

        app.game_utils.wait_frame_stable()

        if prev_page is not None and _is_page_unchanged(prev_page, app.latest_frame):
            break

        prev_page = copy(app.latest_frame)

def _process_modal(app: "AppProcessor"):
    """
    处理意外的模态框
    :param app: app实例
    :return:
    """
    while True:
        modal = _try_get_pass_reward_modal(app, no_body=True)
        if modal is None:
            break

        close_button = modal.cancel_button or modal.confirm_button
        if string_match(modal.modal_title, ModalText.TITLE.MISSION_PASS_PT_ACQUIRED):
            logger.debug(f"Close pass reward modal '{modal.modal_title}'.")
            if close_button is None:
                logger.warning("Mission pass point modal has no actionable button.")
                break
            app.device.click_element(close_button)
            continue

        if modal.cancel_button and not modal.confirm_button:
            logger.debug(f"Close modal '{modal.modal_title}'.")
            app.device.click_element(modal.cancel_button)
            continue

        logger.warning(f"Unsupported pass reward modal '{modal.modal_title}', stop modal loop.")
        break
    app.game_utils.wait_frame_stable(0.9, 1)

def _collect_visible_rewards(app: "AppProcessor"):
    """
    查找并领取当前页奖励
    :param app: app实例
    :return:
    """
    buttons = _find_collect_buttons(app)

    for button in buttons:
        if button.is_disabled():
            continue

        if not app.game_utils.click_element_and_wait_trigger(button, retries=3, timeout=2.5):
            raise TimeoutError(f"Collect button '{button.text}' did not trigger any UI change.")
        _handle_collect_modal(app)
        app.game_utils.wait_for_label(BaseUILabels.CURRENT_LOCATION)
        sleep(1)

def _find_collect_buttons(app: "AppProcessor") -> list:
    """
    查找“领取”按钮
    :param app: app实例
    :return:
    """
    buttons = ButtonList(app.latest_results)
    return [
        btn for btn in buttons
        if string_match(
            btn.text,
            ButtonText.COLLECT,
            MatchConfig(fuzz_threshold=90)
        )
    ]

def _handle_collect_modal(app: "AppProcessor", max_wait: int = 5):
    """
    处理领取后的弹窗
    :param app: app实例
    :param max_wait: 最长等待时间
    :return:
    """
    for i in range(max_wait + 1):
        if i > max_wait:
            raise TimeoutError("Timeout waiting for modal to appear.")

        modal = _try_get_pass_reward_modal(app, no_body=True)
        if modal is not None:
            close_button = modal.cancel_button or modal.confirm_button
            if close_button is None:
                sleep(1)
                continue
            if string_match(modal.modal_title, ModalText.TITLE.RECEIPT_COMPLETED):
                app.device.click_element(close_button)
                return

            if string_match(modal.modal_title, ModalText.TITLE.MISSION_PASS_PT_ACQUIRED):
                app.device.click_element(close_button)
                return

            if string_match(modal.modal_title, ModalText.TITLE.CONNECTION_ERROR):
                app.device.click_element(modal.confirm_button or modal.cancel_button)
            else:
                app.device.click_element(close_button)

        sleep(1)

def _scroll_reward_list(app: "AppProcessor"):
    """
    滑动列表
    :param app: app实例
    :return:
    """
    if isinstance(app.device, Android_App):
        buttons = _find_collect_buttons(app)
        if not buttons:
            return

        h_list = [btn.h for btn in buttons]
        offset = (buttons[0].h - buttons[0].y)
        width, _ = app.device.get_window_size()
        app.device.swipe(
            width // 2,
            max(h_list),
            width // 2,
            min(h_list) - offset,
            0.8
        )
        sleep(0.5)
    else:
        y, x = app.latest_results.frame.shape[:2]
        app.device.scrollY(x // 2, y // 2, -20)

def _is_page_unchanged(prev: np.ndarray, curr: np.ndarray, threshold: float = 0.9) -> bool:
    """
    判断是否翻到尽头
    :param prev: 上一帧
    :param curr: 当前帧
    :param threshold: 阈值
    :return:
    """
    score = compute_ssim_score(prev, curr)
    return score > threshold
