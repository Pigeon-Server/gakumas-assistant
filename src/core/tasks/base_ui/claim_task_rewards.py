from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Game.Components.Button import Button
from src.entity.Game.Components.TabBar import TabBar, TabBarItem, _is_selected_tab_frame
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.main import AppProcessor


def claim_task_rewards(app: "AppProcessor"):
    tab_bar = _get_tab_bar(app)
    for tab in tab_bar:
        _process_single_tab(app, tab)
        sleep(1)

def _get_tab_bar(app: "AppProcessor") -> TabBar:
    """
    获取任务页面中的标签栏（tab bar）。
    """
    tab_bar_elem = app.latest_results.filter_by_label(BaseUILabels.TAB_BAR).first()
    return TabBar(tab_bar_elem)


def _process_single_tab(app: "AppProcessor", tab_item: TabBarItem):
    """
    点击 tab 并处理其对应的任务奖励。
    """
    _switch_to_tab(app, tab_item)

    button = _get_centered_enabled_button(app)
    if button is None:
        # 按钮可能因画面刚切换而被误判为 disabled，重试一次
        sleep(1)
        button = _get_centered_enabled_button(app)

    if button:
        _claim_reward(app, tab_item, button)
    else:
        logger.info(f"{tab_item.text} has no task rewards to be claimed")


def _is_tab_selected_by_highlight(app: "AppProcessor", target_tab: TabBarItem) -> bool:
    """
    基于选中高亮判断目标 tab 是否处于激活状态。
    """
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) == 0:
        return False

    frame_height, frame_width = frame.shape[:2]
    tab_w = max(int(target_tab.w - target_tab.x), 1)
    tab_h = max(int(target_tab.h - target_tab.y), 1)
    x1 = max(int(target_tab.x - max(6, tab_w * 0.2)), 0)
    y1 = max(int(target_tab.y - 2), 0)
    x2 = min(int(target_tab.w + max(6, tab_w * 0.2)), frame_width)
    y2 = min(int(target_tab.h + max(8, tab_h * 1.3)), frame_height)
    if x2 <= x1 or y2 <= y1:
        return False

    selection_frame = frame[y1:y2, x1:x2]
    return _is_selected_tab_frame(selection_frame)


def _switch_to_tab(app: "AppProcessor", tab_item: TabBarItem):
    """
    切换到指定 tab，并等待选中态稳定，减少切页与识别竞争导致的误判。
    """
    switched = app.game_utils.click_element_and_wait_trigger(
        tab_item,
        retries=1,
        timeout=1.2,
        interval=0.1,
        frame_threshold=0.999,
        region_threshold=0.95,
    )
    app.game_utils.wait_frame_stable(min_stable_duration=0.2)
    if _is_tab_selected_by_highlight(app, tab_item):
        return

    # 首次点击后仍未确认选中时再补一次点击，确保每个 tab 至少有一次显式切换尝试。
    app.device.click_element(tab_item)
    app.game_utils.wait_frame_stable(min_stable_duration=0.2)
    if _is_tab_selected_by_highlight(app, tab_item):
        return

    logger.warning(
        f"Task tab may not have switched to target after retry: {tab_item.text} (trigger_detected={switched})"
    )


def _get_centered_enabled_button(app: "AppProcessor"):
    """
    获取屏幕中央可点击的按钮。
    """
    buttons = app.latest_results.filter_by_label(BaseUILabels.BUTTON)
    height, width = app.latest_frame.shape[:2]
    frame_cx = width // 2

    for btn in buttons:
        if frame_cx - 10 < btn.cx < frame_cx + 10:
            button_obj = Button(btn)
            disabled = button_obj.is_disabled()
            logger.debug(f"Centered button: text='{button_obj.text}', cx={btn.cx}, disabled={disabled}")
            if not disabled:
                return btn
    return None


def _claim_reward(app: "AppProcessor", tab: TabBarItem, button: Button):
    """
    点击领奖按钮并处理领奖成功的弹窗。
    """
    if not app.game_utils.click_element_and_wait_trigger(button, retries=3, timeout=2.5):
        raise TimeoutError(f"Task reward button '{button.text}' did not trigger any UI change.")
    modal = app.game_utils.wait_for_modal(ModalText.TITLE.RECEIPT_COMPLETED, no_body=True, timeout=10)
    app.device.click_element(modal.cancel_button)
    app.game_utils.click_on_label(BaseUILabels.CLOSE_BUTTON, timeout=1, interval=0.3)
    app.game_utils.wait_frame_stable()
    logger.info(f"The task reward of {tab.text} has been claimed")
