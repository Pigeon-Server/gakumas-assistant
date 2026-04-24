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
    """在游戏内执行返回操作，逐级关闭覆盖层回到主 gameplay 画面。

    按优先级依次尝试：
    1. 弹窗位置：根据 position 类型决定优先确认还是取消，点击弹窗按钮
    2. CLOSE_BUTTON 标签：点击 YOLO 检测到的关闭按钮
    3. BACK_BTN 标签：点击 YOLO 检测到的返回按钮
    4. OCR 文本匹配：查找「閉じる」或「キャンセル」按钮点击

    Args:
        app: 应用处理器实例，提供 latest_results、game_utils 和 device。

    Returns:
        bool: 成功执行返回操作返回 True，所有方式均失败返回 False。
    """
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
    """从游戏内返回主页（ホーム），最多尝试 max_try 轮。

    每轮按优先级依次尝试：
    1. 检查是否已在主页（TAB_HOME 标签）
    2. 点击 GO_HOME_BTN 标签按钮
    3. OCR 查找「保存中断」/「ホーム」/「引退」按钮
    4. 调用 go_back_in_gameplay 逐级返回
    5. 点击右上角操作按钮

    Args:
        app: 应用处理器实例，提供 latest_results、game_utils 和 device。
        max_try: 最大尝试轮数。每轮执行一个操作后等待 0.8-1.0 秒。默认 4 轮。

    Returns:
        bool: 最终画面出现 TAB_HOME 标签返回 True，超过轮数仍未到达主页返回 False。
    """
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
