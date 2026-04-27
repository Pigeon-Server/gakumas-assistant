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
    """从最新一帧 YOLO 检测结果中提取所有按钮对象。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。

    Returns:
        ButtonList: 当前画面中检测到的所有按钮组成的列表对象。
    """
    return ButtonList(app.latest_results)


def find_button(
    app: "AppProcessor",
    text: str,
    *,
    fuzz_threshold: float = 70,
    use_contains: bool = True,
) -> Button | None:
    """在最新一帧的 YOLO 检测结果中按文本查找按钮。

    从当前画面中检测到的所有按钮里，使用模糊匹配或包含匹配查找与目标文本
    最接近的按钮。匹配失败时返回 None。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。
        text: 目标按钮文本，通常来自 ButtonText 常量或 ProduceText 常量。
        fuzz_threshold: 模糊匹配的最低分数阈值（0-100），低于此分的按钮不视为匹配。默认 70。
        use_contains: 是否启用包含匹配模式。启用时目标文本是按钮文本的子串即可匹配，
            关闭时要求双向包含或 fuzzy ratio。默认 True。

    Returns:
        Button | None: 匹配到的按钮对象，未找到时返回 None。
    """
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
    """判断当前画面中是否存在与目标文本匹配的按钮。

    底层调用 find_button，仅用于布尔判断场景，避免调用方额外写 `is not None`。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。
        text: 目标按钮文本，通常来自 ButtonText 常量或 ProduceText 常量。
        fuzz_threshold: 模糊匹配的最低分数阈值（0-100）。默认 70。
        use_contains: 是否启用包含匹配模式。默认 True。

    Returns:
        bool: 找到匹配按钮时返回 True，否则返回 False。
    """
    return find_button(
        app,
        text,
        fuzz_threshold=fuzz_threshold,
        use_contains=use_contains,
    ) is not None


def wait_frame_stable(app: "AppProcessor", timeout: float = 4.0) -> None:
    """等待画面帧稳定，确保页面过渡动画结束后再进行后续操作。

    通过比较连续帧的相似度来判断画面是否稳定。当连续 2 帧相似度超过 0.985 时
    认为画面已稳定。超时仍未稳定则直接返回，由调用方决定是否重试。

    Args:
        app: 应用处理器实例，提供 game_utils.wait_frame_stable 方法。
        timeout: 最长等待秒数。默认 4.0 秒。
    """
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
    """执行带惯性滑动的操作，并等待画面滑动动画结束。

    与普通 swipe 不同，该函数在滑动后会短暂等待（0.1s），然后调用 wait_frame_stable
    等待页面因惯性滑动产生的过渡动画完成，确保后续操作不会落在错误的帧上。

    Args:
        app: 应用处理器实例，提供 device.swipe 和 game_utils.wait_frame_stable。
        start_x: 滑动起始点的 X 坐标（像素）。
        start_y: 滑动起始点的 Y 坐标（像素）。
        end_x: 滑动结束点的 X 坐标（像素）。
        end_y: 滑动结束点的 Y 坐标（像素）。
        duration: 滑动动作持续时间（秒）。默认 0.45 秒。
        settle_timeout: 等待画面稳定的最长超时时间（秒）。默认 4.0 秒。
        hold_end: 滑动结束时在终点停留的时间（秒）。默认 0.15 秒。
        ease: 滑动缓动函数名称。默认 "out_quad"（减速结束）。
    """
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
    """判断当前页面是否为最终确认页（プロデュース開始前的确认画面）。

    通过排除法+正向条件组合判断：先排除含 NEXT/RESET/AUTO_SELECT 按钮的中间页面，
    再要求同时存在「編成詳細」按钮、「プロデュース開始」按钮，以及至少一种上下文
    标签（支援卡/记忆卡/特别道具），三者同时满足才确认为最终确认页。

    Args:
        app: 应用处理器实例，提供 latest_results 和按钮检测能力。

    Returns:
        bool: 当前页面是最终确认页返回 True，否则返回 False。
    """
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
    """轮询等待最终确认页面出现。

    每隔 0.4 秒检查一次当前页面是否为最终确认页（含編成詳細和プロデュース開始按钮）。
    检测到后额外等待画面稳定 3 秒。超时未出现则返回 False。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。
        timeout: 最长等待秒数。默认 15.0 秒。

    Returns:
        bool: 成功等到最终确认页返回 True，超时返回 False。
    """
    end_time = time() + timeout
    while time() < end_time:
        if is_final_confirm_page(app):
            wait_frame_stable(app, timeout=3.0)
            return True
        sleep(0.4)
    return False


def is_memory_selection_page(app: "AppProcessor") -> bool:
    """判断当前页面是否为记忆卡片选择页。

    通过排除最终确认页（无プロデュース開始按钮），同时要求存在 NEXT、AUTO_SELECT、
    RESET、編成詳細按钮，以及画面中有 MEMORY_CARD 标签，综合判定为记忆选择页。

    Args:
        app: 应用处理器实例，提供 latest_results 和按钮检测能力。

    Returns:
        bool: 当前页面是记忆选择页返回 True，否则返回 False。
    """
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
    """轮询等待记忆卡片选择页面出现。

    每隔 0.4 秒检查一次当前页面是否为记忆选择页（含 NEXT、AUTO_SELECT、RESET、
    編成詳細按钮及 MEMORY_CARD 标签）。检测到后额外等待画面稳定 3 秒。超时未出现
    则返回 False。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。
        timeout: 最长等待秒数。默认 12.0 秒。

    Returns:
        bool: 成功等到记忆选择页返回 True，超时返回 False。
    """
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
    """按确认/取消偏好点击弹窗按钮，失败时自动重试。

    优先点击指定按钮（默认为确认按钮），如果目标按钮不存在则回退到另一个可用按钮。
    每次点击后等待页面过渡动画完成。如果页面未发生变化则重试，最多 retries 次。
    如果画面中不存在弹窗则直接返回 True（无需处理）。

    Args:
        app: 应用处理器实例，提供 game_utils.try_get_modal 和 click_modal_button_and_wait_transition。
        modal: 已获取的弹窗对象，为 None 时自动从当前画面中检测。
        prefer_confirm: 是否优先点击确认按钮。True 时优先 confirm_button，False 时优先 cancel_button。
        retries: 最大重试次数。每次点击后若页面未变化则重新获取弹窗并重试。默认 3 次。
        timeout: 每次点击后等待页面过渡的最长超时时间（秒）。默认 5.0 秒。
        action_name: 日志中标识该操作的名称。默认 "modal action"。

    Returns:
        bool: 成功关闭弹窗返回 True，重试耗尽后仍然失败返回 False。
    """
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
    """点击画面右上角区域的操作按钮并等待画面触发变化。

    从当前画面所有按钮中筛选出中心点位于右上角区域（cx >= 屏幕宽度的 66% 且 cy <= 屏幕高度的 12%）的
    候选按钮，按 Y 坐标升序、X 坐标降序排序后点击最靠右上方的按钮。

    Args:
        app: 应用处理器实例，提供按钮检测和 click_element_and_wait_trigger。
        timeout: 等待画面触发变化的最长超时时间（秒）。默认 6.0 秒。

    Returns:
        bool: 成功找到按钮并点击后画面发生变化返回 True，未找到候选按钮返回 False。
    """
    buttons = get_buttons(app)
    frame = app.latest_frame
    if frame is None or frame.size == 0:
        return False
    height, width = frame.shape[:2]
    cx_threshold = int(width * 0.66)
    cy_threshold = int(height * 0.12)
    candidates = [
        button
        for button in buttons
        if button.cx >= cx_threshold and button.cy <= cy_threshold
    ]
    candidates.sort(key=lambda button: (button.cy, -button.cx))
    if not candidates:
        return False
    return app.game_utils.click_element_and_wait_trigger(
        candidates[0],
        timeout=timeout,
    )
