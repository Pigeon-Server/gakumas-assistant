from time import sleep, time
from typing import TYPE_CHECKING

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.contest_overlay_tools import detect_contest_grade_up_splash, detect_contest_season_overlay
from src.utils.game_tools import get_modal
from src.utils.string_tools import MatchConfig, string_match
from src.utils.task_debug_tools import record_task_step

if TYPE_CHECKING:
    from src.main import AppProcessor


_CONTEST_ENTRY_BUTTON_MATCH = MatchConfig(fuzz_threshold=80, normalize=True)
_CONTEST_ENTRY_STATUS_MATCH = MatchConfig(fuzz_threshold=55, normalize=True)


def _looks_like_contest_entry_text(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if ButtonText.CHALLENGE in normalized:
        return True
    if ButtonText.MAIN_MENU__CONTEST.CONTEST in normalized:
        return True
    if ButtonText.MAIN_MENU__CONTEST.CHALLENGING in normalized:
        return True
    if string_match(normalized, ButtonText.START_CHALLENGE, _CONTEST_ENTRY_STATUS_MATCH):
        return True
    if string_match(normalized, ButtonText.BATTLE_START, _CONTEST_ENTRY_STATUS_MATCH):
        return True
    return False


def _wait_for_contest_entry_button(
    app: "AppProcessor",
    *,
    timeout: float = 4.0,
    interval: float = 0.25,
):
    """等待竞技场入口按钮出现，规避切页后一两帧的检测空窗。"""
    deadline = time() + timeout
    button = _get_contest_entry_button(app)
    if button is not None:
        return button
    while time() < deadline:
        sleep(interval)
        button = _get_contest_entry_button(app)
        if button is not None:
            return button
    return None


def _dismiss_contest_season_overlay_if_present(app: "AppProcessor", reason: str) -> bool:
    """
    关闭竞技场“赛季排行”覆盖层。

    该覆盖层没有标准按钮，因此通过整帧 OCR 识别后点击覆盖层中心，
    并等待覆盖层消失。
    """
    overlay = detect_contest_season_overlay(app.latest_frame, add_debug_box=True)
    if overlay is None:
        return False

    tap_points = [
        (overlay.center_x, overlay.center_y),
        (overlay.center_x, min(app.latest_frame.shape[0] - 1, overlay.bottom - max(20, (overlay.bottom - overlay.top) // 6))),
    ]
    for tap_index, (tap_x, tap_y) in enumerate(tap_points, start=1):
        record_task_step(
            app,
            "goto_contest.dismiss_overlay_tap",
            reason=reason,
            tap_index=tap_index,
            x=int(tap_x),
            y=int(tap_y),
            rank=overlay.rank_text,
        )
        app.device.click(int(tap_x), int(tap_y))
        sleep(1)
        if detect_contest_season_overlay(app.latest_frame, add_debug_box=True) is None:
            record_task_step(app, "goto_contest.dismiss_overlay_done", reason=reason, tap_index=tap_index)
            return True

    record_task_step(app, "goto_contest.dismiss_overlay_failed", reason=reason, rank=overlay.rank_text)
    return False


def _dismiss_contest_grade_up_splash_if_present(app: "AppProcessor", reason: str) -> bool:
    """
    关闭竞技场「グレードUP」演出页。

    真机验证结果表明：该页需要轻触上半屏标题区域，而不是点中间徽章。
    """
    splash = detect_contest_grade_up_splash(app.latest_frame, add_debug_box=True)
    if splash is None:
        return False

    frame_height, frame_width = app.latest_frame.shape[:2]
    tap_points = [
        (splash.title_center_x, splash.title_center_y),
        (frame_width // 2, max(40, int(frame_height * 0.27))),
    ]
    for tap_index, (tap_x, tap_y) in enumerate(tap_points, start=1):
        record_task_step(
            app,
            "goto_contest.dismiss_grade_up_tap",
            reason=reason,
            tap_index=tap_index,
            x=int(tap_x),
            y=int(tap_y),
            title=splash.title_text,
        )
        app.device.click(int(tap_x), int(tap_y))
        sleep(1)
        if detect_contest_grade_up_splash(app.latest_frame, add_debug_box=True) is None:
            record_task_step(app, "goto_contest.dismiss_grade_up_done", reason=reason, tap_index=tap_index)
            return True

    record_task_step(app, "goto_contest.dismiss_grade_up_failed", reason=reason, title=splash.title_text)
    return False


def _settle_contest_blocking_layers(app: "AppProcessor", reason: str) -> bool:
    """
    清理进入竞技场后的阻塞层。

    当前已确认的链路是：
    1. シーズンランキング 覆盖层
    2. グレードUP 演出页
    """
    handled = False
    for _ in range(4):
        if _dismiss_contest_season_overlay_if_present(app, reason):
            handled = True
            continue
        if _dismiss_contest_grade_up_splash_if_present(app, reason):
            handled = True
            continue
        break
    return handled


def _back_home(app: "AppProcessor"):
    if app.game_utils.update_current_location() != GamePageTypes.MAIN_MENU__HOME:
        try:
            app.game_utils.go_home()
            app.game_utils.wait_location_update(GamePageTypes.MAIN_MENU__HOME)
        except (TimeoutError, RuntimeError):
            from src.core.tasks.base_ui.start_game import action__wait_enter_home

            action__wait_enter_home(app)
            app.game_utils.update_current_location(GamePageTypes.MAIN_MENU__HOME)

def _goto_tab_contest(app: "AppProcessor"):
    if app.game_utils.update_current_location() == GamePageTypes.MAIN_MENU__CONTEST:
        return
    _back_home(app)
    if not app.game_utils.wait_for_label(BaseUILabels.TAB_CONTEST):
        raise TimeoutError("Timeout waiting for [tab:contest] to appear.")
    app.game_utils.click_on_label(BaseUILabels.TAB_CONTEST)
    app.game_utils.wait_location_update(GamePageTypes.MAIN_MENU__CONTEST)

def _goto_tab_idol(app: "AppProcessor"):
    if app.game_utils.update_current_location() == GamePageTypes.MAIN_MENU__IDOL:
        return
    _back_home(app)
    if not app.game_utils.wait_for_label(BaseUILabels.TAB_IDOL):
        raise TimeoutError("Timeout waiting for [tab:idol] to appear.")
    app.game_utils.click_on_label(BaseUILabels.TAB_IDOL)
    app.game_utils.wait_location_update(GamePageTypes.MAIN_MENU__IDOL)

def goto__get_expenditure(app: "AppProcessor", candidate_index: int = 0):
    """ 进入“活动费”领取菜单，点击第 candidate_index 个候选按钮 """
    from time import sleep
    _back_home(app)
    if not app.game_utils.wait_for_label(BaseUILabels.HOME_GET_EXPENDITURE):
        raise TimeoutError("Timeout waiting for [home:expenditure] to appear.")
    # wait_for_label 返回后 YOLO 可能已更新帧，用短暂重试避免竞态条件
    candidates = None
    for _ in range(10):
        candidates = app.latest_results.filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        if candidates:
            break
        sleep(0.2)
    if not candidates:
        raise TimeoutError("Failed to locate [home:expenditure] button after label wait.")
    idx = min(candidate_index, len(candidates) - 1)
    expenditure_button = candidates.boxes[idx]
    app.game_utils.click_element_and_wait_trigger(expenditure_button, retries=3, timeout=3.0, interval=0.1)


def goto__work_dispatch_page(app: "AppProcessor"):
    """ 进入任务派遣页面 """
    _back_home(app)
    if not app.game_utils.wait_for_label(BaseUILabels.HOME_DISPATCH_WORK):
        raise TimeoutError("Timeout waiting for [home:dispatch work] to appear.")
    app.game_utils.click_on_label(BaseUILabels.HOME_DISPATCH_WORK)
    app.game_utils.wait_loading()

def goto__gift_page(app: "AppProcessor"):
    """ 进入礼物领取页面 """
    _back_home(app)
    if not app.game_utils.wait_for_label(BaseUILabels.HOME_GIFT_BTN):
        raise TimeoutError("Timeout waiting for [home:gift] to appear.")
    app.game_utils.click_on_label(BaseUILabels.HOME_GIFT_BTN)
    app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.GIFT)

def goto__shop_page(app: "AppProcessor"):
    """ 进入商店页面 """
    _back_home(app)
    if not app.game_utils.wait_for_label(BaseUILabels.HOME_SHOP_BTN):
        raise TimeoutError("Timeout waiting for [home:shop] to appear.")
    app.game_utils.click_on_label(BaseUILabels.HOME_SHOP_BTN)
    app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.SHOP)

def goto__contest_page(app: "AppProcessor"):
    """ 进入竞技场页面 """
    if app.game_utils.update_current_location() == GamePageTypes.CONTEST_TAB.ARENA:
        _settle_contest_blocking_layers(app, "already_in_arena")
        if detect_contest_season_overlay(app.latest_frame, add_debug_box=True) is not None:
            raise TimeoutError("Contest season overlay detected but could not be dismissed.")
        if detect_contest_grade_up_splash(app.latest_frame, add_debug_box=True) is not None:
            raise TimeoutError("Contest grade-up splash detected but could not be dismissed.")
        record_task_step(app, "goto_contest.already_in_arena")
        return
    _goto_tab_contest(app)
    record_task_step(app, "goto_contest.enter_tab")
    if app.game_utils.update_current_location() == GamePageTypes.CONTEST_TAB.ARENA:
        _settle_contest_blocking_layers(app, "entered_arena_from_tab")
        if detect_contest_season_overlay(app.latest_frame, add_debug_box=True) is not None:
            raise TimeoutError("Contest season overlay detected but could not be dismissed.")
        if detect_contest_grade_up_splash(app.latest_frame, add_debug_box=True) is not None:
            raise TimeoutError("Contest grade-up splash detected but could not be dismissed.")
        record_task_step(app, "goto_contest.entered_arena_from_tab")
        return

    last_error: TimeoutError | None = None
    for attempt in range(2):
        contest_button = _wait_for_contest_entry_button(app)
        if contest_button is None:
            record_task_step(
                app,
                "goto_contest.entry_button_missing",
                attempt=attempt + 1,
                location=app.game_utils.update_current_location(),
            )
            if app.game_utils.update_current_location() == GamePageTypes.CONTEST_TAB.ARENA:
                _settle_contest_blocking_layers(app, f"attempt_{attempt + 1}_already_arena")
                record_task_step(app, "goto_contest.entered_arena_without_entry_button", attempt=attempt + 1)
                return
            if attempt == 0:
                continue
            raise TimeoutError("Timeout waiting for contest entry button to appear.")

        record_task_step(
            app,
            "goto_contest.click_entry",
            attempt=attempt + 1,
            text=getattr(contest_button, "text", None),
            cx=int(contest_button.cx),
            cy=int(contest_button.cy),
        )
        if not app.game_utils.click_element_and_wait_trigger(
                contest_button,
                retries=3,
                timeout=2.5,
                interval=0.1,
        ):
            record_task_step(app, "goto_contest.click_entry_no_trigger", attempt=attempt + 1)
            app.device.click_element(contest_button)

        try:
            app.game_utils.wait_loading(timeout=8)
        except TimeoutError as exc:
            last_error = exc
            record_task_step(
                app,
                "goto_contest.wait_loading_timeout",
                attempt=attempt + 1,
                error=str(exc),
            )

        try:
            app.game_utils.wait_location_update(GamePageTypes.CONTEST_TAB.ARENA, timeout=10)
            record_task_step(app, "goto_contest.entered_arena", attempt=attempt + 1)
            return
        except TimeoutError as exc:
            last_error = exc
            record_task_step(
                app,
                "goto_contest.location_timeout",
                attempt=attempt + 1,
                error=str(exc),
            )
            if _settle_contest_blocking_layers(app, f"attempt_{attempt + 1}_timeout"):
                app.game_utils.update_current_location(GamePageTypes.CONTEST_TAB.ARENA)
                record_task_step(app, "goto_contest.entered_arena_via_overlay", attempt=attempt + 1)
                return
            if detect_contest_season_overlay(app.latest_frame, add_debug_box=True) is not None:
                raise TimeoutError("Contest season overlay detected but could not be dismissed.")
            if detect_contest_grade_up_splash(app.latest_frame, add_debug_box=True) is not None:
                raise TimeoutError("Contest grade-up splash detected but could not be dismissed.")
            if app.game_utils.update_current_location() != GamePageTypes.MAIN_MENU__CONTEST:
                continue

    if last_error is not None:
        raise last_error
    raise TimeoutError("Timeout waiting for contest page to open.")


def _get_contest_entry_button(app: "AppProcessor"):
    buttons = ButtonList(app.latest_results)
    if not buttons:
        return None

    if button := buttons.get_button_by_text(
            ButtonText.MAIN_MENU__CONTEST.CONTEST,
            match_config=_CONTEST_ENTRY_BUTTON_MATCH,
    ):
        return button

    text_candidates = []
    for button in buttons:
        if button is None or button.is_disabled():
            continue
        if _looks_like_contest_entry_text(getattr(button, "text", "")):
            text_candidates.append(button)
    if text_candidates:
        return max(
            text_candidates,
            key=lambda item: (
                int(item.w - item.x),
                int(item.h - item.y),
                int(item.cy),
            ),
        )

    frame_height, frame_width = app.latest_frame.shape[:2]
    min_width = int(frame_width * 0.35)
    min_height = int(frame_height * 0.10)
    right_threshold = int(frame_width * 0.58)
    top_threshold = int(frame_height * 0.55)
    bottom_threshold = int(frame_height * 0.88)

    candidates = []
    for button in buttons:
        if button is None or button.is_disabled():
            continue
        button_width = int(button.w - button.x)
        button_height = int(button.h - button.y)
        if button_width < min_width or button_height < min_height:
            continue
        if int(button.cx) < right_threshold:
            continue
        if not top_threshold <= int(button.cy) <= bottom_threshold:
            continue
        candidates.append(button)

    if not candidates:
        # 新版布局可能只有左侧「挑戦中」卡片入口，补充一层宽松几何兜底。
        for button in buttons:
            if button is None or button.is_disabled():
                continue
            button_width = int(button.w - button.x)
            button_height = int(button.h - button.y)
            if button_width < int(frame_width * 0.22) or button_height < int(frame_height * 0.08):
                continue
            if not int(frame_height * 0.50) <= int(button.cy) <= int(frame_height * 0.88):
                continue
            candidates.append(button)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (int(item.w - item.x), int(item.cx), int(item.cy)))

def goto__claim_task_rewards_page(app: "AppProcessor"):
    """ 进入任务奖励领取页面 """
    _back_home(app)
    if not app.game_utils.wait_for_label(BaseUILabels.HOME_DAILY_TASK):
        raise TimeoutError("Timeout waiting for [home:daily_task] to appear.")
    app.game_utils.click_on_label(BaseUILabels.HOME_DAILY_TASK)
    app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.TASK)

def goto__claim_pass_rewards(app: "AppProcessor"):
    """ 进入大月卡奖励领取页面 """
    goto__claim_task_rewards_page(app)
    app.game_utils.click_button(ButtonText.PAGE__TASK_REWARDS.PASS_REWARDS, match_config=MatchConfig(fuzz_threshold=90))
    app.game_utils.wait_loading()
    for i in range(3):
        if not app.latest_results.exists_label(BaseUILabels.MODAL_HEADER) and app.latest_results.exists_all_labels([BaseUILabels.CURRENT_LOCATION, BaseUILabels.BUTTON]):
            break
        if app.latest_results.exists_label(BaseUILabels.MODAL_HEADER):
            modal = app.game_utils.wait_for_modal(None, timeout=5, no_body=True)
            if not modal:
                continue
            action_button = modal.confirm_button or modal.cancel_button
            if action_button is None:
                continue
            app.device.click_element(action_button)
            app.game_utils.wait_loading()
            if app.latest_results.exists_label(BaseUILabels.MODAL_HEADER):
                followup_modal = app.game_utils.wait_for_modal(None, timeout=5, no_body=True)
                if followup_modal is not None:
                    followup_action = followup_modal.cancel_button or followup_modal.confirm_button
                    if followup_action is not None:
                        app.device.click_element(followup_action)
                        app.game_utils.wait_loading()
    if app.latest_results.exists_label(BaseUILabels.MODAL_HEADER):
        modal = get_modal(app.latest_results, True)
        if modal:
            action_button = modal.cancel_button or modal.confirm_button
            if action_button is not None:
                app.device.click_element(action_button)
    app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.PASS_REWARD)

def goto_support_card_list_page(app: "AppProcessor"):
    _goto_tab_idol(app)
    app.game_utils.click_button(ButtonText.PAGE__IDOL.SUPPORT_CARD, match_config=MatchConfig(fuzz_threshold=90))
    app.game_utils.wait_loading()
    app.game_utils.wait_for_label(BaseUILabels.SUPPORT_CARD)

def goto_idol_card_list_page(app: "AppProcessor"):
    """进入 P アイドル育成列表页面"""
    _goto_tab_idol(app)
    app.game_utils.click_button(ButtonText.PAGE__IDOL.IDOL_CULTIVATION, match_config=MatchConfig(fuzz_threshold=85))
    app.game_utils.wait_loading()
    if not app.game_utils.wait_for_label(BaseUILabels.PRODUCT_CARD_SELECTED, timeout=10):
        raise TimeoutError("Timeout waiting for idol card cultivation carousel to appear.")
