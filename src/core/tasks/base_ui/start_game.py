from typing import TYPE_CHECKING
from time import sleep

from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.game_tools import get_modal
from src.utils.logger import logger
from src.utils.string_tools import string_match

if TYPE_CHECKING:
    from src.main import AppProcessor

def action__click_start_game(app: "AppProcessor"):
    """动作：点击启动游戏"""
    if app.game_utils.wait_for_label(BaseUILabels.START_MENU_CLICK_CONTINUE_FLAG):
       app.game_utils.click_on_label(BaseUILabels.START_MENU_CLICK_CONTINUE_FLAG)
    else:
        raise TimeoutError("Timeout waiting for continue flag in the start menu.")

def _handle__modal_boxes(app: "AppProcessor"):
    """处理模态框"""
    logger.debug("_handle__modal_boxes")
    modal = get_modal(app.latest_results)
    if modal is None:
        raise RuntimeError("Failed to parse modal box")

    def _click_action_button():
        action_button = modal.cancel_button or modal.confirm_button
        if action_button is None:
            raise RuntimeError(f"Modal '{modal.modal_title or '<empty>'}' has no actionable button")
        app.device.click_element(action_button)

    if string_match(ModalText.TITLE.CONNECTION_ERROR, modal.modal_title):
        # Token失效
        if string_match(ModalText.BODY.CONNECTION_ERROR_ID.TOKEN_FAIL, modal.modal_body_text):
            logger.warning("Network connection error: Token fail")
            app.device.click_element(modal.cancel_button)
            action__click_start_game(app)
        # 连接超时
        elif string_match(ModalText.BODY.CONNECTION_ERROR_ID.TIMEOUT, modal.modal_body_text):
            logger.warning("Network connection error: Timeout")
            app.device.click_element(modal.confirm_button)
        else:
            logger.warning("Network connection error: Connection error")
            app.device.click_element(modal.confirm_button)
    # 下载新数据
    elif string_match(ModalText.TITLE.DATA_DOWNLOAD, modal.modal_title):
        logger.warning("game requires downloading new data.")
        app.device.click_element(modal.confirm_button)
    elif string_match(modal.modal_title, [ModalText.TITLE.DATA_UPDATE, ModalText.TITLE.DATE_UPDATE]):
        logger.warning(f"Restarting start flow for modal: {modal.modal_title}")
        _click_action_button()
        sleep(1)
        app.game_utils.wait_loading()
        action__click_start_game(app)
        return
    elif string_match(ModalText.TITLE.INIT_ERROR, modal.modal_title):
        logger.error("Game initialization failed.")
        app.device.click_element(modal.cancel_button)
        action__click_start_game(app)
    elif string_match(ModalText.TITLE.INFO_FETCH_FAILED, modal.modal_title):
        logger.warning("Information fetch failed, dismiss modal and retry waiting.")
        _click_action_button()
    elif string_match(ModalText.TITLE.USAGE_EXPIRATION_CONFIRM, modal.modal_title):
        logger.warning("Usage expiration modal detected during start flow, dismissing with cancel")
        if modal.cancel_button is not None:
            app.device.click_element(modal.cancel_button)
        else:
            _click_action_button()
    elif string_match(
        modal.modal_title,
        [
            ModalText.TITLE.EXCHANGE_CONFIRMATION,
            ModalText.TITLE.PURCHASE_CONFIRMATION,
            ModalText.TITLE.WORK_START_CONFIRMATION,
        ],
    ):
        # 启动流程中若残留业务弹窗，优先走取消，避免误确认导致状态漂移
        logger.warning(f"Dismissing non-startup modal during start flow: {modal.modal_title}")
        _click_action_button()
    elif string_match(ModalText.TITLE.SKIP_CONFIRM, modal.modal_title):
        logger.warning("Skip the game story...")
        app.device.click_element(modal.confirm_button)
        sleep(2)
    # 游戏更新
    elif string_match(ModalText.TITLE.GAME_UPDATE, modal.modal_title):
        raise RuntimeWarning("Game requires an update from the App Store. Please update manually.")
    elif modal.confirm_button is None and modal.cancel_button is not None:
        logger.warning(f"Dismissing single-action startup modal: {modal.modal_title}")
        app.device.click_element(modal.cancel_button)
    else:
        raise RuntimeError(f"Unknown modal box: {modal.modal_title or '<empty>'}")
    sleep(1)
    app.game_utils.wait_loading()

def action__wait_enter_home(app: "AppProcessor"):
    """动作：检查主界面标识是否存在"""
    COUNT:int = 0
    REQUIRED_COUNT:int = 3
    non_blocking_locations = {
        GamePageTypes.UNKNOWN,
        GamePageTypes.LOADING,
        GamePageTypes.DOWNLOADING,
        GamePageTypes.START_GAME,
        GamePageTypes.MAIN_MENU__HOME,
    }
    while True:
        current_location = app.game_utils.update_current_location()
        if current_location in {
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_DETAIL,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_CANDIDATE_LIST,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__FORMATION_DETAIL,
        }:
            logger.debug(f"Closing special page before returning home: {current_location}")
            try:
                app.game_utils.back_next_page()
            except Exception as exc:
                logger.warning(f"Failed to close special page while waiting for home: {current_location} ({exc})")
            sleep(1)
            continue
        if current_location in {
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_SELECTION,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__FINAL_CONFIRM,
        }:
            logger.debug(f"Leaving producer page before returning home: {current_location}")
            try:
                app.game_utils.go_home(max_try=2)
            except Exception as exc:
                logger.warning(f"Failed to leave producer page while waiting for home: {current_location} ({exc})")
            sleep(1)
            continue
        if current_location not in non_blocking_locations:
            # 已知处于非主页功能页时，优先主动回主页，避免停在商店/活动页空转
            logger.debug(f"Leaving non-home page while waiting for home: {current_location}")
            try:
                app.game_utils.go_home(max_try=2)
            except Exception as exc:
                logger.warning(f"Failed to leave non-home page while waiting for home: {current_location} ({exc})")
            sleep(1)
            continue
        if close_btn := app.latest_results.filter_by_label(BaseUILabels.CLOSE_BUTTON):
            app.device.click_element(close_btn.first())
            sleep(1)
        elif skip_btn := app.latest_results.filter_by_label(BaseUILabels.SKIP_BUTTON):
            app.device.click_element(skip_btn.first())
            sleep(1)
        elif app.latest_results.filter_by_label(BaseUILabels.MODAL_HEADER):
            _handle__modal_boxes(app)
        elif app.latest_results.filter_by_label(BaseUILabels.TAB_HOME):
            if current_location != GamePageTypes.MAIN_MENU__HOME:
                logger.debug(f"Home tab detected on {current_location}, clicking home tab")
                try:
                    app.game_utils.click_on_label(BaseUILabels.TAB_HOME)
                    app.game_utils.wait_loading()
                except Exception as exc:
                    logger.warning(f"Failed to click home tab while waiting for home: {current_location} ({exc})")
                sleep(1)
                continue
            COUNT = 0
            for _ in range(REQUIRED_COUNT):
                sleep(1)
                if (
                    app.latest_results.filter_by_label(BaseUILabels.TAB_HOME)
                    and app.game_utils.update_current_location() == GamePageTypes.MAIN_MENU__HOME
                ):
                    COUNT += 1
                    continue
                COUNT = 0
                break
            if COUNT >= REQUIRED_COUNT:
                return True
        else:
            height, width = app.latest_frame.shape[:2]
            app.device.click(width // 3, height // 2)
            sleep(1)
