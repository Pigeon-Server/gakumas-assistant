from typing import TYPE_CHECKING

from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.game_tools import get_modal
from src.utils.logger import logger
from src.utils.string_tools import string_match, MatchConfig

if TYPE_CHECKING:
    from src.main import AppProcessor

last_card_name = ""
last_modal = False
last_modal_title = ""

def register_middlewares(processor: "AppProcessor"):

    @processor.task_queue.register_task_middleware()
    @logger.catch
    def _init_location(app: "AppProcessor"):
        if app.game_status_manager.current_location is None and app.latest_results:
            app.game_utils.update_current_location()
        return True

    @processor.task_queue.register_task_middleware()
    @logger.catch
    def _handle_unexpected_modal(app: "AppProcessor"):
        global last_modal, last_modal_title
        if not app.latest_results.exists_label(BaseUILabels.MODAL_HEADER):
            last_modal = False
            last_modal_title = ""
            return True

        if last_modal and not string_match(
            last_modal_title,
            [ModalText.TITLE.CONNECTION_ERROR, ModalText.TITLE.INFO_FETCH_FAILED],
        ):
            return True

        modal = get_modal(app.latest_results, True, quiet=True)
        if modal is None:
            return True

        # 网络/信息获取失败弹窗需要允许重复处理：即使同一弹窗持续存在，也要继续尝试重试按钮。
        if string_match(modal.modal_title, [ModalText.TITLE.CONNECTION_ERROR, ModalText.TITLE.INFO_FETCH_FAILED]):
            logger.warning("Network connection error...")
            action_button = modal.confirm_button or modal.cancel_button
            if action_button is not None:
                app.device.click_element(action_button)
                app.game_utils.wait_loading()
            else:
                logger.warning("Connection modal has no actionable button.")
            last_modal = False
            last_modal_title = modal.modal_title or ""
            return True

        if last_modal:
            return True

        if string_match(modal.modal_title, [ModalText.TITLE.DATA_UPDATE, ModalText.TITLE.DATE_UPDATE], MatchConfig(fuzz_threshold=90)):
            logger.warning("Restart game...")
            if modal.cancel_button is not None:
                app.device.click_element(modal.cancel_button)
            elif modal.confirm_button is not None:
                app.device.click_element(modal.confirm_button)
            app.game_utils.wait_loading()
            app.game_utils.wait_for_label(BaseUILabels.START_MENU_LOGO)
            app.task_queue.insert_task_to_run_queue("start_game")
        last_modal = True
        last_modal_title = modal.modal_title or ""
        return True
