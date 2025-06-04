from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.logger import logger
from src.constants import *
from typing import TYPE_CHECKING

from src.utils.yolo_tools import get_modal

if TYPE_CHECKING:
    from app import AppProcessor

def register_middlewares(processor: "AppProcessor"):
    @processor.register_middleware()
    @logger.catch
    def _init_location(app: "AppProcessor"):
        if app.game_status_manager.current_location is None:
            app.update_current_location()
            app.exec_task()
        return True


    @processor.register_middleware()
    @logger.catch
    def _handle_unexpected_modal(app: "AppProcessor"):
        if app.latest_results.exists_label(base_labels.modal_header):
            modal_header = get_modal(app.latest_results, app.latest_frame, True)
            pass
        return True