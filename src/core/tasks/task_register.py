from src.core.tasks.base_ui.start_game import (
    action__click_start_game,
    handle__network_error_modal_boxes,
    action__check_home_tab_exist
)
from src.utils.game_tools import get_current_location
from time import sleep
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.logger import logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app import AppProcessor

def register_tasks(processor: "AppProcessor"):
    @processor.register_task("start_game", "启动游戏")
    @logger.catch
    def _task__start_game(app: "AppProcessor"):
        TIMEOUT = 30
        if action__click_start_game(app, TIMEOUT) is not False:
            sleep(2)
            app.wait__loading()
            handle__network_error_modal_boxes(app)
        action__check_home_tab_exist(app)
        app.update_current_location()
