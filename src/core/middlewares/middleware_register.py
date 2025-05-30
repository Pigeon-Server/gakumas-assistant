from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.logger import logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app import AppProcessor

def register_middlewares(processor: "AppProcessor"):
    @processor.register_middleware()
    @logger.catch
    def _update_location(app: "AppProcessor"):
        if app.game_status_manager.current_location == GamePageTypes.UNKNOWN:
            app.update_current_location()
