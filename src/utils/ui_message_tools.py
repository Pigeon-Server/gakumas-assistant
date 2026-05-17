from src.constants.websocket_actions import WebsocketActions
from src.core.web.websocket import WebSocketManager
from src.entity.WebSocketData import WebSocketData
from src.utils.i18n_tools import I18nText, serialize_i18n_value

websocket = WebSocketManager()

class UIMessage:

    @staticmethod
    def _send(action, msg: I18nText | str | dict, timeout):
        websocket.broadcast_action_sync(
            action,
            WebSocketData(message={
                "message": serialize_i18n_value(msg),
                "close_delay": timeout,
            })
        )

    def info(self, msg: I18nText | str | dict, timeout=3):
        self._send(WebsocketActions.Message.ShowMessage_Info, msg, timeout)

    def warning(self, msg: I18nText | str | dict, timeout=3):
        self._send(WebsocketActions.Message.ShowMessage_Warning, msg, timeout)

    def error(self, msg: I18nText | str | dict, timeout=3):
        self._send(WebsocketActions.Message.ShowMessage_Error, msg, timeout)

    def success(self, msg: I18nText | str | dict, timeout=3):
        self._send(WebsocketActions.Message.ShowMessage_Success, msg, timeout)
