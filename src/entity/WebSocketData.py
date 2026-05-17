import json
from dataclasses import dataclass

from src.utils.json_encoder import ComplexEncoder
from src.utils.i18n_tools import serialize_i18n_value


@dataclass
class WebSocketData:
    message: str | dict | list | None
    data: bytes | None

    def __init__(self, message: str | dict | list | None = None, data: bytes | None = None):
        self.message = serialize_i18n_value(message)
        self.data = data
