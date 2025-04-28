from dataclasses import dataclass


@dataclass
class WebSocket_Data:
    message: str
    data: bytes
    def __init__(self, message:str | None, data: bytes | None = None):
        self.message = message
        self.data = data