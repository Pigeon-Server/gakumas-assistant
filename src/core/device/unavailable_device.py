from __future__ import annotations

import numpy as np

from src.entity.BaseDevice import BaseDevice
from src.utils.i18n_tools import I18nText, i18n_text


class UnavailableDevice(BaseDevice):
    def __init__(self, reason: I18nText | str, code: str = "device_unavailable"):
        self.reason = reason
        self.code = code

    def __bool__(self) -> bool:
        return False

    def get_unavailable_reason(self) -> I18nText | str:
        return self.reason

    def get_unavailable_code(self) -> str:
        return self.code

    def is_app_focused(self):
        return False

    def is_app_running(self):
        return False

    def start_game(self):
        raise RuntimeError(self.reason)

    def capture(self) -> np.ndarray:
        raise RuntimeError(self.reason)

    def click(self, x, y, el_label=""):
        raise RuntimeError(self.reason)

    def scrollY(self, x, y, scroll_delta):
        raise RuntimeError(self.reason)

    def scrollX(self, x, y, scroll_delta):
        raise RuntimeError(self.reason)
