import platform
from typing import Any

from src.utils.i18n_tools import I18nText, i18n_text

_MACOS_APP_CLASS = None
_MACOS_IMPORT_ERROR = None

if platform.system() == "Darwin":
    try:
        from src.core.device.MacOS.app import MacOS_App as _MACOS_APP_CLASS
    except Exception as exc:
        _MACOS_IMPORT_ERROR = exc


def mac_playtools_mode_is_available() -> bool:
    return _MACOS_APP_CLASS is not None


def get_mac_unavailability_reason() -> I18nText:
    if _MACOS_APP_CLASS is not None:
        return i18n_text("backend.device.mac.available", fallback="")
    if platform.system() != "Darwin":
        return i18n_text(
            "backend.device.mac.unavailable.non_macos",
            fallback="MacPlayTools 模式仅支持 macOS (Apple Silicon)。",
        )
    if _MACOS_IMPORT_ERROR is not None:
        return i18n_text(
            "backend.device.mac.unavailable.import_error",
            fallback=f"MacPlayTools 模式依赖的组件未就绪：{_MACOS_IMPORT_ERROR}",
        )
    return i18n_text(
        "backend.device.mac.unavailable.unknown",
        fallback="MacPlayTools 模式当前不可用。",
    )


def create_mac_device():
    if _MACOS_APP_CLASS is None:
        message = get_mac_unavailability_reason()
        if _MACOS_IMPORT_ERROR is not None:
            raise RuntimeError(str(message)) from _MACOS_IMPORT_ERROR
        raise RuntimeError(str(message))
    return _MACOS_APP_CLASS()


def is_mac_device(device: Any) -> bool:
    return _MACOS_APP_CLASS is not None and isinstance(device, _MACOS_APP_CLASS)
