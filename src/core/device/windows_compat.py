import os
import platform
from typing import Any

from src.utils.i18n_tools import I18nText, i18n_text

_WINDOWS_APP_CLASS = None
_WINDOWS_IMPORT_ERROR = None
_WINDOWS_LOADED = False


def _ensure_windows_loaded():
    global _WINDOWS_APP_CLASS, _WINDOWS_IMPORT_ERROR, _WINDOWS_LOADED
    if _WINDOWS_LOADED:
        return
    _WINDOWS_LOADED = True
    if platform.system() != "Windows":
        return
    # Python 3.8+ compatibility for pywin32
    if hasattr(os, "add_dll_directory"):
        import sys
        for p in sys.path:
            p32 = os.path.join(p, "pywin32_system32")
            if os.path.isdir(p32):
                try:
                    os.add_dll_directory(p32)
                except Exception:
                    pass  # 忽略异常

    try:
        from src.core.device.Windows.app import Windows_App as _cls
        _WINDOWS_APP_CLASS = _cls
    except Exception as exc:
        _WINDOWS_IMPORT_ERROR = exc


def windows_pc_mode_is_available() -> bool:
    _ensure_windows_loaded()
    return _WINDOWS_APP_CLASS is not None


def get_windows_unavailability_reason() -> I18nText:
    _ensure_windows_loaded()
    if _WINDOWS_APP_CLASS is not None:
        return i18n_text("backend.device.windows.available", fallback="")
    if platform.system() != "Windows":
        return i18n_text(
            "backend.device.windows.unavailable.non_windows",
            fallback="PC 模式仅支持 Windows，请在 macOS/Linux 上使用 Phone 模式。",
        )
    if _WINDOWS_IMPORT_ERROR is not None:
        return i18n_text(
            "backend.device.windows.unavailable.import_error",
            fallback=(
                "PC 模式依赖的 Windows 专用组件未就绪（通常是 pywin32 未安装或损坏），"
                "请重新执行 `pip install -r requirements.txt` 后重试。"
            ),
        )
    return i18n_text(
        "backend.device.windows.unavailable.unknown",
        fallback="PC 模式当前不可用。",
    )


def create_windows_device():
    _ensure_windows_loaded()
    if _WINDOWS_APP_CLASS is None:
        message = get_windows_unavailability_reason()
        if _WINDOWS_IMPORT_ERROR is not None:
            raise RuntimeError(str(message)) from _WINDOWS_IMPORT_ERROR
        raise RuntimeError(str(message))
    return _WINDOWS_APP_CLASS()


def is_windows_device(device: Any) -> bool:
    _ensure_windows_loaded()
    return _WINDOWS_APP_CLASS is not None and isinstance(device, _WINDOWS_APP_CLASS)
