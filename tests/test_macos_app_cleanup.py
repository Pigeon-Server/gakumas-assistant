from types import SimpleNamespace

import pytest

from src.core.device.MacOS import app as macos_app_module


class _AdapterStub:
    def __init__(self):
        self.connected = True
        self.disconnected = 0

    def disconnect(self):
        self.disconnected += 1
        self.connected = False


def test_macos_app_close_is_safe_after_port_validation_failure(monkeypatch):
    app = macos_app_module.MacOS_App.__new__(macos_app_module.MacOS_App)

    monkeypatch.setattr(
        macos_app_module,
        "ConfigService",
        lambda: SimpleNamespace(base=SimpleNamespace(playtools_port=0)),
    )

    with pytest.raises(ValueError):
        macos_app_module.MacOS_App.__init__(app)

    app.close()
    app.__del__()  # 验证析构路径不会重复释放
    assert app._adapter is None


def test_macos_app_close_disconnects_adapter_and_clears_reference():
    adapter = _AdapterStub()
    app = macos_app_module.MacOS_App.__new__(macos_app_module.MacOS_App)
    app._adapter = adapter

    app.close()

    assert adapter.disconnected == 1
    assert app._adapter is None
    assert bool(app) is False
