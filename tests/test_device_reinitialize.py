import threading
from types import SimpleNamespace

import pytest

from src.constants.device.device_type import DeviceType
from src.core.device.Android.app import Android_App
from src.main import AppProcessor
import src.main as main_module
from src.utils.i18n_tools import i18n_text, serialize_i18n_value


class _DeviceStub:
    def __init__(self, events: list[str], name: str, available: bool = True):
        self._events = events
        self._name = name
        self._available = available
        self.closed = 0

    def __bool__(self) -> bool:
        return self._available

    def close(self):
        self.closed += 1
        self._events.append(f"close:{self._name}")


class _YoloStub:
    def __init__(self, events: list[str], running: bool):
        self._events = events
        self.running = running
        self.devices = []

    def stop(self):
        self._events.append("stop")
        self.running = False
        return True

    def start(self):
        self._events.append("start")
        self.running = True
        return True

    def set_device(self, device):
        self.devices.append(device)
        self._events.append("set_device")


class _PollingThreadStub:
    def __init__(self, target=None, daemon=True):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.started_with_device = None
        self.started_with_yolo = None
        self.join_timeout = None

    def start(self):
        owner = self.target.__self__
        self.started = True
        self.started_with_device = hasattr(owner, "device")
        self.started_with_yolo = hasattr(owner, "yolo_engine")

    def is_alive(self):
        return self.started

    def join(self, timeout=None):
        self.join_timeout = timeout
        self.started = False


def test_ensure_device_ready_force_closes_old_device_before_recreating():
    events = []
    old_device = _DeviceStub(events, "old")
    new_device = _DeviceStub(events, "new")
    yolo_engine = _YoloStub(events, running=True)

    app = AppProcessor.__new__(AppProcessor)
    app._device_state_lock = threading.RLock()
    app.device = old_device
    app.yolo_engine = yolo_engine

    def _create_device():
        events.append(f"create_after_close:{old_device.closed}")
        return new_device

    app.create_device_instance = _create_device

    ready = AppProcessor.ensure_device_ready(app, force=True, restart_inference=True)

    assert ready is True
    assert app.device is new_device
    assert old_device.closed == 1
    assert yolo_engine.devices == [new_device]
    assert events == [
        "stop",
        "close:old",
        "create_after_close:1",
        "set_device",
        "start",
    ]


def test_init_starts_device_polling_after_device_and_yolo_ready(monkeypatch):
    created_threads = []

    def _create_thread(target=None, daemon=True):
        thread = _PollingThreadStub(target=target, daemon=daemon)
        created_threads.append(thread)
        return thread

    class _ConfigServiceStub:
        def __init__(self):
            self.base = SimpleNamespace(run_mode=DeviceType.PHONE)

        def __call__(self):
            return self

    class _YoloEngineStub:
        def __init__(self, device):
            self.device = device
            self.running = False

        def register_infer_callback(self, _callback):
            return None

        def register_capture_failure_callback(self, _callback):
            return None

    class _TaskServiceStub:
        def __init__(self, _app):
            return None

    class _ResourceUpdateServiceStub:
        def __init__(self, _app):
            return None

        @staticmethod
        def has_required_resources():
            return False

    monkeypatch.setattr(main_module.threading, "Thread", _create_thread)
    monkeypatch.setattr(main_module.AppProcessor, "_init_environment", lambda self: None)
    monkeypatch.setattr(main_module.AppProcessor, "_init_database", staticmethod(lambda: None))
    monkeypatch.setattr(main_module.AppProcessor, "_register_task_services", lambda self: None)
    monkeypatch.setattr(main_module.AppProcessor, "_register_config_listening", lambda self: None)
    monkeypatch.setattr(main_module.AppProcessor, "create_device_instance", lambda self: object())
    monkeypatch.setattr(main_module, "ConfigService", _ConfigServiceStub)
    monkeypatch.setattr(main_module, "YoloInferenceEngine", _YoloEngineStub)
    monkeypatch.setattr(main_module, "DebugTools", lambda: object())
    monkeypatch.setattr(main_module, "TaskService", _TaskServiceStub)
    monkeypatch.setattr(main_module, "GameStatusManager", lambda: object())
    monkeypatch.setattr(main_module, "WebSocketManager", lambda: object())
    monkeypatch.setattr(main_module, "ResourceUpdateService", _ResourceUpdateServiceStub)

    app = AppProcessor()

    assert len(created_threads) == 1
    assert app._device_polling_thread is created_threads[0]
    assert created_threads[0].started is True
    assert created_threads[0].started_with_device is True
    assert created_threads[0].started_with_yolo is True


def test_shutdown_closes_current_device():
    events = []
    device = _DeviceStub(events, "active")
    yolo_engine = _YoloStub(events, running=True)
    polling_thread = _PollingThreadStub()
    polling_thread.started = True

    app = AppProcessor.__new__(AppProcessor)
    app._shutdown_requested = threading.Event()
    app.task_queue = type("TaskQueueStub", (), {"stop": lambda self: events.append("task_stop")})()
    app.yolo_engine = yolo_engine
    app.device = device
    app._device_polling_thread = polling_thread

    AppProcessor.shutdown(app)

    assert app.is_shutdown_requested() is True
    assert device.closed == 1
    assert polling_thread.join_timeout == 3
    assert events == ["task_stop", "stop", "close:active"]


def test_handle_runtime_adb_error_preserves_nested_i18n_reason(monkeypatch):
    """确保 ADB 运行时错误不会把嵌套国际化原因提前压平成字符串。"""
    reason = i18n_text(
        "backend.adb.deviceOffline",
        fallback="ADB 设备当前处于离线状态。请重新连接设备后重试。",
    )

    def _fake_describe_adb_error(_exc: Exception, **_kwargs):
        return "adb_device_disconnected", reason

    app = Android_App.__new__(Android_App)
    app._Android_App__connect_mode = "USB"
    app._Android_App__adb_host = "192.168.1.32"
    app._Android_App__adb_port = 5555
    app._Android_App__adb_serial = "192.168.1.32:5555"
    app._Android_App__adb_device = object()

    monkeypatch.setattr("src.core.device.Android.app.describe_adb_error", _fake_describe_adb_error)
    monkeypatch.setattr(app, "_reset_runtime_services", lambda: None)

    with pytest.raises(RuntimeError):
        app._handle_runtime_adb_error(RuntimeError("offline"), "ADB 初始化")

    unavailable_reason = serialize_i18n_value(app._Android_App__unavailable_reason)
    assert unavailable_reason["key"] == "backend.adb.initFailed"
    assert unavailable_reason["params"]["message"]["key"] == "backend.adb.deviceOffline"
    assert unavailable_reason["params"]["message"]["fallback"] == "ADB 设备当前处于离线状态。请重新连接设备后重试。"
