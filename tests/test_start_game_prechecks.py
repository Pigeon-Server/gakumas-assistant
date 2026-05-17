import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.device.Android.app import Android_App
from src.core.tasks import task_register
from src.core.tasks.task_register import register_tasks
from src.entity.Game.Page.Types.index import GamePageTypes


class _TaskQueueStub:
    def __init__(self):
        self.pre_hooks = {}
        self.tasks = {}

    def register_task(self, task_id, *args, **kwargs):
        def decorator(func):
            self.tasks[task_id] = func
            return func

        return decorator

    def register_pre_queue_start(self):
        def decorator(func):
            self.pre_hooks[func.__name__] = func
            return func

        return decorator


def _build_processor(
        auto_start_game: bool,
        device=None,
        latest_results=None,
        game_utils=None,
):
    processor = SimpleNamespace(
        task_queue=_TaskQueueStub(),
        config_service=lambda: SimpleNamespace(
            base=SimpleNamespace(
                auto_start_game=SimpleNamespace(value=auto_start_game),
            )
        ),
        device=device,
        latest_results=latest_results,
        game_utils=game_utils,
    )
    register_tasks(processor)
    return processor


def test_wait_game_start_skips_when_auto_start_disabled():
    task_register.GAME_RUNNING = False
    processor = _build_processor(
        auto_start_game=False,
        latest_results=SimpleNamespace(
            exists_label=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not inspect startup labels"))
        ),
        game_utils=SimpleNamespace(
            update_current_location=lambda: (_ for _ in ()).throw(AssertionError("should not refresh location"))
        ),
    )

    assert processor.task_queue.pre_hooks["_pre__wait_game_start"]() is True


def test_wait_game_start_accepts_loading_state(monkeypatch):
    monkeypatch.setattr(task_register, "sleep", lambda *_args, **_kwargs: None)
    task_register.GAME_RUNNING = False
    processor = _build_processor(
        auto_start_game=True,
        latest_results=SimpleNamespace(exists_label=lambda *_args, **_kwargs: False),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.LOADING),
    )

    assert processor.task_queue.pre_hooks["_pre__wait_game_start"]() is True


def test_pre_start_game_waits_for_cold_start_process(monkeypatch):
    monkeypatch.setattr(task_register, "sleep", lambda *_args, **_kwargs: None)

    class _DeviceStub:
        def __init__(self):
            self.start_game_calls = 0
            self._running_states = [False, False, True]

        def is_app_running(self):
            state = self._running_states.pop(0)
            return state

        def start_game(self):
            self.start_game_calls += 1

    device = _DeviceStub()
    processor = _build_processor(
        auto_start_game=True,
        device=device,
        latest_results=SimpleNamespace(exists_label=lambda *_args, **_kwargs: False),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
    )

    assert processor.task_queue.pre_hooks["_pre__start_game"]() is True
    assert device.start_game_calls == 1


def test_pre_start_game_brings_windows_game_to_front_even_when_auto_start_is_disabled(monkeypatch):
    monkeypatch.setattr(task_register, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_register, "is_windows_device", lambda _device: True)

    class _WindowsDeviceStub:
        def __init__(self):
            self.bring_to_front_calls = 0
            self._focused = False

        def is_app_running(self):
            return True

        def is_app_focused(self):
            return self._focused

        def bring_to_front(self):
            self.bring_to_front_calls += 1
            self._focused = True

    device = _WindowsDeviceStub()
    processor = _build_processor(
        auto_start_game=False,
        device=device,
        latest_results=SimpleNamespace(exists_label=lambda *_args, **_kwargs: False),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
    )

    assert processor.task_queue.pre_hooks["_pre__start_game"]() is True
    assert device.bring_to_front_calls == 1


def test_pre_start_game_android_does_not_trust_background_process_state(monkeypatch):
    monkeypatch.setattr(task_register, "sleep", lambda *_args, **_kwargs: None)

    class _AndroidDeviceStub(Android_App):
        def __init__(self):
            self.start_game_calls = 0

        def is_app_focused(self):
            return False

        def start_game(self):
            self.start_game_calls += 1

    task_register.GAME_RUNNING = False
    processor = _build_processor(
        auto_start_game=True,
        device=_AndroidDeviceStub(),
        latest_results=SimpleNamespace(exists_label=lambda *_args, **_kwargs: False),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
    )

    assert processor.task_queue.pre_hooks["_pre__start_game"]() is True
    assert task_register.GAME_RUNNING is False
    assert processor.device.start_game_calls == 1


def test_android_start_game_waits_until_app_is_focused(monkeypatch):
    monkeypatch.setattr("src.core.device.Android.app.time.sleep", lambda *_args, **_kwargs: None)

    device = Android_App.__new__(Android_App)
    started_packages = []
    focus_checks = {"count": 0}
    device._Android_App__adb_device = SimpleNamespace(app_start=lambda package: started_packages.append(package))
    device._Android_App__package_name = "com.example.game"

    def _is_app_focused(self):
        focus_checks["count"] += 1
        return focus_checks["count"] >= 3

    monkeypatch.setattr(Android_App, "is_app_focused", _is_app_focused)

    device.start_game()

    assert started_packages == ["com.example.game"]
    assert focus_checks["count"] == 3


def test_android_focus_detection_uses_dumpsys_fallback():
    device = Android_App.__new__(Android_App)
    dumpsys_activity = """
    topResumedActivity=ActivityRecord{fa91d32 u0 com.bandainamcoent.idolmaster_gakuen/com.google.firebase.MessagingUnityPlayerActivity} t185}
    """
    dumpsys_window = """
    mCurrentFocus=Window{81a u0 com.bandainamcoent.idolmaster_gakuen/com.bngames.UnityPlayerActivity}
    """
    shell_calls = []
    device._Android_App__adb_device = SimpleNamespace(
        shell=lambda command: shell_calls.append(command) or (
            dumpsys_activity if command == "dumpsys activity activities" else dumpsys_window
        )
    )
    device._Android_App__u2_device = None
    device._Android_App__package_name = "com.bandainamcoent.idolmaster_gakuen"

    assert device.is_app_focused() is True
    assert shell_calls == [
        "dumpsys activity activities",
    ]
