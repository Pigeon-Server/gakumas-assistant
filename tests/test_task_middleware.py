import sys
from types import SimpleNamespace

import numpy as np


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.middlewares import middleware_register


class _TaskQueueStub:
    def __init__(self):
        self.middlewares = []
        self.inserted = []

    def register_task_middleware(self):
        def decorator(func):
            self.middlewares.append(func)
            return func

        return decorator

    def insert_task_to_run_queue(self, task_id):
        self.inserted.append(task_id)


class _ButtonStub:
    def __init__(self, name: str):
        self.name = name


class _ModalStub:
    def __init__(self, title: str):
        self.modal_title = title
        self.cancel_button = _ButtonStub("cancel")
        self.confirm_button = _ButtonStub("confirm")


class _ResultsStub:
    def __init__(self, has_modal: bool):
        self._has_modal = has_modal
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self._confirm_button = _ButtonStub("confirm")

    def exists_label(self, label):
        return self._has_modal and label == BaseUILabels.MODAL_HEADER

    def filter_by_label(self, label: str):
        if label in {"Universal Confirm button", "Universal Cancel button", "Universal button"}:
            return [self._confirm_button]
        return []


def _build_app(has_modal: bool):
    app = SimpleNamespace()
    app.latest_results = _ResultsStub(has_modal)
    app.device = SimpleNamespace(clicked=[])
    app.device.click_element = lambda button: app.device.clicked.append(button.name)
    app.game_utils = SimpleNamespace(calls=[])
    app.game_utils.wait_loading = lambda: app.game_utils.calls.append("wait_loading")
    app.game_utils.wait_for_label = lambda label: app.game_utils.calls.append(("wait_for_label", label))
    app.task_queue = _TaskQueueStub()
    app.game_status_manager = SimpleNamespace(current_location=None)
    return app


def test_middleware_handles_connection_error_and_blocks_task(monkeypatch):
    processor = SimpleNamespace(task_queue=_TaskQueueStub())
    middleware_register.register_middlewares(processor)
    _, handle_modal = processor.task_queue.middlewares
    middleware_register.last_modal = False
    monkeypatch.setattr(middleware_register, "get_modal", lambda *_args, **_kwargs: _ModalStub(ModalText.TITLE.CONNECTION_ERROR))

    app = _build_app(has_modal=True)

    assert handle_modal(app) is True
    assert app.device.clicked == ["confirm"]
    assert app.game_utils.calls == ["wait_loading"]
    assert app.task_queue.inserted == []


def test_middleware_ocr_uses_frame_not_yolo_results(monkeypatch):
    processor = SimpleNamespace(task_queue=_TaskQueueStub())
    middleware_register.register_middlewares(processor)
    _, handle_modal = processor.task_queue.middlewares
    middleware_register.last_modal = False

    called = {}

    class _OCRServiceStub:
        def ocr(self, img):
            called["img"] = img
            assert isinstance(img, np.ndarray)
            return [SimpleNamespace(text=ModalText.TITLE.CONNECTION_ERROR)]

    monkeypatch.setattr("src.core.inference.ocr_engine.OCRService", _OCRServiceStub)

    app = _build_app(has_modal=True)

    assert handle_modal(app) is True
    assert called["img"] is app.latest_results.frame


def test_middleware_skips_when_latest_results_is_none():
    processor = SimpleNamespace(task_queue=_TaskQueueStub())
    middleware_register.register_middlewares(processor)
    _, handle_modal = processor.task_queue.middlewares
    middleware_register.last_modal = True
    middleware_register.last_modal_title = "前回弹窗"

    app = _build_app(has_modal=False)
    app.latest_results = None

    assert handle_modal(app) is True
    assert middleware_register.last_modal is True
    assert middleware_register.last_modal_title == "前回弹窗"


def test_middleware_parses_unknown_modal_once_until_it_disappears(monkeypatch):
    processor = SimpleNamespace(task_queue=_TaskQueueStub())
    middleware_register.register_middlewares(processor)
    _, handle_modal = processor.task_queue.middlewares
    middleware_register.last_modal = False
    get_modal_calls = {"count": 0}

    def _mock_get_modal(*_args, **_kwargs):
        get_modal_calls["count"] += 1
        return _ModalStub("親愛度アイドル選択")

    monkeypatch.setattr(middleware_register, "get_modal", _mock_get_modal)

    app = _build_app(has_modal=True)

    assert handle_modal(app) is True
    assert handle_modal(app) is True
    assert app.device.clicked == []
    assert get_modal_calls["count"] == 1

    app.latest_results = SimpleNamespace(exists_label=lambda _label: False)
    assert handle_modal(app) is True
    assert middleware_register.last_modal is False


def test_middleware_restarts_game_for_update_modal_when_not_already_running_start_game(monkeypatch):
    processor = SimpleNamespace(task_queue=_TaskQueueStub())
    middleware_register.register_middlewares(processor)
    _, handle_modal = processor.task_queue.middlewares
    middleware_register.last_modal = False
    monkeypatch.setattr(middleware_register, "get_modal", lambda *_args, **_kwargs: _ModalStub(ModalText.TITLE.DATA_UPDATE))

    app = _build_app(has_modal=True)

    assert handle_modal(app) is True
    assert app.device.clicked == ["cancel"]
    assert app.game_utils.calls == ["wait_loading", ("wait_for_label", BaseUILabels.START_MENU_LOGO)]
    assert app.task_queue.inserted == ["start_game"]


def test_middleware_reinserts_start_game_when_update_modal_appears(monkeypatch):
    processor = SimpleNamespace(task_queue=_TaskQueueStub())
    middleware_register.register_middlewares(processor)
    _, handle_modal = processor.task_queue.middlewares
    middleware_register.last_modal = False
    monkeypatch.setattr(middleware_register, "get_modal", lambda *_args, **_kwargs: _ModalStub(ModalText.TITLE.DATA_UPDATE))

    app = _build_app(has_modal=True)

    assert handle_modal(app) is True
    assert app.device.clicked == ["cancel"]
    assert app.game_utils.calls == ["wait_loading", ("wait_for_label", BaseUILabels.START_MENU_LOGO)]
    assert app.task_queue.inserted == ["start_game"]
