import sys
from types import SimpleNamespace

import numpy as np


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCR_Result, OCR_ResultList
from src.core.tasks.base_ui import dispatch_work
from src.entity.Yolo import Yolo_Box, Yolo_Results


def test_dispatch_single_work_skips_rejected_avatar_during_fallback(monkeypatch):
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    avatar1 = Yolo_Box(10, 10, 30, 30, BaseUILabels.AVATAR, frame[10:30, 10:30])
    avatar2 = Yolo_Box(40, 10, 60, 30, BaseUILabels.AVATAR, frame[10:30, 40:60])

    class _LatestResults:
        def filter_by_label(self, label):
            if label == BaseUILabels.AVATAR:
                return Yolo_Results.from_boxes([avatar1, avatar2])
            return Yolo_Results.from_boxes([])

    assign_calls = []

    monkeypatch.setattr(dispatch_work, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dispatch_work, "check_color", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(dispatch_work, "_is_avatar_guaranteed_success", lambda avatar: avatar == avatar1)
    monkeypatch.setattr(dispatch_work, "check_frame_change", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "src.entity.Yolo.Yolo_Results.group_yolo_boxes_by_position",
        lambda self, *_args, **_kwargs: [self],
    )

    def _assign_avatar(_app, avatar=None):
        assign_calls.append(avatar)
        return avatar == avatar2

    monkeypatch.setattr(dispatch_work, "_assign_avatar_to_work", _assign_avatar)

    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_LatestResults(),
        device=SimpleNamespace(scrollY=lambda *_args, **_kwargs: None),
        game_utils=SimpleNamespace(
            wait_loading=lambda: None,
            wait_for_label=lambda *_args, **_kwargs: True,
            wait_frame_stable=lambda: True,
        ),
        debug_tools=SimpleNamespace(
            clear_all=lambda: None,
            add_box=lambda *_args, **_kwargs: None,
            hide=lambda: None,
        ),
    )

    dispatch_work._dispatch_single_work(app)

    assert assign_calls == [avatar1, avatar2]


def test_dispatch_single_work_returns_false_when_avatar_missing(monkeypatch):
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    class _LatestResults:
        def filter_by_label(self, _label):
            return Yolo_Results.from_boxes([])

    monkeypatch.setattr(dispatch_work, "sleep", lambda *_args, **_kwargs: None)

    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_LatestResults(),
        device=SimpleNamespace(scrollY=lambda *_args, **_kwargs: None),
        game_utils=SimpleNamespace(
            wait_loading=lambda: None,
            wait_for_label=lambda *_args, **_kwargs: False,
            try_get_modal=lambda *_args, **_kwargs: None,
            wait_frame_stable=lambda: True,
        ),
        debug_tools=SimpleNamespace(
            clear_all=lambda: None,
            add_box=lambda *_args, **_kwargs: None,
            hide=lambda: None,
        ),
    )

    assert dispatch_work._dispatch_single_work(app) is False


def test_dispatch_single_work_recovers_connection_error_modal(monkeypatch):
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    click_calls = []

    class _LatestResults:
        def filter_by_label(self, _label):
            return Yolo_Results.from_boxes([])

    modal = SimpleNamespace(
        modal_title="通信エラー",
        confirm_button=SimpleNamespace(name="retry"),
        cancel_button=None,
    )

    wait_results = iter([False, False])

    monkeypatch.setattr(dispatch_work, "sleep", lambda *_args, **_kwargs: None)

    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_LatestResults(),
        device=SimpleNamespace(
            click_element=lambda element: click_calls.append(element),
            scrollY=lambda *_args, **_kwargs: None,
        ),
        game_utils=SimpleNamespace(
            wait_loading=lambda: None,
            wait_for_label=lambda *_args, **_kwargs: next(wait_results),
            try_get_modal=lambda *_args, **_kwargs: modal,
            wait_frame_stable=lambda: True,
        ),
        debug_tools=SimpleNamespace(
            clear_all=lambda: None,
            add_box=lambda *_args, **_kwargs: None,
            hide=lambda: None,
        ),
    )

    assert dispatch_work._dispatch_single_work(app) is False
    assert click_calls == [modal.confirm_button]


def test_get_work_action_button_falls_back_to_bottom_ocr(monkeypatch):
    frame = np.full((2400, 1080, 3), 255, dtype=np.uint8)

    class _LatestResults:
        def filter_by_label(self, _label):
            return Yolo_Results.from_boxes([])

    def _fake_ocr(_image):
        return OCR_ResultList([OCR_Result(430, 420, 220, 40, "選択する", 0.99)])

    monkeypatch.setattr(dispatch_work.ocr_service, "ocr", _fake_ocr)

    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_LatestResults(),
    )

    button = dispatch_work._get_work_action_button(app, ("選択する",), timeout=0)

    assert button is not None
    assert 500 <= int(button.cx) <= 580
    assert 1980 <= int(button.y) <= 2060


def test_assign_avatar_to_work_returns_false_when_action_button_missing(monkeypatch):
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    avatar = Yolo_Box(10, 10, 40, 40, BaseUILabels.AVATAR, frame[10:40, 10:40])
    click_calls = []

    monkeypatch.setattr(dispatch_work, "_get_work_action_button", lambda *_args, **_kwargs: None)

    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=SimpleNamespace(exists_label=lambda *_args, **_kwargs: True),
        device=SimpleNamespace(click_element=lambda element: click_calls.append(element)),
        game_utils=SimpleNamespace(try_get_modal=lambda *_args, **_kwargs: None),
        debug_tools=SimpleNamespace(hide=lambda: None),
    )

    assert dispatch_work._assign_avatar_to_work(app, avatar) is False
    assert click_calls == [avatar]


def test_select_work_duration_returns_false_when_no_duration_candidates(monkeypatch):
    frame = np.zeros((240, 120, 3), dtype=np.uint8)
    click_calls = []

    monkeypatch.setattr(dispatch_work, "_get_work_action_button", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dispatch_work.ocr_service, "ocr", lambda *_args, **_kwargs: [])

    config = SimpleNamespace(
        task__dispatch_work=SimpleNamespace(
            reconfigure_work_hours=SimpleNamespace(value=True),
            working_hours=SimpleNamespace(value="8H"),
        )
    )
    app = SimpleNamespace(
        latest_frame=frame,
        device=SimpleNamespace(click_element=lambda element: click_calls.append(element)),
        config_service=lambda: config,
    )

    assert dispatch_work._select_work_duration(app) is False
    assert click_calls == []
