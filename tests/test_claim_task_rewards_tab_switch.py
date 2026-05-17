import sys
from types import SimpleNamespace

import numpy as np


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.tasks.base_ui import claim_task_rewards


def test_switch_to_tab_avoids_extra_click_when_trigger_detected(monkeypatch):
    selected_states = iter([True])
    monkeypatch.setattr(
        claim_task_rewards,
        "_is_tab_selected_by_highlight",
        lambda _app, _tab: next(selected_states),
    )

    trigger_calls = []
    clicked = []
    wait_calls = []
    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            click_element_and_wait_trigger=lambda *_args, **_kwargs: trigger_calls.append("trigger") or True,
            wait_frame_stable=lambda **kwargs: wait_calls.append(kwargs),
        ),
        device=SimpleNamespace(click_element=lambda _tab: clicked.append("click")),
    )

    claim_task_rewards._switch_to_tab(
        app,
        tab_item=SimpleNamespace(x=100, y=100, w=200, h=140, text="ノーマル"),
    )

    assert trigger_calls == ["trigger"]
    assert clicked == []
    assert len(wait_calls) == 1


def test_switch_to_tab_fallback_click_when_trigger_not_detected(monkeypatch):
    selected_states = iter([False, False, True])
    monkeypatch.setattr(
        claim_task_rewards,
        "_is_tab_selected_by_highlight",
        lambda _app, _tab: next(selected_states),
    )

    trigger_calls = []
    clicked = []
    wait_calls = []
    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            click_element_and_wait_trigger=lambda *_args, **_kwargs: trigger_calls.append("trigger") or False,
            wait_frame_stable=lambda **kwargs: wait_calls.append(kwargs),
        ),
        device=SimpleNamespace(click_element=lambda _tab: clicked.append("click")),
    )

    claim_task_rewards._switch_to_tab(
        app,
        tab_item=SimpleNamespace(x=100, y=100, w=200, h=140, text="期間限定"),
    )

    assert trigger_calls == ["trigger"]
    assert clicked == ["click"]
    assert len(wait_calls) == 2


def test_switch_to_tab_still_clicks_once_when_already_selected(monkeypatch):
    monkeypatch.setattr(
        claim_task_rewards,
        "_is_tab_selected_by_highlight",
        lambda _app, _tab: True,
    )

    trigger_calls = []
    clicked = []
    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            click_element_and_wait_trigger=lambda *_args, **_kwargs: trigger_calls.append("trigger") or True,
            wait_frame_stable=lambda **_kwargs: None,
        ),
        device=SimpleNamespace(click_element=lambda _tab: clicked.append("click")),
    )

    claim_task_rewards._switch_to_tab(
        app,
        tab_item=SimpleNamespace(x=100, y=100, w=200, h=140, text="デイリー"),
    )

    assert trigger_calls == ["trigger"]
    assert clicked == []


def test_is_tab_selected_by_highlight_returns_false_for_empty_frame():
    app = SimpleNamespace(latest_frame=None)
    tab_item = SimpleNamespace(x=100, y=100, w=200, h=140)

    assert claim_task_rewards._is_tab_selected_by_highlight(app, tab_item) is False


def test_is_tab_selected_by_highlight_uses_cropped_region(monkeypatch):
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    app = SimpleNamespace(latest_frame=frame)
    tab_item = SimpleNamespace(x=50, y=320, w=110, h=350)
    cropped_shapes = []

    def _fake_selected(frame_region, threshold=0.07):
        cropped_shapes.append(frame_region.shape)
        return True

    monkeypatch.setattr(claim_task_rewards, "_is_selected_tab_frame", _fake_selected)

    assert claim_task_rewards._is_tab_selected_by_highlight(app, tab_item) is True
    assert cropped_shapes and cropped_shapes[0][0] > 0 and cropped_shapes[0][1] > 0
