from types import SimpleNamespace

import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.services.game_utils import GameUtils
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Yolo import Yolo_Box
from src.entity.Yolo import Yolo_Results


class _DeviceStub:
    def __init__(self, app, trigger_on_click: bool):
        self._app = app
        self._trigger_on_click = trigger_on_click
        self.clicks = 0

    def click_element(self, element):
        self.clicks += 1
        if self._trigger_on_click:
            frame = self._app.latest_frame.copy()
            frame[int(element.y):int(element.h), int(element.x):int(element.w)] = 255
            self._app.latest_frame = frame


def _build_app(trigger_on_click: bool):
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    app = SimpleNamespace(latest_frame=frame.copy(), latest_results=SimpleNamespace(frame=frame.copy()))
    app.device = _DeviceStub(app, trigger_on_click=trigger_on_click)
    return app


def _build_button(frame):
    return Yolo_Box(20, 20, 60, 50, BaseUILabels.BUTTON, frame[20:50, 20:60].copy())


def test_click_element_and_wait_trigger_detects_region_change(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=True)
    game_utils = GameUtils(app)
    button = _build_button(app.latest_frame)

    assert game_utils.click_element_and_wait_trigger(button, retries=2, timeout=0.3, interval=0.05)
    assert app.device.clicks == 1


def test_click_element_and_wait_trigger_retries_when_nothing_changes(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    game_utils = GameUtils(app)
    button = _build_button(app.latest_frame)

    assert game_utils.click_element_and_wait_trigger(button, retries=3, timeout=0.2, interval=0.05) is False
    assert app.device.clicks == 3


def _modal(title: str):
    return SimpleNamespace(
        modal_title=title,
        header_box=SimpleNamespace(x=20, y=20, cx=60, cy=20),
        confirm_button=SimpleNamespace(cx=80, cy=95),
        cancel_button=SimpleNamespace(cx=40, cy=95),
    )


def test_wait_modal_transition_returns_when_modal_disappears(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    results_sequence = [
        SimpleNamespace(frame=app.latest_frame.copy()),
        SimpleNamespace(frame=app.latest_frame.copy()),
        SimpleNamespace(frame=app.latest_frame.copy()),
        SimpleNamespace(frame=app.latest_frame.copy()),
    ]
    app.latest_results = results_sequence[0]
    game_utils = GameUtils(app)
    modal_states = iter([
        _modal("購入確認"),
        _modal("購入確認"),
        None,
        None,
    ])

    def _try_get_modal(no_body=True):
        state = next(modal_states)
        if results_sequence:
            app.latest_results = results_sequence.pop(0)
        return state

    monkeypatch.setattr(game_utils, "try_get_modal", _try_get_modal)

    assert game_utils.wait_modal_transition(previous_modal_title="購入確認", timeout=0.5, interval=0.05) is True


def test_click_modal_button_and_wait_transition_uses_current_modal_title(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    results_sequence = [
        SimpleNamespace(frame=app.latest_frame.copy()),
        SimpleNamespace(frame=app.latest_frame.copy()),
        SimpleNamespace(frame=app.latest_frame.copy()),
        SimpleNamespace(frame=app.latest_frame.copy()),
    ]
    app.latest_results = results_sequence[0]
    game_utils = GameUtils(app)
    button = _build_button(app.latest_frame)
    modal_states = iter([
        _modal("購入確認"),
        _modal("購入確認"),
        None,
        None,
    ])

    def _try_get_modal(no_body=True):
        state = next(modal_states)
        if results_sequence:
            app.latest_results = results_sequence.pop(0)
        return state

    monkeypatch.setattr(game_utils, "try_get_modal", _try_get_modal)
    monkeypatch.setattr(game_utils, "click_element_and_wait_trigger", lambda *_args, **_kwargs: True)

    assert game_utils.click_modal_button_and_wait_transition(button, timeout=0.5, interval=0.05) is True


def test_wait_for_modal_requires_stable_modal(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    frame = app.latest_frame.copy()
    app.latest_results = SimpleNamespace(frame=frame.copy())
    game_utils = GameUtils(app)
    modal_states = iter([
        _modal("交換確認"),
        _modal("交換確認"),
    ])
    result_states = iter([
        SimpleNamespace(frame=frame.copy()),
        SimpleNamespace(frame=frame.copy()),
    ])

    def _try_get_modal(no_body=False, require_header=True):
        app.latest_results = next(result_states)
        return next(modal_states)

    monkeypatch.setattr(game_utils, "try_get_modal", _try_get_modal)

    modal = game_utils.wait_for_modal("交換確認", timeout=0.3, interval=0.05)
    assert modal is not None
    assert modal.modal_title == "交換確認"


def test_wait_for_modal_resets_when_modal_position_changes(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    frame = app.latest_frame.copy()
    app.latest_results = SimpleNamespace(frame=frame.copy())
    game_utils = GameUtils(app)

    moving_modal = SimpleNamespace(
        modal_title="交換確認",
        header_box=SimpleNamespace(x=20, y=4, cx=140, cy=20),
        confirm_button=SimpleNamespace(cx=160, cy=95),
        cancel_button=SimpleNamespace(cx=120, cy=95),
    )
    modal_values = [
        _modal("交換確認"),
        moving_modal,
    ]
    result_values = [
        SimpleNamespace(frame=frame.copy()),
        SimpleNamespace(frame=frame.copy()),
    ]
    state = {"index": 0}

    def _try_get_modal(no_body=False, require_header=True):
        index = state["index"]
        app.latest_results = result_values[index % len(result_values)]
        modal = modal_values[index % len(modal_values)]
        state["index"] += 1
        return modal

    monkeypatch.setattr(game_utils, "try_get_modal", _try_get_modal)

    assert game_utils.wait_for_modal("交換確認", timeout=0.3, interval=0.05) is None





def test_back_next_page_waits_for_visible_transition(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=True)
    app.game_status_manager = SimpleNamespace(current_location=GamePageTypes.UNKNOWN)
    back_button = Yolo_Box(20, 20, 60, 50, BaseUILabels.BACK_BTN, app.latest_frame[20:50, 20:60].copy())
    app.latest_results = SimpleNamespace(
        frame=app.latest_frame.copy(),
        filter_by_label=lambda label: Yolo_Results.from_boxes([back_button]) if label == BaseUILabels.BACK_BTN else Yolo_Results.from_boxes([]),
    )
    game_utils = GameUtils(app)
    monkeypatch.setattr(game_utils, "update_current_location", lambda new_location=None: GamePageTypes.UNKNOWN)

    assert game_utils.back_next_page() is True
    assert app.device.clicks == 1


def test_back_next_page_raises_when_click_does_not_change_ui(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    app.game_status_manager = SimpleNamespace(current_location=GamePageTypes.UNKNOWN)
    back_button = Yolo_Box(20, 20, 60, 50, BaseUILabels.BACK_BTN, app.latest_frame[20:50, 20:60].copy())
    app.latest_results = SimpleNamespace(
        frame=app.latest_frame.copy(),
        filter_by_label=lambda label: Yolo_Results.from_boxes([back_button]) if label == BaseUILabels.BACK_BTN else Yolo_Results.from_boxes([]),
    )
    game_utils = GameUtils(app)
    monkeypatch.setattr(game_utils, "update_current_location", lambda new_location=None: GamePageTypes.UNKNOWN)

    try:
        game_utils.back_next_page()
    except TimeoutError as exc:
        assert "did not trigger page transition" in str(exc)
    else:
        raise AssertionError("Expected back_next_page to raise when UI does not change")


def test_go_home_tries_back_button_when_go_home_click_does_not_reach_home(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    app.game_status_manager = SimpleNamespace(current_location=GamePageTypes.HOME_TAB.GIFT)
    go_home_button = Yolo_Box(20, 20, 60, 50, BaseUILabels.GO_HOME_BTN, app.latest_frame[20:50, 20:60].copy())
    back_button = Yolo_Box(70, 20, 100, 50, BaseUILabels.BACK_BTN, app.latest_frame[20:50, 70:100].copy())
    app.latest_results = SimpleNamespace(
        frame=app.latest_frame.copy(),
        filter_by_label=lambda label: (
            Yolo_Results.from_boxes([go_home_button]) if label == BaseUILabels.GO_HOME_BTN
            else Yolo_Results.from_boxes([back_button]) if label == BaseUILabels.BACK_BTN
            else Yolo_Results.from_boxes([])
        ),
    )
    game_utils = GameUtils(app)

    state = {"location": GamePageTypes.HOME_TAB.GIFT}
    clicks = []
    wait_loading_calls = []

    def _update_current_location(new_location=None):
        if new_location is not None:
            state["location"] = new_location
        app.game_status_manager.current_location = state["location"]
        return state["location"]

    def _click_element_and_wait_trigger(element, **_kwargs):
        clicks.append(element.label)
        if element.label == BaseUILabels.BACK_BTN:
            state["location"] = GamePageTypes.MAIN_MENU__HOME
        return True

    monkeypatch.setattr(game_utils, "update_current_location", _update_current_location)
    monkeypatch.setattr(game_utils, "click_element_and_wait_trigger", _click_element_and_wait_trigger)
    monkeypatch.setattr(game_utils, "wait_loading", lambda *args, **kwargs: wait_loading_calls.append(state["location"]))

    game_utils.go_home(max_try=2)

    assert clicks == [BaseUILabels.GO_HOME_BTN, BaseUILabels.BACK_BTN]
    assert wait_loading_calls == [GamePageTypes.HOME_TAB.GIFT, GamePageTypes.MAIN_MENU__HOME]
    assert app.game_status_manager.current_location == GamePageTypes.MAIN_MENU__HOME


def test_back_next_page_uses_cancel_on_memory_detail(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    app.game_status_manager = SimpleNamespace(current_location=GamePageTypes.PRODUCER__MEMORY_DETAIL)
    app.latest_results = SimpleNamespace(
        frame=app.latest_frame.copy(),
        filter_by_label=lambda _label: Yolo_Results.from_boxes([]),
    )
    game_utils = GameUtils(app)

    state = {"location": GamePageTypes.PRODUCER__MEMORY_DETAIL}
    cancel_button = SimpleNamespace(label="cancel", text="キャンセル", x=20, y=20, w=60, h=50, cx=40, cy=35)
    clicks = []

    monkeypatch.setattr(game_utils, "update_current_location", lambda new_location=None: state["location"])
    monkeypatch.setattr(
        game_utils,
        "_find_button_by_text",
        lambda text, **_kwargs: cancel_button if text == "キャンセル" else None,
    )
    monkeypatch.setattr(
        game_utils,
        "click_element_and_wait_trigger",
        lambda element, **_kwargs: clicks.append(element.label) or state.__setitem__("location", GamePageTypes.PRODUCER__MEMORY_CANDIDATE_LIST) or True,
    )
    monkeypatch.setattr(game_utils, "wait_loading", lambda *args, **kwargs: None)

    assert game_utils.back_next_page() is True
    assert clicks == ["cancel"]


def test_go_home_closes_memory_detail_before_using_back_button(monkeypatch):
    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    app = _build_app(trigger_on_click=False)
    app.game_status_manager = SimpleNamespace(current_location=GamePageTypes.PRODUCER__MEMORY_DETAIL)
    back_button = Yolo_Box(70, 20, 100, 50, BaseUILabels.BACK_BTN, app.latest_frame[20:50, 70:100].copy())
    state = {"location": GamePageTypes.PRODUCER__MEMORY_DETAIL}

    def _filter_by_label(label):
        if label == BaseUILabels.BACK_BTN and state["location"] == GamePageTypes.PRODUCER__MEMORY_SELECTION:
            return Yolo_Results.from_boxes([back_button])
        return Yolo_Results.from_boxes([])

    app.latest_results = SimpleNamespace(
        frame=app.latest_frame.copy(),
        filter_by_label=_filter_by_label,
    )
    game_utils = GameUtils(app)

    cancel_button = SimpleNamespace(label="cancel", text="キャンセル", x=20, y=20, w=60, h=50, cx=40, cy=35)
    clicks = []
    wait_loading_calls = []

    def _update_current_location(new_location=None):
        if new_location is not None:
            state["location"] = new_location
        app.game_status_manager.current_location = state["location"]
        return state["location"]

    def _click_element_and_wait_trigger(element, **_kwargs):
        clicks.append(element.label)
        if element.label == "cancel":
            state["location"] = GamePageTypes.PRODUCER__MEMORY_SELECTION
        elif element.label == BaseUILabels.BACK_BTN:
            state["location"] = GamePageTypes.MAIN_MENU__HOME
        return True

    monkeypatch.setattr(game_utils, "update_current_location", _update_current_location)
    monkeypatch.setattr(
        game_utils,
        "_find_button_by_text",
        lambda text, **_kwargs: cancel_button if text == "キャンセル" and state["location"] == GamePageTypes.PRODUCER__MEMORY_DETAIL else None,
    )
    monkeypatch.setattr(game_utils, "click_element_and_wait_trigger", _click_element_and_wait_trigger)
    monkeypatch.setattr(game_utils, "wait_loading", lambda *args, **kwargs: wait_loading_calls.append(state["location"]))

    game_utils.go_home(max_try=3)

    assert clicks == ["cancel", BaseUILabels.BACK_BTN]
    assert wait_loading_calls == [GamePageTypes.PRODUCER__MEMORY_SELECTION, GamePageTypes.MAIN_MENU__HOME]
    assert app.game_status_manager.current_location == GamePageTypes.MAIN_MENU__HOME
