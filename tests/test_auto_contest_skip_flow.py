import sys
from types import SimpleNamespace

import numpy as np
import pytest
import cv2


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _DebugToolsStub:
    def clear_all(self):
        return None


sys.modules["src.utils.logger"] = SimpleNamespace(logger=_LoggerStub())
sys.modules["src.utils.debug_tools"] = SimpleNamespace(DebugTools=_DebugToolsStub)
sys.modules["src.entity.Game.Components.Button"] = SimpleNamespace(ButtonList=object, Button=object)
sys.modules["src.entity.Game.Components.CheckBox"] = SimpleNamespace(CheckBox=object)
sys.modules["src.entity.Game.Components.Contest"] = SimpleNamespace(ContestList=object, ContestItem=object)

from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.base_ui import auto_contest, goto_pages
from src.utils.contest_overlay_tools import (
    ContestGradeUpSplash,
    ContestSeasonOverlay,
    detect_contest_grade_up_splash,
    detect_contest_season_overlay,
)
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.task_debug_tools import get_task_debug_trace
from src.utils.string_tools import string_match


class _BoxList:
    def __init__(self, visible: bool, label: str):
        self._visible = visible
        self._label = label

    def __bool__(self):
        return self._visible

    def first(self):
        if not self._visible:
            raise LookupError(self._label)
        return f"{self._label}-box"


class _SequencedResults:
    def __init__(self, states):
        self._states = states
        self._index = 0

    @property
    def state(self):
        return self._states[min(self._index, len(self._states) - 1)]

    def advance(self):
        self._index += 1

    def filter_by_label(self, label):
        return _BoxList(self.state.get(label, False), label)

    def exists_label(self, label):
        return bool(self.state.get(label, False))


class _DeviceStub:
    def __init__(self):
        self.clicked_elements = []
        self.clicked_points = []

    def click_element(self, element):
        self.clicked_elements.append(element)

    def click(self, x, y):
        self.clicked_points.append((x, y))


class _ButtonStub:
    def __init__(self, x, y, w, h, text="", disabled=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = (x + w) // 2
        self.cy = (y + h) // 2
        self.text = text
        self._disabled = disabled

    def is_disabled(self):
        return self._disabled


class _ButtonListStub:
    def __init__(self, latest_results):
        self._latest_results = latest_results
        self.buttons = list(self._latest_results.state.get("buttons", []))

    def __bool__(self):
        return bool(self.buttons)

    def __len__(self):
        return len(self.buttons)

    def __iter__(self):
        return iter(self.buttons)

    def get_button_by_text(self, text, match_config=None):
        for button in self.buttons:
            if string_match(button.text or "", text, match_config):
                return button

        button_text = self._latest_results.state.get("button_text")
        if button_text is not None and string_match(button_text, text, match_config):
            return f"{button_text}-button"
        return None


# ── _click_skip_until_disappears tests ──


def test_click_skip_waits_for_stable_disappearance(monkeypatch):
    latest_results = _SequencedResults(
        [
            {BaseUILabels.SKIP_BUTTON: True},
            {BaseUILabels.SKIP_BUTTON: False},
            {BaseUILabels.SKIP_BUTTON: True},
            {BaseUILabels.SKIP_BUTTON: False},
            {BaseUILabels.SKIP_BUTTON: False},
            {BaseUILabels.SKIP_BUTTON: False},
        ]
    )
    device = _DeviceStub()
    app = SimpleNamespace(latest_results=latest_results, device=device)

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())

    auto_contest._click_skip_until_disappears(app, timeout=5, interval=0.1, stable_missing=3)

    assert device.clicked_elements == [
        f"{BaseUILabels.SKIP_BUTTON}-box",
        f"{BaseUILabels.SKIP_BUTTON}-box",
    ]


def test_click_skip_raises_timeout_if_button_never_disappears(monkeypatch):
    latest_results = _SequencedResults([{BaseUILabels.SKIP_BUTTON: True}] * 10)
    device = _DeviceStub()
    app = SimpleNamespace(latest_results=latest_results, device=device)

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())

    with pytest.raises(TimeoutError, match="skip button"):
        auto_contest._click_skip_until_disappears(app, timeout=0.3, interval=0.1, stable_missing=3)


# ── _finish_battle tests ──


def test_finish_battle_phase1_center_taps_then_clicks_next(monkeypatch):
    """Phase 1: center taps until NEXT appears, then returns once arena UI is back."""
    latest_results = _SequencedResults([
        {},
        {},
        {BaseUILabels.BUTTON: True, "button_text": ButtonText.NEXT},
        {BaseUILabels.BACK_BTN: True},
    ])
    device = _DeviceStub()
    click_button_calls = []

    def _mock_click_button(text):
        click_button_calls.append(text)
        latest_results.advance()

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            click_button=_mock_click_button,
            wait_for_modal=lambda *a, **k: None,
        ),
    )

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())
    monkeypatch.setattr(auto_contest, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(auto_contest, "_is_contest_detail_page", lambda app: False)

    auto_contest._finish_battle(app)

    assert len(device.clicked_points) == 2
    assert device.clicked_elements == [f"{ButtonText.NEXT}-button"]
    assert click_button_calls == []


def test_finish_battle_phase1_waits_past_old_iteration_cap(monkeypatch):
    """Phase 1 keeps waiting until NEXT appears, even after more than 15 taps."""
    latest_results = _SequencedResults(([{}] * 16) + [
        {BaseUILabels.BUTTON: True, "button_text": ButtonText.NEXT},
        {BaseUILabels.BACK_BTN: True},
    ])
    device = _DeviceStub()
    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(),
    )

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())
    monkeypatch.setattr(auto_contest, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(auto_contest, "_is_contest_detail_page", lambda app: False)

    auto_contest._finish_battle(app)

    assert len(device.clicked_points) == 16
    assert device.clicked_elements == [f"{ButtonText.NEXT}-button"]


def test_finish_battle_full_flow(monkeypatch):
    """Full flow: NEXT → arena BACK_BTN appears → return."""
    latest_results = _SequencedResults([
        {BaseUILabels.BUTTON: True, "button_text": ButtonText.NEXT},
        {BaseUILabels.BACK_BTN: True},
    ])
    device = _DeviceStub()
    click_button_calls = []

    def _mock_click_button(text):
        click_button_calls.append(text)
        latest_results.advance()

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            click_button=_mock_click_button,
            wait_for_modal=lambda *a, **k: None,
        ),
    )

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())
    monkeypatch.setattr(auto_contest, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(auto_contest, "_is_contest_detail_page", lambda app: False)

    auto_contest._finish_battle(app)

    assert device.clicked_elements == [f"{ButtonText.NEXT}-button"]
    assert click_button_calls == []
    assert device.clicked_points == []


def test_finish_battle_phase3_handles_modal(monkeypatch):
    """Phase 3: modal detected → close it → BACK_BTN → return."""
    modal_stub = SimpleNamespace(
        modal_title="評価報酬",
        cancel_button="cancel-btn",
        confirm_button="confirm-btn",
    )
    latest_results = _SequencedResults([
        {BaseUILabels.BUTTON: True, "button_text": ButtonText.NEXT},
        {BaseUILabels.MODAL_HEADER: True},
        {BaseUILabels.BACK_BTN: True},
    ])
    device = _DeviceStub()

    def _mock_click_button(text):
        latest_results.advance()

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            click_button=_mock_click_button,
            wait_for_modal=lambda *a, **k: modal_stub,
        ),
    )

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())
    monkeypatch.setattr(auto_contest, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(auto_contest, "_is_contest_detail_page", lambda app: False)

    auto_contest._finish_battle(app)

    assert device.clicked_elements == [
        f"{ButtonText.NEXT}-button",
        "cancel-btn",
    ]


def test_finish_battle_phase3_detail_page_clicks_back(monkeypatch):
    """Phase 3: BACK_BTN + detail page → click back, then return on arena list."""
    latest_results = _SequencedResults([
        {BaseUILabels.BUTTON: True, "button_text": ButtonText.NEXT},
        {BaseUILabels.BACK_BTN: True},
        {BaseUILabels.BACK_BTN: True},
    ])
    device = _DeviceStub()
    back_called = {"count": 0}
    detail_calls = {"n": 0}

    def _mock_click_button(text):
        latest_results.advance()

    def _try_back(app):
        back_called["count"] += 1
        return True

    def _is_detail(app):
        detail_calls["n"] += 1
        return detail_calls["n"] == 1

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            click_button=_mock_click_button,
            wait_for_modal=lambda *a, **k: None,
        ),
    )

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())
    monkeypatch.setattr(auto_contest, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(auto_contest, "_is_contest_detail_page", _is_detail)
    monkeypatch.setattr(auto_contest, "_try_back_to_contest_list", _try_back)

    auto_contest._finish_battle(app)

    assert back_called["count"] == 1


def test_finish_battle_phase3_back_btn_returns_immediately(monkeypatch):
    """Phase 3: BACK_BTN (not detail page) → return immediately."""
    latest_results = _SequencedResults([
        {BaseUILabels.BUTTON: True, "button_text": ButtonText.NEXT},
        {BaseUILabels.BACK_BTN: True},
    ])
    device = _DeviceStub()

    def _mock_click_button(text):
        latest_results.advance()

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            click_button=_mock_click_button,
            wait_for_modal=lambda *a, **k: None,
        ),
    )

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())
    monkeypatch.setattr(auto_contest, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(auto_contest, "_is_contest_detail_page", lambda app: False)

    auto_contest._finish_battle(app)

    assert device.clicked_elements == [f"{ButtonText.NEXT}-button"]
    assert device.clicked_points == []


def test_finish_battle_phase2_falls_back_to_bottom_primary_button(monkeypatch):
    """Phase 2: EXIT OCR is blank → click the bottom primary button fallback."""
    fallback_button = _ButtonStub(2, 6, 6, 8, text="")
    latest_results = _SequencedResults([
        {BaseUILabels.BUTTON: True, "button_text": ButtonText.NEXT},
        {BaseUILabels.BUTTON: True, "buttons": [fallback_button]},
        {BaseUILabels.BACK_BTN: True},
    ])
    device = _DeviceStub()
    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            click_button=lambda *_args, **_kwargs: None,
            wait_for_modal=lambda *a, **k: None,
        ),
    )

    monkeypatch.setattr(auto_contest, "sleep", lambda *_: latest_results.advance())
    monkeypatch.setattr(auto_contest, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(auto_contest, "_is_contest_detail_page", lambda app: False)

    auto_contest._finish_battle(app)

    assert device.clicked_elements == [f"{ButtonText.NEXT}-button", fallback_button]


def test_get_contest_entry_button_matches_merged_ocr_text(monkeypatch):
    contest_button = _ButtonStub(4, 5, 9, 8, text="1705レートPt4コンテスト210GRADE")
    app = SimpleNamespace(
        latest_results=_SequencedResults([{"buttons": [contest_button]}]),
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
    )

    monkeypatch.setattr(goto_pages, "ButtonList", _ButtonListStub)

    assert goto_pages._get_contest_entry_button(app) is contest_button


def test_get_contest_entry_button_falls_back_to_large_right_tile(monkeypatch):
    left_button = _ButtonStub(0, 5, 3, 8, text="Pランキング")
    road_button = _ButtonStub(0, 6, 3, 9, text="アイドルへの道")
    contest_button = _ButtonStub(6, 6, 10, 9, text="")
    app = SimpleNamespace(
        latest_results=_SequencedResults([{"buttons": [left_button, road_button, contest_button]}]),
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
    )

    monkeypatch.setattr(goto_pages, "ButtonList", _ButtonListStub)

    assert goto_pages._get_contest_entry_button(app) is contest_button


def test_get_contest_entry_button_matches_challenging_status_text(monkeypatch):
    challenge_button = _ButtonStub(1, 5, 5, 9, text="Sアイドルへの道挑戦中")
    other_button = _ButtonStub(6, 5, 9, 8, text="ランキング")
    app = SimpleNamespace(
        latest_results=_SequencedResults([{"buttons": [other_button, challenge_button]}]),
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
    )

    monkeypatch.setattr(goto_pages, "ButtonList", _ButtonListStub)

    assert goto_pages._get_contest_entry_button(app) is challenge_button


def test_goto_contest_page_waits_for_delayed_entry_button(monkeypatch):
    contest_button = _ButtonStub(6, 6, 10, 9, text="Sアイドルへの道挑戦中")
    latest_results = _SequencedResults([{}, {"buttons": [contest_button]}])
    device = _DeviceStub()
    current_location = {"value": GamePageTypes.MAIN_MENU__CONTEST}
    calls = []

    def _update_current_location(new_location=None):
        if new_location is not None:
            current_location["value"] = new_location
        return current_location["value"]

    def _click_element_and_wait_trigger(element, **kwargs):
        calls.append(("trigger", element, kwargs))
        return True

    def _wait_location_update(target_location, **_kwargs):
        current_location["value"] = target_location
        return True

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            update_current_location=_update_current_location,
            click_element_and_wait_trigger=_click_element_and_wait_trigger,
            wait_loading=lambda *_args, **_kwargs: True,
            wait_location_update=_wait_location_update,
        ),
    )

    monkeypatch.setattr(goto_pages, "_goto_tab_contest", lambda _app: None)
    monkeypatch.setattr(goto_pages, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(goto_pages, "sleep", lambda *_: latest_results.advance())

    goto_pages.goto__contest_page(app)

    assert calls[0][0] == "trigger"
    assert calls[0][1] is contest_button
    assert current_location["value"] == GamePageTypes.CONTEST_TAB.ARENA


def test_goto_contest_page_uses_action_trigger_click(monkeypatch):
    contest_button = _ButtonStub(6, 6, 10, 9, text="1705レートPt4コンテスト210GRADE")
    latest_results = _SequencedResults([{"buttons": [contest_button]}])
    device = _DeviceStub()
    current_location = {"value": GamePageTypes.MAIN_MENU__CONTEST}
    calls = []

    def _update_current_location(new_location=None):
        if new_location is not None:
            current_location["value"] = new_location
        return current_location["value"]

    def _click_element_and_wait_trigger(element, **kwargs):
        calls.append(("trigger", element, kwargs))
        return True

    def _wait_loading(timeout=-1):
        calls.append(("wait_loading", timeout))
        return True

    def _wait_location_update(target_location, timeout=15, ignore_loading=True):
        calls.append(("wait_location_update", target_location, timeout, ignore_loading))
        current_location["value"] = target_location
        return True

    def _click_button(*_args, **_kwargs):
        raise AssertionError("goto__contest_page should not use OCR-only click_button here")

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            update_current_location=_update_current_location,
            click_element_and_wait_trigger=_click_element_and_wait_trigger,
            wait_loading=_wait_loading,
            wait_location_update=_wait_location_update,
            click_button=_click_button,
        ),
    )

    monkeypatch.setattr(goto_pages, "_goto_tab_contest", lambda app: None)
    monkeypatch.setattr(goto_pages, "ButtonList", _ButtonListStub)

    goto_pages.goto__contest_page(app)

    assert calls[0][0] == "trigger"
    assert calls[0][1] is contest_button
    assert ("wait_loading", 8) in calls
    assert current_location["value"] == GamePageTypes.CONTEST_TAB.ARENA


def test_goto_contest_page_records_step_trace(monkeypatch):
    contest_button = _ButtonStub(6, 6, 10, 9, text="")
    app = SimpleNamespace(
        latest_results=_SequencedResults([{"buttons": [contest_button]}]),
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
        device=_DeviceStub(),
    )
    current_location = {"value": GamePageTypes.MAIN_MENU__CONTEST}

    def _update_current_location(new_location=None):
        if new_location is not None:
            current_location["value"] = new_location
        return current_location["value"]

    app.game_utils = SimpleNamespace(
        update_current_location=_update_current_location,
        click_element_and_wait_trigger=lambda *_args, **_kwargs: True,
        wait_loading=lambda *_args, **_kwargs: True,
        wait_location_update=lambda target_location, **_kwargs: current_location.update(value=target_location) or True,
    )

    monkeypatch.setattr(goto_pages, "_goto_tab_contest", lambda app: None)
    monkeypatch.setattr(goto_pages, "ButtonList", _ButtonListStub)

    goto_pages.goto__contest_page(app)

    steps = [entry["step"] for entry in get_task_debug_trace(app)]
    assert "goto_contest.click_entry" in steps
    assert "goto_contest.entered_arena" in steps


def test_goto_contest_page_dismisses_season_overlay_after_entry(monkeypatch):
    contest_button = _ButtonStub(6, 6, 10, 9, text="1705レートPt4コンテスト210GRADE")
    latest_results = _SequencedResults([
        {"buttons": [contest_button]},
        {},
    ])
    device = _DeviceStub()
    current_location = {"value": GamePageTypes.MAIN_MENU__CONTEST}
    overlay_visible = {"value": True}
    calls = []

    def _update_current_location(new_location=None):
        if new_location is not None:
            current_location["value"] = new_location
        return current_location["value"]

    def _wait_location_update(target_location, timeout=15, ignore_loading=True):
        calls.append(("wait_location_update", target_location, timeout, ignore_loading))
        raise TimeoutError("Timeout for waiting for location update")

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((20, 10, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            update_current_location=_update_current_location,
            click_element_and_wait_trigger=lambda *_args, **_kwargs: True,
            wait_loading=lambda *_args, **_kwargs: True,
            wait_location_update=_wait_location_update,
        ),
    )

    monkeypatch.setattr(goto_pages, "_goto_tab_contest", lambda app: None)
    monkeypatch.setattr(goto_pages, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(goto_pages, "sleep", lambda *_: latest_results.advance())

    def _detect_overlay(_frame, add_debug_box=False):
        if not overlay_visible["value"]:
            return None
        overlay_visible["value"] = False
        return ContestSeasonOverlay(left=2, top=3, right=8, bottom=15, rank_text="299")

    monkeypatch.setattr(goto_pages, "detect_contest_season_overlay", _detect_overlay)

    goto_pages.goto__contest_page(app)

    assert device.clicked_points == [(5, 9)]
    assert current_location["value"] == GamePageTypes.CONTEST_TAB.ARENA
    steps = [entry["step"] for entry in get_task_debug_trace(app)]
    assert "goto_contest.location_timeout" in steps
    assert "goto_contest.dismiss_overlay_done" in steps
    assert "goto_contest.entered_arena_via_overlay" in steps


def test_goto_contest_page_dismisses_season_overlay_then_grade_up(monkeypatch):
    contest_button = _ButtonStub(6, 6, 10, 9, text="1705レートPt4コンテスト210GRADE")
    latest_results = _SequencedResults([
        {"buttons": [contest_button]},
        {},
        {},
    ])
    device = _DeviceStub()
    current_location = {"value": GamePageTypes.MAIN_MENU__CONTEST}
    season_visible = {"value": True}
    grade_up_visible = {"value": True}

    def _update_current_location(new_location=None):
        if new_location is not None:
            current_location["value"] = new_location
        return current_location["value"]

    def _wait_location_update(*_args, **_kwargs):
        raise TimeoutError("Timeout for waiting for location update")

    app = SimpleNamespace(
        latest_results=latest_results,
        latest_frame=np.zeros((20, 10, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            update_current_location=_update_current_location,
            click_element_and_wait_trigger=lambda *_args, **_kwargs: True,
            wait_loading=lambda *_args, **_kwargs: True,
            wait_location_update=_wait_location_update,
        ),
    )

    monkeypatch.setattr(goto_pages, "_goto_tab_contest", lambda app: None)
    monkeypatch.setattr(goto_pages, "ButtonList", _ButtonListStub)
    monkeypatch.setattr(goto_pages, "sleep", lambda *_: latest_results.advance())

    def _detect_overlay(_frame, add_debug_box=False):
        if season_visible["value"]:
            season_visible["value"] = False
            return ContestSeasonOverlay(left=2, top=3, right=8, bottom=15, rank_text="299")
        return None

    def _detect_grade_up(_frame, add_debug_box=False):
        if season_visible["value"]:
            return None
        if grade_up_visible["value"]:
            grade_up_visible["value"] = False
            return ContestGradeUpSplash(left=2, top=4, right=8, bottom=8, title_text="グレードUP")
        return None

    monkeypatch.setattr(goto_pages, "detect_contest_season_overlay", _detect_overlay)
    monkeypatch.setattr(goto_pages, "detect_contest_grade_up_splash", _detect_grade_up)

    goto_pages.goto__contest_page(app)

    assert device.clicked_points == [(5, 9), (5, 6)]
    assert current_location["value"] == GamePageTypes.CONTEST_TAB.ARENA
    steps = [entry["step"] for entry in get_task_debug_trace(app)]
    assert "goto_contest.dismiss_overlay_done" in steps
    assert "goto_contest.dismiss_grade_up_done" in steps
    assert "goto_contest.entered_arena_via_overlay" in steps


def test_detect_contest_season_overlay_on_sample_image():
    frame = cv2.imread("tests/contest_samples/contest_season_overlay.png")
    overlay = detect_contest_season_overlay(frame)

    assert overlay is not None
    assert overlay.rank_text == "299"
    assert overlay.center_x > 0
    assert overlay.center_y > 0


def test_detect_contest_grade_up_splash_on_sample_image():
    frame = cv2.imread("tests/contest_samples/contest_grade_up.png")
    splash = detect_contest_grade_up_splash(frame)

    assert splash is not None
    assert "グレードUP" in splash.title_text
    assert splash.title_center_x > 0
    assert splash.title_center_y > 0


@pytest.mark.parametrize("jpeg_quality", [85, 65])
def test_detect_contest_season_overlay_with_jpeg_noise(jpeg_quality):
    frame = cv2.imread("tests/contest_samples/contest_season_overlay.png")
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    assert ok
    noisy = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    overlay = detect_contest_season_overlay(noisy)

    assert overlay is not None
    assert overlay.rank_text == "299"


@pytest.mark.parametrize("jpeg_quality", [85, 65])
def test_detect_contest_grade_up_splash_with_jpeg_noise(jpeg_quality):
    frame = cv2.imread("tests/contest_samples/contest_grade_up.png")
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    assert ok
    noisy = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    splash = detect_contest_grade_up_splash(noisy)

    assert splash is not None
    assert "グレードUP" in splash.title_text


@pytest.mark.parametrize("sigma", [8, 16])
def test_detect_contest_season_overlay_with_gaussian_noise(sigma):
    rng = np.random.default_rng(20260418 + sigma)
    frame = cv2.imread("tests/contest_samples/contest_season_overlay.png")
    noise = rng.normal(0, sigma, size=frame.shape).astype(np.float32)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    overlay = detect_contest_season_overlay(noisy)

    assert overlay is not None
    assert overlay.rank_text == "299"


@pytest.mark.parametrize("sigma", [8, 16])
def test_detect_contest_grade_up_splash_with_gaussian_noise(sigma):
    rng = np.random.default_rng(20260418 + sigma + 100)
    frame = cv2.imread("tests/contest_samples/contest_grade_up.png")
    noise = rng.normal(0, sigma, size=frame.shape).astype(np.float32)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    splash = detect_contest_grade_up_splash(noisy)

    assert splash is not None
    assert "グレードUP" in splash.title_text
