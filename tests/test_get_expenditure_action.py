import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.base_ui import get_expenditure


class _ButtonStub:
    def __init__(self, name: str):
        self.name = name


class _ModalStub:
    def __init__(self, title: str, cancel_name: str):
        self.modal_title = title
        self.cancel_button = _ButtonStub(cancel_name)
        self.confirm_button = None


class _LatestResultsStub:
    def __init__(self):
        self.game_utils = None

    def exists_label(self, label):
        if label == BaseUILabels.TAB_HOME:
            return True
        if label == BaseUILabels.MODAL_HEADER:
            return self.game_utils is not None and self.game_utils.active_modal is not None
        return False

    def filter_by_label(self, label):
        if label == BaseUILabels.BUTTON and self.game_utils is not None and self.game_utils.active_modal is not None:
            return [self.game_utils.active_modal.cancel_button]
        return []


class _DeviceStub:
    def __init__(self):
        self.clicked = []

    def click_element(self, element):
        self.clicked.append(element.name)


class _GameUtilsStub:
    def __init__(self):
        self.modals = [
            _ModalStub("親愛度アイドル選択", "cancel_wrong_modal"),
            _ModalStub("活動費", "cancel_expenditure_modal"),
        ]
        self.active_modal = None
        self.close_click_counts = {}
        self.wait_frame_stable_calls = 0

    def wait_for_modal(self, *_args, **_kwargs):
        if self.active_modal is None and self.modals:
            self.active_modal = self.modals.pop(0)
        return self.active_modal

    def try_get_modal(self, *_args, **_kwargs):
        return self.active_modal

    def wait_frame_stable(self, *_args, **_kwargs):
        self.wait_frame_stable_calls += 1
        return True

    def click_modal_button_and_wait_transition(self, button, **_kwargs):
        self._device.click_element(button)
        if self.active_modal is None:
            return True

        modal_title = self.active_modal.modal_title
        self.close_click_counts[modal_title] = self.close_click_counts.get(modal_title, 0) + 1
        if modal_title == "親愛度アイドル選択":
            if self.close_click_counts[modal_title] >= 2:
                self.active_modal = None
                return True
            return False

        self.active_modal = None
        return True


def test_action_claim_expenditure_retries_after_unexpected_modal(monkeypatch):
    monkeypatch.setattr(get_expenditure, "sleep", lambda *_args, **_kwargs: None)

    goto_calls = []

    def _goto__get_expenditure(_app, candidate_index=0):
        goto_calls.append(candidate_index)
        if _app.game_utils.active_modal is None and _app.game_utils.modals:
            _app.game_utils.active_modal = _app.game_utils.modals.pop(0)

    monkeypatch.setattr(
        "src.core.tasks.base_ui.goto_pages.goto__get_expenditure",
        _goto__get_expenditure,
    )

    app = SimpleNamespace(
        game_utils=_GameUtilsStub(),
        device=_DeviceStub(),
        latest_results=_LatestResultsStub(),
    )
    app.game_utils._device = app.device
    app.latest_results.game_utils = app.game_utils

    assert get_expenditure.action__claim_expenditure(app) is True
    assert goto_calls == [0, 1], "Should try index 0 first, then advance to index 1"
    assert app.device.clicked == [
        "cancel_wrong_modal",
        "cancel_wrong_modal",
        "cancel_expenditure_modal",
    ]
    assert app.game_utils.wait_frame_stable_calls == 0
