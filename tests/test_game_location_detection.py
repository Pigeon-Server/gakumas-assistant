from types import SimpleNamespace

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils import game_tools


class _BoxesStub:
    def __init__(self, labels=None):
        self._labels = set(labels or [])
        self.frame = np.zeros((120, 120, 3), dtype=np.uint8)

    def exists_label(self, label):
        return label in self._labels

    def exists_all_labels(self, labels):
        return all(label in self._labels for label in labels)

    def filter_by_label(self, label):
        if label in self._labels:
            return [SimpleNamespace(frame=np.zeros((10, 10, 3), dtype=np.uint8))]
        return []

    def filter_by_labels(self, labels):
        return []


class _ButtonsStub:
    def __init__(self, texts):
        self._texts = set(texts)

    def get_button_by_text(self, text, match_config=None):
        return SimpleNamespace(text=text) if text in self._texts else None


def test_get_current_location_detects_memory_selection(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.MEMORY_CARD})

    monkeypatch.setattr(
        game_tools,
        "ButtonList",
        lambda _boxes: _ButtonsStub({ButtonText.NEXT, ButtonText.AUTO_SELECT, ButtonText.RESET, "編成詳細"}),
    )
    monkeypatch.setattr(game_tools, "_extract_screen_texts", lambda _frame: [])

    assert game_tools.get_current_location(boxes) == GamePageTypes.PRODUCER__MEMORY_SELECTION


def test_get_current_location_detects_memory_candidate_list(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.CLOSE_BUTTON, BaseUILabels.BLANK_SLOT})

    monkeypatch.setattr(game_tools, "ButtonList", lambda _boxes: _ButtonsStub(set()))

    assert game_tools.get_current_location(boxes) == GamePageTypes.PRODUCER__MEMORY_CANDIDATE_LIST


def test_get_current_location_detects_memory_candidate_list_from_ocr(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.BACK_BTN})

    monkeypatch.setattr(game_tools, "ButtonList", lambda _boxes: _ButtonsStub(set()))
    monkeypatch.setattr(game_tools, "_extract_screen_texts", lambda _frame: ["メモリー編成一覧"])

    assert game_tools.get_current_location(boxes) == GamePageTypes.PRODUCER__MEMORY_CANDIDATE_LIST


def test_get_current_location_detects_memory_detail(monkeypatch):
    boxes = _BoxesStub()

    monkeypatch.setattr(
        game_tools,
        "ButtonList",
        lambda _boxes: _ButtonsStub({ButtonText.CONFIRM, "キャンセル"}),
    )
    monkeypatch.setattr(game_tools, "_extract_screen_texts", lambda _frame: ["所持メモリー"])

    assert game_tools.get_current_location(boxes) == GamePageTypes.PRODUCER__MEMORY_DETAIL


def test_get_current_location_detects_final_confirm(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.SUPPORT_CARD})

    monkeypatch.setattr(
        game_tools,
        "ButtonList",
        lambda _boxes: _ButtonsStub({ButtonText.PRODUCE_START, "編成詳細"}),
    )
    monkeypatch.setattr(game_tools, "_extract_screen_texts", lambda _frame: [])

    assert game_tools.get_current_location(boxes) == GamePageTypes.PRODUCER__FINAL_CONFIRM


def test_get_current_location_detects_formation_details(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.TAB_BAR, BaseUILabels.MEMORY_CARD})

    monkeypatch.setattr(game_tools, "ButtonList", lambda _boxes: _ButtonsStub(set()))
    monkeypatch.setattr(game_tools, "_extract_screen_texts", lambda _frame: [])

    assert game_tools.get_current_location(boxes) == GamePageTypes.PRODUCER__FORMATION_DETAIL
