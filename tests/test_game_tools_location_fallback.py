from types import SimpleNamespace

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
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


class _ButtonListStub:
    def __init__(self, texts):
        self._texts = set(texts)

    def __bool__(self):
        return bool(self._texts)

    def get_button_by_text(self, text, match_config=None):
        return SimpleNamespace(text=text) if text in self._texts else None


def test_get_current_location_detects_support_selection_by_page_features(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.SUPPORT_CARD})

    monkeypatch.setattr(game_tools, "ButtonList", lambda _boxes: _ButtonListStub({
        ButtonText.RESET,
        ButtonText.AUTO_SELECT,
        ButtonText.NEXT,
    }))

    assert (
        game_tools.get_current_location(boxes)
        == GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__SUPPORT_SELECTION
    )


def test_get_current_location_does_not_treat_blank_memory_page_as_support(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.BLANK_SLOT})

    monkeypatch.setattr(game_tools, "ButtonList", lambda _boxes: _ButtonListStub({
        ButtonText.RESET,
        ButtonText.AUTO_SELECT,
        ButtonText.NEXT,
    }))

    assert game_tools.get_current_location(boxes) == GamePageTypes.UNKNOWN


def test_get_current_location_detects_memory_selection_by_page_features(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.MEMORY_CARD})

    monkeypatch.setattr(game_tools, "ButtonList", lambda _boxes: _ButtonListStub({
        ButtonText.RESET,
        ButtonText.AUTO_SELECT,
        ButtonText.NEXT,
        ProduceText.FORMATION_DETAILS,
    }))

    assert (
        game_tools.get_current_location(boxes)
        == GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_SELECTION
    )


def test_get_current_location_does_not_treat_memory_slots_without_detail_button_as_memory_page(monkeypatch):
    boxes = _BoxesStub(labels={BaseUILabels.MEMORY_CARD})

    monkeypatch.setattr(game_tools, "ButtonList", lambda _boxes: _ButtonListStub({
        ButtonText.RESET,
        ButtonText.AUTO_SELECT,
        ButtonText.NEXT,
    }))

    assert game_tools.get_current_location(boxes) == GamePageTypes.UNKNOWN
