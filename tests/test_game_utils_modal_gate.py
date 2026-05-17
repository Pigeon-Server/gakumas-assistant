from types import SimpleNamespace

import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.services.game_utils import GameUtils
from src.entity.Yolo import Yolo_Box, Yolo_Results


class _ResultsStub:
    def __init__(self, boxes, has_header=False, shape=(2160, 1080, 3)):
        self.frame = np.zeros(shape, dtype=np.uint8)
        self._boxes = boxes
        self._has_header = has_header

    def filter_by_label(self, label: str):
        if label == BaseUILabels.BUTTON:
            return Yolo_Results.from_boxes([box for box in self._boxes if box.label == BaseUILabels.BUTTON])
        if label == BaseUILabels.MODAL_HEADER and self._has_header:
            return Yolo_Results.from_boxes([
                Yolo_Box(80, 1400, 1000, 1480, BaseUILabels.MODAL_HEADER, self.frame[1400:1480, 80:1000])
            ])
        return Yolo_Results.from_boxes([])

    def exists_label(self, label: str) -> bool:
        return label == BaseUILabels.MODAL_HEADER and self._has_header


def _button(x1, y1, x2, y2):
    frame = np.zeros((max(1, y2 - y1), max(1, x2 - x1), 3), dtype=np.uint8)
    return Yolo_Box(x1, y1, x2, y2, BaseUILabels.BUTTON, frame)


def _build_game_utils(results):
    app = SimpleNamespace(latest_results=results)
    return GameUtils(app)


def test_try_get_modal_skips_parse_without_modal_header_by_default(monkeypatch):
    sentinel = object()
    calls = {"count": 0}

    def _mock_get_modal(*_args, **_kwargs):
        calls["count"] += 1
        return sentinel

    monkeypatch.setattr("src.core.services.game_utils.get_modal", _mock_get_modal)

    results = _ResultsStub([
        _button(120, 1750, 480, 1870),
        _button(600, 1750, 960, 1870),
    ])
    game_utils = _build_game_utils(results)

    assert game_utils.try_get_modal(no_body=True) is None
    assert calls["count"] == 0


def test_try_get_modal_can_opt_in_to_headerless_parse(monkeypatch):
    sentinel = object()
    calls = {"count": 0}

    def _mock_get_modal(*_args, **_kwargs):
        calls["count"] += 1
        return sentinel

    monkeypatch.setattr("src.core.services.game_utils.get_modal", _mock_get_modal)

    results = _ResultsStub([
        _button(120, 1750, 480, 1870),
        _button(600, 1750, 960, 1870),
    ])
    game_utils = _build_game_utils(results)

    assert game_utils.try_get_modal(no_body=True, require_header=False) is sentinel
    assert calls["count"] == 1


def test_try_get_modal_returns_none_when_parser_cannot_build_modal(monkeypatch):
    calls = {"count": 0}

    def _mock_get_modal(*_args, **_kwargs):
        calls["count"] += 1
        return None

    monkeypatch.setattr("src.core.services.game_utils.get_modal", _mock_get_modal)

    results = _ResultsStub([
        _button(120, 1750, 480, 1870),
        _button(600, 1750, 960, 1870),
    ], has_header=True)
    game_utils = _build_game_utils(results)

    assert game_utils.try_get_modal(no_body=True) is None
    assert calls["count"] == 1


def test_try_get_modal_parses_modal_when_header_exists(monkeypatch):
    sentinel = object()

    monkeypatch.setattr("src.core.services.game_utils.get_modal", lambda *_args, **_kwargs: sentinel)

    results = _ResultsStub([
        _button(120, 1750, 480, 1870),
        _button(600, 1750, 960, 1870),
    ], has_header=True)
    game_utils = _build_game_utils(results)

    assert game_utils.try_get_modal(no_body=True) is sentinel


def test_wait_for_modal_does_not_parse_headerless_page_by_default(monkeypatch):
    calls = {"count": 0}

    monkeypatch.setattr("src.core.services.game_utils.sleep", lambda *_args, **_kwargs: None)

    def _mock_get_modal(*_args, **_kwargs):
        calls["count"] += 1
        return object()

    monkeypatch.setattr("src.core.services.game_utils.get_modal", _mock_get_modal)

    results = _ResultsStub([
        _button(120, 1750, 480, 1870),
        _button(600, 1750, 960, 1870),
    ])
    game_utils = _build_game_utils(results)

    assert game_utils.wait_for_modal(None, timeout=1, interval=0.5, no_body=True) is None
    assert calls["count"] == 0
