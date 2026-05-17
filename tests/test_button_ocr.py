import importlib
from types import SimpleNamespace

import numpy as np

from src.core.inference.ocr_engine import OCR_Result, OCR_ResultList

button_module = importlib.import_module("src.entity.Game.Components.Button")


def test_button_ocr_falls_back_to_preprocessed_image(monkeypatch):
    calls = []

    def _fake_ocr(image):
        calls.append(image.shape[:2])
        if image.shape[:2] == (20, 60):
            return OCR_ResultList([])
        return OCR_ResultList([OCR_Result(0, 0, 10, 10, "無料", 0.99)])

    monkeypatch.setattr(button_module.ocr_service, "ocr", _fake_ocr)

    frame = np.full((20, 60, 3), (0, 140, 255), dtype=np.uint8)
    element = SimpleNamespace(x=0, y=0, w=60, h=20, label="Universal button", frame=frame)

    button = button_module.Button(element)

    assert button.text == "無料"
    assert calls[0] == (20, 60)
    assert len(calls) >= 2
