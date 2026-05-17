from types import SimpleNamespace

import cv2
import numpy as np

from src.core.inference.ocr_engine import OCR_Result, OCR_ResultList
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.gameplay import item_select as item_select_module


def _item_box(x1: int, y1: int, x2: int, y2: int):
    return SimpleNamespace(
        x=x1,
        y=y1,
        w=x2,
        h=y2,
        cx=int((x1 + x2) / 2),
        cy=int((y1 + y2) / 2),
        frame=np.zeros((max(1, y2 - y1), max(1, x2 - x1), 3), dtype=np.uint8),
        label="Special Item",
    )


class _ResultsStub:
    def __init__(self, boxes):
        self._boxes = boxes

    def filter_by_label(self, label):
        if isinstance(self._boxes, dict):
            return list(self._boxes.get(label, []))
        return list(self._boxes)


def test_collect_item_select_candidates_reads_titles_from_screen_ocr(monkeypatch):
    boxes = [
        _item_box(120, 760, 320, 980),
        _item_box(720, 760, 920, 980),
    ]
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(boxes),
        clip_manager=None,
    )

    monkeypatch.setattr(item_select_module, "ocr_text", lambda _image: "")
    monkeypatch.setattr(
        item_select_module,
        "_ITEM_SELECT_SCREEN_OCR",
        SimpleNamespace(
            ocr=lambda _frame: OCR_ResultList([
                OCR_Result(170, 520, 440, 44, "受け取るPアイテムを選んでください", 0.99),
                OCR_Result(160, 1020, 170, 42, "测试物品A", 0.99),
                OCR_Result(760, 1025, 170, 42, "测试物品B", 0.99),
                OCR_Result(420, 2010, 200, 52, "受け取る", 0.99),
            ])
        ),
    )
    monkeypatch.setattr(
        item_select_module,
        "resolve_produce_item_identity",
        lambda title, app=None, box=None, index=0, lookup_texts=None: SimpleNamespace(
            action_id=f"produce_item:item_{index}",
            db_id=f"item_{index}",
            source="ocr",
            confidence=0.95,
            display_name=(lookup_texts or [""])[0] or title,
            metadata={"matched_text": (lookup_texts or [""])[0] if lookup_texts else title},
        ),
    )

    candidates = item_select_module.collect_item_select_candidates(app)

    assert [candidate.title for candidate in candidates] == ["测试物品A", "测试物品B"]
    assert [candidate.db_id for candidate in candidates] == ["item_0", "item_1"]
    assert candidates[0].metadata["lookup_texts"] == ["测试物品A"]
    assert candidates[1].metadata["lookup_texts"] == ["测试物品B"]


def test_extract_selected_item_name_prefers_white_panel_title(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 640), (1020, 2030), (245, 245, 245), thickness=-1)
    boxes = [
        _item_box(140, 1520, 340, 1760),
        _item_box(430, 1520, 630, 1760),
        _item_box(720, 1520, 920, 1760),
    ]
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({ProducerLabels.SPECIAL_ITEM: boxes}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        item_select_module,
        "_ITEM_SELECT_SCREEN_OCR",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(84, 36, 320, 50, "はつぼし時計", 0.99),
                OCR_Result(120, 138, 260, 42, "→ 元気+2", 0.99),
            ])
        ),
    )

    name = item_select_module._extract_selected_item_name(app)
    assert name == "はつぼし時計"


def test_extract_selected_item_name_fallback_filters_effect_line(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 700), (1020, 2100), (245, 245, 245), thickness=-1)
    boxes = [
        _item_box(170, 1510, 360, 1750),
        _item_box(720, 1510, 910, 1750),
    ]
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({ProducerLabels.SPECIAL_ITEM: boxes}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        item_select_module,
        "_ITEM_SELECT_SCREEN_OCR",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(120, 24, 240, 44, "→ 元気+2", 0.99),
                OCR_Result(108, 86, 280, 44, "幸運のコイン", 0.99),
            ])
        ),
    )

    name = item_select_module._extract_selected_item_name(app)
    assert name == "幸運のコイン"
