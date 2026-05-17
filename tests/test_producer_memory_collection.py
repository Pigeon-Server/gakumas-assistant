from types import SimpleNamespace

import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCR_Result
from src.core.tasks.producer_challenge.steps.collect import collect_memory_attributes as memory_module
from src.core.tasks.producer_challenge.catalog import get_card_item_catalog
from src.core.tasks.producer_challenge.steps.collect.collect_memory_attributes import (
    CollectMemoryAttributesStep,
    MemorySlotTarget,
)


class _DummyResults:
    def __init__(self, boxes):
        self._boxes = list(boxes)

    def filter_by_label(self, label):
        return [box for box in self._boxes if box.label == label]


def _make_box(cx: int, cy: int, label: str = BaseUILabels.MEMORY_CARD):
    return SimpleNamespace(label=label, cx=cx, cy=cy)


def test_infer_slot_targets_fills_missing_selected_slot():
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_DummyResults(
            [
                _make_box(713, 1573),
                _make_box(227, 1770),
                _make_box(716, 1770),
            ]
        ),
    )

    targets = CollectMemoryAttributesStep._infer_slot_targets(app)

    assert [(target.slot_index, target.cx, target.cy, target.synthetic) for target in targets] == [
        (1, 227, 1573, True),
        (2, 713, 1573, False),
        (3, 227, 1770, False),
        (4, 716, 1770, False),
    ]


def test_extract_memory_stats_reads_values_from_top_block():
    lines = [
        OCR_Result(219, 87, 135, 40, "ボーカル", None),
        OCR_Result(445, 87, 98, 40, "ダンス", None),
        OCR_Result(620, 87, 164, 40, "ビジュアル", None),
        OCR_Result(912, 87, 69, 40, "体カ", None),
        OCR_Result(233, 124, 105, 40, "110", None),
        OCR_Result(445, 124, 98, 40, "182", None),
        OCR_Result(649, 124, 105, 40, "165", None),
        OCR_Result(912, 138, 69, 40, "ろ5", None),
    ]

    stats = CollectMemoryAttributesStep._extract_memory_stats(lines)

    assert stats == {
        "vocal": 110,
        "dance": 182,
        "visual": 165,
        "stamina": 5,
    }


def test_extract_skill_card_page_matches_db_card(monkeypatch):
    produce_card_entry = None
    produce_card_desc = ""
    for entry in get_card_item_catalog():
        if entry.kind != "produce_card":
            continue
        desc = CollectMemoryAttributesStep._get_produce_card_description(entry.id)
        if desc:
            produce_card_entry = entry
            produce_card_desc = desc
            break

    assert produce_card_entry is not None
    assert produce_card_desc

    lines = [
        OCR_Result(64, 313, 368, 47, "獲得可能スキルカード", None),
        OCR_Result(467, 313, 529, 47, "中間審査後に獲得", None),
        OCR_Result(302, 368, 219, 51, produce_card_entry.display_name, None),
        OCR_Result(306, 459, 434, 40, produce_card_desc, None),
        OCR_Result(489, 817, 105, 40, "1/1", None),
    ]

    monkeypatch.setattr(
        CollectMemoryAttributesStep,
        "_extract_detail_lines",
        staticmethod(lambda frame: lines),
    )

    page = CollectMemoryAttributesStep._extract_skill_card_page(np.zeros((2340, 1080, 3), dtype=np.uint8))

    assert page is not None
    assert page["title"] == produce_card_entry.display_name
    assert page["matched_entry"] is not None
    assert page["matched_entry"]["id"] == produce_card_entry.id
    assert page["page_index"] == 1
    assert page["total_pages"] == 1
    assert page["description_match_score"] >= 95


def test_validate_accepts_recoverable_memory_states(monkeypatch):
    step = CollectMemoryAttributesStep()

    monkeypatch.setattr(step, "_get_memory_page_state", lambda app: app.state)

    assert step.validate(SimpleNamespace(state="selection"), SimpleNamespace()) is True
    assert step.validate(SimpleNamespace(state="candidate_list"), SimpleNamespace()) is True
    assert step.validate(SimpleNamespace(state="detail"), SimpleNamespace()) is True
    assert step.validate(SimpleNamespace(state="final_confirm"), SimpleNamespace()) is True
    assert step.validate(SimpleNamespace(state="unknown"), SimpleNamespace()) is False


def test_ensure_memory_selection_page_recovers_detail_then_list(monkeypatch):
    step = CollectMemoryAttributesStep()
    app = SimpleNamespace(state="detail")
    transitions: list[str] = []

    monkeypatch.setattr(
        memory_module,
        "wait_for_memory_selection_page",
        lambda app, timeout=0: app.state == "selection",
    )
    monkeypatch.setattr(
        memory_module,
        "wait_frame_stable",
        lambda app, timeout=0: None,
    )
    monkeypatch.setattr(step, "_get_memory_page_state", lambda app: app.state)

    def _dismiss_detail(app):
        transitions.append("detail")
        app.state = "candidate_list"
        return app.state

    def _dismiss_list(app):
        transitions.append("candidate_list")
        app.state = "selection"
        return True

    monkeypatch.setattr(step, "_dismiss_memory_detail_overlay", _dismiss_detail)
    monkeypatch.setattr(step, "_dismiss_memory_candidate_list", _dismiss_list)

    assert step._ensure_memory_selection_page(app, timeout=2.0) is True
    assert app.state == "selection"
    assert transitions == ["detail", "candidate_list"]


def test_open_current_memory_detail_reopens_list_after_selection_misfire(monkeypatch):
    step = CollectMemoryAttributesStep()
    clicks: list[tuple[int, int]] = []
    open_list_calls = {"count": 0}
    states = iter(["selection", "detail"])

    app = SimpleNamespace(
        device=SimpleNamespace(click=lambda x, y: clicks.append((x, y))),
    )

    monkeypatch.setattr(step, "_wait_for_memory_candidate_list_page", lambda app, timeout=0: True)
    monkeypatch.setattr(step, "_get_memory_detail_hotspots", lambda app: [(100, 200), (300, 400)])
    monkeypatch.setattr(step, "_open_memory_candidate_list", lambda app: open_list_calls.__setitem__("count", open_list_calls["count"] + 1) or True)
    monkeypatch.setattr(step, "_wait_for_memory_page_state", lambda app, allowed_states, timeout=0: next(states))
    monkeypatch.setattr(memory_module, "sleep", lambda *_args, **_kwargs: None)

    assert step._open_current_memory_detail_from_list(app) is True
    assert clicks == [(100, 200), (300, 400)]
    assert open_list_calls["count"] == 1


def test_execute_records_memory_slots_and_skips_detail_collection(monkeypatch):
    step = CollectMemoryAttributesStep()
    ctx = SimpleNamespace(memories=[], memory_attributes=[])
    open_list_calls = {"count": 0}

    monkeypatch.setattr(step, "_get_memory_page_state", lambda app: "selection")
    monkeypatch.setattr(step, "_ensure_memory_selection_page", lambda app, timeout=0: True)
    monkeypatch.setattr(
        step,
        "_infer_slot_targets",
        lambda app: [
            MemorySlotTarget(slot_index=1, cx=120, cy=340),
            MemorySlotTarget(slot_index=2, cx=560, cy=340, synthetic=True),
        ],
    )
    monkeypatch.setattr(
        step,
        "_open_memory_candidate_list",
        lambda app: open_list_calls.__setitem__("count", open_list_calls["count"] + 1) or True,
    )
    monkeypatch.setattr(step, "_advance_to_final_confirm", lambda app, ctx: True)

    assert step.execute(SimpleNamespace(), ctx) is True
    assert ctx.memories == [
        {"slot_index": 1, "selected_cx": 120, "selected_cy": 340, "synthetic": False},
        {"slot_index": 2, "selected_cx": 560, "selected_cy": 340, "synthetic": True},
    ]
    assert ctx.memory_attributes == []
    assert open_list_calls["count"] == 0


def test_execute_returns_immediately_when_already_on_final_confirm(monkeypatch):
    step = CollectMemoryAttributesStep()
    ctx = SimpleNamespace(memories=[], memory_attributes=[])

    monkeypatch.setattr(step, "_get_memory_page_state", lambda app: "final_confirm")
    monkeypatch.setattr(step, "_ensure_memory_selection_page", lambda app, timeout=0: False)

    assert step.execute(SimpleNamespace(), ctx) is True
    assert ctx.memories == []
    assert ctx.memory_attributes == []
