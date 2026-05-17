from types import SimpleNamespace

import numpy as np

from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCR_Result, OCR_ResultList
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import schedule as schedule_module


class _BoxList(list):
    def first(self):
        return self[0]


def _box(x1: int, y1: int, x2: int, y2: int, label: str):
    return SimpleNamespace(
        x=x1,
        y=y1,
        w=x2,
        h=y2,
        cx=int((x1 + x2) / 2),
        cy=int((y1 + y2) / 2),
        frame=np.zeros((max(1, y2 - y1), max(1, x2 - x1), 3), dtype=np.uint8),
        label=label,
    )


class _ResultsStub:
    def __init__(self, mapping):
        self._mapping = {
            label: _BoxList(list(items))
            for label, items in mapping.items()
        }

    def filter_by_label(self, label):
        return self._mapping.get(label, _BoxList())


def test_collect_schedule_action_candidates_prefers_screen_lookup_text_when_box_ocr_is_noisy(monkeypatch):
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            ProducerLabels.PC_ACTION: [
                _box(132, 704, 956, 876, ProducerLabels.PC_ACTION),
                _box(132, 920, 956, 1092, ProducerLabels.PC_ACTION),
            ],
        }),
    )
    ctx = ProduceContext()

    direct_ocr = iter(["is尸)くとんOA", "とsにとんべ'OA"])
    monkeypatch.setattr(schedule_module, "ocr_text", lambda _image: next(direct_ocr))
    monkeypatch.setattr(schedule_module, "_detect_recommended_kind", lambda _app: "unknown")
    monkeypatch.setattr(
        schedule_module,
        "_SCHEDULE_SCREEN_OCR",
        SimpleNamespace(
            ocr=lambda _frame: OCR_ResultList([
                OCR_Result(188, 756, 150, 42, "おでかけ", 0.99),
                OCR_Result(188, 812, 356, 42, "Pポイントを消費して体力を回復", 0.99),
                OCR_Result(188, 972, 92, 42, "休む", 0.99),
                OCR_Result(188, 1028, 292, 42, "体力を回復して次の週へ", 0.99),
            ])
        ),
    )

    candidates = schedule_module.collect_schedule_action_candidates(
        app,
        ctx,
        position="schedule_recommend",
    )

    assert [candidate.title for candidate in candidates] == ["おでかけ", "休む"]
    assert [candidate.action_id for candidate in candidates] == [
        "schedule_action_outing",
        "schedule_action_refresh",
    ]
    assert [candidate.metadata["title_source"] for candidate in candidates] == ["lookup", "lookup"]
    assert candidates[0].metadata["ocr_title"] == "is尸)くとんOA"
    assert candidates[1].metadata["lookup_texts"][0] == "休む"


def test_collect_schedule_action_candidates_keeps_direct_title_when_lookup_is_lower_quality(monkeypatch):
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            ProducerLabels.PC_ACTION: [
                _box(132, 704, 956, 876, ProducerLabels.PC_ACTION),
            ],
        }),
    )
    ctx = ProduceContext()

    monkeypatch.setattr(schedule_module, "ocr_text", lambda _image: "授業")
    monkeypatch.setattr(schedule_module, "_detect_recommended_kind", lambda _app: "unknown")
    monkeypatch.setattr(
        schedule_module,
        "_SCHEDULE_SCREEN_OCR",
        SimpleNamespace(
            ocr=lambda _frame: OCR_ResultList([
                OCR_Result(188, 756, 92, 42, "授業", 0.99),
                OCR_Result(188, 812, 328, 42, "Pポイントを消費して追加効果", 0.99),
            ])
        ),
    )

    candidates = schedule_module.collect_schedule_action_candidates(
        app,
        ctx,
        position="schedule_recommend",
    )

    assert len(candidates) == 1
    assert candidates[0].title == "授業"
    assert candidates[0].action_id == "schedule_action_class"
    assert candidates[0].metadata["title_source"] == "direct"


def test_collect_schedule_action_candidates_normalizes_noisy_class_title(monkeypatch):
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            ProducerLabels.PC_ACTION: [
                _box(132, 704, 956, 876, ProducerLabels.PC_ACTION),
            ],
        }),
    )
    ctx = ProduceContext()

    monkeypatch.setattr(schedule_module, "ocr_text", lambda _image: "Bd授業")
    monkeypatch.setattr(schedule_module, "_detect_recommended_kind", lambda _app: "unknown")
    monkeypatch.setattr(
        schedule_module,
        "_SCHEDULE_SCREEN_OCR",
        SimpleNamespace(ocr=lambda _frame: OCR_ResultList([])),
    )

    candidates = schedule_module.collect_schedule_action_candidates(
        app,
        ctx,
        position="schedule_recommend",
    )

    assert len(candidates) == 1
    assert candidates[0].action_id == "schedule_action_class"
    assert candidates[0].title == "授業"


def test_collect_schedule_action_candidates_schedule_selected_uses_pending_action_id(monkeypatch):
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            ProducerLabels.PC_ACTION: [
                _box(547, 1388, 791, 1668, ProducerLabels.PC_ACTION),
                _box(288, 1389, 532, 1667, ProducerLabels.PC_ACTION),
            ],
        }),
    )
    ctx = ProduceContext()
    ctx.pending_schedule_index = 1
    ctx.pending_schedule_label = "相談"
    ctx.handler_state["pending_schedule_action_id"] = "schedule_action_outing"

    monkeypatch.setattr(schedule_module, "_detect_recommended_kind", lambda _app: "unknown")
    monkeypatch.setattr(
        schedule_module,
        "_resolve_schedule_from_clip",
        lambda _app, box: (
            {
                "action_id": "schedule_action_outing",
                "param_kind": "unknown",
                "rl_action_type": "activity",
            }
            if box.x < 400
            else {
                "action_id": "schedule_action_consult",
                "param_kind": "unknown",
                "rl_action_type": "",
            }
        ),
    )
    probed_ids: list[str] = []
    monkeypatch.setattr(
        schedule_module,
        "_probe_action_info_panel",
        lambda _app, candidate: (
            probed_ids.append(candidate.action_id) or "OUTING_EFFECT"
        ),
    )

    candidates = schedule_module.collect_schedule_action_candidates(
        app,
        ctx,
        position="schedule_selected",
    )

    assert [candidate.action_id for candidate in candidates] == [
        "schedule_action_outing",
        "schedule_action_consult",
    ]
    assert [candidate.selected for candidate in candidates] == [True, False]
    assert ctx.pending_schedule_index == 0
    assert probed_ids == ["schedule_action_outing"]
    assert candidates[0].metadata["effect_text"] == "OUTING_EFFECT"


def test_collect_schedule_action_candidates_lesson_sp_has_canonical_name_and_description(monkeypatch):
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            ProducerLabels.PC_ACTION: [
                _box(132, 704, 956, 876, ProducerLabels.PC_ACTION),
                _box(132, 920, 956, 1092, ProducerLabels.PC_ACTION),
                _box(132, 1136, 956, 1308, ProducerLabels.PC_ACTION),
            ],
        }),
    )
    ctx = ProduceContext()

    direct_ocr = iter(["ボーカルレッスン", "ダンスレッスン", "ビジュアルレッスン"])
    monkeypatch.setattr(schedule_module, "ocr_text", lambda _image: next(direct_ocr))
    monkeypatch.setattr(schedule_module, "_detect_recommended_kind", lambda _app: "unknown")
    monkeypatch.setattr(
        schedule_module,
        "_SCHEDULE_SCREEN_OCR",
        SimpleNamespace(ocr=lambda _frame: OCR_ResultList([])),
    )
    monkeypatch.setattr(schedule_module, "detect_sp_badge", lambda _box: True)

    candidates = schedule_module.collect_schedule_action_candidates(
        app,
        ctx,
        position="schedule_recommend",
    )

    assert [candidate.action_id for candidate in candidates] == [
        "schedule_action_lesson_vocal_sp",
        "schedule_action_lesson_dance_sp",
        "schedule_action_lesson_visual_sp",
    ]
    assert [candidate.title for candidate in candidates] == [
        "ボーカルSPレッスン",
        "ダンスSPレッスン",
        "ビジュアルSPレッスン",
    ]
    assert all("SPレッスン" in str(candidate.metadata.get("description") or "") for candidate in candidates)
