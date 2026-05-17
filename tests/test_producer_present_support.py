from types import SimpleNamespace

import numpy as np
import pytest

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCR_Result, OCR_ResultList
from src.core.tasks.producer_challenge import ui as ui_module
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import schedule as schedule_module
from src.core.tasks.producer_challenge.gameplay.schedule import (
    ScheduleHandler,
    collect_schedule_action_candidates,
)


class _ResultsStub:
    def __init__(self, labels):
        self._labels = set(labels)
        self.frame = np.zeros((2340, 1080, 3), dtype=np.uint8)

    def exists_label(self, label):
        return label in self._labels

    def filter_by_label(self, label):
        return []

    def __bool__(self):
        return True


class _DeviceStub:
    def __init__(self):
        self.clicks = []

    def click(self, x, y, el_label=""):
        self.clicks.append((int(x), int(y), str(el_label or "")))

    def click_element(self, element):
        self.click(element.cx, element.cy, getattr(element, "label", ""))


def _make_present_support_ocr():
    return OCR_ResultList([
        OCR_Result(180, 738, 464, 47, "活動支給・差し入れ選択時、", 0.99),
        OCR_Result(188, 789, 223, 47, "ダンス上昇+12", 0.99),
        OCR_Result(180, 935, 471, 47, "活動支給・差し入れ選択時、", 0.99),
        OCR_Result(188, 987, 215, 40, "ダンス上昇+17", 0.99),
        OCR_Result(180, 1126, 464, 47, "活動支給・差し入れ選択時、", 0.99),
        OCR_Result(188, 1184, 288, 40, "ボーカル上昇+17", 0.99),
    ])


def test_classify_gameplay_state_detects_present_support_selection(monkeypatch):
    results = _ResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.PARAM_VOCAL,
        ProducerLabels.PARAM_DANCE,
        ProducerLabels.PARAM_VISUAL,
    })
    monkeypatch.setattr(
        ui_module,
        "ocr_text",
        lambda _frame: "活動支給 差し入れ選択時 ダンス上昇+12 ダンス上昇+17 ボーカル上昇+17 審査基準",
    )

    phase = ui_module.classify_gameplay_phase(results)
    position = ui_module.classify_pipeline_position(results, phase=phase)

    assert phase == "schedule"
    assert position == GameplayPosition.SCHEDULE_PRESENT_SUPPORT


@pytest.mark.parametrize("frame_text", [
    "活動支給 Pドリンク獲得",
    "活動支給 審査基準 +103",
    "湉動支拾",
])
def test_classify_gameplay_state_detects_present_support_showcase(monkeypatch, frame_text):
    results = _ResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_TARGET,
    })
    monkeypatch.setattr(
        ui_module,
        "ocr_text",
        lambda _frame: frame_text,
    )

    phase = ui_module.classify_gameplay_phase(results)
    position = ui_module.classify_pipeline_position(results, phase=phase)

    assert phase == "schedule"
    assert position == GameplayPosition.SCHEDULE_PRESENT_SUPPORT_SHOWCASE


def test_collect_schedule_action_candidates_reads_present_support_options(monkeypatch):
    monkeypatch.setattr(
        schedule_module,
        "_SCHEDULE_SCREEN_OCR",
        SimpleNamespace(ocr=lambda _frame: _make_present_support_ocr()),
    )
    app = SimpleNamespace(latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8))
    ctx = ProduceContext()

    candidates = collect_schedule_action_candidates(
        app,
        ctx,
        position=GameplayPosition.SCHEDULE_PRESENT_SUPPORT,
    )

    assert [candidate.title for candidate in candidates] == [
        "活動支給・差し入れ選択時、ダンス上昇+12",
        "活動支給・差し入れ選択時、ダンス上昇+17",
        "活動支給・差し入れ選択時、ボーカル上昇+17",
    ]
    assert [candidate.kind for candidate in candidates] == ["dance", "dance", "vocal"]


def test_schedule_handler_selects_present_support_option(monkeypatch):
    monkeypatch.setattr(
        schedule_module,
        "_SCHEDULE_SCREEN_OCR",
        SimpleNamespace(ocr=lambda _frame: _make_present_support_ocr()),
    )
    monkeypatch.setattr(schedule_module, "build_decision_state", lambda *args, **kwargs: {})

    device = _DeviceStub()
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(set()),
        device=device,
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        clip_manager=None,
    )
    ctx = ProduceContext()
    handler = ScheduleHandler()

    result = handler.handle(app, ctx, "schedule", GameplayPosition.SCHEDULE_PRESENT_SUPPORT)

    assert result.status == "ok"
    assert device.clicks == [(299, 812, "OCR_Result")]
    assert ctx.handler_state["unknown_retry_override"]["reason"] == "present_support_selection"


def test_schedule_handler_advances_present_support_showcase():
    device = _DeviceStub()
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(set()),
        device=device,
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        clip_manager=None,
    )
    ctx = ProduceContext()
    handler = ScheduleHandler()

    result = handler.handle(app, ctx, "schedule", GameplayPosition.SCHEDULE_PRESENT_SUPPORT_SHOWCASE)

    assert result.status == "ok"
    assert device.clicks == [(540, 819, "schedule-present-support-showcase")]
    assert ctx.handler_state["unknown_retry_override"]["reason"] == "present_support_showcase"


def test_schedule_handler_event_dialogue_delegates_to_dialogue_step(monkeypatch):
    called = {}

    def _fake_dialogue_step(_app, _ctx, *, position):
        called["position"] = position
        return SimpleNamespace(status="fast_forward")

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.gameplay.dialogue.execute_dialogue_step",
        _fake_dialogue_step,
    )

    device = _DeviceStub()
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(set()),
        device=device,
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        clip_manager=None,
    )
    ctx = ProduceContext()
    handler = ScheduleHandler()

    result = handler.handle(app, ctx, "schedule", "schedule_event_dialogue")

    assert result.status == "ok"
    assert called == {"position": "schedule_event_dialogue"}
    assert device.clicks == []
