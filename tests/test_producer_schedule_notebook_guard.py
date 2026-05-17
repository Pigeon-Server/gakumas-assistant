from types import SimpleNamespace

import numpy as np

from src.constants.game.producer_gameplay import GameplayPosition
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import schedule as schedule_module


class _DeviceStub:
    def __init__(self):
        self.clicks = []

    def click_element(self, element):
        self.clicks.append((getattr(element, "cx", 0), getattr(element, "cy", 0)))


class _ResultsStub:
    def __init__(self, mapping):
        self._mapping = mapping

    def filter_by_label(self, label):
        return list(self._mapping.get(label, []))


def test_schedule_handler_blocks_decision_when_notebook_not_confirmed_closed(monkeypatch):
    called = {"execute": False}

    def _fake_execute(*_args, **_kwargs):
        called["execute"] = True
        return None

    monkeypatch.setattr(schedule_module, "_detect_p_notebook_button", lambda _app: None)
    monkeypatch.setattr(schedule_module, "_detect_p_notebook_close_button", lambda _app: object())
    monkeypatch.setattr(schedule_module, "_close_p_notebook", lambda _app, **_kwargs: False)
    monkeypatch.setattr(schedule_module, "execute_schedule_step", _fake_execute)

    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({}),
        game_utils=SimpleNamespace(wait_frame_stable=lambda **_kwargs: None),
        device=_DeviceStub(),
    )
    ctx = ProduceContext()
    handler = schedule_module.ScheduleHandler()

    result = handler.handle(app, ctx, "schedule", GameplayPosition.SCHEDULE_RECOMMEND)

    assert result.status == "no_action"
    assert called["execute"] is False


def test_schedule_handler_allows_decision_when_controls_visible_even_without_manual_button(monkeypatch):
    called = {"execute": False}

    def _fake_execute(*_args, **_kwargs):
        called["execute"] = True
        return SimpleNamespace(
            status="selected",
            candidate=SimpleNamespace(index=0, title="授業", kind="vocal"),
        )

    monkeypatch.setattr(schedule_module, "_detect_p_notebook_button", lambda _app: None)
    monkeypatch.setattr(schedule_module, "_detect_p_notebook_close_button", lambda _app: object())
    monkeypatch.setattr(schedule_module, "execute_schedule_step", _fake_execute)

    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            schedule_module.ProducerLabels.PC_ACTION: [SimpleNamespace()],
            schedule_module.ProducerLabels.PC_RECOMMEND_ACTION: [SimpleNamespace()],
        }),
        game_utils=SimpleNamespace(wait_frame_stable=lambda **_kwargs: None),
        device=_DeviceStub(),
    )
    ctx = ProduceContext()
    handler = schedule_module.ScheduleHandler()

    result = handler.handle(app, ctx, "schedule", GameplayPosition.SCHEDULE_RECOMMEND)

    assert result.status == "ok"
    assert called["execute"] is True


def test_schedule_handler_skips_notebook_read_after_decision_same_week(monkeypatch):
    monkeypatch.setattr(schedule_module, "_detect_p_notebook_button", lambda _app: object())
    monkeypatch.setattr(
        schedule_module,
        "read_p_notebook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not read notebook")),
    )
    monkeypatch.setattr(
        schedule_module,
        "execute_schedule_step",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="selected",
            candidate=SimpleNamespace(index=0, title="授業", kind="vocal"),
        ),
    )

    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(filter_by_label=lambda _label: []),
        game_utils=SimpleNamespace(wait_frame_stable=lambda **_kwargs: None),
        device=_DeviceStub(),
    )
    ctx = ProduceContext(schedule_notebook_mode="before_decision")
    ctx.current_week = 3
    ctx.handler_state["schedule_action_decided_week"] = 3
    handler = schedule_module.ScheduleHandler()

    result = handler.handle(app, ctx, "schedule", GameplayPosition.SCHEDULE_IDLE)

    assert result.status == "ok"


def test_execute_schedule_step_marks_decision_week(monkeypatch):
    candidate = schedule_module.ScheduleActionCandidate(
        index=0,
        title="授業",
        kind="vocal",
        recommended=False,
        selected=False,
        box=SimpleNamespace(cx=300, cy=600, label="candidate"),
        action_id="schedule_action_class",
        metadata={},
    )

    monkeypatch.setattr(
        schedule_module,
        "collect_schedule_action_candidates",
        lambda *_args, **_kwargs: [candidate],
    )
    monkeypatch.setattr(schedule_module, "decide_schedule_action", lambda *_args, **_kwargs: 0)

    app = SimpleNamespace(device=_DeviceStub())
    ctx = ProduceContext()
    ctx.current_week = 5

    result = schedule_module.execute_schedule_step(
        app,
        ctx,
        position=GameplayPosition.SCHEDULE_IDLE,
    )

    assert result is not None
    assert ctx.handler_state["schedule_action_decided_week"] == 5


def test_execute_schedule_step_reuses_pending_action_without_redeciding(monkeypatch):
    candidate = schedule_module.ScheduleActionCandidate(
        index=0,
        title="活动支给",
        kind="present",
        recommended=False,
        selected=False,
        box=SimpleNamespace(cx=280, cy=620, label="selected"),
        action_id="schedule_action",
        metadata={},
    )
    calls = {"decide": 0}

    monkeypatch.setattr(
        schedule_module,
        "collect_schedule_action_candidates",
        lambda *_args, **_kwargs: [candidate],
    )

    def _fake_decide(*_args, **_kwargs):
        calls["decide"] += 1
        return 0

    monkeypatch.setattr(schedule_module, "decide_schedule_action", _fake_decide)

    app = SimpleNamespace(device=_DeviceStub())
    ctx = ProduceContext()

    first = schedule_module.execute_schedule_step(
        app,
        ctx,
        position=GameplayPosition.SCHEDULE_RECOMMEND,
    )
    second = schedule_module.execute_schedule_step(
        app,
        ctx,
        position=GameplayPosition.SCHEDULE_RECOMMEND,
    )

    assert first is not None and first.status == "selected"
    assert second is not None and second.status == "selected"
    assert calls["decide"] == 1
    assert app.device.clicks == [(280, 620), (280, 620)]
    assert ctx.handler_state["pending_schedule_click_count"] == 2


def test_execute_schedule_step_schedule_selected_confirms_selected_without_redecide(monkeypatch):
    selected_candidate = schedule_module.ScheduleActionCandidate(
        index=0,
        title="活動支給",
        kind="present",
        recommended=False,
        selected=True,
        box=SimpleNamespace(cx=280, cy=620, label="selected"),
        action_id="schedule_action",
        metadata={},
    )
    other_candidate = schedule_module.ScheduleActionCandidate(
        index=1,
        title="相談",
        kind="consult",
        recommended=False,
        selected=False,
        box=SimpleNamespace(cx=760, cy=620, label="other"),
        action_id="schedule_action_consult",
        metadata={},
    )

    monkeypatch.setattr(
        schedule_module,
        "collect_schedule_action_candidates",
        lambda *_args, **_kwargs: [selected_candidate, other_candidate],
    )
    monkeypatch.setattr(
        schedule_module,
        "decide_schedule_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not re-decide")),
    )

    app = SimpleNamespace(device=_DeviceStub())
    ctx = ProduceContext()
    ctx.handler_state["pending_schedule_action_id"] = "schedule_action"

    result = schedule_module.execute_schedule_step(
        app,
        ctx,
        position=GameplayPosition.SCHEDULE_SELECTED,
    )

    assert result is not None
    assert result.status == "confirmed"
    assert result.candidate.index == 0
    assert app.device.clicks == [(280, 620)]
