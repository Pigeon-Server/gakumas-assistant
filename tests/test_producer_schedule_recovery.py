from types import SimpleNamespace

from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import schedule as schedule_module


def _candidate(index: int, title: str, action_id: str, kind: str):
    return schedule_module.ScheduleActionCandidate(
        index=index,
        title=title,
        kind=kind,
        recommended=False,
        selected=False,
        box=None,
        action_id=action_id,
        metadata={"schedule_family": action_id.removeprefix("schedule_action_")},
    )


def test_decide_schedule_action_prefers_refresh_when_stamina_is_critical(monkeypatch):
    ctx = ProduceContext()
    candidates = [
        _candidate(0, "休む", "schedule_action_refresh", "refresh"),
        _candidate(1, "SP课程", "schedule_action_lesson_vocal_sp", "vocal"),
    ]

    monkeypatch.setattr(
        schedule_module,
        "build_decision_state",
        lambda *_args, **_kwargs: {
            "economy": {"stamina": 2, "max_stamina": 35, "p_point": 6},
            "candidates": [
                {"index": 0, "label": "休む", "id": "schedule_action_refresh", "metadata": {"schedule_family": "refresh"}},
                {"index": 1, "label": "SP课程", "id": "schedule_action_lesson_vocal_sp", "metadata": {"schedule_family": "lesson_vocal_sp"}},
            ],
            "legal_actions": [0, 1],
            "llm_actions": [{"index": 0}, {"index": 1}],
            "stage_context": {},
            "llm_snapshot": {"stage_context": {}},
        },
    )
    monkeypatch.setattr(schedule_module, "invoke_decision_strategy", lambda *args, **kwargs: 1)

    chosen = schedule_module.decide_schedule_action(
        SimpleNamespace(),
        ctx,
        candidates,
        position="schedule_idle",
    )

    assert chosen == 0


def test_decide_schedule_action_prefers_outing_when_stamina_low_and_p_point_enough(monkeypatch):
    ctx = ProduceContext()
    candidates = [
        _candidate(0, "おでかけ", "schedule_action_outing", "outing"),
        _candidate(1, "营业", "schedule_action_business", "business"),
    ]

    monkeypatch.setattr(
        schedule_module,
        "build_decision_state",
        lambda *_args, **_kwargs: {
            "economy": {"stamina": 4, "max_stamina": 35, "p_point": 18},
            "candidates": [
                {"index": 0, "label": "おでかけ", "id": "schedule_action_outing", "metadata": {"schedule_family": "outing"}},
                {"index": 1, "label": "营业", "id": "schedule_action_business", "metadata": {"schedule_family": "business"}},
            ],
            "legal_actions": [0, 1],
            "llm_actions": [{"index": 0}, {"index": 1}],
            "stage_context": {},
            "llm_snapshot": {"stage_context": {}},
        },
    )
    monkeypatch.setattr(schedule_module, "invoke_decision_strategy", lambda *args, **kwargs: None)

    chosen = schedule_module.decide_schedule_action(
        SimpleNamespace(),
        ctx,
        candidates,
        position="schedule_idle",
    )

    assert chosen == 0
