from __future__ import annotations

from src.constants.game.produce_enums import ProducePlanType
from src.core.tasks.producer_challenge.gameplay.strategy.algo_strategy import ScheduleAlgoStrategy


def _schedule_decision_state(*, stamina: int, max_stamina: int, weeks_until_gate: int = 3) -> dict:
    return {
        "phase": "schedule",
        "position": "schedule_recommend",
        "candidates": [
            {
                "index": 0,
                "id": "schedule_action_refresh",
                "action_id": "schedule_action_refresh",
                "label": "休む",
                "name": "休む",
                "available": True,
                "metadata": {"schedule_family": "refresh"},
            },
            {
                "index": 1,
                "id": "schedule_action_lesson_dance_sp",
                "action_id": "schedule_action_lesson_dance_sp",
                "label": "授業",
                "name": "授業",
                "available": True,
                "metadata": {"schedule_family": "lesson_dance_sp"},
            },
        ],
        "legal_actions": [0, 1],
        "llm_snapshot": {
            "phase": "schedule",
            "position": "schedule_recommend",
            "scenario": "hajime",
            "difficulty": "regular",
            "week": 10,
            "current_week": 10,
            "stamina": stamina,
            "max_stamina": max_stamina,
            "score": 0,
            "target": 0,
            "resources": {},
            "parameter_stats": {
                "vocal": 400,
                "dance": 486,
                "visual": 591,
                "vocal_max": 1000,
                "dance_max": 1000,
                "visual_max": 1000,
            },
            "planning": {
                "next_gate": {
                    "gate_type": "exam",
                    "weeks_until_gate": weeks_until_gate,
                }
            },
            "hand": [],
            "deck_cards": [],
            "grave_cards": [],
            "hold_cards": [],
            "lost_cards": [],
            "drinks": [],
            "p_items": [],
            "p_point": 254,
        },
    }


def test_schedule_algo_does_not_rest_when_stamina_is_still_healthy() -> None:
    strategy = ScheduleAlgoStrategy()
    decision = strategy(
        app=None,
        ctx=type(
            "_Ctx",
            (),
            {
                "selected_idol_card": type(
                    "_IdolCard",
                    (),
                    {
                        "planType": ProducePlanType.PLAN1.value,
                        "produceVocalGrowthRatePermil": 0,
                        "produceDanceGrowthRatePermil": 1000,
                        "produceVisualGrowthRatePermil": 1500,
                    },
                )(),
                "target_idol_card_id": "",
                "deck_mutations": [],
            },
        )(),
        candidates=[object(), object()],
        decision_state=_schedule_decision_state(stamina=22, max_stamina=30, weeks_until_gate=3),
    )

    assert decision is not None
    assert decision.selected_action_id == "schedule_action_lesson_dance_sp"


def test_schedule_algo_still_prefers_rest_when_stamina_is_critical() -> None:
    strategy = ScheduleAlgoStrategy()
    decision = strategy(
        app=None,
        ctx=type(
            "_Ctx",
            (),
            {
                "selected_idol_card": type(
                    "_IdolCard",
                    (),
                    {
                        "planType": ProducePlanType.PLAN1.value,
                        "produceVocalGrowthRatePermil": 0,
                        "produceDanceGrowthRatePermil": 1000,
                        "produceVisualGrowthRatePermil": 1500,
                    },
                )(),
                "target_idol_card_id": "",
                "deck_mutations": [],
            },
        )(),
        candidates=[object(), object()],
        decision_state=_schedule_decision_state(stamina=4, max_stamina=30, weeks_until_gate=1),
    )

    assert decision is not None
    assert decision.selected_action_id == "schedule_action_refresh"
