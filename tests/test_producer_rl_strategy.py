from __future__ import annotations

from src.core.tasks.producer_challenge.gameplay.strategy.rl_strategy import (
    RLStrategy,
    build_battle_predict_payload,
)


def _battle_decision_state() -> dict:
    return {
        "phase": "lesson",
        "candidates": [
            {
                "index": 0,
                "id": "produce_card:card_alpha:0",
                "db_id": "card_alpha",
                "available": True,
                "metadata": {
                    "upgrade_count": 0,
                    "effect_types": ["ProduceExamEffectType_ExamLessonFix"],
                },
            },
            {
                "index": 1,
                "id": "produce_card:card_beta:0",
                "db_id": "card_beta",
                "available": True,
                "metadata": {
                    "upgrade_count": 0,
                    "effect_types": ["ProduceExamEffectType_ExamReview"],
                },
            },
            {
                "index": 2,
                "id": "end_turn",
                "db_id": "",
                "available": True,
                "metadata": {},
            },
        ],
        "llm_snapshot": {
            "scenario": "hajime",
            "difficulty": "master",
            "turn": 3,
            "remaining": 4,
            "score": 820,
            "target": 2000,
            "stamina": 8,
            "max_stamina": 15,
            "turn_color_label": "ボーカル",
            "parameter_stats": {
                "vocal": 450,
                "dance": 420,
                "visual": 390,
                "vocal_max": 1800,
                "dance_max": 1800,
                "visual_max": 1800,
            },
            "resources": {
                "block": 1,
                "review": 4,
                "aggressive": 2,
                "parameter_buff": 3,
                "lesson_buff": 1,
                "full_power_point": 0,
            },
            "hand": [
                {
                    "db_id": "card_alpha",
                    "name": "アピールA",
                    "category": "active",
                    "rarity": "ProduceCardRarity_R",
                    "upgrade_count": 0,
                    "cost": 3,
                    "description": "测试手牌",
                    "effect_types": ["ProduceExamEffectType_ExamLessonFix"],
                },
            ],
            "drinks": [
                {
                    "id": "drink_alpha",
                    "name": "ドリンクA",
                    "description": "测试饮料",
                    "effect_types": ["ProduceExamEffectType_ExamReview"],
                },
            ],
            "deck_cards": [],
            "grave_cards": [],
            "hold_cards": [],
            "lost_cards": [],
            "zone_counts": {"grave": 0},
            "active_enchants": ["status_alpha"],
        },
    }


def test_build_battle_predict_payload_uses_db_id_fields() -> None:
    """对 RL 服务的 battle payload 应使用稳定的 db_id / action_id。"""

    state, legal_actions = build_battle_predict_payload(_battle_decision_state()) or ({}, [])

    assert state["turn_color"] == "vocal"
    assert state["hand_cards"][0]["db_id"] == "card_alpha"
    assert state["drinks"][0]["db_id"] == "drink_alpha"
    assert legal_actions[0]["action_id"] == "produce_card:card_alpha:0"
    assert legal_actions[0]["db_id"] == "card_alpha"
    assert legal_actions[0]["kind"] == "card"
    assert "label" not in legal_actions[0]


def test_rl_strategy_can_fallback_to_action_identity_when_index_is_invalid() -> None:
    """服务返回越界索引时，仍应按 action_id / db_id 回落到合法候选。"""

    strategy = RLStrategy("http://127.0.0.1:8001")

    class _DummyClient:
        def predict(self, *_args, **_kwargs):
            return {
                "action_index": 99,
                "action_id": "produce_card:card_beta:0",
                "db_id": "card_beta",
                "action_kind": "card",
                "confidence": 1.0,
            }

    strategy._client = _DummyClient()
    result = strategy(
        app=None,
        ctx=None,
        candidates=[object(), object(), object()],
        decision_state=_battle_decision_state(),
    )

    assert result == {
        "action_index": 1,
        "action_id": "produce_card:card_beta:0",
        "db_id": "card_beta",
    }
