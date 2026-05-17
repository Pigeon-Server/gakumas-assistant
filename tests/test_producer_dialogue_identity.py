from types import SimpleNamespace

from src.core.tasks.producer_challenge.gameplay.decision_support import identity as identity_module


def test_description_text_joins_entries_with_separator():
    text = identity_module._description_text(
        [
            {"text": "40%の確率で成功しダンス上昇+110"},
            {"text": "60%の確率で失敗しダンス上昇+45"},
        ]
    )
    assert text == "40%の確率で成功しダンス上昇+110；60%の確率で失敗しダンス上昇+45"


def test_load_dialogue_option_effect_entries_from_game_database(monkeypatch):
    class _FakeDB:
        def get_all_item(self):
            return [
                SimpleNamespace(
                    id="p_s_e_s-dialogue-vocal-20",
                    producePoint=0,
                    stamina=0,
                    produceEffectIds=["p_effect-vocal_addition-0020_0020"],
                    produceDescriptions=[SimpleNamespace(text="ボーカル上昇+20")],
                )
            ]

    monkeypatch.setattr(
        "src.utils.game_database_tools.get_game_database",
        lambda table_name: _FakeDB() if table_name == "ProduceStepEventSuggestion" else None,
    )
    identity_module._dialogue_option_effect_entries = None

    entries = identity_module._load_dialogue_option_effect_entries()

    assert len(entries) == 1
    assert entries[0]["id"] == "p_s_e_s-dialogue-vocal-20"
    assert entries[0]["norm_desc"] == "ボーカル上昇+20"
    assert entries[0]["raw_desc"] == "ボーカル上昇+20"
    identity_module._dialogue_option_effect_entries = None


def test_load_outing_activity_entries_from_game_database(monkeypatch):
    class _FakeDB:
        def get_all_item(self):
            return [
                SimpleNamespace(
                    id="p_s_e_s-event-detail-activity-001-006-01",
                    producePoint=100,
                    produceEffectIds=["p_effect-any"],
                    produceDescriptions=[SimpleNamespace(text="最大体力の30%分回復")],
                )
            ]

    monkeypatch.setattr(
        "src.utils.game_database_tools.get_game_database",
        lambda table_name: _FakeDB() if table_name == "ProduceStepEventSuggestion" else None,
    )
    identity_module._outing_activity_entries = None

    entries = identity_module._load_outing_activity_entries()

    assert len(entries) == 1
    assert entries[0]["id"] == "p_s_e_s-event-detail-activity-001-006-01"
    assert entries[0]["produce_point"] == 100
    assert entries[0]["raw_desc"] == "最大体力の30%分回復"
    identity_module._outing_activity_entries = None


def test_resolve_dialogue_option_identity_matches_db_by_effect(monkeypatch):
    monkeypatch.setattr(
        identity_module,
        "_load_dialogue_option_effect_entries",
        lambda: [
            {
                "id": "p_s_e_s-dialogue-visual-20",
                "produce_point": 0,
                "norm_desc": "ビジュアル上昇20",
                "raw_desc": "ビジュアル上昇+20",
                "effect_ids": ["p_effect-visual_addition-0020_0020"],
                "stamina": 0,
            },
        ],
    )

    resolved = identity_module.resolve_dialogue_option_identity(
        "ビジュアルを重点的に",
        index=2,
        effect_text="ビジュアル上昇+20",
    )

    assert resolved.source == "db_match"
    assert resolved.db_id == "p_s_e_s-dialogue-visual-20"
    assert resolved.action_id == "dialogue_option:p_s_e_s-dialogue-visual-20"
    assert resolved.metadata["dialogue_db_description"] == "ビジュアル上昇+20"
    assert resolved.metadata["description"] == "効果: ビジュアル上昇+20"
    assert resolved.metadata["param_kind"] == "visual"


def test_hydrate_dialogue_candidates_uses_option_effect_for_db_match(monkeypatch):
    monkeypatch.setattr(
        identity_module,
        "_load_dialogue_option_effect_entries",
        lambda: [
            {
                "id": "p_s_e_s-dialogue-dance-30",
                "produce_point": 0,
                "norm_desc": "ダンス上昇30",
                "raw_desc": "ダンス上昇+30",
                "effect_ids": ["p_effect-dance_addition-0030_0030"],
                "stamina": 0,
            },
        ],
    )
    candidate = SimpleNamespace(
        index=0,
        title="ダンスを重点的に",
        action_id="dialogue_option:idx_0",
        db_id="",
        source="ocr",
        confidence=0.0,
        metadata={"option_effect": "ダンス上昇+30"},
    )

    identity_module.hydrate_dialogue_candidates([candidate])

    assert candidate.db_id == "p_s_e_s-dialogue-dance-30"
    assert candidate.action_id == "dialogue_option:p_s_e_s-dialogue-dance-30"
    assert candidate.source == "db_match"
    assert candidate.metadata["dialogue_db_description"] == "ダンス上昇+30"


def test_resolve_dialogue_option_identity_falls_back_without_effect(monkeypatch):
    monkeypatch.setattr(
        identity_module,
        "_load_dialogue_option_effect_entries",
        lambda: [],
    )

    resolved = identity_module.resolve_dialogue_option_identity(
        "ボーカルを重点的に",
        index=0,
        effect_text="",
    )

    assert resolved.source == "ocr"
    assert resolved.db_id == ""
    assert resolved.action_id.startswith("dialogue_option:")
    assert resolved.metadata["param_kind"] == "vocal"
