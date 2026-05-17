from types import SimpleNamespace

import numpy as np

from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import decision as decision_module
from src.core.tasks.producer_challenge.gameplay.llm.prompt_renderer import render
from src.constants.yolo.labels.producer_Labels import ProducerLabels


def _candidate(index: int, card_id: str, name: str, description: str, *, cost: int = 4):
    return SimpleNamespace(
        index=index,
        title=name,
        selected=False,
        box=None,
        action_id=f"produce_card:{card_id}:0",
        db_id=card_id,
        source="ocr",
        confidence=1.0,
        metadata={
            "description": description,
            "category": "ProduceCardCategory_ActiveSkill",
            "upgrade_count": 0,
            "effect_types": ["打分"],
            "cost": cost,
        },
    )


def _schedule_candidate(
    index: int,
    title: str,
    *,
    kind: str,
    recommended: bool = False,
    selected: bool = False,
):
    return SimpleNamespace(
        index=index,
        title=title,
        kind=kind,
        selected=selected,
        recommended=recommended,
        box=None,
        action_id=f"schedule_action:{index}",
        db_id="",
        source="ocr",
        confidence=1.0,
        metadata={"candidate_type": "schedule_action"},
    )


def _dialogue_candidate(index: int, title: str):
    return SimpleNamespace(
        index=index,
        title=title,
        selected=False,
        box=None,
        action_id=f"dialogue_option:{index}",
        db_id="",
        source="ocr",
        confidence=1.0,
        metadata={"candidate_type": "dialogue_option"},
    )


def _drink_entity(drink_id: str, name: str, description: str):
    return {
        "action_id": f"produce_drink:{drink_id}",
        "db_id": drink_id,
        "name": name,
        "source": "clip",
        "confidence": 0.98,
        "metadata": {
            "display_name": name,
            "description": description,
            "effect_types": ["元気"],
        },
    }


def test_nia_scenario_rules_fill_fan_vote_and_rest_limits():
    ctx = ProduceContext()
    ctx.scenario = "nia"
    ctx.current_exam_type = "second_audition"
    ctx.schedule_history = ["レッスン", "お休み", "相談", "休息"]
    ctx.handler_state["fan_votes_current"] = 12000

    payload = decision_module._scenario_rule_payload(ctx)

    assert payload["rest_limit_total"] == 4
    assert payload["rest_limit_used"] == 2
    assert payload["rest_limit_remaining"] == 2
    assert payload["fan_votes_current"] == 12000
    assert payload["fan_votes_next_threshold"] == 14000
    assert payload["fan_votes_gap"] == 2000
    assert payload["audition_stage_current"] == "second_audition"
    assert "休息剩余=2/4" in payload["summary"]


def _drink_candidate(index: int, drink_id: str, name: str, description: str):
    return SimpleNamespace(
        index=index,
        title=name,
        selected=False,
        box=None,
        action_id=f"produce_drink:{drink_id}",
        db_id=drink_id,
        source="clip",
        confidence=0.98,
        metadata={
            "display_name": name,
            "description": description,
            "effect_types": ["ProduceExamEffectType_Block"],
            "drink_score": 32.0,
        },
    )


def _end_turn_candidate(index: int, label: str = "SKIP"):
    return SimpleNamespace(
        index=index,
        title=label,
        selected=False,
        box=None,
        action_id="end_turn",
        db_id="",
        source="yolo",
        confidence=1.0,
        metadata={
            "candidate_type": "end_turn",
            "description": "执行 SKIP，放弃当前回合剩余出牌并直接进入下一回合。",
        },
    )


class _BoxList(list):
    def first(self):
        return self[0]


class _ResultsStub:
    def __init__(self, mapping):
        self._mapping = {
            label: _BoxList(list(items))
            for label, items in mapping.items()
        }

    def exists_label(self, label):
        return bool(self._mapping.get(label))

    def filter_by_label(self, label):
        return self._mapping.get(label, _BoxList())


def test_parse_stamina_text_supports_standard_and_concatenated_ocr():
    assert decision_module._parse_stamina_text("2/15") == (2, 15)
    assert decision_module._parse_stamina_text("215", previous_stamina=2, previous_max_stamina=15) == (2, 15)
    assert decision_module._parse_stamina_text("13", previous_stamina=5, previous_max_stamina=15) == (13, 15)
    assert decision_module._parse_stamina_text("15", previous_stamina=5, previous_max_stamina=15) == (5, 15)
    assert decision_module._parse_stamina_text("5114", previous_stamina=5, previous_max_stamina=14) == (5, 14)
    assert decision_module._parse_stamina_text("1427", previous_stamina=2, previous_max_stamina=7) == (2, 7)
    assert decision_module._parse_stamina_text("14227", previous_stamina=2, previous_max_stamina=7) == (2, 7)


def test_extract_hud_state_uses_current_stamina_as_max_when_upper_unknown(monkeypatch):
    stamina_box = SimpleNamespace(
        frame=np.zeros((120, 320, 3), dtype=np.uint8),
        x=100,
        y=100,
        w=420,
        h=220,
    )
    skill_card_box = SimpleNamespace(frame=np.zeros((20, 20, 3), dtype=np.uint8))
    app = SimpleNamespace(
        latest_results=_ResultsStub({
            ProducerLabels.PC_STAMINA: [stamina_box],
            ProducerLabels.SKILL_CARD_ACTIVE: [skill_card_box],
        }),
        latest_frame=np.zeros((1920, 1080, 3), dtype=np.uint8),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(decision_module, "ocr_text", lambda _img: "")
    parse_calls = iter(((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
    monkeypatch.setattr(
        decision_module,
        "_parse_stamina_text",
        lambda *_args, **_kwargs: next(parse_calls),
    )
    noisy_calls = iter(((0, False), (34, True)))
    monkeypatch.setattr(
        decision_module,
        "_extract_noisy_hud_value",
        lambda *_args, **_kwargs: next(noisy_calls),
    )

    hud = decision_module._extract_hud_state(app)

    assert hud["stamina"] == 34
    assert hud["max_stamina"] == 34


def test_parse_score_bonus_from_bonus_text_prefers_percent_and_turn_prefix_trim():
    assert decision_module._parse_score_bonus_from_bonus_text(
        "ダンス 5519%",
        remaining_turns_text="残り5ターン",
    ) == "519"
    assert decision_module._parse_score_bonus_from_bonus_text("ボーカル 519%") == "519"
    assert decision_module._parse_score_bonus_from_bonus_text("x1.8") == "1.8"


def test_build_llm_snapshot_exam_overrides_suspicious_hud_bonus_with_wheel(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["exam_wheel_info"] = {
        "queue": ["dance"],
        "remaining_turns": 5,
        "current_param": "dance",
        "current_bonus_pct": 519,
        "confidence": "low",
    }
    monkeypatch.setattr(
        decision_module,
        "_sync_virtual_battle_state",
        lambda *_args, **_kwargs: {
            "zones": {"hand": [], "deck": [], "grave": [], "hold": [], "lost": []},
            "resources": {
                "parameter_buff": "",
                "review": "",
                "aggressive": "",
                "block": 0,
                "lesson_buff": "",
                "enthusiastic": "",
                "full_power_point": "",
            },
            "turn_index": 1,
            "play_limit_remaining": 0,
            "play_limit_total_current": 2,
        },
    )
    monkeypatch.setattr(decision_module, "_build_current_deck_snapshot", lambda _ctx: [])
    monkeypatch.setattr(decision_module, "_build_drink_snapshot", lambda _drinks: [])
    monkeypatch.setattr(decision_module, "_build_produce_item_snapshot", lambda _ctx: [])
    monkeypatch.setattr(decision_module, "_build_formation_ability_snapshot", lambda _ctx: [])
    monkeypatch.setattr(decision_module, "_build_formation_event_snapshot", lambda _ctx: [])

    snapshot = decision_module._build_llm_snapshot(
        ctx,
        phase="exam",
        position="lesson_idle",
        hud_state={
            "stamina": 27,
            "max_stamina": 30,
            "genki": 1,
            "score": 0,
            "target_score": 150,
            "remaining_turns": 5,
            "turn_color": "ボーカル",
            "score_bonus": "5519",
            "exam_ranking": "5",
            "p_point": 0,
        },
        resolved_entities=[],
        stage_context={},
    )

    assert snapshot["score_bonus_multiplier"] == "519"


def test_extract_hud_state_keeps_repeated_genki_when_no_half_evidence(monkeypatch):
    stamina_box = SimpleNamespace(
        frame=np.zeros((120, 320, 3), dtype=np.uint8),
        x=100,
        y=100,
        w=420,
        h=220,
    )
    skill_card_box = SimpleNamespace(frame=np.zeros((20, 20, 3), dtype=np.uint8))
    app = SimpleNamespace(
        latest_results=_ResultsStub({
            ProducerLabels.PC_STAMINA: [stamina_box],
            ProducerLabels.SKILL_CARD_ACTIVE: [skill_card_box],
        }),
        latest_frame=np.zeros((1920, 1080, 3), dtype=np.uint8),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(decision_module, "ocr_text", lambda _img: "88")
    parse_calls = iter(((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
    monkeypatch.setattr(
        decision_module,
        "_parse_stamina_text",
        lambda *_args, **_kwargs: next(parse_calls),
    )
    noisy_calls = iter(((88, True), (34, True)))
    monkeypatch.setattr(
        decision_module,
        "_extract_noisy_hud_value",
        lambda *_args, **_kwargs: next(noisy_calls),
    )

    hud = decision_module._extract_hud_state(app)

    assert hud["genki"] == 88


def test_resolve_repeated_digit_ocr_value_is_conservative():
    assert decision_module._resolve_repeated_digit_ocr_value(88, "88", "88") == 88
    assert decision_module._resolve_repeated_digit_ocr_value(88, "88", "8", previous_value=0) == 8
    assert decision_module._resolve_repeated_digit_ocr_value(88, "88", "8", previous_value=80) == 88


def test_extract_noisy_hud_value_handles_duplicate_and_icon_noise():
    assert decision_module._extract_noisy_hud_value("1010", upper_bound=999) == (10, True)
    assert decision_module._extract_noisy_hud_value("227", "7", upper_bound=99) == (27, True)
    assert decision_module._extract_noisy_hud_value("17", "27", previous_value=27, upper_bound=99) == (27, True)


def test_extract_planning_parameter_value_prefers_seed_and_limit_over_sticky_ocr():
    assert decision_module._extract_planning_parameter_value(
        "24292",
        previous_value=242,
        upper_bound=1800,
    ) == (242, True)
    assert decision_module._extract_planning_parameter_value(
        "24292",
        previous_value=0,
        upper_bound=1800,
    ) == (242, True)
    assert decision_module._extract_planning_parameter_value(
        "370/1800 29.0%",
        previous_value=70,
        upper_bound=1800,
    ) == (370, True)
    assert decision_module._extract_planning_parameter_value(
        "370 1800 29.0%",
        previous_value=70,
        upper_bound=1800,
    ) == (370, True)
    assert decision_module._extract_planning_parameter_value(
        "31.4%",
        previous_value=280,
        upper_bound=1800,
    ) == (None, False)


def test_build_decision_state_tracks_virtual_hand_and_resources(monkeypatch):
    ctx = ProduceContext()
    ctx.formation_details = {
        "cards_and_items": {
            "matched_entries": [
                {"kind": "produce_card", "id": "card_a", "name": "卡A"},
                {"kind": "produce_card", "id": "card_b", "name": "卡B"},
                {"kind": "produce_item", "id": "item_x", "name": "道具X"},
            ],
            "produce_item_ids": ["item_x"],
        }
    }
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 20,
            "max_stamina": 35,
            "p_point": 12,
            "target_score": 100,
            "score": 50,
            "remaining_turns": 3,
            "turn_color": "ボーカル",
            "score_bonus": "1.5",
            "exam_ranking": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        decision_module,
        "_enrich_card_metadata",
        lambda card_id, upgrade_count=0: {
            "upgrade_count": upgrade_count,
            "rarity": "R",
            "category": "ProduceCardCategory_ActiveSkill",
            "cost": 4,
            "display_name": f"名称-{card_id}",
            "description": "集中+3 好調2ターン スキルカード使用数追加+1" if card_id == "card_a" else "打分+9",
            "effect_types": ["打分"],
            "trigger_phases": [],
        },
    )
    monkeypatch.setattr(
        decision_module,
        "_enrich_item_metadata",
        lambda item_id: {
            "rarity": "SR",
            "display_name": f"道具-{item_id}",
            "description": "开局获得额外收益",
        },
    )

    snapshot1 = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[
            _candidate(0, "card_a", "卡A", "集中+3 好調2ターン スキルカード使用数追加+1"),
            _candidate(1, "card_b", "卡B", "打分+9"),
        ],
        reason="test_snapshot_1",
    )

    assert snapshot1["llm_snapshot"]["deck_count"] == 2
    assert snapshot1["llm_snapshot"]["p_items"][0]["description"] == "开局获得额外收益"
    assert len(snapshot1["llm_snapshot"]["hand"]) == 2
    assert "Active×2" in snapshot1["llm_snapshot"]["deck_summary"]
    assert snapshot1["llm_snapshot"]["offensive_counts"]["deck"] >= 1

    ctx.record_operation("use_lesson_card", target="卡A", details={"db_id": "card_a"})

    snapshot2 = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[
            _candidate(0, "card_b", "卡B", "打分+9"),
        ],
        reason="test_snapshot_2",
    )

    assert snapshot2["llm_snapshot"]["resources"]["review"] == 3
    assert snapshot2["llm_snapshot"]["resources"]["parameter_buff"] == 2
    assert snapshot2["llm_snapshot"]["play_limit_total"] == 2
    assert snapshot2["llm_snapshot"]["play_limit_remaining"] == 1
    assert snapshot2["llm_snapshot"]["grave_cards"][0]["name"] == "卡A"
    assert isinstance(snapshot2["llm_snapshot"]["offensive_counts"]["grave"], int)
    snapshot_text = render("state_snapshot.j2", **snapshot2["llm_snapshot"])
    assert "牌组摘要:" in snapshot_text
    assert "火力牌统计:" in snapshot_text


def test_realtime_resource_snapshot_overrides_virtual_values(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 18,
            "max_stamina": 35,
            "p_point": 0,
            "target_score": 100,
            "score": 60,
            "remaining_turns": 2,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "2",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        decision_module,
        "_observe_bottom_inventory_drinks",
        lambda _app: ([_drink_entity("drink_guard", "元気茶", "元気+10")], True),
    )
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    decision_module.register_realtime_resource_snapshot(ctx, review=9, block=12)

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[],
        reason="test_realtime_override",
    )

    assert snapshot["llm_snapshot"]["resources"]["review"] == 9
    assert snapshot["llm_snapshot"]["resources"]["block"] == 12
    assert ctx.parameter_state["battle_resources"]["block"] == 12
    assert ctx.economy_state["battle_stamina"] == 18


def test_hud_param_stats_and_p_point_sync_to_schedule_prompt(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 14,
            "max_stamina": 35,
            "p_point": 18,
            "p_point_observed": True,
            "target_score": 120,
            "score": 72,
            "remaining_turns": 2,
            "turn_color": "ボーカル",
            "score_bonus": "1.8",
            "exam_ranking": "",
            "vocal": 123,
            "dance": 234,
            "visual": 345,
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision_module, "_observe_bottom_inventory_drinks", lambda _app: ([], False))
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="schedule",
        position="schedule_idle",
        candidates=[],
        reason="test_hud_param_sync",
    )

    assert ctx.hud_p_point == 18
    assert ctx.economy_state["p_point"] == 18
    assert ctx.parameter_state["vocal"] == 123
    assert ctx.parameter_state["dance"] == 234
    assert ctx.parameter_state["visual"] == 345
    assert snapshot["llm_snapshot"]["parameter_stats"] == {
        "vocal": 123,
        "dance": 234,
        "visual": 345,
        "vocal_max": 1000,
        "dance_max": 1000,
        "visual_max": 1000,
    }
    snapshot_text = render("state_snapshot.j2", **snapshot["llm_snapshot"])
    assert "Pポイント: 18" in snapshot_text
    assert "参数面板: ボーカル=123/1000, ダンス=234/1000, ビジュアル=345/1000" in snapshot_text


def test_lesson_prompt_includes_parameter_panel_with_growth_limit(monkeypatch):
    ctx = ProduceContext()
    ctx.selected_idol_card = SimpleNamespace(planType="ProducePlanType_Plan2")
    ctx.parameter_state = {
        "vocal": 123,
        "dance": 234,
        "visual": 345,
    }
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 14,
            "max_stamina": 35,
            "p_point": 18,
            "p_point_observed": True,
            "target_score": 120,
            "score": 72,
            "remaining_turns": 2,
            "turn_color": "ボーカル",
            "score_bonus": "1.8",
            "exam_ranking": "",
            "vocal": 123,
            "dance": 234,
            "visual": 345,
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision_module, "_observe_bottom_inventory_drinks", lambda _app: ([], False))
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[],
        reason="test_lesson_prompt_hides_schedule_panel",
    )

    snapshot_text = render("state_snapshot.j2", **snapshot["llm_snapshot"])
    assert "Pポイント:" not in snapshot_text
    assert "参数面板: ボーカル=123/1000, ダンス=234/1000, ビジュアル=345/1000" in snapshot_text
    assert "当前流派: ロジック | 核心资源: 好印象 / やる気 / 元気" in snapshot_text


def test_missing_hud_param_observation_keeps_previous_context_values(monkeypatch):
    ctx = ProduceContext()
    ctx.hud_p_point = 9
    ctx.parameter_state = {
        "vocal": 111,
        "dance": 222,
        "visual": 333,
    }
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 12,
            "max_stamina": 35,
            "p_point": 0,
            "p_point_observed": False,
            "target_score": 0,
            "score": 0,
            "remaining_turns": 0,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
            "vocal": None,
            "dance": None,
            "visual": None,
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="schedule",
        position="schedule_idle",
        candidates=[],
        reason="test_keep_previous_hud_values",
    )

    assert ctx.hud_p_point == 9
    assert ctx.economy_state["p_point"] == 9
    assert ctx.parameter_state["vocal"] == 111
    assert ctx.parameter_state["dance"] == 222
    assert ctx.parameter_state["visual"] == 333
    assert snapshot["llm_snapshot"]["parameter_stats"] == {
        "vocal": 111,
        "dance": 222,
        "visual": 333,
        "vocal_max": 1000,
        "dance_max": 1000,
        "visual_max": 1000,
    }


def test_sync_visible_planning_context_updates_p_point_and_param_stats(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace()

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "p_point": 18,
            "p_point_observed": True,
            "vocal": 123,
            "vocal_observed": True,
            "dance": 234,
            "dance_observed": True,
            "visual": 345,
            "visual_observed": True,
        },
    )

    hud_state = decision_module.sync_visible_planning_context(
        app,
        ctx,
        phase="schedule",
        position="schedule_idle",
        reason="test_visible_hud_sync",
    )

    assert hud_state["p_point"] == 18
    assert ctx.hud_p_point == 18
    assert ctx.economy_state["p_point"] == 18
    assert ctx.parameter_state["vocal"] == 123
    assert ctx.parameter_state["dance"] == 234
    assert ctx.parameter_state["visual"] == 345
    assert ctx.parameter_state["vocal_max"] == 1000
    assert ctx.parameter_state["dance_max"] == 1000
    assert ctx.parameter_state["visual_max"] == 1000
    assert ctx.last_sync_reason == "test_visible_hud_sync"


def test_sync_visible_planning_context_preserves_previous_values_when_unobserved(monkeypatch):
    ctx = ProduceContext()
    ctx.hud_p_point = 9
    ctx.economy_state = {"p_point": 9}
    ctx.parameter_state = {"vocal": 111, "dance": 222, "visual": 333}
    app = SimpleNamespace()

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "p_point": 0,
            "p_point_observed": False,
            "vocal": None,
            "vocal_observed": False,
            "dance": None,
            "dance_observed": False,
            "visual": None,
            "visual_observed": False,
        },
    )

    decision_module.sync_visible_planning_context(
        app,
        ctx,
        phase="schedule",
        position="schedule_idle",
        reason="test_visible_hud_keep",
    )

    assert ctx.hud_p_point == 9
    assert ctx.economy_state["p_point"] == 9
    assert ctx.parameter_state["vocal"] == 111
    assert ctx.parameter_state["dance"] == 222
    assert ctx.parameter_state["visual"] == 333


def test_schedule_prompt_keeps_stage_semantics_without_bias_text(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 22,
            "max_stamina": 35,
            "p_point": 18,
            "target_score": 0,
            "score": 0,
            "remaining_turns": 0,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
            "has_progress_hud": True,
            "recommend_action_text": "Da",
            "recommend_action_kind": "dance",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="schedule",
        position="schedule_idle",
        candidates=[
            _schedule_candidate(0, "营业", kind="business"),
            _schedule_candidate(1, "课程", kind="dance", recommended=True),
        ],
        reason="test_schedule_prompt",
    )

    llm_snapshot = snapshot["llm_snapshot"]
    stage_context = llm_snapshot["stage_context"]
    assert stage_context["label"] == "周行动选择"
    assert "system_recommendation" not in stage_context
    assert "recommended_names" not in stage_context
    assert "recommended" not in snapshot["llm_actions"][1]
    assert "operation_meaning" not in snapshot["llm_actions"][0]

    snapshot_text = render("state_snapshot.j2", **llm_snapshot)
    prompt_text = render("action_select.j2", snapshot=snapshot_text, actions=snapshot["llm_actions"])
    assert "阶段语义: 周行动选择" in snapshot_text
    assert "当前可执行: 可从候选周行动中选择一项" in snapshot_text
    assert "系统推荐" not in snapshot_text
    assert "[系统推荐]" not in prompt_text
    assert "操作含义" not in prompt_text
    assert llm_snapshot["schedule_context"]["history"] == []
    assert llm_snapshot["produce_goals"]["summary"] == ""


def test_schedule_lesson_stage_context_uses_course_semantics(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 22,
            "max_stamina": 35,
            "p_point": 18,
            "target_score": 0,
            "score": 0,
            "remaining_turns": 0,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
            "has_progress_hud": True,
            "recommend_action_text": "Vo",
            "recommend_action_kind": "vocal",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="schedule",
        position="schedule_lesson_options",
        candidates=[
            _schedule_candidate(0, "ボーカル", kind="vocal"),
            _schedule_candidate(1, "ダンス", kind="dance"),
        ],
        reason="test_schedule_lesson_options",
    )

    stage_context = snapshot["llm_snapshot"]["stage_context"]
    assert stage_context["id"] == "schedule_lesson_options"
    assert stage_context["label"] == "授业课程选择"
    assert "授業候选" in stage_context["available_action_summary"]
    snapshot_text = render("state_snapshot.j2", **snapshot["llm_snapshot"])
    assert "阶段语义: 授业课程选择" in snapshot_text
    assert "当前可执行: 可从当前授業候选中选择一项课程" in snapshot_text



def test_consult_stage_context_distinguishes_preview_and_confirm(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 22,
            "max_stamina": 35,
            "p_point": 18,
            "target_score": 0,
            "score": 0,
            "remaining_turns": 0,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
            "has_progress_hud": True,
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)

    preview_snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="consult",
        position="consult_enhancement_preview",
        candidates=[
            SimpleNamespace(
                index=0,
                title="アピールの基本",
                selected=False,
                box=None,
                action_id="consult_select_enhancement_target:0",
                db_id="card_a",
                source="ocr",
                confidence=1.0,
                metadata={"consult_action": "consult_select_enhancement_target"},
            ),
        ],
        reason="test_consult_preview",
    )
    ready_snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="consult",
        position="consult_enhancement_ready",
        candidates=[
            SimpleNamespace(
                index=0,
                title="アピールの基本",
                selected=True,
                box=None,
                action_id="consult_confirm_enhancement:0",
                db_id="card_a",
                source="ocr",
                confidence=1.0,
                metadata={"consult_action": "consult_confirm_enhancement"},
            ),
        ],
        reason="test_consult_ready",
    )

    preview_context = preview_snapshot["llm_snapshot"]["stage_context"]
    ready_context = ready_snapshot["llm_snapshot"]["stage_context"]
    assert preview_context["id"] == "consult_enhancement_preview"
    assert preview_context["label"] == "咨询强化预览"
    assert ready_context["id"] == "consult_enhancement_confirm"
    assert ready_context["label"] == "咨询强化确认"



def test_item_select_stage_context_marks_unresolved_probe_mode(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 22,
            "max_stamina": 35,
            "p_point": 18,
            "target_score": 0,
            "score": 0,
            "remaining_turns": 0,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
            "has_progress_hud": True,
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="item_select",
        position="item_select_idle",
        candidates=[
            SimpleNamespace(
                index=0,
                title="未知物品",
                selected=False,
                box=None,
                action_id="item_select:0",
                db_id="",
                source="ocr",
                confidence=0.2,
                metadata={},
            ),
            SimpleNamespace(
                index=1,
                title="アンクルウェイト",
                selected=False,
                box=None,
                action_id="produce_item:item_a",
                db_id="item_a",
                source="db",
                confidence=1.0,
                metadata={},
            ),
        ],
        reason="test_item_select_probe_mode",
    )

    stage_context = snapshot["llm_snapshot"]["stage_context"]
    assert stage_context["id"] == "item_select_probe"
    assert stage_context["label"] == "P物品探查选择"
    assert "探查未识别物品" in stage_context["available_action_summary"]



def test_p_drink_limit_stage_context_mentions_replace_tradeoff(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 22,
            "max_stamina": 35,
            "p_point": 18,
            "target_score": 0,
            "score": 0,
            "remaining_turns": 0,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
            "has_progress_hud": True,
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="p_drink",
        position="p_drink_limit",
        candidates=[
            SimpleNamespace(
                index=0,
                title="受け取らない",
                selected=False,
                box=None,
                action_id="p_drink_limit_skip_new",
                db_id="",
                source="ocr",
                confidence=1.0,
                metadata={"candidate_type": "p_drink_limit", "p_drink_limit_kind": "skip_new_drink"},
            ),
        ],
        reason="test_p_drink_limit_stage_context",
    )

    stage_context = snapshot["llm_snapshot"]["stage_context"]
    assert stage_context["id"] == "p_drink_limit"
    assert stage_context["label"] == "P饮料上限处理"
    assert "放弃新饮料" in stage_context["available_action_summary"]



def test_progress_dialogue_prompt_explains_schedule_event_semantics(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 24,
            "max_stamina": 35,
            "p_point": 20,
            "target_score": 0,
            "score": 0,
            "remaining_turns": 0,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
            "has_progress_hud": True,
            "recommend_action_text": "",
            "recommend_action_kind": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="dialogue",
        position="dialogue_options",
        candidates=[
            _dialogue_candidate(0, "接受建议"),
            _dialogue_candidate(1, "保持原计划"),
        ],
        reason="test_schedule_event_dialogue_prompt",
    )

    stage_context = snapshot["llm_snapshot"]["stage_context"]
    assert stage_context["label"] == "周事件选项"
    assert "周事件分支选择" in stage_context["description"]
    assert "operation_meaning" not in snapshot["llm_actions"][0]

    snapshot_text = render("state_snapshot.j2", **snapshot["llm_snapshot"])
    prompt_text = render("action_select.j2", snapshot=snapshot_text, actions=snapshot["llm_actions"])
    assert "阶段语义: 周事件选项" in snapshot_text
    assert "当前可执行: 可从当前周事件选项中选择一个分支" in snapshot_text
    assert "操作含义" not in prompt_text
    assert "接受建议" in prompt_text
    assert "保持原计划" in prompt_text


def test_lesson_bottom_bar_drinks_sync_into_snapshot_and_followup_schedule_prompt(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 19,
            "max_stamina": 35,
            "p_point": 12,
            "target_score": 120,
            "score": 64,
            "remaining_turns": 2,
            "turn_color": "ビジュアル",
            "score_bonus": "1.8",
            "exam_ranking": "3",
            "has_progress_hud": False,
            "recommend_action_text": "",
            "recommend_action_kind": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        decision_module,
        "_observe_bottom_inventory_drinks",
        lambda _app: (
            [
                _drink_entity("drink_guard", "元気茶", "元気+10"),
                _drink_entity("drink_focus", "集中ブレンド", "集中+4"),
            ],
            True,
        ),
    )
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    lesson_snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[],
        reason="test_lesson_bottom_drinks",
    )

    assert [drink["db_id"] for drink in ctx.recognized_p_drinks] == ["drink_guard", "drink_focus"]
    assert lesson_snapshot["llm_snapshot"]["drinks"][0]["description"] == "元気+10"
    assert lesson_snapshot["llm_snapshot"]["observability"]["drink_inventory_observed"] is True

    lesson_snapshot_text = render("state_snapshot.j2", **lesson_snapshot["llm_snapshot"])
    assert "### P饮料库存 (2瓶)" in lesson_snapshot_text
    assert "元気茶: 元気+10" in lesson_snapshot_text
    assert "底栏饮料库存已观测: 是" in lesson_snapshot_text

    schedule_snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="schedule",
        position="schedule_idle",
        candidates=[],
        reason="test_schedule_known_drinks",
    )
    schedule_snapshot_text = render("state_snapshot.j2", **schedule_snapshot["llm_snapshot"])
    assert "### 已知Pドリンク库存 (2瓶)" in schedule_snapshot_text
    assert "集中ブレンド: 集中+4" in schedule_snapshot_text


def test_battle_snapshot_prefers_legal_drink_over_inventory_clip_guess(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 30,
            "max_stamina": 34,
            "p_point": 0,
            "target_score": 0,
            "score": 0,
            "remaining_turns": 9,
            "turn_color": "ダンス",
            "score_bonus": "",
            "exam_ranking": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        decision_module,
        "_observe_bottom_inventory_drinks",
        lambda _app: ([_drink_entity("drink_wrong", "特制初星精华", "错误库存样本")], True),
    )
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="exam",
        position="exam_idle",
        candidates=[
            _drink_candidate(0, "drink_coffee", "热咖啡", "やる気+3"),
            _end_turn_candidate(1, "结束回合"),
        ],
        reason="test_prefer_legal_drink",
    )

    assert [drink["db_id"] for drink in ctx.recognized_p_drinks] == ["drink_coffee"]
    assert [drink["name"] for drink in snapshot["llm_snapshot"]["drinks"]] == ["热咖啡"]
    assert "特制初星精华" not in render("state_snapshot.j2", **snapshot["llm_snapshot"])


def test_battle_snapshot_marks_zero_stack_conversion_cards_unavailable(monkeypatch):
    ctx = ProduceContext()
    ctx.current_week = 2
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 18,
            "max_stamina": 35,
            "p_point": 0,
            "target_score": 100,
            "score": 60,
            "remaining_turns": 3,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        decision_module,
        "_observe_bottom_inventory_drinks",
        lambda _app: ([_drink_entity("drink_guard", "元気茶", "元気+10")], True),
    )
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[
            _candidate(0, "card_a", "可愛い仕草+", "好印象的120%转为打分"),
            _candidate(1, "card_b", "集中准备", "集中+3"),
        ],
        reason="test_zero_stack_unavailable",
    )

    assert snapshot["llm_actions"][0]["available"] is False
    assert "当前好印象=0" in snapshot["llm_actions"][0]["description"]
    assert snapshot["llm_actions"][1]["available"] is True


def test_battle_action_description_includes_effect_term_hint(monkeypatch):
    ctx = ProduceContext()
    ctx.selected_idol_card = SimpleNamespace(planType="ProducePlanType_Plan1")
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 16,
            "max_stamina": 35,
            "genki": 4,
            "p_point": 0,
            "p_point_observed": False,
            "target_score": 100,
            "score": 30,
            "remaining_turns": 2,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision_module, "_observe_bottom_inventory_drinks", lambda _app: ([], False))
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[
            _candidate(0, "card_a", "集中准备", "集中+3 好調2ターン スキルカード使用数追加+1"),
        ],
        reason="test_effect_term_hints",
    )

    action = snapshot["llm_actions"][0]
    description = action["description"]
    assert "术语提示：" not in description
    assert "集中=每+1都会再追加1点参数/得分基础值" not in description
    assert "好調=会把参数/得分上升量提高50%" not in description
    assert "engine_setup_value" in action["decision_tags"]


def test_battle_snapshot_filters_illegal_actions_by_cost_and_keeps_drinks_out_of_hand(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 3,
            "max_stamina": 35,
            "genki": 0,
            "genki_observed": True,
            "p_point": 0,
            "target_score": 100,
            "score": 42,
            "remaining_turns": 2,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        decision_module,
        "_observe_bottom_inventory_drinks",
        lambda _app: ([_drink_entity("drink_guard", "元気茶", "元気+10")], True),
    )
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[
            _candidate(0, "card_heavy", "重击", "高费打分", cost=6),
            _candidate(1, "card_light", "轻击", "低费打分", cost=2),
            _drink_candidate(2, "drink_guard", "元気茶", "元気+10"),
        ],
        reason="test_legal_action_filter",
    )

    assert snapshot["llm_actions"][0]["available"] is False
    assert snapshot["legal_actions"] == [1, 2]
    assert [card["id"] for card in snapshot["llm_snapshot"]["hand"]] == ["card_heavy", "card_light"]
    assert snapshot["llm_snapshot"]["drinks"][0]["id"] == "drink_guard"
    assert "当前体力只有3" in snapshot["candidates"][0]["metadata"]["unavailable_reason"]


def test_battle_snapshot_treats_genki_as_extra_cost_budget(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 2,
            "max_stamina": 35,
            "genki": 4,
            "genki_observed": True,
            "p_point": 0,
            "target_score": 100,
            "score": 42,
            "remaining_turns": 2,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision_module, "_observe_bottom_inventory_drinks", lambda _app: ([], False))
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="exam",
        position="exam_idle",
        candidates=[
            _candidate(0, "card_heavy", "重击", "高费打分", cost=6),
            _end_turn_candidate(1, "SKIP"),
        ],
        reason="test_genki_cost_budget",
    )

    assert snapshot["llm_actions"][0]["available"] is True
    assert snapshot["legal_actions"] == [0, 1]
    assert snapshot["llm_snapshot"]["genki"] == 4
    assert snapshot["llm_snapshot"]["resources"]["block"] == 4
    assert snapshot["economy"]["battle_genki"] == 4


def test_annotate_battle_candidate_availability_blocks_cards_when_play_limit_is_zero():
    ctx = ProduceContext()
    candidate_payloads = [
        {
            "index": 0,
            "id": "produce_card:card_a:0",
            "db_id": "card_a",
            "name": "卡A",
            "label": "卡A",
            "available": True,
            "metadata": {
                "description": "打分+8",
                "cost": 2,
            },
        },
        {
            "index": 1,
            "id": "produce_drink:drink_guard",
            "db_id": "drink_guard",
            "name": "元気茶",
            "label": "元気茶",
            "available": True,
            "metadata": {
                "description": "元気+10",
            },
        },
    ]
    llm_snapshot = {
        "play_limit_remaining": "0/1",
        "stamina": 30,
        "genki": 0,
        "resources": {},
    }

    decision_module._annotate_battle_candidate_availability(
        ctx,
        phase="lesson",
        candidate_payloads=candidate_payloads,
        llm_snapshot=llm_snapshot,
    )

    assert candidate_payloads[0]["available"] is False
    assert "本回合已没有剩余出牌次数" in candidate_payloads[0]["metadata"]["unavailable_reason"]
    assert candidate_payloads[1]["available"] is True


def test_battle_snapshot_exposes_end_turn_candidate_to_llm(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 4,
            "max_stamina": 10,
            "genki": 0,
            "genki_observed": True,
            "p_point": 0,
            "target_score": 100,
            "score": 42,
            "remaining_turns": 2,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[
            _candidate(0, "card_heavy", "重击", "高费打分", cost=6),
            _end_turn_candidate(1, "SKIP"),
        ],
        reason="test_end_turn_candidate",
    )

    assert snapshot["legal_actions"] == [1]
    assert snapshot["llm_actions"][1]["kind"] == "end_turn"
    assert "operation_meaning" not in snapshot["llm_actions"][1]
    snapshot_text = render("state_snapshot.j2", **snapshot["llm_snapshot"])
    prompt_text = render("action_select.j2", snapshot=snapshot_text, actions=snapshot["llm_actions"])
    assert "也可以选择 SKIP" in snapshot_text
    assert "end_turn" in prompt_text
    assert "操作含义" not in prompt_text


def test_lesson_prompt_does_not_expose_drink_operation_meaning(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 24,
            "max_stamina": 35,
            "genki": 0,
            "genki_observed": True,
            "p_point": 0,
            "target_score": 100,
            "score": 42,
            "remaining_turns": 3,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision_module, "_observe_bottom_inventory_drinks", lambda _app: ([], False))
    monkeypatch.setattr(decision_module, "_enrich_card_metadata", lambda *args, **kwargs: {})

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
        candidates=[
            _drink_candidate(0, "drink_guard", "守护饮料", "元気+8"),
            _end_turn_candidate(1, "SKIP"),
        ],
        reason="test_lesson_drink_operation_meaning",
    )

    assert "operation_meaning" not in snapshot["llm_actions"][0]
    prompt_text = render(
        "action_select.j2",
        snapshot=render("state_snapshot.j2", **snapshot["llm_snapshot"]),
        actions=snapshot["llm_actions"],
    )
    assert "守护饮料" in prompt_text
    assert "元気+8" in prompt_text
    assert "操作含义" not in prompt_text
    assert "确认使用这瓶 P 饮料" not in prompt_text



def test_exam_system_prompt_emphasizes_endgame_pressure_and_skip_tradeoff():
    prompt_text = render("system_exam.j2")

    assert "高压局面下" in prompt_text
    assert "不要把“前期先铺资源”机械套用到残局" in prompt_text
    assert "SKIP 会消耗 1 回合并恢复体力" in prompt_text
    assert "动作前检查" in prompt_text


def test_lesson_system_prompt_emphasizes_goal_pressure():
    prompt_text = render("system_lesson.j2")

    assert "离达成目标 / Perfect 仍有缺口时" in prompt_text
    assert "临近结束时，补牌、额外出牌、直接打分、合适的 SKIP 往往比纯铺垫更重要" in prompt_text


def test_exam_snapshot_surfaces_tactical_focus_summary(monkeypatch):
    ctx = ProduceContext()
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    monkeypatch.setattr(
        decision_module,
        "_extract_hud_state",
        lambda _app: {
            "stamina": 7,
            "max_stamina": 12,
            "genki": 3,
            "genki_observed": True,
            "p_point": 0,
            "target_score": 100,
            "score": 58,
            "remaining_turns": 2,
            "turn_color": "ボーカル",
            "score_bonus": "1.8",
            "exam_ranking": "4",
        },
    )
    monkeypatch.setattr(decision_module, "_annotate_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision_module, "_observe_bottom_inventory_drinks", lambda _app: ([], False))
    monkeypatch.setattr(
        decision_module,
        "_enrich_card_metadata",
        lambda card_id, upgrade_count=0: {
            "upgrade_count": upgrade_count,
            "category": "ProduceCardCategory_ActiveSkill",
            "cost": 4,
            "description": "打分+12",
            "effect_types": ["打分"],
        },
    )

    snapshot = decision_module.build_decision_state(
        app,
        ctx,
        phase="exam",
        position="exam_idle",
        candidates=[
            _candidate(0, "card_score", "高分卡", "打分+12"),
            _end_turn_candidate(1, "SKIP"),
        ],
        reason="test_exam_focus_summary",
    )

    snapshot_text = render("state_snapshot.j2", **snapshot["llm_snapshot"])

    assert "### 考试决策焦点" in snapshot_text
    assert "过线压力: 剩余2回合，分数58/100，当前排名第4位" in snapshot_text
    assert "本回合动作窗口: 剩余スキルカード使用数=1/1" in snapshot_text
    assert "火力储备: 手牌=" in snapshot_text
    assert "资源兑现提醒:" in snapshot_text


def test_followup_exam_snapshot_without_previous_state_still_renders():
    ctx = ProduceContext()

    snapshot = decision_module.build_followup_decision_state(
        ctx,
        phase="exam",
        position="exam_retry_confirm_modal",
        candidates=[_end_turn_candidate(0, "SKIP")],
        reason="test_followup_exam_snapshot_without_previous_state",
    )

    snapshot_text = render("state_snapshot.j2", **snapshot["llm_snapshot"])

    assert "### 考试决策焦点" in snapshot_text
    assert "火力储备: 手牌=0, 牌库=0, 弃牌=0, 保留=0" in snapshot_text
    assert "### 手牌 (0张)" in snapshot_text
    assert "### 牌区计数" in snapshot_text


def test_annotate_candidates_marks_gray_overlay_cards_unavailable():
    frame = np.full((120, 80, 3), 148, dtype=np.uint8)
    candidate = SimpleNamespace(
        index=0,
        title="灰卡",
        selected=False,
        action_id="produce_card:gray_card:0",
        db_id="gray_card",
        metadata={},
        box=SimpleNamespace(
            x=120,
            y=220,
            w=80,
            h=120,
            frame=frame,
        ),
    )
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    decision_module._annotate_candidates(app, phase="lesson", candidates=[candidate])

    assert candidate.metadata["available"] is False
    assert "灰色禁用蒙版" in candidate.metadata["unavailable_reason"]


def test_annotate_candidates_keeps_bright_cards_available():
    frame = np.full((120, 80, 3), 220, dtype=np.uint8)
    candidate = SimpleNamespace(
        index=0,
        title="亮卡",
        selected=False,
        action_id="produce_card:bright_card:0",
        db_id="bright_card",
        metadata={},
        box=SimpleNamespace(
            x=120,
            y=220,
            w=80,
            h=120,
            frame=frame,
        ),
    )
    app = SimpleNamespace(debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None))

    decision_module._annotate_candidates(app, phase="lesson", candidates=[candidate])

    assert candidate.metadata.get("available", True) is True
    assert not candidate.metadata.get("unavailable_reason")
