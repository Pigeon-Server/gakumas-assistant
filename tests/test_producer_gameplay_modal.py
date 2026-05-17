from types import SimpleNamespace

import numpy as np

from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay.modal import ModalHandler


class _DeviceStub:
    def __init__(self):
        self.clicked = []

    def click_element(self, element):
        self.clicked.append(element)


class _ResultsStub:
    def __init__(self, labels: dict[str, list[object]] | None = None):
        self._labels = labels or {}

    def filter_by_label(self, label):
        return list(self._labels.get(label, []))


def _seed_last_exam_decision_state(ctx: ProduceContext) -> None:
    ctx.current_week = 5
    ctx.handler_state["last_decision_state"] = {
        "phase": "exam",
        "position": "exam_idle",
        "llm_snapshot": {
            "phase": "exam",
            "position": "exam_idle",
            "scenario": "hajime",
            "difficulty": "master",
            "week": 5,
            "turn": 4,
            "remaining": 2,
            "max_turns": 6,
            "battle_kind": "exam",
            "battle_kind_label": "試験",
            "score": 12000,
            "target": 15000,
            "ratio": "80%",
            "stamina": 6,
            "max_stamina": 10,
            "genki": 4,
            "play_limit_remaining": 1,
            "play_limit_total": 1,
            "turn_color_label": "vo",
            "turn_color_display_label": "Vo",
            "score_bonus_multiplier": "1.5",
            "exam_ranking": 2,
            "parameter_stats": {"vocal": "", "dance": "", "visual": ""},
            "hand": [
                {"name": "Wよそ見はダメ♪+", "category": "アクティブ", "description": "测试手牌"},
                {"name": "幸せな時間", "category": "メンタル", "description": "测试手牌2"},
            ],
            "deck_count": 12,
            "deck_summary": "测试牌组",
            "deck_cards": [{"name": "可愛い仕草"}],
            "grave_cards": [],
            "hold_cards": [],
            "lost_cards": [],
            "zone_counts": {"deck": 12, "grave": 0, "hold": 0, "lost": 0},
            "offensive_counts": {"hand": 1, "deck": 1, "grave": 0, "hold": 0},
            "reshuffle_hint": "",
            "resources": {
                "parameter_buff": "",
                "review": "",
                "aggressive": "",
                "block": 4,
                "lesson_buff": "",
                "enthusiastic": "",
                "full_power_point": "",
            },
            "stance_desc": "",
            "negatives": "",
            "active_effects": [],
            "active_enchants": [],
            "drinks": [],
            "available_drink_count": 0,
            "used_drink_count": 0,
            "drink_total_count": 0,
            "p_items": [],
            "gimmicks": "",
            "total_counters": {"play_count": 0, "stamina_spent": "", "block_consumed": ""},
            "observability": {
                "deck_order_known": False,
                "resource_panel_parsed": True,
                "exam_ranking_observed": True,
                "turn_color_observed": True,
                "drink_inventory_observed": False,
                "empty_hand_observed": False,
            },
            "stage_context": {"id": "exam_card_play", "is_schedule_context": False},
        },
        "economy": {"stamina": 6, "max_stamina": 10, "p_point": 0},
        "parameters": {"target_score": 15000, "score": 12000, "remaining_turns": 2},
        "inventory": {"p_drinks": []},
        "card_zones": {"hand": []},
        "observability": {"empty_hand_observed": False, "drink_inventory_observed": False},
    }


def test_modal_handler_cancels_invalid_skill_use_modal(monkeypatch):
    clicked = {}

    def _click_modal_action_with_retry(app, modal, *, prefer_confirm, **kwargs):
        clicked["prefer_confirm"] = prefer_confirm
        return True

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.ui.click_modal_action_with_retry",
        _click_modal_action_with_retry,
    )

    modal = SimpleNamespace(
        modal_title="スキルカード使用確認",
        modal_body_text="好印象の値が0のため効果が発動しません。実行しますか?",
        confirm_button=SimpleNamespace(name="confirm"),
        cancel_button=SimpleNamespace(name="cancel"),
    )
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        game_utils=SimpleNamespace(try_get_modal=lambda no_body=False: modal),
    )
    ctx = ProduceContext()
    ctx.current_week = 2
    ctx.handler_state["battle_last_attempted_card"] = {
        "turn_marker": ("lesson", 2, 3),
        "title": "可愛い仕草+",
        "action_id": "produce_card:skill:0",
        "db_id": "skill",
    }

    result = ModalHandler().handle(app, ctx, phase="modal", position="gameplay_modal")

    assert result.status == "ok"
    assert clicked["prefer_confirm"] is False
    assert ctx.handler_state["battle_blocked_cards"]["turn_marker"] == ("lesson", 2, 3)
    assert "produce_card:skill:0" in ctx.handler_state["battle_blocked_cards"]["keys"]


def test_modal_handler_uses_fallback_ocr_to_cancel_invalid_skill_use(monkeypatch):
    cancel_button = SimpleNamespace(name="cancel", cy=2200)
    confirm_button = SimpleNamespace(name="confirm", cy=2200)

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.gameplay.modal._read_modal_fallback_text",
        lambda _app: "スキルカード使用確認 好印象の値が0のため効果が発動しません。実行しますか?",
    )

    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(
            {
                "Universal Cancel button": [cancel_button],
                "Universal Confirm button": [confirm_button],
            }
        ),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        game_utils=SimpleNamespace(try_get_modal=lambda no_body=False: None),
    )
    ctx = ProduceContext()
    ctx.current_week = 2
    ctx.handler_state["battle_last_attempted_card"] = {
        "turn_marker": ("lesson", 2, 3),
        "title": "可愛い仕草+",
        "action_id": "produce_card:skill:0",
        "db_id": "skill",
    }

    result = ModalHandler().handle(app, ctx, phase="modal", position="gameplay_modal")

    assert result.status == "ok"
    assert app.device.clicked == [cancel_button]
    assert "skill" in ctx.handler_state["battle_blocked_cards"]["keys"]


def test_modal_handler_retries_connection_error_instead_of_cancel_fallback(monkeypatch):
    confirm_button = SimpleNamespace(name="confirm", cy=2200)
    cancel_button = SimpleNamespace(name="cancel", cy=2200)

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.gameplay.modal._read_modal_fallback_text",
        lambda _app: "通信エラー 通信中にエラーが発生しました リトライ タイトルへ",
    )

    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(
            {
                "Universal Confirm button": [confirm_button],
                "Universal Cancel button": [cancel_button],
            }
        ),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        game_utils=SimpleNamespace(try_get_modal=lambda no_body=False: None),
    )
    ctx = ProduceContext()
    ctx.handler_state["modal_stuck_count"] = 2

    result = ModalHandler().handle(app, ctx, phase="modal", position="gameplay_modal")

    assert result.status == "ok"
    assert app.device.clicked == [confirm_button]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "connection_error_modal",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_modal_handler_routes_exam_retry_confirm_to_llm_retry(monkeypatch):
    clicked = {}
    captured = {}

    def _click_modal_action_with_retry(app, modal, *, prefer_confirm, **kwargs):
        clicked["prefer_confirm"] = prefer_confirm
        return True

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.ui.click_modal_action_with_retry",
        _click_modal_action_with_retry,
    )

    modal = SimpleNamespace(
        modal_title="再挑戦確認",
        modal_body_text="再挑戦が可能ですが、本当に終了しますか？ あと2回",
        confirm_button=SimpleNamespace(name="produce_end"),
        cancel_button=SimpleNamespace(name="retry"),
    )
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        game_utils=SimpleNamespace(try_get_modal=lambda no_body=False: modal),
    )
    ctx = ProduceContext(difficulty="master")
    _seed_last_exam_decision_state(ctx)

    def _strategy(app, ctx, candidates, decision_state):
        captured["state"] = decision_state
        return 0

    ctx.modal_strategy = _strategy

    result = ModalHandler().handle(
        app,
        ctx,
        phase="modal",
        position="exam_retry_confirm_modal",
    )

    assert result.status == "ok"
    assert clicked["prefer_confirm"] is False
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "exam_retry_confirm_modal",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }
    assert captured["state"]["stage_context"]["id"] == "exam_retry_confirm"
    assert captured["state"]["llm_snapshot"]["hand"][0]["name"] == "Wよそ見はダメ♪+"
    assert captured["state"]["llm_snapshot"]["position"] == "exam_retry_confirm_modal"
    assert captured["state"]["llm_actions"][0]["label"] == "再挑戦"


def test_modal_handler_routes_exam_retry_confirm_to_llm_produce_end(monkeypatch):
    clicked = {}

    def _click_modal_action_with_retry(app, modal, *, prefer_confirm, **kwargs):
        clicked["prefer_confirm"] = prefer_confirm
        return True

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.ui.click_modal_action_with_retry",
        _click_modal_action_with_retry,
    )

    modal = SimpleNamespace(
        modal_title="再挑戦確認",
        modal_body_text="再挑戦が可能ですが、本当に終了しますか？ あと1回",
        confirm_button=SimpleNamespace(name="produce_end"),
        cancel_button=SimpleNamespace(name="retry"),
    )
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        game_utils=SimpleNamespace(try_get_modal=lambda no_body=False: modal),
    )
    ctx = ProduceContext(difficulty="master")
    _seed_last_exam_decision_state(ctx)
    ctx.modal_strategy = lambda app, ctx, candidates, decision_state: 1

    result = ModalHandler().handle(
        app,
        ctx,
        phase="modal",
        position="exam_retry_confirm_modal",
    )

    assert result.status == "ok"
    assert clicked["prefer_confirm"] is True
    assert "unknown_retry_override" not in ctx.handler_state


def test_modal_handler_exam_retry_without_previous_snapshot_defaults_to_retry(monkeypatch):
    clicked = {}
    strategy_called = {"value": False}

    def _click_modal_action_with_retry(app, modal, *, prefer_confirm, **kwargs):
        clicked["prefer_confirm"] = prefer_confirm
        return True

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.ui.click_modal_action_with_retry",
        _click_modal_action_with_retry,
    )

    modal = SimpleNamespace(
        modal_title="再挑戦確認",
        modal_body_text="再挑戦が可能ですが、本当に終了しますか？ あと1回",
        confirm_button=SimpleNamespace(name="produce_end"),
        cancel_button=SimpleNamespace(name="retry"),
    )
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        game_utils=SimpleNamespace(try_get_modal=lambda no_body=False: modal),
    )
    ctx = ProduceContext(difficulty="master")

    def _strategy(_app, _ctx, _candidates, decision_state=None):
        strategy_called["value"] = True
        return 1

    ctx.modal_strategy = _strategy

    result = ModalHandler().handle(
        app,
        ctx,
        phase="modal",
        position="exam_retry_confirm_modal",
    )

    assert result.status == "ok"
    assert clicked["prefer_confirm"] is False
    assert strategy_called["value"] is False
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "exam_retry_confirm_modal",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }
