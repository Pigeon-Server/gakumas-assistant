from __future__ import annotations

from types import SimpleNamespace

from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay.consult import (
    ConsultActionCandidate,
    decide_consult_action,
    execute_consult_step,
)


class _DeviceStub:
    def __init__(self):
        self.clicked = []

    def click_element(self, element):
        self.clicked.append(element)


def _candidate(index: int, kind: str, title: str):
    return ConsultActionCandidate(
        index=index,
        kind=kind,
        title=title,
        box=SimpleNamespace(cx=100 + index, cy=200 + index),
        action_id=f"{kind}:{index}",
        db_id=f"db:{index}",
    )


def test_consult_remove_follows_same_select_then_confirm_skeleton(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["consult_pending_mode"] = "remove"
    app = SimpleNamespace(device=_DeviceStub())

    candidates = [
        _candidate(0, "remove_target", "卡片A"),
        _candidate(1, "confirm_remove", "削除"),
    ]

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.gameplay.consult.detect_consult_actions",
        lambda app, ctx, position: candidates,
    )

    target = execute_consult_step(app, ctx, position="consult_enhancement_preview")
    assert target is not None
    assert target.kind == "remove_target"
    assert ctx.handler_state["consult_pending_mode"] == "remove"
    assert ctx.handler_state["consult_last_subaction"] == "select_remove_target"
    assert ctx.operation_history[-1].action == "consult_select_remove_target"

    # 有 pending target 时，第二步应当优先确认删除，而不是重新选卡
    target_index = decide_consult_action(app, ctx, candidates, position="consult_enhancement_ready")
    assert target_index == 1

    target = execute_consult_step(app, ctx, position="consult_enhancement_ready")
    assert target is not None
    assert target.kind == "confirm_remove"
    assert ctx.operation_history[-1].action == "consult_confirm_remove"


def test_consult_first_decision_will_not_allow_immediate_exit():
    ctx = ProduceContext()
    ctx.consult_strategy = lambda *args, **kwargs: 2
    app = SimpleNamespace()

    candidates = [
        _candidate(0, "exchange", "P50"),
        _candidate(1, "enhance", "強化"),
        _candidate(2, "exit", "終了"),
    ]

    target_index = decide_consult_action(app, ctx, candidates, position="consult_exchange")

    assert target_index == 1


def test_consult_will_retry_same_exchange_when_page_state_is_unchanged():
    ctx = ProduceContext()
    ctx.consult_strategy = lambda *args, **kwargs: 1
    ctx.consult_remaining_p_points = 151
    ctx.handler_state["consult_last_subaction"] = "exchange"
    ctx.handler_state["consult_waiting_exchange_result"] = True
    ctx.handler_state["consult_last_exchange_action_id"] = "consult_exchange_drink:pdrink_00-1-004"
    ctx.handler_state["consult_last_exchange_db_id"] = "pdrink_00-1-004"
    ctx.handler_state["consult_last_exchange_p_points"] = 151
    ctx.handler_state["consult_last_exchange_signature"] = [
        "consult_exchange_drink:pdrink_00-1-004",
    ]
    app = SimpleNamespace()

    candidates = [
        ConsultActionCandidate(
            index=0,
            kind="exchange",
            title="烏龍茶®50",
            box=SimpleNamespace(cx=100, cy=200),
            action_id="consult_exchange_drink:pdrink_00-1-004",
            db_id="pdrink_00-1-004",
        ),
        _candidate(1, "exit", "終了"),
    ]

    target_index = decide_consult_action(app, ctx, candidates, position="consult_exchange")

    assert target_index == 0
    assert ctx.handler_state["consult_exchange_retry_count"] == 1
