from types import SimpleNamespace

import cv2
import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCR_Result, OCR_ResultList
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import p_drink as p_drink_module


class _ResultsStub:
    def __init__(self, mapping):
        self._mapping = dict(mapping or {})

    def filter_by_label(self, label):
        return list(self._mapping.get(label, []))

    def exists_label(self, label):
        return bool(self._mapping.get(label, []))


def _drink_box(x1: int, y1: int, x2: int, y2: int):
    return SimpleNamespace(
        x=x1,
        y=y1,
        w=x2,
        h=y2,
        cx=int((x1 + x2) / 2),
        cy=int((y1 + y2) / 2),
    )


def test_collect_p_drink_limit_action_candidates_includes_skip_and_discard_slots(monkeypatch):
    monkeypatch.setattr(
        p_drink_module,
        "_ocr_limit_controls",
        lambda _frame: [
            {"text": "受け取らない", "x": 420, "y": 1720, "w": 220, "h": 58},
            {"text": "受け取る", "x": 445, "y": 1850, "w": 180, "h": 68},
        ],
    )
    monkeypatch.setattr(
        p_drink_module,
        "_detect_bottom_inventory_drinks",
        lambda _app: [
            SimpleNamespace(cx=130, cy=2230),
            SimpleNamespace(cx=270, cy=2230),
            SimpleNamespace(cx=410, cy=2230),
        ],
    )

    candidates = p_drink_module.collect_p_drink_limit_action_candidates(
        SimpleNamespace(latest_frame=object())
    )

    assert [candidate.action_id for candidate in candidates] == [
        "p_drink_limit_skip_new",
        "p_drink_limit_discard_slot_1",
        "p_drink_limit_discard_slot_2",
        "p_drink_limit_discard_slot_3",
    ]


def test_execute_p_drink_step_limit_page_defaults_to_skip_new(monkeypatch):
    clicks: list[tuple[int, int, str]] = []

    app = SimpleNamespace(
        device=SimpleNamespace(
            click=lambda x, y, el_label="": clicks.append((int(x), int(y), str(el_label))),
            click_element=lambda _element: None,
        ),
        latest_frame=object(),
    )
    ctx = ProduceContext()

    monkeypatch.setattr(p_drink_module, "_is_p_drink_limit_page", lambda _app: True)
    monkeypatch.setattr(
        p_drink_module,
        "collect_p_drink_limit_action_candidates",
        lambda _app, _ctx=None: [
            p_drink_module.PDrinkLimitActionCandidate(
                index=0,
                title="不领取新饮料",
                kind="skip_new_drink",
                action_id="p_drink_limit_skip_new",
                metadata={
                    "checkbox_x": 430,
                    "checkbox_y": 1748,
                    "button_x": 540,
                    "button_y": 1888,
                },
            )
        ],
    )
    monkeypatch.setattr(p_drink_module.time, "sleep", lambda _seconds: None)

    result = p_drink_module.execute_p_drink_step(app, ctx, position="p_drink_selected")

    assert result is not None
    assert result.status == "selected"
    assert clicks == [
        (430, 1748, "p_drink_limit_checkbox"),
        (540, 1888, "p_drink_limit_confirm"),
    ]


def test_decide_p_drink_limit_action_replaces_worst_old_drink_when_new_one_is_stronger(monkeypatch):
    ctx = ProduceContext()
    ctx.hud_stamina = 8
    ctx.hud_max_stamina = 35
    ctx.parameter_state["remaining_turns"] = 2
    ctx.handler_state["pending_new_p_drink"] = {
        "display_name": "集中ブレンド",
        "description": "集中+4 パラメータ上昇量増加",
        "effect_types": ["ProduceExamEffectType_Review", "ProduceExamEffectType_ParameterBuff"],
        "rarity": "SR",
    }
    candidates = [
        p_drink_module.PDrinkLimitActionCandidate(
            index=0,
            title="放弃新饮料「集中ブレンド」",
            kind="skip_new_drink",
            action_id="p_drink_limit_skip_new",
            metadata={"candidate_type": "p_drink_limit"},
        ),
        p_drink_module.PDrinkLimitActionCandidate(
            index=1,
            title="丢弃「ホットコーヒー」并保留新饮料「集中ブレンド」",
            kind="discard_existing_drink",
            action_id="p_drink_limit_discard_slot_1",
            db_id="drink_hot_coffee",
            metadata={
                "display_name": "ホットコーヒー",
                "description": "元気+2",
                "effect_types": ["ProduceExamEffectType_Block"],
            },
        ),
        p_drink_module.PDrinkLimitActionCandidate(
            index=2,
            title="丢弃「スコアドリンク」并保留新饮料「集中ブレンド」",
            kind="discard_existing_drink",
            action_id="p_drink_limit_discard_slot_2",
            db_id="drink_score",
            metadata={
                "display_name": "スコアドリンク",
                "description": "好調+2 スコア上昇",
                "effect_types": ["ProduceExamEffectType_Score"],
            },
        ),
    ]

    monkeypatch.setattr(
        p_drink_module,
        "build_decision_state",
        lambda *_args, **_kwargs: {
            "candidates": [
                {"index": candidate.index, "label": candidate.title, "kind": candidate.kind, "metadata": dict(candidate.metadata)}
                for candidate in candidates
            ],
            "llm_actions": [
                {"index": candidate.index, "label": candidate.title, "kind": candidate.kind}
                for candidate in candidates
            ],
            "stage_context": {},
            "llm_snapshot": {"stage_context": {}},
        },
    )
    monkeypatch.setattr(p_drink_module, "invoke_decision_strategy", lambda *args, **kwargs: None)

    target = p_drink_module.decide_p_drink_limit_action(
        SimpleNamespace(),
        ctx,
        candidates,
    )

    assert target.action_id == "p_drink_limit_discard_slot_1"


def test_execute_p_drink_step_idle_uses_limit_decision_when_disable_and_checkbox_visible(monkeypatch):
    called = {"handled": False}

    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            p_drink_module.ProducerLabels.DISABLE_BUTTON: [SimpleNamespace()],
            p_drink_module.BaseUILabels.CHECKBOX: [SimpleNamespace()],
        }),
    )
    ctx = ProduceContext()

    monkeypatch.setattr(p_drink_module, "_is_p_drink_limit_page", lambda _app: False)

    def _fake_handle(_app, _ctx):
        called["handled"] = True
        return p_drink_module.PDrinkStepResult(status="selected")

    monkeypatch.setattr(p_drink_module, "_handle_p_drink_limit_page", _fake_handle)

    result = p_drink_module.execute_p_drink_step(app, ctx, position="p_drink_idle")

    assert result is not None
    assert result.status == "selected"
    assert called["handled"] is True


def test_click_receive_button_prefers_large_center_button_over_lower_round_buttons(monkeypatch):
    clicked = []

    class _ButtonResultsStub:
        def __init__(self, mapping):
            self._mapping = dict(mapping or {})

        def filter_by_label(self, label):
            return list(self._mapping.get(label, []))

    receive_button = SimpleNamespace(
        x=350, y=1960, w=730, h=2110, cx=540, cy=2035,
        frame=np.ones((150, 380, 3), dtype=np.uint8),
    )
    lower_round_button = SimpleNamespace(
        x=780, y=2240, w=960, h=2420, cx=870, cy=2330,
        frame=np.ones((180, 180, 3), dtype=np.uint8),
    )
    lower_round_button_2 = SimpleNamespace(
        x=120, y=2230, w=300, h=2410, cx=210, cy=2320,
        frame=np.ones((180, 180, 3), dtype=np.uint8),
    )
    app = SimpleNamespace(
        latest_frame=np.ones((2400, 1080, 3), dtype=np.uint8),
        latest_results=_ButtonResultsStub({
            p_drink_module.BaseUILabels.BUTTON: [
                receive_button,
                lower_round_button,
                lower_round_button_2,
            ],
        }),
        device=SimpleNamespace(click_element=lambda element: clicked.append(element)),
    )

    monkeypatch.setattr(
        p_drink_module,
        "ocr_text",
        lambda frame: "受け取る" if frame is receive_button.frame else "",
    )

    assert p_drink_module._click_receive_button(app) is True
    assert clicked == [receive_button]


def test_click_receive_button_falls_back_to_bottom_ocr_when_yolo_button_missing(monkeypatch):
    clicks: list[tuple[int, int, str]] = []

    frame = np.ones((2400, 1080, 3), dtype=np.uint8)
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({}),
        device=SimpleNamespace(click=lambda x, y, el_label="": clicks.append((int(x), int(y), str(el_label)))),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )

    monkeypatch.setattr(
        p_drink_module,
        "_ocr",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(120, 90, 260, 64, "受け取る", 0.99),
            ])
        ),
    )

    assert p_drink_module._click_receive_button(app) is True
    assert clicks == [(120 + int(1080 * 0.18) + 130, 90 + int(2400 * 0.72) + 32, "p_drink_receive_ocr")]


def test_ocr_p_drink_header_prefers_white_panel_title(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (70, 640), (1010, 2040), (246, 246, 246), thickness=-1)
    drinks = [
        _drink_box(150, 1540, 300, 1750),
        _drink_box(460, 1540, 610, 1750),
        _drink_box(770, 1540, 920, 1750),
    ]
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({BaseUILabels.P_DRINK: drinks}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        p_drink_module,
        "_ocr",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(96, 34, 320, 46, "ジンジャーエール", 0.99),
                OCR_Result(118, 138, 220, 40, "→ 元気+2", 0.99),
            ])
        ),
    )

    name, effect = p_drink_module._ocr_p_drink_header(frame, app=app, drink_boxes=drinks)

    assert name == "ジンジャーエール"
    assert effect == "→ 元気+2"


def test_ocr_p_drink_header_fallback_filters_effect_line(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 700), (1020, 2100), (245, 245, 245), thickness=-1)
    drinks = [
        _drink_box(180, 1520, 340, 1740),
        _drink_box(740, 1520, 900, 1740),
    ]
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({BaseUILabels.P_DRINK: drinks}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        p_drink_module,
        "_ocr",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(120, 24, 200, 40, "→ 元気+2", 0.99),
                OCR_Result(110, 88, 280, 40, "スターラテ", 0.99),
            ])
        ),
    )

    name, effect = p_drink_module._ocr_p_drink_header(frame, app=app, drink_boxes=drinks)

    assert name == "スターラテ"
    assert effect == ""
