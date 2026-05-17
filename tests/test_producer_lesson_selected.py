from types import SimpleNamespace

import numpy as np

from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import exam as exam_module
from src.core.tasks.producer_challenge.gameplay import lesson as lesson_module
from src.core.tasks.producer_challenge.gameplay.llm.prompt_renderer import render


class _DeviceStub:
    def __init__(self):
        self.element_clicks = []
        self.point_clicks = []

    def click_element(self, element):
        self.element_clicks.append(getattr(element, "name", ""))

    def click(self, x, y, el_label=""):
        self.point_clicks.append((x, y, el_label))


class _ResultsStub:
    def __init__(self, label_boxes=None):
        self.label_boxes = label_boxes or {}
        self.frame = None

    def exists_label(self, label):
        return bool(self.label_boxes.get(label, []))

    def filter_by_label(self, label):
        return list(self.label_boxes.get(label, []))


class _ResultsSequenceApp:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self._index = 0

    @property
    def latest_results(self):
        if not self._sequence:
            return _ResultsStub()
        idx = min(self._index, len(self._sequence) - 1)
        self._index += 1
        return self._sequence[idx]


def _candidate(index: int, name: str):
    box = SimpleNamespace(
        name=name,
        cx=100 + index * 100,
        cy=200,
        label=name,
        get_COL=lambda: (100 + index * 100, 200),
    )
    return lesson_module.LessonCardCandidate(
        index=index,
        label="Skill Card: Active",
        title=name,
        selected=index == 0,
        box=box,
        action_id=f"produce_card:{name}",
        db_id=f"card_{index}",
        source="ocr",
        confidence=1.0,
        metadata={},
    )


def _end_turn_candidate(index: int, title: str = "SKIP"):
    box = SimpleNamespace(
        name=title,
        x=840,
        y=1540,
        w=960,
        h=1660,
        cx=900,
        cy=1600,
        label=title,
        get_COL=lambda: (900, 1600),
    )
    return lesson_module.LessonCardCandidate(
        index=index,
        label=title,
        title=title,
        selected=False,
        box=box,
        action_id="end_turn",
        db_id="",
        source="yolo",
        confidence=1.0,
        metadata={"candidate_type": "end_turn"},
    )


def _blank_slot_box(index: int):
    x = 67 + index * 145
    y = 2160
    width = 121
    height = 121
    return SimpleNamespace(
        x=x,
        y=y,
        w=x + width,
        h=y + height,
        cx=x + width // 2,
        cy=y + height // 2,
        label=BaseUILabels.BLANK_SLOT,
    )


def test_verify_card_played_requires_stable_clear_polls(monkeypatch):
    # 面板短暂漏检（1帧）后又出现时，不应判定为打出成功。
    app = _ResultsSequenceApp([
        _ResultsStub({ProducerLabels.SKILL_CARD_INFO: [object()]}),
        _ResultsStub({}),
        _ResultsStub({ProducerLabels.SKILL_CARD_INFO: [object()]}),
        _ResultsStub({}),
        _ResultsStub({ProducerLabels.SKILL_CARD_INFO: [object()]}),
    ])
    clock = {"t": 0.0}
    monkeypatch.setattr(lesson_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        lesson_module.time,
        "monotonic",
        lambda: clock.__setitem__("t", clock["t"] + 0.05) or clock["t"],
    )

    assert lesson_module._verify_card_played(app, timeout=0.5) is False


def test_verify_card_played_accepts_three_consecutive_clear_polls(monkeypatch):
    app = _ResultsSequenceApp([
        _ResultsStub({ProducerLabels.SKILL_CARD_INFO: [object()]}),
        _ResultsStub({}),
        _ResultsStub({}),
        _ResultsStub({}),
    ])
    clock = {"t": 0.0}
    monkeypatch.setattr(lesson_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        lesson_module.time,
        "monotonic",
        lambda: clock.__setitem__("t", clock["t"] + 0.05) or clock["t"],
    )

    assert lesson_module._verify_card_played(app, timeout=0.6) is True


def test_wait_info_panel_visibility_accepts_single_open_hit(monkeypatch):
    app = _ResultsSequenceApp([
        _ResultsStub({}),
        _ResultsStub({ProducerLabels.SKILL_CARD_INFO: [object()]}),
        _ResultsStub({}),
    ])
    monkeypatch.setattr(lesson_module.time, "sleep", lambda _s: None)

    ok = lesson_module._wait_info_panel_visibility(
        app,
        expected_visible=True,
        reason="test_single_open_hit",
    )

    assert ok is True


def test_is_info_panel_visible_accepts_white_panel_fallback_without_panel_label(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    results = _ResultsStub(
        {
            ProducerLabels.PC_PROGRESS: [object()],
            ProducerLabels.PC_STAMINA: [object()],
            ProducerLabels.SKILL_CARD_ACTIVE: [
                SimpleNamespace(x=250, y=1800, w=470, h=2100, cx=360, cy=1950, label=ProducerLabels.SKILL_CARD_ACTIVE),
                SimpleNamespace(x=500, y=1800, w=720, h=2100, cx=610, cy=1950, label=ProducerLabels.SKILL_CARD_ACTIVE),
                SimpleNamespace(x=750, y=1800, w=970, h=2100, cx=860, cy=1950, label=ProducerLabels.SKILL_CARD_ACTIVE),
            ],
        }
    )

    monkeypatch.setattr(
        lesson_module,
        "detect_bottom_white_modal_region",
        lambda *_args, **_kwargs: (80, 980, 1000, 2020),
    )

    assert lesson_module._is_info_panel_visible(
        results,
        frame=frame,
        debugger=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    ) is True


def test_wait_info_panel_visibility_pre_close_does_not_use_white_panel_fallback(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)

    class _StaticApp:
        latest_results = _ResultsStub(
            {
                ProducerLabels.PC_PROGRESS: [object()],
                ProducerLabels.PC_STAMINA: [object()],
                ProducerLabels.SKILL_CARD_ACTIVE: [
                    SimpleNamespace(x=250, y=1800, w=470, h=2100, cx=360, cy=1950, label=ProducerLabels.SKILL_CARD_ACTIVE),
                ],
            }
        )
        latest_frame = frame
        debug_tools = SimpleNamespace(add_box=lambda *args, **kwargs: None)

    monkeypatch.setattr(lesson_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        lesson_module,
        "detect_bottom_white_modal_region",
        lambda *_args, **_kwargs: (80, 980, 1000, 2020),
    )

    ok = lesson_module._wait_info_panel_visibility(
        _StaticApp(),
        expected_visible=False,
        reason="test_pre_close_without_fallback",
    )

    assert ok is True


def test_extract_battle_info_panel_name_uses_white_panel_fallback_without_panel_label(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    results = _ResultsStub(
        {
            ProducerLabels.PC_PROGRESS: [object()],
            ProducerLabels.PC_STAMINA: [object()],
            ProducerLabels.SKILL_CARD_ACTIVE: [
                SimpleNamespace(x=250, y=1800, w=470, h=2100, cx=360, cy=1950, label=ProducerLabels.SKILL_CARD_ACTIVE),
                SimpleNamespace(x=500, y=1800, w=720, h=2100, cx=610, cy=1950, label=ProducerLabels.SKILL_CARD_ACTIVE),
                SimpleNamespace(x=750, y=1800, w=970, h=2100, cx=860, cy=1950, label=ProducerLabels.SKILL_CARD_ACTIVE),
            ],
        }
    )

    monkeypatch.setattr(
        lesson_module,
        "detect_bottom_white_modal_region",
        lambda *_args, **_kwargs: (80, 980, 1000, 2020),
    )

    class _OCRResultList(list):
        pass

    class _OCRServiceStub:
        def ocr(self, _image):
            return _OCRResultList(
                [
                    SimpleNamespace(text="エキサイト", x=210, y=70, w=260, h=48),
                    SimpleNamespace(text="-4", x=842, y=68, w=48, h=42),
                    SimpleNamespace(text="パラメータ+6", x=180, y=170, w=320, h=44),
                ]
            )

    monkeypatch.setattr("src.core.inference.ocr_engine.OCRService", _OCRServiceStub)

    name = lesson_module._extract_battle_info_panel_name(
        results,
        frame,
        debugger=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )

    assert name == "エキサイト"


def test_extract_battle_info_panel_name_filters_guide_noise(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    results = _ResultsStub(
        {
            ProducerLabels.PC_PROGRESS: [object()],
            ProducerLabels.PC_STAMINA: [object()],
            ProducerLabels.SKILL_CARD_ACTIVE: [
                SimpleNamespace(x=250, y=1800, w=470, h=2100, cx=360, cy=1950, label=ProducerLabels.SKILL_CARD_ACTIVE),
            ],
        }
    )

    monkeypatch.setattr(
        lesson_module,
        "detect_bottom_white_modal_region",
        lambda *_args, **_kwargs: (80, 980, 1000, 2020),
    )

    class _OCRResultList(list):
        pass

    class _OCRServiceStub:
        def __init__(self):
            self.calls = 0

        def ocr(self, _image):
            self.calls += 1
            if self.calls == 1:
                return _OCRResultList(
                    [
                        SimpleNamespace(text="獲得ガイド", x=720, y=52, w=150, h=40),
                        SimpleNamespace(text="-4", x=842, y=68, w=48, h=42),
                    ]
                )
            return _OCRResultList(
                [
                    SimpleNamespace(text="エキサイト", x=80, y=20, w=260, h=48),
                ]
            )

    monkeypatch.setattr("src.core.inference.ocr_engine.OCRService", _OCRServiceStub)

    name = lesson_module._extract_battle_info_panel_name(
        results,
        frame,
        debugger=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )

    assert name == "エキサイト"


def test_execute_lesson_step_selected_state_reuses_pending_click_point(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    ctx.pending_lesson_card_index = 1
    ctx.pending_lesson_card_label = "第二张牌"
    ctx.handler_state["pending_lesson_click_point"] = (320, 640)
    ctx.handler_state["pending_lesson_action_id"] = "produce_card:第二张牌"
    ctx.handler_state["pending_lesson_db_id"] = "card_1"
    ctx.record_operation = lambda *args, **kwargs: None

    candidates = [_candidate(0, "第一张牌")]
    monkeypatch.setattr(
        lesson_module,
        "collect_lesson_card_candidates",
        lambda _app, _ctx, phase="lesson", position="lesson_selected": candidates,
    )
    monkeypatch.setattr(
        lesson_module,
        "decide_lesson_card",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selected 态首击不应重新走决策")),
    )
    monkeypatch.setattr(lesson_module, "_verify_card_played", lambda _app: True)

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position="lesson_selected",
        phase="lesson",
    )

    assert result is not None
    assert result.status == "used"
    assert app.device.element_clicks == []
    assert app.device.point_clicks == [(320, 640, "confirm_lesson_card")]
    assert ctx.pending_lesson_card_index is None
    assert ctx.pending_lesson_card_label == ""


def test_execute_lesson_step_skips_blocked_pending_card(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    ctx.current_week = 2
    ctx.parameter_state["remaining_turns"] = 3
    ctx.pending_lesson_card_index = 0
    ctx.pending_lesson_card_label = "第一张牌"
    ctx.handler_state["battle_blocked_cards"] = {
        "turn_marker": ("lesson", 2, 3),
        "keys": ["card_0"],
    }
    ctx.record_operation = lambda *args, **kwargs: None

    candidates = [_candidate(0, "第一张牌"), _candidate(1, "第二张牌")]
    monkeypatch.setattr(
        lesson_module,
        "collect_lesson_card_candidates",
        lambda _app, _ctx, phase="lesson", position="lesson_selected": candidates,
    )
    monkeypatch.setattr(
        lesson_module,
        "decide_lesson_card",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(lesson_module, "_verify_card_played", lambda _app: True)

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position="lesson_selected",
        phase="lesson",
    )

    assert result is not None
    assert result.status == "used"
    assert app.device.element_clicks == ["第二张牌"]
    assert app.device.point_clicks == [(200, 200, "confirm_lesson_card")]


def test_execute_lesson_step_idle_double_taps_same_card(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    ctx.record_operation = lambda *args, **kwargs: None

    candidates = [_candidate(0, "第一张牌"), _candidate(1, "第二张牌")]
    monkeypatch.setattr(
        lesson_module,
        "collect_lesson_card_candidates",
        lambda _app, _ctx, phase="lesson", position="lesson_idle": candidates,
    )
    monkeypatch.setattr(lesson_module, "decide_lesson_card", lambda *args, **kwargs: 0)
    monkeypatch.setattr(lesson_module, "_verify_card_played", lambda _app: True)

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position="lesson_idle",
        phase="lesson",
    )

    assert result is not None
    assert result.status == "used"
    assert app.device.element_clicks == ["第一张牌"]
    assert app.device.point_clicks == [(100, 200, "confirm_lesson_card")]


def test_execute_lesson_step_resolves_candidate_by_action_index_instead_of_list_position(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    ctx.record_operation = lambda *args, **kwargs: None

    first = _candidate(0, "第一张牌")
    second = _candidate(1, "第二张牌")
    candidates = [second, first]
    monkeypatch.setattr(
        lesson_module,
        "collect_lesson_card_candidates",
        lambda _app, _ctx, phase="lesson", position="lesson_idle": candidates,
    )
    monkeypatch.setattr(lesson_module, "decide_lesson_card", lambda *args, **kwargs: 1)
    monkeypatch.setattr(lesson_module, "_verify_card_played", lambda _app: True)

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position="lesson_idle",
        phase="lesson",
    )

    assert result is not None
    assert result.status == "used"
    assert result.candidate.title == "第二张牌"
    assert app.device.element_clicks == ["第二张牌"]
    assert app.device.point_clicks == [(200, 200, "confirm_lesson_card")]


def test_execute_lesson_step_refreshes_candidates_when_no_action_is_currently_playable(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    ctx.record_operation = lambda *args, **kwargs: None

    candidates = [_candidate(0, "第一张牌"), _candidate(1, "第二张牌")]
    monkeypatch.setattr(
        lesson_module,
        "collect_lesson_card_candidates",
        lambda _app, _ctx, phase="lesson", position="lesson_idle": candidates,
    )
    decisions = iter([-1, 1])
    monkeypatch.setattr(lesson_module, "decide_lesson_card", lambda *args, **kwargs: next(decisions))
    monkeypatch.setattr(lesson_module, "_verify_card_played", lambda _app: True)

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position="lesson_idle",
        phase="lesson",
    )

    assert result is not None
    assert result.status == "used"
    assert result.candidate.title == "第二张牌"
    assert app.device.element_clicks == ["第二张牌"]
    assert app.device.point_clicks == [
        (540, int(2340 * lesson_module._DESELECT_TAP_Y_RATIO), "deselect_card"),
        (200, 200, "confirm_lesson_card"),
    ]


def test_execute_lesson_step_can_select_bottom_bar_drink(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    operations = []
    ctx.record_operation = lambda *args, **kwargs: operations.append((args, kwargs))

    drink_box = SimpleNamespace(name="元気茶", cx=300, cy=2200, label="P Drink")
    candidates = [
        lesson_module.LessonCardCandidate(
            index=0,
            label="P Drink",
            title="元気茶",
            selected=False,
            box=drink_box,
            action_id="produce_drink_unknown:idx_0",
            db_id="drink_guard",
            source="clip",
            confidence=0.98,
            metadata={"drink_score": 32.0, "description": "元気+10"},
        )
    ]
    monkeypatch.setattr(
        lesson_module,
        "collect_lesson_card_candidates",
        lambda _app, _ctx, phase="lesson", position="lesson_idle": candidates,
    )
    monkeypatch.setattr(lesson_module, "decide_lesson_card", lambda *args, **kwargs: 0)

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position="lesson_idle",
        phase="lesson",
    )

    assert result is not None
    assert result.status == "selected"
    assert app.device.element_clicks == ["元気茶"]
    assert ctx.pending_p_drink_index == 0
    assert ctx.pending_p_drink_label == "元気茶"
    assert operations[0][0][0] == "select_battle_p_drink"


def test_is_battle_empty_hand_observed_via_blank_slots_and_notice(monkeypatch):
    app = SimpleNamespace(
        latest_results=_ResultsStub(
            {
                BaseUILabels.BLANK_SLOT: [_blank_slot_box(0), _blank_slot_box(1), _blank_slot_box(2)],
            }
        ),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )

    monkeypatch.setattr(lesson_module, "ocr_text", lambda _frame: ProduceText.EMPTY_HAND_MESSAGE)

    assert lesson_module._is_battle_empty_hand_observed(app) is True


def test_execute_lesson_step_returns_none_when_empty_hand_is_explicitly_observed(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = ProduceContext()

    monkeypatch.setattr(
        lesson_module,
        "collect_lesson_card_candidates",
        lambda *_args, **_kwargs: [_end_turn_candidate(0)],
    )
    monkeypatch.setattr(lesson_module, "_is_battle_empty_hand_observed", lambda _app: True)
    monkeypatch.setattr(
        lesson_module,
        "decide_lesson_card",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("无手牌时不应直接走正常出牌决策")),
    )

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position="exam_idle",
        phase="exam",
    )

    assert result is None
    assert ctx.observability_state["empty_hand_observed"] is True


def test_is_battle_card_baseline_settled_allows_single_floating_selected_card():
    settled, baseline = lesson_module._is_battle_card_baseline_settled(
        [1700, 1760, 1765],
        tolerance=32,
    )
    assert settled is True
    assert baseline == 1760


def test_wait_battle_card_deal_settle_waits_until_card_baseline_stable(monkeypatch):
    app = SimpleNamespace(
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    sleep_calls = []
    center_sequences = [
        [1700, 1820],
        [1740, 1830],
        [1800, 1810, 1820],
        [1802, 1812, 1822],
        [1801, 1811, 1821],
    ]
    state = {"idx": 0}

    def _next_centers(_results):
        idx = min(state["idx"], len(center_sequences) - 1)
        state["idx"] += 1
        return center_sequences[idx]

    monkeypatch.setattr(lesson_module, "_collect_visible_battle_card_center_ys", _next_centers)
    monkeypatch.setattr(lesson_module, "_resolve_battle_card_baseline_tolerance", lambda _app: (30, 8))
    monkeypatch.setattr(
        lesson_module.time,
        "sleep",
        lambda sec: sleep_calls.append(round(float(sec), 2)),
    )

    lesson_module._wait_battle_card_deal_settle(
        app,
        phase="exam",
        position="exam_idle",
        pending_index=None,
    )

    assert sleep_calls == [round(lesson_module._BATTLE_DEAL_SETTLE_SAMPLE_SLEEP, 2)] * 3


def test_lesson_handler_empty_hand_reuses_decision_engine_for_drink_or_skip(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    operations = []
    ctx.record_operation = lambda *args, **kwargs: operations.append((args, kwargs))

    drink_box = SimpleNamespace(name="元気茶", cx=300, cy=2200, label="P Drink")
    end_turn = _end_turn_candidate(1)
    drink = lesson_module.LessonCardCandidate(
        index=0,
        label="P Drink",
        title="元気茶",
        selected=False,
        box=drink_box,
        action_id="produce_drink:drink_guard",
        db_id="drink_guard",
        source="clip",
        confidence=0.98,
        metadata={"drink_score": 36.0, "description": "使用后补充手牌"},
    )

    monkeypatch.setattr(lesson_module, "execute_lesson_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lesson_module, "build_decision_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        lesson_module,
        "_collect_battle_drink_candidates",
        lambda *_args, **_kwargs: [drink],
    )
    monkeypatch.setattr(
        lesson_module,
        "_collect_battle_end_turn_candidates",
        lambda *_args, **_kwargs: [end_turn],
    )
    monkeypatch.setattr(lesson_module, "decide_lesson_card", lambda *_args, **_kwargs: 0)

    result = lesson_module.LessonHandler().handle(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
    )

    assert result.status == "ok"
    assert "空手牌 fallback" in result.detail
    assert "元気茶" in result.detail
    assert app.device.element_clicks == ["元気茶"]
    assert ctx.pending_p_drink_index == 0
    assert ctx.pending_p_drink_label == "元気茶"
    assert operations == []


def test_exam_handler_empty_hand_reuses_decision_engine_before_local_skip(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    sentinel = exam_module.HandlerResult.ok("exam: empty_hand_fallback")

    monkeypatch.setattr(exam_module, "execute_lesson_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exam_module, "_try_resolve_empty_hand_action", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(
        exam_module,
        "_try_click_skip",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应绕过 LLM/RL fallback 直接 skip")),
    )

    result = exam_module.ExamHandler().handle(
        app,
        ctx,
        phase="exam",
        position="exam_idle",
    )

    assert result is sentinel


def test_decide_lesson_card_uses_drink_when_all_cards_are_unavailable(monkeypatch):
    ctx = ProduceContext()
    card = _candidate(0, "高费卡")
    drink = lesson_module.LessonCardCandidate(
        index=1,
        label="P Drink",
        title="元気茶",
        selected=False,
        box=SimpleNamespace(name="元気茶"),
        action_id="produce_drink:drink_guard",
        db_id="drink_guard",
        source="clip",
        confidence=0.98,
        metadata={"drink_score": 30.0},
    )

    monkeypatch.setattr(
        lesson_module,
        "build_decision_state",
        lambda *_args, **_kwargs: {
            "candidates": [
                {
                    "index": 0,
                    "id": "produce_card:高费卡",
                    "available": False,
                    "metadata": {"unavailable_reason": "体力不足"},
                },
                {
                    "index": 1,
                    "id": "produce_drink:drink_guard",
                    "available": True,
                    "label": "元気茶",
                    "description": "元気+10",
                    "metadata": {"drink_score": 30.0},
                },
            ],
            "legal_actions": [1],
            "llm_snapshot": {"stamina": 2, "max_stamina": 35, "play_limit_remaining": 1},
        },
    )
    monkeypatch.setattr(lesson_module, "invoke_decision_strategy", lambda *args, **kwargs: None)

    chosen = lesson_module.decide_lesson_card(
        SimpleNamespace(),
        ctx,
        [card, drink],
        phase="lesson",
        position="lesson_idle",
    )

    assert chosen == 1


def test_select_forced_battle_drink_index_respects_zero_play_limit_string():
    decision_state = {
        "legal_actions": [0, 1],
        "candidates": [
            {
                "index": 0,
                "id": "produce_card:heavy",
                "available": True,
                "metadata": {"description": "打分+9"},
            },
            {
                "index": 1,
                "id": "produce_drink:drink_guard",
                "available": True,
                "label": "元気茶",
                "description": "元気+10",
                "metadata": {"drink_score": 30.0},
            },
        ],
        "llm_snapshot": {
            "stamina": 30,
            "max_stamina": 30,
            "play_limit_remaining": "0/1",
        },
    }

    chosen = lesson_module._select_forced_battle_drink_index(
        decision_state,
        skip_indices=set(),
    )

    assert chosen == 1


def test_decide_lesson_card_returns_minus_one_when_all_candidates_are_skipped(monkeypatch):
    ctx = ProduceContext()
    card = _candidate(0, "灰卡")

    monkeypatch.setattr(
        lesson_module,
        "build_decision_state",
        lambda *_args, **_kwargs: {
            "candidates": [
                {
                    "index": 0,
                    "id": "produce_card:gray",
                    "available": False,
                    "metadata": {"unavailable_reason": "灰色蒙版禁用"},
                },
            ],
            "legal_actions": [],
            "llm_snapshot": {"stamina": 6, "max_stamina": 35, "play_limit_remaining": 1},
        },
    )
    monkeypatch.setattr(lesson_module, "invoke_decision_strategy", lambda *args, **kwargs: 0)

    chosen = lesson_module.decide_lesson_card(
        SimpleNamespace(),
        ctx,
        [card],
        phase="lesson",
        position="lesson_idle",
    )

    assert chosen == -1


def test_decide_lesson_card_uses_end_turn_when_it_is_only_legal_action(monkeypatch):
    ctx = ProduceContext()
    card = _candidate(0, "高费卡")
    end_turn = _end_turn_candidate(1)

    monkeypatch.setattr(
        lesson_module,
        "build_decision_state",
        lambda *_args, **_kwargs: {
            "candidates": [
                {
                    "index": 0,
                    "id": "produce_card:heavy",
                    "available": False,
                    "metadata": {"unavailable_reason": "体力不足"},
                },
                {
                    "index": 1,
                    "id": "end_turn",
                    "available": True,
                    "label": "SKIP",
                    "metadata": {"candidate_type": "end_turn"},
                },
            ],
            "legal_actions": [1],
            "llm_snapshot": {"stamina": 4, "max_stamina": 10, "play_limit_remaining": 1},
        },
    )
    monkeypatch.setattr(lesson_module, "invoke_decision_strategy", lambda *args, **kwargs: None)

    chosen = lesson_module.decide_lesson_card(
        SimpleNamespace(),
        ctx,
        [card, end_turn],
        phase="lesson",
        position="lesson_idle",
    )

    assert chosen == 1


def test_execute_lesson_step_end_turn_deselects_before_clicking_skip(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()
    ctx.record_operation = lambda *args, **kwargs: None

    end_turn = _end_turn_candidate(0)
    monkeypatch.setattr(
        lesson_module,
        "collect_lesson_card_candidates",
        lambda _app, _ctx, phase="lesson", position="lesson_selected": [end_turn],
    )
    monkeypatch.setattr(lesson_module, "decide_lesson_card", lambda *args, **kwargs: 0)

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position="lesson_selected",
        phase="lesson",
    )

    assert result is not None
    assert result.status == "end_turn"
    assert app.device.point_clicks == [
        (540, int(2340 * lesson_module._DESELECT_TAP_Y_RATIO), "deselect_card"),
        (850, 1600, "battle_end_turn"),
    ]
    assert app.device.element_clicks == []


def test_click_battle_end_turn_uses_yolo_box_edges_not_x2_as_width():
    skip_box = SimpleNamespace(
        x=931,
        y=1530,
        w=1053,
        h=1653,
        cx=992,
        cy=1591,
        label=ProducerLabels.PC_SKIP,
    )
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub({ProducerLabels.PC_SKIP: [skip_box]}),
    )

    assert lesson_module._click_battle_end_turn(app) is True
    assert app.device.point_clicks == [(942, 1591, "battle_end_turn")]
    assert app.device.element_clicks == []


def test_collect_lesson_card_candidates_refreshes_ctx_before_collecting_battle_drinks(monkeypatch):
    card_box = SimpleNamespace(
        name="测试卡",
        cx=120,
        cy=1800,
        frame=np.zeros((200, 120, 3), dtype=np.uint8),
    )
    app = SimpleNamespace(
        latest_results=_ResultsStub(
            {lesson_module.ProducerLabels.SKILL_CARD_ACTIVE: [card_box]}
        ),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = ProduceContext()
    ctx.hud_stamina = 1
    ctx.hud_max_stamina = 20
    ctx.parameter_state["remaining_turns"] = 9
    sync_reasons = []
    observed_values = {}

    monkeypatch.setattr(lesson_module, "ocr_text", lambda _frame: "测试卡")
    monkeypatch.setattr(lesson_module, "hydrate_card_candidates", lambda _app, _candidates: None)
    monkeypatch.setattr(lesson_module, "_resolve_unidentified_cards_via_info_panel", lambda *_args, **_kwargs: None)

    def _fake_build_decision_state(_app, _ctx, *, phase, position, candidates, reason):
        sync_reasons.append((phase, position, reason, len(candidates)))
        _ctx.hud_stamina = 4
        _ctx.hud_max_stamina = 32
        _ctx.parameter_state["remaining_turns"] = 2
        _ctx.parameter_state["battle_resources"] = {"block": 7}
        return {}

    monkeypatch.setattr(lesson_module, "build_decision_state", _fake_build_decision_state)
    monkeypatch.setattr(
        lesson_module,
        "_collect_battle_drink_candidates",
        lambda _app, _ctx, **kwargs: observed_values.update({
            "stamina": _ctx.hud_stamina,
            "max_stamina": _ctx.hud_max_stamina,
            "remaining_turns": _ctx.parameter_state["remaining_turns"],
            "battle_resources": dict(_ctx.parameter_state.get("battle_resources", {})),
        }) or [],
    )

    candidates = lesson_module.collect_lesson_card_candidates(
        app,
        ctx,
        phase="lesson",
        position="lesson_idle",
    )

    assert len(candidates) == 1
    assert candidates[0].title == "测试卡"
    assert sync_reasons == [("lesson", "lesson_idle", "lesson_collect_sync", 1)]
    assert observed_values == {
        "stamina": 4,
        "max_stamina": 32,
        "remaining_turns": 2,
        "battle_resources": {"block": 7},
    }


def test_extract_battle_immediate_score_points_handles_compact_description():
    text = "集中消耗3元気+15打分+30（好調效果会2倍生效）"
    assert lesson_module._extract_battle_immediate_score_points(text) == 30


def test_select_battle_preference_prefers_higher_score_card_on_last_turn():
    decision_state = {
        "llm_snapshot": {
            "remaining": 1,
            "stamina": 30,
            "max_stamina": 30,
            "idol_plan_label": ProduceText.PLAN_SENSE,
            "score": 0,
            "target": 0,
        },
        "candidates": [
            {
                "index": 0,
                "id": "produce_card:bless",
                "available": True,
                "description": "体力消耗4打分+26好調1回合训练中仅限1次",
                "metadata": {},
            },
            {
                "index": 1,
                "id": "produce_card:position_check",
                "available": True,
                "description": "集中消耗3元気+15打分+30（好調效果会2倍生效）元気增加无效2回合训练中仅限1次",
                "metadata": {},
            },
        ],
        "legal_actions": [0, 1],
    }

    index, _score, _reason = lesson_module._select_battle_preference(
        decision_state,
        preferred_indices={0, 1},
        retryable_indices={0, 1},
        end_turn_indices=set(),
        phase="lesson",
    )

    assert index == 1


def test_score_battle_payload_prefers_finishing_card_under_clear_pressure():
    finisher_score, finisher_reasons = lesson_module._score_battle_payload(
        {
            "index": 0,
            "id": "produce_card:finisher",
            "available": True,
            "description": "打分+28",
            "metadata": {},
        },
        llm_snapshot={
            "remaining": 1,
            "stamina": 22,
            "max_stamina": 30,
            "clear_achieved": False,
            "remaining_to_clear": 24,
            "remaining_to_perfect": 0,
            "score": 0,
            "target": 0,
            "idol_plan_label": ProduceText.PLAN_SENSE,
        },
        phase="lesson",
    )
    setup_score, setup_reasons = lesson_module._score_battle_payload(
        {
            "index": 1,
            "id": "produce_card:setup",
            "available": True,
            "description": "好調3回合集中+4",
            "metadata": {},
        },
        llm_snapshot={
            "remaining": 1,
            "stamina": 22,
            "max_stamina": 30,
            "clear_achieved": False,
            "remaining_to_clear": 24,
            "remaining_to_perfect": 0,
            "score": 0,
            "target": 0,
            "idol_plan_label": ProduceText.PLAN_SENSE,
        },
        phase="lesson",
    )

    assert finisher_score > setup_score
    assert any("直接过线" in reason or "CLEAR 压力" in reason for reason in finisher_reasons)
    assert any("铺垫" in reason for reason in setup_reasons)



def test_score_battle_payload_prefers_exam_wheel_matching_output_window():
    focused_score, focused_reasons = lesson_module._score_battle_payload(
        {
            "index": 0,
            "id": "produce_card:visual_burst",
            "available": True,
            "description": "ビジュアル打分+20",
            "metadata": {},
        },
        llm_snapshot={
            "remaining": 2,
            "stamina": 24,
            "max_stamina": 30,
            "score": 90,
            "target": 120,
            "idol_plan_label": ProduceText.PLAN_SENSE,
            "exam_ranking": 4,
            "score_bonus_multiplier": "2.0",
            "exam_wheel": {"current_param": "visual", "bonus_pct": 2000, "confidence": "high"},
        },
        phase="exam",
    )
    off_color_score, off_color_reasons = lesson_module._score_battle_payload(
        {
            "index": 1,
            "id": "produce_card:dance_setup",
            "available": True,
            "description": "ダンス好調2回合集中+3",
            "metadata": {},
        },
        llm_snapshot={
            "remaining": 2,
            "stamina": 24,
            "max_stamina": 30,
            "score": 90,
            "target": 120,
            "idol_plan_label": ProduceText.PLAN_SENSE,
            "exam_ranking": 4,
            "score_bonus_multiplier": "2.0",
            "exam_wheel": {"current_param": "visual", "bonus_pct": 2000, "confidence": "high"},
        },
        phase="exam",
    )

    assert focused_score > off_color_score
    assert any("轮盘参数窗口" in reason or "倍率窗口" in reason for reason in focused_reasons)
    assert not any("轮盘参数窗口" in reason or "倍率窗口" in reason for reason in off_color_reasons)
    assert any("考试压力高" in reason for reason in off_color_reasons)


def test_annotate_battle_preference_keeps_local_preference_out_of_llm_payload():
    decision_state = {
        "candidates": [{"index": 1, "name": "ビジュアル打点", "id": "produce_card:visual"}],
        "llm_actions": [{"index": 1, "label": "ビジュアル打点", "id": "produce_card:visual"}],
        "stage_context": {},
        "llm_snapshot": {
            "phase": "exam",
            "position": "exam_idle",
            "scenario": "",
            "difficulty": "",
            "week": 0,
            "remaining_weeks": None,
            "stage_context": {},
            "idol_plan_label": "",
            "idol_plan_focus": "",
            "idol_plan_description": "",
            "produce_goals": {"summary": "", "exam_criteria": [], "training_tasks": [], "recommended_effects": []},
            "exam_criteria": [],
            "training_tasks": [],
            "recommended_effects": [],
            "schedule_history_summary": "",
            "future_schedule": [],
            "turn": None,
            "max_turns": None,
            "remaining": None,
            "battle_kind_label": "考试",
            "clear_achieved": None,
            "target": None,
            "score": 0,
            "ratio": "0%",
            "stamina": 0,
            "max_stamina": 0,
            "genki": 0,
            "play_limit_remaining": None,
            "play_limit_total": None,
            "turn_color_display_label": "",
            "score_bonus_multiplier": "",
            "exam_ranking": None,
            "parameter_stats": {"vocal": "", "vocal_max": "", "dance": "", "dance_max": "", "visual": "", "visual_max": ""},
            "p_items": [],
            "formation_abilities": [],
            "formation_events": [],
            "hand": [],
            "deck_count": 0,
            "deck_summary": "",
            "deck_cards": [],
            "zone_counts": {"deck": 0, "grave": 0, "hold": 0, "lost": 0},
            "offensive_counts": {"hand": 0, "deck": 0, "grave": 0, "hold": 0},
            "reshuffle_hint": "",
            "grave_cards": [],
            "hold_cards": [],
            "lost_cards": [],
            "resources": {
                "parameter_buff": "",
                "review": "",
                "aggressive": "",
                "block": 0,
                "lesson_buff": "",
                "enthusiastic": "",
                "full_power_point": "",
            },
            "stance_desc": "",
            "negatives": "",
            "active_effects": [],
            "drinks": [],
            "observability": {
                "deck_order_known": False,
                "resource_panel_parsed": False,
                "exam_ranking_observed": False,
                "turn_color_observed": False,
                "drink_inventory_observed": False,
                "empty_hand_observed": False,
            },
        },
    }

    lesson_module._annotate_battle_preference(
        decision_state,
        preferred_index=1,
        reason="当前倍率窗口较高，优先兑现同色输出",
    )

    assert "recommended" not in decision_state["candidates"][0]
    assert "recommended" not in decision_state["llm_actions"][0]
    assert "system_recommendation" not in decision_state["stage_context"]
    assert "system_recommendation" not in decision_state["llm_snapshot"]["stage_context"]
    assert decision_state["local_preference"]["index"] == 1
    assert "倍率窗口较高" in decision_state["local_preference"]["reason"]
    rendered_prompt = render(
        "action_select.j2",
        snapshot=render("state_snapshot.j2", **decision_state["llm_snapshot"]),
        actions=[{"index": 1, "label": "ビジュアル打点", "id": "produce_card:visual"}],
    )
    assert "系统推荐" not in rendered_prompt
    assert "操作含义" not in rendered_prompt
    assert "ビジュアル打点" in rendered_prompt


def test_wait_battle_play_animation_end_waits_until_animation_disappears(monkeypatch):
    app = SimpleNamespace(
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        debug_tools=None,
    )
    sleep_calls = []
    center_sequences = [
        [1200, 1960],  # 浮空动画中
        [1235, 1962],  # 浮空动画中
        [1958, 1970],  # 动画结束，回到手牌基线
    ]
    state = {"idx": 0}

    def _next_centers(_results):
        idx = min(state["idx"], len(center_sequences) - 1)
        state["idx"] += 1
        return center_sequences[idx]

    monkeypatch.setattr(lesson_module, "_collect_visible_battle_card_center_ys", _next_centers)
    monkeypatch.setattr(lesson_module, "_resolve_battle_card_baseline_tolerance", lambda _app: (30, 8))
    monkeypatch.setattr(
        lesson_module.time,
        "sleep",
        lambda sec: sleep_calls.append(round(float(sec), 2)),
    )

    lesson_module._wait_battle_play_animation_end(
        app,
        phase="lesson",
        position="lesson_idle",
        pending_index=None,
    )

    assert sleep_calls == [round(lesson_module._BATTLE_PLAY_ANIMATION_WAIT_SLEEP, 2)] * 3


def test_lesson_handler_yields_when_runtime_frame_has_drifted_to_skill_reward(monkeypatch):
    app = SimpleNamespace(
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        device=_DeviceStub(),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = ProduceContext()
    monkeypatch.setattr(
        lesson_module,
        "detect_gameplay_state",
        lambda _app, _ctx: (GameplayPhase.SKILL_REWARD, GameplayPosition.SKILL_REWARD_SELECTED),
    )

    result = lesson_module.LessonHandler().handle(
        app,
        ctx,
        GameplayPhase.LESSON,
        GameplayPosition.LESSON_IDLE,
    )

    assert result.status == "no_action"
    assert "phase drift" in result.detail
    assert result.sleep_after == 0.0
    assert ctx.gameplay_phase == GameplayPhase.SKILL_REWARD
    assert ctx.gameplay_position == GameplayPosition.SKILL_REWARD_SELECTED


def test_lesson_handler_yields_when_execute_step_detects_phase_drift(monkeypatch):
    app = SimpleNamespace(
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        device=_DeviceStub(),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = ProduceContext()
    ctx.set_phase(GameplayPhase.SKILL_REWARD)
    ctx.set_position(GameplayPosition.SKILL_REWARD_SELECTED)

    monkeypatch.setattr(
        lesson_module.LessonHandler,
        "_detect_phase_drift",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        lesson_module,
        "execute_lesson_step",
        lambda *_args, **_kwargs: lesson_module.LessonStepResult(status="phase_drift"),
    )

    result = lesson_module.LessonHandler().handle(
        app,
        ctx,
        GameplayPhase.LESSON,
        GameplayPosition.LESSON_IDLE,
    )

    assert result.status == "no_action"
    assert "phase drift" in result.detail
    assert result.sleep_after == 0.0


def test_execute_lesson_step_dumps_probe_when_runtime_phase_drift_detected(monkeypatch):
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=_ResultsStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = ProduceContext()
    dumps = []

    monkeypatch.setattr(
        lesson_module,
        "_wait_battle_card_deal_settle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        lesson_module,
        "_wait_battle_play_animation_end",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        lesson_module,
        "detect_gameplay_state",
        lambda *_args, **_kwargs: (GameplayPhase.SKILL_REWARD, GameplayPosition.SKILL_REWARD_SELECTED),
    )
    monkeypatch.setattr(
        lesson_module,
        "_dump_phase_drift_probe",
        lambda *_args, **kwargs: dumps.append(kwargs),
    )

    result = lesson_module.execute_lesson_step(
        app,
        ctx,
        position=GameplayPosition.LESSON_IDLE,
        phase=GameplayPhase.LESSON,
    )

    assert result is not None
    assert result.status == "phase_drift"
    assert ctx.gameplay_phase == GameplayPhase.SKILL_REWARD
    assert ctx.gameplay_position == GameplayPosition.SKILL_REWARD_SELECTED
    assert dumps and dumps[0]["source"] == "execute_entry"
