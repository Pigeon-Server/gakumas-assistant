from types import SimpleNamespace

import numpy as np
import pytest

from src.constants.game.text.general_text import GeneralText
from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.model_type import YoloModelType
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay.dialogue import DialogueHandler, DialogueStepResult
from src.core.tasks.producer_challenge.gameplay.p_drink import PDrinkHandler
from src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop import (
    ProduceGameplayLoopStep,
    _try_external_page_recovery,
    _try_story_dialogue_recovery,
)
from src.core.tasks.producer_challenge import ui as ui_module
from src.core.tasks.producer_challenge.ui import gameplay_state as gameplay_state_module
from src.entity.Game.Page.Types.index import GamePageTypes


class _ResultsStub:
    def __init__(self, name: str):
        self.name = name

    def exists_label(self, _label):
        return False


class _FlakyResultsApp:
    def __init__(self, results_sequence):
        self._results_sequence = list(results_sequence)
        self._read_count = 0
        self.game_utils = SimpleNamespace(try_get_modal=lambda no_body=True: None)

    @property
    def latest_results(self):
        index = min(self._read_count, len(self._results_sequence) - 1)
        self._read_count += 1
        return self._results_sequence[index]


class _PhaseResultsStub:
    def __init__(self, present_labels, *, label_boxes=None):
        self._present_labels = set(present_labels)
        self._label_boxes = dict(label_boxes or {})
        self.frame = np.zeros((2340, 1080, 3), dtype=np.uint8)

    def exists_label(self, label):
        return label in self._present_labels

    def filter_by_label(self, label):
        if label in self._label_boxes:
            return list(self._label_boxes[label])
        return [
            SimpleNamespace(x=0, y=0, w=10, h=10, cx=5, cy=2200, label=label)
        ] if label in self._present_labels else []

    def __bool__(self):
        return True


class _DeviceStub:
    def __init__(self):
        self.clicks = []

    def click(self, x, y, el_label=""):
        self.clicks.append((int(x), int(y), str(el_label or "")))

    def click_element(self, element):
        self.clicks.append(("element", element))


def _build_switchable_app(**kwargs):
    """构造带新切模型入口的测试 app 桩。"""
    app = SimpleNamespace(**kwargs)

    def _switch_yolo_model(model_type, **switch_kwargs):  # noqa: ARG001
        yolo_engine = getattr(app, "yolo_engine", None)
        if hasattr(yolo_engine, "load_model"):
            yolo_engine.load_model(model_type)
        return True

    app.switch_yolo_model = _switch_yolo_model
    return app


def test_detect_gameplay_state_uses_single_results_snapshot(monkeypatch):
    first = _ResultsStub("first")
    second = _ResultsStub("second")
    app = _FlakyResultsApp([first, second])

    def fake_classify_gameplay_phase(results, *, ctx=None):  # noqa: ARG001
        return GameplayPhase.DIALOGUE if results is first else GameplayPhase.UNKNOWN

    def fake_classify_pipeline_position(
        results,
        *,
        modal_title=None,  # noqa: ARG001
        final_confirm=False,  # noqa: ARG001
        ctx=None,  # noqa: ARG001
        phase=None,
    ):
        assert results is first
        assert phase == GameplayPhase.DIALOGUE
        return GameplayPosition.DIALOGUE_OPTIONS

    monkeypatch.setattr(ui_module, "classify_gameplay_phase", fake_classify_gameplay_phase)
    monkeypatch.setattr(ui_module, "classify_pipeline_position", fake_classify_pipeline_position)

    phase, position = ui_module.detect_gameplay_state(app, ProduceContext())

    assert phase == GameplayPhase.DIALOGUE
    assert position == GameplayPosition.DIALOGUE_OPTIONS
    assert app._read_count == 1


def test_classify_gameplay_phase_does_not_fall_back_to_transition_hud_on_skill_reward_selection(monkeypatch):
    results = _PhaseResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.PC_TARGET,
    })
    panel = results.frame[
        int(results.frame.shape[0] * 0.56):int(results.frame.shape[0] * 0.93),
        int(results.frame.shape[1] * 0.05):int(results.frame.shape[1] * 0.95),
    ]
    panel[:] = 242
    monkeypatch.setattr(
        ui_module,
        "ocr_text",
        lambda _frame: "受け取るスキルカードを選んでください。 受け取る 再抽選 あと2回",
    )

    phase = gameplay_state_module.classify_gameplay_phase(results, ctx=ProduceContext())
    position = gameplay_state_module.classify_pipeline_position(results, phase=phase, ctx=ProduceContext())

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_IDLE


def test_classify_gameplay_state_treats_home_produce_resume_card_as_unknown(monkeypatch):
    results = _PhaseResultsStub({ProducerLabels.PC_MENU})
    monkeypatch.setattr(
        ui_module,
        "ocr_text",
        lambda _frame: "プロデュース中 2週目",
    )

    phase = gameplay_state_module.classify_gameplay_phase(results, ctx=ProduceContext())
    position = gameplay_state_module.classify_pipeline_position(
        results,
        phase=phase,
        ctx=ProduceContext(),
    )

    assert phase == GameplayPhase.UNKNOWN
    assert position == GameplayPosition.UNKNOWN


def test_produce_gameplay_loop_rechecks_transient_unknown_before_dispatch(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["pause_on_unknown"] = True
    app = SimpleNamespace()
    step = ProduceGameplayLoopStep()
    states = iter([
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_HUD),
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE),
    ])
    dispatched = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: next(states),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(
            dispatch=lambda _app, _ctx, phase, position: (
                dispatched.append((phase, position))
                or SimpleNamespace(status="exit", detail="done", sleep_after=0.0)
            )
        ),
    )

    assert step.execute(app, ctx) is True
    assert dispatched == [(GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE)]
    assert ctx.consecutive_unknowns == 0


def test_produce_gameplay_loop_syncs_visible_hud_before_dispatch(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["pause_on_unknown"] = True
    app = SimpleNamespace()
    step = ProduceGameplayLoopStep()
    states = iter([
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE),
    ])
    sync_calls = []
    dispatched = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: next(states),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sync_visible_planning_context",
        lambda _app, _ctx, *, phase, position, reason: (
            sync_calls.append((phase, position, reason)),
            _ctx.economy_state.update({"p_point": 18}),
            _ctx.parameter_state.update({"vocal": 123, "dance": 234, "visual": 345}),
        ),
    )

    def _dispatch(_app, _ctx, phase, position):
        assert _ctx.economy_state["p_point"] == 18
        assert _ctx.parameter_state["vocal"] == 123
        assert _ctx.parameter_state["dance"] == 234
        assert _ctx.parameter_state["visual"] == 345
        dispatched.append((phase, position))
        return SimpleNamespace(status="exit", detail="done", sleep_after=0.0)

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(dispatch=_dispatch),
    )

    assert step.execute(app, ctx) is True
    assert sync_calls == [(
        GameplayPhase.SCHEDULE,
        GameplayPosition.SCHEDULE_IDLE,
        "gameplay_loop_visible_hud_sync",
    )]
    assert dispatched == [(GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE)]


def test_produce_gameplay_loop_uses_unknown_retry_override(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["pause_on_unknown"] = True
    ctx.handler_state["unknown_retry_limit"] = 0
    ctx.handler_state["unknown_retry_sleep"] = 0.1
    ctx.handler_state["unknown_retry_override"] = {
        "reason": "confirm_schedule_action",
        "retry_limit": 2,
        "retry_sleep": 0.2,
    }
    app = SimpleNamespace()
    step = ProduceGameplayLoopStep()
    states = iter([
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_EVENT_OPTIONS),
    ])
    dispatched = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: next(states),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(
            dispatch=lambda _app, _ctx, phase, position: (
                dispatched.append((phase, position))
                or SimpleNamespace(status="exit", detail="done", sleep_after=0.0)
            )
        ),
    )

    assert step.execute(app, ctx) is True
    assert dispatched == [(GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_EVENT_OPTIONS)]
    assert "unknown_retry_override" not in ctx.handler_state


def test_produce_gameplay_loop_sets_loading_retry_override(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["pause_on_unknown"] = True
    ctx.handler_state["unknown_retry_limit"] = 0
    ctx.handler_state["unknown_retry_sleep"] = 0.1
    ctx.handler_state["loading_unknown_retry_limit"] = 2
    ctx.handler_state["loading_unknown_retry_sleep"] = 0.2
    app = SimpleNamespace()
    step = ProduceGameplayLoopStep()
    states = iter([
        (GameplayPhase.LOADING, GameplayPosition.TRANSITION_EMPTY),
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE),
    ])
    dispatched = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: next(states),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(
            dispatch=lambda _app, _ctx, phase, position: (
                dispatched.append((phase, position))
                or SimpleNamespace(status="exit", detail="done", sleep_after=0.0)
            )
        ),
    )

    assert step.execute(app, ctx) is True
    assert dispatched == [(GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE)]
    assert "unknown_retry_override" not in ctx.handler_state


def test_produce_gameplay_loop_taps_resume_title_screen(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["pause_on_unknown"] = True
    ctx.handler_state["loading_unknown_retry_limit"] = 2
    ctx.handler_state["loading_unknown_retry_sleep"] = 0.2
    app = SimpleNamespace(device=_DeviceStub())
    step = ProduceGameplayLoopStep()
    states = iter([
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_RESUME_TITLE),
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_RESUME_TITLE),
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_RESUME_TITLE),
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE),
    ])
    dispatched = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: next(states),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(
            dispatch=lambda _app, _ctx, phase, position: (
                dispatched.append((phase, position))
                or SimpleNamespace(status="exit", detail="done", sleep_after=0.0)
            )
        ),
    )

    assert step.execute(app, ctx) is True
    assert app.device.clicks == [(540, 1498, "resume-title-advance")]
    assert dispatched == [(GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE)]
    assert "unknown_retry_override" not in ctx.handler_state


def test_try_external_page_recovery_returns_false_without_external_signal(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "some unrelated screen",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("should not try returning home without external signal"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("should not reopen produce without external signal"),
    )

    assert _try_external_page_recovery(app, ctx) is False
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("model", YoloModelType.PRODUCER),
    ]


def test_try_external_page_recovery_round_trips_home_and_resume(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    resume_results = iter([False, True])
    app = _build_switchable_app(
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: GeneralText.LOGIN_BONUS,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or next(resume_results),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: events.append(("home",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: events.append(("open_produce", timeout)),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("home",),
        ("open_produce", 10),
        ("resume", 8.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "external_page_recovery",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_try_external_page_recovery_treats_story_message_page_as_dialogue(monkeypatch):
    events: list[tuple] = []
    skip_box = SimpleNamespace(name="skip")
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_results=_PhaseResultsStub(
            {BaseUILabels.SKIP_BUTTON},
            label_boxes={BaseUILabels.SKIP_BUTTON: [skip_box]},
        ),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "藤田ことね 世界一可愛い私 1話 メッセージ見ました。あたしに大事な話がある・・・って。 SKIP",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("story page should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("story page should not reopen produce"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [("element", skip_box)]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "story_dialogue_recovery_skip",
        "retry_limit": 8,
        "retry_sleep": 0.7,
    }


def test_try_external_page_recovery_clicks_reward_receive_confirmation(monkeypatch):
    events: list[tuple] = []
    confirm_box = SimpleNamespace(name="receive", cy=1890)
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_results=_PhaseResultsStub(
            {BaseUILabels.BUTTON},
            label_boxes={BaseUILabels.BUTTON: [confirm_box]},
        ),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "活動支給 ホットコーヒー 今やる気+3 受け取る",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("receive confirmation should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("receive confirmation should not reopen produce"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [("element", confirm_box)]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "reward_receive_confirmation",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_try_external_page_recovery_taps_live_tap_to_start_prompt(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "横画面で開始します TAP TO START",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("live tap to start page should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("live tap to start page should not reopen produce"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [(540, 1802, "live-tap-to-start-recovery")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "live_tap_to_start_recovery",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_try_external_page_recovery_clicks_result_memory_generate(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "MEMORY 藤田ことね 生成",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("result memory generate should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("result memory generate should not reopen produce"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: None,
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [(540, 1802, "result-memory-generate-recovery")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_memory_generate_recovery",
        "retry_limit": 2,
        "retry_sleep": 0.3,
    }


def test_try_external_page_recovery_clicks_short_generate_text_without_memory_keyword(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "藤田ことね生成",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("short generate text should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("short generate text should not reopen produce"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: None,
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [(540, 1802, "result-memory-generate-recovery")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_memory_generate_recovery",
        "retry_limit": 2,
        "retry_sleep": 0.3,
    }


def test_try_external_page_recovery_taps_result_chain_tap_page(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "初期ビジュアル上昇+15 ボーカルSPレッスン開始時 TAP",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("result tap page should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("result tap page should not reopen produce"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: None,
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [(540, 1170, "result-chain-tap-recovery")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_chain_tap_recovery_tap",
        "retry_limit": 2,
        "retry_sleep": 0.3,
    }


def test_try_external_page_recovery_clicks_next_button_on_result_chain_tap_page(monkeypatch):
    events: list[tuple] = []
    next_box = SimpleNamespace(name="next")
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "初期ビジュアル上昇+15 TAP 次へ",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("result tap page with next should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("result tap page with next should not reopen produce"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: next_box,
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [("element", next_box)]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_chain_tap_recovery_next",
        "retry_limit": 2,
        "retry_sleep": 0.3,
    }
    assert ctx.handler_state["result_chain_finish_pending"] is True


def test_try_external_page_recovery_closes_result_reward_overlay_with_local_ocr(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2400, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2400, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "で",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )

    def _fake_ocr(image):
        shape = getattr(image, "shape", None)
        if shape == (288, 669, 3):
            return "アイテム獲得"
        if shape == (311, 788, 3):
            return "H.I.F直前キャンペーン交換券"
        if shape == (600, 476, 3):
            return "プロデュース完了"
        return ""

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.ocr_text",
        _fake_ocr,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("result reward overlay should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("result reward overlay should not reopen produce"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: pytest.fail("result reward overlay should not look for next button"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [(540, 2208, "result-reward-overlay-recovery")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_reward_overlay_recovery",
        "retry_limit": 2,
        "retry_sleep": 0.3,
    }


def test_try_external_page_recovery_closes_result_reward_overlay_without_title_ocr(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2400, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2400, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "-",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )

    def _fake_ocr(image):
        shape = getattr(image, "shape", None)
        if shape == (288, 669, 3):
            return ""
        if shape == (311, 788, 3):
            return "キャンペーン交換券"
        if shape == (600, 476, 3):
            return "プロデュース完了"
        return ""

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.ocr_text",
        _fake_ocr,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("result reward overlay should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("result reward overlay should not reopen produce"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: pytest.fail("result reward overlay should not look for next button"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [(540, 2208, "result-reward-overlay-recovery")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_reward_overlay_recovery",
        "retry_limit": 2,
        "retry_sleep": 0.3,
    }


def test_try_external_page_recovery_clicks_next_on_result_closeout_summary(monkeypatch):
    events: list[tuple] = []
    next_box = SimpleNamespace(name="next")
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2400, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2400, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "歩Ply56ユロファン数936,5861+(2284」卜進劫次へプロデュース履歴",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )

    def _fake_ocr(image):
        shape = getattr(image, "shape", None)
        if shape == (576, 497, 3):
            return "プロデュース完了"
        if shape == (552, 519, 3):
            return "アチーブメント進捗"
        if shape == (288, 540, 3):
            return "次へ"
        if shape == (288, 324, 3):
            return "プロデュース履歴"
        return ""

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.ocr_text",
        _fake_ocr,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: next_box,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("result closeout summary should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("result closeout summary should not reopen produce"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [("element", next_box)]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_closeout_summary_recovery_next",
        "retry_limit": 2,
        "retry_sleep": 0.3,
    }


def test_try_external_page_recovery_taps_result_closeout_summary_without_button_match(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2400, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2400, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "次へプロデュース履歴",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )

    def _fake_ocr(image):
        shape = getattr(image, "shape", None)
        if shape == (576, 497, 3):
            return "プロデュース完了"
        if shape == (552, 519, 3):
            return ""
        if shape == (288, 540, 3):
            return "次へ"
        if shape == (288, 324, 3):
            return "プロデュース履歴"
        return ""

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.ocr_text",
        _fake_ocr,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("result closeout summary should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("result closeout summary should not reopen produce"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [(540, 2160, "result-closeout-summary-recovery")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_closeout_summary_recovery_tap",
        "retry_limit": 2,
        "retry_sleep": 0.3,
    }


def test_try_external_page_recovery_marks_result_complete_when_home_after_next(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    ctx.handler_state["result_chain_finish_pending"] = True
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(
            update_current_location=lambda: GamePageTypes.MAIN_MENU__HOME
        ),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "ホーム プロデュース",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("result completion on home should not wait enter home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("result completion on home should not reopen produce"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: pytest.fail("result completion on home should not look for next button"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == []
    assert "result_chain_finish_pending" not in ctx.handler_state
    assert ctx.handler_state["result_chain_completed"] is True


def test_try_external_page_recovery_marks_result_complete_on_producer_entry_after_finish(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    ctx.handler_state["produce_finishing_pending"] = True
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_PhaseResultsStub({BaseUILabels.PRODUCER_REGULAR}),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(
            update_current_location=lambda: GamePageTypes.HOME_TAB.PRODUCER
        ),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "初 レギュラー プロ マスター",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("producer entry completion should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("producer entry completion should not reopen produce"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == []
    assert ctx.handler_state["result_chain_completed"] is True


def test_try_external_page_recovery_keeps_strict_failure_on_producer_entry_without_finish_state(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_PhaseResultsStub({BaseUILabels.PRODUCER_REGULAR}),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(
            update_current_location=lambda: GamePageTypes.HOME_TAB.PRODUCER
        ),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "初 レギュラー",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: events.append(("wait_home", None)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: events.append(("open_produce", timeout)),
    )

    with pytest.raises(RuntimeError, match="外页恢复失败"):
        _try_external_page_recovery(app, ctx)
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("wait_home", None),
        ("open_produce", 10),
        ("resume", 8.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert "result_chain_completed" not in ctx.handler_state


def test_try_external_page_recovery_marks_result_complete_after_reopen_to_producer_entry(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    ctx.handler_state["produce_finishing_pending"] = True

    states = iter([
        GamePageTypes.START_GAME,
        GamePageTypes.HOME_TAB.PRODUCER,
    ])
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_PhaseResultsStub({BaseUILabels.PRODUCER_REGULAR}),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(
            update_current_location=lambda: next(states)
        ),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "プロデュース 初 レギュラー",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: events.append(("wait_home", None)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: events.append(("open_produce", timeout)),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("wait_home", None),
        ("open_produce", 10),
        ("resume", 8.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert ctx.handler_state["result_chain_completed"] is True


def test_try_external_page_recovery_marks_result_complete_after_reopen_to_producer_location_only(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    ctx.handler_state["produce_finishing_pending"] = True

    states = iter([
        GamePageTypes.START_GAME,
        GamePageTypes.HOME_TAB.PRODUCER,
    ])
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_PhaseResultsStub(set()),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(
            update_current_location=lambda: next(states)
        ),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "プロデュース",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: events.append(("wait_home", None)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: events.append(("open_produce", timeout)),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("wait_home", None),
        ("open_produce", 10),
        ("resume", 8.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert ctx.handler_state["result_chain_completed"] is True


def test_try_external_page_recovery_does_not_tap_result_page_on_known_home(monkeypatch):
    events: list[tuple] = []
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(frame=np.zeros((2340, 1080, 3), dtype=np.uint8)),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(
            update_current_location=lambda: GamePageTypes.MAIN_MENU__HOME
        ),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "初期ビジュアル上昇+15 TAP",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or (timeout >= 8.0),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: events.append(("wait_home", None)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: events.append(("open_produce", timeout)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.find_button",
        lambda *_args, **_kwargs: pytest.fail("known home should not look for result next button"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("wait_home", None),
        ("open_produce", 10),
        ("resume", 8.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == []
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "external_page_recovery",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_try_external_page_recovery_uses_back_button_before_giving_up(monkeypatch):
    events: list[tuple] = []
    back_box = SimpleNamespace(name="back")
    ctx = ProduceContext()
    app = _build_switchable_app(
        latest_results=_PhaseResultsStub(
            {BaseUILabels.BACK_BTN},
            label_boxes={BaseUILabels.BACK_BTN: [back_box]},
        ),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(update_current_location=lambda: GamePageTypes.UNKNOWN),
        yolo_engine=SimpleNamespace(
            load_model=lambda model_type: events.append(("model", model_type))
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.collect_frame_text",
        lambda _results: "WLAN 保存済 102-private 102-public",
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.resume_resumable_produce",
        lambda _app, timeout=0.0: events.append(("resume", timeout)) or False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.action__wait_enter_home",
        lambda _app: pytest.fail("back button recovery should not go home"),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.open_produce_entry_from_home",
        lambda _app, timeout=0.0: pytest.fail("back button recovery should not reopen produce"),
    )

    assert _try_external_page_recovery(app, ctx) is True
    assert events == [
        ("model", YoloModelType.BASE_UI),
        ("resume", 1.0),
        ("model", YoloModelType.PRODUCER),
    ]
    assert app.device.clicks == [("element", back_box)]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "external_back_button_recovery",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_produce_gameplay_loop_attempts_external_recovery_before_pausing(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["pause_on_unknown"] = True
    ctx.handler_state["unknown_retry_limit"] = 0
    app = SimpleNamespace()
    step = ProduceGameplayLoopStep()
    states = iter([
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE),
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE),
    ])
    dispatched = []
    recoveries = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: next(states),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop._try_external_page_recovery",
        lambda _app, _ctx: recoveries.append(True) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(
            dispatch=lambda _app, _ctx, phase, position: (
                dispatched.append((phase, position))
                or SimpleNamespace(status="exit", detail="done", sleep_after=0.0)
            )
        ),
    )

    assert step.execute(app, ctx) is True
    assert recoveries == [True]
    assert dispatched == [(GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE)]


def test_produce_gameplay_loop_stops_when_result_chain_completed(monkeypatch):
    ctx = ProduceContext()
    ctx.max_gameplay_loops = 1
    app = SimpleNamespace()
    step = ProduceGameplayLoopStep()

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop._try_external_page_recovery",
        lambda _app, _ctx: _ctx.handler_state.__setitem__("result_chain_completed", True) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(dispatch=lambda *args, **kwargs: pytest.fail("dispatcher should not run")),
    )

    assert step.execute(app, ctx) is True
    assert "result_chain_completed" not in ctx.handler_state


def test_produce_gameplay_loop_rechecks_after_failed_external_probe(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["pause_on_unknown"] = True
    ctx.handler_state["unknown_retry_limit"] = 0
    app = SimpleNamespace()
    step = ProduceGameplayLoopStep()
    states = iter([
        (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE),
        (GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE),
    ])
    dispatched = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: next(states),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop._try_external_page_recovery",
        lambda _app, _ctx: False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(
            dispatch=lambda _app, _ctx, phase, position: (
                dispatched.append((phase, position))
                or SimpleNamespace(status="exit", detail="done", sleep_after=0.0)
            )
        ),
    )

    assert step.execute(app, ctx) is True
    assert dispatched == [(GameplayPhase.SCHEDULE, GameplayPosition.SCHEDULE_IDLE)]


def test_dialogue_handler_sets_retry_override_after_fast_forward(monkeypatch):
    handler = DialogueHandler()
    ctx = ProduceContext()
    ctx.handler_state["dialogue_transition_unknown_retry_limit"] = 3
    ctx.handler_state["dialogue_transition_unknown_retry_sleep"] = 0.9

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.gameplay.dialogue.execute_dialogue_step",
        lambda _app, _ctx, position: DialogueStepResult(status="fast_forward"),
    )

    result = handler.handle(SimpleNamespace(), ctx, GameplayPhase.DIALOGUE, GameplayPosition.DIALOGUE_CONTINUE)

    assert result.status == "ok"
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "dialogue_fast_forward",
        "retry_limit": 3,
        "retry_sleep": 0.9,
    }


def test_story_dialogue_recovery_prefers_skip_over_fast_forward(monkeypatch):
    device = _DeviceStub()
    ff_box = SimpleNamespace(cx=200, cy=210, x=150, y=180, w=250, h=240)
    skip_box = SimpleNamespace(cx=300, cy=310, x=260, y=280, w=360, h=340)
    app = SimpleNamespace(
        latest_results=_PhaseResultsStub(
            {BaseUILabels.PLOT_FAST_FORWARD_BUTTON, BaseUILabels.SKIP_BUTTON},
            label_boxes={
                BaseUILabels.PLOT_FAST_FORWARD_BUTTON: [ff_box],
                BaseUILabels.SKIP_BUTTON: [skip_box],
            },
        ),
        device=device,
    )
    ctx = ProduceContext()

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop._looks_like_story_dialogue_recovery_page",
        lambda _results, _text: True,
    )

    assert _try_story_dialogue_recovery(app, ctx, "dummy text") is True
    assert device.clicks == [("element", skip_box)]


def test_story_dialogue_recovery_does_not_toggle_fast_forward_repeatedly(monkeypatch):
    device = _DeviceStub()
    ff_box = SimpleNamespace(cx=200, cy=210, x=150, y=180, w=250, h=240)
    app = SimpleNamespace(
        latest_results=_PhaseResultsStub(
            {BaseUILabels.PLOT_FAST_FORWARD_BUTTON},
            label_boxes={BaseUILabels.PLOT_FAST_FORWARD_BUTTON: [ff_box]},
        ),
        device=device,
    )
    ctx = ProduceContext()
    ctx.handler_state["dialogue_fast_forward_enabled"] = True

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop._looks_like_story_dialogue_recovery_page",
        lambda _results, _text: True,
    )

    assert _try_story_dialogue_recovery(app, ctx, "dummy text") is True
    assert device.clicks == [(540, 1919, "story-dialogue-recovery")]


def test_story_dialogue_recovery_respects_orange_enabled_fast_forward(monkeypatch):
    device = _DeviceStub()
    ff_box = SimpleNamespace(cx=200, cy=210, x=150, y=180, w=250, h=240)
    app = SimpleNamespace(
        latest_results=_PhaseResultsStub(
            {BaseUILabels.PLOT_FAST_FORWARD_BUTTON},
            label_boxes={BaseUILabels.PLOT_FAST_FORWARD_BUTTON: [ff_box]},
        ),
        device=device,
    )
    ctx = ProduceContext()

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop._looks_like_story_dialogue_recovery_page",
        lambda _results, _text: True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.probe_fast_forward_enabled_state",
        lambda *_args, **_kwargs: (True, 0.38),
    )

    assert _try_story_dialogue_recovery(app, ctx, "dummy text") is True
    assert device.clicks == [(540, 1919, "story-dialogue-recovery")]
    assert ctx.handler_state["dialogue_fast_forward_enabled"] is True


def test_produce_gameplay_loop_pauses_after_confirmed_unknown_page(monkeypatch):
    ctx = ProduceContext()
    ctx.handler_state["pause_on_unknown"] = True
    app = SimpleNamespace()
    step = ProduceGameplayLoopStep()

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.detect_gameplay_state",
        lambda _app, _ctx: (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_HUD),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop.build_default_dispatcher",
        lambda: SimpleNamespace(dispatch=lambda *args, **kwargs: pytest.fail("dispatcher should not run")),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop._try_external_page_recovery",
        lambda _app, _ctx: False,
    )

    with pytest.raises(RuntimeError, match="遇到未识别页面，已暂停等待分析"):
        step.execute(app, ctx)
    assert ctx.consecutive_unknowns == 1


def test_progress_only_dialogue_text_is_classified_as_dialogue_continue(monkeypatch):
    results = _PhaseResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_TARGET,
        ProducerLabels.PC_P_POINT,
    })
    ctx = ProduceContext()

    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "もちろんやらせていただきますッ!",
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.DIALOGUE
    assert position == GameplayPosition.DIALOGUE_CONTINUE


def test_title_logo_screen_is_classified_as_resume_transition(monkeypatch):
    results = _PhaseResultsStub(set())
    ctx = ProduceContext()

    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "学園アイドルマスター",
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.UNKNOWN
    assert position == GameplayPosition.TRANSITION_RESUME_TITLE


def test_bottom_bar_p_drink_does_not_block_dialogue_continue(monkeypatch):
    results = _PhaseResultsStub(
        {
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_TARGET,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.P_DRINK,
        },
        label_boxes={
            ProducerLabels.P_DRINK: [SimpleNamespace(cy=2200)],
        },
    )
    ctx = ProduceContext()

    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "でもコンプだしならよく使うし、ちょーどいいかも？",
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.DIALOGUE
    assert position == GameplayPosition.DIALOGUE_CONTINUE


def test_skill_card_info_showcase_is_classified_as_skill_reward_showcase(monkeypatch):
    showcase_card = SimpleNamespace(x=420, y=1040, w=660, h=1440, cx=540, cy=1240)
    results = _PhaseResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_TARGET,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.SKILL_CARD_MENTAL,
        ProducerLabels.SKILL_CARD_INFO,
    }, label_boxes={
        ProducerLabels.SKILL_CARD_MENTAL: [showcase_card],
        ProducerLabels.SKILL_CARD_INFO: [showcase_card],
    })
    ctx = ProduceContext()

    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "心のアルバム パラメータ+3 元気+3 次のターン、スキルカードを引く レッスン中1回",
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_SHOWCASE


def test_single_param_action_info_is_lesson_summary_showcase(monkeypatch):
    action_info = SimpleNamespace(x=80, y=1668, w=1030, h=1932, cx=555, cy=1800)
    dance = SimpleNamespace(x=406, y=1427, w=673, h=1598, cx=539, cy=1512)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PC_ACTION_INFO,
            ProducerLabels.PARAM_DANCE,
        },
        label_boxes={
            ProducerLabels.PC_ACTION_INFO: [action_info],
            ProducerLabels.PARAM_DANCE: [dance],
        },
    )
    ctx = ProduceContext()
    debug_boxes: list[dict] = []
    debugger = SimpleNamespace(
        add_box=lambda x, y, w, h, **kwargs: debug_boxes.append(
            {"x": x, "y": y, "w": w, "h": h, **kwargs}
        )
    )

    monkeypatch.setattr(ui_module, "collect_frame_text", lambda _results: pytest.fail("不应走整帧 OCR"))
    monkeypatch.setattr(gameplay_state_module, "collect_frame_text", lambda _results: pytest.fail("不应走整帧 OCR"))
    monkeypatch.setattr(gameplay_state_module, "DebugTools", lambda: debugger)

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.LESSON
    assert position == GameplayPosition.LESSON_SUMMARY_SHOWCASE
    assert {box["label"] for box in debug_boxes} == {
        "lesson_summary:action_info",
        f"lesson_summary:{ProducerLabels.PARAM_DANCE}",
    }


def test_single_param_short_text_showcase_without_action_info_is_lesson_summary_showcase(monkeypatch):
    vocal = SimpleNamespace(x=72, y=1428, w=318, h=1596, cx=195, cy=1512)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PARAM_VOCAL,
        },
        label_boxes={
            ProducerLabels.PARAM_VOCAL: [vocal],
        },
    )
    ctx = ProduceContext()
    debug_boxes: list[dict] = []
    debugger = SimpleNamespace(
        add_box=lambda x, y, w, h, **kwargs: debug_boxes.append(
            {"x": x, "y": y, "w": w, "h": h, **kwargs}
        )
    )
    roi_texts: list[str] = []

    def _fake_ocr(image):
        roi_texts.append(f"{image.shape[1]}x{image.shape[0]}")
        return "Vo.が43上昇した!"

    monkeypatch.setattr(ui_module, "ocr_text", _fake_ocr)
    monkeypatch.setattr(gameplay_state_module, "ocr_text", _fake_ocr)
    monkeypatch.setattr(gameplay_state_module, "DebugTools", lambda: debugger)

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.LESSON
    assert position == GameplayPosition.LESSON_SUMMARY_SHOWCASE
    assert roi_texts == ["993x468"]
    assert {box["label"] for box in debug_boxes} == {
        "lesson_summary:ocr_roi",
        f"lesson_summary:{ProducerLabels.PARAM_VOCAL}",
    }


def test_param_only_layout_without_action_info_is_lesson_summary_showcase(monkeypatch):
    vocal = SimpleNamespace(x=72, y=1428, w=318, h=1596, cx=195, cy=1512)
    dance = SimpleNamespace(x=406, y=1427, w=673, h=1598, cx=539, cy=1512)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PARAM_VOCAL,
            ProducerLabels.PARAM_DANCE,
        },
        label_boxes={
            ProducerLabels.PARAM_VOCAL: [vocal],
            ProducerLabels.PARAM_DANCE: [dance],
        },
    )
    ctx = ProduceContext()
    debug_boxes: list[dict] = []
    debugger = SimpleNamespace(
        add_box=lambda x, y, w, h, **kwargs: debug_boxes.append(
            {"x": x, "y": y, "w": w, "h": h, **kwargs}
        )
    )

    monkeypatch.setattr(ui_module, "ocr_text", lambda _image: pytest.fail("不应走 OCR"))
    monkeypatch.setattr(gameplay_state_module, "ocr_text", lambda _image: pytest.fail("不应走 OCR"))
    monkeypatch.setattr(gameplay_state_module, "DebugTools", lambda: debugger)

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.LESSON
    assert position == GameplayPosition.LESSON_SUMMARY_SHOWCASE
    assert {box["label"] for box in debug_boxes} == {
        "lesson_summary:param_only_layout",
        f"lesson_summary:{ProducerLabels.PARAM_VOCAL}",
        f"lesson_summary:{ProducerLabels.PARAM_DANCE}",
    }


def test_short_text_only_showcase_without_any_detection_is_lesson_summary_showcase(monkeypatch):
    results = _PhaseResultsStub(set())
    ctx = ProduceContext()
    debug_boxes: list[dict] = []
    debugger = SimpleNamespace(
        add_box=lambda x, y, w, h, **kwargs: debug_boxes.append(
            {"x": x, "y": y, "w": w, "h": h, **kwargs}
        )
    )

    monkeypatch.setattr(ui_module, "ocr_text", lambda _image: "Vo.が43上昇した!")
    monkeypatch.setattr(gameplay_state_module, "ocr_text", lambda _image: "Vo.が43上昇した!")
    monkeypatch.setattr(gameplay_state_module, "DebugTools", lambda: debugger)

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.LESSON
    assert position == GameplayPosition.LESSON_SUMMARY_SHOWCASE
    assert any(box["label"] == "lesson_summary:ocr_roi" for box in debug_boxes)


def test_lesson_p_drink_showcase_without_detection_is_p_drink_showcase(monkeypatch):
    results = _PhaseResultsStub(set())
    results.frame[:] = 120
    results.frame[int(results.frame.shape[0] * 0.55):int(results.frame.shape[0] * 0.86), int(results.frame.shape[1] * 0.06):int(results.frame.shape[1] * 0.94)] = 235
    ctx = ProduceContext()
    ctx.last_stable_position = GameplayPosition.LESSON_SUMMARY_SHOWCASE
    debug_boxes: list[dict] = []
    debugger = SimpleNamespace(
        add_box=lambda x, y, w, h, **kwargs: debug_boxes.append(
            {"x": x, "y": y, "w": w, "h": h, **kwargs}
        )
    )

    monkeypatch.setattr(
        ui_module,
        "ocr_text",
        lambda _image: "Vo.レッスン Pポイント 初星水 獲得 パラメータ+10",
    )
    monkeypatch.setattr(
        gameplay_state_module,
        "ocr_text",
        lambda _image: "Vo.レッスン Pポイント 初星水 獲得 パラメータ+10",
    )
    monkeypatch.setattr(gameplay_state_module, "DebugTools", lambda: debugger)

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.P_DRINK
    assert position == GameplayPosition.P_DRINK_SHOWCASE
    assert any(box["label"] == "lesson_p_drink_showcase:panel" for box in debug_boxes)


def test_p_drink_showcase_handler_advances_with_safe_tap():
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
    )
    ctx = ProduceContext()

    result = PDrinkHandler().handle(
        app,
        ctx,
        GameplayPhase.P_DRINK,
        GameplayPosition.P_DRINK_SHOWCASE,
    )

    assert result.status == "ok"
    assert app.device.clicks == [(540, 1919, "p_drink_showcase_advance")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "p_drink_showcase_transition",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_memory_effect_single_card_with_info_panel_is_showcase_without_ocr(monkeypatch):
    top_card = SimpleNamespace(x=370, y=860, w=710, h=1460, cx=540, cy=1160)
    info_panel = SimpleNamespace(x=120, y=1500, w=960, h=2060, cx=540, cy=1780)
    results = _PhaseResultsStub(
        {
            ProducerLabels.SKILL_CARD_ACTIVE,
            ProducerLabels.SKILL_CARD_INFO,
        },
        label_boxes={
            ProducerLabels.SKILL_CARD_ACTIVE: [top_card],
            ProducerLabels.SKILL_CARD_INFO: [info_panel],
        },
    )
    ctx = ProduceContext()

    monkeypatch.setattr(ui_module, "collect_frame_text", lambda _results: "")

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_SHOWCASE


def test_action_info_tap_result_page_is_classified_as_result_memory_page(monkeypatch):
    action_info = SimpleNamespace(x=111, y=985, w=990, h=1114, cx=550, cy=1049)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PC_ACTION_INFO,
        },
        label_boxes={
            ProducerLabels.PC_ACTION_INFO: [action_info],
        },
    )
    ctx = ProduceContext()

    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "ダンスSPレッスン開始時、35%の確率で 消費体力減少2ターン TAP",
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.RESULT
    assert position == GameplayPosition.RESULT_MEMORY_PAGE


def test_skill_reward_detail_with_receive_button_is_showcase(monkeypatch):
    top_card = SimpleNamespace(x=370, y=860, w=710, h=1460, cx=540, cy=1160)
    info_panel = SimpleNamespace(x=120, y=1500, w=960, h=2060, cx=540, cy=1780)
    receive_button = SimpleNamespace(x=330, y=1880, w=750, h=2040, cx=540, cy=1960)
    redraw_button = SimpleNamespace(x=820, y=1880, w=1020, h=2040, cx=920, cy=1960)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_TARGET,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.SKILL_CARD_ACTIVE,
            ProducerLabels.SKILL_CARD_INFO,
            ProducerLabels.CONFIRM_BUTTON,
            BaseUILabels.BUTTON,
        },
        label_boxes={
            ProducerLabels.SKILL_CARD_ACTIVE: [top_card],
            ProducerLabels.SKILL_CARD_INFO: [info_panel],
            ProducerLabels.CONFIRM_BUTTON: [receive_button],
            BaseUILabels.BUTTON: [redraw_button],
        },
    )
    ctx = ProduceContext()

    monkeypatch.setattr(ui_module, "collect_frame_text", lambda _results: "")

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_SHOWCASE


def test_low_single_skill_card_stays_reward_selected(monkeypatch):
    low_card = SimpleNamespace(x=420, y=1700, w=660, h=2120, cx=540, cy=1910)
    results = _PhaseResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_TARGET,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.PC_ACTION_INFO,
        ProducerLabels.SKILL_CARD_MENTAL,
        ProducerLabels.CONFIRM_BUTTON,
    }, label_boxes={
        ProducerLabels.SKILL_CARD_MENTAL: [low_card],
    })
    ctx = ProduceContext()
    monkeypatch.setattr(ui_module, "collect_frame_text", lambda _results: "ワクワクが止まらないを強化しました")

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_SELECTED


def test_skill_reward_select_prompt_forces_idle(monkeypatch):
    low_card = SimpleNamespace(x=420, y=1700, w=660, h=2120, cx=540, cy=1910)
    button_box = SimpleNamespace(x=860, y=1900, w=1020, h=2040, cx=940, cy=1970, label=BaseUILabels.BUTTON)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_TARGET,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.SKILL_CARD_MENTAL,
            BaseUILabels.BUTTON,
        },
        label_boxes={
            ProducerLabels.SKILL_CARD_MENTAL: [low_card],
            BaseUILabels.BUTTON: [button_box],
        },
    )
    results.frame[
        int(results.frame.shape[0] * 0.56):int(results.frame.shape[0] * 0.93),
        int(results.frame.shape[1] * 0.05):int(results.frame.shape[1] * 0.95),
    ] = 240
    ctx = ProduceContext()
    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "受け取るスキルカードを選んでください 受け取る 再抽選",
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_IDLE


def test_skill_reward_receive_button_overrides_select_prompt(monkeypatch):
    low_card = SimpleNamespace(x=420, y=1700, w=660, h=2120, cx=540, cy=1910)
    receive_box = SimpleNamespace(x=300, y=1900, w=640, h=2040, cx=470, cy=1970, label=BaseUILabels.BUTTON)
    redraw_box = SimpleNamespace(x=860, y=1900, w=1020, h=2040, cx=940, cy=1970, label=BaseUILabels.BUTTON)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_TARGET,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.SKILL_CARD_MENTAL,
            BaseUILabels.BUTTON,
        },
        label_boxes={
            ProducerLabels.SKILL_CARD_MENTAL: [low_card],
            BaseUILabels.BUTTON: [receive_box, redraw_box],
        },
    )
    results.frame[
        int(results.frame.shape[0] * 0.56):int(results.frame.shape[0] * 0.93),
        int(results.frame.shape[1] * 0.05):int(results.frame.shape[1] * 0.95),
    ] = 240
    ctx = ProduceContext()
    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "受け取るスキルカードを選んでください 受け取る 再抽選",
    )
    monkeypatch.setattr(
        ui_module,
        "collect_button_like_texts",
        lambda _results: ["受け取る", "あと3回 再抽選"],
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_SELECTED


def test_skill_reward_selected_layout_without_confirm_button(monkeypatch):
    raised_card = SimpleNamespace(x=230, y=1460, w=430, h=1660, cx=330, cy=1560, label=ProducerLabels.SKILL_CARD_MENTAL)
    mid_card = SimpleNamespace(x=440, y=1540, w=635, h=1740, cx=538, cy=1640, label=ProducerLabels.SKILL_CARD_MENTAL)
    right_card = SimpleNamespace(x=650, y=1540, w=845, h=1740, cx=748, cy=1640, label=ProducerLabels.SKILL_CARD_MENTAL)
    drink_box = SimpleNamespace(x=70, y=2220, w=185, h=2338, cx=128, cy=2280, label=ProducerLabels.P_DRINK)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_TARGET,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.SKILL_CARD_MENTAL,
            ProducerLabels.P_DRINK,
        },
        label_boxes={
            ProducerLabels.SKILL_CARD_MENTAL: [raised_card, mid_card, right_card],
            ProducerLabels.P_DRINK: [drink_box],
        },
    )
    results.frame[
        int(results.frame.shape[0] * 0.56):int(results.frame.shape[0] * 0.93),
        int(results.frame.shape[1] * 0.05):int(results.frame.shape[1] * 0.95),
    ] = 240
    ctx = ProduceContext()
    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "受け取るスキルカードを選んでください 前途洋々 パラメータ+8 元気+7 受け取る 獲得ガイド",
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_SELECTED


def test_skill_reward_redraw_only_button_stays_idle(monkeypatch):
    low_card = SimpleNamespace(x=420, y=1700, w=660, h=2120, cx=540, cy=1910)
    button_box = SimpleNamespace(x=860, y=1900, w=1020, h=2040, cx=940, cy=1970, label=BaseUILabels.BUTTON)
    results = _PhaseResultsStub(
        {
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_TARGET,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.SKILL_CARD_MENTAL,
            BaseUILabels.BUTTON,
        },
        label_boxes={
            ProducerLabels.SKILL_CARD_MENTAL: [low_card],
            BaseUILabels.BUTTON: [button_box],
        },
    )
    results.frame[
        int(results.frame.shape[0] * 0.56):int(results.frame.shape[0] * 0.93),
        int(results.frame.shape[1] * 0.05):int(results.frame.shape[1] * 0.95),
    ] = 240
    ctx = ProduceContext()
    monkeypatch.setattr(ui_module, "collect_frame_text", lambda _results: "受け取る 再抽選 あと2回")
    monkeypatch.setattr(ui_module, "collect_button_like_texts", lambda _results: ["再抽選"])

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.SKILL_REWARD
    assert position == GameplayPosition.SKILL_REWARD_IDLE


def test_p_drink_receive_confirmation_is_classified_as_p_drink_selected(monkeypatch):
    results = _PhaseResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_TARGET,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.P_DRINK,
        ProducerLabels.CONFIRM_BUTTON,
    })
    ctx = ProduceContext()
    ctx.last_stable_position = GameplayPosition.P_DRINK_SELECTED

    monkeypatch.setattr(
        ui_module,
        "collect_frame_text",
        lambda _results: "活動支給 ホットコーヒー 今やる 受け取る",
    )

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)
    position = ui_module.classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == GameplayPhase.P_DRINK
    assert position == GameplayPosition.P_DRINK_SELECTED


def test_loading_screen_is_classified_as_loading(monkeypatch):
    results = _PhaseResultsStub(set())
    ctx = ProduceContext()

    monkeypatch.setattr(ui_module, "ocr_text", lambda _crop: "NOW LOADING...")

    phase = ui_module.classify_gameplay_phase(results, ctx=ctx)

    assert phase == GameplayPhase.LOADING
