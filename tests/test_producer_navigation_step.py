import cv2
import numpy as np
import pytest

from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.game.text.produce_text import ProduceText
from src.core.exceptions.TaskException import TaskUserMessage
from types import SimpleNamespace

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.steps.entry.navigate_to_produce import (
    NavigateToProduceStep,
    _find_gameplay_menu_button,
    _find_gameplay_retire_menu_entry,
    _has_gameplay_retire_menu,
    _looks_like_resume_lesson_p_drink_showcase,
    _looks_like_resume_lesson_summary_showcase,
    resume_resumable_produce,
)
from src.entity.Yolo import Yolo_Box


def test_navigate_to_produce_dismisses_residual_modal_before_going_home(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()
    modal = SimpleNamespace(modal_title="ボイス再生確認")

    class _ResultsStub:
        @staticmethod
        def exists_label(_label):
            return False

    app = SimpleNamespace(
        latest_results=_ResultsStub(),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: modal if not events else None,
            go_home=lambda: events.append(("go_home",)),
            update_current_location=lambda: "MAIN_MENU__HOME",
            wait_loading=lambda: events.append(("wait_loading",)),
            wait_for_label=lambda label, timeout=0: events.append(("wait_label", label)) or True,
            click_on_label=lambda label: events.append(("click_label", label)),
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.click_modal_action_with_retry",
        lambda _app, current_modal, **kwargs: events.append(("dismiss_modal", current_modal.modal_title)) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, _text, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: len([event for event in events if event[0] == "click_label"]) > 0,
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_enter_from_home_entry",
        staticmethod(lambda _app, _ctx: events.append(("enter_from_home_entry",)) or True),
    )

    assert step.execute(app, SimpleNamespace(resume_interrupted=False)) is True
    assert events[0] == ("dismiss_modal", "ボイス再生確認")
    assert ("enter_from_home_entry",) in events


def test_navigate_to_produce_confirms_retire_residual_modal(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()
    modal = SimpleNamespace(modal_title=ProduceText.PRODUCE_RETIRE_CONFIRM)

    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: modal if not events else None,
        ),
        # 模拟非主页场景，latest_results 不包含主页培育按钮
        latest_results=SimpleNamespace(
            exists_label=lambda label: False,
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.click_modal_action_with_retry",
        lambda _app, current_modal, **kwargs: events.append(
            ("dismiss_modal", current_modal.modal_title, kwargs.get("prefer_confirm"))
        ) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, _text, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )

    step._dismiss_residual_modal(app)

    assert events == [("dismiss_modal", ProduceText.PRODUCE_RETIRE_CONFIRM, True)]


def test_navigate_to_produce_closes_menu_overlay_before_generic_modal_action(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()
    modal = SimpleNamespace(modal_title="メニュー")
    close_button = SimpleNamespace(text=ButtonText.CLOSE)

    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: modal if not events else None,
            click_element_and_wait_trigger=lambda button, timeout=0: events.append(
                ("click_element", getattr(button, "text", ""), timeout)
            ) or True,
        ),
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, text, **kwargs: close_button if text == ButtonText.CLOSE else None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.click_modal_action_with_retry",
        lambda *_args, **_kwargs: events.append(("generic_modal_action",)) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )

    step._dismiss_residual_modal(app)

    assert ("click_element", ButtonText.CLOSE, 3.0) in events
    assert ("generic_modal_action",) not in events


def test_navigate_to_produce_recovers_from_start_game_before_resume_probe(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()
    location_state = {"count": 0}

    def _update_current_location():
        location_state["count"] += 1
        return "START_GAME" if location_state["count"] == 1 else "MAIN_MENU__HOME"

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            update_current_location=_update_current_location,
        ),
    )
    ctx = SimpleNamespace(
        resume_interrupted=True,
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_dismiss_residual_modal",
        lambda _self, _app: events.append(("dismiss_residual_modal",)),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_recover_from_start_game",
        staticmethod(lambda _app: events.append(("recover_from_start_game",))),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_wait_for_main_home_signal",
        classmethod(lambda _cls, _app, **_kwargs: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_resume_interrupted",
        staticmethod(lambda _app, _ctx: events.append(("try_resume_interrupted",)) or False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_enter_from_home_entry",
        staticmethod(lambda _app, _ctx: events.append(("enter_from_home_entry",)) or True),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(lambda _app, _ctx: events.append(("try_detect_existing_gameplay",)) or False),
    )

    assert step.execute(app, ctx) is True
    assert ("recover_from_start_game",) in events
    assert ("enter_from_home_entry",) in events
    assert ("try_detect_existing_gameplay",) not in events


def test_navigate_to_produce_retires_resumable_produce_then_reenters(monkeypatch):
    events: list[tuple] = []
    state = {"screen": "home"}
    step = NavigateToProduceStep()
    resume_modal = SimpleNamespace(modal_title=ProduceText.PRODUCE_RESUME)
    confirm_modal = SimpleNamespace(modal_title="リタイア確認")

    class _ResultsStub:
        @staticmethod
        def exists_label(_label):
            return False

    def _try_get_modal(no_body=True, require_header=False):
        if state["screen"] == "resume_modal":
            return resume_modal
        if state["screen"] == "retire_confirm":
            return confirm_modal
        return None

    def _click_on_label(label):
        events.append(("click_label", label))
        if label == BaseUILabels.HOME_PRODUCE_BTN:
            if state["screen"] == "home":
                state["screen"] = "resume_modal"
            elif state["screen"] == "home_after_retire":
                state["screen"] = "scenario"

    def _wait_for_label(label, timeout=0):
        events.append(("wait_label", label, timeout))
        return label == BaseUILabels.HOME_PRODUCE_BTN

    def _click_element_and_wait_trigger(button, timeout=0):
        events.append(("click_element", getattr(button, "text", ""), timeout))
        if getattr(button, "text", "") == ButtonText.RETIRE:
            state["screen"] = "retire_confirm"
            return True
        return False

    app = SimpleNamespace(
        latest_results=_ResultsStub(),
        game_utils=SimpleNamespace(
            try_get_modal=_try_get_modal,
            go_home=lambda: events.append(("go_home",)),
            wait_loading=lambda: events.append(("wait_loading", state["screen"])),
            wait_for_label=_wait_for_label,
            click_on_label=_click_on_label,
            click_element_and_wait_trigger=_click_element_and_wait_trigger,
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, text, **kwargs: SimpleNamespace(text=text) if (
            state["screen"] == "resume_modal"
            and text in (ButtonText.RETIRE, ButtonText.CANCEL, ButtonText.PRODUCE_RESUME)
        ) else None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.click_modal_action_with_retry",
        lambda _app, current_modal, **kwargs: (
            events.append(("confirm_modal", current_modal.modal_title)),
            state.update({"screen": "home_after_retire"}),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: state["screen"] == "scenario",
    )

    assert step.execute(app, SimpleNamespace()) is True
    assert ("click_element", ButtonText.RETIRE, 3.0) in events
    assert ("confirm_modal", "リタイア確認") in events
    assert events.count(("click_label", BaseUILabels.HOME_PRODUCE_BTN)) == 2


def test_navigate_to_produce_retires_active_gameplay_after_go_home_failure(monkeypatch):
    events: list[tuple] = []
    state = {"go_home_calls": 0}
    step = NavigateToProduceStep()

    class _ResultsStub:
        @staticmethod
        def exists_label(_label):
            return False

    def _go_home():
        state["go_home_calls"] += 1
        events.append(("go_home", state["go_home_calls"]))
        if state["go_home_calls"] == 1:
            raise RuntimeError("Going home failed")

    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            go_home=_go_home,
            wait_loading=lambda: events.append(("wait_loading",)),
            wait_for_label=lambda label, timeout=0: events.append(("wait_label", label, timeout)) or True,
            click_on_label=lambda label: events.append(("click_label", label)),
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, _text, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._retire_active_gameplay_produce",
        lambda _app: events.append(("retire_active_gameplay",)) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.go_back_in_gameplay",
        lambda _app: False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: len([event for event in events if event[0] == "click_label"]) > 0,
    )

    assert step.execute(app, SimpleNamespace()) is True
    assert ("retire_active_gameplay",) in events
    assert events.count(("go_home", 1)) == 1
    assert events.count(("go_home", 2)) == 1
    assert ("click_label", BaseUILabels.HOME_PRODUCE_BTN) in events


def test_navigate_to_produce_tries_gameplay_back_before_retire_chain(monkeypatch):
    events: list[tuple] = []
    state = {"go_home_calls": 0}
    step = NavigateToProduceStep()

    class _ResultsStub:
        @staticmethod
        def exists_label(_label):
            return False

    def _go_home():
        state["go_home_calls"] += 1
        events.append(("go_home", state["go_home_calls"]))
        if state["go_home_calls"] == 1:
            raise RuntimeError("Going home failed")

    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            go_home=_go_home,
            wait_loading=lambda: events.append(("wait_loading",)),
            wait_for_label=lambda label, timeout=0: events.append(("wait_label", label, timeout)) or True,
            click_on_label=lambda label: events.append(("click_label", label)),
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, _text, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.go_back_in_gameplay",
        lambda _app: events.append(("go_back_in_gameplay",)) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._retire_active_gameplay_produce",
        lambda _app: events.append(("retire_active_gameplay",)) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: len([event for event in events if event[0] == "click_label"]) > 0,
    )

    assert step.execute(app, SimpleNamespace()) is True
    assert ("go_back_in_gameplay",) in events
    assert ("retire_active_gameplay",) not in events
    assert events.count(("go_home", 1)) == 1
    assert events.count(("go_home", 2)) == 1


def test_detect_gameplay_menu_button_from_capture():
    frame = cv2.imread("tests/produce_gameplay_captures/current_screen.png")

    target = _find_gameplay_menu_button(frame)

    assert target is not None
    assert 900 <= target.cx <= 1020
    assert target.cy >= 2140


def test_find_gameplay_retire_menu_entry_prefers_retire_candidate(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    suspend_box = Yolo_Box(120, 1920, 220, 1980, "ocr:中断", frame[1920:1980, 120:220].copy())
    setting_box = Yolo_Box(860, 1920, 950, 1980, "ocr:設定", frame[1920:1980, 860:950].copy())
    retire_box = Yolo_Box(580, 2050, 760, 2100, "ocr:図リタイア", frame[2050:2100, 580:760].copy())
    ranking_box = Yolo_Box(800, 2050, 980, 2100, "ocr:ランキング", frame[2050:2100, 800:980].copy())

    def _fake_collect(_frame, *, left_ratio, top_ratio, right_ratio, bottom_ratio):
        assert top_ratio == 0.78
        assert bottom_ratio == 0.92
        if left_ratio >= 0.5:
            return [
                ("設定", setting_box, 1.0),
                ("図リタイア", retire_box, 0.8),
                ("ランキング", ranking_box, 0.9),
            ]
        return [
            ("中断", suspend_box, 1.0),
            ("設定", setting_box, 1.0),
            ("図リタイア", retire_box, 0.8),
            ("ランキング", ranking_box, 0.9),
        ]

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._collect_ocr_candidates",
        _fake_collect,
    )

    assert _has_gameplay_retire_menu(frame) is True
    assert _find_gameplay_retire_menu_entry(frame) == retire_box


def test_resume_resumable_produce_clicks_resume_button(monkeypatch):
    events: list[tuple] = []
    state = {"modal_checks": 0}
    resume_modal = SimpleNamespace(modal_title=ProduceText.PRODUCE_RESUME)

    def _try_get_modal(no_body=True):
        state["modal_checks"] += 1
        return resume_modal if state["modal_checks"] >= 2 else None

    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            try_get_modal=_try_get_modal,
            click_element_and_wait_trigger=lambda button, timeout=0: events.append(
                ("click_element", getattr(button, "text", ""), timeout)
            ) or True,
            wait_loading=lambda: events.append(("wait_loading",)),
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, text, **kwargs: SimpleNamespace(text=text) if text == ButtonText.PRODUCE_RESUME else None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )

    assert resume_resumable_produce(app, timeout=2.0) is True
    assert ("click_element", ButtonText.PRODUCE_RESUME, 3.0) in events
    assert ("wait_loading",) in events
    assert ("wait_frame_stable", 3.0) in events


def test_navigate_to_produce_confirms_destroying_production_data_modal_in_resume_mode(monkeypatch):
    events: list[tuple] = []
    state = {"screen": "home"}
    step = NavigateToProduceStep()
    destroy_modal = SimpleNamespace(
        modal_title=ModalText.TITLE.DESTROYING_PRODUCTION_DATA,
        confirm_button=SimpleNamespace(name="confirm"),
        cancel_button=SimpleNamespace(name="cancel"),
    )

    class _ResultsStub:
        @staticmethod
        def exists_label(_label):
            return False

    def _try_get_modal(no_body=True, require_header=False):
        if state["screen"] == "destroy_modal":
            return destroy_modal
        return None

    def _click_on_label(label):
        events.append(("click_label", label))
        if label == BaseUILabels.HOME_PRODUCE_BTN:
            state["screen"] = "destroy_modal"

    app = SimpleNamespace(
        latest_results=_ResultsStub(),
        game_utils=SimpleNamespace(
            try_get_modal=_try_get_modal,
            go_home=lambda: events.append(("go_home",)),
            wait_loading=lambda: events.append(("wait_loading", state["screen"])),
            wait_for_label=lambda label, timeout=0: events.append(("wait_label", label, timeout)) or True,
            click_on_label=_click_on_label,
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, _text, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.click_modal_action_with_retry",
        lambda _app, current_modal, **kwargs: (
            events.append(("confirm_modal", current_modal.modal_title, kwargs.get("prefer_confirm"))),
            state.update({"screen": "scenario"}),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: state["screen"] == "scenario",
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(lambda _app, _ctx: False),
    )

    assert step.execute(app, SimpleNamespace(resume_interrupted=True)) is True
    assert ("confirm_modal", ModalText.TITLE.DESTROYING_PRODUCTION_DATA, True) in events
    assert ("wait_loading", "scenario") in events
    assert ("wait_frame_stable", 3.0) in events


def test_navigate_to_produce_cancels_destroying_production_data_modal_when_disallowed(monkeypatch):
    events: list[tuple] = []
    state = {"screen": "home"}
    step = NavigateToProduceStep()
    destroy_modal = SimpleNamespace(
        modal_title=ModalText.TITLE.DESTROYING_PRODUCTION_DATA,
        confirm_button=SimpleNamespace(name="confirm"),
        cancel_button=SimpleNamespace(name="cancel"),
    )

    class _ResultsStub:
        @staticmethod
        def exists_label(_label):
            return False

    def _try_get_modal(no_body=True, require_header=False):
        if state["screen"] == "destroy_modal":
            return destroy_modal
        return None

    def _click_on_label(label):
        events.append(("click_label", label))
        if label == BaseUILabels.HOME_PRODUCE_BTN:
            state["screen"] = "destroy_modal"

    app = SimpleNamespace(
        latest_results=_ResultsStub(),
        game_utils=SimpleNamespace(
            try_get_modal=_try_get_modal,
            go_home=lambda: events.append(("go_home", state["screen"])) or state.update({"screen": "home"}),
            wait_loading=lambda: events.append(("wait_loading", state["screen"])),
            wait_for_label=lambda label, timeout=0: events.append(("wait_label", label, timeout)) or True,
            click_on_label=_click_on_label,
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.find_button",
        lambda _app, _text, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.click_modal_action_with_retry",
        lambda _app, current_modal, **kwargs: (
            events.append(("cancel_modal", current_modal.modal_title, kwargs.get("prefer_confirm"))),
            state.update({"screen": "home"}),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: state["screen"] == "scenario",
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(lambda _app, _ctx: False),
    )

    with pytest.raises(TaskUserMessage, match="已取消并返回主页"):
        step.execute(
            app,
            SimpleNamespace(
                resume_interrupted=True,
                allow_destroy_production_data=False,
            ),
        )

    assert ("cancel_modal", ModalText.TITLE.DESTROYING_PRODUCTION_DATA, False) in events
    assert ("go_home", "home") in events


def test_try_detect_existing_gameplay_does_not_accept_transition_position(monkeypatch):
    events: list[tuple] = []
    app = SimpleNamespace(
        yolo_engine=SimpleNamespace(model_type="BASE_UI"),
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
        switch_yolo_model=lambda model_type, settle_seconds=0.0: events.append(
            ("switch_yolo_model", model_type, settle_seconds)
        ),
    )
    ctx = SimpleNamespace(
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.detect_gameplay_state",
        lambda _app, _ctx: (GameplayPhase.UNKNOWN, GameplayPosition.TRANSITION_EMPTY),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )

    assert NavigateToProduceStep._try_detect_existing_gameplay(app, ctx) is False
    assert ctx.resumed_from_interrupt is False
    assert ctx.resume_pipeline_step == ""
    assert events[-1] == ("switch_yolo_model", "BASE_UI", 1.0)


def test_try_detect_existing_gameplay_accepts_result_position(monkeypatch):
    app = SimpleNamespace(
        yolo_engine=SimpleNamespace(model_type="BASE_UI"),
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
        switch_yolo_model=lambda *_args, **_kwargs: True,
    )
    ctx = SimpleNamespace(
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.detect_gameplay_state",
        lambda _app, _ctx: (GameplayPhase.UNKNOWN, GameplayPosition.RESULT_MEMORY_PAGE),
    )

    assert NavigateToProduceStep._try_detect_existing_gameplay(app, ctx) is True
    assert ctx.resumed_from_interrupt is True
    assert ctx.resume_pipeline_step == "produce_gameplay_loop"


def test_resume_lesson_summary_showcase_ocr_fallback_accepts_zero_detection(monkeypatch):
    frame = np.zeros((2400, 1080, 3), dtype=np.uint8)
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.ocr_text",
        lambda _image: "Vo.が43上昇した!",
    )

    assert _looks_like_resume_lesson_summary_showcase(app) is True


def test_resume_lesson_summary_showcase_visual_bubble_fallback_accepts_param_only(monkeypatch):
    frame = np.full((2400, 1080, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (80, 1830), (1000, 2160), (250, 250, 250), thickness=-1)
    boxes = [
        SimpleNamespace(label=ProducerLabels.PARAM_VOCAL),
        SimpleNamespace(label=ProducerLabels.PARAM_DANCE),
    ]
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=SimpleNamespace(
            exists_label=lambda label: label == ProducerLabels.PARAM_VOCAL,
            boxes=boxes,
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.ocr_text",
        lambda _image: "",
    )

    assert _looks_like_resume_lesson_summary_showcase(app) is True


def test_try_detect_existing_gameplay_accepts_lesson_summary_ocr_fallback(monkeypatch):
    app = SimpleNamespace(
        yolo_engine=SimpleNamespace(model_type="BASE_UI"),
        latest_frame=np.zeros((2400, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
        switch_yolo_model=lambda *_args, **_kwargs: True,
    )
    ctx = SimpleNamespace(
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.detect_gameplay_state",
        lambda _app, _ctx: (GameplayPhase.UNKNOWN, GameplayPosition.UNKNOWN),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._looks_like_resume_lesson_summary_showcase",
        lambda _app: True,
    )

    assert NavigateToProduceStep._try_detect_existing_gameplay(app, ctx) is True
    assert ctx.resumed_from_interrupt is True
    assert ctx.resume_pipeline_step == "produce_gameplay_loop"


def test_resume_lesson_p_drink_showcase_ocr_fallback_accepts_zero_detection(monkeypatch):
    frame = np.full((2400, 1080, 3), 120, dtype=np.uint8)
    frame[int(frame.shape[0] * 0.55):int(frame.shape[0] * 0.92), int(frame.shape[1] * 0.06):int(frame.shape[1] * 0.94)] = 235
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.ocr_text",
        lambda _image: "初星水 獲得 パラメータ+10",
    )

    assert _looks_like_resume_lesson_p_drink_showcase(app) is True


def test_try_detect_existing_gameplay_accepts_lesson_p_drink_showcase_fallback(monkeypatch):
    app = SimpleNamespace(
        yolo_engine=SimpleNamespace(model_type="BASE_UI"),
        latest_frame=np.zeros((2400, 1080, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
        switch_yolo_model=lambda *_args, **_kwargs: True,
    )
    ctx = SimpleNamespace(
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.detect_gameplay_state",
        lambda _app, _ctx: (GameplayPhase.UNKNOWN, GameplayPosition.UNKNOWN),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._looks_like_resume_lesson_summary_showcase",
        lambda _app: False,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._looks_like_resume_lesson_p_drink_showcase",
        lambda _app: True,
    )

    assert NavigateToProduceStep._try_detect_existing_gameplay(app, ctx) is True
    assert ctx.resumed_from_interrupt is True
    assert ctx.resume_pipeline_step == "produce_gameplay_loop"


def test_navigate_to_produce_clicks_home_entry_in_resume_mode_when_on_main_home(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            exists_label=lambda _label: False,
        ),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            update_current_location=lambda: "MAIN_MENU__HOME",
        ),
    )
    ctx = SimpleNamespace(
        resume_interrupted=True,
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_resume_interrupted",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_dismiss_residual_modal",
        lambda _self, _app: events.append(("dismiss_residual_modal",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.open_produce_entry_from_home",
        lambda _app, **_kwargs: events.append(("open_produce_entry_from_home",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_confirm_destroying_production_data_modal",
        staticmethod(lambda _app, _ctx: False),
    )
    app.switch_yolo_model = lambda model_type, settle_seconds=0.0, **_kwargs: events.append(
        ("switch_yolo_model", model_type, settle_seconds)
    )

    assert step.execute(app, ctx) is True
    assert ("switch_yolo_model", "BASE_UI", 1.0) in events
    assert ("open_produce_entry_from_home",) in events
    assert ("wait_frame_stable", 2.0) in events


def test_navigate_to_produce_uses_home_ui_fallback_when_location_misclassified(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            exists_label=lambda label: label in {
                BaseUILabels.TAB_HOME,
                BaseUILabels.HOME_PRODUCE_BTN,
            },
        ),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            update_current_location=lambda: "PASS_REWARD",
        ),
    )
    ctx = SimpleNamespace(
        resume_interrupted=True,
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_resume_interrupted",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_confirm_destroying_production_data_modal",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_dismiss_residual_modal",
        lambda _self, _app: events.append(("dismiss_residual_modal",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.open_produce_entry_from_home",
        lambda _app, **_kwargs: events.append(("open_produce_entry_from_home",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    app.switch_yolo_model = lambda model_type, settle_seconds=0.0, **_kwargs: events.append(
        ("switch_yolo_model", model_type, settle_seconds)
    )

    assert step.execute(app, ctx) is True
    assert ("switch_yolo_model", "BASE_UI", 1.0) in events
    assert ("open_produce_entry_from_home",) in events


def test_navigate_to_produce_rechecks_home_after_resume_probe_cleanup(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()
    state = {"home": False}

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            exists_label=lambda label: state["home"] and label in {
                BaseUILabels.TAB_HOME,
                BaseUILabels.HOME_PRODUCE_BTN,
            },
            filter_by_label=lambda _label: [],
        ),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            update_current_location=lambda: "UNKNOWN",
        ),
    )
    ctx = SimpleNamespace(
        resume_interrupted=True,
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: False,
    )
    probe_state = {"count": 0}

    def _fake_try_detect(_app, _ctx):
        probe_state["count"] += 1
        return False

    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(_fake_try_detect),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_resume_interrupted",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_confirm_destroying_production_data_modal",
        staticmethod(lambda _app, _ctx: False),
    )

    def _fake_dismiss(_self, _app):
        events.append(("dismiss_residual_modal", probe_state["count"]))
        if probe_state["count"] >= 1:
            state["home"] = True

    monkeypatch.setattr(
        NavigateToProduceStep,
        "_dismiss_residual_modal",
        _fake_dismiss,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.open_produce_entry_from_home",
        lambda _app, **_kwargs: events.append(("open_produce_entry_from_home",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    app.switch_yolo_model = lambda model_type, settle_seconds=0.0, **_kwargs: events.append(
        ("switch_yolo_model", model_type, settle_seconds)
    )

    assert step.execute(app, ctx) is True
    assert probe_state["count"] >= 1
    assert ("switch_yolo_model", "BASE_UI", 1.0) in events
    assert ("open_produce_entry_from_home",) in events


def test_navigate_to_produce_home_entry_reuses_resume_modal_after_click(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            exists_label=lambda label: label in {
                BaseUILabels.TAB_HOME,
                BaseUILabels.HOME_PRODUCE_BTN,
            },
        ),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            update_current_location=lambda: "PASS_REWARD",
        ),
        switch_yolo_model=lambda model_type, settle_seconds=0.0, **_kwargs: events.append(
            ("switch_yolo_model", model_type, settle_seconds)
        ),
    )
    ctx = SimpleNamespace(
        resume_interrupted=True,
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_dismiss_residual_modal",
        lambda _self, _app: events.append(("dismiss_residual_modal",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.open_produce_entry_from_home",
        lambda _app, **_kwargs: events.append(("open_produce_entry_from_home",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )

    def _fake_try_resume(_app, _ctx):
        events.append(("try_resume_interrupted",))
        return len([event for event in events if event[0] == "open_produce_entry_from_home"]) > 0

    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_resume_interrupted",
        staticmethod(_fake_try_resume),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_confirm_destroying_production_data_modal",
        staticmethod(lambda _app, _ctx: events.append(("confirm_destroy_modal",)) or False),
    )

    assert step.execute(app, ctx) is True
    resume_probe_indexes = [
        index for index, event in enumerate(events) if event == ("try_resume_interrupted",)
    ]
    assert len(resume_probe_indexes) >= 2
    assert events.index(("open_produce_entry_from_home",)) < resume_probe_indexes[-1]
    assert ("switch_yolo_model", "BASE_UI", 1.0) in events


def test_navigate_to_produce_resume_prefers_home_entry_before_gameplay_probe(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            exists_label=lambda label: label in {
                BaseUILabels.TAB_HOME,
                BaseUILabels.HOME_PRODUCE_BTN,
            },
        ),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            update_current_location=lambda: "PASS_REWARD",
        ),
        switch_yolo_model=lambda model_type, settle_seconds=0.0, **_kwargs: events.append(
            ("switch_yolo_model", model_type, settle_seconds)
        ),
    )
    ctx = SimpleNamespace(
        resume_interrupted=True,
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_dismiss_residual_modal",
        lambda _self, _app: events.append(("dismiss_residual_modal",)),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_resume_interrupted",
        staticmethod(lambda _app, _ctx: events.append(("try_resume_interrupted",)) or False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_confirm_destroying_production_data_modal",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(lambda _app, _ctx: events.append(("try_detect_existing_gameplay",)) or False),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.open_produce_entry_from_home",
        lambda _app, **_kwargs: events.append(("open_produce_entry_from_home",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )

    assert step.execute(app, ctx) is True
    assert ("open_produce_entry_from_home",) in events
    assert ("try_detect_existing_gameplay",) not in events


def test_navigate_to_produce_waits_briefly_for_home_signal_before_gameplay_probe(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()
    state = {"home_checks": 0}

    def _exists_label(label):
        if label in {BaseUILabels.TAB_HOME, BaseUILabels.HOME_PRODUCE_BTN}:
            state["home_checks"] += 1
            return state["home_checks"] >= 3
        return False

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            exists_label=_exists_label,
        ),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            update_current_location=lambda: "PASS_REWARD",
        ),
        switch_yolo_model=lambda model_type, settle_seconds=0.0, **_kwargs: events.append(
            ("switch_yolo_model", model_type, settle_seconds)
        ),
    )
    ctx = SimpleNamespace(
        resume_interrupted=True,
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_dismiss_residual_modal",
        lambda _self, _app: events.append(("dismiss_residual_modal",)),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_resume_interrupted",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_confirm_destroying_production_data_modal",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_detect_existing_gameplay",
        staticmethod(lambda _app, _ctx: events.append(("try_detect_existing_gameplay",)) or False),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.open_produce_entry_from_home",
        lambda _app, **_kwargs: events.append(("open_produce_entry_from_home",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.sleep",
        lambda _seconds: None,
    )

    assert step.execute(app, ctx) is True
    assert ("open_produce_entry_from_home",) in events
    assert ("try_detect_existing_gameplay",) not in events


def test_navigate_to_produce_prefers_home_signal_before_location_ocr(monkeypatch):
    events: list[tuple] = []
    step = NavigateToProduceStep()

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            exists_label=lambda label: label in {
                BaseUILabels.TAB_HOME,
                BaseUILabels.HOME_GIFT_BTN,
            },
        ),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: None,
            update_current_location=lambda: (_ for _ in ()).throw(AssertionError("不应先调用 update_current_location")),
        ),
        switch_yolo_model=lambda model_type, settle_seconds=0.0, **_kwargs: events.append(
            ("switch_yolo_model", model_type, settle_seconds)
        ),
    )
    ctx = SimpleNamespace(
        resume_interrupted=True,
        resumed_from_interrupt=False,
        resume_pipeline_step="",
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce._is_on_scenario_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_dismiss_residual_modal",
        lambda _self, _app: events.append(("dismiss_residual_modal",)),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_try_resume_interrupted",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        NavigateToProduceStep,
        "_confirm_destroying_production_data_modal",
        staticmethod(lambda _app, _ctx: False),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.open_produce_entry_from_home",
        lambda _app, **_kwargs: events.append(("open_produce_entry_from_home",)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.navigate_to_produce.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )

    assert step.execute(app, ctx) is True
    assert ("open_produce_entry_from_home",) in events
