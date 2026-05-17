from __future__ import annotations

from types import SimpleNamespace

from src.constants.game.producer_gameplay import GameplayPhase
from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.model_type import YoloModelType
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals import HandleStartupModalsStep


class _YoloEngineStub:
    def __init__(self):
        self.loaded_models: list[str] = []

    def load_model(self, model_type: str):
        self.loaded_models.append(model_type)


class _GameUtilsStub:
    def __init__(self, modals):
        self._modals = list(modals)
        self.clicked_buttons: list[str] = []

    def try_get_modal(self, no_body=True):  # noqa: ARG002
        if self._modals:
            return self._modals.pop(0)
        return None

    def click_button(self, text, match_config=None):  # noqa: ARG002
        self.clicked_buttons.append(text)


def _build_app(modal_titles: list[str]):
    modals = [SimpleNamespace(modal_title=title) for title in modal_titles]
    app = SimpleNamespace(
        yolo_engine=_YoloEngineStub(),
        game_utils=_GameUtilsStub(modals),
        latest_results=None,
    )

    # 测试桩补齐新的公共切模型入口，避免还停留在旧的 load_model 调用方式。
    def _switch_yolo_model(model_type: str, **kwargs):  # noqa: ARG001
        app.yolo_engine.load_model(model_type)
        return True

    app.switch_yolo_model = _switch_yolo_model
    return app


def test_handle_startup_modals_directly_enters_gameplay(monkeypatch):
    app = _build_app(["ボイス再生確認", "コミュ早送り設定", "演出スキップ設定"])
    ctx = ProduceContext()
    step = HandleStartupModalsStep()
    modal_targets: list[str] = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.click_modal_action_with_retry",
        lambda _app, modal, **kwargs: modal_targets.append(modal.modal_title) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.is_final_confirm_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        HandleStartupModalsStep,
        "_wait_for_gameplay_phase",
        staticmethod(lambda _app, _ctx, timeout=30: str(GameplayPhase.DIALOGUE)),
    )

    assert step.execute(app, ctx) is True
    assert YoloModelType.PRODUCER in app.yolo_engine.loaded_models
    assert modal_targets == ["ボイス再生確認", "コミュ早送り設定", "演出スキップ設定"]
    assert ctx.gameplay_phase == str(GameplayPhase.DIALOGUE)
    assert not app.game_utils.clicked_buttons


def test_handle_startup_modals_retries_start_after_returning_final_confirm(monkeypatch):
    app = _build_app([])
    ctx = ProduceContext()
    step = HandleStartupModalsStep()

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.click_modal_action_with_retry",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        HandleStartupModalsStep,
        "_dismiss_startup_modals_with_base_ui",
        staticmethod(lambda _app, _ctx, timeout=25: 0),
    )
    # is_final_confirm_page 返回 True，触发二次点击开始
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.is_final_confirm_page",
        lambda _app: True,
    )
    monkeypatch.setattr(
        HandleStartupModalsStep,
        "_wait_for_gameplay_phase",
        staticmethod(lambda _app, _ctx, timeout=30: str(GameplayPhase.SCHEDULE)),
    )

    assert step.execute(app, ctx) is True
    assert YoloModelType.PRODUCER in app.yolo_engine.loaded_models
    assert app.game_utils.clicked_buttons == [ButtonText.PRODUCE_START]
    assert ctx.gameplay_phase == str(GameplayPhase.SCHEDULE)
    assert any(op.action == "confirm_produce_start_again" for op in ctx.operation_history)


def test_handle_startup_modals_raises_when_no_gameplay(monkeypatch):
    app = _build_app(["ボイス再生確認"])
    ctx = ProduceContext()
    step = HandleStartupModalsStep()

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.click_modal_action_with_retry",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        HandleStartupModalsStep,
        "_dismiss_startup_modals_with_base_ui",
        staticmethod(lambda _app, _ctx, timeout=25: 1),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.is_final_confirm_page",
        lambda _app: False,
    )
    monkeypatch.setattr(
        HandleStartupModalsStep,
        "_wait_for_gameplay_phase",
        staticmethod(lambda _app, _ctx, timeout=30: ""),
    )

    try:
        step.execute(app, ctx)
    except TimeoutError as exc:
        assert "gameplay" in str(exc).lower() or "首帧" in str(exc)
    else:
        raise AssertionError("expected TimeoutError")


def test_wait_for_gameplay_phase_handles_delayed_modal(monkeypatch):
    """_wait_for_gameplay_phase 在等待期间遇到弹窗应尝试确认后继续等待。"""
    app = _build_app([])
    ctx = ProduceContext()
    modal = SimpleNamespace(modal_title="ボイス再生確認")
    # 当 phase=MODAL 时 try_get_modal 被调用，第一次返回 modal
    modal_sequence = iter([modal, None])
    phase_sequence = iter([
        GameplayPhase.UNKNOWN,
        GameplayPhase.MODAL,
        GameplayPhase.UNKNOWN,
        GameplayPhase.DIALOGUE,
    ])
    confirmed_titles: list[str] = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        app.game_utils,
        "try_get_modal",
        lambda no_body=True: next(modal_sequence, None),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.detect_gameplay_phase",
        lambda _app, _ctx: next(phase_sequence),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.runtime.handle_startup_modals.click_modal_action_with_retry",
        lambda _app, modal, **kwargs: confirmed_titles.append(modal.modal_title) or True,
    )

    detected = HandleStartupModalsStep._wait_for_gameplay_phase(app, ctx, timeout=6)

    assert detected == str(GameplayPhase.DIALOGUE)
    assert confirmed_titles == ["ボイス再生確認"]
