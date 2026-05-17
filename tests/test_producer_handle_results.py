from types import SimpleNamespace

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.yolo.model_type import YoloModelType
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.steps.finalize import handle_results as handle_results_module
from src.core.tasks.producer_challenge.steps.finalize.handle_results import HandleResultsStep
from src.core.tasks.producer_challenge.steps.runtime import produce_gameplay_loop as gameplay_loop_module


class _OCRResultListStub:
    def __init__(self, items):
        self._items = list(items)

    def __bool__(self):
        return bool(self._items)

    def __iter__(self):
        return iter(self._items)

    def search(self, queries, _config):
        assert not isinstance(queries, tuple)
        if isinstance(queries, str):
            expected = {queries}
        else:
            expected = set(queries)
        return _OCRResultListStub([item for item in self._items if item.text in expected])


def test_is_result_detail_page_detects_result_detail_variants():
    labels = [
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.PC_ACTION_INFO,
        ProducerLabels.SKILL_CARD_ACTIVE,
        ProducerLabels.SKILL_CARD_INFO,
    ]

    assert HandleResultsStep._is_result_detail_page(
        "可愛い仕草+ 戻す",
        labels,
    )
    assert HandleResultsStep._is_result_detail_page(
        "可愛い仕草の強化を戻しました",
        labels,
    )
    assert not HandleResultsStep._is_result_detail_page(
        "可愛い仕草+",
        labels,
    )


def test_click_ocr_text_prefers_rightmost_match(monkeypatch):
    clicked = {}
    app = SimpleNamespace(
        device=SimpleNamespace(
            click=lambda x, y, label="": clicked.update({"x": x, "y": y, "label": label})
        )
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    matches = [
        SimpleNamespace(text=ButtonText.BACK, cx=20, cy=50),
        SimpleNamespace(text=ButtonText.BACK, cx=80, cy=45),
    ]

    monkeypatch.setattr(
        handle_results_module,
        "_RESULT_OCR",
        SimpleNamespace(ocr=lambda _frame: _OCRResultListStub(matches)),
    )

    clicked_text = HandleResultsStep._click_ocr_text(
        app,
        frame,
        (ButtonText.BACK,),
        prefer_rightmost=True,
    )

    assert clicked_text == ButtonText.BACK
    assert clicked == {"x": 80, "y": 45, "label": f"ocr:{ButtonText.BACK}"}


def test_detect_post_result_state_returns_resume(monkeypatch):
    monkeypatch.setattr(
        handle_results_module,
        "detect_gameplay_state",
        lambda _app, _ctx: ("schedule", "schedule_idle"),
    )

    app = SimpleNamespace(
        latest_results=SimpleNamespace(exists_label=lambda _label: False),
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
    )
    ctx = SimpleNamespace()

    assert HandleResultsStep._detect_post_result_state(app, ctx) == (
        "resume",
        "schedule",
        "schedule_idle",
    )


def test_detect_post_result_state_keeps_generic_modal_inside_result_chain(monkeypatch):
    monkeypatch.setattr(
        handle_results_module,
        "detect_gameplay_state",
        lambda _app, _ctx: (GameplayPhase.MODAL, GameplayPosition.GAMEPLAY_MODAL),
    )

    app = SimpleNamespace(
        latest_results=SimpleNamespace(exists_label=lambda _label: False),
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
    )

    assert HandleResultsStep._detect_post_result_state(
        app,
        SimpleNamespace(),
        frame_text="確認",
        labels=["Universal Modal Header", "Universal Confirm button"],
    ) == ("result", "", "")


def test_detect_post_result_state_allows_specific_gameplay_modal_resume(monkeypatch):
    monkeypatch.setattr(
        handle_results_module,
        "detect_gameplay_state",
        lambda _app, _ctx: (
            GameplayPhase.MODAL,
            GameplayPosition.EXAM_RETRY_CONFIRM_MODAL,
        ),
    )

    app = SimpleNamespace(
        latest_results=SimpleNamespace(exists_label=lambda _label: False),
        latest_frame=np.zeros((10, 10, 3), dtype=np.uint8),
    )

    assert HandleResultsStep._detect_post_result_state(
        app,
        SimpleNamespace(),
        frame_text="再挑戦確認",
        labels=["Universal Modal Header", "Universal Confirm button"],
    ) == ("resume", "modal", "exam_retry_confirm_modal")


def test_execute_resumes_gameplay_loop_when_result_chain_returns_to_gameplay(monkeypatch):
    class _Ctx:
        def __init__(self):
            self.gameplay_phase = ""
            self.gameplay_position = ""
            self.handler_state = {}

        def set_phase(self, phase):
            self.gameplay_phase = phase

        def set_position(self, position):
            self.gameplay_position = position

    ctx = _Ctx()
    loaded_models = []
    gameplay_runs = []
    state_sequence = iter([
        ("resume", "schedule", "schedule_idle"),
        ("home", "", ""),
    ])

    monkeypatch.setattr(
        HandleResultsStep,
        "_skip_result_screens",
        staticmethod(lambda _app, _ctx, timeout=120: None),
    )
    monkeypatch.setattr(
        HandleResultsStep,
        "_detect_post_result_state",
        staticmethod(lambda _app, _ctx: next(state_sequence)),
    )
    monkeypatch.setattr(
        gameplay_loop_module.ProduceGameplayLoopStep,
        "execute",
        lambda self, _app, _ctx: gameplay_runs.append((_ctx.gameplay_phase, _ctx.gameplay_position)) or True,
    )
    monkeypatch.setattr(
        HandleResultsStep,
        "_wait_for_home",
        staticmethod(lambda _app, timeout=40: True),
    )

    app = SimpleNamespace(
        yolo_engine=SimpleNamespace(load_model=lambda model: loaded_models.append(model)),
        game_utils=SimpleNamespace(go_home=lambda max_try=5: None),
    )
    app.switch_yolo_model = lambda model_type, **kwargs: loaded_models.append(model_type) or True

    assert HandleResultsStep().execute(app, ctx)
    assert gameplay_runs == [("schedule", "schedule_idle")]
    assert loaded_models == [
        YoloModelType.BASE_UI,
        YoloModelType.PRODUCER,
        YoloModelType.BASE_UI,
    ]
