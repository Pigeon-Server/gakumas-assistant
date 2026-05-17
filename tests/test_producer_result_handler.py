from types import SimpleNamespace

import numpy as np

from src.core.inference.ocr_engine import OCR_Result, OCR_ResultList
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import handler_base as handler_base_module


class _DeviceStub:
    def __init__(self):
        self.clicks = []

    def click(self, x, y, label=""):
        self.clicks.append((int(x), int(y), str(label)))

    def click_element(self, _element):
        raise AssertionError("该用例不应走 click_element 分支")


class _OCRStub:
    def __init__(self, results):
        self._results = results

    def ocr(self, _frame):
        return OCR_ResultList(self._results)


def test_result_exam_failure_choose_next_when_retry_remaining_zero():
    ctx = ProduceContext()
    assert handler_base_module.ResultHandler._should_choose_next_on_exam_failure(ctx, "あと0回")


def test_result_exam_failure_choose_next_when_retry_stuck():
    ctx = ProduceContext()
    ctx.handler_state[handler_base_module._RESULT_EXAM_FAILURE_RETRY_KEY] = 3
    assert handler_base_module.ResultHandler._should_choose_next_on_exam_failure(ctx, "あと2回")


def test_result_exam_failure_ocr_retry_clicks_retry_button(monkeypatch):
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=SimpleNamespace(frame=frame),
        latest_frame=frame,
    )
    ctx = ProduceContext()
    ctx.handler_state[handler_base_module._RESULT_EXAM_FAILURE_CENTER_TAP_LIMIT_KEY] = 0

    monkeypatch.setattr("src.core.tasks.producer_challenge.ui.find_button", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.core.tasks.producer_challenge.ui.collect_frame_text", lambda *_args, **_kwargs: "不合格 あと2回")
    monkeypatch.setattr(handler_base_module.DebugTools, "add_box", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        handler_base_module,
        "_RESULT_SCREEN_OCR",
        _OCRStub(
            [
                OCR_Result(100, 1700, 220, 80, "再挑戦", 0.95),
                OCR_Result(760, 1700, 180, 80, "次へ", 0.93),
            ]
        ),
    )

    assert handler_base_module.ResultHandler._handle_result_exam_failure(app, ctx) is True
    assert app.device.clicks
    assert app.device.clicks[-1][2] == "result-exam-failure-retry"
    assert ctx.handler_state[handler_base_module._RESULT_EXAM_FAILURE_RETRY_KEY] == 1


def test_result_exam_failure_ocr_next_resets_retry_counter(monkeypatch):
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=SimpleNamespace(frame=frame),
        latest_frame=frame,
    )
    ctx = ProduceContext()
    ctx.handler_state[handler_base_module._RESULT_EXAM_FAILURE_RETRY_KEY] = 4
    ctx.handler_state[handler_base_module._RESULT_EXAM_FAILURE_CENTER_TAP_LIMIT_KEY] = 0

    monkeypatch.setattr("src.core.tasks.producer_challenge.ui.find_button", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.core.tasks.producer_challenge.ui.collect_frame_text", lambda *_args, **_kwargs: "不合格 あと0回")
    monkeypatch.setattr(handler_base_module.DebugTools, "add_box", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        handler_base_module,
        "_RESULT_SCREEN_OCR",
        _OCRStub(
            [
                OCR_Result(760, 1700, 180, 80, "次へ", 0.93),
            ]
        ),
    )

    assert handler_base_module.ResultHandler._handle_result_exam_failure(app, ctx) is True
    assert app.device.clicks
    assert app.device.clicks[-1][2] == "result-exam-failure-next"
    assert ctx.handler_state[handler_base_module._RESULT_EXAM_FAILURE_RETRY_KEY] == 0


def test_result_exam_failure_prefers_center_tap_first(monkeypatch):
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    frame[420:1550, 80:1000] = (245, 245, 245)
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=SimpleNamespace(frame=frame),
        latest_frame=frame,
    )
    ctx = ProduceContext()
    ctx.handler_state[handler_base_module._RESULT_EXAM_FAILURE_CENTER_TAP_LIMIT_KEY] = 2

    monkeypatch.setattr("src.core.tasks.producer_challenge.ui.find_button", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.core.tasks.producer_challenge.ui.collect_frame_text", lambda *_args, **_kwargs: "不合格 あと2回")
    monkeypatch.setattr(handler_base_module.DebugTools, "add_box", lambda *_args, **_kwargs: None)
    def _strict_add_point(self, cx, cy, *, radius=6, color=(0, 0, 255), alpha=0.8, duration=60.0):
        return None
    monkeypatch.setattr(handler_base_module.DebugTools, "add_point", _strict_add_point)

    assert handler_base_module.ResultHandler._handle_result_exam_failure(app, ctx) is True
    assert app.device.clicks
    assert app.device.clicks[-1][2] == "result-exam-failure-outside"
    assert ctx.handler_state[handler_base_module._RESULT_EXAM_FAILURE_CENTER_TAP_KEY] == 1


def test_click_outside_result_panel_hits_outside_white_box(monkeypatch):
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    # 模拟考试失败页中央灰白面板
    frame[420:1550, 80:1000] = (245, 245, 245)
    app = SimpleNamespace(
        device=_DeviceStub(),
        latest_results=SimpleNamespace(frame=frame),
        latest_frame=frame,
    )
    monkeypatch.setattr(handler_base_module.DebugTools, "add_box", lambda *_args, **_kwargs: None)
    def _strict_add_point(self, cx, cy, *, radius=6, color=(0, 0, 255), alpha=0.8, duration=60.0):
        return None
    monkeypatch.setattr(handler_base_module.DebugTools, "add_point", _strict_add_point)

    ok = handler_base_module.ResultHandler._click_outside_result_panel(
        app,
        label="result-exam-failure-outside",
    )

    assert ok is True
    assert app.device.clicks
    click_x, click_y, click_label = app.device.clicks[-1]
    assert click_label == "result-exam-failure-outside"
    # 关键约束：必须点击灰白框外
    assert not (80 <= click_x <= 1000 and 420 <= click_y <= 1550)


def test_result_memory_page_uses_fast_unknown_retry_override():
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    app = SimpleNamespace(
        device=SimpleNamespace(click_element=lambda _element: None, click=lambda *_args, **_kwargs: None),
        latest_results=SimpleNamespace(filter_by_label=lambda _label: []),
        latest_frame=frame,
    )
    ctx = ProduceContext()

    result = handler_base_module.ResultHandler().handle(
        app,
        ctx,
        phase="result",
        position="result_memory_page",
    )

    assert result.status == "ok"
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_chain_pending_transition:result_memory_page",
        "retry_limit": handler_base_module._RESULT_CHAIN_FAST_UNKNOWN_RETRY_LIMIT,
        "retry_sleep": handler_base_module._RESULT_CHAIN_FAST_UNKNOWN_RETRY_SLEEP,
    }


def test_result_generic_page_keeps_long_unknown_retry_override():
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    app = SimpleNamespace(
        device=SimpleNamespace(click_element=lambda _element: None, click=lambda *_args, **_kwargs: None),
        latest_results=SimpleNamespace(filter_by_label=lambda _label: []),
        latest_frame=frame,
    )
    ctx = ProduceContext()

    result = handler_base_module.ResultHandler().handle(
        app,
        ctx,
        phase="result",
        position="result",
    )

    assert result.status == "ok"
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "result_midgame_transition",
        "retry_limit": 10,
        "retry_sleep": 1.0,
    }
