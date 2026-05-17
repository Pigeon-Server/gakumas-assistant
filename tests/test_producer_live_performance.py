from types import SimpleNamespace

import numpy as np

from src.constants.game.producer_gameplay import GameplayPosition
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import live_performance as live_module


class _DeviceStub:
    def __init__(self):
        self.clicks = []

    def click(self, x, y, el_label=""):
        self.clicks.append((int(x), int(y), str(el_label or "")))


def test_detect_tap_to_start_only_ocr_bottom_center_roi(monkeypatch):
    frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
    seen_shapes: list[tuple[int, int, int]] = []

    def _fake_ocr_text(image):
        seen_shapes.append(tuple(image.shape))
        return "tap to start"

    monkeypatch.setattr(live_module, "ocr_text", _fake_ocr_text)

    assert live_module._detect_tap_to_start(frame) is True
    probe_window = live_module._build_tap_to_start_probe_window(frame)
    assert probe_window is not None
    expected_h = probe_window.y2 - probe_window.y1
    expected_w = probe_window.x2 - probe_window.x1
    assert seen_shapes == [(expected_h, expected_w, 3)]


def test_classify_live_position_skips_ocr_when_probe_disabled(monkeypatch):
    frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
    ocr_calls = {"count": 0}

    def _fake_ocr_text(_image):
        ocr_calls["count"] += 1
        return "TAP TO START"

    monkeypatch.setattr(live_module, "ocr_text", _fake_ocr_text)

    position = live_module.classify_live_position(frame, should_probe_tap=False)

    assert position == GameplayPosition.LIVE_PERFORMING
    assert ocr_calls["count"] == 0


def test_live_handler_marks_tap_confirmed_after_repeated_performing(monkeypatch):
    handler = live_module.LivePerformanceHandler()
    frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
    app = SimpleNamespace(latest_frame=frame, device=_DeviceStub(), debug_tools=None)
    ctx = ProduceContext()

    monkeypatch.setattr(
        live_module,
        "classify_live_position",
        lambda *_args, **_kwargs: GameplayPosition.LIVE_PERFORMING,
    )

    for _ in range(live_module._LIVE_TAP_EARLY_PROBE_LIMIT):
        result = handler.handle(
            app,
            ctx,
            phase="live_performance",
            position=GameplayPosition.LIVE_PERFORMING,
        )
        assert result.status == "ok"

    assert ctx.handler_state["live_wait_count"] == live_module._LIVE_TAP_EARLY_PROBE_LIMIT
    assert ctx.handler_state[live_module._LIVE_TAP_CONFIRMED_KEY] is True


def test_live_handler_resets_tap_confirmed_when_finished():
    handler = live_module.LivePerformanceHandler()
    ctx = ProduceContext()
    ctx.handler_state["live_wait_count"] = 7
    ctx.handler_state[live_module._LIVE_TAP_CONFIRMED_KEY] = True

    result = handler._handle_finished(SimpleNamespace(), ctx)

    assert result.status == "ok"
    assert ctx.handler_state["live_wait_count"] == 0
    assert ctx.handler_state[live_module._LIVE_TAP_CONFIRMED_KEY] is False
