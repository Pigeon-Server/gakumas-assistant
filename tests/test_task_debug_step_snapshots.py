from types import SimpleNamespace

import numpy as np

from src.utils.task_debug_tools import (
    bind_task_debug_context,
    clear_task_debug_context,
    get_task_recent_step_snapshots,
    record_task_scroll,
    record_task_step,
    record_task_swipe,
    reset_task_debug_trace,
)


def test_record_task_step_collects_frame_click_and_recognition():
    app = SimpleNamespace(
        latest_frame=np.zeros((32, 32, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(
            boxes=[
                SimpleNamespace(
                    label="PrimaryButton",
                    x=10,
                    y=12,
                    w=20,
                    h=8,
                    cx=20,
                    cy=16,
                )
            ],
            results=SimpleNamespace(scores=[0.9876]),
        ),
    )
    reset_task_debug_trace(app, "auto_contest")

    record_task_step(app, "device.click", x=123, y=456, label="挑战按钮", source="android")

    snapshots = get_task_recent_step_snapshots(app, limit=1)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["step"] == "device.click"
    assert snapshot["captures"]["click"]["x"] == 123
    assert snapshot["captures"]["click"]["y"] == 456
    assert snapshot["captures"]["recognition"]["count"] == 1
    assert snapshot["captures"]["recognition"]["detections"][0]["label"] == "PrimaryButton"
    assert "frame" in snapshot["images"]
    assert len(snapshot["images"]["frame"]["bytes"]) > 0


def test_record_task_swipe_and_scroll_collectors():
    app = SimpleNamespace(
        latest_frame=np.zeros((16, 16, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(boxes=[], results=None),
    )
    reset_task_debug_trace(app, "auto_producer")
    bind_task_debug_context(app, "auto_producer", "自动培育")
    try:
        record_task_swipe(10, 20, 110, 220, duration=0.35, source="android.maatouch")
        record_task_scroll(320, 480, direction="down", delta=360, source="android")
    finally:
        clear_task_debug_context()

    snapshots = get_task_recent_step_snapshots(app, limit=2)
    assert len(snapshots) == 2
    swipe_snapshot, scroll_snapshot = snapshots

    assert swipe_snapshot["step"] == "device.swipe"
    assert swipe_snapshot["captures"]["swipe"]["start_x"] == 10
    assert swipe_snapshot["captures"]["swipe"]["end_y"] == 220
    assert swipe_snapshot["captures"]["swipe"]["duration"] == 0.35

    assert scroll_snapshot["step"] == "device.scroll"
    assert scroll_snapshot["captures"]["scroll"]["x"] == 320
    assert scroll_snapshot["captures"]["scroll"]["direction"] == "down"
    assert scroll_snapshot["captures"]["scroll"]["delta"] == 360.0
