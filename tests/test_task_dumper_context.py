import json
from types import SimpleNamespace

import numpy as np

from src.utils import task_dumper
from src.utils.task_debug_tools import (
    bind_task_debug_context,
    clear_task_debug_context,
    record_task_scroll,
    record_task_step,
    record_task_swipe,
    reset_task_debug_trace,
)


class _ButtonStub:
    def __init__(self, x, y, w, h, text="", disabled=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = (x + w) // 2
        self.cy = (y + h) // 2
        self.text = text
        self._disabled = disabled

    def is_disabled(self):
        return self._disabled


def test_dump_task_failure_includes_trace_and_button_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(task_dumper, "resolve_log_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(task_dumper, "_cleanup_old_dumps", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        task_dumper,
        "_get_button_list",
        lambda _results: [_ButtonStub(10, 20, 30, 40, text="", disabled=False)],
    )
    monkeypatch.setattr(
        task_dumper,
        "_collect_runtime_diagnostics",
        lambda _app, _task: {
            "host": {"system": "Darwin"},
            "device": {"run_mode": "Phone"},
            "yolo": {"model_type": "PRODUCER"},
            "ocr": {"runtime_backend": "rapidocr"},
            "decision": {"backend": "llm"},
        },
    )

    app = SimpleNamespace(
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(results=None, boxes=[]),
        game_status_manager=SimpleNamespace(current_location="ARENA"),
    )
    reset_task_debug_trace(app, "auto_contest")
    record_task_step(app, "auto_contest.finish.phase2.click_primary_fallback", text="")

    task = SimpleNamespace(
        id="auto_contest",
        task_name="自动每日竞技场",
        status="FAILED",
        timeout=None,
        get_timeout=lambda: None,
        get_start_time=lambda: 123,
    )

    task_dumper.dump_task_failure(app, task, TimeoutError("stuck"))

    dump_dirs = [path for path in tmp_path.joinpath("dumps").iterdir() if path.is_dir()]
    assert len(dump_dirs) == 1

    meta = json.loads(dump_dirs[0].joinpath("meta.json").read_text(encoding="utf-8"))
    assert meta["debug"]["task_trace"][-1]["step"] == "auto_contest.finish.phase2.click_primary_fallback"
    assert meta["ui"]["frame_shape"] == [8, 8, 3]
    assert meta["ui"]["buttons"][0]["text"] == ""
    assert meta["ui"]["buttons"][0]["disabled"] is False
    assert len(meta["debug"]["recent_steps"]) == 1
    step_meta_path = dump_dirs[0].joinpath(meta["debug"]["recent_steps"][0]["meta_file"])
    assert step_meta_path.exists()
    step_meta = json.loads(step_meta_path.read_text(encoding="utf-8"))
    assert step_meta["step"] == "auto_contest.finish.phase2.click_primary_fallback"
    assert "frame" in step_meta["images"]
    assert meta["runtime"]["device"]["run_mode"] == "Phone"


def test_dump_task_failure_exports_swipe_and_scroll_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(task_dumper, "resolve_log_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(task_dumper, "_cleanup_old_dumps", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_dumper, "_collect_runtime_diagnostics", lambda _app, _task: {})

    app = SimpleNamespace(
        latest_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        latest_results=SimpleNamespace(results=None, boxes=[]),
        game_status_manager=SimpleNamespace(current_location="ARENA"),
    )
    reset_task_debug_trace(app, "auto_producer")
    bind_task_debug_context(app, "auto_producer", "自动培育")
    try:
        record_task_swipe(12, 24, 120, 240, duration=0.4, source="mac_playtools")
        record_task_scroll(300, 500, direction="down", delta=420, source="mac_playtools")
    finally:
        clear_task_debug_context()

    task = SimpleNamespace(
        id="auto_producer",
        task_name="自动培育",
        status="FAILED",
        timeout=None,
        get_timeout=lambda: None,
        get_start_time=lambda: 456,
    )

    task_dumper.dump_task_failure(app, task, RuntimeError("gesture export"))
    dump_dirs = [path for path in tmp_path.joinpath("dumps").iterdir() if path.is_dir()]
    assert len(dump_dirs) == 1

    meta = json.loads(dump_dirs[0].joinpath("meta.json").read_text(encoding="utf-8"))
    assert len(meta["debug"]["recent_steps"]) == 2

    swipe_meta_path = dump_dirs[0].joinpath(meta["debug"]["recent_steps"][0]["meta_file"])
    scroll_meta_path = dump_dirs[0].joinpath(meta["debug"]["recent_steps"][1]["meta_file"])
    swipe_meta = json.loads(swipe_meta_path.read_text(encoding="utf-8"))
    scroll_meta = json.loads(scroll_meta_path.read_text(encoding="utf-8"))

    assert swipe_meta["exports"]["swipe"]["distance"] > 0
    assert scroll_meta["exports"]["scroll"]["direction"] == "down"
