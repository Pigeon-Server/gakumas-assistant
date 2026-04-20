from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import local
from typing import Any, Callable

import cv2
import numpy as np

from src.utils.logger import logger

_TASK_DEBUG_TRACE_MAXLEN = 200
_TASK_STEP_SNAPSHOT_MAXLEN = 80
_TASK_STEP_CAPTURE_MAX_WIDTH = 960
_TASK_STEP_CAPTURE_QUALITY = 75

_TaskStepCollector = Callable[[Any, dict], dict | None]
_task_step_snapshot_collectors: dict[str, _TaskStepCollector] = {}
_task_debug_context = local()


def _make_json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return str(value)


def register_task_step_snapshot_collector(name: str):
    """注册任务步骤快照采集器。"""
    collector_name = str(name).strip()
    if not collector_name:
        raise ValueError("collector name cannot be empty")

    def decorator(func: _TaskStepCollector):
        _task_step_snapshot_collectors[collector_name] = func
        return func

    return decorator


def bind_task_debug_context(app, task_id: str, task_name: str = ""):
    """绑定当前线程的任务上下文，供设备层记录点击步骤。"""
    _task_debug_context.app = app
    _task_debug_context.task_id = task_id
    _task_debug_context.task_name = task_name


def clear_task_debug_context():
    """清理当前线程任务上下文。"""
    for key in ("app", "task_id", "task_name"):
        if hasattr(_task_debug_context, key):
            delattr(_task_debug_context, key)


def _get_bound_task_app():
    return getattr(_task_debug_context, "app", None)


def reset_task_debug_trace(
    app,
    task_id: str | None = None,
    maxlen: int = _TASK_DEBUG_TRACE_MAXLEN,
):
    trace = deque(maxlen=maxlen)
    setattr(app, "_task_debug_trace", trace)
    setattr(app, "_task_debug_task_id", task_id)
    setattr(app, "_task_recent_step_snapshots", deque(maxlen=_TASK_STEP_SNAPSHOT_MAXLEN))
    return trace


def _encode_step_frame(frame: np.ndarray) -> bytes | None:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    target = frame
    height, width = frame.shape[:2]
    if width > _TASK_STEP_CAPTURE_MAX_WIDTH:
        ratio = _TASK_STEP_CAPTURE_MAX_WIDTH / float(width)
        target = cv2.resize(frame, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg",
        target,
        [cv2.IMWRITE_JPEG_QUALITY, _TASK_STEP_CAPTURE_QUALITY],
    )
    if not ok:
        return None
    return encoded.tobytes()


def _serialize_step_yolo(results) -> list[dict]:
    boxes = getattr(results, "boxes", None)
    raw = getattr(results, "results", None)
    if not boxes:
        return []
    serialized = []
    for idx, box in enumerate(list(boxes)[:40]):
        item = {
            "label": getattr(box, "label", ""),
            "x": int(getattr(box, "x", 0)),
            "y": int(getattr(box, "y", 0)),
            "w": int(getattr(box, "w", 0)),
            "h": int(getattr(box, "h", 0)),
            "cx": int(getattr(box, "cx", 0)),
            "cy": int(getattr(box, "cy", 0)),
        }
        scores = getattr(raw, "scores", None)
        if scores is not None and idx < len(scores):
            item["confidence"] = round(float(scores[idx]), 4)
        serialized.append(item)
    return serialized


def _capture_step_snapshot(app, entry: dict):
    snapshots = getattr(app, "_task_recent_step_snapshots", None)
    if snapshots is None or not isinstance(snapshots, deque):
        snapshots = deque(maxlen=_TASK_STEP_SNAPSHOT_MAXLEN)
        setattr(app, "_task_recent_step_snapshots", snapshots)

    snapshot = {
        "timestamp": entry["timestamp"],
        "step": entry["step"],
        "data": entry.get("data", {}),
        "captures": {},
        "images": {},
    }

    for name, collector in _task_step_snapshot_collectors.items():
        try:
            result = collector(app, entry)
        except Exception as exc:
            snapshot["captures"][name] = {
                "capture_error": f"{type(exc).__name__}: {exc}",
            }
            logger.debug(f"Task step collector '{name}' failed: {exc}")
            continue
        if not result:
            continue
        data = result.get("data")
        if data is not None:
            snapshot["captures"][name] = _make_json_safe(data)
        image_bytes = result.get("image_bytes")
        if image_bytes:
            snapshot["images"][name] = {
                "ext": str(result.get("image_ext") or "jpg"),
                "bytes": image_bytes,
            }

    snapshots.append(snapshot)
    return snapshot


def record_task_step(app, step: str, **data):
    trace = getattr(app, "_task_debug_trace", None)
    if trace is None or not isinstance(trace, deque):
        trace = reset_task_debug_trace(app)

    entry = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "step": step,
    }
    if data:
        entry["data"] = _make_json_safe(data)
    trace.append(entry)
    _capture_step_snapshot(app, entry)
    return entry


def record_task_click(x, y, label: str = "", source: str = ""):
    """记录设备点击步骤，自动挂到当前任务 trace。"""
    app = _get_bound_task_app()
    if app is None:
        return None
    payload = {
        "x": int(x),
        "y": int(y),
    }
    if label:
        payload["label"] = str(label)
    if source:
        payload["source"] = str(source)
    return record_task_step(app, "device.click", **payload)


def record_task_swipe(
    start_x,
    start_y,
    end_x,
    end_y,
    duration: float | None = None,
    source: str = "",
):
    """记录设备滑动步骤，自动挂到当前任务 trace。"""
    app = _get_bound_task_app()
    if app is None:
        return None
    payload = {
        "start_x": int(start_x),
        "start_y": int(start_y),
        "end_x": int(end_x),
        "end_y": int(end_y),
    }
    if duration is not None:
        payload["duration"] = float(duration)
    if source:
        payload["source"] = str(source)
    return record_task_step(app, "device.swipe", **payload)


def record_task_scroll(
    x,
    y,
    direction: str = "",
    delta: int | float | None = None,
    source: str = "",
):
    """记录设备滚动步骤，自动挂到当前任务 trace。"""
    app = _get_bound_task_app()
    if app is None:
        return None
    payload = {
        "x": int(x),
        "y": int(y),
    }
    if direction:
        payload["direction"] = str(direction)
    if delta is not None:
        payload["delta"] = float(delta)
    if source:
        payload["source"] = str(source)
    return record_task_step(app, "device.scroll", **payload)


def get_task_debug_trace(app) -> list[dict]:
    trace = getattr(app, "_task_debug_trace", None)
    if trace is None:
        return []
    return list(trace)


def get_task_recent_step_snapshots(app, limit: int = 5) -> list[dict]:
    snapshots = getattr(app, "_task_recent_step_snapshots", None)
    if snapshots is None:
        return []
    items = list(snapshots)
    if limit <= 0:
        return items
    return items[-limit:]


@register_task_step_snapshot_collector("frame")
def _capture_frame(app, _entry):
    frame = getattr(app, "latest_frame", None)
    if frame is None:
        return None
    encoded = _encode_step_frame(frame)
    if encoded is None:
        return {
            "data": {
                "shape": list(map(int, frame.shape)) if hasattr(frame, "shape") else None,
            }
        }
    return {
        "data": {
            "shape": list(map(int, frame.shape)) if hasattr(frame, "shape") else None,
            "format": "jpg",
            "quality": _TASK_STEP_CAPTURE_QUALITY,
        },
        "image_bytes": encoded,
        "image_ext": "jpg",
    }


@register_task_step_snapshot_collector("click")
def _capture_click(_app, entry):
    if entry.get("step") != "device.click":
        return None
    data = entry.get("data") or {}
    if "x" not in data or "y" not in data:
        return None
    return {
        "data": {
            "x": data.get("x"),
            "y": data.get("y"),
            "label": data.get("label", ""),
            "source": data.get("source", ""),
        }
    }


@register_task_step_snapshot_collector("swipe")
def _capture_swipe(_app, entry):
    if entry.get("step") != "device.swipe":
        return None
    data = entry.get("data") or {}
    if "start_x" not in data or "start_y" not in data or "end_x" not in data or "end_y" not in data:
        return None
    return {
        "data": {
            "start_x": data.get("start_x"),
            "start_y": data.get("start_y"),
            "end_x": data.get("end_x"),
            "end_y": data.get("end_y"),
            "duration": data.get("duration"),
            "source": data.get("source", ""),
        }
    }


@register_task_step_snapshot_collector("scroll")
def _capture_scroll(_app, entry):
    if entry.get("step") != "device.scroll":
        return None
    data = entry.get("data") or {}
    if "x" not in data or "y" not in data:
        return None
    return {
        "data": {
            "x": data.get("x"),
            "y": data.get("y"),
            "direction": data.get("direction", ""),
            "delta": data.get("delta"),
            "source": data.get("source", ""),
        }
    }


@register_task_step_snapshot_collector("recognition")
def _capture_recognition(app, _entry):
    results = getattr(app, "latest_results", None)
    if results is None:
        return None
    yolo_items = _serialize_step_yolo(results)
    if not yolo_items:
        return {
            "data": {
                "count": 0,
            }
        }
    return {
        "data": {
            "count": len(yolo_items),
            "detections": yolo_items,
        }
    }
