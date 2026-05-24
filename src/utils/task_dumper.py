"""
任务失败时的现场转储工具。

在任务超时或异常失败时，自动保存：
- 最后一帧截图 (PNG)
- YOLO 标注帧 (PNG)
- YOLO 检测结果 (JSON)
- 任务信息 & 异常堆栈 (JSON)

转储目录: logs/dumps/<task_id>_<timestamp>/
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np

from src.utils.logger import logger
from src.utils.runtime_paths import resolve_log_path
from src.utils.task_debug_tools import get_task_debug_trace, get_task_recent_step_snapshots

if TYPE_CHECKING:
    from src.entity.Task import Task
    from src.entity.Yolo import Yolo_Results
    from src.main import AppProcessor

# 最多保留的 dump 目录数量，超出时删除最旧的
MAX_DUMPS = 50
RECENT_STEP_LIMIT = 5

_TaskDumpStepExporter = Callable[[dict], dict | None]
_task_dump_step_exporters: dict[str, _TaskDumpStepExporter] = {}


def register_task_dump_step_exporter(name: str):
    """注册任务步骤导出器。"""
    exporter_name = str(name).strip()
    if not exporter_name:
        raise ValueError("exporter name cannot be empty")

    def decorator(func: _TaskDumpStepExporter):
        _task_dump_step_exporters[exporter_name] = func
        return func

    return decorator


def _cleanup_old_dumps(dumps_root: str):
    """保留最新的 MAX_DUMPS 个 dump 目录，删除多余的。"""
    try:
        entries = []
        for name in os.listdir(dumps_root):
            full = os.path.join(dumps_root, name)
            if os.path.isdir(full):
                entries.append((os.path.getmtime(full), full))
        if len(entries) <= MAX_DUMPS:
            return
        entries.sort()
        for _, path in entries[: len(entries) - MAX_DUMPS]:
            import shutil
            shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
            logger.debug(f"TaskDumper: 操作失败: {e}")



def _serialize_yolo_results(results: "Yolo_Results") -> list:
    """将 Yolo_Results 中的检测框序列化为可 JSON 化的列表。"""
    items = []
    raw = results.results  # ONNXYoloResult
    for i, box in enumerate(results.boxes):
        entry = {
            "label": box.label,
            "x": int(box.x),
            "y": int(box.y),
            "w": int(box.w),
            "h": int(box.h),
            "cx": int(box.cx),
            "cy": int(box.cy),
        }
        # 置信度来自原始推理结果
        if raw is not None and hasattr(raw, "scores") and i < len(raw.scores):
            entry["confidence"] = round(float(raw.scores[i]), 4)
        items.append(entry)
    return items


def _get_button_list(results):
    from src.entity.Game.Components.Button import ButtonList

    return ButtonList(results)


def _serialize_button_snapshot(results: "Yolo_Results") -> tuple[list, str | None]:
    try:
        buttons = _get_button_list(results)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

    items = []
    for button in buttons:
        disabled = None
        try:
            disabled = button.is_disabled()
        except Exception as e:
            logger.debug(f"TaskDumper: 操作失败: {e}")

        items.append({
            "text": button.text,
            "x": int(button.x),
            "y": int(button.y),
            "w": int(button.w),
            "h": int(button.h),
            "cx": int(button.cx),
            "cy": int(button.cy),
            "disabled": disabled,
        })
    return items, None


def _safe_step_file_fragment(step_name: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(step_name))
    normalized = normalized.strip("_")
    if not normalized:
        return "unknown"
    return normalized[:48]


def _collect_yolo_runtime_diagnostics(app: "AppProcessor") -> dict:
    yolo_engine = getattr(app, "yolo_engine", None)
    if yolo_engine is None:
        return {}

    engine = getattr(yolo_engine, "_engine", None)
    session = getattr(engine, "_engine", None)
    providers = []
    if session is not None and hasattr(session, "get_providers"):
        providers = list(session.get_providers())

    model_dir = str(getattr(engine, "_model_dir", "") or "")
    model_file = str(getattr(engine, "_model_file", "") or "")
    model_path = ""
    if model_dir and model_file:
        model_path = str(Path(model_dir) / model_file)

    model_meta = getattr(engine, "_model_meta", None)
    export_meta = getattr(engine, "_export_meta", None)

    return {
        "model_type": str(getattr(yolo_engine, "model_type", "") or ""),
        "model_name": str(getattr(engine, "_model_name", "") or ""),
        "model_path": model_path,
        "output_layout": str(getattr(engine, "_output_layout", "") or ""),
        "execution_providers": providers,
        "model_meta": {
            "imgsz": list(getattr(model_meta, "imgsz", []) or []),
            "labels_count": len(dict(getattr(model_meta, "names", {}) or {})),
        },
        "export_meta": {
            "version": str(getattr(export_meta, "version", "") or ""),
            "end2end": bool(getattr(export_meta, "end2end", False)),
            "args": dict(getattr(export_meta, "args", {}) or {}),
        },
    }


def _collect_ocr_runtime_diagnostics(app: "AppProcessor") -> dict:
    from src.core.inference.ocr_backends.factory import (
        get_requested_ocr_backend,
        resolve_ocr_backend_candidates,
    )
    from src.core.inference.ocr_engine import OCRLoader

    requested_backend = str(get_requested_ocr_backend() or "")
    candidates = list(resolve_ocr_backend_candidates(requested_backend))
    active_backend = ""
    try:
        active_backend = str(OCRLoader().backend_name or "")
    except Exception as exc:
        active_backend = f"init_error:{type(exc).__name__}"

    return {
        "configured_backend": str(getattr(app.config_service().base, "ocr_backend", "") or ""),
        "requested_backend": requested_backend,
        "runtime_backend": active_backend,
        "candidate_backends": candidates,
    }


def _collect_device_runtime_diagnostics(app: "AppProcessor") -> dict:
    device = getattr(app, "device", None)
    base_info = {
        "run_mode": str(getattr(app.config_service().base, "run_mode", "") or ""),
        "device_class": type(device).__name__ if device is not None else "",
    }
    if device is None:
        return base_info

    diagnostics = {}
    getter = getattr(device, "get_diagnostics", None)
    if callable(getter):
        try:
            diagnostics = dict(getter() or {})
        except Exception as exc:
            diagnostics = {
                "collect_error": f"{type(exc).__name__}: {exc}",
            }
    return {
        **base_info,
        **diagnostics,
    }


def _collect_decision_runtime_diagnostics(app: "AppProcessor", task: "Task") -> dict:
    base = app.config_service().base
    backend = str(getattr(base, "producer_decision_backend", "llm") or "llm")
    info = {
        "backend": backend,
        "is_auto_producer_task": task.id == "auto_producer",
    }
    if backend == "rl_battle":
        info["rl"] = {
            "inference_base_url": str(getattr(base, "rl_inference_base_url", "") or ""),
            "inference_timeout": float(getattr(base, "rl_inference_timeout", 0.0) or 0.0),
        }
    else:
        info["llm"] = {
            "base_url": str(getattr(base, "llm_base_url", "") or ""),
            "model": str(getattr(base, "llm_model", "") or ""),
            "timeout": float(getattr(base, "llm_timeout", 0.0) or 0.0),
            "max_tokens": int(getattr(base, "llm_max_tokens", 0) or 0),
            "num_ctx": int(getattr(base, "llm_num_ctx", 0) or 0),
            "temperature": float(getattr(base, "llm_temperature", 0.0) or 0.0),
        }
    return info


def _collect_host_runtime_diagnostics() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "process": {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
        },
    }


def _collect_runtime_diagnostics(app: "AppProcessor", task: "Task") -> dict:
    return {
        "host": _collect_host_runtime_diagnostics(),
        "device": _collect_device_runtime_diagnostics(app),
        "yolo": _collect_yolo_runtime_diagnostics(app),
        "ocr": _collect_ocr_runtime_diagnostics(app),
        "decision": _collect_decision_runtime_diagnostics(app, task),
    }


def _dump_recent_steps(dump_dir: Path, app: "AppProcessor") -> list[dict]:
    recent_steps = get_task_recent_step_snapshots(app, limit=RECENT_STEP_LIMIT)
    if not recent_steps:
        return []

    steps_dir = dump_dir / "recent_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    exported_steps: list[dict] = []

    for index, snapshot in enumerate(recent_steps, start=1):
        step_name = str(snapshot.get("step") or "unknown")
        prefix = f"step_{index:02d}_{_safe_step_file_fragment(step_name)}"
        step_meta = {
            "timestamp": snapshot.get("timestamp"),
            "step": step_name,
            "data": snapshot.get("data", {}),
            "captures": snapshot.get("captures", {}),
            "images": {},
            "exports": {},
        }

        images = snapshot.get("images", {})
        for image_name, image_item in images.items():
            image_ext = str(image_item.get("ext") or "jpg")
            image_filename = f"{prefix}_{image_name}.{image_ext}"
            image_path = steps_dir / image_filename
            image_path.write_bytes(image_item.get("bytes") or b"")
            step_meta["images"][image_name] = image_filename

        for exporter_name, exporter in _task_dump_step_exporters.items():
            try:
                exported = exporter(snapshot)
                if exported:
                    step_meta["exports"][exporter_name] = exported
            except Exception as exc:
                step_meta["exports"][exporter_name] = {
                    "export_error": f"{type(exc).__name__}: {exc}",
                }

        meta_filename = f"{prefix}.json"
        (steps_dir / meta_filename).write_text(
            json.dumps(step_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        exported_steps.append({
            "step": step_name,
            "timestamp": snapshot.get("timestamp"),
            "meta_file": f"recent_steps/{meta_filename}",
            "images": step_meta["images"],
        })

    return exported_steps


def dump_task_failure(
    app: "AppProcessor",
    task: "Task",
    exception: Optional[BaseException] = None,
) -> str | None:
    """
    保存任务失败的诊断现场。

    Parameters
    ----------
    app : AppProcessor
        应用主处理器，用于获取最新帧和 YOLO 结果。
    task : Task
        失败的任务对象。
    exception : BaseException, optional
        导致失败的异常。
    """
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_dir = Path(resolve_log_path("dumps", f"{task.id}_{ts}"))
        dump_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. 最后一帧截图 ──
        frame: Optional[np.ndarray] = getattr(app, "latest_frame", None)
        if frame is not None:
            cv2.imwrite(str(dump_dir / "last_frame.png"), frame)
        frame_shape = list(map(int, frame.shape)) if frame is not None else None

        # ── 2. YOLO 检测结果 & 标注帧 ──
        results: Optional["Yolo_Results"] = getattr(app, "latest_results", None)
        yolo_data = []
        button_snapshot = []
        button_snapshot_error = None
        if results is not None:
            yolo_data = _serialize_yolo_results(results)
            button_snapshot, button_snapshot_error = _serialize_button_snapshot(results)
            # 标注帧
            raw = results.results
            if raw is not None and hasattr(raw, "plot"):
                try:
                    annotated = raw.plot()
                    cv2.imwrite(str(dump_dir / "annotated_frame.png"), annotated)
                except Exception as e:
            logger.debug(f"TaskDumper: 操作失败: {e}")


        recent_steps = _dump_recent_steps(dump_dir, app)

        # ── 3. 任务 & 状态元数据 ──
        meta = {
            "timestamp": ts,
            "task": {
                "id": task.id,
                "name": task.task_name,
                "status": task.status,
                "timeout": task.timeout,
                "runtime_timeout": task.get_timeout(),
                "start_time": task.get_start_time(),
            },
            "game": {
                "current_location": getattr(
                    getattr(app, "game_status_manager", None),
                    "current_location",
                    None,
                ),
            },
            "ui": {
                "frame_shape": frame_shape,
                "buttons": button_snapshot,
            },
            "debug": {
                "task_trace": get_task_debug_trace(app),
                "recent_steps": recent_steps,
            },
            "runtime": _collect_runtime_diagnostics(app, task),
            "yolo_detections": yolo_data,
        }
        if button_snapshot_error is not None:
            meta["ui"]["button_snapshot_error"] = button_snapshot_error

        # ── 4. 异常堆栈 ──
        if exception is not None:
            meta["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                ),
            }

        with open(dump_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"Task failure dump saved to {dump_dir}")

        # 清理旧 dump
        dumps_root = str(resolve_log_path("dumps"))
        _cleanup_old_dumps(dumps_root)
        return str(dump_dir)

    except Exception as e:
        logger.warning(f"Failed to save task dump: {e}")
        return None


@register_task_dump_step_exporter("click")
def _export_step_click(snapshot: dict):
    captures = snapshot.get("captures") or {}
    click_capture = captures.get("click")
    if not click_capture:
        return None
    return {
        "x": click_capture.get("x"),
        "y": click_capture.get("y"),
        "label": click_capture.get("label", ""),
        "source": click_capture.get("source", ""),
    }


@register_task_dump_step_exporter("recognition")
def _export_step_recognition(snapshot: dict):
    captures = snapshot.get("captures") or {}
    recognition = captures.get("recognition")
    if not recognition:
        return None
    return {
        "count": recognition.get("count", 0),
        "detections": recognition.get("detections", []),
    }


@register_task_dump_step_exporter("swipe")
def _export_step_swipe(snapshot: dict):
    captures = snapshot.get("captures") or {}
    swipe_capture = captures.get("swipe")
    if not swipe_capture:
        return None
    start_x = swipe_capture.get("start_x")
    start_y = swipe_capture.get("start_y")
    end_x = swipe_capture.get("end_x")
    end_y = swipe_capture.get("end_y")
    if None in {start_x, start_y, end_x, end_y}:
        return None
    distance = round(float(((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5), 2)
    return {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "duration": swipe_capture.get("duration"),
        "distance": distance,
        "source": swipe_capture.get("source", ""),
    }


@register_task_dump_step_exporter("scroll")
def _export_step_scroll(snapshot: dict):
    captures = snapshot.get("captures") or {}
    scroll_capture = captures.get("scroll")
    if not scroll_capture:
        return None
    return {
        "x": scroll_capture.get("x"),
        "y": scroll_capture.get("y"),
        "direction": scroll_capture.get("direction", ""),
        "delta": scroll_capture.get("delta"),
        "source": scroll_capture.get("source", ""),
    }
