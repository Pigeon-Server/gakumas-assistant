from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.utils.logger import logger
from src.utils.runtime_paths import resolve_log_path

if TYPE_CHECKING:
    from src.entity.Task import Task
    from src.main import AppProcessor

GITHUB_ISSUE_URL = "https://github.com/Pigeon-Server/gakumas-assistant/issues"
QQ_GROUP_NUMBER = "328346267"

_TEXT_FILE_SUFFIXES = {
    ".log",
    ".txt",
    ".json",
    ".md",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".csv",
}
_LOG_LINE_TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \|"
)
_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r'(?i)("(?:api[_-]?key|token|pf_access_token|authorization|open_id|viewer_id|serial)"\s*:\s*")([^"]+)(")'),
        r"\1[REDACTED]\3",
    ),
    (
        re.compile(r"(?i)\b((?:api[_-]?key|token|pf_access_token|authorization|open_id|viewer_id|serial)\s*[:=]\s*)([^\s,;]+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[REDACTED_IP]",
    ),
    (
        re.compile(r"/Users/[^/\s]+"),
        "/Users/[REDACTED_USER]",
    ),
    (
        re.compile(r"/home/[^/\s]+"),
        "/home/[REDACTED_USER]",
    ),
    (
        re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"),
        r"C:\\Users\\[REDACTED_USER]",
    ),
]

_FailureBundleCollector = Callable[[Path, "AppProcessor", "Task", Path, BaseException | None], None]
_failure_bundle_collectors: dict[str, _FailureBundleCollector] = {}
_package_download_refs: dict[str, Path] = {}
_package_download_lock = threading.RLock()
_MAX_PACKAGE_DOWNLOAD_REFS = 200


def register_task_failure_bundle_collector(name: str):
    """注册任务失败打包的额外内容采集器。"""
    collector_name = str(name).strip()
    if not collector_name:
        raise ValueError("collector name cannot be empty")

    def decorator(func: _FailureBundleCollector):
        _failure_bundle_collectors[collector_name] = func
        return func

    return decorator


def register_task_failure_package_download(package_path: str | Path):
    """
    注册失败包下载引用，返回可用于前端直链下载的元信息。
    链接生命周期：应用进程存活期间有效，重启后失效。
    """
    if not package_path:
        return None
    package_file = Path(package_path)
    if not package_file.exists() or not package_file.is_file():
        return None

    package_id = uuid.uuid4().hex
    with _package_download_lock:
        stale_keys = [key for key, value in _package_download_refs.items() if not value.exists()]
        for stale_key in stale_keys:
            _package_download_refs.pop(stale_key, None)
        if len(_package_download_refs) >= _MAX_PACKAGE_DOWNLOAD_REFS:
            # 使用插入顺序淘汰最旧的一批，避免内存中无限累积下载引用。
            for old_key in list(_package_download_refs.keys())[: len(_package_download_refs) - _MAX_PACKAGE_DOWNLOAD_REFS + 1]:
                _package_download_refs.pop(old_key, None)
        _package_download_refs[package_id] = package_file
    return {
        "package_id": package_id,
        "package_path": str(package_file),
        "download_url": f"/api/task/failure_package/download/{package_id}",
    }


def resolve_task_failure_package_download(package_id: str) -> Path | None:
    if not package_id:
        return None
    with _package_download_lock:
        package_path = _package_download_refs.get(package_id)
    if package_path is None:
        return None
    if not package_path.exists() or not package_path.is_file():
        with _package_download_lock:
            _package_download_refs.pop(package_id, None)
        return None
    return package_path


def _anonymize_text(text: str) -> str:
    redacted = text
    for pattern, replacement in _REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_FILE_SUFFIXES


def _copy_with_anonymization(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _is_text_file(src):
        raw_text = src.read_text(encoding="utf-8", errors="ignore")
        dst.write_text(_anonymize_text(raw_text), encoding="utf-8")
        return
    shutil.copy2(src, dst)


def _copy_tree_with_anonymization(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        return
    for source in src_dir.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(src_dir)
        target = dst_dir / relative
        _copy_with_anonymization(source, target)


def _parse_log_line_timestamp(line: str) -> datetime | None:
    match = _LOG_LINE_TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S.%f")


def _extract_task_logs(task: "Task") -> str:
    log_root = Path(resolve_log_path())
    if not log_root.exists():
        return ""
    log_files = sorted([path for path in log_root.glob("*.log") if path.is_file()])
    if not log_files:
        return ""

    start_at = None
    end_at = datetime.now()
    start_ts = task.get_start_time()
    if isinstance(start_ts, (int, float)) and start_ts not in (-1, None) and start_ts > 0:
        start_at = datetime.fromtimestamp(start_ts)

    collected: list[str] = []
    for file_path in log_files:
        in_window = start_at is None
        with file_path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                ts = _parse_log_line_timestamp(line)
                if ts is not None:
                    in_window = (start_at is None or ts >= start_at) and ts <= end_at
                if in_window:
                    collected.append(line)

    if collected:
        return "".join(collected)

    # 没有命中时间窗口时，回退到最近日志尾部，避免打包空文件。
    last_file = log_files[-1]
    with last_file.open("r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
    return "".join(lines[-800:])


def _write_bundle_manifest(
    bundle_dir: Path,
    task: "Task",
    dump_dir: Path,
    exception: BaseException | None,
):
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "anonymized": True,
        "task": {
            "id": task.id,
            "name": task.task_name,
            "status": task.status,
            "start_time": task.get_start_time(),
            "runtime_timeout": task.get_timeout(),
        },
        "dump_dir": str(dump_dir),
        "feedback": {
            "github_issues": GITHUB_ISSUE_URL,
            "qq_group": QQ_GROUP_NUMBER,
        },
    }
    if exception is not None:
        manifest["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
        }
    (bundle_dir / "manifest.json").write_text(
        _anonymize_text(json.dumps(manifest, ensure_ascii=False, indent=2)),
        encoding="utf-8",
    )


def build_task_failure_package(
    app: "AppProcessor",
    task: "Task",
    dump_dir: str | Path | None,
    exception: BaseException | None = None,
) -> str | None:
    """构建任务失败反馈压缩包（自动匿名化）。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_root = Path(resolve_log_path("issue_packages"))
    package_root.mkdir(parents=True, exist_ok=True)
    staging_dir = package_root / f"{task.id}_{ts}_staging"
    archive_base = package_root / f"{task.id}_{ts}"
    archive_path = archive_base.with_suffix(".zip")

    dump_path = Path(dump_dir) if dump_dir else Path()

    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        if dump_path.exists():
            _copy_tree_with_anonymization(dump_path, staging_dir / "dump")

        task_logs = _extract_task_logs(task)
        logs_dir = staging_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "task.log").write_text(_anonymize_text(task_logs), encoding="utf-8")

        extras_root = staging_dir / "extras"
        for collector_name, collector in _failure_bundle_collectors.items():
            collector_output = extras_root / collector_name
            collector_output.mkdir(parents=True, exist_ok=True)
            try:
                collector(collector_output, app, task, dump_path, exception)
            except Exception as exc:
                logger.warning(f"Collect failure bundle extra '{collector_name}' failed: {exc}")

        _write_bundle_manifest(staging_dir, task, dump_path, exception)

        if archive_path.exists():
            archive_path.unlink()
        final_zip = shutil.make_archive(str(archive_base), "zip", root_dir=staging_dir)
        logger.info(f"Task failure package saved to {final_zip}")
        return final_zip
    except Exception as exc:
        logger.warning(f"Build task failure package failed: {exc}")
        return None
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


@register_task_failure_bundle_collector("auto_producer_thought_chain")
def _collect_auto_producer_thought_chain(
    output_dir: Path,
    _app: "AppProcessor",
    task: "Task",
    _dump_dir: Path,
    _exception: BaseException | None,
):
    if task.id != "auto_producer":
        return
    from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper

    session_dir = DecisionDumper.get_instance().session_dir
    if session_dir is None:
        return
    source = Path(session_dir)
    if not source.is_absolute():
        source = Path(os.getcwd()) / source
    if not source.exists():
        return
    _copy_tree_with_anonymization(source, output_dir / "decision_dump")
