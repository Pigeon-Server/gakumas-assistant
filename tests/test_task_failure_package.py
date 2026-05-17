from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from src.utils import task_failure_package


def test_build_task_failure_package_anonymizes_dump_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        task_failure_package,
        "resolve_log_path",
        lambda *parts: tmp_path.joinpath(*parts),
    )

    now = datetime.now().replace(microsecond=0)
    start_at = now - timedelta(minutes=1)

    log_line = (
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}.000 | ERROR    | MainThread | "
        f"src.core.services.task_service:_run_task_inner:1 - "
        "token=abc123 open_id=9527 ip=192.168.1.10 "
        "email=foo@example.com path=/Users/tester/projects/gakumas\n"
    )
    (tmp_path / f"{now.strftime('%Y-%m-%d')}.log").write_text(log_line, encoding="utf-8")

    dump_dir = tmp_path / "dumps" / "auto_contest_20260420_120000"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.joinpath("meta.json").write_text(
        '{"account":"foo@example.com","path":"/Users/tester/.gakumas-assistant","token":"secret"}',
        encoding="utf-8",
    )

    task = SimpleNamespace(
        id="auto_contest",
        task_name="自动每日竞技场",
        status="FAILED",
        get_start_time=lambda: int(start_at.timestamp()),
        get_timeout=lambda: None,
    )

    package_path = task_failure_package.build_task_failure_package(
        app=SimpleNamespace(),
        task=task,
        dump_dir=dump_dir,
        exception=RuntimeError("boom"),
    )
    assert package_path is not None
    assert Path(package_path).exists()

    with ZipFile(package_path, "r") as zf:
        task_log = zf.read("logs/task.log").decode("utf-8")
        dump_meta = zf.read("dump/meta.json").decode("utf-8")
        manifest = zf.read("manifest.json").decode("utf-8")

    assert "abc123" not in task_log
    assert "192.168.1.10" not in task_log
    assert "foo@example.com" not in task_log
    assert "/Users/tester" not in task_log
    assert "[REDACTED]" in task_log
    assert "[REDACTED_EMAIL]" in task_log
    assert "/Users/[REDACTED_USER]" in task_log

    assert "secret" not in dump_meta
    assert "foo@example.com" not in dump_meta
    assert "/Users/tester" not in dump_meta

    assert task_failure_package.GITHUB_ISSUE_URL in manifest
    assert task_failure_package.QQ_GROUP_NUMBER in manifest


def test_register_and_resolve_task_failure_package_download(tmp_path):
    package_file = tmp_path / "task_failure.zip"
    package_file.write_bytes(b"zip-bytes")

    download = task_failure_package.register_task_failure_package_download(package_file)
    assert download is not None
    assert download["package_id"]
    assert download["package_path"] == str(package_file)
    assert download["download_url"].endswith(download["package_id"])

    resolved = task_failure_package.resolve_task_failure_package_download(download["package_id"])
    assert resolved == package_file

    package_file.unlink()
    assert task_failure_package.resolve_task_failure_package_download(download["package_id"]) is None
