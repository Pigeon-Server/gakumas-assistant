import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from queue import Queue


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.services.task_service import TaskService
from src.core.exceptions.TaskException import TaskUserMessage
from src.constants.task_status import TaskStatus
from src.entity.Task import Task


def test_task_trace_executes_middleware_only_on_line_events(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)

    service = TaskService(SimpleNamespace(is_resource_ready=lambda: False))
    middleware_calls = {"count": 0}
    service._exec_task_middleware = lambda: middleware_calls.__setitem__("count", middleware_calls["count"] + 1) or True

    task = Task(
        id="trace_test",
        task_name="trace_test",
        enable=True,
        disabled_middleware=False,
        function=lambda: None,
        timeout=10,
    )
    trace = service._make_trace(task, Event())

    fake_filename = str(Path(TaskService._src_path) / "core" / "tasks" / "fake_task.py")
    frame = SimpleNamespace(f_code=SimpleNamespace(co_filename=fake_filename))

    trace(frame, "call", None)
    trace(frame, "line", None)
    trace(frame, "return", None)

    assert middleware_calls["count"] == 1


def test_task_trace_skips_middleware_while_model_switching(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)

    app = SimpleNamespace(
        is_resource_ready=lambda: False,
        yolo_engine=SimpleNamespace(is_model_switching=True),
    )
    service = TaskService(app)
    middleware_calls = {"count": 0}
    service._exec_task_middleware = lambda: middleware_calls.__setitem__("count", middleware_calls["count"] + 1) or True

    task = Task(
        id="trace_model_switch",
        task_name="trace_model_switch",
        enable=True,
        disabled_middleware=False,
        function=lambda: None,
        timeout=10,
    )
    trace = service._make_trace(task, Event())

    fake_filename = str(Path(TaskService._src_path) / "core" / "tasks" / "fake_task.py")
    frame = SimpleNamespace(f_code=SimpleNamespace(co_filename=fake_filename))

    trace(frame, "line", None)

    assert middleware_calls["count"] == 0


def test_stop_keeps_pending_tasks_pending(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)

    service = TaskService(SimpleNamespace(
        is_resource_ready=lambda: True,
        ensure_device_ready=lambda restart_inference=True: True,
        yolo_engine=SimpleNamespace(pause=lambda: None),
        device=None,
    ))

    task1 = Task(
        id="pending_1",
        task_name="pending_1",
        enable=True,
        disabled_middleware=True,
        function=lambda: None,
        timeout=10,
        status=TaskStatus.PENDING,
    )
    task2 = Task(
        id="pending_2",
        task_name="pending_2",
        enable=True,
        disabled_middleware=True,
        function=lambda: None,
        timeout=10,
        status=TaskStatus.PENDING,
    )

    service._task_list.extend([task1, task2])
    service._task_queue = Queue()
    service._task_queue.put(task1)
    service._task_queue.put(task2)
    service._queue_status = True

    assert service.stop() is True
    assert task1.status == TaskStatus.PENDING
    assert task2.status == TaskStatus.PENDING
    assert service._task_queue.empty()


def test_handle_task_failure_broadcasts_download_link(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)
    service = TaskService(SimpleNamespace(is_resource_ready=lambda: False))

    monkeypatch.setattr(
        "src.core.services.task_service.dump_task_failure",
        lambda _app, _task, _exc: "/tmp/dump_case",
    )
    monkeypatch.setattr(
        "src.core.services.task_service.build_task_failure_package",
        lambda _app, _task, _dump, _exc: "/tmp/failure_case.zip",
    )
    monkeypatch.setattr(
        "src.core.services.task_service.register_task_failure_package_download",
        lambda _path: {
            "package_id": "pkg_test",
            "download_url": "/api/task/failure_package/download/pkg_test",
        },
    )

    captured = {}

    def _capture(action, data=None):
        captured["action"] = action
        captured["payload"] = {} if data is None else data.message

    monkeypatch.setattr("src.core.services.task_service.websocket_manager.broadcast_action_sync", _capture)

    task = SimpleNamespace(
        id="auto_producer",
        task_name="自动培育",
        status=TaskStatus.FAILED,
    )
    service._handle_task_failure(task, RuntimeError("boom"))

    payload = captured["payload"]
    assert payload["package_path"] == "/tmp/failure_case.zip"
    assert payload["package_id"] == "pkg_test"
    assert payload["package_download_url"].endswith("/pkg_test")


def test_run_task_inner_marks_canceled_when_failure_finalize_is_interrupted(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)
    service = TaskService(SimpleNamespace(is_resource_ready=lambda: False))

    task = Task(
        id="failure_interrupted",
        task_name="failure_interrupted",
        enable=True,
        disabled_middleware=True,
        function=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        timeout=10,
    )
    monkeypatch.setattr(service, "_handle_task_failure_with_cancel_guard", lambda *_args, **_kwargs: False)

    service._run_task_inner(task, Event())

    assert task.status == TaskStatus.CANCELED


def test_run_task_inner_broadcasts_user_message_without_dump(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)
    service = TaskService(SimpleNamespace(is_resource_ready=lambda: False))

    task = Task(
        id="user_message",
        task_name="user_message",
        enable=True,
        disabled_middleware=True,
        function=lambda: (_ for _ in ()).throw(TaskUserMessage("show this message")),
        timeout=10,
    )

    broadcasts = []
    dump_called = {"value": False}

    monkeypatch.setattr(
        "src.core.services.task_service.websocket_manager.broadcast_action_sync",
        lambda action, data=None: broadcasts.append(
            (action, {} if data is None else data.message)
        ),
    )
    monkeypatch.setattr(
        service,
        "_handle_task_failure_with_cancel_guard",
        lambda *_args, **_kwargs: dump_called.__setitem__("value", True),
    )

    service._run_task_inner(task, Event())

    assert task.status == TaskStatus.CANCELED
    assert dump_called["value"] is False
    execution_errors = [
        payload for action, payload in broadcasts
        if action == "task:execution_error"
    ]
    assert execution_errors
    assert execution_errors[-1]["error_message"] == "show this message"
    assert execution_errors[-1]["dump_dir"] == ""
