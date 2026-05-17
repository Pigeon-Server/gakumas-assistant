import sys
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.task_status import TaskStatus
from src.entity.Game.Page.Types.index import GamePageTypes
from src.core.tasks.base_ui import refresh_skill_storage
from src.core.services.task_service import TaskService
from src.entity.Task import Task


def test_navigate_to_page_suspends_until_user_switches_page(monkeypatch):
    calls = []
    locations = [
        GamePageTypes.HOME_TAB.PASS_REWARD,
        GamePageTypes.SUB_MENU.PRODUCER_ILLUSTRATED,
    ]

    monkeypatch.setattr(
        refresh_skill_storage,
        "message_tools",
        SimpleNamespace(info=lambda message, timeout=3: calls.append(("info", message, timeout))),
    )
    monkeypatch.setattr(refresh_skill_storage, "TabBar", lambda _box: [SimpleNamespace(text="スキルカード")])

    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            update_current_location=lambda: locations.pop(0),
            wait_for_label=lambda _label: True,
        ),
        latest_results=SimpleNamespace(filter_by_label=lambda _label: SimpleNamespace(first=lambda: object())),
        device=SimpleNamespace(click_element=lambda *_args, **_kwargs: calls.append("click_tab")),
        task_queue=SimpleNamespace(
            suspend_running_task=lambda: calls.append("suspend_running_task"),
        ),
    )

    assert refresh_skill_storage.__navigate_to_page(app) is True
    assert calls == [
        ("info", "任务已挂起，请手动切换到图鉴页面", 30),
        "suspend_running_task",
        "click_tab",
    ]


def test_navigate_to_page_enters_skill_tab_when_already_on_illustrated_page(monkeypatch):
    calls = []

    monkeypatch.setattr(
        refresh_skill_storage,
        "message_tools",
        SimpleNamespace(info=lambda message, timeout=3: calls.append(("info", message, timeout))),
    )
    monkeypatch.setattr(refresh_skill_storage, "TabBar", lambda _box: [SimpleNamespace(text="スキルカード")])

    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            update_current_location=lambda: GamePageTypes.SUB_MENU.PRODUCER_ILLUSTRATED,
            wait_for_label=lambda _label: True,
        ),
        latest_results=SimpleNamespace(filter_by_label=lambda _label: SimpleNamespace(first=lambda: object())),
        device=SimpleNamespace(click_element=lambda *_args, **_kwargs: calls.append("click_tab")),
        task_queue=SimpleNamespace(suspend_running_task=lambda: calls.append("suspend_running_task")),
    )

    assert refresh_skill_storage.__navigate_to_page(app) is True
    assert calls == [
        "click_tab",
    ]


def test_processor_resumes_same_task_after_self_suspend(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)

    app = SimpleNamespace(
        is_resource_ready=lambda: True,
        ensure_device_ready=lambda restart_inference=True: True,
        yolo_engine=SimpleNamespace(pause=lambda: None),
        device=None,
    )
    service = TaskService(app)
    calls = {"entered": 0, "after_resume": 0}
    task_namespace = {}
    task_filename = str(Path(TaskService._src_path) / "core" / "tasks" / "_test_self_suspend_task.py")
    exec(
        compile(
            """
def _task_body(_app):
    calls["entered"] += 1
    service.suspend_running_task(update_status=False)
    calls["after_resume"] += 1
    return True
""",
            task_filename,
            "exec",
        ),
        {
            "TaskStatus": TaskStatus,
            "calls": calls,
            "service": service,
        },
        task_namespace,
    )
    task_body = task_namespace["_task_body"]

    task = Task(
        id="refresh-skill-storage-test",
        task_name="Refresh Skill Storage Test",
        enable=True,
        disabled_middleware=True,
        function=task_body,
        timeout=10,
        allow_manual_resume=True,
    )
    service._task_list.append(task)
    service._task_queue = Queue()
    service._task_queue.put(task)
    service._queue_status = True
    service._resume_event.set()

    worker = threading.Thread(target=service._processor_task_queue, daemon=True)
    worker.start()

    for _ in range(50):
        if service.queue_status() == TaskStatus.SUSPENDED:
            break
        worker.join(0.02)
    assert service.queue_status() == TaskStatus.SUSPENDED
    assert calls["entered"] == 1
    assert calls["after_resume"] == 0

    service.resume_suspended_task()
    worker.join(2)

    assert not worker.is_alive()
    assert calls["entered"] == 1
    assert calls["after_resume"] == 1
    assert task.status == TaskStatus.SUCCESS
