import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.services.task_service import TaskService
from src.core.tasks.task_register import register_tasks


def test_get_task_list_uses_real_registered_tasks_before_resources_ready(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)
    app = SimpleNamespace(is_resource_ready=lambda: False)
    service = TaskService(app)
    register_tasks(SimpleNamespace(task_queue=service))

    task_list = service.get_task_list()

    assert "start_game" in task_list
    assert "claim_task_rewards" in task_list
    assert "void_task" not in task_list
    assert task_list["refresh_skill_storage"]["manual_only"] is True


def test_start_game_task_keeps_middleware_enabled(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)
    service = TaskService(SimpleNamespace(is_resource_ready=lambda: False))
    register_tasks(SimpleNamespace(task_queue=service))

    start_game_task = service._find_task("start_game")

    assert start_game_task is not None
    assert start_game_task.disabled_middleware is False


def test_get_expenditure_timeout_is_extended_for_post_startup_loading(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)
    service = TaskService(SimpleNamespace(is_resource_ready=lambda: False))
    register_tasks(SimpleNamespace(task_queue=service))

    get_expenditure_task = service._find_task("get_expenditure")

    assert get_expenditure_task is not None
    assert get_expenditure_task.timeout == 60


def test_get_task_list_is_empty_when_nothing_is_registered(monkeypatch):
    monkeypatch.setattr(TaskService, "_init_auto_startup_scheduler", lambda self: None)
    service = TaskService(SimpleNamespace(is_resource_ready=lambda: False))

    assert service.get_task_list() == {}
