import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.tasks.task_register import register_tasks


class _FakeTaskQueue:
    def __init__(self):
        self.task_ids = []
        self.pre_hooks = []

    def register_task(self, task_id, *args, **kwargs):
        def decorator(func):
            self.task_ids.append(task_id)
            return func

        return decorator

    def register_pre_queue_start(self):
        def decorator(func):
            self.pre_hooks.append(func.__name__)
            return func

        return decorator


def test_register_tasks_is_safe_before_resources_ready():
    processor = SimpleNamespace(task_queue=_FakeTaskQueue())

    register_tasks(processor)

    assert "start_game" in processor.task_queue.task_ids
    assert "claim_task_rewards" in processor.task_queue.task_ids
    assert "refresh_skill_storage" in processor.task_queue.task_ids
    assert processor.task_queue.pre_hooks
