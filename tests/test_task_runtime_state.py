import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.task_status import TaskStatus
from src.entity.Task import Task


def _dummy_task():
    return None


def test_task_runtime_state_resets_start_time_and_runtime_timeout():
    task = Task(
        id="demo",
        task_name="Demo",
        enable=True,
        disabled_middleware=False,
        function=_dummy_task,
        timeout=12,
        status=TaskStatus.FAILED,
    )
    task._start_time = 123456
    task._runtime_timeout = 99

    task.reset_runtime_state()

    assert task.get_start_time() == -1
    assert task.get_timeout() == 12.0


def test_update_start_time_does_not_change_status_and_reinitializes_runtime_timeout():
    task = Task(
        id="demo",
        task_name="Demo",
        enable=True,
        disabled_middleware=False,
        function=_dummy_task,
        timeout=8,
        status=TaskStatus.PENDING,
    )
    task._runtime_timeout = 100

    start_time = task.update_start_time()

    assert start_time == task.get_start_time()
    assert task.status == TaskStatus.PENDING
    assert task.get_timeout() == 8.0


def test_extend_timeout_only_affects_runtime_timeout():
    task = Task(
        id="demo",
        task_name="Demo",
        enable=True,
        disabled_middleware=False,
        function=_dummy_task,
        timeout=5,
    )
    task.reset_runtime_state()

    task.extend_timeout(2.5)

    assert task.timeout == 5
    assert task.get_timeout() == 7.5
