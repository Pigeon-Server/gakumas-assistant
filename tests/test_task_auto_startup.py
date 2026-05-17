import sys
from types import SimpleNamespace
from datetime import datetime


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.services.task_service import TaskService
from src.entity.Config import Config


def test_auto_startup_time_accepts_hh_mm():
    config = Config()
    payload = config.to_json_dict()
    payload["base"]["auto_startup_time"]["value"] = "07:30"

    status, errors = config.from_json_dict(payload)

    assert status is True
    assert errors == []
    assert config.base.auto_startup_time.value == "07:30"


def test_auto_startup_time_rejects_invalid_format():
    config = Config()
    payload = config.to_json_dict()
    payload["base"]["auto_startup_time"]["value"] = "7:30"

    status, errors = config.from_json_dict(payload)

    assert status is False
    assert any(error.section == "base" and error.field == "auto_startup_time" for error in errors)


def test_get_next_auto_startup_datetime_uses_same_day_when_future():
    now = datetime(2026, 3, 24, 8, 15, 30)

    next_run = TaskService._get_next_auto_startup_datetime(now, "12:00")

    assert next_run == datetime(2026, 3, 24, 12, 0, 0)


def test_get_next_auto_startup_datetime_uses_same_day_within_same_minute():
    now = datetime(2026, 3, 24, 12, 0, 30)

    next_run = TaskService._get_next_auto_startup_datetime(now, "12:00")

    assert next_run == datetime(2026, 3, 24, 12, 0, 0)


def test_get_next_auto_startup_datetime_rolls_to_next_day_when_time_passed():
    now = datetime(2026, 3, 24, 12, 0, 1)

    next_run = TaskService._get_next_auto_startup_datetime(now.replace(minute=1), "12:00")

    assert next_run == datetime(2026, 3, 25, 12, 0, 0)
