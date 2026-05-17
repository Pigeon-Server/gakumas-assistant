import sys
from datetime import datetime
from types import SimpleNamespace

from peewee import SqliteDatabase


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.entity.Config import Config
from src.models.config import ConfigModel


def test_load_config_prefers_code_verify_over_stale_database_metadata():
    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx([ConfigModel]):
        test_db.connect()
        test_db.create_tables([ConfigModel])
        ConfigModel.create(
            key="base.android_screen_capture_service",
            value="ADB",
            verify="ADB|uiautomator2|DroidCast|Bin",
            use_verify=True,
            last_modified_time=datetime.now(),
        )

        config = ConfigModel.load_config()

        assert config.base.android_screen_capture_service.verify == "ADB|uiautomator2|DroidCast|scrcpy"
        assert config.base.android_screen_capture_service.use_verify is True


def test_update_database_refreshes_stale_verify_metadata():
    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx([ConfigModel]):
        test_db.connect()
        test_db.create_tables([ConfigModel])
        ConfigModel.create(
            key="base.android_screen_capture_service",
            value="ADB",
            verify="ADB|uiautomator2|DroidCast|Bin",
            use_verify=True,
            last_modified_time=datetime.now(),
        )

        ConfigModel.update_database()
        row = ConfigModel.get(ConfigModel.key == "base.android_screen_capture_service")

        assert row.verify == "ADB|uiautomator2|DroidCast|scrcpy"
        assert row.use_verify is True


def test_resource_update_check_period_accepts_supported_value():
    config = Config()
    payload = config.to_json_dict()
    payload["base"]["resource_update_check_period"]["value"] = "weekly"

    status, errors = config.from_json_dict(payload)

    assert status is True
    assert errors == []
    assert config.base.resource_update_check_period.value == "weekly"


def test_ocr_backend_accepts_vision_value():
    config = Config()
    payload = config.to_json_dict()
    payload["base"]["ocr_backend"]["value"] = "vision"

    status, errors = config.from_json_dict(payload)

    assert status is True
    assert errors == []
    assert config.base.ocr_backend.value == "vision"


def test_llm_insight_visible_if_uses_base_decision_backend_paths():
    config = Config()
    payload = config.to_json_dict()
    visible_if = payload["base"]["llm_insight_max_tokens"]["ui"]["visible_if"]

    assert visible_if == {
        "__or__": [
            {"base.schedule_decision_backend": "llm"},
            {"base.battle_decision_backend": "llm"},
            {"base.other_decision_backend": "llm"},
        ],
    }


def test_from_json_dict_migrates_legacy_producer_decision_backend():
    config = Config()
    payload = {
        "base": {
            "producer_decision_backend": {
                "value": "rl_battle",
            },
        },
    }

    status, errors = config.from_json_dict(payload)

    assert status is True
    assert errors == []
    assert config.base.schedule_decision_backend.value == "llm"
    assert config.base.battle_decision_backend.value == "rl_battle"
    assert config.base.other_decision_backend.value == "llm"


def test_load_config_migrates_legacy_producer_decision_backend():
    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx([ConfigModel]):
        test_db.connect()
        test_db.create_tables([ConfigModel])
        ConfigModel.create(
            key="base.producer_decision_backend",
            value="rl_battle",
            verify="llm|rl_battle",
            use_verify=True,
            last_modified_time=datetime.now(),
        )

        config = ConfigModel.load_config()

        assert config.base.schedule_decision_backend.value == "llm"
        assert config.base.battle_decision_backend.value == "rl_battle"
        assert config.base.other_decision_backend.value == "llm"
