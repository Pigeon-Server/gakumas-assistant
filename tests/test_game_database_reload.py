import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.utils.game_database_tools import _BaseYamlDatabase, _SingletonByFileMeta, reload_loaded_game_databases


class DummyYamlDatabase(_BaseYamlDatabase):
    loc_cls = None

    def _load_objects(self, entries):
        return [SimpleNamespace(**entry) for entry in entries]


def test_reload_loaded_game_databases_preserves_instance_identity(tmp_path):
    old_instances = dict(_SingletonByFileMeta._instances)
    _SingletonByFileMeta._instances = {}
    try:
        yaml_path = tmp_path / "Dummy.yaml"
        yaml_path.write_text("- id: first\n  value: before\n", encoding="utf-8")

        db = DummyYamlDatabase(str(yaml_path))

        yaml_path.write_text("- id: first\n  value: after\n", encoding="utf-8")

        reload_loaded_game_databases()

        assert DummyYamlDatabase(str(yaml_path)) is db
        assert db.get_by_id("first").value == "after"
    finally:
        _SingletonByFileMeta._instances = old_instances
