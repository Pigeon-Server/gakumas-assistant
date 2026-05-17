import json
from pathlib import Path

from src.utils import runtime_paths


def _clear_runtime_path_caches():
    for func_name in (
        "get_runtime_root",
        "get_runtime_metadata_path",
        "get_storage_mode",
        "get_user_data_root",
        "get_storage_root",
        "get_cache_root",
        "get_data_root",
        "get_log_root",
        "get_managed_resource_root",
    ):
        cache_clear = getattr(getattr(runtime_paths, func_name), "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


def test_storage_root_defaults_to_runtime_root(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runtime_paths, "get_runtime_root", lambda: runtime_root)
    monkeypatch.delenv("GAKUMAS_STORAGE_MODE", raising=False)
    _clear_runtime_path_caches()

    assert runtime_paths.get_storage_mode() == runtime_paths.STORAGE_MODE_PORTABLE
    assert runtime_paths.get_storage_root() == runtime_root
    assert runtime_paths.resolve_data_path("db.sqlite3") == runtime_root / "data" / "db.sqlite3"


def test_storage_root_uses_user_home_for_merged_build(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    metadata_path = runtime_root / runtime_paths.RUNTIME_METADATA_FILE_NAME
    metadata_path.write_text(
        json.dumps({"storage_mode": runtime_paths.STORAGE_MODE_MERGED}, ensure_ascii=False),
        encoding="utf-8",
    )
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runtime_paths, "get_runtime_root", lambda: runtime_root)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("GAKUMAS_STORAGE_MODE", raising=False)
    _clear_runtime_path_caches()

    assert runtime_paths.get_storage_mode() == runtime_paths.STORAGE_MODE_MERGED
    assert runtime_paths.get_storage_root() == fake_home / ".gakumas-assistant"
    assert runtime_paths.resolve_log_path("debug") == fake_home / ".gakumas-assistant" / "logs" / "debug"
