import importlib
import json
import os
import sys
from threading import Event, Lock
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _FakeConfigService:
    def __init__(self):
        self.base = SimpleNamespace(
            enabled_check_resource_updates=False,
            check_resource_updates_on_startup=False,
            resource_update_check_period="daily",
        )

    def add_listener(self, keys, callback):
        return None


class _FakeWebSocketManager:
    def broadcast_action_sync(self, *args, **kwargs):
        return None


class _FakeUIMessage:
    def success(self, *args, **kwargs):
        return None


def _load_resource_update_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.utils.logger", SimpleNamespace(logger=_LoggerStub()))
    monkeypatch.setitem(
        sys.modules,
        "src.core.services.config_service",
        SimpleNamespace(ConfigService=_FakeConfigService),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.core.web.websocket",
        SimpleNamespace(WebSocketManager=_FakeWebSocketManager),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.utils.ui_message_tools",
        SimpleNamespace(UIMessage=_FakeUIMessage),
    )
    sys.modules.pop("src.core.services.resource_update_service", None)
    return importlib.import_module("src.core.services.resource_update_service")


def _build_service(module, tmp_path, git_executable="git"):
    service = module.ResourceUpdateService.__new__(module.ResourceUpdateService)
    service._git_executable = git_executable
    service._app = SimpleNamespace(data_path=str(tmp_path / "data"))
    service._session = None
    service._status_lock = Lock()
    service._operation_lock = Lock()
    service._refresh_event = Event()
    service._next_check_at = None
    service._status = {}
    (tmp_path / "data").mkdir(exist_ok=True)
    return service


def _bind_repository_paths(monkeypatch, service, repository, repo_path):
    monkeypatch.setattr(service, "_get_repository_mutable_path", lambda repository_obj: repo_path)
    monkeypatch.setattr(service, "_get_repository_active_path", lambda repository_obj: repo_path)


def test_select_update_method_prefers_git_when_available(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    repo_path = tmp_path / "assets" / "repo"
    (repo_path / ".git").mkdir(parents=True)

    service = _build_service(module, tmp_path, git_executable="git")

    assert service._select_update_method(str(repo_path)) == "git"

    service._git_executable = None

    assert service._select_update_method(str(repo_path)) == "snapshot"


def test_resolve_git_dir_supports_git_file(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    repo_path = tmp_path / "repo"
    actual_git_dir = tmp_path / ".git" / "modules" / "repo"
    repo_path.mkdir()
    actual_git_dir.mkdir(parents=True)
    (repo_path / ".git").write_text("gitdir: ../.git/modules/repo\n", encoding="utf-8")

    assert module.ResourceUpdateService._resolve_git_dir(str(repo_path)) == str(actual_git_dir.resolve())


def test_read_git_ref_uses_packed_refs(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    git_dir = tmp_path / "gitdir"
    git_dir.mkdir()
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        "abcdef1234567890 refs/heads/main\n",
        encoding="utf-8",
    )

    assert module.ResourceUpdateService._read_git_ref(str(git_dir), "refs/heads/main") == "abcdef1234567890"


def test_update_repository_prefers_git_and_persists_metadata(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    repo_relative_path = "assets/test-repo"
    repo_path = tmp_path / repo_relative_path
    (repo_path / ".git").mkdir(parents=True)
    repository = module.ResourceRepository(
        name="test-repo",
        path=repo_relative_path,
        owner="owner",
        repo="repo",
    )
    service = _build_service(module, tmp_path, git_executable="git")
    _bind_repository_paths(monkeypatch, service, repository, repo_path)
    calls = {}

    monkeypatch.setattr(
        service,
        "_update_repository_with_git",
        lambda repository_obj, repo_path_obj, branch_name_obj: calls.update(
            {"repo_path": repo_path_obj, "branch": branch_name_obj}
        ),
    )
    monkeypatch.setattr(
        service,
        "_read_local_git_version",
        lambda repo_path_obj: {
            "commit": "abcdef1234567890",
            "branch": "main",
            "source": "git",
        },
    )

    service._update_repository(
        repository,
        "ignored",
        "ignored",
        current_step=1,
        total_steps=1,
        attempt=1,
    )

    metadata_path = tmp_path / "data" / module.ResourceUpdateService.VERSION_STATE_DIR_NAME / "test-repo.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert os.path.normpath(calls["repo_path"]) == os.path.normpath(str(repo_path))
    assert calls["branch"] == "ignored"
    assert metadata["commit"] == "abcdef1234567890"
    assert metadata["branch"] == "main"
    assert metadata["source"] == "git"


def test_update_repository_falls_back_to_snapshot_without_git(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    repo_relative_path = "assets/test-snapshot"
    repo_path = tmp_path / repo_relative_path
    (repo_path / ".git").mkdir(parents=True)
    (repo_path / "old.txt").write_text("old", encoding="utf-8")
    staged_source = tmp_path / "staged-source"
    staged_source.mkdir()
    (staged_source / "new.txt").write_text("new", encoding="utf-8")
    repository = module.ResourceRepository(
        name="test-snapshot",
        path=repo_relative_path,
        owner="owner",
        repo="repo",
    )
    service = _build_service(module, tmp_path, git_executable=None)
    _bind_repository_paths(monkeypatch, service, repository, repo_path)

    monkeypatch.setattr(
        service,
        "_download_repository_snapshot",
        lambda repository_obj, commit_sha, workdir, **kwargs: str(staged_source),
    )

    service._update_repository(
        repository,
        "1234567890abcdef",
        "main",
        current_step=1,
        total_steps=1,
        attempt=1,
    )

    metadata_path = tmp_path / "data" / module.ResourceUpdateService.VERSION_STATE_DIR_NAME / "test-snapshot.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert (repo_path / "new.txt").read_text(encoding="utf-8") == "new"
    assert (repo_path / ".git").is_dir()
    assert metadata["commit"] == "1234567890abcdef"
    assert metadata["branch"] == "main"
    assert metadata["source"] == "snapshot"


def test_manual_check_updates_resets_periodic_timer_on_success(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    module.config_service.base.enabled_check_resource_updates = True
    module.config_service.base.resource_update_check_period = "weekly"
    service = _build_service(module, tmp_path, git_executable=None)
    monkeypatch.setattr(service, "get_missing_required_resources", lambda: [])
    service._status = service._build_empty_status()

    monkeypatch.setattr(
        service,
        "_inspect_repository",
        lambda repository: {
            "name": repository.name,
            "path": repository.path,
            "exists": True,
            "dirty": False,
            "has_update": False,
            "local_commit": "abc",
            "remote_commit": "abc",
            "local_commit_short": "abc",
            "remote_commit_short": "abc",
            "local_branch": "main",
            "remote_branch": "main",
            "version_source": "git",
            "update_method": "git",
            "error": "",
        },
    )

    success, _, status = service.manual_check_updates()

    assert success is True
    assert status["next_check_at"] is not None
    assert service._refresh_event.is_set() is True


def test_manual_check_updates_does_not_reset_timer_when_check_has_errors(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    module.config_service.base.enabled_check_resource_updates = True
    service = _build_service(module, tmp_path, git_executable=None)
    monkeypatch.setattr(service, "get_missing_required_resources", lambda: [])
    service._status = service._build_empty_status()
    service._set_next_check_at(None)

    monkeypatch.setattr(
        service,
        "_inspect_repository",
        lambda repository: {
            "name": repository.name,
            "path": repository.path,
            "exists": True,
            "dirty": False,
            "has_update": False,
            "local_commit": "",
            "remote_commit": "",
            "local_commit_short": "",
            "remote_commit_short": "",
            "local_branch": "",
            "remote_branch": "",
            "version_source": "",
            "update_method": "snapshot",
            "error": f"{repository.name} failed",
        },
    )

    success, message, status = service.manual_check_updates()

    assert success is True
    assert message.fallback
    assert "部分仓库检查失败" in message.fallback
    assert status["next_check_at"] is None
    assert status["last_error"]


def test_manual_check_updates_keeps_last_error_empty_when_no_repository_errors(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    service = _build_service(module, tmp_path, git_executable=None)
    monkeypatch.setattr(service, "get_missing_required_resources", lambda: [])
    service._status = service._build_empty_status()

    monkeypatch.setattr(
        service,
        "_inspect_repository",
        lambda repository: {
            "name": repository.name,
            "path": repository.path,
            "exists": True,
            "dirty": False,
            "has_update": False,
            "local_commit": "abc",
            "remote_commit": "abc",
            "local_commit_short": "abc",
            "remote_commit_short": "abc",
            "local_branch": "main",
            "remote_branch": "main",
            "version_source": "git",
            "update_method": "git",
            "error": "",
        },
    )

    success, _, status = service.manual_check_updates()

    assert success is True
    assert status["last_error"] == ""


def test_update_repository_with_git_force_resets_and_cleans_worktree(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    repository = module.ResourceRepository(
        name="test-repo",
        path="assets/test-repo",
        owner="owner",
        repo="repo",
    )
    repo_path = str(tmp_path / repository.path)
    commands = []
    service = _build_service(module, tmp_path, git_executable="git")

    monkeypatch.setattr(service, "_is_configured_submodule", lambda relative_path: False)
    monkeypatch.setattr(
        service,
        "_run_git",
        lambda args, timeout=30: commands.append((args, timeout)) or "",
    )

    service._update_repository_with_git(repository, repo_path, "main")

    assert commands == [
        (["-C", repo_path, "fetch", "--prune", "origin"], 300),
        (["-C", repo_path, "reset", "--hard", "origin/main"], 300),
        (["-C", repo_path, "clean", "-fd"], 300),
    ]


def test_update_repository_with_git_force_updates_submodule(monkeypatch, tmp_path):
    module = _load_resource_update_module(monkeypatch)
    repository = module.ResourceRepository(
        name="test-submodule",
        path="assets/test-submodule",
        owner="owner",
        repo="repo",
    )
    repo_path = str(tmp_path / repository.path)
    commands = []
    service = _build_service(module, tmp_path, git_executable="git")

    monkeypatch.setattr(service, "_is_configured_submodule", lambda relative_path: True)
    monkeypatch.setattr(
        service,
        "_run_git",
        lambda args, timeout=30: commands.append((args, timeout)) or "",
    )

    service._update_repository_with_git(repository, repo_path, "main")

    assert commands == [
        (["-C", repo_path, "clean", "-fd"], 300),
        (["submodule", "update", "--remote", "--init", "--force", "--", repository.path], 300),
        (["-C", repo_path, "clean", "-fd"], 300),
    ]
