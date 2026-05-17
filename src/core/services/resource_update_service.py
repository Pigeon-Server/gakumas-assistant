import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Event, Lock, Thread
from time import sleep
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

import requests

from src.constants.resource_repositories import (
    RESOURCE_REPOSITORIES,
    ResourceRepositoryDefinition,
)
from src.constants.task_status import TaskStatus
from src.constants.websocket_actions import WebsocketActions
from src.core.services.config_service import ConfigService
from src.core.web.websocket import WebSocketManager
from src.entity.WebSocketData import WebSocketData
from src.utils.i18n_tools import i18n_text, serialize_i18n_value
from src.utils.logger import logger
from src.utils.runtime_paths import (
    get_runtime_root,
    resolve_existing_resource_path,
    resolve_managed_resource_path,
    resolve_runtime_path,
    resolve_runtime_str,
)

if TYPE_CHECKING:
    from src.main import AppProcessor


config_service = ConfigService()
websocket_manager = WebSocketManager()
ResourceRepository = ResourceRepositoryDefinition


@dataclass(frozen=True)
class ResolvedRepositoryRemote:
    url: str
    owner: str
    repo: str
    supports_github_api: bool


class ResourceUpdateService:
    RESOURCE_REPOSITORIES = RESOURCE_REPOSITORIES
    REQUEST_TIMEOUT = 30
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    DOWNLOAD_RETRY_LIMIT = 3
    DOWNLOAD_RETRY_BASE_DELAY_SECONDS = 3
    VERSION_STATE_DIR_NAME = "resource_repository_versions"

    def __init__(self, app: "AppProcessor"):
        self._app = app
        self._started = False
        self._status_lock = Lock()
        self._operation_lock = Lock()
        self._refresh_event = Event()
        self._checker_thread: Optional[Thread] = None
        self._next_check_at: Optional[datetime] = None
        self._force_check_requested = False
        self._status = self._build_empty_status()
        self._session = requests.Session()
        self._git_executable = shutil.which("git")
        self._session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "User-Agent": "gakumas-assistant-resource-updater",
            }
        )
        config_service.add_listener(
            [
                "base.enabled_check_resource_updates",
                "base.check_resource_updates_on_startup",
                "base.resource_update_check_period",
                *[
                    repository.config_key
                    for repository in self.RESOURCE_REPOSITORIES
                    if repository.config_key
                ],
            ],
            self._on_config_changed,
        )

    def start(self):
        if self._started:
            return
        self._started = True
        self._checker_thread = Thread(target=self._checker_loop, daemon=True)
        self._checker_thread.start()

    def get_status(self):
        with self._status_lock:
            snapshot = deepcopy(self._status)
        return self._merge_status_metadata(snapshot)

    @staticmethod
    def _get_repository_parts(repository: ResourceRepository) -> tuple[str, ...]:
        return tuple(part for part in repository.path.replace("\\", "/").split("/") if part)

    def _get_repository_runtime_path(self, repository: ResourceRepository):
        return resolve_runtime_path(*self._get_repository_parts(repository))

    def _get_repository_managed_path(self, repository: ResourceRepository):
        return resolve_managed_resource_path(*self._get_repository_parts(repository))

    def _get_repository_active_path(self, repository: ResourceRepository):
        managed_path = self._get_repository_managed_path(repository)
        if managed_path.exists():
            return managed_path
        runtime_path = self._get_repository_runtime_path(repository)
        if runtime_path.exists():
            return runtime_path
        return managed_path

    def _get_repository_mutable_path(self, repository: ResourceRepository):
        runtime_path = self._get_repository_runtime_path(repository)
        if self._can_use_git(str(runtime_path)):
            return runtime_path
        return self._get_repository_managed_path(repository)

    def _get_repository_remote_url(self, repository: ResourceRepository) -> str:
        configured_url = ""
        if repository.config_key:
            try:
                configured_url = str(config_service.item(repository.config_key).value or "").strip()
            except Exception:
                configured_url = ""
        return configured_url or repository.default_url or f"https://github.com/{repository.owner}/{repository.repo}.git"

    @staticmethod
    def _parse_github_repository_url(repository_url: str) -> tuple[str, str]:
        ssh_match = re.match(r"^git@(?P<host>[^:]+):(?P<path>.+)$", repository_url.strip())
        if ssh_match:
            host = ssh_match.group("host").lower()
            path = ssh_match.group("path")
        else:
            parsed = urlparse(repository_url)
            host = (parsed.netloc or "").lower()
            path = parsed.path or ""
        if host not in {"github.com", "www.github.com"}:
            return "", ""
        path = re.sub(r"/+$", "", path)
        path_parts = [part for part in path.split("/") if part]
        if len(path_parts) < 2:
            return "", ""
        owner = path_parts[0].strip()
        repo = path_parts[1].strip()
        if repo.endswith(".git"):
            repo = repo[:-4]
        if not owner or not repo:
            return "", ""
        return owner, repo

    def _resolve_repository_remote(self, repository: ResourceRepository) -> ResolvedRepositoryRemote:
        repository_url = self._get_repository_remote_url(repository).strip()
        if not repository_url:
            raise RuntimeError(f"{repository.name} 未配置仓库 URL")
        owner, repo = self._parse_github_repository_url(repository_url)
        return ResolvedRepositoryRemote(
            url=repository_url,
            owner=owner,
            repo=repo,
            supports_github_api=bool(owner and repo),
        )

    @staticmethod
    def _classify_repository_remote_error(error_message: str) -> str:
        normalized = " ".join(str(error_message or "").split()).strip()
        if not normalized:
            return "仓库不存在、无访问权限，或网络不可用"
        lower = normalized.lower()
        if any(pattern in lower for pattern in ["repository not found", "not found", "http 404", " 404", "404："]):
            return "仓库不存在或当前地址无访问权限"
        if any(
            pattern in lower
            for pattern in [
                "authentication failed",
                "access denied",
                "permission denied",
                "forbidden",
                "http 403",
                " 403",
                "could not read username",
                "requires authentication",
            ]
        ):
            return "仓库无访问权限，可能需要认证"
        if any(
            pattern in lower
            for pattern in [
                "unable to access",
                "could not resolve host",
                "name or service not known",
                "network is unreachable",
                "failed to connect",
                "connection timed out",
                "timed out",
                "connection refused",
                "connection reset",
            ]
        ):
            return "无法访问仓库地址，请检查 URL 和网络连接"
        return normalized

    def _build_repository_remote_error(
        self,
        repository: ResourceRepository,
        remote_url: str,
        action: str,
        error: Exception | str,
    ) -> RuntimeError:
        detail = self._classify_repository_remote_error(str(error))
        suffix = f"（{remote_url}）" if remote_url else ""
        return RuntimeError(f"{repository.name} {action}失败：{detail}{suffix}")

    def get_missing_required_resources(self):
        missing_resources = []
        for repository in self.RESOURCE_REPOSITORIES:
            required_paths = repository.iter_required_relative_paths()
            if not required_paths:
                active_path = self._get_repository_active_path(repository)
                if not active_path.exists():
                    missing_resources.append(
                        {
                            "name": repository.name,
                            "path": repository.path,
                            "required_count": 1,
                            "missing_count": 1,
                            "missing_paths": ["."],
                        }
                    )
                continue
            missing_paths = [
                relative_path
                for relative_path in required_paths
                if not resolve_existing_resource_path(
                    *self._get_repository_parts(repository),
                    *relative_path.split("/"),
                ).exists()
            ]
            if missing_paths:
                missing_resources.append(
                    {
                        "name": repository.name,
                        "path": repository.path,
                        "required_count": len(required_paths),
                        "missing_count": len(missing_paths),
                        "missing_paths": missing_paths,
                    }
                )
        return missing_resources

    def has_required_resources(self) -> bool:
        return not self.get_missing_required_resources()

    def _build_empty_progress(self):
        return {
            "active": False,
            "phase": "",
            "title": "",
            "message": "",
            "repository": "",
            "repository_path": "",
            "current_step": 0,
            "total_steps": 0,
            "step_percent": 0.0,
            "percent": 0.0,
            "bytes_downloaded": 0,
            "bytes_total": 0,
            "attempt": 0,
            "max_attempts": self.DOWNLOAD_RETRY_LIMIT,
            "retry_wait_seconds": 0,
        }

    def apply_updates(self):
        if not self._operation_lock.acquire(blocking=False):
            return False, i18n_text("resource.progress.operationLocked", fallback="当前正在检查或更新资源仓库"), self.get_status()
        try:
            if self._app.task_queue.queue_status() != TaskStatus.PENDING:
                return False, i18n_text("resource.progress.taskRunning", fallback="任务执行中，无法更新资源仓库"), self.get_status()
            self._publish_status(
                {
                    **self.get_status(),
                    "updating": True,
                    "checking": False,
                    "last_error": "",
                    "progress": self._build_empty_progress(),
                }
            )
            current_status = self._check_updates_locked(updating=True)
            repository_status_map = {
                repository_status["path"]: dict(repository_status)
                for repository_status in current_status["repositories"]
            }
            repositories_to_update = [
                repo for repo in current_status["repositories"] if repo["has_update"] and not repo["error"]
            ]
            missing_paths = {item["path"] for item in self.get_missing_required_resources()}
            for repository in self.RESOURCE_REPOSITORIES:
                if repository.path in missing_paths and repository.path not in {repo["path"] for repo in repositories_to_update}:
                    repositories_to_update.append(
                        {
                            "path": repository.path,
                            "remote_commit": "",
                            "remote_branch": "",
                        }
                    )
            if not repositories_to_update:
                current_status["updating"] = False
                current_status["progress"] = self._build_empty_progress()
                self._publish_status(current_status)
                return True, i18n_text("resource.progress.upToDate", fallback="资源仓库已经是最新版本"), current_status

            update_errors = []
            total_steps = len(repositories_to_update)
            for index, repo_status in enumerate(repositories_to_update, start=1):
                repository = self._get_repository_by_path(repo_status["path"])
                try:
                    self._update_repository_with_retry(
                        repository,
                        repo_status.get("remote_commit", ""),
                        repo_status.get("remote_branch", ""),
                        current_step=index,
                        total_steps=total_steps,
                    )
                    repository_status_map[repository.path] = self._build_local_repository_status(
                        repository,
                        existing_status=repository_status_map.get(repository.path),
                    )
                except Exception as exc:
                    logger.error(f"Update resource repository '{repository.name}' failed: {exc}")
                    update_errors.append(f"{repository.name}: {exc}")

            if update_errors:
                current_status = self._check_updates_locked(updating=True)
                current_status["updating"] = False
                current_status["progress"] = self._build_empty_progress()
                current_status["last_error"] = i18n_text(
                    "resource.progress.updateFailed",
                    fallback="；".join(update_errors),
                    error="；".join(update_errors),
                )
                self._publish_status(current_status)
                return False, i18n_text(
                    "resource.progress.updateFailed",
                    fallback=f"资源仓库更新失败：{current_status['last_error']}",
                    error=serialize_i18n_value(current_status["last_error"]),
                ), current_status

            self._publish_progress(
                phase="reload_database",
                title=i18n_text("resource.progress.reloadingDatabase", fallback="正在重载游戏数据库"),
                message=i18n_text("resource.progress.reloadingDatabaseMessage", fallback="资源下载完成，正在重载游戏数据库和相关服务。"),
                current_step=total_steps,
                total_steps=total_steps,
                step_percent=100,
            )
            if self.has_required_resources():
                try:
                    self._app.reload_game_database()
                except Exception as exc:
                    logger.error(f"Reload game database after resource update failed: {exc}")
                    failed_status = self._build_status_from_repositories(
                        repositories=[
                            self._build_local_repository_status(
                                repository,
                                existing_status=repository_status_map.get(repository.path),
                            )
                            for repository in self.RESOURCE_REPOSITORIES
                        ],
                        checking=False,
                        updating=False,
                        last_error=i18n_text(
                            "resource.progress.reloadFailed",
                            fallback=f"资源已更新，但重载游戏数据库失败：{exc}",
                            error=str(exc),
                        ),
                        last_checked_at=datetime.now().isoformat(timespec="seconds"),
                    )
                    failed_status["progress"] = self._build_empty_progress()
                    self._publish_status(failed_status)
                    return False, failed_status["last_error"], failed_status
            checked_at = datetime.now()
            if self._is_periodic_check_enabled():
                self._schedule_next_check(checked_at)
                self._refresh_event.set()
            final_status = self._build_status_from_repositories(
                repositories=[
                    self._build_local_repository_status(
                        repository,
                        existing_status=repository_status_map.get(repository.path),
                    )
                    for repository in self.RESOURCE_REPOSITORIES
                ],
                checking=False,
                updating=False,
                last_error="",
                last_checked_at=checked_at.isoformat(timespec="seconds"),
            )
            final_status["progress"] = self._build_empty_progress()
            self._publish_status(final_status)
            if final_status["required_resources_ready"]:
                # 资源更新后触发 CLIP 预训练（后台异步）
                self._trigger_clip_training_after_update()
                return True, i18n_text("resource.progress.updateCompleted", fallback="资源仓库更新完成，游戏数据库已重新加载"), final_status
            return False, i18n_text("resource.progress.resourcesMissing", fallback="资源下载未完成，仍缺少运行所需资源"), final_status
        except Exception as exc:
            logger.error(f"Apply resource updates failed: {exc}")
            current_status = self._build_status_from_repositories(
                repositories=self.get_status().get("repositories", []),
                checking=False,
                updating=False,
                last_error=i18n_text(
                    "resource.progress.updateFailed",
                    fallback=str(exc),
                    error=str(exc),
                ),
                last_checked_at=self.get_status().get("last_checked_at"),
            )
            current_status["progress"] = self._build_empty_progress()
            self._publish_status(current_status)
            return False, i18n_text("resource.progress.updateFailed", fallback=f"资源仓库更新失败：{exc}", error=str(exc)), current_status
        finally:
            self._operation_lock.release()

    def _checker_loop(self):
        startup_check_pending = True
        while True:
            self._publish_status(self._merge_status_metadata(self.get_status()))
            if startup_check_pending:
                startup_check_pending = False
                if self._should_check_on_startup() and self.has_required_resources():
                    self.check_updates(reset_timer_always=self._is_periodic_check_enabled())
                    continue
                self._schedule_next_check()
                self._publish_status(self._merge_status_metadata(self.get_status()))
            if self._consume_force_check_requested():
                self.check_updates(
                    reset_timer_always=self._is_periodic_check_enabled(),
                    allow_missing_resources=True,
                )
                continue
            if not self._is_periodic_check_enabled():
                if self._wait_for_refresh(60):
                    continue
                continue
            next_check_at = self._ensure_next_check_at()
            wait_seconds = max((next_check_at - datetime.now()).total_seconds(), 0)
            if self._wait_for_refresh(wait_seconds):
                continue
            self.check_updates(reset_timer_always=True)

    def manual_check_updates(self):
        if not self._operation_lock.acquire(blocking=False):
            return False, "当前正在检查或更新资源仓库", self.get_status()
        try:
            if not self.has_required_resources():
                status = self._publish_status(
                    {
                        **self.get_status(),
                        "checking": False,
                        "updating": False,
                        "last_error": "",
                    }
                )
                return True, "缺少运行所需资源，请先下载所需资源", status
            status = self._check_updates_locked(reset_timer_on_success=self._is_periodic_check_enabled())
            if self._is_periodic_check_enabled() and not status["last_error"]:
                self._refresh_event.set()
            if status["last_error"]:
                return True, i18n_text("resource.progress.checkCompletedWithErrors", fallback=f"资源仓库检查完成，但部分仓库检查失败：{status['last_error']}", error=serialize_i18n_value(status["last_error"])), status
            if status["has_update"]:
                return True, i18n_text("resource.progress.checkHasUpdate", fallback="资源仓库检查完成，发现可用更新"), status
            return True, i18n_text("resource.progress.checkUpToDate", fallback="资源仓库已经是最新版本"), status
        except Exception as exc:
            logger.error(f"Manual resource update check failed: {exc}")
            current_status = self._merge_status_metadata(self.get_status())
            current_status["last_error"] = i18n_text("resource.progress.checkFailed", fallback=str(exc), error=str(exc))
            self._publish_status(current_status)
            return False, i18n_text("resource.progress.checkFailed", fallback=f"资源仓库检查失败：{exc}", error=str(exc)), current_status
        finally:
            self._operation_lock.release()

    def check_updates(
        self,
        reset_timer_on_success: bool = False,
        reset_timer_always: bool = False,
        allow_missing_resources: bool = False,
    ):
        if not self._operation_lock.acquire(blocking=False):
            return self.get_status()
        try:
            if not allow_missing_resources and not self.has_required_resources():
                if reset_timer_always or reset_timer_on_success:
                    self._schedule_next_check()
                status = self._publish_status(
                    {
                        **self.get_status(),
                    "checking": False,
                    "updating": False,
                    "last_error": "",
                }
                )
                return status
            return self._check_updates_locked(
                reset_timer_on_success=reset_timer_on_success,
                reset_timer_always=reset_timer_always,
            )
        finally:
            self._operation_lock.release()

    def _check_updates_locked(
        self,
        updating: bool = False,
        reset_timer_on_success: bool = False,
        reset_timer_always: bool = False,
    ):
        self._publish_status(
            self._merge_status_metadata({
                **self.get_status(),
                "checking": True,
                "updating": updating,
                "last_error": "",
            })
        )
        repositories = []
        errors = []
        for repository in self.RESOURCE_REPOSITORIES:
            repository_status = self._inspect_repository(repository)
            repositories.append(repository_status)
            if repository_status["error"]:
                errors.append(f"{repository.name}: {repository_status['error']}")
        checked_at = datetime.now()
        if reset_timer_always or (reset_timer_on_success and not errors):
            self._schedule_next_check(checked_at)
        last_error = ""
        if errors:
            last_error = i18n_text(
                "resource.progress.checkFailed",
                fallback="；".join(errors),
                error="；".join(errors),
            )
        status = self._build_status_from_repositories(
            repositories=repositories,
            checking=False,
            updating=updating,
            last_error=last_error,
            last_checked_at=checked_at.isoformat(timespec="seconds"),
        )
        self._publish_status(status)
        return status

    @staticmethod
    def _now_iso():
        return datetime.now().isoformat(timespec="seconds")

    def _wait_for_refresh(self, timeout_seconds: int):
        if self._refresh_event.wait(timeout=max(timeout_seconds, 0)):
            self._refresh_event.clear()
            return True
        return False

    def _request_force_check(self):
        with self._status_lock:
            self._force_check_requested = True
        self._refresh_event.set()

    def _consume_force_check_requested(self) -> bool:
        with self._status_lock:
            should_force_check = self._force_check_requested
            self._force_check_requested = False
        return should_force_check

    def _trigger_clip_training_after_update(self):
        """资源更新完成后，在后台下载游戏资源图片并触发 CLIP 预训练。"""
        from src.core.services import game_asset_service

        clip_manager = self._app.clip_manager
        if not game_asset_service._is_gom_available():
            logger.debug("[ResourceUpdate] GkmasObjectManager 不可用，跳过 CLIP 预训练")
            return
        if not config_service().base.enable_game_asset_download.value:
            logger.debug("[ResourceUpdate] 游戏资源下载未启用，跳过 CLIP 预训练")
            return

        def _worker():
            for download_fn, subdir in [
                (game_asset_service.download_support_card_images, game_asset_service.SUPPORT_CARD_SUBDIR),
                (game_asset_service.download_idol_card_images, game_asset_service.IDOL_CARD_SUBDIR),
                (game_asset_service.download_item_images, game_asset_service.ITEM_SUBDIR),
                (game_asset_service.download_skill_card_images, game_asset_service.SKILL_CARD_SUBDIR),
            ]:
                try:
                    download_fn()
                    if clip_manager is not None:
                        game_asset_service.train_clip_from_game_assets(clip_manager, subdir)
                except Exception as exc:
                    logger.warning(f"[ResourceUpdate] CLIP 预训练失败 ({subdir}): {exc}")

        Thread(target=_worker, daemon=True).start()
        logger.info("[ResourceUpdate] 已启动后台 CLIP 预训练任务")

    def _on_config_changed(self, key: str, old_value, new_value):
        logger.info(f"Reload resource update checker because '{key}' changed: {old_value!r} -> {new_value!r}")
        self._set_next_check_at(None)
        if key in {
            repository.config_key
            for repository in self.RESOURCE_REPOSITORIES
            if repository.config_key
        }:
            self._request_force_check()
            return
        self._refresh_event.set()

    def _build_empty_status(self):
        return self._merge_status_metadata(
            {
                "checking": False,
                "updating": False,
                "has_update": False,
                "last_checked_at": None,
                "last_error": "",
                "update_signature": "",
                "repositories": [],
                "progress": self._build_empty_progress(),
            }
        )

    def _build_status_from_repositories(
        self,
        repositories: list[dict],
        checking: bool,
        updating: bool,
        last_error: str,
        last_checked_at: Optional[str],
    ):
        update_signature = self._build_update_signature(repositories)
        current_progress = self.get_status().get("progress", self._build_empty_progress())
        return self._merge_status_metadata(
            {
                "checking": checking,
                "updating": updating,
                "has_update": any(repo["has_update"] and not repo["error"] for repo in repositories),
                "last_checked_at": last_checked_at,
                "last_error": last_error,
                "update_signature": update_signature,
                "repositories": repositories,
                "progress": current_progress if updating else self._build_empty_progress(),
            }
        )

    @staticmethod
    def _build_update_signature(repositories: list[dict]):
        changed_repositories = [
            f"{repo['path']}:{repo.get('local_commit', '')}:{repo.get('remote_commit', '')}"
            for repo in repositories
            if repo.get("has_update") and not repo.get("error")
        ]
        if not changed_repositories:
            return ""
        content = "|".join(sorted(changed_repositories)).encode("utf-8")
        import hashlib

        return hashlib.sha1(content).hexdigest()

    def _publish_status(self, status: dict):
        normalized_status = self._merge_status_metadata(status)
        with self._status_lock:
            self._status = deepcopy(normalized_status)
            snapshot = deepcopy(self._status)
        websocket_manager.broadcast_action_sync(
            WebsocketActions.ResourceUpdate.StatusChanged,
            WebSocketData(message=snapshot),
        )
        return snapshot

    def _merge_status_metadata(self, status: dict):
        metadata = self._build_status_metadata()
        return {
            **status,
            **metadata,
        }

    def _build_status_metadata(self):
        missing_required_resources = self.get_missing_required_resources()
        return {
            "enabled": self._is_periodic_check_enabled(),
            "check_on_startup": self._should_check_on_startup(),
            "check_period": self._get_check_period(),
            "interval_minutes": self._get_check_interval_seconds() // 60,
            "next_check_at": self._get_next_check_at_iso(),
            "required_resources_ready": not missing_required_resources,
            "bootstrap_required": bool(missing_required_resources),
            "missing_required_resources": missing_required_resources,
        }

    def _publish_progress(
        self,
        repository: Optional[ResourceRepository] = None,
        *,
        phase: str,
        title,
        message,
        current_step: int = 0,
        total_steps: int = 0,
        step_percent: float = 0.0,
        bytes_downloaded: int = 0,
        bytes_total: int = 0,
        attempt: int = 0,
        max_attempts: Optional[int] = None,
        retry_wait_seconds: int = 0,
        active: bool = True,
    ):
        total_steps = max(total_steps, 0)
        current_step = max(current_step, 0)
        step_percent = max(0.0, min(float(step_percent), 100.0))
        overall_percent = 0.0
        if total_steps > 0 and current_step > 0:
            completed_steps = max(current_step - 1, 0)
            overall_percent = ((completed_steps + (step_percent / 100.0)) / total_steps) * 100.0

        progress = {
            "active": active,
            "phase": phase,
            "title": title,
            "message": message,
            "repository": "" if repository is None else i18n_text(f"resource.repository.{repository.path.replace('/', '.')}", fallback=repository.name),
            "repository_path": "" if repository is None else repository.path,
            "current_step": current_step,
            "total_steps": total_steps,
            "step_percent": round(step_percent, 2),
            "percent": round(overall_percent, 2),
            "bytes_downloaded": int(max(bytes_downloaded, 0)),
            "bytes_total": int(max(bytes_total, 0)),
            "attempt": int(max(attempt, 0)),
            "max_attempts": int(max_attempts or self.DOWNLOAD_RETRY_LIMIT),
            "retry_wait_seconds": int(max(retry_wait_seconds, 0)),
        }
        current_status = self.get_status()
        return self._publish_status(
            {
                **current_status,
                "checking": False,
                "updating": active or current_status.get("updating", False),
                "progress": progress,
            }
        )

    def _get_check_period(self):
        return str(config_service.base.resource_update_check_period)

    def _get_check_interval_seconds(self):
        period = self._get_check_period()
        return {
            "daily": 24 * 60 * 60,
            "every_3_days": 3 * 24 * 60 * 60,
            "weekly": 7 * 24 * 60 * 60,
        }.get(period, 24 * 60 * 60)

    def _is_periodic_check_enabled(self):
        return bool(config_service.base.enabled_check_resource_updates)

    def _should_check_on_startup(self):
        return bool(config_service.base.check_resource_updates_on_startup)

    def _get_next_check_at(self):
        with self._status_lock:
            return self._next_check_at

    def _get_next_check_at_iso(self):
        next_check_at = self._get_next_check_at()
        if next_check_at is None:
            return None
        return next_check_at.isoformat(timespec="seconds")

    def _set_next_check_at(self, next_check_at: Optional[datetime]):
        with self._status_lock:
            self._next_check_at = next_check_at

    def _schedule_next_check(self, base_time: Optional[datetime] = None):
        if not self._is_periodic_check_enabled():
            self._set_next_check_at(None)
            return None
        if base_time is None:
            base_time = datetime.now()
        next_check_at = base_time + timedelta(seconds=self._get_check_interval_seconds())
        self._set_next_check_at(next_check_at)
        return next_check_at

    def _ensure_next_check_at(self):
        next_check_at = self._get_next_check_at()
        if next_check_at is not None:
            return next_check_at
        scheduled = self._schedule_next_check()
        if scheduled is None:
            raise RuntimeError("Periodic resource update check is not enabled")
        return scheduled

    def _inspect_repository(self, repository: ResourceRepository):
        active_path = self._get_repository_active_path(repository)
        mutable_path = self._get_repository_mutable_path(repository)
        status = {
            "name": i18n_text(f"resource.repository.{repository.path.replace('/', '.')}", fallback=repository.name),
            "path": repository.path,
            "exists": active_path.exists(),
            "dirty": False,
            "has_update": False,
            "local_commit": "",
            "remote_commit": "",
            "local_commit_short": "",
            "remote_commit_short": "",
            "local_branch": "",
            "remote_branch": "",
            "version_source": "",
            "update_method": "",
            "error": "",
        }
        try:
            update_method = self._select_update_method(str(mutable_path))
            if update_method == "git":
                local_version = self._read_local_git_version(str(mutable_path))
                remote_version = self._fetch_remote_git_version(repository, str(mutable_path))
                dirty = self._has_git_dirty_changes(str(mutable_path))
            else:
                local_version = self._read_local_version(repository, str(active_path))
                remote_version = self._fetch_remote_version(repository)
                dirty = False
            local_commit = local_version.get("commit", "")
            remote_commit = remote_version.get("commit", "")
            status.update(
                {
                    "dirty": dirty,
                    "local_commit": local_commit,
                    "remote_commit": remote_commit,
                    "local_commit_short": local_commit[:7],
                    "remote_commit_short": remote_commit[:7],
                    "local_branch": local_version.get("branch", ""),
                    "remote_branch": remote_version.get("branch", ""),
                    "version_source": local_version.get("source", ""),
                    "update_method": update_method,
                    "has_update": bool(remote_commit) and (not local_commit or local_commit != remote_commit),
                }
            )
        except Exception as exc:
            status["error"] = i18n_text("resource.progress.repositoryError", fallback=str(exc), error=str(exc))
        return status

    def _build_local_repository_status(
        self,
        repository: ResourceRepository,
        existing_status: Optional[dict] = None,
    ):
        active_path = self._get_repository_active_path(repository)
        mutable_path = self._get_repository_mutable_path(repository)
        update_method = self._select_update_method(str(mutable_path))
        if update_method == "git":
            local_version = self._read_local_git_version(str(mutable_path))
            dirty = self._has_git_dirty_changes(str(mutable_path))
        else:
            local_version = self._read_local_version(repository, str(active_path))
            dirty = False

        local_commit = local_version.get("commit", "")
        remote_commit = (
            (existing_status or {}).get("remote_commit")
            or local_commit
        )
        remote_branch = (
            (existing_status or {}).get("remote_branch")
            or local_version.get("branch", "")
        )
        return {
            "name": i18n_text(f"resource.repository.{repository.path.replace('/', '.')}", fallback=repository.name),
            "path": repository.path,
            "exists": active_path.exists(),
            "dirty": dirty,
            "has_update": bool(remote_commit) and local_commit != remote_commit,
            "local_commit": local_commit,
            "remote_commit": remote_commit,
            "local_commit_short": local_commit[:7],
            "remote_commit_short": remote_commit[:7],
            "local_branch": local_version.get("branch", ""),
            "remote_branch": remote_branch,
            "version_source": local_version.get("source", ""),
            "update_method": update_method,
            "error": i18n_text("resource.progress.noError", fallback=""),
        }

    def _update_repository_with_retry(
        self,
        repository: ResourceRepository,
        commit_sha: str,
        branch_name: str,
        *,
        current_step: int,
        total_steps: int,
    ):
        last_error = None
        for attempt in range(1, self.DOWNLOAD_RETRY_LIMIT + 1):
            try:
                self._update_repository(
                    repository,
                    commit_sha,
                    branch_name,
                    current_step=current_step,
                    total_steps=total_steps,
                    attempt=attempt,
                )
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Update resource repository '{repository.name}' attempt {attempt}/{self.DOWNLOAD_RETRY_LIMIT} failed: {exc}"
                )
                if attempt >= self.DOWNLOAD_RETRY_LIMIT:
                    break
                retry_wait_seconds = self.DOWNLOAD_RETRY_BASE_DELAY_SECONDS * attempt
                self._publish_progress(
                    repository,
                    phase="retry_waiting",
                    title=i18n_text("resource.progress.retrying", fallback=f"正在重试下载 {repository.name}", repository=repository.name),
                    message=i18n_text("resource.progress.retryingMessage", fallback=f"{repository.name} 下载失败，将在 {retry_wait_seconds} 秒后自动重试。", repository=repository.name, seconds=retry_wait_seconds),
                    current_step=current_step,
                    total_steps=total_steps,
                    step_percent=0,
                    attempt=attempt,
                    max_attempts=self.DOWNLOAD_RETRY_LIMIT,
                    retry_wait_seconds=retry_wait_seconds,
                )
                sleep(retry_wait_seconds)
        raise RuntimeError(str(i18n_text("resource.progress.retryExceeded", fallback=f"已自动重试 {self.DOWNLOAD_RETRY_LIMIT} 次，最后一次错误：{last_error}", error=str(last_error))))

    def _update_repository(
        self,
        repository: ResourceRepository,
        commit_sha: str,
        branch_name: str,
        *,
        current_step: int,
        total_steps: int,
        attempt: int,
    ):
        repo_path = self._get_repository_mutable_path(repository)
        update_method = self._select_update_method(str(repo_path))
        if update_method == "git":
            self._publish_progress(
                repository,
                phase="git_fetch",
                title=i18n_text(
                    "resource.progress.updatingRepository",
                    fallback=f"正在更新 {repository.name}",
                    repository=repository.name,
                ),
                message=i18n_text(
                    "resource.progress.updatingRepositoryWithGit",
                    fallback=f"正在通过 Git 更新 {repository.name}。",
                    repository=repository.name,
                ),
                current_step=current_step,
                total_steps=total_steps,
                step_percent=10,
                attempt=attempt,
                max_attempts=self.DOWNLOAD_RETRY_LIMIT,
            )
            self._update_repository_with_git(repository, str(repo_path), branch_name)
            updated_version = self._read_local_git_version(str(repo_path))
            self._write_local_version(
                repository,
                updated_version.get("commit", ""),
                updated_version.get("branch", ""),
                source="git",
            )
            self._publish_progress(
                repository,
                phase="completed",
                title=i18n_text(
                    "resource.progress.repositoryUpdated",
                    fallback=f"{repository.name} 已更新",
                    repository=repository.name,
                ),
                message=i18n_text(
                    "resource.progress.repositorySynced",
                    fallback=f"{repository.name} 已同步到最新版本。",
                    repository=repository.name,
                ),
                current_step=current_step,
                total_steps=total_steps,
                step_percent=100,
                attempt=attempt,
                max_attempts=self.DOWNLOAD_RETRY_LIMIT,
            )
            logger.success(
                f"Updated resource repository by git: {repository.path} -> {updated_version.get('commit', '')[:7]}"
            )
            return
        if not commit_sha:
            remote_version = self._fetch_remote_version(repository)
            commit_sha = remote_version.get("commit", "")
            branch_name = remote_version.get("branch", "")
        if not commit_sha:
            raise RuntimeError(f"{repository.name} 缺少远端提交信息")
        with tempfile.TemporaryDirectory(prefix=f"{repository.name}_", dir=self._app.data_path) as workdir:
            staged_source = self._download_repository_snapshot(
                repository,
                commit_sha,
                workdir,
                current_step=current_step,
                total_steps=total_steps,
                attempt=attempt,
            )
            self._publish_progress(
                repository,
                phase="replace",
                title=i18n_text(
                    "resource.progress.installingRepository",
                    fallback=f"正在安装 {repository.name}",
                    repository=repository.name,
                ),
                message=i18n_text(
                    "resource.progress.writingRepository",
                    fallback=f"正在写入 {repository.name} 到本地资源目录。",
                    repository=repository.name,
                ),
                current_step=current_step,
                total_steps=total_steps,
                step_percent=92,
                attempt=attempt,
                max_attempts=self.DOWNLOAD_RETRY_LIMIT,
            )
            self._replace_repository_directory(repo_path, staged_source)
        self._write_local_version(repository, commit_sha, branch_name, source="snapshot")
        self._publish_progress(
            repository,
            phase="completed",
            title=i18n_text(
                "resource.progress.repositoryUpdated",
                fallback=f"{repository.name} 已更新",
                repository=repository.name,
            ),
            message=i18n_text(
                "resource.progress.repositoryUpdatedToLatest",
                fallback=f"{repository.name} 已更新到最新版本。",
                repository=repository.name,
            ),
            current_step=current_step,
            total_steps=total_steps,
            step_percent=100,
            attempt=attempt,
            max_attempts=self.DOWNLOAD_RETRY_LIMIT,
        )
        logger.success(f"Updated resource repository: {repository.path} -> {commit_sha[:7]}")

    def _download_repository_snapshot(
        self,
        repository: ResourceRepository,
        commit_sha: str,
        workdir: str,
        *,
        current_step: int,
        total_steps: int,
        attempt: int,
    ):
        remote = self._resolve_repository_remote(repository)
        self._publish_progress(
            repository,
            phase="download_prepare",
            title=i18n_text(
                "resource.progress.preparingRepositoryDownload",
                fallback=f"正在准备下载 {repository.name}",
                repository=repository.name,
            ),
            message=i18n_text(
                "resource.progress.preparingRepositoryDownloadMessage",
                fallback=f"正在准备从 {remote.url} 下载 {repository.name}。",
                repository=repository.name,
                url=remote.url,
            ),
            current_step=current_step,
            total_steps=total_steps,
            step_percent=5,
            attempt=attempt,
            max_attempts=self.DOWNLOAD_RETRY_LIMIT,
        )
        if remote.supports_github_api:
            return self._download_repository_snapshot_from_github(
                repository,
                remote,
                commit_sha,
                workdir,
                current_step=current_step,
                total_steps=total_steps,
                attempt=attempt,
            )
        if not self._git_executable:
            raise RuntimeError(
                f"{repository.name} 当前配置的仓库 URL 不是有效的 GitHub 仓库地址，且系统未安装 git：{remote.url}"
            )
        return self._download_repository_snapshot_with_git(
            repository,
            remote,
            commit_sha,
            workdir,
            current_step=current_step,
            total_steps=total_steps,
            attempt=attempt,
        )

    def _replace_repository_directory(self, target_path, staged_source: str):
        target_path = str(target_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        backup_root = tempfile.mkdtemp(prefix="resource_backup_", dir=self._app.data_path)
        backup_path = os.path.join(backup_root, os.path.basename(target_path))
        try:
            if os.path.exists(target_path):
                shutil.move(target_path, backup_path)
            shutil.move(staged_source, target_path)
        except Exception:
            if os.path.exists(target_path):
                shutil.rmtree(target_path, ignore_errors=True)
            if os.path.exists(backup_path):
                shutil.move(backup_path, target_path)
            shutil.rmtree(backup_root, ignore_errors=True)
            raise
        preserved_git = os.path.join(backup_path, ".git")
        if os.path.exists(preserved_git) and not os.path.exists(os.path.join(target_path, ".git")):
            try:
                shutil.move(preserved_git, os.path.join(target_path, ".git"))
            except Exception as exc:
                logger.warning(f"Preserve git metadata for '{target_path}' failed: {exc}")
        shutil.rmtree(backup_root, ignore_errors=True)

    def _download_repository_snapshot_from_github(
        self,
        repository: ResourceRepository,
        remote: ResolvedRepositoryRemote,
        commit_sha: str,
        workdir: str,
        *,
        current_step: int,
        total_steps: int,
        attempt: int,
    ):
        zip_url = f"https://api.github.com/repos/{remote.owner}/{remote.repo}/zipball/{commit_sha}"
        zip_path = os.path.join(workdir, f"{repository.name}.zip")
        response = self._session.get(
            zip_url,
            timeout=self.REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=True,
        )
        if not response.ok:
            raise self._build_repository_remote_error(
                repository,
                remote.url,
                "下载资源",
                self._format_http_error(response),
            )
        total_size = int(response.headers.get("Content-Length") or 0)
        downloaded_size = 0
        with open(zip_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=self.DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    file_obj.write(chunk)
                    downloaded_size += len(chunk)
                    step_percent = 10.0
                    if total_size > 0:
                        step_percent = 10.0 + min(downloaded_size / total_size, 1.0) * 70.0
                    self._publish_progress(
                        repository,
                        phase="downloading",
                        title=i18n_text(
                            "resource.progress.downloadingRepository",
                            fallback=f"正在下载 {repository.name}",
                            repository=repository.name,
                        ),
                        message=i18n_text(
                            "resource.progress.downloadingRepositoryFromGithub",
                            fallback=f"正在从 GitHub 下载 {repository.name} 资源包。",
                            repository=repository.name,
                        ),
                        current_step=current_step,
                        total_steps=total_steps,
                        step_percent=step_percent,
                        bytes_downloaded=downloaded_size,
                        bytes_total=total_size,
                        attempt=attempt,
                        max_attempts=self.DOWNLOAD_RETRY_LIMIT,
                    )
        extract_root = os.path.join(workdir, "extract")
        os.makedirs(extract_root, exist_ok=True)
        self._publish_progress(
            repository,
            phase="extracting",
            title=i18n_text(
                "resource.progress.extractingRepository",
                fallback=f"正在解压 {repository.name}",
                repository=repository.name,
            ),
            message=i18n_text(
                "resource.progress.extractingRepositoryMessage",
                fallback=f"正在解压 {repository.name} 资源包。",
                repository=repository.name,
            ),
            current_step=current_step,
            total_steps=total_steps,
            step_percent=85,
            bytes_downloaded=downloaded_size,
            bytes_total=total_size,
            attempt=attempt,
            max_attempts=self.DOWNLOAD_RETRY_LIMIT,
        )
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_root)
        entries = [
            os.path.join(extract_root, name)
            for name in os.listdir(extract_root)
            if not name.startswith("__MACOSX")
        ]
        if len(entries) == 1 and os.path.isdir(entries[0]):
            return entries[0]
        if entries:
            return extract_root
        raise RuntimeError(f"{repository.name} 下载包解压后为空")

    def _download_repository_snapshot_with_git(
        self,
        repository: ResourceRepository,
        remote: ResolvedRepositoryRemote,
        commit_sha: str,
        workdir: str,
        *,
        current_step: int,
        total_steps: int,
        attempt: int,
    ):
        clone_path = os.path.join(workdir, repository.name)
        self._publish_progress(
            repository,
            phase="downloading",
            title=i18n_text(
                "resource.progress.downloadingRepository",
                fallback=f"正在下载 {repository.name}",
                repository=repository.name,
            ),
            message=i18n_text(
                "resource.progress.downloadingRepositoryWithGit",
                fallback=f"正在通过 Git 下载 {repository.name} 资源包。",
                repository=repository.name,
            ),
            current_step=current_step,
            total_steps=total_steps,
            step_percent=40,
            attempt=attempt,
            max_attempts=self.DOWNLOAD_RETRY_LIMIT,
        )
        clone_args = ["clone", "--depth", "1"]
        try:
            remote_version = self._fetch_remote_git_version_from_target(remote.url)
        except Exception as exc:
            raise self._build_repository_remote_error(repository, remote.url, "检查仓库", exc) from exc
        target_branch = remote_version.get("branch", "")
        if target_branch:
            clone_args.extend(["--branch", target_branch])
        clone_args.extend([remote.url, clone_path])
        try:
            self._run_git(clone_args, timeout=600)
        except Exception as exc:
            raise self._build_repository_remote_error(repository, remote.url, "下载资源", exc) from exc
        if commit_sha:
            local_commit = self._run_git(["-C", clone_path, "rev-parse", "HEAD"])
            if local_commit != commit_sha:
                try:
                    self._run_git(["-C", clone_path, "fetch", "--depth", "1", remote.url, commit_sha], timeout=600)
                except Exception as exc:
                    raise self._build_repository_remote_error(repository, remote.url, "下载资源", exc) from exc
                self._run_git(["-C", clone_path, "checkout", "--detach", "FETCH_HEAD"], timeout=300)
        git_dir = os.path.join(clone_path, ".git")
        if os.path.isdir(git_dir):
            shutil.rmtree(git_dir, ignore_errors=True)
        elif os.path.isfile(git_dir):
            os.unlink(git_dir)
        return clone_path

    def _fetch_remote_version(self, repository: ResourceRepository):
        remote = self._resolve_repository_remote(repository)
        if not remote.supports_github_api:
            if not self._git_executable:
                raise RuntimeError(
                    f"{repository.name} 当前配置的仓库 URL 不是有效的 GitHub 仓库地址，且系统未安装 git：{remote.url}"
                )
            try:
                return self._fetch_remote_git_version_from_target(remote.url)
            except Exception as exc:
                raise self._build_repository_remote_error(repository, remote.url, "检查仓库", exc) from exc
        repo_url = f"https://api.github.com/repos/{remote.owner}/{remote.repo}"
        try:
            repo_info = self._get_json(repo_url)
            branch_name = repo_info.get("default_branch") or "main"
            commit_info = self._get_json(f"{repo_url}/commits/{branch_name}")
        except Exception as exc:
            raise self._build_repository_remote_error(repository, remote.url, "检查仓库", exc) from exc
        commit_sha = commit_info.get("sha", "")
        if not commit_sha:
            raise RuntimeError(f"{repository.name} 未返回默认分支最新提交")
        return {
            "commit": commit_sha,
            "branch": branch_name,
            "source": "github",
        }

    def _read_local_version(self, repository: ResourceRepository, repo_path: str):
        if self._can_use_git(repo_path):
            git_version = self._read_local_git_version(repo_path)
            if git_version:
                return git_version
        metadata = self._read_local_version_metadata(repository)
        if metadata:
            metadata.setdefault("source", "metadata")
            return metadata
        git_version = self._read_git_version(repo_path)
        if git_version:
            return git_version
        return {
            "commit": "",
            "branch": "",
            "source": "unknown",
        }

    def _read_local_version_metadata(self, repository: ResourceRepository):
        metadata_path = self._get_metadata_path(repository)
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return {
            "commit": str(data.get("commit", "")),
            "branch": str(data.get("branch", "")),
            "source": "metadata",
        }

    def _write_local_version(
        self,
        repository: ResourceRepository,
        commit_sha: str,
        branch_name: str,
        source: str = "metadata",
    ):
        metadata_path = self._get_metadata_path(repository)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "name": repository.name,
                    "path": repository.path,
                    "owner": repository.owner,
                    "repo": repository.repo,
                    "commit": commit_sha,
                    "branch": branch_name,
                    "source": source,
                    "updated_at": self._now_iso(),
                },
                file_obj,
                ensure_ascii=False,
                indent=2,
            )

    def _get_metadata_path(self, repository: ResourceRepository):
        return os.path.join(
            self._app.data_path,
            self.VERSION_STATE_DIR_NAME,
            f"{repository.name}.json",
        )

    def _select_update_method(self, repo_path: str):
        return "git" if self._can_use_git(repo_path) else "snapshot"

    def _can_use_git(self, repo_path: str):
        return bool(self._git_executable) and bool(self._resolve_git_dir(repo_path))

    def _read_local_git_version(self, repo_path: str):
        if not self._can_use_git(repo_path):
            return self._read_git_version(repo_path)
        commit_sha = self._run_git(["-C", repo_path, "rev-parse", "HEAD"])
        branch_name = self._run_git(["-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"])
        if branch_name == "HEAD":
            branch_name = self._read_remote_head_branch(repo_path)
        return {
            "commit": commit_sha,
            "branch": branch_name,
            "source": "git",
        }

    def _fetch_remote_git_version(self, repository: ResourceRepository, repo_path: str):
        remote = self._resolve_repository_remote(repository)
        try:
            return self._fetch_remote_git_version_from_target(remote.url, repo_path=repo_path)
        except Exception as exc:
            raise self._build_repository_remote_error(repository, remote.url, "检查仓库", exc) from exc

    def _fetch_remote_git_version_from_target(self, target: str, repo_path: str = ""):
        args = ["ls-remote", "--symref", target, "HEAD"]
        if repo_path:
            args = ["-C", repo_path, *args]
        output = self._run_git(args)
        branch_name = ""
        commit_sha = ""
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("ref:"):
                parts = line.split()
                if len(parts) >= 3 and parts[-1] == "HEAD":
                    branch_name = parts[1].split("/")[-1]
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[-1] == "HEAD":
                commit_sha = parts[0]
        if not commit_sha:
            raise RuntimeError("无法获取远端 HEAD")
        return {
            "commit": commit_sha,
            "branch": branch_name,
            "source": "git",
        }

    def _read_remote_head_branch(self, repo_path: str):
        try:
            output = self._run_git(["-C", repo_path, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"])
        except Exception:
            return ""
        return output.split("/", 1)[-1].strip()

    def _has_git_dirty_changes(self, repo_path: str):
        if not self._can_use_git(repo_path):
            return False
        return bool(self._run_git(["-C", repo_path, "status", "--porcelain"]))

    def _update_repository_with_git(
        self,
        repository: ResourceRepository,
        repo_path: str,
        remote_branch: str = "",
    ):
        remote = self._resolve_repository_remote(repository)
        try:
            self._run_git(["-C", repo_path, "fetch", "--prune", remote.url], timeout=300)
        except Exception as exc:
            raise self._build_repository_remote_error(repository, remote.url, "更新仓库", exc) from exc
        self._run_git(["-C", repo_path, "reset", "--hard", "FETCH_HEAD"], timeout=300)
        self._run_git(["-C", repo_path, "clean", "-fd"], timeout=300)

    @staticmethod
    def _is_configured_submodule(relative_path: str):
        gitmodules_path = resolve_runtime_str(".gitmodules")
        if not os.path.exists(gitmodules_path):
            return False
        with open(gitmodules_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()
        return f"path = {relative_path}" in content

    @staticmethod
    def _read_git_version(repo_path: str):
        git_dir = ResourceUpdateService._resolve_git_dir(repo_path)
        if not git_dir:
            return None
        head_path = os.path.join(git_dir, "HEAD")
        if not os.path.exists(head_path):
            return None
        with open(head_path, "r", encoding="utf-8") as file_obj:
            head_value = file_obj.read().strip()
        if not head_value:
            return None
        if head_value.startswith("ref: "):
            ref_name = head_value[5:].strip()
            commit_sha = ResourceUpdateService._read_git_ref(git_dir, ref_name)
            if not commit_sha:
                return None
            return {
                "commit": commit_sha,
                "branch": ref_name.split("/")[-1],
                "source": "git",
            }
        return {
            "commit": head_value,
            "branch": "",
            "source": "git",
        }

    @staticmethod
    def _resolve_git_dir(repo_path: str):
        git_entry = os.path.join(repo_path, ".git")
        if os.path.isdir(git_entry):
            return git_entry
        if not os.path.isfile(git_entry):
            return None
        with open(git_entry, "r", encoding="utf-8") as file_obj:
            first_line = file_obj.readline().strip()
        if not first_line.startswith("gitdir:"):
            return None
        git_dir = first_line.split(":", 1)[1].strip()
        if not os.path.isabs(git_dir):
            git_dir = os.path.abspath(os.path.join(repo_path, git_dir))
        return git_dir

    @staticmethod
    def _read_git_ref(git_dir: str, ref_name: str):
        ref_path = os.path.join(git_dir, *ref_name.split("/"))
        if os.path.exists(ref_path):
            with open(ref_path, "r", encoding="utf-8") as file_obj:
                return file_obj.read().strip()
        packed_refs_path = os.path.join(git_dir, "packed-refs")
        if not os.path.exists(packed_refs_path):
            return ""
        with open(packed_refs_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("^"):
                    continue
                try:
                    commit_sha, packed_ref = line.split(" ", 1)
                except ValueError:
                    continue
                if packed_ref == ref_name:
                    return commit_sha.strip()
        return ""

    def _get_repository_by_path(self, relative_path: str):
        repository = next(
            (repo for repo in self.RESOURCE_REPOSITORIES if repo.path == relative_path),
            None,
        )
        if repository is None:
            raise KeyError(f"Unknown resource repository path: {relative_path}")
        return repository

    def _get_json(self, url: str):
        response = self._session.get(url, timeout=self.REQUEST_TIMEOUT)
        if not response.ok:
            raise RuntimeError(self._format_http_error(response))
        return response.json()

    def _run_git(self, args: list[str], timeout: int = 30):
        if not self._git_executable:
            raise RuntimeError("未找到 git，可执行文件")
        result = subprocess.run(
            [self._git_executable, *args],
            cwd=str(get_runtime_root()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            message = stderr or stdout or f"git {' '.join(args)} 执行失败"
            raise RuntimeError(message)
        return (result.stdout or "").strip()

    @staticmethod
    def _format_http_error(response: requests.Response):
        message = ""
        try:
            data = response.json()
            message = data.get("message", "")
        except Exception:
            message = response.text.strip()
        message = message or f"HTTP {response.status_code}"
        return f"GitHub 请求失败（{response.status_code}）：{message}"
