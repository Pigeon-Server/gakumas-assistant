import ctypes
import inspect
import os
import traceback
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from queue import Queue
from typing import Callable, Optional, TYPE_CHECKING, List
from threading import Thread, Event, Lock, current_thread
from time import time, sleep

try:
    from pyautogui import FailSafeException
except Exception:
    class FailSafeException(Exception):
        pass

from src.constants.task_status import TaskStatus
from src.constants.websocket_actions import WebsocketActions
from src.core.device.windows_compat import is_windows_device
from src.core.exceptions.TaskException import UserCancelTask, TaskTimeout, TaskUserMessage
from src.core.services.config_service import ConfigService
from src.core.web.websocket import WebSocketManager
from src.entity.Task import Task
from src.entity.WebSocketData import WebSocketData
from src.utils.debug_tools import DebugTools
from src.utils.i18n_tools import i18n_text, serialize_i18n_value
from src.utils.logger import logger
from src.utils.runtime_paths import resolve_runtime_str
from src.utils.task_debug_tools import (
    bind_task_debug_context,
    clear_task_debug_context,
    record_task_step,
    reset_task_debug_trace,
)
from src.utils.task_dumper import dump_task_failure
from src.utils.task_failure_package import (
    GITHUB_ISSUE_URL,
    QQ_GROUP_NUMBER,
    build_task_failure_package,
    register_task_failure_package_download,
)

if TYPE_CHECKING:
    from src.main import AppProcessor

config_service = ConfigService()
websocket_manager = WebSocketManager()
debug_tools = DebugTools()


def _raise_in_thread(thread_id: int, exc_type: type):
    """
    向目标线程注入异常（CPython API）。
    线程会在下一条 Python 字节码执行时抛出该异常。
    配合 BaseException 子类使用，可穿透 except Exception 块。
    """
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(exc_type)
    )
    if ret == 0:
        logger.warning(f"Thread {thread_id} not found (may have already exited)")
    elif ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
        raise SystemError("PyThreadState_SetAsyncExc affected multiple threads")

class TaskService:
    AUTO_STARTUP_TIME_FORMAT = "%H:%M"

    def __init__(self, app: "AppProcessor"):
        self._app = app
        self._task_queue: Queue = Queue()
        self._task_list: List[Task] = []
        self._task_pre_function: list[Callable] = []
        self._task_middleware_function: list[Callable] = []
        self._worker_thread: Optional[Thread] = None
        self._current_running_task: Optional[Task] = None
        self._suspend_target_task: Optional[Task] = None
        self._queue_status: bool = False
        # 线程安全信号
        self._stop_event = Event()
        self._resume_event = Event()
        self._resume_event.set()  # 初始为非挂起
        self._suspended = Event()
        self._task_runner_thread: Optional[Thread] = None
        self._auto_startup_refresh_event = Event()
        self._auto_startup_state_lock = Lock()
        self._auto_startup_next_run: Optional[datetime] = None
        self._init_auto_startup_scheduler()

    # ========== 注册 ==========

    def register_task(
            self,
            task_id: str,
            task_name: str,
            timeout: int | None = None,
            disabled_middleware: bool = False,
            manual_only: bool = False,
            hide: bool = False,
            allow_manual_suspend: bool = False,
            allow_manual_resume: bool = False
    ):
        """注册任务"""
        def decorator(func: Callable):
            if self._find_task(task_id):
                raise RuntimeError(f"Duplicate task name: '{task_id}'")
            enable = task_id not in config_service.base.disabled_tasks
            self._task_list.append(Task(
                id=task_id,
                task_name=task_name,
                enable=enable,
                disabled_middleware=disabled_middleware,
                manual_only=manual_only,
                hide=hide,
                allow_manual_suspend=allow_manual_suspend,
                allow_manual_resume=allow_manual_resume,
                function=func,
                timeout=timeout
            ))
            logger.debug(f"register task: {task_name}({task_id})")
        return decorator

    def register_pre_queue_start(self):
        """注册任务队列执行前预执行"""
        def decorator(func: Callable):
            if func not in self._task_pre_function:
                logger.debug(f"register task pre function: {func.__name__}")
                self._task_pre_function.append(func)
        return decorator

    def register_task_middleware(self):
        """注册任务中间件"""
        def decorator(func: Callable):
            if func not in self._task_middleware_function:
                logger.debug(f"register middleware: {func.__name__}")
                self._task_middleware_function.append(func)
        return decorator

    # ========== 队列控制 ==========

    def start_queue(self, task_id: str = None):
        """启动任务队列"""
        if self._queue_status:
            return False
        if not self._app.is_resource_ready():
            logger.warning("Task queue start rejected: required game resources are not ready")
            return False
        if not self._app.ensure_device_ready(restart_inference=True):
            logger.warning(f"Task queue start rejected: {self._app.get_device_status().get('message', 'device unavailable')}")
            return False
        self._queue_status = True
        self._stop_event.clear()
        self._resume_event.set()
        self._suspended.clear()
        self._suspend_target_task = None
        debug_tools.clear_all()
        with self._task_queue.mutex:
            self._task_queue.queue.clear()
        if task_id:
            if task := self._find_task(task_id):
                task.update_status(TaskStatus.PENDING)
                self._task_queue.put(task)
            else:
                logger.warning(f"Task '{task_id}' not found.")
                self._queue_status = False
                return False
        else:
            for task in self._get_enable_tasks():
                if not task.manual_only:
                    task.update_status(TaskStatus.PENDING)
                    self._task_queue.put(task)
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = Thread(target=self._processor_task_queue, daemon=True)
            self._worker_thread.start()
        else:
            logger.error("Worker thread is already alive.")
            self._queue_status = False
            return False
        return True

    def start_queue_from(self, task_id: str):
        """从指定任务开始启动后续自动任务队列"""
        if self._queue_status:
            return False
        if not self._app.is_resource_ready():
            logger.warning("Task queue start from rejected: required game resources are not ready")
            return False
        if not self._app.ensure_device_ready(restart_inference=True):
            logger.warning(
                f"Task queue start from rejected: {self._app.get_device_status().get('message', 'device unavailable')}"
            )
            return False
        start_index = -1
        for index, task in enumerate(self._task_list):
            if task.id == task_id:
                start_index = index
                break
        if start_index < 0:
            logger.warning(f"Task '{task_id}' not found.")
            return False
        start_task = self._task_list[start_index]
        if start_task.manual_only:
            logger.warning(f"Task '{task_id}' is manual only and cannot start queue from here.")
            return False

        self._queue_status = True
        self._stop_event.clear()
        self._resume_event.set()
        self._suspended.clear()
        self._suspend_target_task = None
        debug_tools.clear_all()
        with self._task_queue.mutex:
            self._task_queue.queue.clear()

        queued = False
        for task in self._task_list[start_index:]:
            if not task.enable or task.manual_only:
                continue
            task.update_status(TaskStatus.PENDING)
            self._task_queue.put(task)
            queued = True
        if not queued:
            logger.warning(f"No executable task found from '{task_id}'.")
            self._queue_status = False
            return False
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = Thread(target=self._processor_task_queue, daemon=True)
            self._worker_thread.start()
        else:
            logger.error("Worker thread is already alive.")
            self._queue_status = False
            return False
        return True

    def stop(self):
        """停止任务队列"""
        if not self._queue_status:
            return False
        self._stop_event.set()
        # 如果处于挂起，先唤醒让线程能响应异常
        self._resume_event.set()
        self._suspended.clear()
        # 向任务线程注入 UserCancelTask（BaseException 子类，穿透 except Exception）
        if self._task_runner_thread is not None and self._task_runner_thread.is_alive():
            _raise_in_thread(self._task_runner_thread.ident, UserCancelTask)
        with self._task_queue.mutex:
            self._task_queue.queue.clear()
        return True

    def _broadcast_runtime_snapshot(self):
        try:
            self._app.broadcast_app_status()
        except Exception as exc:
            logger.debug(f"Skip app status broadcast from task service: {exc}")

    def suspend_running_task(self, update_status: bool = True) -> Task:
        """挂起当前正在运行的任务"""
        if self._suspended.is_set():
            raise RuntimeError("Task already suspended")
        target = self._current_running_task
        if not target or target.status != TaskStatus.RUNNING:
            target = next((t for t in self._task_list if t.status == TaskStatus.RUNNING), None)
        if not target:
            raise RuntimeError("No running task to suspend")

        def _apply_suspend_state(applied_event: Optional[Event] = None):
            try:
                self._suspend_target_task = target
                self._suspended.set()
                self._resume_event.clear()  # trace 中的 wait() 将阻塞任务线程
                target.update_status(TaskStatus.SUSPENDED)
                self._current_running_task = None
                if update_status:
                    websocket_manager.broadcast_action_sync(WebsocketActions.TaskService.TaskQueueSuspend)
                self._broadcast_runtime_snapshot()
                logger.debug(f"Suspend task: {target.task_name}({target.id})")
            finally:
                if applied_event is not None:
                    applied_event.set()

        if self._task_runner_thread is not None and current_thread().ident == self._task_runner_thread.ident:
            applied_event = Event()
            Thread(target=_apply_suspend_state, args=(applied_event,), daemon=True).start()
            applied_event.wait()
        else:
            _apply_suspend_state()
        return target

    def request_suspend_running_task(self, update_status: bool = True) -> Task:
        """
        由任务代码主动请求挂起当前任务。
        保留该接口兼容旧调用点，语义与 `suspend_running_task()` 相同：
        当前执行流会在调用点原地挂起，恢复后继续向下执行。
        """
        return self.suspend_running_task(update_status=update_status)

    def resume_suspended_task(self):
        """恢复被挂起的任务"""
        if not self._suspended.is_set():
            raise RuntimeError("Current not task suspended")
        target = self._suspend_target_task
        self._suspended.clear()
        self._suspend_target_task = None
        if self._task_runner_thread is not None and self._task_runner_thread.is_alive():
            self._current_running_task = target
            target.update_status(TaskStatus.RUNNING)
        else:
            self._current_running_task = None
        self._resume_event.set()  # 唤醒任务线程
        websocket_manager.broadcast_action_sync(WebsocketActions.TaskService.TaskQueueStart)
        self._broadcast_runtime_snapshot()
        logger.debug(f"Resume task: {target.task_name}({target.id})")

    def insert_task_to_run_queue(self, task_id: str):
        """插队执行：挂起当前 -> 执行新任务 -> 恢复旧任务"""
        if self._suspended.is_set():
            logger.error("System is already suspended, cannot insert.")
            return False
        insert_task = self._find_task(task_id)
        if not insert_task:
            logger.error(f"Task '{task_id}' does not exist.")
            return False
        logger.debug(f"Insert task: {insert_task.task_name}({insert_task.id})")
        try:
            self.suspend_running_task()
        except RuntimeError as e:
            logger.error(f"Try suspend current running task error: {e}")
            return False

        def _insert_runner():
            try:
                self._task_thread(insert_task)
            finally:
                logger.debug("Resume suspended task")
                self.resume_suspended_task()

        self._current_running_task = insert_task
        self._broadcast_runtime_snapshot()
        Thread(target=_insert_runner, daemon=True).start()
        return True

    def _init_auto_startup_scheduler(self):
        """初始化每日自动执行调度器。"""
        config_service.add_listener(
            ["base.enabled_auto_startup", "base.auto_startup_time"],
            self._on_auto_startup_config_changed,
        )
        self._auto_startup_thread = Thread(target=self._auto_startup_scheduler_loop, daemon=True)
        self._auto_startup_thread.start()

    def _on_auto_startup_config_changed(self, key: str, old_value, new_value):
        logger.info(f"Reload auto startup schedule because '{key}' changed: {old_value!r} -> {new_value!r}")
        self._auto_startup_refresh_event.set()

    @classmethod
    def _parse_auto_startup_time(cls, value: str):
        return datetime.strptime(value, cls.AUTO_STARTUP_TIME_FORMAT)

    @classmethod
    def _get_next_auto_startup_datetime(cls, now: datetime, time_text: str) -> datetime:
        schedule_time = cls._parse_auto_startup_time(time_text)
        next_run = now.replace(
            hour=schedule_time.hour,
            minute=schedule_time.minute,
            second=0,
            microsecond=0,
        )
        current_minute = now.replace(second=0, microsecond=0)
        if next_run < current_minute:
            next_run += timedelta(days=1)
        return next_run

    def _set_auto_startup_next_run(self, next_run: Optional[datetime]):
        with self._auto_startup_state_lock:
            self._auto_startup_next_run = next_run

    def get_auto_startup_next_run(self) -> Optional[datetime]:
        with self._auto_startup_state_lock:
            return self._auto_startup_next_run

    def _wait_for_auto_startup_refresh(self, timeout: float) -> bool:
        if self._auto_startup_refresh_event.wait(timeout=max(timeout, 0)):
            self._auto_startup_refresh_event.clear()
            return True
        return False

    def _auto_startup_scheduler_loop(self):
        """每日自动执行任务队列。"""
        while True:
            if not config_service.base.enabled_auto_startup:
                self._set_auto_startup_next_run(None)
                self._wait_for_auto_startup_refresh(60)
                continue
            try:
                next_run = self._get_next_auto_startup_datetime(
                    datetime.now(),
                    config_service.base.auto_startup_time,
                )
            except ValueError:
                self._set_auto_startup_next_run(None)
                logger.error(
                    f"Invalid auto startup time: {config_service.base.auto_startup_time!r}, "
                    f"expected format {self.AUTO_STARTUP_TIME_FORMAT}"
                )
                self._wait_for_auto_startup_refresh(60)
                continue
            self._set_auto_startup_next_run(next_run)
            logger.info(f"Next auto startup scheduled at {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            wait_seconds = (next_run - datetime.now()).total_seconds()
            if self._wait_for_auto_startup_refresh(wait_seconds):
                continue
            self._set_auto_startup_next_run(None)
            if self._queue_status:
                logger.warning("Skip auto startup because task queue is already running")
                continue
            logger.info("Auto startup triggered, starting task queue")
            if not self.start_queue():
                logger.warning("Auto startup failed to start task queue")

    # ========== 内部实现 ==========

    # 项目 src 目录的绝对路径，用于 trace 白名单过滤
    _src_path: str = resolve_runtime_str("src")
    _self_file: str = os.path.abspath(__file__)

    MIDDLEWARE_WHITELIST = [
        Path("src/core/tasks").as_posix(),
        Path("src/core/services/game_utils.py").as_posix(),
    ]

    @staticmethod
    @lru_cache(maxsize=256)
    def _should_trace(filename: str) -> bool:
        """
        判断是否需要在该文件中响应挂起。
        只对 src/ 下的代码生效，排除 task_service 自身和 logger。
        """
        if TaskService._src_path not in filename:
            return False
        if filename == TaskService._self_file:
            return False
        if "logger" in filename:
            return False
        return True

    @staticmethod
    @lru_cache(maxsize=256)
    def _is_middleware_code(filename: str) -> bool:
        """判断文件是否在中间件白名单路径中。"""
        posix_path = Path(filename).as_posix()
        return any(wl in posix_path for wl in TaskService.MIDDLEWARE_WHITELIST)

    def _make_trace(self, task: Task, timeout_event: Event):
        """
        轻量 trace，两个职责：
        1. 挂起时阻塞任务线程（白名单文件中 Event.wait()）
        2. 中间件拦截（仅在 tasks/game_utils 路径下，中间件返回 False 时阻塞）
        急停仍由 PyThreadState_SetAsyncExc 处理，超时由 watchdog 设置事件后在 trace 安全点抛出。
        """
        resume_event = self._resume_event
        should_trace = self._should_trace
        is_middleware = self._is_middleware_code
        exec_middleware = self._exec_task_middleware
        disabled_middleware = task.disabled_middleware

        def _wait_and_extend_while_blocked(condition, step=0.2):
            """
            当 condition() 为 True 时阻塞，并把等待时间补偿到运行时超时。
            中间件返回 False 时属于“系统等待”而非业务执行，不应吞掉任务 timeout。
            """
            added = 0.0
            while condition():
                if timeout_event.is_set():
                    raise TaskTimeout(task)
                resume_event.wait()
                sleep(step)
                added += step
            if added > 0:
                task.extend_timeout(added)

        def _trace(frame, event, arg):
            filename = frame.f_code.co_filename
            if not should_trace(filename):
                return _trace
            if event != "line":
                return _trace
            if timeout_event.is_set():
                raise TaskTimeout(task)
            # 挂起检查
            resume_event.wait()
            # 中间件拦截：仅在白名单路径中执行，返回 False 时阻塞任务
            if not disabled_middleware and is_middleware(filename):
                _wait_and_extend_while_blocked(lambda: not exec_middleware(), step=0.2)
            return _trace

        return _trace

    def _watchdog(self, task: Task, timeout_event: Event):
        """
        看门狗线程：周期性检查超时、Windows焦点、执行中间件。
        与任务线程完全解耦，不侵入任务代码。
        """
        suspend_start: Optional[float] = None

        while not self._stop_event.is_set():
            if task.status not in (TaskStatus.RUNNING, TaskStatus.SUSPENDED):
                break

            # 挂起状态：记录时长用于超时补偿
            if self._suspended.is_set() and self._suspend_target_task == task:
                if suspend_start is None:
                    suspend_start = time()
                sleep(0.2)
                continue
            else:
                if suspend_start is not None:
                    elapsed = time() - suspend_start
                    task.extend_timeout(elapsed)
                    suspend_start = None

            # 超时检查
            task_timeout = task.get_timeout()
            start_time = task.get_start_time()
            if task_timeout not in (None, -1) and start_time not in (None, -1, None):
                if (time() - start_time) > task_timeout:
                    logger.warning(f"Task '{task.task_name}' timed out")
                    timeout_event.set()
                    break

            # Windows 焦点检查
            if is_windows_device(self._app.device):
                if not self._app.device.is_app_focused():
                    focus_lost_start = time()
                    while not self._app.device.is_app_focused():
                        if self._stop_event.is_set():
                            return
                        sleep(0.1)
                    waited = time() - focus_lost_start
                    if waited > 0:
                        task.extend_timeout(waited)

            sleep(0.5)

    def _exec_task_middleware(self):
        """执行中间件"""
        flag: bool = True
        for func in self._task_middleware_function:
            if func(self._app) is False:
                logger.debug(f"{func.__name__} return false")
                flag = False
        return flag

    def _processor_task_queue(self):
        """任务队列处理器"""
        websocket_manager.broadcast_action_sync(WebsocketActions.TaskService.TaskQueueStart)
        self._broadcast_runtime_snapshot()
        try:
            for func in self._task_pre_function:
                if func() is False:
                    raise RuntimeError(f"Pre-function {func.__name__} returned False")
        except Exception as e:
            logger.error(f"Task queue pre-check failed: {e}")
            websocket_manager.broadcast_action_sync(WebsocketActions.TaskService.TaskQueueStop)
            self._queue_status = False
            self._app.yolo_engine.pause()
            self._broadcast_runtime_snapshot()
            return

        try:
            while not self._task_queue.empty():
                if self._stop_event.is_set():
                    break
                task = self._task_queue.get()
                self._current_running_task = task
                self._broadcast_runtime_snapshot()
                logger.info(f"Run task: {task.task_name}({task.id})")
                self._task_thread(task)
                if task.status == TaskStatus.FAILED:
                    logger.warning(f"Task {task.task_name}({task.id}) failed, clearing queue.")
                    with self._task_queue.mutex:
                        self._task_queue.queue.clear()
                    break
        finally:
            websocket_manager.broadcast_action_sync(WebsocketActions.TaskService.TaskQueueStop)
            self._queue_status = False
            self._app.yolo_engine.pause()
            self._current_running_task = None
            self._broadcast_runtime_snapshot()
            logger.debug("Task queue processor exited")

    def _task_thread(self, task: Task):
        """执行单个任务：启动任务线程 + watchdog 线程"""
        task.reset_runtime_state()
        task.update_start_time()
        task.update_status(TaskStatus.RUNNING)
        reset_task_debug_trace(self._app, task.id)
        record_task_step(
            self._app,
            "task.run.start",
            task_id=task.id,
            task_name=task.task_name,
            timeout=task.get_timeout(),
        )
        timeout_event = Event()
        runner = Thread(target=self._run_task_inner, args=(task, timeout_event), daemon=True)
        self._task_runner_thread = runner
        watchdog = Thread(target=self._watchdog, args=(task, timeout_event), daemon=True)
        runner.start()
        watchdog.start()
        runner.join()
        self._task_runner_thread = None

    def _broadcast_task_execution_error(
        self,
        task: Task,
        exception: BaseException,
        dump_dir: str | None,
        package_path: str | None,
        package_download: dict | None = None,
    ):
        package_download = package_download or {}
        websocket_manager.broadcast_action_sync(
            WebsocketActions.TaskService.TaskExecutionError,
            WebSocketData(message={
                "task_id": task.id,
                "task_name": i18n_text(
                    f"backend.task.names.{task.id}",
                    fallback=task.task_name,
                ),
                "status": task.status,
                "error_type": type(exception).__name__,
                "error_message": getattr(exception, "message", None) or str(exception),
                "dump_dir": dump_dir or "",
                "package_path": package_path or "",
                "package_id": package_download.get("package_id", ""),
                "package_download_url": package_download.get("download_url", ""),
                "feedback": {
                    "github_issues": GITHUB_ISSUE_URL,
                    "qq_group": QQ_GROUP_NUMBER,
                },
            }),
        )

    def _handle_task_failure(self, task: Task, exception: BaseException):
        dump_dir = dump_task_failure(self._app, task, exception)
        package_path = build_task_failure_package(self._app, task, dump_dir, exception)
        package_download = register_task_failure_package_download(package_path) if package_path else None
        self._broadcast_task_execution_error(
            task,
            exception,
            dump_dir,
            package_path,
            package_download=package_download,
        )

    def _handle_task_failure_with_cancel_guard(self, task: Task, exception: BaseException) -> bool:
        """
        处理失败收尾，并兜底拦截用户在收尾阶段触发的取消信号。

        返回 True 表示失败收尾完整执行；
        返回 False 表示收尾过程中被取消，调用方应将任务状态改为 CANCELED。
        """
        try:
            self._handle_task_failure(task, exception)
            return True
        except (UserCancelTask, FailSafeException):
            return False

    def _run_task_inner(self, task: Task, timeout_event: Event):
        """任务实际执行（在独立线程中，供 PyThreadState_SetAsyncExc 定位）"""
        import sys
        bind_task_debug_context(self._app, task.id, task.task_name)
        try:
            sys.settrace(self._make_trace(task, timeout_event))
            func = task.function
            sig = inspect.signature(func)
            result = func(self._app) if len(sig.parameters) > 0 else func()
            sys.settrace(None)
            if timeout_event.is_set():
                raise TaskTimeout(task)
            if result in TaskStatus.__dict__.keys():
                task.update_status(result)
            elif result is False:
                task.update_status(TaskStatus.CANCELED)
            else:
                task.update_status(TaskStatus.SUCCESS)
        except (UserCancelTask, FailSafeException):
            sys.settrace(None)
            task.update_status(TaskStatus.CANCELED)
            record_task_step(self._app, "task.run.canceled", task_id=task.id)
            logger.warning(f"Task '{task.task_name}({task.id})' cancelled")
        except TaskUserMessage as e:
            sys.settrace(None)
            task.update_status(TaskStatus.CANCELED)
            record_task_step(
                self._app,
                "task.run.user_message",
                task_id=task.id,
                error_type=type(e).__name__,
                error=str(e),
            )
            self._broadcast_task_execution_error(
                task,
                e,
                None,
                None,
            )
            logger.warning(f"Task '{task.task_name}({task.id})' stopped with user message: {e}")
        except TaskTimeout as e:
            timeout_event.clear()
            sys.settrace(None)
            task.update_status(TaskStatus.FAILED)
            record_task_step(
                self._app,
                "task.run.timeout",
                task_id=task.id,
                error_type=type(e).__name__,
                error=str(e),
            )
            logger.exception(f"Task '{task.task_name}({task.id})' timed out")
            if not self._handle_task_failure_with_cancel_guard(task, e):
                task.update_status(TaskStatus.CANCELED)
                record_task_step(self._app, "task.run.canceled", task_id=task.id)
                logger.warning(f"Task '{task.task_name}({task.id})' cancelled")
        except Exception as e:
            sys.settrace(None)
            task.update_status(TaskStatus.FAILED)
            record_task_step(
                self._app,
                "task.run.exception",
                task_id=task.id,
                error_type=type(e).__name__,
                error=str(e),
            )
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()
            logger.exception(f"Task '{task.task_name}({task.id})' failed:\n{tb_str}")
            if not self._handle_task_failure_with_cancel_guard(task, e):
                task.update_status(TaskStatus.CANCELED)
                record_task_step(self._app, "task.run.canceled", task_id=task.id)
                logger.warning(f"Task '{task.task_name}({task.id})' cancelled")
        finally:
            task.update_end_time()
            record_task_step(
                self._app,
                "task.run.finish",
                task_id=task.id,
                status=task.status,
            )
            sys.settrace(None)
            clear_task_debug_context()
            logger.info(f"Task '{task.task_name}({task.id})' status: {task.status}")

    # ========== 查询方法 ==========

    def _find_task(self, task_id: str):
        return next((task for task in self._task_list if task.id == task_id), None)

    def _get_enable_tasks(self):
        return [task for task in self._task_list if task.enable]

    def _get_task_names(self):
        return [task.id for task in self._task_list]

    def get_current_running_task(self) -> Task:
        return self._current_running_task

    def get_current_suspend_task(self) -> Task:
        return self._suspend_target_task

    def get_task_list(self):
        return {
            task.id: {
                "description": i18n_text(
                    f"backend.task.names.{task.id}",
                    fallback=task.task_name,
                ),
                "enable": task.enable,
                "last_run_time": task.last_run_time,
                "start_time": task.get_start_time(),
                "status": task.status,
                "manual_only": task.manual_only,
                "allow_manual_resume": task.allow_manual_resume,
                "allow_manual_suspend": task.allow_manual_suspend,
            }
            for task in self._task_list if task.hide is False
        }

    def disable_task(self, task_id: str):
        if task := self._find_task(task_id):
            task.enable = False
            return True
        return False

    def enable_task(self, task_id: str):
        if task := self._find_task(task_id):
            task.enable = True
            return True
        return False

    def queue_status(self):
        if self._queue_status:
            if self._suspended.is_set():
                return TaskStatus.SUSPENDED
            return TaskStatus.RUNNING
        return TaskStatus.PENDING
