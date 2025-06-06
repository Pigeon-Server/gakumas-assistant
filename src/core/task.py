import sys
from dataclasses import dataclass
from functools import partial
from queue import Queue
from typing import Callable
from threading import Thread
from time import time

from src.utils.logger import logger

class TaskStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

@dataclass
class Task:
    name: str
    description: str
    enable: bool
    function: Callable
    timeout: int
    status: str = TaskStatus.PENDING
    _start_time: int | None = -1
    _end_time: int | None = -1
    last_run_time: float = 0

    def __init__(self, name: str, description: str, enable: bool, function: Callable, timeout: int = 30):
        """
        :param name: 任务名
        :param description: 任务介绍
        :param enable: 是否启用
        :param function: 方法
        :param timeout: 任务超时事件
        """
        self.name = name
        self.description = description
        self.enable = enable
        self.function = function
        self.timeout = timeout

    def update_start_time(self):
        self._start_time = int(time())
        self.last_run_time = self._start_time
        self._end_time = None
        return self._start_time

    def update_end_time(self):
        self._end_time = int(time())
        return self._end_time

    def get_start_time(self):
        return self._start_time

class TaskQueue:
    _app = None
    _task_queue = Queue()
    _task_list = []
    _run_lock: bool = False
    _worker_thread: Thread

    def __init__(self, app):
        self._app = app
        self._worker_thread = Thread(target=self._processor_task_queue, daemon=True)
        self._worker_thread.start()

    def reg_task(self, task_name: str, task_description: str, task_func: Callable, timeout: int | None = None):
        """
        注册任务
        """
        if self._find_task(task_name):
            raise RuntimeError(f"Duplicate task name: {task_name}")
        self._task_list.append(Task(task_name, task_description, True, task_func, timeout))

    def exec_task(self):
        """执行任务"""
        if self._run_lock:
            return False  # 如果已有任务在执行，则不重复启动
        self._run_lock = True
        logger.debug("start exec task queue")
        if not self._task_queue.empty():
            self._task_queue.queue.clear()
        for task in self._get_enable_tasks():
            self._task_queue.put(task)
        if not self._worker_thread.is_alive():
            self._worker_thread = Thread(target=self._processor_task_queue, daemon=True)
            self._worker_thread.start()
        return True

    def _processor_task_queue(self):
        """任务队列处理器，确保任务按顺序执行"""
        while True:
            if self._task_queue.empty():
                self._run_lock = False
                logger.debug("[Exit]Task queue is empty")
                break

            task = self._task_queue.get()
            logger.info(f"Run task: {task.name}")
            self._task_thread(task)

    def _trace_calls(self, frame, event, arg, task: Task):
        if task.timeout and task.timeout != -1 and task.get_start_time() - int(time()) > task.timeout:
            task.status = TaskStatus.FAILED
            raise TimeoutError(f"Task {task.name} execution timed out.")
        if frame.f_code.co_name != task.function.__name__:
            return partial(self._trace_calls, task=task)
        # elif event == 'line':
        #     lineno = frame.f_lineno
        #     filename = frame.f_globals["__file__"]
        #     logger.debug(f"[LINE] {filename}:{lineno}")
        return partial(self._trace_calls, task=task)

    def _task_thread(self, task: Task):
        """执行任务"""
        logger.debug(f"Executing task: {task.name}")

        def wrapper():
            try:
                task.status = TaskStatus.RUNNING
                sys.settrace(partial(self._trace_calls, task=task))
                if task.function(self._app) is False:
                    task.status = TaskStatus.FAILED
                else:
                    task.status = TaskStatus.SUCCESS
            except Exception as e:
                task.status = TaskStatus.FAILED
                self._task_queue.queue.clear()
                # logger.error(f"Task {task.name} failed: {e}")
                raise RuntimeError(f"Task {task.name} failed.")
            finally:
                task.update_end_time()
                sys.settrace(None)

        worker = Thread(target=wrapper, daemon=True)
        worker.start()
        worker.join()

    def _find_task(self, task_name: str):
        """查找任务"""
        return next((task for task in self._task_list if task.name == task_name), None)

    def _get_enable_tasks(self):
        """获取启用的任务"""
        return [task for task in self._task_list if task.enable]

    def get_task_list(self):
        """获取所有任务列表"""
        return [
            {
                task.name: {
                "description": task.description,
                "enable": task.enable,
                "last_run_time": task.last_run_time,
                "start_time": task.get_start_time(),
                "status": task.status,
                }
            }
            for task in self._task_list
        ]

    def disable_task(self, task_name: str):
        """禁用任务"""
        if task := self._find_task(task_name):
            task.enable = False
            return True
        return False

    def enable_task(self, task_name: str):
        """启用任务"""
        if task := self._find_task(task_name):
            task.enable = True
            return True
        return False

    def stop(self):
        """停止任务队列"""
        if self._run_lock is False or self._task_queue.empty():
            return False
        # 清空任务队列
        self._task_queue.queue.clear()
        # 释放执行锁
        self._run_lock = False
        return True
