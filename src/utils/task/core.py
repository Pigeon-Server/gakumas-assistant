from queue import Queue
from typing import Callable, List, Dict

class TaskQueue:
    """任务队列管理器"""
    def __init__(self):
        self.queue = Queue()
        self.enabled_tasks = set()
        self.running = False

    def add_task(self, task_name: str, *args):
        """向任务队列添加任务"""
        if task_name in self.enabled_tasks:
            self.queue.put((task_name, args))

    def process_tasks(self, processor):
        """按顺序处理任务"""
        while not self.queue.empty() and self.running:
            task_name, args = self.queue.get()
            if task_name in processor.task_registry:
                func, _ = processor.task_registry[task_name]
                if not func(processor, *args):
                    logger.warning(f"任务 {task_name} 执行失败")

    def start(self):
        """启动任务队列"""
        self.running = True

    def stop(self):
        """停止任务队列"""
        self.running = False

    def clear(self):
        """清空任务队列"""
        with self.queue.mutex:
            self.queue.queue.clear()

    def enable_task(self, task_name: str):
        """启用任务"""
        self.enabled_tasks.add(task_name)

    def disable_task(self, task_name: str):
        """禁用任务"""
        self.enabled_tasks.discard(task_name)

    def get_enabled_tasks(self) -> List[str]:
        """查询已启用任务"""
        return list(self.enabled_tasks)

    def get_registered_tasks(self) -> Dict[str, str]:
        """查询所有已注册任务"""
        return {name: desc for name, (_, desc) in AppProcessor.task_registry.items()}