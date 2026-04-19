from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

class UserCancelTask(BaseException):
    """
    继承 BaseException 而非 Exception，
    使其能穿透任务代码中的 except Exception 块，
    确保急停信号不会被业务代码意外吞掉。
    """
    def __init__(self, task: "Task" = None):
        self.task = task

    def __str__(self):
        if self.task:
            return f"User cancel task {self.task.id}"
        return "User cancel task"

class TaskTimeout(BaseException):
    """同 UserCancelTask，继承 BaseException 确保不被 except Exception 捕获。"""
    def __init__(self, task: "Task" = None):
        self.task = task

    def __str__(self):
        if self.task:
            return f"Task '{self.task.id}' execution timed out."
        return "Task execution timed out."
