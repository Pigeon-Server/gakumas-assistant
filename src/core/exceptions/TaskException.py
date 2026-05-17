from typing import TYPE_CHECKING

from src.utils.i18n_tools import I18nText, serialize_i18n_value

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


class TaskUserMessage(BaseException):
    """
    受控中止当前任务，并向前端展示一条用户可读消息。

    与普通异常不同，这类异常不应触发失败 dump / 失败包，
    而是由 TaskService 直接广播 message 后结束任务。
    """

    def __init__(self, message: I18nText | str | dict, task: "Task" = None):
        self.task = task
        self.message = serialize_i18n_value(message)

    def __str__(self):
        if isinstance(self.message, dict):
            return self.message.get("fallback") or self.message.get("key", "")
        return str(self.message)
