from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


class ProduceStep(ABC):
    """培育流程步骤基类，约定 validate / execute 两阶段接口。"""

    step_name: str = "unnamed_step"
    skip_on_resume: bool = False  # 恢复中断模式下是否跳过此步骤

    @abstractmethod
    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """执行当前步骤的主体逻辑。

        Args:
            app: 当前应用处理器，用于执行点击、OCR、YOLO 检测和页面等待。
            ctx: 在整条培育流水线中共享的上下文对象，步骤可从中读取配置并写回结果。

        Returns:
            bool: 返回 True 表示步骤成功完成；返回 False 或抛出异常都会让流水线中断。
        """
        ...

    def validate(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """检查当前步骤的前置条件是否满足。

        Args:
            app: 当前应用处理器，可用于读取当前页面状态。
            ctx: 当前培育上下文，可用于判断上一步是否已经写入必要数据。

        Returns:
            bool: 默认总是返回 True；子类可覆盖为更严格的页面或上下文校验。
        """
        return True

    def __repr__(self):
        """返回步骤对象的简要调试表示，便于日志定位当前执行节点。"""
        return f"<{self.__class__.__name__} step_name={self.step_name!r}>"
