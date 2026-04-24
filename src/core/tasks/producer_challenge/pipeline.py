from typing import TYPE_CHECKING, List

from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


class ProducePipeline:
    """
    培育流程流水线。

    按顺序执行一组 ProduceStep，在它们之间传递共享的 ProduceContext。
    任意一步的 validate 或 execute 失败都会中止流水线并抛出异常。
    """

    def __init__(self, steps: List[ProduceStep] | None = None):
        """初始化流水线，并登记需要依次执行的步骤列表。

        Args:
            steps: 预先组装好的步骤列表；传入 None 时会创建空流水线，
                由调用方后续通过 add_step 逐步追加。
        """
        self.steps: List[ProduceStep] = steps or []

    def add_step(self, step: ProduceStep) -> "ProducePipeline":
        """向流水线尾部追加一个步骤，并返回自身以支持链式组装。

        Args:
            step: 需要加入流水线的 ProduceStep 实例。该步骤会在 run 中
                按追加顺序执行，并共享同一个 ProduceContext。

        Returns:
            ProducePipeline: 当前流水线对象，便于连续调用 add_step。
        """
        self.steps.append(step)
        return self

    def run(self, app: "AppProcessor", ctx: "ProduceContext"):
        """按顺序执行整条培育流水线。

        Args:
            app: 当前设备与识别能力的统一入口，供每个步骤执行点击、OCR、
                YOLO 检测和页面等待等操作。
            ctx: 在整条流水线中共享的培育上下文。步骤会持续向其中写入
                选择结果、运行期状态和断点恢复信息。

        Raises:
            RuntimeError: 任一步骤的 validate 失败，或 execute 返回 False 时抛出，
                用于立即中断培育流程并把失败点暴露给上层调用方。

        Notes:
            - 当 ctx.resumed_from_interrupt 为 True 且步骤声明 skip_on_resume 时，
              该步骤会被跳过，避免恢复中断培育时重复执行编成相关步骤。
            - 每一步开始、完成、跳过都会写入日志，便于排查卡死点。
        """
        total = len(self.steps)
        for idx, step in enumerate(self.steps, 1):
            tag = f"[{idx}/{total}] {step.step_name}"

            # 恢复中断模式下跳过编成相关步骤
            if getattr(ctx, "resumed_from_interrupt", False) and getattr(step, "skip_on_resume", False):
                logger.info(f"{tag} — 恢复中断模式，跳过")
                continue

            if not step.validate(app, ctx):
                raise RuntimeError(f"{tag} — 前置条件检查失败")

            logger.info(f"{tag} — 开始执行")
            if not step.execute(app, ctx):
                raise RuntimeError(f"{tag} — 执行失败")
            logger.success(f"{tag} — 完成")

        logger.success("ProducePipeline 全部步骤执行完毕")
