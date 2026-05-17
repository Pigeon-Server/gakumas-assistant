from typing import TYPE_CHECKING, List

from src.core.tasks.producer_challenge.gameplay.strategy.llm_strategy import LLMStrategy
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


def _flush_llm_session(ctx: "ProduceContext") -> None:
    """流水线异常退出时也尽量收尾整局 LLM 会话。"""
    strategy = ctx.schedule_strategy
    if isinstance(strategy, LLMStrategy):
        strategy.flush_session(ctx)


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

    def _resolve_resume_index(self, ctx: "ProduceContext") -> tuple[str, int]:
        """根据上下文里的恢复目标，解析当前应跳转到的步骤索引。"""
        resume_step = str(getattr(ctx, "resume_pipeline_step", "") or "").strip()
        if not getattr(ctx, "resumed_from_interrupt", False) or not resume_step:
            return "", -1

        for idx, step in enumerate(self.steps):
            if step.step_name == resume_step:
                return resume_step, idx

        logger.warning(f"恢复步骤 {resume_step!r} 不在当前流水线中，将从头执行")
        return resume_step, -1

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
        """
        total = len(self.steps)

        try:
            for idx, step in enumerate(self.steps, 1):
                tag = f"[{idx}/{total}] {step.step_name}"
                resume_step, resume_index = self._resolve_resume_index(ctx)

                if resume_index >= 0 and idx - 1 < resume_index:
                    logger.info(f"{tag} — 恢复到 {resume_step}，跳过前置步骤")
                    continue

                if not step.validate(app, ctx):
                    raise RuntimeError(f"{tag} — 前置条件检查失败")

                logger.info(f"{tag} — 开始执行")
                if not step.execute(app, ctx):
                    raise RuntimeError(f"{tag} — 执行失败")
                logger.success(f"{tag} — 完成")
                ctx.last_pipeline_step = step.step_name
        except Exception:
            _flush_llm_session(ctx)
            raise

        logger.success("ProducePipeline 全部步骤执行完毕")
