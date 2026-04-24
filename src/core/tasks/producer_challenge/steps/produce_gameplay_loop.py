"""Step 11: 培育主循环 — 基于 handler 调度器的可扩展 gameplay loop。

这是培育过程中最核心的循环步骤，处理从第一周行程选择到
生产结束之间的所有游戏交互。

当前支持的画面类型（handler 可通过 dispatcher 动态注册/替换）:
  - schedule:     周行程选择画面 → 选择/确认行程
  - dialogue:     对话/剧情选项 → 双击选项/快进
  - lesson:       レッスン → 手牌选择/出牌
  - exam:         試験/オーディション → 手牌选择（与 lesson 共用机制）
  - skill_reward: 技能卡奖励选择 → 选卡→确认
  - p_drink:      P饮料选择 → 选饮料→确认
  - consult:      相談交换页 → 交换/強化/削除
  - modal:        弹窗 → 确认关闭
  - effect_chain: 过场展示（奖励/メモリー/角色过场）→ 点击推进
  - result:       结果画面 → 退出循环

扩展方式:
  - 新增 handler: 在 gameplay/ 下创建模块，注册到 dispatcher
  - 替换 handler: dispatcher.unregister() + register()
  - 调整优先级: 修改 handler 的 priority 属性
"""

from time import sleep
from typing import TYPE_CHECKING

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge.context import GameplayPhase
from src.core.tasks.producer_challenge.gameplay.common import click_relative_point
from src.core.tasks.producer_challenge.gameplay import build_default_dispatcher
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    detect_gameplay_phase,
    get_pipeline_position,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


def _try_back_button_recovery(app: "AppProcessor") -> bool:
    """检测并点击 Back Button 以恢复误入的子画面（手牌库等）。

    当画面处于 UNKNOWN 且检测到 Back Button 时调用。
    Returns:
        True 表示已点击 Back Button。
    """
    results = app.latest_results
    if results is None:
        return False
    back_buttons = list(results.filter_by_label(BaseUILabels.BACK_BTN))
    if back_buttons:
        logger.info("recovery: 检测到 Back Button，点击返回")
        app.device.click_element(back_buttons[0])
        return True
    return False


class ProduceGameplayLoopStep(ProduceStep):
    """定义 ProduceGameplayLoopStep 的结构化数据。
    """
    step_name = "produce_gameplay_loop"

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """执行当前步骤的完整流程。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        logger.info("进入培育主循环")
        ctx.set_phase(GameplayPhase.SCHEDULE)
        ctx.last_pipeline_step = self.step_name

        dispatcher = build_default_dispatcher()
        max_unknown = 20         # 连续 unknown 阈值（触发安全退出）
        total_actions = 0        # 总操作次数

        while total_actions < ctx.max_gameplay_loops:
            sleep(0.8)

            phase = detect_gameplay_phase(app, ctx)
            position = get_pipeline_position(app, ctx)
            ctx.set_phase(phase)
            ctx.set_position(position)
            logger.debug(f"[Loop {total_actions}] phase={phase}, position={position}")

            # 连续无法识别画面安全阈值
            if ctx.consecutive_unknowns >= max_unknown:
                logger.error(f"连续 {max_unknown} 次无法识别画面，安全退出循环")
                raise RuntimeError("培育主循环: 连续无法识别画面阈值超出")

            # 分发到对应 handler
            result = dispatcher.dispatch(app, ctx, phase, position)

            if result.status == "exit":
                logger.info(f"主循环退出: {result.detail}")
                return True

            if result.status == "unhandled":
                logger.warning(f"无 handler 匹配: phase={phase}, position={position}")
                # 优先尝试 Back Button 恢复（可能误入手牌库等子画面）
                if _try_back_button_recovery(app):
                    sleep(1.0)
                else:
                    click_relative_point(app, x_ratio=0.5, y_ratio=0.35, label="unhandled-advance")
                    sleep(1.0)
            elif result.sleep_after > 0:
                sleep(result.sleep_after)

            total_actions += 1

        raise RuntimeError(f"培育主循环: 达到最大循环次数 {ctx.max_gameplay_loops}")
