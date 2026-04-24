"""过场展示 / 效果链 handler。

处理稳定 gameplay 阶段之间出现的各类展示和过渡画面:
  - 奖励展示 (P点+103, 技能卡获得, P饮料获得)
  - メモリー効果（回忆效果）
  - 角色立绘过场
  - 達成 / CLEAR 覆盖展示
  - レッスン结算
  - 支援事件效果
  - 活動支給 奖励链

这些画面通常只需点击即可推进到链中的下一帧。
本 handler 检测到无法识别的画面，但 HUD 元素表明仍处于
游戏中，则通过点击推进。

优先级较低 — 仅在无其他 handler 匹配且 HUD 表明仍在 produce 流程中时激活。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import GameplayPhase, TRANSITION_POSITIONS
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.shared.common import click_relative_point
from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayHandler,
    HandlerResult,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_HUD_LABELS = (
    ProducerLabels.PC_PROGRESS,
    ProducerLabels.PC_TRAINING_SCORE,
    ProducerLabels.PC_TRAINING_REMAINING,
    ProducerLabels.PC_STAMINA,
    ProducerLabels.PC_P_POINT,
    ProducerLabels.PC_TARGET,
)


def _has_hud_elements(app: "AppProcessor") -> bool:
    """检查是否可见 HUD 元素 — 表明当前处于活跃 gameplay 中。"""
    results = app.latest_results
    if results is None:
        return False
    return any(results.exists_label(label) for label in _HUD_LABELS)


class EffectChainHandler(GameplayHandler):
    """推进效果链 / 过渡 / 展示画面。

    当 phase 为 unknown 但 position 或 HUD 元素表明仍在 produce 流程中时激活。
    通过 ``ctx.handler_state["effect_chain_depth"]`` 跟踪连续推进次数，
    以便监控异常长链。
    """

    phase_tag = GameplayPhase.UNKNOWN
    priority = 10  # 高于 fallback AdvanceHandler，低于所有常规 handler

    def can_handle(self, app, ctx, phase, position):
        """判断当前画面是否应由该处理器接管。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        if phase != GameplayPhase.UNKNOWN:
            return False
        return position in TRANSITION_POSITIONS or _has_hud_elements(app)

    def handle(self, app, ctx, phase, position):
        """执行处理器主逻辑并返回处理结果。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            返回执行结果对象，具体类型见函数注解。
        """
        depth = ctx.handler_state.get("effect_chain_depth", 0) + 1
        ctx.handler_state["effect_chain_depth"] = depth

        logger.debug(f"effect_chain: advancing (depth={depth}, position={position})")

        # 优先点击检测到的按钮（如结果画面的“次へ”）。
        results = app.latest_results
        confirm_btn = results.filter_by_label(ProducerLabels.CONFIRM_BUTTON)
        if confirm_btn:
            app.device.click_element(confirm_btn.first())
            return HandlerResult.ok("effect_chain advance (confirm)", sleep_after=1.0)

        # 其次点击通用按钮（取最靠下的按钮）
        buttons = results.filter_by_label(ProducerLabels.BUTTON)
        if buttons:
            bottom_btn = max(buttons, key=lambda b: b.cy)
            app.device.click_element(bottom_btn)
            return HandlerResult.ok("effect_chain advance (button)", sleep_after=1.0)

        # 无按钮时点击屏幕中央推进过场动画
        click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="effect-chain-advance")
        return HandlerResult.ok("effect_chain advance", sleep_after=1.0)
