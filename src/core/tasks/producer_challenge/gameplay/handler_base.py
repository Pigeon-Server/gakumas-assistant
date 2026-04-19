"""可扩展的 gameplay handler 基础设施。

架构:
  - GameplayHandler: 所有阶段 handler 的抽象基类。
  - HandlerResult: handler 执行后的返回类型。
  - GameplayDispatcher: 将 phase 路由到对应 handler 的注册中心。

新增 gameplay 阶段的步骤（如 NIA オーディション）:
  1. （可选）在 `src/constants/game/producer_gameplay.py` 中添加阶段值
  2. （可选）在 ui.py 的 classify_gameplay_phase() 中添加检测规则
  3. 在 gameplay/ 下新建模块，继承 GameplayHandler
  4. 在 __init__.py 的 build_default_dispatcher() 中注册

每个 handler 必须实现:
  - can_handle(app, ctx, phase, position) -> bool
  - handle(app, ctx, phase, position) -> HandlerResult

调度器按 priority 从高到低尝试所有 handler，
委托给第一个 can_handle() 返回 True 的 handler。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.core.tasks.producer_challenge.shared.common import click_relative_point

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


# ────────────────────────────────────────────────────────────
# Handler 返回值
# ────────────────────────────────────────────────────────────

@dataclass
class HandlerResult:
    """handler 执行后的返回值。

    状态:
      ok        — 操作成功执行
      waiting   — 暂无操作，下次循环重试
      exit      — gameplay 循环应当终止（如结果画面）
      no_action — handler 匹配但无法执行（无候选元素等）
      unhandled — 没有 handler 匹配
    """
    status: str
    detail: str = ""
    sleep_after: float = 0.5

    @staticmethod
    def ok(detail: str = "", sleep_after: float = 0.5) -> "HandlerResult":
        return HandlerResult("ok", detail, sleep_after)

    @staticmethod
    def waiting(detail: str = "", sleep_after: float = 1.0) -> "HandlerResult":
        return HandlerResult("waiting", detail, sleep_after)

    @staticmethod
    def exit(detail: str = "") -> "HandlerResult":
        return HandlerResult("exit", detail, 0.0)

    @staticmethod
    def no_action(detail: str = "", sleep_after: float = 0.8) -> "HandlerResult":
        return HandlerResult("no_action", detail, sleep_after)

    @staticmethod
    def unhandled() -> "HandlerResult":
        return HandlerResult("unhandled", "", 0.0)


# ────────────────────────────────────────────────────────────
# Handler 抽象基类
# ────────────────────────────────────────────────────────────

class GameplayHandler(ABC):
    """所有 gameplay 阶段 handler 的抽象基类。

    子类需覆盖 ``phase_tag`` 和 ``priority``。

    推荐优先级范围:
      95   结果画面 / 退出检测
      90   弹窗覆盖层
      50   常规 gameplay 阶段
      10   过场效果链
      -100 兜底（点击推进）
    """

    phase_tag: str = ""
    priority: int = 50

    @abstractmethod
    def can_handle(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        phase: str,
        position: str,
    ) -> bool:
        """判断此 handler 是否应处理当前画面，返回 True 表示匹配。"""
        ...

    @abstractmethod
    def handle(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        phase: str,
        position: str,
    ) -> HandlerResult:
        """执行操作。仅在 can_handle() 返回 True 时调用。"""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} phase={self.phase_tag!r} priority={self.priority}>"


# ────────────────────────────────────────────────────────────
# 调度器
# ────────────────────────────────────────────────────────────

class GameplayDispatcher:
    """将 gameplay 画面帧路由到已注册 handler 的调度器。

    按 priority 从高到低依次尝试 handler，
    第一个 can_handle() 返回 True 的 handler 处理该帧。

    用法::

        dispatcher = GameplayDispatcher()
        dispatcher.register(ScheduleHandler())
        dispatcher.register(DialogueHandler())
        result = dispatcher.dispatch(app, ctx, phase, position)
    """

    def __init__(self) -> None:
        self._handlers: List[GameplayHandler] = []

    def register(self, handler: GameplayHandler) -> "GameplayDispatcher":
        """注册 handler，按 priority 降序重新排序。"""
        self._handlers.append(handler)
        self._handlers.sort(key=lambda h: -h.priority)
        return self

    def unregister(self, handler_type: type) -> "GameplayDispatcher":
        """移除指定类型的所有 handler。"""
        self._handlers = [h for h in self._handlers if not isinstance(h, handler_type)]
        return self

    def dispatch(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        phase: str,
        position: str,
    ) -> HandlerResult:
        """找到第一个匹配的 handler 并执行。"""
        for handler in self._handlers:
            if handler.can_handle(app, ctx, phase, position):
                # 非 modal 阶段成功处理时，重置 modal 卡住计数器
                if handler.phase_tag != "modal":
                    ctx.handler_state.pop("modal_stuck_count", None)
                result = handler.handle(app, ctx, phase, position)
                return result
        return HandlerResult.unhandled()

    @property
    def handlers(self) -> List[GameplayHandler]:
        """当前已注册的 handler（按 priority 排序）。"""
        return list(self._handlers)


# ────────────────────────────────────────────────────────────
# 内置通用 handler
# ────────────────────────────────────────────────────────────

class ResultHandler(GameplayHandler):
    """检测到结果画面时标记培育进入收尾阶段。

    收尾分两阶段：
      1. produce_finishing_pending: 已决定结束培育，但仍在 PRODUCER 模型下处理
         （考试结果、LIVE演出、记忆生成等）
      2. produce_finishing: 到达记忆卡面选择等后期通用 UI 页面，切 BASE_UI 推进回主页
    """

    phase_tag = GameplayPhase.RESULT
    priority = 95

    # 到达这些位置时切换到 BASE_UI 收尾（记忆卡面选择及之后的通用 UI）
    _BASEUI_POSITIONS = {
        GameplayPosition.RESULT_MEMORY_PAGE,
        GameplayPosition.RESULT_REWARD_SUMMARY,
        GameplayPosition.RESULT_ACHIEVEMENT_PROGRESS,
        GameplayPosition.RESULT_EVENT_REWARD_PROGRESS,
    }

    # 这些位置说明培育已结束，但还在 PRODUCER 可处理的阶段（点击推进即可）
    _PENDING_POSITIONS = {
        GameplayPosition.RESULT_FINAL_EVALUATION,
        GameplayPosition.RESULT_MEMORY_GENERATION,
    }

    def can_handle(self, app, ctx, phase, position):
        return phase == GameplayPhase.RESULT

    def handle(self, app, ctx, phase, position):
        # 到达记忆卡面选择等后期页面 → 切 BASE_UI 收尾
        if position in self._BASEUI_POSITIONS:
            ctx.handler_state["produce_finishing"] = True
            logger.info("result: 到达 {} → 标记 produce_finishing，将切换 BASE_UI", position)
            return HandlerResult.ok(f"result ({position}) → produce_finishing", sleep_after=0.5)

        # 培育已结束的中间页面（最终评价、记忆生成等）→ 标记 pending，继续 PRODUCER 推进
        if position in self._PENDING_POSITIONS:
            ctx.handler_state["produce_finishing_pending"] = True
            logger.info("result: 到达 {} → 标记 produce_finishing_pending，继续 PRODUCER 推进", position)

        # 所有非 BASE_UI 的结果页面：优先点击 Confirm 按钮，否则点中心推进
        from src.constants.yolo.labels.producer_Labels import ProducerLabels
        confirm_boxes = list(app.latest_results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
        if confirm_boxes:
            box = confirm_boxes[0]
            app.device.click(box.cx, box.cy, "result-confirm")
        else:
            click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="result-advance")
        # 结果页后常有切页动画 / 对话过渡，给更长的 unknown 重试
        ctx.handler_state["unknown_retry_override"] = {
            "reason": "result_midgame_transition",
            "retry_limit": 10,
            "retry_sleep": 1.0,
        }
        return HandlerResult.ok(f"result ({position}) → advance", sleep_after=0.8)


class AdvanceHandler(GameplayHandler):
    """兜底 handler：点击屏幕中央推进未知/加载画面。

    最低优先级 — 仅在无其他 handler 匹配时激活。
    """

    phase_tag = ""
    priority = -100

    def can_handle(self, app, ctx, phase, position):
        return True  # catch-all

    def handle(self, app, ctx, phase, position):
        from src.utils.logger import logger
        logger.debug(f"advance: tap to progress (phase={phase}, position={position})")
        click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="advance")
        return HandlerResult.ok("advance tap", sleep_after=1.0)
