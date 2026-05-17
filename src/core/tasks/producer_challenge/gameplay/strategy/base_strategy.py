from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.core.tasks.producer_challenge.gameplay.strategy.algo_strategy_types import DecisionResult
    from src.main import AppProcessor


class ScheduleDecisionStrategy(ABC):
    """周行程决策策略基类。"""

    @abstractmethod
    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> "DecisionResult | None":
        pass


class BattleDecisionStrategy(ABC):
    """战斗决策策略基类（レッスン/試験）。"""

    @abstractmethod
    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> "DecisionResult | None":
        pass


class OtherDecisionStrategy(ABC):
    """其他决策策略基类（对话/P饮料/技能奖励/咨询/道具选择）。"""

    @abstractmethod
    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> "DecisionResult | None":
        pass
