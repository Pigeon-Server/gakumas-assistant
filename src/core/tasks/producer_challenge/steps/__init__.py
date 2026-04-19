# ── entry：进入培育前的页面导航与剧本/难度选择 ──
from src.core.tasks.producer_challenge.steps.entry import (
    NavigateToProduceStep,
    SelectDifficultyStep,
    SelectScenarioStep,
)

# ── setup：正式开跑前的编成与开始确认 ──
from src.core.tasks.producer_challenge.steps.setup import (
    ConfirmAndStartStep,
    SelectIdolCardStep,
    SelectMemoriesStep,
    SelectSupportCardsStep,
)

# ── collect：开始前的详情采集 ──
from src.core.tasks.producer_challenge.steps.collect import (
    CollectFormationDetailsStep,
    CollectMemoryAttributesStep,
)

# ── runtime：进入培育后的启动弹窗与主循环 ──
from src.core.tasks.producer_challenge.steps.runtime import (
    HandleStartupModalsStep,
    ProduceGameplayLoopStep,
)

# ── finalize：培育结束后的结果链处理 ──
from src.core.tasks.producer_challenge.steps.finalize import HandleResultsStep

__all__ = [
    "NavigateToProduceStep",
    "SelectScenarioStep",
    "SelectDifficultyStep",
    "SelectIdolCardStep",
    "SelectSupportCardsStep",
    "SelectMemoriesStep",
    "CollectMemoryAttributesStep",
    "CollectFormationDetailsStep",
    "ConfirmAndStartStep",
    "HandleStartupModalsStep",
    "ProduceGameplayLoopStep",
    "HandleResultsStep",
]
