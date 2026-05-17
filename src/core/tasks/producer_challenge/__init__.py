from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.pipeline import ProducePipeline
from src.core.tasks.producer_challenge.steps import (
    NavigateToProduceStep,
    SelectScenarioStep,
    SelectDifficultyStep,
    SelectIdolCardStep,
    SelectSupportCardsStep,
    SelectMemoriesStep,
    CollectMemoryAttributesStep,
    CollectFormationDetailsStep,
    ConfirmAndStartStep,
    HandleStartupModalsStep,
    ProduceGameplayLoopStep,
    HandleResultsStep,
)


def build_produce_pipeline() -> ProducePipeline:
    """构建完整的培育流程流水线（从导航到培育结束返回主页）"""
    return ProducePipeline([
        # 从主页导航到剧本选择页
        NavigateToProduceStep(),
        # 选择剧本（初 / NIA）
        SelectScenarioStep(),
        # 选择难度（Regular/Pro/Master/Legend, NIA Pro/Master）
        SelectDifficultyStep(),
        # 选择偶像卡
        SelectIdolCardStep(),
        # 支援卡编成
        SelectSupportCardsStep(),
        # 记忆编成（含レンタル复选框同步）
        SelectMemoriesStep(),
        # 采集记忆卡属性（編成詳細 → メモリー Tab）
        CollectMemoryAttributesStep(),
        # 采集完整编成详情
        CollectFormationDetailsStep(),
        # 处理加成道具 → 点击プロデュース開始
        ConfirmAndStartStep(),
        # 处理启动弹窗（语音/快进/跳过设置）→ 切换 PRODUCER 模型
        HandleStartupModalsStep(),
        # 培育主循环（行程选择/对话/レッスン/試験/P饮料）
        ProduceGameplayLoopStep(),
        # 结果画面处理 → 切回 BASE_UI 模型 → 返回主页
        HandleResultsStep(),
    ])


__all__ = [
    "ProduceContext",
    "ProducePipeline",
    "build_produce_pipeline",
]