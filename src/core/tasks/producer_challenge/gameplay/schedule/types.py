from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScheduleActionCandidate:
    """定义 ScheduleActionCandidate 的结构化数据。

    Attributes:
        index: 候选项在当前列表中的序号（通常从上到下或从左到右）。
        title: 候选项主标题文本，通常来自 OCR 或预设文案。
        kind: 候选项类型标签，用于规则筛选与优先级判断。
        recommended: 是否与周行动预览提示的属性类型一致（True 表示一致）。
        selected: 是否为当前已选中项（True 表示已选中）。
        box: 候选项对应的检测框，用于点击、裁剪和可视化调试。
        action_id: 标准化动作标识，用于在决策层与执行层之间关联同一操作。
        db_id: 数据库中的实体 ID；为空通常表示当前候选项尚未完成实体识别。
        source: 候选项来源标记（如 OCR、DB、fallback），便于排查识别链路。
        confidence: 当前识别或匹配结果的置信度，数值越高代表结果越可靠。
        metadata: 扩展元数据，保存额外识别信息与决策辅助字段。
    """
    index: int
    title: str
    kind: str
    recommended: bool
    selected: bool
    box: Any = field(repr=False, default=None)
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleStepResult:
    """定义 ScheduleStepResult 的结构化数据。

    Attributes:
        status: 步骤执行状态（如 selected/confirmed/skipped）。
        candidate: 本步骤最终选中的候选项对象。
    """
    status: str
    candidate: ScheduleActionCandidate
