"""策略洞察数据类 — 替代 dict 传递。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.strategy_insight import ProduceStrategyInsight


@dataclass
class InsightData:
    """策略洞察的数据载体，替代原来在各模块间传递的 dict。"""

    id: int = 0
    insight_type: str = ""
    scenario: str = ""
    phase: str = ""
    position: str = ""
    strategy_description: str = ""
    setup_pattern: str = ""
    when_to_apply: str = ""
    when_not_to_apply: str = ""
    decision_family: str = ""
    idol_plan_type: str = ""
    parameter_priority: str = ""
    next_gate_key: str = ""
    route_bias_key: str = ""
    resource_pressure_key: str = ""
    build_plan_key: str = ""
    buff_pattern: str = ""
    outcome_type: str = ""
    outcome_notes: str = ""
    validation_status: str = "draft"
    evidence_count: int = 0
    similarity_reasons: list[str] = field(default_factory=list)

    def to_store_dict(self) -> dict[str, object]:
        """转为 InsightStore.save_insight() 接受的字典。"""
        return {
            "insight_type": self.insight_type,
            "scenario": self.scenario,
            "phase": self.phase,
            "position": self.position,
            "strategy_description": self.strategy_description,
            "setup_pattern": self.setup_pattern,
            "when_to_apply": self.when_to_apply,
            "when_not_to_apply": self.when_not_to_apply,
            "decision_family": self.decision_family,
            "idol_plan_type": self.idol_plan_type,
            "parameter_priority": self.parameter_priority,
            "next_gate_key": self.next_gate_key,
            "route_bias_key": self.route_bias_key,
            "resource_pressure_key": self.resource_pressure_key,
            "build_plan_key": self.build_plan_key,
            "buff_pattern": self.buff_pattern,
            "outcome_type": self.outcome_type,
            "outcome_notes": self.outcome_notes,
            "session_id": "",
        }

    @staticmethod
    def from_row(row: "ProduceStrategyInsight") -> InsightData:
        """从 peewee model row 构建。"""
        return InsightData(
            id=int(row.id),
            insight_type=str(row.insight_type),
            scenario=str(row.scenario),
            phase=str(row.phase),
            strategy_description=str(row.strategy_description),
            setup_pattern=str(row.setup_pattern),
            when_to_apply=str(row.when_to_apply),
            when_not_to_apply=str(row.when_not_to_apply),
            decision_family=str(row.decision_family),
            idol_plan_type=str(row.idol_plan_type),
            next_gate_key=str(row.next_gate_key),
            outcome_type=str(row.outcome_type),
            validation_status=str(row.validation_status),
            evidence_count=int(row.evidence_count),
        )


@dataclass
class InsightSelectResult:
    """第一轮调用的结果：选中的洞察 ID 列表。"""

    selected_ids: list[int] = field(default_factory=list)
    raw_response: str = ""
