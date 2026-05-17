"""策略洞察模型 — 存储 LLM 生成的抽象策略洞察。"""

from __future__ import annotations

from datetime import datetime

from peewee import (
    AutoField,
    DateTimeField,
    IntegerField,
    TextField,
)

from src.models.base import BaseModel


class ProduceStrategyInsight(BaseModel):
    """LLM 生成的抽象策略洞察。

    只存储提炼后的策略知识，不存储具体卡名/操作名。
    """

    id = AutoField(primary_key=True)

    # 分类
    insight_type = TextField(default="")        # "step" / "phase"
    scenario = TextField(default="")            # hajime / nia
    phase = TextField(default="")               # schedule / lesson / exam / ...
    position = TextField(default="")

    # 策略内容（LLM 生成的抽象描述）
    strategy_description = TextField(default="")
    setup_pattern = TextField(default="")
    when_to_apply = TextField(default="")
    when_not_to_apply = TextField(default="")

    # 战术层标签（用于检索匹配）
    decision_family = TextField(default="")
    idol_plan_type = TextField(default="")
    parameter_priority = TextField(default="")
    next_gate_key = TextField(default="")
    route_bias_key = TextField(default="")
    resource_pressure_key = TextField(default="")
    build_plan_key = TextField(default="")
    buff_pattern = TextField(default="")

    # 结果
    outcome_type = TextField(default="")
    outcome_notes = TextField(default="")

    # 生命周期管理
    validation_status = TextField(default="draft")
    evidence_count = IntegerField(default=0)
    contradiction_count = IntegerField(default=0)
    last_used_at = DateTimeField(null=True)
    last_outcome_type = TextField(default="")
    session_id = TextField(default="")
    created_at = DateTimeField(default=datetime.now)
    deleted_at = DateTimeField(null=True)

    class Meta:
        table_name = "produce_strategy_insights"
        indexes = (
            (("scenario", "phase", "decision_family"), False),
            (("validation_status",), False),
        )
