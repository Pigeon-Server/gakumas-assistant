"""策略洞察存储层 — ProduceStrategyInsight 的 CRUD + 检索。"""

from __future__ import annotations

from datetime import datetime

from src.models.strategy_insight import ProduceStrategyInsight
from src.core.tasks.producer_challenge.gameplay.llm.insight_data import InsightData


class InsightStore:
    """策略洞察的增删改查与相似度检索。直接使用全局 db。"""

    def save_insight(self, data: InsightData) -> int:
        row = ProduceStrategyInsight.create(**data.to_store_dict())
        return int(row.id)

    def update_insight(self, insight_id: int, updates: dict[str, object]) -> None:
        (
            ProduceStrategyInsight
            .update(**updates)
            .where(ProduceStrategyInsight.id == insight_id)
            .execute()
        )

    def soft_delete(self, insight_id: int) -> None:
        self.update_insight(insight_id, {"deleted_at": datetime.now()})

    def record_usage(self, insight_id: int, outcome_type: str) -> None:
        row = ProduceStrategyInsight.get_or_none(ProduceStrategyInsight.id == insight_id)
        if row is None:
            return
        row.evidence_count += 1
        row.last_used_at = datetime.now()
        row.last_outcome_type = outcome_type
        if outcome_type == "success":
            row.validation_status = self._promote_status(row.validation_status, row.evidence_count)
        row.save()

    def record_contradiction(self, insight_id: int) -> None:
        row = ProduceStrategyInsight.get_or_none(ProduceStrategyInsight.id == insight_id)
        if row is None:
            return
        row.contradiction_count += 1
        if row.contradiction_count > row.evidence_count * 0.6:
            row.validation_status = "disputed"
        row.save()

    def retrieve_insights(
        self,
        state: dict[str, object],
        *,
        insight_type: str = "",
        limit: int = 5,
    ) -> list[InsightData]:
        snapshot = state.get("llm_snapshot", {})
        if not isinstance(snapshot, dict):
            snapshot = {}
        scenario = str(snapshot.get("scenario") or state.get("scenario") or "")
        phase = str(state.get("phase") or "")
        decision_family = self._decision_family(phase)
        idol_plan_type = str(snapshot.get("idol_plan_type") or "")
        planning = snapshot.get("planning", {})
        if not isinstance(planning, dict):
            planning = {}
        next_gate = planning.get("next_gate", {})
        if not isinstance(next_gate, dict):
            next_gate = {}
        next_gate_key = str(next_gate.get("gate_type") or "")
        resource_pressure = planning.get("resource_pressure", {})
        if not isinstance(resource_pressure, dict):
            resource_pressure = {}
        resource_pressure_key = str(resource_pressure.get("summary") or "")
        build_plan = planning.get("build_plan", {})
        if not isinstance(build_plan, dict):
            build_plan = {}
        build_plan_key = str(build_plan.get("summary") or "")

        q = (
            ProduceStrategyInsight
            .select()
            .where(
                ProduceStrategyInsight.deleted_at.is_null(),
                ProduceStrategyInsight.validation_status != "expired",
            )
        )
        if scenario:
            q = q.where(ProduceStrategyInsight.scenario == scenario)
        if insight_type:
            q = q.where(ProduceStrategyInsight.insight_type == insight_type)
        elif phase:
            q = q.where(
                (ProduceStrategyInsight.phase == phase)
                | (ProduceStrategyInsight.phase == "")
            )

        rows = list(q.order_by(ProduceStrategyInsight.created_at.desc()).limit(limit * 3))

        scored: list[tuple[float, InsightData]] = []
        for row in rows:
            score = 0.0
            reasons: list[str] = []
            if row.decision_family and row.decision_family == decision_family:
                score += 3.0
                reasons.append("decision_family")
            if row.idol_plan_type and row.idol_plan_type == idol_plan_type:
                score += 2.5
                reasons.append("idol_plan_type")
            if row.next_gate_key and row.next_gate_key == next_gate_key:
                score += 2.0
                reasons.append("next_gate")
            if row.route_bias_key and row.route_bias_key == idol_plan_type:
                score += 1.5
                reasons.append("route_bias")
            if row.resource_pressure_key and row.resource_pressure_key == resource_pressure_key:
                score += 1.5
                reasons.append("resource_pressure")
            if row.build_plan_key and row.build_plan_key == build_plan_key:
                score += 1.5
                reasons.append("build_plan")
            if row.validation_status == "verified":
                score += 1.0
            elif row.validation_status == "trusted":
                score += 0.5
            elif row.validation_status == "disputed":
                score -= 2.0
            if score < 2.0:
                continue
            data = InsightData.from_row(row)
            data.similarity_reasons = reasons
            scored.append((score, data))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def get_by_ids(self, ids: list[int]) -> list[InsightData]:
        if not ids:
            return []
        rows = list(
            ProduceStrategyInsight
            .select()
            .where(
                ProduceStrategyInsight.id.in_(ids),
                ProduceStrategyInsight.deleted_at.is_null(),
            )
        )
        return [InsightData.from_row(row) for row in rows]

    def get_review_candidates(self, scenario: str, limit: int = 20) -> list[InsightData]:
        rows = list(
            ProduceStrategyInsight
            .select()
            .where(
                ProduceStrategyInsight.deleted_at.is_null(),
                ProduceStrategyInsight.scenario == scenario,
                ProduceStrategyInsight.validation_status.not_in(["expired", "disputed"]),
            )
            .order_by(ProduceStrategyInsight.created_at.desc())
            .limit(limit)
        )
        return [InsightData.from_row(row) for row in rows]

    def get_session_insights(self, session_id: str) -> list[InsightData]:
        if not session_id:
            return []
        rows = list(
            ProduceStrategyInsight
            .select()
            .where(
                ProduceStrategyInsight.session_id == session_id,
                ProduceStrategyInsight.deleted_at.is_null(),
            )
        )
        return [InsightData.from_row(row) for row in rows]

    @staticmethod
    def _promote_status(current: str, evidence_count: int) -> str:
        if current == "draft" and evidence_count >= 2:
            return "trusted"
        if current == "trusted" and evidence_count >= 5:
            return "verified"
        return current

    @staticmethod
    def _decision_family(phase: str) -> str:
        mapping = {
            "schedule": "schedule",
            "skill_reward": "reward",
            "p_drink": "inventory",
            "item_select": "inventory",
            "consult": "consult",
            "lesson": "combat",
            "exam": "combat",
        }
        return mapping.get(phase, phase or "unknown")


_insight_store: InsightStore | None = None


def get_insight_store() -> InsightStore:
    global _insight_store
    if _insight_store is None:
        _insight_store = InsightStore()
    return _insight_store
