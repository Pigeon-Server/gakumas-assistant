"""策略洞察生成器 — 后台异步调用 LLM 生成抽象策略洞察。

职责:
  - submit_step_insight(): 每次决策后，后台生成单步洞察
  - submit_phase_insights(): 局末，后台生成阶段洞察（从结果反推）
  - submit_review(): 局末，后台自审已有洞察
  - wait_all(): 等待所有后台任务完成
"""

from __future__ import annotations

import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from src.core.tasks.producer_challenge.gameplay.llm.insight_store import get_insight_store
from src.core.tasks.producer_challenge.gameplay.llm.insight_data import InsightData
from src.core.tasks.producer_challenge.gameplay.llm.llm_caller import (
    extract_final_text,
)
from src.core.tasks.producer_challenge.gameplay.llm.prompt_renderer import render as render_template
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext


class InsightGenerator:
    """后台策略洞察生成器。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        max_tokens: int | None = None,
        num_ctx: int = 8192,
        reasoning_effort: str = "medium",
        temperature: float = 0.2,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx
        self._reasoning_effort = reasoning_effort
        self._temperature = temperature
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="insight")
        self._pending: list[Future] = []

    # ── 公开接口 ────────────────────────────────────

    def submit_step_insight(
        self,
        ctx: "ProduceContext",
        decision_state: dict[str, Any],
        outcome_type: str,
        outcome_notes: str = "",
    ) -> None:
        if not self._should_generate(ctx):
            return
        self._submit(self._generate_step_insight, ctx, decision_state, outcome_type, outcome_notes)

    def submit_phase_insights(self, ctx: "ProduceContext") -> None:
        if not self._should_generate(ctx):
            return
        self._submit(self._generate_phase_insights, ctx)

    def submit_review(self, ctx: "ProduceContext") -> None:
        if not self._should_generate(ctx):
            return
        self._submit(self._generate_review, ctx)

    def wait_all(self, timeout: float = 30) -> None:
        for f in self._pending:
            try:
                f.result(timeout=timeout)
            except Exception as exc:
                logger.warning("[Insight] 后台任务异常: {}", exc)
        self._pending.clear()

    # ── 单步洞察 ───────────────────────────────────

    def _generate_step_insight(
        self,
        ctx: "ProduceContext",
        decision_state: dict[str, Any],
        outcome_type: str,
        outcome_notes: str,
    ) -> None:
        try:
            snapshot = decision_state.get("llm_snapshot", {})
            planning = dict(snapshot.get("planning", {}) or {})
            resources = dict(snapshot.get("resources", {}) or {})
            deck_plan = dict(planning.get("deck_plan", {}) or {})
            p_items = snapshot.get("p_items") or []

            recent_steps = self._extract_recent_steps(ctx)
            candidates = self._extract_candidates(decision_state)
            chosen_tags = decision_state.get("chosen_decision_tags", "")
            if not chosen_tags:
                chosen_idx = decision_state.get("chosen_index")
                for c in candidates:
                    if c.get("index") == chosen_idx:
                        chosen_tags = c.get("decision_tags", "")
                        break

            user_prompt = render_template(
                "insight_step.j2",
                idol_plan_type=str(snapshot.get("idol_plan_type") or ""),
                parameter_priority=str(snapshot.get("parameter_priority") or ""),
                idol_plan_focus=str(snapshot.get("idol_plan_focus") or ""),
                resources=resources,
                deck_plan_summary=str(deck_plan.get("summary") or ""),
                deck_plan_needs=", ".join(deck_plan.get("needs") or []),
                p_items_desc=", ".join(str(i.get("description") or i.get("name") or "") for i in p_items),
                stamina=int(snapshot.get("stamina") or 0),
                max_stamina=int(snapshot.get("max_stamina") or 0),
                p_point=int(snapshot.get("p_point") or 0),
                resource_pressure=str(planning.get("resource_pressure", {}).get("summary") or ""),
                recent_steps=recent_steps,
                phase=str(decision_state.get("phase") or ""),
                position=str(decision_state.get("position") or ""),
                decision_goal=str(decision_state.get("decision_goal") or ""),
                candidates=candidates,
                chosen_tags=chosen_tags,
                outcome_type=outcome_type,
                outcome_notes=outcome_notes,
            )

            result = self._call_llm(user_prompt, "system_insight_generator.j2")
            if result:
                self._save_insight(result, ctx, decision_state, "step", outcome_type, outcome_notes)
        except Exception as exc:
            logger.warning("[Insight] 单步洞察生成异常: {}", exc)

    # ── 阶段洞察 ───────────────────────────────────

    def _generate_phase_insights(self, ctx: "ProduceContext") -> None:
        try:
            snapshot = ctx.handler_state.get("last_decision_state", {}).get("llm_snapshot", {})
            exam_info = ctx.handler_state.get("exam_result_info") or {}
            param = ctx.parameter_state or {}
            timeline = self._extract_full_timeline(ctx)
            mutations = self._extract_deck_mutations(ctx)

            result_summary = self._build_result_summary(exam_info, param, ctx)

            user_prompt = render_template(
                "insight_phase.j2",
                scenario=str(ctx.scenario or "hajime"),
                difficulty=str(ctx.difficulty or "regular"),
                idol_plan_type=str(snapshot.get("idol_plan_type") or ""),
                idol_plan_focus=str(snapshot.get("idol_plan_focus") or ""),
                parameter_priority=str(snapshot.get("parameter_priority") or ""),
                exam_rank=int(exam_info.get("player_rank") or 0),
                exam_total=int(len(exam_info.get("all_rankings") or []) or 6),
                exam_score=int(exam_info.get("player_score") or 0),
                exam_passed=bool(exam_info.get("passed", False)),
                vocal=int(param.get("vocal") or 0),
                dance=int(param.get("dance") or 0),
                visual=int(param.get("visual") or 0),
                final_week=int(ctx.current_week or 0),
                timeline=timeline,
                deck_mutations=mutations,
                result_summary=result_summary,
            )

            result = self._call_llm(user_prompt, "system_insight_generator.j2")
            if result:
                self._save_insight(result, ctx, {}, "phase", "completed", result_summary)
        except Exception as exc:
            logger.warning("[Insight] 阶段洞察生成异常: {}", exc)

    # ── LLM 自审 ───────────────────────────────────

    def _generate_review(self, ctx: "ProduceContext") -> None:
        try:
            store = get_insight_store()
            scenario = str(ctx.scenario or "hajime")
            session_id = str(ctx.handler_state.get("llm_session_state", {}).get("session_id") or "")

            new_insights = store.get_session_insights(session_id)
            existing = store.get_review_candidates(scenario, limit=20)
            existing = [e for e in existing if e.id > 0]

            if not existing:
                return

            timeline = self._extract_full_timeline(ctx)
            exam_info = ctx.handler_state.get("exam_result_info") or {}
            param = ctx.parameter_state or {}
            result_summary = self._build_result_summary(exam_info, param, ctx)

            # 渲染自审 prompt
            new_dicts = [{"id": i.id, "strategy_description": i.strategy_description,
                          "when_to_apply": i.when_to_apply, "when_not_to_apply": i.when_not_to_apply}
                         for i in new_insights]
            existing_dicts = [{"id": i.id, "validation_status": i.validation_status,
                               "evidence_count": i.evidence_count, "contradiction_count": 0,
                               "strategy_description": i.strategy_description,
                               "when_to_apply": i.when_to_apply, "when_not_to_apply": i.when_not_to_apply}
                              for i in existing]

            user_prompt = render_template(
                "insight_review.j2",
                new_insights=new_dicts,
                existing_insights=existing_dicts,
                timeline=timeline,
                result_summary=result_summary,
            )

            raw = self._call_llm(user_prompt, "system_insight_reviewer.j2")
            if not raw:
                return

            judgments = self._parse_review_result(raw)
            for j in judgments:
                iid = j.get("insight_id")
                judgment = j.get("judgment")
                if not iid or not judgment:
                    continue
                if judgment == "keep":
                    store.update_insight(iid, {})
                elif judgment == "update_conditions":
                    updates: dict[str, Any] = {}
                    if j.get("updated_when_to_apply"):
                        updates["when_to_apply"] = j["updated_when_to_apply"]
                    if j.get("updated_when_not_to_apply"):
                        updates["when_not_to_apply"] = j["updated_when_not_to_apply"]
                    if updates:
                        store.update_insight(iid, updates)
                elif judgment == "mark_disputed":
                    store.record_contradiction(iid)
                elif judgment == "delete":
                    store.soft_delete(iid)
        except Exception as exc:
            logger.warning("[Insight] 自审异常: {}", exc)

    # ── LLM 调用 ───────────────────────────────────

    def _call_llm(self, user_prompt: str, system_template: str) -> str:
        system_prompt = render_template(system_template)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._num_ctx:
            kwargs["extra_body"] = {"options": {"num_ctx": self._num_ctx}}
        try:
            response = self._client.chat.completions.create(**kwargs)
            text = extract_final_text(response)
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            return cleaned
        except Exception as exc:
            logger.warning("[Insight] LLM 调用失败: {}", exc)
            return ""

    # ── 数据提取 ───────────────────────────────────

    def _extract_recent_steps(self, ctx: "ProduceContext", window: int = 5) -> list[dict[str, str]]:
        ops = list(ctx.operation_history or [])[-window:]
        steps: list[dict[str, str]] = []
        for op in ops:
            detail = op.details
            steps.append({
                "week": str(detail.get("week") or ""),
                "phase": str(op.phase),
                "decision_goal": str(detail.get("decision_goal") or ""),
                "decision_tags": str(detail.get("decision_tags") or ""),
                "cost": str(detail.get("cost_summary") or ""),
                "gain": str(detail.get("gain_summary") or ""),
                "outcome": str(detail.get("outcome") or "unknown"),
            })
        return steps

    def _extract_full_timeline(self, ctx: "ProduceContext") -> list[dict[str, str]]:
        return self._extract_recent_steps(ctx, window=200)

    def _extract_candidates(self, decision_state: dict[str, Any]) -> list[dict[str, Any]]:
        actions = decision_state.get("llm_actions") or []
        candidates: list[dict[str, Any]] = []
        for a in actions:
            if not a.get("available", True):
                continue
            candidates.append({
                "index": a.get("index", 0),
                "decision_tags": ", ".join(a.get("decision_tags") or []),
                "cost": str(a.get("cost_summary") or ""),
                "gain": str(a.get("gain_summary") or ""),
            })
        return candidates[:12]

    def _extract_deck_mutations(self, ctx: "ProduceContext") -> list[dict[str, str]]:
        mutations = list(ctx.deck_mutations or [])
        result: list[dict[str, str]] = []
        for m in mutations[-20:]:
            result.append({
                "week": str(m.get("week") or ""),
                "type": str(m.get("type") or ""),
                "kind": str(m.get("kind") or ""),
                "detail": str(m.get("detail") or m.get("name") or ""),
            })
        return result

    def _build_result_summary(
        self, exam_info: dict, param: dict, ctx: "ProduceContext"
    ) -> str:
        parts: list[str] = []
        if exam_info:
            rank = exam_info.get("player_rank")
            score = exam_info.get("player_score")
            passed = exam_info.get("passed")
            parts.append(f"考试: 排名={rank}, 分数={score}, 合格={passed}")
        if param:
            parts.append(f"参数: vocal={param.get('vocal',0)}, dance={param.get('dance',0)}, visual={param.get('visual',0)}")
        parts.append(f"周数: {ctx.current_week}")
        return " | ".join(parts)

    def _save_insight(
        self,
        llm_output: str,
        ctx: "ProduceContext",
        decision_state: dict[str, Any],
        insight_type: str,
        outcome_type: str,
        outcome_notes: str,
    ) -> None:
        parsed = self._parse_json(llm_output)
        if not parsed:
            return

        snapshot = decision_state.get("llm_snapshot", {}) or {}
        planning = snapshot.get("planning", {})
        if not isinstance(planning, dict):
            planning = {}
        next_gate = planning.get("next_gate", {})
        if not isinstance(next_gate, dict):
            next_gate = {}
        resource_pressure = planning.get("resource_pressure", {})
        if not isinstance(resource_pressure, dict):
            resource_pressure = {}
        build_plan = planning.get("build_plan", {})
        if not isinstance(build_plan, dict):
            build_plan = {}

        data = InsightData(
            insight_type=insight_type,
            scenario=str(ctx.scenario or "hajime"),
            phase=str(decision_state.get("phase") or ""),
            position=str(decision_state.get("position") or ""),
            strategy_description=str(parsed.get("strategy_description") or ""),
            setup_pattern=str(parsed.get("setup_pattern") or ""),
            when_to_apply=str(parsed.get("when_to_apply") or ""),
            when_not_to_apply=str(parsed.get("when_not_to_apply") or ""),
            decision_family=self._decision_family(str(decision_state.get("phase") or "")),
            idol_plan_type=str(snapshot.get("idol_plan_type") or ""),
            parameter_priority=str(snapshot.get("parameter_priority") or ""),
            next_gate_key=str(next_gate.get("gate_type") or ""),
            route_bias_key=str(snapshot.get("idol_plan_type") or ""),
            resource_pressure_key=str(resource_pressure.get("summary") or ""),
            build_plan_key=str(build_plan.get("summary") or ""),
            outcome_type=outcome_type,
            outcome_notes=outcome_notes,
        )

        if not data.strategy_description:
            return

        store = get_insight_store()
        store.save_insight(data)
        logger.info("[Insight] 保存{}洞察: {}", insight_type, data.strategy_description[:60])

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_review_result(text: str) -> list[dict[str, Any]]:
        text = text.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []

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

    def _should_generate(self, ctx: "ProduceContext") -> bool:
        try:
            from src.core.tasks.producer_challenge.gameplay.llm.config import get_insight_config
            cfg = get_insight_config()
            return bool(cfg.get("enabled", True))
        except Exception:
            return False

    def _submit(self, fn: Any, *args: Any) -> None:
        try:
            future = self._executor.submit(fn, *args)
            self._pending.append(future)
        except Exception as exc:
            logger.warning("[Insight] 提交后台任务失败: {}", exc)
