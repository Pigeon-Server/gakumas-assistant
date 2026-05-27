"""LLM 消息构建器 — prompt 渲染 + 会话消息组装 + 滚动压缩。

职责:
  - 渲染系统 prompt 和用户 prompt（Jinja2 模板）
  - 组装完整的会话消息列表（system → rolling → insights → recent → current）
  - 滚动摘要压缩（LLM 生成 + 本地兜底）
  - 策略洞察文本构建
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.logger import logger
from src.core.tasks.producer_challenge.gameplay.llm.prompt_renderer import render as _render_template
from src.core.tasks.producer_challenge.gameplay.llm.session_state import SessionManager
from src.core.tasks.producer_challenge.gameplay.llm.insight_data import InsightData

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext  # codeql[py/unused-import]

# 阶段 → 系统提示词模板映射
_SYSTEM_TEMPLATE_MAP: dict[str, str] = {
    "lesson": "system_lesson.j2",
    "exam": "system_exam.j2",
    "schedule": "system_schedule.j2",
    "dialogue": "system_dialogue.j2",
    "skill_reward": "system_skill_reward.j2",
    "p_drink": "system_p_drink.j2",
    "consult": "system_consult.j2",
    "item_select": "system_item_select.j2",
}


# ── Prompt 渲染 ──────────────────────────────────────


def build_system_prompt(phase: str, snapshot: dict[str, Any] | None = None) -> str:
    template_name = _SYSTEM_TEMPLATE_MAP.get(phase, "system_default.j2")
    kwargs = dict(snapshot) if snapshot else {}
    try:
        return _render_template(template_name, **kwargs)
    except Exception as exc:
        logger.warning("[LLM] 渲染系统模板 {} 失败: {}", template_name, exc)
        return _render_template("system_default.j2")


def build_user_prompt(
    state: dict[str, Any],
    strategy_insights: list[InsightData] | None = None,
) -> str:
    snapshot = state.get("llm_snapshot", {})
    llm_actions = state.get("llm_actions") or []
    strategy_insights = list(strategy_insights or [])

    try:
        rendered_snapshot = _render_template("state_snapshot.j2", **snapshot)
    except Exception as exc:
        logger.warning("[LLM] 渲染 state_snapshot.j2 失败: {}", exc)
        rendered_snapshot = f"## 当前局面\n（模板渲染失败: {exc}）"

    insight_dicts = [
        {"strategy_description": i.strategy_description, "when_to_apply": i.when_to_apply, "when_not_to_apply": i.when_not_to_apply}
        for i in strategy_insights
    ]

    try:
        return _render_template(
            "action_select.j2",
            snapshot=rendered_snapshot,
            actions=llm_actions,
            strategy_insights=insight_dicts,
        )
    except Exception as exc:
        logger.warning("[LLM] 渲染 action_select.j2 失败: {}", exc)
        parts = [rendered_snapshot, "\n## 当前动作"]
        for a in llm_actions:
            idx = a.get("index", 0)
            label = a.get("label", "?")
            desc = a.get("description", "")
            line = f"{idx}: {label}"
            if desc:
                line += f" - {desc}"
            parts.append(line)
        if insight_dicts:
            parts.append("\n## 历史策略洞察")
            for insight in insight_dicts:
                parts.append(f"- 策略: {insight.get('strategy_description', '')}")
                parts.append(f"  适用: {insight.get('when_to_apply', '')}")
                parts.append(f"  不适用: {insight.get('when_not_to_apply', '')}")
        parts.append("\n请选择当前最优动作，只输出动作编号：")
        return "\n".join(parts)


# ── 消息组装 ─────────────────────────────────────────


def prepare_messages(
    *,
    system_prompt: str,
    current_prompt: str,
    state: dict[str, Any],
    selected_insights: list[InsightData],
    session: SessionManager,
    ctx: "ProduceContext | None",
) -> list[dict[str, str]]:
    """组装同局共享会话消息，并在必要时自动压缩。"""
    maybe_compact(state=state, system_prompt=system_prompt, current_prompt=current_prompt,
                  session=session, ctx=ctx)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    rolling_msg = session.build_rolling_summary_message()
    if rolling_msg:
        messages.append({"role": "assistant", "content": rolling_msg})

    insight_text = build_insight_reference_text(selected_insights)
    if insight_text:
        messages.append({"role": "assistant", "content": insight_text})

    recent_text = session.build_recent_turns_text()
    if recent_text:
        messages.append({"role": "assistant", "content": recent_text})

    messages.append({"role": "user", "content": current_prompt})

    metrics = calculate_prompt_metrics(messages, session)
    session.state.last_prompt_metrics = metrics
    session.mirror_to_context(ctx)
    return messages


def maybe_compact(
    *,
    state: dict[str, Any],
    system_prompt: str,
    current_prompt: str,
    session: SessionManager,
    ctx: "ProduceContext | None",
    summarizer_fn: Any = None,
) -> None:
    """预测式上下文预算检查，必要时触发压缩。"""
    estimated = _estimate_context_tokens(system_prompt, current_prompt, session)
    if not session.should_compact(estimated):
        session.state.last_prompt_metrics = {"estimated_tokens": estimated}
        session.mirror_to_context(ctx)
        return

    logger.info("[LLM] 预测上下文压力过高，触发滚动压缩 estimated_tokens={} threshold={}",
                estimated, max(int(session.num_ctx * session.compression_trigger_ratio), 1))
    planning_note = build_current_planning_note(state)
    if summarizer_fn:
        summarizer_fn(current_planning_note=planning_note)
        session.mirror_to_context(ctx)
        return

    session.trim_recent_turns()
    session.mirror_to_context(ctx)


# ── 策略洞察文本构建 ─────────────────────────────────


def build_insight_reference_text(insights: list[InsightData]) -> str:
    if not insights:
        return ""
    lines = ["以下是来自过往培育经验的策略洞察，仅供参考（不是本局已发生事实）："]
    for item in insights:
        if not item.strategy_description:
            continue
        lines.append(f"- 策略: {item.strategy_description}")
        if item.when_to_apply:
            lines.append(f"  适用: {item.when_to_apply}")
        if item.when_not_to_apply:
            lines.append(f"  不适用: {item.when_not_to_apply}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Planning 提取 ────────────────────────────────────


def build_current_planning_note(state: dict[str, Any]) -> str:
    snapshot = dict(state.get("llm_snapshot", {}) or {})
    planning = dict(snapshot.get("planning", {}) or {})
    lines: list[str] = []
    route_bias = dict(planning.get("route_bias", {}) or {})
    if route_bias.get("summary"):
        lines.append(f"路线倾向: {route_bias['summary']}")
    current_objectives = dict(planning.get("current_objectives", {}) or {})
    if current_objectives.get("summary"):
        lines.append(f"当前目标: {current_objectives['summary']}")
    next_gate = dict(planning.get("next_gate", {}) or {})
    if next_gate:
        gate_parts = [str(next_gate.get("gate_label") or next_gate.get("gate_type") or "")]
        if next_gate.get("weeks_until_gate") is not None:
            gate_parts.append(f"剩余{next_gate.get('weeks_until_gate')}周")
        if next_gate.get("readiness_summary"):
            gate_parts.append(str(next_gate.get("readiness_summary") or ""))
        gate_line = " | ".join(part for part in gate_parts if part)
        if gate_line:
            lines.append(f"下个门槛: {gate_line}")
    resource_pressure = dict(planning.get("resource_pressure", {}) or {})
    if resource_pressure.get("summary"):
        lines.append(f"资源压力: {resource_pressure['summary']}")
    time_pressure = dict(planning.get("time_pressure", {}) or {})
    if time_pressure.get("summary"):
        lines.append(f"时间压力: {time_pressure['summary']}")
    build_plan = dict(planning.get("build_plan", {}) or {})
    if build_plan.get("summary"):
        lines.append(f"成长规划: {build_plan['summary']}")
    deck_plan = dict(planning.get("deck_plan", {}) or {})
    if deck_plan.get("summary"):
        lines.append(f"牌组规划: {deck_plan['summary']}")
    inventory_plan = dict(planning.get("inventory_plan", {}) or {})
    if inventory_plan.get("summary"):
        lines.append(f"库存规划: {inventory_plan['summary']}")
    return "\n".join(line for line in lines if line)


# ── 指标与工具 ───────────────────────────────────────


def calculate_prompt_metrics(messages: list[dict[str, str]], session: SessionManager) -> dict[str, Any]:
    estimated_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    return {
        "estimated_tokens": estimated_tokens,
        "num_ctx": int(session.num_ctx or 0),
        "compression_trigger_ratio": session.compression_trigger_ratio,
        "threshold_tokens": max(int(session.num_ctx * session.compression_trigger_ratio), 1),
        "message_count": len(messages),
        "recent_turn_count": len(session.state.recent_turns),
        "has_rolling_summary": bool(session.state.rolling_summary),
        "compression_count": session.state.compression_count,
    }


def estimate_tokens(text: str) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return 0
    return max((len(normalized) + 1) // 2, 1)


def sanitize_summary_text(text: str) -> str:
    """过滤摘要中不该长期保留的 UI 与噪声内容。"""
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(token in lowered for token in ("reasoning_content", "<think", "</think", "bbox", "box=", "x=", "y=", "cx=", "cy=")):
            continue
        if any(token in line for token in ("[已选中]", "系统推荐", "操作含义")):
            continue
        lines.append(line)
    sanitized = "\n".join(lines).strip()
    if len(sanitized) > 4000:
        sanitized = sanitized[:4000].rstrip()
    return sanitized


# ── 内部辅助 ─────────────────────────────────────────


def _estimate_context_tokens(
    system_prompt: str,
    current_prompt: str,
    session: SessionManager,
) -> int:
    total = estimate_tokens(system_prompt) + estimate_tokens(current_prompt)
    rolling = session.state.rolling_summary
    if rolling:
        total += estimate_tokens(rolling)
    recent_text = session.build_recent_turns_text()
    if recent_text:
        total += estimate_tokens(recent_text)
    return total
