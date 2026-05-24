"""局内 LLM 会话状态 — 数据类 + 滚动压缩管理。

职责:
  - _LLMCallDetails: 单次 LLM 调用的详情快照
  - _LLMSessionTurn: 一次决策的压缩表示（用于 recent turns）
  - _LLMSessionState: 整局共享的会话状态
  - SessionManager: 会话生命周期管理（reset/flush/mirror/压缩）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext


# ── 数据类 ───────────────────────────────────────────


@dataclass
class LLMCallDetails:
    """单次 LLM 调用的详情快照，供 dump 系统使用。"""

    system_prompt: str = ""
    user_prompt: str = ""
    raw_content: str = ""
    raw_reasoning: str = ""
    cleaned_output: str = ""
    elapsed_sec: float = 0.0


@dataclass
class LLMSessionTurn:
    """局内一次决策的压缩表示。"""

    summary_id: int | None = None
    phase: str = ""
    position: str = ""
    decision_family: str = ""
    chosen_action_id: str = ""
    chosen_canonical_name: str = ""
    decision_goal: str = ""
    why_this_action: str = ""
    why_not_others: str = ""
    effective_when: str = ""
    ineffective_when: str = ""
    summary_text: str = ""
    validation_status: str = ""
    fallback_used: bool = False
    outcome_type: str = ""
    outcome_notes: str = ""

    def to_compact_text(self) -> str:
        """转换为供滚动压缩使用的短文本。"""
        title = f"phase={self.phase or 'unknown'}"
        if self.position:
            title += f" position={self.position}"
        action_name = self.chosen_canonical_name or self.chosen_action_id or "未知动作"
        parts = [
            title,
            f"目标: {self.decision_goal}" if self.decision_goal else "",
            f"选择: {action_name}",
            self.why_this_action,
            f"未选原因: {self.why_not_others}" if self.why_not_others else "",
            f"适用: {self.effective_when}" if self.effective_when else "",
            f"边界: {self.ineffective_when}" if self.ineffective_when else "",
            f"执行结果: {self.outcome_type} {self.outcome_notes}".strip()
            if (self.outcome_type or self.outcome_notes)
            else "",
            "该步最终走了 fallback 覆盖。" if self.fallback_used else "",
        ]
        return "\n".join(part for part in parts if part)

    def to_recent_memory_line(self) -> str:
        """转换为 recent turns 注入消息的一行。"""
        action_name = self.chosen_canonical_name or self.chosen_action_id or "未知动作"
        parts = [
            f"[{self.phase or 'unknown'}] {action_name}",
            self.decision_goal,
            self.why_this_action or self.summary_text,
        ]
        if self.outcome_type or self.outcome_notes:
            outcome = f"结果={self.outcome_type}" if self.outcome_type else "结果已回写"
            if self.outcome_notes:
                outcome += f" ({self.outcome_notes})"
            parts.append(outcome)
        if self.fallback_used:
            parts.append("最终由 fallback 执行")
        return "；".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "phase": self.phase,
            "position": self.position,
            "decision_family": self.decision_family,
            "chosen_action_id": self.chosen_action_id,
            "chosen_canonical_name": self.chosen_canonical_name,
            "decision_goal": self.decision_goal,
            "why_this_action": self.why_this_action,
            "why_not_others": self.why_not_others,
            "effective_when": self.effective_when,
            "ineffective_when": self.ineffective_when,
            "summary_text": self.summary_text,
            "validation_status": self.validation_status,
            "fallback_used": self.fallback_used,
            "outcome_type": self.outcome_type,
            "outcome_notes": self.outcome_notes,
        }


@dataclass
class LLMSessionState:
    """同一局内共享的会话状态。"""

    session_id: str = ""
    rolling_summary: str = ""
    recent_turns: list[LLMSessionTurn] = field(default_factory=list)
    compression_count: int = 0
    last_prompt_metrics: dict[str, Any] = field(default_factory=dict)
    last_summary_id: int | None = None
    last_recorded_summary_id: int | None = None
    last_compacted_summary_id: int | None = None


# ── 会话管理器 ────────────────────────────────────────


class SessionManager:
    """会话生命周期管理：初始化、镜像、压缩、flush。"""

    def __init__(
        self,
        *,
        recent_turn_window: int = 3,
        compression_trigger_ratio: float = 0.75,
        num_ctx: int = 8192,
        summary_target_tokens: int = 768,
        historical_summary_limit: int = 3,
    ):
        self.recent_turn_window = max(recent_turn_window, 1)
        self.compression_trigger_ratio = min(max(compression_trigger_ratio, 0.4), 0.95)
        self.num_ctx = num_ctx
        self.summary_target_tokens = max(summary_target_tokens, 128)
        self.historical_summary_limit = max(historical_summary_limit, 0)
        self.state = LLMSessionState()

    # ── 生命周期 ────────────────────────────────────

    def reset(self, ctx: "ProduceContext | None" = None, *, session_id: str = "") -> None:
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state = LLMSessionState(session_id=session_id)
        self.mirror_to_context(ctx)

    def flush(self, ctx: "ProduceContext | None" = None, *, compact_fn: Any = None) -> None:
        """整局结束时折叠所有 recent turns。"""
        if self.state.recent_turns and compact_fn:
            compact_fn(current_planning_note="", compact_all=True)
        self.mirror_to_context(ctx)

    def mirror_to_context(self, ctx: "ProduceContext | None") -> None:
        if ctx is None:
            return
        handler_state = ctx.handler_state
        if not isinstance(handler_state, dict):
            return
        s = self.state
        handler_state["llm_session_state"] = {
            "session_id": s.session_id,
            "rolling_summary": s.rolling_summary,
            "recent_turns": [t.to_dict() for t in s.recent_turns],
            "compression_count": s.compression_count,
            "last_prompt_metrics": dict(s.last_prompt_metrics or {}),
            "last_summary_id": s.last_summary_id,
            "last_recorded_summary_id": s.last_recorded_summary_id,
            "last_compacted_summary_id": s.last_compacted_summary_id,
        }

    def ensure_ready(self, ctx: "ProduceContext | None" = None) -> None:
        if not self.state.session_id:
            self.reset(ctx)
        else:
            self.mirror_to_context(ctx)

    # ── Recent Turn 管理 ────────────────────────────

    def upsert_recent_turn(self, summary: dict[str, Any]) -> None:
        turn = self._build_turn(summary)
        turns = self.state.recent_turns
        if turns and turn.summary_id is not None and turns[-1].summary_id == turn.summary_id:
            turns[-1] = turn
        else:
            turns.append(turn)
            overflow = len(turns) - max(self.recent_turn_window + 3, self.recent_turn_window)
            if overflow > 0:
                del turns[:overflow]
        self.state.last_summary_id = turn.summary_id
        self.state.last_recorded_summary_id = turn.summary_id

    def refresh_from_ctx(self, ctx: "ProduceContext | None") -> None:
        """按最近 summary id 回读最新结构化摘要。已迁移到洞察系统，此方法保留为空。"""

    # ── 压缩 ────────────────────────────────────────

    def should_compact(self, estimated_tokens: int) -> bool:
        threshold = max(int(self.num_ctx * self.compression_trigger_ratio), 1)
        return estimated_tokens >= threshold

    def prepare_compact_args(
        self, *, compact_all: bool = False
    ) -> tuple[list[LLMSessionTurn], list[LLMSessionTurn]] | None:
        """返回 (compact_turns, keep_turns)，无需压缩时返回 None。"""
        turns = list(self.state.recent_turns)
        if not turns:
            return None
        if compact_all:
            return turns, []
        if len(turns) > self.recent_turn_window:
            return turns[: -self.recent_turn_window], turns[-self.recent_turn_window :]
        if self.state.rolling_summary:
            ct = turns[:-1] if len(turns) > 1 else turns
            kt = turns[-1:] if len(turns) > 1 else []
            return ct, kt
        return None

    def apply_compact_result(self, new_summary: str, keep_turns: list[LLMSessionTurn], last_compacted_id: int | None) -> None:
        self.state.rolling_summary = new_summary.strip()
        self.state.recent_turns = keep_turns
        self.state.compression_count += 1
        self.state.last_compacted_summary_id = last_compacted_id

    def trim_recent_turns(self) -> None:
        if len(self.state.recent_turns) > self.recent_turn_window:
            self.state.recent_turns = self.state.recent_turns[-self.recent_turn_window :]

    # ── 序列化 ──────────────────────────────────────

    def build_recent_turns_text(self) -> str:
        if not self.state.recent_turns:
            return ""
        lines = ["以下是本局最近几步尚未压缩的关键决策："]
        for turn in self.state.recent_turns[-self.recent_turn_window :]:
            lines.append(f"- {turn.to_recent_memory_line()}")
        return "\n".join(lines)

    def build_rolling_summary_message(self) -> str | None:
        text = str(self.state.rolling_summary or "").strip()
        if not text:
            return None
        return "以下是当前这一局已压缩的持续记忆，请把它视为本局已发生事实的摘要，不要改写成别的意思：\n" + text

    # ── 内部 ────────────────────────────────────────

    @staticmethod
    def _build_turn(summary: dict[str, Any]) -> LLMSessionTurn:
        return LLMSessionTurn(
            summary_id=_safe_int(summary.get("id")),
            phase=str(summary.get("phase") or ""),
            position=str(summary.get("position") or ""),
            decision_family=str(summary.get("decision_family") or ""),
            chosen_action_id=str(summary.get("chosen_action_id") or ""),
            chosen_canonical_name=str(summary.get("chosen_canonical_name") or ""),
            decision_goal=str(summary.get("decision_goal") or ""),
            why_this_action=str(summary.get("why_this_action") or ""),
            why_not_others=str(summary.get("why_not_others") or ""),
            effective_when=str(summary.get("effective_when") or ""),
            ineffective_when=str(summary.get("ineffective_when") or ""),
            summary_text=str(summary.get("summary_text") or ""),
            validation_status=str(summary.get("validation_status") or ""),
            fallback_used=bool(summary.get("fallback_used", False)),
            outcome_type=str(summary.get("latest_outcome_type") or ""),
            outcome_notes=str(summary.get("latest_outcome_notes") or ""),
        )


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
