"""决策 Dump 系统 — 将 LLM 决策全链路信息写入 JSON 文件，便于离线分析。

每次 LLM 决策会生成一份 JSON 文件，包含:
  - 时间戳、阶段(phase)、位置(position)
  - 游戏局面快照 (llm_snapshot)
  - 系统提示词 / 用户提示词
  - LLM 原始响应 (content + reasoning)
  - 最终选择的动作索引和候选信息
  - 耗时统计

文件保存位置: logs/debug/decisions/{session_id}/
文件名格式: {seq:04d}_{phase}.json

使用方式:
  dumper = DecisionDumper.get_instance()
  dumper.start_session()          # 每次培育开始时调用
  dumper.record(...)              # 每次 LLM 决策后调用
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import logger

# dump 文件根目录
_DUMP_ROOT = Path("logs/debug/decisions")


@dataclass
class DecisionRecord:
    """单次决策的完整记录。"""

    # ── 基本标识 ──
    seq: int                            # 本次 session 内的序号
    timestamp: str                      # ISO 8601 时间戳
    phase: str                          # 阶段：lesson / exam / schedule / consult / ...
    position: str                       # 更细的位置标识
    week: int = 0                       # 当前周数
    revision: int = 0                   # 状态修订号（state_revision）

    # ── 局面快照 ──
    llm_snapshot: dict[str, Any] = field(default_factory=dict)
    stage_context: dict[str, Any] = field(default_factory=dict)
    decision_explanation: dict[str, Any] = field(default_factory=dict)

    # ── 候选动作 ──
    candidates: list[dict[str, Any]] = field(default_factory=list)
    llm_actions: list[dict[str, Any]] = field(default_factory=list)
    legal_actions: list[int] = field(default_factory=list)

    # ── LLM 调用 ──
    system_prompt: str = ""
    user_prompt: str = ""
    llm_raw_content: str = ""           # LLM content 字段原文
    llm_raw_reasoning: str = ""         # LLM reasoning/思维链 原文
    llm_cleaned_output: str = ""        # 清理后的最终输出
    llm_model: str = ""                 # 模型名称
    llm_elapsed_sec: float = 0.0        # LLM 调用耗时

    # ── 最终决策 ──
    chosen_index: int | None = None     # LLM 返回的动作索引
    resolved_index: int | None = None   # 最终执行的动作索引（可能被兜底覆盖）
    resolved_action_id: str = ""        # 最终执行的 action_id
    resolved_name: str = ""             # 最终执行的候选名称
    fallback_used: bool = False         # 是否使用了兜底策略
    fallback_reason: str = ""           # 兜底原因

    # ── 总耗时（含 build_state + LLM + resolve）──
    total_elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "position": self.position,
            "week": self.week,
            "revision": self.revision,
            "llm_snapshot": self.llm_snapshot,
            "stage_context": self.stage_context,
            "decision_explanation": self.decision_explanation,
            "candidates": self.candidates,
            "llm_actions": self.llm_actions,
            "legal_actions": self.legal_actions,
            "llm_call": {
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt,
                "raw_content": self.llm_raw_content,
                "raw_reasoning": self.llm_raw_reasoning,
                "cleaned_output": self.llm_cleaned_output,
                "model": self.llm_model,
                "elapsed_sec": round(self.llm_elapsed_sec, 3),
            },
            "summary_id": None,
            "decision": {
                "chosen_index": self.chosen_index,
                "resolved_index": self.resolved_index,
                "resolved_action_id": self.resolved_action_id,
                "resolved_name": self.resolved_name,
                "fallback_used": self.fallback_used,
                "fallback_reason": self.fallback_reason,
            },
            "total_elapsed_sec": round(self.total_elapsed_sec, 3),
        }


class DecisionDumper:
    """决策 Dump 管理器（单例）。

    每次培育启动时调用 start_session() 创建新的子目录；
    每次 LLM 决策完成后调用 record() 写入一条记录。
    """

    _instance: DecisionDumper | None = None

    def __init__(self):
        self._session_dir: Path | None = None
        self._seq: int = 0
        self._enabled: bool = True
        self._records: list[DecisionRecord] = []

    @classmethod
    def get_instance(cls) -> DecisionDumper:
        """获取instance并返回结果。

        Returns:
            DecisionDumper: 返回值类型见注解，语义由函数用途决定。
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def record_count(self) -> int:
        return self._seq

    def start_session(self, session_id: str = "") -> Path:
        """开始新的 dump session，创建输出目录。

        Args:
            session_id: 可选标识，默认使用时间戳。

        Returns:
            session 目录路径。
        """
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = _DUMP_ROOT / session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._records = []
        logger.info("[DecisionDump] 新 session: {}", self._session_dir)
        return self._session_dir

    def record(
        self,
        *,
        decision_state: dict[str, Any],
        system_prompt: str = "",
        user_prompt: str = "",
        llm_raw_content: str = "",
        llm_raw_reasoning: str = "",
        llm_cleaned_output: str = "",
        llm_model: str = "",
        llm_elapsed_sec: float = 0.0,
        chosen_index: int | None = None,
        resolved_index: int | None = None,
        resolved_action_id: str = "",
        resolved_name: str = "",
        fallback_used: bool = False,
        fallback_reason: str = "",
        total_elapsed_sec: float = 0.0,
    ) -> DecisionRecord | None:
        """记录一次完整决策并写入 JSON 文件。

        Args:
            decision_state: build_decision_state() 返回的完整状态。
            其余参数: LLM 调用的详细信息。

        Returns:
            DecisionRecord 或 None（如果未启用/未初始化 session）。
        """
        if not self._enabled:
            return None
        if self._session_dir is None:
            self.start_session()

        phase = str(decision_state.get("phase", "unknown"))
        position = str(decision_state.get("position", ""))

        rec = DecisionRecord(
            seq=self._seq,
            timestamp=datetime.now().isoformat(),
            phase=phase,
            position=position,
            week=int(decision_state.get("week", 0)),
            revision=int(decision_state.get("revision", 0)),
            llm_snapshot=_safe_serialize(decision_state.get("llm_snapshot", {})),
            stage_context=_safe_serialize(decision_state.get("stage_context", {})),
            decision_explanation=_safe_serialize(decision_state.get("decision_explanation", {})),
            candidates=_safe_serialize(decision_state.get("candidates", [])),
            llm_actions=_safe_serialize(decision_state.get("llm_actions", [])),
            legal_actions=list(decision_state.get("legal_actions", [])),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            llm_raw_content=llm_raw_content,
            llm_raw_reasoning=llm_raw_reasoning,
            llm_cleaned_output=llm_cleaned_output,
            llm_model=llm_model,
            llm_elapsed_sec=llm_elapsed_sec,
            chosen_index=chosen_index,
            resolved_index=resolved_index,
            resolved_action_id=resolved_action_id,
            resolved_name=resolved_name,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            total_elapsed_sec=total_elapsed_sec,
        )

        # 写入 JSON 文件（防御性创建目录，避免被外部清理后写入失败）
        filename = f"{self._seq:04d}_{phase}.json"
        filepath = self._session_dir / filename
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(rec.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(
                "[DecisionDump] #{} {} → {} (idx={}, {:.1f}s)",
                self._seq, phase, resolved_name or f"#{resolved_index}",
                resolved_index, total_elapsed_sec,
            )
        except Exception as exc:
            logger.warning("[DecisionDump] 写入失败 {}: {}", filepath, exc)

        self._records.append(rec)
        self._seq += 1
        return rec

    def update_last_resolved(
        self,
        *,
        resolved_index: int,
        resolved_name: str = "",
        resolved_action_id: str = "",
        fallback_used: bool = False,
        fallback_reason: str = "",
    ):
        """更新最近一条记录的最终执行结果（当调用方覆盖了 LLM 的原始决策时使用）。

        场景: LLM 返回 None 或非法索引后，调用方通过本地兜底逻辑选择了另一个动作。
        """
        if not self._enabled or not self._records:
            return
        rec = self._records[-1]
        rec.resolved_index = resolved_index
        rec.resolved_name = resolved_name or rec.resolved_name
        rec.resolved_action_id = resolved_action_id or rec.resolved_action_id
        rec.fallback_used = fallback_used or rec.fallback_used
        rec.fallback_reason = fallback_reason or rec.fallback_reason
        # 覆写 JSON 文件
        if self._session_dir:
            filename = f"{rec.seq:04d}_{rec.phase}.json"
            filepath = self._session_dir / filename
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(rec.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.debug("[DecisionDump] 更新写入失败 {}: {}", filepath, exc)

    def get_summary(self) -> dict[str, Any]:
        """返回当前 session 的统计摘要。"""
        if not self._records:
            return {"total": 0}
        phase_counts: dict[str, int] = {}
        total_llm_time = 0.0
        fallback_count = 0
        for rec in self._records:
            phase_counts[rec.phase] = phase_counts.get(rec.phase, 0) + 1
            total_llm_time += rec.llm_elapsed_sec
            if rec.fallback_used:
                fallback_count += 1
        return {
            "total": len(self._records),
            "by_phase": phase_counts,
            "total_llm_time_sec": round(total_llm_time, 1),
            "avg_llm_time_sec": round(total_llm_time / len(self._records), 2),
            "fallback_count": fallback_count,
            "session_dir": str(self._session_dir),
        }

    def write_summary(self):
        """将统计摘要写入 session 目录。"""
        if self._session_dir is None:
            return
        summary = self.get_summary()
        filepath = self._session_dir / "_summary.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info("[DecisionDump] 摘要已写入: {}", filepath)
        except Exception as exc:
            logger.warning("[DecisionDump] 摘要写入失败: {}", exc)


def _safe_serialize(obj: Any) -> Any:
    """将对象转换为 JSON 可序列化格式，处理不可序列化的类型。"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    # numpy 类型
    type_name = type(obj).__name__
    if "int" in type_name and hasattr(obj, "item"):
        return int(obj.item())
    if "float" in type_name and hasattr(obj, "item"):
        return float(obj.item())
    if "ndarray" in type_name:
        return f"<ndarray shape={getattr(obj, 'shape', '?')}>"
    # 其余类型转字符串
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
