"""LLM 决策策略 — 通过 OpenAI 兼容 API 为各阶段提供智能决策。

内部结构：
- ScheduleLLMStrategy: 周行程决策
- BattleLLMStrategy: 战斗决策（lesson/exam）
- OtherLLMStrategy: 其他决策（dialogue/p_drink/consult 等）
- LLMStrategy: 统一入口，内部持有3个子策略，按 phase 路由
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import TYPE_CHECKING, Any, Sequence

from openai import OpenAI

from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper
from src.core.tasks.producer_challenge.gameplay.llm.config import get_insight_config, get_llm_config
from src.core.tasks.producer_challenge.gameplay.llm.insight_data import InsightData
from src.core.tasks.producer_challenge.gameplay.llm.llm_caller import (
    apply_backend_compat_options,
    extract_final_text,
    extract_raw_fields,
    get_candidate_name,
    normalize_legacy_think,
    normalize_reasoning_effort,
    parse_action_index,
    parse_decision_reasoning,
)
from src.core.tasks.producer_challenge.gameplay.llm.message_builder import (
    build_system_prompt,
    build_user_prompt,
    prepare_messages,
    sanitize_summary_text,
)
from src.core.tasks.producer_challenge.gameplay.llm.session_state import (
    LLMCallDetails,
    SessionManager,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


# ── 工具函数 ───────────────────────────────────────────────


_DEFAULT_LLM_CONFIG: dict[str, Any] = {
    "base_url": "http://192.168.100.10:11434/v1/",
    "model": "gpt-oss:20b",
    "api_key": "ollama",
    "timeout": 60.0,
    "temperature": 0.3,
    "max_tokens": None,
    "reasoning_effort": "medium",
    "num_ctx": 8192,
}

_DEFAULT_INSIGHT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "base_url": "",
    "model": "",
    "api_key": "",
    "timeout": 120.0,
    "max_tokens": None,
    "num_ctx": 8192,
    "reasoning_effort": "medium",
    "temperature": 0.2,
}


@dataclass(frozen=True)
class RuntimeLLMConfig:
    """LLM 主调用的运行时配置。"""

    base_url: str
    model: str
    api_key: str
    timeout: float
    temperature: float
    max_tokens: int | None
    reasoning_effort: str
    num_ctx: int


@dataclass(frozen=True)
class RuntimeInsightConfig:
    """策略洞察生成器的运行时配置。"""

    enabled: bool
    base_url: str
    model: str
    api_key: str
    timeout: float
    max_tokens: int | None
    num_ctx: int
    reasoning_effort: str
    temperature: float


def _coerce_int(value: Any, default: int = 0) -> int:
    """将任意值转换为整数，失败时返回默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """将任意值转换为浮点数，失败时返回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_positive_int(value: Any) -> int | None:
    """将正整数配置转换为可选整数，0 或空值表示自动。"""
    number = _coerce_int(value, 0)
    return number if number > 0 else None


def _normalize_base_url(value: Any, default: str) -> str:
    """标准化 OpenAI 兼容接口地址。"""
    text = str(value or default).strip()
    return (text or default).rstrip("/")


def _read_llm_config() -> dict[str, Any]:
    """读取主 LLM 配置，读取失败时返回内置默认值。"""
    config = dict(_DEFAULT_LLM_CONFIG)
    try:
        config.update(get_llm_config())
    except Exception as exc:
        logger.debug("[LLM] 读取主配置失败，使用默认值: {}", exc)
    return config


def _read_insight_config() -> dict[str, Any]:
    """读取洞察生成配置，读取失败时返回内置默认值。"""
    config = dict(_DEFAULT_INSIGHT_CONFIG)
    try:
        config.update(get_insight_config())
    except Exception as exc:
        logger.debug("[LLM] 读取洞察配置失败，使用默认值: {}", exc)
    return config


def _build_runtime_llm_config(overrides: dict[str, Any]) -> RuntimeLLMConfig:
    """按“默认值 < 全局配置 < 显式覆盖”的顺序生成主 LLM 配置。"""
    config = _read_llm_config()
    config.update(overrides)
    return RuntimeLLMConfig(
        base_url=_normalize_base_url(config.get("base_url"), _DEFAULT_LLM_CONFIG["base_url"]),
        model=str(config.get("model") or _DEFAULT_LLM_CONFIG["model"]),
        api_key=str(config.get("api_key") if config.get("api_key") is not None else _DEFAULT_LLM_CONFIG["api_key"]),
        timeout=_coerce_float(config.get("timeout"), float(_DEFAULT_LLM_CONFIG["timeout"])),
        temperature=_coerce_float(config.get("temperature"), float(_DEFAULT_LLM_CONFIG["temperature"])),
        max_tokens=_coerce_positive_int(config.get("max_tokens")),
        reasoning_effort=normalize_reasoning_effort(config.get("reasoning_effort")),
        num_ctx=max(_coerce_int(config.get("num_ctx"), int(_DEFAULT_LLM_CONFIG["num_ctx"])), 0),
    )


def _build_runtime_insight_config(
    llm_config: RuntimeLLMConfig,
    overrides: dict[str, Any],
) -> RuntimeInsightConfig:
    """按当前主 LLM 配置生成洞察生成器配置。"""
    config = _read_insight_config()
    config.update(overrides)
    return RuntimeInsightConfig(
        enabled=bool(config.get("enabled", True)),
        base_url=_normalize_base_url(config.get("base_url") or llm_config.base_url, llm_config.base_url),
        model=str(config.get("model") or llm_config.model),
        api_key=str(config.get("api_key") if config.get("api_key") is not None else llm_config.api_key),
        timeout=_coerce_float(config.get("timeout"), 120.0),
        max_tokens=_coerce_positive_int(config.get("max_tokens")),
        num_ctx=_coerce_positive_int(config.get("num_ctx")) or llm_config.num_ctx,
        reasoning_effort=normalize_reasoning_effort(config.get("reasoning_effort")),
        temperature=_coerce_float(config.get("temperature"), 0.2),
    )


def _set_override(overrides: dict[str, Any], key: str, value: Any) -> None:
    """只记录显式传入的覆盖值。"""
    if value is not None:
        overrides[key] = value


# ── 洞察选择（三个子策略共用）──────────────────────────────


def _select_relevant_insights_impl(
    phase: str,
    decision_state: dict[str, Any],
    parent: "LLMStrategy",
) -> list[InsightData]:
    """共用的洞察选择逻辑。"""
    try:
        from src.core.tasks.producer_challenge.gameplay.llm.insight_store import get_insight_store
        store = get_insight_store()
        insight_index = store.retrieve_insights(decision_state, limit=10)
        if not insight_index:
            return []

        from src.core.tasks.producer_challenge.gameplay.llm.prompt_renderer import render as render_template
        snapshot = decision_state.get("llm_snapshot", {})
        planning = snapshot.get("planning", {})
        if not isinstance(planning, dict):
            planning = {}
        next_gate = planning.get("next_gate", {})
        if not isinstance(next_gate, dict):
            next_gate = {}
        resource_pressure = planning.get("resource_pressure", {})
        if not isinstance(resource_pressure, dict):
            resource_pressure = {}

        insight_dicts = [
            {"id": i.id, "strategy_description": i.strategy_description, "when_to_apply": i.when_to_apply}
            for i in insight_index
        ]
        user_prompt = render_template(
            "insight_select.j2",
            phase=str(phase or ""),
            position=str(decision_state.get("position") or ""),
            idol_plan_type=str(snapshot.get("idol_plan_type") or ""),
            parameter_priority=str(snapshot.get("parameter_priority") or ""),
            route_bias=str(snapshot.get("idol_plan_label") or ""),
            next_gate=str(next_gate.get("gate_label") or next_gate.get("gate_type") or ""),
            weeks_until_gate=next_gate.get("weeks_until_gate"),
            resource_pressure=str(resource_pressure.get("summary") or ""),
            decision_goal=str(decision_state.get("decision_goal") or ""),
            insights=insight_dicts,
        )

        system_prompt = render_template("system_insight_selector.j2")
        client = parent._get_client()
        kwargs: dict[str, Any] = {
            "model": parent.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        apply_backend_compat_options(kwargs, reasoning_effort=parent.reasoning_effort, num_ctx=parent.num_ctx)
        response = client.chat.completions.create(**kwargs)
        text = extract_final_text(response)
        ids = _parse_insight_ids(text)
        if not ids:
            return []
        return store.get_by_ids(ids)
    except Exception as exc:
        logger.debug("[LLM] 洞察选择失败: {}", exc)
        return []


def _parse_insight_ids(text: str) -> list[int]:
    ids: list[int] = []
    for match in re.findall(r"\d+", text):
        try:
            ids.append(int(match))
        except ValueError:
            continue
    return ids[:3]


# ── Schedule LLM 策略 ───────────────────────────────────────


class ScheduleLLMStrategy:
    """周行程 LLM 决策。"""

    __slots__ = ("_parent",)

    def __init__(self, parent: "LLMStrategy") -> None:
        self._parent = parent

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> int | None:
        phase = str(decision_state.get("phase") or "")
        if phase != "schedule":
            return None

        self._parent._session.ensure_ready(ctx)

        prompt = build_user_prompt(decision_state)
        if not prompt:
            return None

        selected_insights = _select_relevant_insights_impl(phase, decision_state, self._parent)

        try:
            t0 = time.monotonic()
            self._parent._last_call_details = None
            action_index, reasoning = self._parent._call_and_parse(
                prompt, decision_state, ctx=ctx, selected_insights=selected_insights
            )
            elapsed = time.monotonic() - t0
            self._parent._call_count += 1
            self._parent._total_latency += elapsed

            if action_index is not None:
                legal = decision_state.get("legal_actions", [])
                if legal and action_index not in legal:
                    logger.warning(f"[LLM][schedule] 返回索引 {action_index} 不在合法动作 {legal} 中，忽略")
                    self._parent._dump_decision(
                        decision_state=decision_state, chosen_index=action_index,
                        resolved_index=None, fallback_used=True,
                        fallback_reason=f"非法索引 {action_index} (合法: {legal})",
                        decision_reasoning=reasoning, total_elapsed=elapsed,
                    )
                    return None
                candidate_name = get_candidate_name(decision_state, action_index)
                logger.info(f"[LLM][schedule] 决策: 选择 #{action_index} ({candidate_name}) [{elapsed:.1f}s]")
                logger.info(f"[LLM][schedule] 理由: {reasoning.get('why_this', '')[:200]}")
                self._parent._dump_decision(
                    decision_state=decision_state, chosen_index=action_index,
                    resolved_index=action_index, resolved_name=candidate_name,
                    decision_reasoning=reasoning, total_elapsed=elapsed,
                )
                self._parent._record_insight_usage(selected_insights, "success")
                self._parent._submit_insight_with_reasoning(ctx, decision_state, reasoning, "success")
                return action_index

            logger.debug(f"[LLM][schedule] 决策: 无法解析结果，交给 fallback")
            self._parent._dump_decision(
                decision_state=decision_state, chosen_index=None, resolved_index=None,
                fallback_used=True, fallback_reason="LLM 返回无法解析",
                decision_reasoning=reasoning, total_elapsed=elapsed,
            )
            self._parent._record_insight_usage(selected_insights, "failure")
            return None
        except Exception as exc:
            logger.warning(f"[LLM][schedule] 决策出错: {exc}")
            return None


# ── Battle LLM 策略 ─────────────────────────────────────────


class BattleLLMStrategy:
    """战斗 LLM 决策（lesson/exam）。"""

    __slots__ = ("_parent",)

    def __init__(self, parent: "LLMStrategy") -> None:
        self._parent = parent

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> int | None:
        phase = str(decision_state.get("phase") or "")
        if phase not in ("lesson", "exam"):
            return None

        self._parent._session.ensure_ready(ctx)

        prompt = build_user_prompt(decision_state)
        if not prompt:
            return None

        selected_insights = _select_relevant_insights_impl(phase, decision_state, self._parent)

        try:
            t0 = time.monotonic()
            self._parent._last_call_details = None
            action_index, reasoning = self._parent._call_and_parse(
                prompt, decision_state, ctx=ctx, selected_insights=selected_insights
            )
            elapsed = time.monotonic() - t0
            self._parent._call_count += 1
            self._parent._total_latency += elapsed

            if action_index is not None:
                legal = decision_state.get("legal_actions", [])
                if legal and action_index not in legal:
                    logger.warning(f"[LLM][battle] 返回索引 {action_index} 不在合法动作 {legal} 中，忽略")
                    self._parent._dump_decision(
                        decision_state=decision_state, chosen_index=action_index,
                        resolved_index=None, fallback_used=True,
                        fallback_reason=f"非法索引 {action_index} (合法: {legal})",
                        decision_reasoning=reasoning, total_elapsed=elapsed,
                    )
                    return None
                candidate_name = get_candidate_name(decision_state, action_index)
                logger.info(f"[LLM][battle] {phase} 决策: 选择 #{action_index} ({candidate_name}) [{elapsed:.1f}s]")
                logger.info(f"[LLM][battle] 理由: {reasoning.get('why_this', '')[:200]}")
                self._parent._dump_decision(
                    decision_state=decision_state, chosen_index=action_index,
                    resolved_index=action_index, resolved_name=candidate_name,
                    decision_reasoning=reasoning, total_elapsed=elapsed,
                )
                self._parent._record_insight_usage(selected_insights, "success")
                self._parent._submit_insight_with_reasoning(ctx, decision_state, reasoning, "success")
                return action_index

            logger.debug(f"[LLM][battle] {phase} 决策: 无法解析结果，交给 fallback")
            self._parent._dump_decision(
                decision_state=decision_state, chosen_index=None, resolved_index=None,
                fallback_used=True, fallback_reason="LLM 返回无法解析",
                decision_reasoning=reasoning, total_elapsed=elapsed,
            )
            self._parent._record_insight_usage(selected_insights, "failure")
            return None
        except Exception as exc:
            logger.warning(f"[LLM][battle] {phase} 决策出错: {exc}")
            return None


# ── Other LLM 策略 ──────────────────────────────────────────


class OtherLLMStrategy:
    """其他 LLM 决策（dialogue/p_drink/consult/item_select/modal 等）。"""

    __slots__ = ("_parent",)

    def __init__(self, parent: "LLMStrategy") -> None:
        self._parent = parent

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> int | None:
        phase = str(decision_state.get("phase") or "")
        if phase in ("schedule", "lesson", "exam"):
            return None

        self._parent._session.ensure_ready(ctx)

        prompt = build_user_prompt(decision_state)
        if not prompt:
            return None

        selected_insights = _select_relevant_insights_impl(phase, decision_state, self._parent)

        try:
            t0 = time.monotonic()
            self._parent._last_call_details = None
            action_index, reasoning = self._parent._call_and_parse(
                prompt, decision_state, ctx=ctx, selected_insights=selected_insights
            )
            elapsed = time.monotonic() - t0
            self._parent._call_count += 1
            self._parent._total_latency += elapsed

            if action_index is not None:
                legal = decision_state.get("legal_actions", [])
                if legal and action_index not in legal:
                    logger.warning(f"[LLM][other] {phase} 返回索引 {action_index} 不在合法动作 {legal} 中，忽略")
                    self._parent._dump_decision(
                        decision_state=decision_state, chosen_index=action_index,
                        resolved_index=None, fallback_used=True,
                        fallback_reason=f"非法索引 {action_index} (合法: {legal})",
                        decision_reasoning=reasoning, total_elapsed=elapsed,
                    )
                    return None
                candidate_name = get_candidate_name(decision_state, action_index)
                logger.info(f"[LLM][other] {phase} 决策: 选择 #{action_index} ({candidate_name}) [{elapsed:.1f}s]")
                logger.info(f"[LLM][other] 理由: {reasoning.get('why_this', '')[:200]}")
                self._parent._dump_decision(
                    decision_state=decision_state, chosen_index=action_index,
                    resolved_index=action_index, resolved_name=candidate_name,
                    decision_reasoning=reasoning, total_elapsed=elapsed,
                )
                self._parent._record_insight_usage(selected_insights, "success")
                self._parent._submit_insight_with_reasoning(ctx, decision_state, reasoning, "success")
                return action_index

            logger.debug(f"[LLM][other] {phase} 决策: 无法解析结果，交给 fallback")
            self._parent._dump_decision(
                decision_state=decision_state, chosen_index=None, resolved_index=None,
                fallback_used=True, fallback_reason="LLM 返回无法解析",
                decision_reasoning=reasoning, total_elapsed=elapsed,
            )
            self._parent._record_insight_usage(selected_insights, "failure")
            return None
        except Exception as exc:
            logger.warning(f"[LLM][other] {phase} 决策出错: {exc}")
            return None


# ── Phase 路由映射 ───────────────────────────────────────────


_PHASE_TO_STRATEGY = {
    "schedule": "schedule",
    "lesson": "battle",
    "exam": "battle",
    "dialogue": "other",
    "p_drink": "other",
    "skill_reward": "other",
    "consult": "other",
    "item_select": "other",
    "modal": "other",
}


# ── 统一 LLM 策略入口 ───────────────────────────────────────


class LLMStrategy:
    """通过 OpenAI 兼容 API 做游戏决策的统一策略。

    内部持有3个子策略，按 decision_state["phase"] 路由：
    - ScheduleLLMStrategy: schedule
    - BattleLLMStrategy: lesson / exam
    - OtherLLMStrategy: dialogue / p_drink / skill_reward / consult / item_select / modal
    """

    _EMPTY_CONTENT_MAX_RETRIES = 2

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        num_ctx: int | None = None,
        summarizer_model: str | None = None,
        summarizer_base_url: str | None = None,
        summarizer_api_key: str | None = None,
        summarizer_timeout: float | None = None,
        compression_trigger_ratio: float = 0.75,
        summary_target_tokens: int = 768,
        recent_turn_window: int = 3,
        historical_summary_limit: int = 3,
        think: Any | None = None,
        insight_base_url: str | None = None,
        insight_model: str | None = None,
        insight_api_key: str | None = None,
        insight_timeout: float | None = None,
        insight_max_tokens: int | None = None,
        insight_num_ctx: int | None = None,
        insight_reasoning_effort: str | None = None,
        insight_temperature: float | None = None,
    ) -> None:
        self._llm_overrides: dict[str, Any] = {}
        _set_override(self._llm_overrides, "base_url", base_url)
        _set_override(self._llm_overrides, "model", model)
        _set_override(self._llm_overrides, "api_key", api_key)
        _set_override(self._llm_overrides, "timeout", timeout)
        _set_override(self._llm_overrides, "temperature", temperature)
        _set_override(self._llm_overrides, "max_tokens", max_tokens)
        _set_override(self._llm_overrides, "reasoning_effort", reasoning_effort)
        _set_override(self._llm_overrides, "num_ctx", num_ctx)
        if think is not None and str(think).strip() != "":
            self._llm_overrides["reasoning_effort"] = normalize_legacy_think(think)

        self._summarizer_overrides: dict[str, Any] = {}
        _set_override(self._summarizer_overrides, "model", summarizer_model)
        _set_override(self._summarizer_overrides, "base_url", summarizer_base_url)
        _set_override(self._summarizer_overrides, "api_key", summarizer_api_key)
        _set_override(self._summarizer_overrides, "timeout", summarizer_timeout)

        self._insight_overrides: dict[str, Any] = {}
        _set_override(self._insight_overrides, "base_url", insight_base_url)
        _set_override(self._insight_overrides, "model", insight_model)
        _set_override(self._insight_overrides, "api_key", insight_api_key)
        _set_override(self._insight_overrides, "timeout", insight_timeout)
        _set_override(self._insight_overrides, "max_tokens", insight_max_tokens)
        _set_override(self._insight_overrides, "num_ctx", insight_num_ctx)
        _set_override(self._insight_overrides, "reasoning_effort", insight_reasoning_effort)
        _set_override(self._insight_overrides, "temperature", insight_temperature)

        self.base_url = ""
        self.model = ""
        self.api_key = ""
        self.timeout = 0.0
        self.temperature = 0.0
        self.max_tokens: int | None = None
        self.reasoning_effort = "medium"
        self.num_ctx = 0
        self.summarizer_model = ""
        self.summarizer_base_url = ""
        self.summarizer_api_key = ""
        self.summarizer_timeout = 0.0
        self._client: OpenAI | None = None
        self._summarizer_client: OpenAI | None = None
        self._client_signature: tuple[str, str, float] | None = None
        self._summarizer_client_signature: tuple[str, str, float] | None = None
        self._insight_signature: tuple[bool, str, str, str, float, int | None, int, str, float] | None = None
        self._call_count = 0
        self._total_latency = 0.0
        self._last_call_details: LLMCallDetails | None = None
        initial_num_ctx = (
            _coerce_int(num_ctx, int(_DEFAULT_LLM_CONFIG["num_ctx"]))
            if num_ctx is not None
            else int(_DEFAULT_LLM_CONFIG["num_ctx"])
        )
        self._session = SessionManager(
            recent_turn_window=recent_turn_window,
            compression_trigger_ratio=compression_trigger_ratio,
            num_ctx=initial_num_ctx,
            summary_target_tokens=summary_target_tokens,
            historical_summary_limit=historical_summary_limit,
        )
        self._insight_gen: Any = None
        self._insight_enabled = True
        self._insight_base_url = ""
        self._insight_model = ""
        self._insight_api_key = ""
        self._insight_timeout = 0.0
        self._insight_max_tokens: int | None = None
        self._insight_num_ctx = 0
        self._insight_reasoning_effort = "medium"
        self._insight_temperature = 0.0

        # 内部3个子策略
        self._schedule = ScheduleLLMStrategy(self)
        self._battle = BattleLLMStrategy(self)
        self._other = OtherLLMStrategy(self)
        self._refresh_runtime_config()

    # ── Client 管理 ───────────────────────────────────────

    def _refresh_runtime_config(self) -> None:
        """从全局配置刷新运行时参数，并在连接参数变化时丢弃旧客户端。"""
        llm_config = _build_runtime_llm_config(self._llm_overrides)
        client_signature = (llm_config.base_url, llm_config.api_key, llm_config.timeout)
        if self._client_signature is not None and self._client_signature != client_signature:
            self._client = None
        self._client_signature = client_signature

        self.base_url = llm_config.base_url
        self.model = llm_config.model
        self.api_key = llm_config.api_key
        self.timeout = llm_config.timeout
        self.temperature = llm_config.temperature
        self.max_tokens = llm_config.max_tokens
        self.reasoning_effort = llm_config.reasoning_effort
        self.num_ctx = llm_config.num_ctx
        self._session.num_ctx = llm_config.num_ctx or int(_DEFAULT_LLM_CONFIG["num_ctx"])

        self.summarizer_model = str(self._summarizer_overrides.get("model") or llm_config.model)
        self.summarizer_base_url = _normalize_base_url(
            self._summarizer_overrides.get("base_url") or llm_config.base_url,
            llm_config.base_url,
        )
        self.summarizer_api_key = str(
            self._summarizer_overrides.get("api_key")
            if self._summarizer_overrides.get("api_key") is not None
            else llm_config.api_key
        )
        self.summarizer_timeout = _coerce_float(
            self._summarizer_overrides.get("timeout"),
            llm_config.timeout,
        )
        summarizer_signature = (
            self.summarizer_base_url,
            self.summarizer_api_key,
            self.summarizer_timeout,
        )
        if (
            self._summarizer_client_signature is not None
            and self._summarizer_client_signature != summarizer_signature
        ):
            self._summarizer_client = None
        self._summarizer_client_signature = summarizer_signature

        insight_config = _build_runtime_insight_config(llm_config, self._insight_overrides)
        insight_signature = (
            insight_config.enabled,
            insight_config.base_url,
            insight_config.model,
            insight_config.api_key,
            insight_config.timeout,
            insight_config.max_tokens,
            insight_config.num_ctx,
            insight_config.reasoning_effort,
            insight_config.temperature,
        )
        if self._insight_signature is not None and self._insight_signature != insight_signature:
            self._dispose_insight_generator()
        self._insight_signature = insight_signature
        self._insight_enabled = insight_config.enabled
        self._insight_base_url = insight_config.base_url
        self._insight_model = insight_config.model
        self._insight_api_key = insight_config.api_key
        self._insight_timeout = insight_config.timeout
        self._insight_max_tokens = insight_config.max_tokens
        self._insight_num_ctx = insight_config.num_ctx
        self._insight_reasoning_effort = insight_config.reasoning_effort
        self._insight_temperature = insight_config.temperature

    def _dispose_insight_generator(self) -> None:
        """等待旧洞察任务结束并丢弃生成器引用。"""
        if self._insight_gen is None:
            return
        try:
            self._insight_gen.wait_all()
        except Exception as exc:
            logger.debug("[LLM] 释放旧 InsightGenerator 失败: {}", exc)
        self._insight_gen = None

    def _get_client(self) -> OpenAI:
        """获取主 LLM 客户端，必要时按最新配置重建。"""
        self._refresh_runtime_config()
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._client

    def _get_summarizer_client(self) -> OpenAI:
        """获取会话压缩用客户端，未单独配置时复用主客户端。"""
        self._refresh_runtime_config()
        if (
            self.summarizer_base_url == self.base_url
            and self.summarizer_api_key == self.api_key
            and self.summarizer_timeout == self.timeout
        ):
            return self._get_client()
        if self._summarizer_client is None:
            self._summarizer_client = OpenAI(
                base_url=self.summarizer_base_url,
                api_key=self.summarizer_api_key,
                timeout=self.summarizer_timeout,
            )
        return self._summarizer_client

    def _get_insight_generator(self) -> Any:
        """获取策略洞察生成器，关闭配置时返回 None。"""
        self._refresh_runtime_config()
        if not self._insight_enabled:
            self._dispose_insight_generator()
            return None
        if self._insight_gen is None:
            try:
                from src.core.tasks.producer_challenge.gameplay.llm.insight_generator import InsightGenerator
                self._insight_gen = InsightGenerator(
                    base_url=self._insight_base_url,
                    model=self._insight_model,
                    api_key=self._insight_api_key,
                    timeout=self._insight_timeout,
                    max_tokens=self._insight_max_tokens,
                    num_ctx=self._insight_num_ctx,
                    reasoning_effort=self._insight_reasoning_effort,
                    temperature=self._insight_temperature,
                )
            except Exception as exc:
                logger.warning("[LLM] 初始化 InsightGenerator 失败: {}", exc)
        return self._insight_gen

    # ── 会话生命周期 ──────────────────────────────────────

    def reset_session(self, ctx: "ProduceContext | None" = None, *, session_id: str = "") -> None:
        """重置当前局的 LLM 会话状态。"""
        self._refresh_runtime_config()
        self._session.reset(ctx, session_id=session_id)

    def flush_session(self, ctx: "ProduceContext | None" = None, *, force_compact: bool = True) -> None:
        """在局结束时写回会话记忆并提交洞察任务。"""
        self._refresh_runtime_config()
        if force_compact and self._session.state.recent_turns:
            self._compact_session_memory(current_planning_note="", compact_all=True)
        self._session.mirror_to_context(ctx)
        gen = self._get_insight_generator()
        if gen:
            gen.submit_phase_insights(ctx)
            gen.submit_review(ctx)
            gen.wait_all()

    # ── 统一策略入口 ─────────────────────────────────────

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> int | None:
        """按当前阶段路由到具体 LLM 子策略。"""
        if not candidates or decision_state is None:
            return None
        self._refresh_runtime_config()

        phase = str(decision_state.get("phase") or "")
        target = _PHASE_TO_STRATEGY.get(phase, "schedule")

        if target == "battle":
            return self._battle(app, ctx, candidates, decision_state)
        elif target == "other":
            return self._other(app, ctx, candidates, decision_state)
        else:
            return self._schedule(app, ctx, candidates, decision_state)

    # ── 洞察选择 ─────────────────────────────────────────

    def _record_insight_usage(self, insights: list[InsightData], outcome_type: str) -> None:
        """记录被引用洞察的使用结果。"""
        if not insights:
            return
        try:
            from src.core.tasks.producer_challenge.gameplay.llm.insight_store import get_insight_store
            store = get_insight_store()
            for insight in insights:
                store.record_usage(insight.id, outcome_type)
        except Exception as e:
            logger.debug(f"LLMStrategy: 操作失败: {e}")


    # ── 决策 Dump ───────────────────────────────────────

    def _dump_decision(
        self, *, decision_state: dict[str, Any], chosen_index: int | None,
        resolved_index: int | None, resolved_name: str = "", resolved_action_id: str = "",
        fallback_used: bool = False, fallback_reason: str = "",
        decision_reasoning: dict[str, str] | None = None,
        total_elapsed: float = 0.0,
    ) -> None:
        """把单次 LLM 决策详情写入调试 dump。"""
        try:
            dumper = DecisionDumper.get_instance()
            if not dumper.enabled:
                return
            details = self._last_call_details or LLMCallDetails()
            if not resolved_action_id and resolved_index is not None:
                for c in decision_state.get("candidates", []):
                    if c.get("index") == resolved_index:
                        resolved_action_id = str(c.get("id") or c.get("action_id") or "")
                        if not resolved_name:
                            resolved_name = str(c.get("name") or c.get("label") or "")
                        break
            if decision_reasoning:
                explanation = dict(decision_state.get("decision_explanation") or {})
                explanation["why_this_action"] = decision_reasoning.get("why_this", "")
                explanation["why_not_others"] = decision_reasoning.get("why_not_others", "")
                decision_state["decision_explanation"] = explanation
            dumper.record(
                decision_state=decision_state, system_prompt=details.system_prompt,
                user_prompt=details.user_prompt, llm_raw_content=details.raw_content,
                llm_raw_reasoning=details.raw_reasoning, llm_cleaned_output=details.cleaned_output,
                llm_model=self.model, llm_elapsed_sec=details.elapsed_sec,
                chosen_index=chosen_index, resolved_index=resolved_index,
                resolved_action_id=resolved_action_id, resolved_name=resolved_name,
                fallback_used=fallback_used, fallback_reason=fallback_reason,
                total_elapsed_sec=total_elapsed,
            )
        except Exception as exc:
            logger.debug("[LLM] dump 写入异常: {}", exc)

    def _submit_insight_with_reasoning(
        self, ctx: "ProduceContext", decision_state: dict[str, Any],
        reasoning: dict[str, str], outcome_type: str,
    ) -> None:
        """将本次决策理由提交给后台洞察生成器。"""
        gen = self._get_insight_generator()
        if not gen:
            return
        decision_state["chosen_decision_reasoning"] = reasoning
        gen.submit_step_insight(ctx, decision_state, outcome_type)

    # ── LLM 调用 ─────────────────────────────────────────

    def _call_and_parse(
        self, prompt: str, state: dict[str, Any], *,
        ctx: "ProduceContext | None" = None,
        selected_insights: list[InsightData] | None = None,
    ) -> tuple[int | None, dict[str, str]]:
        """调用 LLM 并解析动作编号与理由。"""
        client = self._get_client()
        legal = state.get("legal_actions", [])
        phase = state.get("phase", "unknown")
        snapshot = state.get("llm_snapshot", {})
        system_prompt = build_system_prompt(phase, snapshot)
        base_messages = prepare_messages(
            system_prompt=system_prompt, current_prompt=prompt, state=state,
            selected_insights=list(selected_insights or []),
            session=self._session, ctx=ctx,
        )

        base_kwargs: dict[str, Any] = {"model": self.model, "temperature": self.temperature}
        if self.max_tokens is not None:
            base_kwargs["max_tokens"] = self.max_tokens
        apply_backend_compat_options(base_kwargs, reasoning_effort=self.reasoning_effort, num_ctx=self.num_ctx)

        logger.debug("[LLM] reasoning_effort={}", self.reasoning_effort)
        logger.debug("[LLM] ====== SYSTEM PROMPT [{}] ======\n{}", phase, system_prompt)
        logger.debug("[LLM] ====== USER PROMPT ======\n{}", prompt)
        logger.debug("[LLM] 会话指标: {}", json.dumps(self._session.state.last_prompt_metrics, ensure_ascii=False))

        details = LLMCallDetails(system_prompt=system_prompt, user_prompt=prompt)
        t0 = time.monotonic()
        response = None
        final_text = ""

        for attempt in range(1 + self._EMPTY_CONTENT_MAX_RETRIES):
            messages = list(base_messages)
            if attempt > 0:
                legal_hint = ", ".join(str(x) for x in legal) if legal else "无"
                messages.append({"role": "user", "content": (
                    "你上次没有给出最终答案。现在只输出 JSON，不要附加任何其他文本。"
                    f"合法动作: {legal_hint}。\n"
                    '{"why_this":"理由","why_not_others":"理由","action_index":编号}'
                )})
            kwargs = {**base_kwargs, "messages": messages}
            if attempt > 0:
                kwargs["temperature"] = 0.0
                kwargs["max_tokens"] = min(int(self.max_tokens or 64), 64)
                apply_backend_compat_options(kwargs, reasoning_effort=self.reasoning_effort,
                                             num_ctx=self.num_ctx, for_retry=True)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                message = str(exc).lower()
                if any(kw in message for kw in ("reasoning_effort", "think", "unsupported", "extra_body")):
                    logger.debug("[LLM] reasoning/兼容参数不支持，回退到最小请求重试")
                    kwargs.pop("reasoning_effort", None)
                    kwargs.pop("extra_body", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            if response and response.choices:
                raw_content, raw_reasoning = extract_raw_fields(response)
                details.raw_content = raw_content
                details.raw_reasoning = raw_reasoning
                logger.info("[LLM] content({} 字符): {}", len(raw_content), repr(raw_content[:300]))
                if raw_reasoning:
                    logger.info("[LLM] reasoning({} 字符): {}", len(raw_reasoning), repr(raw_reasoning[:300]))
                finish_reason = str(response.choices[0].finish_reason or "")
                if finish_reason:
                    logger.debug("[LLM] finish_reason={}", finish_reason)

            final_text = extract_final_text(response)
            if final_text:
                break
            if attempt >= self._EMPTY_CONTENT_MAX_RETRIES:
                logger.warning("[LLM] 重试 {} 次后 content 仍为空，交给 fallback", attempt)
                details.elapsed_sec = time.monotonic() - t0
                self._last_call_details = details
                return None, {"why_this": "", "why_not_others": ""}
            logger.warning("[LLM] content 为空(第{}次)，reasoning={} 字符，追加提示重试",
                           attempt + 1, len(details.raw_reasoning))

        details.elapsed_sec = time.monotonic() - t0
        if not final_text:
            self._last_call_details = details
            return None, {"why_this": "", "why_not_others": ""}
        cleaned = re.sub(r"<think>.*?", "", final_text, flags=re.IGNORECASE | re.DOTALL).strip()
        details.cleaned_output = cleaned
        logger.info("[LLM] 清理后最终输出: [{}]", cleaned[:200])
        self._last_call_details = details
        reasoning = parse_decision_reasoning(cleaned)
        action_index = parse_action_index(cleaned, legal)
        return action_index, reasoning

    def _compact_session_memory(self, current_planning_note: str, compact_all: bool = False) -> None:
        """压缩当前局的短期会话记忆。"""
        try:
            summarizer = self._get_summarizer_client()
            recent = self._session.state.recent_turns
            if not recent:
                return
            to_compact = [r for r in recent if not r.get("_compact", False)]
            if not to_compact:
                return
            oldest = to_compact[0]
            messages = [
                {"role": "system", "content": "你是一个摘要助手。请将以下对话历史压缩为简洁的摘要，保留关键决策、状态变化和理由。"},
                {"role": "user", "content": f"对话历史:\n{oldest.get('user_prompt', '')}\n\n模型回复:\n{oldest.get('llm_response', '')}"},
            ]
            kwargs: dict[str, Any] = {"model": self.summarizer_model, "messages": messages, "temperature": 0.3}
            apply_backend_compat_options(kwargs, reasoning_effort="low", num_ctx=self.num_ctx)
            resp = summarizer.chat.completions.create(**kwargs)
            summary_text = extract_final_text(resp)
            if summary_text:
                summary_text = sanitize_summary_text(summary_text)
            self._session.state.pending_summary = summary_text or ""
            for r in to_compact:
                r["_compact"] = True
        except Exception as exc:
            logger.debug("[LLM] 会话压缩失败: {}", exc)

    # ── 统计 ─────────────────────────────────────────────

    @property
    def stats(self) -> str:
        avg = (self._total_latency / self._call_count) if self._call_count else 0
        return f"calls={self._call_count}, avg_latency={avg:.1f}s"


# ── 工厂与注入 ─────────────────────────────────────────────


def create_llm_strategy(
    base_url: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMStrategy:
    """创建 LLM 策略，未显式传入的参数会动态读取全局配置。"""
    return LLMStrategy(base_url=base_url, model=model, **kwargs)


def inject_llm_strategy(
    ctx: "ProduceContext",
    strategy: LLMStrategy | None = None,
    *,
    base_url: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMStrategy:
    """把同一个 LLM 策略注入到所有决策阶段。"""
    if strategy is None:
        strategy = create_llm_strategy(base_url=base_url, model=model, **kwargs)
    ctx.schedule_strategy = strategy
    ctx.lesson_strategy = strategy
    ctx.exam_strategy = strategy
    ctx.dialogue_strategy = strategy
    ctx.skill_reward_strategy = strategy
    ctx.p_drink_strategy = strategy
    ctx.consult_strategy = strategy
    ctx.item_select_strategy = strategy
    ctx.modal_strategy = strategy
    dumper = DecisionDumper.get_instance()
    dumper.start_session()
    strategy.reset_session(ctx)
    logger.debug(f"[LLM] 策略已注入所有决策字段 | model={strategy.model} base_url={strategy.base_url}")
    return strategy


__all__ = [
    "LLMStrategy",
    "ScheduleLLMStrategy",
    "BattleLLMStrategy",
    "OtherLLMStrategy",
    "create_llm_strategy",
    "inject_llm_strategy",
]
