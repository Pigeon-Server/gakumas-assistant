"""LLM 决策策略 — 通过 OpenAI 兼容 API 为各阶段提供智能决策。
策略回调签名：strategy(app, ctx, candidates, decision_state) → int | dict | None
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from openai import OpenAI

from src.utils.logger import logger
from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

# ── Jinja2 模板：按阶段选择系统提示词 ────────────────
from src.core.tasks.producer_challenge.gameplay.llm.prompt_renderer import render as _render_template

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


@dataclass
class _LLMCallDetails:
    """_call_and_parse 内部保存的 LLM 调用详情，供 __call__ 读取后写入 dump。"""
    system_prompt: str = ""
    user_prompt: str = ""
    raw_content: str = ""
    raw_reasoning: str = ""
    cleaned_output: str = ""
    elapsed_sec: float = 0.0


class LLMStrategy:
    """通过 OpenAI 兼容 API 做游戏决策的统一策略。

    所有阶段（schedule/lesson/exam/dialogue/skill_reward/p_drink/consult）
    共用同一个实例，根据 decision_state["phase"] 自动切换提示词。
    """

    def __init__(
        self,
        base_url: str = "http://192.168.100.10:11434/v1/",
        model: str = "gpt-oss:20b",
        api_key: str = "ollama",
        timeout: float = 60.0,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        think: str = "low",
        num_ctx: int = 8192,
    ):
        """处理init并返回结果。

        Args:
            base_url: 用于提供base、url相关输入。
            model: 用于提供model相关输入。
            api_key: 用于提供api、key相关输入。
            timeout: 用于提供timeout相关输入。
            temperature: 用于提供temperature相关输入。
            max_tokens: 用于提供max、tokens相关输入。
            think: 用于提供think相关输入。
            num_ctx: 用于提供num、ctx相关输入。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        # max_tokens 为 None 时不传给 API，让模型自行控制（避免 thinking 占满 token 预算导致 content 为空）
        self.max_tokens = max_tokens
        self.think = think
        self.num_ctx = num_ctx
        self._client: OpenAI | None = None
        self._call_count = 0
        self._total_latency = 0.0
        # 最近一次 LLM 调用的详情（供 dump 系统使用）
        self._last_call_details: _LLMCallDetails | None = None

    def _get_client(self) -> OpenAI:
        """获取client并返回结果。

        Returns:
            OpenAI: 返回值类型见注解，语义由函数用途决定。
        """
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    # ── 策略回调入口（供 invoke_decision_strategy 调用）──

    def __call__(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> int | None:
        """策略回调入口。返回候选动作索引或 None（交给后续 fallback）。"""
        if not candidates or decision_state is None:
            return None

        phase = decision_state.get("phase", "unknown")
        prompt = self._build_prompt(decision_state)
        if not prompt:
            return None

        try:
            t0 = time.monotonic()
            self._last_call_details = None
            action_index = self._call_and_parse(prompt, decision_state)
            elapsed = time.monotonic() - t0
            self._call_count += 1
            self._total_latency += elapsed

            candidate_name = ""
            if action_index is not None:
                # 验证索引合法性
                legal = decision_state.get("legal_actions", [])
                if legal and action_index not in legal:
                    logger.warning(
                        f"[LLM] 返回索引 {action_index} 不在合法动作 {legal} 中，忽略"
                    )
                    # dump: LLM 返回了非法索引
                    self._dump_decision(
                        decision_state=decision_state,
                        chosen_index=action_index,
                        resolved_index=None,
                        fallback_used=True,
                        fallback_reason=f"非法索引 {action_index} (合法: {legal})",
                        total_elapsed=elapsed,
                    )
                    return None
                candidate_name = self._get_candidate_name(
                    decision_state, action_index
                )
                logger.info(
                    f"[LLM] {phase} 决策: 选择 #{action_index}"
                    f" ({candidate_name}) [{elapsed:.1f}s]"
                )
                # dump: 正常决策
                self._dump_decision(
                    decision_state=decision_state,
                    chosen_index=action_index,
                    resolved_index=action_index,
                    resolved_name=candidate_name,
                    total_elapsed=elapsed,
                )
                return action_index

            logger.debug(f"[LLM] {phase} 决策: 无法解析结果，交给 fallback")
            # dump: LLM 无法解析
            self._dump_decision(
                decision_state=decision_state,
                chosen_index=None,
                resolved_index=None,
                fallback_used=True,
                fallback_reason="LLM 返回无法解析",
                total_elapsed=elapsed,
            )
            return None

        except Exception as exc:
            logger.warning(f"[LLM] {phase} 决策出错: {exc}")
            return None

    def _dump_decision(
        self,
        *,
        decision_state: dict[str, Any],
        chosen_index: int | None,
        resolved_index: int | None,
        resolved_name: str = "",
        resolved_action_id: str = "",
        fallback_used: bool = False,
        fallback_reason: str = "",
        total_elapsed: float = 0.0,
    ):
        """处理dump、decision并返回结果。

        Args:
            decision_state: 决策快照，包含上下文、候选项与当前理由。
            chosen_index: 用于提供chosen、index相关输入。
            resolved_index: 用于提供resolved、index相关输入。
            resolved_name: 用于提供resolved、name相关输入。
            resolved_action_id: 业务对象标识符，用于索引或匹配目标实体。
            fallback_used: 用于提供fallback、used相关输入。
            fallback_reason: 用于提供fallback、reason相关输入。
            total_elapsed: 用于提供total、elapsed相关输入。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
        try:
            dumper = DecisionDumper.get_instance()
            if not dumper.enabled:
                return
            details = self._last_call_details or _LLMCallDetails()
            # 从候选中提取 action_id
            if not resolved_action_id and resolved_index is not None:
                for c in decision_state.get("candidates", []):
                    if c.get("index") == resolved_index:
                        resolved_action_id = str(c.get("id") or c.get("action_id") or "")
                        if not resolved_name:
                            resolved_name = str(c.get("name") or c.get("label") or "")
                        break
            dumper.record(
                decision_state=decision_state,
                system_prompt=details.system_prompt,
                user_prompt=details.user_prompt,
                llm_raw_content=details.raw_content,
                llm_raw_reasoning=details.raw_reasoning,
                llm_cleaned_output=details.cleaned_output,
                llm_model=self.model,
                llm_elapsed_sec=details.elapsed_sec,
                chosen_index=chosen_index,
                resolved_index=resolved_index,
                resolved_action_id=resolved_action_id,
                resolved_name=resolved_name,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                total_elapsed_sec=total_elapsed,
            )
        except Exception as exc:
            logger.debug("[LLM] dump 写入异常: {}", exc)

    # ── Prompt 构建 ──────────────────────────────────────

    def _build_system_prompt(self, phase: str, snapshot: dict[str, Any] | None = None) -> str:
        """根据阶段选择对应的系统提示词模板。
        部分系统模板（如 schedule）需要传入 snapshot 中的上下文数据。
        """
        template_name = _SYSTEM_TEMPLATE_MAP.get(phase, "system_default.j2")
        kwargs = dict(snapshot) if snapshot else {}
        try:
            return _render_template(template_name, **kwargs)
        except Exception as exc:
            logger.warning("[LLM] 渲染系统模板 {} 失败: {}", template_name, exc)
            return _render_template("system_default.j2")

    def _build_prompt(self, state: dict[str, Any]) -> str:
        """使用 Jinja2 模板渲染用户提示词（游戏状态 + 候选动作）。"""
        snapshot = state.get("llm_snapshot", {})
        llm_actions = state.get("llm_actions") or []

        # 渲染局面快照
        try:
            rendered_snapshot = _render_template("state_snapshot.j2", **snapshot)
        except Exception as exc:
            logger.warning("[LLM] 渲染 state_snapshot.j2 失败: {}", exc)
            rendered_snapshot = f"## 当前局面\n（模板渲染失败: {exc}）"

        # 渲染完整用户提示词（快照 + 动作列表 + 决策指令）
        try:
            return _render_template(
                "action_select.j2",
                snapshot=rendered_snapshot,
                actions=llm_actions,
            )
        except Exception as exc:
            logger.warning("[LLM] 渲染 action_select.j2 失败: {}", exc)
            # 最小回退：拼接快照和简单动作列表
            parts = [rendered_snapshot, "\n## 合法动作"]
            for a in llm_actions:
                idx = a.get("index", 0)
                label = a.get("label", "?")
                desc = a.get("description", "")
                line = f"{idx}: {label}"
                if desc:
                    line += f" - {desc}"
                parts.append(line)
            parts.append("\n请选择当前最优动作，只输出动作编号：")
            return "\n".join(parts)

    @staticmethod
    def _get_candidate_name(state: dict[str, Any], index: int) -> str:
        """获取候选项、name并返回结果。

        Args:
            state: 用于提供状态相关输入。
            index: 用于提供index相关输入。

        Returns:
            str: 处理后的文本结果。
        """
        for c in state.get("candidates", []):
            if c.get("index") == index:
                return c.get("name", c.get("label", f"#{index}"))
        return f"#{index}"

    # ── LLM 调用 ────────────────────────────────────────

    # content 为空时最多重试次数
    _EMPTY_CONTENT_MAX_RETRIES = 2

    def _call_and_parse(
        self,
        prompt: str,
        state: dict[str, Any],
    ) -> int | None:
        """调用 LLM 并解析返回的动作编号。

        当模型只输出 reasoning 但 content 为空时（thinking 占满 token 预算），
        会自动追加提示重试，最多重试 _EMPTY_CONTENT_MAX_RETRIES 次。
        """
        client = self._get_client()
        legal = state.get("legal_actions", [])
        phase = state.get("phase", "unknown")

        # 根据阶段渲染系统提示词（部分模板需要 snapshot 数据）
        snapshot = state.get("llm_snapshot", {})
        system_prompt = self._build_system_prompt(phase, snapshot)

        # 构建消息列表（重试时会追加 assistant/user 消息）
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # 构建基础请求参数
        base_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        # max_tokens 为 None 时不传，让模型自行管理 thinking / content 的 token 分配
        if self.max_tokens is not None:
            base_kwargs["max_tokens"] = self.max_tokens

        # Ollama think 参数
        if self.think and self.think != "false":
            think_value = self.think if self.think in ("low", "medium", "high") else True
            base_kwargs["extra_body"] = {"think": think_value, "options": {"num_ctx": self.num_ctx}}
        elif self.num_ctx:
            base_kwargs["extra_body"] = {"options": {"num_ctx": self.num_ctx}}

        # 调试：打印系统提示词和用户提示词
        logger.debug("[LLM] ====== SYSTEM PROMPT [{}] ======\n{}", phase, system_prompt)
        logger.debug("[LLM] ====== USER PROMPT ======\n{}", prompt)

        # 初始化本次调用详情
        details = _LLMCallDetails(system_prompt=system_prompt, user_prompt=prompt)
        t0 = time.monotonic()

        # 重试循环：content 为空但有 reasoning 时追加提示重试
        response = None
        for attempt in range(1 + self._EMPTY_CONTENT_MAX_RETRIES):
            kwargs = {**base_kwargs, "messages": list(messages)}
            if attempt > 0:
                # 二次重试时强制进入“短输出直答”模式，避免继续长链思考。
                kwargs["temperature"] = 0.0
                kwargs["max_tokens"] = min(int(self.max_tokens or 64), 64)
                extra_body = dict(kwargs.get("extra_body") or {})
                # 关闭 think，让模型直接给最终编号。
                extra_body.pop("think", None)
                if extra_body:
                    kwargs["extra_body"] = extra_body
                else:
                    kwargs.pop("extra_body", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                # 若 think 参数不支持，回退不带 think 的请求
                if "think" in str(exc).lower() or "unsupported" in str(exc).lower():
                    logger.debug("[LLM] think 参数不支持，回退重试")
                    kwargs.pop("extra_body", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            # 提取原始 response 关键字段
            if response and response.choices:
                _msg_dict = self._coerce_dict(
                    response.choices[0].message if hasattr(response.choices[0], "message") else response.choices[0]
                )
                details.raw_content = self._coerce_text(_msg_dict.get("content"))
                details.raw_reasoning = self._coerce_text(
                    _msg_dict.get("reasoning_content") or _msg_dict.get("reasoning")
                )
                logger.info("[LLM] content({} 字符): {}", len(details.raw_content), repr(details.raw_content[:300]))
                if details.raw_reasoning:
                    logger.info("[LLM] reasoning({} 字符): {}", len(details.raw_reasoning), repr(details.raw_reasoning[:300]))
                finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "")
                if finish_reason:
                    logger.debug("[LLM] finish_reason={}", finish_reason)

            final_text = self._extract_final_text(response)
            if final_text:
                break  # 成功拿到 content

            if attempt >= self._EMPTY_CONTENT_MAX_RETRIES:
                logger.warning(
                    "[LLM] 重试 {} 次后 content 仍为空，交给 fallback",
                    attempt,
                )
                details.elapsed_sec = time.monotonic() - t0
                self._last_call_details = details
                return None

            # content 为空即重试：无论是否携带 reasoning，都要求模型直接输出编号。
            logger.warning(
                "[LLM] content 为空(第{}次)，reasoning={} 字符，追加提示重试",
                attempt + 1,
                len(details.raw_reasoning),
            )
            # 不把 reasoning 回灌给模型，避免它沿着思考链继续“只思考不作答”。
            legal_hint = ", ".join(str(x) for x in legal) if legal else "无"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "你上次没有给出最终答案。现在只输出一个合法动作编号（纯数字）。"
                        f"合法动作: {legal_hint}。禁止解释，禁止输出思考过程。"
                    ),
                },
            ]

        details.elapsed_sec = time.monotonic() - t0

        if not final_text:
            self._last_call_details = details
            return None

        # 去除推理标签（content 里可能还有 <think> 标签）
        cleaned = re.sub(
            r"<think>.*?</think>", "", final_text, flags=re.IGNORECASE | re.DOTALL
        )
        cleaned = cleaned.strip()
        details.cleaned_output = cleaned
        logger.info("[LLM] 清理后最终输出: [{}]", cleaned[:200])

        # 保存调用详情供 dump 使用
        self._last_call_details = details

        # 解析动作编号
        return self._parse_action_index(cleaned, legal)

    @staticmethod
    def _coerce_dict(value: Any) -> dict[str, Any]:
        """将 pydantic 对象或 dict 统一转为 dict。"""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump(mode="json")
            return dumped if isinstance(dumped, dict) else {}
        if hasattr(value, "to_dict"):
            dumped = value.to_dict()
            return dumped if isinstance(dumped, dict) else {}
        return getattr(value, "__dict__", {})

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """从各种格式中提取纯文本（content 字段可能是 str / list / dict）。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    item_type = str(item.get("type") or "")
                    if item_type and item_type not in {"text", "output_text"}:
                        continue
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        if isinstance(value, dict):
            return str(value.get("text") or value.get("content") or "")
        return str(value)

    @classmethod
    def _extract_final_text(cls, response: Any) -> str:
        """从 OpenAI 响应中提取最终输出文本。

        关键：思考内容(reasoning/reasoning_content)和最终输出(content)必须分离，
        只返回 content 部分。思考内容里的数字不应被用于决策。
        参考 train/gakumas_rl/gakumas_rl/llm_player.py 的实现。
        """
        if not response or not response.choices:
            return ""
        choice = response.choices[0] if isinstance(response.choices, list) else response.choices
        message_raw = getattr(choice, "message", choice)
        message = cls._coerce_dict(message_raw)

        # 最终输出：只从 content 字段取
        raw_text = cls._coerce_text(message.get("content"))
        if raw_text.strip():
            return raw_text.strip()

        # content 为空时不应回退到 reasoning — 那是思考过程不是最终回答
        # 记录 reasoning 长度以便调试
        reasoning = cls._coerce_text(
            message.get("reasoning_content") or message.get("reasoning")
        )
        if reasoning.strip():
            logger.warning(
                "[LLM] content 为空但有 reasoning ({} 字符) — 模型可能只思考未输出最终回答",
                len(reasoning),
            )
        return ""

    @staticmethod
    def _parse_action_index(text: str, legal_actions: list[int]) -> int | None:
        """解析操作、index并返回结果。

        Args:
            text: 待处理文本，通常来源于 OCR 或配置。
            legal_actions: 用于提供legal、actions相关输入。

        Returns:
            int | None: 返回值类型见注解。
        """
        if not text:
            return None

        # 尝试第一行精确匹配
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            exact = re.fullmatch(r"\D*(\d+)\D*", lines[0])
            if exact:
                idx = int(exact.group(1))
                if not legal_actions or idx in legal_actions:
                    return idx

        # 回退：找到文本中所有数字，从前往后匹配（优先取第一个合法数字）
        numbers = re.findall(r"\d+", text)
        for num_str in numbers:
            idx = int(num_str)
            if not legal_actions or idx in legal_actions:
                return idx

        return None

    # ── 工具方法 ────────────────────────────────────────

    @property
    def stats(self) -> str:
        """汇总当前策略运行统计信息。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
        avg = (self._total_latency / self._call_count) if self._call_count else 0
        return f"calls={self._call_count}, avg_latency={avg:.1f}s"


def create_llm_strategy(
    base_url: str = "http://192.168.100.10:11434/v1/",
    model: str = "gpt-oss:20b",
    **kwargs: Any,
) -> LLMStrategy:
    """创建 LLM 策略实例的便捷工厂函数。"""
    return LLMStrategy(base_url=base_url, model=model, **kwargs)


def inject_llm_strategy(
    ctx: "ProduceContext",
    strategy: LLMStrategy | None = None,
    *,
    base_url: str = "http://192.168.100.10:11434/v1/",
    model: str = "gpt-oss:20b",
    **kwargs: Any,
) -> LLMStrategy:
    """创建 LLM 策略并注入到 ProduceContext 的所有决策字段。

    同时初始化 DecisionDumper session，开始记录决策。

    Returns:
        注入的 LLMStrategy 实例。
    """
    if strategy is None:
        strategy = create_llm_strategy(base_url=base_url, model=model, **kwargs)

    ctx.schedule_strategy = strategy
    ctx.lesson_strategy = strategy
    ctx.exam_strategy = strategy
    ctx.dialogue_strategy = strategy
    ctx.skill_reward_strategy = strategy
    ctx.p_drink_strategy = strategy
    ctx.consult_strategy = strategy
    ctx.modal_strategy = strategy

    # 初始化决策 dump session
    dumper = DecisionDumper.get_instance()
    dumper.start_session()

    logger.debug(
        f"[LLM] 策略已注入所有决策字段 | model={strategy.model} "
        f"base_url={strategy.base_url}"
    )
    return strategy
