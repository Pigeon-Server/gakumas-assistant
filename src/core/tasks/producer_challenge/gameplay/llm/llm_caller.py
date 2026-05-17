"""LLM 调用工具 — 响应解析 + 后端兼容 + 动作编号提取。

职责:
  - 从 OpenAI 兼容响应中提取最终文本
  - 兼容各种响应格式（pydantic / dict / list）
  - 解析动作编号
  - 后端兼容参数处理（Ollama think / reasoning_effort）
"""

from __future__ import annotations

import re
from typing import Any

from src.utils.logger import logger


# ── 响应文本提取 ─────────────────────────────────────


def coerce_dict(value: Any) -> dict[str, Any]:
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


def coerce_text(value: Any) -> str:
    """从各种格式中提取纯文本。"""
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


def extract_final_text(response: Any) -> str:
    """从 OpenAI 响应中提取最终输出文本。

    只取 content，不回退到 reasoning（思考过程不是最终回答）。
    """
    if not response or not response.choices:
        return ""
    choice = response.choices[0] if isinstance(response.choices, list) else response.choices
    message = coerce_dict(choice.message)

    raw_text = coerce_text(message.get("content"))
    if raw_text.strip():
        return raw_text.strip()

    reasoning = coerce_text(
        message.get("reasoning_content") or message.get("reasoning")
    )
    if reasoning.strip():
        logger.warning(
            "[LLM] content 为空但有 reasoning ({} 字符) — 模型可能只思考未输出最终回答",
            len(reasoning),
        )
    return ""


def extract_raw_fields(response: Any) -> tuple[str, str]:
    """从响应中提取 (raw_content, raw_reasoning)。"""
    if not response or not response.choices:
        return "", ""
    choice = response.choices[0] if isinstance(response.choices, list) else response.choices
    message = coerce_dict(choice.message)
    return coerce_text(message.get("content")), coerce_text(
        message.get("reasoning_content") or message.get("reasoning")
    )


# ── 动作编号解析 ─────────────────────────────────────


def parse_action_index(text: str, legal_actions: list[int]) -> int | None:
    """从 LLM 输出中解析动作编号。JSON 优先，正则兜底。"""
    if not text:
        return None

    # 优先从 JSON 解析
    parsed = _try_parse_decision_json(text)
    if parsed is not None:
        idx = parsed.get("action_index")
        if isinstance(idx, int) and (not legal_actions or idx in legal_actions):
            return idx

    # 回退：正则提取最后一个数字
    numbers = re.findall(r"\d+", text)
    for num_str in reversed(numbers):
        idx = int(num_str)
        if not legal_actions or idx in legal_actions:
            return idx
    return None


def parse_decision_reasoning(text: str) -> dict[str, str]:
    """从 LLM 输出中提取决策理由。JSON 优先，正则兜底。

    返回 {"why_this": "...", "why_not_others": "..."}
    """
    if not text:
        return {"why_this": "", "why_not_others": ""}

    # 优先从 JSON 解析
    parsed = _try_parse_decision_json(text)
    if parsed is not None:
        return {
            "why_this": str(parsed.get("why_this") or ""),
            "why_not_others": str(parsed.get("why_not_others") or ""),
        }

    # 回退：从自由文本提取
    why_this = ""
    why_not = ""
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^(?:理由|why_this)[：:]\s*(.+)", stripped)
        if m:
            why_this = m.group(1)
        m = re.match(r"^(?:why_not_others)[：:]\s*(.+)", stripped)
        if m:
            why_not = m.group(1)
    if not why_this:
        # 最后兜底：取去掉编号行的剩余文本
        lines = [l.strip() for l in text.splitlines() if l.strip() and not re.match(r"^\d+$", l.strip())]
        why_this = " ".join(lines[:3])
    return {"why_this": why_this, "why_not_others": why_not}


def _try_parse_decision_json(text: str) -> dict[str, object] | None:
    """尝试从文本中提取决策 JSON。支持 ```json 块 和裸 JSON。"""
    import json as _json

    # 尝试 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(1))
        except _json.JSONDecodeError:
            pass

    # 尝试裸 JSON
    m = re.search(r"\{[^{}]*\"action_index\"\s*:\s*\d+[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return _json.loads(m.group())
        except _json.JSONDecodeError:
            pass

    return None


# ── 后端兼容 ─────────────────────────────────────────


def normalize_legacy_think(value: Any) -> str:
    """兼容旧 think 配置并映射到 reasoning_effort。"""
    text = str(value or "").strip().lower()
    if text in {"false", "off", "none", "0"}:
        return "low"
    if text in {"true", "on", "1", "high"}:
        return "high"
    if text in {"low", "medium", "high", "xhigh"}:
        return text
    return "medium"


def normalize_reasoning_effort(value: Any) -> str:
    """标准化 OpenAI 风格的思考强度配置。"""
    effort = str(value or "medium").strip().lower()
    if effort not in {"low", "medium", "high", "xhigh"}:
        return "medium"
    return effort


def apply_backend_compat_options(
    kwargs: dict[str, Any],
    *,
    reasoning_effort: str = "medium",
    num_ctx: int = 0,
    for_retry: bool = False,
) -> None:
    """优先使用 OpenAI 风格参数，必要时再附加兼容后端参数。"""
    kwargs["reasoning_effort"] = reasoning_effort
    extra_body = dict(kwargs.get("extra_body") or {})
    if num_ctx:
        extra_body["options"] = {**dict(extra_body.get("options") or {}), "num_ctx": int(num_ctx)}
    if not for_retry:
        extra_body["think"] = reasoning_effort
    else:
        extra_body.pop("think", None)
    if extra_body:
        kwargs["extra_body"] = extra_body


def get_candidate_name(state: dict[str, Any], index: int) -> str:
    """根据索引读取候选名称。"""
    for c in state.get("candidates", []):
        if c.get("index") == index:
            return c.get("name", c.get("label", f"#{index}"))
    return f"#{index}"
