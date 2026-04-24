from __future__ import annotations

from typing import Any

from src.constants.game.text.produce_text import ProduceText


_CRITICAL_STAMINA_RATIO = 0.18
_LOW_STAMINA_RATIO = 0.32


def _schedule_payload_family(payload: dict[str, Any]) -> str:
    """处理日程、载荷、family并返回结果。

    Args:
        payload: 用于提供载荷相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    metadata = dict(payload.get("metadata", {}) or {})
    family = str(metadata.get("schedule_family") or "").strip()
    if family:
        return family
    action_id = str(payload.get("id") or payload.get("action_id") or "")
    title = str(payload.get("label") or payload.get("title") or "")
    if "refresh" in action_id or ProduceText.REST in title:
        return "refresh"
    if (
        "outing" in action_id
        or ProduceText.OUTING in title
        or ProduceText.GO_OUT in title
    ):
        return "outing"
    return ""


def _select_low_stamina_recovery_action(
    decision_state: dict[str, Any],
) -> tuple[int, str] | None:
    """选择low、stamina、recovery、操作并返回结果。

    Args:
        decision_state: 决策快照，包含上下文、候选项与当前理由。

    Returns:
        tuple[int, str] | None: 返回值类型见注解，语义由函数用途决定。
    """
    economy = dict(decision_state.get("economy", {}) or {})
    stamina = int(economy.get("stamina") or 0)
    max_stamina = int(economy.get("max_stamina") or 0)
    p_point = int(economy.get("p_point") or 0)
    if max_stamina <= 0:
        return None

    stamina_ratio = float(stamina) / max(max_stamina, 1)
    critical = stamina <= 2 or stamina_ratio <= _CRITICAL_STAMINA_RATIO
    low = stamina <= 4 or stamina_ratio <= _LOW_STAMINA_RATIO
    if not (critical or low):
        return None

    legal_actions = {int(index) for index in decision_state.get("legal_actions", [])}
    refresh_payloads: list[dict[str, Any]] = []
    outing_payloads: list[dict[str, Any]] = []
    for payload in decision_state.get("candidates", []):
        index = int(payload.get("index", -1))
        if index not in legal_actions:
            continue
        family = _schedule_payload_family(payload)
        if family == "refresh":
            refresh_payloads.append(payload)
        elif family == "outing":
            outing_payloads.append(payload)

    if critical:
        if refresh_payloads:
            return (
                int(refresh_payloads[0]["index"]),
                f"当前体力过低，按手册应优先{ProduceText.REST_ACTION}回体，避免下一周直接暴毙。",
            )
        if outing_payloads and p_point > 0:
            return (
                int(outing_payloads[0]["index"]),
                f"当前体力过低，且 {ProduceText.P_POINT} 足够，优先{ProduceText.OUTING}回体更稳。",
            )
    if low:
        if outing_payloads and p_point > 0:
            return (
                int(outing_payloads[0]["index"]),
                f"当前体力偏低，且有 {ProduceText.P_POINT}，可优先{ProduceText.OUTING}回体并顺带争取额外收益。",
            )
        if refresh_payloads:
            return (
                int(refresh_payloads[0]["index"]),
                f"当前体力偏低，先{ProduceText.REST_ACTION}补体力比继续高消耗行动更稳。",
            )
    return None


def _annotate_low_stamina_recovery_preference(
    decision_state: dict[str, Any],
    *,
    preferred_index: int,
    reason: str,
) -> None:
    """补充标注low、stamina、recovery、preference并返回结果。

    Args:
        decision_state: 决策快照，包含上下文、候选项与当前理由。
        preferred_index: 用于提供preferred、index相关输入。
        reason: 用于提供reason相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    payloads = list(decision_state.get("candidates", []) or [])
    label = f"候选 {preferred_index}"
    for payload in payloads:
        if int(payload.get("index", -1)) == preferred_index:
            payload["recommended"] = True
            label = str(payload.get("label") or payload.get("title") or label)
            break

    for payload in decision_state.get("llm_actions", []) or []:
        if int(payload.get("index", -1)) == preferred_index:
            payload["recommended"] = True

    stage_context = dict(decision_state.get("stage_context", {}) or {})
    stage_context["system_recommendation"] = f"系统当前推荐优先考虑：{label}。{reason}"
    decision_state["stage_context"] = stage_context
    llm_snapshot = decision_state.get("llm_snapshot")
    if isinstance(llm_snapshot, dict):
        llm_snapshot["stage_context"] = stage_context
