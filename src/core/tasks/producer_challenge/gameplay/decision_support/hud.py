from __future__ import annotations

import re
from typing import Any

from src.constants.game.text.produce_text import ProduceText
from src.utils.string_tools import fullwidth_to_halfwidth

_NUMBER_RE = re.compile(r"\d+")
_STAMINA_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _extract_first_int(text: str) -> int:
    """提取first、int并返回结果。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        int: 计算得到的数值结果。
    """
    match = _NUMBER_RE.search(text or "")
    return int(match.group()) if match else 0


def _match_any_variant(text_upper: str, variants: tuple[str, ...]) -> bool:
    """检查 text_upper 是否包含 variants 中的任意一个（大小写不敏感）。"""
    return any(value.upper() in text_upper for value in variants)


def _parse_progress_circle(score_text: str) -> dict | None:
    """尝试将 PC_TRAINING_SCORE 的 OCR 文本解析为进度圆圈信息。"""
    if not score_text:
        return None
    normalized = (score_text or "").replace(" ", "").replace("　", "").upper()
    has_made = _match_any_variant(normalized, ProduceText.PROGRESS_MADE_OCR_VARIANTS)
    has_perfect = _match_any_variant(normalized, ProduceText.PROGRESS_PERFECT_OCR_VARIANTS)
    has_clear = _match_any_variant(normalized, ProduceText.PROGRESS_CLEAR_OCR_VARIANTS)
    if not has_made and not has_clear and not has_perfect:
        return None

    made_end = 0
    for variant in ProduceText.PROGRESS_MADE_OCR_VARIANTS:
        idx = normalized.find(variant.upper())
        if idx >= 0:
            made_end = max(made_end, idx + len(variant))
    number = (
        _extract_first_int(normalized[made_end:])
        if made_end > 0
        else _extract_first_int(score_text)
    )
    if has_perfect:
        return {
            "clear_achieved": True,
            "remaining_to_clear": 0,
            "remaining_to_perfect": number,
        }
    return {
        "clear_achieved": False,
        "remaining_to_clear": number,
        "remaining_to_perfect": 0,
    }


def _build_noisy_stamina_candidates(digits: str) -> list[int]:
    """从可能粘连/夹噪的数字串里枚举候选当前体力。"""
    if not digits:
        return []
    candidates: list[int] = [int(digits)]
    if len(digits) >= 2:
        candidates.extend(int(digits[index:]) for index in range(1, len(digits)))
        candidates.extend(int(digits[:index]) for index in range(1, len(digits)))
        candidates.extend(
            int(digits[start:end])
            for start in range(len(digits))
            for end in range(start + 1, len(digits) + 1)
            if end - start <= 2
        )
        candidates.extend(
            int(digits[:index] + digits[index + 1 :])
            for index in range(len(digits))
            if digits[:index] + digits[index + 1 :]
        )
    deduped: list[int] = []
    seen: set[int] = set()
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _parse_stamina_text(
    text: str,
    *,
    previous_stamina: int = 0,
    previous_max_stamina: int = 0,
) -> tuple[int, int]:
    """解析stamina、text并返回结果。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。
        previous_stamina: 用于提供previous、stamina相关输入。
        previous_max_stamina: 用于提供previous、max、stamina相关输入。

    Returns:
        tuple[int, int]: 返回值类型见注解。
    """
    normalized = fullwidth_to_halfwidth(str(text or ""))
    match = _STAMINA_RE.search(normalized)
    if match:
        return int(match.group(1)), int(match.group(2))

    digit_groups = re.findall(r"\d+", normalized)
    if not digit_groups:
        return 0, 0
    digits = "".join(digit_groups)
    has_slash = "/" in normalized

    if previous_max_stamina > 0:
        max_text = str(previous_max_stamina)
        if digits == max_text and 0 < previous_stamina < previous_max_stamina:
            return previous_stamina, previous_max_stamina
        if digits.endswith(max_text) and len(digits) > len(max_text):
            current_text = digits[:-len(max_text)]
            candidate_values = _build_noisy_stamina_candidates(current_text)
            valid_candidates = [
                value
                for value in candidate_values
                if 0 <= value <= previous_max_stamina
            ]
            if valid_candidates:
                current_value = min(
                    valid_candidates,
                    key=lambda value: (abs(value - previous_stamina), -value),
                )
                return current_value, previous_max_stamina
        inferred_current = int(digits)
        if 0 <= inferred_current <= previous_max_stamina:
            return inferred_current, previous_max_stamina
        if not has_slash:
            return previous_stamina, previous_max_stamina

    if len(digits) >= 3:
        current_value = int(digits[:-2])
        max_value = int(digits[-2:])
        if 0 < max_value <= 99 and 0 <= current_value <= max_value:
            return current_value, max_value
    if len(digits) >= 2:
        current_value = int(digits[:-1])
        max_value = int(digits[-1:])
        if 0 < max_value <= 9 and 0 <= current_value <= max_value:
            return current_value, max_value
    return int(digits), 0


def _build_noisy_hud_value_candidates(digits: str) -> list[tuple[int, int]]:
    """从单值 HUD 的 OCR 文本里枚举去噪候选，priority 越小越可信。"""
    if not digits:
        return []
    candidates: list[tuple[int, int]] = []
    seen: set[int] = set()

    def _add(value: int, priority: int) -> None:
        """处理add并返回结果。

        Args:
            value: 用于提供value相关输入。
            priority: 用于提供priority相关输入。

        Returns:
            None: 仅产生副作用，不返回业务值。
        """
        if value in seen:
            return
        seen.add(value)
        candidates.append((value, priority))

    _add(int(digits), 0)
    if len(digits) >= 2 and len(digits) % 2 == 0:
        half = len(digits) // 2
        if digits[:half] == digits[half:]:
            _add(int(digits[:half]), 1)
    for index in range(1, len(digits)):
        _add(int(digits[index:]), 2)
    for index in range(len(digits) - 1, 0, -1):
        _add(int(digits[:index]), 3)
    max_window = min(len(digits) - 1, 3)
    for window in range(max_window, 0, -1):
        for start in range(0, len(digits) - window + 1):
            _add(int(digits[start : start + window]), 4)
    return candidates


def _extract_noisy_hud_value(
    *texts: str,
    previous_value: int = 0,
    upper_bound: int = 0,
) -> tuple[int, bool]:
    """综合多份裁切 OCR，提取 battle HUD 中的单个数值。"""
    candidate_items: list[tuple[int, int, int, int]] = []
    has_digits = False
    for source_index, text in enumerate(texts):
        normalized = fullwidth_to_halfwidth(str(text or ""))
        digit_groups = re.findall(r"\d+", normalized)
        if not digit_groups:
            continue
        has_digits = True
        digits = "".join(digit_groups)
        for value, priority in _build_noisy_hud_value_candidates(digits):
            if upper_bound > 0 and value > upper_bound:
                continue
            candidate_items.append((value, source_index, priority, len(str(value))))
    if not candidate_items:
        return 0, has_digits
    if previous_value > 0:
        candidate_items.sort(
            key=lambda item: (
                abs(item[0] - previous_value),
                item[1],
                item[2],
                -item[3],
                item[0],
            )
        )
    else:
        candidate_items.sort(
            key=lambda item: (
                item[1],
                item[2],
                -item[3],
                item[0],
            )
        )
    return candidate_items[0][0], True


def _resolve_repeated_digit_ocr_value(
    value: int,
    *texts: str,
    previous_value: int = 0,
) -> int:
    """保守修正重复数字抖动，避免把真实高值（如 88）误折叠成 8。"""
    if value <= 0:
        return value

    normalized_digits: list[str] = []
    for text in texts:
        digits = "".join(re.findall(r"\d+", fullwidth_to_halfwidth(str(text or ""))))
        if digits:
            normalized_digits.append(digits)
    if not normalized_digits:
        return value

    for digits in normalized_digits:
        if len(digits) < 2 or len(digits) % 2 != 0:
            continue
        half = len(digits) // 2
        if digits[:half] != digits[half:]:
            continue
        full_value = int(digits)
        half_value = int(digits[:half])
        if full_value != value:
            continue

        full_support = sum(1 for item in normalized_digits if item == digits)
        half_support = sum(1 for item in normalized_digits if item == digits[:half])

        # 首帧只在“半值有直接证据且全值证据弱”时折叠，防止真实 88 被改成 8。
        if previous_value <= 0:
            if half_support > 0 and full_support <= 1:
                return half_value
            continue

        # 非首帧：只有历史值明显更接近半值、且半值有直接证据时才折叠。
        if half_support <= 0:
            continue
        if abs(previous_value - half_value) + 1 < abs(previous_value - full_value):
            return half_value

    return value


def _get_parameter_seed_value(ctx: Any, key: str) -> int:
    """优先使用已同步参数，其次回退到偶像卡主库基础值。"""
    if ctx is None:
        return 0
    current_value = ctx.parameter_state.get(key)
    if isinstance(current_value, int) and current_value > 0:
        return current_value
    selected_idol_card = getattr(ctx, "selected_idol_card", None)
    if selected_idol_card is None:
        return 0
    field_name = {
        "vocal": "produceVocal",
        "dance": "produceDance",
        "visual": "produceVisual",
    }.get(key, "")
    if not field_name:
        return 0
    return int(getattr(selected_idol_card, field_name, 0) or 0)


def _extract_planning_parameter_value(
    *texts: str,
    previous_value: int = 0,
    upper_bound: int = 0,
) -> tuple[int | None, bool]:
    """提取周规划 HUD 参数值，并利用数据库上限抑制粘连脏 OCR。"""
    filtered_texts: list[str] = []
    for text in texts:
        normalized = fullwidth_to_halfwidth(str(text or ""))
        if not normalized.strip():
            continue
        # 百分比片段（如 31.4%）若不包含上限锚点，通常不是参数本体值，避免污染。
        if upper_bound > 0 and ("%" in normalized or "％" in normalized):
            digit_groups = re.findall(r"\d+", normalized)
            has_upper_anchor = any(int(group) == upper_bound for group in digit_groups if group)
            if "/" not in normalized and not has_upper_anchor:
                continue
        filtered_texts.append(normalized)

    if not filtered_texts:
        return None, False

    if upper_bound > 0:
        for normalized in filtered_texts:
            slash_match = re.search(r"(\d+)\s*/\s*(\d+)", normalized)
            if slash_match:
                current_value = int(slash_match.group(1))
                max_value = int(slash_match.group(2))
                if max_value == upper_bound and 0 < current_value <= upper_bound:
                    return current_value, True
            digit_groups = re.findall(r"\d+", normalized)
            if not digit_groups:
                continue
            for index, group in enumerate(digit_groups):
                if int(group) != upper_bound or index <= 0:
                    continue
                prev_group = digit_groups[index - 1]
                if not prev_group:
                    continue
                candidate = int(prev_group)
                if 0 < candidate <= upper_bound:
                    return candidate, True
            first_group = digit_groups[0]
            # 三位及以上数值优先按“主参数值”处理，避免被历史值吸附到 70/31 这类截断结果。
            if len(first_group) >= 3:
                candidate = int(first_group)
                if 0 < candidate <= upper_bound:
                    return candidate, True

    max_digits = len(str(upper_bound)) if upper_bound > 0 else 0
    if previous_value <= 0 and max_digits > 0:
        for normalized in filtered_texts:
            digit_groups = re.findall(r"\d+", normalized)
            if not digit_groups:
                continue
            digits = "".join(digit_groups)
            if len(digits) <= max_digits:
                continue
            for prefix_len in range(max_digits, 0, -1):
                candidate = int(digits[:prefix_len])
                if 0 < candidate <= upper_bound:
                    return candidate, True

    value, has_digits = _extract_noisy_hud_value(
        *filtered_texts,
        previous_value=previous_value,
        upper_bound=upper_bound,
    )
    if not has_digits:
        return None, False
    return (value if value > 0 else None), True


def _extract_first_int_from_texts(*texts: str) -> int:
    """提取first、int、from、texts并返回结果。

    Args:
        *texts: 用于提供texts相关输入。

    Returns:
        int: 计算得到的数值结果。
    """
    for text in texts:
        value = _extract_first_int(text)
        if value > 0:
            return value
    return 0


def _build_parameter_stats_payload(ctx: Any) -> dict[str, Any]:
    """构建parameter、stats、结构化载荷并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        dict: 结构化结果字典。
    """
    parameter_limit = int(getattr(ctx, "parameter_growth_limit", 0) or 0)
    return {
        "vocal": ctx.parameter_state.get("vocal", "") or "",
        "dance": ctx.parameter_state.get("dance", "") or "",
        "visual": ctx.parameter_state.get("visual", "") or "",
        "vocal_max": parameter_limit or "",
        "dance_max": parameter_limit or "",
        "visual_max": parameter_limit or "",
    }


__all__ = [
    "_build_noisy_hud_value_candidates",
    "_build_noisy_stamina_candidates",
    "_build_parameter_stats_payload",
    "_extract_first_int",
    "_extract_first_int_from_texts",
    "_extract_noisy_hud_value",
    "_extract_planning_parameter_value",
    "_get_parameter_seed_value",
    "_match_any_variant",
    "_parse_progress_circle",
    "_parse_stamina_text",
    "_resolve_repeated_digit_ocr_value",
]
