from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Sequence

from src.constants.game.text.produce_text import ProduceText
from src.core.tasks.producer_challenge.shared.common import (
    infer_param_kind,
    normalize_lookup_text,
)
from src.utils.logger import logger


@dataclass(frozen=True)
class CandidateResolution:
    """定义 CandidateResolution 的结构化数据。

    Attributes:
        action_id: 标准化动作标识，用于在决策层与执行层之间关联同一操作。
        candidate_type: 候选项类别（如 schedule_action、dialogue_option），用于后续分支处理。
        db_id: 数据库中的实体 ID；为空通常表示当前候选项尚未完成实体识别。
        display_name: 展示给日志/策略的可读名称。
        source: 候选项来源标记（如 OCR、DB、fallback），便于排查识别链路。
        confidence: 当前识别或匹配结果的置信度，数值越高代表结果越可靠。
        metadata: 扩展元数据，保存额外识别信息与决策辅助字段。
    """
    action_id: str
    candidate_type: str
    db_id: str = ""
    display_name: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduleActionSpec:
    """定义 ScheduleActionSpec 的结构化数据。

    Attributes:
        action_id: 标准化动作标识，用于在决策层与执行层之间关联同一操作。
        aliases: 可匹配的别名集合，用于提高 OCR 文本匹配容错率。
        rl_action_type: 对接 RL 动作空间时使用的动作类别标识。
        todo: 尚未完成能力的说明或后续补全备注。
        confidence: 当前识别或匹配结果的置信度，数值越高代表结果越可靠。
    """
    action_id: str
    aliases: tuple[str, ...]
    rl_action_type: str = ""
    todo: str = ""
    confidence: float = 0.95


_SLUG_CLEANUP_RE = re.compile(r"[^a-z0-9_]+")

_SCHEDULE_ACTION_SPECS: tuple[ScheduleActionSpec, ...] = (
    ScheduleActionSpec(
        action_id="schedule_action_special_guidance",
        aliases=(ProduceText.SPECIAL_GUIDANCE,),
        todo="TODO: 缺少特別指導真实采集图与稳定界面判据，当前仅补 action_id，未实现专用 handler。",
        confidence=0.75,
    ),
    ScheduleActionSpec(
        action_id="schedule_action_customize",
        aliases=(ProduceText.CUSTOMIZE,),
        todo="TODO: 缺少カスタマイズ真实采集图与稳定界面判据，当前仅补 action_id，未实现专用 handler。",
        confidence=0.75,
    ),
    ScheduleActionSpec(
        action_id="schedule_action_audition_finale",
        aliases=(ProduceText.FINALE,),
    ),
    ScheduleActionSpec(
        action_id="schedule_action_audition_second",
        aliases=(ProduceText.SECOND_AUDITION,),
    ),
    ScheduleActionSpec(
        action_id="schedule_action_audition_first",
        aliases=(ProduceText.FIRST_AUDITION,),
    ),
    ScheduleActionSpec(
        action_id="schedule_action_audition",
        aliases=(ProduceText.AUDITION,),
    ),
    ScheduleActionSpec(
        action_id="schedule_action_consult",
        aliases=(ProduceText.CONSULT,),
    ),
    ScheduleActionSpec(
        action_id="schedule_action_present_support",
        aliases=(ProduceText.PRESENT_SUPPORT,),
        rl_action_type="present",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_fan_present",
        aliases=(ProduceText.FAN_PRESENT,),
        rl_action_type="present",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_business_corporate",
        aliases=(ProduceText.BUSINESS_CORPORATE,),
        rl_action_type="business",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_business_municipal",
        aliases=(ProduceText.BUSINESS_MUNICIPAL,),
        rl_action_type="business",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_business_resort",
        aliases=(ProduceText.BUSINESS_RESORT,),
        rl_action_type="business",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_business_commercial",
        aliases=(ProduceText.BUSINESS_COMMERCIAL,),
        rl_action_type="business",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_business",
        aliases=(ProduceText.BUSINESS,),
        rl_action_type="business",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_outing",
        aliases=(ProduceText.OUTING, ProduceText.GO_OUT),
        rl_action_type="activity",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_class",
        aliases=(ProduceText.CLASS,),
        rl_action_type="activity",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_activity",
        aliases=(ProduceText.ACTIVITY,),
        rl_action_type="activity",
    ),
    ScheduleActionSpec(
        action_id="schedule_action_refresh",
        aliases=(ProduceText.REST_ACTION, ProduceText.REST, "refresh"),
        rl_action_type="refresh",
    ),
)

_outing_activity_entries: list[dict[str, Any]] | None = None
_dialogue_option_effect_entries: list[dict[str, Any]] | None = None
_OUTING_WHITESPACE_RE = re.compile(r"[\s\n\r　]+")
_OUTING_GENERIC_ID_RE = re.compile(r"^p_s_e_s-event-detail-activity-\d+-\d+-\d+$")
_OUTING_MATCH_THRESHOLD = 0.5
_DIALOGUE_MATCH_THRESHOLD = 0.52

_LESSON_OPTION_MAP: dict[str, dict[str, str]] = {
    "vocal": {
        "action_id": "lesson_option_vocal_normal",
        "rl_action_type": "lesson_vocal_normal",
        "display_name": ProduceText.VOCAL_LESSON,
    },
    "dance": {
        "action_id": "lesson_option_dance_normal",
        "rl_action_type": "lesson_dance_normal",
        "display_name": ProduceText.DANCE_LESSON,
    },
    "visual": {
        "action_id": "lesson_option_visual_normal",
        "rl_action_type": "lesson_visual_normal",
        "display_name": ProduceText.VISUAL_LESSON,
    },
}


def _clean_description_text(text: str) -> str:
    """清洗描述、text并返回结果。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        str: 处理后的文本结果。
    """
    cleaned = (
        str(text or "")
        .replace("<nobr>", "")
        .replace("</nobr>", "")
        .replace("<br>", "；")
        .replace("<br/>", "；")
        .replace("<br />", "；")
        .replace("\t", " ")
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*([,，。；：、）])", r"\1", cleaned)
    cleaned = re.sub(r"([（(])\s+", r"\1", cleaned)
    return cleaned.strip()


def _description_text(entries: Any) -> str:
    """处理描述、文本并返回结果。

    Args:
        entries: 用于提供entries相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    if not entries:
        return ""
    parts: list[str] = []
    for raw_entry in entries:
        entry = raw_entry or {}
        if isinstance(entry, dict):
            text = _clean_description_text(str(entry.get("text") or ""))
        else:
            text = _clean_description_text(str(getattr(entry, "text", "") or ""))
        if text:
            parts.append(text)
    # 多段效果文本用分号连接，避免 OCR/DB 拼接成 "+11060%" 这类歧义串。
    return _clean_description_text("；".join(parts))


def _humanize_runtime_text(text: str) -> str:
    """处理humanize、runtime、文本并返回结果。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        str: 处理后的文本结果。
    """
    cleaned = _clean_description_text(text)
    replacements = (
        ("干劲", ProduceText.YARUKI),
        ("好调", ProduceText.GOOD_CONDITION),
        ("绝好调", ProduceText.EXCELLENT_CONDITION),
        ("元气", ProduceText.GENKI),
        ("强气", ProduceText.STRONG_SPIRIT),
        ("弱气", ProduceText.WEAK_SPIRIT),
        ("热意", ProduceText.ENTHUSIASM),
        ("全力值", ProduceText.FULL_POWER_POINT),
        ("技能卡使用数追加", ProduceText.SKILL_CARD_USE_COUNT_UP),
        ("体力回复", ProduceText.STAMINA_RECOVERY),
    )
    for before, after in replacements:
        cleaned = cleaned.replace(before, after)
    cleaned = re.sub(r"[；]{2,}", "；", cleaned)
    return cleaned.strip("； ")


def _normalize_effect_lookup_text(text: str) -> str:
    # 统一描述文本归一化，降低 OCR 噪声（空格、全半角、标点）对匹配的影响。
    """规范化`effect_lookup_text`。"""
    return normalize_lookup_text(_humanize_runtime_text(_clean_description_text(text)))


def _slugify_text(text: str | None, *, fallback: str) -> str:
    """处理slugify、文本并返回结果。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。
        fallback: 用于提供fallback相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    normalized = normalize_lookup_text(text)
    slug = _SLUG_CLEANUP_RE.sub("_", normalized.lower()).strip("_")
    return slug or fallback


def _build_unknown_action_id(prefix: str, text: str | None, *, index: int) -> str:
    """构建unknown、操作、id并返回结果。

    Args:
        prefix: 用于提供prefix相关输入。
        text: 待处理文本，通常来源于 OCR 或配置。
        index: 用于提供index相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    return f"{prefix}:{_slugify_text(text, fallback=f'idx_{index}')}"


def _matches_schedule_alias(raw_title: str, normalized_title: str, alias: str) -> bool:
    """判断日程、alias是否成立。

    Args:
        raw_title: 用于提供raw、title相关输入。
        normalized_title: 用于提供normalized、title相关输入。
        alias: 用于提供alias相关输入。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    if not alias:
        return False
    return alias in raw_title or normalize_lookup_text(alias) in normalized_title


def _schedule_lesson_display_name(param_kind: str, variant: str) -> str:
    """处理日程`schedule_lesson_display_name`。"""
    lesson_name_map = {
        "vocal": ProduceText.VOCAL,
        "dance": ProduceText.DANCE,
        "visual": ProduceText.VISUAL,
    }
    lesson_name = lesson_name_map.get(param_kind, ProduceText.LESSON)
    if variant == "sp":
        return f"{lesson_name}SP{ProduceText.LESSON}"
    if variant == "hard":
        return f"{ProduceText.HARD_LESSON}{lesson_name}{ProduceText.LESSON}"
    return f"{lesson_name}{ProduceText.LESSON}"


def _build_schedule_description(
    *,
    action_id: str,
    rl_action_type: str,
    param_kind: str,
) -> str:
    """构建日程、描述并返回结果。

    Args:
        action_id: 业务对象标识符，用于索引或匹配目标实体。
        rl_action_type: 用于提供rl、操作、type相关输入。
        param_kind: 用于提供param、kind相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    action_desc = {
        "schedule_action_consult": "相談；技能卡交换/强化/删除等牌组调整。",
        "schedule_action_present_support": "活動支給；通常可从差し入れ中选择一次即时收益。",
        "schedule_action_fan_present": "差し入れ；通常可从候选慰问品中选择一次即时收益。",
        "schedule_action_business": "営業；获取粉丝或参数收益，通常会消耗体力。",
        "schedule_action_business_corporate": "企業イベント営業；偏向粉丝/参数收益，通常会消耗体力。",
        "schedule_action_business_municipal": "自治体イベント営業；偏向粉丝/参数收益，通常会消耗体力。",
        "schedule_action_business_resort": "リゾート施設営業；偏向粉丝/参数收益，通常会消耗体力。",
        "schedule_action_business_commercial": "商業施設営業；偏向粉丝/参数收益，通常会消耗体力。",
        "schedule_action_outing": "おでかけ；消耗Pポイント并获取恢复/强化等收益。",
        "schedule_action_class": "授業；主に体力を消費しパラメータを獲得します。",
        "schedule_action_activity": "活動；通常可获取粉丝或阶段收益，可能消耗体力。",
        "schedule_action_refresh": "休む；恢复体力。",
    }
    direct = action_desc.get(action_id, "")
    if direct:
        return direct
    if rl_action_type.startswith("lesson_"):
        axis_map = {
            "vocal": ProduceText.VOCAL,
            "dance": ProduceText.DANCE,
            "visual": ProduceText.VISUAL,
        }
        axis = axis_map.get(param_kind, "对应")
        if rl_action_type.endswith("_sp"):
            return f"SPレッスン；主に体力を消費し{axis}パラメータを大きく伸ばします。"
        if rl_action_type.endswith("_hard"):
            return f"追い込みレッスン；主に体力を消費し{axis}パラメータを重点提升。"
        return f"通常レッスン；主に体力を消費し{axis}パラメータを獲得します。"
    if rl_action_type.startswith("self_lesson_"):
        return "自主レッスン；主に体力を消費し对应参数を獲得します。"
    return ""


def _resolve_schedule_spec(
    spec: ScheduleActionSpec,
    *,
    raw_title: str,
    metadata: dict[str, Any],
) -> CandidateResolution:
    """解析并确定`schedule_spec`。"""
    spec_metadata = dict(metadata)
    spec_metadata["schedule_family"] = spec.action_id.removeprefix("schedule_action_")
    spec_metadata["supported"] = not bool(spec.todo)
    if spec.rl_action_type:
        spec_metadata["rl_action_type"] = spec.rl_action_type
    description = _build_schedule_description(
        action_id=spec.action_id,
        rl_action_type=spec.rl_action_type,
        param_kind=str(spec_metadata.get("param_kind") or "unknown"),
    )
    if description:
        spec_metadata["description"] = description
    if spec.todo:
        spec_metadata["todo"] = spec.todo
    # 周行动名称统一使用主别名，避免 OCR 噪声（如 "Bd授業"）直接透传到决策层。
    display = spec.aliases[0] if spec.aliases else raw_title
    return CandidateResolution(
        action_id=spec.action_id,
        candidate_type="schedule_action",
        display_name=display,
        source="todo" if spec.todo else "heuristic",
        confidence=spec.confidence,
        metadata=spec_metadata,
    )


def resolve_schedule_action_identity(
    title: str,
    kind: str,
    *,
    index: int = 0,
    is_sp: bool = False,
) -> CandidateResolution:
    """解析并补全日程、操作、标识并返回结果。

    Args:
        title: 用于提供title相关输入。
        kind: 用于提供kind相关输入。
        index: 用于提供index相关输入。
        is_sp: 用于提供is、sp相关输入。

    Returns:
        CandidateResolution: 返回值类型见注解。
    """
    raw_title = str(title or "")
    normalized_title = normalize_lookup_text(raw_title)
    metadata: dict[str, Any] = {
        "title": raw_title,
        "param_kind": kind or infer_param_kind(raw_title),
    }

    for spec in _SCHEDULE_ACTION_SPECS:
        if raw_title == spec.action_id:
            return _resolve_schedule_spec(spec, raw_title=raw_title, metadata=metadata)
        if any(_matches_schedule_alias(raw_title, normalized_title, alias) for alias in spec.aliases):
            return _resolve_schedule_spec(spec, raw_title=raw_title, metadata=metadata)

    param_kind = metadata["param_kind"]
    if ProduceText.SELF_LESSON in raw_title:
        variant = "sp" if ("SP" in raw_title.upper() or is_sp) else "normal"
        metadata["rl_action_type"] = (
            f"self_lesson_{param_kind}_{variant}" if param_kind != "unknown" else ""
        )
        metadata["description"] = _build_schedule_description(
            action_id="schedule_action_self_lesson",
            rl_action_type=str(metadata.get("rl_action_type") or ""),
            param_kind=param_kind,
        )
        return CandidateResolution(
            action_id=(
                f"schedule_action_self_lesson_{param_kind}_{variant}"
                if param_kind != "unknown"
                else _build_unknown_action_id("schedule_action_self_lesson_unknown", raw_title, index=index)
            ),
            candidate_type="schedule_action",
            display_name=_schedule_lesson_display_name(param_kind, variant),
            source="heuristic",
            confidence=0.95,
            metadata=metadata,
        )

    if ProduceText.HARD_LESSON in raw_title:
        metadata["rl_action_type"] = (
            f"lesson_{param_kind}_hard" if param_kind != "unknown" else ""
        )
        metadata["description"] = _build_schedule_description(
            action_id="schedule_action_lesson_hard",
            rl_action_type=str(metadata.get("rl_action_type") or ""),
            param_kind=param_kind,
        )
        return CandidateResolution(
            action_id=(
                f"schedule_action_lesson_{param_kind}_hard"
                if param_kind != "unknown"
                else _build_unknown_action_id("schedule_action_lesson_hard_unknown", raw_title, index=index)
            ),
            candidate_type="schedule_action",
            display_name=_schedule_lesson_display_name(param_kind, "hard"),
            source="heuristic",
            confidence=0.95,
            metadata=metadata,
        )

    if "SP" in raw_title.upper() or "ＳＰ" in raw_title or is_sp:
        metadata["rl_action_type"] = (
            f"lesson_{param_kind}_sp" if param_kind != "unknown" else ""
        )
        metadata["description"] = _build_schedule_description(
            action_id="schedule_action_lesson_sp",
            rl_action_type=str(metadata.get("rl_action_type") or ""),
            param_kind=param_kind,
        )
        return CandidateResolution(
            action_id=(
                f"schedule_action_lesson_{param_kind}_sp"
                if param_kind != "unknown"
                else _build_unknown_action_id("schedule_action_lesson_sp_unknown", raw_title, index=index)
            ),
            candidate_type="schedule_action",
            display_name=_schedule_lesson_display_name(param_kind, "sp"),
            source="heuristic",
            confidence=0.95,
            metadata=metadata,
        )

    if ProduceText.LESSON in raw_title or param_kind != "unknown":
        metadata["rl_action_type"] = (
            f"lesson_{param_kind}_normal" if param_kind != "unknown" else ""
        )
        metadata["description"] = _build_schedule_description(
            action_id="schedule_action_lesson_normal",
            rl_action_type=str(metadata.get("rl_action_type") or ""),
            param_kind=param_kind,
        )
        return CandidateResolution(
            action_id=(
                f"schedule_action_lesson_{param_kind}_normal"
                if param_kind != "unknown"
                else _build_unknown_action_id("schedule_action_lesson_unknown", raw_title, index=index)
            ),
            candidate_type="schedule_action",
            display_name=_schedule_lesson_display_name(param_kind, "normal"),
            source="heuristic",
            confidence=0.95,
            metadata=metadata,
        )

    return CandidateResolution(
        action_id=_build_unknown_action_id("schedule_action", raw_title, index=index),
        candidate_type="schedule_action",
        display_name=raw_title,
        source="heuristic",
        confidence=0.5,
        metadata=metadata,
    )


def _fallback_dialogue_option_identity(
    title: str,
    *,
    index: int,
    param_kind: str = "unknown",
) -> CandidateResolution:
    """解析`fallback_dialogue_option_identity`。"""
    metadata: dict[str, Any] = {}
    if param_kind and param_kind != "unknown":
        metadata["param_kind"] = param_kind
    return CandidateResolution(
        action_id=f"dialogue_option:{_slugify_text(title, fallback=f'idx_{index}')}",
        candidate_type="dialogue_option",
        display_name=title,
        source="ocr",
        confidence=0.75 if title else 0.0,
        metadata=metadata,
    )


def _load_dialogue_option_effect_entries() -> list[dict[str, Any]]:
    """懒加载周事件选项效果 DB 条目（单例）。"""
    global _dialogue_option_effect_entries
    if _dialogue_option_effect_entries is not None:
        return _dialogue_option_effect_entries

    from src.utils.game_database_tools import get_game_database

    data = list(get_game_database("ProduceStepEventSuggestion").get_all_item() or [])
    if not data:
        _dialogue_option_effect_entries = []
        return _dialogue_option_effect_entries

    seen: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in data:
        entry_id = str(getattr(entry, "id", "") or "")
        raw_desc = _humanize_runtime_text(_description_text(getattr(entry, "produceDescriptions", None)))
        norm_desc = _normalize_effect_lookup_text(raw_desc)
        if not entry_id or not norm_desc:
            continue
        produce_point = int(getattr(entry, "producePoint", 0) or 0)
        key = (produce_point, norm_desc)
        candidate = {
            "id": entry_id,
            "produce_point": produce_point,
            "norm_desc": norm_desc,
            "raw_desc": raw_desc,
            "effect_ids": list(getattr(entry, "produceEffectIds", []) or []),
            "stamina": int(getattr(entry, "stamina", 0) or 0),
        }
        existing = seen.get(key)
        # 优先保留效果轴信息更完整的条目；同分时取 id 字典序较小者，保证稳定性。
        if existing is None:
            seen[key] = candidate
            continue
        if len(candidate["effect_ids"]) > len(existing.get("effect_ids", [])):
            seen[key] = candidate
            continue
        if len(candidate["effect_ids"]) == len(existing.get("effect_ids", [])) and candidate["id"] < existing["id"]:
            seen[key] = candidate

    _dialogue_option_effect_entries = list(seen.values())
    logger.info(
        "dialogue DB: 加载 {} 条唯一效果条目（来自 {} 条原始记录）",
        len(_dialogue_option_effect_entries),
        len(data),
    )
    return _dialogue_option_effect_entries


def resolve_dialogue_option_identity(
    title: str,
    *,
    index: int,
    effect_text: str = "",
    p_cost: int | None = None,
) -> CandidateResolution:
    """解析并补全对话、option、标识并返回结果。

    Args:
        title: 用于提供title相关输入。
        index: 用于提供index相关输入。
        effect_text: 用于提供效果、text相关输入。
        p_cost: 用于提供p、cost相关输入。

    Returns:
        CandidateResolution: 返回值类型见注解。
    """
    param_kind = infer_param_kind(title)
    normalized_effect = _normalize_effect_lookup_text(effect_text)
    if not normalized_effect:
        return _fallback_dialogue_option_identity(title, index=index, param_kind=param_kind)

    entries = _load_dialogue_option_effect_entries()
    if not entries:
        return _fallback_dialogue_option_identity(title, index=index, param_kind=param_kind)

    best_entry: dict[str, Any] | None = None
    best_score = 0.0
    for entry in entries:
        if p_cost is not None and int(entry.get("produce_point", 0) or 0) != int(p_cost):
            continue
        score = SequenceMatcher(None, normalized_effect, str(entry.get("norm_desc") or "")).ratio()
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= _DIALOGUE_MATCH_THRESHOLD:
        db_id = str(best_entry.get("id") or "")
        db_desc = str(best_entry.get("raw_desc") or "")
        metadata: dict[str, Any] = {
            "dialogue_match_score": best_score,
            "dialogue_db_description": db_desc,
            "dialogue_effect_ids": list(best_entry.get("effect_ids") or []),
            "dialogue_stamina_cost": int(best_entry.get("stamina", 0) or 0),
        }
        if param_kind and param_kind != "unknown":
            metadata["param_kind"] = param_kind
        if p_cost is not None:
            metadata["p_cost"] = int(p_cost)
        if db_desc:
            metadata["description"] = f"効果: {db_desc}"
        return CandidateResolution(
            action_id=f"dialogue_option:{db_id}",
            candidate_type="dialogue_option",
            db_id=db_id,
            display_name=title,
            source="db_match",
            confidence=best_score,
            metadata=metadata,
        )

    return _fallback_dialogue_option_identity(title, index=index, param_kind=param_kind)


def _load_outing_activity_entries() -> list[dict[str, Any]]:
    """懒加载おでかけ活動 DB 条目（单例）。"""
    global _outing_activity_entries
    if _outing_activity_entries is not None:
        return _outing_activity_entries

    from src.utils.game_database_tools import get_game_database

    data = list(get_game_database("ProduceStepEventSuggestion").get_all_item() or [])
    if not data:
        _outing_activity_entries = []
        return _outing_activity_entries

    seen: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in data:
        entry_id = str(getattr(entry, "id", "") or "")
        if "activity" not in entry_id:
            continue
        p_cost = int(getattr(entry, "producePoint", 0) or 0)
        desc_parts = [str(getattr(item, "text", "") or "") for item in (getattr(entry, "produceDescriptions", []) or []) if str(getattr(item, "text", "") or "")]
        raw_desc = "".join(desc_parts)
        norm_desc = _OUTING_WHITESPACE_RE.sub("", raw_desc)

        key = (p_cost, norm_desc)
        is_generic = bool(_OUTING_GENERIC_ID_RE.match(entry_id))
        if key not in seen or (is_generic and not seen[key].get("is_generic")):
            seen[key] = {
                "id": entry_id,
                "produce_point": p_cost,
                "norm_desc": norm_desc,
                "raw_desc": raw_desc,
                "effect_ids": list(getattr(entry, "produceEffectIds", []) or []),
                "is_generic": is_generic,
            }

    _outing_activity_entries = list(seen.values())
    logger.info(
        "outing DB: 加载 {} 条唯一活動条目（来自 {} 条原始记录）",
        len(_outing_activity_entries),
        sum(1 for entry in data if "activity" in str(getattr(entry, "id", "") or "")),
    )
    return _outing_activity_entries


def resolve_outing_option_identity(
    *,
    p_cost: int | None,
    effect_text: str,
    title: str = "",
    index: int = 0,
) -> CandidateResolution:
    """解析并补全outing、option、标识并返回结果。

    Args:
        p_cost: 用于提供p、cost相关输入。
        effect_text: 用于提供效果、text相关输入。
        title: 用于提供title相关输入。
        index: 用于提供index相关输入。

    Returns:
        CandidateResolution: 返回值类型见注解。
    """
    if not effect_text:
        return resolve_dialogue_option_identity(title, index=index)

    entries = _load_outing_activity_entries()
    if not entries:
        return resolve_dialogue_option_identity(title, index=index)

    ocr_normalized = _OUTING_WHITESPACE_RE.sub("", effect_text)
    best_entry: dict[str, Any] | None = None
    best_score = 0.0

    for entry in entries:
        if p_cost is not None and entry["produce_point"] != p_cost:
            continue
        score = SequenceMatcher(None, ocr_normalized, entry["norm_desc"]).ratio()
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= _OUTING_MATCH_THRESHOLD:
        db_id = best_entry["id"]
        logger.debug(
            "outing DB: 选项 #{} '{}' → {} (score={:.2f}, P={})",
            index,
            (effect_text or "")[:30],
            db_id,
            best_score,
            p_cost,
        )
        return CandidateResolution(
            action_id=f"outing_activity:{db_id}",
            candidate_type="outing_activity",
            db_id=db_id,
            display_name=title,
            source="db_match",
            confidence=best_score,
            metadata={
                "outing_match_score": best_score,
                "outing_db_description": best_entry["raw_desc"],
                "outing_effect_ids": best_entry["effect_ids"],
            },
        )

    logger.debug(
        "outing DB: 选项 #{} '{}' 未匹配 (best_score={:.2f}, P={})",
        index,
        (effect_text or "")[:30],
        best_score,
        p_cost,
    )
    return resolve_dialogue_option_identity(title, index=index)


def resolve_lesson_option_identity(
    kind: str,
    *,
    stamina_cost: int | None = None,
    effect_text: str = "",
    index: int = 0,
) -> CandidateResolution:
    """解析授業課程選項的 action_id 和 rl_action_type。"""
    spec = _LESSON_OPTION_MAP.get(kind)
    if spec is not None:
        metadata: dict[str, Any] = {
            "param_kind": kind,
            "rl_action_type": spec["rl_action_type"],
            "lesson_option": True,
        }
        if stamina_cost is not None:
            metadata["stamina_cost"] = stamina_cost
        if effect_text:
            metadata["effect_text"] = effect_text
        return CandidateResolution(
            action_id=spec["action_id"],
            candidate_type="lesson_option",
            display_name=spec["display_name"],
            source="probe",
            confidence=1.0,
            metadata=metadata,
        )

    return CandidateResolution(
        action_id=f"lesson_option_unknown_{index}",
        candidate_type="lesson_option",
        display_name=f"授業選項{index + 1}",
        source="unknown",
        confidence=0.3,
        metadata={
            "param_kind": "unknown",
            "lesson_option": True,
            "stamina_cost": stamina_cost,
        },
    )


def _apply_resolution(candidate: Any, resolution: CandidateResolution) -> None:
    """处理apply、resolution并返回结果。

    Args:
        candidate: 单个候选项对象。
        resolution: 用于提供resolution相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    candidate.action_id = resolution.action_id
    candidate.db_id = resolution.db_id
    candidate.source = resolution.source
    candidate.confidence = resolution.confidence
    existing_metadata = getattr(candidate, "metadata", None)
    if existing_metadata is None:
        existing_metadata = {}
        candidate.metadata = existing_metadata
    existing_metadata.update(
        {
            "candidate_type": resolution.candidate_type,
            "source": resolution.source,
            **resolution.metadata,
        }
    )
    if resolution.db_id:
        existing_metadata.pop("unresolved", None)
    if resolution.display_name and hasattr(candidate, "title"):
        current_title = str(getattr(candidate, "title", "") or "")
        is_internal_id = current_title.startswith("schedule_action_")
        # schedule_action 没有 db_id，也要用规范化名称覆盖噪声 OCR。
        if (
            resolution.candidate_type == "schedule_action"
            or resolution.db_id
            or not current_title
            or is_internal_id
        ):
            candidate.title = resolution.display_name


def hydrate_schedule_candidates(candidates: Sequence[Any]) -> None:
    """处理hydrate、日程、candidates并返回结果。

    Args:
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", None) or {}
        resolution = resolve_schedule_action_identity(
            getattr(candidate, "title", ""),
            getattr(candidate, "kind", ""),
            index=getattr(candidate, "index", 0),
            is_sp=bool(metadata.get("is_sp")),
        )
        _apply_resolution(candidate, resolution)


def hydrate_dialogue_candidates(candidates: Sequence[Any]) -> None:
    """处理hydrate、对话、candidates并返回结果。

    Args:
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {}) or {}
        effect_text = str(
            metadata.get("option_effect")
            or metadata.get("outing_effect")
            or metadata.get("dialogue_db_description")
            or ""
        )
        p_cost_raw = metadata.get("p_cost")
        p_cost: int | None = None
        if p_cost_raw is not None and str(p_cost_raw).strip():
            try:
                p_cost = int(p_cost_raw)
            except (TypeError, ValueError):
                p_cost = None
        resolution = resolve_dialogue_option_identity(
            getattr(candidate, "title", ""),
            index=getattr(candidate, "index", 0),
            effect_text=effect_text,
            p_cost=p_cost,
        )
        _apply_resolution(candidate, resolution)


def hydrate_outing_candidates(candidates: Sequence[Any]) -> None:
    """おでかけ選項の DB ID 解析（探査后调用）。"""
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {}) or {}
        effect_text = str(metadata.get("outing_effect") or "")
        if not effect_text:
            continue
        resolution = resolve_outing_option_identity(
            p_cost=metadata.get("p_cost"),
            effect_text=effect_text,
            title=getattr(candidate, "title", ""),
            index=getattr(candidate, "index", 0),
        )
        if resolution.db_id:
            _apply_resolution(candidate, resolution)


def hydrate_lesson_candidates(candidates: Sequence[Any]) -> None:
    """处理hydrate、课程、candidates并返回结果。

    Args:
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {}) or {}
        kind = getattr(candidate, "kind", "") or metadata.get("lesson_stat", "unknown")
        resolution = resolve_lesson_option_identity(
            kind,
            stamina_cost=metadata.get("stamina_cost"),
            effect_text=str(metadata.get("lesson_effect") or ""),
            index=getattr(candidate, "index", 0),
        )
        _apply_resolution(candidate, resolution)
        if resolution.display_name and hasattr(candidate, "title"):
            candidate.title = resolution.display_name


__all__ = [
    "CandidateResolution",
    "ScheduleActionSpec",
    "_apply_resolution",
    "_build_unknown_action_id",
    "_clean_description_text",
    "_description_text",
    "_humanize_runtime_text",
    "_load_outing_activity_entries",
    "hydrate_dialogue_candidates",
    "hydrate_lesson_candidates",
    "hydrate_outing_candidates",
    "hydrate_schedule_candidates",
    "resolve_dialogue_option_identity",
    "resolve_lesson_option_identity",
    "resolve_outing_option_identity",
    "resolve_schedule_action_identity",
]
