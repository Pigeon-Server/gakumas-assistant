from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
import re
from typing import Any, Sequence

import cv2

from src.constants.game.producer_gameplay import GameplayPhase
from src.constants.game.text.produce_text import ProduceText
from src.core.tasks.producer_challenge.catalog import match_card_and_item_entries
from src.utils.game_database_tools import (
    GakumasDatabase_ProduceCardDataUtils,
    GakumasDatabase_ProduceDrinkDataUtils,
    GakumasDatabase_ProduceItemDataUtils,
)
from src.utils.logger import logger
from src.utils.runtime_paths import resolve_data_str
from src.utils.string_tools import fullwidth_to_halfwidth

from .identity import _description_text, _humanize_runtime_text

_NUMBER_RE = re.compile(r"\d+")

_DRINK_EFFECT_SCORE_WEIGHTS = {
    "ProduceExamEffectType_Score": 38.0,
    "ProduceExamEffectType_ParameterBuff": 34.0,
    "ProduceExamEffectType_Review": 30.0,
    "ProduceExamEffectType_Aggressive": 28.0,
    "ProduceExamEffectType_Block": 26.0,
    "ProduceExamEffectType_Enthusiastic": 18.0,
    "ProduceExamEffectType_FullPowerPoint": 16.0,
    "ProduceExamEffectType_FullPower": 16.0,
}
_DRINK_DESCRIPTION_SCORE_RULES = (
    (ProduceText.SKILL_CARD_USE_COUNT_UP, 34.0),
    (ProduceText.PARAMETER_UP_INCREASE, 28.0),
    (ProduceText.GOOD_CONDITION, 18.0),
    (ProduceText.CONCENTRATION, 16.0),
    (ProduceText.GOOD_IMPRESSION, 16.0),
    (ProduceText.GENKI, 15.0),
    (ProduceText.STAMINA_RECOVERY, 22.0),
    (ProduceText.STAMINA_CONSUMPTION, 12.0),
    (ProduceText.ENTHUSIASM, 10.0),
    (ProduceText.FULL_POWER_POINT, 10.0),
)
_DRINK_RARITY_BONUS = {
    "SSR": 8.0,
    "SR": 5.0,
    "R": 2.0,
}
_PLAN_TYPE_METADATA = {
    "ProducePlanType_Plan1": {
        "label": ProduceText.PLAN_SENSE,
        "focus": " / ".join(ProduceText.BATTLE_SENSE_TOKENS),
        "description": (
            f"官方主轴是{ProduceText.GOOD_CONDITION}、{ProduceText.CONCENTRATION}"
            f"与{ProduceText.EXCELLENT_CONDITION}，偏向放大单次参数/得分收益。"
        ),
    },
    "ProducePlanType_Plan2": {
        "label": ProduceText.PLAN_LOGIC,
        "focus": " / ".join(ProduceText.BATTLE_LOGIC_TOKENS),
        "description": (
            f"官方主轴是{ProduceText.GOOD_IMPRESSION}与{ProduceText.YARUKI}，"
            "偏向回合结算收益与续航。"
        ),
    },
    "ProducePlanType_Plan3": {
        "label": ProduceText.PLAN_ANOMALY,
        "focus": " / ".join(ProduceText.BATTLE_ANOMALY_TOKENS),
        "description": (
            f"官方主轴是{ProduceText.FULL_POWER}、{ProduceText.STRONG_SPIRIT}、"
            f"{ProduceText.CONSERVE_POWER}与{ProduceText.ENTHUSIASM}，"
            "偏向指针切换和爆发回合。"
        ),
    },
}

_SP_PINK_RATIO_THRESHOLD = 0.03
_SP_COMP_RATIO_THRESHOLD = 0.015
_SP_COOL_RATIO_THRESHOLD = 0.01
_SP_WHITE_RATIO_THRESHOLD = 0.012
_SP_COLOR_RATIO_THRESHOLD = 0.045


def score_produce_drink_metadata(
    metadata: dict[str, Any] | None,
    *,
    phase: str = "",
    stamina: int = 0,
    max_stamina: int = 0,
    remaining_turns: int = 0,
) -> float:
    """根据主库效果轴和描述，为 P 饮料给一个通用价值分。"""
    payload = dict(metadata or {})
    description = str(payload.get("description") or "")
    effect_types = [str(value or "") for value in payload.get("effect_types", []) or []]
    rarity = str(payload.get("rarity") or "").upper()
    phase_key = phase.value if hasattr(phase, "value") else str(phase)

    score = _DRINK_RARITY_BONUS.get(rarity, 0.0)
    for effect_type in effect_types:
        for token, weight in _DRINK_EFFECT_SCORE_WEIGHTS.items():
            if token in effect_type:
                score += weight
                break

    normalized_description = fullwidth_to_halfwidth(description)
    for keyword, weight in _DRINK_DESCRIPTION_SCORE_RULES:
        if keyword in normalized_description:
            score += weight

    numeric_values = [int(value) for value in _NUMBER_RE.findall(normalized_description)]
    if numeric_values:
        score += min(max(numeric_values), 30) * 0.45

    stamina_ratio = (
        float(stamina) / max(int(max_stamina), 1)
        if int(max_stamina or 0) > 0
        else 1.0
    )
    if stamina_ratio <= 0.35 and any(
        token in normalized_description
        for token in (
            *ProduceText.BATTLE_RECOVERY_TOKENS,
            ProduceText.BLOCK,
        )
    ):
        score += 18.0
    if phase_key in {GameplayPhase.LESSON, GameplayPhase.EXAM} and remaining_turns <= 2 and any(
        token in normalized_description
        for token in (
            ProduceText.GOOD_CONDITION,
            ProduceText.CONCENTRATION,
            ProduceText.GOOD_IMPRESSION,
            ProduceText.SKILL_CARD_USE_COUNT_UP,
            ProduceText.PARAMETER_UP_INCREASE,
        )
    ):
        score += 12.0
    return score


def _plan_type_payload(plan_type: Any) -> dict[str, str]:
    """处理养成路线、type、载荷并返回结果。

    Args:
        plan_type: 用于提供养成路线、type相关输入。

    Returns:
        dict: 结构化结果字典。
    """
    raw_value = str(plan_type or "").strip()
    metadata = _PLAN_TYPE_METADATA.get(raw_value, {})
    return {
        "type": raw_value,
        "label": str(metadata.get("label") or ""),
        "focus": str(metadata.get("focus") or ""),
        "description": str(metadata.get("description") or ""),
    }


def _localized_humanized_description(
    table_name: str,
    item_id: str,
    fallback_entries: Any = None,
    *,
    upgrade_count: int | None = None,
) -> str:
    """将条目描述做本地化与可读化处理，输出可用于决策的文本。

    Args:
        table_name: 本项目数据库表名（`ProduceCard` / `ProduceDrink` / `ProduceItem`）。
        item_id: 目标条目 ID。
        fallback_entries: 数据库未命中时使用的回退描述数据。
        upgrade_count: 强化次数，用于读取强化后的条目描述。

    Returns:
        str: 处理后的文本结果。
    """
    entries = _resolve_localized_descriptions(
        table_name=table_name,
        item_id=item_id,
        upgrade_count=upgrade_count,
    ) or fallback_entries
    return _humanize_runtime_text(_description_text(entries))


@lru_cache(maxsize=1)
def _produce_card_db() -> GakumasDatabase_ProduceCardDataUtils:
    """获取本项目技能卡数据库实例（缓存）。"""
    return GakumasDatabase_ProduceCardDataUtils()


@lru_cache(maxsize=1)
def _produce_drink_db() -> GakumasDatabase_ProduceDrinkDataUtils:
    """获取本项目 P 饮料数据库实例（缓存）。"""
    return GakumasDatabase_ProduceDrinkDataUtils()


@lru_cache(maxsize=1)
def _produce_item_db() -> GakumasDatabase_ProduceItemDataUtils:
    """获取本项目 P 物品数据库实例（缓存）。"""
    return GakumasDatabase_ProduceItemDataUtils()


def _lookup_card_row(card_id: str, *, upgrade_count: int | None = None) -> Any | None:
    """按卡牌 ID（及强化次数）查询技能卡实体。"""
    if not card_id:
        return None
    db = _produce_card_db()
    if upgrade_count is not None:
        payload = db.get_by_id(f"{card_id}.{int(upgrade_count)}")
        if payload is not None:
            return payload
    payload = db.get_by_id(f"{card_id}.0")
    if payload is not None:
        return payload
    return db.get_by_raw_id(card_id)


def _lookup_drink_row(drink_id: str) -> Any | None:
    """按饮料 ID 查询 P 饮料实体。"""
    if not drink_id:
        return None
    return _produce_drink_db().get_by_id(str(drink_id))


def _lookup_item_row(item_id: str) -> Any | None:
    """按物品 ID 查询 P 物品实体。"""
    if not item_id:
        return None
    return _produce_item_db().get_by_id(str(item_id))


def _display_name(payload: Any) -> str:
    """获取实体展示名（优先本地化，失败回退原名）。"""
    loc = getattr(payload, "localization", None)
    return str((getattr(loc, "name", None) if loc else None) or getattr(payload, "name", "") or "")


def _raw_name(payload: Any) -> str:
    """获取实体原始名称。"""
    return str(getattr(payload, "name", "") or "")


def _effect_types_from_effect_groups(effect_groups: Sequence[Any] | None) -> list[str]:
    """从 EffectGroup 列表提取效果类型集合。"""
    result: list[str] = []
    if not effect_groups:
        return result
    for group in effect_groups:
        if group is None:
            continue
        for values in (
            [getattr(group, "examEffectType", "")],
            [getattr(group, "produceEffectType", "")],
            list(getattr(group, "examEffectTypes", []) or []),
            list(getattr(group, "produceEffectTypes", []) or []),
        ):
            for value in values:
                effect = str(value or "").strip()
                if effect and effect not in result:
                    result.append(effect)
    return result


def _resolve_localized_descriptions(
    *,
    table_name: str,
    item_id: str,
    upgrade_count: int | None = None,
) -> Any:
    """读取本项目数据库中的本地化描述片段。"""
    if not item_id:
        return None
    table = str(table_name or "").strip()
    if table == "ProduceCard":
        payload = _lookup_card_row(item_id, upgrade_count=upgrade_count)
    elif table == "ProduceDrink":
        payload = _lookup_drink_row(item_id)
    elif table == "ProduceItem":
        payload = _lookup_item_row(item_id)
    else:
        payload = None
    if payload is None:
        return None
    loc = getattr(payload, "localization", None)
    return (getattr(loc, "produceDescriptions", None) if loc else None) or getattr(payload, "produceDescriptions", None)


def _match_catalog_entry_from_texts(
    texts: Sequence[str],
    *,
    expected_kind: str | None = None,
) -> dict[str, Any] | None:
    """匹配`catalog_entry_from_texts`。"""
    normalized_texts = [str(text or "").strip() for text in texts if str(text or "").strip()]
    if not normalized_texts:
        return None
    matches = match_card_and_item_entries(normalized_texts, threshold=72)
    if expected_kind is not None:
        matches = [entry for entry in matches if entry["kind"] == expected_kind]
    if not matches:
        return None
    matches.sort(key=lambda entry: float(entry.get("score") or 0.0), reverse=True)
    return matches[0]


def _match_catalog_entry(
    title: str,
    *,
    expected_kind: str | None = None,
) -> dict[str, Any] | None:
    """匹配catalog、entry并返回结果。

    Args:
        title: 用于提供title相关输入。
        expected_kind: 用于提供expected、kind相关输入。

    Returns:
        dict: 结构化结果字典。
    """
    return _match_catalog_entry_from_texts([title], expected_kind=expected_kind)


def _enrich_card_metadata(card_id: str, *, upgrade_count: int = 0) -> dict[str, Any]:
    """补全`card_metadata`信息。"""
    row = _lookup_card_row(card_id, upgrade_count=upgrade_count)
    if row is None:
        return {
            "upgrade_count": int(upgrade_count),
            "description": "",
        }
    play_effect_types: list[str] = []
    for effect in list(getattr(row, "playEffects", []) or []):
        effect_cls = getattr(effect, "produceExamEffectCls", None)
        effect_type = str(getattr(effect_cls, "effectType", "") or "")
        if effect_type and effect_type not in play_effect_types:
            play_effect_types.append(effect_type)
    for effect in list(getattr(row, "moveProduceExamEffectClss", []) or []):
        effect_type = str(getattr(effect, "effectType", "") or "")
        if effect_type and effect_type not in play_effect_types:
            play_effect_types.append(effect_type)
    for effect in _effect_types_from_effect_groups(getattr(row, "effectGroupClss", [])):
        if effect not in play_effect_types:
            play_effect_types.append(effect)

    trigger_phases: list[str] = []
    trigger_candidates = [getattr(row, "playProduceExamTriggerCls", None)] + list(
        getattr(row, "moveProduceExamTriggerClss", []) or []
    )
    for trigger in trigger_candidates:
        if trigger is None:
            continue
        for phase in list(getattr(trigger, "phaseTypes", []) or []):
            phase_text = str(phase or "")
            if phase_text and phase_text not in trigger_phases:
                trigger_phases.append(phase_text)

    real_upgrade_count = int(getattr(row, "upgradeCount", upgrade_count) or 0)
    return {
        "upgrade_count": real_upgrade_count,
        "rarity": str(getattr(row, "rarity", "") or ""),
        "category": str(getattr(row, "category", "") or ""),
        "plan_type": str(getattr(row, "planType", "") or ""),
        "plan_type_label": _plan_type_payload(getattr(row, "planType", "")).get("label", ""),
        "cost_type": str(getattr(row, "costType", "") or ""),
        "cost": int(getattr(row, "stamina", 0) or 0),
        "display_name": _display_name(row),
        "raw_name": _raw_name(row),
        "description": _localized_humanized_description(
            "ProduceCard",
            card_id,
            getattr(row, "produceDescriptions", None),
            upgrade_count=real_upgrade_count,
        ),
        "effect_types": play_effect_types,
        "trigger_phases": trigger_phases,
    }


def _enrich_drink_metadata(drink_id: str) -> dict[str, Any]:
    """补全`drink_metadata`信息。"""
    row = _lookup_drink_row(drink_id)
    if row is None:
        return {}
    return {
        "rarity": str(getattr(row, "rarity", "") or ""),
        "plan_type": str(getattr(row, "planType", "") or ""),
        "plan_type_label": _plan_type_payload(getattr(row, "planType", "")).get("label", ""),
        "display_name": _display_name(row),
        "raw_name": _raw_name(row),
        "description": _localized_humanized_description(
            "ProduceDrink",
            drink_id,
            getattr(row, "produceDescriptions", None),
        ),
        "effect_types": _effect_types_from_effect_groups(getattr(row, "effectGroupClss", [])),
    }


def _enrich_item_metadata(item_id: str) -> dict[str, Any]:
    """补全`item_metadata`信息。"""
    row = _lookup_item_row(item_id)
    if row is None:
        return {}
    return {
        "rarity": str(getattr(row, "rarity", "") or ""),
        "display_name": _display_name(row),
        "raw_name": _raw_name(row),
        "description": _localized_humanized_description(
            "ProduceItem",
            item_id,
            getattr(row, "produceDescriptions", None),
        ),
    }


def detect_sp_badge(action_box: Any) -> bool:
    """检测 PC_ACTION 框左上角区域是否存在 SP 渐变徽章。"""
    frame = getattr(action_box, "frame", None)
    if frame is None or frame.size == 0:
        return False

    height, width = frame.shape[:2]
    # SP 徽章位于左上角，包含紫/红到蓝的渐变与白色 "SP" 字样。
    roi = frame[: max(1, int(height * 0.36)), : max(1, int(width * 0.32))]
    if roi.size == 0:
        return False

    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    area = float(roi.shape[0] * roi.shape[1])

    saturated_mask = cv2.inRange(hsv, (0, 70, 85), (180, 255, 255))
    warm_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (145, 90, 90), (180, 255, 255)),
        cv2.inRange(hsv, (0, 90, 90), (14, 255, 255)),
    )
    cool_mask = cv2.inRange(hsv, (84, 90, 90), (145, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 190), (180, 70, 255))

    warm_ratio = cv2.countNonZero(warm_mask) / area
    cool_ratio = cv2.countNonZero(cool_mask) / area
    white_ratio = cv2.countNonZero(white_mask) / area
    color_ratio = cv2.countNonZero(saturated_mask) / area

    if warm_ratio < _SP_PINK_RATIO_THRESHOLD and cool_ratio < _SP_COOL_RATIO_THRESHOLD:
        return False
    if white_ratio < _SP_WHITE_RATIO_THRESHOLD or color_ratio < _SP_COLOR_RATIO_THRESHOLD:
        return False

    color_union = cv2.bitwise_or(warm_mask, cool_mask)
    contours, _ = cv2.findContours(color_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max((cv2.contourArea(contour) for contour in contours), default=0)
    comp_ratio = largest / area

    return comp_ratio >= _SP_COMP_RATIO_THRESHOLD


def _learn_card_clip_from_db_id(app: Any, image: Any, card_id: str, *, upgrade_count: int = 0) -> None:
    """将技能卡图像写入 CLIP 记忆库，建立 `card_id(+upgrade_count)` 到图像特征的映射。

    Args:
        app: 应用处理器实例，用于获取 `clip_manager`。
        image: 待写入记忆库的卡面图像（numpy 数组）。
        card_id: 技能卡数据库 ID（不含强化后缀）。
        upgrade_count: 技能卡强化次数；优先按该强化级别查表，失败时回退到 `.0`。

    Returns:
        None: 仅执行 CLIP 记忆学习副作用，不返回业务值。
    """
    if image is None or getattr(image, "size", 0) <= 0 or not card_id:
        return
    clip_manager = getattr(app, "clip_manager", None)
    if clip_manager is None:
        return
    skill_card_clip = getattr(clip_manager, "skill_card_clip", None)
    if skill_card_clip is None:
        return
    try:
        from src.utils.game_database_tools import GakumasDatabase_ProduceCardDataUtils

        payload = GakumasDatabase_ProduceCardDataUtils().get_by_id(f"{card_id}.{int(upgrade_count)}")
        if payload is None:
            payload = GakumasDatabase_ProduceCardDataUtils().get_by_id(f"{card_id}.0")
        if payload is None:
            return
        skill_card_clip.add_to_memory(image, payload, similarity_threshold=0.98)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"producer decision: 技能卡 CLIP 学习失败 {card_id}: {exc}")


def _learn_drink_clip_from_db_id(app: Any, image: Any, drink_id: str) -> None:
    """将 P 饮料图像写入 CLIP 记忆库，建立 `drink_id` 到图像特征的映射。

    Args:
        app: 应用处理器实例，用于获取 `clip_manager`。
        image: 待写入记忆库的饮料图像（numpy 数组）。
        drink_id: P 饮料数据库 ID。

    Returns:
        None: 仅执行 CLIP 记忆学习副作用，不返回业务值。
    """
    if image is None or getattr(image, "size", 0) <= 0 or not drink_id:
        return
    clip_manager = getattr(app, "clip_manager", None)
    if clip_manager is None:
        return
    produce_drink_clip = getattr(clip_manager, "produce_drink_clip", None)
    if produce_drink_clip is None:
        return
    try:
        from src.utils.game_database_tools import GakumasDatabase_ProduceDrinkDataUtils

        payload = GakumasDatabase_ProduceDrinkDataUtils().get_by_id(str(drink_id))
        if payload is None:
            return
        produce_drink_clip.add_to_memory(image, payload, similarity_threshold=0.98)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"producer decision: P饮料 CLIP 学习失败 {drink_id}: {exc}")


def _learn_item_clip_from_db_id(app: Any, image: Any, item_id: str) -> None:
    """将 P 物品图像写入 CLIP 记忆库，建立 `item_id` 到图像特征的映射。

    Args:
        app: 应用处理器实例，用于获取 `clip_manager`。
        image: 待写入记忆库的物品图像（numpy 数组）。
        item_id: P 物品数据库 ID。

    Returns:
        None: 仅执行 CLIP 记忆学习副作用，不返回业务值。
    """
    if image is None or getattr(image, "size", 0) <= 0 or not item_id:
        return
    clip_manager = getattr(app, "clip_manager", None)
    if clip_manager is None:
        return
    produce_item_clip = getattr(clip_manager, "produce_item_clip", None)
    if produce_item_clip is None:
        return
    try:
        from src.utils.game_database_tools import GakumasDatabase_ProduceItemDataUtils

        payload = GakumasDatabase_ProduceItemDataUtils().get_by_id(str(item_id))
        if payload is None:
            return
        produce_item_clip.add_to_memory(image, payload, similarity_threshold=0.98)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"producer decision: P物品 CLIP 学习失败 {item_id}: {exc}")


def _auto_collect_unresolved_entity_image(box: Any, index: int) -> None:
    """CLIP 识别失败时自动采集未识别的实体图像，用于后续人工标注和学习。"""
    frame = getattr(box, "frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return
    try:
        collect_dir = resolve_data_str("CLIP", "unresolved_consult")
        os.makedirs(collect_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(collect_dir, f"entity_{timestamp}_{index}.png")
        cv2.imwrite(path, frame)
        logger.info(f"[CLIP] 未识别实体已采集至: {path}")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[CLIP] 自动采集失败: {exc}")


__all__ = [
    "_auto_collect_unresolved_entity_image",
    "_enrich_card_metadata",
    "_enrich_drink_metadata",
    "_enrich_item_metadata",
    "_learn_card_clip_from_db_id",
    "_learn_drink_clip_from_db_id",
    "_learn_item_clip_from_db_id",
    "_match_catalog_entry",
    "_match_catalog_entry_from_texts",
    "_plan_type_payload",
    "detect_sp_badge",
    "score_produce_drink_metadata",
]
