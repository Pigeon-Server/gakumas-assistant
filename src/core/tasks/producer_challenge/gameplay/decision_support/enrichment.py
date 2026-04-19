from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import cv2

from src.constants.game.producer_gameplay import GameplayPhase
from src.constants.game.text.produce_text import ProduceText
from src.core.tasks.producer_challenge.catalog import match_card_and_item_entries
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
    repository = _get_rl_repository()
    if repository is None or not item_id:
        return _humanize_runtime_text(_description_text(fallback_entries))
    loc_map = repository.load_localization(table_name)
    row = {}
    if upgrade_count is not None:
        row = loc_map.get(f"{item_id}.{int(upgrade_count)}", {})
    if not row:
        row = loc_map.get(str(item_id), {})
    entries = row.get("produceDescriptions") or fallback_entries
    return _humanize_runtime_text(_description_text(entries))


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "train" / "gakumas_rl").exists():
            return parent
    return current.parents[6]


def _ensure_rl_package_on_path() -> None:
    rl_root = _repo_root() / "train" / "gakumas_rl"
    if rl_root.exists():
        rl_root_str = str(rl_root)
        if rl_root_str not in sys.path:
            sys.path.insert(0, rl_root_str)


@lru_cache(maxsize=1)
def _get_rl_repository():
    try:
        _ensure_rl_package_on_path()
        from gakumas_rl.service import get_repository

        return get_repository()
    except Exception as exc:  # noqa: BLE001 - 缺依赖时需要回退到 OCR/本地 catalog
        logger.debug(f"producer decision: 无法加载 gakumas_rl 主数据仓库，回退文本匹配: {exc}")
        return None


def _lookup_card_row(card_id: str, *, upgrade_count: int | None = None) -> dict[str, Any] | None:
    repository = _get_rl_repository()
    if repository is None or not card_id:
        return None
    if upgrade_count is not None:
        return repository.card_row_by_upgrade(card_id, upgrade_count)
    return repository.canonical_card_row(card_id)


def _lookup_named_row(table_name: str, item_id: str) -> dict[str, Any] | None:
    repository = _get_rl_repository()
    if repository is None or not item_id:
        return None
    table = getattr(repository, table_name, None)
    if table is None:
        table = repository.load_table(
            "ProduceDrink" if table_name == "produce_drinks" else "ProduceItem"
        )
    return table.first(item_id)


def _match_catalog_entry_from_texts(
    texts: Sequence[str],
    *,
    expected_kind: str | None = None,
) -> dict[str, Any] | None:
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
    return _match_catalog_entry_from_texts([title], expected_kind=expected_kind)


def _enrich_card_metadata(card_id: str, *, upgrade_count: int = 0) -> dict[str, Any]:
    row = _lookup_card_row(card_id, upgrade_count=upgrade_count)
    repository = _get_rl_repository()
    if row is None or repository is None:
        return {
            "upgrade_count": int(upgrade_count),
            "description": "",
        }
    return {
        "upgrade_count": int(row.get("upgradeCount") or upgrade_count or 0),
        "rarity": str(row.get("rarity") or ""),
        "category": str(row.get("category") or ""),
        "plan_type": str(row.get("planType") or ""),
        "plan_type_label": _plan_type_payload(row.get("planType")).get("label", ""),
        "cost_type": str(row.get("costType") or ""),
        "cost": int(row.get("stamina") or 0),
        "display_name": repository.card_name(row),
        "raw_name": repository.raw_card_name(row),
        "description": _localized_humanized_description(
            "ProduceCard",
            card_id,
            row.get("produceDescriptions"),
            upgrade_count=int(row.get("upgradeCount") or upgrade_count or 0),
        ),
        "effect_types": repository.card_axis_effect_types(row),
        "trigger_phases": repository.card_trigger_phases(row),
    }


def _enrich_drink_metadata(drink_id: str) -> dict[str, Any]:
    row = _lookup_named_row("produce_drinks", drink_id)
    repository = _get_rl_repository()
    if row is None or repository is None:
        return {}
    return {
        "rarity": str(row.get("rarity") or ""),
        "plan_type": str(row.get("planType") or ""),
        "plan_type_label": _plan_type_payload(row.get("planType")).get("label", ""),
        "display_name": repository.drink_name(row),
        "raw_name": repository.raw_drink_name(row),
        "description": _localized_humanized_description(
            "ProduceDrink",
            drink_id,
            row.get("produceDescriptions"),
        ),
        "effect_types": repository.drink_axis_effect_types(row),
    }


def _enrich_item_metadata(item_id: str) -> dict[str, Any]:
    row = _lookup_named_row("produce_items", item_id)
    repository = _get_rl_repository()
    if row is None or repository is None:
        return {}
    return {
        "rarity": str(row.get("rarity") or ""),
        "display_name": repository.item_name(row),
        "raw_name": repository.raw_item_name(row),
        "description": _localized_humanized_description(
            "ProduceItem",
            item_id,
            row.get("produceDescriptions"),
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
