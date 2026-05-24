from __future__ import annotations

import copy
from collections import Counter
import re
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import numpy as np

from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.general_text import GeneralText
from src.constants.game.text.produce_text import ProduceText
from src.core.tasks.producer_challenge.shared.common import (
    infer_param_kind,
    ocr_text,
)
from src.utils.logger import logger
from src.utils.debug_tools import DebugTools
from src.utils.string_tools import fullwidth_to_halfwidth, normalize_ocr_jp
from src.core.tasks.producer_challenge.gameplay.exam_ranking import get_exam_ranking_value
from .decision_support import (
    CandidateResolution,
    _auto_collect_unresolved_entity_image,
    _append_exam_snapshot_details,
    _apply_resolution,
    _build_stage_context,
    _build_unknown_action_id,
    _build_parameter_stats_payload,
    _compute_remaining_weeks,
    _enrich_card_metadata,
    _enrich_drink_metadata,
    _enrich_item_metadata,
    _extract_first_int,
    _extract_noisy_hud_value,
    _extract_planning_parameter_value,
    _get_parameter_seed_value,
    _learn_card_clip_from_db_id,
    _learn_drink_clip_from_db_id,
    _learn_item_clip_from_db_id,
    _match_catalog_entry,
    _match_catalog_entry_from_texts,
    _parse_progress_circle,
    _parse_stamina_text,
    _plan_type_payload,
    _serialize_box,
    _sync_virtual_battle_state,
    _resolve_repeated_digit_ocr_value,
    is_produce_card_action_id,
    is_produce_drink_action_id,
    register_realtime_resource_snapshot,
    serialize_candidate,
)

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_SCORE_BONUS_PARAM_PERCENT_RE = re.compile(
    r"(?:ボーカル|ダンス|ビジュアル|VO|DA|VI)\D{0,6}(\d{2,6})\s*[%％]",
    re.IGNORECASE,
)
_SCORE_BONUS_PERCENT_RE = re.compile(r"(\d{2,6})\s*[%％]")
_SCORE_BONUS_X_RE = re.compile(r"[x×X]\s*(\d{1,4}(?:\.\d+)?)")
_PERCENT_BASED_RESOURCE_PATTERNS = (
    re.compile(
        rf"({'|'.join(ProduceText.STATUS_VALUE_TOKENS)})\s*の\s*\d+\s*[%％]"
    ),
    re.compile(
        rf"({'|'.join(ProduceText.STATUS_VALUE_TOKENS)})\s*的\s*\d+\s*[%％]"
    ),
)
_SNAPSHOT_RESOURCE_KEY_BY_LABEL = {
    ProduceText.GOOD_IMPRESSION: "aggressive",
    ProduceText.CONCENTRATION: "review",
    ProduceText.GOOD_CONDITION: "parameter_buff",
    ProduceText.GENKI: "block",
    ProduceText.ENTHUSIASM: "enthusiastic",
    ProduceText.FULL_POWER_POINT: "full_power_point",
}
_SNAPSHOT_CARD_CATEGORY_NAMES = {
    "ProduceCardCategory_ActiveSkill": "Active",
    "ProduceCardCategory_MentalSkill": "Mental",
    "ProduceCardCategory_Trouble": "Trouble",
}

_SNAPSHOT_DESCRIPTION_NOISE_PATTERN = re.compile(r"\s*[；;]\s*")
_SNAPSHOT_DESCRIPTION_PARENS_PATTERN = re.compile(r"\s*[（(]\s*")
_SNAPSHOT_DESCRIPTION_PARENS_CLOSE_PATTERN = re.compile(r"\s*[）)]\s*")
_SNAPSHOT_DESCRIPTION_PLUS_PATTERN = re.compile(r"\s*[＋+]\s*")
_SNAPSHOT_DESCRIPTION_MULTI_SPACE_PATTERN = re.compile(r"\s+")

_CARD_CATEGORY_DISPLAY_OVERRIDES = {
    ProduceText.CARD_TYPE_ACTIVE: "Active",
    ProduceText.CARD_TYPE_MENTAL: "Mental",
    ProduceText.CARD_TYPE_TROUBLE: "Trouble",
}

_DECK_SUMMARY_EMPTY_TEXT = "未知"


_OFFENSIVE_EFFECT_KEYWORDS = (
    "ExamLesson",
    "ExamLessonFix",
    "ProduceExamEffectType_Score",
    "打分",
    "固定打分",
    "スコア",
)
_OFFENSIVE_DESCRIPTION_KEYWORDS = (
    "打分",
    "固定打分",
    ProduceText.SCORE,
    ProduceText.PARAMETER,
)


def _parse_score_bonus_from_bonus_text(
    bonus_text: str,
    *,
    remaining_turns_text: str = "",
) -> str:
    """从奖励指示文本中提取分数倍率，优先百分比格式。"""
    normalized = normalize_ocr_jp(fullwidth_to_halfwidth(str(bonus_text or "")))
    if not normalized.strip():
        return ""

    percent_values = [
        int(value)
        for value in _SCORE_BONUS_PARAM_PERCENT_RE.findall(normalized)
    ]
    if not percent_values:
        percent_values = [int(value) for value in _SCORE_BONUS_PERCENT_RE.findall(normalized)]
    valid_values = [value for value in percent_values if 50 <= value <= 9999]
    if valid_values:
        bonus_value = max(valid_values)
        remaining_turns = _extract_first_int(remaining_turns_text)
        # OCR 粘连时常见 "5"+"519%" => "5519%"，优先剥离前缀回合数。
        if remaining_turns > 0 and bonus_value >= 1000:
            bonus_str = str(int(bonus_value))
            prefix = str(int(remaining_turns))
            if bonus_str.startswith(prefix) and len(bonus_str) > len(prefix):
                tail = int(bonus_str[len(prefix):])
                if 50 <= tail <= 9999:
                    bonus_value = tail
        return str(int(bonus_value))

    x_match = _SCORE_BONUS_X_RE.search(normalized)
    if x_match:
        return str(x_match.group(1))
    return ""

_VISUAL_DISABLED_LOWER_HSV = (0, 0, 0)
_VISUAL_DISABLED_UPPER_HSV = (179, 255, 155)
_VISUAL_DISABLED_MASK_RATIO = 0.74
_HUD_GENKI_SHIELD_LOWER_HSV = (62, 0, 175)
_HUD_GENKI_SHIELD_UPPER_HSV = (93, 190, 255)
_HUD_STAMINA_HEART_LOWER_HSV = (0, 39, 97)
_HUD_STAMINA_HEART_UPPER_HSV = (88, 169, 255)
_HUD_STAMINA_BAR_LOWER_HSV = (19, 143, 0)
_HUD_STAMINA_BAR_UPPER_HSV = (100, 186, 255)
_EFFECT_TERM_HINTS = (
    (ProduceText.SKILL_CARD_USE_COUNT_UP, "本回合可以多打一张技能卡"),
    (ProduceText.PARAMETER_UP_INCREASE, "会抬高后续参数/得分型动作的收益"),
    (ProduceText.STAMINA_RECOVERY, "会直接回复体力，缓解低体力卡手"),
    (ProduceText.EXCELLENT_CONDITION, "会按当前好調层数进一步放大好調收益"),
    (ProduceText.GOOD_IMPRESSION, "会在回合结束按层数结算一次收益，并在回合开始时-1"),
    (ProduceText.YARUKI, "每+1都会提高元気的获取量"),
    (ProduceText.FULL_POWER_POINT, "累计到10会在下回合进入全力，并额外+1出牌次数"),
    (ProduceText.FULL_POWER, "会大幅提高参数/得分收益，并额外+1出牌次数"),
    (ProduceText.STRONG_SPIRIT, "会提高参数/得分收益，但也会增加体力消耗"),
    (ProduceText.CONSERVE_POWER, "会降低当前收益和体力消耗，解除时返还热意/元気/出牌次数"),
    (ProduceText.ENTHUSIASM, "每+1都会再追加1点参数/得分基础值，回合结束归零"),
    (ProduceText.GENKI, "会优先代替体力支付，且不能带到下一场レッスン/試験"),
    (ProduceText.CONCENTRATION, "每+1都会再追加1点参数/得分基础值，不会自然衰减"),
    (ProduceText.GOOD_CONDITION, "会把参数/得分上升量提高50%，并随回合衰减"),
)
def _current_idol_plan_payload(ctx: "ProduceContext") -> dict[str, str]:
    """读取当前偶像卡的养成路线信息并构建载荷。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        dict: 结构化结果字典。
    """
    idol_card = getattr(ctx, "selected_idol_card", None)
    # 回退: 从主数据库按配置的目标偶像卡 ID 查询
    if idol_card is None:
        target_id = getattr(ctx, "target_idol_card_id", "") or ""
        if target_id:
            try:
                from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils
                idol_card = GakumasDatabase_IdolCardDataUtils().get_by_id(target_id)
                if idol_card is not None:
                    ctx.selected_idol_card = idol_card
            except Exception:  # 数据库查询失败时忽略
                pass
    if idol_card is None:
        return _plan_type_payload("")
    return _plan_type_payload(getattr(idol_card, "planType", ""))


def _build_parameter_priority(ctx: "ProduceContext") -> str:
    """根据偶像卡成长率计算属性优先级排序（如 'visual > dance > vocal'）。
    优先使用 ctx.selected_idol_card；若为空则从主数据库按 target_idol_card_id 查询。
    """
    idol_card = getattr(ctx, "selected_idol_card", None)

    # 回退: 从主数据库按配置的目标偶像卡 ID 查询
    if idol_card is None:
        target_id = getattr(ctx, "target_idol_card_id", "") or ""
        if target_id:
            try:
                from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils
                idol_card = GakumasDatabase_IdolCardDataUtils().get_by_id(target_id)
                if idol_card is not None:
                    # 同时回填 ctx，后续调用不再重复查询
                    ctx.selected_idol_card = idol_card
            except Exception:
                pass

    if idol_card is None:
        return ""
    growth = {
        "vocal": int(getattr(idol_card, "produceVocalGrowthRatePermil", 0) or 0),
        "dance": int(getattr(idol_card, "produceDanceGrowthRatePermil", 0) or 0),
        "visual": int(getattr(idol_card, "produceVisualGrowthRatePermil", 0) or 0),
    }
    sorted_params = sorted(growth.items(), key=lambda x: x[1], reverse=True)
    return " > ".join(p[0] for p in sorted_params)


def _build_consult_session_summary(ctx: "ProduceContext") -> dict[str, Any]:
    """从 handler_state 和 operation_history 构建当前相談 session 的操作摘要。"""
    handler = ctx.handler_state
    used_enhancement = bool(handler.get("consult_auto_used_enhancement"))
    used_remove = bool(handler.get("consult_auto_used_remove"))
    # 统计本次相談中已完成的兑换操作
    exchanged_items: list[str] = []
    for op in reversed(ctx.operation_history):
        if op.phase != GameplayPhase.CONSULT:
            break
        if op.action == "consult_exchange":
            name = op.target or (op.details or {}).get("db_id", "")
            if name:
                exchanged_items.append(name)
    exchanged_items.reverse()
    return {
        "used_enhancement": used_enhancement,
        "used_remove": used_remove,
        "exchanged_items": exchanged_items,
        "actions_taken": len(exchanged_items) + int(used_enhancement) + int(used_remove),
    }


def _build_effect_term_hints(text: str) -> list[str]:
    """构建效果、术语、hints并返回结果。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    remaining_text = fullwidth_to_halfwidth(str(text or ""))
    hints: list[str] = []
    for token, hint in _EFFECT_TERM_HINTS:
        if token not in remaining_text:
            continue
        hints.append(f"{token}={hint}")
        remaining_text = remaining_text.replace(token, " ")
    return hints


def _coerce_candidate_metadata(candidate: Any) -> dict[str, Any]:
    """转换`candidate_metadata`格式。"""
    metadata = getattr(candidate, "metadata", None)
    if metadata is None:
        metadata = {}
        setattr(candidate, "metadata", metadata)
    return metadata


def mark_candidate_unavailable(candidate: Any, *, reason: str) -> None:
    """把候选标记为当前不可用，供序列化和本地 fallback 共用。"""
    reason_text = str(reason or "").strip()
    if not reason_text:
        return
    metadata = _coerce_candidate_metadata(candidate)
    metadata["available"] = False
    metadata["unavailable_reason"] = reason_text


def _resolve_card_from_clip(app: "AppProcessor", box: Any) -> CandidateResolution | None:
    """解析并确定`card_from_clip`。"""
    clip_manager = getattr(app, "clip_manager", None)
    if clip_manager is None or box is None or getattr(box, "frame", None) is None:
        return None
    skill_card_clip = getattr(clip_manager, "skill_card_clip", None)
    if skill_card_clip is None:
        return None
    matched = None
    similarity_thresholds = (0.98, 0.94)
    try:
        for threshold in similarity_thresholds:
            matched = skill_card_clip.retrieve(box.frame, similarity_threshold=threshold)
            if matched is not None:
                if threshold < similarity_thresholds[0]:
                    logger.info(
                        "producer decision: 技能卡 CLIP 通过放宽阈值命中 threshold={} card_id={} name={!r}",
                        threshold,
                        getattr(matched, "id", ""),
                        getattr(matched, "name", ""),
                    )
                break
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"producer decision: 技能卡 CLIP 识别失败，回退 OCR: {exc}")
        return None
    if matched is None:
        return None

    card_id = str(getattr(matched, "id", "") or "")
    upgrade_count = int(getattr(matched, "upgradeCount", 0) or 0)
    metadata = _enrich_card_metadata(card_id, upgrade_count=upgrade_count)
    display_name = metadata.get("raw_name") or getattr(matched, "name", "") or ""
    return CandidateResolution(
        action_id=f"produce_card:{card_id}:{upgrade_count}",
        candidate_type="produce_card",
        db_id=card_id,
        display_name=str(display_name),
        source="clip",
        confidence=1.0,
        metadata=metadata,
    )


def _resolve_drink_from_clip(app: "AppProcessor", box: Any) -> CandidateResolution | None:
    """解析并确定`drink_from_clip`。"""
    clip_manager = getattr(app, "clip_manager", None)
    if clip_manager is None or box is None or getattr(box, "frame", None) is None:
        return None
    produce_drink_clip = getattr(clip_manager, "produce_drink_clip", None)
    if produce_drink_clip is None:
        return None
    try:
        matched = produce_drink_clip.retrieve(box.frame)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"producer decision: P饮料 CLIP 识别失败，回退 OCR: {exc}")
        return None
    if matched is None:
        return None

    drink_id = str(getattr(matched, "id", "") or "")
    metadata = _enrich_drink_metadata(drink_id)
    display_name = metadata.get("raw_name") or getattr(matched, "name", "") or ""
    return CandidateResolution(
        action_id=f"produce_drink:{drink_id}",
        candidate_type="produce_drink",
        db_id=drink_id,
        display_name=str(display_name),
        source="clip",
        confidence=1.0,
        metadata=metadata,
    )


def _resolve_item_from_clip(app: "AppProcessor", box: Any) -> CandidateResolution | None:
    """解析并确定`item_from_clip`。"""
    clip_manager = getattr(app, "clip_manager", None)
    if clip_manager is None or box is None or getattr(box, "frame", None) is None:
        return None
    produce_item_clip = getattr(clip_manager, "produce_item_clip", None)
    if produce_item_clip is None:
        return None
    try:
        matched = produce_item_clip.retrieve(box.frame)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"producer decision: P物品 CLIP 识别失败，回退 OCR: {exc}")
        return None
    if matched is None:
        return None

    item_id = str(getattr(matched, "id", "") or "")
    metadata = _enrich_item_metadata(item_id)
    display_name = metadata.get("raw_name") or getattr(matched, "name", "") or ""
    return CandidateResolution(
        action_id=f"produce_item:{item_id}",
        candidate_type="produce_item",
        db_id=item_id,
        display_name=str(display_name),
        source="clip",
        confidence=1.0,
        metadata=metadata,
    )


def resolve_produce_card_identity(
    app: "AppProcessor",
    *,
    title: str,
    box: Any,
    index: int,
) -> CandidateResolution:
    """解析并补全produce、卡牌、标识并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        title: 用于提供title相关输入。
        box: 单个检测框对象。
        index: 用于提供index相关输入。

    Returns:
        CandidateResolution: 返回值类型见注解。
    """
    clip_resolution = _resolve_card_from_clip(app, box)
    if clip_resolution is not None:
        return clip_resolution

    matched = _match_catalog_entry(title, expected_kind="produce_card")
    if matched is not None:
        card_id = str(matched["id"])
        metadata = _enrich_card_metadata(card_id, upgrade_count=0)
        display_name = metadata.get("raw_name") or str(matched.get("raw_name") or matched.get("name") or "")
        _learn_card_clip_from_db_id(app, getattr(box, "frame", None), card_id, upgrade_count=0)
        return CandidateResolution(
            action_id=f"produce_card:{card_id}:0",
            candidate_type="produce_card",
            db_id=card_id,
            display_name=str(display_name),
            source="ocr",
            confidence=float(matched.get("score") or 0.0) / 100.0,
            metadata=metadata,
        )

    return CandidateResolution(
        action_id=_build_unknown_action_id("produce_card_unknown", title, index=index),
        candidate_type="produce_card",
        display_name=title,
        source="unresolved",
        confidence=0.0,
        metadata={"unresolved": True},
    )


def resolve_produce_drink_identity(
    title: str,
    *,
    app: "AppProcessor | None" = None,
    box: Any = None,
    index: int,
    allow_ocr_fallback: bool = True,
    min_ocr_confidence: float = 0.0,
) -> CandidateResolution:
    """解析并补全produce、饮料、标识并返回结果。

    Args:
        title: 用于提供title相关输入。
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        box: 单个检测框对象。
        index: 用于提供index相关输入。
        allow_ocr_fallback: 用于提供allow、OCR、fallback相关输入。
        min_ocr_confidence: 用于提供min、OCR、confidence相关输入。

    Returns:
        CandidateResolution: 返回值类型见注解。
    """
    clip_resolution = _resolve_drink_from_clip(app, box) if app is not None else None
    if clip_resolution is not None:
        return clip_resolution

    if not allow_ocr_fallback:
        return CandidateResolution(
            action_id=_build_unknown_action_id("produce_drink_unknown", title, index=index),
            candidate_type="produce_drink",
            display_name=title,
            source="unresolved",
            confidence=0.0,
            metadata={"unresolved": True},
        )

    matched = _match_catalog_entry(title, expected_kind="produce_drink")
    if matched is not None:
        drink_id = str(matched["id"])
        matched_confidence = float(matched.get("score") or 0.0) / 100.0
        if matched_confidence < float(min_ocr_confidence or 0.0):
            return CandidateResolution(
                action_id=_build_unknown_action_id("produce_drink_unknown", title, index=index),
                candidate_type="produce_drink",
                display_name=title,
                source="unresolved",
                confidence=0.0,
                metadata={"unresolved": True},
            )
        metadata = _enrich_drink_metadata(drink_id)
        display_name = metadata.get("raw_name") or str(matched.get("raw_name") or matched.get("name") or "")
        if app is not None and box is not None:
            _learn_drink_clip_from_db_id(app, getattr(box, "frame", None), drink_id)
        return CandidateResolution(
            action_id=f"produce_drink:{drink_id}",
            candidate_type="produce_drink",
            db_id=drink_id,
            display_name=str(display_name),
            source="ocr",
            confidence=matched_confidence,
            metadata=metadata,
        )

    return CandidateResolution(
        action_id=_build_unknown_action_id("produce_drink_unknown", title, index=index),
        candidate_type="produce_drink",
        display_name=title,
        source="unresolved",
        confidence=0.0,
        metadata={"unresolved": True},
    )


def resolve_produce_item_identity(
    title: str,
    *,
    app: "AppProcessor | None" = None,
    box: Any = None,
    index: int,
    lookup_texts: Sequence[str] | None = None,
) -> CandidateResolution:
    """解析并补全produce、道具、标识并返回结果。

    Args:
        title: 用于提供title相关输入。
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        box: 单个检测框对象。
        index: 用于提供index相关输入。
        lookup_texts: 用于提供lookup、texts相关输入。

    Returns:
        CandidateResolution: 返回值类型见注解。
    """
    clip_resolution = _resolve_item_from_clip(app, box) if app is not None else None
    if clip_resolution is not None:
        return clip_resolution

    match_inputs = [title, *(lookup_texts or ())]
    matched = (
        _match_catalog_entry_from_texts(match_inputs, expected_kind="produce_item")
        if lookup_texts
        else _match_catalog_entry(title, expected_kind="produce_item")
    )
    if matched is not None:
        item_id = str(matched["id"])
        metadata = {
            **_enrich_item_metadata(item_id),
            "matched_text": str(matched.get("matched_text") or ""),
        }
        display_name = (
            metadata.get("raw_name")
            or str(matched.get("raw_name") or matched.get("name") or "")
            or ""
        )
        if app is not None and box is not None:
            _learn_item_clip_from_db_id(app, getattr(box, "frame", None), item_id)
        return CandidateResolution(
            action_id=f"produce_item:{item_id}",
            candidate_type="produce_item",
            db_id=item_id,
            display_name=str(display_name),
            source="ocr",
            confidence=float(matched.get("score") or 0.0) / 100.0,
            metadata=metadata,
        )

    return CandidateResolution(
        action_id=_build_unknown_action_id("produce_item_unknown", title, index=index),
        candidate_type="produce_item",
        display_name=title or next((text for text in (lookup_texts or ()) if text), ""),
        source="unresolved",
        confidence=0.0,
        metadata={
            "unresolved": True,
            "lookup_texts": [str(text) for text in (lookup_texts or ()) if str(text or "").strip()],
        },
    )


def resolve_produce_entity_identity(
    title: str,
    *,
    app: "AppProcessor | None" = None,
    box: Any = None,
    index: int,
    icon_box: Any = None,
    entity_type_hint: str = "",
) -> CandidateResolution:
    # 1. OCR 文本匹配
    """解析并补全produce、entity、标识并返回结果。

    Args:
        title: 用于提供title相关输入。
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        box: 单个检测框对象。
        index: 用于提供index相关输入。
        icon_box: 用于提供icon、box相关输入。
        entity_type_hint: 用于提供entity、type、提示相关输入。

    Returns:
        CandidateResolution: 返回值类型见注解。
    """
    matched = _match_catalog_entry(title) if title.strip() else None
    if matched is not None:
        kind = str(matched.get("kind") or "")
        if kind == "produce_card":
            if app is None:
                return CandidateResolution(
                    action_id=_build_unknown_action_id("produce_card_unknown", title, index=index),
                    candidate_type="produce_card",
                    display_name=title,
                    source="unresolved",
                    confidence=0.0,
                    metadata={"unresolved": True},
                )
            return resolve_produce_card_identity(app, title=title, box=box, index=index)
        if kind == "produce_drink":
            return resolve_produce_drink_identity(title, app=app, box=box, index=index)
        if kind == "produce_item":
            return resolve_produce_item_identity(title, app=app, box=box, index=index)

    # 2. 尝试 CLIP 视觉识别（仅标准阈值，宁可不识别也不能误识别）
    if app is not None:
        # 选择最佳 CLIP 输入图像：优先使用内层图标框（更干净），否则用整个 box
        clip_box = icon_box if icon_box is not None else box

        if clip_box is not None:
            # 根据类型提示优先尝试对应的 CLIP 服务（标准阈值）
            if entity_type_hint == "produce_drink":
                result = _resolve_drink_from_clip(app, clip_box)
                if result is not None:
                    return result
            elif entity_type_hint == "produce_card":
                result = _resolve_card_from_clip(app, clip_box)
                if result is not None:
                    return result

            # 无提示或提示的 CLIP 未命中 → 按优先级尝试所有 CLIP 服务（标准阈值）
            for resolver in (_resolve_drink_from_clip, _resolve_card_from_clip, _resolve_item_from_clip):
                result = resolver(app, clip_box)
                if result is not None:
                    return result

        # 3. 所有识别均失败 → 自动采集未识别图像供后续人工标注
        collect_box = clip_box if clip_box is not None else box
        if collect_box is not None:
            _auto_collect_unresolved_entity_image(collect_box, index)

    return CandidateResolution(
        action_id=_build_unknown_action_id("produce_entity_unknown", title, index=index),
        candidate_type="produce_entity",
        display_name=title,
        source="unresolved",
        confidence=0.0,
        metadata={"unresolved": True},
    )


def hydrate_card_candidates(
    app: "AppProcessor",
    candidates: Sequence[Any],
) -> None:
    """处理hydrate、卡牌、candidates并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    for candidate in candidates:
        resolution = resolve_produce_card_identity(
            app,
            title=getattr(candidate, "title", ""),
            box=getattr(candidate, "box", None),
            index=getattr(candidate, "index", 0),
        )
        _apply_resolution(candidate, resolution)


def hydrate_p_drink_candidates(app: "AppProcessor", candidates: Sequence[Any]) -> None:
    """处理hydrate、p、饮料、candidates并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    for candidate in candidates:
        resolution = resolve_produce_drink_identity(
            getattr(candidate, "title", ""),
            app=app,
            box=getattr(candidate, "box", None),
            index=getattr(candidate, "index", 0),
        )
        _apply_resolution(candidate, resolution)


def hydrate_consult_candidates(
    app: "AppProcessor",
    candidates: Sequence[Any],
) -> None:
    """处理hydrate、consult、candidates并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    for candidate in candidates:
        kind = getattr(candidate, "kind", "")
        title = getattr(candidate, "title", "")
        index = getattr(candidate, "index", 0)
        if kind in {"enhancement_target", "remove_target"}:
            resolution = resolve_produce_card_identity(app, title=title, box=getattr(candidate, "box", None), index=index)
            consult_action = (
                "consult_select_remove_target"
                if kind == "remove_target"
                else "consult_select_enhancement_target"
            )
            resolution = CandidateResolution(
                action_id=f"{consult_action}:{resolution.db_id or index}",
                candidate_type="consult_action",
                db_id=resolution.db_id,
                display_name=resolution.display_name or title,
                source=resolution.source,
                confidence=resolution.confidence,
                metadata={
                    **resolution.metadata,
                    "consult_action": consult_action,
                },
            )
        elif kind == "enhance":
            resolution = CandidateResolution(
                action_id="consult_open_enhancement",
                candidate_type="consult_action",
                display_name=title or GeneralText.ENHANCE,
                source="yolo",
                confidence=1.0,
                metadata={"consult_action": "consult_open_enhancement"},
            )
        elif kind == "delete":
            resolution = CandidateResolution(
                action_id="consult_open_remove",
                candidate_type="consult_action",
                display_name=title or ProduceText.SKILL_CARD_REMOVE,
                source="yolo",
                confidence=1.0,
                metadata={"consult_action": "consult_open_remove"},
            )
        elif kind == "confirm_enhancement":
            resolution = CandidateResolution(
                action_id="consult_confirm_enhancement",
                candidate_type="consult_action",
                display_name=title or ProduceText.ENHANCE_CONFIRM,
                source="yolo",
                confidence=1.0,
                metadata={"consult_action": "consult_confirm_enhancement"},
            )
        elif kind == "confirm_remove":
            resolution = CandidateResolution(
                action_id="consult_confirm_remove",
                candidate_type="consult_action",
                display_name=title or ProduceText.SKILL_CARD_REMOVE,
                source="yolo",
                confidence=1.0,
                metadata={"consult_action": "consult_confirm_remove"},
            )
        elif kind == "exit":
            resolution = CandidateResolution(
                action_id="consult_exit",
                candidate_type="consult_action",
                display_name=title or ButtonText.EXIT,
                source="ocr",
                confidence=0.9,
                metadata={"consult_action": "consult_exit"},
            )
        else:
            entry_resolution = resolve_produce_entity_identity(
                title,
                app=app,
                box=getattr(candidate, "box", None),
                index=index,
                icon_box=getattr(candidate, "icon_box", None),
                entity_type_hint=getattr(candidate, "entity_type_hint", ""),
            )
            consult_action = GameplayPosition.CONSULT_EXCHANGE
            if entry_resolution.candidate_type == "produce_drink":
                consult_action = "consult_exchange_drink"
            elif entry_resolution.candidate_type == "produce_card":
                consult_action = "consult_exchange_card"
            elif entry_resolution.candidate_type == "produce_item":
                consult_action = "consult_exchange_item"
            # 将价格信息附加到元数据
            price = (getattr(candidate, "metadata", {}) or {}).get("price", "")
            resolution = CandidateResolution(
                action_id=f"{consult_action}:{entry_resolution.db_id or index}",
                candidate_type="consult_action",
                db_id=entry_resolution.db_id,
                display_name=entry_resolution.display_name or title,
                source=entry_resolution.source or "clip",
                confidence=entry_resolution.confidence,
                metadata={
                    **entry_resolution.metadata,
                    "consult_action": consult_action,
                    **({"price": price} if price else {}),
                },
            )
        _apply_resolution(candidate, resolution)


def _looks_like_visually_disabled_card(box: Any) -> bool:
    """通过 HSV 阈值识别“当前无法打出”的禁用态卡牌。"""
    frame = getattr(box, "frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return False
    if len(frame.shape) < 3:
        return False
    height, width = frame.shape[:2]
    if height < 24 or width < 24:
        return False

    crop = frame[
        int(height * 0.14):int(height * 0.88),
        int(width * 0.08):int(width * 0.92),
    ]
    if crop.size <= 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(_VISUAL_DISABLED_LOWER_HSV, dtype=np.uint8),
        np.array(_VISUAL_DISABLED_UPPER_HSV, dtype=np.uint8),
    )
    # 轻量去噪，提升 JPG 压缩噪点下的稳定性。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    disabled_ratio = float(np.mean(mask > 0))
    return disabled_ratio >= _VISUAL_DISABLED_MASK_RATIO


def _annotate_candidates(app: "AppProcessor", *, phase: str, candidates: Sequence[Any]) -> None:
    """补充标注候选项并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        phase: 当前 gameplay 阶段标识。
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    debugger = getattr(app, "debug_tools", None) or DebugTools()
    phase_color = {
        GameplayPhase.SCHEDULE: (255, 215, 0),
        GameplayPhase.DIALOGUE: (0, 180, 255),
        GameplayPhase.LESSON: (0, 220, 120),
        GameplayPhase.EXAM: (255, 120, 0),
        GameplayPhase.SKILL_REWARD: (160, 120, 255),
        GameplayPhase.P_DRINK: (255, 0, 160),
        GameplayPhase.CONSULT: (255, 80, 80),
    }.get(phase, (200, 200, 200))
    for candidate in candidates:
        box = getattr(candidate, "box", None)
        coords = _serialize_box(box)
        if coords is None:
            continue
        metadata = _coerce_candidate_metadata(candidate)
        if (
            phase in {GameplayPhase.LESSON, GameplayPhase.EXAM}
            and is_produce_card_action_id(getattr(candidate, "action_id", ""))
            and _looks_like_visually_disabled_card(box)
        ):
            mark_candidate_unavailable(
                candidate,
                reason="卡面呈现灰色禁用蒙版，当前条件下无法打出",
            )
            metadata = _coerce_candidate_metadata(candidate)
        label_core = getattr(candidate, "db_id", "") or getattr(candidate, "action_id", "") or getattr(candidate, "title", "") or getattr(candidate, "kind", "")
        unavailable_reason = str(metadata.get("unavailable_reason") or "").strip()
        debugger.add_box(
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            label=(
                f"{phase}:{getattr(candidate, 'index', 0)} {str(label_core)[:24]}"
                f"{' [不可用]' if unavailable_reason else ''}"
            ),
            color=(255, 80, 80) if unavailable_reason else phase_color,
            alpha=0.15,
            duration=3.0,
            font_size=18,
        )


def _extract_hud_state(app: "AppProcessor") -> dict[str, Any]:
    """提取HUD、状态并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        dict: 结构化结果字典。
    """
    ctx = getattr(app, "_produce_decision_ctx", None)
    results = getattr(app, "latest_results", None)
    if results is None:
        return {
            "stamina": 0,
            "max_stamina": 0,
            "stamina_observed": False,
            "genki": 0,
            "genki_observed": False,
            "p_point": 0,
            "p_point_observed": False,
            "target_score": 0,
            "target_score_observed": False,
            "score": 0,
            "score_observed": False,
            "remaining_turns": 0,
            "remaining_turns_observed": False,
            "turn_color": "",
            "score_bonus": "",
            "exam_ranking": "",
            "vocal": None,
            "vocal_observed": False,
            "dance": None,
            "dance_observed": False,
            "visual": None,
            "visual_observed": False,
            "has_progress_hud": False,
            "recommend_action_text": "",
            "recommend_action_kind": "",
        }

    def _ocr_first(label: str) -> str:
        """处理OCR、first并返回结果。

        Args:
            label: 用于提供label相关输入。

        Returns:
            str: 处理后的文本结果。
        """
        boxes = results.filter_by_label(label)
        if not boxes:
            return ""
        return ocr_text(boxes.first().frame)

    def _ocr_region(x1_ratio: float, y1_ratio: float, x2_ratio: float, y2_ratio: float) -> str:
        """处理OCR、region并返回结果。

        Args:
            x1_ratio: 用于提供x1、ratio相关输入。
            y1_ratio: 用于提供y1、ratio相关输入。
            x2_ratio: 用于提供x2、ratio相关输入。
            y2_ratio: 用于提供y2、ratio相关输入。

        Returns:
            str: 处理后的文本结果。
        """
        frame = getattr(app, "latest_frame", None)
        if frame is None or frame.size == 0:
            return ""
        h, w = frame.shape[:2]
        x1 = int(w * x1_ratio)
        y1 = int(h * y1_ratio)
        x2 = int(w * x2_ratio)
        y2 = int(h * y2_ratio)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        return ocr_text(crop)

    debugger = getattr(app, "debug_tools", None) or DebugTools()

    def _ocr_box_region(
        box,
        *,
        x1_ratio: float,
        y1_ratio: float,
        x2_ratio: float,
        y2_ratio: float,
        debug_label: str = "",
    ) -> str:
        """处理OCR、检测框、region并返回结果。

        Args:
            box: 单个检测框对象。
            x1_ratio: 用于提供x1、ratio相关输入。
            y1_ratio: 用于提供y1、ratio相关输入。
            x2_ratio: 用于提供x2、ratio相关输入。
            y2_ratio: 用于提供y2、ratio相关输入。
            debug_label: 用于提供debug、label相关输入。

        Returns:
            str: 处理后的文本结果。
        """
        frame = getattr(box, "frame", None)
        if frame is None or frame.size == 0:
            return ""
        h, w = frame.shape[:2]
        x1 = int(w * x1_ratio)
        y1 = int(h * y1_ratio)
        x2 = int(w * x2_ratio)
        y2 = int(h * y2_ratio)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        if debug_label:
            box_x = int(getattr(box, "x", 0))
            box_y = int(getattr(box, "y", 0))
            debugger.add_box(
                box_x + x1,
                box_y + y1,
                box_x + x2,
                box_y + y2,
                label=debug_label,
                color=(80, 220, 120),
                alpha=0.15,
                duration=2.5,
                font_size=16,
            )
        return ocr_text(crop)

    def _ocr_box_text_right_of_color_anchor(
        box,
        *,
        lower_color: tuple[int, int, int],
        upper_color: tuple[int, int, int],
        search_y1_ratio: float,
        search_y2_ratio: float,
        min_area_ratio: float,
        min_aspect: float,
        max_aspect: float,
        x_padding: int,
        y_padding: int,
        min_x1_ratio: float,
        debug_label: str = "",
    ) -> str:
        """处理OCR、检测框、文本、right、of、color、anchor并返回结果。

        Args:
            box: 单个检测框对象。
            lower_color: 用于提供lower、color相关输入。
            upper_color: 用于提供upper、color相关输入。
            search_y1_ratio: 用于提供search、y1、ratio相关输入。
            search_y2_ratio: 用于提供search、y2、ratio相关输入。
            min_area_ratio: 用于提供min、area、ratio相关输入。
            min_aspect: 用于提供min、aspect相关输入。
            max_aspect: 用于提供max、aspect相关输入。
            x_padding: 用于提供x、padding相关输入。
            y_padding: 用于提供y、padding相关输入。
            min_x1_ratio: 用于提供min、x1、ratio相关输入。
            debug_label: 用于提供debug、label相关输入。

        Returns:
            str: 处理后的文本结果。
        """
        frame = getattr(box, "frame", None)
        if frame is None or frame.size == 0:
            return ""
        h, w = frame.shape[:2]
        search_y1 = int(h * search_y1_ratio)
        search_y2 = int(h * search_y2_ratio)
        search_crop = frame[search_y1:search_y2, :]
        if search_crop.size == 0:
            return ""
        hsv = cv2.cvtColor(search_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(lower_color, dtype=np.uint8),
            np.array(upper_color, dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = float(mask.shape[0] * mask.shape[1]) * min_area_ratio
        anchor_box: tuple[int, int, int, int] | None = None
        anchor_area = 0
        for label_idx in range(1, num_labels):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            component_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            component_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_area or component_w <= 0 or component_h <= 0:
                continue
            aspect = component_w / max(1, component_h)
            if aspect < min_aspect or aspect > max_aspect:
                continue
            if area >= anchor_area:
                anchor_box = (x, y, component_w, component_h)
                anchor_area = area
        if anchor_box is None:
            return ""
        anchor_x, anchor_y, anchor_w, anchor_h = anchor_box
        x1 = min(
            w - 1,
            max(int(w * min_x1_ratio), anchor_x + anchor_w + x_padding),
        )
        y1 = max(0, search_y1 + anchor_y - y_padding)
        y2 = min(h, search_y1 + anchor_y + anchor_h + y_padding)
        crop = frame[y1:y2, x1:w]
        # 有些截图里锚点已经贴近最右侧，右侧裁切会过窄，此时改为锚点内右半区 OCR。
        min_text_width = max(8, int(w * 0.04))
        if crop.size == 0 or crop.shape[1] < min_text_width:
            fallback_x1 = max(0, anchor_x + int(anchor_w * 0.45))
            fallback_x2 = min(w, anchor_x + anchor_w + max(x_padding, 2))
            fallback_y1 = max(0, search_y1 + anchor_y - y_padding)
            fallback_y2 = min(h, search_y1 + anchor_y + anchor_h + y_padding)
            fallback = frame[fallback_y1:fallback_y2, fallback_x1:fallback_x2]
            if fallback.size == 0:
                return ""
            if debug_label:
                box_x = int(getattr(box, "x", 0))
                box_y = int(getattr(box, "y", 0))
                debugger.add_box(
                    box_x + fallback_x1,
                    box_y + fallback_y1,
                    box_x + fallback_x2,
                    box_y + fallback_y2,
                    label=f"{debug_label}_tight",
                    color=(120, 180, 255),
                    alpha=0.18,
                    duration=2.5,
                    font_size=16,
                )
            return ocr_text(fallback)
        if debug_label:
            box_x = int(getattr(box, "x", 0))
            box_y = int(getattr(box, "y", 0))
            debugger.add_box(
                box_x + anchor_x,
                box_y + search_y1 + anchor_y,
                box_x + anchor_x + anchor_w,
                box_y + search_y1 + anchor_y + anchor_h,
                label=f"{debug_label}_anchor",
                color=(255, 200, 0),
                alpha=0.12,
                duration=2.5,
                font_size=16,
            )
            debugger.add_box(
                box_x + x1,
                box_y + y1,
                box_x + w,
                box_y + y2,
                label=debug_label,
                color=(80, 220, 120),
                alpha=0.15,
                duration=2.5,
                font_size=16,
            )
        return ocr_text(crop)

    battle_like_hud = any(
        results.exists_label(label)
        for label in (
            ProducerLabels.PC_BONUS_INDICATOR,
            ProducerLabels.PC_TRAINING_SCORE,
            ProducerLabels.SKILL_CARD_ACTIVE,
            ProducerLabels.SKILL_CARD_MENTAL,
            ProducerLabels.SKILL_CARD_TRAP,
        )
    )

    stamina_text = _ocr_first(ProducerLabels.PC_STAMINA)
    previous_stamina = int(getattr(app, "_last_produce_hud_stamina", 0) or 0)
    previous_max_stamina = int(getattr(app, "_last_produce_hud_max_stamina", 0) or 0)
    previous_genki = int(getattr(app, "_last_produce_hud_genki", 0) or 0)
    stamina_value = 0
    max_stamina_value = 0
    genki_value = 0
    stamina_observed = False
    genki_observed = False
    stamina_boxes = results.filter_by_label(ProducerLabels.PC_STAMINA)
    if battle_like_hud and stamina_boxes:
        stamina_box = stamina_boxes.first()
        # 盾标位于右上区域，右侧数字是元气值。
        genki_text_shield = _ocr_box_text_right_of_color_anchor(
            stamina_box,
            lower_color=_HUD_GENKI_SHIELD_LOWER_HSV,
            upper_color=_HUD_GENKI_SHIELD_UPPER_HSV,
            search_y1_ratio=0.00,
            search_y2_ratio=0.62,
            min_area_ratio=0.006,
            min_aspect=0.55,
            max_aspect=2.40,
            x_padding=4,
            y_padding=4,
            min_x1_ratio=0.56,
            debug_label="pc_genki_shield_hsv",
        )
        # 体力条同样在上半区，可作为元气行数字的辅助锚点。
        genki_text_bar = _ocr_box_text_right_of_color_anchor(
            stamina_box,
            lower_color=_HUD_STAMINA_BAR_LOWER_HSV,
            upper_color=_HUD_STAMINA_BAR_UPPER_HSV,
            search_y1_ratio=0.00,
            search_y2_ratio=0.52,
            min_area_ratio=0.015,
            min_aspect=2.0,
            max_aspect=20.0,
            x_padding=4,
            y_padding=6,
            min_x1_ratio=0.46,
            debug_label="pc_genki_bar_hsv",
        )
        genki_text_color = _ocr_box_text_right_of_color_anchor(
            stamina_box,
            lower_color=(40, 80, 80),
            upper_color=(110, 255, 255),
            search_y1_ratio=0.00,
            search_y2_ratio=0.55,
            min_area_ratio=0.03,
            min_aspect=2.0,
            max_aspect=20.0,
            x_padding=4,
            y_padding=6,
            min_x1_ratio=0.50,
            debug_label="pc_genki_color",
        )
        # 绿心固定在体力条下方，右侧数字是体力值。
        stamina_text_heart = _ocr_box_text_right_of_color_anchor(
            stamina_box,
            lower_color=_HUD_STAMINA_HEART_LOWER_HSV,
            upper_color=_HUD_STAMINA_HEART_UPPER_HSV,
            search_y1_ratio=0.34,
            search_y2_ratio=1.00,
            min_area_ratio=0.004,
            min_aspect=0.50,
            max_aspect=2.50,
            x_padding=4,
            y_padding=6,
            min_x1_ratio=0.28,
            debug_label="pc_stamina_heart_hsv",
        )
        genki_text = _ocr_box_region(
            stamina_box,
            x1_ratio=0.54,
            y1_ratio=0.02,
            x2_ratio=0.98,
            y2_ratio=0.48,
            debug_label="pc_genki_hud",
        )
        genki_text_alt = _ocr_box_region(
            stamina_box,
            x1_ratio=0.42,
            y1_ratio=0.08,
            x2_ratio=0.98,
            y2_ratio=0.56,
            debug_label="pc_genki_hud_alt",
        )
        # 右上角圆形资源徽标（常见为元気）有时不会被上面两块裁切覆盖，单独补一块读取。
        genki_badge_text = _ocr_box_region(
            stamina_box,
            x1_ratio=0.70,
            y1_ratio=0.00,
            x2_ratio=1.00,
            y2_ratio=0.70,
            debug_label="pc_genki_badge",
        )
        stamina_text_color = _ocr_box_text_right_of_color_anchor(
            stamina_box,
            lower_color=(30, 80, 80),
            upper_color=(85, 255, 255),
            search_y1_ratio=0.45,
            search_y2_ratio=1.00,
            min_area_ratio=0.02,
            min_aspect=0.5,
            max_aspect=2.0,
            x_padding=4,
            y_padding=6,
            min_x1_ratio=0.38,
            debug_label="pc_stamina_color",
        )
        stamina_lower_text = _ocr_box_region(
            stamina_box,
            x1_ratio=0.42,
            y1_ratio=0.40,
            x2_ratio=0.98,
            y2_ratio=0.98,
            debug_label="pc_stamina_hud",
        )
        stamina_lower_text_alt = _ocr_box_region(
            stamina_box,
            x1_ratio=0.48,
            y1_ratio=0.46,
            x2_ratio=0.98,
            y2_ratio=0.98,
            debug_label="pc_stamina_hud_alt",
        )
        stamina_lower_text_legacy = _ocr_box_region(
            stamina_box,
            x1_ratio=0.54,
            y1_ratio=0.48,
            x2_ratio=0.98,
            y2_ratio=0.98,
            debug_label="pc_stamina_hud_legacy",
        )
        genki_value, genki_has_digits = _extract_noisy_hud_value(
            genki_text_shield,
            genki_text_bar,
            genki_text_color,
            genki_text,
            genki_text_alt,
            genki_badge_text,
            previous_value=previous_genki,
            upper_bound=999,
        )
        genki_value = _resolve_repeated_digit_ocr_value(
            genki_value,
            genki_text_shield,
            genki_text_bar,
            genki_text_color,
            genki_text,
            genki_text_alt,
            genki_badge_text,
            previous_value=previous_genki,
        )
        stamina_value, stamina_has_digits = _extract_noisy_hud_value(
            stamina_text_heart,
            stamina_text_color,
            stamina_lower_text,
            stamina_lower_text_alt,
            stamina_lower_text_legacy,
            previous_value=previous_stamina,
            upper_bound=previous_max_stamina or 99,
        )
        stamina_candidates = (
            stamina_text,
            stamina_text_color,
            stamina_lower_text,
            stamina_lower_text_alt,
            stamina_lower_text_legacy,
        )
        parsed_stamina = 0
        parsed_max_stamina = 0
        for candidate_text in stamina_candidates:
            parsed_current, parsed_max = _parse_stamina_text(
                candidate_text,
                previous_stamina=previous_stamina,
                previous_max_stamina=previous_max_stamina,
            )
            if parsed_current > 0 and parsed_stamina <= 0:
                parsed_stamina = parsed_current
            if parsed_max > 0:
                parsed_max_stamina = parsed_max
                break
        if not stamina_has_digits and previous_stamina > 0:
            stamina_value = previous_stamina
        elif stamina_value <= 0 and parsed_stamina > 0:
            stamina_value = parsed_stamina
        if not genki_has_digits and previous_genki > 0:
            genki_value = previous_genki
        stamina_observed = any(
            str(text or "").strip()
            for text in (
                stamina_text_heart,
                stamina_lower_text,
                stamina_lower_text_alt,
                stamina_lower_text_legacy,
                stamina_text_color,
                stamina_text,
            )
        )
        genki_observed = any(
            str(text or "").strip()
            for text in (
                genki_text_shield,
                genki_text_bar,
                genki_text_color,
                genki_text,
                genki_text_alt,
                genki_badge_text,
            )
        )
        max_stamina_value = parsed_max_stamina or previous_max_stamina
        if max_stamina_value <= 0 and stamina_value > 0:
            # lesson 画面常只显示当前体力，不显示上限；避免出现 34/0 这种无意义状态。
            max_stamina_value = stamina_value
        if max_stamina_value > 0 and stamina_value > max_stamina_value:
            max_stamina_value = stamina_value
    else:
        stamina_value, max_stamina_value = _parse_stamina_text(
            stamina_text,
            previous_stamina=previous_stamina,
            previous_max_stamina=previous_max_stamina,
        )
        stamina_observed = bool(str(stamina_text or "").strip())
        genki_value = previous_genki
        genki_observed = False
    if stamina_observed:
        setattr(app, "_last_produce_hud_stamina", stamina_value)
        if max_stamina_value > 0:
            setattr(app, "_last_produce_hud_max_stamina", max_stamina_value)
    if genki_observed:
        setattr(app, "_last_produce_hud_genki", genki_value)

    bonus_text = _ocr_first(ProducerLabels.PC_BONUS_INDICATOR)
    recommend_text = _ocr_first(ProducerLabels.PC_RECOMMEND_ACTION)
    target_text = _ocr_first(ProducerLabels.PC_TARGET)
    score_text = _ocr_first(ProducerLabels.PC_TRAINING_SCORE)
    remaining_turns_text = _ocr_first(ProducerLabels.PC_TRAINING_REMAINING)
    turn_color = ""
    for token, display in (
        ("Vo", ProduceText.VOCAL),
        ("Da", ProduceText.DANCE),
        ("Vi", ProduceText.VISUAL),
        (ProduceText.VOCAL, ProduceText.VOCAL),
        (ProduceText.DANCE, ProduceText.DANCE),
        (ProduceText.VISUAL, ProduceText.VISUAL),
    ):
        if token and token in bonus_text:
            turn_color = display
            break

    score_bonus = ""
    if bonus_text:
        score_bonus = _parse_score_bonus_from_bonus_text(
            bonus_text,
            remaining_turns_text=remaining_turns_text,
        )

    # 排名从上下文读取（由 exam.py 每回合提取并存入）
    exam_ranking_str = get_exam_ranking_value(ctx) if ctx else ""
    p_point_text = _ocr_first(ProducerLabels.PC_P_POINT)
    parameter_upper_bound = int(getattr(ctx, "parameter_growth_limit", 0) or 0)
    vocal_value, vocal_observed = _extract_planning_parameter_value(
        _ocr_first(ProducerLabels.PARAM_VOCAL),
        previous_value=_get_parameter_seed_value(ctx, "vocal"),
        upper_bound=parameter_upper_bound,
    )
    dance_value, dance_observed = _extract_planning_parameter_value(
        _ocr_first(ProducerLabels.PARAM_DANCE),
        previous_value=_get_parameter_seed_value(ctx, "dance"),
        upper_bound=parameter_upper_bound,
    )
    visual_value, visual_observed = _extract_planning_parameter_value(
        _ocr_first(ProducerLabels.PARAM_VISUAL),
        previous_value=_get_parameter_seed_value(ctx, "visual"),
        upper_bound=parameter_upper_bound,
    )

    # ── 解析进度圆圈（课程画面） ──
    # score_text 可能形如“PERFECTまで175CLEAR”或“CLEARまで10”。
    progress_info = _parse_progress_circle(score_text)
    if progress_info is not None:
        # 进度圆圈模式: 数字是"距离目标的剩余分数"，不是当前累计分数
        score_value = 0
        score_observed = False
    else:
        # 普通模式: 数字就是当前分数（日程/考核画面）
        score_value = _extract_first_int(score_text)
        score_observed = bool(str(score_text or "").strip())

    return {
        "stamina": stamina_value,
        "max_stamina": max_stamina_value,
        "stamina_observed": stamina_observed,
        "genki": genki_value,
        "genki_observed": genki_observed,
        "p_point": _extract_first_int(p_point_text),
        "p_point_observed": bool(str(p_point_text or "").strip()),
        "target_score": _extract_first_int(target_text),
        "target_score_observed": bool(str(target_text or "").strip()),
        "score": score_value,
        "score_observed": score_observed,
        "remaining_turns": _extract_first_int(remaining_turns_text),
        "remaining_turns_observed": bool(str(remaining_turns_text or "").strip()),
        "turn_color": turn_color,
        "score_bonus": score_bonus,
        "exam_ranking": exam_ranking_str,
        "vocal": vocal_value,
        "vocal_observed": vocal_observed,
        "dance": dance_value,
        "dance_observed": dance_observed,
        "visual": visual_value,
        "visual_observed": visual_observed,
        "has_progress_hud": results.exists_label(ProducerLabels.PC_PROGRESS),
        "recommend_action_text": recommend_text,
        "recommend_action_kind": infer_param_kind(recommend_text) if recommend_text else "",
        # 课程进度圆圈解析结果
        "progress_circle": progress_info,
    }


def sync_visible_planning_context(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    phase: str,
    position: str,
    reason: str = "visible_hud_sync",
) -> dict[str, Any]:
    """把当前帧能稳定读到的周规划 HUD 尽快同步到上下文。"""
    setattr(app, "_produce_decision_ctx", ctx)
    setattr(app, "_produce_decision_ctx", ctx)
    hud_state = _extract_hud_state(app)
    updated_fields: list[str] = []

    if bool(hud_state.get("p_point_observed", False)):
        ctx.hud_p_point = int(hud_state.get("p_point") or 0)
        ctx.consult_remaining_p_points = ctx.hud_p_point
        ctx.economy_state = {
            **ctx.economy_state,
            "p_point": ctx.hud_p_point,
        }
        updated_fields.append("p_point")

    next_parameter_state = dict(ctx.parameter_state)
    for key in ("vocal", "dance", "visual"):
        if not bool(hud_state.get(f"{key}_observed", False)):
            continue
        next_parameter_state[key] = hud_state.get(key)
        updated_fields.append(key)
    parameter_limit = int(getattr(ctx, "parameter_growth_limit", 0) or 0)
    if parameter_limit > 0:
        for key in ("vocal", "dance", "visual"):
            next_parameter_state[f"{key}_max"] = parameter_limit
    if updated_fields:
        ctx.parameter_state = next_parameter_state
        ctx.last_sync_reason = reason
        logger.debug(
            "hud: 快速同步周规划上下文 phase={} position={} updated={} p_point={} params={}",
            phase,
            position,
            updated_fields,
            ctx.hud_p_point,
            {
                "vocal": ctx.parameter_state.get("vocal", ""),
                "dance": ctx.parameter_state.get("dance", ""),
                "visual": ctx.parameter_state.get("visual", ""),
            },
        )
    return hud_state


def _canonical_entity_name(entity: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    """返回非 UI 决策链路使用的实体原始名称。"""
    metadata = metadata if metadata is not None else dict(entity.get("metadata", {}) or {})
    return str(
        metadata.get("raw_name")
        or entity.get("name")
        or ""
    )


def _build_hand_snapshot(resolved_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建hand、snapshot并返回结果。

    Args:
        resolved_entities: 用于提供resolved、entities相关输入。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    entries: list[dict[str, Any]] = []
    for entity in resolved_entities:
        metadata = dict(entity.get("metadata", {}) or {})
        payload = _snapshot_card_payload(
            card_id=str(entity.get("db_id") or ""),
            name=_canonical_entity_name(entity, metadata),
            metadata=metadata,
        )
        payload["db_id"] = entity.get("db_id") or ""
        payload["rarity"] = metadata.get("rarity") or ""
        entries.append(payload)
    return entries


def _build_initial_deck_snapshot(ctx: "ProduceContext") -> list[dict[str, Any]]:
    """构建initial、deck、snapshot并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    card_details = dict((ctx.formation_details or {}).get("cards_and_items", {}) or {})
    entries = list(card_details.get("matched_entries", []) or [])
    deck_entries: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") != "produce_card":
            continue
        card_id = str(entry.get("id") or "")
        metadata = _enrich_card_metadata(card_id, upgrade_count=0)
        deck_entries.append(
            _snapshot_card_payload(
                card_id=card_id,
                name=str(entry.get("name") or ""),
                metadata=metadata,
            )
        )
    return deck_entries


def _build_current_deck_snapshot(ctx: "ProduceContext") -> list[dict[str, Any]]:
    """从初始牌组快照出发，叠加 deck_mutations 返回当前实际牌组。"""
    deck = _build_initial_deck_snapshot(ctx)

    # 按 card_id 索引增量強化记录
    enhance_map: dict[str, int] = {}
    acquired: list[dict[str, Any]] = []
    removed_ids: set[str] = set()

    for m in ctx.deck_mutations:
        mt = m.get("type")
        card_id = str(m.get("card_id") or "")
        kind = str(m.get("kind", "produce_card"))
        if not card_id:
            continue
        if mt == "enhance":
            enhance_map[card_id] = enhance_map.get(card_id, 0) + int(m.get("upgrade_count", 1))
        elif mt == "acquire" and kind == "produce_card":
            acquired.append(m)
        elif mt == "remove" and kind == "produce_card":
            removed_ids.add(card_id)

    # 应用強化: 更新 upgrade_count 并重新获取元数据
    if enhance_map:
        for entry in deck:
            cid = str(entry.get("id") or "")
            if cid in enhance_map:
                uc = min(enhance_map[cid], 3)
                metadata = _enrich_card_metadata(cid, upgrade_count=uc)
                updated_entry = _snapshot_card_payload(
                    card_id=cid,
                    name=str(entry.get("name") or ""),
                    metadata=metadata,
                )
                entry.update(updated_entry)
                entry["upgrade_count"] = uc

    # 应用削除
    if removed_ids:
        deck = [e for e in deck if str(e.get("id") or "") not in removed_ids]

    # 应用获取
    for m in acquired:
        card_id = str(m.get("card_id") or "")
        if card_id in removed_ids:
            continue
        metadata = _enrich_card_metadata(card_id, upgrade_count=0)
        deck.append(
            _snapshot_card_payload(
                card_id=card_id,
                name=str(m.get("name") or ""),
                metadata=metadata,
            )
        )

    return deck


def _build_produce_item_snapshot(ctx: "ProduceContext") -> list[dict[str, Any]]:
    """构建produce、道具、snapshot并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    card_details = dict((ctx.formation_details or {}).get("cards_and_items", {}) or {})
    item_ids = list(card_details.get("produce_item_ids", []) or [])
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item_id in item_ids:
        sid = str(item_id)
        seen_ids.add(sid)
        metadata = _enrich_item_metadata(sid)
        items.append({
            "id": sid,
            "name": str(metadata.get("raw_name") or ""),
            "description": metadata.get("description") or "",
            "rarity": metadata.get("rarity") or "",
        })
    # 叠加 deck_mutations 中新获取的 produce_item
    for m in ctx.deck_mutations:
        if m.get("type") == "acquire" and m.get("kind") == "produce_item":
            mid = str(m.get("card_id") or "")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                metadata = _enrich_item_metadata(mid)
                items.append({
                    "id": mid,
                    "name": str(metadata.get("raw_name") or m.get("name") or ""),
                    "description": metadata.get("description") or "",
                    "rarity": metadata.get("rarity") or "",
                })
    return items


def _build_formation_ability_snapshot(ctx: "ProduceContext") -> list[dict[str, Any]]:
    """从编成详情中提取已匹配的支援/P偶像能力，供 LLM 使用。"""
    abilities_data = dict((ctx.formation_details or {}).get("abilities", {}) or {})
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section_key in ("p_idol_abilities", "lesson_support", "support_abilities"):
        section = abilities_data.get(section_key, {})
        if not isinstance(section, dict):
            continue
        matched = section.get("matched_entries") or []
        for entry in matched:
            eid = str(entry.get("id") or "")
            name = str(entry.get("name") or eid)
            if not eid or eid in seen:
                continue
            seen.add(eid)
            entries.append({"name": name, "section": section_key})
    return entries


def _build_formation_event_snapshot(ctx: "ProduceContext") -> list[dict[str, Any]]:
    """从编成详情中提取支援卡事件信息，供 LLM 使用。"""
    events_data = dict((ctx.formation_details or {}).get("events", {}) or {})
    support_cards = events_data.get("support_cards") or []
    result: list[dict[str, Any]] = []
    for sc in support_cards:
        card_name = sc.get("name") or sc.get("name_ja") or sc.get("id", "")
        events = sc.get("events") or []
        event_lines: list[str] = []
        for ev in events:
            title = ev.get("title") or ev.get("title_ja") or f"Event#{ev.get('number', '?')}"
            descs = ev.get("descriptions") or []
            desc_text = ", ".join(descs) if descs else ""
            if desc_text:
                event_lines.append(f"{title}: {desc_text}")
            else:
                event_lines.append(title)
        result.append({"card_name": card_name, "events": event_lines})
    return result


def _build_drink_snapshot(drink_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建饮料、snapshot并返回结果。

    Args:
        drink_entities: 用于提供饮料、entities相关输入。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    drinks: list[dict[str, Any]] = []
    for entity in drink_entities:
        metadata = dict(entity.get("metadata", {}) or {})
        drinks.append({
            "id": entity.get("db_id") or "",
            "name": _canonical_entity_name(entity, metadata),
            "description": metadata.get("description") or "",
            "rarity": metadata.get("rarity") or "",
            "effect_types": list(metadata.get("effect_types", []) or []),
        })
    return drinks


def _snapshot_card_category_name(value: Any) -> str:
    """构建快照`snapshot_card_category_name`。"""
    category = str(value or "").strip()
    if category in _SNAPSHOT_CARD_CATEGORY_NAMES:
        return _SNAPSHOT_CARD_CATEGORY_NAMES[category]
    if category in _CARD_CATEGORY_DISPLAY_OVERRIDES:
        return _CARD_CATEGORY_DISPLAY_OVERRIDES[category]
    if category.startswith("ProduceCardCategory_"):
        suffix = category.removeprefix("ProduceCardCategory_")
        suffix = suffix.removesuffix("Skill")
        return suffix or "未知"
    return category or "未知"


def _format_snapshot_description(value: Any) -> str:
    """清理快照里的卡牌说明文本，减少 OCR/数据库拼接噪声。"""
    text = normalize_ocr_jp(fullwidth_to_halfwidth(str(value or ""))).strip()
    if not text:
        return ""
    text = _SNAPSHOT_DESCRIPTION_PARENS_PATTERN.sub("(", text)
    text = _SNAPSHOT_DESCRIPTION_PARENS_CLOSE_PATTERN.sub(")", text)
    text = _SNAPSHOT_DESCRIPTION_PLUS_PATTERN.sub("+", text)
    text = _SNAPSHOT_DESCRIPTION_NOISE_PATTERN.sub(" ", text)
    text = _SNAPSHOT_DESCRIPTION_MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip(" ;；")



def _snapshot_card_payload(
    *,
    card_id: str,
    name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """统一构建快照中的技能卡载荷。"""
    return {
        "id": card_id,
        "name": str(metadata.get("raw_name") or name or ""),
        "description": _format_snapshot_description(metadata.get("description") or ""),
        "category": _snapshot_card_category_name(metadata.get("category") or ""),
        "cost": metadata.get("cost") or 0,
        "effect_types": list(metadata.get("effect_types", []) or []),
        "upgrade_count": int(metadata.get("upgrade_count") or 0),
    }


def _is_offensive_snapshot_card(card: dict[str, Any]) -> bool:
    """判断offensive、snapshot、卡牌是否成立。

    Args:
        card: 用于提供卡牌相关输入。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    effect_types = [
        str(value or "")
        for value in card.get("effect_types", []) or []
    ]
    if any(
        keyword in effect_type
        for effect_type in effect_types
        for keyword in _OFFENSIVE_EFFECT_KEYWORDS
    ):
        return True

    description = str(card.get("description") or "")
    return any(keyword in description for keyword in _OFFENSIVE_DESCRIPTION_KEYWORDS)


def _count_offensive_snapshot_cards(cards: list[dict[str, Any]]) -> int:
    """统计`offensive_snapshot_cards`数量。"""
    return sum(1 for card in cards if _is_offensive_snapshot_card(card))


def _build_snapshot_deck_summary(cards: list[dict[str, Any]]) -> str:
    """构建snapshot、deck、摘要并返回结果。

    Args:
        cards: 用于提供卡牌相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    if not cards:
        return _DECK_SUMMARY_EMPTY_TEXT

    category_counts: Counter[str] = Counter()
    total_cost = 0.0
    cost_count = 0
    for card in cards:
        category_counts[_snapshot_card_category_name(card.get("category"))] += 1
        if card.get("cost") not in (None, ""):
            total_cost += float(card.get("cost") or 0)
            cost_count += 1

    category_text = ", ".join(
        f"{category}×{count}"
        for category, count in category_counts.most_common()
    )
    if cost_count <= 0:
        return f"分类: {category_text}"
    avg_cost = total_cost / max(cost_count, 1)
    return f"分类: {category_text} | 平均消耗:{avg_cost:.1f}"


def _build_snapshot_reshuffle_hint(
    *,
    deck_cards: list[dict[str, Any]],
    grave_cards: list[dict[str, Any]],
    offensive_counts: dict[str, int],
) -> str:
    """构建snapshot、reshuffle、提示并返回结果。

    Args:
        deck_cards: 用于提供deck、卡牌相关输入。
        grave_cards: 用于提供grave、卡牌相关输入。
        offensive_counts: 用于提供offensive、counts相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    if len(deck_cards) <= 2 and grave_cards:
        return f"牌库仅剩{len(deck_cards)}张；下次抽牌大概率会把弃牌堆洗回。"
    if offensive_counts.get("deck", 0) <= 0 and offensive_counts.get("grave", 0) > 0:
        return "当前牌库几乎没有火力牌，后续主要依赖洗回弃牌堆后的再抽。"
    return ""


def _observe_bottom_inventory_drinks(
    app: "AppProcessor",
) -> tuple[list[dict[str, Any]], bool]:
    """观察课内底栏 P 饮料库存，并整理成可复用的实体列表。"""
    results = getattr(app, "latest_results", None)
    frame = getattr(app, "latest_frame", None)
    if results is None or frame is None or getattr(frame, "size", 0) <= 0:
        return [], False

    frame_height = int(frame.shape[0])
    boxes = sorted(
        (
            box
            for box in results.filter_by_label(ProducerLabels.P_DRINK)
            if getattr(box, "cy", 0) >= frame_height * 0.88
        ),
        key=lambda item: item.cx,
    )
    debugger = getattr(app, "debug_tools", None) or DebugTools()
    observed: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        resolution = resolve_produce_drink_identity(
            "",
            app=app,
            box=box,
            index=index,
        )
        metadata = dict(resolution.metadata or {})
        display_name = (
            metadata.get("raw_name")
            or resolution.display_name
            or f"Pドリンク#{index + 1}"
        )
        observed.append({
            "action_id": resolution.action_id,
            "db_id": resolution.db_id,
            "name": str(metadata.get("raw_name") or resolution.display_name or ""),
            "source": resolution.source,
            "confidence": resolution.confidence,
            "metadata": metadata,
        })
        coords = _serialize_box(box)
        if coords is None:
            continue
        label_core = resolution.db_id or display_name
        debugger.add_box(
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            label=f"inventory_drink:{index} {str(label_core)[:24]}",
            color=(255, 0, 160),
            alpha=0.18,
            duration=3.0,
            font_size=18,
        )
    return observed, True


def _summarize_exam_criteria(ctx: "ProduceContext") -> list[str]:
    """将育成页采集到的审查基准压缩为适合提示词的摘要行。"""
    criteria = dict(getattr(ctx, "exam_criteria", {}) or {})
    if not criteria:
        return []
    lines: list[str] = []
    for key in ("vocal", "dance", "visual"):
        value = criteria.get(key)
        if value in (None, "", 0):
            continue
        label = {
            "vocal": "ボーカル",
            "dance": "ダンス",
            "visual": "ビジュアル",
        }.get(key, key)
        lines.append(f"{label}: {value}")
    for key in ("special", "note", "summary"):
        value = str(criteria.get(key) or "").strip()
        if value:
            lines.append(value)
    return lines[:4]


def _summarize_training_tasks(ctx: "ProduceContext") -> list[str]:
    """将育成课题列表压缩为简短摘要，避免整段 OCR 原文撑爆 prompt。"""
    tasks = list(getattr(ctx, "training_tasks", []) or [])
    summaries: list[str] = []
    for task in tasks:
        if isinstance(task, str):
            text = task.strip()
            if text:
                summaries.append(text)
            continue
        if not isinstance(task, dict):
            continue
        parts: list[str] = []
        title = str(task.get("title") or task.get("label") or task.get("name") or "").strip()
        if title:
            parts.append(title)
        condition = str(task.get("condition") or task.get("target") or task.get("summary") or "").strip()
        if condition and condition not in parts:
            parts.append(condition)
        reward = str(task.get("reward") or "").strip()
        if reward:
            parts.append(f"报酬: {reward}")
        status = str(task.get("status") or "").strip()
        if status:
            parts.append(f"状态: {status}")
        if parts:
            summaries.append(" | ".join(parts))
    return summaries[:5]


def _build_produce_goal_snapshot(ctx: "ProduceContext") -> dict[str, Any]:
    """构建主流程长期目标摘要，供所有阶段共享。"""
    exam_criteria = _summarize_exam_criteria(ctx)
    training_tasks = _summarize_training_tasks(ctx)
    summary_parts: list[str] = []
    if exam_criteria:
        summary_parts.append(f"審査基準: {' / '.join(exam_criteria)}")
    if training_tasks:
        summary_parts.append(f"育成課題: {' / '.join(training_tasks[:2])}")
    return {
        "exam_criteria": exam_criteria,
        "training_tasks": training_tasks,
        "summary": "；".join(summary_parts),
    }


def _build_schedule_context_snapshot(ctx: "ProduceContext") -> dict[str, Any]:
    """整理周流程历史与未来日程，帮助模型理解长期规划压力。"""
    history = [str(item or "").strip() for item in (ctx.schedule_history or []) if str(item or "").strip()]
    notebook_entries = list(ctx.handler_state.get("p_notebook_schedule") or [])
    future_schedule: list[str] = []
    raw_entries: list[dict[str, Any]] = []
    for entry in notebook_entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("completed"):
            continue
        actions = [str(item or "").strip() for item in (entry.get("actions") or []) if str(item or "").strip()]
        special_event = str(entry.get("special_event") or "").strip()
        week = int(entry.get("week") or 0)
        label = str(entry.get("label") or entry.get("week_label") or entry.get("title") or "").strip()
        if not label and week > 0:
            label = f"第{week}周"
        parts = [part for part in [label, special_event] if part]
        if actions:
            parts.append(" / ".join(actions[:3]))
        if parts:
            future_schedule.append(" | ".join(parts))
        raw_entries.append({
            "week": week,
            "label": label,
            "actions": actions,
            "special_event": special_event,
            "raw_text": str(entry.get("raw_text") or "").strip(),
        })
    return {
        "history": history[-8:],
        "future_schedule": future_schedule[:8],
        "history_summary": " → ".join(history[-5:]) if history else "",
        "raw_entries": raw_entries[:8],
    }


def _scenario_rule_payload(ctx: "ProduceContext") -> dict[str, Any]:
    """构建剧本制度约束摘要。"""
    scenario = str(ctx.scenario or "").strip().lower()
    produce_group_type = str((ctx.produce_metadata or {}).get("produce_group_type") or "")
    handler_state = dict(getattr(ctx, "handler_state", {}) or {})
    rest_limit_total: int | None = None
    rest_limit_used: int | None = _optional_int(
        handler_state.get("rest_limit_used")
        or handler_state.get("nia_rest_used")
        or getattr(ctx, "rest_limit_used", None)
    )
    rest_rule = ""
    if scenario == "nia":
        rest_limit_total = 4
        if rest_limit_used is None:
            rest_limit_used = _infer_rest_used_from_history(ctx)
        rest_rule = "NIA 中休息总次数上限为 4，前四周不可休息。"
    elif scenario == "hajime" and str(ctx.difficulty or "").strip().lower() == "master":
        rest_rule = "初 Master 前四周不可休息。"
    rest_limit_remaining = (
        max(int(rest_limit_total) - int(rest_limit_used or 0), 0)
        if rest_limit_total is not None and rest_limit_used is not None
        else None
    )
    fan_votes_current = _optional_int(
        handler_state.get("fan_votes_current")
        or handler_state.get("fan_vote_current")
        or handler_state.get("nia_fan_votes")
        or getattr(ctx, "fan_votes_current", None)
        or getattr(ctx, "fan_votes", None)
    )
    audition_stage_current = str(
        getattr(ctx, "current_exam_type", "")
        or handler_state.get("audition_stage_current")
        or ""
    )
    audition_stage_next = str(handler_state.get("audition_stage_next") or "")
    fan_votes_next_threshold = None
    fan_votes_gap = None
    if scenario == "nia":
        fan_votes_next_threshold = _next_nia_fan_vote_threshold(
            fan_votes_current=fan_votes_current,
            audition_stage=audition_stage_current or audition_stage_next,
        )
        if fan_votes_current is not None and fan_votes_next_threshold is not None:
            fan_votes_gap = max(int(fan_votes_next_threshold) - int(fan_votes_current), 0)
    scenario_label = {
        "nia": "N.I.A",
        "hajime": "定期公演『初』",
    }.get(scenario, scenario or "未知")
    summary_parts = [rest_rule] if rest_rule else []
    if fan_votes_current is not None:
        summary_parts.append(f"fan vote={fan_votes_current}")
    if fan_votes_next_threshold is not None:
        summary_parts.append(f"下个 fan vote 门槛={fan_votes_next_threshold}")
    if rest_limit_remaining is not None:
        summary_parts.append(f"休息剩余={rest_limit_remaining}/{rest_limit_total}")
    return {
        "scenario": scenario,
        "scenario_label": scenario_label,
        "produce_group_type": produce_group_type,
        "rest_limit_total": rest_limit_total,
        "rest_limit_used": rest_limit_used,
        "rest_limit_remaining": rest_limit_remaining,
        "fan_votes_current": fan_votes_current,
        "fan_votes_next_threshold": fan_votes_next_threshold,
        "fan_votes_gap": fan_votes_gap,
        "audition_stage_current": audition_stage_current,
        "audition_stage_next": audition_stage_next,
        "weeks_until_gate": _compute_remaining_weeks(ctx),
        "summary": "；".join(part for part in summary_parts if part),
    }


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _infer_rest_used_from_history(ctx: "ProduceContext") -> int:
    count = 0
    for entry in list(getattr(ctx, "schedule_history", []) or []):
        text = str(entry or "")
        if ProduceText.REST_ACTION in text or ProduceText.REST in text or "休息" in text or "お休み" in text:
            count += 1
    return count


def _next_nia_fan_vote_threshold(
    *,
    fan_votes_current: int | None,
    audition_stage: str,
) -> int | None:
    stage = str(audition_stage or "").lower()
    if "final" in stage or "finale" in stage or "最終" in stage:
        thresholds = (28000, 40000, 57000)
    elif "second" in stage or "2" in stage or "二" in stage:
        thresholds = (14000, 25000)
    elif "first" in stage or "1" in stage or "一" in stage:
        thresholds = (4000, 9000)
    else:
        thresholds = (4000, 9000, 14000, 25000, 28000, 40000, 57000)
    if fan_votes_current is None:
        return thresholds[0] if thresholds else None
    for threshold in thresholds:
        if int(fan_votes_current) < int(threshold):
            return int(threshold)
    return None


def _build_next_gate_snapshot(
    ctx: "ProduceContext",
    *,
    remaining_weeks: int | None,
    exam_criteria: list[str],
    training_tasks: list[str],
) -> dict[str, Any]:
    """构建下一道门槛摘要。"""
    scenario = str(ctx.scenario or "").strip().lower()
    current_exam_type = str(getattr(ctx, "current_exam_type", "") or "").strip()
    gate_type = ""
    gate_label = ""
    if current_exam_type:
        gate_type = current_exam_type
        gate_label = current_exam_type
    elif scenario == "nia":
        gate_type = "audition"
        gate_label = "下一次 audition / finale 门槛"
    else:
        gate_type = "exam"
        gate_label = "下一次审查门槛"
    requirements = [*exam_criteria[:3], *training_tasks[:2]]
    blocking_gaps = requirements[:3]
    readiness_parts: list[str] = []
    if remaining_weeks is not None:
        readiness_parts.append(f"距下个门槛约剩{remaining_weeks}周")
    if blocking_gaps:
        readiness_parts.append(f"当前主要约束: {' / '.join(blocking_gaps[:2])}")
    return {
        "gate_type": gate_type,
        "gate_label": gate_label,
        "weeks_until_gate": remaining_weeks,
        "requirements": requirements,
        "readiness_summary": "；".join(readiness_parts),
        "blocking_gaps": blocking_gaps,
    }


def _build_resource_pressure_snapshot(
    ctx: "ProduceContext",
    *,
    hud_state: dict[str, Any],
    known_drinks: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建资源压力摘要。"""
    stamina = int(hud_state.get("stamina") or ctx.hud_stamina or 0)
    max_stamina = int(hud_state.get("max_stamina") or ctx.hud_max_stamina or 0)
    p_point = int(hud_state.get("p_point") or ctx.hud_p_point or 0)
    stamina_ratio = (stamina / max(max_stamina, 1)) if max_stamina > 0 else None
    stamina_pressure = "unknown"
    if stamina_ratio is not None:
        if stamina_ratio <= 0.35:
            stamina_pressure = "high"
        elif stamina_ratio <= 0.65:
            stamina_pressure = "medium"
        else:
            stamina_pressure = "low"
    p_point_pressure = "low" if p_point >= 60 else "medium" if p_point >= 20 else "high"
    drink_pressure = "low" if known_drinks else "medium"
    summary_parts: list[str] = []
    if stamina_ratio is not None:
        summary_parts.append(f"体力压力={stamina_pressure}({stamina}/{max_stamina or '?'})")
    if p_point >= 0:
        summary_parts.append(f"P点压力={p_point_pressure}({p_point})")
    summary_parts.append(f"饮料余量={'有库存' if known_drinks else '无库存'}")
    return {
        "stamina_pressure": stamina_pressure,
        "p_point_pressure": p_point_pressure,
        "drink_pressure": drink_pressure,
        "stamina_ratio": round(stamina_ratio, 3) if stamina_ratio is not None else None,
        "summary": "；".join(summary_parts),
    }


def _build_time_pressure_snapshot(
    ctx: "ProduceContext",
    *,
    phase_key: str,
    remaining_weeks: int | None,
    remaining_turns: int | None,
) -> dict[str, Any]:
    """构建时间窗口压力摘要。"""
    pressure = "unknown"
    summary_parts: list[str] = []
    if phase_key in {GameplayPhase.LESSON, GameplayPhase.EXAM} and remaining_turns is not None:
        if remaining_turns <= 1:
            pressure = "high"
        elif remaining_turns <= 3:
            pressure = "medium"
        else:
            pressure = "low"
        summary_parts.append(f"回合窗口={pressure}(剩余{remaining_turns}回合)")
    elif remaining_weeks is not None:
        if remaining_weeks <= 2:
            pressure = "high"
        elif remaining_weeks <= 5:
            pressure = "medium"
        else:
            pressure = "low"
        summary_parts.append(f"周窗口={pressure}(剩余{remaining_weeks}周)")
    if str(ctx.current_exam_type or "").strip():
        summary_parts.append(f"当前考试阶段={ctx.current_exam_type}")
    return {
        "pressure": pressure,
        "remaining_weeks": remaining_weeks,
        "remaining_turns": remaining_turns,
        "summary": "；".join(summary_parts),
    }


def _build_build_plan_snapshot(
    ctx: "ProduceContext",
    *,
    parameter_priority: str,
    parameter_stats: dict[str, Any],
    next_gate: dict[str, Any],
) -> dict[str, Any]:
    """构建属性成长/毕业面板导向规划。"""
    focus_stats: list[str] = []
    for key, label in (("vocal", "Vo"), ("dance", "Da"), ("visual", "Vi")):
        value = parameter_stats.get(key, "")
        if value not in (None, ""):
            max_value = parameter_stats.get(f"{key}_max", "")
            focus_stats.append(f"{label}={value}{f'/{max_value}' if max_value not in (None, '') else ''}")
    summary_parts: list[str] = []
    if parameter_priority:
        summary_parts.append(f"成长优先级={parameter_priority}")
    if focus_stats:
        summary_parts.append(f"当前面板={' / '.join(focus_stats)}")
    blocking_gaps = list(next_gate.get("blocking_gaps") or [])
    if blocking_gaps:
        summary_parts.append(f"近期门槛缺口={' / '.join(blocking_gaps[:2])}")
    return {
        "priority": parameter_priority,
        "current_stats": focus_stats,
        "blocking_gaps": blocking_gaps,
        "summary": "；".join(summary_parts),
    }


def _build_deck_plan_snapshot(
    *,
    deck_cards: list[dict[str, Any]],
    offensive_counts: dict[str, int],
    hand_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建牌组结构规划。"""
    total_cards = len(deck_cards)
    offensive_total = int(offensive_counts.get("deck", 0)) + int(offensive_counts.get("hand", 0))
    support_total = max(total_cards - int(offensive_counts.get("deck", 0)), 0)
    needs: list[str] = []
    if offensive_total <= max(2, total_cards // 4):
        needs.append("补直接得分/火力")
    if support_total <= max(2, total_cards // 5):
        needs.append("补启动或资源铺垫")
    if total_cards >= 18:
        needs.append("避免继续稀释牌组")
    if not needs:
        needs.append("维持当前牌组结构，优先补兑现链完整度")
    hand_names = [str(card.get("name") or "") for card in hand_entries[:4] if str(card.get("name") or "").strip()]
    return {
        "deck_count": total_cards,
        "offensive_total": offensive_total,
        "support_total": support_total,
        "current_hand_focus": hand_names,
        "needs": needs,
        "summary": "；".join(needs[:3]),
    }


def _build_inventory_plan_snapshot(
    *,
    known_drinks: list[dict[str, Any]],
    produce_items: list[dict[str, Any]],
    remaining_weeks: int | None,
) -> dict[str, Any]:
    """构建饮料 / P物品使用窗口规划。"""
    needs: list[str] = []
    if not known_drinks:
        needs.append("库存缺少可立即兑现的饮料")
    if remaining_weeks is not None and remaining_weeks <= 2:
        needs.append("优先保留能在短窗口内直接兑现的库存")
    if produce_items:
        needs.append("比较 P物品长期收益与当前门槛压力是否一致")
    if not needs:
        needs.append("优先保留泛用爆发或保命资源")
    return {
        "drink_count": len(known_drinks),
        "item_count": len(produce_items),
        "needs": needs,
        "summary": "；".join(needs[:3]),
    }


def _build_recent_outcomes_snapshot(ctx: "ProduceContext") -> list[str]:
    """压缩最近几步已发生的关键操作，用作短期规划参照。"""
    result: list[str] = []
    for op in list(ctx.operation_history or [])[-5:]:
        action = str(getattr(op, "action", "") or "").strip()
        if not action:
            continue
        target = str(getattr(op, "target", "") or "").strip()
        phase = str(getattr(op, "phase", "") or "").strip()
        line = f"{phase}:{action}"
        if target:
            line = f"{line}({target})"
        result.append(line)
    return result


def _build_planning_snapshot(
    ctx: "ProduceContext",
    *,
    phase_key: str,
    hud_state: dict[str, Any],
    known_drinks: list[dict[str, Any]],
    produce_items: list[dict[str, Any]],
    parameter_stats: dict[str, Any],
    hand_entries: list[dict[str, Any]],
    deck_cards: list[dict[str, Any]],
    offensive_counts: dict[str, int],
    produce_goals: dict[str, Any],
    schedule_context: dict[str, Any],
    remaining_weeks: int | None,
    remaining_turns: int | None,
    parameter_priority: str,
) -> dict[str, Any]:
    """构建统一长线规划结构。"""
    scenario_rules = _scenario_rule_payload(ctx)
    next_gate = _build_next_gate_snapshot(
        ctx,
        remaining_weeks=remaining_weeks,
        exam_criteria=list(produce_goals.get("exam_criteria") or []),
        training_tasks=list(produce_goals.get("training_tasks") or []),
    )
    resource_pressure = _build_resource_pressure_snapshot(
        ctx,
        hud_state=hud_state,
        known_drinks=known_drinks,
    )
    time_pressure = _build_time_pressure_snapshot(
        ctx,
        phase_key=phase_key,
        remaining_weeks=remaining_weeks,
        remaining_turns=remaining_turns,
    )
    build_plan = _build_build_plan_snapshot(
        ctx,
        parameter_priority=parameter_priority,
        parameter_stats=parameter_stats,
        next_gate=next_gate,
    )
    deck_plan = _build_deck_plan_snapshot(
        deck_cards=deck_cards,
        offensive_counts=offensive_counts,
        hand_entries=hand_entries,
    )
    inventory_plan = _build_inventory_plan_snapshot(
        known_drinks=known_drinks,
        produce_items=produce_items,
        remaining_weeks=remaining_weeks,
    )
    route_bias = {
        "idol_plan_type": _current_idol_plan_payload(ctx)["type"],
        "idol_plan_label": _current_idol_plan_payload(ctx)["label"],
        "parameter_priority": parameter_priority,
        "summary": "；".join(
            part for part in [
                _current_idol_plan_payload(ctx)["label"],
                f"属性成长优先级={parameter_priority}" if parameter_priority else "",
            ] if part
        ),
    }
    current_objectives = {
        "produce_goal_summary": str(produce_goals.get("summary") or ""),
        "training_tasks": list(produce_goals.get("training_tasks") or []),
        "exam_criteria": list(produce_goals.get("exam_criteria") or []),
        "summary": str(produce_goals.get("summary") or ""),
    }
    recent_outcomes = _build_recent_outcomes_snapshot(ctx)
    schedule_future = list(schedule_context.get("future_schedule") or [])
    summary_parts = [
        str(build_plan.get("summary") or ""),
        str(deck_plan.get("summary") or ""),
        str(inventory_plan.get("summary") or ""),
        str(next_gate.get("readiness_summary") or ""),
    ]
    return {
        "scenario_rules": scenario_rules,
        "current_objectives": current_objectives,
        "next_gate": next_gate,
        "resource_pressure": resource_pressure,
        "time_pressure": time_pressure,
        "route_bias": route_bias,
        "build_plan": build_plan,
        "deck_plan": deck_plan,
        "inventory_plan": inventory_plan,
        "recent_outcomes": recent_outcomes,
        "future_schedule": schedule_future,
        "schedule_history_summary": str(schedule_context.get("history_summary") or ""),
        "consult_session": _build_consult_session_summary(ctx) if phase_key == GameplayPhase.CONSULT else None,
        "summary": "；".join([part for part in summary_parts if part]),
    }


def _build_llm_actions(
    candidate_payloads: list[dict[str, Any]],
    *,
    phase: str,
    position: str,
    stage_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """构建llm、actions并返回结果。

    Args:
        candidate_payloads: 用于提供候选项、payloads相关输入。
        phase: 当前 gameplay 阶段标识。
        position: 当前阶段下的细分画面位置标识。
        stage_context: 用于提供stage、context相关输入。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    actions: list[dict[str, Any]] = []
    for payload in candidate_payloads:
        phase_key = phase.value if hasattr(phase, "value") else str(phase)
        metadata = dict(payload.get("metadata", {}) or {})
        consult_action = str(payload.get("type") or "")

        # ── 相談: 判断候选是否为实体类（兑换 / 強化选卡 / 削除选卡） ──
        # 这些候选代表具体的游戏实体（卡/饮料/物品），必须有 db_id
        is_consult_entity = (
            phase_key == GameplayPhase.CONSULT
            and (
                consult_action.startswith("consult_exchange")
                or consult_action in {
                    "consult_select_enhancement_target",
                    "consult_select_remove_target",
                }
            )
        )
        # ── 战斗 (lesson/exam): 技能卡必须有 db_id，未识别卡不参与自动决策 ──
        is_battle_card = (
            phase_key in {GameplayPhase.LESSON, GameplayPhase.EXAM}
            and is_produce_card_action_id(payload.get("id"))
        )
        # ── 技能卡奖励: 允许无 db_id 候选进入 LLM（否则会出现“合法动作为空”） ──
        is_skill_reward_card = (
            phase_key == GameplayPhase.SKILL_REWARD
            and is_produce_card_action_id(payload.get("id"))
        )
        is_unresolved_skill_reward_card = bool(is_skill_reward_card and not str(payload.get("db_id") or ""))
        # ── 技能卡奖励再抽選: 特殊非实体候选 ──
        is_skill_reward_redraw = (
            phase_key == GameplayPhase.SKILL_REWARD
            and bool(metadata.get("is_redraw"))
        )
        # ── P物品选択: 实体类，有 db_id 则走数据库描述 ──
        is_item_select_entity = (
            phase_key == GameplayPhase.ITEM_SELECT
            and str(payload.get("id") or "").startswith("produce_item:")
        )
        # ── P饮料选择: 有 db_id 的饮料走数据库描述 ──
        is_p_drink_entity = (
            phase_key == GameplayPhase.P_DRINK
            and is_produce_drink_action_id(payload.get("id"))
        )
        # ── 外出活動: outing probe 匹配到 DB ID 的選項 ──
        is_outing_entity = (
            str(metadata.get("candidate_type") or "") == "outing_activity"
            and bool(payload.get("db_id"))
        )
        db_id = str(payload.get("db_id") or "")
        # 全链路 DB ID 传递: 相談实体与战斗出牌仍要求 db_id；技能卡奖励例外（可用 OCR 名称做兜底）
        if (is_consult_entity or is_battle_card) and not db_id:
            continue
        # 技能卡奖励: 未识别卡（无 db_id 且非再抽選）也跳过
        if phase_key == GameplayPhase.SKILL_REWARD and not is_skill_reward_card and not is_skill_reward_redraw and not db_id:
            continue
        # 相談兑换: P 点不足的候选直接过滤，避免 LLM 花时间分析买不起的选项
        if is_consult_entity and consult_action.startswith("consult_exchange"):
            price_str = str(metadata.get("price") or "")
            price_val = int(re.search(r"\d+", price_str).group()) if re.search(r"\d+", price_str) else 0
            current_p = int(stage_context.get("p_point") or 0)
            if price_val > 0 and current_p < price_val:
                continue

        is_entity = (
            is_consult_entity
            or is_battle_card
            or (is_skill_reward_card and bool(db_id))
            or (is_item_select_entity and bool(db_id))
            or is_outing_entity
            or (is_p_drink_entity and bool(db_id))
        )

        # ── 描述构建 ──
        if is_skill_reward_redraw:
            # 再抽選: 构建带剩余次数的描述
            remaining = int(metadata.get("redraw_remaining") or 0)
            description = f"再抽選（あと{remaining}回）— 消耗一次再抽選机会，刷新全部候选技能卡"
        elif is_unresolved_skill_reward_card:
            description = (
                str(metadata.get("description") or "").strip()
                or "信息面板 OCR 已识别名称，但暂未匹配主数据库；按名称进行相对选择。"
            )
        elif is_outing_entity:
            # 外出活動: DB 描述 + P 成本
            canonical_name = str(
                metadata.get("raw_name")
                or payload.get("name")
                or ""
            )
            p_cost = metadata.get("p_cost")
            # DB 匹配成功时使用 DB 描述，失败时使用 OCR 效果描述。
            outing_db_desc = str(metadata.get("outing_db_description") or "")
            outing_effect = str(metadata.get("outing_effect") or "")
            parts: list[str] = [canonical_name] if canonical_name else []
            if p_cost is not None:
                parts.append(f"消耗: {p_cost}P")
            else:
                parts.append("免费")
            desc_text = outing_db_desc or outing_effect
            if desc_text:
                parts.append(f"効果: {desc_text}")
            description = " | ".join(parts)
        elif is_entity:
            # 实体类: 所有描述/属性均从数据库查询结果获取，不使用 OCR 原文
            raw_name = str(
                metadata.get("raw_name")
                or payload.get("name")
                or ""
            )
            db_description = str(metadata.get("description") or "")
            # 属性明细（全部来自 _enrich_card_metadata / _enrich_drink_metadata）
            detail_parts: list[str] = []
            upgrade_count = metadata.get("upgrade_count")
            if upgrade_count is not None:
                detail_parts.append(f"等级: {int(upgrade_count)}")
            price = str(metadata.get("price") or "")
            if price:
                detail_parts.append(f"价格: {price}P")
            rarity = str(metadata.get("rarity") or "")
            if rarity:
                rarity_short = rarity.rsplit("_", 1)[-1] if "_" in rarity else rarity
                detail_parts.append(f"稀有度: {rarity_short}")
            plan_label = str(metadata.get("plan_type_label") or "")
            if plan_label:
                detail_parts.append(f"适性: {plan_label}")
            category = str(metadata.get("category") or "")
            if category:
                cat_name = _SNAPSHOT_CARD_CATEGORY_NAMES.get(category, "")
                if cat_name:
                    detail_parts.append(f"分类: {cat_name}")
            cost = int(metadata.get("cost") or 0)
            if cost:
                detail_parts.append(f"消耗体力: {cost}")
            # 组装: "属性1 | 属性2；效果描述"（display_name 已在 label 中，不重复）
            description = " | ".join(detail_parts) if detail_parts else ""
            if db_description:
                description = f"{description}；{db_description}" if description else db_description

            # ── 强化目标: 追加强化后收益对比，帮助 LLM 判断是否值得 ──
            if consult_action == "consult_select_enhancement_target" and upgrade_count is not None:
                next_uc = int(upgrade_count) + 1
                if next_uc <= 3:
                    next_meta = _enrich_card_metadata(db_id, upgrade_count=next_uc)
                    next_desc = str(next_meta.get("description") or "")
                    next_name = str(next_meta.get("raw_name") or "")
                    if next_desc and next_desc != db_description:
                        description = f"{description}；【強化後→{next_name}】{next_desc}"
                    elif next_name:
                        description = f"{description}；【強化後→{next_name}】效果不变"
                else:
                    description = f"{description}；已满级，无法再強化"
        else:
            # 非实体类（强化/削除/退出按钮等 + 周行动）: 保持原有逻辑
            description = (
                metadata.get("description")
                or payload.get("name")
                or ""
            )

        # ── 周行动: 附加效果描述（来自信息面板探查或数据库） ──
        if phase_key == GameplayPhase.SCHEDULE:
            effect_text = str(metadata.get("effect_text") or "").strip()
            # 授業選項: 探査効果描述存放在 lesson_effect 字段
            lesson_effect = str(metadata.get("lesson_effect") or "").strip()
            if lesson_effect and not effect_text:
                effect_text = lesson_effect
            schedule_name = str(
                metadata.get("raw_name")
                or payload.get("name")
                or ""
            ).strip()
            rl_type = str(metadata.get("rl_action_type") or "").strip()
            sched_parts: list[str] = []
            if schedule_name and schedule_name != description:
                sched_parts.append(schedule_name)
            if rl_type:
                sched_parts.append(f"类型: {rl_type}")
            # 授業選項: 附加体力消耗
            stamina_cost = metadata.get("stamina_cost")
            if stamina_cost is not None and metadata.get("lesson_option"):
                sched_parts.append(f"消耗体力: {stamina_cost}")
            if sched_parts:
                prefix = " | ".join(sched_parts)
                description = f"{prefix}；{description}" if description else prefix
            if effect_text and effect_text not in description:
                description = f"{description}；効果: {effect_text}" if description else f"効果: {effect_text}"

        unavailable_reason = str(metadata.get("unavailable_reason") or "").strip()
        if unavailable_reason and unavailable_reason not in description:
            description = (
                f"{description}；注意：{unavailable_reason}"
                if description
                else unavailable_reason
            )
        # ── 标签: 实体类用 db_id（RL 对接），外出用可读名（LLM 不需要内部 ID） ──
        if is_outing_entity:
            # 外出: LLM 看到可读名称，db_id 仅供 RL 外部消费
            label = str(
                metadata.get("raw_name")
                or payload.get("name")
                or payload.get("label")
                or ""
            )
        elif is_entity:
            # 战斗卡/实体类: label 用可读名称（LLM 需要看懂卡名），db_id 已在 payload 中保留供 RL 使用
            label = str(
                metadata.get("raw_name")
                or payload.get("name")
                or payload.get("label")
                or ""
            )
        elif phase_key == GameplayPhase.CONSULT:
            label = str(payload.get("id") or payload.get("name") or payload.get("label") or "")
        else:
            label = str(payload.get("name") or payload.get("label") or "")

        decision_tags = _infer_action_decision_tags(
            phase_key=phase_key,
            payload=payload,
            metadata=metadata,
            description=description,
        )
        cost_summary = _build_action_cost_summary(metadata, description)
        gain_summary = _build_action_gain_summary(metadata, description, decision_tags)
        actions.append({
            "index": int(payload.get("index", 0)),
            "kind": consult_action if phase_key == GameplayPhase.CONSULT else payload.get("type", ""),
            "canonical_name": label,
            "label": label,
            "description": description,
            "available": bool(payload.get("available", True)),
            "availability_reason": unavailable_reason,
            "decision_tags": decision_tags,
            "cost_summary": cost_summary,
            "gain_summary": gain_summary,
        })
    return actions


def _infer_action_decision_tags(
    *,
    phase_key: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    description: str,
) -> list[str]:
    """把候选动作抽象成可复用的功能维度，供日志和复盘检索使用。"""
    text = "；".join(
        part
        for part in (
            str(payload.get("type") or ""),
            str(metadata.get("candidate_type") or ""),
            str(metadata.get("rl_action_type") or ""),
            str(metadata.get("category") or ""),
            str(description or ""),
            " / ".join(str(item or "") for item in metadata.get("effect_types", []) or []),
        )
        if part
    )
    tags: list[str] = []

    def add(tag: str, *needles: str) -> None:
        if tag in tags:
            return
        if not needles or any(needle and needle in text for needle in needles):
            tags.append(tag)

    if phase_key == GameplayPhase.SCHEDULE:
        add("parameter_growth_value", "lesson", "レッスン", "授業", "vocal", "dance", "visual")
        add("stamina_recovery_value", ProduceText.REST, ProduceText.REST_ACTION, ProduceText.OUTING, ProduceText.GO_OUT, "回復")
        add("p_point_economy_value", "Pポイント", "P点", "P")
        add("fan_vote_progress_value", ProduceText.BUSINESS, "営業", "fan", "投票")
        add("deck_quality_value", ProduceText.CONSULT, "相談", "強化", "削除")
    elif phase_key == GameplayPhase.SKILL_REWARD:
        add("deck_quality_value")
    elif phase_key == GameplayPhase.CONSULT:
        add("deck_quality_value", "consult", "強化", "削除", "ProduceCard", "skill")
        add("inventory_flexibility_value", "ProduceDrink", "P飲料", "Pドリンク")
        add("p_point_economy_value", "価格", "price", "P")
    elif phase_key in {GameplayPhase.P_DRINK, GameplayPhase.ITEM_SELECT}:
        add("inventory_flexibility_value")

    add("immediate_score_gain", "Score", ProduceText.PARAMETER_UP_INCREASE, "スコア", "得分", "パラメータ")
    add("engine_setup_value", ProduceText.GOOD_CONDITION, ProduceText.CONCENTRATION, ProduceText.GOOD_IMPRESSION, ProduceText.YARUKI, ProduceText.FULL_POWER_POINT, ProduceText.FULL_POWER)
    add("stability_value", ProduceText.GENKI, ProduceText.STAMINA_RECOVERY, ProduceText.BLOCK, "回復", "軽減", "削減")
    add("stamina_recovery_value", ProduceText.STAMINA_RECOVERY, "体力回復", "回復")
    add("deck_quality_value", "強化", "削除", "交換", "生成", "draw", "手札")
    add("gate_preparation_value", "gate", "門", "試験", "オーディション", "audition", "選抜", "投票")
    return tags


def _build_action_cost_summary(metadata: dict[str, Any], description: str) -> str:
    parts: list[str] = []
    price = str(metadata.get("price") or "").strip()
    if price:
        parts.append(f"{price}P")
    cost = metadata.get("cost")
    if cost:
        parts.append(f"体力{int(cost)}")
    stamina_cost = metadata.get("stamina_cost")
    if stamina_cost is not None:
        parts.append(f"体力{stamina_cost}")
    if not parts and ("消耗" in description or "消費" in description):
        parts.append("描述中含资源消耗")
    return " / ".join(parts)


def _build_action_gain_summary(
    metadata: dict[str, Any],
    description: str,
    decision_tags: list[str],
) -> str:
    if not description:
        return ""
    tag_text = " / ".join(decision_tags[:4])
    rarity = str(metadata.get("rarity") or "")
    prefix = rarity.rsplit("_", 1)[-1] if rarity else ""
    parts = [part for part in (prefix, tag_text) if part]
    return "；".join(parts)


def _decision_family_for_phase(phase: str) -> str:
    phase_key = phase.value if hasattr(phase, "value") else str(phase)
    if phase_key == GameplayPhase.SCHEDULE:
        return "schedule"
    if phase_key == GameplayPhase.SKILL_REWARD:
        return "reward"
    if phase_key in {GameplayPhase.P_DRINK, GameplayPhase.ITEM_SELECT}:
        return "inventory"
    if phase_key == GameplayPhase.CONSULT:
        return "consult"
    if phase_key in {GameplayPhase.LESSON, GameplayPhase.EXAM}:
        return "combat"
    return phase_key or "unknown"


def _build_decision_explanation(
    *,
    phase: str,
    position: str,
    llm_snapshot: dict[str, Any],
    stage_context: dict[str, Any],
    llm_actions: list[dict[str, Any]],
    legal_actions: list[int],
) -> dict[str, Any]:
    """构建解释型日志字段；这是决策输入摘要，不伪造模型事后理由。"""
    planning = dict(llm_snapshot.get("planning", {}) or {})
    next_gate = dict(planning.get("next_gate", {}) or {})
    resource_pressure = dict(planning.get("resource_pressure", {}) or {})
    time_pressure = dict(planning.get("time_pressure", {}) or {})
    build_plan = dict(planning.get("build_plan", {}) or {})
    deck_plan = dict(planning.get("deck_plan", {}) or {})
    inventory_plan = dict(planning.get("inventory_plan", {}) or {})

    available = [action for action in llm_actions if bool(action.get("available", True))]
    unavailable = [action for action in llm_actions if not bool(action.get("available", True))]
    candidate_roles: dict[str, int] = {}
    for action in available:
        for tag in list(action.get("decision_tags") or []):
            candidate_roles[tag] = candidate_roles.get(tag, 0) + 1

    goal_parts = [
        str(dict(planning.get("current_objectives", {}) or {}).get("summary") or ""),
        str(next_gate.get("readiness_summary") or ""),
        str(stage_context.get("label") or ""),
    ]
    conditions = {
        "phase": phase,
        "position": position,
        "week": llm_snapshot.get("week"),
        "remaining_weeks": llm_snapshot.get("remaining_weeks"),
        "remaining_turns": llm_snapshot.get("remaining"),
        "stamina": llm_snapshot.get("stamina"),
        "max_stamina": llm_snapshot.get("max_stamina"),
        "p_point": llm_snapshot.get("p_point"),
        "score": llm_snapshot.get("score"),
        "target": llm_snapshot.get("target"),
        "fan_votes": llm_snapshot.get("fan_votes"),
        "next_gate": next_gate,
        "candidate_role_counts": candidate_roles,
    }
    constraints: list[str] = []
    if unavailable:
        constraints.append(
            "不可选动作: "
            + " / ".join(
                f"{action.get('index')}={action.get('availability_reason') or 'unavailable'}"
                for action in unavailable[:8]
            )
        )
    if legal_actions:
        constraints.append("只能选择可选动作编号: " + ", ".join(str(index) for index in legal_actions))
    if time_pressure.get("summary"):
        constraints.append(str(time_pressure.get("summary")))
    if resource_pressure.get("summary"):
        constraints.append(str(resource_pressure.get("summary")))

    comparison = [
        {
            "index": action.get("index"),
            "canonical_name": action.get("canonical_name") or action.get("label"),
            "decision_tags": list(action.get("decision_tags") or []),
            "cost_summary": str(action.get("cost_summary") or ""),
            "gain_summary": str(action.get("gain_summary") or ""),
        }
        for action in available[:12]
    ]
    trace = [
        f"decision_family={_decision_family_for_phase(phase)}",
        f"legal_count={len(legal_actions)}",
        f"unavailable_count={len(unavailable)}",
    ]
    if next_gate.get("weeks_until_gate") is not None:
        trace.append(f"weeks_until_gate={next_gate.get('weeks_until_gate')}")
    if resource_pressure.get("key"):
        trace.append(f"resource_pressure={resource_pressure.get('key')}")
    if time_pressure.get("key"):
        trace.append(f"time_pressure={time_pressure.get('key')}")

    return {
        "decision_family": _decision_family_for_phase(phase),
        "decision_goal": "；".join(part for part in goal_parts if part),
        "decision_constraints": constraints,
        "decision_conditions": conditions,
        "comparison_summary": comparison,
        "why_this_action": "",
        "why_not_others": "",
        "decision_rule_trace": trace,
        "effective_when": "当前目标、资源压力、时间窗口和候选功能角色相近时可参考。",
        "ineffective_when": "候选可用性、剩余周/回合、下个门槛或资源压力明显不同时不应直接套用。",
        "build_plan_key": str(build_plan.get("key") or build_plan.get("summary") or ""),
        "deck_plan_key": str(deck_plan.get("key") or deck_plan.get("summary") or ""),
        "inventory_plan_key": str(inventory_plan.get("key") or inventory_plan.get("summary") or ""),
        "next_gate_key": str(next_gate.get("gate_type") or next_gate.get("gate_label") or ""),
        "resource_pressure_key": str(resource_pressure.get("key") or resource_pressure.get("summary") or ""),
        "route_bias_key": str(dict(planning.get("route_bias", {}) or {}).get("idol_plan_type") or ""),
        "candidate_signature": "|".join(
            str(action.get("canonical_name") or action.get("label") or action.get("index"))
            for action in llm_actions
        ),
        "candidate_role_signature": "|".join(sorted(candidate_roles)),
    }


def _blocked_battle_card_keys(
    ctx: "ProduceContext",
    *,
    phase: str,
    llm_snapshot: dict[str, Any],
) -> set[str]:
    """处理blocked、battle、卡牌、keys并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        phase: 当前 gameplay 阶段标识。
        llm_snapshot: 用于提供llm、snapshot相关输入。

    Returns:
        set[str]: 返回值类型见注解。
    """
    blocked_state = dict(ctx.handler_state.get("battle_blocked_cards", {}) or {})
    current_marker = (
        str(phase or ""),
        int(ctx.current_week or 0),
        int(llm_snapshot.get("remaining") or -1),
    )
    if blocked_state.get("turn_marker") != current_marker:
        ctx.handler_state.pop("battle_blocked_cards", None)
        return set()
    return {
        str(key)
        for key in blocked_state.get("keys", [])
        if str(key or "").strip()
    }


def _zero_resource_dependency_reason(
    description: str,
    *,
    resources: dict[str, Any],
) -> str:
    """生成`zero_resource_dependency_reason`。"""
    normalized = fullwidth_to_halfwidth(str(description or ""))
    if not normalized:
        return ""
    for pattern in _PERCENT_BASED_RESOURCE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        resource_label = str(match.group(1) or "")
        resource_key = _SNAPSHOT_RESOURCE_KEY_BY_LABEL.get(resource_label, "")
        if not resource_key:
            continue
        if int(resources.get(resource_key) or 0) <= 0:
            return (
                f"当前{resource_label}=0，这张牌的主要效果依赖该资源，"
                "当前回合先不要使用"
            )
    return ""


def _insufficient_cost_reason(
    metadata: dict[str, Any],
    *,
    llm_snapshot: dict[str, Any],
) -> str:
    """生成`insufficient_cost_reason`。"""
    cost = int(metadata.get("cost") or 0)
    if cost <= 0:
        return ""
    current_stamina = int(llm_snapshot.get("stamina") or 0)
    current_genki = int(
        ((llm_snapshot.get("resources") or {}).get("block"))
        or llm_snapshot.get("genki")
        or 0
    )
    description = str(metadata.get("description") or "")
    direct_stamina_only = any(
        token in description
        for token in ProduceText.DIRECT_STAMINA_COST_HINT_TOKENS
    )
    available_cost_budget = current_stamina if direct_stamina_only else current_stamina + current_genki
    if available_cost_budget >= cost:
        return ""
    if direct_stamina_only or current_genki <= 0:
        return f"当前体力只有{current_stamina}，但这张牌需要消耗{cost}体力，当前无法使用"
    return (
        f"当前体力只有{current_stamina}，元气只有{current_genki}，可用于支付的总量只有"
        f"{available_cost_budget}，但这张牌需要消耗{cost}体力，当前无法使用"
    )


def _parse_play_limit_remaining(llm_snapshot: dict[str, Any]) -> int:
    """解析本回合剩余出牌次数；缺失时默认 1，显式 0 必须保留。"""
    raw_value = llm_snapshot.get("play_limit_remaining")
    if raw_value is None:
        return 1
    normalized = fullwidth_to_halfwidth(str(raw_value)).strip()
    if not normalized:
        return 1
    match = re.search(r"\d+", normalized)
    if match is None:
        return 1
    return int(match.group())


def _annotate_battle_candidate_availability(
    ctx: "ProduceContext",
    *,
    phase: str,
    candidate_payloads: list[dict[str, Any]],
    llm_snapshot: dict[str, Any],
) -> None:
    """补充标注battle、候选项、availability并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        phase: 当前 gameplay 阶段标识。
        candidate_payloads: 用于提供候选项、payloads相关输入。
        llm_snapshot: 用于提供llm、snapshot相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    phase_key = phase.value if hasattr(phase, "value") else str(phase)
    if phase_key not in {GameplayPhase.LESSON, GameplayPhase.EXAM}:
        return
    blocked_keys = _blocked_battle_card_keys(ctx, phase=phase_key, llm_snapshot=llm_snapshot)
    resources = dict(llm_snapshot.get("resources", {}) or {})
    for payload in candidate_payloads:
        action_id = str(payload.get("id") or "")
        metadata = dict(payload.get("metadata", {}) or {})
        if is_produce_drink_action_id(action_id):
            unavailable_reason = ""
            if not str(payload.get("db_id") or "").strip():
                unavailable_reason = str(metadata.get("unavailable_reason") or "未识别出饮料 db_id，当前不能使用该饮料").strip()
            elif not bool(payload.get("available", True)):
                unavailable_reason = str(metadata.get("unavailable_reason") or "当前饮料不可用").strip()
            elif bool(metadata.get("identity_unresolved", False)):
                unavailable_reason = str(metadata.get("unavailable_reason") or "饮料识别未完成，当前不能使用该饮料").strip()
            elif bool(metadata.get("probe_failed", False)):
                unavailable_reason = str(metadata.get("unavailable_reason") or "饮料详情探测失败，当前不能使用该饮料").strip()
            elif bool(metadata.get("modal_open_timeout", False)):
                unavailable_reason = str(metadata.get("unavailable_reason") or "未稳定检测到饮料详情模态，当前不能使用该饮料").strip()
            elif bool(metadata.get("modal_title_mismatch", False)):
                unavailable_reason = str(metadata.get("unavailable_reason") or "当前模态不是饮料详情，当前不能使用该饮料").strip()
            if unavailable_reason:
                payload["available"] = False
                payload["unavailable_reason"] = unavailable_reason
                metadata["available"] = False
                metadata["unavailable_reason"] = unavailable_reason
                payload["metadata"] = metadata
            continue
        if not is_produce_card_action_id(action_id):
            continue
        description = str(
            metadata.get("description")
            or payload.get("name")
            or payload.get("label")
            or ""
        )
        candidate_keys = {
            str(value)
            for value in (
                action_id,
                payload.get("db_id"),
                payload.get("name"),
                payload.get("label"),
            )
            if str(value or "").strip()
        }
        unavailable_reason = ""
        if blocked_keys and candidate_keys & blocked_keys:
            unavailable_reason = "上一轮已确认当前条件下效果不会发动，本回合先不要再用这张牌"
        elif not bool(payload.get("available", True)):
            unavailable_reason = str(metadata.get("unavailable_reason") or "").strip()
        else:
            unavailable_reason = _insufficient_cost_reason(
                metadata,
                llm_snapshot=llm_snapshot,
            )
        if not unavailable_reason:
            unavailable_reason = _zero_resource_dependency_reason(
                description,
                resources=resources,
            )
        if not unavailable_reason:
            continue
        payload["available"] = False
        payload["unavailable_reason"] = unavailable_reason
        metadata["available"] = False
        metadata["unavailable_reason"] = unavailable_reason
        payload["metadata"] = metadata


def _build_llm_snapshot(
    ctx: "ProduceContext",
    *,
    phase: str,
    position: str,
    hud_state: dict[str, Any],
    resolved_entities: list[dict[str, Any]],
    stage_context: dict[str, Any],
) -> dict[str, Any]:
    """构建llm、snapshot并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        phase: 当前 gameplay 阶段标识。
        position: 当前阶段下的细分画面位置标识。
        hud_state: 用于提供HUD、状态相关输入。
        resolved_entities: 用于提供resolved、entities相关输入。
        stage_context: 用于提供stage、context相关输入。

    Returns:
        dict: 结构化结果字典。
    """
    phase_key = phase.value if hasattr(phase, "value") else str(phase)
    known_deck = _build_current_deck_snapshot(ctx)
    hand_entries = _build_hand_snapshot(resolved_entities) if phase_key in {GameplayPhase.LESSON, GameplayPhase.EXAM} else []
    virtual_state = _sync_virtual_battle_state(
        ctx,
        hud_state=hud_state,
        known_deck=known_deck,
        observed_hand=hand_entries,
    ) if phase_key in {GameplayPhase.LESSON, GameplayPhase.EXAM} else None
    idol_plan = _current_idol_plan_payload(ctx)
    if virtual_state is not None:
        hand_entries = list(virtual_state["zones"]["hand"])
    known_drinks = _build_drink_snapshot(ctx.recognized_p_drinks)
    current_deck_zone = list(virtual_state["zones"]["deck"]) if virtual_state is not None else known_deck
    grave_cards = list(virtual_state["zones"]["grave"]) if virtual_state is not None else []
    hold_cards = list(virtual_state["zones"]["hold"]) if virtual_state is not None else []
    lost_cards = list(virtual_state["zones"]["lost"]) if virtual_state is not None else []
    deck_cards = known_deck
    deck_cards_for_counts = current_deck_zone or known_deck
    offensive_counts = {
        "hand": _count_offensive_snapshot_cards(hand_entries),
        "deck": _count_offensive_snapshot_cards(deck_cards_for_counts),
        "grave": _count_offensive_snapshot_cards(grave_cards),
        "hold": _count_offensive_snapshot_cards(hold_cards),
    }
    deck_summary = _build_snapshot_deck_summary(deck_cards)
    reshuffle_hint = _build_snapshot_reshuffle_hint(
        deck_cards=deck_cards_for_counts,
        grave_cards=grave_cards,
        offensive_counts=offensive_counts,
    )
    # 优先使用本帧观测到的 target_score，未观测时回退到 ctx 缓存的上次值
    effective_target = hud_state.get("target_score") or getattr(ctx, "hud_target_score", 0) or 0

    # ── 课程进度圆圈信息 ──
    progress_circle = hud_state.get("progress_circle")  # _parse_progress_circle 的结果
    if progress_circle is not None:
        # 进度圆圈模式: score=0, 使用 remaining_to_clear / remaining_to_perfect
        clear_achieved = progress_circle["clear_achieved"]
        remaining_to_clear = progress_circle["remaining_to_clear"]
        remaining_to_perfect = progress_circle["remaining_to_perfect"]
    else:
        clear_achieved = None
        remaining_to_clear = 0
        remaining_to_perfect = 0

    produce_goals = _build_produce_goal_snapshot(ctx)
    schedule_context = _build_schedule_context_snapshot(ctx)
    remaining_weeks = _compute_remaining_weeks(ctx)
    parameter_priority = _build_parameter_priority(ctx)
    parameter_stats = _build_parameter_stats_payload(ctx)
    produce_items = _build_produce_item_snapshot(ctx)
    remaining_turns = int(hud_state.get("remaining_turns") or 0) if phase_key in {GameplayPhase.LESSON, GameplayPhase.EXAM} else None
    planning = _build_planning_snapshot(
        ctx,
        phase_key=phase_key,
        hud_state=hud_state,
        known_drinks=known_drinks,
        produce_items=produce_items,
        parameter_stats=parameter_stats,
        hand_entries=hand_entries,
        deck_cards=current_deck_zone,
        offensive_counts=offensive_counts,
        produce_goals=produce_goals,
        schedule_context=schedule_context,
        remaining_weeks=remaining_weeks,
        remaining_turns=remaining_turns,
        parameter_priority=parameter_priority,
    )

    snapshot = {
        "phase": phase_key,
        "position": position,
        "stage_context": stage_context,
        "scenario": ctx.scenario,
        "difficulty": ctx.difficulty,
        "week": ctx.current_week,
        "remaining_weeks": remaining_weeks,
        "produce_goals": produce_goals,
        "planning": planning,
        "exam_criteria": produce_goals["exam_criteria"],
        "training_tasks": produce_goals["training_tasks"],
        "schedule_context": schedule_context,
        "schedule_history": schedule_context["history"],
        "future_schedule": schedule_context["future_schedule"],
        "schedule_history_summary": schedule_context["history_summary"],
        "idol_plan_type": idol_plan["type"],
        "idol_plan_label": idol_plan["label"],
        "idol_plan_focus": idol_plan["focus"],
        "idol_plan_description": idol_plan["description"],
        "parameter_priority": parameter_priority,
        "turn": virtual_state["turn_index"] if virtual_state is not None else (ctx.lesson_turns_played + 1 if hud_state.get("remaining_turns") else None),
        "remaining": hud_state.get("remaining_turns", 0),
        "max_turns": None,
        "battle_kind": "exam" if phase_key == GameplayPhase.EXAM else "lesson",
        "battle_kind_label": ProduceText.EXAM if phase_key == GameplayPhase.EXAM else ProduceText.LESSON,
        "score": hud_state.get("score", 0),
        "target": effective_target,
        "ratio": (
            f"{(hud_state.get('score', 0) / max(effective_target, 1)):.0%}"
            if effective_target
            else "未知"
        ),
        # 课程进度圆圈
        "clear_achieved": clear_achieved,
        "remaining_to_clear": remaining_to_clear,
        "remaining_to_perfect": remaining_to_perfect,
        "p_point": hud_state.get("p_point", 0),
        "stamina": hud_state.get("stamina", 0),
        "max_stamina": hud_state.get("max_stamina", 0),
        "genki": hud_state.get("genki", 0),
        "play_limit_remaining": virtual_state["play_limit_remaining"] if virtual_state is not None else None,
        "play_limit_total": virtual_state["play_limit_total_current"] if virtual_state is not None else None,
        "turn_color_label": hud_state.get("turn_color", ""),
        "turn_color_display_label": hud_state.get("turn_color", ""),
        "score_bonus_multiplier": hud_state.get("score_bonus", ""),
        "exam_ranking": hud_state.get("exam_ranking", ""),
        "parameter_stats": parameter_stats,
        "hand": hand_entries,
        "deck_count": len(deck_cards),
        "deck_summary": deck_summary,
        "deck_cards": deck_cards,
        "grave_cards": grave_cards,
        "hold_cards": hold_cards,
        "lost_cards": lost_cards,
        "zone_counts": {
            "deck": len(deck_cards),
            "grave": len(grave_cards),
            "hold": len(hold_cards),
            "lost": len(lost_cards),
        },
        "offensive_counts": offensive_counts,
        "reshuffle_hint": reshuffle_hint,
        "resources": {
            "parameter_buff": virtual_state["resources"]["parameter_buff"] if virtual_state is not None else "",
            "review": virtual_state["resources"]["review"] if virtual_state is not None else "",
            "aggressive": virtual_state["resources"]["aggressive"] if virtual_state is not None else "",
            "block": virtual_state["resources"]["block"] if virtual_state is not None else "",
            "lesson_buff": virtual_state["resources"]["lesson_buff"] if virtual_state is not None else "",
            "enthusiastic": virtual_state["resources"]["enthusiastic"] if virtual_state is not None else "",
            "full_power_point": virtual_state["resources"]["full_power_point"] if virtual_state is not None else "",
        },
        "stance_desc": "",
        "negatives": "",
        "active_effects": [],
        "active_enchants": [],
        "drinks": known_drinks,
        "available_drink_count": len(known_drinks),
        "used_drink_count": 0,
        "drink_total_count": len(known_drinks),
        "p_items": produce_items,
        "formation_abilities": _build_formation_ability_snapshot(ctx),
        "formation_events": _build_formation_event_snapshot(ctx),
        "gimmicks": "",
        "fan_votes": planning["scenario_rules"].get("fan_votes_current"),
        "total_counters": {
            "play_count": 0,
            "stamina_spent": "",
            "block_consumed": "",
        },
        "observability": {
            "deck_order_known": False,
            "resource_panel_parsed": virtual_state is not None,
            "exam_ranking_observed": bool(hud_state.get("exam_ranking")),
            "turn_color_observed": bool(hud_state.get("turn_color")),
            "drink_inventory_observed": bool(ctx.observability_state.get("drink_inventory_observed", False)),
            "empty_hand_observed": bool(ctx.observability_state.get("empty_hand_observed", False)),
        },
    }
    # 考试轮盘队列 + 加成倍率（供 LLM 规划后续回合）
    if phase_key == GameplayPhase.EXAM:
        _append_exam_snapshot_details(snapshot, ctx)
        wheel_info = dict(snapshot.get("exam_wheel") or {})
        wheel_bonus = _extract_first_int(str(wheel_info.get("bonus_pct") or ""))
        hud_bonus = _extract_first_int(str(snapshot.get("score_bonus_multiplier") or ""))
        wheel_confidence = str(wheel_info.get("confidence") or "low")
        if wheel_bonus > 0:
            should_override_bonus = hud_bonus <= 0
            if not should_override_bonus and hud_bonus > 0:
                mismatch = abs(hud_bonus - wheel_bonus)
                if wheel_confidence in {"high", "medium"} and mismatch >= 120:
                    should_override_bonus = True
                elif wheel_confidence == "low":
                    remaining_turns = int(snapshot.get("remaining") or 0)
                    prefix_suspect = False
                    if remaining_turns > 0:
                        hud_bonus_str = str(int(hud_bonus))
                        turn_prefix = str(int(remaining_turns))
                        if hud_bonus_str.startswith(turn_prefix) and len(hud_bonus_str) > len(turn_prefix):
                            tail = int(hud_bonus_str[len(turn_prefix):])
                            if abs(tail - wheel_bonus) <= 120:
                                prefix_suspect = True
                    if prefix_suspect or (hud_bonus >= 2000 and mismatch >= 400):
                        should_override_bonus = True
            if should_override_bonus:
                snapshot["score_bonus_multiplier"] = str(int(wheel_bonus))
                logger.debug(
                    "[考试倍率] 使用轮盘倍率覆盖 HUD 值: hud={} wheel={} (confidence={})",
                    hud_bonus,
                    wheel_bonus,
                    wheel_confidence,
                )
    # 相談 session 操作摘要（告知 LLM 本次相談已做了什么、还能做什么）
    if phase_key == GameplayPhase.CONSULT:
        snapshot["consult_session"] = _build_consult_session_summary(ctx)
    return snapshot


def build_decision_state(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    phase: str,
    position: str,
    candidates: Sequence[Any],
    reason: str = "decision",
) -> dict[str, Any]:
    """构建决策、状态并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        phase: 当前 gameplay 阶段标识。
        position: 当前阶段下的细分画面位置标识。
        candidates: 候选项列表，供策略或规则选择目标动作。
        reason: 用于提供reason相关输入。

    Returns:
        dict: 结构化结果字典。
    """
    hud_state = _extract_hud_state(app)
    _annotate_candidates(app, phase=phase, candidates=candidates)
    candidate_payloads = [serialize_candidate(candidate, phase=phase) for candidate in candidates]
    observed_inventory_drinks: list[dict[str, Any]] = []
    drink_inventory_observed = False
    if phase in {GameplayPhase.LESSON, GameplayPhase.EXAM}:
        observed_inventory_drinks, drink_inventory_observed = _observe_bottom_inventory_drinks(app)
    resolved_entities = [payload for payload in candidate_payloads if payload.get("db_id")]
    unresolved_entities = [payload for payload in candidate_payloads if not payload.get("db_id")]
    resolved_card_entities = [
        payload
        for payload in resolved_entities
        if is_produce_card_action_id(payload.get("id"))
    ]
    resolved_drink_entities = [
        payload
        for payload in resolved_entities
        if is_produce_drink_action_id(payload.get("id"))
    ]

    ctx.hud_stamina = hud_state["stamina"]
    if hud_state["max_stamina"] > 0:
        ctx.hud_max_stamina = hud_state["max_stamina"]
    if bool(hud_state.get("p_point_observed", True)):
        ctx.hud_p_point = hud_state["p_point"]
        ctx.consult_remaining_p_points = hud_state["p_point"]
    # 仅在本帧实际观测到目标分数时才更新；课程打牌画面 PC_TARGET 不可检测，
    # 保留上一次（日程页面等）观测到的值
    if hud_state.get("target_score_observed") and hud_state["target_score"] > 0:
        ctx.hud_target_score = hud_state["target_score"]
    ctx.economy_state = {
        "stamina": ctx.hud_stamina,
        "max_stamina": ctx.hud_max_stamina,
        "p_point": ctx.hud_p_point,
    }
    next_parameter_state = {
        "target_score": ctx.hud_target_score,
        "score": hud_state["score"],
        "remaining_turns": hud_state["remaining_turns"],
        "turn_color": hud_state["turn_color"],
        "score_bonus": hud_state["score_bonus"],
        "exam_ranking": hud_state["exam_ranking"],
    }
    for key in ("vocal", "dance", "visual"):
        value = hud_state.get(key)
        if value is not None:
            next_parameter_state[key] = value
        elif key in ctx.parameter_state:
            next_parameter_state[key] = ctx.parameter_state[key]
    parameter_limit = int(getattr(ctx, "parameter_growth_limit", 0) or 0)
    if parameter_limit > 0:
        for key in ("vocal", "dance", "visual"):
            next_parameter_state[f"{key}_max"] = parameter_limit
    ctx.parameter_state = next_parameter_state
    ctx.last_sync_reason = reason
    ctx.state_revision += 1

    if phase in {GameplayPhase.LESSON, GameplayPhase.EXAM} and bool(hud_state.get("genki_observed", False)):
        register_realtime_resource_snapshot(
            ctx,
            block=int(hud_state.get("genki") or 0),
        )

    if phase in {GameplayPhase.LESSON, GameplayPhase.EXAM}:
        ctx.recognized_hand_cards = resolved_card_entities
        ctx.card_zone_state = {
            "hand": resolved_card_entities,
        }
        # 战斗场景优先使用“本轮合法动作”中的饮料，避免底栏图标 CLIP 误识别导致
        # snapshot 与 legal actions 不一致；仅在没有合法饮料时回退到底栏库存观测。
        if resolved_drink_entities:
            ctx.recognized_p_drinks = list(resolved_drink_entities)
        elif drink_inventory_observed:
            reliable_inventory_drinks = [
                entry
                for entry in observed_inventory_drinks
                if str(entry.get("db_id") or "").strip()
            ]
            ctx.recognized_p_drinks = list(reliable_inventory_drinks)
        inventory_observed = bool(resolved_drink_entities) or bool(drink_inventory_observed)
        ctx.inventory_state = {
            **ctx.inventory_state,
            "p_drinks": list(ctx.recognized_p_drinks),
        }
        ctx.observability_state = {
            **ctx.observability_state,
            "draw_pile_order_known": False,
            "drink_inventory_observed": inventory_observed,
        }
    elif phase == GameplayPhase.P_DRINK:
        ctx.recognized_p_drinks = resolved_entities
        ctx.inventory_state = {
            **ctx.inventory_state,
            "p_drinks": resolved_entities,
        }
        ctx.observability_state = {
            **ctx.observability_state,
            "drink_inventory_observed": True,
        }
    elif phase == GameplayPhase.CONSULT:
        ctx.recognized_produce_items = resolved_entities
        ctx.inventory_state = {
            **ctx.inventory_state,
            "consult_candidates": resolved_entities,
        }

    ctx.unresolved_clip_entities = unresolved_entities

    snapshot = {
        "revision": ctx.state_revision,
        "phase": phase,
        "position": position,
        "week": ctx.current_week,
        "remaining_weeks": _compute_remaining_weeks(ctx),
        "scenario": ctx.scenario,
        "difficulty": ctx.difficulty,
        "produce_id": ctx.produce_id,
        "produce_group_id": ctx.produce_group_id,
        "economy": dict(ctx.economy_state),
        "parameters": dict(ctx.parameter_state),
        "inventory": dict(ctx.inventory_state),
        "card_zones": dict(ctx.card_zone_state),
        "observability": dict(ctx.observability_state),
        "candidates": candidate_payloads,
    }
    stage_context = _build_stage_context(
        phase=phase,
        position=position,
        hud_state=hud_state,
        candidate_payloads=candidate_payloads,
    )
    produce_goals = _build_produce_goal_snapshot(ctx)
    schedule_context = _build_schedule_context_snapshot(ctx)
    if produce_goals["summary"]:
        stage_context["produce_goal_summary"] = produce_goals["summary"]
    elif schedule_context["history_summary"]:
        stage_context["produce_goal_summary"] = schedule_context["history_summary"]
    if schedule_context["history_summary"]:
        stage_context["schedule_history_summary"] = schedule_context["history_summary"]
    if schedule_context["future_schedule"]:
        stage_context["future_schedule"] = schedule_context["future_schedule"]
    if schedule_context["history"]:
        stage_context["schedule_history"] = schedule_context["history"]
    # 将 P 手账日程数据注入 stage_context（供 LLM 未来规划参考）。
    if phase == GameplayPhase.SCHEDULE:
        notebook_entries = list(ctx.handler_state.get("p_notebook_schedule") or [])
        if notebook_entries:
            stage_context["future_schedule"] = notebook_entries
            stage_context["schedule_history"] = list(ctx.schedule_history or [])
            stage_context["schedule_history_summary"] = schedule_context["history_summary"]
            stage_context["produce_goal_summary"] = produce_goals["summary"]
    snapshot["stage_context"] = stage_context
    snapshot["llm_snapshot"] = _build_llm_snapshot(
        ctx,
        phase=phase,
        position=position,
        hud_state=hud_state,
        resolved_entities=(
            resolved_card_entities
            if phase in {GameplayPhase.LESSON, GameplayPhase.EXAM}
            else resolved_entities
        ),
        stage_context=stage_context,
    )
    battle_resources = dict(snapshot["llm_snapshot"].get("resources", {}) or {})
    current_stamina = int(snapshot["llm_snapshot"].get("stamina") or ctx.hud_stamina)
    current_max_stamina = int(snapshot["llm_snapshot"].get("max_stamina") or ctx.hud_max_stamina)
    current_genki = int(
        battle_resources.get("block")
        or snapshot["llm_snapshot"].get("genki")
        or 0
    )
    ctx.economy_state = {
        **ctx.economy_state,
        "battle_stamina": current_stamina,
        "battle_max_stamina": current_max_stamina,
        "battle_genki": current_genki,
    }
    ctx.parameter_state = {
        **ctx.parameter_state,
        "battle_resources": battle_resources,
        "battle_block": battle_resources.get("block", ""),
        "battle_review": battle_resources.get("review", ""),
        "battle_aggressive": battle_resources.get("aggressive", ""),
        "battle_parameter_buff": battle_resources.get("parameter_buff", ""),
    }
    snapshot["economy"] = dict(ctx.economy_state)
    snapshot["parameters"] = dict(ctx.parameter_state)
    _annotate_battle_candidate_availability(
        ctx,
        phase=phase,
        candidate_payloads=candidate_payloads,
        llm_snapshot=snapshot["llm_snapshot"],
    )
    for candidate, payload in zip(candidates, candidate_payloads, strict=False):
        metadata = _coerce_candidate_metadata(candidate)
        if "available" in payload:
            metadata["available"] = bool(payload.get("available", True))
        if payload.get("unavailable_reason"):
            metadata["unavailable_reason"] = str(payload.get("unavailable_reason") or "")
    snapshot["llm_actions"] = _build_llm_actions(
        candidate_payloads,
        phase=phase,
        position=position,
        stage_context=stage_context,
    )
    snapshot["legal_actions"] = [
        payload["index"]
        for payload in candidate_payloads
        if bool(payload.get("available", True))
    ]
    snapshot["decision_explanation"] = _build_decision_explanation(
        phase=phase,
        position=position,
        llm_snapshot=snapshot["llm_snapshot"],
        stage_context=stage_context,
        llm_actions=snapshot["llm_actions"],
        legal_actions=snapshot["legal_actions"],
    )
    snapshot["resolved_entities"] = resolved_entities
    snapshot["unresolved_entities"] = unresolved_entities
    ctx.handler_state["last_decision_state"] = snapshot
    return snapshot


def build_followup_decision_state(
    ctx: "ProduceContext",
    *,
    phase: str,
    position: str,
    candidates: Sequence[Any],
    reason: str = "followup_decision",
) -> dict[str, Any]:
    """基于上一份稳定快照，为覆盖层/确认页重组一份可供 LLM 使用的决策状态。"""
    phase_key = phase.value if hasattr(phase, "value") else str(phase)
    position_key = position.value if hasattr(position, "value") else str(position)
    previous_state = copy.deepcopy(ctx.handler_state.get("last_decision_state", {}) or {})
    previous_snapshot = dict(previous_state.get("llm_snapshot", {}) or {})
    candidate_payloads = [serialize_candidate(candidate, phase=phase_key) for candidate in candidates]
    stage_context = _build_stage_context(
        phase=phase_key,
        position=position_key,
        hud_state={
            "has_progress_hud": bool(previous_snapshot.get("stage_context", {}).get("is_schedule_context", False)),
            "recommend_action_kind": "",
            "recommend_action_text": "",
        },
        candidate_payloads=candidate_payloads,
    )

    produce_goals = _build_produce_goal_snapshot(ctx)
    schedule_context = _build_schedule_context_snapshot(ctx)
    if produce_goals["summary"]:
        stage_context["produce_goal_summary"] = produce_goals["summary"]
    elif schedule_context["history_summary"]:
        stage_context["produce_goal_summary"] = schedule_context["history_summary"]
    if schedule_context["history_summary"]:
        stage_context["schedule_history_summary"] = schedule_context["history_summary"]
    if schedule_context["future_schedule"]:
        stage_context["future_schedule"] = schedule_context["future_schedule"]
    if schedule_context["history"]:
        stage_context["schedule_history"] = schedule_context["history"]

    remaining_weeks = previous_snapshot.get("remaining_weeks") or _compute_remaining_weeks(ctx)
    parameter_priority = _build_parameter_priority(ctx)
    idol_plan = _current_idol_plan_payload(ctx)
    llm_snapshot = {
        **previous_snapshot,
        "phase": phase_key,
        "position": position_key,
        "stage_context": stage_context,
        "scenario": previous_snapshot.get("scenario", ctx.scenario),
        "difficulty": previous_snapshot.get("difficulty", ctx.difficulty),
        "week": previous_snapshot.get("week", ctx.current_week),
        "remaining_weeks": remaining_weeks,
        "produce_goals": produce_goals,
        "exam_criteria": produce_goals["exam_criteria"],
        "training_tasks": produce_goals["training_tasks"],
        "schedule_context": schedule_context,
        "schedule_history": schedule_context["history"],
        "future_schedule": schedule_context["future_schedule"],
        "schedule_history_summary": schedule_context["history_summary"],
        "idol_plan_type": previous_snapshot.get("idol_plan_type", idol_plan["type"]),
        "idol_plan_label": previous_snapshot.get("idol_plan_label", idol_plan["label"]),
        "idol_plan_focus": previous_snapshot.get("idol_plan_focus", idol_plan["focus"]),
        "idol_plan_description": previous_snapshot.get("idol_plan_description", idol_plan["description"]),
        "parameter_priority": previous_snapshot.get("parameter_priority", parameter_priority),
    }
    llm_snapshot.setdefault("fan_votes", None)
    previous_planning = dict(previous_snapshot.get("planning", {}) or {})
    planning = {
        **previous_planning,
        "scenario_rules": {
            **dict(previous_planning.get("scenario_rules", {}) or {}),
            **_scenario_rule_payload(ctx),
        },
        "current_objectives": {
            "produce_goal_summary": str(produce_goals.get("summary") or ""),
            "training_tasks": list(produce_goals.get("training_tasks") or []),
            "exam_criteria": list(produce_goals.get("exam_criteria") or []),
            "summary": str(produce_goals.get("summary") or ""),
        },
        "next_gate": {
            **dict(previous_planning.get("next_gate", {}) or {}),
            **_build_next_gate_snapshot(
                ctx,
                remaining_weeks=remaining_weeks,
                exam_criteria=list(produce_goals.get("exam_criteria") or []),
                training_tasks=list(produce_goals.get("training_tasks") or []),
            ),
        },
        "resource_pressure": dict(previous_planning.get("resource_pressure", {}) or {}),
        "time_pressure": dict(previous_planning.get("time_pressure", {}) or {}),
        "build_plan": dict(previous_planning.get("build_plan", {}) or {}),
        "deck_plan": dict(previous_planning.get("deck_plan", {}) or {}),
        "inventory_plan": dict(previous_planning.get("inventory_plan", {}) or {}),
        "recent_outcomes": list(previous_planning.get("recent_outcomes", []) or []),
        "future_schedule": list(schedule_context.get("future_schedule") or []),
        "schedule_history_summary": str(schedule_context.get("history_summary") or ""),
    }
    planning["route_bias"] = {
        **dict(previous_planning.get("route_bias", {}) or {}),
        "idol_plan_type": llm_snapshot.get("idol_plan_type", ""),
        "idol_plan_label": llm_snapshot.get("idol_plan_label", ""),
        "parameter_priority": llm_snapshot.get("parameter_priority", ""),
        "summary": "；".join(
            part for part in [
                str(llm_snapshot.get("idol_plan_label") or ""),
                f"属性成长优先级={llm_snapshot.get('parameter_priority')}" if str(llm_snapshot.get("parameter_priority") or "") else "",
            ] if part
        ),
    }
    planning["scenario_rules"].setdefault("weeks_until_gate", remaining_weeks)
    planning["scenario_rules"].setdefault("summary", "")
    planning["next_gate"].setdefault("weeks_until_gate", remaining_weeks)
    planning["next_gate"].setdefault("readiness_summary", "")
    planning["time_pressure"].setdefault("remaining_weeks", remaining_weeks)
    planning["time_pressure"].setdefault("summary", f"周窗口未知(剩余{remaining_weeks}周)" if remaining_weeks is not None else "")
    planning["resource_pressure"].setdefault("summary", "")
    planning["build_plan"].setdefault("summary", "")
    planning["deck_plan"].setdefault("summary", "")
    planning["inventory_plan"].setdefault("summary", "")
    planning["scenario_rules"]["audition_stage_current"] = str(
        getattr(ctx, "current_exam_type", "")
        or planning["scenario_rules"].get("audition_stage_current")
        or ""
    )
    planning["summary"] = "；".join(
        [
            part
            for part in (
                str(planning.get("current_objectives", {}).get("summary") or ""),
                str(planning.get("next_gate", {}).get("readiness_summary") or ""),
                str(planning.get("build_plan", {}).get("summary") or ""),
                str(planning.get("deck_plan", {}).get("summary") or ""),
                str(planning.get("inventory_plan", {}).get("summary") or ""),
            )
            if part
        ]
    )
    llm_snapshot["planning"] = planning
    llm_snapshot["fan_votes"] = planning["scenario_rules"].get("fan_votes_current")
    # 相談 session 摘要传递
    if phase_key == GameplayPhase.CONSULT:
        llm_snapshot["consult_session"] = _build_consult_session_summary(ctx)
        llm_snapshot["planning"]["consult_session"] = llm_snapshot["consult_session"]
    else:
        llm_snapshot.setdefault("consult_session", previous_snapshot.get("consult_session"))
        llm_snapshot["planning"]["consult_session"] = llm_snapshot.get("consult_session")
    if phase_key in {GameplayPhase.LESSON, GameplayPhase.EXAM}:
        llm_snapshot.setdefault(
            "battle_kind",
            "exam" if phase_key == GameplayPhase.EXAM else "lesson",
        )
        llm_snapshot.setdefault(
            "battle_kind_label",
            ProduceText.EXAM if phase_key == GameplayPhase.EXAM else ProduceText.LESSON,
        )
        llm_snapshot.setdefault("turn", None)
        llm_snapshot.setdefault("remaining", None)
        llm_snapshot.setdefault("max_turns", None)
        llm_snapshot.setdefault("score", 0)
        llm_snapshot.setdefault("target", 0)
        llm_snapshot.setdefault("ratio", "0%")
        llm_snapshot.setdefault("stamina", 0)
        llm_snapshot.setdefault("max_stamina", 0)
        llm_snapshot.setdefault("genki", 0)
        llm_snapshot.setdefault("play_limit_remaining", None)
        llm_snapshot.setdefault("play_limit_total", None)
        llm_snapshot.setdefault("turn_color_label", "")
        llm_snapshot.setdefault("turn_color_display_label", "")
        llm_snapshot.setdefault("score_bonus_multiplier", "")
        llm_snapshot.setdefault("exam_ranking", "")
        llm_snapshot.setdefault("deck_count", 0)
        llm_snapshot.setdefault("deck_summary", "未知")
        llm_snapshot.setdefault("reshuffle_hint", "")
        llm_snapshot.setdefault("stance_desc", "")
        llm_snapshot.setdefault("negatives", "")
        llm_snapshot.setdefault("gimmicks", "")
        llm_snapshot.setdefault("available_drink_count", 0)
        llm_snapshot.setdefault("used_drink_count", 0)
        llm_snapshot.setdefault("drink_total_count", 0)
        llm_snapshot.setdefault("hand", [])
        llm_snapshot.setdefault("deck_cards", [])
        llm_snapshot.setdefault("grave_cards", [])
        llm_snapshot.setdefault("hold_cards", [])
        llm_snapshot.setdefault("lost_cards", [])
        llm_snapshot.setdefault("active_effects", [])
        llm_snapshot.setdefault("active_enchants", [])
        llm_snapshot.setdefault("drinks", [])
        llm_snapshot.setdefault("p_items", [])
        llm_snapshot.setdefault("formation_abilities", _build_formation_ability_snapshot(ctx))
        llm_snapshot.setdefault("formation_events", _build_formation_event_snapshot(ctx))
        parameter_stats = {
            "vocal": "",
            "dance": "",
            "visual": "",
            "vocal_max": int(getattr(ctx, "parameter_growth_limit", 0) or 0) or "",
            "dance_max": int(getattr(ctx, "parameter_growth_limit", 0) or 0) or "",
            "visual_max": int(getattr(ctx, "parameter_growth_limit", 0) or 0) or "",
            **dict(llm_snapshot.get("parameter_stats", {}) or {}),
        }
        llm_snapshot["parameter_stats"] = parameter_stats
        zone_counts = {
            "deck": 0,
            "grave": 0,
            "hold": 0,
            "lost": 0,
            **dict(llm_snapshot.get("zone_counts", {}) or {}),
        }
        llm_snapshot["zone_counts"] = zone_counts
        offensive_counts = {
            "hand": 0,
            "deck": 0,
            "grave": 0,
            "hold": 0,
            **dict(llm_snapshot.get("offensive_counts", {}) or {}),
        }
        llm_snapshot["offensive_counts"] = offensive_counts
        resources = {
            "parameter_buff": "",
            "review": "",
            "aggressive": "",
            "block": "",
            "lesson_buff": "",
            "enthusiastic": "",
            "full_power_point": "",
            **dict(llm_snapshot.get("resources", {}) or {}),
        }
        llm_snapshot["resources"] = resources
        total_counters = {
            "play_count": 0,
            "stamina_spent": "",
            "block_consumed": "",
            **dict(llm_snapshot.get("total_counters", {}) or {}),
        }
        llm_snapshot["total_counters"] = total_counters
        observability = {
            "deck_order_known": False,
            "resource_panel_parsed": False,
            "exam_ranking_observed": False,
            "turn_color_observed": False,
            "drink_inventory_observed": False,
            "empty_hand_observed": False,
            **dict(llm_snapshot.get("observability", {}) or {}),
        }
        llm_snapshot["observability"] = observability


    snapshot = {
        **previous_state,
        "phase": phase_key,
        "position": position_key,
        "candidates": candidate_payloads,
        "stage_context": stage_context,
        "llm_snapshot": llm_snapshot,
    }
    snapshot["llm_actions"] = _build_llm_actions(
        candidate_payloads,
        phase=phase_key,
        position=position_key,
        stage_context=stage_context,
    )
    snapshot["legal_actions"] = [
        payload["index"]
        for payload in candidate_payloads
        if bool(payload.get("available", True))
    ]
    snapshot["decision_explanation"] = _build_decision_explanation(
        phase=phase_key,
        position=position_key,
        llm_snapshot=llm_snapshot,
        stage_context=stage_context,
        llm_actions=snapshot["llm_actions"],
        legal_actions=snapshot["legal_actions"],
    )
    ctx.last_sync_reason = reason
    ctx.handler_state["last_decision_state"] = snapshot
    return snapshot
