"""算法决策策略 — 重构版本。

基于游戏数据库的可信数据源进行决策：
1. 所有复杂数据使用dataclass传递
2. 所有枚举值使用Enum类型
3. 使用效果评估体系统一评分
4. 禁止硬编码坐标、强模板匹配
"""

from __future__ import annotations

from collections import Counter
import re
from typing import TYPE_CHECKING, Any, Sequence

from src.constants.game.produce_enums import (
    AttributeType,
    ExamEffectType,
    ProduceCardCategory,
    ProducePlanType,
)
from src.utils.game_database_tools import (
    GakumasDatabase_IdolCardDataUtils,
    GakumasDatabase_ProduceCardDataUtils,
    GakumasDatabase_ProduceDrinkDataUtils,
    GakumasDatabase_ProduceItemDataUtils,
)
from src.utils.i18n_tools import i18n_text
from src.utils.logger import logger

from .algo_effect_evaluator import EffectEvaluator
from .algo_strategy_types import (
    ActionScoreBreakdown,
    BattleContext,
    CandidateAction,
    CardEffectInfo,
    DecisionResult,
    DeckCardState,
    DeckState,
    DrinkInfo,
    ExamBonusInfo,
    GatePlan,
    IdolPlanInfo,
    InventoryState,
    ItemInfo,
    MacroPlan,
    ParameterState,
    ResourceState,
    ScheduleContext,
    StrategyInput,
    TurnInfo,
)

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


_PLAN1 = ProducePlanType.PLAN1
_PLAN2 = ProducePlanType.PLAN2
_PLAN3 = ProducePlanType.PLAN3


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_plan_type(raw_plan: Any) -> ProducePlanType:
    try:
        return ProducePlanType(str(raw_plan))
    except ValueError:
        return ProducePlanType.PLAN1


def _parse_attribute_type(raw_attr: Any) -> AttributeType | None:
    attr_lower = str(raw_attr or "").lower()
    if "vocal" in attr_lower or "ボーカル" in attr_lower:
        return AttributeType.VOCAL
    if "dance" in attr_lower or "ダンス" in attr_lower:
        return AttributeType.DANCE
    if "visual" in attr_lower or "ビジュアル" in attr_lower:
        return AttributeType.VISUAL
    return None


def _extract_effect_types(entity: Any) -> tuple[str, ...]:
    effect_types: list[str] = []
    groups = getattr(entity, "effectGroupClss", None) or []
    for group in groups:
        exam_effect_types = getattr(group, "examEffectTypes", None) or []
        effect_types.extend(str(effect_type) for effect_type in exam_effect_types if str(effect_type).strip())
    return tuple(effect_types)


def _build_idol_plan_info(ctx: ProduceContext) -> IdolPlanInfo:
    idol_card = getattr(ctx, "selected_idol_card", None)
    if idol_card is None:
        target_id = getattr(ctx, "target_idol_card_id", "") or ""
        if target_id:
            try:
                db = GakumasDatabase_IdolCardDataUtils()
                idol_card = db.get_by_id(target_id)
                if idol_card is None:
                    idol_card = db.get_by_raw_id(target_id)
            except Exception:
                idol_card = None
    if idol_card is None:
        raise ValueError(i18n_text("backend.task.idolCardInfoNotFound", fallback="无法获取偶像卡信息"))
    return IdolPlanInfo(
        plan_type=_parse_plan_type(getattr(idol_card, "planType", "")),
        vocal_growth_permil=_coerce_int(getattr(idol_card, "produceVocalGrowthRatePermil", 0)),
        dance_growth_permil=_coerce_int(getattr(idol_card, "produceDanceGrowthRatePermil", 0)),
        visual_growth_permil=_coerce_int(getattr(idol_card, "produceVisualGrowthRatePermil", 0)),
    )


def _build_exam_bonus_info(ctx: ProduceContext) -> ExamBonusInfo | None:
    try:
        from src.core.tasks.producer_challenge.gameplay.exam_prep import get_exam_prep_bonuses

        raw = get_exam_prep_bonuses(ctx)
    except Exception:
        raw = None
    if not raw:
        return None
    return ExamBonusInfo(
        vocal_bonus_permil=_coerce_int(raw.get("vocal_bonus_pct")),
        dance_bonus_permil=_coerce_int(raw.get("dance_bonus_pct")),
        visual_bonus_permil=_coerce_int(raw.get("visual_bonus_pct")),
    )


def _build_resource_state(snapshot: dict[str, Any]) -> ResourceState:
    resources = dict(snapshot.get("resources") or {})
    return ResourceState(
        stamina=_coerce_int(snapshot.get("stamina")),
        max_stamina=max(_coerce_int(snapshot.get("max_stamina"), 1), 1),
        score=_coerce_int(snapshot.get("score")),
        parameter_buff=_coerce_int(resources.get("parameter_buff")),
        concentration=_coerce_int(resources.get("concentration") or resources.get("lesson_buff")),
        review=_coerce_int(resources.get("review")),
        aggressive=_coerce_int(resources.get("aggressive")),
        block=_coerce_int(resources.get("block")),
        enthusiastic=_coerce_int(resources.get("enthusiastic")),
        full_power_point=_coerce_int(resources.get("full_power_point")),
        lesson_buff=_coerce_int(resources.get("lesson_buff")),
    )


def _build_parameter_state(snapshot: dict[str, Any]) -> ParameterState:
    param_stats = dict(snapshot.get("parameter_stats") or {})
    return ParameterState(
        vocal_current=_coerce_int(param_stats.get("vocal")),
        vocal_max=_coerce_int(param_stats.get("vocal_max")),
        dance_current=_coerce_int(param_stats.get("dance")),
        dance_max=_coerce_int(param_stats.get("dance_max")),
        visual_current=_coerce_int(param_stats.get("visual")),
        visual_max=_coerce_int(param_stats.get("visual_max")),
    )


def _build_turn_info(snapshot: dict[str, Any]) -> TurnInfo:
    remaining = _coerce_int(snapshot.get("remaining"))
    if remaining <= 0:
        turn = _coerce_int(snapshot.get("turn"), 1)
        max_turns = _coerce_int(snapshot.get("max_turns"))
        remaining = max(max_turns - turn + 1, 0) if max_turns > 0 else 0
    turn_color = _parse_attribute_type(snapshot.get("turn_color_display_label") or snapshot.get("turn_color_label"))
    return TurnInfo(
        current_turn=_coerce_int(snapshot.get("turn"), 1),
        max_turns=_coerce_int(snapshot.get("max_turns")),
        remaining_turns=remaining,
        turn_color=turn_color,
    )


def _build_card_effect_info_from_db(card_id: str, upgrade_count: int = 0) -> CardEffectInfo | None:
    try:
        db = GakumasDatabase_ProduceCardDataUtils()
        card = db.get_by_id(f"{card_id}.{upgrade_count}")
        if card is None:
            card = db.get_by_raw_id(card_id)
        if card is None:
            return None
        return CardEffectInfo(
            card_id=str(card.id),
            card_name=str(getattr(card.localization, "name", "") if hasattr(card, "localization") else getattr(card, "name", "")),
            rarity=str(card.rarity),
            category=ProduceCardCategory(card.category) if card.category else ProduceCardCategory.UNKNOWN,
            upgrade_count=_coerce_int(getattr(card, "upgradeCount", upgrade_count)),
            stamina_cost=_coerce_int(getattr(card, "stamina", 0)),
            effect_types=_extract_effect_types(card),
            description="",
            plan_type=_parse_plan_type(getattr(card, "planType", "")),
        )
    except Exception as exc:
        logger.warning("[AlgoStrategy] 无法从数据库读取卡牌 {}: {}", card_id, exc)
        return None


def _build_drink_info_from_db(drink_id: str) -> DrinkInfo | None:
    try:
        db = GakumasDatabase_ProduceDrinkDataUtils()
        drink = db.get_by_id(drink_id) or db.get_by_raw_id(drink_id)
        if drink is None:
            return None
        return DrinkInfo(
            drink_id=str(drink.id),
            drink_name=str(getattr(drink.localization, "name", "") if hasattr(drink, "localization") else getattr(drink, "name", "")),
            rarity=str(drink.rarity),
            effect_types=_extract_effect_types(drink),
            description="",
            plan_type=_parse_plan_type(getattr(drink, "planType", "")),
        )
    except Exception as exc:
        logger.warning("[AlgoStrategy] 无法从数据库读取饮料 {}: {}", drink_id, exc)
        return None


def _build_item_info_from_db(item_id: str) -> ItemInfo | None:
    try:
        db = GakumasDatabase_ProduceItemDataUtils()
        item = db.get_by_id(item_id) or db.get_by_raw_id(item_id)
        if item is None:
            return None
        return ItemInfo(
            item_id=str(item.id),
            item_name=str(getattr(item.localization, "name", "") if hasattr(item, "localization") else getattr(item, "name", "")),
            rarity=str(item.rarity),
            effect_types=_extract_effect_types(item),
            description="",
            plan_type=_parse_plan_type(getattr(item, "planType", "")),
            is_exam_effect=bool(getattr(item, "isExamEffect", False)),
        )
    except Exception as exc:
        logger.warning("[AlgoStrategy] 无法从数据库读取物品 {}: {}", item_id, exc)
        return None


def _derive_candidate_kind(payload: dict[str, Any], phase: str) -> str:
    kind = str(payload.get("kind") or payload.get("type") or payload.get("entity_kind") or "")
    if kind:
        if kind == "delete":
            return "delete"
        return kind
    metadata = dict(payload.get("metadata") or {})
    action_id = str(payload.get("id") or payload.get("action_id") or "")
    if phase == "p_drink":
        limit_kind = str(metadata.get("p_drink_limit_kind") or "")
        if limit_kind:
            return limit_kind
    if phase == "consult":
        consult_action = str(metadata.get("consult_action") or "")
        if "remove" in consult_action:
            return "delete"
        if "enhance" in consult_action:
            return "enhance"
        if "exchange" in consult_action:
            return "exchange"
        if "exit" in consult_action:
            return "exit"
        if "remove" in action_id:
            return "delete"
        if "enhance" in action_id:
            return "enhance"
        if "exchange" in action_id:
            return "exchange"
        if "exit" in action_id:
            return "exit"
    if phase == "dialogue":
        return "dialogue"
    return ""


def _normalize_action_payload(payload: dict[str, Any], phase: str = "") -> CandidateAction:
    metadata = dict(payload.get("metadata") or {})
    return CandidateAction(
        index=_coerce_int(payload.get("index", -1)),
        action_id=str(payload.get("id") or payload.get("action_id") or ""),
        db_id=str(payload.get("db_id") or ""),
        name=str(payload.get("name") or payload.get("label") or payload.get("title") or ""),
        available=bool(payload.get("available", True)),
        kind=_derive_candidate_kind(payload, phase),
        metadata=metadata,
    )


def _build_gate_plan(snapshot: dict[str, Any], idol_plan: IdolPlanInfo) -> GatePlan:
    planning = dict(snapshot.get("planning") or {})
    next_gate = dict(planning.get("next_gate") or {})
    weeks_until_gate = max(_coerce_int(next_gate.get("weeks_until_gate")), 0)
    gate_type = str(next_gate.get("gate_type") or "")
    return GatePlan(
        gate_type=gate_type,
        weeks_until_gate=weeks_until_gate,
        is_key_window=weeks_until_gate <= 2,
        preserve_stamina=weeks_until_gate <= 1,
        preserve_p_points=gate_type.lower() == "consult",
        preferred_attributes=tuple(
            attr for attr, _ in idol_plan.attribute_priority if attr in {AttributeType.VOCAL, AttributeType.DANCE, AttributeType.VISUAL}
        ),
    )


def _build_deck_state(inventory: InventoryState) -> DeckState:
    cards = []
    all_cards = list(inventory.hand_cards) + list(inventory.deck_cards) + list(inventory.grave_cards) + list(inventory.hold_cards) + list(inventory.lost_cards)
    grouped: dict[str, int] = Counter(card.card_id for card in all_cards)
    zone_index = 0
    for zone, zone_cards in (
        ("hand", inventory.hand_cards),
        ("deck", inventory.deck_cards),
        ("grave", inventory.grave_cards),
        ("hold", inventory.hold_cards),
        ("lost", inventory.lost_cards),
    ):
        for card in zone_cards:
            duplicate_size = grouped.get(card.card_id, 1)
            if any(et in card.effect_types for et in (ExamEffectType.REVIEW.value, ExamEffectType.CONCENTRATION.value, ExamEffectType.PARAMETER_BUFF.value, ExamEffectType.FULL_POWER_POINT.value, ExamEffectType.ENTHUSIASTIC.value)):
                ownership_value = 1.0
            else:
                ownership_value = 0.3 if duplicate_size > 1 else 0.0
            if zone == "hand":
                play_value = 1.0
            elif zone == "deck":
                play_value = 0.4
            elif zone == "grave":
                play_value = 0.2
            else:
                play_value = 0.0
            cards.append(
                DeckCardState(
                    card=card,
                    zone=zone,
                    zone_index=zone_index,
                    duplicate_group_size=duplicate_size,
                    ownership_value=ownership_value,
                    play_value=play_value,
                )
            )
            zone_index += 1
    total_cards = len(cards)
    unique_cards = len({card.card.card_id for card in cards})
    duplicate_groups = sum(1 for count in grouped.values() if count > 1)
    recovery_cards = sum(1 for card in cards if any("Block" in et for et in card.card.effect_types))
    extra_play_cards = sum(1 for card in cards if any("PlayCountUp" in et for et in card.card.effect_types))
    engine_cards = sum(1 for card in cards if any(et in card.card.effect_types for et in (ExamEffectType.REVIEW.value, ExamEffectType.CONCENTRATION.value, ExamEffectType.PARAMETER_BUFF.value, ExamEffectType.FULL_POWER_POINT.value, ExamEffectType.ENTHUSIASTIC.value)))
    purity_score = unique_cards / max(total_cards, 1)
    engine_density = engine_cards / max(total_cards, 1)
    return DeckState(
        cards=tuple(cards),
        total_cards=total_cards,
        unique_cards=unique_cards,
        duplicate_groups=duplicate_groups,
        purity_score=purity_score,
        engine_density=engine_density,
        recovery_cards=recovery_cards,
        extra_play_cards=extra_play_cards,
    )


def _build_inventory_state(ctx: ProduceContext, snapshot: dict[str, Any]) -> InventoryState:
    hand_cards: list[CardEffectInfo] = []
    deck_cards: list[CardEffectInfo] = []
    grave_cards: list[CardEffectInfo] = []
    hold_cards: list[CardEffectInfo] = []
    lost_cards: list[CardEffectInfo] = []

    def _append_cards(target: list[CardEffectInfo], key: str) -> None:
        for card_data in list(snapshot.get(key) or []):
            card_id = str(card_data.get("id") or card_data.get("db_id") or "")
            if not card_id:
                continue
            card_info = _build_card_effect_info_from_db(card_id, _coerce_int(card_data.get("upgrade_count")))
            if card_info is not None:
                target.append(card_info)

    _append_cards(hand_cards, "hand")
    _append_cards(deck_cards, "deck_cards")
    _append_cards(grave_cards, "grave_cards")
    _append_cards(hold_cards, "hold_cards")
    _append_cards(lost_cards, "lost_cards")

    p_drinks: list[DrinkInfo] = []
    for drink_data in list(snapshot.get("p_drinks") or snapshot.get("drinks") or []):
        drink_id = str(drink_data.get("id") or drink_data.get("db_id") or "")
        if not drink_id:
            continue
        drink_info = _build_drink_info_from_db(drink_id)
        if drink_info is not None:
            p_drinks.append(drink_info)

    p_items: list[ItemInfo] = []
    for item_data in list(snapshot.get("p_items") or []):
        item_id = str(item_data.get("id") or item_data.get("db_id") or "")
        if not item_id:
            continue
        item_info = _build_item_info_from_db(item_id)
        if item_info is not None:
            p_items.append(item_info)

    p_points = _coerce_int(snapshot.get("p_points") or snapshot.get("p_point") or ctx.consult_remaining_p_points)
    return InventoryState(
        hand_cards=tuple(hand_cards),
        deck_cards=tuple(deck_cards),
        grave_cards=tuple(grave_cards),
        hold_cards=tuple(hold_cards),
        lost_cards=tuple(lost_cards),
        p_drinks=tuple(p_drinks),
        p_items=tuple(p_items),
        p_points=p_points,
    )


def _build_schedule_context(idol_plan: IdolPlanInfo, resources: ResourceState, parameters: ParameterState, inventory: InventoryState, gate_plan: GatePlan, snapshot: dict[str, Any]) -> ScheduleContext:
    return ScheduleContext(
        idol_plan=idol_plan,
        resources=resources,
        parameters=parameters,
        inventory=inventory,
        weeks_until_gate=gate_plan.weeks_until_gate,
        gate_type=gate_plan.gate_type,
        current_week=_coerce_int(snapshot.get("current_week") or snapshot.get("week")),
    )


def _build_battle_context(idol_plan: IdolPlanInfo, exam_bonus: ExamBonusInfo | None, resources: ResourceState, turn_info: TurnInfo, inventory: InventoryState, is_exam: bool) -> BattleContext:
    return BattleContext(
        idol_plan=idol_plan,
        exam_bonus=exam_bonus,
        resources=resources,
        turn_info=turn_info,
        inventory=inventory,
        is_exam=is_exam,
    )


def _build_macro_plan(gate_plan: GatePlan, resources: ResourceState | None, parameters: ParameterState | None, inventory: InventoryState) -> MacroPlan:
    purity_priority = 1.0 if inventory.deck_composition.total_cards >= 18 else 0.6
    should_prioritize_purity = inventory.deck_composition.purity_score < 0.8
    should_preserve_resources = gate_plan.is_key_window or (resources.stamina_ratio < 0.3 if resources else False)
    should_push_burst = bool(parameters and parameters.total_gap <= 120 and not should_preserve_resources)
    if gate_plan.weeks_until_gate <= 1:
        main_goal = "保线备战"
        secondary_goal = "保资源"
    elif should_prioritize_purity:
        main_goal = "去稀释"
        secondary_goal = "补核心引擎"
    else:
        main_goal = "补主轴"
        secondary_goal = "维持资源"
    return MacroPlan(
        main_goal=main_goal,
        secondary_goal=secondary_goal,
        gate_plan=gate_plan,
        should_prioritize_purity=should_prioritize_purity,
        should_preserve_resources=should_preserve_resources,
        should_push_burst=should_push_burst,
        safety_priority=1.0 if should_preserve_resources else 0.5,
        purity_priority=purity_priority,
    )


def _build_strategy_input(app: AppProcessor, ctx: ProduceContext, candidates: Sequence[Any], decision_state: dict[str, Any]) -> StrategyInput:
    snapshot = dict(decision_state.get("llm_snapshot") or {})
    phase = str(decision_state.get("phase") or snapshot.get("phase") or "")
    position = str(decision_state.get("position") or snapshot.get("position") or "")
    idol_plan = _build_idol_plan_info(ctx)
    exam_bonus = _build_exam_bonus_info(ctx) if phase == "exam" else None
    resources = _build_resource_state(snapshot)
    parameters = _build_parameter_state(snapshot)
    turn_info = _build_turn_info(snapshot)
    inventory = _build_inventory_state(ctx, snapshot)
    gate_plan = _build_gate_plan(snapshot, idol_plan)
    macro_plan = _build_macro_plan(gate_plan, resources, parameters, inventory)
    schedule_context = _build_schedule_context(idol_plan, resources, parameters, inventory, gate_plan, snapshot)
    battle_context = _build_battle_context(idol_plan, exam_bonus, resources, turn_info, inventory, phase == "exam") if phase in {"lesson", "exam"} else None
    deck_state = _build_deck_state(inventory)
    normalized_candidates = tuple(_normalize_action_payload(payload, phase) for payload in list(decision_state.get("candidates") or []))
    legal_actions = frozenset(_coerce_int(index) for index in decision_state.get("legal_actions") or [])
    return StrategyInput(
        phase=phase,
        position=position,
        idol_plan=idol_plan,
        resources=resources,
        parameters=parameters,
        turn_info=turn_info,
        inventory=inventory,
        deck_state=deck_state,
        gate_plan=gate_plan,
        macro_plan=macro_plan,
        battle_context=battle_context,
        schedule_context=schedule_context,
        exam_bonus=exam_bonus,
        current_week=_coerce_int(snapshot.get("current_week") or snapshot.get("week")),
        weeks_until_gate=gate_plan.weeks_until_gate,
        decision_reason=str(decision_state.get("decision_explanation") or decision_state.get("reason") or ""),
        candidates=normalized_candidates,
        legal_actions=legal_actions,
    )


def _action_breakdown_for_schedule(action_id: str, input_data: StrategyInput, payload: dict[str, Any]) -> ActionScoreBreakdown:
    score = ActionScoreBreakdown()
    resources = input_data.resources
    parameters = input_data.parameters
    gate_plan = input_data.gate_plan
    if resources is not None and resources.stamina_ratio < 0.25:
        if "outing" in action_id or "refresh" in action_id:
            score.safety += 80.0
        else:
            score.volatility_penalty += 40.0
    if gate_plan.is_key_window:
        if "lesson" in action_id or "sp" in action_id:
            score.macro_alignment += 70.0
        else:
            score.macro_alignment -= 20.0
    if parameters is not None:
        if gate_plan.preferred_attributes:
            attr = _parse_attribute_type(action_id)
            if attr in gate_plan.preferred_attributes:
                score.immediate_gain += 25.0 + max(parameters.attribute_gap(attr), 0) * 0.2
        score.future_value += min(parameters.total_gap, 200) * 0.05
    if input_data.macro_plan.should_prioritize_purity:
        score.purity_impact += 15.0 if "lesson" in action_id else -10.0
    return score




def _parse_price_value(value: Any) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else 0


def _has_effect(effect_types: Sequence[Any], effect_type: ExamEffectType) -> bool:
    target = effect_type.value
    for raw in effect_types:
        if target in str(raw):
            return True
    return False


def _decision_tags_from_metadata(candidate: CandidateAction) -> set[str]:
    tags = candidate.metadata.get("decision_tags") or []
    return {str(tag) for tag in tags if str(tag).strip()}


def _estimate_drink_reserve_value(input_data: StrategyInput, drink: DrinkInfo | None) -> float:
    if drink is None:
        return 0.0
    resources = input_data.resources
    evaluation = EffectEvaluator.evaluate_drink(
        drink,
        input_data.idol_plan.plan_type,
        resources.stamina_ratio if resources else 1.0,
        {
            "parameter_buff": resources.parameter_buff if resources else 0,
            "concentration": resources.concentration if resources else 0,
            "review": resources.review if resources else 0,
            "aggressive": resources.aggressive if resources else 0,
            "block": resources.block if resources else 0,
            "enthusiastic": resources.enthusiastic if resources else 0,
            "full_power_point": resources.full_power_point if resources else 0,
        },
    )
    reserve_bonus = 0.0
    if input_data.gate_plan.is_key_window:
        reserve_bonus += 80.0
    if input_data.gate_plan.preserve_stamina and _has_effect(drink.effect_types, ExamEffectType.BLOCK):
        reserve_bonus += 140.0
    if input_data.macro_plan.should_push_burst and _has_effect(drink.effect_types, ExamEffectType.PLAY_COUNT_UP):
        reserve_bonus += 120.0
    return evaluation.total_score + reserve_bonus


def _find_inventory_drink(input_data: StrategyInput, db_id: str) -> DrinkInfo | None:
    for drink in input_data.inventory.p_drinks:
        if drink.drink_id == db_id:
            return drink
    return None


def _build_pending_drink_info(payload: dict[str, Any]) -> DrinkInfo | None:
    drink_id = str(payload.get("db_id") or "")
    if drink_id:
        resolved = _build_drink_info_from_db(drink_id)
        if resolved is not None:
            return resolved
    effect_types = tuple(
        str(effect_type)
        for effect_type in list(payload.get("effect_types") or [])
        if str(effect_type or "").strip()
    )
    if not effect_types and not str(payload.get("name") or payload.get("title") or "").strip():
        return None
    return DrinkInfo(
        drink_id=drink_id,
        drink_name=str(payload.get("name") or payload.get("title") or drink_id or ""),
        rarity=str(payload.get("rarity") or ""),
        effect_types=effect_types,
        description=str(payload.get("description") or ""),
        plan_type=_parse_plan_type(payload.get("plan_type") or payload.get("planType") or ""),
    )


def _candidate_card_from_metadata(candidate: CandidateAction) -> CardEffectInfo | None:
    if candidate.db_id:
        return _build_card_effect_info_from_db(candidate.db_id, _coerce_int(candidate.metadata.get("upgrade_count")))
    effect_types = tuple(
        str(effect_type)
        for effect_type in list(candidate.metadata.get("effect_types") or [])
        if str(effect_type or "").strip()
    )
    if not effect_types and not candidate.name:
        return None
    try:
        category = ProduceCardCategory(str(candidate.metadata.get("category") or ProduceCardCategory.UNKNOWN.value))
    except ValueError:
        category = ProduceCardCategory.UNKNOWN
    return CardEffectInfo(
        card_id=str(candidate.metadata.get("raw_id") or candidate.action_id or candidate.index),
        card_name=str(candidate.metadata.get("raw_name") or candidate.name or candidate.action_id),
        rarity=str(candidate.metadata.get("rarity") or ""),
        category=category,
        upgrade_count=_coerce_int(candidate.metadata.get("upgrade_count")),
        stamina_cost=_coerce_int(candidate.metadata.get("cost") or candidate.metadata.get("stamina_cost")),
        effect_types=effect_types,
        description=str(candidate.metadata.get("description") or ""),
        plan_type=_parse_plan_type(candidate.metadata.get("plan_type") or candidate.metadata.get("planType") or ""),
    )


def _candidate_item_from_metadata(candidate: CandidateAction) -> ItemInfo | None:
    if candidate.db_id:
        return _build_item_info_from_db(candidate.db_id)
    effect_types = tuple(
        str(effect_type)
        for effect_type in list(candidate.metadata.get("effect_types") or [])
        if str(effect_type or "").strip()
    )
    if not effect_types and not candidate.name:
        return None
    return ItemInfo(
        item_id=str(candidate.metadata.get("raw_id") or candidate.action_id or candidate.index),
        item_name=str(candidate.metadata.get("raw_name") or candidate.name or candidate.action_id),
        rarity=str(candidate.metadata.get("rarity") or ""),
        effect_types=effect_types,
        description=str(candidate.metadata.get("description") or ""),
        plan_type=_parse_plan_type(candidate.metadata.get("plan_type") or candidate.metadata.get("planType") or ""),
        is_exam_effect=bool(candidate.metadata.get("is_exam_effect") or candidate.metadata.get("isExamEffect")),
    )


def _evaluate_schedule_action(action_id: str, input_data: StrategyInput, payload: dict[str, Any]) -> ActionScoreBreakdown:
    score = _action_breakdown_for_schedule(action_id, input_data, payload)
    if "lesson" in action_id and "self" not in action_id:
        score.immediate_gain += 40.0
    elif "outing" in action_id:
        score.resource_conversion += 20.0
    elif "business" in action_id:
        score.future_value += 15.0
    elif "refresh" in action_id:
        resources = input_data.resources
        stamina_ratio = resources.stamina_ratio if resources is not None else 1.0
        stamina_deficit = max(0.0, 1.0 - stamina_ratio)

        # “休む”不应在健康体力时凭固定高分碾压正常周行动；
        # 这里改为按体力缺口给收益，真正的低体力强制回体仍由 schedule/recovery.py 负责。
        score.safety += stamina_deficit * 36.0

        # 临近关口且本周确实缺体力时，再额外提高休息优先级。
        if input_data.gate_plan.preserve_stamina:
            score.safety += 18.0
        elif input_data.gate_plan.is_key_window and stamina_ratio < 0.5:
            score.safety += 8.0

        # 低体力且关口临近时，休息需要能压过“无脑冲 SP 课”的固定加成。
        if input_data.gate_plan.is_key_window and stamina_ratio < 0.25:
            score.safety += 120.0
        elif stamina_ratio < 0.18:
            score.safety += 80.0

        # 体力健康时休息会损失行动窗口，给明确机会成本，避免在 20+/30 体力时频繁乱休息。
        if stamina_ratio >= 0.6:
            score.opportunity_cost += (stamina_ratio - 0.6) * 120.0
        elif stamina_ratio >= 0.5 and input_data.gate_plan.is_key_window:
            score.opportunity_cost += 8.0
    return score


def _evaluate_battle_action(card: CardEffectInfo, input_data: StrategyInput) -> ActionScoreBreakdown:
    evaluation = EffectEvaluator.evaluate_card_for_battle(card, input_data.battle_context)
    score = ActionScoreBreakdown(
        immediate_gain=evaluation.base_score,
        macro_alignment=evaluation.plan_match_bonus,
        safety=evaluation.context_bonus,
        play_value=0.0,
    )
    if input_data.resources is not None and input_data.resources.stamina < card.stamina_cost:
        score.volatility_penalty += card.stamina_cost * 80.0
    if card.is_trouble:
        score.volatility_penalty += 800.0
    if input_data.macro_plan.should_push_burst and card.category == ProduceCardCategory.ACTIVE_SKILL:
        score.engine_gain += 60.0
    return score


def _evaluate_reward_action(card: CardEffectInfo, input_data: StrategyInput) -> ActionScoreBreakdown:
    evaluation = EffectEvaluator.evaluate_card_for_reward(
        card,
        input_data.idol_plan.plan_type,
        input_data.inventory.deck_composition,
        input_data.inventory.hand_cards + input_data.inventory.deck_cards,
    )
    return ActionScoreBreakdown(
        immediate_gain=evaluation.base_score,
        macro_alignment=evaluation.plan_match_bonus,
        safety=evaluation.context_bonus,
        purity_impact=evaluation.duplicate_penalty * -1.0,
        ownership_value=evaluation.total_score * 0.05,
    )


def _evaluate_drink_action(drink: DrinkInfo, input_data: StrategyInput) -> ActionScoreBreakdown:
    evaluation = EffectEvaluator.evaluate_drink(
        drink,
        input_data.idol_plan.plan_type,
        input_data.resources.stamina_ratio if input_data.resources else 1.0,
        {
            "parameter_buff": input_data.resources.parameter_buff if input_data.resources else 0,
            "concentration": input_data.resources.concentration if input_data.resources else 0,
            "review": input_data.resources.review if input_data.resources else 0,
            "aggressive": input_data.resources.aggressive if input_data.resources else 0,
            "block": input_data.resources.block if input_data.resources else 0,
            "enthusiastic": input_data.resources.enthusiastic if input_data.resources else 0,
            "full_power_point": input_data.resources.full_power_point if input_data.resources else 0,
        },
    )
    reserve_value = _estimate_drink_reserve_value(input_data, drink)
    score = ActionScoreBreakdown(
        immediate_gain=evaluation.base_score,
        macro_alignment=evaluation.plan_match_bonus,
        safety=evaluation.context_bonus,
        ownership_value=evaluation.rarity_bonus,
        future_value=reserve_value * 0.05,
    )
    if input_data.gate_plan.is_key_window:
        score.macro_alignment += 30.0
    if input_data.macro_plan.should_preserve_resources and _has_effect(drink.effect_types, ExamEffectType.BLOCK):
        score.safety += 40.0
    return score


def _evaluate_item_action(item: ItemInfo, input_data: StrategyInput) -> ActionScoreBreakdown:
    evaluation = EffectEvaluator.evaluate_item(item, input_data.idol_plan.plan_type)
    weeks = max(input_data.gate_plan.weeks_until_gate, 0)
    score = ActionScoreBreakdown(
        immediate_gain=evaluation.base_score,
        macro_alignment=evaluation.plan_match_bonus,
        safety=evaluation.context_bonus,
        ownership_value=evaluation.rarity_bonus,
        future_value=evaluation.base_score * 0.08 + min(weeks * 20.0, 80.0),
    )
    if item.is_exam_effect and weeks <= 2:
        score.macro_alignment += 80.0
    if input_data.macro_plan.should_push_burst and any(
        _has_effect(item.effect_types, effect_type)
        for effect_type in (ExamEffectType.PLAY_COUNT_UP, ExamEffectType.PARAMETER_UP_INCREASE)
    ):
        score.engine_gain += 70.0
    return score


def _evaluate_consult_action(candidate: CandidateAction, input_data: StrategyInput) -> ActionScoreBreakdown:
    metadata = candidate.metadata
    kind = candidate.kind
    consult_action = str(metadata.get("consult_action") or candidate.action_id)
    tags = _decision_tags_from_metadata(candidate)
    score = ActionScoreBreakdown(safety=10.0)

    if kind == "exit" or consult_action == "consult_exit":
        score.safety += 25.0
        if input_data.macro_plan.should_preserve_resources:
            score.macro_alignment += 15.0
        else:
            score.opportunity_cost += 10.0
        return score

    if kind == "exchange" or consult_action.startswith("consult_exchange"):
        price = _parse_price_value(metadata.get("price"))
        score.resource_conversion += 35.0
        score.future_value += 20.0
        if "deck_quality_value" in tags:
            score.engine_gain += 30.0
        if "inventory_flexibility_value" in tags:
            score.future_value += 20.0
        if input_data.gate_plan.preserve_p_points:
            score.opportunity_cost += price * 0.5
        else:
            score.opportunity_cost += price * 0.25
        if input_data.inventory.p_points < price:
            score.volatility_penalty += 9999.0
        return score

    if kind in {"enhance", "delete"} or consult_action in {"consult_open_enhancement", "consult_open_remove"}:
        score.macro_alignment += 40.0
        score.future_value += 30.0
        if consult_action == "consult_open_remove" or kind == "delete":
            score.purity_impact += 50.0 if input_data.macro_plan.should_prioritize_purity else 20.0
        else:
            score.engine_gain += 45.0
            if input_data.macro_plan.should_push_burst:
                score.macro_alignment += 20.0
        return score

    if consult_action == "consult_select_remove_target":
        score.purity_impact += 70.0 if input_data.macro_plan.should_prioritize_purity else 35.0
        if "deck_quality_value" in tags:
            score.future_value += 20.0
        return score

    if consult_action == "consult_select_enhancement_target":
        score.engine_gain += 70.0
        score.future_value += 35.0
        if input_data.macro_plan.should_push_burst:
            score.macro_alignment += 25.0
        return score

    if consult_action == "consult_confirm_remove":
        score.purity_impact += 45.0
        score.future_value += 20.0
        return score

    if consult_action == "consult_confirm_enhancement":
        score.engine_gain += 50.0
        score.future_value += 25.0
        return score

    score.context_bonus += 5.0
    return score


def _evaluate_dialogue_action(candidate: CandidateAction, input_data: StrategyInput) -> ActionScoreBreakdown:
    metadata = candidate.metadata
    description = str(metadata.get("description") or "")
    tags = _decision_tags_from_metadata(candidate)
    p_cost = _coerce_int(metadata.get("p_cost"))
    text = " ".join((candidate.name, description, str(metadata.get("gain_summary") or ""))).lower()
    score = ActionScoreBreakdown(safety=12.0)

    if p_cost > 0:
        score.opportunity_cost += p_cost * 0.8 if input_data.gate_plan.preserve_p_points else p_cost * 0.35
        if input_data.inventory.p_points < p_cost:
            score.volatility_penalty += 9999.0

    if "stamina_recovery_value" in tags and input_data.resources is not None:
        gain = (1.0 - input_data.resources.stamina_ratio) * 70.0
        score.safety += gain
        score.immediate_gain += gain * 0.5
    if "parameter_growth_value" in tags and input_data.parameters is not None:
        score.immediate_gain += min(input_data.parameters.total_gap * 0.12, 60.0)
        score.macro_alignment += 20.0
    if "deck_quality_value" in tags:
        score.future_value += 35.0
    if "p_point_economy_value" in tags:
        score.resource_conversion += 25.0
    if "random" in text or "ランダム" in description:
        score.volatility_penalty += 35.0
    if input_data.gate_plan.is_key_window and "stability_value" in tags:
        score.safety += 20.0
    return score


def _evaluate_p_drink_limit_action(candidate: CandidateAction, input_data: StrategyInput) -> ActionScoreBreakdown:
    metadata = candidate.metadata
    new_drink = _build_pending_drink_info(dict(metadata.get("new_drink") or {}))
    new_value = _estimate_drink_reserve_value(input_data, new_drink)
    score = ActionScoreBreakdown(safety=10.0)

    if candidate.kind == "skip_new_drink":
        score.future_value += max(sum(_estimate_drink_reserve_value(input_data, drink) for drink in input_data.inventory.p_drinks) * 0.02, 5.0)
        if input_data.gate_plan.is_key_window:
            score.safety += 20.0
        score.opportunity_cost += max(new_value * 0.03, 0.0)
        return score

    existing = _find_inventory_drink(input_data, candidate.db_id)
    existing_value = _estimate_drink_reserve_value(input_data, existing)
    diff = new_value - existing_value
    score.future_value += diff * 0.06
    if diff > 0:
        score.macro_alignment += 25.0
    else:
        score.opportunity_cost += abs(diff) * 0.06 + 20.0
    if input_data.gate_plan.preserve_stamina and existing is not None and _has_effect(existing.effect_types, ExamEffectType.BLOCK):
        score.opportunity_cost += 30.0
    return score


def _evaluate_other_action(candidate: CandidateAction, input_data: StrategyInput) -> tuple[ActionScoreBreakdown, str] | None:
    phase = input_data.phase
    if phase == "skill_reward":
        card = _candidate_card_from_metadata(candidate)
        if card is None:
            return None
        return _evaluate_reward_action(card, input_data), card.card_name
    if phase == "p_drink":
        if candidate.kind in {"skip_new_drink", "discard_existing_drink"}:
            return _evaluate_p_drink_limit_action(candidate, input_data), candidate.name or candidate.action_id
        drink = _build_drink_info_from_db(candidate.db_id) if candidate.db_id else _build_pending_drink_info(candidate.metadata)
        if drink is None:
            return None
        return _evaluate_drink_action(drink, input_data), drink.drink_name
    if phase == "item_select":
        item = _candidate_item_from_metadata(candidate)
        if item is None:
            return None
        return _evaluate_item_action(item, input_data), item.item_name
    if phase == "consult":
        return _evaluate_consult_action(candidate, input_data), candidate.name or candidate.action_id
    if phase == "dialogue":
        return _evaluate_dialogue_action(candidate, input_data), candidate.name or candidate.action_id
    return ActionScoreBreakdown(safety=1.0), candidate.name or candidate.action_id


def _pick_best_candidate(input_data: StrategyInput, kind: str) -> DecisionResult | None:
    best_result: DecisionResult | None = None
    top_choices: list[tuple[float, str, CandidateAction, ActionScoreBreakdown]] = []
    for candidate in input_data.candidates:
        if not candidate.available:
            continue
        if input_data.legal_actions and candidate.index not in input_data.legal_actions:
            continue
        if kind == "battle":
            if not candidate.is_card:
                continue
            card = _candidate_card_from_metadata(candidate)
            if card is None:
                continue
            score = _evaluate_battle_action(card, input_data)
            reason = f"{card.card_name}"
        elif kind == "schedule":
            score = _evaluate_schedule_action(candidate.action_id, input_data, candidate.metadata)
            reason = candidate.name or candidate.action_id
        else:
            evaluated = _evaluate_other_action(candidate, input_data)
            if evaluated is None:
                continue
            score, reason = evaluated

        result = DecisionResult(
            selected_index=candidate.index,
            selected_action_id=candidate.action_id,
            score=score.total_score,
            reason=reason,
        )
        top_choices.append((result.score, reason, candidate, score))
        if best_result is None or result.score > best_result.score:
            best_result = result

    if top_choices:
        top_choices.sort(key=lambda item: item[0], reverse=True)
        summary = []
        for total, reason, candidate, breakdown in top_choices[:3]:
            summary.append(
                f"#{candidate.index}:{reason} total={total:.1f} macro={breakdown.macro_alignment:.1f} future={breakdown.future_value:.1f} purity={breakdown.purity_impact:.1f} safety={breakdown.safety:.1f} cost={breakdown.opportunity_cost:.1f}"
            )
        logger.info("[AlgoStrategy][{}] Top choices: {}", input_data.phase or kind, " | ".join(summary))
    return best_result


class BattleAlgoStrategy:
    """战斗决策策略（レッスン/試験）"""

    def __call__(
        self,
        app: AppProcessor,
        ctx: ProduceContext,
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> DecisionResult | None:
        if not candidates or decision_state is None:
            return None
        phase = str(decision_state.get("phase") or "")
        if phase not in ("lesson", "exam"):
            return None
        input_data = _build_strategy_input(app, ctx, candidates, decision_state)
        return _pick_best_candidate(input_data, "battle")


class ScheduleAlgoStrategy:
    """周选择决策策略"""

    def __call__(
        self,
        app: AppProcessor,
        ctx: ProduceContext,
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> DecisionResult | None:
        if not candidates or decision_state is None:
            return None
        phase = str(decision_state.get("phase") or "")
        if phase != "schedule":
            return None
        input_data = _build_strategy_input(app, ctx, candidates, decision_state)
        return _pick_best_candidate(input_data, "schedule")


class OtherAlgoStrategy:
    """其他决策策略：对话/P饮料/技能奖励/咨询/道具选择"""

    def __call__(
        self,
        app: AppProcessor,
        ctx: ProduceContext,
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> DecisionResult | None:
        if not candidates or decision_state is None:
            return None
        phase = str(decision_state.get("phase") or "")
        if phase == "skill_reward":
            kind = "skill_reward"
        elif phase == "p_drink":
            kind = "p_drink"
        elif phase == "item_select":
            kind = "item_select"
        elif phase == "consult":
            kind = "consult"
        else:
            kind = "dialogue"
        input_data = _build_strategy_input(app, ctx, candidates, decision_state)
        return _pick_best_candidate(input_data, kind)


class AlgoStrategy:
    """统一算法决策策略入口"""

    def __init__(self):
        self._battle = BattleAlgoStrategy()
        self._schedule = ScheduleAlgoStrategy()
        self._other = OtherAlgoStrategy()

    def __call__(
        self,
        app: AppProcessor,
        ctx: ProduceContext,
        candidates: Sequence[Any],
        decision_state: dict[str, Any] | None = None,
    ) -> DecisionResult | None:
        if decision_state is None:
            return None
        phase = str(decision_state.get("phase") or "")
        try:
            if phase in ("lesson", "exam"):
                return self._battle(app, ctx, candidates, decision_state)
            if phase == "schedule":
                return self._schedule(app, ctx, candidates, decision_state)
            return self._other(app, ctx, candidates, decision_state)
        except ValueError as exc:
            logger.warning("[AlgoStrategy] 决策输入不完整，放弃本次自动决策: {}", exc)
            return None


def inject_algo_strategy(ctx: ProduceContext) -> tuple:
    """注入算法策略到上下文（兼容旧接口）"""
    schedule_strategy = ScheduleAlgoStrategy()
    battle_strategy = BattleAlgoStrategy()
    other_strategy = OtherAlgoStrategy()

    ctx.schedule_strategy = schedule_strategy
    ctx.lesson_strategy = battle_strategy
    ctx.exam_strategy = battle_strategy
    ctx.dialogue_strategy = other_strategy
    ctx.p_drink_strategy = other_strategy
    ctx.skill_reward_strategy = other_strategy
    ctx.consult_strategy = other_strategy
    ctx.item_select_strategy = other_strategy
    ctx.modal_strategy = other_strategy

    return schedule_strategy, battle_strategy, other_strategy


__all__ = [
    "AlgoStrategy",
    "BattleAlgoStrategy",
    "ScheduleAlgoStrategy",
    "OtherAlgoStrategy",
    "inject_algo_strategy",
]
