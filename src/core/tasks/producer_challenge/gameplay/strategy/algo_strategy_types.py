"""算法决策所需的数据类型定义。

所有复杂数据传递必须使用dataclass，禁止使用dict。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.constants.game.produce_enums import (
    AttributeType,
    ProduceCardCategory,
    ProducePlanType,
)


@dataclass(frozen=True)
class IdolPlanInfo:
    """偶像计划类型信息（从IdolCard读取）"""

    plan_type: ProducePlanType
    vocal_growth_permil: int
    dance_growth_permil: int
    visual_growth_permil: int

    @property
    def plan_label(self) -> str:
        return self.plan_type.label

    @property
    def primary_attribute(self) -> AttributeType:
        growth_map = {
            AttributeType.VOCAL: self.vocal_growth_permil,
            AttributeType.DANCE: self.dance_growth_permil,
            AttributeType.VISUAL: self.visual_growth_permil,
        }
        return max(growth_map.items(), key=lambda x: x[1])[0]

    @property
    def attribute_priority(self) -> list[tuple[AttributeType, int]]:
        return sorted(
            [
                (AttributeType.VOCAL, self.vocal_growth_permil),
                (AttributeType.DANCE, self.dance_growth_permil),
                (AttributeType.VISUAL, self.visual_growth_permil),
            ],
            key=lambda x: x[1],
            reverse=True,
        )


@dataclass(frozen=True)
class ExamBonusInfo:
    """考试加成信息（从exam_prep读取）"""

    vocal_bonus_permil: int
    dance_bonus_permil: int
    visual_bonus_permil: int

    @property
    def total_bonus(self) -> int:
        return self.vocal_bonus_permil + self.dance_bonus_permil + self.visual_bonus_permil

    @property
    def primary_attribute(self) -> AttributeType:
        bonus_map = {
            AttributeType.VOCAL: self.vocal_bonus_permil,
            AttributeType.DANCE: self.dance_bonus_permil,
            AttributeType.VISUAL: self.visual_bonus_permil,
        }
        return max(bonus_map.items(), key=lambda x: x[1])[0]


@dataclass(frozen=True)
class ResourceState:
    """战斗资源状态"""

    stamina: int
    max_stamina: int
    score: int
    parameter_buff: int
    concentration: int
    review: int
    aggressive: int
    block: int
    enthusiastic: int
    full_power_point: int
    lesson_buff: int

    @property
    def stamina_ratio(self) -> float:
        return self.stamina / max(self.max_stamina, 1)


@dataclass(frozen=True)
class ParameterState:
    """参数状态"""

    vocal_current: int
    vocal_max: int
    dance_current: int
    dance_max: int
    visual_current: int
    visual_max: int

    @property
    def vocal_gap(self) -> int:
        return max(self.vocal_max - self.vocal_current, 0)

    @property
    def dance_gap(self) -> int:
        return max(self.dance_max - self.dance_current, 0)

    @property
    def visual_gap(self) -> int:
        return max(self.visual_max - self.visual_current, 0)

    @property
    def total_gap(self) -> int:
        return self.vocal_gap + self.dance_gap + self.visual_gap

    def attribute_gap(self, attr: AttributeType) -> int:
        if attr == AttributeType.VOCAL:
            return self.vocal_gap
        if attr == AttributeType.DANCE:
            return self.dance_gap
        if attr == AttributeType.VISUAL:
            return self.visual_gap
        return 0


@dataclass(frozen=True)
class TurnInfo:
    """回合信息"""

    current_turn: int
    max_turns: int
    remaining_turns: int
    turn_color: AttributeType | None

    @property
    def is_final_turns(self) -> bool:
        return self.remaining_turns <= 2


@dataclass(frozen=True)
class CardEffectInfo:
    """卡牌效果信息（从游戏数据库ProduceCard读取）"""

    card_id: str
    card_name: str
    rarity: str
    category: ProduceCardCategory
    upgrade_count: int
    stamina_cost: int
    effect_types: tuple[str, ...]
    description: str
    plan_type: ProducePlanType

    @property
    def is_active(self) -> bool:
        return self.category == ProduceCardCategory.ACTIVE_SKILL

    @property
    def is_mental(self) -> bool:
        return self.category == ProduceCardCategory.MENTAL_SKILL

    @property
    def is_trouble(self) -> bool:
        return self.category == ProduceCardCategory.TROUBLE


@dataclass(frozen=True)
class DrinkInfo:
    """P饮料信息（从游戏数据库ProduceDrink读取）"""

    drink_id: str
    drink_name: str
    rarity: str
    effect_types: tuple[str, ...]
    description: str
    plan_type: ProducePlanType


@dataclass(frozen=True)
class ItemInfo:
    """P物品信息（从游戏数据库ProduceItem读取）"""

    item_id: str
    item_name: str
    rarity: str
    effect_types: tuple[str, ...]
    description: str
    plan_type: ProducePlanType
    is_exam_effect: bool


@dataclass(frozen=True)
class InventoryState:
    """库存与牌区实体集合。"""

    hand_cards: tuple[CardEffectInfo, ...]
    deck_cards: tuple[CardEffectInfo, ...]
    grave_cards: tuple[CardEffectInfo, ...]
    hold_cards: tuple[CardEffectInfo, ...]
    lost_cards: tuple[CardEffectInfo, ...]
    p_drinks: tuple[DrinkInfo, ...]
    p_items: tuple[ItemInfo, ...]
    p_points: int

    @property
    def all_cards(self) -> tuple[CardEffectInfo, ...]:
        return (
            self.hand_cards
            + self.deck_cards
            + self.grave_cards
            + self.hold_cards
            + self.lost_cards
        )

    @property
    def deck_composition(self) -> DeckComposition:
        total_cards = len(self.all_cards)
        unique_cards = len({card.card_id for card in self.all_cards})
        sense_cards = sum(1 for card in self.all_cards if card.plan_type == ProducePlanType.PLAN1)
        logic_cards = sum(1 for card in self.all_cards if card.plan_type == ProducePlanType.PLAN2)
        anomaly_cards = sum(1 for card in self.all_cards if card.plan_type == ProducePlanType.PLAN3)
        immediate_output_cards = sum(
            1
            for card in self.all_cards
            if any(effect_type in card.effect_types for effect_type in ("score", "vocal", "dance", "visual"))
        )
        status_stacking_cards = sum(
            1
            for card in self.all_cards
            if any(effect_type in card.effect_types for effect_type in ("review", "concentration", "parameterBuff", "fullPowerPoint", "enthusiastic"))
        )
        recovery_cards = sum(
            1
            for card in self.all_cards
            if any("block" in effect_type.lower() for effect_type in card.effect_types)
        )
        extra_play_cards = sum(
            1
            for card in self.all_cards
            if any("playcountup" in effect_type.lower() for effect_type in card.effect_types)
        )
        return DeckComposition(
            total_cards=total_cards,
            sense_cards=sense_cards,
            logic_cards=logic_cards,
            anomaly_cards=anomaly_cards,
            immediate_output_cards=immediate_output_cards,
            status_stacking_cards=status_stacking_cards,
            recovery_cards=recovery_cards,
            extra_play_cards=extra_play_cards,
            purity_score=unique_cards / max(total_cards, 1),
        )


@dataclass(frozen=True)
class DeckCardState:
    """牌组卡片状态。"""

    card: CardEffectInfo
    zone: str
    zone_index: int
    duplicate_group_size: int = 1
    ownership_value: float = 0.0
    play_value: float = 0.0

    @property
    def is_hand(self) -> bool:
        return self.zone == "hand"

    @property
    def is_deck(self) -> bool:
        return self.zone == "deck"

    @property
    def is_grave(self) -> bool:
        return self.zone == "grave"

    @property
    def is_hold(self) -> bool:
        return self.zone == "hold"

    @property
    def is_lost(self) -> bool:
        return self.zone == "lost"

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_group_size > 1


@dataclass(frozen=True)
class DeckComposition:
    """牌组构成分析"""

    total_cards: int
    sense_cards: int
    logic_cards: int
    anomaly_cards: int
    immediate_output_cards: int
    status_stacking_cards: int
    recovery_cards: int
    extra_play_cards: int
    purity_score: float

    @property
    def sense_ratio(self) -> float:
        return self.sense_cards / max(self.total_cards, 1)

    @property
    def logic_ratio(self) -> float:
        return self.logic_cards / max(self.total_cards, 1)

    @property
    def anomaly_ratio(self) -> float:
        return self.anomaly_cards / max(self.total_cards, 1)

    @property
    def duplicate_density(self) -> float:
        return 1.0 - self.purity_score


@dataclass(frozen=True)
class DeckState:
    """完整牌组状态。"""

    cards: tuple[DeckCardState, ...]
    total_cards: int
    unique_cards: int
    duplicate_groups: int
    purity_score: float
    engine_density: float
    recovery_cards: int
    extra_play_cards: int

    @property
    def duplicate_density(self) -> float:
        return 1.0 - self.purity_score

    @property
    def hand_cards(self) -> tuple[DeckCardState, ...]:
        return tuple(card for card in self.cards if card.is_hand)

    @property
    def deck_cards(self) -> tuple[DeckCardState, ...]:
        return tuple(card for card in self.cards if card.is_deck)

    @property
    def grave_cards(self) -> tuple[DeckCardState, ...]:
        return tuple(card for card in self.cards if card.is_grave)

    @property
    def hold_cards(self) -> tuple[DeckCardState, ...]:
        return tuple(card for card in self.cards if card.is_hold)

    @property
    def lost_cards(self) -> tuple[DeckCardState, ...]:
        return tuple(card for card in self.cards if card.is_lost)


@dataclass(frozen=True)
class GatePlan:
    """关口规划。"""

    gate_type: str
    weeks_until_gate: int
    is_key_window: bool
    preserve_stamina: bool
    preserve_p_points: bool
    preferred_attributes: tuple[AttributeType, ...] = ()


@dataclass(frozen=True)
class MacroPlan:
    """宏观规划。"""

    main_goal: str
    secondary_goal: str
    gate_plan: GatePlan
    should_prioritize_purity: bool
    should_preserve_resources: bool
    should_push_burst: bool
    safety_priority: float = 0.0
    purity_priority: float = 0.0


@dataclass
class ActionScoreBreakdown:
    """动作评分拆分。"""

    macro_alignment: float = 0.0
    immediate_gain: float = 0.0
    engine_gain: float = 0.0
    resource_conversion: float = 0.0
    future_value: float = 0.0
    purity_impact: float = 0.0
    safety: float = 0.0
    volatility_penalty: float = 0.0
    opportunity_cost: float = 0.0
    ownership_value: float = 0.0
    play_value: float = 0.0
    context_bonus: float = 0.0

    @property
    def total_score(self) -> float:
        return (
            self.macro_alignment
            + self.immediate_gain
            + self.engine_gain
            + self.resource_conversion
            + self.future_value
            + self.purity_impact
            + self.safety
            + self.ownership_value
            + self.play_value
            + self.context_bonus
            - self.volatility_penalty
            - self.opportunity_cost
        )


@dataclass(frozen=True)
class ScheduleContext:
    """周选择决策上下文"""

    idol_plan: IdolPlanInfo
    resources: ResourceState
    parameters: ParameterState
    inventory: InventoryState
    weeks_until_gate: int
    gate_type: str
    current_week: int

    @property
    def is_pre_exam(self) -> bool:
        return self.weeks_until_gate == 1

    @property
    def is_low_stamina(self) -> bool:
        return self.resources.stamina_ratio < 0.25


@dataclass(frozen=True)
class BattleContext:
    """战斗决策上下文"""

    idol_plan: IdolPlanInfo
    exam_bonus: ExamBonusInfo | None
    resources: ResourceState
    turn_info: TurnInfo
    inventory: InventoryState
    is_exam: bool

    @property
    def is_critical_turn(self) -> bool:
        return self.turn_info.is_final_turns or self.resources.stamina_ratio < 0.2


@dataclass(frozen=True)
class CandidateAction:
    """候选动作"""

    index: int
    action_id: str
    db_id: str
    name: str
    available: bool
    kind: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_end_turn(self) -> bool:
        return self.action_id == "end_turn"

    @property
    def is_card(self) -> bool:
        return self.action_id.startswith("produce_card:")

    @property
    def is_drink(self) -> bool:
        return self.action_id.startswith("produce_drink:")


@dataclass(frozen=True)
class StrategyInput:
    """统一决策输入。"""

    phase: str
    position: str
    idol_plan: IdolPlanInfo
    resources: ResourceState | None
    parameters: ParameterState | None
    turn_info: TurnInfo | None
    inventory: InventoryState
    deck_state: DeckState
    gate_plan: GatePlan
    macro_plan: MacroPlan
    battle_context: BattleContext | None
    schedule_context: ScheduleContext | None
    exam_bonus: ExamBonusInfo | None = None
    current_week: int = 0
    weeks_until_gate: int = 0
    decision_reason: str = ""
    candidates: tuple[CandidateAction, ...] = ()
    legal_actions: frozenset[int] = frozenset()


@dataclass(frozen=True)
class DecisionResult:
    """决策结果"""

    selected_index: int
    selected_action_id: str
    score: float
    reason: str

    @property
    def index(self) -> int:
        return self.selected_index


__all__ = [
    "ActionScoreBreakdown",
    "BattleContext",
    "CandidateAction",
    "CardEffectInfo",
    "DeckCardState",
    "DeckComposition",
    "DeckState",
    "DecisionResult",
    "DrinkInfo",
    "ExamBonusInfo",
    "GatePlan",
    "IdolPlanInfo",
    "InventoryState",
    "ItemInfo",
    "MacroPlan",
    "ParameterState",
    "ResourceState",
    "ScheduleContext",
    "StrategyInput",
    "TurnInfo",
]
