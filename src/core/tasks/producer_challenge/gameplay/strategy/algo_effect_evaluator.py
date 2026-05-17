"""卡牌效果评估体系。

基于游戏数据库的effect_types和ProduceText常量，构建统一的效果评估系统。
支持重复卡牌的价值计算（如好印象体系中的多张同效果卡）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.constants.game.produce_enums import ExamEffectType, ProducePlanType
from src.constants.game.text.produce_text import ProduceText

if TYPE_CHECKING:
    from .algo_strategy_types import (
        BattleContext,
        CardEffectInfo,
        DeckComposition,
        DrinkInfo,
        ItemInfo,
    )


# 效果类型基础权重
EFFECT_BASE_WEIGHTS = {
    # 即时输出类（高优先级）
    ExamEffectType.SCORE: 300.0,
    ExamEffectType.VOCAL: 250.0,
    ExamEffectType.DANCE: 250.0,
    ExamEffectType.VISUAL: 250.0,
    # 状态类（中优先级）
    ExamEffectType.PARAMETER_BUFF: 1250.0,  # 好調
    ExamEffectType.CONCENTRATION: 2500.0,  # 集中
    ExamEffectType.REVIEW: 2100.0,  # 好印象
    ExamEffectType.AGGRESSIVE: 540.0,  # やる気
    ExamEffectType.BLOCK: 320.0,  # 元気
    ExamEffectType.ENTHUSIASTIC: 800.0,  # 熱意
    ExamEffectType.FULL_POWER_POINT: 1500.0,  # 全力値
    # 特殊效果（超高优先级）
    ExamEffectType.PLAY_COUNT_UP: 2600.0,  # 追加使用
    ExamEffectType.PARAMETER_UP_INCREASE: 1600.0,  # 参数上升量增加
}

# 计划类型对应的核心效果
PLAN_CORE_EFFECTS = {
    ProducePlanType.PLAN1: {  # 感性
        ExamEffectType.PARAMETER_BUFF,
        ExamEffectType.CONCENTRATION,
    },
    ProducePlanType.PLAN2: {  # 理性
        ExamEffectType.REVIEW,
        ExamEffectType.AGGRESSIVE,
    },
    ProducePlanType.PLAN3: {  # 非凡
        ExamEffectType.FULL_POWER_POINT,
        ExamEffectType.ENTHUSIASTIC,
    },
}

# 稀有度加成
RARITY_BONUS = {
    "SSR": 35.0,
    "SR": 25.0,
    "R": 15.0,
    "N": 5.0,
}


@dataclass(frozen=True)
class EffectEvaluation:
    """效果评估结果"""

    base_score: float  # 基础分数
    plan_match_bonus: float  # 计划匹配加成
    rarity_bonus: float  # 稀有度加成
    context_bonus: float  # 上下文加成（体力、回合数等）
    duplicate_penalty: float  # 重复惩罚（已有多张同效果卡）

    @property
    def total_score(self) -> float:
        """总分"""
        return (
            self.base_score
            + self.plan_match_bonus
            + self.rarity_bonus
            + self.context_bonus
            - self.duplicate_penalty
        )


class EffectEvaluator:
    """效果评估器"""

    @staticmethod
    def evaluate_card_for_battle(
        card: CardEffectInfo,
        context: BattleContext,
    ) -> EffectEvaluation:
        """评估卡牌在战斗中的价值

        Args:
            card: 卡牌信息
            context: 战斗上下文

        Returns:
            EffectEvaluation: 评估结果
        """
        base_score = 0.0
        plan_match_bonus = 0.0
        context_bonus = 0.0

        # 基础效果分数
        for effect_type in card.effect_types:
            try:
                effect_enum = ExamEffectType(effect_type)
                base_score += EFFECT_BASE_WEIGHTS.get(effect_enum, 0.0)
            except ValueError:
                # 不是标准效果类型，跳过
                continue

        # 计划匹配加成
        plan_core = PLAN_CORE_EFFECTS.get(context.idol_plan.plan_type, set())
        for effect_type in card.effect_types:
            try:
                effect_enum = ExamEffectType(effect_type)
                if effect_enum in plan_core:
                    plan_match_bonus += 400.0
            except ValueError:
                continue

        # 上下文加成
        # 1. 体力不足时，回复卡价值提升
        if context.resources.stamina_ratio < 0.3:
            for effect_type in card.effect_types:
                if ExamEffectType.BLOCK.value in effect_type:
                    context_bonus += 1500.0 * (1.0 - context.resources.stamina_ratio)

        # 2. 最后几回合，即时输出卡价值提升
        if context.turn_info.is_final_turns:
            for effect_type in card.effect_types:
                if any(
                    et.value in effect_type
                    for et in [
                        ExamEffectType.SCORE,
                        ExamEffectType.VOCAL,
                        ExamEffectType.DANCE,
                        ExamEffectType.VISUAL,
                    ]
                ):
                    context_bonus += 1600.0

        # 3. 状态不足时，对应状态卡价值提升
        if context.idol_plan.plan_type == ProducePlanType.PLAN1:
            if context.resources.concentration < 3:
                for effect_type in card.effect_types:
                    if ExamEffectType.CONCENTRATION.value in effect_type:
                        context_bonus += 600.0
        elif context.idol_plan.plan_type == ProducePlanType.PLAN2:
            if context.resources.review < 3:
                for effect_type in card.effect_types:
                    if ExamEffectType.REVIEW.value in effect_type:
                        context_bonus += 600.0
        elif context.idol_plan.plan_type == ProducePlanType.PLAN3:
            if context.resources.enthusiastic < 3:
                for effect_type in card.effect_types:
                    if ExamEffectType.ENTHUSIASTIC.value in effect_type:
                        context_bonus += 550.0

        # 稀有度加成
        rarity_bonus = RARITY_BONUS.get(card.rarity.upper(), 0.0)

        # 重复惩罚（暂时为0，需要根据手牌库计算）
        duplicate_penalty = 0.0

        return EffectEvaluation(
            base_score=base_score,
            plan_match_bonus=plan_match_bonus,
            rarity_bonus=rarity_bonus,
            context_bonus=context_bonus,
            duplicate_penalty=duplicate_penalty,
        )

    @staticmethod
    def evaluate_card_for_reward(
        card: CardEffectInfo,
        plan_type: ProducePlanType,
        deck_composition: DeckComposition,
        existing_cards: tuple[CardEffectInfo, ...] = (),
    ) -> EffectEvaluation:
        """评估卡牌作为奖励的价值

        Args:
            card: 卡牌信息
            plan_type: 计划类型
            deck_composition: 当前牌组构成
            existing_cards: 已有的卡牌列表（用于计算重复惩罚）

        Returns:
            EffectEvaluation: 评估结果
        """
        base_score = 0.0
        plan_match_bonus = 0.0
        context_bonus = 0.0

        # 基础效果分数
        for effect_type in card.effect_types:
            try:
                effect_enum = ExamEffectType(effect_type)
                base_score += EFFECT_BASE_WEIGHTS.get(effect_enum, 0.0) * 0.5
            except ValueError:
                continue

        # 计划匹配加成
        plan_core = PLAN_CORE_EFFECTS.get(plan_type, set())
        for effect_type in card.effect_types:
            try:
                effect_enum = ExamEffectType(effect_type)
                if effect_enum in plan_core:
                    plan_match_bonus += 350.0
            except ValueError:
                continue

        # 牌组构成加成
        # 如果当前计划类型的卡牌不足，优先补充
        if plan_type == ProducePlanType.PLAN1 and deck_composition.sense_ratio < 0.4:
            if card.plan_type == ProducePlanType.PLAN1:
                context_bonus += 200.0
        elif plan_type == ProducePlanType.PLAN2 and deck_composition.logic_ratio < 0.4:
            if card.plan_type == ProducePlanType.PLAN2:
                context_bonus += 200.0
        elif plan_type == ProducePlanType.PLAN3 and deck_composition.anomaly_ratio < 0.4:
            if card.plan_type == ProducePlanType.PLAN3:
                context_bonus += 200.0

        # 稀有度加成
        rarity_bonus = RARITY_BONUS.get(card.rarity.upper(), 0.0)

        # 重复惩罚：统计已有卡牌中相同效果的数量
        duplicate_penalty = EffectEvaluator._calculate_duplicate_penalty(card, existing_cards)

        return EffectEvaluation(
            base_score=base_score,
            plan_match_bonus=plan_match_bonus,
            rarity_bonus=rarity_bonus,
            context_bonus=context_bonus,
            duplicate_penalty=duplicate_penalty,
        )

    @staticmethod
    def _calculate_duplicate_penalty(
        card: CardEffectInfo,
        existing_cards: tuple[CardEffectInfo, ...],
    ) -> float:
        """计算重复卡牌的调整分数

        对于好印象体系等，多张同效果卡是**必需的**（一回合只能打一张，但需要多回合打出）。
        因此：
        - 核心效果卡（好印象/集中/全力等）：前5张不惩罚，甚至加分
        - 非核心效果卡：超过3张后开始惩罚

        Args:
            card: 待评估的卡牌
            existing_cards: 已有的卡牌列表

        Returns:
            调整分数（负数表示扣分，正数表示加分）
        """
        if not existing_cards:
            return 0.0

        # 判断是否是核心状态堆叠卡
        is_core_stacking = any(
            effect_type in card.effect_types
            for effect_type in [
                ExamEffectType.REVIEW.value,  # 好印象
                ExamEffectType.CONCENTRATION.value,  # 集中
                ExamEffectType.PARAMETER_BUFF.value,  # 好調
                ExamEffectType.FULL_POWER_POINT.value,  # 全力値
                ExamEffectType.ENTHUSIASTIC.value,  # 熱意
            ]
        )

        # 统计已有卡牌中相同效果类型的数量
        card_effect_set = set(card.effect_types)
        similar_count = 0

        for existing in existing_cards:
            existing_effect_set = set(existing.effect_types)
            # 如果效果类型有50%以上重叠，认为是相似卡牌
            overlap = len(card_effect_set & existing_effect_set)
            total = len(card_effect_set | existing_effect_set)
            if total > 0 and overlap / total >= 0.5:
                similar_count += 1

        if is_core_stacking:
            # 核心状态堆叠卡：多张是好事
            # 0-2张：轻微加分（鼓励构建体系）
            # 3-5张：正常（体系完整）
            # 6+张：轻微惩罚（过度冗余）
            if similar_count <= 2:
                return -100.0  # 负数表示扣分，这里返回负数让penalty减少总分，所以用负负得正
            elif similar_count <= 5:
                return 0.0  # 无惩罚
            else:
                return (similar_count - 5) * 150.0  # 轻微惩罚
        else:
            # 非核心卡：过多重复不好
            # 0-2张：无惩罚
            # 3-4张：轻度惩罚
            # 5+张：重度惩罚
            if similar_count <= 2:
                return 0.0
            elif similar_count <= 4:
                return (similar_count - 2) * 150.0
            else:
                return 300.0 + (similar_count - 4) * 250.0

    @staticmethod
    def evaluate_drink(
        drink: DrinkInfo,
        plan_type: ProducePlanType,
        stamina_ratio: float,
        current_resources: dict[str, int],
    ) -> EffectEvaluation:
        """评估P饮料的价值

        Args:
            drink: 饮料信息
            plan_type: 计划类型
            stamina_ratio: 体力比率
            current_resources: 当前资源状态

        Returns:
            EffectEvaluation: 评估结果
        """
        base_score = 0.0
        plan_match_bonus = 0.0
        context_bonus = 0.0

        # 基础效果分数
        for effect_type in drink.effect_types:
            try:
                effect_enum = ExamEffectType(effect_type)
                base_score += EFFECT_BASE_WEIGHTS.get(effect_enum, 0.0) * 0.3
            except ValueError:
                continue

        # 计划匹配加成
        plan_core = PLAN_CORE_EFFECTS.get(plan_type, set())
        for effect_type in drink.effect_types:
            try:
                effect_enum = ExamEffectType(effect_type)
                if effect_enum in plan_core:
                    plan_match_bonus += 250.0
            except ValueError:
                continue

        # 上下文加成：体力不足时，回复类饮料价值提升
        if stamina_ratio < 0.4:
            for effect_type in drink.effect_types:
                if ExamEffectType.BLOCK.value in effect_type:
                    context_bonus += 500.0 * (1.0 - stamina_ratio)

        # 稀有度加成
        rarity_bonus = RARITY_BONUS.get(drink.rarity.upper(), 0.0)

        return EffectEvaluation(
            base_score=base_score,
            plan_match_bonus=plan_match_bonus,
            rarity_bonus=rarity_bonus,
            context_bonus=context_bonus,
            duplicate_penalty=0.0,
        )

    @staticmethod
    def evaluate_item(
        item: ItemInfo,
        plan_type: ProducePlanType,
    ) -> EffectEvaluation:
        """评估P物品的价值

        Args:
            item: 物品信息
            plan_type: 计划类型

        Returns:
            EffectEvaluation: 评估结果
        """
        base_score = 0.0
        plan_match_bonus = 0.0

        # 基础效果分数
        for effect_type in item.effect_types:
            try:
                effect_enum = ExamEffectType(effect_type)
                base_score += EFFECT_BASE_WEIGHTS.get(effect_enum, 0.0) * 0.4
            except ValueError:
                continue

        # 计划匹配加成
        plan_core = PLAN_CORE_EFFECTS.get(plan_type, set())
        for effect_type in item.effect_types:
            try:
                effect_enum = ExamEffectType(effect_type)
                if effect_enum in plan_core:
                    plan_match_bonus += 300.0
            except ValueError:
                continue

        # 考试效果物品额外加成
        context_bonus = 200.0 if item.is_exam_effect else 0.0

        # 稀有度加成
        rarity_bonus = RARITY_BONUS.get(item.rarity.upper(), 0.0)

        return EffectEvaluation(
            base_score=base_score,
            plan_match_bonus=plan_match_bonus,
            rarity_bonus=rarity_bonus,
            context_bonus=context_bonus,
            duplicate_penalty=0.0,
        )


__all__ = [
    "EffectEvaluation",
    "EffectEvaluator",
    "EFFECT_BASE_WEIGHTS",
    "PLAN_CORE_EFFECTS",
]
