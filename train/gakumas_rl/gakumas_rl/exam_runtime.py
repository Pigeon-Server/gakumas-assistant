"""考试阶段运行时，负责根据主数据解释卡牌、饮料、触发器与状态效果。"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np

from .data import MasterDataRepository, ScenarioSpec
from .idol_config import build_initial_exam_deck
from .loadout import IdolLoadout
from .produce_item_interpreter import RuntimeExamStatusEnchantSpec
from .reward_config import RewardConfig, build_reward_config


CARD_ZONE_MAP = {
    'ProduceCardPositionType_Deck': 'deck',
    'ProduceCardPositionType_DeckAll': 'deck',
    'ProduceCardPositionType_DeckGrave': 'deck_grave',
    'ProduceCardPositionType_Hand': 'hand',
    'ProduceCardPositionType_Hold': 'hold',
    'ProduceCardPositionType_Lost': 'lost',
    'ProduceCardPositionType_NotLost': 'not_lost',
    'ProduceCardPositionType_Playing': 'playing',
    'ProduceCardPositionType_Target': 'target',
}

MOVE_POSITION_MAP = {
    'ProduceCardMovePositionType_DeckFirst': 'deck_first',
    'ProduceCardMovePositionType_DeckLast': 'deck_last',
    'ProduceCardMovePositionType_DeckRandom': 'deck_random',
    'ProduceCardMovePositionType_Grave': 'grave',
    'ProduceCardMovePositionType_Hand': 'hand',
    'ProduceCardMovePositionType_Hold': 'hold',
    'ProduceCardMovePositionType_Lost': 'lost',
    'ProduceCardMovePositionType_Unknown': 'grave',
}

PHASE_TURN_VALUE = {
    'ProduceExamPhaseType_ExamStartTurn',
    'ProduceExamPhaseType_ExamEndTurn',
    'ProduceExamPhaseType_ExamEndTurnInterval',
    'ProduceExamPhaseType_ExamTurnInterval',
    'ProduceExamPhaseType_ExamTurnTimer',
}

SCALAR_RESOURCE_TYPES = {
    'ProduceExamEffectType_ExamCardPlayAggressive': 'aggressive',
    'ProduceExamEffectType_ExamReview': 'review',
    'ProduceExamEffectType_ExamLessonBuff': 'lesson_buff',
    'ProduceExamEffectType_ExamFullPowerPoint': 'full_power_point',
}

DURATION_RESOURCE_TYPES = {
    'ProduceExamEffectType_ExamParameterBuff': 'parameter_buff',
}

COST_RESOURCE_MAP = {
    'ExamCostType_ExamCardPlayAggressive': 'aggressive',
    'ExamCostType_ExamReview': 'review',
    'ExamCostType_ExamParameterBuff': 'parameter_buff',
    'ExamCostType_ExamParameterBuffMultiplePerTurn': 'parameter_buff_multiple_per_turn',
    'ExamCostType_ExamLessonBuff': 'lesson_buff',
    'ExamCostType_ExamFullPowerPoint': 'full_power_point',
}

GROW_EFFECT_COST_RESOURCE_MAP = {
    'ProduceCardGrowEffectType_CostAdd': 'stamina',
    'ProduceCardGrowEffectType_CostReduce': 'stamina',
    'ProduceCardGrowEffectType_CostAggressiveAdd': 'aggressive',
    'ProduceCardGrowEffectType_CostAggressiveReduce': 'aggressive',
    'ProduceCardGrowEffectType_CostReviewAdd': 'review',
    'ProduceCardGrowEffectType_CostReviewReduce': 'review',
    'ProduceCardGrowEffectType_CostParameterBuffAdd': 'parameter_buff',
    'ProduceCardGrowEffectType_CostParameterBuffReduce': 'parameter_buff',
    'ProduceCardGrowEffectType_CostParameterBuffMultiplePerTurnReduce': 'parameter_buff_multiple_per_turn',
    'ProduceCardGrowEffectType_CostLessonBuffAdd': 'lesson_buff',
    'ProduceCardGrowEffectType_CostLessonBuffReduce': 'lesson_buff',
    'ProduceCardGrowEffectType_CostFullPowerPointAdd': 'full_power_point',
    'ProduceCardGrowEffectType_CostFullPowerPointReduce': 'full_power_point',
    'ProduceCardGrowEffectType_CostPenetrateAdd': 'penetrate',
    'ProduceCardGrowEffectType_CostPenetrateReduce': 'penetrate',
}

LESSON_EFFECT_TYPES = {
    'ProduceExamEffectType_ExamLessonFix',
    'ProduceExamEffectType_ExamLessonDependExamReview',
    'ProduceExamEffectType_ExamLessonDependExamCardPlayAggressive',
    'ProduceExamEffectType_ExamLessonDependBlock',
    'ProduceExamEffectType_ExamLessonDependParameterBuff',
    'ProduceExamEffectType_ExamLessonDependPlayCardCountSum',
    'ProduceExamEffectType_ExamLessonDependStaminaConsumptionSum',
    'ProduceExamEffectType_ExamLessonDependBlockConsumptionSum',
    'ProduceExamEffectType_ExamLessonDependBlockAndSearchCount',
    'ProduceExamEffectType_ExamLessonPerSearchCount',
    'ProduceExamEffectType_ExamLessonFullPowerPoint',
    'ProduceExamEffectType_ExamLessonAddMultipleLessonBuff',
    'ProduceExamEffectType_ExamMultipleLessonBuffLesson',
    'ProduceExamEffectType_ExamLessonAddMultipleParameterBuff',
}

STANCE_PHASES = {
    'ProduceExamPhaseType_ExamStanceChangeConcentration': 'concentration',
    'ProduceExamPhaseType_ExamStanceChangeFullPower': 'full_power',
    'ProduceExamPhaseType_ExamStanceChangePreservation': 'preservation',
}

EXAM_REWARD_MODES = ('score', 'clear')
FULL_POWER_POINT_THRESHOLD = 10.0
STATUS_CHANGE_TRIGGER_ORIGINS = {'card', 'drink'}
TURN_COLOR_ORDER = ('vocal', 'dance', 'visual')
TURN_COLOR_INDEX = {color: index for index, color in enumerate(TURN_COLOR_ORDER)}
TURN_COLOR_LABELS = {
    'vocal': 'Vocal',
    'dance': 'Dance',
    'visual': 'Visual',
}

PLAN_REWARD_FAMILY = {
    'ProducePlanType_Plan1': 'sense',
    'ProducePlanType_Plan2': 'logic',
    'ProducePlanType_Plan3': 'anomaly',
}

REWARD_PROFILE_CONFIGS: dict[str, dict[str, float]] = {
    'score': {
        'shape_scale': 1.85,
        'goal_weight': 0.95,
        'eval_weight': 1.55,
        'archetype_weight': 1.20,
        'risk_weight': 0.60,
        'efficiency_weight': 0.20,
        'turn_window_weight': 0.55,
        'efficiency_gate': 0.92,
        'efficiency_overshoot_penalty': 0.12,
        'terminal_pass_reward': 5.5,
        'terminal_eval_weight': 5.0,
        'terminal_stamina_weight': 0.45,
        'terminal_speed_weight': 0.40,
        'terminal_failure_weight': 5.0,
        'terminal_force_end_bonus': 1.2,
        'terminal_nia_bonus': 1.25,
        'lesson_clear_reward': 4.0,
        'lesson_perfect_reward': 6.5,
        'overshoot_penalty': 0.55,
    },
    'clear': {
        'shape_scale': 1.55,
        'goal_weight': 1.45,
        'eval_weight': 1.00,
        'archetype_weight': 0.75,
        'risk_weight': 1.10,
        'efficiency_weight': 0.45,
        'turn_window_weight': 0.30,
        'efficiency_gate': 0.75,
        'efficiency_overshoot_penalty': 1.10,
        'terminal_pass_reward': 7.2,
        'terminal_eval_weight': 3.8,
        'terminal_stamina_weight': 0.90,
        'terminal_speed_weight': 0.95,
        'terminal_failure_weight': 7.0,
        'terminal_force_end_bonus': 1.7,
        'terminal_nia_bonus': 1.10,
        'lesson_clear_reward': 5.4,
        'lesson_perfect_reward': 8.2,
        'overshoot_penalty': 2.20,
    },
}

NEGATIVE_TIMED_EFFECT_TYPES = {
    'ProduceExamEffectType_ExamBlockAddDown',
    'ProduceExamEffectType_ExamBlockRestriction',
    'ProduceExamEffectType_ExamGimmickLessonDebuff',
    'ProduceExamEffectType_ExamGimmickParameterDebuff',
    'ProduceExamEffectType_ExamGimmickSlump',
    'ProduceExamEffectType_ExamPanic',
    'ProduceExamEffectType_ExamStaminaRecoverRestriction',
}

ANTI_DEBUFF_EFFECT_TYPES = {
    'ProduceExamEffectType_ExamBlockAddDown',
    'ProduceExamEffectType_ExamBlockRestriction',
    'ProduceExamEffectType_ExamGimmickLessonDebuff',
    'ProduceExamEffectType_ExamGimmickParameterDebuff',
    'ProduceExamEffectType_ExamGimmickPlayCardLimit',
    'ProduceExamEffectType_ExamGimmickSleepy',
    'ProduceExamEffectType_ExamGimmickSlump',
    'ProduceExamEffectType_ExamGimmickStartTurnCardDrawDown',
    'ProduceExamEffectType_ExamPanic',
    'ProduceExamEffectType_ExamStaminaRecoverRestriction',
}


@dataclass
class ExamActionCandidate:
    """环境包装层暴露出的一个考试动作。"""

    label: str
    kind: str
    payload: dict[str, Any]


@dataclass
class RuntimeCard:
    """考试运行时中的可变卡牌实例。"""

    uid: int
    card_id: str
    upgrade_count: int
    base_card: dict[str, Any]
    grow_effect_ids: list[str] = field(default_factory=list)
    card_status_enchant_id: str = ''
    transient_effect_ids: list[str] = field(default_factory=list)
    transient_trigger_ids: list[str] = field(default_factory=list)
    play_count_bonus: int = 0

    def effect_ids(self) -> list[str]:
        """返回这张运行时卡当前生效的出牌效果 id 列表。"""

        effect_ids = [str(effect.get('produceExamEffectId') or '') for effect in self.base_card.get('playEffects', [])]
        effect_ids.extend(self.transient_effect_ids)
        return [value for value in effect_ids if value]

    def trigger_ids(self) -> list[str]:
        """返回这张运行时卡当前绑定的触发器 id 列表。"""

        trigger_ids = []
        if self.base_card.get('playProduceExamTriggerId'):
            trigger_ids.append(str(self.base_card['playProduceExamTriggerId']))
        trigger_ids.extend(self.transient_trigger_ids)
        return [value for value in trigger_ids if value]

    def clone(self, uid: int | None = None) -> 'RuntimeCard':
        """复制一张运行时卡，并可选替换 uid。"""

        return RuntimeCard(
            uid=self.uid if uid is None else uid,
            card_id=self.card_id,
            upgrade_count=self.upgrade_count,
            base_card=self.base_card,
            grow_effect_ids=list(self.grow_effect_ids),
            card_status_enchant_id=self.card_status_enchant_id,
            transient_effect_ids=list(self.transient_effect_ids),
            transient_trigger_ids=list(self.transient_trigger_ids),
            play_count_bonus=self.play_count_bonus,
        )


@dataclass
class TimedExamEffect:
    """带持续回合或次数限制的运行时考试效果。"""

    uid: int
    effect: dict[str, Any]
    remaining_turns: int | None
    remaining_count: int | None
    source: str


@dataclass
class TriggeredEnchant:
    """已挂载到场上的状态附魔及其触发配置。"""

    uid: int
    enchant_id: str
    trigger_id: str
    effect_ids: list[str]
    remaining_turns: int | None
    remaining_count: int | None
    source: str
    source_identity: str = ''


@dataclass
class ScheduledEffect:
    """将在未来某个回合结算的延迟效果。"""

    effect_id: str
    fire_turn: int
    remaining_count: int | None


@dataclass
class ExamEvent:
    """运行时事件记录。"""

    turn: int
    event_type: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class CardSelection:
    """检索结果，包含命中的卡牌和候选池大小。"""

    selected: list[RuntimeCard]
    pool_size: int


def format_audition_row_selector(row: dict[str, Any] | None) -> str | None:
    """把考试主数据行格式化成 battle runtime 可消费的显式 selector。"""

    if not row:
        return None
    return f"{str(row.get('id') or '')}:{int(row.get('number') or 0)}"


def default_audition_row_selector(
    repository: MasterDataRepository,
    scenario: ScenarioSpec,
    stage_type: str | None = None,
    loadout: IdolLoadout | None = None,
    fan_votes: float | None = None,
) -> str | None:
    """为外层兼容调用提供一个稳定的默认 battle row selector。"""

    audition_difficulty_id = str(loadout.stat_profile.audition_difficulty_id or '') if loadout is not None else ''
    selected_row = repository.select_audition_row(
        scenario,
        stage_type or scenario.default_stage,
        audition_difficulty_id=audition_difficulty_id or None,
        fan_votes=fan_votes,
    )
    return format_audition_row_selector(selected_row)


class ExamRuntime:
    """数据驱动的考试战斗运行时。

    模型只看到结构化卡牌特征，卡名仅保留在调试标签里；实际逻辑依赖效果类型、
    触发 phase 和检索定义来推进。
    """

    def __init__(
        self,
        repository: MasterDataRepository,
        scenario: ScenarioSpec,
        stage_type: str | None = None,
        seed: int | None = None,
        deck: list[dict[str, Any]] | None = None,
        drinks: list[dict[str, Any]] | None = None,
        initial_status_enchant_ids: list[str] | None = None,
        initial_status_enchants: Iterable[RuntimeExamStatusEnchantSpec | dict[str, Any]] | None = None,
        loadout: IdolLoadout | None = None,
        starting_stamina: float | None = None,
        exam_score_bonus_multiplier: float | None = None,
        fan_votes: float | None = None,
        reward_mode: str = 'score',
        reward_config: RewardConfig | None = None,
        audition_row_id: str | None = None,
        battle_kind: str | None = None,
        lesson_type: str | None = None,
        lesson_types: Iterable[str] | None = None,
        lesson_post_clear_types: Iterable[str] | None = None,
        lesson_sequence: Iterable[str] | None = None,
        lesson_target_value: float | None = None,
        lesson_perfect_value: float | None = None,
        lesson_perfect_recovery_per_turn: float = 0.0,
        turn_limit: int | None = None,
    ):
        """初始化考试运行时，并根据偶像卡选择难度 profile、初始 deck 与开场附魔。"""

        if reward_mode not in EXAM_REWARD_MODES:
            raise ValueError(f'Unsupported exam reward mode: {reward_mode}')

        self.repository = repository
        self.scenario = scenario
        self.reward_config: RewardConfig = reward_config or build_reward_config(reward_mode)
        self.stage_type = stage_type or scenario.default_stage
        self.np_random = np.random.default_rng(seed)
        self.exam_setting = repository.load_table('ExamSetting').first('p_exam_setting-1') or {}
        self.card_searches = repository.load_table('ProduceCardSearch')
        self.grow_effects = repository.load_table('ProduceCardGrowEffect')
        self.card_status_enchants = repository.load_table('ProduceCardStatusEnchant')
        self.exam_gimmicks = repository.load_table('ProduceExamGimmickEffectGroup')
        self.random_pools = repository.load_table('ProduceCardRandomPool')
        self.card_pools = repository.load_table('ProduceCardPool')
        self.loadout = loadout
        self.starting_stamina = float(starting_stamina) if starting_stamina is not None else None
        self.reward_mode = reward_mode
        self.audition_row_id = str(audition_row_id or '')
        self.battle_kind = self._resolve_battle_kind(battle_kind)
        self.explicit_fan_votes = (
            float(fan_votes)
            if fan_votes is not None and self._fan_vote_enabled_for_mode()
            else None
        )
        self.selected_battle_row = self._resolve_selected_battle_row()
        self.profile = self._build_battle_profile(self.selected_battle_row)
        self.fan_votes = self._resolve_initial_fan_votes(self.explicit_fan_votes)
        self.initial_deck_rows = list(deck or build_initial_exam_deck(repository, scenario, rng=self.np_random, loadout=loadout))
        self.initial_drinks = list(
            drinks
            or repository.build_drink_inventory(
                scenario,
                rng=self.np_random,
                plan_type=loadout.stat_profile.plan_type if loadout is not None else None,
            )
        )
        self.default_lesson_type = str(lesson_type or self._infer_stage_lesson_type(self.stage_type))
        resolved_lesson_types = tuple(str(value) for value in (lesson_types or ()) if str(value or ''))
        if not resolved_lesson_types:
            resolved_lesson_types = (self.default_lesson_type,)
        self.default_lesson_types = resolved_lesson_types
        self.lesson_post_clear_types = tuple(
            str(value)
            for value in (lesson_post_clear_types or ())
            if str(value or '')
        )
        self.lesson_sequence = tuple(str(value) for value in (lesson_sequence or ()) if str(value or ''))
        self.lesson_target_value = (
            float(lesson_target_value)
            if lesson_target_value is not None
            else (float(self.profile.get('base_score') or 0.0) if self._uses_clear_training_rules() else None)
        )
        self.lesson_perfect_value = float(lesson_perfect_value) if lesson_perfect_value is not None else None
        self.lesson_perfect_recovery_per_turn = float(lesson_perfect_recovery_per_turn or 0.0)
        combined_enchants: dict[tuple[str, str, str], dict[str, Any]] = {}
        for enchant_id in initial_status_enchant_ids or []:
            if enchant_id:
                combined_enchants[(str(enchant_id), 'produce', '')] = {
                    'enchant_id': str(enchant_id),
                    'effect_turn': None,
                    'effect_count': None,
                    'source': 'produce',
                }
        for enchant_spec in initial_status_enchants or ():
            if isinstance(enchant_spec, RuntimeExamStatusEnchantSpec):
                payload = {
                    'enchant_id': str(enchant_spec.enchant_id),
                    'effect_turn': enchant_spec.effect_turn,
                    'effect_count': enchant_spec.effect_count,
                    'source': str(enchant_spec.source or 'produce'),
                    'source_identity': str(enchant_spec.source_identity or ''),
                }
            else:
                payload = {
                    'enchant_id': str(enchant_spec.get('enchant_id') or ''),
                    'effect_turn': enchant_spec.get('effect_turn'),
                    'effect_count': enchant_spec.get('effect_count'),
                    'source': str(enchant_spec.get('source') or 'produce'),
                    'source_identity': str(enchant_spec.get('source_identity') or ''),
                }
            if not payload['enchant_id']:
                continue
            key = (payload['enchant_id'], payload['source'], payload['source_identity'])
            combined_enchants[key] = payload
        if loadout is not None:
            loadout_specs = getattr(loadout, 'exam_status_enchant_specs', ())
            if loadout_specs:
                for spec in loadout_specs:
                    if not spec.enchant_id:
                        continue
                    source_identity = str(spec.source_identity or loadout.produce_item_id or spec.enchant_id)
                    key = (str(spec.enchant_id), 'produce_item', source_identity)
                    combined_enchants[key] = {
                        'enchant_id': str(spec.enchant_id),
                        'effect_turn': spec.effect_turn,
                        'effect_count': spec.effect_count,
                        'source': 'produce_item',
                        'source_identity': source_identity,
                    }
            else:
                for enchant_id in loadout.exam_status_enchant_ids:
                    if enchant_id:
                        key = (str(enchant_id), 'produce_item', str(loadout.produce_item_id or enchant_id))
                        combined_enchants[key] = {
                            'enchant_id': str(enchant_id),
                            'effect_turn': None,
                            'effect_count': None,
                            'source': 'produce_item',
                            'source_identity': str(loadout.produce_item_id or enchant_id),
                        }
        self.initial_status_enchants = list(combined_enchants.values())

        self._uid_counter = 0
        self.turn = 0
        self.max_turns = int(turn_limit or self.profile.get('turns') or scenario.exam_turns)
        self.terminated = False
        self.last_info: dict[str, Any] = {}

        self.deck: deque[RuntimeCard] = deque()
        self.hand: list[RuntimeCard] = []
        self.grave: list[RuntimeCard] = []
        self.hold: list[RuntimeCard] = []
        self.lost: list[RuntimeCard] = []
        self.playing: list[RuntimeCard] = []
        self.current_card: RuntimeCard | None = None
        self.drinks: list[dict[str, Any]] = []

        self.score = 0.0
        self.stamina = 0.0
        self.max_stamina = 0.0
        self.stance = 'neutral'
        self.stance_level = 0
        self.stance_locked = False
        self.play_limit = 1
        self.start_turn_draw_penalty = 0
        self.extra_turns = 0
        self.panic_cost_overrides: dict[int, float] = {}
        parameter_values = np.array(
            [
                float(loadout.stat_profile.vocal) if loadout is not None else 0.0,
                float(loadout.stat_profile.dance) if loadout is not None else 0.0,
                float(loadout.stat_profile.visual) if loadout is not None else 0.0,
            ],
            dtype=np.float32,
        )
        parameter_limit = float(getattr(scenario, 'parameter_growth_limit', 0.0) or 0.0)
        if parameter_limit > 0:
            parameter_values = np.clip(parameter_values, 0.0, parameter_limit)
        else:
            parameter_values = np.clip(parameter_values, 0.0, None)
        self.parameter_stats = (float(parameter_values[0]), float(parameter_values[1]), float(parameter_values[2]))
        resolved_base_score_bonus = (
            float(exam_score_bonus_multiplier)
            if exam_score_bonus_multiplier is not None
            else float(loadout.exam_score_bonus_multiplier)
            if loadout is not None
            else self._default_score_bonus_multiplier()
        )
        if self.scenario.route_type == 'nia':
            resolved_base_score_bonus *= self._fan_vote_score_multiplier()
        self.base_score_bonus_multiplier = max(resolved_base_score_bonus, 0.25)
        self.current_turn_color = ''
        self.turn_color_history: list[str] = []
        self.score_bonus_multiplier = self.base_score_bonus_multiplier

        self.resources: dict[str, float] = defaultdict(float)
        self.turn_counters: Counter[str] = Counter()
        self.total_counters: Counter[str] = Counter()
        self.search_history: Counter[str] = Counter()
        self.active_effects: list[TimedExamEffect] = []
        self.active_enchants: list[TriggeredEnchant] = []
        self.scheduled_effects: list[ScheduledEffect] = []
        self.gimmick_rows: list[dict[str, Any]] = []
        self._resolving_enchant_uids: set[int] = set()
        self._resolved_gimmick_keys: set[tuple[str, int, int]] = set()
        self.forbidden_card_search_ids: Counter[str] = Counter()
        self.lesson_cleared = False
        self.clear_state = 'ongoing'
        self.event_log: list[ExamEvent] = []

        # ── 奖励追踪状态 ──
        self._milestone_flags: dict[str, bool] = {
            '25': False, '50': False, '75': False, '100': False,
        }
        self._consecutive_end_turns: int = 0
        self._prev_score: float = 0.0
        self._prev_resource_stock: float = 0.0

    def _record_event(self, event_type: str, detail: dict[str, Any] | None = None) -> None:
        """记录一条运行时事件。"""

        self.event_log.append(ExamEvent(turn=self.turn, event_type=event_type, detail=detail or {}))

    def _next_uid(self) -> int:
        """生成运行时对象使用的递增 uid。"""

        self._uid_counter += 1
        return self._uid_counter

    def _resolve_battle_kind(self, battle_kind: str | None) -> str:
        """解析当前战斗是考试/课程哪一类。"""

        normalized = str(battle_kind or '').strip().lower()
        if normalized in {'lesson', 'exam', 'audition'}:
            return normalized
        return 'lesson' if 'Lesson' in str(self.stage_type or '') else 'exam'

    def _infer_stage_lesson_type(self, stage_type: str | None) -> str:
        """从 stepType 推断固定课程类型；考试未提供序列时返回 Unknown。"""

        stage = str(stage_type or '')
        if 'LessonVocal' in stage:
            return 'ProduceStepLessonType_LessonVocal'
        if 'LessonDance' in stage:
            return 'ProduceStepLessonType_LessonDance'
        if 'LessonVisual' in stage:
            return 'ProduceStepLessonType_LessonVisual'
        if 'LessonSp' in stage:
            return 'ProduceStepLessonType_LessonSp'
        return 'ProduceStepLessonType_Unknown'

    def _resolve_selected_battle_row(self) -> dict[str, Any] | None:
        """按显式 row id 或偶像卡难度配置选择本局实际考试行。"""

        if self.battle_kind == 'lesson':
            return None
        all_rows = self.repository.audition_rows(self.scenario, self.stage_type)
        if self.audition_row_id:
            matched = []
            for row in all_rows:
                row_id = str(row.get('id') or '')
                row_number = int(row.get('number') or 0)
                row_selector = f'{row_id}:{row_number}'
                battle_config_id = str(row.get('produceExamBattleConfigId') or '')
                if self.audition_row_id in {row_selector, battle_config_id}:
                    matched.append(row)
                elif self.audition_row_id == row_id and len([item for item in all_rows if str(item.get('id') or '') == row_id]) == 1:
                    matched.append(row)
            if not matched:
                raise ValueError(f'Audition row not found for stage {self.stage_type}: {self.audition_row_id}')
            if len(matched) > 1:
                examples = ', '.join(
                    f"{str(row.get('id') or '')}:{int(row.get('number') or 0)}"
                    for row in matched[:5]
                )
                raise ValueError(
                    f'Audition selector matched multiple rows for stage {self.stage_type}: '
                    f'{self.audition_row_id}. use one of [{examples}]'
                )
            return matched[0]
        audition_difficulty_id = str(self.loadout.stat_profile.audition_difficulty_id or '') if self.loadout is not None else ''
        rows = self.repository.audition_rows(
            self.scenario,
            self.stage_type,
            audition_difficulty_id=audition_difficulty_id or None,
        )
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]
        if self.scenario.route_type == 'nia' and self.explicit_fan_votes is not None:
            selected = self.repository.select_audition_row(
                self.scenario,
                self.stage_type,
                audition_difficulty_id=audition_difficulty_id or None,
                fan_votes=self.explicit_fan_votes,
            )
            if selected is not None:
                return selected
        row_ids = ', '.join(str(format_audition_row_selector(row) or '') for row in rows[:5])
        raise ValueError(
            f'Ambiguous audition rows for stage {self.stage_type}; '
            f'provide loadout.audition_difficulty_id or audition_row_id. examples=[{row_ids}]'
        )

    def _build_battle_profile(self, stage_row: dict[str, Any] | None) -> dict[str, float]:
        """从显式关卡行构造 battle profile；无关卡行时回退到聚合配置。"""

        if stage_row is not None:
            config = self.repository.battle_config_map.get(str(stage_row.get('produceExamBattleConfigId') or '')) or {}
            weight_vector = np.array(
                [
                    float(config.get('vocal') or self.scenario.score_weights[0]),
                    float(config.get('dance') or self.scenario.score_weights[1]),
                    float(config.get('visual') or self.scenario.score_weights[2]),
                ],
                dtype=np.float32,
            )
            weight_sum = float(weight_vector.sum())
            if weight_sum > 0:
                weight_vector = weight_vector / weight_sum
            return {
                'base_score': float(stage_row.get('baseScore') or 0.0),
                'force_end_score': float(stage_row.get('forceEndScore') or 0.0),
                'rank_threshold': float(stage_row.get('rankThreshold') or 0.0),
                'parameter_baseline': float(stage_row.get('parameterBaseLine') or 0.0),
                'fan_vote_baseline': float(stage_row.get('voteCountBaseLine') or 0.0),
                'fan_vote_requirement': float(stage_row.get('voteCount') or 0.0),
                'turns': float(config.get('turn') or self.scenario.exam_turns),
                'vocal_weight': float(weight_vector[0]),
                'dance_weight': float(weight_vector[1]),
                'visual_weight': float(weight_vector[2]),
            }
        profile = dict(self.repository.stage_thresholds.get((self.scenario.produce_id, self.stage_type), {}))
        if profile:
            return profile
        return {
            'base_score': 2000.0,
            'force_end_score': 0.0,
            'rank_threshold': 3.0,
            'parameter_baseline': 180.0 if self.scenario.route_type == 'nia' else 140.0,
            'fan_vote_baseline': 0.0,
            'fan_vote_requirement': 0.0,
            'turns': float(self.scenario.exam_turns),
            'vocal_weight': self.scenario.score_weights[0],
            'dance_weight': self.scenario.score_weights[1],
            'visual_weight': self.scenario.score_weights[2],
        }

    def _fan_vote_enabled_for_mode(self) -> bool:
        """考试模式下始终保留 NIA fan vote 相关规则。"""

        return self.battle_kind != 'lesson' and self.scenario.route_type == 'nia'

    def _turn_color_enabled(self) -> bool:
        """仅 lesson 模式禁用考试回合颜色。"""

        return self.battle_kind != 'lesson'

    def _uses_clear_training_rules(self) -> bool:
        """lesson 与 clear reward 共享训练课目标规则。"""

        return self.battle_kind == 'lesson' or self.reward_mode == 'clear'

    def _current_clear_target(self) -> float:
        """返回当前训练课 clear 目标值。"""

        if not self._uses_clear_training_rules():
            return 0.0
        if self.lesson_target_value is not None:
            return max(float(self.lesson_target_value), 0.0)
        return max(float(self.profile.get('base_score') or 0.0), 0.0)

    def _current_perfect_target(self) -> float:
        """返回当前训练课 perfect 目标值。"""

        if not self._uses_clear_training_rules():
            return 0.0
        if self.lesson_perfect_value is not None:
            return max(float(self.lesson_perfect_value), 0.0)
        clear_target = self._current_clear_target()
        if self.reward_mode == 'clear' and clear_target > 0:
            return clear_target * 2.0
        return 0.0

    def _reported_fan_vote_baseline(self) -> float:
        """返回当前考试的 fan vote 基准。"""

        if not self._fan_vote_enabled_for_mode():
            return 0.0
        return float(self.profile.get('fan_vote_baseline') or 0.0)

    def _reported_fan_vote_requirement(self) -> float:
        """返回当前考试的 fan vote 门槛。"""

        if not self._fan_vote_enabled_for_mode():
            return 0.0
        return float(self.profile.get('fan_vote_requirement') or 0.0)

    def _resolve_initial_fan_votes(self, explicit_fan_votes: float | None) -> float:
        """解析当前考试上下文的初始 fan vote。"""

        if not self._fan_vote_enabled_for_mode():
            return 0.0
        if explicit_fan_votes is not None:
            return max(float(explicit_fan_votes), 0.0)
        baseline = self._reported_fan_vote_baseline()
        if baseline > 0:
            return baseline
        requirement = self._reported_fan_vote_requirement()
        return max(requirement, 0.0)

    def _current_lesson_type(self) -> str:
        """返回当前战斗上下文中的课程类型。"""

        lesson_types = self._current_lesson_types()
        if lesson_types:
            return lesson_types[0]
        if self.lesson_sequence and 1 <= self.turn <= len(self.lesson_sequence):
            return self.lesson_sequence[self.turn - 1]
        return self.default_lesson_type

    def _current_lesson_types(self) -> tuple[str, ...]:
        """返回当前 lesson 上下文有效的课程类型集合。"""

        if self.lesson_sequence and 1 <= self.turn <= len(self.lesson_sequence):
            return (self.lesson_sequence[self.turn - 1],)
        if self.battle_kind == 'lesson' and self.clear_state in {'cleared', 'perfect'} and self.lesson_post_clear_types:
            return self.lesson_post_clear_types
        return self.default_lesson_types

    def _lesson_target_remaining(self) -> float:
        """课程模式下离清课目标还差多少。"""

        if not self._uses_clear_training_rules():
            return 0.0
        target = self._current_clear_target()
        return max(target - self.score, 0.0) if target > 0 else 0.0

    def _lesson_perfect_remaining(self) -> float:
        """课程模式下离 Perfect Lesson 还差多少。"""

        if not self._uses_clear_training_rules():
            return 0.0
        perfect = self._current_perfect_target()
        return max(perfect - self.score, 0.0) if perfect > 0 else 0.0

    def _remaining_turns_after_current_action(self) -> int:
        """当前动作结算后仍然剩余的回合数。"""

        return max(self.max_turns - self.turn, 0)

    def _perfect_finish_recovery_turns(self) -> int:
        """Perfect 结算时按手册返回应折算回体的剩余回合数。"""

        if self.battle_kind == 'lesson':
            return max(self.max_turns - self.turn + 1, 0)
        return self._remaining_turns_after_current_action()

    def _update_clear_state_after_score_change(self) -> None:
        """按课程目标/Perfect 门槛或考试最高分更新本局通关状态。

        帮助文档：試験に挑んだ際に、スコアが獲得できる最大値に到達すると、
        挑んだ試験は自動的に合格となり終了します。
        """

        # 考试模式下，达到 force_end_score 时自动合格结束
        if self.battle_kind != 'lesson' and not self._uses_clear_training_rules():
            force_end_score = float(self.profile.get('force_end_score') or 0.0)
            if force_end_score > 0 and self.score >= force_end_score:
                if self.clear_state != 'force_end':
                    self.clear_state = 'force_end'
                    self._record_event('force_end_reached', {
                        'score': self.score,
                        'force_end_score': force_end_score,
                    })
                self.terminated = True
            return

        if not self._uses_clear_training_rules():
            return
        target = self._current_clear_target()
        perfect = self._current_perfect_target()
        if perfect > 0 and self.score >= perfect:
            self.lesson_cleared = True
            if self.clear_state != 'perfect':
                recovery = self._perfect_finish_recovery_turns() * self.lesson_perfect_recovery_per_turn
                if recovery > 0:
                    self.stamina = min(self.max_stamina, self.stamina + recovery)
            self.clear_state = 'perfect'
            self.terminated = True
            return
        if target > 0 and self.score >= target:
            self.lesson_cleared = True
            self.clear_state = 'cleared'
            if self.reward_mode == 'clear' and self.battle_kind != 'lesson':
                self.terminated = True

    def reset(self) -> None:
        """重置考试战斗状态并进入第一个回合。"""

        self.event_log = []
        deck_cards = self._build_runtime_deck(self.initial_deck_rows)
        self.np_random.shuffle(deck_cards)
        self.turn = 0
        self.terminated = False
        self.last_info = {}
        self.deck = deque(deck_cards)
        self.hand = []
        self.grave = []
        self.hold = []
        self.lost = []
        self.playing = []
        self.current_card = None
        self.drinks = [dict(row) for row in self.initial_drinks]

        self.score = 0.0
        default_stamina = 12.0 if self.scenario.route_type == 'first_star' else 15.0
        loadout_stamina = float(self.loadout.stat_profile.stamina) if self.loadout is not None else 0.0
        self.max_stamina = loadout_stamina if loadout_stamina > 0 else default_stamina
        opening_stamina = self.max_stamina if self.starting_stamina is None else self.starting_stamina
        self.stamina = float(np.clip(opening_stamina, 0.0, self.max_stamina))
        self.stance = 'neutral'
        self.stance_level = 0
        self.stance_locked = False
        self.play_limit = self._base_play_limit()
        self.start_turn_draw_penalty = 0
        self.extra_turns = 0
        self.panic_cost_overrides = {}
        self.current_turn_color = ''
        self.turn_color_history = []
        self.score_bonus_multiplier = self.base_score_bonus_multiplier

        self.resources = defaultdict(float)
        self.resources['block'] = 0.0
        self.resources['full_power_point'] = 0.0
        self.resources['stamina_consumption_down'] = 0.0
        self.resources['parameter_buff_multiple_per_turn'] = 0.0
        self.resources['panic'] = 0.0
        self.resources['slump'] = 0.0
        self.resources['enthusiastic'] = 0.0

        self.turn_counters = Counter()
        self.total_counters = Counter()
        self.search_history = Counter()
        self.active_effects = []
        self.active_enchants = []
        self.scheduled_effects = []
        self.gimmick_rows = self._load_stage_gimmicks()
        self._resolving_enchant_uids = set()
        self._resolved_gimmick_keys = set()
        self.forbidden_card_search_ids = Counter()
        self.lesson_cleared = False
        self.clear_state = 'ongoing'
        self._milestone_flags = {'25': False, '50': False, '75': False, '100': False}
        self._consecutive_end_turns = 0
        self._prev_score = 0.0
        self._prev_resource_stock = 0.0
        self._sync_stance_resources()
        self._sync_effect_resources()
        self._sync_forbidden_search_resources()

        for enchant_spec in self.initial_status_enchants:
            self._register_initial_enchant(
                str(enchant_spec['enchant_id']),
                source=str(enchant_spec.get('source') or 'produce'),
                remaining_turns=enchant_spec.get('effect_turn'),
                remaining_count=enchant_spec.get('effect_count'),
                source_identity=str(enchant_spec.get('source_identity') or ''),
            )

        self._dispatch_phase('ProduceExamPhaseType_ExamStartExam')
        self._start_turn()

    def legal_actions(self) -> list[ExamActionCandidate]:
        """枚举当前可执行的出牌、饮料和结束回合动作。"""

        if self.terminated:
            return [ExamActionCandidate(label='结束', kind='noop', payload={'kind': 'noop'})]

        candidates: list[ExamActionCandidate] = []
        for card in self.hand:
            if self._can_play_card(card):
                candidates.append(
                    ExamActionCandidate(
                        label=self._card_label(card),
                        kind='card',
                        payload={'kind': 'card', 'uid': card.uid},
                    )
                )
        for index, drink in enumerate(self.drinks):
            if self._can_use_drink(drink):
                candidates.append(
                    ExamActionCandidate(
                        label=self.repository.drink_name(drink),
                        kind='drink',
                        payload={'kind': 'drink', 'index': index},
                    )
                )
        end_turn_label = 'SKIP' if self.battle_kind == 'lesson' else '结束回合'
        candidates.append(ExamActionCandidate(label=end_turn_label, kind='end_turn', payload={'kind': 'end_turn'}))
        return candidates

    def step(self, action: ExamActionCandidate) -> tuple[float, dict[str, Any]]:
        """执行一个考试动作，并返回增量奖励与状态摘要。"""

        rc = self.reward_config
        reward_before = self._reward_signal()
        score_before = self.score
        resource_stock_before = self._resource_stock()

        # ── 执行动作 ──
        skipped_turn = False
        if action.kind == 'card':
            card = self._remove_hand_card(int(action.payload['uid']))
            if card is None:
                raise KeyError(f'Card uid {action.payload["uid"]} is not in hand')
            self._play_card(card)
            self._consecutive_end_turns = 0
            if not self.terminated and not self._has_remaining_play_window():
                self._end_turn(skipped=False)
        elif action.kind == 'drink':
            self._use_drink(int(action.payload['index']))
            self._consecutive_end_turns = 0
        elif action.kind == 'end_turn':
            skipped_turn = self.turn_counters['play_count'] <= 0
            self._consecutive_end_turns += 1
            if not self.terminated:
                self._end_turn(skipped=skipped_turn)
        else:
            self.terminated = True

        reward_after = self._reward_signal()

        # ── 基础潜势差分奖励 ──
        reward = reward_after - reward_before

        # ── 动作类型微奖励 ──
        if action.kind == 'card' and rc.card_play_reward != 0.0:
            reward += rc.card_play_reward
        if action.kind == 'drink' and rc.drink_use_reward != 0.0:
            reward += rc.drink_use_reward

        # ── 分数差分密集奖励 ──
        if rc.score_delta_scale != 0.0:
            target = float(self.profile.get('base_score') or self._target_score() or 1.0)
            score_delta = (self.score - score_before) / max(target, 1.0)
            reward += score_delta * rc.score_delta_scale

        # ── 资源增量密集奖励 ──
        if rc.resource_gain_scale != 0.0:
            resource_delta = self._resource_stock() - resource_stock_before
            if resource_delta > 0:
                reward += self._resource_curve(resource_delta, 10.0) * rc.resource_gain_scale

        # ── 里程碑一次性奖励 ──
        reward += self._check_milestones()

        # ── 截断 / 惩罚 ──
        if action.kind == 'end_turn' and skipped_turn and rc.skip_turn_penalty != 0.0:
            reward += rc.skip_turn_penalty
        if rc.consecutive_end_turn_penalty != 0.0 and self._consecutive_end_turns >= 2:
            reward += rc.consecutive_end_turn_penalty * (self._consecutive_end_turns - 1)
        if self.terminated and self.stamina <= 0 and rc.stamina_death_penalty != 0.0:
            reward += rc.stamina_death_penalty

        # ── 全局缩放与裁剪 ──
        reward *= rc.reward_scale
        if rc.reward_clip > 0:
            reward = max(min(reward, rc.reward_clip), -rc.reward_clip)

        # ── 更新追踪 ──
        self._prev_score = self.score
        self._prev_resource_stock = self._resource_stock()

        target_score = self._target_score()
        info = {
            'action': action.label,
            'kind': action.kind,
            'reward_mode': self.reward_mode,
            'battle_kind': self.battle_kind,
            'score': self.score,
            'target_score': target_score,
            'score_ratio': self.score / max(target_score, 1.0),
            'stamina': self.stamina,
            'turn': self.turn,
            'stance': self.stance,
            'turn_color': self.current_turn_color,
            'turn_color_label': self.turn_color_label(),
            'fan_votes': self.fan_votes,
            'fan_vote_baseline': self._reported_fan_vote_baseline(),
            'fan_vote_requirement': self._reported_fan_vote_requirement(),
            'clear_state': self.clear_state,
            'deck': len(self.deck),
            'hand': len(self.hand),
            'grave': len(self.grave),
            'hold': len(self.hold),
            'lost': len(self.lost),
        }
        if self._uses_clear_training_rules():
            info['lesson_cleared'] = self.lesson_cleared
            info['lesson_target_remaining'] = self._lesson_target_remaining()
            info['lesson_perfect_remaining'] = self._lesson_perfect_remaining()
        self.last_info = info
        return reward, info

    def _check_milestones(self) -> float:
        """检查分数里程碑并返回一次性奖励。"""

        rc = self.reward_config
        primary_ratio = self._primary_goal_ratio()
        bonus = 0.0
        for threshold, key, reward_val in (
            (0.25, '25', rc.milestone_25_reward),
            (0.50, '50', rc.milestone_50_reward),
            (0.75, '75', rc.milestone_75_reward),
            (1.00, '100', rc.milestone_100_reward),
        ):
            if reward_val != 0.0 and not self._milestone_flags[key] and primary_ratio >= threshold:
                self._milestone_flags[key] = True
                bonus += reward_val
        return bonus

    def _target_score(self) -> float:
        """返回当前考试阶段的目标分数。"""

        if self._uses_clear_training_rules():
            clear_target = self._current_clear_target()
            perfect_target = self._current_perfect_target()
            if self.battle_kind == 'lesson' and self.lesson_cleared and perfect_target > 0:
                return max(perfect_target, 1.0)
            if clear_target > 0:
                return max(clear_target, 1.0)
        return float(self.profile.get('base_score') or 1.0)

    def _future_gimmick_count(self) -> int:
        """统计当前回合之后仍未触发的场地 gimmick 数量。"""

        return sum(1 for row in self.gimmick_rows if int(row.get('startTurn') or 0) > self.turn)

    def _resource_stock(self) -> float:
        """估算当前保留下来的正向资源库存，用于 clear 模式的节奏奖励。"""

        return (
            self.resources['review'] * 0.8
            + self.resources['aggressive'] * 0.8
            + self.resources['parameter_buff']
            + self.resources['lesson_buff']
            + self.resources['block'] * 0.3
            + self.resources['concentration'] * 0.4
            + self.resources['full_power_point'] * 0.3
        )

    def _reward_profile_config(self) -> RewardConfig:
        """返回当前生效的奖励配置对象。"""

        return self.reward_config

    def _plan_reward_family(self) -> str:
        """把偶像 plan type 归并成奖励层使用的资源流派。"""

        if self.loadout is None:
            return 'common'
        plan_type = str(self.loadout.stat_profile.plan_type or '')
        return PLAN_REWARD_FAMILY.get(plan_type, 'common')

    def _turn_progress_ratio(self) -> float:
        """返回当前局面已消耗的回合进度。"""

        return min(max(self.turn / max(self.max_turns, 1), 0.0), 1.0)

    def _remaining_turn_ratio(self) -> float:
        """返回当前局面剩余回合占比。"""

        return max(self.max_turns - self.turn + 1, 0) / max(self.max_turns, 1)

    def _stamina_ratio(self) -> float:
        """返回当前体力占比。"""

        return self.stamina / max(self.max_stamina, 1.0)

    def _primary_goal_ratio(self) -> float:
        """返回当前局面对主要通关目标的进度比。"""

        if self._uses_clear_training_rules():
            target = float(self._current_clear_target() or self._target_score() or 0.0)
        else:
            target = float(self.profile.get('base_score') or self._target_score() or 0.0)
        return self.score / max(target, 1.0)

    def _secondary_goal_ratio(self) -> float:
        """返回当前局面对次级终局目标的进度比。"""

        if self._uses_clear_training_rules():
            target = float(self._current_perfect_target() or self._current_clear_target() or self._target_score() or 0.0)
            return self.score / max(target, 1.0)
        force_end_score = float(self.profile.get('force_end_score') or 0.0)
        if force_end_score > 0:
            return self.score / max(force_end_score, 1.0)
        return self._primary_goal_ratio()

    def _fan_vote_reference(self) -> float:
        """返回 NIA fan vote 的主数据基准值。"""

        if not self._fan_vote_enabled_for_mode():
            return 1.0
        baseline = self._reported_fan_vote_baseline()
        if baseline > 0:
            return baseline
        requirement = self._reported_fan_vote_requirement()
        if requirement > 0:
            return requirement
        return 1.0

    def _fan_vote_progress(self) -> float:
        """返回当前 fan vote 相对主数据基准的进度。"""

        if not self._fan_vote_enabled_for_mode():
            return 0.0
        return max(self.fan_votes, 0.0) / max(self._fan_vote_reference(), 1.0)

    def _fan_vote_score_multiplier(self) -> float:
        """把 fan vote 进度映射到 NIA 局内得分倍率。"""

        if not self._fan_vote_enabled_for_mode():
            return 1.0
        progress = max(self._fan_vote_progress(), 0.0)
        if progress <= 1.0:
            return 0.80 + 0.20 * math.sqrt(progress)
        bonus = 0.15 * math.log1p(progress - 1.0) / math.log(3.0)
        return min(1.0 + bonus, 1.20)

    def _fan_vote_gain_stage_scale(self) -> float:
        """估算当前 NIA 难度下 fan vote 奖励的量级。"""

        baseline = self._reported_fan_vote_baseline()
        requirement = self._reported_fan_vote_requirement()
        return max(requirement * 0.50, baseline * 0.15, 800.0)

    def estimate_fan_vote_gain(self, score: float | None = None) -> float:
        """按当前总分估算本场 NIA 结束时可获得的 fan vote。"""

        if not self._fan_vote_enabled_for_mode():
            return 0.0
        current_score = max(float(self.score if score is None else score), 0.0)
        base_score = float(self.profile.get('base_score') or 0.0)
        score_ratio = current_score / max(base_score, 1.0)
        quality = min(self._score_value_curve(score_ratio), 1.4)
        return self._fan_vote_gain_stage_scale() * (0.35 + 0.65 * quality)

    def _fan_vote_gain_value(self) -> float:
        """把 fan vote 奖励估算压成 reward 可用的有界值。"""

        if not self._fan_vote_enabled_for_mode():
            return 0.0
        return min(self.estimate_fan_vote_gain() / max(self._fan_vote_gain_stage_scale(), 1.0), 1.4)

    def _clear_finish_value(self) -> float:
        """clear 终止时动态估算 perfect 段的潜在补偿价值。"""

        if self.reward_mode != 'clear':
            return 0.0
        stock_value = self._resource_curve(self._resource_stock(), 26.0)
        stamina_value = self._stamina_ratio()
        turn_value = self._remaining_turn_ratio()
        if self.clear_state == 'perfect':
            return 0.90 + stock_value * 0.85 + stamina_value * 0.20 + turn_value * 0.25
        if self.clear_state == 'cleared':
            return stock_value * (0.70 + 0.30 * turn_value) + stamina_value * 0.35 + turn_value * 0.25
        return 0.0

    def _score_value_curve(self, ratio: float) -> float:
        """把原始得分比压成边际递减的终局价值近似。"""

        normalized = max(float(ratio), 0.0)
        progress = min(normalized, 1.0)
        overshoot = max(normalized - 1.0, 0.0)
        return progress + (math.log1p(overshoot * 3.0) / math.log(4.0)) * 0.35

    def _resource_curve(self, value: float, soft_cap: float) -> float:
        """对资源库存做边际递减压缩，避免囤积奖励线性膨胀。"""

        clipped = max(float(value), 0.0)
        if clipped <= 0.0:
            return 0.0
        return min(math.log1p(clipped) / math.log1p(max(soft_cap, 1.0)), 1.5)

    def _judging_alignment(self) -> float:
        """估算当前三维和审查权重的匹配程度。"""

        if self.battle_kind == 'lesson':
            return 1.0 / 3.0
        weights = np.array(
            [
                float(self.profile.get('vocal_weight') or 0.0),
                float(self.profile.get('dance_weight') or 0.0),
                float(self.profile.get('visual_weight') or 0.0),
            ],
            dtype=np.float32,
        )
        weight_sum = float(weights.sum())
        stats = np.clip(np.array(self.parameter_stats, dtype=np.float32), 0.0, None)
        stat_sum = float(stats.sum())
        if weight_sum <= 1e-6 or stat_sum <= 1e-6:
            return 1.0 / 3.0
        return float(np.dot(stats / stat_sum, weights / weight_sum))

    def _turn_window_value(self) -> float:
        """把当前回合颜色窗口转换成 reward 可用的局面价值。"""

        if self.battle_kind == 'lesson':
            return 0.0
        base_multiplier = max(self.base_score_bonus_multiplier, 0.25)
        color_ratio = self._effective_score_bonus_multiplier() / base_multiplier
        return float(np.clip(math.tanh((color_ratio - 1.0) * 1.4), -0.75, 0.75))

    def _phi_goal(self) -> float:
        """潜势函数：离真实目标还有多远。"""

        primary_ratio = self._primary_goal_ratio()
        secondary_ratio = self._secondary_goal_ratio()
        progress = min(primary_ratio, 1.0)
        secondary_progress = min(secondary_ratio, 1.0)
        pace_gap = progress - self._turn_progress_ratio()
        finish_bonus = 0.0
        if self._uses_clear_training_rules():
            if self.clear_state == 'cleared':
                finish_bonus += 0.35 + self._clear_finish_value() * 0.15
            elif self.clear_state == 'perfect':
                finish_bonus += 0.80 + self._clear_finish_value() * 0.10
        else:
            if primary_ratio >= 1.0:
                finish_bonus += 0.25 + self._remaining_turn_ratio() * 0.35
            force_end_score = float(self.profile.get('force_end_score') or 0.0)
            if force_end_score > 0 and self.score >= force_end_score:
                finish_bonus += 0.55
        return progress * 1.15 + secondary_progress * 0.35 + pace_gap * 0.55 + finish_bonus

    def _phi_eval(self, config: RewardConfig) -> float:
        """潜势函数：当前局面的终局评价边际价值。"""

        primary_ratio = self._primary_goal_ratio()
        score_value = self._score_value_curve(primary_ratio)
        judging_alignment = self._judging_alignment()
        turn_window_value = self._turn_window_value()
        alignment_weight = float(config.judging_alignment_weight)
        value = score_value * (0.70 + judging_alignment * alignment_weight)
        value += turn_window_value * float(config.turn_window_weight)
        if self.reward_mode == 'clear':
            value += self._clear_finish_value() * 0.20
        if self._fan_vote_enabled_for_mode():
            value += self._fan_vote_gain_value() * 0.30
        return value

    def _phi_archetype(self) -> float:
        """潜势函数：按 plan 感知未来资源的可兑现价值。"""

        remaining_turn_ratio = self._remaining_turn_ratio()
        delayed_scale = 0.35 + 0.65 * remaining_turn_ratio
        window_scale = 1.0 if self.reward_mode == 'clear' else 0.65 + 0.35 * max(self._turn_window_value(), 0.0)
        family = self._plan_reward_family()
        family_scale = {
            'sense': self.reward_config.sense_resource_scale,
            'logic': self.reward_config.logic_resource_scale,
            'anomaly': self.reward_config.anomaly_resource_scale,
        }.get(family, 1.0)
        if family == 'sense':
            return family_scale * (
                self._resource_curve(self.resources['parameter_buff'], 8.0) * (1.10 * window_scale)
                + self._resource_curve(self.resources['review'], 14.0) * (0.95 * delayed_scale)
                + self._resource_curve(self.resources['parameter_buff_multiple_per_turn'], 3.0) * delayed_scale
                + self._resource_curve(self.resources['lesson_buff'], 10.0) * 0.45
            )
        if family == 'logic':
            return family_scale * (
                self._resource_curve(self.resources['aggressive'], 18.0) * (1.05 * delayed_scale)
                + self._resource_curve(self.resources['block'], 18.0) * (0.90 * delayed_scale)
                + self._resource_curve(self.resources['lesson_buff'], 10.0) * 0.35
                + self._resource_curve(self.resources['stamina_consumption_down'], 6.0) * 0.25
            )
        if family == 'anomaly':
            full_power_progress = self._resource_curve(self.resources['full_power_point'], FULL_POWER_POINT_THRESHOLD)
            if self.resources['full_power_point'] >= FULL_POWER_POINT_THRESHOLD:
                full_power_progress += 0.25
            return family_scale * (
                self._resource_curve(self.resources['concentration'], 2.0) * (0.95 * window_scale)
                + self._resource_curve(self.resources['preservation'], 3.0) * (0.85 * delayed_scale)
                + self._resource_curve(self.resources['over_preservation'], 2.0) * delayed_scale
                + full_power_progress * 1.15
                + self._resource_curve(self.resources['enthusiastic'], 10.0) * (0.65 + 0.35 * max(window_scale, delayed_scale))
            )
        return (
            self._resource_curve(self.resources['parameter_buff'], 6.0) * 0.65
            + self._resource_curve(self.resources['review'], 10.0) * 0.55
            + self._resource_curve(self.resources['aggressive'], 10.0) * 0.55
            + self._resource_curve(self.resources['lesson_buff'], 10.0) * 0.40
        )

    def _phi_risk(self) -> float:
        """潜势函数：失败风险、负面状态和体力断线风险。"""
        negative_penalty = (
            self.resources['sleepy'] * 0.30
            + self.resources['panic'] * 0.28
            + self.resources['slump'] * 0.24
            + self.resources['active_skill_forbidden'] * 0.35
            + self.resources['mental_skill_forbidden'] * 0.35
        )
        tempo_pressure = max(self._turn_progress_ratio() - min(self._primary_goal_ratio(), 1.20), 0.0)
        low_stamina = max(0.35 - self._stamina_ratio(), 0.0) / 0.35
        future_gimmick_pressure = min(self._future_gimmick_count(), 3) * 0.08 * max(1.0 - self._primary_goal_ratio(), 0.0)
        remaining_drinks = sum(1 for drink in self.drinks if not drink.get('_consumed'))
        safety = self._stamina_ratio() * 0.75 + min(remaining_drinks, 2) * 0.12
        return safety - negative_penalty - tempo_pressure * 0.90 - low_stamina * 0.80 - future_gimmick_pressure

    def _phi_efficiency(self, config: RewardConfig) -> float:
        """潜势函数：在主要目标可达后，鼓励更高效地收官。"""

        efficiency_gate = float(config.efficiency_gate)
        gate = max(min((self._primary_goal_ratio() - efficiency_gate) / max(1.0 - efficiency_gate, 1e-6), 1.0), 0.0)
        if gate <= 0.0:
            return 0.0
        remaining_drinks = sum(1 for drink in self.drinks if not drink.get('_consumed'))
        spare_value = (
            self._stamina_ratio() * 0.70
            + self._remaining_turn_ratio() * 0.45
            + min(remaining_drinks, 2) * 0.18
            + min(self._resource_stock(), 12.0) / 12.0 * 0.15
        )
        overshoot = max(self._primary_goal_ratio() - 1.0, 0.0)
        return gate * spare_value - overshoot * float(config.efficiency_overshoot_penalty)

    def _potential_value(self, config: RewardConfig) -> float:
        """统一潜势函数；step() 会对它做差分得到 shaping。"""

        return (
            float(config.goal_weight) * self._phi_goal()
            + float(config.eval_weight) * self._phi_eval(config)
            + float(config.archetype_weight) * self._phi_archetype()
            + float(config.risk_weight) * self._phi_risk()
            + float(config.efficiency_weight) * self._phi_efficiency(config)
        )

    def _terminal_utility(self, config: RewardConfig) -> float:
        """终局效用：只在回合真正结束时发放。"""

        if not self.terminated:
            return 0.0
        stamina_term = self._stamina_ratio() * float(config.terminal_stamina_weight)
        speed_term = self._remaining_turn_ratio() * float(config.terminal_speed_weight)
        overshoot_penalty = float(config.overshoot_penalty)
        if self._uses_clear_training_rules():
            clear_target = float(self._current_clear_target() or self._target_score() or 0.0)
            clear_ratio = self.score / max(clear_target, 1.0)
            perfect_ratio = self._secondary_goal_ratio()
            utility = self._score_value_curve(perfect_ratio) * float(config.terminal_eval_weight)
            if self.clear_state == 'perfect':
                utility += float(config.lesson_perfect_reward) + speed_term + self._clear_finish_value() * 0.25
            elif self.clear_state == 'cleared':
                utility += float(config.lesson_clear_reward) + self._clear_finish_value()
            else:
                utility -= float(config.terminal_failure_weight) * min(max(1.0 - clear_ratio, 0.0), 1.0)
            utility += stamina_term
            utility -= max(perfect_ratio - 1.0, 0.0) * overshoot_penalty
            return utility

        primary_ratio = self._primary_goal_ratio()
        utility = self._score_value_curve(primary_ratio) * float(config.terminal_eval_weight)
        if primary_ratio >= 1.0:
            utility += float(config.terminal_pass_reward) + speed_term
        else:
            utility -= float(config.terminal_failure_weight) * min(max(1.0 - primary_ratio, 0.0), 1.0)
        utility += stamina_term
        utility -= max(primary_ratio - 1.0, 0.0) * overshoot_penalty
        force_end_score = float(self.profile.get('force_end_score') or 0.0)
        if force_end_score > 0 and self.score >= force_end_score:
            utility += float(config.terminal_force_end_bonus)
        if self._fan_vote_enabled_for_mode():
            utility += self._fan_vote_gain_value() * float(config.terminal_nia_bonus)
        return utility

    def _utility_reward_signal(self) -> float:
        """统一奖励信号：终局效用加 potential-based shaping。"""

        config = self._reward_profile_config()
        return self._terminal_utility(config) + float(config.shape_scale) * self._potential_value(config)

    def _score_reward_signal(self) -> float:
        """高分导向 reward，内部走统一终局效用 + shaping 框架。"""

        return self._utility_reward_signal()

    def _clear_reward_signal(self) -> float:
        """过线导向 reward，内部走统一终局效用 + shaping 框架。"""

        return self._utility_reward_signal()

    def _reward_signal(self) -> float:
        """按 reward mode 构造供训练使用的平滑奖励信号。"""

        if self.reward_mode == 'clear':
            return self._clear_reward_signal()
        return self._score_reward_signal()

    def _build_runtime_deck(self, card_rows: Iterable[dict[str, Any]]) -> list[RuntimeCard]:
        """把主数据卡行转换成可变的运行时卡实例。"""

        cards = []
        for row in card_rows:
            grow_effect_ids = [
                str(value)
                for value in row.get('produceCardGrowEffectIds', []) or row.get('growEffectIds', [])
                if value
            ]
            runtime_card = RuntimeCard(
                uid=self._next_uid(),
                card_id=str(row.get('id')),
                upgrade_count=int(row.get('upgradeCount') or 0),
                base_card=row,
                grow_effect_ids=list(grow_effect_ids),
                card_status_enchant_id=str(row.get('produceCardStatusEnchantId') or ''),
            )
            cards.append(runtime_card)
            self._record_event('card_acquired', {
                'card_id': runtime_card.card_id,
                'card_name': self.repository.card_name(row),
                'upgrade_count': runtime_card.upgrade_count,
                'destination': 'deck',
                'source': 'initial_deck',
            })
            initial_add_count = 0
            for grow_effect_id in grow_effect_ids:
                grow_row = self.grow_effects.first(str(grow_effect_id))
                if grow_row and str(grow_row.get('effectType') or '') == 'ProduceCardGrowEffectType_InitialAdd':
                    initial_add_count += 1
            for _ in range(initial_add_count):
                duplicate = runtime_card.clone(uid=self._next_uid())
                cards.append(duplicate)
                self._record_event('card_acquired', {
                    'card_id': duplicate.card_id,
                    'card_name': self.repository.card_name(row),
                    'upgrade_count': duplicate.upgrade_count,
                    'destination': 'deck',
                    'source': 'initial_add',
                })
        return cards

    def _default_score_bonus_multiplier(self) -> float:
        """根据偶像属性与亲爱度估算默认分数倍率。"""

        if self.loadout is None:
            return 1.0
        weights = np.array(
            [
                float(self.profile.get('vocal_weight') or self.scenario.score_weights[0]),
                float(self.profile.get('dance_weight') or self.scenario.score_weights[1]),
                float(self.profile.get('visual_weight') or self.scenario.score_weights[2]),
            ],
            dtype=np.float32,
        )
        stats = np.array(self.parameter_stats, dtype=np.float32)
        weighted_parameter = float(np.dot(stats, weights))
        baseline = float(self.profile.get('parameter_baseline') or 0.0)
        if baseline <= 0.0:
            baseline = 180.0 if self.scenario.route_type == 'nia' else 140.0
        dearness_ratio = 1.0 + min(max(int(self.loadout.dearness_level), 0), 20) * 0.01
        return max((weighted_parameter / max(baseline, 1.0)) * dearness_ratio, 0.25)

    def _turn_color_probabilities(self) -> np.ndarray:
        """根据当前培育属性计算考试回合颜色分布。"""

        stats = np.clip(np.array(self.parameter_stats, dtype=np.float32), 0.0, None)
        total = float(stats.sum())
        if total <= 0:
            return np.full(len(TURN_COLOR_ORDER), 1.0 / len(TURN_COLOR_ORDER), dtype=np.float32)
        return stats / total

    def _roll_turn_color(self) -> str:
        """按当前属性权重为新回合抽取颜色。"""

        probabilities = self._turn_color_probabilities()
        index = int(self.np_random.choice(len(TURN_COLOR_ORDER), p=probabilities))
        return TURN_COLOR_ORDER[index]

    def turn_color_label(self) -> str:
        """返回当前回合颜色的人类可读标签。"""

        return TURN_COLOR_LABELS.get(self.current_turn_color, '')

    def turn_color_one_hot(self) -> np.ndarray:
        """把当前回合颜色编码成 one-hot；无颜色时返回全零。"""

        encoded = np.zeros(len(TURN_COLOR_ORDER), dtype=np.float32)
        index = TURN_COLOR_INDEX.get(self.current_turn_color)
        if index is not None:
            encoded[index] = 1.0
        return encoded

    def _effective_score_bonus_multiplier(self) -> float:
        """根据当前回合颜色把局内基准倍率换算成实际得分倍率。"""

        if not self._turn_color_enabled():
            return self.base_score_bonus_multiplier
        color_index = TURN_COLOR_INDEX.get(self.current_turn_color)
        if color_index is None:
            return self.base_score_bonus_multiplier
        stats = np.clip(np.array(self.parameter_stats, dtype=np.float32), 0.0, None)
        if color_index >= len(stats):
            return self.base_score_bonus_multiplier
        expected_stat = float(np.dot(stats, self._turn_color_probabilities()))
        if expected_stat <= 1e-6:
            return self.base_score_bonus_multiplier
        selected_stat = float(stats[color_index])
        return max(self.base_score_bonus_multiplier * (selected_stat / expected_stat), 0.0)

    def _refresh_turn_score_bonus_multiplier(self) -> None:
        """同步当前回合的实际得分倍率。"""

        self.score_bonus_multiplier = self._effective_score_bonus_multiplier()

    def _base_play_limit(self) -> int:
        """返回当前基础每回合可出牌次数。"""

        bonus = 0
        for timed in self.active_effects:
            if str(timed.effect.get('effectType') or '') == 'ProduceExamEffectType_ExamPlayableValueAdd':
                bonus += int(round(self._count_value(timed.effect)))
        return 1 + max(bonus, 0)

    def _score_gain(self, value: float) -> float:
        """对分数类收益统一套用考试倍率。"""

        return float(value) * self.score_bonus_multiplier

    def _raw_effect_value(self, effect: dict[str, Any]) -> float:
        """读取效果主数值，不附带运行时修正。"""

        return float(effect.get('effectValue1') or 0)

    def _timed_effects_of_type(self, effect_type: str) -> list[TimedExamEffect]:
        """按效果类型筛选当前激活中的持续效果。"""

        return [item for item in self.active_effects if str(item.effect.get('effectType') or '') == effect_type]

    def _has_timed_effect(self, effect_type: str) -> bool:
        """判断某个持续效果当前是否在场。"""

        return any(str(item.effect.get('effectType') or '') == effect_type for item in self.active_effects)

    def _timed_effect_stack_value(self, effect_type: str, *, default_value: float = 1.0) -> float:
        """累加同类持续效果的主数值，供追加触发类效果复用。"""

        total = 0.0
        for timed in self._timed_effects_of_type(effect_type):
            value = self._raw_effect_value(timed.effect)
            total += value if value > 0 else default_value
        return total

    def _current_card_grow_total(self, grow_effect_type: str) -> float:
        """汇总当前出牌卡上某种成长效果的数值。"""

        if self.current_card is None:
            return 0.0
        total = 0.0
        for grow_effect in self._card_grow_rows(self.current_card):
            if str(grow_effect.get('effectType') or '') == grow_effect_type:
                total += float(grow_effect.get('value') or 0)
        return total

    def _current_card_ratio_bonus(self, grow_effect_type: str) -> float:
        """把当前卡的成长效果加成换算成千分比。"""

        return max(self._current_card_grow_total(grow_effect_type), 0.0) / 1000.0

    def _adjust_direct_gain(self, value: float, add_grow_type: str = '', reduce_grow_type: str = '') -> float:
        """把卡牌成长对直接收益的加减成统一折算出来。"""

        updated = float(value)
        if add_grow_type:
            updated += self._current_card_grow_total(add_grow_type)
        if reduce_grow_type:
            updated -= self._current_card_grow_total(reduce_grow_type)
        if value > 0:
            return max(updated, 1.0 if reduce_grow_type else 0.0)
        return max(updated, 0.0)

    def _card_matches_search(self, card: RuntimeCard, search_id: str) -> bool:
        """判断当前卡是否命中给定的检索条件。"""

        if not search_id:
            return False
        selection = self._search_cards(search_id, acting_card=card, target_card=card)
        return any(candidate.uid == card.uid for candidate in selection.selected)

    def _consume_timed_effect_uid(self, uid: int) -> None:
        """消费一层按次数生效的持续效果。"""

        next_effects: list[TimedExamEffect] = []
        for timed in self.active_effects:
            if timed.uid == uid and timed.remaining_count is not None:
                timed.remaining_count -= 1
            if timed.remaining_count is not None and timed.remaining_count <= 0:
                continue
            next_effects.append(timed)
        self.active_effects = next_effects
        self._sync_effect_resources()

    def _matched_play_count_buff_effects(self, card: RuntimeCard) -> list[TimedExamEffect]:
        """返回对当前卡生效的追加发动持续效果。"""

        matched: list[TimedExamEffect] = []
        for timed in self.active_effects:
            if str(timed.effect.get('effectType') or '') != 'ProduceExamEffectType_ExamCardSearchEffectPlayCountBuff':
                continue
            search_id = str(timed.effect.get('produceCardSearchId') or '')
            if self._card_matches_search(card, search_id):
                matched.append(timed)
        return matched

    def _matching_search_stamina_overrides(self, card: RuntimeCard) -> list[TimedExamEffect]:
        """返回对当前卡生效的消耗体力覆写效果。"""

        matched: list[TimedExamEffect] = []
        for timed in self.active_effects:
            if str(timed.effect.get('effectType') or '') != 'ProduceExamEffectType_ExamSearchPlayCardStaminaConsumptionChange':
                continue
            search_id = str(timed.effect.get('produceCardSearchId') or '')
            if self._card_matches_search(card, search_id):
                matched.append(timed)
        return matched

    def _card_repeat_bonus(self, card: RuntimeCard) -> int:
        """计算当前卡会被额外重复发动几次。"""

        bonus = 0
        for timed in self._matched_play_count_buff_effects(card):
            bonus += int(round(self._raw_effect_value(timed.effect)))
        return max(bonus, 0)

    def _consume_card_play_buffs(self, card: RuntimeCard) -> None:
        """在卡牌结算后消费按次数触发的目标卡持续效果。"""

        consume_uids = {
            timed.uid
            for timed in self._matched_play_count_buff_effects(card) + self._matching_search_stamina_overrides(card)
            if timed.remaining_count is not None
        }
        for uid in sorted(consume_uids):
            self._consume_timed_effect_uid(uid)

    def _effective_card_trigger_ids(self, card: RuntimeCard) -> list[str]:
        """解析卡牌当前实际使用的出牌触发器。"""

        trigger_ids = []
        if card.base_card.get('playProduceExamTriggerId'):
            trigger_ids.append(str(card.base_card['playProduceExamTriggerId']))
        trigger_ids.extend(card.transient_trigger_ids)
        for grow_effect in self._card_grow_rows(card):
            if str(grow_effect.get('effectType') or '') != 'ProduceCardGrowEffectType_PlayTriggerChange':
                continue
            next_trigger = str(grow_effect.get('playProduceExamTriggerId') or '')
            if not next_trigger:
                continue
            target_ids = {str(value) for value in grow_effect.get('targetPlayEffectProduceExamTriggerIds', []) if value}
            if target_ids:
                if not trigger_ids or not any(trigger_id in target_ids for trigger_id in trigger_ids):
                    continue
            trigger_ids = [next_trigger]
        return [value for value in trigger_ids if value]

    def _resolved_card_play_effects(self, card: RuntimeCard) -> list[dict[str, str]]:
        """解析卡牌当前实际会结算的直接出牌效果与触发器。"""

        resolved = [
            {
                'effect_id': str(play_effect.get('produceExamEffectId') or ''),
                'trigger_id': str(play_effect.get('produceExamTriggerId') or ''),
            }
            for play_effect in card.base_card.get('playEffects', [])
            if play_effect.get('produceExamEffectId')
        ]
        resolved.extend({'effect_id': effect_id, 'trigger_id': ''} for effect_id in card.transient_effect_ids if effect_id)
        for grow_effect in self._card_grow_rows(card):
            effect_type = str(grow_effect.get('effectType') or '')
            if effect_type == 'ProduceCardGrowEffectType_EffectAdd' and grow_effect.get('playProduceExamEffectId'):
                resolved.append({'effect_id': str(grow_effect['playProduceExamEffectId']), 'trigger_id': ''})
            elif effect_type == 'ProduceCardGrowEffectType_EffectChange' and grow_effect.get('playProduceExamEffectId'):
                next_effect_id = str(grow_effect['playProduceExamEffectId'])
                target_ids = {str(value) for value in grow_effect.get('targetPlayProduceExamEffectIds', []) if value}
                if target_ids:
                    for item in resolved:
                        if item['effect_id'] in target_ids:
                            item['effect_id'] = next_effect_id
                elif resolved:
                    for item in resolved:
                        item['effect_id'] = next_effect_id
                else:
                    resolved.append({'effect_id': next_effect_id, 'trigger_id': ''})
            elif effect_type == 'ProduceCardGrowEffectType_PlayEffectTriggerChange' and grow_effect.get('playEffectProduceExamTriggerId'):
                next_trigger_id = str(grow_effect['playEffectProduceExamTriggerId'])
                target_ids = {str(value) for value in grow_effect.get('targetPlayEffectProduceExamTriggerIds', []) if value}
                for item in resolved:
                    if not target_ids or item['trigger_id'] in target_ids:
                        item['trigger_id'] = next_trigger_id
        return [item for item in resolved if item['effect_id']]

    def _grow_cost_resource_key(self, effect_type: str, fallback_cost_type: str) -> str:
        """把成长效果名称映射到对应的体力或资源消耗槽。"""

        mapped = GROW_EFFECT_COST_RESOURCE_MAP.get(effect_type)
        if mapped:
            return mapped
        return COST_RESOURCE_MAP.get(fallback_cost_type, fallback_cost_type)

    def _sync_effect_resources(self) -> None:
        """把持续效果投影到观测和 reward 使用的资源槽。"""

        self.resources['parameter_buff_multiple_per_turn'] = float(len(self._timed_effects_of_type('ProduceExamEffectType_ExamParameterBuffMultiplePerTurn')))
        self.resources['panic'] = float(len(self._timed_effects_of_type('ProduceExamEffectType_ExamPanic')))
        self.resources['slump'] = float(len(self._timed_effects_of_type('ProduceExamEffectType_ExamGimmickSlump')))
        self.resources['stamina_consumption_down'] = float(
            any(
                str(item.effect.get('effectType') or '')
                in {
                    'ProduceExamEffectType_ExamStaminaConsumptionDown',
                    'ProduceExamEffectType_ExamStaminaConsumptionDownFix',
                    'ProduceExamEffectType_ExamSearchPlayCardStaminaConsumptionChange',
                }
                for item in self.active_effects
            )
        )

    def _sync_forbidden_search_resources(self) -> None:
        """把搜索型禁卡状态压缩回观测资源槽。"""

        active = 0
        mental = 0
        for search_id, count in self.forbidden_card_search_ids.items():
            if count <= 0:
                continue
            search = self.card_searches.first(search_id)
            categories = {str(value) for value in (search or {}).get('cardCategories', []) if value}
            if categories and categories.issubset({'ProduceCardCategory_ActiveSkill'}):
                active += count
            if categories and categories.issubset({'ProduceCardCategory_MentalSkill'}):
                mental += count
        self.resources['active_skill_forbidden'] = float(active)
        self.resources['mental_skill_forbidden'] = float(mental)

    def _matches_forbidden_search(self, card: RuntimeCard) -> bool:
        """检查一张卡是否命中当前场上的禁卡搜索条件。"""

        return any(
            count > 0 and self._card_matches_search(card, search_id)
            for search_id, count in self.forbidden_card_search_ids.items()
        )

    def _status_change_origin(self, source: str | None) -> str:
        """把内部 effect source 归一化为状态变化来源。"""

        normalized = str(source or '')
        return normalized if normalized in STATUS_CHANGE_TRIGGER_ORIGINS else 'other'

    def _ceil_positive(self, value: float) -> float:
        """把正向浮点增量按帮助文档要求向上取整。"""

        numeric = float(value)
        if numeric <= 0:
            return 0.0
        return float(math.ceil(numeric - 1e-9))

    def _positive_count(self, value: float) -> int:
        """把正向增量转换为状态变化 phase 的整数值。"""

        return int(self._ceil_positive(value))

    def _dispatch_status_change(self, amount: float, effect_types: list[str], origin: str) -> None:
        """统一派发状态变化事件，并保留合法触发来源。"""

        phase_value = self._positive_count(amount)
        if phase_value <= 0:
            return
        self._dispatch_phase(
            'ProduceExamPhaseType_ExamStatusChange',
            phase_value=phase_value,
            effect_types=effect_types,
            status_change_origin=self._status_change_origin(origin),
        )

    def _compose_referenced_gain(self, base: float = 0.0, referenced: float = 0.0) -> float:
        """把参照值带来的增量按帮助文档规则向上取整，再叠加固定底值。"""

        return max(float(base), 0.0) + self._ceil_positive(referenced)

    def _sync_stance_resources(self) -> None:
        """把指针状态同步到观测中的数值槽。"""

        self.resources['concentration'] = float(self.stance_level if self.stance == 'concentration' else 0.0)
        self.resources['preservation'] = float(self.stance_level if self.stance == 'preservation' else 0.0)
        self.resources['over_preservation'] = 1.0 if self.stance == 'preservation' and self.stance_level >= 3 else 0.0

    def _parameter_buff_gain_value(self, effect: dict[str, Any]) -> float:
        """好调按持续回合数结算，而不是按 effectValue1。"""

        base_turns = float(effect.get('effectTurn') or 0)
        return self._apply_scalar_modifiers('ProduceExamEffectType_ExamParameterBuff', base_turns)

    def _consume_parameter_buff_multiple(self, amount: float) -> None:
        """消耗绝好调层数，并从持续效果列表中移除对应实例。"""

        remaining = int(round(amount))
        if remaining <= 0:
            return
        next_effects: list[TimedExamEffect] = []
        for timed in self.active_effects:
            effect_type = str(timed.effect.get('effectType') or '')
            if effect_type == 'ProduceExamEffectType_ExamParameterBuffMultiplePerTurn' and remaining > 0:
                remaining -= 1
                continue
            next_effects.append(timed)
        self.active_effects = next_effects
        self._sync_effect_resources()

    def _consume_anti_debuff(self, effect_type: str) -> bool:
        """低下状态无效会抵消一次负面状态或场地 debuff。"""

        if effect_type not in ANTI_DEBUFF_EFFECT_TYPES:
            return False
        if self.resources['anti_debuff'] <= 0:
            return False
        self.resources['anti_debuff'] = max(self.resources['anti_debuff'] - 1.0, 0.0)
        return True

    def _gain_enthusiastic(self, amount: float) -> float:
        """结算热意增加量，并套用热意相关修饰。"""

        updated = max(float(amount), 0.0)
        for timed in self.active_effects:
            modifier_type = str(timed.effect.get('effectType') or '')
            if modifier_type == 'ProduceExamEffectType_ExamEnthusiasticAdditive':
                updated += self._raw_effect_value(timed.effect)
            elif modifier_type == 'ProduceExamEffectType_ExamEnthusiasticMultiple':
                updated *= 1.0 + self._ratio_value(timed.effect)
        updated = max(updated, 0.0)
        self.resources['enthusiastic'] += updated
        return updated

    def _gain_block(
        self,
        amount: float,
        effect_type: str = 'ProduceExamEffectType_ExamBlock',
        status_change_origin: str = 'other',
    ) -> float:
        """统一处理元气增长，应用干劲、弱气和元气无效等修正。"""

        delta = self._apply_scalar_modifiers('ProduceExamEffectType_ExamBlock', amount)
        self.resources['block'] += delta
        if delta > 0:
            self._dispatch_status_change(delta, [effect_type], origin=status_change_origin)
        return delta

    def _panic_stamina_value(self, card: RuntimeCard) -> float:
        """在随机状态下，为当前回合缓存这张卡的随机体力消耗。"""

        if card.uid not in self.panic_cost_overrides:
            candidates = list(self.exam_setting.get('produceExamPanicStaminaCandidates') or [1, 2, 3])
            self.panic_cost_overrides[card.uid] = float(self.np_random.choice(candidates))
        return self.panic_cost_overrides[card.uid]

    def _selected_audition_row(self) -> dict[str, Any] | None:
        """按偶像卡对应难度组挑选当前考试配置行。"""

        return self.selected_battle_row

    def _load_stage_gimmicks(self) -> list[dict[str, Any]]:
        """读取当前考试关卡对应的 gimmick 列表。"""

        if self.battle_kind == 'lesson':
            return []
        stage_row = self._selected_audition_row()
        gimmick_id = str(stage_row.get('produceExamGimmickEffectGroupId') or '') if stage_row else ''
        rows = [row for row in self.exam_gimmicks.rows if str(row.get('id')) == gimmick_id] if gimmick_id else []
        rows.sort(key=lambda row: int(row.get('priority') or 0))
        return rows

    def _discard_hand_to_grave(self) -> None:
        """把回合结束后未使用的手牌全部弃到弃牌堆。"""

        if not self.hand:
            return
        self.grave.extend(self.hand)
        self.hand = []

    def _send_to_hand_or_top_deck(self, card: RuntimeCard) -> bool:
        """手牌已满时把新卡改为放到牌堆顶。"""

        hand_limit = int(self.exam_setting.get('handLimit') or 5)
        if len(self.hand) >= hand_limit:
            self.deck.appendleft(card)
            return False
        self.hand.append(card)
        return True

    def _is_offensive_card(self, card: RuntimeCard) -> bool:
        """判断一张卡是否属于可直接输出的主动技能。"""

        return str(card.base_card.get('category') or '') == 'ProduceCardCategory_ActiveSkill'

    def _start_turn(self) -> None:
        """推进到新回合，并处理开场抽牌与阶段效果。"""

        self.turn += 1
        self.turn_counters = Counter()
        self.play_limit = self._base_play_limit()
        self.panic_cost_overrides = {}
        self.forbidden_card_search_ids = Counter()
        self._sync_forbidden_search_resources()
        if self.turn > self.max_turns and self.extra_turns <= 0:
            self.current_turn_color = ''
            self._refresh_turn_score_bonus_multiplier()
            self.terminated = True
            return
        if self.extra_turns > 0 and self.turn > self.max_turns:
            self.extra_turns -= 1
        if not self._turn_color_enabled():
            self.current_turn_color = ''
        else:
            self.current_turn_color = self._roll_turn_color()
            self.turn_color_history.append(self.current_turn_color)
            self._record_event('turn_color_assigned', {
                'color': self.current_turn_color,
                'label': self.turn_color_label(),
            })
        self._refresh_turn_score_bonus_multiplier()

        # 好印象和好调都会在新回合开始时自然衰减。
        if self.resources['review'] > 0:
            self.resources['review'] = max(self.resources['review'] - 1.0, 0.0)
        if self.resources['parameter_buff'] > 0:
            self.resources['parameter_buff'] = max(self.resources['parameter_buff'] - 1.0, 0.0)

        self._decay_turn_effects()
        if self.resources['full_power_point'] >= FULL_POWER_POINT_THRESHOLD and self.stance != 'full_power':
            self.resources['full_power_point'] = max(self.resources['full_power_point'] - FULL_POWER_POINT_THRESHOLD, 0.0)
            self._enter_full_power()
        self._apply_gimmicks_for_turn(self.turn)
        if self.terminated:
            return
        self._dispatch_phase('ProduceExamPhaseType_ExamStartTurn', phase_value=self.turn)
        if self.terminated:
            return
        self._dispatch_phase('ProduceExamPhaseType_ExamTurnTimer', phase_value=self.turn)
        if self.terminated:
            return
        self._dispatch_interval_phase('ProduceExamPhaseType_ExamTurnInterval', self.turn)
        if self.terminated:
            return

        draw_count = int(self.exam_setting.get('turnStartDistribute') or 3) - self.start_turn_draw_penalty
        draw_count = max(draw_count, 0)
        self.start_turn_draw_penalty = 0
        self._draw(draw_count)
        # “次のターン、手札...” 这类链式效果要在新回合抽牌后触发，否则会打到空手牌。
        self._fire_scheduled_effects()
        if self.terminated:
            return
        self._sync_effect_resources()
        self._sync_stance_resources()

    def _remove_hand_card(self, uid: int) -> RuntimeCard | None:
        """按 uid 从手牌中取出一张卡。"""

        for index, card in enumerate(self.hand):
            if card.uid == uid:
                return self.hand.pop(index)
        return None

    def _can_play_card(self, card: RuntimeCard) -> bool:
        """检查当前状态下这张卡是否允许打出。"""

        if not self._has_remaining_play_window():
            return False
        if self._matches_forbidden_search(card):
            return False
        if not self._card_play_condition_matches(card):
            return False
        cost_stamina, cost_force = self._card_stamina_components(card)
        if self.stamina < cost_force:
            return False
        if self.stamina + self.resources['block'] < cost_stamina + cost_force:
            return False
        for resource_key, amount in self._card_resource_costs(card).items():
            if self.resources[resource_key] < amount:
                return False
        return True

    def _has_remaining_play_window(self) -> bool:
        """当前回合是否仍允许继续行动。"""

        return self.turn_counters['play_count'] < self.play_limit

    def _can_use_drink(self, drink: dict[str, Any]) -> bool:
        """饮料只能在本回合仍有剩余出牌窗口时使用。"""

        if drink.get('_consumed'):
            return False
        return self._has_remaining_play_window()

    def _refresh_play_limit(self) -> None:
        """把持续效果带来的额外出牌次数同步到当前回合。"""

        self.play_limit = max(self.play_limit, self._base_play_limit())

    def _card_play_condition_matches(self, card: RuntimeCard) -> bool:
        """判断卡面触发条件是否允许当前出牌。"""

        trigger_ids = self._effective_card_trigger_ids(card)
        if not trigger_ids:
            return True
        for trigger_id in trigger_ids:
            trigger = self.repository.exam_trigger_map.get(trigger_id)
            if not trigger:
                continue
            phase_types = [str(value) for value in trigger.get('phaseTypes', []) if value]
            if not phase_types:
                continue
            for phase_type in phase_types:
                if phase_type == 'ProduceExamPhaseType_None':
                    event_phase_value = 0
                else:
                    event_phase_value = self.turn
                event = {
                    'phase_type': phase_type,
                    'phase_value': event_phase_value,
                    'acting_card': card,
                    'effect_types': self.repository.card_exam_effect_types(card.base_card),
                }
                if self._trigger_matches(trigger, event, acting_card=card, target_card=card):
                    return True
        return False

    def _play_card(self, card: RuntimeCard) -> None:
        """执行出牌流程，并分发所有相关 phase 与效果。"""

        self.current_card = card
        self.playing = [card]

        cost_stamina, cost_force = self._card_stamina_components(card)
        spent = self._spend_stamina(
            cost_stamina,
            cost_force,
            phase_type='ProduceExamPhaseType_ExamStaminaReduceCard',
            status_change_origin='card',
        )
        self.total_counters['stamina_spent'] += int(round(spent))
        self.turn_counters['stamina_spent'] += int(round(spent))
        self._dispatch_phase('ProduceExamPhaseType_ExamStaminaReduceCard', effect_types=['ProduceExamEffectType_ExamStaminaReduce'])
        self._dispatch_phase('ProduceExamPhaseType_ExamBuffConsume')
        for key, value in self._card_resource_costs(card).items():
            if key == 'parameter_buff_multiple_per_turn':
                self._consume_parameter_buff_multiple(value)
            else:
                self.resources[key] = max(self.resources[key] - value, 0.0)

        self._dispatch_phase('ProduceExamPhaseType_StartExamPlay', acting_card=card)
        self._dispatch_phase('ProduceExamPhaseType_StartPlay', acting_card=card)
        self._dispatch_phase('ProduceExamPhaseType_ExamCardPlay', acting_card=card)

        resolved_effect_specs = []
        for play_effect in self._resolved_card_play_effects(card):
            trigger_id = str(play_effect.get('trigger_id') or '')
            if trigger_id:
                trigger = self.repository.exam_trigger_map.get(trigger_id)
                event = {
                    'phase_type': 'ProduceExamPhaseType_ExamCardPlay',
                    'phase_value': self.turn,
                    'acting_card': card,
                    'effect_types': self.repository.card_exam_effect_types(card.base_card),
                }
                if not trigger or not self._trigger_matches(trigger, event, acting_card=card, target_card=card):
                    continue
            resolved_effect_specs.append(play_effect)

        card_repeat_total = 1 + self._card_repeat_bonus(card)
        lesson_repeat_bonus = int(round(self._current_card_grow_total('ProduceCardGrowEffectType_LessonCountAdd')))
        lesson_repeat_bonus -= int(round(self._current_card_grow_total('ProduceCardGrowEffectType_LessonCountReduce')))
        for _ in range(max(card_repeat_total, 1)):
            for play_effect in resolved_effect_specs:
                effect = self.repository.exam_effect_map.get(str(play_effect['effect_id']))
                if not effect:
                    continue
                effect_type = str(effect.get('effectType') or '')
                apply_times = 1
                if effect_type in LESSON_EFFECT_TYPES:
                    apply_times = max(1 + lesson_repeat_bonus, 1)
                for _ in range(apply_times):
                    self._apply_exam_effect(effect, source='card')
                    if self.terminated:
                        break
                if self.terminated:
                    break
            if self.terminated:
                break
        self._consume_card_play_buffs(card)

        self.turn_counters['play_count'] += 1
        self.total_counters['play_count'] += 1
        self.search_history[str(card.card_id)] += 1
        self._dispatch_interval_phase('ProduceExamPhaseType_ExamPlayTurnCountInterval', self.turn_counters['play_count'], acting_card=card)
        self._dispatch_interval_phase('ProduceExamPhaseType_ExamPlayCountInterval', self.total_counters['play_count'], acting_card=card)
        self._dispatch_phase('ProduceExamPhaseType_ExamSearchCardPlay', acting_card=card)
        self._dispatch_phase(
            'ProduceExamPhaseType_ExamCardPlayAfter',
            acting_card=card,
            effect_types=self.repository.card_exam_effect_types(card.base_card),
        )
        self._dispatch_interval_phase('ProduceExamPhaseType_ExamPlayCountIntervalAfter', self.total_counters['play_count'], acting_card=card)
        self._move_runtime_card(card, self._card_move_destination(card))
        self.playing = []
        self.current_card = None

    def _use_drink(self, index: int) -> None:
        """消耗一瓶饮料并应用其考试效果。"""

        drink = self.drinks[index]
        if not self._can_use_drink(drink):
            raise ValueError('Drink cannot be used without remaining play count')
        drink['_consumed'] = True
        self._record_event('drink_consumed', {
            'drink_index': index,
            'drink_name': self.repository.drink_name(drink),
        })
        for drink_effect_id in drink.get('produceDrinkEffectIds', []):
            drink_effect = self.repository.drink_effect_map.get(str(drink_effect_id))
            if not drink_effect:
                continue
            effect_id = str(drink_effect.get('produceExamEffectId') or '')
            effect = self.repository.exam_effect_map.get(effect_id)
            if effect:
                self._apply_exam_effect(effect, source='drink')
            if self.terminated:
                break
        self._refresh_play_limit()

    def _end_turn(self, skipped: bool = False) -> None:
        """结算回合末收益、弃手并进入下一回合。"""

        if skipped:
            self._dispatch_phase('ProduceExamPhaseType_ExamTurnSkip', phase_value=self.turn)
        review_activation_count = max(1 + int(round(self._timed_effect_stack_value('ProduceExamEffectType_ExamReviewCountAdd'))), 1)
        self.score += self._score_gain(self._apply_score_value_modifiers(self.resources['review'])) * review_activation_count
        self._update_clear_state_after_score_change()
        self._dispatch_phase('ProduceExamPhaseType_ExamEndTurn', phase_value=self.turn)
        self._dispatch_interval_phase('ProduceExamPhaseType_ExamEndTurnInterval', self.turn)
        if self.stance == 'full_power':
            self._release_full_power()
        self._discard_hand_to_grave()
        self.resources['enthusiastic'] = 0.0
        recovery = float(self.exam_setting.get('examTurnEndRecoveryStamina') or 0)
        if self.battle_kind != 'lesson' or skipped:
            self.stamina = min(self.max_stamina, self.stamina + recovery)
        self.panic_cost_overrides = {}
        if self.terminated:
            return
        if skipped and self.turn >= self.max_turns and self.extra_turns <= 0:
            self.terminated = True
            return
        self._start_turn()

    def _card_stamina_components(self, card: RuntimeCard) -> tuple[float, float]:
        """计算卡牌的实际体力消耗和强制体力消耗。"""

        base_cost = float(card.base_card.get('stamina') or 0)
        force_cost = float(card.base_card.get('forceStamina') or 0)
        fallback_cost_type = str(card.base_card.get('costType') or 'ExamCostType_Unknown')
        for grow_effect in self._card_grow_rows(card):
            effect_type = str(grow_effect.get('effectType') or '')
            delta = float(grow_effect.get('value') or 0)
            resource_key = self._grow_cost_resource_key(effect_type, fallback_cost_type)
            if resource_key == 'stamina':
                if effect_type.endswith('Add'):
                    base_cost += delta
                elif effect_type.endswith('Reduce'):
                    base_cost -= delta
            elif resource_key == 'penetrate':
                if effect_type.endswith('Add'):
                    force_cost += delta
                elif effect_type.endswith('Reduce'):
                    force_cost -= delta
        for timed in self._matching_search_stamina_overrides(card):
            base_cost = float(timed.effect.get('effectValue1') or 0)
            force_cost = 0.0
        for timed in self.active_effects:
            effect_type = str(timed.effect.get('effectType') or '')
            if effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionAdd':
                base_cost *= 2.0
                force_cost *= 2.0
            elif effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionDown':
                base_cost *= 0.5
                force_cost *= 0.5
            elif effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionAddFix':
                delta = self._raw_effect_value(timed.effect)
                base_cost += delta
                force_cost += delta
            elif effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionDownFix':
                delta = self._raw_effect_value(timed.effect)
                base_cost -= delta
                force_cost -= delta
        if self._has_timed_effect('ProduceExamEffectType_ExamPanic'):
            base_cost = self._panic_stamina_value(card)
            force_cost = 0.0
        base_cost = max(base_cost, 0.0)
        force_cost = max(force_cost, 0.0)
        stamina_multiple = 1.0
        if self.stance == 'concentration':
            key = f'examConcentrationStaminaMultiplePermil{max(self.stance_level, 1)}'
            stamina_multiple = float(self.exam_setting.get(key) or 1000) / 1000.0
            penetrate_key = f'examConcentrationStaminaPenetrateReduce{max(self.stance_level, 1)}'
            force_cost += float(self.exam_setting.get(penetrate_key) or 0)
        elif self.stance == 'preservation':
            if self.stance_level >= 3:
                stamina_multiple = float(self.exam_setting.get('examOverPreservationStaminaMultiplePermil') or 1000) / 1000.0
            else:
                key = f'examPreservationStaminaMultiplePermil{max(self.stance_level, 1)}'
                stamina_multiple = float(self.exam_setting.get(key) or 1000) / 1000.0
        base_cost *= stamina_multiple
        force_cost *= stamina_multiple
        return max(base_cost, 0.0), max(force_cost, 0.0)

    def _spend_stamina(
        self,
        amount: float,
        force_amount: float = 0.0,
        phase_type: str | None = None,
        status_change_origin: str = 'other',
    ) -> float:
        """先消耗护盾，再扣除实际体力。"""

        blocked = min(self.resources['block'], amount)
        if blocked > 0:
            self.resources['block'] -= blocked
            self.total_counters['block_consumed'] += int(round(blocked))
        direct_damage = max(amount - blocked, 0.0) + force_amount
        if direct_damage > 0:
            self.stamina = max(self.stamina - direct_damage, 0.0)
            self._dispatch_phase(
                phase_type or 'ProduceExamPhaseType_ExamStaminaReduce',
                effect_types=['ProduceExamEffectType_ExamStaminaReduce'],
            )
            self._dispatch_status_change(direct_damage, ['ProduceExamEffectType_ExamStaminaReduce'], origin=status_change_origin)
        return blocked + direct_damage

    def _card_resource_costs(self, card: RuntimeCard) -> dict[str, float]:
        """计算卡牌在当前成长效果下的资源消耗。"""

        costs: dict[str, float] = defaultdict(float)
        cost_type = str(card.base_card.get('costType') or 'ExamCostType_Unknown')
        if cost_type != 'ExamCostType_Unknown':
            resource_key = COST_RESOURCE_MAP.get(cost_type, cost_type)
            if resource_key not in {'stamina', 'penetrate', 'ExamCostType_Unknown'}:
                costs[resource_key] += float(card.base_card.get('costValue') or 0)
        for grow_effect in self._card_grow_rows(card):
            effect_type = str(grow_effect.get('effectType') or '')
            value = float(grow_effect.get('value') or 0)
            if not effect_type.startswith('ProduceCardGrowEffectType_Cost'):
                continue
            resource_key = self._grow_cost_resource_key(effect_type, cost_type)
            if resource_key in {'stamina', 'penetrate', 'ExamCostType_Unknown', ''}:
                continue
            if effect_type.endswith('Add'):
                costs[resource_key] += value
            elif effect_type.endswith('Reduce'):
                costs[resource_key] -= value
        return {key: max(value, 0.0) for key, value in costs.items()}

    def _decay_turn_effects(self) -> None:
        """在回合推进时衰减持续效果和附魔回合数。"""

        next_effects: list[TimedExamEffect] = []
        for timed in self.active_effects:
            remaining_turns = timed.remaining_turns
            if remaining_turns is not None:
                remaining_turns -= 1
            if remaining_turns is not None and remaining_turns <= 0:
                continue
            timed.remaining_turns = remaining_turns
            next_effects.append(timed)
        self.active_effects = next_effects

        next_enchants: list[TriggeredEnchant] = []
        for enchant in self.active_enchants:
            remaining_turns = enchant.remaining_turns
            if remaining_turns is not None:
                remaining_turns -= 1
            if remaining_turns is not None and remaining_turns <= 0:
                continue
            enchant.remaining_turns = remaining_turns
            next_enchants.append(enchant)
        self.active_enchants = next_enchants
        self._sync_effect_resources()

    def _dispatch_phase(
        self,
        phase_type: str,
        phase_value: int | None = None,
        acting_card: RuntimeCard | None = None,
        effect_types: list[str] | None = None,
        status_change_origin: str | None = None,
    ) -> None:
        """在单个考试 phase 下运行所有激活中的触发源。"""

        if self.turn > 0:
            self._apply_gimmicks_for_turn(self.turn)
            if self.terminated:
                return
        event = {
            'phase_type': phase_type,
            'phase_value': self.turn if phase_value is None and phase_type in PHASE_TURN_VALUE else phase_value,
            'acting_card': acting_card,
            'effect_types': effect_types or [],
            'status_change_origin': self._status_change_origin(status_change_origin),
        }

        fired_produce_items: set[tuple[str, str, int]] = set()
        for enchant in list(self.active_enchants):
            if enchant.uid in self._resolving_enchant_uids:
                continue
            trigger = self.repository.exam_trigger_map.get(enchant.trigger_id)
            timing_key = (
                str(enchant.source_identity or enchant.enchant_id),
                phase_type,
                int(event.get('phase_value') or 0),
            )
            if enchant.source == 'produce_item' and timing_key in fired_produce_items:
                continue
            if trigger and self._trigger_matches(trigger, event):
                self._resolving_enchant_uids.add(enchant.uid)
                self._record_event('enchant_triggered', {
                    'enchant_id': enchant.enchant_id,
                    'trigger_phase': phase_type,
                    'effect_ids': list(enchant.effect_ids),
                    'source': enchant.source,
                })
                try:
                    for effect_id in enchant.effect_ids:
                        effect = self.repository.exam_effect_map.get(str(effect_id))
                        if effect:
                            self._apply_exam_effect(effect, source=f'enchant:{enchant.enchant_id}')
                finally:
                    self._resolving_enchant_uids.discard(enchant.uid)
                if enchant.remaining_count is not None:
                    enchant.remaining_count -= 1
                if enchant.source == 'produce_item':
                    fired_produce_items.add(timing_key)
            if enchant.remaining_count is not None and enchant.remaining_count <= 0:
                self.active_enchants = [item for item in self.active_enchants if item.uid != enchant.uid]
            if self.terminated:
                return

        for card in list(self.hand) + list(self.hold) + list(self.deck) + list(self.grave):
            if not card.card_status_enchant_id:
                continue
            enchant_row = self.card_status_enchants.first(card.card_status_enchant_id)
            if not enchant_row:
                continue
            trigger = self.repository.exam_trigger_map.get(str(enchant_row.get('produceExamTriggerId') or ''))
            if trigger and self._trigger_matches(trigger, event, acting_card=acting_card, target_card=card):
                self._apply_card_status_enchant(card, enchant_row)
            if self.terminated:
                return

    def _dispatch_interval_phase(
        self,
        phase_type: str,
        counter_value: int,
        acting_card: RuntimeCard | None = None,
    ) -> None:
        """按间隔类条件逐次分发对应 phase。"""

        if counter_value <= 0:
            return
        for interval in self.repository.interval_phase_values.get(phase_type, ()):
            if counter_value % interval != 0:
                continue
            self._dispatch_phase(phase_type, phase_value=interval, acting_card=acting_card)
            if self.terminated:
                return

    def _trigger_matches(
        self,
        trigger: dict[str, Any],
        event: dict[str, Any],
        acting_card: RuntimeCard | None = None,
        target_card: RuntimeCard | None = None,
    ) -> bool:
        """判断某条触发器是否命中当前 phase 事件。"""

        phase_types = [str(value) for value in trigger.get('phaseTypes', []) if value]
        if phase_types and event['phase_type'] not in phase_types:
            return False

        if event['phase_type'] == 'ProduceExamPhaseType_ExamStatusChange':
            if str(event.get('status_change_origin') or '') not in STATUS_CHANGE_TRIGGER_ORIGINS:
                return False
        phase_values = [int(value) for value in trigger.get('phaseValues', []) if value not in (None, 0)]
        if phase_values:
            current_phase_value = int(event.get('phase_value') or 0)
            if event['phase_type'] == 'ProduceExamPhaseType_ExamStatusChange':
                if not any(current_phase_value >= value for value in phase_values):
                    return False
            elif current_phase_value not in phase_values:
                return False

        lesson_type = str(trigger.get('lessonType') or 'ProduceStepLessonType_Unknown')
        if lesson_type != 'ProduceStepLessonType_Unknown' and lesson_type not in set(self._current_lesson_types()):
            return False

        if trigger.get('effectTypes'):
            required = {str(value) for value in trigger.get('effectTypes', []) if value}
            if required and not required.intersection(set(event.get('effect_types') or [])):
                return False

        if not self._trigger_field_status_matches(trigger):
            return False
        if not self._trigger_card_search_matches(trigger, acting_card=acting_card, target_card=target_card):
            return False
        return True

    def _trigger_field_status_matches(self, trigger: dict[str, Any]) -> bool:
        """检查触发器里的场地状态条件是否成立。"""

        status_types = [str(value) for value in trigger.get('fieldStatusTypes', []) if value]
        status_values = [float(value) for value in trigger.get('fieldStatusValues', []) if value not in (None, '')]
        check_types = [str(value) for value in trigger.get('fieldStatusCheckTypes', []) if value]
        search_ids = [str(value) for value in trigger.get('fieldStatusProduceCardSearchIds', []) if value]

        for index, status_type in enumerate(status_types):
            current_value = self._field_status_value(status_type, search_ids[index] if index < len(search_ids) else '')
            expected_value = status_values[index] if index < len(status_values) else 1.0
            check_type = check_types[index] if index < len(check_types) else 'ProduceExamTriggerCheckType_Unknown'
            reverse_threshold = status_type.endswith('MultipleDown') or 'LessMultiple' in status_type
            if check_type == 'ProduceExamTriggerCheckType_Not':
                if reverse_threshold:
                    if current_value <= expected_value:
                        return False
                elif current_value >= expected_value:
                    return False
            else:
                if reverse_threshold:
                    if current_value > expected_value:
                        return False
                elif current_value < expected_value:
                    return False
        return True

    def _field_status_value(self, field_status_type: str, search_id: str) -> float:
        """读取指定场地状态类型在当前战斗中的数值。"""

        stamina_ratio = float(self.stamina / max(self.max_stamina, 1.0)) * 1000.0
        lesson_progress = float(self.score / max(self.profile.get('base_score') or 1.0, 1.0)) * 1000.0
        mapping = {
            'ProduceExamFieldStatusType_BlockUp': self.resources['block'],
            'ProduceExamFieldStatusType_CardPlayAggressiveUp': self.resources['aggressive'],
            'ProduceExamFieldStatusType_CardSearchCountUp': float(self._search_cards(search_id).pool_size if search_id else 0),
            'ProduceExamFieldStatusType_ConditionThresholdMultipleDown': lesson_progress,
            'ProduceExamFieldStatusType_ConcentrationChangeCountUp': float(self.total_counters['stance_concentration']),
            'ProduceExamFieldStatusType_ConcentrationUp': self.resources['concentration'],
            'ProduceExamFieldStatusType_FullPowerChangeCountUp': float(self.total_counters['stance_full_power']),
            'ProduceExamFieldStatusType_FullPowerPointGetSumUp': float(self.total_counters['full_power_point_gained']),
            'ProduceExamFieldStatusType_FullPowerPointUp': self.resources['full_power_point'],
            'ProduceExamFieldStatusType_FullPowerUp': 1.0 if self.stance == 'full_power' else 0.0,
            'ProduceExamFieldStatusType_LessonBuffUp': self.resources['lesson_buff'],
            'ProduceExamFieldStatusType_NoBlock': 1.0 if self.resources['block'] <= 0 else 0.0,
            'ProduceExamFieldStatusType_NoStance': 1.0 if self.stance == 'neutral' else 0.0,
            'ProduceExamFieldStatusType_ParameterBuff': self.resources['parameter_buff'],
            'ProduceExamFieldStatusType_ParameterBuffMultiplePerTurnUp': self.resources['parameter_buff_multiple_per_turn'],
            'ProduceExamFieldStatusType_ParameterBuffUp': self.resources['parameter_buff'],
            'ProduceExamFieldStatusType_PlayCardLesson': float(self.turn_counters['lesson_plays']),
            'ProduceExamFieldStatusType_PlayCardSkill': float(self.turn_counters['play_count']),
            'ProduceExamFieldStatusType_PreservationChangeCountUp': float(self.total_counters['stance_preservation']),
            'ProduceExamFieldStatusType_PreservationUp': self.resources['preservation'],
            'ProduceExamFieldStatusType_RemainingTurn': float(max(self.max_turns - self.turn + 1, 0)),
            'ProduceExamFieldStatusType_ReviewUp': self.resources['review'],
            'ProduceExamFieldStatusType_StaminaConsumptionDown': self.resources['stamina_consumption_down'],
            'ProduceExamFieldStatusType_StaminaLessMultiple': stamina_ratio,
            'ProduceExamFieldStatusType_StaminaUpMultiple': stamina_ratio,
            'ProduceExamFieldStatusType_StanceChangeCountUp': float(self.total_counters['stance_changes']),
            'ProduceExamFieldStatusType_TurnProgressUp': float(self.turn),
        }
        return float(mapping.get(field_status_type, 0.0))

    def _trigger_card_search_matches(
        self,
        trigger: dict[str, Any],
        acting_card: RuntimeCard | None,
        target_card: RuntimeCard | None,
    ) -> bool:
        """检查触发器里的卡牌搜索条件是否成立。"""

        search_id = str(trigger.get('produceCardSearchId') or '')
        if not search_id:
            return True
        selection = self._search_cards(search_id, acting_card=acting_card, target_card=target_card)
        lower = int(trigger.get('lowerSearchCount') or 0)
        upper = int(trigger.get('upperSearchCount') or 0)
        if lower and selection.pool_size < lower:
            return False
        if upper and selection.pool_size > upper:
            return False
        return selection.pool_size > 0 or (lower == 0 and upper == 0)

    def _apply_card_status_enchant(self, card: RuntimeCard, enchant_row: dict[str, Any]) -> None:
        """把卡牌状态附魔展开为临时附魔变更或成长效果。"""

        for grow_effect_id in enchant_row.get('produceCardGrowEffectIds', []):
            grow_effect = self.grow_effects.first(str(grow_effect_id))
            if not grow_effect:
                continue
            effect_type = str(grow_effect.get('effectType') or '')
            if effect_type == 'ProduceCardGrowEffectType_CardStatusEnchantChange':
                card.card_status_enchant_id = str(grow_effect.get('produceCardStatusEnchantId') or '')
                continue
            card.grow_effect_ids.append(str(grow_effect_id))

    def _search_cards(
        self,
        search_id: str,
        acting_card: RuntimeCard | None = None,
        target_card: RuntimeCard | None = None,
    ) -> CardSelection:
        """按 ProduceCardSearch 定义检索运行时卡牌。"""

        search = self.card_searches.first(search_id)
        if not search:
            return CardSelection(selected=[], pool_size=0)
        pool = self._pool_for_search(search, acting_card=acting_card, target_card=target_card)
        pool = [card for card in pool if self._matches_card_search(card, search)]
        pool_size = len(pool)
        limit_count = int(search.get('limitCount') or 0)
        if limit_count <= 0:
            limit_count = pool_size
        order_type = str(search.get('orderType') or 'ProduceCardOrderType_Unknown')
        if order_type == 'ProduceCardOrderType_Random' and pool:
            count = min(pool_size, max(limit_count, 1))
            indices = self.np_random.choice(pool_size, size=count, replace=False)
            selected = [pool[int(index)] for index in np.atleast_1d(indices)]
        else:
            selected = pool[:limit_count]
        return CardSelection(selected=selected, pool_size=pool_size)

    def _pool_for_search(
        self,
        search: dict[str, Any],
        acting_card: RuntimeCard | None,
        target_card: RuntimeCard | None,
    ) -> list[RuntimeCard]:
        """根据搜索区域决定检索池来源。"""

        zone = CARD_ZONE_MAP.get(str(search.get('cardPositionType') or 'ProduceCardPositionType_DeckAll'), 'deck')
        if search.get('isSelf') and acting_card is not None:
            return [acting_card]
        if zone == 'deck':
            return list(self.deck)
        if zone == 'deck_grave':
            return list(self.deck) + list(self.grave)
        if zone == 'hand':
            return list(self.hand)
        if zone == 'hold':
            return list(self.hold)
        if zone == 'lost':
            return list(self.lost)
        if zone == 'not_lost':
            return list(self.deck) + list(self.hand) + list(self.grave) + list(self.hold)
        if zone == 'playing':
            return list(self.playing)
        if zone == 'target':
            return [target_card] if target_card is not None else ([acting_card] if acting_card is not None else [])
        return list(self.deck)

    def _matches_card_search(self, card: RuntimeCard, search: dict[str, Any]) -> bool:
        """判断单张运行时卡是否命中搜索条件。"""

        base_card = card.base_card
        rarities = {str(value) for value in search.get('cardRarities', []) if value}
        if rarities and str(base_card.get('rarity')) not in rarities:
            return False
        produce_card_ids = {str(value) for value in search.get('produceCardIds', []) if value}
        if produce_card_ids and card.card_id not in produce_card_ids:
            return False
        upgrade_counts = {int(value) for value in search.get('upgradeCounts', []) if value is not None}
        if upgrade_counts and card.upgrade_count not in upgrade_counts:
            return False
        categories = {str(value) for value in search.get('cardCategories', []) if value}
        if categories and str(base_card.get('category')) not in categories:
            return False
        plan_type = str(search.get('planType') or 'ProducePlanType_Unknown')
        if plan_type != 'ProducePlanType_Unknown' and str(base_card.get('planType')) != plan_type:
            return False
        search_tag = str(search.get('cardSearchTag') or '')
        if search_tag and search_tag != str(base_card.get('searchTag') or ''):
            return False
        cost_type = str(search.get('costType') or 'ExamCostType_Unknown')
        if cost_type != 'ExamCostType_Unknown' and str(base_card.get('costType')) != cost_type:
            return False
        exam_effect_type = str(search.get('examEffectType') or 'ProduceExamEffectType_Unknown')
        if exam_effect_type != 'ProduceExamEffectType_Unknown' and exam_effect_type not in set(self.repository.card_exam_effect_types(base_card)):
            return False
        effect_group_ids = {str(value) for value in search.get('effectGroupIds', []) if value}
        if effect_group_ids and not effect_group_ids.intersection({str(value) for value in base_card.get('effectGroupIds', []) if value}):
            return False
        if search.get('isCustomized') and not card.grow_effect_ids:
            return False
        return True

    def _apply_exam_effect(self, effect: dict[str, Any], source: str) -> None:
        """应用一条考试效果。

        这里按效果类型家族分支处理，卡牌、附魔、计时器和数值修饰都直接来自主数据字段，
        不依赖卡名硬编码。
        """

        effect_type = str(effect.get('effectType') or '')
        timed_effect_types = {
            'ProduceExamEffectType_ExamAggressiveAdditive',
            'ProduceExamEffectType_ExamAggressiveValueMultiple',
            'ProduceExamEffectType_ExamBlockAddDown',
            'ProduceExamEffectType_ExamBlockRestriction',
            'ProduceExamEffectType_ExamBlockValueMultiple',
            'ProduceExamEffectType_ExamCardSearchEffectPlayCountBuff',
            'ProduceExamEffectType_ExamEnthusiasticAdditive',
            'ProduceExamEffectType_ExamEnthusiasticMultiple',
            'ProduceExamEffectType_ExamFullPowerPointAdditive',
            'ProduceExamEffectType_ExamLessonBuffAdditive',
            'ProduceExamEffectType_ExamLessonBuffMultiple',
            'ProduceExamEffectType_ExamLessonValueMultiple',
            'ProduceExamEffectType_ExamLessonValueMultipleDependReviewOrAggressive',
            'ProduceExamEffectType_ExamLessonValueMultipleDown',
            'ProduceExamEffectType_ExamParameterBuffAdditive',
            'ProduceExamEffectType_ExamParameterBuffMultiplePerTurn',
            'ProduceExamEffectType_ExamPlayableValueAdd',
            'ProduceExamEffectType_ExamReviewCountAdd',
            'ProduceExamEffectType_ExamReviewMultiple',
            'ProduceExamEffectType_ExamSearchPlayCardStaminaConsumptionChange',
            'ProduceExamEffectType_ExamStaminaConsumptionAdd',
            'ProduceExamEffectType_ExamStaminaConsumptionAddFix',
            'ProduceExamEffectType_ExamStaminaConsumptionDown',
            'ProduceExamEffectType_ExamStaminaConsumptionDownFix',
            'ProduceExamEffectType_ExamStaminaRecoverRestriction',
        }
        if effect_type in timed_effect_types:
            if self._consume_anti_debuff(effect_type):
                return
            register_times = 1
            effect_row = dict(effect)
            if source == 'card' and effect_type == 'ProduceExamEffectType_ExamParameterBuffMultiplePerTurn':
                register_times += int(round(self._current_card_grow_total('ProduceCardGrowEffectType_ParameterBuffMultiplePerTurnAdd')))
            if source == 'card' and effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionDown':
                effect_row['effectTurn'] = int(effect.get('effectTurn') or 0) + int(round(self._current_card_grow_total('ProduceCardGrowEffectType_StaminaConsumptionDownTurnAdd')))
            for _ in range(max(register_times, 1)):
                self._register_timed_effect(dict(effect_row), source)
            return
        if effect_type == 'ProduceExamEffectType_ExamStatusEnchant':
            self._apply_status_enchant(effect, source)
            return
        if effect_type == 'ProduceExamEffectType_ExamEffectTimer':
            self._schedule_effect(effect)
            return
        if effect_type == 'ProduceExamEffectType_ExamAddGrowEffect':
            self._add_grow_effect(effect)
            return
        if effect_type in {
            'ProduceExamEffectType_ExamCardCreateId',
            'ProduceExamEffectType_ExamCardCreateSearch',
            'ProduceExamEffectType_ExamCardDuplicate',
            'ProduceExamEffectType_ExamCardMove',
            'ProduceExamEffectType_ExamCardUpgrade',
            'ProduceExamEffectType_ExamForcePlayCardSearch',
        }:
            self._apply_card_operation(effect)
            return
        if effect_type in DURATION_RESOURCE_TYPES:
            resource_key = DURATION_RESOURCE_TYPES[effect_type]
            delta = self._parameter_buff_gain_value(effect)
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ParameterBuffTurnAdd')
            self.resources[resource_key] += delta
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type in SCALAR_RESOURCE_TYPES:
            resource_key = SCALAR_RESOURCE_TYPES[effect_type]
            delta = self._direct_value(effect)
            if source == 'card' and effect_type == 'ProduceExamEffectType_ExamCardPlayAggressive':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_AggressiveAdd')
            elif source == 'card' and effect_type == 'ProduceExamEffectType_ExamReview':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ReviewAdd')
            elif source == 'card' and effect_type == 'ProduceExamEffectType_ExamLessonBuff':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_LessonBuffAdd')
            elif source == 'card' and effect_type == 'ProduceExamEffectType_ExamFullPowerPoint':
                delta = self._adjust_direct_gain(
                    delta,
                    add_grow_type='ProduceCardGrowEffectType_FullPowerPointAdd',
                    reduce_grow_type='ProduceCardGrowEffectType_FullPowerPointReduce',
                )
            self.resources[resource_key] += delta
            if resource_key == 'full_power_point':
                self.total_counters['full_power_point_gained'] += self._positive_count(delta)
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamConcentration':
            self._enter_concentration(int(max(self._raw_effect_value(effect), 1.0)))
            return
        if effect_type == 'ProduceExamEffectType_ExamPreservation':
            self._enter_preservation(int(max(self._raw_effect_value(effect), 1.0)))
            return
        if effect_type == 'ProduceExamEffectType_ExamOverPreservation':
            self._enter_preservation(3)
            return
        if effect_type == 'ProduceExamEffectType_ExamCardDraw':
            self._draw(int(max(1.0, round(self._direct_value(effect)))))
            return
        if effect_type == 'ProduceExamEffectType_ExamFullPower':
            self._enter_full_power()
            return
        if effect_type == 'ProduceExamEffectType_ExamStanceReset':
            self._reset_stance()
            return
        if effect_type == 'ProduceExamEffectType_StanceLock':
            self.stance_locked = True
            return
        if effect_type == 'ProduceExamEffectType_ExamExtraTurn':
            self.extra_turns += 1
            return
        if effect_type == 'ProduceExamEffectType_ExamAntiDebuff':
            self.resources['anti_debuff'] += self._count_value(effect)
            return
        if effect_type == 'ProduceExamEffectType_ExamDebuffRecover':
            self._clear_negative_effects()
            return
        if effect_type in {
            'ProduceExamEffectType_ExamStaminaDamage',
            'ProduceExamEffectType_ExamStaminaReduce',
            'ProduceExamEffectType_ExamStaminaReduceFix',
        }:
            self._spend_stamina(
                self._direct_value(effect),
                phase_type='ProduceExamPhaseType_ExamStaminaReduce',
                status_change_origin=source,
            )
            return
        if effect_type in {'ProduceExamEffectType_ExamStaminaRecover', 'ProduceExamEffectType_ExamStaminaRecoverFix'}:
            if self._has_timed_effect('ProduceExamEffectType_ExamStaminaRecoverRestriction'):
                return
            self.stamina = min(self.max_stamina, self.stamina + self._direct_value(effect))
            return
        if effect_type == 'ProduceExamEffectType_ExamStaminaRecoverMultiple':
            if self._has_timed_effect('ProduceExamEffectType_ExamStaminaRecoverRestriction'):
                return
            self.stamina = min(self.max_stamina, self.stamina + self.max_stamina * self._ratio_value(effect))
            return
        if effect_type in {'ProduceExamEffectType_ExamBlock', 'ProduceExamEffectType_ExamBlockFix'}:
            delta = self._raw_effect_value(effect)
            if source == 'card':
                delta = self._adjust_direct_gain(
                    delta,
                    add_grow_type='ProduceCardGrowEffectType_BlockAdd',
                    reduce_grow_type='ProduceCardGrowEffectType_BlockReduce',
                )
            self._gain_block(delta, effect_type=effect_type, status_change_origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamBlockDependBlockConsumptionSum':
            delta = self._ceil_positive(self.total_counters['block_consumed'] * self._ratio_value(effect))
            if source == 'card':
                delta = self._adjust_direct_gain(
                    delta,
                    add_grow_type='ProduceCardGrowEffectType_BlockAdd',
                    reduce_grow_type='ProduceCardGrowEffectType_BlockReduce',
                )
            self._gain_block(delta, effect_type=effect_type, status_change_origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamBlockAddMultipleAggressive':
            base_delta = self._compose_referenced_gain(
                base=max(self._raw_effect_value(effect), 0.0),
                referenced=self.resources['aggressive'] * (float(effect.get('effectValue2') or 0) / 1000.0),
            )
            if source == 'card':
                base_delta = self._adjust_direct_gain(
                    base_delta,
                    add_grow_type='ProduceCardGrowEffectType_BlockAdd',
                    reduce_grow_type='ProduceCardGrowEffectType_BlockReduce',
                )
            self._gain_block(base_delta, effect_type=effect_type, status_change_origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamBlockDown':
            self.resources['block'] *= max(1.0 - self._ratio_value(effect), 0.0)
            return
        if effect_type == 'ProduceExamEffectType_ExamBlockPerUseCardCount':
            delta = self.turn_counters['play_count'] * self._raw_effect_value(effect)
            if source == 'card':
                delta = self._adjust_direct_gain(
                    delta,
                    add_grow_type='ProduceCardGrowEffectType_BlockAdd',
                    reduce_grow_type='ProduceCardGrowEffectType_BlockReduce',
                )
            self._gain_block(delta, effect_type=effect_type, status_change_origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamBlockDependExamReview':
            delta = self._ceil_positive(self.resources['review'] * self._ratio_value(effect))
            if source == 'card':
                delta = self._adjust_direct_gain(
                    delta,
                    add_grow_type='ProduceCardGrowEffectType_BlockAdd',
                    reduce_grow_type='ProduceCardGrowEffectType_BlockReduce',
                )
            self._gain_block(delta, effect_type=effect_type, status_change_origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamAggressiveReduce':
            self.resources['aggressive'] = max(self.resources['aggressive'] - self._direct_value(effect), 0.0)
            return
        if effect_type in {'ProduceExamEffectType_ExamReview', 'ProduceExamEffectType_ExamReviewAdditive'}:
            delta = self._direct_value(effect)
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ReviewAdd')
            self.resources['review'] += delta
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamReviewDependExamCardPlayAggressive':
            delta = self._ceil_positive(self.resources['aggressive'] * self._ratio_value(effect))
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ReviewAdd')
            self.resources['review'] += delta
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamReviewDependExamBlock':
            delta = self._ceil_positive(self.resources['block'] * self._ratio_value(effect))
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ReviewAdd')
            self.resources['review'] += delta
            if float(effect.get('effectValue2') or 0) >= 1000:
                self.resources['block'] = 0.0
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamReviewPerSearchCount':
            search_count = self._search_cards(str(effect.get('produceCardSearchId') or '')).pool_size
            delta = self._ceil_positive(search_count * (float(effect.get('effectValue2') or effect.get('effectValue1') or 0) / 1000.0))
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ReviewAdd')
            self.resources['review'] += delta
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamReviewReduce':
            self.resources['review'] = max(self.resources['review'] - self._direct_value(effect), 0.0)
            return
        if effect_type == 'ProduceExamEffectType_ExamReviewValueMultiple':
            self.resources['review'] *= 1.0 + self._ratio_value(effect)
            return
        if effect_type == 'ProduceExamEffectType_ExamParameterBuffReduce':
            self.resources['parameter_buff'] = max(self.resources['parameter_buff'] - self._direct_value(effect), 0.0)
            return
        if effect_type == 'ProduceExamEffectType_ExamParameterBuffMultiplePerTurnReduce':
            self._consume_parameter_buff_multiple(self._direct_value(effect))
            return
        if effect_type == 'ProduceExamEffectType_ExamParameterBuffDependLessonBuff':
            delta = self._ceil_positive(self.resources['lesson_buff'] * self._ratio_value(effect))
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ParameterBuffTurnAdd')
            self.resources['parameter_buff'] += delta
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamLessonBuff':
            delta = self._direct_value(effect)
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_LessonBuffAdd')
            self.resources['lesson_buff'] += delta
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamLessonBuffReduce':
            self.resources['lesson_buff'] = max(self.resources['lesson_buff'] - self._direct_value(effect), 0.0)
            return
        if effect_type == 'ProduceExamEffectType_ExamLessonBuffDependParameterBuff':
            delta = self._ceil_positive(self.resources['parameter_buff'] * self._ratio_value(effect))
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_LessonBuffAdd')
            self.resources['lesson_buff'] += delta
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type == 'ProduceExamEffectType_ExamLessonBuffPerSearchCount':
            search_count = self._search_cards(str(effect.get('produceCardSearchId') or '')).pool_size
            delta = self._ceil_positive(search_count * (float(effect.get('effectValue2') or effect.get('effectValue1') or 0) / 1000.0))
            if source == 'card':
                delta = self._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_LessonBuffAdd')
            self.resources['lesson_buff'] += delta
            self._dispatch_status_change(delta, [effect_type], origin=source)
            return
        if effect_type.startswith('ProduceExamEffectType_ExamLesson') or effect_type == 'ProduceExamEffectType_ExamMultipleLessonBuffLesson':
            lesson_value = self._resolve_lesson_effect_value(effect, from_card=source == 'card')
            self.score += self._score_gain(lesson_value)
            self.turn_counters['lesson_plays'] += 1
            self.total_counters['lesson_plays'] += 1
            self._update_clear_state_after_score_change()
            return
        if effect_type == 'ProduceExamEffectType_ExamItemFireLimitAdd':
            delta = self._direct_value(effect)
            self.resources['item_fire_limit_add'] += delta
            for enchant in self.active_enchants:
                if enchant.source != 'produce_item' or enchant.remaining_count is None:
                    continue
                enchant.remaining_count += int(round(delta))
            return
        if effect_type == 'ProduceExamEffectType_ExamGimmickPlayCardLimit':
            if self._consume_anti_debuff(effect_type):
                return
            search_id = str(effect.get('produceCardSearchId') or '')
            if not search_id:
                return
            self.forbidden_card_search_ids[search_id] += max(self._positive_count(self._count_value(effect)), 1)
            self._sync_forbidden_search_resources()
            return
        if effect_type == 'ProduceExamEffectType_ExamGimmickStartTurnCardDrawDown':
            if self._consume_anti_debuff(effect_type):
                return
            self.start_turn_draw_penalty += int(round(self._raw_effect_value(effect)))
            return
        if effect_type == 'ProduceExamEffectType_ExamGimmickSleepy':
            if self._consume_anti_debuff(effect_type):
                return
            self.resources['sleepy'] += self._raw_effect_value(effect)
            return
        if effect_type in {
            'ProduceExamEffectType_ExamGimmickSlump',
            'ProduceExamEffectType_ExamGimmickParameterDebuff',
            'ProduceExamEffectType_ExamGimmickLessonDebuff',
            'ProduceExamEffectType_ExamPanic',
        }:
            if self._consume_anti_debuff(effect_type):
                return
            self._register_timed_effect(effect, source)
            return
        if effect_type == 'ProduceExamEffectType_ExamFullPowerPointReduce':
            self.resources['full_power_point'] = max(self.resources['full_power_point'] - self._direct_value(effect), 0.0)
            return
        if effect_type == 'ProduceExamEffectType_ExamHandGraveCountCardDraw':
            self._draw(len(self.hand) + len(self.grave))
            return
        self._register_timed_effect(effect, source)

    def _register_initial_enchant(
        self,
        enchant_id: str,
        source: str,
        remaining_turns: int | None = None,
        remaining_count: int | None = None,
        source_identity: str = '',
    ) -> None:
        """注册开场自带的状态附魔，并保留来源提供的次数与回合数。"""

        enchant_row = self.repository.exam_status_enchant_map.get(str(enchant_id))
        if not enchant_row:
            return
        self.active_enchants.append(
            TriggeredEnchant(
                uid=self._next_uid(),
                enchant_id=str(enchant_row.get('id')),
                trigger_id=str(enchant_row.get('produceExamTriggerId') or ''),
                effect_ids=[str(value) for value in enchant_row.get('produceExamEffectIds', []) if value],
                remaining_turns=remaining_turns,
                remaining_count=remaining_count,
                source=source,
                source_identity=source_identity,
            )
        )

    def _register_timed_effect(self, effect: dict[str, Any], source: str) -> None:
        """把持续型考试效果挂入运行时效果列表。"""

        remaining_turns = int(effect.get('effectTurn') or 0)
        if remaining_turns < 0:
            remaining_turns = None
        elif remaining_turns == 0:
            remaining_turns = 1
        remaining_count = int(effect.get('effectCount') or 0)
        if remaining_count <= 0:
            remaining_count = None
        self.active_effects.append(
            TimedExamEffect(
                uid=self._next_uid(),
                effect=effect,
                remaining_turns=remaining_turns,
                remaining_count=remaining_count,
                source=source,
            )
        )
        self._sync_effect_resources()
        if str(effect.get('effectType') or '') == 'ProduceExamEffectType_ExamPlayableValueAdd':
            self._refresh_play_limit()

    def _apply_status_enchant(self, effect: dict[str, Any], source: str) -> None:
        """把考试效果转换成可触发的状态附魔实例。"""

        enchant_row = self.repository.exam_status_enchant_map.get(str(effect.get('produceExamStatusEnchantId') or ''))
        if not enchant_row:
            return
        remaining_turns = int(effect.get('effectTurn') or 0)
        if remaining_turns < 0:
            remaining_turns = None
        elif remaining_turns == 0:
            remaining_turns = 1
        remaining_count = int(effect.get('effectCount') or 0)
        if remaining_count <= 0:
            remaining_count = None
        self.active_enchants.append(
            TriggeredEnchant(
                uid=self._next_uid(),
                enchant_id=str(enchant_row.get('id')),
                trigger_id=str(enchant_row.get('produceExamTriggerId') or ''),
                effect_ids=[str(value) for value in enchant_row.get('produceExamEffectIds', []) if value],
                remaining_turns=remaining_turns,
                remaining_count=remaining_count,
                source=source,
                source_identity=source,
            )
        )

    def _schedule_effect(self, effect: dict[str, Any]) -> None:
        """登记延迟触发的链式考试效果。"""

        chain_id = str(effect.get('chainProduceExamEffectId') or '')
        if not chain_id:
            return
        delay = max(int(effect.get('effectValue1') or 1), 1)
        self.scheduled_effects.append(
            ScheduledEffect(
                effect_id=chain_id,
                fire_turn=self.turn + delay,
                remaining_count=int(effect.get('effectCount') or 0) or None,
            )
        )
        self._record_event('effect_scheduled', {
            'effect_id': chain_id,
            'fire_turn': self.turn + delay,
        })

    def _add_grow_effect(self, effect: dict[str, Any]) -> None:
        """为检索到的运行时卡牌追加成长效果。"""

        selection = self._search_cards(str(effect.get('produceCardSearchId') or ''))
        grow_ids = [str(value) for value in effect.get('produceCardGrowEffectIds', []) if value]
        for card in selection.selected:
            card.grow_effect_ids.extend(grow_ids)
            initial_adds = 0
            for grow_id in grow_ids:
                grow_row = self.grow_effects.first(grow_id)
                if grow_row and str(grow_row.get('effectType') or '') == 'ProduceCardGrowEffectType_InitialAdd':
                    initial_adds += 1
            for _ in range(initial_adds):
                self._insert_card(card.clone(uid=self._next_uid()), 'deck_last')

    def _apply_card_operation(self, effect: dict[str, Any]) -> None:
        """执行造卡、复制、移动、升级、强制打出等卡牌操作。"""

        effect_type = str(effect.get('effectType') or '')
        if effect_type == 'ProduceExamEffectType_ExamCardCreateId':
            self._create_card_by_id(effect)
        elif effect_type == 'ProduceExamEffectType_ExamCardCreateSearch':
            self._create_card_by_search(effect)
        elif effect_type == 'ProduceExamEffectType_ExamCardDuplicate':
            selection = self._search_cards(str(effect.get('produceCardSearchId') or ''))
            destination = MOVE_POSITION_MAP.get(str(effect.get('movePositionType') or ''), 'deck_last')
            for card in selection.selected:
                self._insert_card(card.clone(uid=self._next_uid()), destination)
        elif effect_type == 'ProduceExamEffectType_ExamCardMove':
            selection = self._search_cards(str(effect.get('produceCardSearchId') or ''))
            destination = MOVE_POSITION_MAP.get(str(effect.get('movePositionType') or ''), 'grave')
            for card in selection.selected:
                self._move_runtime_card(card, destination)
        elif effect_type == 'ProduceExamEffectType_ExamCardUpgrade':
            selection = self._search_cards(str(effect.get('produceCardSearchId') or ''))
            for card in selection.selected:
                upgraded_row = self._lookup_card_row(card.card_id, card.upgrade_count + 1)
                if upgraded_row is not None:
                    card.base_card = upgraded_row
                    card.upgrade_count = int(upgraded_row.get('upgradeCount') or card.upgrade_count + 1)
        elif effect_type == 'ProduceExamEffectType_ExamForcePlayCardSearch':
            selection = self._search_cards(str(effect.get('produceCardSearchId') or ''))
            if selection.selected:
                forced_card = selection.selected[0]
                self._detach_card(forced_card)
                self._play_card(forced_card)

    def _create_card_by_id(self, effect: dict[str, Any]) -> None:
        """按显式卡牌 id 创建运行时卡并放入目标区域。"""

        target_id = str(effect.get('targetProduceCardId') or '')
        if not target_id:
            return
        card_row = self._lookup_card_row(target_id, int(effect.get('targetUpgradeCount') or 0))
        if card_row is None:
            return
        runtime_card = RuntimeCard(
            uid=self._next_uid(),
            card_id=target_id,
            upgrade_count=int(card_row.get('upgradeCount') or 0),
            base_card=card_row,
        )
        destination = MOVE_POSITION_MAP.get(str(effect.get('movePositionType') or ''), 'deck_last')
        self._record_event('card_acquired', {
            'card_id': target_id,
            'card_name': self.repository.card_name(card_row),
            'upgrade_count': runtime_card.upgrade_count,
            'destination': destination,
            'source': 'create_by_id',
        })
        self._insert_card(runtime_card, destination)

    def _create_card_by_search(self, effect: dict[str, Any]) -> None:
        """按检索配置随机生成卡牌并插入目标区域。"""

        search_row = self.card_searches.first(str(effect.get('produceCardSearchId') or ''))
        if not search_row:
            return
        destination = MOVE_POSITION_MAP.get(str(effect.get('movePositionType') or ''), 'deck_last')
        for card_row in self._sample_card_specs_from_search(search_row, effect):
            runtime_card = RuntimeCard(
                uid=self._next_uid(),
                card_id=str(card_row.get('id')),
                upgrade_count=int(card_row.get('upgradeCount') or 0),
                base_card=card_row,
            )
            self._record_event('card_acquired', {
                'card_id': runtime_card.card_id,
                'card_name': self.repository.card_name(card_row),
                'upgrade_count': runtime_card.upgrade_count,
                'destination': destination,
                'source': 'create_by_search',
            })
            self._insert_card(runtime_card, destination)

    def _fire_scheduled_effects(self) -> None:
        """结算当前回合到期的延迟效果。"""

        remaining: list[ScheduledEffect] = []
        for index, scheduled in enumerate(self.scheduled_effects):
            if scheduled.fire_turn > self.turn:
                remaining.append(scheduled)
                continue
            effect = self.repository.exam_effect_map.get(scheduled.effect_id)
            if effect:
                self._apply_exam_effect(effect, source='timer')
            if scheduled.remaining_count is not None:
                scheduled.remaining_count -= 1
            if scheduled.remaining_count is None or scheduled.remaining_count > 0:
                remaining.append(scheduled)
            if self.terminated:
                remaining.extend(self.scheduled_effects[index + 1:])
                break
        self.scheduled_effects = remaining

    def _apply_gimmicks_for_turn(self, turn: int) -> None:
        """在指定回合应用满足条件的考场机制效果。"""

        for row in self.gimmick_rows:
            if int(row.get('startTurn') or 0) != turn:
                continue
            gimmick_key = (
                str(row.get('id') or ''),
                int(row.get('priority') or 0),
                int(row.get('startTurn') or 0),
            )
            if gimmick_key in self._resolved_gimmick_keys:
                continue
            self._resolved_gimmick_keys.add(gimmick_key)
            if not self._gimmick_condition_matches(row):
                continue
            effect = self.repository.exam_effect_map.get(str(row.get('produceExamEffectId') or ''))
            if effect:
                self._record_event('gimmick_fired', {
                    'gimmick_id': str(row.get('id') or ''),
                    'start_turn': int(row.get('startTurn') or 0),
                    'effect_id': str(row.get('produceExamEffectId') or ''),
                })
                self._apply_exam_effect(effect, source='gimmick')
                if self.terminated:
                    return

    def _gimmick_condition_matches(self, row: dict[str, Any]) -> bool:
        """判断考场机制的场地条件是否满足。"""

        field_status_type = str(row.get('fieldStatusType') or 'ProduceExamFieldStatusType_Unknown')
        if field_status_type == 'ProduceExamFieldStatusType_Unknown':
            return True
        current_value = self._field_status_value(field_status_type, str(row.get('fieldStatusProduceCardSearchId') or ''))
        expected_value = float(row.get('fieldStatusValue') or 1)
        check_type = str(row.get('fieldStatusCheckType') or 'ProduceExamTriggerCheckType_Unknown')
        reverse_threshold = field_status_type.endswith('MultipleDown') or 'LessMultiple' in field_status_type
        if check_type == 'ProduceExamTriggerCheckType_Not':
            return current_value > expected_value if reverse_threshold else current_value < expected_value
        return current_value <= expected_value if reverse_threshold else current_value >= expected_value

    def _clear_negative_effects(self) -> None:
        """清理可被移除的负面状态资源。"""

        self.resources['sleepy'] = 0.0
        self.forbidden_card_search_ids = Counter()
        self.resources['active_skill_forbidden'] = 0.0
        self.resources['mental_skill_forbidden'] = 0.0
        self.active_effects = [
            item
            for item in self.active_effects
            if str(item.effect.get('effectType') or '') not in NEGATIVE_TIMED_EFFECT_TYPES
        ]
        self.panic_cost_overrides = {}
        self._sync_effect_resources()

    def _card_move_destination(self, card: RuntimeCard) -> str:
        """解析卡牌打出后的最终落点，包含成长效果修正。"""

        move_type = str(card.base_card.get('playMovePositionType') or 'ProduceCardMovePositionType_Grave')
        for grow_effect in self._card_grow_rows(card):
            if str(grow_effect.get('effectType') or '') == 'ProduceCardGrowEffectType_PlayMovePositionTypeChange':
                move_type = str(grow_effect.get('playMovePositionType') or move_type)
        return MOVE_POSITION_MAP.get(move_type, 'grave')

    def _move_runtime_card(self, card: RuntimeCard, destination: str) -> None:
        """把一张运行时卡移动到指定区域，并触发对应 phase。"""

        self._detach_card(card)
        if destination == 'hand':
            self._send_to_hand_or_top_deck(card)
            move_phase = 'ProduceExamPhaseType_ExamCardMoveHand'
        elif destination == 'hold':
            self.hold.append(card)
            move_phase = None
        elif destination == 'lost':
            self.lost.append(card)
            move_phase = 'ProduceExamPhaseType_ExamCardMoveLost'
        elif destination == 'deck_first':
            self.deck.appendleft(card)
            move_phase = None
        elif destination == 'deck_random':
            deck_list = list(self.deck)
            index = int(self.np_random.integers(0, len(deck_list) + 1))
            deck_list.insert(index, card)
            self.deck = deque(deck_list)
            move_phase = None
        elif destination == 'deck_last':
            self.deck.append(card)
            move_phase = None
        else:
            self.grave.append(card)
            move_phase = 'ProduceExamPhaseType_ExamCardMoveGrave'
        if move_phase:
            self._dispatch_phase(move_phase, acting_card=card)

    def _detach_card(self, card: RuntimeCard) -> None:
        """把卡牌从当前所在区域摘除。"""

        for zone in (self.hand, self.grave, self.hold, self.lost, self.playing):
            for index, current in enumerate(zone):
                if current.uid == card.uid:
                    zone.pop(index)
                    return
        deck_list = list(self.deck)
        for index, current in enumerate(deck_list):
            if current.uid == card.uid:
                deck_list.pop(index)
                self.deck = deque(deck_list)
                return

    def _draw(self, count: int) -> None:
        """抽取指定数量的卡牌，并处理洗回牌堆。"""

        hand_limit = int(self.exam_setting.get('handLimit') or 5)
        for _ in range(count):
            if len(self.hand) >= hand_limit:
                return
            if not self.deck:
                if not self.grave:
                    return
                self._reshuffle_grave_into_deck()
            if not self.deck:
                return
            self.hand.append(self.deck.popleft())

    def _reshuffle_grave_into_deck(self) -> None:
        """把弃牌区洗回牌堆，遵守主数据中的洗牌配置。"""

        cards = list(self.grave)
        self.grave = []
        if cards:
            if self.exam_setting.get('fixMoveCardShuffleDeckEnable'):
                self.np_random.shuffle(cards)
            self.deck = deque(cards)

    def _card_label(self, card: RuntimeCard) -> str:
        """生成调试日志使用的卡牌标签。"""

        return f'{self.repository.card_name(card.base_card)}[{card.upgrade_count}]'

    def _lookup_card_row(self, card_id: str, upgrade_count: int) -> dict[str, Any] | None:
        """按卡牌 id 和强化次数查找最匹配的主数据行。"""

        return self.repository.card_row_by_upgrade(card_id, upgrade_count, fallback_to_canonical=True)

    def _card_grow_rows(self, card: RuntimeCard) -> list[dict[str, Any]]:
        """解析当前运行时卡牌挂载的成长效果行。"""

        rows = []
        for effect_id in card.grow_effect_ids:
            row = self.grow_effects.first(str(effect_id))
            if row:
                rows.append(row)
        return rows

    def _insert_card(self, card: RuntimeCard, destination: str) -> None:
        """把新建或复制的卡牌插入指定区域。"""

        if destination == 'hand':
            self._send_to_hand_or_top_deck(card)
        elif destination == 'hold':
            self.hold.append(card)
        elif destination == 'lost':
            self.lost.append(card)
        elif destination == 'deck_first':
            self.deck.appendleft(card)
        elif destination == 'deck_random':
            deck_list = list(self.deck)
            index = int(self.np_random.integers(0, len(deck_list) + 1))
            deck_list.insert(index, card)
            self.deck = deque(deck_list)
        elif destination == 'deck_last':
            self.deck.append(card)
        else:
            self.grave.append(card)

    def _sample_card_specs_from_search(self, search: dict[str, Any], effect: dict[str, Any]) -> list[dict[str, Any]]:
        """根据检索定义抽样待创建的卡牌规格。"""

        card_rows: list[dict[str, Any]] = []
        explicit_ids = [str(value) for value in search.get('produceCardIds', []) if value]
        upgrade_counts = [int(value) for value in search.get('upgradeCounts', []) if value is not None]
        if explicit_ids:
            for card_id in explicit_ids:
                card_row = self._lookup_card_row(card_id, upgrade_counts[0] if upgrade_counts else 0)
                if card_row is not None:
                    card_rows.append(card_row)
        elif search.get('produceCardRandomPoolId'):
            ratios = self.random_pools.all(str(search.get('produceCardRandomPoolId')))
            weighted_rows = []
            weights = []
            for ratio in ratios:
                card_row = self._lookup_card_row(str(ratio.get('produceCardId')), int(ratio.get('upgradeCount') or 0))
                if card_row is None:
                    continue
                weighted_rows.append(card_row)
                weights.append(float(ratio.get('ratio') or 1.0))
            if weighted_rows:
                pick_count = max(int(effect.get('pickCountMax') or effect.get('pickCountMin') or 1), 1)
                probabilities = np.array(weights, dtype=np.float64) / max(sum(weights), 1.0)
                indices = self.np_random.choice(len(weighted_rows), size=pick_count, replace=True, p=probabilities)
                card_rows.extend(weighted_rows[int(index)] for index in np.atleast_1d(indices))
        elif search.get('produceCardPoolId'):
            pool_row = self.card_pools.first(str(search.get('produceCardPoolId')))
            if pool_row:
                weighted_rows = []
                weights = []
                for ratio in pool_row.get('produceCardRatios', []):
                    card_row = self._lookup_card_row(str(ratio.get('id')), int(ratio.get('upgradeCount') or 0))
                    if card_row is None:
                        continue
                    weighted_rows.append(card_row)
                    weights.append(float(ratio.get('ratio') or 1.0))
                if weighted_rows:
                    pick_count = max(int(effect.get('pickCountMax') or effect.get('pickCountMin') or 1), 1)
                    probabilities = np.array(weights, dtype=np.float64) / max(sum(weights), 1.0)
                    indices = self.np_random.choice(len(weighted_rows), size=pick_count, replace=True, p=probabilities)
                    card_rows.extend(weighted_rows[int(index)] for index in np.atleast_1d(indices))
        return card_rows

    def _resolve_lesson_effect_value(self, effect: dict[str, Any], from_card: bool = False) -> float:
        """按资源、检索数量和 stance 结算课程分数效果。"""

        effect_type = str(effect.get('effectType') or '')
        base_value = self._direct_value(effect)
        ratio_value = self._ratio_value(effect)
        if from_card and effect_type == 'ProduceExamEffectType_ExamLessonDependExamReview':
            ratio_value += self._current_card_ratio_bonus('ProduceCardGrowEffectType_LessonDependExamReviewAdd')
        elif from_card and effect_type == 'ProduceExamEffectType_ExamLessonDependExamCardPlayAggressive':
            ratio_value += self._current_card_ratio_bonus('ProduceCardGrowEffectType_LessonDependExamCardPlayAggressiveAdd')
        elif from_card and effect_type == 'ProduceExamEffectType_ExamLessonDependBlock':
            ratio_value += self._current_card_ratio_bonus('ProduceCardGrowEffectType_LessonDependBlockAdd')
        search_count = self._search_cards(str(effect.get('produceCardSearchId') or '')).pool_size
        if effect_type == 'ProduceExamEffectType_ExamLessonFix':
            value = base_value
        elif effect_type == 'ProduceExamEffectType_ExamLessonDependExamReview':
            value = self._ceil_positive(self.resources['review'] * ratio_value)
        elif effect_type == 'ProduceExamEffectType_ExamLessonDependExamCardPlayAggressive':
            value = self._ceil_positive(self.resources['aggressive'] * ratio_value)
        elif effect_type == 'ProduceExamEffectType_ExamLessonDependBlock':
            value = self._ceil_positive(self.resources['block'] * ratio_value)
        elif effect_type == 'ProduceExamEffectType_ExamLessonDependParameterBuff':
            value = self._ceil_positive(self.resources['parameter_buff'] * ratio_value)
        elif effect_type == 'ProduceExamEffectType_ExamLessonDependPlayCardCountSum':
            value = self.total_counters['play_count'] * max(base_value, 1.0)
        elif effect_type == 'ProduceExamEffectType_ExamLessonDependStaminaConsumptionSum':
            value = self._ceil_positive(self.total_counters['stamina_spent'] * ratio_value)
        elif effect_type == 'ProduceExamEffectType_ExamLessonDependBlockConsumptionSum':
            value = self._ceil_positive(self.total_counters['block_consumed'] * ratio_value)
        elif effect_type == 'ProduceExamEffectType_ExamLessonDependBlockAndSearchCount':
            extra_ratio = float(effect.get('effectValue2') or 0) / 1000.0
            value = self._compose_referenced_gain(
                base=search_count * max(base_value, 1.0),
                referenced=self.resources['block'] * extra_ratio,
            )
        elif effect_type == 'ProduceExamEffectType_ExamLessonPerSearchCount':
            value = self._compose_referenced_gain(
                base=max(base_value, 1.0),
                referenced=search_count * (float(effect.get('effectValue2') or 0) / 1000.0),
            )
        elif effect_type == 'ProduceExamEffectType_ExamLessonFullPowerPoint':
            value = self._ceil_positive(self.resources['full_power_point'] * max(base_value, 1.0))
        elif effect_type in {'ProduceExamEffectType_ExamLessonAddMultipleLessonBuff', 'ProduceExamEffectType_ExamMultipleLessonBuffLesson'}:
            extra_ratio = float(effect.get('effectValue2') or effect.get('effectValue1') or 0) / 1000.0
            value = self._compose_referenced_gain(
                base=max(base_value, 1.0),
                referenced=self.resources['lesson_buff'] * extra_ratio,
            )
        elif effect_type == 'ProduceExamEffectType_ExamLessonAddMultipleParameterBuff':
            extra_ratio = float(effect.get('effectValue2') or 0) / 1000.0
            value = self._compose_referenced_gain(
                base=max(base_value, 1.0),
                referenced=self.resources['parameter_buff'] * extra_ratio,
            )
        else:
            value = base_value
        if from_card:
            value = self._adjust_direct_gain(
                value,
                add_grow_type='ProduceCardGrowEffectType_LessonAdd',
                reduce_grow_type='ProduceCardGrowEffectType_LessonReduce',
            )
        return max(self._apply_score_value_modifiers(value), 0.0)

    def _apply_score_value_modifiers(self, value: float) -> float:
        """把集中、热意、好调、指针和场地 debuff 统一叠加到得分值上。"""

        updated = max(value, 0.0)
        updated += self.resources['lesson_buff']
        updated += self.resources['enthusiastic']
        consumed_effects: list[TimedExamEffect] = []
        for timed in list(self.active_effects):
            modifier_type = str(timed.effect.get('effectType') or '')
            used = False
            if modifier_type == 'ProduceExamEffectType_ExamLessonValueMultiple':
                updated *= 1.0 + self._ratio_value(timed.effect)
                used = True
            elif modifier_type == 'ProduceExamEffectType_ExamLessonValueMultipleDown':
                updated *= max(1.0 - self._ratio_value(timed.effect), 0.0)
                used = True
            elif modifier_type == 'ProduceExamEffectType_ExamLessonValueMultipleDependReviewOrAggressive':
                per_stack = float(self.exam_setting.get('examLessonValueMultipleDependReviewOrAggressiveMultiplePermil') or 20) / 1000.0
                max_ratio = float(self.exam_setting.get('examLessonValueMultipleDependReviewOrAggressiveMaxPermil') or 500) / 1000.0
                updated *= 1.0 + min(min(self.resources['review'], self.resources['aggressive']) * per_stack, max_ratio)
                used = True
            elif modifier_type == 'ProduceExamEffectType_ExamGimmickLessonDebuff':
                updated = max(updated - self._raw_effect_value(timed.effect), 0.0)
                used = True
            elif modifier_type == 'ProduceExamEffectType_ExamGimmickParameterDebuff':
                factor = float(self.exam_setting.get('examGimmickParameterDebuffPermil') or 0) / 1000.0
                updated *= max(1.0 - factor, 0.0)
                used = True
            elif modifier_type == 'ProduceExamEffectType_ExamGimmickSlump':
                updated = 0.0
                used = True
            if used and timed.remaining_count is not None:
                consumed_effects.append(timed)
        for timed in consumed_effects:
            timed.remaining_count -= 1
        self.active_effects = [item for item in self.active_effects if item.remaining_count is None or item.remaining_count > 0]
        self._sync_effect_resources()

        if self.resources['parameter_buff'] > 0:
            parameter_multiple = float(self.exam_setting.get('examParameterBuffPermil') or 1500) / 1000.0
            if self.resources['parameter_buff_multiple_per_turn'] > 0:
                extra_ratio = float(self.exam_setting.get('examParameterBuffMultiplePerTurnPermil') or 0) / 1000.0
                parameter_multiple += self.resources['parameter_buff'] * self.resources['parameter_buff_multiple_per_turn'] * extra_ratio
            updated *= parameter_multiple
        if self.stance == 'concentration':
            key = f'examConcentrationLessonValueMultiplePermil{max(self.stance_level, 1)}'
            updated *= float(self.exam_setting.get(key) or self.exam_setting.get('examConcentrationLessonValueMultiplePermil') or 1000) / 1000.0
        elif self.stance == 'preservation':
            if self.stance_level >= 3:
                updated *= float(self.exam_setting.get('examOverPreservationLessonValueMultiplePermil') or 1000) / 1000.0
            else:
                key = f'examPreservationLessonValueMultiplePermil{max(self.stance_level, 1)}'
                updated *= float(self.exam_setting.get(key) or 1000) / 1000.0
        elif self.stance == 'full_power':
            updated *= float(self.exam_setting.get('examFullPowerLessonValueMultiplePermil') or 1000) / 1000.0
        return max(updated, 0.0)

    def _apply_scalar_modifiers(self, effect_type: str, amount: float) -> float:
        """对好调、元气、集中等资源应用场上修饰。"""

        updated = float(amount)
        if effect_type == 'ProduceExamEffectType_ExamBlock':
            if self._has_timed_effect('ProduceExamEffectType_ExamBlockRestriction'):
                return 0.0
            updated += self.resources['aggressive']
            updated -= self.resources['sleepy']
        for timed in self.active_effects:
            modifier_type = str(timed.effect.get('effectType') or '')
            if effect_type == 'ProduceExamEffectType_ExamCardPlayAggressive':
                if modifier_type == 'ProduceExamEffectType_ExamAggressiveAdditive':
                    updated += self._raw_effect_value(timed.effect)
                elif modifier_type == 'ProduceExamEffectType_ExamAggressiveValueMultiple':
                    updated *= 1.0 + self._ratio_value(timed.effect)
            elif effect_type == 'ProduceExamEffectType_ExamBlock':
                if modifier_type == 'ProduceExamEffectType_ExamBlockAddDown':
                    updated *= max(1.0 - float(self.exam_setting.get('examBlockAddDownPermil') or 0) / 1000.0, 0.0)
                elif modifier_type == 'ProduceExamEffectType_ExamBlockValueMultiple':
                    updated *= 1.0 + self._ratio_value(timed.effect)
            elif effect_type in {'ProduceExamEffectType_ExamReview', 'ProduceExamEffectType_ExamReviewAdditive'}:
                if modifier_type == 'ProduceExamEffectType_ExamReviewMultiple':
                    updated *= 1.0 + self._ratio_value(timed.effect)
            elif effect_type == 'ProduceExamEffectType_ExamParameterBuff':
                if modifier_type == 'ProduceExamEffectType_ExamParameterBuffAdditive':
                    updated *= 1.0 + self._ratio_value(timed.effect)
            elif effect_type == 'ProduceExamEffectType_ExamLessonBuff':
                if modifier_type == 'ProduceExamEffectType_ExamLessonBuffAdditive':
                    updated += self._raw_effect_value(timed.effect)
                elif modifier_type == 'ProduceExamEffectType_ExamLessonBuffMultiple':
                    updated *= 1.0 + self._ratio_value(timed.effect)
            elif effect_type == 'ProduceExamEffectType_ExamFullPowerPoint' and modifier_type == 'ProduceExamEffectType_ExamFullPowerPointAdditive':
                updated += self._raw_effect_value(timed.effect)
        return max(updated, 0.0)

    def _apply_preservation_release(self, target_stance: str) -> None:
        """温存/悠闲解除时，根据阶段发放对应奖励。"""

        if self.stance != 'preservation' or self.stance_level <= 0:
            return
        if self.stance_level >= 3:
            self._gain_block(float(self.exam_setting.get('overPreservationReleaseBlockAdd') or 0), effect_type='ProduceExamEffectType_ExamBlockFix')
            self.play_limit += int(round(float(self.exam_setting.get('overPreservationReleasePlayableValueAdd') or 0)))
            if target_stance == 'full_power':
                self._add_lesson_grow_effect_to_all_cards(float(self.exam_setting.get('overPreservationReleaseToFullPowerGrowEffectLessonAdd') or 0))
            else:
                self._gain_enthusiastic(float(self.exam_setting.get('overPreservationReleaseEnthusiastic') or 0))
            return
        enthusiastic_key = f'preservationReleaseEnthusiastic{self.stance_level}'
        block_key = f'preservationReleaseBlockAdd{self.stance_level}'
        playable_key = f'preservationReleasePlayableValueAdd{self.stance_level}'
        self._gain_enthusiastic(float(self.exam_setting.get(enthusiastic_key) or 0))
        block_add = float(self.exam_setting.get(block_key) or 0)
        if block_add > 0:
            self._gain_block(block_add, effect_type='ProduceExamEffectType_ExamBlockFix')
        self.play_limit += int(round(float(self.exam_setting.get(playable_key) or 0)))

    def _move_hold_cards_to_hand(self) -> None:
        """全力进场时把保留区的卡尽量送回手牌。"""

        held_cards = list(self.hold)
        self.hold = []
        for card in held_cards:
            self._send_to_hand_or_top_deck(card)

    def _add_lesson_grow_effect_to_all_cards(self, value: float) -> None:
        """给所有运行时技能卡追加固定打分成长效果。"""

        amount = int(round(value))
        if amount <= 0:
            return
        grow_effect_id = f'g_effect-lesson_add-{amount}'
        if self.grow_effects.first(grow_effect_id) is None:
            return
        seen: set[int] = set()
        for zone in (self.deck, self.hand, self.grave, self.hold, self.lost, self.playing):
            for card in zone:
                if card.uid in seen:
                    continue
                seen.add(card.uid)
                if grow_effect_id not in card.grow_effect_ids:
                    card.grow_effect_ids.append(grow_effect_id)

    def _enter_concentration(self, level: int) -> None:
        """进入强气指针；重复进入时提升到更高阶段。"""

        if self.stance == 'full_power' and self.stance_locked:
            return
        target_level = min(max(int(level), 1), 2)
        if self.stance == 'concentration':
            if target_level <= 1 and self.stance_level == 1:
                target_level = 2
            else:
                target_level = max(self.stance_level, target_level)
            self._set_stance('concentration', target_level)
            return
        if self.stance == 'preservation':
            self._apply_preservation_release('concentration')
        self._set_stance('concentration', target_level)

    def _enter_preservation(self, level: int) -> None:
        """进入温存/悠闲指针；重复进入时提升到更高阶段。"""

        if self.stance == 'full_power' and self.stance_locked:
            return
        target_level = 3 if int(level) >= 3 else min(max(int(level), 1), 2)
        if self.stance == 'preservation':
            if self.stance_level >= 3 and target_level < 3:
                return
            if target_level >= 3:
                self._set_stance('preservation', 3)
                return
            if target_level <= 1 and self.stance_level == 1:
                target_level = 2
            else:
                target_level = max(self.stance_level, target_level)
            self._set_stance('preservation', target_level)
            return
        self._set_stance('preservation', target_level)

    def _enter_full_power(self) -> None:
        """进入全力，结算温存解除奖励并移动保留区。"""

        if self.stance == 'full_power':
            return
        if self.stance == 'preservation':
            self._apply_preservation_release('full_power')
        if not self._set_stance('full_power', 1):
            return
        self.stance_locked = True
        self.play_limit += int(round(float(self.exam_setting.get('fullPowerPlayableValueAdd') or 0)))
        self._move_hold_cards_to_hand()

    def _release_full_power(self) -> None:
        """在回合结束时解除全力锁定。"""

        if self.stance != 'full_power':
            return
        self.stance_locked = False
        self._set_stance('neutral', 0)

    def _reset_stance(self) -> None:
        """手动解除当前指针状态。"""

        if self.stance == 'full_power' and self.stance_locked:
            return
        if self.stance == 'preservation':
            self._apply_preservation_release('neutral')
        self._set_stance('neutral', 0)

    def _set_stance(self, stance: str, level: int = 1) -> bool:
        """切换当前指针状态，并补发相关触发 phase。"""

        if self.stance_locked and self.stance != stance:
            return False
        normalized_level = 0 if stance == 'neutral' else max(int(level), 1)
        previous = self.stance
        previous_level = self.stance_level
        if previous == stance and previous_level == normalized_level:
            return False
        self.stance = stance
        self.stance_level = normalized_level
        self.total_counters['stance_changes'] += 1
        if stance == 'concentration':
            self.total_counters['stance_concentration'] += 1
        elif stance == 'full_power':
            self.total_counters['stance_full_power'] += 1
        elif stance == 'preservation':
            self.total_counters['stance_preservation'] += 1
        self._dispatch_interval_phase('ProduceExamPhaseType_ExamStanceChangeCountInterval', self.total_counters['stance_changes'])
        phase = next((phase for phase, mapped in STANCE_PHASES.items() if mapped == stance), None)
        if phase:
            self._dispatch_phase(phase)
        if previous == 'concentration' and stance != 'concentration':
            self._dispatch_phase('ProduceExamPhaseType_ExamStanceChangeFromConcentration')
        if previous == 'full_power' and stance != 'full_power':
            self._dispatch_phase('ProduceExamPhaseType_ExamStanceChangeFromFullPower')
        self._sync_stance_resources()
        return True

    def _lesson_type_for_card(self, card: RuntimeCard | None) -> str:
        """兼容旧调用，统一返回战斗上下文中的课程类型。"""

        return self._current_lesson_type()

    def _direct_value(self, effect: dict[str, Any]) -> float:
        """解析效果的直接数值，并套用标量资源修正。"""

        effect_type = str(effect.get('effectType') or '')
        base_value = self._raw_effect_value(effect)
        if effect_type in DURATION_RESOURCE_TYPES:
            return self._parameter_buff_gain_value(effect)
        if effect_type in SCALAR_RESOURCE_TYPES or effect_type in {
            'ProduceExamEffectType_ExamStaminaDamage',
            'ProduceExamEffectType_ExamStaminaRecover',
            'ProduceExamEffectType_ExamStaminaRecoverFix',
            'ProduceExamEffectType_ExamStaminaReduce',
            'ProduceExamEffectType_ExamBlock',
            'ProduceExamEffectType_ExamBlockFix',
            'ProduceExamEffectType_ExamCardDraw',
        }:
            return self._apply_scalar_modifiers(effect_type, base_value)
        if base_value == 0:
            base_value = float(effect.get('effectCount') or 0)
        if base_value == 0:
            base_value = 1.0
        return max(base_value, 0.0)

    def _ratio_value(self, effect: dict[str, Any]) -> float:
        """把千分比字段转换成 0-1 浮点比例。"""

        value = float(effect.get('effectValue1') or effect.get('effectValue2') or 0)
        return max(value, 0.0) / 1000.0 if value > 0 else 0.0

    def _count_value(self, effect: dict[str, Any]) -> float:
        """解析效果次数字段，缺省时按 1 处理。"""

        return float(effect.get('effectCount') or effect.get('effectValue1') or 1)
