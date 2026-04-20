"""Gymnasium 环境封装，把运行时状态整理成固定张量观测。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import LessonTrainingSpec, MasterDataRepository, ScenarioSpec
from .exam_runtime import ExamActionCandidate, ExamRuntime, RuntimeCard, default_audition_row_selector
from .idol_config import augment_loadout_with_produce_items, build_idol_loadout, build_initial_exam_deck
from .loadout import ExamEpisodeRandomizationConfig, IdolLoadout
from .manual_exam_setups import ManualExamSetupDataset, ResolvedManualExamSetup
from .produce_runtime import ProduceRuntime
from .reward_config import RewardConfig, build_reward_config


@dataclass
class ActionView:
    """供掩码策略网络消费的定长动作槽表示。"""

    label: str
    kind: str
    feature: np.ndarray
    payload: dict[str, Any]


def _sanitize_float32_array(values: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    """把观测里的 NaN / Inf 压回有限范围。"""

    sanitized = np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=upper,
        neginf=lower,
    )
    return np.clip(sanitized, lower, upper).astype(np.float32, copy=False)


@dataclass(frozen=True)
class ManualEpisodeSelection:
    """Resolved manual record picked for the current episode."""

    setup: ResolvedManualExamSetup
    loadout: IdolLoadout | None
    stage_type: str


def _next_seed(rng: np.random.Generator) -> int:
    """从环境 RNG 派生一个子种子，保证 episode 内外随机性可复现。"""

    return int(rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))


def _bounded_positive(value: float, scale: float) -> float:
    """把非负数平滑压到 [0, 1) 区间，避免极端资源值打穿观测空间。"""

    numeric = max(float(value), 0.0)
    base = max(float(scale), 1e-6)
    return numeric / (numeric + base)


def _bounded_signed(value: float, scale: float) -> float:
    """把有符号数平滑压到 (-1, 1) 区间。"""

    numeric = float(value)
    base = max(float(scale), 1e-6)
    return numeric / (abs(numeric) + base)


_LOADOUT_PLAN_TYPES = (
    'ProducePlanType_Common',
    'ProducePlanType_Plan1',
    'ProducePlanType_Plan2',
    'ProducePlanType_Plan3',
)


def _is_card_lesson_action_type(action_type: str) -> bool:
    """独立 lesson 模式只接受需要打牌的课程动作。"""

    normalized = str(action_type or '')
    return normalized.startswith('lesson_')


class GakumasPlanningEnv(gym.Env):
    """把培育规划运行时包装成 Gym 环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        repository: MasterDataRepository,
        scenario: ScenarioSpec,
        seed: int | None = None,
        idol_loadout: IdolLoadout | None = None,
        include_action_labels_in_step_info: bool = False,
    ):
        """初始化培育环境，并固定动作槽与观测维度。"""

        super().__init__()
        self.repository = repository
        self.taxonomy = repository.taxonomy
        self.scenario = scenario
        self.idol_loadout = idol_loadout
        self.include_action_labels_in_step_info = include_action_labels_in_step_info
        self._initial_seed = seed
        self._seed_consumed = False
        self.runtime = ProduceRuntime(repository, scenario, seed=seed, idol_loadout=self.idol_loadout)
        self.route_one_hot = np.array(
            [1.0, 0.0] if scenario.route_type == 'first_star' else [0.0, 1.0],
            dtype=np.float32,
        )
        self.max_actions = len(scenario.action_types)
        self.global_feature_names = (
            'step_ratio',
            'remaining_step_ratio',
            'audition_progress',
            'stamina_ratio',
            'produce_point_ratio',
            'fan_vote_ratio',
            'vocal_ratio',
            'dance_ratio',
            'visual_ratio',
            'vocal_growth',
            'dance_growth',
            'visual_growth',
            'deck_quality',
            'drink_quality',
            'deck_size_ratio',
            'drink_inventory_ratio',
            'exam_enchant_count',
            'refresh_used_ratio',
            'last_exam_score_ratio',
            'route_first_star',
            'route_nia',
            'score_weight_vocal',
            'score_weight_dance',
            'score_weight_visual',
            'audition_difficulty_bonus',
            'audition_parameter_bonus',
            'producer_level_ratio',
            'idol_rank_ratio',
            'dearness_level_ratio',
            'exam_score_bonus_multiplier',
            'is_shop_phase',
            'shop_card_modified_in_visit',
            'has_pending_audition',
            'shop_card_modify_count',
            'remaining_audition_ratio',
            'activity_produce_point_bonus',
            'business_vote_bonus',
            'lesson_present_point_bonus',
            'support_event_point_bonus',
            'support_event_stat_bonus',
            'support_event_stamina_bonus',
            'audition_vote_bonus',
            'audition_turn_modifier',
            'before_audition_refresh_penalty',
            'generic_sp_rate_bonus',
            'vocal_sp_rate_bonus',
            'dance_sp_rate_bonus',
            'visual_sp_rate_bonus',
            'shop_discount',
            'reward_card_count_bonus',
            'customize_slots',
            'exclude_count_bonus',
            'reroll_count_bonus',
            'card_upgrade_probability_bonus',
        )
        self.global_dim = len(self.global_feature_names)
        self.action_feature_dim = (
            len(self.taxonomy.action_types)
            + len(self.taxonomy.produce_effect_types)
            + len(self.taxonomy.exam_effect_types)
            + len(self.taxonomy.card_categories)
            + len(self.taxonomy.card_rarities)
            + len(self.taxonomy.card_cost_types)
            + 17
        )
        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Dict(
            {
                'global': spaces.Box(-20.0, 20.0, shape=(self.global_dim,), dtype=np.float32),
                'action_features': spaces.Box(
                    -20.0,
                    20.0,
                    shape=(self.max_actions, self.action_feature_dim),
                    dtype=np.float32,
                ),
                'action_mask': spaces.Box(0.0, 1.0, shape=(self.max_actions,), dtype=np.float32),
            }
        )
        self._candidates: list[ActionView] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """重置环境并返回首个培育观测。"""

        episode_seed = seed if seed is not None else (self._initial_seed if not self._seed_consumed else None)
        super().reset(seed=episode_seed)
        self._seed_consumed = True
        runtime_seed = _next_seed(self.np_random)
        self.runtime = ProduceRuntime(self.repository, self.scenario, seed=runtime_seed, idol_loadout=self.idol_loadout)
        self.runtime.reset()
        obs = self._build_observation()
        info = {
            'scenario': self.scenario.scenario_id,
            'action_labels': [candidate.label for candidate in self._candidates],
        }
        return obs, info

    def _produce_effect_types(self, candidate) -> list[str]:
        """汇总动作候选里显式和隐式的 ProduceEffectType。"""

        effect_types = set(candidate.effect_types)
        for effect_id in candidate.produce_effect_ids + candidate.success_effect_ids + candidate.fail_effect_ids:
            row = self.repository.produce_effects.first(str(effect_id))
            if row and row.get('produceEffectType'):
                effect_types.add(str(row['produceEffectType']))
        return sorted(effect_types)

    def _global_observation(self) -> np.ndarray:
        """构造培育全局状态向量。"""

        state = self.runtime.state
        max_refresh = max(self.scenario.max_refresh_count, 1)
        parameter_scale = max(float(getattr(self.scenario, 'parameter_growth_limit', 0.0) or 0.0), 1.0)
        global_values = {
            'step_ratio': state['step'] / max(self.scenario.steps, 1),
            'remaining_step_ratio': max(self.scenario.steps - state['step'], 0) / max(self.scenario.steps, 1),
            'audition_progress': state['audition_index'] / max(len(self.scenario.audition_sequence), 1),
            'stamina_ratio': state['stamina'] / max(state['max_stamina'], 1.0),
            'produce_point_ratio': state['produce_points'] / 150.0,
            'fan_vote_ratio': state['fan_votes'] / 5000.0,
            'vocal_ratio': state['vocal'] / parameter_scale,
            'dance_ratio': state['dance'] / parameter_scale,
            'visual_ratio': state['visual'] / parameter_scale,
            'vocal_growth': state['vocal_growth'],
            'dance_growth': state['dance_growth'],
            'visual_growth': state['visual_growth'],
            'deck_quality': state['deck_quality'] / 20.0,
            'drink_quality': state['drink_quality'] / 10.0,
            'deck_size_ratio': len(self.runtime.deck) / 40.0,
            'drink_inventory_ratio': len(self.runtime.drinks) / max(self.scenario.drink_limit, 1),
            'exam_enchant_count': len(self.runtime.exam_status_enchant_ids) / 20.0,
            'refresh_used_ratio': state['refresh_used'] / max_refresh,
            'last_exam_score_ratio': state['last_exam_score'] / 5000.0,
            'route_first_star': self.route_one_hot[0],
            'route_nia': self.route_one_hot[1],
            'score_weight_vocal': self.scenario.score_weights[0],
            'score_weight_dance': self.scenario.score_weights[1],
            'score_weight_visual': self.scenario.score_weights[2],
            'audition_difficulty_bonus': state['audition_difficulty_bonus'],
            'audition_parameter_bonus': state['audition_parameter_bonus'],
            'producer_level_ratio': state['producer_level'] / 60.0,
            'idol_rank_ratio': state['idol_rank'] / 20.0,
            'dearness_level_ratio': state['dearness_level'] / 20.0,
            'exam_score_bonus_multiplier': state['exam_score_bonus_multiplier'] / 4.0,
            'is_shop_phase': 1.0 if self.runtime.pre_audition_phase == 'shop' else 0.0,
            'shop_card_modified_in_visit': float(state.get('shop_card_modified_in_visit') or 0.0),
            'has_pending_audition': 1.0 if self.runtime.pending_audition_stage else 0.0,
            'shop_card_modify_count': _bounded_positive(float(state.get('shop_card_modify_count') or 0.0), 4.0),
            'remaining_audition_ratio': max(len(self.runtime.checkpoints) - state['audition_index'], 0) / max(len(self.runtime.checkpoints), 1),
            'activity_produce_point_bonus': _bounded_signed(float(state.get('activity_produce_point_bonus') or 0.0), 0.5),
            'business_vote_bonus': _bounded_signed(float(state.get('business_vote_bonus') or 0.0), 0.5),
            'lesson_present_point_bonus': _bounded_signed(float(state.get('lesson_present_point_bonus') or 0.0), 0.5),
            'support_event_point_bonus': _bounded_signed(float(state.get('support_event_point_bonus') or 0.0), 0.5),
            'support_event_stat_bonus': _bounded_signed(float(state.get('support_event_stat_bonus') or 0.0), 0.5),
            'support_event_stamina_bonus': _bounded_signed(float(state.get('support_event_stamina_bonus') or 0.0), 0.5),
            'audition_vote_bonus': _bounded_signed(float(state.get('audition_vote_bonus') or 0.0), 0.5),
            'audition_turn_modifier': _bounded_signed(float(state.get('audition_turn_modifier') or 0.0), 2.0),
            'before_audition_refresh_penalty': _bounded_signed(float(state.get('before_audition_refresh_penalty') or 0.0), 0.5),
            'generic_sp_rate_bonus': _bounded_signed(float(state.get('generic_sp_rate_bonus') or 0.0), 0.5),
            'vocal_sp_rate_bonus': _bounded_signed(float(state.get('vocal_sp_rate_bonus') or 0.0), 0.5),
            'dance_sp_rate_bonus': _bounded_signed(float(state.get('dance_sp_rate_bonus') or 0.0), 0.5),
            'visual_sp_rate_bonus': _bounded_signed(float(state.get('visual_sp_rate_bonus') or 0.0), 0.5),
            'shop_discount': _bounded_signed(float(state.get('shop_discount') or 0.0), 0.5),
            'reward_card_count_bonus': _bounded_positive(float(state.get('reward_card_count_bonus') or 0.0), 2.0),
            'customize_slots': _bounded_positive(float(state.get('customize_slots') or 0.0), 4.0),
            'exclude_count_bonus': _bounded_positive(float(state.get('exclude_count_bonus') or 0.0), 3.0),
            'reroll_count_bonus': _bounded_positive(float(state.get('reroll_count_bonus') or 0.0), 3.0),
            'card_upgrade_probability_bonus': _bounded_positive(float(state.get('card_upgrade_probability_bonus') or 0.0), 0.5),
        }
        return np.array(
            [float(global_values[name]) for name in self.global_feature_names],
            dtype=np.float32,
        )

    def _candidate_feature(self, candidate) -> np.ndarray:
        """把单个培育动作编码成定长特征。"""

        effect_types = self._produce_effect_types(candidate)
        state = self.runtime.state
        numeric = np.array(
            [
                candidate.stamina_delta / max(state['max_stamina'], 1.0),
                candidate.produce_point_delta / 100.0,
                candidate.success_probability,
                len(candidate.produce_effect_ids) / 8.0,
                len(candidate.success_effect_ids) / 8.0,
                len(candidate.fail_effect_ids) / 8.0,
                1.0 if candidate.produce_card_id else 0.0,
                len(effect_types) / 8.0,
                1.0 if candidate.available else 0.0,
                state['stamina'] / max(state['max_stamina'], 1.0),
                state['deck_quality'] / 20.0,
                state['drink_quality'] / 10.0,
                len(candidate.exam_effect_types) / 8.0,
                1.0 if candidate.resource_type == 'ProduceResourceType_ProduceCard' else 0.0,
                1.0 if candidate.resource_type == 'ProduceResourceType_ProduceDrink' else 0.0,
                1.0 if candidate.target_deck_index >= 0 else 0.0,
                float(candidate.resource_level) / 2.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                self.taxonomy.encode_actions([candidate.action_type]),
                self.taxonomy.encode_produce_effects(effect_types),
                self.taxonomy.encode_exam_effects(candidate.exam_effect_types),
                self.taxonomy.encode_categories([candidate.card_category] if candidate.card_category else []),
                self.taxonomy.encode_rarities([candidate.card_rarity] if candidate.card_rarity else []),
                self.taxonomy.encode_cost_types([candidate.card_cost_type] if candidate.card_cost_type else []),
                numeric,
            ]
        ).astype(np.float32)

    def _build_candidates(self) -> list[ActionView]:
        """把培育候选动作编码成固定槽位。"""

        runtime_candidates = self.runtime.legal_actions()
        return [
            ActionView(
                label=candidate.label,
                kind=candidate.action_type,
                feature=self._candidate_feature(candidate),
                payload={'available': candidate.available, 'index': index},
            )
            for index, candidate in enumerate(runtime_candidates)
        ]

    def _build_observation(self) -> dict[str, np.ndarray]:
        """刷新候选动作并拼装培育观测。"""

        self._candidates = self._build_candidates()
        return {
            'global': self._global_observation(),
            'action_features': np.stack([candidate.feature for candidate in self._candidates]).astype(np.float32),
            'action_mask': self.action_masks().astype(np.float32),
        }

    def action_masks(self) -> np.ndarray:
        """为 MaskablePPO 提供当前动作掩码。"""

        if not self._candidates:
            self._candidates = self._build_candidates()
        return np.array([bool(candidate.payload.get('available', False)) for candidate in self._candidates], dtype=bool)

    def step(self, action: int):
        """执行一个培育动作槽。"""

        candidate = self._candidates[action]
        if not candidate.payload.get('available', False):
            obs = self._build_observation()
            info = {
                'scenario': self.scenario.scenario_id,
                'invalid_action': True,
            }
            if self.include_action_labels_in_step_info:
                info['action_labels'] = [item.label for item in self._candidates]
            return obs, -0.25, False, False, info

        reward, terminated, runtime_info = self.runtime.step(int(candidate.payload['index']))
        obs = self._build_observation()
        info = {'scenario': self.scenario.scenario_id}
        if self.include_action_labels_in_step_info:
            info['action_labels'] = [item.label for item in self._candidates]
        info.update(runtime_info)
        return obs, float(reward), bool(terminated), False, info


class GakumasExamEnv(gym.Env):
    """把考试运行时包装成 Gym 环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        repository: MasterDataRepository,
        scenario: ScenarioSpec,
        battle_kind: str = 'exam',
        stage_type: str | None = None,
        max_hand_cards: int = 48,
        max_drinks: int | None = None,
        seed: int | None = None,
        idol_loadout: IdolLoadout | None = None,
        base_loadout_config: dict[str, Any] | None = None,
        episode_randomization: ExamEpisodeRandomizationConfig | None = None,
        exam_reward_mode: str = 'score',
        exam_starting_stamina_mode: str = 'full',
        exam_starting_stamina_min_ratio: float = 0.6,
        exam_starting_stamina_max_ratio: float = 1.0,
        lesson_action_type: str | None = None,
        lesson_action_types: tuple[str, ...] | list[str] | None = None,
        lesson_level_index: int | None = None,
        include_action_labels_in_step_info: bool = False,
        include_deck_features: bool = False,
        manual_setup_dataset: ManualExamSetupDataset | None = None,
        initial_deck_guaranteed_effect_counts: dict[str, int] | None = None,
        initial_deck_forced_card_groups: dict[str, tuple[str, ...]] | None = None,
        reward_config: RewardConfig | None = None,
    ):
        """初始化考试环境，并预分配手牌和饮料动作槽。"""

        super().__init__()
        self.repository = repository
        self.taxonomy = repository.taxonomy
        self.scenario = scenario
        normalized_battle_kind = str(battle_kind or 'exam').strip().lower()
        if normalized_battle_kind not in {'exam', 'lesson'}:
            raise ValueError(f'Unsupported battle kind: {battle_kind}')
        self.battle_kind = normalized_battle_kind
        self.stage_type = stage_type or scenario.default_stage
        self.max_hand_cards = max_hand_cards
        self.max_drinks = max(max_drinks or scenario.drink_limit, scenario.drink_limit)
        self.idol_loadout = idol_loadout
        self.base_loadout_config = dict(base_loadout_config or {})
        self.episode_randomization = episode_randomization or ExamEpisodeRandomizationConfig()
        self.exam_reward_mode = 'clear' if self.battle_kind == 'lesson' else exam_reward_mode
        self.reward_config = reward_config or build_reward_config(self.exam_reward_mode)
        if exam_starting_stamina_mode not in {'full', 'random'}:
            raise ValueError(f'Unsupported exam starting stamina mode: {exam_starting_stamina_mode}')
        self.exam_starting_stamina_mode = exam_starting_stamina_mode
        self.exam_starting_stamina_min_ratio = float(exam_starting_stamina_min_ratio)
        self.exam_starting_stamina_max_ratio = float(exam_starting_stamina_max_ratio)
        requested_lesson_action_type = str(lesson_action_type or '').strip()
        requested_lesson_action_pool = tuple(
            str(value)
            for value in (lesson_action_types or ())
            if str(value or '')
        )
        self.fixed_lesson_action_type = requested_lesson_action_type
        self.lesson_action_pool = self._resolve_lesson_action_pool(
            requested_lesson_action_type,
            requested_lesson_action_pool,
        )
        self.lesson_level_index = max(int(lesson_level_index or 0), 0)
        self.current_lesson_action_type = requested_lesson_action_type if self.battle_kind == 'lesson' else ''
        self.current_lesson_spec: LessonTrainingSpec | None = None
        self.include_action_labels_in_step_info = include_action_labels_in_step_info
        self.include_deck_features = include_deck_features
        self.manual_setup_dataset = manual_setup_dataset
        self.initial_deck_guaranteed_effect_counts = dict(initial_deck_guaranteed_effect_counts or {})
        self.initial_deck_forced_card_groups = {
            str(effect_type): tuple(str(card_id) for card_id in card_ids if str(card_id or '').strip())
            for effect_type, card_ids in (initial_deck_forced_card_groups or {}).items()
            if tuple(str(card_id) for card_id in card_ids if str(card_id or '').strip())
        }
        self.randomize_stage_type = False
        self.current_stage_type = self.stage_type
        self.current_loadout = idol_loadout
        self._initial_seed = seed
        self._seed_consumed = False
        self.idol_card_pool = tuple(
            str(value)
            for value in self.base_loadout_config.get('idol_card_ids', ())
            if str(value or '')
        )
        if not self.idol_card_pool and self.idol_loadout is not None:
            self.idol_card_pool = (self.idol_loadout.idol_card_id,)
        self._idol_loadout_cache: dict[tuple[Any, ...], IdolLoadout] = {}
        self.current_manual_setup: ResolvedManualExamSetup | None = None
        self.stage_type_ids = self._resolve_stage_type_ids()
        self.stage_context_dim = len(self.stage_type_ids) + 1
        self.loadout_character_ids = self._resolve_loadout_character_ids()
        self.loadout_rarities = self._resolve_loadout_rarities()
        self.loadout_context_dim = 4 + len(self.loadout_character_ids) + len(self.loadout_rarities) + len(self.scenario.focus_effect_types) + 4
        self.base_fan_votes = (
            float(self.base_loadout_config.get('fan_votes'))
            if self.base_loadout_config.get('fan_votes') is not None
            else None
        )
        initial_rng = np.random.default_rng(seed)
        if self.battle_kind == 'lesson' and self.current_loadout is None:
            self.current_loadout = self._sample_episode_loadout(initial_rng)
        self.runtime = self._build_runtime(
            runtime_seed=seed,
            rng=initial_rng,
            loadout=self.current_loadout,
            stage_type=self.current_stage_type,
            starting_stamina=None,
        )
        self.max_actions = self.max_hand_cards + self.max_drinks + 1
        self.deck_feature_dim = (
            len(self.taxonomy.exam_effect_types)
            + 2  # avg evaluation, avg stamina cost
            + len(self.taxonomy.card_cost_types)
            + len(self.taxonomy.card_categories)
        ) if self.include_deck_features else 0
        self._refresh_episode_randomization_flags()
        self.global_dim = 48 + self.stage_context_dim + self.loadout_context_dim + self.deck_feature_dim
        self.action_feature_dim = (
            len(self.taxonomy.action_types)
            + len(self.taxonomy.exam_effect_types)
            + len(self.taxonomy.trigger_phases)
            + len(self.taxonomy.card_categories)
            + len(self.taxonomy.card_rarities)
            + len(self.taxonomy.card_cost_types)
            + 14
        )
        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Dict(
            {
                'global': spaces.Box(-20.0, 20.0, shape=(self.global_dim,), dtype=np.float32),
                'action_features': spaces.Box(
                    -20.0,
                    20.0,
                    shape=(self.max_actions, self.action_feature_dim),
                    dtype=np.float32,
                ),
                'action_mask': spaces.Box(0.0, 1.0, shape=(self.max_actions,), dtype=np.float32),
            }
        )
        self._candidates: list[ActionView] = []
        self._candidate_mask = np.zeros(self.max_actions, dtype=bool)
        self._action_type_size = len(self.taxonomy.action_types)
        self._exam_effect_size = len(self.taxonomy.exam_effect_types)
        self._trigger_phase_size = len(self.taxonomy.trigger_phases)
        self._category_size = len(self.taxonomy.card_categories)
        self._rarity_size = len(self.taxonomy.card_rarities)
        self._cost_type_size = len(self.taxonomy.card_cost_types)
        effect_start = self._action_type_size
        trigger_start = effect_start + self._exam_effect_size
        self._exam_effect_slice = slice(effect_start, trigger_start)
        self._trigger_phase_slice = slice(trigger_start, trigger_start + self._trigger_phase_size)
        self._card_static_prefix_cache: dict[str, np.ndarray] = {}
        self._drink_static_prefix_cache: dict[str, np.ndarray] = {}
        self._card_action_vector = self.taxonomy.encode_actions(['card'])
        self._drink_action_vector = self.taxonomy.encode_actions(['drink'])
        self._end_turn_action_vector = self.taxonomy.encode_actions(['end_turn'])
        self._zero_exam_effect_vector = np.zeros(self._exam_effect_size, dtype=np.float32)
        self._zero_trigger_phase_vector = np.zeros(self._trigger_phase_size, dtype=np.float32)
        self._zero_category_vector = np.zeros(self._category_size, dtype=np.float32)
        self._zero_cost_type_vector = np.zeros(self._cost_type_size, dtype=np.float32)
        self._empty_card_feature = np.zeros(self.action_feature_dim, dtype=np.float32)
        self._empty_drink_feature = np.zeros(self.action_feature_dim, dtype=np.float32)
        self._end_turn_prefix = np.concatenate(
            [
                self._end_turn_action_vector,
                self._zero_exam_effect_vector,
                self._zero_trigger_phase_vector,
                self._zero_category_vector,
                np.zeros(self._rarity_size, dtype=np.float32),
                self._zero_cost_type_vector,
            ]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """在每次考试开始时采样整局固定的上下文，并构造首个观测。"""

        episode_seed = seed if seed is not None else (self._initial_seed if not self._seed_consumed else None)
        super().reset(seed=episode_seed)
        self._seed_consumed = True
        focus_effect_type = None
        if options and options.get('focus_effect_type'):
            focus_effect_type = str(options['focus_effect_type'])
        rng = self.np_random
        if self.randomize_stage_type and self.scenario.audition_sequence:
            self.current_stage_type = str(rng.choice(self.scenario.audition_sequence))
        else:
            self.current_stage_type = self.stage_type
        runtime_seed = _next_seed(rng)
        self.current_loadout = self._sample_episode_loadout(rng)
        self.current_manual_setup = None
        starting_stamina = self._sample_episode_starting_stamina(rng)
        selected_stage_type = self.current_stage_type
        manual_selection = self._resolve_manual_episode_selection(rng)
        if manual_selection is not None:
            self.current_manual_setup = manual_selection.setup
            self.current_loadout = manual_selection.loadout
            selected_stage_type = manual_selection.stage_type
            self.current_stage_type = selected_stage_type
            deck = list(manual_selection.setup.deck_rows)
            drinks = list(manual_selection.setup.drink_rows)
        else:
            deck = build_initial_exam_deck(
                self.repository,
                self.scenario,
                focus_effect_type=focus_effect_type,
                rng=rng,
                loadout=self.current_loadout,
                guaranteed_effect_counts=self.initial_deck_guaranteed_effect_counts,
                forced_card_groups=self.initial_deck_forced_card_groups,
            )
            drinks = self.repository.build_drink_inventory(
                self.scenario,
                rng=rng,
                plan_type=self.current_loadout.stat_profile.plan_type if self.current_loadout is not None else None,
            )
        self.runtime = self._build_runtime(
            runtime_seed=runtime_seed,
            rng=rng,
            loadout=self.current_loadout,
            stage_type=selected_stage_type,
            deck=deck,
            drinks=drinks,
            starting_stamina=starting_stamina,
        )
        self.runtime.reset()
        obs = self._build_observation()
        info = {
            'scenario': self.scenario.scenario_id,
            'stage_type': self.current_stage_type,
            'battle_kind': self.runtime.battle_kind,
            'turn_color': self.runtime.current_turn_color,
            'turn_color_label': self.runtime.turn_color_label(),
            'fan_votes': self.runtime.fan_votes,
            'fan_vote_baseline': self.runtime._reported_fan_vote_baseline(),
            'fan_vote_requirement': self.runtime._reported_fan_vote_requirement(),
            'action_labels': [candidate.label for candidate in self._candidates],
        }
        if self.current_lesson_spec is not None:
            info.update(
                {
                    'lesson_action_type': self.current_lesson_spec.action_type,
                    'lesson_name': self.current_lesson_spec.name,
                    'lesson_level_index': self.current_lesson_spec.level_index,
                    'lesson_target_value': self.current_lesson_spec.clear_target,
                    'lesson_perfect_value': self.current_lesson_spec.perfect_target,
                }
            )
        if self.current_loadout is not None:
            info['episode_context'] = {
                'idol_card_id': self.current_loadout.idol_card_id,
                'character_id': self.current_loadout.stat_profile.character_id,
                'plan_type': self.current_loadout.stat_profile.plan_type,
                'rarity': self.current_loadout.metadata.get('rarity'),
                'use_after_item': self.current_loadout.use_after_item,
                'produce_item_id': self.current_loadout.produce_item_id,
                'exam_score_bonus_multiplier': self.current_loadout.exam_score_bonus_multiplier,
                'vocal': self.current_loadout.stat_profile.vocal,
                'dance': self.current_loadout.stat_profile.dance,
                'visual': self.current_loadout.stat_profile.visual,
                'stamina': self.current_loadout.stat_profile.stamina,
                'starting_stamina': self.runtime.stamina,
                'starting_stamina_mode': self.exam_starting_stamina_mode,
                'fan_votes': self.runtime.fan_votes,
                'exam_status_enchant_ids': list(self.current_loadout.exam_status_enchant_ids),
            }
        if self.current_manual_setup is not None:
            info['manual_setup'] = {
                'label': self.current_manual_setup.record.label,
                'source_path': self.current_manual_setup.record.source_path,
                'line_number': self.current_manual_setup.record.line_number,
                'deck_size': len(self.current_manual_setup.deck_rows),
                'drink_count': len(self.current_manual_setup.drink_rows),
                'produce_item_ids': list(self.current_manual_setup.produce_item_ids),
            }
        return obs, info

    def _resolve_lesson_action_pool(
        self,
        requested_action_type: str,
        requested_action_pool: tuple[str, ...],
    ) -> tuple[str, ...]:
        """解析 lesson 模式允许采样的课程动作集合。"""

        if self.battle_kind != 'lesson':
            return ()
        if requested_action_type:
            if not _is_card_lesson_action_type(requested_action_type):
                raise ValueError(f'Unsupported lesson action type: {requested_action_type}')
            return (requested_action_type,)
        if requested_action_pool:
            resolved_pool = tuple(
                action_type
                for action_type in requested_action_pool
                if _is_card_lesson_action_type(action_type)
            )
            if not resolved_pool:
                raise ValueError('Lesson action pool is empty after filtering non-card lesson actions')
            return resolved_pool
        default_pool = tuple(
            action_type
            for action_type in self.scenario.action_types
            if _is_card_lesson_action_type(action_type) and not str(action_type).endswith('_hard')
        )
        if default_pool:
            return default_pool
        fallback_pool = tuple(action_type for action_type in self.scenario.action_types if _is_card_lesson_action_type(action_type))
        if fallback_pool:
            return fallback_pool
        raise ValueError(f'No card-play lesson actions available for scenario {self.scenario.scenario_id}')

    def _sample_episode_lesson_spec(
        self,
        rng: np.random.Generator,
        loadout: IdolLoadout | None,
    ) -> LessonTrainingSpec | None:
        """为 lesson 模式按主数据抽样当前 episode 的课程规格。"""

        if self.battle_kind != 'lesson':
            self.current_lesson_action_type = ''
            self.current_lesson_spec = None
            return None
        if loadout is None:
            raise ValueError('Lesson mode requires a resolved idol loadout')
        action_type = self.fixed_lesson_action_type
        if not action_type:
            action_type = str(rng.choice(self.lesson_action_pool))
        self.current_lesson_action_type = action_type
        self.current_lesson_spec = self.repository.resolve_lesson_training_spec(
            self.scenario,
            action_type=action_type,
            loadout=loadout,
            level_index=self.lesson_level_index or None,
            rng=rng,
        )
        return self.current_lesson_spec

    def _build_runtime(
        self,
        *,
        runtime_seed: int | None,
        rng: np.random.Generator,
        loadout: IdolLoadout | None,
        stage_type: str,
        deck: list[dict[str, Any]] | None = None,
        drinks: list[dict[str, Any]] | None = None,
        starting_stamina: float | None = None,
    ) -> ExamRuntime:
        """按当前 env mode 构建实际运行的 battle runtime。"""

        lesson_spec = self._sample_episode_lesson_spec(rng, loadout)
        fan_votes = self._sample_episode_fan_votes() if self.exam_reward_mode != 'clear' else None
        audition_row_id = None
        battle_kind = self.battle_kind
        lesson_kwargs: dict[str, Any] = {}
        if battle_kind == 'lesson' and lesson_spec is not None:
            exam_setting = self.repository.load_table('ExamSetting').first('p_exam_setting-1') or {}
            lesson_kwargs = {
                'lesson_type': lesson_spec.lesson_type,
                'lesson_types': lesson_spec.lesson_trigger_types,
                'lesson_post_clear_types': lesson_spec.lesson_post_clear_types,
                'lesson_target_value': lesson_spec.clear_target,
                'lesson_perfect_value': lesson_spec.perfect_target,
                'lesson_perfect_recovery_per_turn': float(exam_setting.get('examTurnEndRecoveryStamina') or 0),
                'turn_limit': lesson_spec.turn_limit,
            }
            fan_votes = None
        else:
            audition_row_id = default_audition_row_selector(
                self.repository,
                self.scenario,
                stage_type=stage_type,
                loadout=loadout,
                fan_votes=fan_votes,
            )
        return ExamRuntime(
            self.repository,
            self.scenario,
            stage_type=stage_type,
            seed=runtime_seed,
            deck=deck,
            drinks=drinks,
            loadout=loadout,
            starting_stamina=starting_stamina,
            exam_score_bonus_multiplier=loadout.exam_score_bonus_multiplier if loadout is not None else None,
            fan_votes=fan_votes,
            audition_row_id=audition_row_id,
            reward_mode=self.exam_reward_mode,
            reward_config=self.reward_config,
            battle_kind=battle_kind,
            **lesson_kwargs,
        )

    def _episode_max_stamina(self) -> float:
        """返回当前 episode 的考试最大体力。"""

        default_stamina = 12.0 if self.scenario.route_type == 'first_star' else 15.0
        if self.current_loadout is None:
            return default_stamina
        loadout_stamina = float(self.current_loadout.stat_profile.stamina or 0.0)
        return loadout_stamina if loadout_stamina > 0 else default_stamina

    def _sample_episode_starting_stamina(self, rng: np.random.Generator) -> float | None:
        """为 exam-only 环境采样开场体力。"""

        if self.exam_starting_stamina_mode != 'random':
            return None
        low = max(self.exam_starting_stamina_min_ratio, 0.0)
        high = max(self.exam_starting_stamina_max_ratio, 0.0)
        if low > high:
            low, high = high, low
        ratio = low if np.isclose(low, high) else float(rng.uniform(low, high))
        return self._episode_max_stamina() * ratio

    def _sample_episode_fan_votes(self) -> float | None:
        """为 exam-only 环境提供 NIA fan vote 上下文。"""

        if self.battle_kind != 'exam':
            return None
        if self.scenario.route_type != 'nia':
            return None
        if self.base_fan_votes is None:
            return None
        return max(float(self.base_fan_votes), 0.0)

    def _resolve_loadout_character_ids(self) -> tuple[str, ...]:
        """解析当前训练池中的角色列表，用于全局观测编码。"""

        idol_cards = self.repository.load_table('IdolCard')
        character_ids = {
            str(row.get('characterId') or '')
            for idol_card_id in self.idol_card_pool
            for row in [idol_cards.first(idol_card_id) or {}]
            if str(row.get('characterId') or '')
        }
        if self.idol_loadout is not None and self.idol_loadout.stat_profile.character_id:
            character_ids.add(self.idol_loadout.stat_profile.character_id)
        return tuple(sorted(character_ids))

    def _resolve_stage_type_ids(self) -> tuple[str, ...]:
        """汇总当前场景可能出现的 stage_type，用于显式编码考试舞台。"""

        seen: set[str] = set()
        stage_types: list[str] = []
        for raw_value in (self.stage_type, self.scenario.default_stage, *(self.scenario.audition_sequence or ())):
            stage_type = str(raw_value or '').strip()
            if not stage_type or stage_type in seen:
                continue
            seen.add(stage_type)
            stage_types.append(stage_type)
        return tuple(stage_types)

    def _resolve_loadout_rarities(self) -> tuple[str, ...]:
        """解析当前训练池中的偶像稀有度列表。"""

        idol_cards = self.repository.load_table('IdolCard')
        rarities = {
            str(row.get('rarity') or '')
            for idol_card_id in self.idol_card_pool
            for row in [idol_cards.first(idol_card_id) or {}]
            if str(row.get('rarity') or '')
        }
        if self.idol_loadout is not None:
            rarity = str(self.idol_loadout.metadata.get('rarity') or '')
            if rarity:
                rarities.add(rarity)
        return tuple(sorted(rarities))

    def _refresh_episode_randomization_flags(self) -> None:
        """根据当前配置刷新 episode 级随机化开关。"""

        self.randomize_stage_type = bool(
            self.battle_kind == 'exam'
            and self.episode_randomization.randomize_stage_type
            and self.stage_type is None
        )

    def update_episode_randomization(
        self,
        *,
        enabled: bool | None = None,
        randomize_stage_type: bool | None = None,
        randomize_use_after_item: bool | None = None,
    ) -> None:
        """在不中断训练对象的情况下调整后续 episode 的随机化开关。"""

        self.episode_randomization = replace(
            self.episode_randomization,
            enabled=self.episode_randomization.enabled if enabled is None else bool(enabled),
            randomize_stage_type=(
                self.episode_randomization.randomize_stage_type
                if randomize_stage_type is None
                else bool(randomize_stage_type)
            ),
            randomize_use_after_item=(
                self.episode_randomization.randomize_use_after_item
                if randomize_use_after_item is None
                else bool(randomize_use_after_item)
            ),
        )
        self._refresh_episode_randomization_flags()

    def _sample_episode_loadout(self, rng: np.random.Generator) -> IdolLoadout | None:
        """在每次考试 reset 时抽样一套整局固定的局外上下文。"""

        sampled_idol_card_id = str(self.base_loadout_config.get('idol_card_id') or '')
        if not sampled_idol_card_id and self.idol_card_pool:
            sampled_idol_card_id = str(rng.choice(self.idol_card_pool))
        loadout = self._resolve_loadout_for_idol_card(sampled_idol_card_id, rng)
        return self._apply_episode_randomization(loadout, rng)

    def _resolve_loadout_for_idol_card(
        self,
        idol_card_id: str,
        rng: np.random.Generator,
    ) -> IdolLoadout | None:
        """Resolve one base loadout, respecting env-level fixed params and cache."""

        loadout = self.idol_loadout
        config = self.base_loadout_config
        sampled_idol_card_id = str(idol_card_id or '')
        if sampled_idol_card_id:
            sampled_use_after_item = config.get('use_after_item')
            if self.episode_randomization.randomize_use_after_item:
                sampled_use_after_item = bool(rng.integers(0, 2))
            cache_key = (
                sampled_idol_card_id,
                int(config.get('producer_level') or 35),
                int(config.get('idol_rank') or 0),
                int(config.get('dearness_level') or 0),
                sampled_use_after_item,
                config.get('exam_score_bonus_multiplier'),
            )
            loadout = self._idol_loadout_cache.get(cache_key)
            if loadout is None:
                loadout = build_idol_loadout(
                    self.repository,
                    self.scenario,
                    idol_card_id=sampled_idol_card_id,
                    producer_level=int(config.get('producer_level') or 35),
                    idol_rank=int(config.get('idol_rank') or 0),
                    dearness_level=int(config.get('dearness_level') or 0),
                    use_after_item=sampled_use_after_item,
                    exam_score_bonus_multiplier=config.get('exam_score_bonus_multiplier'),
                )
                self._idol_loadout_cache[cache_key] = loadout
        return loadout

    def _apply_episode_randomization(
        self,
        loadout: IdolLoadout | None,
        rng: np.random.Generator,
    ) -> IdolLoadout | None:
        """Apply per-episode stat / multiplier jitter to a resolved base loadout."""

        if loadout is None or not self.episode_randomization.enabled:
            return loadout

        profile = loadout.stat_profile
        base_stats = np.array([profile.vocal, profile.dance, profile.visual], dtype=np.float32)
        focus = np.array(self.scenario.score_weights, dtype=np.float32)
        focus_sum = float(focus.sum())
        focus = focus / focus_sum if focus_sum > 0 else np.full(3, 1.0 / 3.0, dtype=np.float32)
        stat_jitter = max(float(self.episode_randomization.stat_jitter_ratio), 0.0)
        score_jitter = max(float(self.episode_randomization.score_bonus_jitter_ratio), 0.0)
        stat_scales = rng.uniform(1.0 - stat_jitter, 1.0 + stat_jitter, size=3).astype(np.float32)
        stat_scales += focus * (stat_jitter * 0.25)
        stat_scales = np.clip(stat_scales, 0.35, None)
        sampled_stats = np.maximum(base_stats * stat_scales, 1.0)
        parameter_limit = float(getattr(self.scenario, 'parameter_growth_limit', 0.0) or 0.0)
        if parameter_limit > 0:
            sampled_stats = np.minimum(sampled_stats, parameter_limit)
        base_focus_score = float(np.dot(base_stats, focus))
        sampled_focus_score = float(np.dot(sampled_stats, focus))
        focus_ratio = sampled_focus_score / max(base_focus_score, 1e-6)
        score_scale = focus_ratio * float(rng.uniform(1.0 - score_jitter, 1.0 + score_jitter))
        sampled_profile = replace(
            profile,
            vocal=float(sampled_stats[0]),
            dance=float(sampled_stats[1]),
            visual=float(sampled_stats[2]),
        )
        return replace(
            loadout,
            stat_profile=sampled_profile,
            exam_score_bonus_multiplier=max(loadout.exam_score_bonus_multiplier * score_scale, 0.25),
        )

    def _resolve_manual_episode_selection(
        self,
        rng: np.random.Generator,
    ) -> ManualEpisodeSelection | None:
        """Pick one manual setup record and align loadout / stage with it."""

        if self.manual_setup_dataset is None:
            return None
        fixed_idol_card_id = str(self.base_loadout_config.get('idol_card_id') or '')
        record = self.manual_setup_dataset.sample(
            rng,
            scenario_tokens=(self.base_loadout_config.get('scenario'), self.scenario.scenario_id),
            fixed_idol_card_id=fixed_idol_card_id,
        )
        resolved = self.manual_setup_dataset.resolve(self.repository, record)
        resolved_loadout = self.current_loadout
        if record.idol_card_id:
            resolved_loadout = self._resolve_loadout_for_idol_card(record.idol_card_id, rng)
            resolved_loadout = self._apply_episode_randomization(resolved_loadout, rng)
        if resolved_loadout is not None and resolved.produce_item_ids:
            resolved_loadout = augment_loadout_with_produce_items(
                self.repository,
                resolved_loadout,
                resolved.produce_item_ids,
                replace_existing=True,
            )
        stage_type = str(record.stage_type or self.current_stage_type)
        return ManualEpisodeSelection(
            setup=resolved,
            loadout=resolved_loadout,
            stage_type=stage_type,
        )

    def _stance_one_hot(self) -> np.ndarray:
        """把当前 stance 编码成 one-hot。"""

        mapping = {
            'neutral': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'concentration': np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            'full_power': np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            'preservation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        return mapping.get(self.runtime.stance, mapping['neutral'])

    def _remaining_gimmick_features(self) -> tuple[float, float]:
        """提取剩余 gimmick 数量和最近触发距离。"""

        future_turns = sorted(
            int(row.get('startTurn') or 0)
            for row in self.runtime.gimmick_rows
            if int(row.get('startTurn') or 0) > self.runtime.turn
        )
        if not future_turns:
            return 0.0, 0.0
        next_delta = max(future_turns[0] - self.runtime.turn, 0)
        return len(future_turns) / 12.0, next_delta / max(self.runtime.max_turns, 1)

    def _loadout_context_feature(self) -> np.ndarray:
        """把当前偶像 loadout 编码进全局观测，支持 all-idol 混训。"""

        if self.current_loadout is None:
            return np.zeros(self.loadout_context_dim, dtype=np.float32)

        profile = self.current_loadout.stat_profile
        rarity = str(self.current_loadout.metadata.get('rarity') or '')
        plan_one_hot = np.array([1.0 if profile.plan_type == plan_type else 0.0 for plan_type in _LOADOUT_PLAN_TYPES], dtype=np.float32)
        character_one_hot = np.array([1.0 if profile.character_id == character_id else 0.0 for character_id in self.loadout_character_ids], dtype=np.float32)
        rarity_one_hot = np.array([1.0 if rarity == value else 0.0 for value in self.loadout_rarities], dtype=np.float32)
        focus_effect_one_hot = np.array(
            [1.0 if profile.exam_effect_type == effect_type else 0.0 for effect_type in self.scenario.focus_effect_types],
            dtype=np.float32,
        )
        numeric = np.array(
            [
                self.current_loadout.producer_level / 60.0,
                self.current_loadout.idol_rank / 20.0,
                self.current_loadout.dearness_level / 20.0,
                1.0 if self.current_loadout.use_after_item else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                plan_one_hot,
                character_one_hot,
                rarity_one_hot,
                focus_effect_one_hot,
                numeric,
            ]
        ).astype(np.float32)

    def _stage_type_feature(self) -> np.ndarray:
        """把当前考试舞台编码进全局观测，避免多 stage 混训时上下文缺失。"""

        feature = np.zeros(self.stage_context_dim, dtype=np.float32)
        try:
            index = self.stage_type_ids.index(self.current_stage_type)
        except ValueError:
            index = len(self.stage_type_ids)
        feature[index] = 1.0
        return feature

    def _global_observation(self) -> np.ndarray:
        """构造考试全局状态向量。"""

        target_score = float(self.runtime._target_score() or 1.0)
        score_gap = target_score - self.runtime.score
        remaining_drinks = sum(1 for drink in self.runtime.drinks if not drink.get('_consumed'))
        hand_overflow = max(len(self.runtime.hand) - self.max_hand_cards, 0)
        remaining_gimmicks, next_gimmick_delta = self._remaining_gimmick_features()
        stance = self._stance_one_hot()
        base_core = np.array(
            [
                self.runtime.turn / max(self.runtime.max_turns, 1),
                max(self.runtime.max_turns - self.runtime.turn + 1, 0) / max(self.runtime.max_turns, 1),
                _bounded_positive(self.runtime.score, max(target_score, 1.0)),
                _bounded_signed(score_gap, max(target_score, 1.0)),
                self.runtime.stamina / max(self.runtime.max_stamina, 1.0),
                len(self.runtime.hand) / max(self.max_hand_cards, 1),
                _bounded_positive(hand_overflow, 8.0),
                len(self.runtime.deck) / 40.0,
                len(self.runtime.grave) / 40.0,
                len(self.runtime.hold) / 20.0,
                len(self.runtime.lost) / 20.0,
                remaining_drinks / max(self.max_drinks, 1),
                _bounded_positive(self.runtime.resources['block'], 30.0),
                _bounded_positive(self.runtime.resources['review'], 30.0),
                _bounded_positive(self.runtime.resources['aggressive'], 30.0),
                _bounded_positive(self.runtime.resources['concentration'], 30.0),
                _bounded_positive(self.runtime.resources['full_power_point'], 30.0),
                _bounded_positive(self.runtime.resources['parameter_buff'], 30.0),
                _bounded_positive(self.runtime.resources['lesson_buff'], 30.0),
                _bounded_positive(self.runtime.resources['preservation'], 30.0),
                _bounded_positive(self.runtime.resources['over_preservation'], 30.0),
                _bounded_positive(self.runtime.resources['enthusiastic'], 30.0),
                _bounded_positive(self.runtime.resources['sleepy'], 10.0),
                _bounded_positive(self.runtime.resources['panic'], 10.0),
                _bounded_positive(self.runtime.resources['stamina_consumption_down'], 10.0),
                _bounded_positive(self.runtime.turn_counters['play_count'], 10.0),
                _bounded_positive(self.runtime.play_limit, 10.0),
                stance[0],
                stance[1],
                stance[2],
                stance[3],
                self.runtime.profile['vocal_weight'],
                self.runtime.profile['dance_weight'],
                self.runtime.profile['visual_weight'],
                _bounded_positive(self.runtime.extra_turns, 5.0),
                _bounded_positive(len(self.runtime.active_effects), 20.0),
                _bounded_positive(len(self.runtime.active_enchants), 20.0),
                _bounded_positive(self.runtime.score_bonus_multiplier, 4.0),
                _bounded_positive(self.runtime.parameter_stats[0], 1200.0),
                _bounded_positive(self.runtime.parameter_stats[1], 1200.0),
                _bounded_positive(self.runtime.parameter_stats[2], 1200.0),
                _bounded_positive(self.runtime.fan_votes, max(self.runtime._fan_vote_reference(), 1.0)),
                _bounded_positive(self.runtime._reported_fan_vote_requirement(), max(self.runtime._fan_vote_reference(), 1.0)),
                remaining_gimmicks,
                next_gimmick_delta,
            ],
            dtype=np.float32,
        )
        base = np.concatenate([base_core, self.runtime.turn_color_one_hot()]).astype(np.float32)
        parts = [base, self._stage_type_feature(), self._loadout_context_feature()]
        if self.include_deck_features:
            parts.append(self._deck_composition_feature())
        return np.concatenate(parts).astype(np.float32)

    def _deck_composition_feature(self) -> np.ndarray:
        """聚合牌库组成特征（不泄露抽牌顺序）。"""

        n = max(len(self.runtime.deck), 1)
        effect_type_counts = np.zeros(len(self.taxonomy.exam_effect_types), dtype=np.float32)
        cost_type_counts = np.zeros(len(self.taxonomy.card_cost_types), dtype=np.float32)
        category_counts = np.zeros(len(self.taxonomy.card_categories), dtype=np.float32)
        total_evaluation = 0.0
        total_stamina = 0.0
        for card in self.runtime.deck:
            for et in self.repository.card_exam_effect_types(card.base_card):
                idx = self.taxonomy.exam_effect_index.get(et)
                if idx is not None:
                    effect_type_counts[idx] += 1.0
            ct = str(card.base_card.get('costType') or '')
            ct_idx = self.taxonomy.cost_index.get(ct)
            if ct_idx is not None:
                cost_type_counts[ct_idx] += 1.0
            cat = str(card.base_card.get('category') or '')
            cat_idx = self.taxonomy.category_index.get(cat)
            if cat_idx is not None:
                category_counts[cat_idx] += 1.0
            total_evaluation += float(card.base_card.get('evaluation') or 0)
            total_stamina += float(card.base_card.get('stamina') or 0)
        effect_type_counts /= n
        cost_type_counts /= n
        category_counts /= n
        numeric = np.array([total_evaluation / n / 100.0, total_stamina / n / 10.0], dtype=np.float32)
        return np.concatenate([effect_type_counts, numeric, cost_type_counts, category_counts]).astype(np.float32)

    def _score_gap_ratio(self) -> float:
        """返回平滑归一化后的相对目标分数缺口。"""

        target_score = float(self.runtime._target_score() or 1.0)
        return _bounded_signed(target_score - self.runtime.score, max(target_score, 1.0))

    def _transient_card_effect_types(self, card: RuntimeCard) -> list[str]:
        """收集运行时卡牌的动态附加考试效果类型。"""

        effect_types: list[str] = []
        for effect_id in card.transient_effect_ids:
            effect_row = self.repository.exam_effect_map.get(str(effect_id))
            if effect_row and effect_row.get('effectType'):
                effect_types.append(str(effect_row['effectType']))
        return effect_types

    def _transient_card_trigger_phases(self, card: RuntimeCard) -> list[str]:
        """收集运行时卡牌的动态附加 phase。"""

        phases: list[str] = []
        for trigger_id in card.transient_trigger_ids:
            trigger = self.repository.exam_trigger_map.get(str(trigger_id))
            if trigger:
                phases.extend(str(value) for value in trigger.get('phaseTypes', []) if value)
        return phases

    def _card_static_prefix(self, card: RuntimeCard) -> np.ndarray:
        """返回仅依赖卡面静态信息的编码前缀。"""

        cache_key = str(card.card_id)
        cached = self._card_static_prefix_cache.get(cache_key)
        if cached is None:
            cached = np.concatenate(
                [
                    self._card_action_vector,
                    self.taxonomy.encode_exam_effects(self.repository.card_exam_effect_types(card.base_card)),
                    self.taxonomy.encode_trigger_phases(self.repository.card_trigger_phases(card.base_card)),
                    self.taxonomy.encode_categories([str(card.base_card.get('category') or '')]),
                    self.taxonomy.encode_rarities([str(card.base_card.get('rarity') or '')]),
                    self.taxonomy.encode_cost_types([str(card.base_card.get('costType') or '')]),
                ]
            ).astype(np.float32)
            self._card_static_prefix_cache[cache_key] = cached
        return cached

    def _drink_static_prefix(self, drink: dict[str, Any]) -> np.ndarray:
        """返回仅依赖饮料静态信息的编码前缀。"""

        cache_key = str(drink.get('id') or '')
        cached = self._drink_static_prefix_cache.get(cache_key)
        if cached is None:
            cached = np.concatenate(
                [
                    self._drink_action_vector,
                    self.taxonomy.encode_exam_effects(self.repository.drink_exam_effect_types(drink)),
                    self._zero_trigger_phase_vector,
                    self._zero_category_vector,
                    self.taxonomy.encode_rarities([str(drink.get('rarity') or '')]),
                    self._zero_cost_type_vector,
                ]
            ).astype(np.float32)
            self._drink_static_prefix_cache[cache_key] = cached
        return cached

    def _candidate_shared_metrics(self) -> tuple[float, float, float, float]:
        """收集同一帧内所有动作共享的数值特征，避免重复计算。"""

        return (
            self._score_gap_ratio(),
            _bounded_positive(self.runtime.resources['review'], 20.0),
            _bounded_positive(self.runtime.resources['block'], 20.0),
            _bounded_positive(self.runtime.resources['aggressive'], 20.0),
        )

    def _card_feature(
        self,
        card: RuntimeCard,
        slot_index: int,
        playable: bool,
        shared_metrics: tuple[float, float, float, float],
    ) -> np.ndarray:
        """把手牌槽位编码成模型输入特征。"""

        prefix = self._card_static_prefix(card).copy()
        effect_types = self._transient_card_effect_types(card)
        if effect_types:
            prefix[self._exam_effect_slice] += self.taxonomy.encode_exam_effects(effect_types)
        trigger_phases = self._transient_card_trigger_phases(card)
        if trigger_phases:
            prefix[self._trigger_phase_slice] += self.taxonomy.encode_trigger_phases(trigger_phases)
        resource_cost = sum(self.runtime._card_resource_costs(card).values())
        customized = 1.0 if (card.grow_effect_ids or card.transient_effect_ids or card.transient_trigger_ids or card.card_status_enchant_id) else 0.0
        score_gap_ratio, review_resource, block_resource, aggressive_resource = shared_metrics
        numeric = np.array(
            [
                _bounded_positive(float(card.base_card.get('stamina') or 0), 10.0),
                _bounded_positive(float(card.base_card.get('forceStamina') or 0), 10.0),
                _bounded_positive(resource_cost, 10.0),
                _bounded_positive(float(card.base_card.get('evaluation') or 0), 100.0),
                1.0 if playable else 0.0,
                _bounded_positive(card.upgrade_count, 4.0),
                _bounded_positive(card.play_count_bonus, 4.0),
                _bounded_positive(len(card.grow_effect_ids), 4.0),
                customized,
                score_gap_ratio,
                review_resource,
                block_resource,
                aggressive_resource,
                (slot_index + 1) / max(self.max_hand_cards, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([prefix, numeric]).astype(np.float32)

    def _drink_feature(
        self,
        drink: dict[str, Any],
        slot_index: int,
        shared_metrics: tuple[float, float, float, float],
    ) -> np.ndarray:
        """把饮料槽位编码成模型输入特征。"""

        score_gap_ratio, review_resource, block_resource, aggressive_resource = shared_metrics
        numeric = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 if not drink.get('_consumed') else 0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                score_gap_ratio,
                review_resource,
                block_resource,
                aggressive_resource,
                (slot_index + 1) / max(self.max_drinks, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([self._drink_static_prefix(drink), numeric]).astype(np.float32)

    def _end_turn_feature(self, shared_metrics: tuple[float, float, float, float]) -> np.ndarray:
        """构造结束回合动作的特征向量。"""

        score_gap_ratio, review_resource, block_resource, aggressive_resource = shared_metrics
        numeric = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                score_gap_ratio,
                review_resource,
                block_resource,
                aggressive_resource,
                1.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([self._end_turn_prefix, numeric]).astype(np.float32)

    def _build_candidates(self) -> list[ActionView]:
        """把当前手牌、饮料和结束回合动作展开为固定动作槽。"""

        candidates: list[ActionView] = []
        candidate_mask: list[bool] = []
        shared_metrics = self._candidate_shared_metrics()

        for index in range(self.max_hand_cards):
            if index < len(self.runtime.hand):
                card = self.runtime.hand[index]
                available = self.runtime._can_play_card(card)
                candidates.append(
                    ActionView(
                        label=self.runtime._card_label(card),
                        kind='card',
                        feature=self._card_feature(card, index, available, shared_metrics),
                        payload={'kind': 'card', 'uid': card.uid, 'available': available},
                    )
                )
                candidate_mask.append(available)
            else:
                candidates.append(
                    ActionView(
                        label='empty',
                        kind='card',
                        feature=self._empty_card_feature,
                        payload={'kind': 'card', 'available': False},
                    )
                )
                candidate_mask.append(False)

        for index in range(self.max_drinks):
            if index < len(self.runtime.drinks):
                drink = self.runtime.drinks[index]
                available = self.runtime._can_use_drink(drink)
                label = self.repository.drink_name(drink) if not drink.get('_consumed') else f"{self.repository.drink_name(drink)}(used)"
                candidates.append(
                    ActionView(
                        label=label,
                        kind='drink',
                        feature=self._drink_feature(drink, index, shared_metrics),
                        payload={'kind': 'drink', 'index': index, 'available': available},
                    )
                )
                candidate_mask.append(available)
            else:
                candidates.append(
                    ActionView(
                        label='no_drink',
                        kind='drink',
                        feature=self._empty_drink_feature,
                        payload={'kind': 'drink', 'available': False},
                    )
                )
                candidate_mask.append(False)

        end_turn_label = 'SKIP' if self.battle_kind == 'lesson' else 'end_turn'
        candidates.append(
            ActionView(
                label=end_turn_label,
                kind='end_turn',
                feature=self._end_turn_feature(shared_metrics),
                payload={'kind': 'end_turn', 'available': True},
            )
        )
        candidate_mask.append(True)
        self._candidate_mask = np.asarray(candidate_mask, dtype=bool)
        return candidates

    def _build_observation(self) -> dict[str, np.ndarray]:
        """刷新考试观测。"""

        self._candidates = self._build_candidates()
        raw_obs = {
            'global': self._global_observation(),
            'action_features': np.stack([candidate.feature for candidate in self._candidates]).astype(np.float32),
            'action_mask': self._candidate_mask.astype(np.float32),
        }
        return self._sanitize_observation(raw_obs)

    def _sanitize_observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """统一清洗考试观测，避免数值异常传进策略网络。"""

        return {
            'global': _sanitize_float32_array(obs['global'], lower=-20.0, upper=20.0),
            'action_features': _sanitize_float32_array(obs['action_features'], lower=-20.0, upper=20.0),
            'action_mask': _sanitize_float32_array(obs['action_mask'], lower=0.0, upper=1.0),
        }

    def _has_numeric_instability(self) -> bool:
        """检查 runtime 是否已经出现 NaN / Inf。"""

        critical_values: list[float] = [
            float(self.runtime.score),
            float(self.runtime.stamina),
            float(self.runtime.max_stamina),
            float(self.runtime.turn),
            float(self.runtime.extra_turns),
            float(self.runtime.score_bonus_multiplier),
        ]
        critical_values.extend(float(value) for value in self.runtime.parameter_stats)
        critical_values.extend(float(value) for value in self.runtime.resources.values())
        return any(not np.isfinite(value) for value in critical_values)

    def _numeric_safeguard_penalty(self) -> float:
        """数值异常时统一返回一个有限负奖励。"""

        clip_value = float(getattr(self.reward_config, 'reward_clip', 0.0) or 0.0)
        return -max(clip_value, 10.0)

    def action_masks(self) -> np.ndarray:
        """为 MaskablePPO 提供当前动作掩码。"""

        if not self._candidates:
            self._candidates = self._build_candidates()
        return self._candidate_mask

    def step(self, action: int):
        """执行一个考试动作槽。"""

        candidate = self._candidates[action]
        if not candidate.payload.get('available', False):
            obs = self._build_observation()
            info = {
                'scenario': self.scenario.scenario_id,
                'stage_type': self.current_stage_type,
                'invalid_action': True,
            }
            if self.include_action_labels_in_step_info:
                info['action_labels'] = [item.label for item in self._candidates]
            return obs, self.reward_config.invalid_action_penalty, False, False, info

        if candidate.kind == 'card':
            runtime_action = ExamActionCandidate(
                label=candidate.label,
                kind='card',
                payload={'kind': 'card', 'uid': int(candidate.payload['uid'])},
            )
        elif candidate.kind == 'drink':
            runtime_action = ExamActionCandidate(
                label=candidate.label,
                kind='drink',
                payload={'kind': 'drink', 'index': int(candidate.payload['index'])},
            )
        else:
            runtime_action = ExamActionCandidate(
                label='结束回合',
                kind='end_turn',
                payload={'kind': 'end_turn'},
            )

        reward, runtime_info = self.runtime.step(runtime_action)
        numeric_instability = self._has_numeric_instability()
        if not np.isfinite(float(reward)) or numeric_instability:
            raw_reward = float(reward)
            reward = self._numeric_safeguard_penalty()
            self.runtime.terminated = True
            runtime_info = dict(runtime_info)
            runtime_info['numeric_safeguard_triggered'] = True
            runtime_info['numeric_reward_raw'] = raw_reward
            runtime_info['numeric_state_unstable'] = bool(numeric_instability)
        obs = self._build_observation()
        info = {
            'scenario': self.scenario.scenario_id,
            'stage_type': self.current_stage_type,
            'hand_overflow': max(len(self.runtime.hand) - self.max_hand_cards, 0),
        }
        if self.include_action_labels_in_step_info:
            info['action_labels'] = [item.label for item in self._candidates]
        info.update(runtime_info)
        return obs, float(reward), bool(self.runtime.terminated), False, info
