"""无状态服务层，统一 CLI、API 与训练后端的装配逻辑。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .data import MasterDataRepository
from .deck_constraints import normalize_forced_card_groups, normalize_guaranteed_effect_counts
from .exam_runtime import ExamRuntime, default_audition_row_selector
from .idol_config import (
    build_idol_loadout,
    build_initial_exam_deck,
    build_weighted_card_pool,
    list_available_produce_card_conversions,
    list_trainable_idol_card_ids,
)
from .loadout import ExamEpisodeRandomizationConfig, IdolLoadout
from .manual_exam_setups import load_manual_exam_setup_dataset
from .produce_runtime import ProduceRuntime
from .reward_config import RewardConfig, build_reward_config

SCENARIO_ALIASES = {
    'first_star_regular': 'produce-001',
    'first_star_pro': 'produce-002',
    'first_star_master': 'produce-003',
    'first_star_legend': 'produce-006',
    'nia_pro': 'produce-004',
    'nia_master': 'produce-005',
}


@dataclass(frozen=True)
class LoadoutConfig:
    """无状态服务输入用的偶像配置。"""

    idol_card_id: str = ''
    producer_level: int = 35
    idol_rank: int = 0
    dearness_level: int = 0
    use_after_item: bool | None = None
    produce_card_conversion_after_ids: tuple[str, ...] = ()
    exam_score_bonus_multiplier: float | None = None
    fan_votes: float | None = None
    exam_randomize_context: bool = False
    exam_stat_jitter_ratio: float = 0.10
    exam_score_bonus_jitter_ratio: float = 0.05
    exam_randomize_use_after_item: bool = False
    exam_randomize_stage_type: bool = False


@lru_cache(maxsize=1)
def get_repository() -> MasterDataRepository:
    """缓存主数据仓库，避免重复加载 YAML。"""

    return MasterDataRepository()


@lru_cache(maxsize=32)
def get_scenario(scenario_name: str):
    """按别名解析并缓存场景定义。"""

    repository = get_repository()
    scenario_id = SCENARIO_ALIASES.get(scenario_name, scenario_name)
    return repository.build_scenario(scenario_id)


@lru_cache(maxsize=256)
def resolve_loadout(
    scenario_name: str,
    idol_card_id: str,
    producer_level: int = 35,
    idol_rank: int = 0,
    dearness_level: int = 0,
    use_after_item: bool | None = None,
    produce_card_conversion_after_ids: tuple[str, ...] = (),
    exam_score_bonus_multiplier: float | None = None,
) -> IdolLoadout | None:
    """按场景和偶像参数解析 loadout，并做 LRU 缓存。"""

    if not idol_card_id:
        return None
    repository = get_repository()
    scenario = get_scenario(scenario_name)
    return build_idol_loadout(
        repository,
        scenario,
        idol_card_id=idol_card_id,
        producer_level=producer_level,
        idol_rank=idol_rank,
        dearness_level=dearness_level,
        use_after_item=use_after_item,
        selected_produce_card_conversion_after_ids=produce_card_conversion_after_ids,
        exam_score_bonus_multiplier=exam_score_bonus_multiplier,
    )


@lru_cache(maxsize=64)
def resolve_idol_card_pool(scenario_name: str) -> tuple[str, ...]:
    """返回场景下默认用于通用训练的偶像卡池。"""

    repository = get_repository()
    scenario = get_scenario(scenario_name)
    return list_trainable_idol_card_ids(repository, scenario)



def build_loadout_from_config(scenario_name: str, config: LoadoutConfig | None) -> IdolLoadout | None:
    """把可选配置对象转换为内部 loadout。"""

    if config is None or not config.idol_card_id:
        return None
    return resolve_loadout(
        scenario_name,
        config.idol_card_id,
        producer_level=int(config.producer_level),
        idol_rank=int(config.idol_rank),
        dearness_level=int(config.dearness_level),
        use_after_item=config.use_after_item,
        produce_card_conversion_after_ids=tuple(config.produce_card_conversion_after_ids),
        exam_score_bonus_multiplier=config.exam_score_bonus_multiplier,
    )


def build_env_from_config(config: dict[str, Any]):
    """根据配置创建 planning / exam / lesson 环境实例。"""

    repository = get_repository()
    scenario_name = str(config.get('scenario') or 'nia_master')
    scenario = get_scenario(scenario_name)
    requested_idol_card_id = str(config.get('idol_card_id') or '')
    if requested_idol_card_id.lower() == 'all':
        requested_idol_card_id = ''
    requested_idol_card_ids = tuple(
        str(value)
        for value in (config.get('idol_card_ids') or [])
        if str(value or '')
    )
    manual_setup_paths = tuple(
        str(value)
        for value in (
            config.get('manual_exam_setup_paths')
            or config.get('manual_exam_setup_data')
            or []
        )
        if str(value or '')
    )
    manual_setup_dataset = load_manual_exam_setup_dataset(manual_setup_paths)
    initial_deck_guaranteed_effect_counts = normalize_guaranteed_effect_counts(
        config.get('initial_deck_guaranteed_effect_counts')
        if config.get('initial_deck_guaranteed_effect_counts') is not None
        else config.get('guarantee_card_effects')
    )
    initial_deck_forced_card_groups = normalize_forced_card_groups(
        config.get('initial_deck_forced_card_groups')
        if config.get('initial_deck_forced_card_groups') is not None
        else (
            config.get('force_card_groups')
            if config.get('force_card_groups') is not None
            else config.get('force_card_ids')
        )
    )
    loadout_config = LoadoutConfig(
        idol_card_id=requested_idol_card_id,
        producer_level=int(config.get('producer_level') or 35),
        idol_rank=int(config.get('idol_rank') or 0),
        dearness_level=int(config.get('dearness_level') or 0),
        use_after_item=config.get('use_after_item'),
        produce_card_conversion_after_ids=tuple(
            str(value)
            for value in (config.get('produce_card_conversion_after_ids') or [])
            if str(value or '')
        ),
        exam_score_bonus_multiplier=config.get('exam_score_bonus_multiplier'),
        fan_votes=float(config.get('fan_votes')) if config.get('fan_votes') is not None else None,
        exam_randomize_context=bool(config.get('exam_randomize_context') or False),
        exam_stat_jitter_ratio=float(config.get('exam_stat_jitter_ratio') or 0.10),
        exam_score_bonus_jitter_ratio=float(config.get('exam_score_bonus_jitter_ratio') or 0.05),
        exam_randomize_use_after_item=bool(config.get('exam_randomize_use_after_item') or False),
        exam_randomize_stage_type=bool(config.get('exam_randomize_stage_type') or False),
    )
    loadout = build_loadout_from_config(scenario_name, loadout_config)
    idol_card_pool = requested_idol_card_ids or ((loadout_config.idol_card_id,) if loadout_config.idol_card_id else resolve_idol_card_pool(scenario_name))
    mode = str(config.get('mode') or 'exam')
    seed = config.get('seed')
    include_action_labels_in_step_info = bool(config.get('include_action_labels_in_step_info') or False)
    if mode == 'planning':
        from .envs import GakumasPlanningEnv

        return GakumasPlanningEnv(
            repository,
            scenario,
            seed=seed,
            idol_loadout=loadout,
            include_action_labels_in_step_info=include_action_labels_in_step_info,
        )
    from .envs import GakumasExamEnv

    battle_kind = 'lesson' if mode == 'lesson' else 'exam'
    effective_reward_mode = 'clear' if battle_kind == 'lesson' else str(config.get('exam_reward_mode') or 'score')
    reward_config = build_reward_config(
        reward_mode=effective_reward_mode,
        reward_config_path=config.get('reward_config_path'),
        overrides=config.get('reward_overrides'),
    )
    env = GakumasExamEnv(
        repository,
        scenario,
        battle_kind=battle_kind,
        stage_type=config.get('stage_type'),
        seed=seed,
        idol_loadout=loadout,
        base_loadout_config={
            'scenario': scenario_name,
            'idol_card_id': loadout_config.idol_card_id,
            'idol_card_ids': idol_card_pool,
            'producer_level': loadout_config.producer_level,
            'idol_rank': loadout_config.idol_rank,
            'dearness_level': loadout_config.dearness_level,
            'use_after_item': loadout_config.use_after_item,
            'produce_card_conversion_after_ids': loadout_config.produce_card_conversion_after_ids,
            'exam_score_bonus_multiplier': loadout_config.exam_score_bonus_multiplier,
            'fan_votes': loadout_config.fan_votes,
        },
        episode_randomization=ExamEpisodeRandomizationConfig(
            enabled=loadout_config.exam_randomize_context,
            stat_jitter_ratio=loadout_config.exam_stat_jitter_ratio,
            score_bonus_jitter_ratio=loadout_config.exam_score_bonus_jitter_ratio,
            randomize_use_after_item=loadout_config.exam_randomize_use_after_item,
            randomize_stage_type=loadout_config.exam_randomize_stage_type,
        ),
        exam_reward_mode=effective_reward_mode,
        exam_starting_stamina_mode=str(config.get('exam_starting_stamina_mode') or 'full'),
        exam_starting_stamina_min_ratio=float(config.get('exam_starting_stamina_min_ratio') or 0.6),
        exam_starting_stamina_max_ratio=float(config.get('exam_starting_stamina_max_ratio') or 1.0),
        lesson_action_type=config.get('lesson_action_type'),
        lesson_action_types=tuple(
            str(value)
            for value in (config.get('lesson_action_types') or [])
            if str(value or '')
        ),
        lesson_level_index=int(config.get('lesson_level_index') or 0),
        include_action_labels_in_step_info=include_action_labels_in_step_info,
        include_deck_features=bool(config.get('include_deck_features') or False),
        manual_setup_dataset=manual_setup_dataset,
        initial_deck_guaranteed_effect_counts=initial_deck_guaranteed_effect_counts,
        initial_deck_forced_card_groups=initial_deck_forced_card_groups,
        reward_config=reward_config,
    )

    if config.get('llm_reward_enabled'):
        from .llm_reward import LLMRewardConfig, LLMRewardShaper

        llm_config = LLMRewardConfig(
            enabled=True,
            model=config.get('llm_reward_model', 'qwen3:4b'),
            base_url=config.get('llm_reward_base_url', 'http://localhost:11434'),
            reward_weight=float(config.get('llm_reward_weight', 0.3)),
            eval_interval=int(config.get('llm_reward_eval_interval', 1)),
        )
        env = LLMRewardShaper(env, llm_config)

    return env

def loadout_summary(scenario_name: str, config: LoadoutConfig | None = None) -> dict[str, Any]:
    """返回 loadout 摘要、初始卡组和随机池预览。"""

    repository = get_repository()
    scenario = get_scenario(scenario_name)
    loadout = build_loadout_from_config(scenario_name, config)
    if loadout is None:
        idol_card_pool = resolve_idol_card_pool(scenario_name)
        return {
            'scenario': scenario.scenario_id,
            'loadout': None,
            'idol_card_pool_size': len(idol_card_pool),
            'idol_card_pool_preview': list(idol_card_pool[:16]),
        }
    rng = np.random.default_rng(0)
    initial_deck = build_initial_exam_deck(repository, scenario, loadout=loadout, rng=rng)
    weighted_pool = build_weighted_card_pool(
        repository,
        scenario,
        focus_effect_type=loadout.stat_profile.exam_effect_type,
        loadout=loadout,
    )
    return {
        'scenario': scenario.scenario_id,
        'loadout': {
            'idol_card_id': loadout.idol_card_id,
            'producer_level': loadout.producer_level,
            'idol_rank': loadout.idol_rank,
            'dearness_level': loadout.dearness_level,
            'use_after_item': loadout.use_after_item,
            'produce_item_id': loadout.produce_item_id,
            'produce_card_conversions': [asdict(spec) for spec in loadout.produce_card_conversions],
            'exam_status_enchant_ids': list(loadout.exam_status_enchant_ids),
            'exam_score_bonus_multiplier': loadout.exam_score_bonus_multiplier,
            'metadata': dict(loadout.metadata),
            'stat_profile': asdict(loadout.stat_profile),
            'deck_archetype': asdict(loadout.deck_archetype) if loadout.deck_archetype is not None else None,
            'produce_skills': [asdict(skill) for skill in loadout.produce_skills],
        },
        'available_produce_card_conversions': [
            asdict(spec) for spec in list_available_produce_card_conversions(repository, loadout.producer_level)
        ],
        'initial_deck': [
            {
                'id': str(card.get('id') or ''),
                'name': repository.card_name(card),
                'effect_types': repository.card_exam_effect_types(card),
            }
            for card in initial_deck
        ],
        'weighted_pool_preview': [
            {
                'id': str(card.get('id') or ''),
                'name': repository.card_name(card),
                'effect_types': repository.card_exam_effect_types(card),
            }
            for card in weighted_pool[:24]
        ],
    }


def _serialize_exam_actions(runtime: ExamRuntime) -> list[dict[str, Any]]:
    """把考试动作列表序列化成便于调试的结构。"""

    actions = []
    for index, action in enumerate(runtime.legal_actions()):
        available = action.kind == 'end_turn'
        if action.kind == 'card':
            uid = int(action.payload.get('uid', -1))
            card = next((item for item in runtime.hand if item.uid == uid), None)
            available = card is not None and runtime._can_play_card(card)  # type: ignore[attr-defined]
        elif action.kind == 'drink':
            drink_index = int(action.payload.get('index') or 0)
            available = (
                drink_index < len(runtime.drinks)
                and runtime._can_use_drink(runtime.drinks[drink_index])  # type: ignore[attr-defined]
            )
        actions.append(
            {
                'index': index,
                'label': action.label,
                'kind': action.kind,
                'payload': dict(action.payload),
                'available': available,
            }
        )
    return actions


def _serialize_exam_state(runtime: ExamRuntime) -> dict[str, Any]:
    """提取考试运行时的核心状态快照。"""

    return {
        'reward_mode': runtime.reward_mode,
        'battle_kind': runtime.battle_kind,
        'score': runtime.score,
        'stamina': runtime.stamina,
        'max_stamina': runtime.max_stamina,
        'turn': runtime.turn,
        'max_turns': runtime.max_turns,
        'terminated': runtime.terminated,
        'stance': runtime.stance,
        'turn_color': runtime.current_turn_color,
        'turn_color_label': runtime.turn_color_label(),
        'fan_votes': runtime.fan_votes,
        'fan_vote_baseline': runtime._reported_fan_vote_baseline(),
        'fan_vote_requirement': runtime._reported_fan_vote_requirement(),
        'clear_state': runtime.clear_state,
        'lesson_target_remaining': runtime._lesson_target_remaining(),
        'lesson_perfect_remaining': runtime._lesson_perfect_remaining(),
        'lesson_target_value': runtime._current_clear_target(),
        'lesson_perfect_value': runtime._current_perfect_target(),
        'play_limit': runtime.play_limit,
        'score_bonus_multiplier': runtime.score_bonus_multiplier,
        'parameter_stats': {
            'vocal': runtime.parameter_stats[0],
            'dance': runtime.parameter_stats[1],
            'visual': runtime.parameter_stats[2],
        },
        'profile': dict(runtime.profile),
        'target_score': runtime._target_score(),
        'resources': dict(runtime.resources),
        'zones': {
            'deck': len(runtime.deck),
            'hand': len(runtime.hand),
            'grave': len(runtime.grave),
            'hold': len(runtime.hold),
            'lost': len(runtime.lost),
        },
        'hand_cards': [runtime._card_label(card) for card in runtime.hand],
        'drinks': [
            {
                'name': runtime.repository.drink_name(drink),
                'consumed': bool(drink.get('_consumed')),
            }
            for drink in runtime.drinks
        ],
        'gimmicks': [
            {
                'start_turn': int(row.get('startTurn') or 0),
                'effect_id': str(row.get('produceExamEffectId') or ''),
            }
            for row in runtime.gimmick_rows
        ],
        'deck_cards': [
            {
                'uid': card.uid,
                'card_id': card.card_id,
                'name': runtime.repository.card_name(card.base_card),
                'upgrade_count': card.upgrade_count,
                'effect_types': runtime.repository.card_exam_effect_types(card.base_card),
                'category': str(card.base_card.get('category') or ''),
                'cost_type': str(card.base_card.get('costType') or ''),
                'stamina': float(card.base_card.get('stamina') or 0),
            }
            for card in runtime.deck
        ],
        'event_log': [
            {'turn': e.turn, 'event_type': e.event_type, 'detail': e.detail}
            for e in runtime.event_log
        ],
    }


def _serialize_planning_state(runtime: ProduceRuntime) -> dict[str, Any]:
    """提取培育运行时的核心状态快照。"""

    return {
        'state': dict(runtime.state),
        'deck': [runtime.repository.card_name(card) for card in runtime.deck[:20]],
        'drinks': [runtime.repository.drink_name(drink) for drink in runtime.drinks],
        'exam_status_enchant_ids': list(runtime.exam_status_enchant_ids),
    }


def _serialize_planning_actions(runtime: ProduceRuntime) -> list[dict[str, Any]]:
    """序列化当前周可执行的培育动作。"""

    return [
        {
            'index': index,
            'label': candidate.label,
            'action_type': candidate.action_type,
            'available': candidate.available,
            'success_probability': candidate.success_probability,
            'produce_card_id': candidate.produce_card_id,
            'effect_types': list(candidate.effect_types),
        }
        for index, candidate in enumerate(runtime.legal_actions())
    ]


def _choose_planning_action(runtime: ProduceRuntime) -> int | None:
    """用简单启发式挑选一个培育动作，便于自动跑。"""

    candidates = runtime.legal_actions()
    best_index = None
    best_score = -1e9
    for index, candidate in enumerate(candidates):
        if not candidate.available:
            continue
        score = candidate.success_probability
        score += candidate.produce_point_delta * 0.05
        score += len(candidate.effect_types) * 0.02
        score += len(candidate.exam_effect_types) * 0.01
        if candidate.action_type == 'refresh' and runtime.state['stamina'] < runtime.state['max_stamina'] * 0.35:
            score += 1.0
        if candidate.action_type.startswith('shop_buy_card_'):
            score += 0.3
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _choose_exam_action(runtime: ExamRuntime, scenario_name: str, loadout: IdolLoadout | None, seed: int | None = None):
    """复用培育运行时中的启发式出牌器，为考试自动跑提供动作。"""

    planner = ProduceRuntime(get_repository(), get_scenario(scenario_name), seed=seed, idol_loadout=loadout)
    return planner._choose_exam_action(runtime)


def simulate_planning(
    scenario_name: str,
    actions: list[int] | None = None,
    auto_policy: str | None = None,
    auto_steps: int = 0,
    seed: int | None = None,
    loadout_config: LoadoutConfig | None = None,
) -> dict[str, Any]:
    """执行一次无状态培育模拟，可混合手动动作与自动策略。"""

    repository = get_repository()
    scenario = get_scenario(scenario_name)
    loadout = build_loadout_from_config(scenario_name, loadout_config)
    runtime = ProduceRuntime(repository, scenario, seed=seed, idol_loadout=loadout)
    runtime.reset()
    history: list[dict[str, Any]] = []
    for action_index in actions or []:
        candidates = runtime.legal_actions()
        if action_index < 0 or action_index >= len(candidates):
            raise IndexError(f'Invalid planning action index: {action_index}')
        reward, terminated, info = runtime.step(action_index)
        history.append({'action_index': action_index, 'reward': reward, 'terminated': terminated, 'info': info})
        if terminated:
            break
    if auto_policy:
        for _ in range(max(auto_steps, 0)):
            if runtime.state['step'] >= runtime.state['max_steps']:
                break
            if auto_policy == 'first_valid':
                legal_actions = runtime.legal_actions()
                next_index = next((index for index, item in enumerate(legal_actions) if item.available), None)
            else:
                next_index = _choose_planning_action(runtime)
            if next_index is None:
                break
            reward, terminated, info = runtime.step(next_index)
            history.append({'action_index': next_index, 'reward': reward, 'terminated': terminated, 'info': info})
            if terminated:
                break
    return {
        'scenario': scenario.scenario_id,
        'loadout': loadout_summary(scenario_name, loadout_config).get('loadout'),
        'history': history,
        'state': _serialize_planning_state(runtime),
        'legal_actions': _serialize_planning_actions(runtime),
    }


def simulate_exam(
    scenario_name: str,
    stage_type: str | None = None,
    actions: list[int] | None = None,
    auto_policy: str | None = None,
    auto_steps: int = 0,
    seed: int | None = None,
    loadout_config: LoadoutConfig | None = None,
    focus_effect_type: str | None = None,
    exam_reward_mode: str = 'score',
) -> dict[str, Any]:
    """执行一次无状态考试模拟，可混合手动动作与自动策略。"""

    repository = get_repository()
    scenario = get_scenario(scenario_name)
    loadout = build_loadout_from_config(scenario_name, loadout_config)
    rng = np.random.default_rng(seed)
    deck = build_initial_exam_deck(repository, scenario, focus_effect_type=focus_effect_type, rng=rng, loadout=loadout)
    drinks = repository.build_drink_inventory(
        scenario,
        rng=rng,
        plan_type=loadout.stat_profile.plan_type if loadout is not None else None,
    )
    runtime = ExamRuntime(
        repository,
        scenario,
        stage_type=stage_type,
        seed=seed,
        deck=deck,
        drinks=drinks,
        loadout=loadout,
        exam_score_bonus_multiplier=loadout.exam_score_bonus_multiplier if loadout is not None else None,
        fan_votes=(
            float(loadout_config.fan_votes)
            if exam_reward_mode != 'clear' and loadout_config is not None and loadout_config.fan_votes is not None
            else None
        ),
        audition_row_id=default_audition_row_selector(
            repository,
            scenario,
            stage_type=stage_type,
            loadout=loadout,
            fan_votes=(
                float(loadout_config.fan_votes)
                if exam_reward_mode != 'clear' and loadout_config is not None and loadout_config.fan_votes is not None
                else None
            ),
        ),
        reward_mode=exam_reward_mode,
    )
    runtime.reset()
    history: list[dict[str, Any]] = []
    for action_index in actions or []:
        candidates = runtime.legal_actions()
        if action_index < 0 or action_index >= len(candidates):
            raise IndexError(f'Invalid exam action index: {action_index}')
        action = candidates[action_index]
        reward, info = runtime.step(action)
        history.append({'action_index': action_index, 'label': action.label, 'reward': reward, 'info': info})
        if runtime.terminated:
            break
    if auto_policy:
        for _ in range(max(auto_steps, 0)):
            if runtime.terminated:
                break
            if auto_policy == 'end_turn':
                action = runtime.legal_actions()[-1]
            else:
                action = _choose_exam_action(runtime, scenario_name, loadout, seed=seed)
            if action is None:
                break
            candidates = runtime.legal_actions()
            selected_action = action if hasattr(action, 'label') else candidates[-1]
            reward, info = runtime.step(selected_action)
            history.append({'action_index': None, 'label': selected_action.label, 'reward': reward, 'info': info})
            if runtime.terminated:
                break
    return {
        'scenario': scenario.scenario_id,
        'stage_type': runtime.stage_type,
        'loadout': loadout_summary(scenario_name, loadout_config).get('loadout'),
        'history': history,
        'state': _serialize_exam_state(runtime),
        'legal_actions': [
            {
                'index': index,
                'label': action.label,
                'kind': action.kind,
                'payload': dict(action.payload),
            }
            for index, action in enumerate(runtime.legal_actions())
        ],
    }


