"""考试策略回放工具，运行一局考试并输出 JSON trace 文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .data import RUNS_DIR
from .model import tensorize_observation
from .service import LoadoutConfig, build_env_from_config

_RESOURCE_KEYS = (
    'block',
    'review',
    'aggressive',
    'concentration',
    'full_power_point',
    'parameter_buff',
    'lesson_buff',
    'preservation',
    'over_preservation',
    'enthusiastic',
    'sleepy',
    'panic',
)

_PRIMARY_RESOURCE_KEYS = (
    'review',
    'block',
    'aggressive',
    'concentration',
    'parameter_buff',
    'lesson_buff',
    'full_power_point',
    'preservation',
    'enthusiastic',
    'sleepy',
    'panic',
)


class PolicyRunner:
    """统一封装不同后端的单步推理接口。"""

    backend_name = 'unknown'

    def predict(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> int:
        raise NotImplementedError

    def close(self) -> None:
        """释放后端资源。"""


class SB3PolicyRunner(PolicyRunner):
    """SB3 / sb3-contrib checkpoint 推理器。"""

    backend_name = 'sb3'

    def __init__(self, checkpoint_path: Path):
        last_error: Exception | None = None
        self.model = None
        self.maskable = False

        try:
            from sb3_contrib import MaskablePPO

            self.model = MaskablePPO.load(str(checkpoint_path))
            self.maskable = True
            return
        except Exception as exc:  # pragma: no cover - 依赖环境差异较大
            last_error = exc

        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(str(checkpoint_path))
            self.maskable = False
            return
        except Exception as exc:  # pragma: no cover - 依赖环境差异较大
            last_error = exc

        raise RuntimeError(f'Failed to load SB3 checkpoint: {checkpoint_path}') from last_error

    def predict(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> int:
        kwargs: dict[str, Any] = {}
        if self.maskable:
            kwargs['action_masks'] = obs['action_mask']
        try:
            action, _ = self.model.predict(obs, deterministic=deterministic, **kwargs)
        except TypeError:
            action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)


class RLlibPolicyRunner(PolicyRunner):
    """RLlib checkpoint 推理器。"""

    backend_name = 'rllib'

    def __init__(self, checkpoint_path: Path, env_config: dict[str, Any]):
        try:
            import ray
            from ray.rllib.algorithms.ppo import PPOConfig
            from ray.tune.registry import register_env
        except ModuleNotFoundError as exc:  # pragma: no cover - 依赖环境差异较大
            raise RuntimeError('RLlib demo requires ray[rllib] in the active environment.') from exc

        from .rllib_model import register_rllib_model

        self._started_ray = False
        self._ray = ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)
            self._started_ray = True

        env_name = f"gakumas_rl_demo_{hashlib.sha1(json.dumps(env_config, sort_keys=True).encode('utf-8')).hexdigest()[:12]}"
        register_env(env_name, lambda config: build_env_from_config(config))
        self._api_stack = _resolve_rllib_api_stack(checkpoint_path)
        self._torch = None
        if self._api_stack == 'new':
            from .rllib_model import build_rllib_module_spec

            self._torch = __import__('torch')
            config = (
                PPOConfig()
                .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
                .environment(env=env_name, env_config=env_config, disable_env_checking=True)
                .framework('torch')
                .rl_module(
                    rl_module_spec=build_rllib_module_spec(_resolve_rllib_custom_model_config(checkpoint_path))
                )
                .env_runners(num_env_runners=0)
            )
        else:
            model_name = register_rllib_model()
            config = (
                PPOConfig()
                .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                .environment(env=env_name, env_config=env_config, disable_env_checking=True)
                .framework('torch')
                .training(
                    model={
                        'custom_model': model_name,
                        'custom_model_config': _resolve_rllib_custom_model_config(checkpoint_path),
                    },
                )
                .resources(num_gpus=0.0)
                .env_runners(num_env_runners=0)
            )
        try:
            self.algo = config.build_algo()
        except AttributeError:  # pragma: no cover - 兼容旧版 RLlib
            self.algo = config.build()
        self.algo.restore(str(checkpoint_path))

    def predict(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> int:
        if self._api_stack == 'new':
            from ray.rllib.core.columns import Columns

            module = self.algo.get_module()
            if module is None:
                raise RuntimeError('RLlib new API checkpoint 缺少可用 RLModule。')
            device = next(module.parameters()).device
            batched_obs = tensorize_observation(obs, device)
            with self._torch.no_grad():
                output = module.forward_inference({Columns.OBS: batched_obs})
                dist_cls = module.get_inference_action_dist_cls()
                dist = dist_cls.from_logits(output[Columns.ACTION_DIST_INPUTS])
                if deterministic:
                    dist = dist.to_deterministic()
                action = dist.sample()
            return int(action.reshape(-1)[0].item())

        action = self.algo.compute_single_action(obs, explore=not deterministic)
        if isinstance(action, tuple):
            action = action[0]
        return int(action)

    def close(self) -> None:
        if hasattr(self, 'algo'):
            self.algo.stop()
        if self._started_ray:
            self._ray.shutdown()


def _load_rllib_ctor_config(checkpoint_path: Path) -> dict[str, Any] | None:
    """从 RLlib checkpoint 的构造参数中恢复算法配置。"""

    ctor_args_path = checkpoint_path / 'class_and_ctor_args.pkl'
    if not ctor_args_path.exists():
        return None
    try:
        payload = pickle.loads(ctor_args_path.read_bytes())
        args, kwargs = payload.get('ctor_args_and_kwargs', ((), {}))
        if args and isinstance(args[0], dict):
            return args[0]
        config = kwargs.get('config') if isinstance(kwargs, dict) else None
        if isinstance(config, dict):
            return config
    except Exception:
        return None
    return None


def _resolve_rllib_custom_model_config(checkpoint_path: Path) -> dict[str, Any]:
    """优先从训练元数据或 checkpoint 中恢复 RLlib 自定义模型配置。"""

    metadata_path = checkpoint_path.parent.parent / 'training_metadata.json'
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding='utf-8'))
            model_config = payload.get('config', {}).get('custom_model_config')
            if isinstance(model_config, dict) and model_config:
                return model_config
        except Exception:
            pass

    algorithm_state_path = checkpoint_path / 'algorithm_state.pkl'
    if algorithm_state_path.exists():
        try:
            payload = pickle.loads(algorithm_state_path.read_bytes())
            model_config = payload.get('config', {}).get('model', {}).get('custom_model_config')
            if isinstance(model_config, dict) and model_config:
                return model_config
        except Exception:
            pass

    ctor_config = _load_rllib_ctor_config(checkpoint_path)
    if ctor_config:
        model_config = ctor_config.get('model', {}).get('custom_model_config')
        if isinstance(model_config, dict) and model_config:
            return model_config
        rl_module_spec = ctor_config.get('_rl_module_spec')
        model_config = getattr(rl_module_spec, 'model_config', None)
        if isinstance(model_config, dict) and model_config:
            return model_config

    return {'hidden_dim': 256}


def _resolve_rllib_api_stack(checkpoint_path: Path) -> str:
    """优先从训练元数据推断 RLlib checkpoint 属于 old 还是 new API stack。"""

    metadata_path = checkpoint_path.parent.parent / 'training_metadata.json'
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding='utf-8'))
            api_stack = str(payload.get('config', {}).get('api_stack') or '').strip().lower()
            if api_stack in {'old', 'new'}:
                return api_stack
        except Exception:
            pass

    algorithm_state_path = checkpoint_path / 'algorithm_state.pkl'
    if algorithm_state_path.exists():
        try:
            payload = pickle.loads(algorithm_state_path.read_bytes())
            config = payload.get('config', {})
            if bool(config.get('enable_rl_module_and_learner')) and bool(config.get('enable_env_runner_and_connector_v2')):
                return 'new'
        except Exception:
            pass

    ctor_config = _load_rllib_ctor_config(checkpoint_path)
    if ctor_config:
        if bool(ctor_config.get('enable_rl_module_and_learner')) and bool(ctor_config.get('enable_env_runner_and_connector_v2')):
            return 'new'

    return 'old'


class TorchPolicyRunner(PolicyRunner):
    """本地 Torch actor-critic checkpoint 推理器。"""

    backend_name = 'torch'

    def __init__(self, checkpoint_path: Path, env: Any):
        import torch

        from .model import MaskedPolicyValueNet, tensorize_observation

        self._torch = torch
        self._tensorize_observation = tensorize_observation
        self.device = torch.device('cpu')
        global_dim = int(env.observation_space['global'].shape[0])
        action_dim = int(env.observation_space['action_features'].shape[-1])
        self.model = MaskedPolicyValueNet(global_dim, action_dim).to(self.device)
        payload = torch.load(checkpoint_path, map_location=self.device)
        state_dict = payload.get('model_state_dict', payload)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def predict(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> int:
        obs_tensor = self._tensorize_observation(obs, self.device)
        with self._torch.no_grad():
            logits, _ = self.model.forward(
                obs_tensor['global'],
                obs_tensor['action_features'],
                obs_tensor['action_mask'],
            )
            if deterministic:
                action = self._torch.argmax(logits, dim=-1)
            else:
                distribution = self._torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
        return int(action.item())


class HeuristicPolicyRunner(PolicyRunner):
    """没有 checkpoint 时的简易回放策略。"""

    backend_name = 'heuristic'

    def __init__(self, env: Any):
        self.env = env

    def predict(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> int:
        legal_indices = np.flatnonzero(np.asarray(obs['action_mask'], dtype=np.float32) > 0.5)
        if legal_indices.size == 0:
            return int(len(obs['action_mask']) - 1)
        for index in legal_indices.tolist():
            candidate = self.env._candidates[int(index)]
            if candidate.kind == 'card':
                return int(index)
        return int(legal_indices[-1])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='运行一局考试并输出 JSON trace 文件。')
    parser.add_argument('--checkpoint', type=Path, help='模型 checkpoint 路径；不传则使用内置启发式策略。')
    parser.add_argument('--backend', choices=('auto', 'sb3', 'rllib', 'torch', 'heuristic'), default='auto')
    parser.add_argument('--scenario', default='nia_master')
    parser.add_argument('--mode', default='exam', choices=('exam', 'lesson'))
    parser.add_argument('--stage-type', default=None)
    parser.add_argument('--exam-reward-mode', choices=('score', 'clear'), default='score')
    parser.add_argument('--lesson-action-type', default='')
    parser.add_argument('--lesson-level-index', type=int, default=0)
    parser.add_argument('--idol-card-id', default='')
    parser.add_argument('--producer-level', type=int, default=35)
    parser.add_argument('--idol-rank', type=int, default=0)
    parser.add_argument('--dearness-level', type=int, default=0)
    parser.add_argument('--use-after-item', action='store_true')
    parser.add_argument('--exam-score-bonus-multiplier', type=float, default=None)
    parser.add_argument('--fan-votes', type=float, default=None)
    parser.add_argument('--exam-randomize-context', action='store_true')
    parser.add_argument('--exam-randomize-stage-type', action='store_true')
    parser.add_argument('--exam-randomize-use-after-item', action='store_true')
    parser.add_argument(
        '--manual-exam-setup',
        action='append',
        default=[],
        help='manual exam setup jsonl; when provided, replay samples real deck/drink/item records',
    )
    parser.add_argument(
        '--guarantee-card-effect',
        action='append',
        default=[],
        help='ensure at least N cards of one effect tag, e.g. review=3, 打分=4, 元气=2',
    )
    parser.add_argument(
        '--force-card',
        action='append',
        default=[],
        help='force player-axis cards into the random initial deck via JSON or @json file, e.g. {"好印象":["p_card..."]} or {"干劲":["p_card..."]}',
    )
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-steps', type=int, default=64)
    parser.add_argument('--output', type=Path, help='输出 JSON 路径，默认写到 checkpoint/run 同级目录。')
    parser.add_argument('--stochastic', action='store_true', help='使用随机策略采样，而不是确定性动作。')
    return parser.parse_args(argv)


def _default_output_path(checkpoint_path: Path | None, backend: str, scenario: str) -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if checkpoint_path is None:
        return RUNS_DIR / f'demo_{backend}_{scenario}_{stamp}.json'
    parent = checkpoint_path.parent if checkpoint_path.is_dir() else checkpoint_path.parent
    return parent / f'demo_{backend}_{scenario}_{stamp}.json'


def _infer_backend(checkpoint_path: Path | None, requested_backend: str) -> str:
    if requested_backend != 'auto':
        return requested_backend
    if checkpoint_path is None:
        return 'heuristic'
    if checkpoint_path.suffix == '.pt':
        return 'torch'
    if checkpoint_path.is_dir() or checkpoint_path.name.startswith('checkpoint'):
        return 'rllib'
    return 'sb3'


def _build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    loadout = LoadoutConfig(
        idol_card_id=args.idol_card_id,
        producer_level=args.producer_level,
        idol_rank=args.idol_rank,
        dearness_level=args.dearness_level,
        use_after_item=True if getattr(args, 'use_after_item', False) else None,
        exam_score_bonus_multiplier=getattr(args, 'exam_score_bonus_multiplier', None),
        fan_votes=getattr(args, 'fan_votes', None),
        exam_randomize_context=bool(args.exam_randomize_context),
        exam_randomize_use_after_item=bool(args.exam_randomize_use_after_item),
        exam_randomize_stage_type=bool(args.exam_randomize_stage_type),
    )
    return {
        'mode': args.mode,
        'scenario': args.scenario,
        'stage_type': args.stage_type,
        'idol_card_id': loadout.idol_card_id,
        'producer_level': loadout.producer_level,
        'idol_rank': loadout.idol_rank,
        'dearness_level': loadout.dearness_level,
        'use_after_item': loadout.use_after_item,
        'exam_score_bonus_multiplier': loadout.exam_score_bonus_multiplier,
        'fan_votes': loadout.fan_votes,
        'exam_randomize_context': loadout.exam_randomize_context,
        'exam_randomize_stage_type': loadout.exam_randomize_stage_type,
        'exam_randomize_use_after_item': loadout.exam_randomize_use_after_item,
        'exam_reward_mode': args.exam_reward_mode,
        'lesson_action_type': args.lesson_action_type,
        'lesson_level_index': args.lesson_level_index,
        'manual_exam_setup_paths': list(args.manual_exam_setup),
        'guarantee_card_effects': list(args.guarantee_card_effect),
        'force_card_groups': list(args.force_card),
    }


def _short_enum(value: str | None) -> str:
    raw = str(value or '').strip()
    if not raw:
        return ''
    for prefix in (
        'ProduceExamEffectType_',
        'ProduceExamPhaseType_',
        'ProduceCardCategory_',
        'ProduceCardRarity_',
        'ProduceCardCostType_',
        'ProduceStepType_',
        'ExamDescriptionType_',
    ):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def _dedupe_keep_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text_value = str(value).strip()
        if not text_value or text_value in seen:
            continue
        seen.add(text_value)
        result.append(text_value)
    return result


def _description_texts(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    values = []
    for item in payload.get('produceDescriptions', []):
        if not item:
            continue
        text = str(item.get('text') or '').replace('<nobr>', '').replace('</nobr>', '').strip()
        if text:
            values.append(text)
    return _dedupe_keep_order(values)


def _description_token(item: dict[str, Any]) -> str:
    text = str(item.get('text') or '').replace('<nobr>', '').replace('</nobr>', '').strip()
    if text:
        return text
    target_id = str(item.get('targetId') or '')
    exam_description_type = str(item.get('examDescriptionType') or '')
    if target_id == 'Description_ProduceCardIsInitial' or exam_description_type == 'ExamDescriptionType_CustomizeInitialAdd':
        return '开始时加入手牌'
    if target_id == 'Description_LessonCountAdd_CountSection':
        return '课程中'
    return ''


def _join_description_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ''
    text = ' '.join(token for token in tokens if token)
    for marker in (' + ', ' - ', ' × ', ' / ', ' % ', ' )', '( ', ' ]', '[ '):
        compact = marker.strip()
        text = text.replace(marker, compact)
    text = text.replace('> <', '><')
    return ' '.join(text.split()).strip()


def _should_flush_description_line(tokens: list[str]) -> bool:
    line = _join_description_tokens(tokens)
    if not line:
        return False
    markers = ('+', '-', '×', '%', '开始时加入手牌', '不可重复', '课程中仅限')
    return any(marker in line for marker in markers) or any(char.isdigit() for char in line)


def _assembled_description_lines(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    lines: list[str] = []
    current_tokens: list[str] = []
    standalone_tokens = {'开始时加入手牌', '不可重复', '课程中仅限1次'}
    for item in payload.get('produceDescriptions', []):
        if not item:
            continue
        raw_text = str(item.get('text') or '').replace('<nobr>', '').replace('</nobr>', '').strip()
        desc_type = str(item.get('produceDescriptionType') or '')
        token = _description_token(item)
        if token in standalone_tokens:
            line = _join_description_tokens(current_tokens)
            if line:
                lines.append(line)
            current_tokens = []
            lines.append(token)
            continue
        if desc_type == 'ProduceDescriptionType_PlainText' and not raw_text:
            if _should_flush_description_line(current_tokens):
                line = _join_description_tokens(current_tokens)
                if line:
                    lines.append(line)
                current_tokens = []
            continue
        if desc_type in {'ProduceDescriptionType_ProduceExamEffectType', 'ProduceDescriptionType_ProduceDescription', 'ProduceDescriptionType_ProduceDescriptionName'} and current_tokens and _should_flush_description_line(current_tokens):
            line = _join_description_tokens(current_tokens)
            if line:
                lines.append(line)
            current_tokens = []
        if token:
            current_tokens.append(token)
    tail = _join_description_tokens(current_tokens)
    if tail:
        lines.append(tail)
    return _dedupe_keep_order(lines)


def _effect_summary(repository: Any, effect_id: str) -> dict[str, Any]:
    if not effect_id:
        return {'id': '', 'type': '', 'texts': []}
    row = repository.exam_effect_map.get(str(effect_id))
    if not row:
        return {'id': str(effect_id), 'type': str(effect_id), 'texts': []}
    return {
        'id': str(effect_id),
        'type': _short_enum(row.get('effectType')),
        'texts': _description_texts(row),
    }


def _trigger_summary(repository: Any, trigger_id: str) -> dict[str, Any]:
    if not trigger_id:
        return {'id': '', 'phases': [], 'texts': []}
    row = repository.exam_trigger_map.get(str(trigger_id))
    if not row:
        return {'id': str(trigger_id), 'phases': [], 'texts': []}
    return {
        'id': str(trigger_id),
        'phases': [_short_enum(value) for value in row.get('phaseTypes', []) if value],
        'texts': _description_texts(row),
    }


def _effect_pairs_to_lines(effect_pairs: list[dict[str, Any]]) -> list[str]:
    """将 effect_pairs 转为可读文本行，用于 preview_lines 字段。"""
    lines: list[str] = []
    for pair in effect_pairs:
        effect = pair.get('effect') or {}
        trigger = pair.get('trigger') or {}
        effect_name = str(effect.get('type') or effect.get('id') or '').strip()
        trigger_name = ', '.join(trigger.get('phases', []))
        head = f'{trigger_name} -> {effect_name}' if trigger_name and effect_name else (effect_name or trigger_name)
        texts = effect.get('texts') or trigger.get('texts') or []
        if texts:
            lines.append(f'{head}: {texts[0]}' if head else texts[0])
        elif head:
            lines.append(head)
    return _dedupe_keep_order(lines)


def _card_cost_badges(runtime: Any, card: Any) -> list[str]:
    badges: list[str] = []
    stamina = float(card.base_card.get('stamina') or 0)
    if stamina:
        badges.append(f'体力 {stamina:.0f}')
    force_stamina = float(card.base_card.get('forceStamina') or 0)
    if force_stamina:
        badges.append(f'强制体力 {force_stamina:.0f}')
    try:
        resource_costs = runtime._card_resource_costs(card)
    except Exception:
        resource_costs = {}
    for key, value in resource_costs.items():
        if not value:
            continue
        badges.append(f'{_short_enum(key)} {float(value):.0f}')
    return badges


def _card_summary(runtime: Any, slot: int, card: Any) -> dict[str, Any]:
    repository = runtime.repository
    card_id = str(card.card_id)
    localization = repository.produce_card_localization.get(card_id, {})
    original_name = str(card.base_card.get('name') or '')
    display_name = str(localization.get('name') or original_name or repository.card_name(card.base_card))
    resolved_pairs = runtime._resolved_card_play_effects(card)
    effect_pairs = [
        {
            'effect': _effect_summary(repository, str(pair.get('effect_id') or '')),
            'trigger': _trigger_summary(repository, str(pair.get('trigger_id') or '')) if pair.get('trigger_id') else None,
        }
        for pair in resolved_pairs
    ]
    trigger_summaries = [_trigger_summary(repository, trigger_id) for trigger_id in runtime._effective_card_trigger_ids(card)]
    description_lines = _assembled_description_lines(localization) or _assembled_description_lines(card.base_card)
    original_description_lines = _assembled_description_lines(card.base_card) if localization else []
    preview_lines = description_lines or _effect_pairs_to_lines(effect_pairs)
    return {
        'slot': slot,
        'label': display_name,
        'original_name': original_name,
        'card_id': card_id,
        'available': bool(runtime._can_play_card(card)),
        'stamina': float(card.base_card.get('stamina') or 0),
        'evaluation': float(card.base_card.get('evaluation') or 0),
        'rarity': _short_enum(card.base_card.get('rarity')),
        'category': _short_enum(card.base_card.get('category')),
        'upgrade_count': int(card.upgrade_count),
        'cost_badges': _card_cost_badges(runtime, card),
        'description_lines': description_lines,
        'original_description_lines': original_description_lines,
        'effect_pairs': effect_pairs,
        'trigger_summaries': trigger_summaries,
        'preview_lines': preview_lines[:3],
    }


def _drink_summary(runtime: Any, slot: int, drink: dict[str, Any]) -> dict[str, Any]:
    repository = runtime.repository
    drink_id = str(drink.get('id') or '')
    localization = repository.produce_drink_localization.get(drink_id, {})
    effect_pairs = []
    for drink_effect_id in drink.get('produceDrinkEffectIds', []):
        drink_effect = repository.drink_effect_map.get(str(drink_effect_id))
        if not drink_effect:
            continue
        effect_pairs.append(
            {
                'effect': _effect_summary(repository, str(drink_effect.get('produceExamEffectId') or '')),
                'trigger': None,
            }
        )
    description_lines = _assembled_description_lines(localization) or _effect_pairs_to_lines(effect_pairs)
    return {
        'slot': slot,
        'label': repository.drink_name(drink),
        'drink_id': drink_id,
        'consumed': bool(drink.get('_consumed')),
        'rarity': _short_enum(drink.get('rarity')),
        'description_lines': description_lines,
        'effect_pairs': effect_pairs,
        'preview_lines': description_lines[:3],
    }


def _gimmick_summary(runtime: Any, row: dict[str, Any]) -> dict[str, Any]:
    effect_id = str(row.get('produceExamEffectId') or '')
    effect = _effect_summary(runtime.repository, effect_id)
    # 优先使用 gimmick 行自身的 produceDescriptions（YAML 原始数据已包含可读文本）
    lines = _assembled_description_lines(row)
    # 回退：尝试本地化数据
    if not lines:
        gimmick_id = str(row.get('id') or '')
        priority = int(row.get('priority') or 0)
        loc = runtime.repository.load_localization('ProduceExamGimmickEffectGroup')
        loc_key = f'{gimmick_id}:{priority}'
        loc_row = loc.get(loc_key) or loc.get(gimmick_id)
        if loc_row:
            lines = _assembled_description_lines(loc_row)
    # 最终回退：使用 effect 信息
    if not lines:
        lines = effect.get('texts') or ([str(effect.get('type'))] if effect.get('type') else [])
    return {
        'start_turn': int(row.get('startTurn') or 0),
        'effect_id': effect_id,
        'effect_type': effect.get('type', ''),
        'text_lines': lines,
        'is_positive': bool(row.get('isPositive', True)),
    }


def _snapshot(env: Any, obs: dict[str, np.ndarray]) -> dict[str, Any]:
    runtime = env.runtime
    hand_cards = [_card_summary(runtime, index, card) for index, card in enumerate(runtime.hand)]
    drinks = [_drink_summary(runtime, index, drink) for index, drink in enumerate(runtime.drinks)]
    actions: list[dict[str, Any]] = []
    for index, candidate in enumerate(env._candidates):
        slot_group = 'system'
        slot_index = 0
        if index < env.max_hand_cards:
            slot_group = 'hand'
            slot_index = index
        elif index < env.max_hand_cards + env.max_drinks:
            slot_group = 'drink'
            slot_index = index - env.max_hand_cards
        actions.append(
            {
                'index': index,
                'label': candidate.label,
                'kind': candidate.kind,
                'available': bool(candidate.payload.get('available', False)),
                'slot_group': slot_group,
                'slot_index': slot_index,
            }
        )
    return {
        'turn': int(runtime.turn),
        'max_turns': int(runtime.max_turns),
        'score': float(runtime.score),
        'target_score': float(runtime._target_score()),
        'stamina': float(runtime.stamina),
        'max_stamina': float(runtime.max_stamina),
        'stance': str(runtime.stance),
        'turn_color': str(runtime.current_turn_color),
        'turn_color_label': runtime.turn_color_label(),
        'play_limit': int(runtime.play_limit),
        'score_bonus_multiplier': float(runtime.score_bonus_multiplier),
        'resources': {key: float(runtime.resources.get(key, 0.0)) for key in _RESOURCE_KEYS},
        'parameter_stats': {
            'vocal': float(runtime.parameter_stats[0]),
            'dance': float(runtime.parameter_stats[1]),
            'visual': float(runtime.parameter_stats[2]),
        },
        'zones': {
            'deck': int(len(runtime.deck)),
            'hand': int(len(runtime.hand)),
            'grave': int(len(runtime.grave)),
            'hold': int(len(runtime.hold)),
            'lost': int(len(runtime.lost)),
        },
        'gimmicks': [_gimmick_summary(runtime, row) for row in runtime.gimmick_rows],
        'hand_cards': hand_cards,
        'drinks': drinks,
        'legal_actions': actions,
        'action_mask': [int(value > 0.5) for value in np.asarray(obs['action_mask']).tolist()],
    }


def _load_policy_runner(backend: str, checkpoint_path: Path | None, env: Any, env_config: dict[str, Any]) -> PolicyRunner:
    if backend == 'heuristic':
        return HeuristicPolicyRunner(env)
    if checkpoint_path is None:
        raise ValueError('checkpoint is required when backend is sb3, rllib, or torch')
    if backend == 'sb3':
        return SB3PolicyRunner(checkpoint_path)
    if backend == 'rllib':
        return RLlibPolicyRunner(checkpoint_path, env_config)
    if backend == 'torch':
        return TorchPolicyRunner(checkpoint_path, env)
    raise ValueError(f'Unsupported backend: {backend}')


def _choose_action(policy: PolicyRunner, obs: dict[str, np.ndarray], deterministic: bool) -> tuple[int, str | None, int]:
    predicted_index = int(policy.predict(obs, deterministic=deterministic))
    mask = np.asarray(obs['action_mask'], dtype=np.float32)
    fallback_reason = None
    if predicted_index < 0 or predicted_index >= mask.shape[0] or mask[predicted_index] <= 0.5:
        legal_indices = np.flatnonzero(mask > 0.5)
        if legal_indices.size == 0:
            return int(mask.shape[0] - 1), f'no_legal_action_from_prediction:{predicted_index}', predicted_index
        fallback_index = int(legal_indices[-1])
        fallback_reason = f'invalid_prediction:{predicted_index}'
        return fallback_index, fallback_reason, predicted_index
    return predicted_index, fallback_reason, predicted_index


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else None
    backend = _infer_backend(checkpoint_path, args.backend)
    env_config = _build_env_config(args)
    env = build_env_from_config(env_config)
    obs, info = env.reset(seed=args.seed)
    policy = _load_policy_runner(backend, checkpoint_path, env, env_config)
    trace: list[dict[str, Any]] = []
    total_reward = 0.0
    final_state: dict[str, Any] | None = None
    try:
        for step_index in range(max(int(args.max_steps), 1)):
            before = _snapshot(env, obs)
            action_index, fallback_reason, predicted_index = _choose_action(policy, obs, deterministic=not args.stochastic)
            action = before['legal_actions'][action_index]
            next_obs, reward, terminated, truncated, step_info = env.step(action_index)
            after = _snapshot(env, next_obs)
            total_reward += float(reward)
            trace.append(
                {
                    'step_index': step_index,
                    'predicted_index': predicted_index,
                    'selected_index': action_index,
                    'action': action,
                    'fallback_reason': fallback_reason,
                    'reward': float(reward),
                    'terminated': bool(terminated),
                    'truncated': bool(truncated),
                    'before': before,
                    'after': after,
                    'info': step_info,
                }
            )
            obs = next_obs
            if terminated or truncated:
                break
        final_state = trace[-1]['after'] if trace else _snapshot(env, obs)
    finally:
        policy.close()
        env.close()
    if final_state is None:
        final_state = trace[-1]['after'] if trace else _snapshot(env, obs)
    report = {
        'summary': {
            'backend': backend,
            'checkpoint': str(checkpoint_path) if checkpoint_path else None,
            'scenario': args.scenario,
            'reward_mode': args.exam_reward_mode,
            'seed': int(args.seed),
            'total_reward': float(total_reward),
            'final_score': float(final_state['score']),
            'target_score': float(final_state['target_score']),
            'turns_played': len(trace),
            'terminated': bool(trace[-1]['terminated']) if trace else False,
        },
        'reset_info': info,
        'trace': trace,
    }
    output_path = (args.output or _default_output_path(checkpoint_path, backend, args.scenario)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix('.json')
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    report['artifacts'] = {'json': str(output_path)}
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_demo(args)
    print(json.dumps(report['artifacts'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
