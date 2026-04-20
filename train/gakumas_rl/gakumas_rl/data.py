"""RL 训练使用的主数据仓库、效果分类与场景装配工具。"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import yaml

from .deck_constraints import summarize_exam_effect_axes


def _detect_package_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'pyproject.toml').exists() and (parent / 'gakumas_rl').exists():
            return parent
    return current.parents[1]


PACKAGE_ROOT = _detect_package_root()
ROOT_DIR = Path(os.getenv('GAKUMAS_RL_ROOT_DIR') or PACKAGE_ROOT)
ASSETS_DIR = Path(os.getenv('GAKUMAS_RL_ASSETS_DIR') or (ROOT_DIR / 'assets' / 'gakumasu-diff'))
LOCALIZATION_DIR = Path(
    os.getenv('GAKUMAS_RL_LOCALIZATION_DIR')
    or (ROOT_DIR / 'assets' / 'GakumasTranslationData' / 'local-files' / 'masterTrans')
)
OUTPUT_DIR = Path(os.getenv('GAKUMAS_RL_OUTPUT_DIR') or ROOT_DIR)
RUNS_DIR = OUTPUT_DIR / 'runs'
TRAJECTORIES_DIR = OUTPUT_DIR / 'trajectories'
CHECKPOINTS_DIR = OUTPUT_DIR / 'checkpoints'
MASTERDATA_CACHE_VERSION = 1

_SHARED_TABLE_CACHE: dict[tuple[str, int, int, int], 'TableIndex'] = {}
_SHARED_LOCALIZATION_CACHE: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = {}

_ALLOWED_EXAM_PLAN_TYPES = {
    'ProducePlanType_Common',
    'ProducePlanType_Plan1',
    'ProducePlanType_Plan2',
    'ProducePlanType_Plan3',
}


def _allowed_plan_types(plan_type: str | None = None) -> set[str]:
    normalized = str(plan_type or '')
    if normalized and normalized in _ALLOWED_EXAM_PLAN_TYPES:
        if normalized == 'ProducePlanType_Common':
            return {'ProducePlanType_Common'}
        return {'ProducePlanType_Common', normalized}
    return set(_ALLOWED_EXAM_PLAN_TYPES)


def _upgrade_count(row: dict[str, Any]) -> int:
    return int(row.get('upgradeCount') or 0)


def _card_cache_key(row: dict[str, Any]) -> tuple[str, int]:
    """为卡牌主数据行生成稳定缓存键。"""

    return (str(row.get('id') or ''), _upgrade_count(row))


def _random_upgrade_weight(upgrade_count: int) -> float:
    if upgrade_count <= 0:
        return 0.30
    if upgrade_count == 1:
        return 0.55
    return 0.15 * (0.25 ** max(upgrade_count - 2, 0))


def _source_cache_key(path: Path) -> tuple[str, int, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return (str(resolved), int(stat.st_mtime_ns), int(stat.st_size), MASTERDATA_CACHE_VERSION)


@dataclass(frozen=True)
class TableIndex:
    """按 `id` 聚合后的主数据表索引，便于 O(1) 查询。"""

    name: str
    rows: list[dict[str, Any]]
    by_id: dict[str, list[dict[str, Any]]]

    def first(self, item_id: str) -> dict[str, Any] | None:
        """返回指定 `id` 的第一条记录。"""

        items = self.by_id.get(item_id)
        return items[0] if items else None

    def all(self, item_id: str) -> list[dict[str, Any]]:
        """返回指定 `id` 的全部记录副本。"""

        return list(self.by_id.get(item_id, []))


@dataclass(frozen=True)
class ScenarioSpec:
    """训练环境共享的场景配置快照。"""

    scenario_id: str
    produce_id: str
    produce_name: str
    group_id: str
    route_type: str
    parameter_growth_limit: float
    steps: int
    action_point_quantity: int
    max_refresh_count: int
    drink_limit: int
    audition_sequence: tuple[str, ...]
    action_types: tuple[str, ...]
    focus_effect_types: tuple[str, ...]
    score_weights: tuple[float, float, float]
    exam_turns: int
    default_stage: str
    reward_weights: dict[str, float]


@dataclass(frozen=True)
class LessonTrainingSpec:
    """独立 lesson 训练模式使用的课程主数据快照。"""

    action_type: str
    lesson_kind: str
    stat_type: str
    lesson_type: str
    lesson_trigger_types: tuple[str, ...]
    lesson_post_clear_types: tuple[str, ...]
    source_row_id: str
    source_level_id: str
    name: str
    clear_target: float
    perfect_target: float
    turn_limit: int
    level_index: int


@dataclass(frozen=True)
class EffectTaxonomy:
    """把效果类型、触发阶段和卡片元数据编码成固定索引。"""

    exam_effect_types: tuple[str, ...]
    produce_effect_types: tuple[str, ...]
    trigger_phases: tuple[str, ...]
    action_types: tuple[str, ...]
    card_categories: tuple[str, ...]
    card_rarities: tuple[str, ...]
    card_cost_types: tuple[str, ...]

    @cached_property
    def exam_effect_index(self) -> dict[str, int]:
        """考试效果类型到向量下标的映射。"""

        return {value: idx for idx, value in enumerate(self.exam_effect_types)}

    @cached_property
    def produce_effect_index(self) -> dict[str, int]:
        """培养效果类型到向量下标的映射。"""

        return {value: idx for idx, value in enumerate(self.produce_effect_types)}

    @cached_property
    def trigger_phase_index(self) -> dict[str, int]:
        """考试触发阶段到向量下标的映射。"""

        return {value: idx for idx, value in enumerate(self.trigger_phases)}

    @cached_property
    def action_index(self) -> dict[str, int]:
        """动作类型到向量下标的映射。"""

        return {value: idx for idx, value in enumerate(self.action_types)}

    @cached_property
    def category_index(self) -> dict[str, int]:
        """卡牌类别到向量下标的映射。"""

        return {value: idx for idx, value in enumerate(self.card_categories)}

    @cached_property
    def rarity_index(self) -> dict[str, int]:
        """卡牌稀有度到向量下标的映射。"""

        return {value: idx for idx, value in enumerate(self.card_rarities)}

    @cached_property
    def cost_index(self) -> dict[str, int]:
        """卡牌费用类型到向量下标的映射。"""

        return {value: idx for idx, value in enumerate(self.card_cost_types)}

    def encode(self, values: Iterable[str], index_map: dict[str, int], size: int) -> np.ndarray:
        """把离散标签集合编码成计数向量。"""

        vector = np.zeros(size, dtype=np.float32)
        for value in values:
            idx = index_map.get(value)
            if idx is not None:
                vector[idx] += 1.0
        return vector

    def encode_exam_effects(self, values: Iterable[str]) -> np.ndarray:
        """编码考试效果类型集合。"""

        return self.encode(values, self.exam_effect_index, len(self.exam_effect_types))

    def encode_produce_effects(self, values: Iterable[str]) -> np.ndarray:
        """编码培养效果类型集合。"""

        return self.encode(values, self.produce_effect_index, len(self.produce_effect_types))

    def encode_trigger_phases(self, values: Iterable[str]) -> np.ndarray:
        """编码考试触发阶段集合。"""

        return self.encode(values, self.trigger_phase_index, len(self.trigger_phases))

    def encode_actions(self, values: Iterable[str]) -> np.ndarray:
        """编码动作类型集合。"""

        return self.encode(values, self.action_index, len(self.action_types))

    def encode_categories(self, values: Iterable[str]) -> np.ndarray:
        """编码卡牌类别集合。"""

        return self.encode(values, self.category_index, len(self.card_categories))

    def encode_rarities(self, values: Iterable[str]) -> np.ndarray:
        """编码卡牌稀有度集合。"""

        return self.encode(values, self.rarity_index, len(self.card_rarities))

    def encode_cost_types(self, values: Iterable[str]) -> np.ndarray:
        """编码卡牌费用类型集合。"""

        return self.encode(values, self.cost_index, len(self.card_cost_types))


class MasterDataRepository:
    """RL 代码使用的无副作用主数据视图与派生统计仓库。"""

    def __init__(
        self,
        root_dir: Path | None = None,
        assets_dir: Path | None = None,
        localization_dir: Path | None = None,
    ):
        """初始化主数据目录。

        优先级依次为显式参数、环境变量、独立包根目录下的 `assets/`。
        """

        self.root_dir = Path(root_dir) if root_dir else ROOT_DIR
        derived_assets_dir = self.root_dir / 'assets' / 'gakumasu-diff'
        derived_localization_dir = self.root_dir / 'assets' / 'GakumasTranslationData' / 'local-files' / 'masterTrans'
        self.assets_dir = Path(assets_dir) if assets_dir else (ASSETS_DIR if root_dir is None else derived_assets_dir)
        self.localization_dir = (
            Path(localization_dir) if localization_dir else (LOCALIZATION_DIR if root_dir is None else derived_localization_dir)
        )
        self.cache_dir = self.root_dir / '.gakumas_rl_cache'
        self._table_cache: dict[str, TableIndex] = {}
        self._localization_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._card_exam_effect_types_cache: dict[tuple[str, int], tuple[str, ...]] = {}
        self._card_trigger_phases_cache: dict[tuple[str, int], tuple[str, ...]] = {}
        self._drink_exam_effect_types_cache: dict[str, tuple[str, ...]] = {}
        self._weighted_card_pool_cache: dict[tuple[str, str, str], tuple[dict[str, Any], ...]] = {}
        self._canonical_card_row_cache: dict[str, dict[str, Any] | None] = {}
        self._card_row_by_upgrade_cache: dict[str, dict[int, dict[str, Any]]] = {}
        self._card_variant_sampling_cache: dict[str, tuple[tuple[dict[str, Any], ...], np.ndarray]] = {}

    # 原始主数据与本地化都先经过这里的清洗和缓存入口。
    def _sanitize_yaml(self, text: str) -> str:
        """清洗 YAML 文本中的控制字符与制表符。"""

        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
        return text.replace('\t', '    ')

    def load_table(self, table_name: str) -> TableIndex:
        """按需加载主数据表，并缓存成 `TableIndex`。"""

        cached = self._table_cache.get(table_name)
        if cached is not None:
            return cached

        table_path = self.assets_dir / f'{table_name}.yaml'
        if not table_path.exists():
            raise FileNotFoundError(
                f'Master table not found: {table_path}. '
                '请复制 `assets/gakumasu-diff` 到独立包目录，或设置 GAKUMAS_RL_ASSETS_DIR。'
            )
        source_key = _source_cache_key(table_path)
        shared = _SHARED_TABLE_CACHE.get(source_key)
        if shared is not None:
            self._table_cache[table_name] = shared
            return shared
        disk_cached = self._read_disk_cache(self._serialized_cache_path('tables', table_name), source_key)
        if isinstance(disk_cached, TableIndex):
            self._table_cache[table_name] = disk_cached
            _SHARED_TABLE_CACHE[source_key] = disk_cached
            return disk_cached
        text = self._sanitize_yaml(table_path.read_text(encoding='utf-8'))
        rows = yaml.load(text, Loader=yaml.CSafeLoader) or []
        by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            item_id = row.get('id')
            if item_id is not None:
                by_id[str(item_id)].append(row)
        index = TableIndex(name=table_name, rows=rows, by_id=dict(by_id))
        self._table_cache[table_name] = index
        _SHARED_TABLE_CACHE[source_key] = index
        self._write_disk_cache(self._serialized_cache_path('tables', table_name), source_key, index)
        return index

    def load_localization(self, table_name: str) -> dict[str, dict[str, Any]]:
        """加载本地翻译表，并按 `id` 建立映射。"""

        cached = self._localization_cache.get(table_name)
        if cached is not None:
            return cached

        path = self.localization_dir / f'{table_name}.json'
        if not path.exists():
            self._localization_cache[table_name] = {}
            return {}
        source_key = _source_cache_key(path)
        shared = _SHARED_LOCALIZATION_CACHE.get(source_key)
        if shared is not None:
            self._localization_cache[table_name] = shared
            return shared
        disk_cached = self._read_disk_cache(self._serialized_cache_path('localization', table_name), source_key)
        if isinstance(disk_cached, dict):
            self._localization_cache[table_name] = disk_cached
            _SHARED_LOCALIZATION_CACHE[source_key] = disk_cached
            return disk_cached
        payload = json.loads(path.read_text(encoding='utf-8'))
        mapping: dict[str, dict[str, Any]] = {}
        for row in payload.get('data', []):
            row_id = row.get('id')
            if row_id is not None:
                mapping[str(row_id)] = row
                # 当存在 upgradeCount 时，额外存储复合键 "{id}.{upgradeCount}"
                # 以区分同 id 不同升级等级的本地化名称
                upgrade_count = row.get('upgradeCount')
                if upgrade_count is not None:
                    mapping[f'{row_id}.{int(upgrade_count)}'] = row
        self._localization_cache[table_name] = mapping
        _SHARED_LOCALIZATION_CACHE[source_key] = mapping
        self._write_disk_cache(self._serialized_cache_path('localization', table_name), source_key, mapping)
        return mapping

    def _serialized_cache_path(self, namespace: str, table_name: str) -> Path:
        return self.cache_dir / namespace / f'{table_name}.pkl'

    def _read_disk_cache(self, cache_path: Path, source_key: tuple[str, int, int, int]) -> Any:
        if not cache_path.exists():
            return None
        try:
            with cache_path.open('rb') as handle:
                payload = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if tuple(payload.get('source_key') or ()) != source_key:
            return None
        return payload.get('value')

    def _write_disk_cache(self, cache_path: Path, source_key: tuple[str, int, int, int], value: Any) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
            with temp_path.open('wb') as handle:
                pickle.dump({'source_key': source_key, 'value': value}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.replace(cache_path)
        except OSError:
            return

    @cached_property
    def produce_cards(self) -> TableIndex:
        return self.load_table('ProduceCard')

    @cached_property
    def produce_drinks(self) -> TableIndex:
        return self.load_table('ProduceDrink')

    @cached_property
    def produce_items(self) -> TableIndex:
        return self.load_table('ProduceItem')

    @cached_property
    def drink_effects(self) -> TableIndex:
        return self.load_table('ProduceDrinkEffect')

    @cached_property
    def exam_effects(self) -> TableIndex:
        return self.load_table('ProduceExamEffect')

    @cached_property
    def exam_status_enchants(self) -> TableIndex:
        return self.load_table('ProduceExamStatusEnchant')

    @cached_property
    def exam_triggers(self) -> TableIndex:
        return self.load_table('ProduceExamTrigger')

    @cached_property
    def effect_groups(self) -> TableIndex:
        return self.load_table('EffectGroup')

    @cached_property
    def produce_effects(self) -> TableIndex:
        return self.load_table('ProduceEffect')

    @cached_property
    def produces(self) -> TableIndex:
        return self.load_table('Produce')

    @cached_property
    def produce_groups(self) -> TableIndex:
        return self.load_table('ProduceGroup')

    @cached_property
    def produce_settings(self) -> TableIndex:
        return self.load_table('ProduceSetting')

    @cached_property
    def produce_initial_decks(self) -> TableIndex:
        return self.load_table('ProduceInitialDeck')

    @cached_property
    def exam_initial_decks(self) -> TableIndex:
        return self.load_table('ExamInitialDeck')

    @cached_property
    def audition_difficulties(self) -> TableIndex:
        return self.load_table('ProduceStepAuditionDifficulty')

    @cached_property
    def battle_configs(self) -> TableIndex:
        return self.load_table('ProduceExamBattleConfig')

    @cached_property
    def battle_score_configs(self) -> TableIndex:
        return self.load_table('ProduceExamBattleScoreConfig')

    @cached_property
    def auto_evaluations(self) -> TableIndex:
        return self.load_table('ProduceExamAutoEvaluation')

    @cached_property
    def auto_trigger_evaluations(self) -> TableIndex:
        return self.load_table('ProduceExamAutoTriggerEvaluation')

    @cached_property
    def auto_play_card_evaluations(self) -> TableIndex:
        return self.load_table('ProduceExamAutoPlayCardEvaluation')

    @cached_property
    def lesson_levels(self) -> TableIndex:
        return self.load_table('ProduceStepLessonLevel')

    @cached_property
    def lesson_rows(self) -> TableIndex:
        return self.load_table('ProduceStepLesson')

    @cached_property
    def effect_group_map(self) -> dict[str, dict[str, Any]]:
        return {row['id']: row for row in self.effect_groups.rows}

    @cached_property
    def exam_effect_map(self) -> dict[str, dict[str, Any]]:
        return {row['id']: row for row in self.exam_effects.rows}

    @cached_property
    def exam_status_enchant_map(self) -> dict[str, dict[str, Any]]:
        return {row['id']: row for row in self.exam_status_enchants.rows}

    @cached_property
    def exam_trigger_map(self) -> dict[str, dict[str, Any]]:
        return {row['id']: row for row in self.exam_triggers.rows}

    @cached_property
    def drink_effect_map(self) -> dict[str, dict[str, Any]]:
        return {row['id']: row for row in self.drink_effects.rows}

    @cached_property
    def battle_config_map(self) -> dict[str, dict[str, Any]]:
        return {row['id']: row for row in self.battle_configs.rows}

    @cached_property
    def battle_score_config_map(self) -> dict[str, dict[str, Any]]:
        return {row['id']: row for row in self.battle_score_configs.rows}

    @cached_property
    def produce_localization(self) -> dict[str, dict[str, Any]]:
        return self.load_localization('Produce')

    @cached_property
    def lesson_profile_stats(self) -> dict[str, float]:
        """统计普通、SP、Hard 课程的典型目标值。"""

        buckets: dict[str, list[float]] = defaultdict(list)
        for row in self.lesson_levels.rows:
            lesson_id = str(row.get('id', ''))
            target_value = float(row.get('resultTargetValueLimit') or 0)
            if '-normal-' in lesson_id:
                buckets['normal'].append(target_value)
            elif '-sp-' in lesson_id:
                buckets['sp'].append(target_value)
            elif '-hard-' in lesson_id:
                buckets['hard'].append(target_value)
        return {key: float(np.mean(values)) for key, values in buckets.items() if values}

    def _lesson_effect_slug(self, loadout: Any | None) -> tuple[str, str]:
        """把偶像卡主流派映射到 lesson row 使用的效果 slug。"""

        exam_effect_type = ''
        if loadout is not None and getattr(loadout, 'stat_profile', None) is not None:
            exam_effect_type = str(getattr(loadout.stat_profile, 'exam_effect_type', '') or '')
        effect_slug_map = {
            'ProduceExamEffectType_ExamCardPlayAggressive': ('exam_card_play_aggressive_01', 'card_play_aggressive'),
            'ProduceExamEffectType_ExamConcentration': ('exam_concentration_01', 'concentration'),
            'ProduceExamEffectType_ExamFullPower': ('exam_full_power_01', 'full_power'),
            'ProduceExamEffectType_ExamLessonBuff': ('exam_lesson_buff_01', 'lesson_buff'),
            'ProduceExamEffectType_ExamParameterBuff': ('exam_parameter_buff_01', 'parameter_buff'),
            'ProduceExamEffectType_ExamReview': ('exam_review_01', 'review'),
        }
        resolved = effect_slug_map.get(exam_effect_type)
        if resolved is not None:
            return resolved
        return ('exam_parameter_buff_01', 'parameter_buff')

    def resolve_lesson_training_spec(
        self,
        scenario: ScenarioSpec,
        *,
        action_type: str,
        loadout: Any | None = None,
        level_index: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> LessonTrainingSpec:
        """按课程动作类型和偶像流派解析 lesson battle 所需的主数据。"""

        normalized_action_type = str(action_type or '').strip()
        if not normalized_action_type.startswith('lesson_'):
            raise ValueError(f'Unsupported lesson action type for lesson mode: {normalized_action_type}')

        parts = normalized_action_type.split('_')
        if len(parts) < 3:
            raise ValueError(f'Invalid lesson action type: {normalized_action_type}')

        stat_type = parts[1]
        lesson_kind = parts[2]
        stat_to_lesson_type = {
            'vocal': 'ProduceStepLessonType_LessonVocal',
            'dance': 'ProduceStepLessonType_LessonDance',
            'visual': 'ProduceStepLessonType_LessonVisual',
        }
        lesson_type = stat_to_lesson_type.get(stat_type)
        if lesson_type is None:
            raise ValueError(f'Unsupported lesson stat type: {normalized_action_type}')

        effect_slug, legend_effect_slug = self._lesson_effect_slug(loadout)
        prefix_candidates: list[str] = []
        if scenario.produce_id == 'produce-001':
            prefix_candidates.append('p_step_lesson-produce-001')
        elif scenario.produce_id == 'produce-002':
            prefix_candidates.append('p_step_lesson-produce-002')
        elif scenario.produce_id == 'produce-003':
            if lesson_kind == 'hard':
                prefix_candidates.append('p_step_lesson-produce-003')
            prefix_candidates.append('p_step_lesson-produce-002')
        elif scenario.produce_id in {'produce-004', 'produce-005'}:
            prefix_candidates.append('p_step_lesson-produce-002')
        elif scenario.produce_id == 'produce-006':
            if lesson_kind in {'normal', 'sp'}:
                prefix_candidates.append('p_step_lesson-produce_006')
            if lesson_kind == 'hard':
                prefix_candidates.append('p_step_lesson-produce-003')
            prefix_candidates.append('p_step_lesson-produce-002')
        else:
            prefix_candidates.append(f'p_step_lesson-{scenario.produce_id}')

        available_rows: list[tuple[int, dict[str, Any]]] = []
        seen_row_ids: set[str] = set()
        for prefix in prefix_candidates:
            if prefix.endswith('produce_006'):
                stem = f'{prefix}-{legend_effect_slug}-{lesson_kind}-'
            else:
                stem = f'{prefix}-{effect_slug}-{lesson_kind}-'
            for row in self.lesson_rows.rows:
                row_id = str(row.get('id') or '')
                if not row_id.startswith(stem) or row_id in seen_row_ids:
                    continue
                suffix = row_id[len(stem):]
                if not suffix.isdigit():
                    continue
                seen_row_ids.add(row_id)
                available_rows.append((int(suffix), row))
        if not available_rows:
            raise KeyError(
                f'No lesson rows found for scenario={scenario.produce_id}, action_type={normalized_action_type}, effect_slug={effect_slug}'
            )
        available_rows.sort(key=lambda item: (item[0], str(item[1].get('id') or '')))

        resolved_level_index = int(level_index or 0)
        if resolved_level_index > 0:
            matched = next((row for index, row in available_rows if index == resolved_level_index), None)
            if matched is None:
                supported = ', '.join(str(index) for index, _ in available_rows)
                raise KeyError(
                    f'Lesson level {resolved_level_index} is unavailable for {normalized_action_type}; supported=[{supported}]'
                )
            selected_level_index = resolved_level_index
            selected_row = matched
        else:
            sampler = rng or np.random.default_rng(0)
            selected_index = int(sampler.integers(0, len(available_rows)))
            selected_level_index, selected_row = available_rows[selected_index]

        level_id = str(selected_row.get('produceStepLessonLevelId') or '')
        level_row = self.lesson_levels.first(level_id)
        if level_row is None:
            raise KeyError(f'Lesson level row not found: {level_id}')

        lesson_trigger_types = (lesson_type,)
        if lesson_kind == 'sp':
            lesson_trigger_types = (lesson_type, 'ProduceStepLessonType_LessonSp')
        lesson_post_clear_types: tuple[str, ...] = ()
        if lesson_kind == 'hard':
            lesson_post_clear_types = tuple(stat_to_lesson_type.values())

        return LessonTrainingSpec(
            action_type=normalized_action_type,
            lesson_kind=lesson_kind,
            stat_type=stat_type,
            lesson_type=lesson_type,
            lesson_trigger_types=lesson_trigger_types,
            lesson_post_clear_types=lesson_post_clear_types,
            source_row_id=str(selected_row.get('id') or ''),
            source_level_id=level_id,
            name=str(selected_row.get('name') or normalized_action_type),
            clear_target=float(level_row.get('successThreshold') or 0.0),
            perfect_target=float(level_row.get('resultTargetValueLimit') or 0.0),
            turn_limit=int(level_row.get('limitTurn') or 0),
            level_index=selected_level_index,
        )

    @cached_property
    def produce_effect_stats(self) -> dict[str, dict[str, float]]:
        """按培养效果类型汇总主数据里的量级统计。"""

        buckets: dict[str, list[float]] = defaultdict(list)
        for row in self.produce_effects.rows:
            effect_type = str(row.get('produceEffectType'))
            values = [
                row.get('effectValueMin') or 0,
                row.get('effectValueMax') or 0,
                row.get('pickCountMin') or 0,
                row.get('pickCountMax') or 0,
            ]
            non_zero = [abs(float(value)) for value in values if value not in (None, 0, '')]
            magnitude = float(np.mean(non_zero)) if non_zero else 1.0
            buckets[effect_type].append(magnitude)
        stats = {}
        for effect_type, values in buckets.items():
            stats[effect_type] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
            }
        return stats

    @cached_property
    def exam_effect_priors(self) -> dict[tuple[str, int], float]:
        """从客户端自动估值表提取考试效果先验分。"""

        priors: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in self.auto_evaluations.rows:
            key = (str(row.get('examEffectType')), int(row.get('remainingTerm') or 0))
            priors[key].append(float(row.get('evaluation') or 0))
        return {key: float(np.mean(values)) for key, values in priors.items()}

    @cached_property
    def trigger_priors(self) -> dict[str, float]:
        """从自动估值表提取触发器系数先验。"""

        priors: dict[str, list[float]] = defaultdict(list)
        for row in self.auto_trigger_evaluations.rows:
            trigger_id = str(row.get('examStatusEnchantProduceExamTriggerId') or '')
            if not trigger_id:
                continue
            priors[trigger_id].append(float(row.get('coefficientPermil') or 0) / 1000.0)
        return {key: float(np.mean(values)) for key, values in priors.items()}

    @cached_property
    def card_play_priors(self) -> dict[str, float]:
        """从客户端自动出牌估值表提取卡牌先验。"""

        priors: dict[str, list[float]] = defaultdict(list)
        for row in self.auto_play_card_evaluations.rows:
            priors[str(row.get('produceCardId'))].append(float(row.get('evaluation') or 0))
        return {key: float(np.mean(values)) for key, values in priors.items()}

    @cached_property
    def stage_thresholds(self) -> dict[tuple[str, str], dict[str, float]]:
        """聚合同场景同阶段的考试阈值、回合数和属性权重。"""

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.audition_difficulties.rows:
            grouped[(str(row.get('produceId')), str(row.get('stepType')))].append(row)
        aggregated: dict[tuple[str, str], dict[str, float]] = {}
        for key, rows in grouped.items():
            battle_turns = []
            battle_focus = []
            base_scores = [float(row.get('baseScore') or 0) for row in rows]
            force_scores = [float(row.get('forceEndScore') or 0) for row in rows]
            rank_thresholds = [float(row.get('rankThreshold') or 0) for row in rows]
            parameter_baselines = [float(row.get('parameterBaseLine') or 0) for row in rows]
            fan_vote_baselines = [float(row.get('voteCountBaseLine') or 0) for row in rows]
            fan_vote_requirements = [float(row.get('voteCount') or 0) for row in rows]
            for row in rows:
                config = self.battle_config_map.get(str(row.get('produceExamBattleConfigId') or ''))
                if config:
                    battle_turns.append(float(config.get('turn') or 0))
                    battle_focus.append(
                        [
                            float(config.get('vocal') or 0),
                            float(config.get('dance') or 0),
                            float(config.get('visual') or 0),
                        ]
                    )
            weight_vector = np.mean(battle_focus, axis=0) if battle_focus else np.array([1.0, 1.0, 1.0])
            weight_vector = weight_vector / max(float(weight_vector.sum()), 1.0)
            aggregated[key] = {
                'base_score': float(np.median(base_scores)) if base_scores else 0.0,
                'force_end_score': float(np.median(force_scores)) if force_scores else 0.0,
                'rank_threshold': float(np.median(rank_thresholds)) if rank_thresholds else 0.0,
                'parameter_baseline': float(np.median(parameter_baselines)) if parameter_baselines else 0.0,
                'fan_vote_baseline': float(np.median(fan_vote_baselines)) if fan_vote_baselines else 0.0,
                'fan_vote_requirement': float(np.median(fan_vote_requirements)) if fan_vote_requirements else 0.0,
                'turns': float(np.median(battle_turns)) if battle_turns else 9.0,
                'vocal_weight': float(weight_vector[0]),
                'dance_weight': float(weight_vector[1]),
                'visual_weight': float(weight_vector[2]),
            }
        return aggregated

    @cached_property
    def taxonomy(self) -> EffectTaxonomy:
        """从主数据自动抽取训练使用的统一离散词表。"""

        exam_effect_types: set[str] = set()
        produce_effect_types: set[str] = set()
        trigger_phases: set[str] = set()
        card_categories: set[str] = set()
        card_rarities: set[str] = set()
        card_cost_types: set[str] = set()

        for row in self.effect_groups.rows:
            exam_effect_types.update(str(value) for value in row.get('examEffectTypes', []) if value)
            produce_effect_types.update(str(value) for value in row.get('produceEffectTypes', []) if value)

        for row in self.exam_effects.rows:
            effect_type = row.get('effectType')
            if effect_type:
                exam_effect_types.add(str(effect_type))

        for row in self.produce_effects.rows:
            effect_type = row.get('produceEffectType')
            if effect_type:
                produce_effect_types.add(str(effect_type))

        for row in self.exam_triggers.rows:
            trigger_phases.update(str(value) for value in row.get('phaseTypes', []) if value)

        for row in self.produce_cards.rows:
            category = row.get('category')
            rarity = row.get('rarity')
            cost_type = row.get('costType')
            if category:
                card_categories.add(str(category))
            if rarity:
                card_rarities.add(str(rarity))
            if cost_type:
                card_cost_types.add(str(cost_type))

        shop_card_action_types = tuple(f'shop_buy_card_{index}' for index in range(1, 5))
        shop_drink_action_types = tuple(f'shop_buy_drink_{index}' for index in range(1, 5))
        shop_upgrade_action_types = tuple(f'shop_upgrade_card_{index}' for index in range(1, 5))
        shop_delete_action_types = tuple(f'shop_delete_card_{index}' for index in range(1, 5))
        action_types = (
            'lesson_vocal_normal',
            'lesson_dance_normal',
            'lesson_visual_normal',
            'lesson_vocal_sp',
            'lesson_dance_sp',
            'lesson_visual_sp',
            'lesson_vocal_hard',
            'lesson_dance_hard',
            'lesson_visual_hard',
            'self_lesson_vocal_normal',
            'self_lesson_vocal_sp',
            'self_lesson_dance_normal',
            'self_lesson_dance_sp',
            'self_lesson_visual_normal',
            'self_lesson_visual_sp',
            'activity',
            'business',
            'present',
            'refresh',
            *shop_card_action_types,
            *shop_drink_action_types,
            *shop_upgrade_action_types,
            *shop_delete_action_types,
            'pre_audition_continue',
            'drink',
            'card',
            'end_turn',
        )
        return EffectTaxonomy(
            exam_effect_types=tuple(sorted(exam_effect_types)),
            produce_effect_types=tuple(sorted(produce_effect_types)),
            trigger_phases=tuple(sorted(trigger_phases)),
            action_types=action_types,
            card_categories=tuple(sorted(card_categories)),
            card_rarities=tuple(sorted(card_rarities)),
            card_cost_types=tuple(sorted(card_cost_types)),
        )

    def produce_name(self, produce_id: str) -> str:
        """返回培养路线的本地化名称。"""

        row = self.produces.first(produce_id)
        if not row:
            return produce_id
        loc = self.produce_localization.get(produce_id, {})
        return str(loc.get('name') or row.get('name') or produce_id)

    @cached_property
    def produce_card_localization(self) -> dict[str, dict[str, Any]]:
        return self.load_localization('ProduceCard')

    @cached_property
    def produce_drink_localization(self) -> dict[str, dict[str, Any]]:
        return self.load_localization('ProduceDrink')

    @cached_property
    def produce_item_localization(self) -> dict[str, dict[str, Any]]:
        return self.load_localization('ProduceItem')

    def card_name(self, card: dict[str, Any]) -> str:
        """返回卡牌的优先本地化名称（区分升级等级）。"""

        item_id = str(card.get('id') or '')
        upgrade_count = card.get('upgradeCount')
        # 优先使用 "{id}.{upgradeCount}" 复合键查找精确升级等级的翻译
        if upgrade_count is not None:
            loc = self.produce_card_localization.get(f'{item_id}.{int(upgrade_count)}', {})
            if loc:
                return str(loc.get('name') or card.get('name') or item_id)
        loc = self.produce_card_localization.get(item_id, {})
        return str(loc.get('name') or card.get('name') or item_id)

    def raw_card_name(self, card: dict[str, Any]) -> str:
        """返回卡牌主数据库中的原始名称。"""

        item_id = str(card.get('id') or '')
        return str(card.get('name') or item_id)

    def canonical_card_row(self, card_id: str) -> dict[str, Any] | None:
        """返回该卡 id 的最低强化版本。"""

        normalized_card_id = str(card_id or '')
        cached = self._canonical_card_row_cache.get(normalized_card_id)
        if cached is not None or normalized_card_id in self._canonical_card_row_cache:
            return cached
        rows = self.produce_cards.all(normalized_card_id)
        resolved = min(rows, key=_upgrade_count) if rows else None
        self._canonical_card_row_cache[normalized_card_id] = resolved
        return resolved

    def card_row_by_upgrade(self, card_id: str, upgrade_count: int, *, fallback_to_canonical: bool = True) -> dict[str, Any] | None:
        """按 card id + upgradeCount 查找具体卡面。"""

        normalized_card_id = str(card_id or '')
        upgrade_map = self._card_row_by_upgrade_cache.get(normalized_card_id)
        if upgrade_map is None:
            upgrade_map = {
                _upgrade_count(row): row
                for row in self.produce_cards.all(normalized_card_id)
            }
            self._card_row_by_upgrade_cache[normalized_card_id] = upgrade_map
        resolved = upgrade_map.get(int(upgrade_count))
        if resolved is not None:
            return resolved
        if fallback_to_canonical:
            return self.canonical_card_row(normalized_card_id)
        return None

    def sample_random_card_variant(self, card_id: str, rng: np.random.Generator) -> dict[str, Any] | None:
        """为随机获得的新卡按常见强化分布抽样 upgradeCount。"""

        normalized_card_id = str(card_id or '')
        cached = self._card_variant_sampling_cache.get(normalized_card_id)
        if cached is None:
            rows = tuple(sorted(self.produce_cards.all(normalized_card_id), key=_upgrade_count))
            if not rows:
                return None
            weights = np.array([_random_upgrade_weight(_upgrade_count(row)) for row in rows], dtype=np.float64)
            weights = weights / max(weights.sum(), 1e-8)
            cached = (rows, weights)
            self._card_variant_sampling_cache[normalized_card_id] = cached
        rows, weights = cached
        if not rows:
            return None
        return rows[int(rng.choice(len(rows), p=weights))]

    def card_axis_effect_types(self, card: dict[str, Any]) -> list[str]:
        """将原始 effectType 投影为用户可理解的轴标签。"""

        return list(summarize_exam_effect_axes(self.card_exam_effect_types(card)))

    def drink_name(self, drink: dict[str, Any]) -> str:
        """返回饮料的优先本地化名称。"""

        item_id = str(drink.get('id') or '')
        loc = self.produce_drink_localization.get(item_id, {})
        return str(loc.get('name') or drink.get('name') or item_id)

    def raw_drink_name(self, drink: dict[str, Any]) -> str:
        """返回饮料主数据库中的原始名称。"""

        item_id = str(drink.get('id') or '')
        return str(drink.get('name') or item_id)

    def drink_axis_effect_types(self, drink: dict[str, Any]) -> list[str]:
        """将饮料原始 effectType 投影为用户可理解的轴标签。"""

        return list(summarize_exam_effect_axes(self.drink_exam_effect_types(drink)))

    def item_name(self, item: dict[str, Any]) -> str:
        """返回 P 道具的优先本地化名称。"""

        item_id = str(item.get('id') or '')
        loc = self.produce_item_localization.get(item_id, {})
        return str(loc.get('name') or item.get('name') or item_id)

    def raw_item_name(self, item: dict[str, Any]) -> str:
        """返回 P 道具主数据库中的原始名称。"""

        item_id = str(item.get('id') or '')
        return str(item.get('name') or item_id)

    def build_scenario(self, scenario_id: str) -> ScenarioSpec:
        """把主数据里的 `Produce` 配置装配成训练场景对象。"""

        produce = self.produces.first(scenario_id)
        if produce is None:
            raise KeyError(f'Unknown produce id: {scenario_id}')
        group = next(
            (row for row in self.produce_groups.rows if scenario_id in row.get('produceIds', [])),
            None,
        )
        if group is None:
            raise KeyError(f'Unable to find produce group for {scenario_id}')
        setting = self.produce_settings.first(str(produce.get('produceSettingId')))
        route_type = 'nia' if str(group.get('type')) == 'ProduceType_NextIdolAudition' else 'first_star'
        audition_sequence = (
            'ProduceStepType_AuditionMid1',
            'ProduceStepType_AuditionMid2',
            'ProduceStepType_AuditionFinal',
        ) if route_type == 'nia' else (
            'ProduceStepType_AuditionMid1',
            'ProduceStepType_AuditionFinal',
        )
        action_types = (
            'lesson_vocal_normal',
            'lesson_dance_normal',
            'lesson_visual_normal',
            'lesson_vocal_sp',
            'lesson_dance_sp',
            'lesson_visual_sp',
            'refresh',
        )
        if route_type == 'first_star':
            action_types = action_types + (
                'lesson_vocal_hard',
                'lesson_dance_hard',
                'lesson_visual_hard',
            )
        if route_type == 'nia':
            shop_card_action_types = tuple(f'shop_buy_card_{index}' for index in range(1, 5))
            shop_drink_action_types = tuple(f'shop_buy_drink_{index}' for index in range(1, 5))
            shop_upgrade_action_types = tuple(f'shop_upgrade_card_{index}' for index in range(1, 5))
            shop_delete_action_types = tuple(f'shop_delete_card_{index}' for index in range(1, 5))
            action_types = action_types + (
                'self_lesson_vocal_normal',
                'self_lesson_vocal_sp',
                'self_lesson_dance_normal',
                'self_lesson_dance_sp',
                'self_lesson_visual_normal',
                'self_lesson_visual_sp',
                'activity',
                'business',
                'present',
                *shop_card_action_types,
                *shop_drink_action_types,
                *shop_upgrade_action_types,
                *shop_delete_action_types,
                'pre_audition_continue',
            )
        first_stage = self.stage_thresholds.get((scenario_id, audition_sequence[0]), {})
        weight_vector = np.array(
            [
                first_stage.get('vocal_weight', 1.0 / 3.0),
                first_stage.get('dance_weight', 1.0 / 3.0),
                first_stage.get('visual_weight', 1.0 / 3.0),
            ],
            dtype=np.float32,
        )
        return ScenarioSpec(
            scenario_id=scenario_id,
            produce_id=scenario_id,
            produce_name=self.produce_name(scenario_id),
            group_id=str(group.get('id')),
            route_type=route_type,
            parameter_growth_limit=float(produce.get('idolCardParameterGrowthLimit') or 0.0),
            steps=int(produce.get('steps') or 0),
            action_point_quantity=int(produce.get('actionPointQuantity') or 0),
            max_refresh_count=int(produce.get('maxRefreshCount') or 0),
            drink_limit=int((setting or {}).get('produceDrinkPossessLimit') or 3),
            audition_sequence=audition_sequence,
            action_types=action_types,
            focus_effect_types=(
                'ProduceExamEffectType_ExamParameterBuff',
                'ProduceExamEffectType_ExamLessonBuff',
                'ProduceExamEffectType_ExamReview',
                'ProduceExamEffectType_ExamCardPlayAggressive',
                'ProduceExamEffectType_ExamConcentration',
            ),
            score_weights=(float(weight_vector[0]), float(weight_vector[1]), float(weight_vector[2])),
            exam_turns=int(first_stage.get('turns') or 9),
            default_stage=audition_sequence[-1],
            reward_weights={
                'stat': 0.0025 if route_type == 'nia' else 0.003,
                'vote': 0.0012 if route_type == 'nia' else 0.0003,
                'points': 0.0010,
                'deck': 0.09,
                'drink': 0.06,
                'stamina': 0.015,
                'clear': 1.2 if route_type == 'nia' else 0.9,
            },
        )

    def list_supported_scenarios(self) -> list[str]:
        """列出当前训练代码显式支持的场景 id。"""

        return [row['id'] for row in self.produces.rows if row.get('id') in {'produce-001', 'produce-002', 'produce-003', 'produce-004', 'produce-005', 'produce-006'}]

    def _resolve_effect_group_types(self, effect_group_ids: Iterable[str], key: str) -> list[str]:
        values: list[str] = []
        for group_id in effect_group_ids:
            group = self.effect_group_map.get(str(group_id))
            if group:
                values.extend(str(value) for value in group.get(key, []) if value)
        return values

    @cached_property
    def interval_phase_values(self) -> dict[str, tuple[int, ...]]:
        """按 phase_type 预编译所有正整数 interval phase 值，避免运行时反复全表扫描。"""

        phase_type_to_values: dict[str, set[int]] = defaultdict(set)
        for trigger in self.exam_triggers.rows:
            phase_types = [str(item) for item in trigger.get('phaseTypes', []) if item]
            positive_values = [int(value) for value in trigger.get('phaseValues', []) if int(value or 0) > 0]
            if not phase_types or not positive_values:
                continue
            for phase_type in phase_types:
                phase_type_to_values[phase_type].update(positive_values)
        return {
            phase_type: tuple(sorted(values))
            for phase_type, values in phase_type_to_values.items()
        }

    def card_exam_effect_types(self, card: dict[str, Any]) -> list[str]:
        """汇总单张卡牌显式和间接声明的考试效果类型。"""

        cache_key = _card_cache_key(card)
        cached = self._card_exam_effect_types_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        types: list[str] = []
        for effect_item in card.get('playEffects', []):
            effect_id = str(effect_item.get('produceExamEffectId') or '')
            effect = self.exam_effect_map.get(effect_id)
            if effect and effect.get('effectType'):
                types.append(str(effect['effectType']))
        for effect_id in card.get('moveProduceExamEffectIds', []):
            effect = self.exam_effect_map.get(str(effect_id))
            if effect and effect.get('effectType'):
                types.append(str(effect['effectType']))
        types.extend(self._resolve_effect_group_types(card.get('effectGroupIds', []), 'examEffectTypes'))
        cached_types = tuple(types)
        self._card_exam_effect_types_cache[cache_key] = cached_types
        return list(cached_types)

    def card_trigger_phases(self, card: dict[str, Any]) -> list[str]:
        """返回卡牌主动触发器声明的 phase 列表。"""

        cache_key = _card_cache_key(card)
        cached = self._card_trigger_phases_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        trigger = self.exam_trigger_map.get(str(card.get('playProduceExamTriggerId') or ''))
        if not trigger:
            return []
        cached_phases = tuple(str(value) for value in trigger.get('phaseTypes', []) if value)
        self._card_trigger_phases_cache[cache_key] = cached_phases
        return list(cached_phases)

    def drink_exam_effect_types(self, drink: dict[str, Any]) -> list[str]:
        """汇总饮料显式和间接声明的考试效果类型。"""

        drink_id = str(drink.get('id') or '')
        cached = self._drink_exam_effect_types_cache.get(drink_id)
        if cached is not None:
            return list(cached)

        types: list[str] = []
        for drink_effect_id in drink.get('produceDrinkEffectIds', []):
            drink_effect = self.drink_effect_map.get(str(drink_effect_id))
            if not drink_effect:
                continue
            exam_effect = self.exam_effect_map.get(str(drink_effect.get('produceExamEffectId') or ''))
            if exam_effect and exam_effect.get('effectType'):
                types.append(str(exam_effect['effectType']))
        types.extend(self._resolve_effect_group_types(drink.get('effectGroupIds', []), 'examEffectTypes'))
        cached_types = tuple(types)
        if drink_id:
            self._drink_exam_effect_types_cache[drink_id] = cached_types
        return list(cached_types)

    def weighted_card_pool(
        self,
        scenario: ScenarioSpec,
        focus_effect_type: str | None = None,
        plan_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """构造默认考试卡池，并按主数据先验做降序排序。"""

        cache_key = (scenario.produce_id, str(focus_effect_type or ''), str(plan_type or ''))
        cached = self._weighted_card_pool_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        candidates = []
        allowed_plan_types = _allowed_plan_types(plan_type)
        for card_id, rows in self.produce_cards.by_id.items():
            if not card_id:
                continue
            card = min(rows, key=lambda row: int(row.get('upgradeCount') or 0))
            if card.get('libraryHidden'):
                continue
            if str(card.get('planType') or 'ProducePlanType_Common') not in allowed_plan_types:
                continue
            if card.get('isLimited'):
                continue
            effect_types = self.card_exam_effect_types(card)
            score = float(card.get('evaluation') or 0) / 100.0
            score += self.card_play_priors.get(str(card.get('id')), 0.0) / 100.0
            if focus_effect_type and focus_effect_type in effect_types:
                score += 3.0
            if any(value in effect_types for value in scenario.focus_effect_types):
                score += 1.0
            candidates.append((score, card))
        candidates.sort(key=lambda item: item[0], reverse=True)
        cached_pool = tuple(card for _, card in candidates)
        self._weighted_card_pool_cache[cache_key] = cached_pool
        return list(cached_pool)

    def build_initial_exam_deck(
        self,
        scenario: ScenarioSpec,
        focus_effect_type: str | None = None,
        deck_size: int = 15,
        rng: np.random.Generator | None = None,
        plan_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """从初始卡组主数据和权重卡池拼装默认考试牌组。"""

        rng = rng or np.random.default_rng()
        initial_rows = [
            row
            for row in self.produce_initial_decks.rows
            if str(row.get('produceId')) == scenario.produce_id
        ]
        if focus_effect_type:
            preferred = [row for row in initial_rows if str(row.get('examEffectType')) == focus_effect_type]
            if preferred:
                initial_rows = preferred
        if not initial_rows:
            raise RuntimeError(f'No initial deck found for {scenario.produce_id}')
        base_row = initial_rows[int(rng.integers(0, len(initial_rows)))]
        exam_initial_deck = self.exam_initial_decks.first(str(base_row.get('examInitialDeckId')))
        deck: list[dict[str, Any]] = []
        for card_id in exam_initial_deck.get('produceCardIds', []):
            card = self.canonical_card_row(str(card_id))
            if card:
                deck.append(card)
        weighted_pool = self.weighted_card_pool(scenario, focus_effect_type, plan_type=plan_type)
        seen_keys = Counter(str(card.get('id')) for card in deck)
        rank_weights = np.array([1.0 / math.sqrt(index + 1.0) for index in range(len(weighted_pool))], dtype=np.float64)
        while len(deck) < deck_size and weighted_pool:
            available_indices: list[int] = []
            available_weights: list[float] = []
            for index, card in enumerate(weighted_pool):
                card_id = str(card.get('id'))
                if card.get('noDeckDuplication') and seen_keys[card_id] > 0:
                    continue
                if seen_keys[card_id] >= 2:
                    continue
                available_indices.append(index)
                available_weights.append(float(rank_weights[index]))
            if not available_indices:
                break
            probabilities = np.array(available_weights, dtype=np.float64)
            probabilities = probabilities / max(probabilities.sum(), 1e-8)
            selected_index = int(rng.choice(len(available_indices), p=probabilities))
            selected_card = weighted_pool[available_indices[selected_index]]
            sampled_variant = self.sample_random_card_variant(str(selected_card.get('id') or ''), rng)
            if sampled_variant is not None:
                selected_card = sampled_variant
            deck.append(selected_card)
            seen_keys[str(selected_card.get('id'))] += 1
        return deck[:deck_size]

    def build_drink_inventory(
        self,
        scenario: ScenarioSpec,
        max_items: int | None = None,
        rng: np.random.Generator | None = None,
        plan_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """为场景分层抽样初始饮料库存。

        按质量排序后，前 1/3 中取 ceil(limit/2) 瓶，剩余从后 2/3 中取，
        保证至少有 1 瓶强饮料，减少全取弱饮料的灾难性情况。
        """

        rng = rng or np.random.default_rng()
        limit = max_items or scenario.drink_limit
        allowed_plan_types = _allowed_plan_types(plan_type)
        drinks = [
            row
            for row in self.produce_drinks.rows
            if not row.get('libraryHidden')
            and str(row.get('planType') or 'ProducePlanType_Common') in allowed_plan_types
        ]
        drinks.sort(key=lambda row: (len(self.drink_exam_effect_types(row)), row.get('rarity', '')), reverse=True)
        if len(drinks) <= limit:
            return drinks

        # 分层抽样：前 1/3 为强饮料，后 2/3 为普通饮料
        top_boundary = max(len(drinks) // 3, 1)
        top_tier = drinks[:top_boundary]
        rest_tier = drinks[top_boundary:]

        top_count = min(math.ceil(limit / 2), len(top_tier))
        rest_count = min(limit - top_count, len(rest_tier))

        top_indices = rng.choice(len(top_tier), size=top_count, replace=False)
        selected = [top_tier[int(i)] for i in top_indices]

        if rest_count > 0 and len(rest_tier) > 0:
            rest_indices = rng.choice(len(rest_tier), size=rest_count, replace=False)
            selected.extend(rest_tier[int(i)] for i in rest_indices)

        return selected

    def audition_rows(
        self,
        scenario: ScenarioSpec,
        stage_type: str | None = None,
        audition_difficulty_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """返回指定场景和阶段对应的考试难度行。"""

        stage = stage_type or scenario.default_stage
        rows = [
            row
            for row in self.audition_difficulties.rows
            if str(row.get('produceId')) == scenario.produce_id and str(row.get('stepType')) == stage
        ]
        if audition_difficulty_id:
            filtered = [row for row in rows if str(row.get('id') or '') == audition_difficulty_id]
            if filtered:
                rows = filtered
        rows.sort(key=lambda row: int(row.get('number') or 0))
        return rows

    def select_audition_row(
        self,
        scenario: ScenarioSpec,
        stage_type: str | None = None,
        audition_difficulty_id: str | None = None,
        fan_votes: float | None = None,
    ) -> dict[str, Any] | None:
        """按 fan vote 门槛选择当前可进入的考试难度行。"""

        rows = self.audition_rows(scenario, stage_type, audition_difficulty_id=audition_difficulty_id)
        if not rows:
            return None
        if fan_votes is None or str(getattr(scenario, 'route_type', '') or '') != 'nia':
            return rows[0]
        accessible = [
            row
            for row in rows
            if float(row.get('voteCount') or 0.0) <= max(float(fan_votes), 0.0)
        ]
        if accessible:
            return accessible[-1]
        return rows[0]

    def battle_profile(
        self,
        scenario: ScenarioSpec,
        stage_type: str | None = None,
        audition_difficulty_id: str | None = None,
    ) -> dict[str, float]:
        """解析阶段考试使用的分数阈值、回合数和属性权重。"""

        stage = stage_type or scenario.default_stage
        rows = self.audition_rows(scenario, stage, audition_difficulty_id=audition_difficulty_id)
        if rows:
            selected = rows[len(rows) // 2]
            config = self.battle_config_map.get(str(selected.get('produceExamBattleConfigId') or '')) or {}
            weight_vector = np.array(
                [
                    float(config.get('vocal') or scenario.score_weights[0]),
                    float(config.get('dance') or scenario.score_weights[1]),
                    float(config.get('visual') or scenario.score_weights[2]),
                ],
                dtype=np.float32,
            )
            weight_sum = float(weight_vector.sum())
            if weight_sum > 0:
                weight_vector = weight_vector / weight_sum
            return {
                'base_score': float(selected.get('baseScore') or 0.0),
                'force_end_score': float(selected.get('forceEndScore') or 0.0),
                'rank_threshold': float(selected.get('rankThreshold') or 0.0),
                'parameter_baseline': float(selected.get('parameterBaseLine') or 0.0),
                'fan_vote_baseline': float(selected.get('voteCountBaseLine') or 0.0),
                'fan_vote_requirement': float(selected.get('voteCount') or 0.0),
                'turns': float(config.get('turn') or scenario.exam_turns),
                'vocal_weight': float(weight_vector[0]),
                'dance_weight': float(weight_vector[1]),
                'visual_weight': float(weight_vector[2]),
            }
        profile = dict(self.stage_thresholds.get((scenario.produce_id, stage), {}))
        if not profile:
            profile = {
                'base_score': 2000.0,
                'force_end_score': 0.0,
                'rank_threshold': 3.0,
                'parameter_baseline': 180.0 if scenario.route_type == 'nia' else 140.0,
                'fan_vote_baseline': 0.0,
                'fan_vote_requirement': 0.0,
                'turns': float(scenario.exam_turns),
                'vocal_weight': scenario.score_weights[0],
                'dance_weight': scenario.score_weights[1],
                'visual_weight': scenario.score_weights[2],
            }
        return profile

    def exam_effect_value(self, effect: dict[str, Any]) -> float:
        """估算单个考试效果的数值量级，用于启发式打分。"""

        values = [
            effect.get('effectValue1') or 0,
            effect.get('effectValue2') or 0,
            effect.get('effectCount') or 0,
            effect.get('effectTurn') or 0,
            effect.get('pickCountMin') or 0,
            effect.get('pickCountMax') or 0,
        ]
        values = [abs(float(value)) for value in values if value not in (None, '', 0)]
        return float(np.mean(values)) if values else 1.0
