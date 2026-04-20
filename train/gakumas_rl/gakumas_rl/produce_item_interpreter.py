"""Data-driven interpreter for produce-phase P items and challenge items."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .data import MasterDataRepository, ScenarioSpec

_PARAMETER_THRESHOLD_RE = re.compile(r'-(vocal|dance|visual)-(\d{4})_0000')
_STAMINA_RATIO_RE = re.compile(r'-stamina_ratio-(\d{4})_0000')
_PRODUCE_CARD_COUNT_RE = re.compile(r'-produce_card_count-(\d{4})_0000')

_LESSON_TRIGGER_TOKENS = (
    'lesson_vocal_sp',
    'lesson_dance_sp',
    'lesson_visual_sp',
    'lesson_vocal',
    'lesson_dance',
    'lesson_visual',
    'lesson_sp',
    'lesson_hard',
    'lesson',
)

SUPPORTED_RUNTIME_PRODUCE_ITEM_PHASES = frozenset(
    {
        'ProducePhaseType_ProduceStart',
        'ProducePhaseType_StartShop',
        'ProducePhaseType_StartCustomize',
        'ProducePhaseType_EndShop',
        'ProducePhaseType_BuyShopItemProduceDrink',
        'ProducePhaseType_CustomizeProduceCard',
        'ProducePhaseType_StartLesson',
        'ProducePhaseType_EndLesson',
        'ProducePhaseType_EndLessonBeforePresent',
        'ProducePhaseType_StartPresent',
        'ProducePhaseType_EndPresent',
        'ProducePhaseType_EndStepEventSchool',
        'ProducePhaseType_EndStepEventActivity',
        'ProducePhaseType_EndStepEventBusiness',
        'ProducePhaseType_EndBeforeAuditionRefresh',
        'ProducePhaseType_StartAudition',
        'ProducePhaseType_StartAuditionMid1',
        'ProducePhaseType_StartAuditionMid2',
        'ProducePhaseType_StartAuditionFinal',
        'ProducePhaseType_EndAudition',
        'ProducePhaseType_StartRefresh',
        'ProducePhaseType_GetProduceCard',
        'ProducePhaseType_UpgradeProduceCard',
        'ProducePhaseType_DeleteProduceCard',
        'ProducePhaseType_ChangeProduceCard',
        'ProducePhaseType_GetProduceDrink',
        'ProducePhaseType_GetProduceItem',
    }
)

SUPPORTED_SCENARIO_TAGS = frozenset({'', 'nia_master', 'hajime_legend', 'hajime_legend_ssr'})


@dataclass(frozen=True)
class RuntimeExamStatusEnchantSpec:
    """Structured enchant handoff from produce runtime to exam runtime."""

    enchant_id: str
    effect_turn: int | None = None
    effect_count: int | None = None
    source: str = 'produce'
    source_identity: str = ''


@dataclass(frozen=True)
class ResolvedProduceItemEffect:
    """Resolved wrapper row for a single produce item effect."""

    item_effect_id: str
    effect_type: str
    produce_effect_id: str = ''
    enchant_id: str = ''
    effect_turn: int | None = None
    effect_count: int | None = None


@dataclass(frozen=True)
class ResolvedProduceItemTrigger:
    """Parsed trigger contract for a produce item."""

    trigger_id: str
    phase_type: str
    lesson_filter: str = ''
    card_search_id: str = ''
    business_reward_kind: str = ''
    scenario_tag: str = ''
    parameter_thresholds: tuple[tuple[str, float], ...] = ()
    stamina_ratio_min: float | None = None
    produce_card_count_min: int | None = None


@dataclass(frozen=True)
class ResolvedProduceItem:
    """Master-data-backed produce item description used by runtime."""

    item_id: str
    name: str
    trigger: ResolvedProduceItemTrigger | None
    effects: tuple[ResolvedProduceItemEffect, ...]
    fire_limit: int
    fire_interval: int
    is_challenge: bool = False
    is_exam_effect: bool = False


@dataclass
class ActiveProduceItem:
    """Runtime state for an owned produce item."""

    spec: ResolvedProduceItem
    source: str = 'loadout'
    fire_count: int = 0
    cooldown_remaining: int = 0

    @property
    def item_id(self) -> str:
        return self.spec.item_id

    @property
    def trigger(self) -> ResolvedProduceItemTrigger | None:
        return self.spec.trigger


class ProduceItemInterpreter:
    """Resolve produce items from master data and match trigger conditions."""

    def __init__(self, repository: MasterDataRepository):
        self.repository = repository
        self.item_table = repository.load_table('ProduceItem')
        self.item_effect_table = repository.load_table('ProduceItemEffect')
        self.produce_effect_table = repository.load_table('ProduceEffect')
        self.produce_trigger_table = repository.load_table('ProduceTrigger')
        self.card_search_table = repository.load_table('ProduceCardSearch')
        self.effect_group_table = repository.load_table('EffectGroup')
        self._resolved_items: dict[str, ResolvedProduceItem | None] = {}
        self._card_search_ids = sorted(
            (str(row.get('id') or '') for row in self.card_search_table.rows if str(row.get('id') or '')),
            key=len,
            reverse=True,
        )

    def resolve_item(self, item_id: str) -> ResolvedProduceItem | None:
        """Resolve one `ProduceItem` row into a reusable runtime spec."""

        cached = self._resolved_items.get(item_id)
        if item_id in self._resolved_items:
            return cached
        item_row = self.item_table.first(item_id)
        if item_row is None:
            self._resolved_items[item_id] = None
            return None
        trigger_id = str(item_row.get('produceTriggerId') or '')
        trigger = self._parse_trigger(trigger_id) if trigger_id else None
        effect_ids = []
        for skill in item_row.get('skills', []) or []:
            effect_id = str(skill.get('produceItemEffectId') or '')
            if effect_id:
                effect_ids.append(effect_id)
        if not effect_ids:
            effect_ids = [str(value) for value in item_row.get('produceItemEffectIds', []) if value]
        effects: list[ResolvedProduceItemEffect] = []
        for effect_id in effect_ids:
            row = self.item_effect_table.first(effect_id)
            if not row:
                continue
            effect_turn = int(row.get('effectTurn') or 0)
            effect_count = int(row.get('effectCount') or 0)
            effects.append(
                ResolvedProduceItemEffect(
                    item_effect_id=effect_id,
                    effect_type=str(row.get('effectType') or ''),
                    produce_effect_id=str(row.get('produceEffectId') or ''),
                    enchant_id=str(row.get('produceExamStatusEnchantId') or ''),
                    effect_turn=None if effect_turn < 0 else effect_turn,
                    effect_count=None if effect_count <= 0 else effect_count,
                )
            )
        resolved = ResolvedProduceItem(
            item_id=item_id,
            name=str(item_row.get('name') or item_id),
            trigger=trigger,
            effects=tuple(effects),
            fire_limit=max(int(item_row.get('fireLimit') or 0), 0),
            fire_interval=max(int(item_row.get('fireInterval') or 0), 0),
            is_challenge=bool(item_row.get('isChallenge')),
            is_exam_effect=bool(item_row.get('isExamEffect')),
        )
        self._resolved_items[item_id] = resolved
        return resolved

    def activate_item(self, item_id: str, *, source: str = 'loadout') -> ActiveProduceItem | None:
        """Create runtime state for an owned item."""

        resolved = self.resolve_item(item_id)
        if resolved is None:
            return None
        return ActiveProduceItem(spec=resolved, source=source)

    def card_matches_search(self, card: dict[str, Any], search_id: str) -> bool:
        """Check whether a single card row matches a `ProduceCardSearch`."""

        if not search_id:
            return True
        search = self.card_search_table.first(search_id)
        if not search:
            return False
        produce_card_ids = {str(value) for value in search.get('produceCardIds', []) if value}
        if produce_card_ids and str(card.get('id') or '') not in produce_card_ids:
            return False
        categories = {str(value) for value in search.get('cardCategories', []) if value}
        if categories and str(card.get('category') or '') not in categories:
            return False
        rarities = {str(value) for value in search.get('cardRarities', []) if value}
        if rarities and str(card.get('rarity') or '') not in rarities:
            return False
        upgrade_counts = {int(value) for value in search.get('upgradeCounts', []) if value is not None}
        if upgrade_counts and int(card.get('upgradeCount') or 0) not in upgrade_counts:
            return False
        plan_type = str(search.get('planType') or 'ProducePlanType_Unknown')
        if plan_type != 'ProducePlanType_Unknown' and str(card.get('planType') or '') != plan_type:
            return False
        search_tag = str(search.get('cardSearchTag') or '')
        if search_tag and str(card.get('searchTag') or '') != search_tag:
            return False
        cost_type = str(search.get('costType') or 'ExamCostType_Unknown')
        if cost_type != 'ExamCostType_Unknown' and str(card.get('costType') or '') != cost_type:
            return False
        effect_group_ids = {str(value) for value in search.get('effectGroupIds', []) if value}
        if effect_group_ids:
            card_group_ids = {str(value) for value in card.get('effectGroupIds', []) if value}
            if not effect_group_ids.issubset(card_group_ids):
                return False
        exam_effect_type = str(search.get('examEffectType') or 'ProduceExamEffectType_Unknown')
        if exam_effect_type != 'ProduceExamEffectType_Unknown':
            matched = False
            for effect_group_id in card.get('effectGroupIds', []) or []:
                group = self.effect_group_table.first(str(effect_group_id))
                if not group:
                    continue
                if exam_effect_type in {str(value) for value in group.get('examEffectTypes', []) if value}:
                    matched = True
                    break
            if not matched:
                return False
        return True

    def should_fire(
        self,
        item: ActiveProduceItem,
        *,
        phase_type: str,
        scenario: ScenarioSpec,
        state: dict[str, Any],
        deck: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate phase, counters, and parsed trigger conditions."""

        trigger = item.trigger
        if trigger is None or trigger.phase_type != phase_type:
            return False
        if item.spec.fire_limit > 0 and item.fire_count >= item.spec.fire_limit:
            return False
        if item.cooldown_remaining > 0:
            item.cooldown_remaining -= 1
            return False
        if not self._scenario_tag_matches(trigger.scenario_tag, scenario):
            return False
        for key, threshold in trigger.parameter_thresholds:
            if float(state.get(key) or 0.0) < threshold:
                return False
        if trigger.stamina_ratio_min is not None:
            max_stamina = max(float(state.get('max_stamina') or 0.0), 1.0)
            if float(state.get('stamina') or 0.0) / max_stamina < trigger.stamina_ratio_min:
                return False
        if trigger.produce_card_count_min is not None and len(deck) < trigger.produce_card_count_min:
            return False
        phase_context = context or {}
        if trigger.lesson_filter and not self._lesson_filter_matches(trigger.lesson_filter, str(phase_context.get('action_type') or '')):
            return False
        if trigger.business_reward_kind and str(phase_context.get('business_reward_kind') or '') != trigger.business_reward_kind:
            return False
        if trigger.card_search_id:
            card = phase_context.get('card')
            if not isinstance(card, dict) or not self.card_matches_search(card, trigger.card_search_id):
                return False
        return True

    def mark_fired(self, item: ActiveProduceItem) -> None:
        """Advance item counters after one successful trigger."""

        item.fire_count += 1
        item.cooldown_remaining = max(item.spec.fire_interval - 1, 0)

    def _parse_trigger(self, trigger_id: str) -> ResolvedProduceItemTrigger:
        trigger_row = self.produce_trigger_table.first(trigger_id) or {}
        phase_type = str(trigger_row.get('phaseType') or 'ProducePhaseType_Unknown')
        lesson_filter = ''
        for token in _LESSON_TRIGGER_TOKENS:
            if f'-{token}' in trigger_id:
                lesson_filter = token
                break
        card_search_id = self._extract_card_search_id(trigger_id)
        stamina_ratio_min = None
        match = _STAMINA_RATIO_RE.search(trigger_id)
        if match:
            stamina_ratio_min = float(match.group(1)) / 1000.0
        produce_card_count_min = None
        match = _PRODUCE_CARD_COUNT_RE.search(trigger_id)
        if match:
            produce_card_count_min = float(match.group(1))
        parameter_thresholds = tuple(
            (match.group(1), float(match.group(2)))
            for match in _PARAMETER_THRESHOLD_RE.finditer(trigger_id)
        )
        scenario_tag = ''
        if '-for_' in trigger_id:
            scenario_tag = trigger_id.rsplit('-for_', 1)[1]
        business_reward_kind = ''
        for kind in ('produce_card', 'produce_drink', 'produce_point'):
            if trigger_id.endswith(f'-{kind}'):
                business_reward_kind = kind
                break
        return ResolvedProduceItemTrigger(
            trigger_id=trigger_id,
            phase_type=phase_type,
            lesson_filter=lesson_filter,
            card_search_id=card_search_id,
            business_reward_kind=business_reward_kind,
            scenario_tag=scenario_tag,
            parameter_thresholds=parameter_thresholds,
            stamina_ratio_min=stamina_ratio_min,
            produce_card_count_min=produce_card_count_min,
        )

    def _extract_card_search_id(self, trigger_id: str) -> str:
        prefix_index = trigger_id.find('p_card_search-')
        if prefix_index < 0:
            return ''
        tail = trigger_id[prefix_index:]
        for search_id in self._card_search_ids:
            if tail.startswith(search_id):
                return search_id
        return ''

    def _lesson_filter_matches(self, lesson_filter: str, action_type: str) -> bool:
        if not action_type:
            return False
        is_lesson = action_type.startswith('lesson_') or action_type.startswith('self_lesson_')
        if lesson_filter == 'lesson':
            return is_lesson
        if lesson_filter == 'lesson_sp':
            return action_type.endswith('_sp')
        if lesson_filter == 'lesson_hard':
            return action_type.endswith('_hard')
        if lesson_filter in {'lesson_vocal', 'lesson_dance', 'lesson_visual'}:
            stat_type = lesson_filter.split('_', 1)[1]
            parts = action_type.split('_')
            matched_stat = parts[1] if action_type.startswith('lesson_') else (parts[2] if len(parts) > 2 else '')
            return is_lesson and matched_stat == stat_type
        if lesson_filter in {'lesson_vocal_sp', 'lesson_dance_sp', 'lesson_visual_sp'}:
            stat_type = lesson_filter.split('_')[1]
            parts = action_type.split('_')
            matched_stat = parts[1] if action_type.startswith('lesson_') else (parts[2] if len(parts) > 2 else '')
            return is_lesson and matched_stat == stat_type and action_type.endswith('_sp')
        return False

    def _scenario_tag_matches(self, scenario_tag: str, scenario: ScenarioSpec) -> bool:
        if not scenario_tag:
            return True
        if scenario_tag == 'nia_master':
            return scenario.route_type == 'nia' and scenario.parameter_growth_limit >= 2600.0
        if scenario_tag == 'hajime_legend_ssr':
            return scenario.route_type == 'first_star' and scenario.parameter_growth_limit >= 2800.0
        if scenario_tag == 'hajime_legend':
            return scenario.route_type == 'first_star' and scenario.parameter_growth_limit < 2800.0
        return False
