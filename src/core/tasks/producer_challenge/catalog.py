from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

from rapidfuzz import fuzz

from src.utils.game_database_tools import (
    GakumasDatabase_ProduceCardDataUtils,
    GakumasDatabase_ProduceDrinkDataUtils,
    GakumasDatabase_ProduceItemDataUtils,
    GakumasDatabase_ProduceSkillDataUtils,
    GakumasDatabase_SupportCardDataUtils,
    _concat_produce_descriptions,
    build_support_card_events,
    build_support_card_skill_descriptions,
    get_game_database,
)
from src.core.tasks.producer_challenge.shared.common import normalize_lookup_text

_PRODUCE_ROUTE_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("hajime", "regular"): ("produce_group-001", "produce-001"),
    ("hajime", "pro"): ("produce_group-001", "produce-002"),
    ("hajime", "master"): ("produce_group-001", "produce-003"),
    ("hajime", "legend"): ("produce_group-001", "produce-006"),
    ("nia", "pro"): ("produce_group-002", "produce-004"),
    ("nia", "master"): ("produce_group-002", "produce-005"),
}


@dataclass(frozen=True)
class ProduceRouteDefinition:
    scenario: str
    difficulty: str
    produce_id: str
    produce_name: str
    produce_group_id: str
    produce_group_name: str
    produce_group_type: str
    parameter_growth_limit: int

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "produce_id": self.produce_id,
            "produce_name": self.produce_name,
            "produce_group_id": self.produce_group_id,
            "produce_group_name": self.produce_group_name,
            "produce_group_type": self.produce_group_type,
            "parameter_growth_limit": self.parameter_growth_limit,
        }


@dataclass(frozen=True)
class CatalogEntry:
    kind: str
    id: str
    display_name: str
    lookup_texts: tuple[str, ...]
    metadata: dict[str, Any]


def _score_lookup_match(source: str, candidate: str) -> float:
    if not source or not candidate:
        return 0.0
    if source == candidate:
        return 100.0
    if source in candidate or candidate in source:
        shorter = min(len(source), len(candidate))
        longer = max(len(source), len(candidate))
        coverage = shorter / max(longer, 1)
        return 92.0 + coverage * 8.0
    return float(fuzz.ratio(source, candidate))


def _dedupe_strings(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _best_match_for_texts(
    texts: Sequence[str],
    entries: Sequence[CatalogEntry],
    threshold: float,
    preferred_group_id: str | None = None,
) -> tuple[CatalogEntry | None, str | None, float]:
    best_entry: CatalogEntry | None = None
    best_text: str | None = None
    best_score = 0.0

    normalized_entries = [
        (
            entry,
            tuple(normalize_lookup_text(candidate) for candidate in entry.lookup_texts if candidate),
        )
        for entry in entries
    ]

    for raw_text in texts:
        normalized_text = normalize_lookup_text(raw_text)
        if len(normalized_text) < 2:
            continue
        for entry, normalized_candidates in normalized_entries:
            if not normalized_candidates:
                continue
            score = max(_score_lookup_match(normalized_text, candidate) for candidate in normalized_candidates)
            group_ids = entry.metadata.get("produce_group_ids", [])
            if preferred_group_id and group_ids:
                if preferred_group_id in group_ids:
                    score += 4.0
                else:
                    score -= 2.0
            if score > best_score:
                best_entry = entry
                best_text = raw_text
                best_score = score

    if best_entry is None or best_score < threshold:
        return None, None, 0.0
    return best_entry, best_text, best_score


def _match_lines_against_catalog(
    texts: Sequence[str],
    entries: Sequence[CatalogEntry],
    threshold: float,
    preferred_group_id: str | None = None,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for text in texts:
        entry, matched_text, score = _best_match_for_texts(
            [text], entries, threshold=threshold, preferred_group_id=preferred_group_id
        )
        if entry is None or matched_text is None:
            continue
        key = (entry.kind, entry.id)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        matched.append(
            {
                "kind": entry.kind,
                "id": entry.id,
                "name": entry.display_name,
                "matched_text": matched_text,
                "score": round(score, 2),
                "metadata": entry.metadata,
            }
        )
    return matched


@lru_cache(maxsize=1)
def get_produce_route_definitions() -> dict[tuple[str, str], ProduceRouteDefinition]:
    produce_db = get_game_database("Produce")
    produce_group_db = get_game_database("ProduceGroup")
    definitions: dict[tuple[str, str], ProduceRouteDefinition] = {}

    for (scenario, difficulty), (group_id, produce_id) in _PRODUCE_ROUTE_MAP.items():
        produce = produce_db.get_by_id(produce_id)
        if produce is None:
            raise KeyError(f"Produce '{produce_id}' not found in game database")
        group = produce_group_db.get_by_id(group_id)
        if group is None:
            raise KeyError(f"ProduceGroup '{group_id}' not found in game database")
        if produce_id not in getattr(group, "produceIds", []):
            raise ValueError(f"Produce '{produce_id}' is not linked to group '{group_id}'")

        definitions[(scenario, difficulty)] = ProduceRouteDefinition(
            scenario=scenario,
            difficulty=difficulty,
            produce_id=produce_id,
            produce_name=produce.name,
            produce_group_id=group_id,
            produce_group_name=group.name,
            produce_group_type=group.type,
            parameter_growth_limit=int(getattr(produce, "idolCardParameterGrowthLimit", 0) or 0),
        )
    return definitions


def resolve_produce_route(scenario: str, difficulty: str) -> ProduceRouteDefinition:
    key = ((scenario or "").lower(), (difficulty or "").lower())
    try:
        return get_produce_route_definitions()[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported produce route: scenario={scenario!r}, difficulty={difficulty!r}") from exc


@lru_cache(maxsize=1)
def get_memory_tag_catalog() -> tuple[CatalogEntry, ...]:
    memory_tag_db = get_game_database("MemoryTag")
    entries: list[CatalogEntry] = []
    for tag in memory_tag_db.get_all_item():
        names = _dedupe_strings(
            [
                getattr(tag, "defaultName", None),
                getattr(getattr(tag, "localization", None), "defaultName", None),
            ]
        )
        if not names:
            continue
        entries.append(
            CatalogEntry(
                kind="memory_tag",
                id=tag.id,
                display_name=names[0],
                lookup_texts=names,
                metadata={
                    "asset_id": getattr(tag, "assetId", ""),
                    "order": getattr(tag, "order", 0),
                },
            )
        )
    return tuple(entries)


@lru_cache(maxsize=1)
def get_memory_ability_catalog() -> tuple[CatalogEntry, ...]:
    memory_ability_db = get_game_database("MemoryAbility")
    produce_skill_db = GakumasDatabase_ProduceSkillDataUtils()
    by_description: dict[str, dict[str, Any]] = {}

    for ability in memory_ability_db.get_all_item():
        skill_key = f"{ability.skillId}.{ability.level}"
        skill = produce_skill_db.get_by_id(skill_key)
        if skill is None:
            continue
        source = skill.localization if getattr(skill, "localization", None) else skill
        description = _concat_produce_descriptions(getattr(source, "produceDescriptions", []))
        # Also collect Japanese description for OCR matching
        description_ja = ""
        if getattr(skill, "localization", None):
            description_ja = _concat_produce_descriptions(getattr(skill, "produceDescriptions", []))
        if not description and not description_ja:
            continue

        primary = description_ja or description
        normalized_description = normalize_lookup_text(primary)
        bucket = by_description.setdefault(
            normalized_description,
            {
                "display_name": description or description_ja,
                "lookup_texts": set(),
                "candidates": [],
                "produce_group_ids": set(),
            },
        )
        if description:
            bucket["lookup_texts"].add(description)
        if description_ja:
            bucket["lookup_texts"].add(description_ja)
        produce_group_ids = list(getattr(ability, "produceGroupIds", []) or [])
        bucket["produce_group_ids"].update(produce_group_ids)
        bucket["candidates"].append(
            {
                "memory_ability_id": ability.id,
                "produce_skill_id": ability.skillId,
                "produce_skill_level": ability.level,
                "evaluation": ability.evaluation,
                "produce_group_ids": produce_group_ids,
            }
        )

    entries: list[CatalogEntry] = []
    for normalized_description, bucket in by_description.items():
        candidates = bucket["candidates"]
        first_candidate = candidates[0]
        entries.append(
            CatalogEntry(
                kind="memory_ability",
                id=first_candidate["memory_ability_id"],
                display_name=bucket["display_name"],
                lookup_texts=_dedupe_strings(bucket["lookup_texts"]),
                metadata={
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "produce_group_ids": sorted(bucket["produce_group_ids"]),
                },
            )
        )
    return tuple(entries)


@lru_cache(maxsize=1)
def get_card_item_catalog() -> tuple[CatalogEntry, ...]:
    produce_card_db = GakumasDatabase_ProduceCardDataUtils()
    produce_item_db = GakumasDatabase_ProduceItemDataUtils()
    produce_drink_db = GakumasDatabase_ProduceDrinkDataUtils()
    entries: dict[tuple[str, str], CatalogEntry] = {}

    def _collect_lookup_names(obj: Any) -> tuple[str, tuple[str, ...]]:
        """Return (display_name, lookup_texts) including both JP and localized names."""
        raw_name = getattr(obj, "name", "") or ""
        loc_name = getattr(getattr(obj, "localization", None), "name", None) or ""
        display = loc_name or raw_name
        names = _dedupe_strings([n for n in (raw_name, loc_name) if n])
        return display, names

    def add_entry(kind: str, obj: Any, display_name: str, lookup_texts: tuple[str, ...], metadata: dict[str, Any]):
        key = (kind, display_name)
        if key in entries:
            existing = entries[key]
            existing_ids = existing.metadata.setdefault("candidate_ids", [])
            if obj.id not in existing_ids:
                existing_ids.append(obj.id)
            return
        entries[key] = CatalogEntry(
            kind=kind,
            id=obj.id,
            display_name=display_name,
            lookup_texts=lookup_texts,
            metadata={"candidate_ids": [obj.id], **metadata},
        )

    for card in produce_card_db.get_all_item():
        if getattr(card, "upgradeCount", 0) != 0:
            continue
        display, lookups = _collect_lookup_names(card)
        if not display:
            continue
        add_entry(
            "produce_card",
            card,
            display,
            lookups,
            {
                "rarity": getattr(card, "rarity", ""),
                "plan_type": getattr(card, "planType", ""),
                "category": getattr(card, "category", ""),
                "asset_id": getattr(card, "assetId", ""),
            },
        )

    for item in produce_item_db.get_all_item():
        display, lookups = _collect_lookup_names(item)
        if not display:
            continue
        add_entry(
            "produce_item",
            item,
            display,
            lookups,
            {
                "rarity": getattr(item, "rarity", ""),
                "plan_type": getattr(item, "planType", ""),
                "asset_id": getattr(item, "assetId", ""),
            },
        )

    for drink in produce_drink_db.get_all_item():
        display, lookups = _collect_lookup_names(drink)
        if not display:
            continue
        add_entry(
            "produce_drink",
            drink,
            display,
            lookups,
            {
                "rarity": getattr(drink, "rarity", ""),
                "plan_type": getattr(drink, "planType", ""),
                "asset_id": getattr(drink, "assetId", ""),
            },
        )

    return tuple(entries.values())


@lru_cache(maxsize=1)
def get_support_ability_catalog() -> tuple[CatalogEntry, ...]:
    skill_descs = build_support_card_skill_descriptions()
    by_description: dict[str, dict[str, Any]] = {}

    for support_card_id, slots in skill_descs.items():
        for slot in slots:
            order = slot.get("order", 0)
            for level in slot.get("levels", []):
                description = level.get("description", "")
                description_ja = level.get("description_ja", "")
                if not description and not description_ja:
                    continue
                primary = description_ja or description
                normalized_description = normalize_lookup_text(primary)
                bucket = by_description.setdefault(
                    normalized_description,
                    {
                        "display_name": description or description_ja,
                        "lookup_texts": set(),
                        "support_card_ids": set(),
                        "sources": [],
                    },
                )
                if description:
                    bucket["lookup_texts"].add(description)
                if description_ja:
                    bucket["lookup_texts"].add(description_ja)
                bucket["support_card_ids"].add(support_card_id)
                bucket["sources"].append(
                    {
                        "support_card_id": support_card_id,
                        "order": order,
                        "card_level": level.get("cardLevel", 1),
                        "skill_level": level.get("skillLevel", 1),
                    }
                )

    entries: list[CatalogEntry] = []
    for bucket in by_description.values():
        entries.append(
            CatalogEntry(
                kind="support_ability",
                id=f"support_ability:{normalize_lookup_text(bucket['display_name'])}",
                display_name=bucket["display_name"],
                lookup_texts=_dedupe_strings(bucket["lookup_texts"]),
                metadata={
                    "support_card_ids": sorted(bucket["support_card_ids"]),
                    "sources": bucket["sources"],
                },
            )
        )
    return tuple(entries)


@lru_cache(maxsize=1)
def get_support_event_catalog() -> tuple[CatalogEntry, ...]:
    events = build_support_card_events()
    by_title: dict[str, dict[str, Any]] = {}

    for support_card_id, event_list in events.items():
        for event in event_list:
            title = event.get("title", "")
            title_ja = event.get("title_ja", "")
            if not title and not title_ja:
                continue
            primary_title = title_ja or title
            normalized_title = normalize_lookup_text(primary_title)
            bucket = by_title.setdefault(
                normalized_title,
                {
                    "display_name": title or title_ja,
                    "lookup_texts": set(),
                    "support_card_ids": set(),
                    "candidates": [],
                },
            )
            if title_ja:
                bucket["lookup_texts"].add(title_ja)
            if title:
                bucket["lookup_texts"].add(title)
            bucket["support_card_ids"].add(support_card_id)
            bucket["candidates"].append(
                {
                    "support_card_id": support_card_id,
                    "number": event.get("number", 0),
                    "support_card_level": event.get("supportCardLevel", 1),
                    "descriptions": event.get("descriptions", []),
                    "evaluation": title_ja or title,
                },
            )

    entries: list[CatalogEntry] = []
    for bucket in by_title.values():
        entries.append(
            CatalogEntry(
                kind="support_event",
                id=f"support_event:{normalize_lookup_text(bucket['display_name'])}",
                display_name=bucket["display_name"],
                lookup_texts=_dedupe_strings(bucket["lookup_texts"]),
                metadata={
                    "support_card_ids": sorted(bucket["support_card_ids"]),
                    "candidates": bucket["candidates"],
                },
            )
        )
    return tuple(entries)


def match_memory_tags(texts: Sequence[str], threshold: float = 70) -> list[dict[str, Any]]:
    return _match_lines_against_catalog(texts, get_memory_tag_catalog(), threshold)


def match_memory_abilities(
    texts: Sequence[str],
    produce_group_id: str | None = None,
    threshold: float = 72,
) -> list[dict[str, Any]]:
    return _match_lines_against_catalog(
        texts,
        get_memory_ability_catalog(),
        threshold,
        preferred_group_id=produce_group_id,
    )


def match_card_and_item_entries(texts: Sequence[str], threshold: float = 72) -> list[dict[str, Any]]:
    return _match_lines_against_catalog(texts, get_card_item_catalog(), threshold)


def match_support_abilities(texts: Sequence[str], threshold: float = 74) -> list[dict[str, Any]]:
    return _match_lines_against_catalog(texts, get_support_ability_catalog(), threshold)


def match_support_events(texts: Sequence[str], threshold: float = 65) -> list[dict[str, Any]]:
    return _match_lines_against_catalog(texts, get_support_event_catalog(), threshold)


# ---------------------------------------------------------------------------
# Support card name catalog — matches OCR'd support card names from the
# イベント tab yellow bars to database support card IDs.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_support_card_name_catalog() -> tuple[CatalogEntry, ...]:
    """Build a catalog of support card names for OCR matching."""
    db = GakumasDatabase_SupportCardDataUtils()
    events_map = build_support_card_events()
    entries: list[CatalogEntry] = []
    for sc in db._data:
        if not getattr(sc, "id", None):
            continue
        name_ja = getattr(sc, "name", "") or ""
        loc = getattr(sc, "localization", None)
        name_loc = getattr(loc, "name", "") if loc else ""
        if not name_ja and not name_loc:
            continue
        lookup: set[str] = set()
        if name_ja:
            lookup.add(name_ja)
        if name_loc:
            lookup.add(name_loc)

        sc_events = events_map.get(sc.id, [])
        event_summaries = []
        for ev in sc_events:
            summary: dict[str, Any] = {
                "number": ev.get("number", 0),
                "title": ev.get("title", ""),
                "descriptions": ev.get("descriptions", []),
            }
            if ev.get("title_ja"):
                summary["title_ja"] = ev["title_ja"]
            event_summaries.append(summary)

        entries.append(
            CatalogEntry(
                kind="support_card",
                id=sc.id,
                display_name=name_loc or name_ja,
                lookup_texts=_dedupe_strings(lookup),
                metadata={
                    "name_ja": name_ja,
                    "events": event_summaries,
                },
            )
        )
    return tuple(entries)


def match_support_card_names(texts: Sequence[str], threshold: float = 68) -> list[dict[str, Any]]:
    """Match OCR texts against support card names. Used for the event tab."""
    return _match_lines_against_catalog(texts, get_support_card_name_catalog(), threshold)
