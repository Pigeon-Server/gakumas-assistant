from src.core.tasks.producer_challenge.catalog import (
    get_card_item_catalog,
    get_memory_ability_catalog,
    get_support_ability_catalog,
    get_support_card_name_catalog,
    get_support_event_catalog,
    match_memory_abilities,
    match_memory_tags,
    match_support_abilities,
    match_support_events,
    resolve_produce_route,
)


def test_resolve_produce_route_uses_main_database():
    route = resolve_produce_route("hajime", "legend")
    assert route.produce_id == "produce-006"
    assert route.produce_group_id == "produce_group-001"
    assert route.produce_name == "レジェンド"
    assert route.produce_group_name


def test_match_memory_tags_with_japanese_labels():
    matches = match_memory_tags(["ボーカル", "集中", "無関係テキスト"])
    matched_ids = {match["id"] for match in matches}
    assert "memory_tag_01" in matched_ids
    assert "memory_tag_15" in matched_ids


def test_match_memory_abilities_with_catalog_description():
    entry = next(item for item in get_memory_ability_catalog() if item.display_name)
    matches = match_memory_abilities([entry.display_name])
    assert any(match["id"] == entry.id for match in matches)


def test_match_support_abilities_with_catalog_description():
    entry = next(item for item in get_support_ability_catalog() if item.display_name)
    matches = match_support_abilities([entry.display_name])
    assert any(match["id"] == entry.id for match in matches)


def test_match_support_events_with_catalog_title():
    entry = next(item for item in get_support_event_catalog() if item.display_name)
    matches = match_support_events([entry.display_name])
    assert any(match["id"] == entry.id for match in matches)


def test_catalog_display_names_prefer_japanese_main_database_values():
    card_entry = next(
        item
        for item in get_card_item_catalog()
        if item.kind == "produce_card"
        and item.display_name
        and any(text != item.display_name for text in item.lookup_texts)
    )
    assert card_entry.lookup_texts[0] == card_entry.display_name

    memory_ability_entry = next(
        item
        for item in get_memory_ability_catalog()
        if item.display_name
        and any(text != item.display_name for text in item.lookup_texts)
    )
    assert memory_ability_entry.lookup_texts[0] == memory_ability_entry.display_name

    support_event_entry = next(
        item
        for item in get_support_event_catalog()
        if item.display_name
        and any(text != item.display_name for text in item.lookup_texts)
    )
    assert support_event_entry.lookup_texts[0] == support_event_entry.display_name

    support_card_entry = next(
        item
        for item in get_support_card_name_catalog()
        if item.display_name
        and any(text != item.display_name for text in item.lookup_texts)
    )
    assert support_card_entry.lookup_texts[0] == support_card_entry.display_name
    assert support_card_entry.metadata["name_ja"] == support_card_entry.display_name
