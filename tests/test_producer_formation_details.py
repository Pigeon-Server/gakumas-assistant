from types import SimpleNamespace

from src.core.tasks.producer_challenge.catalog import (
    get_card_item_catalog,
    get_memory_ability_catalog,
    match_memory_abilities,
)
from src.core.tasks.producer_challenge.steps.collect.collect_formation_details import CollectFormationDetailsStep


def _get_produce_card_match():
    entry = next(item for item in get_card_item_catalog() if item.kind == "produce_card")
    return {
        "kind": entry.kind,
        "id": entry.id,
        "name": entry.display_name,
        "matched_text": entry.display_name,
        "score": 100.0,
        "metadata": entry.metadata,
    }


def _get_memory_ability_match():
    entry = next(item for item in get_memory_ability_catalog() if item.display_name)
    return match_memory_abilities([entry.display_name])[0]


def test_build_card_item_details_extracts_skill_card_summaries():
    produce_card = _get_produce_card_match()

    details = CollectFormationDetailsStep._build_card_item_details(
        [
            "プロデュース中獲得",
            "中間試験/1次オーディション後に獲得",
            produce_card["name"],
            "イベント",
        ]
    )

    assert details["skill_card_summaries"]
    assert details["produce_card_ids"] == [produce_card["id"]]
    summary = details["skill_card_summaries"][0]
    assert summary["title"] == produce_card["name"]
    assert summary["matched_entry"]["id"] == produce_card["id"]
    assert summary["matched_entry_id"] == produce_card["id"]
    assert summary["source_kind"] == "earned_during_produce"
    assert summary["gain_timing"] == "mid_exam_or_first_audition"
    assert "中間試験/1次オーディション後に獲得" in summary["phase_texts"]


def test_build_ability_details_falls_back_to_raw_memory_matches():
    ability_entry = next(item for item in get_memory_ability_catalog() if item.display_name)

    details = CollectFormationDetailsStep._build_ability_details(
        ["アビリティ", ability_entry.display_name, "イベント"],
        SimpleNamespace(produce_group_id=None),
    )

    assert details["memory_abilities"]["match_scope"] == "raw_texts_fallback"
    assert details["memory_abilities"]["entry_ids"] == [ability_entry.id]
    assert any(match["id"] == ability_entry.id for match in details["memory_abilities"]["matched_entries"])


def test_build_memory_fallback_combines_abilities_and_skill_cards():
    ability_match = _get_memory_ability_match()
    produce_card = _get_produce_card_match()

    summaries = CollectFormationDetailsStep._build_memory_fallback(
        {
            "skill_card_summaries": [
                {
                    "page_index": 1,
                    "total_pages": 1,
                    "phase_texts": ["中間試験/1次オーディション後に獲得"],
                    "title": produce_card["name"],
                    "raw_texts": ["中間試験/1次オーディション後に獲得", produce_card["name"]],
                    "effect_texts": [],
                    "matched_entry": produce_card,
                    "db_description": "",
                    "description_match_score": 0.0,
                    "source_kind": "earned_during_produce",
                    "source_text": "プロデュース中獲得",
                    "gain_timing": "mid_exam_or_first_audition",
                }
            ]
        },
        {
            "memory_abilities": {
                "matched_entries": [ability_match],
            }
        },
        produce_group_id="produce_group-001",
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["source"] == "formation-details-summary"
    assert summary["abilities"] == [ability_match]
    assert summary["memory_ability_ids"] == [ability_match["id"]]
    assert summary["skill_card_count"] == 1
    assert summary["skill_card_ids"] == [produce_card["id"]]
    assert summary["gain_timing"] == "mid_exam_or_first_audition"
    assert summary["card_source"] == "earned_during_produce"
    assert summary["card_acquisition_texts"] == ["中間試験/1次オーディション後に獲得"]
    assert summary["produce_group_id"] == "produce_group-001"


def test_extract_skill_card_summaries_keeps_generic_sections_when_card_name_unmatched():
    summaries = CollectFormationDetailsStep._extract_memory_skill_card_summaries(
        [
            "プロデュース開始時から所持",
            "アピールの基本",
            "プロデュース中獲得",
            "中間試験/1次オーディション後に獲得",
            "強化確認",
        ]
    )

    assert len(summaries) == 2
    assert summaries[0]["source_kind"] == "initial_owned"
    assert summaries[0]["raw_texts"] == ["プロデュース開始時から所持", "アピールの基本"]
    assert summaries[1]["source_kind"] == "earned_during_produce"
    assert summaries[1]["gain_timing"] == "mid_exam_or_first_audition"
    assert "中間試験/1次オーディション後に獲得" in summaries[1]["phase_texts"]


def test_build_memory_fallback_uses_unmatched_raw_texts_as_aggregate_entry():
    summaries = CollectFormationDetailsStep._build_memory_fallback(
        {
            "skill_card_summaries": [
                {
                    "phase_texts": ["中間試験/1次オーディション後に獲得"],
                    "raw_texts": ["プロデュース中獲得", "中間試験/1次オーディション後に獲得", "強化確認"],
                    "source_kind": "earned_during_produce",
                    "gain_timing": "mid_exam_or_first_audition",
                }
            ]
        },
        {
            "memory_abilities": {
                "matched_entries": [],
                "raw_texts": ["レッスン中1回", "やる気+2"],
            }
        },
        produce_group_id="produce_group-001",
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["source"] == "formation-details-summary"
    assert summary["abilities"] == []
    assert summary["skill_card_count"] == 1
    assert summary["gain_timing"] == "mid_exam_or_first_audition"
    assert summary["card_acquisition_texts"] == ["中間試験/1次オーディション後に獲得"]
    assert "レッスン中1回" in summary["raw_texts"]
