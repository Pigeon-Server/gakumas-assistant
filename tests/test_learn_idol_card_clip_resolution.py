import numpy as np
import pytest

import src.core.tasks.base_ui.learn_idol_card_clip as learn_idol_card_clip
from src.constants.game.text.button_text import ButtonText
from src.utils.string_tools import string_match


def test_resolve_idol_card_from_texts_uses_character_to_disambiguate_duplicate_titles():
    card = learn_idol_card_clip._resolve_idol_card_from_texts(["R", "学園生活", "花海佑芽"])

    assert card is not None
    assert card.id == "i_card-hume-1-000"
    assert card.name == "学園生活"
    assert f"{card.characterCls.lastName}{card.characterCls.firstName}" == "花海佑芽"


def test_resolve_idol_card_from_texts_ignores_noise_and_matches_normalized_title():
    card = learn_idol_card_clip._resolve_idol_card_from_texts(
        ["SR", "ー番星", "十王星南", "ぐ", "7", "L』"]
    )

    assert card is not None
    assert card.id == "i_card-jsna-2-000"
    assert card.name == "一番星"
    assert f"{card.characterCls.lastName}{card.characterCls.firstName}" == "十王星南"


def test_resolve_idol_card_from_texts_returns_none_for_ambiguous_title_without_character():
    assert learn_idol_card_clip._resolve_idol_card_from_texts(["学園生活"]) is None


def test_finalize_invisible_tracks_queues_unresolved_tracks_for_late_resolution():
    track = learn_idol_card_clip._CarouselTrack(
        temp_id=21,
        images=[np.zeros((12, 12, 3), dtype=np.uint8)],
    )
    pending_tracks = []

    kept, learned = learn_idol_card_clip._finalize_invisible_tracks(
        app=object(),
        tracks_by_slot={0: track},
        visible_offsets=set(),
        pending_tracks=pending_tracks,
    )

    assert kept == {}
    assert learned == 0
    assert pending_tracks == [track]


def test_late_resolve_track_with_clip_prefers_majority_vote(monkeypatch):
    track = learn_idol_card_clip._CarouselTrack(
        temp_id=22,
        images=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
    )
    target = learn_idol_card_clip._resolve_idol_card_from_texts(["一番星", "十王星南"])
    other = learn_idol_card_clip._resolve_idol_card_from_texts(["学園生活", "花海佑芽"])
    results = [target, other, target]

    monkeypatch.setattr(
        learn_idol_card_clip,
        "_try_clip_identify",
        lambda _app, _image: results.pop(0),
    )

    resolved = learn_idol_card_clip._late_resolve_track_with_clip(object(), track)

    assert resolved is not None
    assert resolved.id == target.id


@pytest.mark.parametrize("text", ["Pアイドル一覧", "Pアイドルー覧", "アイドル一覧", "アイドルー覧"])
def test_idol_list_button_uses_canonical_constant_with_string_match(text):
    assert string_match(
        text,
        ButtonText.PAGE__IDOL.IDOL_LIST,
        learn_idol_card_clip._IDOL_LIST_BUTTON_MATCH,
    )
