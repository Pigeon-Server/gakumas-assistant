from types import SimpleNamespace

import numpy as np

from src.core.tasks.producer_challenge.gameplay import decision as decision_module


def test_resolve_produce_drink_identity_prefers_clip(monkeypatch):
    matched = SimpleNamespace(
        id="pdrink_001",
        name="アイスコーヒー",
        localization=SimpleNamespace(name="冰咖啡"),
    )
    app = SimpleNamespace(
        clip_manager=SimpleNamespace(
            produce_drink_clip=SimpleNamespace(retrieve=lambda _image: matched)
        )
    )

    monkeypatch.setattr(
        decision_module,
        "_enrich_drink_metadata",
        lambda _drink_id: {
            "display_name": "冰咖啡",
            "description": "集中+3",
            "effect_types": ["集中"],
            "rarity": "SR",
        },
    )

    resolution = decision_module.resolve_produce_drink_identity(
        "",
        app=app,
        box=SimpleNamespace(frame=np.zeros((8, 8, 3), dtype=np.uint8)),
        index=0,
    )

    assert resolution.db_id == "pdrink_001"
    assert resolution.source == "clip"
    assert resolution.display_name == "冰咖啡"


def test_resolve_produce_drink_identity_learns_after_ocr_match(monkeypatch):
    calls: list[str] = []
    app = SimpleNamespace(
        clip_manager=SimpleNamespace(
            produce_drink_clip=SimpleNamespace(
                retrieve=lambda _image: None,
                add_to_memory=lambda _image, payload, similarity_threshold=0.98, save_image=False: calls.append(payload.id) or True,
            )
        )
    )

    monkeypatch.setattr(
        decision_module,
        "_match_catalog_entry",
        lambda title, expected_kind=None: {
            "id": "pdrink_002",
            "name": "ビタミンドリンク",
            "kind": "produce_drink",
            "score": 99,
        } if expected_kind == "produce_drink" else None,
    )
    monkeypatch.setattr(
        decision_module,
        "_enrich_drink_metadata",
        lambda _drink_id: {
            "display_name": "维生素饮料",
            "description": "好調2ターン",
            "effect_types": ["好調"],
            "rarity": "R",
        },
    )

    payload = SimpleNamespace(id="pdrink_002")
    monkeypatch.setattr(
        "src.utils.game_database_tools.GakumasDatabase_ProduceDrinkDataUtils",
        lambda: SimpleNamespace(get_by_id=lambda _id: payload),
    )

    resolution = decision_module.resolve_produce_drink_identity(
        "ビタミンドリンク",
        app=app,
        box=SimpleNamespace(frame=np.zeros((8, 8, 3), dtype=np.uint8)),
        index=0,
    )

    assert resolution.db_id == "pdrink_002"
    assert resolution.source == "ocr"
    assert calls == ["pdrink_002"]


def test_resolve_produce_item_identity_learns_after_ocr_match(monkeypatch):
    calls: list[str] = []
    app = SimpleNamespace(
        clip_manager=SimpleNamespace(
            produce_item_clip=SimpleNamespace(
                retrieve=lambda _image: None,
                add_to_memory=lambda _image, payload, similarity_threshold=0.98, save_image=False: calls.append(payload.id) or True,
            )
        )
    )

    monkeypatch.setattr(
        decision_module,
        "_match_catalog_entry",
        lambda title, expected_kind=None: {
            "id": "pitem_001",
            "name": "测试P物品",
            "kind": "produce_item",
            "score": 98,
        } if expected_kind == "produce_item" else None,
    )
    monkeypatch.setattr(
        decision_module,
        "_enrich_item_metadata",
        lambda _item_id: {
            "display_name": "测试P物品",
            "description": "获得额外收益",
            "rarity": "SR",
        },
    )

    payload = SimpleNamespace(id="pitem_001")
    monkeypatch.setattr(
        "src.utils.game_database_tools.GakumasDatabase_ProduceItemDataUtils",
        lambda: SimpleNamespace(get_by_id=lambda _id: payload),
    )

    resolution = decision_module.resolve_produce_item_identity(
        "测试P物品",
        app=app,
        box=SimpleNamespace(frame=np.zeros((8, 8, 3), dtype=np.uint8)),
        index=0,
    )

    assert resolution.db_id == "pitem_001"
    assert resolution.source == "ocr"
    assert calls == ["pitem_001"]


def test_resolve_produce_item_identity_uses_lookup_texts_when_title_empty(monkeypatch):
    app = SimpleNamespace(
        clip_manager=SimpleNamespace(
            produce_item_clip=SimpleNamespace(retrieve=lambda _image: None)
        )
    )

    monkeypatch.setattr(
        decision_module,
        "_match_catalog_entry_from_texts",
        lambda texts, expected_kind=None: {
            "id": "pitem_777",
            "name": "候选P物品",
            "kind": "produce_item",
            "score": 96,
            "matched_text": "候选P物品",
        } if expected_kind == "produce_item" and "候选P物品" in texts else None,
    )
    monkeypatch.setattr(
        decision_module,
        "_enrich_item_metadata",
        lambda _item_id: {
            "display_name": "候选P物品",
            "description": "测试效果",
            "rarity": "SR",
        },
    )
    monkeypatch.setattr(
        decision_module,
        "_learn_item_clip_from_db_id",
        lambda _app, _image, _item_id: None,
    )

    resolution = decision_module.resolve_produce_item_identity(
        "",
        app=app,
        box=SimpleNamespace(frame=np.zeros((8, 8, 3), dtype=np.uint8)),
        index=0,
        lookup_texts=["候选P物品", "别的噪声"],
    )

    assert resolution.db_id == "pitem_777"
    assert resolution.display_name == "候选P物品"
    assert resolution.metadata["matched_text"] == "候选P物品"


def test_resolve_produce_card_identity_learns_after_ocr_match(monkeypatch):
    calls: list[str] = []
    app = SimpleNamespace(
        clip_manager=SimpleNamespace(
            skill_card_clip=SimpleNamespace(
                retrieve=lambda _image: None,
                add_to_memory=lambda _image, payload, similarity_threshold=0.98, save_image=False: calls.append(payload.id) or True,
            )
        )
    )

    monkeypatch.setattr(
        decision_module,
        "_match_catalog_entry",
        lambda title, expected_kind=None: {
            "id": "card_001",
            "name": "测试技能卡",
            "kind": "produce_card",
            "score": 97,
        } if expected_kind == "produce_card" else None,
    )
    monkeypatch.setattr(
        decision_module,
        "_enrich_card_metadata",
        lambda _card_id, upgrade_count=0: {
            "display_name": "测试技能卡",
            "description": "打分+9",
            "category": "ProduceCardCategory_ActiveSkill",
            "upgrade_count": upgrade_count,
            "effect_types": ["打分"],
        },
    )

    payload = SimpleNamespace(id="card_001", upgradeCount=0)
    monkeypatch.setattr(
        "src.utils.game_database_tools.GakumasDatabase_ProduceCardDataUtils",
        lambda: SimpleNamespace(get_by_id=lambda _id: payload),
    )

    resolution = decision_module.resolve_produce_card_identity(
        app,
        title="测试技能卡",
        box=SimpleNamespace(frame=np.zeros((8, 8, 3), dtype=np.uint8)),
        index=0,
    )

    assert resolution.db_id == "card_001"
    assert resolution.source == "ocr"
    assert calls == ["card_001"]
