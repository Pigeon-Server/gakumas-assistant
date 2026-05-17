from types import SimpleNamespace

import numpy as np

from src.core.services import game_asset_service


class _FakeIdolCardDb:
    def __init__(self, entries):
        self._entries = list(entries)
        self._by_id = {entry.id: entry for entry in self._entries}

    def get_all_item(self):
        return list(self._entries)

    def get_by_id(self, entry_id):
        return self._by_id.get(entry_id)


class _LoggerCapture:
    def __init__(self):
        self.debug_messages = []
        self.info_messages = []

    def debug(self, message, *args, **kwargs):
        self.debug_messages.append(message)

    def info(self, message, *args, **kwargs):
        self.info_messages.append(message)

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_train_clip_from_game_assets_idol_supports_asset_id_filenames(monkeypatch, tmp_path):
    asset_dir = tmp_path / "idol_cards"
    asset_dir.mkdir()
    (asset_dir / "cidol-test-001.png").write_bytes(b"legacy")
    (asset_dir / "i_card-test-002.png").write_bytes(b"canonical")

    payloads = [
        SimpleNamespace(id="i_card-test-001", assetId="cidol-test-001"),
        SimpleNamespace(id="i_card-test-002", assetId="cidol-test-002"),
    ]
    calls = []
    clip_manager = SimpleNamespace(
        idol_card_clip=SimpleNamespace(
            add_to_memory=lambda _image, payload, save_image=False: calls.append(payload.id) or True
        )
    )

    monkeypatch.setattr(game_asset_service, "_get_asset_subdir", lambda _subdir: asset_dir)
    monkeypatch.setattr("src.utils.game_database_tools.GakumasDatabase_IdolCardDataUtils", lambda: _FakeIdolCardDb(payloads))

    import cv2
    monkeypatch.setattr(cv2, "imread", lambda _path: np.zeros((8, 8, 3), dtype=np.uint8))

    added = game_asset_service.train_clip_from_game_assets(
        clip_manager,
        game_asset_service.IDOL_CARD_SUBDIR,
    )

    assert added == 2
    assert calls == ["i_card-test-001", "i_card-test-002"]


def test_train_clip_from_game_assets_summarizes_missing_idol_entries(monkeypatch, tmp_path):
    asset_dir = tmp_path / "idol_cards"
    asset_dir.mkdir()
    (asset_dir / "i_card-valid.png").write_bytes(b"valid")
    (asset_dir / "i_card-stale.png").write_bytes(b"stale-id")
    (asset_dir / "cidol-stale.png").write_bytes(b"stale-asset")

    payload = SimpleNamespace(id="i_card-valid", assetId="cidol-valid")
    clip_manager = SimpleNamespace(
        idol_card_clip=SimpleNamespace(add_to_memory=lambda *_args, **_kwargs: True)
    )
    logger = _LoggerCapture()

    monkeypatch.setattr(game_asset_service, "_get_asset_subdir", lambda _subdir: asset_dir)
    monkeypatch.setattr("src.utils.game_database_tools.GakumasDatabase_IdolCardDataUtils", lambda: _FakeIdolCardDb([payload]))
    monkeypatch.setattr(game_asset_service, "logger", logger)

    import cv2
    monkeypatch.setattr(cv2, "imread", lambda _path: np.zeros((8, 8, 3), dtype=np.uint8))

    added = game_asset_service.train_clip_from_game_assets(
        clip_manager,
        game_asset_service.IDOL_CARD_SUBDIR,
    )

    assert added == 1
    assert all("No DB entry for" not in message for message in logger.debug_messages)
    assert any(
        "skipped 2 cache files without current DB entry" in message
        for message in logger.debug_messages
    )
