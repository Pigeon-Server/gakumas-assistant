import sys
from types import SimpleNamespace

import numpy as np


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.tasks.producer_challenge.steps.setup.select_idol_card import SelectIdolCardStep
from src.core.tasks.base_ui.learn_idol_card_clip import _IdolListThumbnailBox


def test_confirm_grid_target_selected_accepts_delayed_ocr_hit(monkeypatch):
    target_card = SimpleNamespace(id="target", name="Target")
    other_card = SimpleNamespace(id="other", name="Other")
    ocr_results = iter(
        [
            (other_card, ["Other"]),
            (target_card, ["Target"]),
        ]
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card._ocr_match_grid_selected_card",
        lambda _app: next(ocr_results),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card.sleep",
        lambda *_args, **_kwargs: None,
    )

    resolved = SelectIdolCardStep._confirm_grid_target_selected(
        app=SimpleNamespace(),
        target_id="target",
        attempts=3,
    )

    assert resolved == target_card


def test_search_grid_stops_after_clip_hit_is_confirmed(monkeypatch):
    step = SelectIdolCardStep()
    target_card = SimpleNamespace(id="target", name="Target")
    ctx = SimpleNamespace(selected_idol_card=None)
    click_calls = []

    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    app = SimpleNamespace(
        latest_frame=frame,
        device=SimpleNamespace(click=lambda x, y, _label="": click_calls.append((x, y))),
        game_utils=SimpleNamespace(wait_frame_stable=lambda **_kwargs: None),
        clip_manager=SimpleNamespace(idol_card_clip=SimpleNamespace()),
    )

    boxes = [
        _IdolListThumbnailBox(x1=100, y1=700, x2=240, y2=920),
        _IdolListThumbnailBox(x1=260, y1=700, x2=400, y2=920),
    ]
    clip_hits = iter([target_card, None])

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card._detect_idol_list_thumbnail_boxes",
        lambda _frame: boxes,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card._try_clip_identify",
        lambda _app, _thumb: next(clip_hits),
    )
    monkeypatch.setattr(
        step,
        "_confirm_grid_target_selected",
        lambda _app, _target_id, attempts=4: target_card,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card._ocr_match_grid_selected_card",
        lambda _app: (None, []),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card.sleep",
        lambda *_args, **_kwargs: None,
    )

    found = step._search_idol_list_grid(app, "target", ctx)

    assert found is True
    assert ctx.selected_idol_card == target_card
    assert len(click_calls) == 1


def test_clip_learn_variant_uses_grid_trimmed_image_and_relaxed_threshold(monkeypatch):
    card = SimpleNamespace(id="target", name="Target")
    raw_image = np.ones((80, 60, 3), dtype=np.uint8)
    trimmed_image = np.ones((40, 30, 3), dtype=np.uint8)
    captured = {}

    def _add_variant_to_memory(image, payload, similarity_threshold=0.0, augment=False, **_kwargs):
        captured["shape"] = image.shape
        captured["payload"] = payload
        captured["threshold"] = similarity_threshold
        captured["augment"] = augment
        return True

    app = SimpleNamespace(
        clip_manager=SimpleNamespace(
            idol_card_clip=SimpleNamespace(add_variant_to_memory=_add_variant_to_memory)
        )
    )

    monkeypatch.setattr(
        SelectIdolCardStep,
        "_prepare_grid_thumbnail_for_learning",
        staticmethod(lambda _image: trimmed_image),
    )

    SelectIdolCardStep._clip_learn_variant(app, True, raw_image, card)

    assert captured["shape"] == trimmed_image.shape
    assert captured["payload"] == card
    assert captured["threshold"] == 0.94
    assert captured["augment"] is False
