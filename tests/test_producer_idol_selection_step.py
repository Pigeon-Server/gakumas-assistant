from types import SimpleNamespace

from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge.steps.setup.select_idol_card import SelectIdolCardStep


def test_select_idol_card_default_selection_advances_to_support_page(monkeypatch):
    events: list[tuple] = []
    step = SelectIdolCardStep()
    ctx = SimpleNamespace(target_idol_card_id="", selected_idol_card=None)
    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            click_button=lambda text, match_config=None: events.append(("click", text)),
            wait_loading=lambda: events.append(("wait_loading",)),
        ),
    )

    monkeypatch.setattr(step, "_remember_current_selection", lambda _app, _ctx: events.append(("remember",)))
    monkeypatch.setattr(step, "_wait_for_support_selection_page", lambda _app: events.append(("wait_support",)) or True)

    assert step.execute(app, ctx) is True
    assert events == [
        ("remember",),
        ("click", ButtonText.NEXT),
        ("wait_loading",),
        ("wait_support",),
    ]


def test_select_idol_card_uses_ocr_fallback_when_clip_misses(monkeypatch):
    step = SelectIdolCardStep()
    target_card = SimpleNamespace(id="i_card-fktn-3-000", name="Target Card")
    ctx = SimpleNamespace(target_idol_card_id="i_card-fktn-3-000", selected_idol_card=None)
    app = SimpleNamespace()

    advance_calls: list[tuple] = []

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card.idol_card_db.get_by_id",
        lambda _card_id: target_card,
    )
    monkeypatch.setattr(step, "_extract_selected_card_image", lambda _app: object())
    monkeypatch.setattr(step, "_advance_to_support_selection", lambda _app: True)
    monkeypatch.setattr(step, "_rewind_to_head", lambda _app: 0)
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card._resolve_current_selected_idol_card",
        lambda _app, _card_image: (target_card, ["Target Card"]),
    )
    monkeypatch.setattr(
        step,
        "_advance_carousel",
        lambda _app, _card_image: advance_calls.append(("advance",)) or True,
    )

    assert step.execute(app, ctx) is True
    assert ctx.selected_idol_card is target_card
    assert advance_calls == []


def test_select_idol_card_rewinds_one_after_transient_target_match(monkeypatch):
    step = SelectIdolCardStep()
    target_card = SimpleNamespace(id="i_card-fktn-3-000", name="Target Card")
    overshot_card = SimpleNamespace(id="i_card-kcna-3-000", name="Overshot Card")
    ctx = SimpleNamespace(target_idol_card_id="i_card-fktn-3-000", selected_idol_card=None)
    move_calls: list[tuple[str, bool]] = []
    app = SimpleNamespace()

    resolve_results = iter(
        [
            (target_card, ["Target Card"]),
            (overshot_card, ["Overshot Card"]),
            (target_card, ["Target Card"]),
        ]
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card.idol_card_db.get_by_id",
        lambda _card_id: target_card,
    )
    monkeypatch.setattr(step, "_extract_selected_card_image", lambda _app: object())
    monkeypatch.setattr(step, "_advance_to_support_selection", lambda _app: True)
    monkeypatch.setattr(step, "_rewind_to_head", lambda _app: 0)
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card._resolve_current_selected_idol_card",
        lambda _app, _card_image: next(resolve_results),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card.wait_for_idol_card_carousel_stable",
        lambda _app, stable_count=0, timeout=0: True,
    )
    monkeypatch.setattr(
        step,
        "_move_carousel",
        lambda _app, direction, allow_swipe_fallback: move_calls.append((direction, allow_swipe_fallback)) or True,
    )

    assert step.execute(app, ctx) is True
    assert ctx.selected_idol_card is target_card
    assert move_calls == [("prev", False)]


def test_wait_for_support_selection_page_uses_support_slot_signal(monkeypatch):
    step = SelectIdolCardStep()
    states = iter(
        [
            {
                BaseUILabels.PRODUCT_CARD_SELECTED,
            },
            {
                BaseUILabels.SUPPORT_CARD,
                BaseUILabels.PRODUCE_CARD_VISUAL,
            },
        ]
    )
    current_labels = next(states)

    class _ResultsStub:
        def exists_label(self, label):
            return label in current_labels

    def _fake_sleep(_seconds):
        nonlocal current_labels
        current_labels = next(states, current_labels)

    app = SimpleNamespace(latest_results=_ResultsStub())

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card.sleep",
        _fake_sleep,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.setup.select_idol_card.wait_frame_stable",
        lambda _app, timeout=0: None,
    )

    assert step._wait_for_support_selection_page(app) is True
