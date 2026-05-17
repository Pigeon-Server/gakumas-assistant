from types import SimpleNamespace

from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge import ui as ui_module
from src.core.tasks.producer_challenge.steps.setup import select_memories as memory_module
from src.core.tasks.producer_challenge.steps.setup import select_support_cards as support_module
from src.core.tasks.producer_challenge.steps.setup.select_memories import SelectMemoriesStep
from src.core.tasks.producer_challenge.steps.setup.select_support_cards import SelectSupportCardsStep


def test_parse_preset_index_accepts_noisy_button_text():
    assert ui_module.parse_preset_index(":1/20") == (1, 20)
    assert ui_module.parse_preset_index(" = 03 / 20 ") == (3, 20)
    assert ui_module.parse_preset_index("编成1") is None


def test_build_preset_swipe_paths_uses_card_rows():
    boxes = [
        SimpleNamespace(x=41, w=403, cy=1569),
        SimpleNamespace(x=380, w=700, cy=1569),
        SimpleNamespace(x=40, w=361, cy=1768),
    ]

    paths = ui_module.build_preset_swipe_paths(boxes, frame_width=1080)

    assert paths == [
        (601, 1569, 139, 1569),
        (601, 1768, 139, 1768),
    ]


def test_select_preset_by_horizontal_swipe_moves_forward(monkeypatch):
    app = SimpleNamespace(current_index=1)
    swipes: list[tuple[int, int, int, int]] = []

    monkeypatch.setattr(ui_module, "get_current_preset_index", lambda app: (app.current_index, 20))
    monkeypatch.setattr(
        ui_module,
        "get_preset_swipe_paths",
        lambda app, card_labels: [(601, 1569, 139, 1569)],
    )

    def _fake_swipe(app, start_x, start_y, end_x, end_y, **_kwargs):
        swipes.append((start_x, start_y, end_x, end_y))
        app.current_index += 1 if start_x > end_x else -1

    monkeypatch.setattr(ui_module, "inertial_swipe", _fake_swipe)

    assert ui_module.select_preset_by_horizontal_swipe(
        app,
        3,
        card_labels=(BaseUILabels.SUPPORT_CARD,),
        description="支援卡编成",
    ) is True
    assert app.current_index == 3
    assert swipes == [
        (601, 1569, 139, 1569),
        (601, 1569, 139, 1569),
    ]


def test_select_preset_by_horizontal_swipe_moves_backward(monkeypatch):
    app = SimpleNamespace(current_index=4)
    swipes: list[tuple[int, int, int, int]] = []

    monkeypatch.setattr(ui_module, "get_current_preset_index", lambda app: (app.current_index, 20))
    monkeypatch.setattr(
        ui_module,
        "get_preset_swipe_paths",
        lambda app, card_labels: [(601, 1569, 139, 1569)],
    )

    def _fake_swipe(app, start_x, start_y, end_x, end_y, **_kwargs):
        swipes.append((start_x, start_y, end_x, end_y))
        app.current_index += 1 if start_x > end_x else -1

    monkeypatch.setattr(ui_module, "inertial_swipe", _fake_swipe)

    assert ui_module.select_preset_by_horizontal_swipe(
        app,
        2,
        card_labels=(BaseUILabels.MEMORY_CARD,),
        description="记忆编成",
    ) is True
    assert app.current_index == 2
    assert swipes == [
        (139, 1569, 601, 1569),
        (139, 1569, 601, 1569),
    ]


def test_support_preset_select_uses_swipe_and_advances(monkeypatch):
    events: list[tuple] = []

    class _Results:
        @staticmethod
        def exists_label(label):
            return label == BaseUILabels.MEMORY_CARD

    app = SimpleNamespace(
        latest_results=_Results(),
        game_utils=SimpleNamespace(
            click_button=lambda text, match_config=None: events.append(("click", text)),
            wait_loading=lambda: events.append(("wait_loading",)),
        ),
    )
    ctx = SimpleNamespace(support_card_preset_index=3)

    monkeypatch.setattr(
        support_module,
        "select_preset_by_horizontal_swipe",
        lambda app, target_index, **kwargs: events.append(
            ("swipe", target_index, kwargs["card_labels"], kwargs["description"])
        )
        or True,
    )
    monkeypatch.setattr(support_module, "sleep", lambda *_args, **_kwargs: None)

    assert SelectSupportCardsStep()._preset_select(app, ctx) is True
    assert events[0] == ("swipe", 3, (BaseUILabels.SUPPORT_CARD,), "支援卡编成")
    assert ("click", ButtonText.NEXT) in events


def test_memory_preset_select_uses_swipe_and_keeps_page(monkeypatch):
    events: list[tuple] = []
    step = SelectMemoriesStep()
    ctx = SimpleNamespace(memory_preset_index=4, use_rental=True)

    monkeypatch.setattr(
        step,
        "_sync_rental_checkbox",
        lambda app, ctx: events.append(("sync_rental", ctx.use_rental)),
    )
    monkeypatch.setattr(
        memory_module,
        "select_preset_by_horizontal_swipe",
        lambda app, target_index, **kwargs: events.append(
            ("swipe", target_index, kwargs["card_labels"], kwargs["description"])
        )
        or True,
    )
    monkeypatch.setattr(memory_module, "wait_for_memory_selection_page", lambda app, timeout=0: True)

    assert step._preset_select(SimpleNamespace(), ctx) is True
    assert events == [
        ("swipe", 4, (BaseUILabels.MEMORY_CARD,), "记忆编成"),
    ]
