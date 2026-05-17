from types import SimpleNamespace

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.core.tasks.base_ui import auto_purchase


class _ClickableStub:
    def __init__(self, name: str, text: str = "", disabled: bool = False):
        self.name = name
        self.text = text
        self._disabled = disabled

    def is_disabled(self) -> bool:
        return self._disabled


class _ModalStub:
    def __init__(self, confirm_button: _ClickableStub, body: str):
        self.confirm_button = confirm_button
        self.modal_body_text = body


class _DeviceStub:
    def __init__(self):
        self.clicked: list[str] = []

    def click_element(self, element):
        self.clicked.append(element.name)


class _GameUtilsStub:
    def __init__(self, modal_bodies=None):
        self.calls: list = []
        self._modal_bodies = list(modal_bodies or [])
        self._confirm_index = 0
        self._location_wait_failures = {}
        self._modal = None
        self._app = None

    def wait_frame_stable(self, *args, **kwargs):
        self.calls.append("wait_frame_stable")

    def wait_loading(self):
        self.calls.append("wait_loading")

    def wait_for_modal(self, *_args, **_kwargs):
        self.calls.append("wait_for_modal")
        if not self._modal_bodies:
            return None
        self._confirm_index += 1
        return _ModalStub(
            _ClickableStub(f"confirm:{self._confirm_index}"),
            self._modal_bodies.pop(0),
        )

    def back_next_page(self):
        self.calls.append("back_next_page")

    def click_button(self, text, **_kwargs):
        self.calls.append(("click_button", text))

    def wait_location_update(self, location, timeout=10):
        self.calls.append(("wait_location_update", location, timeout))
        if self._location_wait_failures.get(location, 0) > 0:
            self._location_wait_failures[location] -= 1
            raise TimeoutError("location timeout")

    def update_current_location(self, location=None):
        self.calls.append(("update_current_location", location))

    def try_get_modal(self, *_args, **_kwargs):
        self.calls.append("try_get_modal")
        return self._modal

    def click_modal_button_and_wait_transition(self, button, previous_modal_title=None, timeout=5, interval=0.2):
        self.calls.append(("click_modal_button_and_wait_transition", getattr(button, "name", None), previous_modal_title))
        if self._app is not None and hasattr(self._app, "advance_results"):
            self._app.advance_results()
        self._modal = None
        return True

    def wait_for_label(self, label, timeout=10, interval=0.2):
        self.calls.append(("wait_for_label", label, timeout, interval))
        if self._app is not None and hasattr(self._app, "advance_results"):
            self._app.advance_results()
        return True


class _ButtonListFactory:
    def __init__(self, buttons):
        self._buttons = list(buttons)

    def __call__(self, _results):
        button = self._buttons.pop(0) if self._buttons else None
        return SimpleNamespace(get_button_by_text=lambda _text, **_kwargs: button)


def _build_app(refresh_shop: bool, use_gem_refresh: bool, modal_bodies=None):
    config = SimpleNamespace(
        task__auto_purchase=SimpleNamespace(
            refresh_shop=SimpleNamespace(value=refresh_shop),
            use_gem_refresh=SimpleNamespace(value=use_gem_refresh),
            daily_buy_list=SimpleNamespace(value=[]),
        )
    )
    return SimpleNamespace(
        config_service=lambda: config,
        latest_frame=np.zeros((100, 100, 3), dtype=np.uint8),
        latest_results=object(),
        device=_DeviceStub(),
        game_utils=_GameUtilsStub(modal_bodies=modal_bodies),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None, show=lambda: None, clear_all=lambda: None),
    )


class _DetectionStub:
    def __init__(self, items=None, groups=None, exists=False):
        self._items = list(items or [])
        self._groups = list(groups or [])
        self._exists = exists or bool(self._items)

    def __bool__(self):
        return self._exists

    def __len__(self):
        return len(self._items)

    def find_containing_groups(self, *_args, **_kwargs):
        return list(self._groups)


class _ResultsStub:
    def __init__(self, *, has_modal=False, has_quantity_selector=False, has_items=False):
        self._has_modal = has_modal
        self._has_quantity_selector = has_quantity_selector
        self._has_items = has_items

    def exists_label(self, label):
        return label == auto_purchase.BaseUILabels.MODAL_HEADER and self._has_modal

    def filter_by_labels(self, labels):
        label_set = set(labels)
        if auto_purchase.BaseUILabels.ITEM in label_set and auto_purchase.BaseUILabels.CARD_COMMODITY in label_set:
            if self._has_items:
                return _DetectionStub(items=[object()], groups=[object()], exists=True)
            return _DetectionStub()
        quantity_labels = {
            auto_purchase.BaseUILabels.QUANTITY_SELECTOR,
            auto_purchase.BaseUILabels.QUANTITY_SELECTOR_ADDED,
            auto_purchase.BaseUILabels.QUANTITY_SELECTOR_REDUCED,
        }
        if label_set & quantity_labels:
            return _DetectionStub(items=[object()] if self._has_quantity_selector else [], exists=self._has_quantity_selector)
        return _DetectionStub()


class _SequenceApp:
    def __init__(self, results, game_utils):
        self._results = list(results)
        self._index = 0
        self.game_utils = game_utils
        self.device = _DeviceStub()
        self.debug_tools = SimpleNamespace(add_box=lambda *args, **kwargs: None, show=lambda: None, clear_all=lambda: None)

    @property
    def latest_results(self):
        return self._results[self._index]

    def advance_results(self):
        if self._index < len(self._results) - 1:
            self._index += 1


class _FilterByLabelStub:
    def __init__(self, element):
        self._element = element

    def first(self):
        return self._element


class _ResultsWithTabBarStub:
    def __init__(self, tabbar_element=None):
        self._tabbar_element = tabbar_element

    def filter_by_label(self, label):
        if label == auto_purchase.BaseUILabels.TAB_BAR:
            return _FilterByLabelStub(self._tabbar_element)
        return _FilterByLabelStub(None)


class _TabBarStub:
    def __init__(self, element):
        self._items = [SimpleNamespace(name="tab0")] if element is not None else []

    def __iter__(self):
        return iter(self._items)

    def __bool__(self):
        return bool(self._items)


class _DailyExchangeRetryApp:
    def __init__(self, results):
        self._results = list(results)
        self._result_index = 0
        self.game_utils = _GameUtilsStub()
        self.game_utils._app = self
        self.debug_tools = SimpleNamespace(add_box=lambda *args, **kwargs: None, show=lambda: None, clear_all=lambda: None)

    @property
    def latest_results(self):
        return self._results[self._result_index]

    def advance_results(self):
        if self._result_index < len(self._results) - 1:
            self._result_index += 1


def test_free_shop_refresh_returns_to_shop_root_then_reopens_daily_exchange(monkeypatch):
    app = _build_app(refresh_shop=True, use_gem_refresh=False)
    exchange_calls = {"count": 0}
    wait_exchange_item_groups_calls = {"count": 0}

    def _exchange_items(_app, _commodity_target):
        exchange_calls["count"] += 1
        if exchange_calls["count"] > 2:
            raise AssertionError("free refresh loop did not terminate as expected")

    def _wait_exchange_item_groups(_app):
        wait_exchange_item_groups_calls["count"] += 1
        return object(), [object()]

    monkeypatch.setattr(auto_purchase, "_exchange_items", _exchange_items)
    monkeypatch.setattr(auto_purchase, "_wait_exchange_item_groups", _wait_exchange_item_groups)
    monkeypatch.setattr(
        auto_purchase,
        "ButtonList",
        _ButtonListFactory(
            [
                _ClickableStub("free_refresh", ButtonText.FREE),
                None,
            ]
        ),
    )

    auto_purchase._handle_tabbar__manny_exchange(app, commodity_target=[])

    assert exchange_calls["count"] == 2
    assert app.device.clicked == ["free_refresh"]
    assert wait_exchange_item_groups_calls["count"] == 1
    assert "back_next_page" in app.game_utils.calls
    assert ("wait_location_update", auto_purchase.GamePageTypes.HOME_TAB.SHOP, 10) in app.game_utils.calls
    assert ("click_button", ButtonText.SHOP.DAILY_EXCHANGE) in app.game_utils.calls
    assert (
        "wait_location_update",
        auto_purchase.GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.DAILY_EXCHANGE,
        10,
    ) in app.game_utils.calls


def test_paid_shop_refresh_consumes_remaining_attempts_and_reopens_daily_exchange(monkeypatch):
    app = _build_app(
        refresh_shop=True,
        use_gem_refresh=True,
        modal_bodies=[
            "更新可能回数 あと2回",
            "更新可能回数 あと2回",
        ],
    )
    exchange_calls = {"count": 0}
    wait_exchange_item_groups_calls = {"count": 0}

    def _exchange_items(_app, _commodity_target):
        exchange_calls["count"] += 1
        if exchange_calls["count"] > 3:
            raise AssertionError("paid refresh loop did not stop when attempts reached zero")

    def _wait_exchange_item_groups(_app):
        wait_exchange_item_groups_calls["count"] += 1
        return object(), [object()]

    monkeypatch.setattr(auto_purchase, "_exchange_items", _exchange_items)
    monkeypatch.setattr(auto_purchase, "_wait_exchange_item_groups", _wait_exchange_item_groups)
    monkeypatch.setattr(
        auto_purchase,
        "ButtonList",
        _ButtonListFactory(
            [
                _ClickableStub("paid_refresh", "リスト更新:450"),
                _ClickableStub("paid_refresh", "リスト更新:450"),
                _ClickableStub("paid_refresh", "リスト更新:450"),
            ]
        ),
    )

    auto_purchase._handle_tabbar__manny_exchange(app, commodity_target=[])

    assert exchange_calls["count"] == 3
    assert app.device.clicked == [
        "paid_refresh",
        "confirm:1",
        "paid_refresh",
        "confirm:2",
    ]
    assert app.game_utils.calls.count("wait_loading") == 4
    assert wait_exchange_item_groups_calls["count"] == 2
    assert app.game_utils.calls.count("back_next_page") == 2
    assert app.game_utils.calls.count(("click_button", ButtonText.SHOP.DAILY_EXCHANGE)) == 2


def test_reset_daily_exchange_to_top_falls_back_to_button_layout_when_shop_location_ocr_times_out(monkeypatch):
    app = _build_app(refresh_shop=True, use_gem_refresh=False)
    app.game_utils._location_wait_failures[auto_purchase.GamePageTypes.HOME_TAB.SHOP] = 1
    wait_exchange_item_groups_calls = {"count": 0}

    def _wait_exchange_item_groups(_app):
        wait_exchange_item_groups_calls["count"] += 1
        return object(), [object()]

    monkeypatch.setattr(auto_purchase, "_wait_exchange_item_groups", _wait_exchange_item_groups)
    monkeypatch.setattr(
        auto_purchase,
        "ButtonList",
        _ButtonListFactory([
            _ClickableStub("pack_tab", ButtonText.SHOP.PACK),
        ]),
    )

    auto_purchase._reset_daily_exchange_to_top(app)

    assert "back_next_page" in app.game_utils.calls
    assert "wait_loading" in app.game_utils.calls
    assert app.game_utils.calls.count("wait_frame_stable") == 3
    assert ("update_current_location", auto_purchase.GamePageTypes.HOME_TAB.SHOP) in app.game_utils.calls
    assert ("click_button", ButtonText.SHOP.DAILY_EXCHANGE) in app.game_utils.calls
    assert wait_exchange_item_groups_calls["count"] == 1


def test_wait_exchange_item_groups_clicks_max_quantity_modal_before_detecting_items(monkeypatch):
    game_utils = _GameUtilsStub()
    app = _SequenceApp(
        results=[
            _ResultsStub(has_modal=True, has_quantity_selector=True, has_items=False),
            _ResultsStub(has_modal=False, has_quantity_selector=False, has_items=True),
        ],
        game_utils=game_utils,
    )
    game_utils._app = app
    game_utils._modal = SimpleNamespace(
        modal_title="確認",
        cancel_button=_ClickableStub("keep_quantity"),
        confirm_button=_ClickableStub("set_max_quantity"),
    )

    monkeypatch.setattr(auto_purchase, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auto_purchase,
        "ButtonList",
        lambda _results: SimpleNamespace(
            get_button_by_text=lambda text, match_config=None: (
                _ClickableStub("set_max_quantity")
                if text == ButtonText.SHOP.SET_MAX_QUANTITY
                else None
            )
        ),
    )

    item_commodity, item_groups = auto_purchase._wait_exchange_item_groups(
        app,
        timeout=0.5,
        interval=0.1,
    )

    assert item_commodity
    assert len(item_groups) == 1
    assert "try_get_modal" in game_utils.calls
    assert (
        "click_modal_button_and_wait_transition",
        "set_max_quantity",
        "確認",
    ) in game_utils.calls


def test_action_daily_exchange_falls_back_to_current_tab_when_tabbar_missing(monkeypatch):
    app = _build_app(refresh_shop=False, use_gem_refresh=False)
    exchange_calls = {"count": 0}

    monkeypatch.setattr(auto_purchase, "_wait_exchange_item_groups", lambda _app: (object(), [object()]))
    monkeypatch.setattr(auto_purchase, "_detect_daily_exchange_tabbar", lambda _app: None)
    monkeypatch.setattr(
        auto_purchase,
        "_exchange_items",
        lambda _app, _commodity_target: exchange_calls.__setitem__("count", exchange_calls["count"] + 1),
    )

    ok = auto_purchase.action__daily_exchange(app)

    assert ok is True
    assert exchange_calls["count"] == 1
    assert ("click_button", ButtonText.SHOP.DAILY_EXCHANGE) in app.game_utils.calls


def test_detect_daily_exchange_tabbar_retries_and_recovers_with_multiframe_results(monkeypatch):
    app = _DailyExchangeRetryApp([
        _ResultsWithTabBarStub(None),
        _ResultsWithTabBarStub(None),
        _ResultsWithTabBarStub(SimpleNamespace(x=10, y=20, w=110, h=50)),
    ])

    monkeypatch.setattr(auto_purchase, "TabBar", _TabBarStub)

    tabbar = auto_purchase._detect_daily_exchange_tabbar(app, retries=3)

    assert tabbar is not None
    assert bool(tabbar) is True
    assert app._result_index == 2
    assert app.game_utils.calls.count(("wait_for_label", auto_purchase.BaseUILabels.TAB_BAR, 2, 0.2)) == 2
