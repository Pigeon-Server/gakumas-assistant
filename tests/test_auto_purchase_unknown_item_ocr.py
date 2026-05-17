import sys
from types import SimpleNamespace

from src.constants.game.text.modal_text import ModalText

class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.tasks.base_ui import auto_purchase


def test_resolve_item_from_modal_ocr_falls_back_to_modal_body_title(monkeypatch):
    item_info = object()
    modal_body = object()
    matched_item = SimpleNamespace(name="[光景]篠澤広のピース")

    def _ocr_modal_item_candidates(image, limit=5):
        if image is item_info:
            return ["デュースアイドルの才能開花や"]
        if image is modal_body:
            return ["[光景]篠澤広のピース"]
        return []

    def _search(candidate, match_config=None):
        if candidate == "[光景]篠澤広のピース":
            return True, matched_item
        return False, None

    monkeypatch.setattr(auto_purchase, "_ocr_modal_item_candidates", _ocr_modal_item_candidates)
    monkeypatch.setattr(auto_purchase.item_db, "search", _search)

    status, db_result, matched_text = auto_purchase._resolve_item_from_modal_ocr(item_info, modal_body)

    assert status is True
    assert db_result == matched_item
    assert matched_text == "[光景]篠澤広のピース"


def test_resolve_item_from_modal_ocr_matches_db_with_first_two_lines(monkeypatch):
    item_info = object()
    modal_body = object()
    matched_item = SimpleNamespace(
        name="ロジックノート（ビジュアル）",
        description="プラン：ロジックのプロデュースアイドルの特訓で使用するアイテム",
    )

    def _ocr_modal_item_candidates(image, limit=5):
        if image is item_info:
            return [
                "ロジックノート(ビジュアル)",
                "プラン:ロジックのプロデュースアイドルの特訓で使用するアイテム",
                "イドルの特訓で使用するアイテム",
            ]
        if image is modal_body:
            return []
        return []

    monkeypatch.setattr(auto_purchase, "_ocr_modal_item_candidates", _ocr_modal_item_candidates)
    monkeypatch.setattr(
        auto_purchase.item_db,
        "search",
        lambda candidate, match_config=None: (False, None),
    )
    monkeypatch.setattr(auto_purchase.item_db, "get_all_item", lambda: [matched_item])

    status, db_result, matched_text = auto_purchase._resolve_item_from_modal_ocr(item_info, modal_body)

    assert status is True
    assert db_result == matched_item
    assert matched_text == (
        "ロジックノート(ビジュアル)"
        "プラン:ロジックのプロデュースアイドルの特訓で使用するアイテム"
    )


class _ClickableStub:
    def __init__(self, name: str, text: str = "", disabled: bool = False):
        self.name = name
        self.text = text
        self._disabled = disabled

    def is_disabled(self) -> bool:
        return self._disabled


class _GameUtilsStub:
    def __init__(self, transition_results, fallback_modal=None, stable_modal=None):
        self._transition_results = list(transition_results)
        self._fallback_modal = fallback_modal
        self._stable_modal = stable_modal
        self.calls = []

    def click_modal_button_and_wait_transition(self, button, **kwargs):
        self.calls.append(("transition", button.name, kwargs["previous_modal_title"]))
        return self._transition_results.pop(0)

    def wait_frame_stable(self):
        self.calls.append(("wait_frame_stable",))

    def try_get_modal(self, no_body=False):
        self.calls.append(("try_get_modal", no_body))
        return self._fallback_modal

    def wait_for_modal(self, title, timeout=2, interval=0.2, no_body=False):
        self.calls.append(("wait_for_modal", title, timeout, interval, no_body))
        return self._stable_modal if self._stable_modal is not None else self._fallback_modal


class _UnknownItemGameUtilsStub:
    def __init__(self, modal):
        self._modal = modal
        self.calls = []

    def wait_loading(self):
        self.calls.append("wait_loading")

    def wait_frame_stable(self):
        self.calls.append("wait_frame_stable")

    def wait_for_modal(self, _title):
        self.calls.append("wait_for_modal")
        modal, self._modal = self._modal, None
        return modal

    def click_modal_button_and_wait_transition(self, button, **kwargs):
        self.calls.append(("transition", button.name, kwargs.get("previous_modal_title")))
        return True

    def wait_label_exist(self, label):
        self.calls.append(("wait_label_exist", label))
        return True


class _DeviceStub:
    def __init__(self):
        self.clicked = []

    def click_element(self, element):
        self.clicked.append(element.name)


class _BoxStub:
    def __init__(self, name: str, x=10, y=20, w=30, h=40, frame=None):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = frame if frame is not None else [[0]]


def test_confirm_exchange_modal_cancels_when_ap_is_insufficient():
    modal = SimpleNamespace(
        modal_title=ModalText.TITLE.EXCHANGE_CONFIRMATION,
        modal_body_text="所持している「AP」が不足しています",
        confirm_button=_ClickableStub("confirm", "決定", disabled=True),
        cancel_button=_ClickableStub("cancel", "キャンセル"),
    )
    game_utils = _GameUtilsStub([True], stable_modal=modal)
    app = SimpleNamespace(game_utils=game_utils)

    assert auto_purchase._confirm_exchange_modal(app, modal, "入手ノート数アップ") is True
    assert game_utils.calls == [
        ("wait_for_modal", ModalText.TITLE.EXCHANGE_CONFIRMATION, 2, 0.2, False),
        ("transition", "cancel", ModalText.TITLE.EXCHANGE_CONFIRMATION),
        ("wait_frame_stable",),
    ]


def test_confirm_exchange_modal_closes_stuck_modal_after_failed_confirm():
    fallback_modal = SimpleNamespace(
        modal_title=ModalText.TITLE.EXCHANGE_CONFIRMATION,
        modal_body_text="",
        confirm_button=_ClickableStub("confirm", "決定"),
        cancel_button=_ClickableStub("cancel", "キャンセル"),
    )
    game_utils = _GameUtilsStub([False, True], fallback_modal=fallback_modal, stable_modal=fallback_modal)
    app = SimpleNamespace(game_utils=game_utils)

    assert auto_purchase._confirm_exchange_modal(app, fallback_modal, "入手ノート数アップ") is True
    assert game_utils.calls == [
        ("wait_for_modal", ModalText.TITLE.EXCHANGE_CONFIRMATION, 2, 0.2, False),
        ("transition", "confirm", ModalText.TITLE.EXCHANGE_CONFIRMATION),
        ("wait_for_modal", ModalText.TITLE.EXCHANGE_CONFIRMATION, 2, 0.2, False),
        ("transition", "confirm", ModalText.TITLE.EXCHANGE_CONFIRMATION),
        ("wait_frame_stable",),
    ]

def test_handle_unknown_item_clicks_item_inner_instead_of_group_box(monkeypatch):
    modal = SimpleNamespace(
        modal_body=object(),
        cancel_button=_ClickableStub("cancel", "キャンセル"),
    )
    app = SimpleNamespace(
        device=_DeviceStub(),
        game_utils=_UnknownItemGameUtilsStub(modal),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        clip_manager=SimpleNamespace(
            item_clip=SimpleNamespace(add_to_memory=lambda *args, **kwargs: None)
        ),
    )
    item_box = _BoxStub("item_box", x=1, y=1, w=500, h=300)
    item_inner = _BoxStub("item_inner", x=100, y=100, w=200, h=180)
    matched_item = SimpleNamespace(name="未在目标中的道具")

    monkeypatch.setattr(auto_purchase, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(auto_purchase, "_read_daily_exchange_money_multiframe", lambda _app: 0)
    monkeypatch.setattr(auto_purchase, "modal_body_extract_item_info", lambda _modal_body: (object(), object()))
    monkeypatch.setattr(
        auto_purchase,
        "_resolve_item_from_modal_ocr",
        lambda _item_info, _modal_body: (True, matched_item, matched_item.name),
    )

    ok = auto_purchase._handle_unknown_item(
        app=app,
        item_box=item_box,
        item_inner=item_inner,
        commodity_target=["其他道具"],
        index=2,
    )

    assert ok is True
    assert app.device.clicked[0] == "item_inner"
    assert "item_box" not in app.device.clicked
