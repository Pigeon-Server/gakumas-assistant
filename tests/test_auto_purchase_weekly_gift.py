from types import SimpleNamespace

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.core.tasks.base_ui import auto_purchase


class _ClickableStub:
    def __init__(self, name: str, text: str = "", disabled: bool = False):
        self.name = name
        self.text = text
        self._disabled = disabled
        self.x = 0
        self.y = 0
        self.w = 100
        self.h = 40

    def is_disabled(self) -> bool:
        return self._disabled


class _ModalStub:
    def __init__(self, title: str, confirm_button=None, cancel_button=None):
        self.modal_title = title
        self.confirm_button = confirm_button
        self.cancel_button = cancel_button


class _DeviceStub:
    def __init__(self):
        self.clicked: list[str] = []

    def click_element(self, element):
        self.clicked.append(element.name)


def test_claim_weekly_gift_closes_disabled_confirm_modal():
    device = _DeviceStub()
    confirm_modal = _ModalStub(
        ModalText.TITLE.PURCHASE_CONFIRMATION,
        confirm_button=_ClickableStub("confirm", ButtonText.CONFIRM, disabled=True),
        cancel_button=_ClickableStub("cancel", ButtonText.CLOSE),
    )
    wait_frame_stable_calls = []

    app = SimpleNamespace(
        device=device,
        game_utils=SimpleNamespace(
            wait_for_modal=lambda *_args, **_kwargs: confirm_modal,
            click_modal_button_and_wait_transition=lambda button, **_kwargs: device.click_element(button) or True,
            wait_frame_stable=lambda: wait_frame_stable_calls.append("wait_frame_stable"),
        ),
    )

    button = _ClickableStub("weekly_free", ButtonText.FREE)

    assert auto_purchase._claim_weekly_gift_button(app, button) is False
    assert device.clicked == ["weekly_free", "cancel"]
    assert wait_frame_stable_calls == ["wait_frame_stable"]


def test_receive_weekly_gift_dismisses_residual_modal_before_back(monkeypatch):
    device = _DeviceStub()
    modal_queue = [
        _ModalStub(
            ModalText.TITLE.PURCHASE_CONFIRMATION,
            confirm_button=_ClickableStub("confirm", ButtonText.CONFIRM),
            cancel_button=_ClickableStub("cancel", ButtonText.CLOSE),
        ),
        None,
    ]
    calls = []

    monkeypatch.setattr(auto_purchase, "_collect_visible_weekly_gifts", lambda _app: 0)
    monkeypatch.setattr(auto_purchase, "_scroll_weekly_gift_page", lambda _app: calls.append("scroll"))
    monkeypatch.setattr(auto_purchase, "check_frame_change", lambda *_args, **_kwargs: True)

    app = SimpleNamespace(
        latest_frame=np.zeros((100, 100, 3), dtype=np.uint8),
        device=device,
        game_utils=SimpleNamespace(
            click_button=lambda text, **_kwargs: calls.append(("click_button", text)),
            wait_location_update=lambda location, timeout=10: calls.append(("wait_location_update", location, timeout)),
            wait_frame_stable=lambda: calls.append("wait_frame_stable"),
            try_get_modal=lambda no_body=False: modal_queue.pop(0),
            click_modal_button_and_wait_transition=lambda button, **_kwargs: device.click_element(button) or True,
            back_next_page=lambda: calls.append("back_next_page"),
            wait_loading=lambda: calls.append("wait_loading"),
            update_current_location=lambda location=None: calls.append(("update_current_location", location)),
        ),
    )

    auto_purchase.action__receive_weekly_gift(app)

    assert device.clicked == ["cancel"]
    assert "back_next_page" in calls
    assert ("click_button", ButtonText.SHOP.PACK) in calls
    assert ("update_current_location", auto_purchase.GamePageTypes.HOME_TAB.SHOP) in calls
