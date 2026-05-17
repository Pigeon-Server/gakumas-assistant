import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.core.tasks.base_ui import claim_pass_rewards


def test_process_modal_skips_modal_parse_without_modal_header(monkeypatch):
    try_get_modal_calls = {"count": 0}
    wait_frame_stable_calls = []

    monkeypatch.setattr(
        claim_pass_rewards,
        "ButtonList",
        lambda _results: SimpleNamespace(get_button_by_text=lambda *_args, **_kwargs: None),
    )

    app = SimpleNamespace(
        latest_results=SimpleNamespace(exists_label=lambda _label: False),
        game_utils=SimpleNamespace(
            wait_frame_stable=lambda *args, **kwargs: wait_frame_stable_calls.append((args, kwargs)),
            try_get_modal=lambda *_args, **_kwargs: try_get_modal_calls.__setitem__("count", try_get_modal_calls["count"] + 1),
        ),
        device=SimpleNamespace(click_element=lambda *_args, **_kwargs: None),
    )

    claim_pass_rewards._process_modal(app)

    assert try_get_modal_calls["count"] == 0
    assert len(wait_frame_stable_calls) == 1


def test_handle_collect_modal_skips_modal_parse_without_modal_header(monkeypatch):
    try_get_modal_calls = {"count": 0}

    monkeypatch.setattr(claim_pass_rewards, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        claim_pass_rewards,
        "ButtonList",
        lambda _results: SimpleNamespace(get_button_by_text=lambda *_args, **_kwargs: None),
    )

    app = SimpleNamespace(
        latest_results=SimpleNamespace(exists_label=lambda _label: False),
        game_utils=SimpleNamespace(
            try_get_modal=lambda *_args, **_kwargs: try_get_modal_calls.__setitem__("count", try_get_modal_calls["count"] + 1),
        ),
        device=SimpleNamespace(click_element=lambda *_args, **_kwargs: None),
    )

    claim_pass_rewards._handle_collect_modal(app, max_wait=0)

    assert try_get_modal_calls["count"] == 0


def test_process_modal_parses_headerless_close_modal(monkeypatch):
    wait_frame_stable_calls = []
    try_get_modal_calls = []
    clicked = []
    close_button = SimpleNamespace(name=ButtonText.CLOSE)
    modal = SimpleNamespace(
        modal_title=ModalText.TITLE.MISSION_PASS_PT_ACQUIRED,
        cancel_button=close_button,
        confirm_button=None,
    )
    responses = [modal, None]

    monkeypatch.setattr(
        claim_pass_rewards,
        "ButtonList",
        lambda _results: SimpleNamespace(get_button_by_text=lambda *_args, **_kwargs: close_button),
    )

    app = SimpleNamespace(
        latest_results=SimpleNamespace(exists_label=lambda _label: False),
        game_utils=SimpleNamespace(
            wait_frame_stable=lambda *args, **kwargs: wait_frame_stable_calls.append((args, kwargs)),
            try_get_modal=lambda *args, **kwargs: (try_get_modal_calls.append((args, kwargs)), responses.pop(0))[1],
        ),
        device=SimpleNamespace(click_element=lambda button: clicked.append(button.name)),
    )

    claim_pass_rewards._process_modal(app)

    assert clicked == [ButtonText.CLOSE]
    assert try_get_modal_calls == [
        ((), {"no_body": True, "require_header": False}),
        ((), {"no_body": True, "require_header": False}),
    ]
    assert len(wait_frame_stable_calls) == 1


def test_handle_collect_modal_closes_mission_pass_point_modal(monkeypatch):
    monkeypatch.setattr(claim_pass_rewards, "sleep", lambda *_args, **_kwargs: None)

    clicked = []
    close_button = SimpleNamespace(name=ButtonText.CLOSE)
    modal = SimpleNamespace(
        modal_title=ModalText.TITLE.MISSION_PASS_PT_ACQUIRED,
        cancel_button=close_button,
        confirm_button=None,
    )

    monkeypatch.setattr(
        claim_pass_rewards,
        "ButtonList",
        lambda _results: SimpleNamespace(get_button_by_text=lambda *_args, **_kwargs: close_button),
    )

    app = SimpleNamespace(
        latest_results=SimpleNamespace(exists_label=lambda _label: False),
        game_utils=SimpleNamespace(try_get_modal=lambda *_args, **_kwargs: modal),
        device=SimpleNamespace(click_element=lambda button: clicked.append(button.name)),
    )

    claim_pass_rewards._handle_collect_modal(app, max_wait=0)

    assert clicked == [ButtonText.CLOSE]
