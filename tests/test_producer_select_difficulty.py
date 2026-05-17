from types import SimpleNamespace

import pytest

from src.constants.game.text.modal_text import ModalText
from src.constants.game.text.produce_text import ProduceText
from src.core.exceptions.TaskException import TaskUserMessage
from src.core.tasks.producer_challenge.steps.entry.select_difficulty import SelectDifficultyStep


def test_wait_idol_selection_page_cancels_ap_recovery_when_disallowed(monkeypatch):
    events: list[tuple] = []
    modal = SimpleNamespace(
        modal_title=ModalText.TITLE.CONFIRM,
        confirm_button=SimpleNamespace(name="confirm"),
        cancel_button=SimpleNamespace(name="cancel"),
    )

    class _ResultsStub:
        @staticmethod
        def exists_label(_label):
            return False

    app = SimpleNamespace(
        latest_frame=object(),
        latest_results=_ResultsStub(),
        game_utils=SimpleNamespace(
            try_get_modal=lambda no_body=True, require_header=False: modal,
            go_home=lambda: events.append(("go_home",)),
        ),
    )

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.select_difficulty.ocr_text",
        lambda _frame: ProduceText.AP_SHORTAGE,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.select_difficulty.click_modal_action_with_retry",
        lambda _app, current_modal, **kwargs: events.append(
            ("cancel_modal", current_modal.modal_title, kwargs.get("prefer_confirm"))
        ) or True,
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.select_difficulty.wait_frame_stable",
        lambda _app, timeout=0: events.append(("wait_frame_stable", timeout)),
    )
    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.steps.entry.select_difficulty.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(TaskUserMessage, match="已取消并返回主页"):
        SelectDifficultyStep._wait_idol_selection_page(
            app,
            SimpleNamespace(allow_ap_recovery=False),
        )

    assert ("cancel_modal", ModalText.TITLE.CONFIRM, False) in events
    assert ("go_home",) in events
