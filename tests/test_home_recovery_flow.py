import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.entity.Game.Page.Types.index import GamePageTypes
from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks import task_register
from src.core.tasks.base_ui.start_game import action__wait_enter_home, _handle__modal_boxes
from src.core.tasks.task_register import register_tasks
from src.core.tasks.base_ui import goto_pages


class _TaskQueueStub:
    def __init__(self):
        self.tasks = {}

    def register_task(self, task_id, *args, **kwargs):
        def decorator(func):
            self.tasks[task_id] = func
            return func

        return decorator

    def register_pre_queue_start(self):
        def decorator(func):
            return func

        return decorator


def test_start_game_waits_for_home_when_location_is_unknown(monkeypatch):
    processor = SimpleNamespace(task_queue=_TaskQueueStub())
    register_tasks(processor)
    task = processor.task_queue.tasks["start_game"]

    monkeypatch.setattr(task_register, "sleep", lambda *_args, **_kwargs: None)

    calls = {"click_start_game": 0, "wait_enter_home": 0}

    monkeypatch.setattr(
        "src.core.tasks.base_ui.start_game.action__click_start_game",
        lambda _app: calls.__setitem__("click_start_game", calls["click_start_game"] + 1),
    )
    monkeypatch.setattr(
        "src.core.tasks.base_ui.start_game.action__wait_enter_home",
        lambda _app: calls.__setitem__("wait_enter_home", calls["wait_enter_home"] + 1),
    )

    updates = [GamePageTypes.UNKNOWN, GamePageTypes.MAIN_MENU__HOME]

    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            update_current_location=lambda: updates.pop(0),
            wait_loading=lambda: None,
        )
    )

    task(app)

    assert calls == {
        "click_start_game": 0,
        "wait_enter_home": 1,
    }


def test_back_home_falls_back_to_wait_enter_home_after_location_timeout(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "src.core.tasks.base_ui.start_game.action__wait_enter_home",
        lambda _app: calls.append("wait_enter_home"),
    )

    def _update_current_location(new_location=None):
        calls.append(("update_current_location", new_location))
        return GamePageTypes.UNKNOWN if new_location is None else new_location

    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            update_current_location=_update_current_location,
            go_home=lambda: calls.append("go_home"),
            wait_location_update=lambda _target: (_ for _ in ()).throw(TimeoutError("timeout")),
        )
    )

    goto_pages._back_home(app)

    assert calls == [
        ("update_current_location", None),
        "go_home",
        "wait_enter_home",
        ("update_current_location", GamePageTypes.MAIN_MENU__HOME),
    ]


def test_back_home_falls_back_to_wait_enter_home_after_go_home_failure(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "src.core.tasks.base_ui.start_game.action__wait_enter_home",
        lambda _app: calls.append("wait_enter_home"),
    )

    def _update_current_location(new_location=None):
        calls.append(("update_current_location", new_location))
        return GamePageTypes.UNKNOWN if new_location is None else new_location

    app = SimpleNamespace(
        game_utils=SimpleNamespace(
            update_current_location=_update_current_location,
            go_home=lambda: (_ for _ in ()).throw(RuntimeError("Going home failed")),
            wait_location_update=lambda _target: calls.append(("wait_location_update", _target)),
        )
    )

    goto_pages._back_home(app)

    assert calls == [
        ("update_current_location", None),
        "wait_enter_home",
        ("update_current_location", GamePageTypes.MAIN_MENU__HOME),
    ]


def test_wait_enter_home_closes_special_producer_page(monkeypatch):
    monkeypatch.setattr("src.core.tasks.base_ui.start_game.sleep", lambda *_args, **_kwargs: None)

    state = {"location": GamePageTypes.PRODUCER__MEMORY_DETAIL}
    calls = []

    def _update_current_location():
        return state["location"]

    def _back_next_page():
        calls.append("back_next_page")
        state["location"] = GamePageTypes.MAIN_MENU__HOME
        return True

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            filter_by_label=lambda label: [object()] if label == BaseUILabels.TAB_HOME and state["location"] == GamePageTypes.MAIN_MENU__HOME else []
        ),
        latest_frame=SimpleNamespace(shape=(1920, 1080, 3)),
        device=SimpleNamespace(click_element=lambda *_args, **_kwargs: None, click=lambda *_args, **_kwargs: None),
        game_utils=SimpleNamespace(
            update_current_location=_update_current_location,
            back_next_page=_back_next_page,
            go_home=lambda *args, **kwargs: calls.append("go_home"),
        ),
    )

    action__wait_enter_home(app)

    assert calls == ["back_next_page"]


def test_wait_enter_home_leaves_shop_sub_page_before_confirming_home(monkeypatch):
    monkeypatch.setattr("src.core.tasks.base_ui.start_game.sleep", lambda *_args, **_kwargs: None)

    state = {"location": GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.DAILY_EXCHANGE}
    calls = []

    def _update_current_location():
        return state["location"]

    def _go_home(max_try=2):
        calls.append(("go_home", max_try))
        state["location"] = GamePageTypes.MAIN_MENU__HOME
        return True

    app = SimpleNamespace(
        latest_results=SimpleNamespace(
            filter_by_label=lambda label: [object()] if label == BaseUILabels.TAB_HOME and state["location"] == GamePageTypes.MAIN_MENU__HOME else []
        ),
        latest_frame=SimpleNamespace(shape=(1920, 1080, 3)),
        device=SimpleNamespace(click_element=lambda *_args, **_kwargs: None, click=lambda *_args, **_kwargs: None),
        game_utils=SimpleNamespace(
            update_current_location=_update_current_location,
            back_next_page=lambda: calls.append("back_next_page"),
            go_home=_go_home,
            click_on_label=lambda label: calls.append(("click_on_label", label)),
            wait_loading=lambda: calls.append("wait_loading"),
        ),
    )

    action__wait_enter_home(app)

    assert calls and calls[0] == ("go_home", 2)


def test_handle_modal_boxes_dismisses_unknown_single_action_modal(monkeypatch):
    monkeypatch.setattr("src.core.tasks.base_ui.start_game.sleep", lambda *_args, **_kwargs: None)

    clicked = []
    wait_loading_calls = []
    modal = SimpleNamespace(
        modal_title="親愛度アイドル選択",
        modal_body_text="",
        confirm_button=None,
        cancel_button=SimpleNamespace(name="close"),
    )
    monkeypatch.setattr("src.core.tasks.base_ui.start_game.get_modal", lambda *_args, **_kwargs: modal)

    app = SimpleNamespace(
        latest_results=SimpleNamespace(),
        device=SimpleNamespace(click_element=lambda button: clicked.append(button.name)),
        game_utils=SimpleNamespace(wait_loading=lambda: wait_loading_calls.append("wait_loading")),
    )

    _handle__modal_boxes(app)

    assert clicked == ["close"]
    assert wait_loading_calls == ["wait_loading"]


def test_handle_modal_boxes_restarts_start_flow_for_data_update(monkeypatch):
    monkeypatch.setattr("src.core.tasks.base_ui.start_game.sleep", lambda *_args, **_kwargs: None)

    clicked = []
    wait_loading_calls = []
    start_click_calls = []
    modal = SimpleNamespace(
        modal_title="データ更新",
        modal_body_text="",
        confirm_button=None,
        cancel_button=SimpleNamespace(name="cancel"),
    )
    monkeypatch.setattr("src.core.tasks.base_ui.start_game.get_modal", lambda *_args, **_kwargs: modal)
    monkeypatch.setattr(
        "src.core.tasks.base_ui.start_game.action__click_start_game",
        lambda _app: start_click_calls.append("action__click_start_game"),
    )

    app = SimpleNamespace(
        latest_results=SimpleNamespace(),
        device=SimpleNamespace(click_element=lambda button: clicked.append(button.name)),
        game_utils=SimpleNamespace(wait_loading=lambda: wait_loading_calls.append("wait_loading")),
    )

    _handle__modal_boxes(app)

    assert clicked == ["cancel"]
    assert wait_loading_calls == ["wait_loading"]
    assert start_click_calls == ["action__click_start_game"]


def test_handle_modal_boxes_dismisses_exchange_confirmation_modal(monkeypatch):
    monkeypatch.setattr("src.core.tasks.base_ui.start_game.sleep", lambda *_args, **_kwargs: None)

    clicked = []
    wait_loading_calls = []
    modal = SimpleNamespace(
        modal_title=ModalText.TITLE.EXCHANGE_CONFIRMATION,
        modal_body_text="",
        confirm_button=SimpleNamespace(name="confirm"),
        cancel_button=SimpleNamespace(name="cancel"),
    )
    monkeypatch.setattr("src.core.tasks.base_ui.start_game.get_modal", lambda *_args, **_kwargs: modal)

    app = SimpleNamespace(
        latest_results=SimpleNamespace(),
        device=SimpleNamespace(click_element=lambda button: clicked.append(button.name)),
        game_utils=SimpleNamespace(wait_loading=lambda: wait_loading_calls.append("wait_loading")),
    )

    _handle__modal_boxes(app)

    assert clicked == ["cancel"]
    assert wait_loading_calls == ["wait_loading"]
