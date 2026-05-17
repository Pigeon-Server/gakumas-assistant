from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.tasks.producer_challenge.context import GameplayPhase, ProduceContext
from src.core.tasks.producer_challenge import ui as ui_module
from src.entity.Game.Page.Types.index import GamePageTypes


def _load_debug_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_produce_debug.py"
    spec = spec_from_file_location("run_produce_debug_test", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


run_produce_debug = _load_debug_module()


class _ResultsStub:
    def __init__(self, name: str, box_count: int):
        self.name = name
        self._box_count = box_count
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self._labels = set()

    def __len__(self):
        return self._box_count

    def __iter__(self):
        return iter([])

    def exists_label(self, label):
        return label in self._labels


class _ProducerPageResultsStub(_ResultsStub):
    def __init__(self, name: str, box_count: int, labels):
        super().__init__(name, box_count)
        self._labels = set(labels)


class _OnlyLoopProbeApp:
    def __init__(self, locations, results_sequence, *, frame=None, ocr_texts=None):
        self._locations = list(locations)
        self._results_sequence = list(results_sequence)
        self._location_read_count = 0
        self._results_read_count = 0
        self.latest_frame = frame if frame is not None else np.zeros((2400, 1080, 3), dtype=np.uint8)
        self._ocr_texts = list(ocr_texts or [])
        self._ocr_read_count = 0
        self.game_utils = SimpleNamespace(update_current_location=self._update_current_location)
        self.ocr_service = SimpleNamespace(ocr=self._ocr)

    def _update_current_location(self):
        index = min(self._location_read_count, len(self._locations) - 1)
        self._location_read_count += 1
        return self._locations[index]

    def _ocr(self, _frame):
        index = min(self._ocr_read_count, len(self._ocr_texts) - 1) if self._ocr_texts else 0
        self._ocr_read_count += 1
        text = self._ocr_texts[index] if self._ocr_texts else ""
        return [SimpleNamespace(text=text)] if text else []

    @property
    def latest_results(self):
        index = min(self._results_read_count, len(self._results_sequence) - 1)
        self._results_read_count += 1
        return self._results_sequence[index]


def test_select_capture_snapshot_prefers_latest_max_box_sample():
    first = _ResultsStub("first", 0)
    second = _ResultsStub("second", 12)
    third = _ResultsStub("third", 12)

    selected = run_produce_debug._select_capture_snapshot([first, second, third])

    assert selected is third


def test_select_phase_probe_prefers_latest_non_unknown_sample():
    first = _ResultsStub("first", 0)
    second = _ResultsStub("second", 12)
    third = _ResultsStub("third", 0)

    selected = run_produce_debug._select_phase_probe([
        (GameplayPhase.UNKNOWN, "transition_empty", first),
        (GameplayPhase.SCHEDULE, "schedule_idle", second),
        (GameplayPhase.UNKNOWN, "transition_empty", third),
    ])

    assert selected == (GameplayPhase.SCHEDULE, "schedule_idle", second)


def test_probe_gameplay_state_uses_retry_window_and_returns_stable_phase(monkeypatch):
    samples = [
        _ResultsStub("first", 0),
        _ResultsStub("second", 12),
        _ResultsStub("third", 0),
    ]
    ctx = ProduceContext()
    ctx.handler_state["unknown_retry_limit"] = 2
    ctx.handler_state["unknown_retry_sleep"] = 0.4
    app = SimpleNamespace()
    ensure_calls = []
    wait_calls = []

    def fake_classify(results, *, ctx=None):  # noqa: ARG001
        if results is samples[1]:
            return GameplayPhase.SCHEDULE, "schedule_idle"
        return GameplayPhase.UNKNOWN, "transition_empty"

    monkeypatch.setattr(
        run_produce_debug,
        "_ensure_debug_model",
        lambda _app, *, model_type: ensure_calls.append(model_type) or samples[0],
    )
    wait_sequence = iter(samples[1:])
    monkeypatch.setattr(
        run_produce_debug,
        "_wait_for_fresh_results",
        lambda _app, _previous_results, timeout=2.0: wait_calls.append(timeout) or next(wait_sequence, None),
    )
    monkeypatch.setattr(ui_module, "classify_gameplay_state", fake_classify)

    phase, position, results = run_produce_debug._probe_gameplay_state(
        app,
        ctx,
        model_type="PRODUCER",
    )

    assert ensure_calls == ["PRODUCER"]
    assert wait_calls == [2.0]
    assert phase == GameplayPhase.SCHEDULE
    assert position == "schedule_idle"
    assert results is samples[1]


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (GamePageTypes.HOME_TAB.PRODUCER, True),
        (GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__IDOL_SELECTION, True),
        (GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__SUPPORT_SELECTION, True),
        (GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_SELECTION, True),
        (GamePageTypes.UNKNOWN, False),
        (GamePageTypes.MAIN_MENU__HOME, False),
    ],
)
def test_is_only_loop_blocked_location(location, expected):
    assert run_produce_debug._is_only_loop_blocked_location(location) is expected


def test_assert_only_loop_starts_in_gameplay_blocks_stable_producer_entry(monkeypatch):
    monkeypatch.setattr(run_produce_debug.time, "sleep", lambda _seconds: None)
    results = _ProducerPageResultsStub(
        "producer_entry",
        4,
        labels={"Producer Challenge: Regular"},
    )
    monkeypatch.setattr(
        run_produce_debug,
        "_probe_gameplay_state",
        lambda *_args, **_kwargs: (GameplayPhase.SCHEDULE, "schedule_idle", results),
    )
    app = _OnlyLoopProbeApp(
        [GamePageTypes.HOME_TAB.PRODUCER] * 3,
        [results, results, results],
    )

    with pytest.raises(RuntimeError, match="当前已处于 producer 入口页或选卡页"):
        run_produce_debug._assert_only_loop_starts_in_gameplay(
            app,
            ProduceContext(),
            probe_count=3,
            probe_interval=0.0,
            required_hits=2,
        )


def test_assert_only_loop_starts_in_gameplay_blocks_unknown_then_producer(monkeypatch):
    monkeypatch.setattr(run_produce_debug.time, "sleep", lambda _seconds: None)
    empty = _ProducerPageResultsStub("empty", 0, labels=set())
    producer = _ProducerPageResultsStub(
        "producer_entry",
        4,
        labels={"Producer Challenge: Pro"},
    )
    monkeypatch.setattr(
        run_produce_debug,
        "_probe_gameplay_state",
        lambda *_args, **_kwargs: (GameplayPhase.SCHEDULE, "schedule_idle", producer),
    )
    app = _OnlyLoopProbeApp(
        [
            GamePageTypes.UNKNOWN,
            GamePageTypes.HOME_TAB.PRODUCER,
            GamePageTypes.HOME_TAB.PRODUCER,
        ],
        [empty, producer, producer],
    )

    with pytest.raises(RuntimeError, match="当前已处于 producer 入口页或选卡页"):
        run_produce_debug._assert_only_loop_starts_in_gameplay(
            app,
            ProduceContext(),
            probe_count=3,
            probe_interval=0.0,
            required_hits=2,
        )


def test_assert_only_loop_starts_in_gameplay_allows_single_false_positive(monkeypatch):
    monkeypatch.setattr(run_produce_debug.time, "sleep", lambda _seconds: None)
    producer = _ProducerPageResultsStub(
        "producer_entry",
        4,
        labels={"Producer Challenge: Master"},
    )
    gameplay_like = _ProducerPageResultsStub("gameplay", 6, labels=set())
    monkeypatch.setattr(
        run_produce_debug,
        "_probe_gameplay_state",
        lambda *_args, **_kwargs: (GameplayPhase.SCHEDULE, "schedule_idle", gameplay_like),
    )
    app = _OnlyLoopProbeApp(
        [
            GamePageTypes.HOME_TAB.PRODUCER,
            GamePageTypes.UNKNOWN,
            GamePageTypes.UNKNOWN,
        ],
        [producer, gameplay_like, gameplay_like],
    )

    run_produce_debug._assert_only_loop_starts_in_gameplay(
        app,
        ProduceContext(),
        probe_count=3,
        probe_interval=0.0,
        required_hits=2,
    )


def test_assert_only_loop_starts_in_gameplay_blocks_unknown_phase(monkeypatch):
    monkeypatch.setattr(run_produce_debug.time, "sleep", lambda _seconds: None)
    gameplay_like = _ProducerPageResultsStub("unknown", 0, labels=set())
    monkeypatch.setattr(
        run_produce_debug,
        "_probe_gameplay_state",
        lambda *_args, **_kwargs: (GameplayPhase.UNKNOWN, "transition_empty", gameplay_like),
    )
    app = _OnlyLoopProbeApp(
        [GamePageTypes.UNKNOWN] * 3,
        [gameplay_like, gameplay_like, gameplay_like],
    )

    with pytest.raises(RuntimeError, match="当前画面未识别为 gameplay 局内"):
        run_produce_debug._assert_only_loop_starts_in_gameplay(
            app,
            ProduceContext(),
            probe_count=3,
            probe_interval=0.0,
            required_hits=2,
        )


def test_looks_like_only_loop_blocked_producer_page_by_ocr_detects_entry(monkeypatch):
    app = _OnlyLoopProbeApp(
        [GamePageTypes.UNKNOWN],
        [_ProducerPageResultsStub("empty", 0, labels=set())],
        ocr_texts=["プロデュース", "レギュラー プロ マスター"],
    )

    assert run_produce_debug._looks_like_only_loop_blocked_producer_page_by_ocr(app) is True


def test_assert_only_loop_starts_in_gameplay_blocks_ocr_detected_producer_entry(monkeypatch):
    monkeypatch.setattr(run_produce_debug.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        run_produce_debug,
        "_probe_gameplay_state",
        lambda *_args, **_kwargs: (GameplayPhase.SCHEDULE, "schedule_idle", None),
    )
    app = _OnlyLoopProbeApp(
        [GamePageTypes.UNKNOWN] * 3,
        [_ProducerPageResultsStub("empty", 0, labels=set())] * 3,
        ocr_texts=[
            "プロデュース",
            "レギュラー プロ マスター",
            "プロデュース",
            "レギュラー プロ マスター",
            "プロデュース",
            "レギュラー プロ マスター",
        ],
    )

    with pytest.raises(RuntimeError, match="当前已处于 producer 入口页或选卡页"):
        run_produce_debug._assert_only_loop_starts_in_gameplay(
            app,
            ProduceContext(),
            probe_count=3,
            probe_interval=0.0,
            required_hits=2,
        )
