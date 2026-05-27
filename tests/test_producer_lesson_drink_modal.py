from types import SimpleNamespace

import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import lesson as lesson_module


class _ResultsStub:
    def __init__(self, label_boxes=None):
        self.label_boxes = label_boxes or {}

    def filter_by_label(self, label):
        return list(self.label_boxes.get(label, []))


class _HeaderBox(SimpleNamespace):
    pass


class _DrinkBox(SimpleNamespace):
    pass


def _header_box(text: str, x1: int = 80, y1: int = 1260, x2: int = 1000, y2: int = 1385):
    box = _HeaderBox(x=x1, y=y1, w=x2, h=y2)
    box.frame = SimpleNamespace(_ocr_text=text, size=1)
    box._ocr_text = text
    return box


def _drink_box(x1: int, y1: int, x2: int, y2: int):
    box = _DrinkBox(x=x1, y=y1, w=x2, h=y2, cx=(x1 + x2) // 2, cy=(y1 + y2) // 2)
    box.frame = np.zeros((max(y2 - y1, 1), max(x2 - x1, 1), 3), dtype=np.uint8)
    return box


def _fake_ocr_text(image):
    if image is None:
        return ""
    if hasattr(image, "_ocr_text"):
        return getattr(image, "_ocr_text", "")
    return ""


def _fake_enrich_drink_metadata(db_id: str):
    return {"display_name": f"drink:{db_id}", "db_id": db_id}


def _fake_apply_resolution(cand, resolution):
    cand.db_id = resolution.db_id
    cand.action_id = resolution.action_id
    cand.title = resolution.display_name
    cand.source = resolution.source
    cand.confidence = resolution.confidence
    cand.metadata.update(resolution.metadata or {})
    cand.metadata["db_id"] = resolution.db_id
    cand.metadata["display_name"] = resolution.display_name
    cand.metadata["available"] = True
    cand.metadata.pop("identity_unresolved", None)


def _fake_score_produce_drink_metadata(_metadata, **_kwargs):
    return 42.0


def _make_drink_candidate(index: int = 0, title: str = "P饮料1"):
    return lesson_module.LessonCardCandidate(
        index=index,
        label=ProducerLabels.P_DRINK,
        title=title,
        selected=False,
        box=_drink_box(100, 1750, 200, 1840),
        action_id="",
        db_id="",
        metadata={"battle_drink_slot": index + 1, "candidate_type": "battle_p_drink", "raw_ocr_title": title},
    )


def _make_drink_payload(*, db_id: str = "", available: bool = True, metadata: dict | None = None):
    meta = {"available": available, **(metadata or {})}
    return {
        "id": f"produce_drink:{db_id}" if db_id else "produce_drink_unknown:test",
        "db_id": db_id,
        "available": available,
        "metadata": meta,
        "name": meta.get("display_name", "饮料"),
        "label": ProducerLabels.P_DRINK,
    }


def _make_card_payload():
    return {
        "id": "produce_card:test:0",
        "db_id": "test",
        "available": True,
        "metadata": {"available": True, "description": "打分;+;10"},
        "name": "测试卡",
        "label": ProducerLabels.SKILL_CARD_ACTIVE,
    }


def _make_llm_snapshot():
    return {
        "play_count": {"remaining": 1},
        "resources": {},
        "stamina": 30,
        "max_stamina": 30,
        "genki": 0,
    }


class _SeqResultsApp:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self._index = 0
        self.device = SimpleNamespace(click_element=lambda _el: None)
        self.latest_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.debug_tools = None

    @property
    def latest_results(self):
        idx = min(self._index, len(self._sequence) - 1)
        self._index += 1
        return self._sequence[idx]


def _box(x1: int, y1: int, x2: int, y2: int):
    return SimpleNamespace(x=x1, y=y1, w=x2, h=y2)


def _ocr_item(x: int, y: int, w: int, h: int, text: str, confidence: float):
    return SimpleNamespace(x=x, y=y, w=w, h=h, text=text, confidence=confidence)


def test_extract_drink_modal_name_candidates_prefers_name_over_effect_text(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    results = _ResultsStub({
        BaseUILabels.MODAL_HEADER: [_header_box("Pドリンク詳細")],
        ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
    })
    ocr_items = [
        _ocr_item(32, 20, 220, 46, "Pドリンク詳細", 0.99),
        _ocr_item(228, 136, 286, 48, "ルイボスティー", 0.42),
        _ocr_item(770, 130, 92, 42, "捨てる", 0.91),
        _ocr_item(230, 248, 210, 42, "好印象+3", 0.95),
    ]

    class _FakeOCRService:
        def ocr(self, _img):
            return ocr_items

    monkeypatch.setattr("src.core.inference.ocr_engine.OCRService", _FakeOCRService)
    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)

    names = lesson_module._extract_drink_modal_name_candidates(results, frame)

    assert names
    assert names[0] == "ルイボスティー"
    assert "好印象+3" not in names


def test_extract_drink_modal_name_candidates_prefers_white_panel_merged_title(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    results = _ResultsStub({
        BaseUILabels.MODAL_HEADER: [_header_box("Pドリンク詳細", 120, 1180, 960, 1300)],
        ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
    })

    class _OCRResultList(list):
        def auto_merge_lines(self, **_kwargs):
            return list(self)

    merged_lines = _OCRResultList([
        _ocr_item(180, 120, 280, 52, "ミックススムージー", 0.85),
        _ocr_item(700, 122, 96, 42, "捨てる", 0.95),
        _ocr_item(190, 250, 210, 42, "元気+8", 0.92),
    ])
    fallback_items = [
        _ocr_item(20, 20, 180, 40, "Pドリンク詳細", 0.99),
        _ocr_item(700, 122, 96, 42, "捨てる", 0.95),
    ]

    class _FakeOCRService:
        def __init__(self):
            self.calls = 0

        def ocr(self, _img):
            self.calls += 1
            if self.calls == 1:
                return merged_lines
            return fallback_items

    monkeypatch.setattr("src.core.inference.ocr_engine.OCRService", _FakeOCRService)
    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)
    monkeypatch.setattr(
        lesson_module,
        "detect_bottom_white_modal_region",
        lambda *_args, **_kwargs: (100, 1320, 980, 2050),
    )

    names = lesson_module._extract_drink_modal_name_candidates(results, frame)

    assert names
    assert names[0] == "ミックススムージー"


def test_ensure_drink_cache_scope_clears_stale_cache_on_week_change():
    ctx = ProduceContext()
    ctx.current_week = 2
    ctx.handler_state[lesson_module._DRINK_CACHE_SCOPE_KEY] = ("lesson", 1)
    ctx.handler_state[lesson_module._DRINK_CACHE_KEY] = {(120, 1800): {"db_id": "pdrink_00-2-001"}}
    ctx.handler_state[lesson_module._DRINK_PROBE_COUNT_KEY] = {(120, 1800): 2}

    lesson_module._ensure_drink_cache_scope(ctx, phase="lesson")

    assert ctx.handler_state[lesson_module._DRINK_CACHE_SCOPE_KEY] == ("lesson", 2)
    assert ctx.handler_state[lesson_module._DRINK_CACHE_KEY] == {}
    assert ctx.handler_state[lesson_module._DRINK_PROBE_COUNT_KEY] == {}


def test_wait_drink_modal_visibility_requires_stable_consecutive_frames(monkeypatch):
    modal_only = _ResultsStub({
        BaseUILabels.MODAL_HEADER: [_header_box("Pドリンク詳細")],
    })
    modal_with_cancel = _ResultsStub({
        BaseUILabels.MODAL_HEADER: [_header_box("Pドリンク詳細")],
        ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
    })
    no_modal = _ResultsStub({})
    app = _SeqResultsApp([
        modal_only,
        no_modal,
        modal_with_cancel,
        modal_with_cancel,
    ])
    monkeypatch.setattr(lesson_module.time, "sleep", lambda _sec: None)
    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)

    ok = lesson_module._wait_drink_modal_visibility(
        app,
        expected_visible=True,
        require_action_button=True,
        reason="test_stable_open",
    )

    assert ok is True


def test_wait_drink_modal_visibility_accepts_single_open_hit_when_header_ocr_is_dirty(monkeypatch):
    dirty_modal = _ResultsStub({
        BaseUILabels.MODAL_HEADER: [_header_box("-「")],
        ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
    })
    no_modal = _ResultsStub({})
    app = _SeqResultsApp([
        no_modal,
        dirty_modal,
        no_modal,
    ])
    monkeypatch.setattr(lesson_module.time, "sleep", lambda _sec: None)
    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)

    ok = lesson_module._wait_drink_modal_visibility(
        app,
        expected_visible=True,
        require_action_button=True,
        reason="test_dirty_header_open",
    )

    assert ok is True


def test_resolve_unidentified_drinks_waits_modal_close_then_open(monkeypatch):
    candidate = _make_drink_candidate()
    app = SimpleNamespace(
        latest_results=_ResultsStub({
            BaseUILabels.MODAL_HEADER: [_header_box("Pドリンク詳細")],
            ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
        }),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        device=SimpleNamespace(click_element=lambda _box: None),
        debug_tools=None,
    )
    ctx = ProduceContext()
    ctx.current_week = 3

    wait_calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)
    monkeypatch.setattr(lesson_module, "_enrich_drink_metadata", _fake_enrich_drink_metadata)
    monkeypatch.setattr(lesson_module, "score_produce_drink_metadata", _fake_score_produce_drink_metadata)
    monkeypatch.setattr(lesson_module, "_is_drink_modal_title", lambda _text: True)
    monkeypatch.setattr(
        lesson_module,
        "_wait_drink_modal_visibility",
        lambda *_args, expected_visible, require_action_button=False, reason="", **_kwargs: (
            wait_calls.append((expected_visible, require_action_button)) or True
        ),
    )
    monkeypatch.setattr(
        lesson_module,
        "_extract_drink_modal_name_candidates",
        lambda *_args, **_kwargs: ["ルイボスティー"],
    )
    monkeypatch.setattr(
        lesson_module,
        "resolve_produce_drink_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            db_id="pdrink_02-1-006",
            action_id="produce_drink:pdrink_02-1-006",
            display_name="ルイボスティー",
            source="ocr_modal",
            confidence=0.95,
            metadata={"display_name": "ルイボスティー", "db_id": "pdrink_02-1-006"},
        ),
    )
    monkeypatch.setattr(lesson_module, "_apply_resolution", _fake_apply_resolution)
    monkeypatch.setattr(lesson_module, "_learn_drink_clip_from_db_id", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lesson_module, "_cancel_drink_modal", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(lesson_module.time, "sleep", lambda _sec: None)

    lesson_module._resolve_unidentified_drinks_via_modal(app, ctx, [candidate])

    assert wait_calls[:2] == [(False, False), (True, True)]
    assert candidate.db_id == "pdrink_02-1-006"
    assert candidate.action_id == "produce_drink:pdrink_02-1-006"
    assert candidate.metadata["available"] is True


def test_resolve_unidentified_drinks_allows_dirty_header_when_modal_name_is_readable(monkeypatch):
    candidate = _make_drink_candidate()
    results = _ResultsStub({
        BaseUILabels.MODAL_HEADER: [_header_box("-「")],
        ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
    })
    app = SimpleNamespace(
        latest_results=results,
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        device=SimpleNamespace(click_element=lambda _box: None),
        debug_tools=None,
    )
    ctx = ProduceContext()
    ctx.current_week = 3

    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)
    monkeypatch.setattr(lesson_module, "_enrich_drink_metadata", _fake_enrich_drink_metadata)
    monkeypatch.setattr(lesson_module, "score_produce_drink_metadata", _fake_score_produce_drink_metadata)
    monkeypatch.setattr(
        lesson_module,
        "_wait_drink_modal_visibility",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        lesson_module,
        "_extract_drink_modal_name_candidates",
        lambda *_args, **_kwargs: ["ルイボスティー"],
    )
    monkeypatch.setattr(
        lesson_module,
        "resolve_produce_drink_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            db_id="pdrink_02-1-006",
            action_id="produce_drink:pdrink_02-1-006",
            display_name="ルイボスティー",
            source="ocr_modal",
            confidence=0.95,
            metadata={"display_name": "ルイボスティー", "db_id": "pdrink_02-1-006"},
        ),
    )
    monkeypatch.setattr(lesson_module, "_apply_resolution", _fake_apply_resolution)
    monkeypatch.setattr(lesson_module, "_learn_drink_clip_from_db_id", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lesson_module, "_cancel_drink_modal", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(lesson_module.time, "sleep", lambda _sec: None)

    lesson_module._resolve_unidentified_drinks_via_modal(app, ctx, [candidate])

    assert candidate.db_id == "pdrink_02-1-006"
    assert candidate.metadata["available"] is True
    assert candidate.metadata.get("modal_title_mismatch") is None


def test_is_drink_modal_visible_accepts_header_ocr_variant(monkeypatch):
    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)
    monkeypatch.setattr(lesson_module, "_is_drink_modal_title", lambda text: "pドリンク詳绅" in lesson_module.normalize_lookup_text(text) or "pドリンク詳細" in lesson_module.normalize_lookup_text(text))
    results = _ResultsStub({
        BaseUILabels.MODAL_HEADER: [_header_box("Pドリンク詳绅")],
        ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
    })

    assert lesson_module._is_drink_modal_visible(results, require_action_button=True) is True


def test_is_drink_modal_visible_accepts_header_and_buttons_as_fallback(monkeypatch):
    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)
    results = _ResultsStub({
        BaseUILabels.MODAL_HEADER: [_header_box("-「")],
        ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
    })

    assert lesson_module._is_drink_modal_visible(
        results,
        require_action_button=True,
        allow_header_fallback=True,
    ) is True


def test_resolve_unidentified_drink_marks_unavailable_when_db_id_missing(monkeypatch):
    candidate = _make_drink_candidate()
    app = SimpleNamespace(
        latest_results=_ResultsStub({
            BaseUILabels.MODAL_HEADER: [_header_box("Pドリンク詳細")],
            ProducerLabels.CANCEL_BUTTON: [_box(160, 2060, 480, 2180)],
        }),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        device=SimpleNamespace(click_element=lambda _box: None),
        debug_tools=None,
    )
    ctx = ProduceContext()
    ctx.current_week = 3

    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)
    monkeypatch.setattr(
        lesson_module,
        "_wait_drink_modal_visibility",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        lesson_module,
        "_extract_drink_modal_name_candidates",
        lambda *_args, **_kwargs: ["ルイボスティー"],
    )
    monkeypatch.setattr(
        lesson_module,
        "resolve_produce_drink_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            db_id="",
            action_id="produce_drink_unknown:test",
            display_name="ルイボスティー",
            source="unresolved",
            confidence=0.0,
            metadata={"unresolved": True},
        ),
    )
    monkeypatch.setattr(lesson_module, "_cancel_drink_modal", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(lesson_module.time, "sleep", lambda _sec: None)

    lesson_module._resolve_unidentified_drinks_via_modal(app, ctx, [candidate])

    assert candidate.db_id == ""
    assert candidate.metadata["available"] is False
    assert candidate.metadata["identity_unresolved"] is True
    assert "db_id" in candidate.metadata
    assert candidate.metadata["db_id"] == ""


def test_decision_annotate_battle_candidate_availability_disables_unresolved_drink():
    ctx = ProduceContext()
    payloads = [
        _make_drink_payload(db_id="", available=True, metadata={"identity_unresolved": True}),
        _make_card_payload(),
    ]

    from src.core.tasks.producer_challenge.gameplay import decision as decision_module

    decision_module._annotate_battle_candidate_availability(
        ctx,
        phase="lesson",
        candidate_payloads=payloads,
        llm_snapshot=_make_llm_snapshot(),
    )

    assert payloads[0]["available"] is False
    assert "不能使用该饮料" in payloads[0]["unavailable_reason"]
    assert payloads[1]["available"] is True


def test_filter_unavailable_battle_drinks_removes_unresolved_candidates():
    resolved = _make_drink_candidate(index=0)
    resolved.db_id = "pdrink_02-1-006"
    resolved.action_id = "produce_drink:pdrink_02-1-006"
    resolved.metadata.update({"available": True, "display_name": "ルイボスティー", "db_id": resolved.db_id})
    unresolved = _make_drink_candidate(index=1)
    unresolved.metadata.update({"available": False, "identity_unresolved": True})

    filtered = lesson_module._filter_unavailable_battle_drinks([resolved, unresolved])

    assert [item.db_id for item in filtered] == ["pdrink_02-1-006"]


def test_collect_battle_drink_candidates_filters_unknown_title_to_neutral_name(monkeypatch):
    ctx = ProduceContext()
    ctx.hud_stamina = 30
    ctx.hud_max_stamina = 30
    ctx.parameter_state = {"remaining_turns": 5}
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    drink = _drink_box(100, 2100, 220, 2220)
    drink._ocr_text = "fncathls"
    drink.frame = drink
    results = _ResultsStub({ProducerLabels.P_DRINK: [drink]})
    app = SimpleNamespace(latest_frame=frame, latest_results=results, debug_tools=None, device=SimpleNamespace(click_element=lambda _box: None))

    monkeypatch.setattr(lesson_module, "ocr_text", _fake_ocr_text)
    monkeypatch.setattr(lesson_module, "_enrich_drink_metadata", _fake_enrich_drink_metadata)
    monkeypatch.setattr(lesson_module, "score_produce_drink_metadata", _fake_score_produce_drink_metadata)
    monkeypatch.setattr(
        lesson_module,
        "resolve_produce_drink_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            db_id="",
            action_id="produce_drink_unknown:test",
            display_name="",
            source="unresolved",
            confidence=0.0,
            metadata={"unresolved": True},
        ),
    )
    monkeypatch.setattr(lesson_module, "_apply_drink_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lesson_module, "_save_drink_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lesson_module, "_resolve_unidentified_drinks_via_modal", lambda *_args, **_kwargs: None)

    candidates = lesson_module._collect_battle_drink_candidates(app, ctx, phase="lesson", start_index=0)

    assert len(candidates) == 1
    assert candidates[0].title == "未识别P饮料#1"
    assert candidates[0].db_id == ""
    assert candidates[0].metadata["available"] is False
    assert candidates[0].metadata["raw_ocr_title"] == "fncathls"
    assert candidates[0].metadata.get("drink_score") is None
    assert lesson_module._filter_unavailable_battle_drinks(candidates) == []
