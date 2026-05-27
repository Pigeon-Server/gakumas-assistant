from types import SimpleNamespace

import cv2
import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCR_Result, OCR_ResultList
from src.core.tasks.producer_challenge.gameplay import decision as decision_module
from src.core.tasks.producer_challenge.gameplay import skill_reward as skill_reward_module
from src.core.tasks.producer_challenge.shared import common as shared_common_module


class _BoxList(list):
    def first(self):
        return self[0]


def _box(x1: int, y1: int, x2: int, y2: int, label: str):
    return SimpleNamespace(
        x=x1,
        y=y1,
        w=x2,
        h=y2,
        cx=int((x1 + x2) / 2),
        cy=int((y1 + y2) / 2),
        frame=np.zeros((max(1, y2 - y1), max(1, x2 - x1), 3), dtype=np.uint8),
        label=label,
    )


class _ResultsStub:
    def __init__(self, mapping):
        self._mapping = {
            label: _BoxList(list(items))
            for label, items in mapping.items()
        }

    def filter_by_label(self, label):
        return self._mapping.get(label, _BoxList())

    def exists_label(self, label):
        return bool(self.filter_by_label(label))


def test_extract_card_name_from_info_panel_prefers_panel_top_line(monkeypatch):
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (70, 520), (1010, 1800), (242, 242, 242), thickness=-1)
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        skill_reward_module,
        "_SKILL_REWARD_PANEL_OCR",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(72, 28, 360, 42, "ワクワクが止まらない+", 0.99),
                OCR_Result(710, 28, 64, 42, "❤-2", 0.99),
                OCR_Result(88, 118, 250, 36, "元気+2", 0.99),
            ])
        ),
    )

    name = skill_reward_module._extract_card_name_from_info_panel(
        app,
        card_boxes=[_box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)],
    )

    assert name == "ワクワクが止まらない+"


def test_extract_card_name_from_info_panel_fallback_when_panel_line_is_effect(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 700), (1020, 2100), (245, 245, 245), thickness=-1)
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        skill_reward_module,
        "_SKILL_REWARD_PANEL_OCR",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(72, 28, 260, 42, "→ 好印象+4", 0.99),
                OCR_Result(88, 118, 250, 36, "元気+2", 0.99),
            ])
        ),
    )
    name = skill_reward_module._extract_card_name_from_info_panel(
        app,
        card_boxes=[_box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)],
    )

    assert name == ""


def test_extract_card_name_from_info_panel_rejects_gibberish_lines(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 700), (1020, 2100), (245, 245, 245), thickness=-1)
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        skill_reward_module,
        "_SKILL_REWARD_PANEL_OCR",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(72, 28, 120, 42, "↑A", 0.99),
                OCR_Result(280, 28, 90, 42, "W", 0.99),
                OCR_Result(420, 28, 180, 42, "7-455M", 0.99),
            ])
        ),
    )
    monkeypatch.setattr(skill_reward_module, "ocr_text", lambda _img: "↑A")

    name = skill_reward_module._extract_card_name_from_info_panel(
        app,
        card_boxes=[_box(260, 1550, 470, 1860, ProducerLabels.SKILL_CARD_INFO)],
    )

    assert name == ""


def test_extract_card_name_from_info_panel_uses_detected_modal_region(monkeypatch):
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    card_boxes = [_box(220, 1260, 430, 1560, ProducerLabels.SKILL_CARD_INFO)]
    monkeypatch.setattr(
        skill_reward_module,
        "detect_bottom_white_modal_region",
        lambda *_args, **_kwargs: (70, 620, 1010, 1800),
    )

    def _mock_panel_ocr(img):
        return OCR_ResultList([
            OCR_Result(72, 28, 360, 42, "オトメゴコロ+", 0.99),
            OCR_Result(88, 118, 250, 36, "元気+2", 0.99),
        ])

    monkeypatch.setattr(skill_reward_module, "_SKILL_REWARD_PANEL_OCR", SimpleNamespace(ocr=_mock_panel_ocr))

    name = skill_reward_module._extract_card_name_from_info_panel(app, card_boxes)

    assert name == "オトメゴコロ+"


def test_extract_card_name_from_info_panel_hsv_fallback_when_default_detector_misses(monkeypatch):
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    cv2.rectangle(frame, (70, 660), (1010, 2100), (245, 245, 245), thickness=-1)
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub({}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    card_boxes = [
        _box(160, 1540, 360, 1820, ProducerLabels.SKILL_CARD_INFO),
        _box(420, 1540, 620, 1820, ProducerLabels.SKILL_CARD_INFO),
        _box(680, 1540, 880, 1820, ProducerLabels.SKILL_CARD_INFO),
    ]
    monkeypatch.setattr(
        skill_reward_module,
        "detect_bottom_white_modal_region",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        skill_reward_module,
        "_SKILL_REWARD_PANEL_OCR",
        SimpleNamespace(
            ocr=lambda _img: OCR_ResultList([
                OCR_Result(72, 28, 320, 42, "ワクワクが止まらない+", 0.99),
                OCR_Result(88, 118, 250, 36, "元気+2", 0.99),
            ])
        ),
    )

    name = skill_reward_module._extract_card_name_from_info_panel(app, card_boxes)

    assert name == "ワクワクが止まらない+"


def test_detect_bottom_white_modal_region_avoids_overhigh_white_contour():
    frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    # 干扰白区（过高，不应被选中）
    cv2.rectangle(frame, (80, 420), (1000, 610), (245, 245, 245), thickness=-1)
    # 真实底部白模态
    cv2.rectangle(frame, (70, 660), (1010, 2100), (245, 245, 245), thickness=-1)
    row_boxes = [
        _box(160, 1540, 360, 1820, ProducerLabels.SKILL_CARD_INFO),
        _box(420, 1540, 620, 1820, ProducerLabels.SKILL_CARD_INFO),
        _box(680, 1540, 880, 1820, ProducerLabels.SKILL_CARD_INFO),
    ]

    rect = shared_common_module.detect_bottom_white_modal_region(
        frame,
        row_boxes=row_boxes,
        debug_tools=None,
        debug_label="test_modal",
    )

    assert rect is not None
    assert rect[1] >= 620


def test_wait_skill_reward_cards_settle_waits_until_baseline_stable(monkeypatch):
    app = SimpleNamespace(
        latest_results=_ResultsStub({}),
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    sleep_calls = []
    center_sequences = [
        [1680, 1840],
        [1720, 1835],
        [1790, 1810, 1830],
        [1792, 1812, 1832],
        [1791, 1811, 1831],
    ]
    state = {"idx": 0}

    def _next_centers(_results):
        idx = min(state["idx"], len(center_sequences) - 1)
        state["idx"] += 1
        return center_sequences[idx]

    monkeypatch.setattr(skill_reward_module, "_collect_skill_reward_card_center_ys", _next_centers)
    monkeypatch.setattr(skill_reward_module, "_resolve_skill_reward_baseline_tolerance", lambda _app: (28, 9))
    monkeypatch.setattr(skill_reward_module, "sleep", lambda sec: sleep_calls.append(round(float(sec), 2)))

    skill_reward_module._wait_skill_reward_cards_settle(
        app,
        position="skill_reward_idle",
    )

    assert sleep_calls == [round(skill_reward_module._SKILL_REWARD_SETTLE_POLL_SLEEP, 2)] * 3


def test_collect_skill_reward_candidates_dedups_overlapped_labels(monkeypatch):
    overlap_active = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)
    overlap_info = _box(268, 1358, 478, 1668, ProducerLabels.SKILL_CARD_INFO)
    separate = _box(620, 1350, 830, 1660, ProducerLabels.SKILL_CARD_INFO)
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            ProducerLabels.SKILL_CARD_INFO: [overlap_active, overlap_info, separate],
        }),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = SimpleNamespace(pending_skill_reward_index=None)
    monkeypatch.setattr(skill_reward_module, "hydrate_card_candidates", lambda *_args, **_kwargs: None)

    candidates = skill_reward_module.collect_skill_reward_candidates(
        app,
        ctx,
        position="skill_reward_idle",
    )

    assert candidates == []


def test_collect_skill_reward_candidates_prefers_clickable_card_boxes_over_info_boxes(monkeypatch):
    active_left = _box(180, 1320, 380, 1640, BaseUILabels.SKILL_CARD_ACTIVE)
    active_right = _box(600, 1320, 800, 1640, BaseUILabels.SKILL_CARD_MENTAL)
    info_left = _box(160, 1500, 420, 1860, ProducerLabels.SKILL_CARD_INFO)
    info_right = _box(580, 1500, 840, 1860, ProducerLabels.SKILL_CARD_INFO)
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            BaseUILabels.SKILL_CARD_ACTIVE: [active_left],
            BaseUILabels.SKILL_CARD_MENTAL: [active_right],
            ProducerLabels.SKILL_CARD_INFO: [info_left, info_right],
        }),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = SimpleNamespace(pending_skill_reward_index=None)
    monkeypatch.setattr(skill_reward_module, "hydrate_card_candidates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "ocr_text", lambda _img: "")

    candidates = skill_reward_module.collect_skill_reward_candidates(
        app,
        ctx,
        position="skill_reward_idle",
    )

    assert len(candidates) == 2
    assert [candidate.label for candidate in candidates] == [
        BaseUILabels.SKILL_CARD_ACTIVE,
        BaseUILabels.SKILL_CARD_MENTAL,
    ]
    assert [candidate.box.cy for candidate in candidates] == [active_left.cy, active_right.cy]


def test_collect_skill_reward_candidates_ignores_info_boxes_when_no_clickable_cards(monkeypatch):
    info_left = _box(160, 1500, 420, 1860, ProducerLabels.SKILL_CARD_INFO)
    info_right = _box(580, 1500, 840, 1860, ProducerLabels.SKILL_CARD_INFO)
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            ProducerLabels.SKILL_CARD_INFO: [info_left, info_right],
        }),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = SimpleNamespace(pending_skill_reward_index=None)
    monkeypatch.setattr(skill_reward_module, "hydrate_card_candidates", lambda *_args, **_kwargs: None)

    candidates = skill_reward_module.collect_skill_reward_candidates(
        app,
        ctx,
        position="skill_reward_idle",
    )

    assert candidates == []


def test_collect_skill_reward_candidates_filters_multi_row_noise(monkeypatch):
    # 三行噪声候选（4 + 4 + 2），应只保留计数最多且更靠下的一行（中间行）。
    row_top = [
        _box(119, 1044, 301, 1225, BaseUILabels.SKILL_CARD_ACTIVE),
        _box(338, 1044, 518, 1224, BaseUILabels.SKILL_CARD_ACTIVE),
        _box(558, 1043, 739, 1224, BaseUILabels.SKILL_CARD_ACTIVE),
        _box(778, 1044, 960, 1225, BaseUILabels.SKILL_CARD_ACTIVE),
    ]
    row_mid = [
        _box(119, 1264, 302, 1444, BaseUILabels.SKILL_CARD_MENTAL),
        _box(338, 1265, 518, 1443, BaseUILabels.SKILL_CARD_MENTAL),
        _box(559, 1265, 740, 1444, BaseUILabels.SKILL_CARD_MENTAL),
        _box(777, 1264, 960, 1444, BaseUILabels.SKILL_CARD_MENTAL),
    ]
    row_bottom = [
        _box(559, 1484, 738, 1659, BaseUILabels.SKILL_CARD_MENTAL),
        _box(778, 1485, 960, 1660, BaseUILabels.SKILL_CARD_MENTAL),
    ]
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({
            BaseUILabels.SKILL_CARD_ACTIVE: row_top,
            BaseUILabels.SKILL_CARD_MENTAL: [*row_mid, *row_bottom],
        }),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
    )
    ctx = SimpleNamespace(pending_skill_reward_index=None)
    monkeypatch.setattr(skill_reward_module, "hydrate_card_candidates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "ocr_text", lambda _img: "6A")

    candidates = skill_reward_module.collect_skill_reward_candidates(
        app,
        ctx,
        position="skill_reward_idle",
    )

    assert len(candidates) == 4
    assert all(1260 <= int(c.box.y) <= 1270 for c in candidates)
    # 噪声 OCR 标题应被清洗，避免乱码直接进入 LLM 提示词。
    assert all(c.title == "" for c in candidates)


def test_probe_unresolved_cards_saves_unmatched_samples(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_INFO,
        title="",
        selected=False,
        box=card,
    )
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({ProducerLabels.SKILL_CARD_INFO: [card]}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        device=SimpleNamespace(click_element=lambda _box: None),
    )

    saved_payloads: list[tuple[int, str]] = []
    monkeypatch.setattr(skill_reward_module, "sleep", lambda _sec: None)
    monkeypatch.setattr(
        skill_reward_module,
        "_extract_card_name_from_info_panel",
        lambda *_args, **_kwargs: "↓ 好印象消費2",
    )
    monkeypatch.setattr(
        skill_reward_module,
        "_match_skill_reward_card_entry",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        skill_reward_module,
        "_save_unresolved_skill_reward_probe",
        lambda _app, cand, *, card_name: saved_payloads.append((cand.index, card_name)),
    )

    from src.utils import game_database_tools as db_tools
    monkeypatch.setattr(
        db_tools,
        "GakumasDatabase_ProduceCardDataUtils",
        SimpleNamespace,
    )

    skill_reward_module._probe_unresolved_cards(app, [candidate])

    assert candidate.title == "↓ 好印象消費2"
    assert saved_payloads == [(0, "↓ 好印象消費2")]


def test_probe_unresolved_cards_clears_noisy_thumbnail_title_when_panel_ocr_empty(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_INFO,
        title="65+1",
        selected=False,
        box=card,
        metadata={"raw_candidate_title": "65+1"},
    )
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({ProducerLabels.SKILL_CARD_INFO: [card]}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        device=SimpleNamespace(click_element=lambda _box: None),
    )

    saved_payloads: list[tuple[int, str]] = []
    monkeypatch.setattr(skill_reward_module, "sleep", lambda _sec: None)
    monkeypatch.setattr(
        skill_reward_module,
        "_extract_card_name_from_info_panel",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        skill_reward_module,
        "_save_unresolved_skill_reward_probe",
        lambda _app, cand, *, card_name: saved_payloads.append((cand.index, card_name)),
    )

    from src.utils import game_database_tools as db_tools
    monkeypatch.setattr(
        db_tools,
        "GakumasDatabase_ProduceCardDataUtils",
        SimpleNamespace,
    )

    skill_reward_module._probe_unresolved_cards(app, [candidate])

    assert candidate.title == ""
    assert saved_payloads == [(0, "")]


def test_execute_skill_reward_selected_without_pending_does_not_redo_selection(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)
    card_2 = _box(560, 1350, 770, 1660, ProducerLabels.SKILL_CARD_INFO)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_INFO,
        title="",
        selected=False,
        box=card,
    )
    candidate_2 = skill_reward_module.SkillRewardCandidate(
        index=1,
        label=ProducerLabels.SKILL_CARD_INFO,
        title="",
        selected=False,
        box=card_2,
    )
    clicks: list[int] = []
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({ProducerLabels.SKILL_CARD_INFO: [card, card_2]}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        device=SimpleNamespace(click_element=lambda box: clicks.append(int(getattr(box, "cx", 0)))),
    )
    class _Ctx(SimpleNamespace):
        def record_operation(self, *_args, **_kwargs):
            return None

        def mutate_deck_acquire(self, *_args, **_kwargs):
            return None

        def clear_skill_reward_pending(self):
            self.pending_skill_reward_index = None
            self.pending_skill_reward_label = ""
            self.handler_state.pop("pending_skill_reward_db_id", None)

    ctx = _Ctx(
        pending_skill_reward_index=None,
        pending_skill_reward_label="",
        handler_state={},
    )
    probe_calls: list[int] = []
    decide_calls = {"count": 0}
    confirm_calls = {"count": 0}
    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_cards_settle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "collect_skill_reward_candidates", lambda *_args, **_kwargs: [candidate, candidate_2])
    def _mock_probe(*_args, **_kwargs):
        probe_calls.append(1)
        candidate.title = "静かな意志"
    monkeypatch.setattr(
        skill_reward_module,
        "_probe_unresolved_cards",
        _mock_probe,
    )
    def _mock_decide(*_args, **_kwargs):
        decide_calls["count"] += 1
        return 0
    monkeypatch.setattr(skill_reward_module, "decide_skill_reward", _mock_decide)
    monkeypatch.setattr(
        skill_reward_module,
        "_click_confirm_button",
        lambda *_args, **_kwargs: confirm_calls.__setitem__("count", confirm_calls["count"] + 1) or True,
    )
    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_confirm_consumed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(skill_reward_module, "sleep", lambda _sec: None)

    result = skill_reward_module.execute_skill_reward_step(app, ctx, position="skill_reward_selected")

    assert result is not None
    assert result.status == "confirmed"
    assert len(probe_calls) == 1
    assert decide_calls["count"] == 0
    assert confirm_calls["count"] == 1
    assert clicks == []


def test_execute_skill_reward_selected_without_pending_probes_unresolved_single_card_first(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_INFO,
        title="",
        selected=False,
        box=card,
        db_id="",
    )
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({ProducerLabels.SKILL_CARD_INFO: [card]}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        device=SimpleNamespace(click_element=lambda _box: None),
    )

    class _Ctx(SimpleNamespace):
        def record_operation(self, *_args, **_kwargs):
            return None

        def mutate_deck_acquire(self, *_args, **_kwargs):
            return None

        def clear_skill_reward_pending(self):
            self.pending_skill_reward_index = None
            self.pending_skill_reward_label = ""
            self.handler_state.pop("pending_skill_reward_db_id", None)

    ctx = _Ctx(
        pending_skill_reward_index=None,
        pending_skill_reward_label="",
        handler_state={},
    )
    probe_calls = {"count": 0}
    confirm_calls = {"count": 0}

    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_cards_settle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "collect_skill_reward_candidates", lambda *_args, **_kwargs: [candidate])

    def _mock_probe(*_args, **_kwargs):
        probe_calls["count"] += 1
        candidate.title = "静かな意志"
        candidate.db_id = "p_card-01-act-1_023"

    monkeypatch.setattr(skill_reward_module, "_probe_unresolved_cards", _mock_probe)
    monkeypatch.setattr(
        skill_reward_module,
        "_click_confirm_button",
        lambda *_args, **_kwargs: confirm_calls.__setitem__("count", confirm_calls["count"] + 1) or True,
    )
    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_confirm_consumed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(skill_reward_module, "sleep", lambda _sec: None)

    result = skill_reward_module.execute_skill_reward_step(app, ctx, position="skill_reward_selected")

    assert result is not None
    assert result.status == "confirmed"
    assert probe_calls["count"] == 1
    assert confirm_calls["count"] == 1


def test_execute_skill_reward_selected_without_pending_does_not_confirm_unresolved_single_card(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_INFO,
        title="",
        selected=False,
        box=card,
        db_id="",
    )
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({ProducerLabels.SKILL_CARD_INFO: [card]}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        device=SimpleNamespace(click_element=lambda _box: None),
    )
    ctx = SimpleNamespace(
        pending_skill_reward_index=None,
        pending_skill_reward_label="",
        handler_state={},
    )
    confirm_calls = {"count": 0}

    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_cards_settle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "collect_skill_reward_candidates", lambda *_args, **_kwargs: [candidate])
    monkeypatch.setattr(skill_reward_module, "_probe_unresolved_cards", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        skill_reward_module,
        "_click_confirm_button",
        lambda *_args, **_kwargs: confirm_calls.__setitem__("count", confirm_calls["count"] + 1) or True,
    )
    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_confirm_consumed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(skill_reward_module, "sleep", lambda _sec: None)

    result = skill_reward_module.execute_skill_reward_step(app, ctx, position="skill_reward_selected")

    assert result is None
    assert confirm_calls["count"] == 0


def test_click_confirm_button_ignores_redraw_button(monkeypatch):
    receive_btn = _box(320, 1920, 640, 2050, "btn")
    redraw_btn = _box(820, 1880, 1020, 2010, "btn")
    app = SimpleNamespace(
        latest_results=_ResultsStub({
            BaseUILabels.BUTTON: [receive_btn, redraw_btn],
        }),
        device=SimpleNamespace(click_element=lambda box: setattr(app, "_clicked", box)),
    )
    app._clicked = None

    def _mock_ocr(frame):
        if frame is receive_btn.frame:
            return "受け取る"
        if frame is redraw_btn.frame:
            return "あと3回 再抽選"
        return ""

    monkeypatch.setattr(skill_reward_module, "ocr_text", _mock_ocr)

    assert skill_reward_module._click_confirm_button(app) is True
    assert app._clicked is receive_btn


def test_execute_skill_reward_idle_selects_and_confirms_receive(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_MENTAL)
    confirm = _box(320, 1920, 760, 2050, ProducerLabels.CONFIRM_BUTTON)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_MENTAL,
        title="勢い任せ",
        selected=False,
        box=card,
        db_id="p_card-01-act-1_023",
    )
    clicks = []

    class _App:
        latest_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        debug_tools = SimpleNamespace(add_box=lambda *args, **kwargs: None)

        def __init__(self):
            self._latest_results = _ResultsStub({ProducerLabels.SKILL_CARD_MENTAL: [card]})
            self.device = SimpleNamespace(click_element=self._click_element)

        @property
        def latest_results(self):
            return self._latest_results

        def _click_element(self, box):
            clicks.append(box)
            if box is card:
                self._latest_results = _ResultsStub({ProducerLabels.CONFIRM_BUTTON: [confirm]})
            elif box is confirm:
                self._latest_results = _ResultsStub({})

    class _Ctx(SimpleNamespace):
        def record_operation(self, name, target=None, details=None):
            self.operations.append((name, target, details or {}))

        def mutate_deck_acquire(self, db_id, **kwargs):
            self.acquired.append((db_id, kwargs))

        def clear_skill_reward_pending(self):
            self.pending_skill_reward_index = None
            self.pending_skill_reward_label = ""
            self.handler_state.pop("pending_skill_reward_db_id", None)

    app = _App()
    ctx = _Ctx(
        pending_skill_reward_index=None,
        pending_skill_reward_label="",
        handler_state={},
        operations=[],
        acquired=[],
    )

    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_cards_settle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "collect_skill_reward_candidates", lambda *_args, **_kwargs: [candidate])
    monkeypatch.setattr(skill_reward_module, "_append_redraw_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "decide_skill_reward", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(skill_reward_module, "sleep", lambda _sec: None)

    result = skill_reward_module.execute_skill_reward_step(app, ctx, position="skill_reward_idle")

    assert result is not None
    assert result.status == "confirmed"
    assert clicks == [card, confirm]
    assert [operation[0] for operation in ctx.operations] == ["select_skill_reward", "confirm_skill_reward"]
    assert ctx.acquired == [("p_card-01-act-1_023", {
        "kind": "produce_card",
        "name": "勢い任せ",
        "source": "skill_reward",
    })]
    assert ctx.pending_skill_reward_index is None
    assert ctx.pending_skill_reward_label == ""
    assert ctx.handler_state["unknown_retry_override"]["reason"] == "skill_reward_confirmed_transition"


def test_execute_skill_reward_retries_when_receive_click_does_not_leave_page(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_MENTAL)
    confirm = _box(320, 1920, 760, 2050, ProducerLabels.CONFIRM_BUTTON)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_MENTAL,
        title="勢い任せ",
        selected=False,
        box=card,
        db_id="p_card-01-act-1_023",
    )
    clicks = []

    class _App:
        latest_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        debug_tools = SimpleNamespace(add_box=lambda *args, **kwargs: None)

        def __init__(self):
            self._confirm_clicks = 0
            self._latest_results = _ResultsStub({ProducerLabels.SKILL_CARD_MENTAL: [card]})
            self.device = SimpleNamespace(click_element=self._click_element)

        @property
        def latest_results(self):
            return self._latest_results

        def _click_element(self, box):
            clicks.append(box)
            if box is card:
                self._latest_results = _ResultsStub({ProducerLabels.CONFIRM_BUTTON: [confirm]})
                return
            if box is confirm:
                self._confirm_clicks += 1
                if self._confirm_clicks == 1:
                    self._latest_results = _ResultsStub({ProducerLabels.CONFIRM_BUTTON: [confirm]})
                else:
                    self._latest_results = _ResultsStub({})

    class _Ctx(SimpleNamespace):
        def record_operation(self, name, target=None, details=None):
            self.operations.append((name, target, details or {}))

        def mutate_deck_acquire(self, *_args, **_kwargs):
            return None

        def clear_skill_reward_pending(self):
            self.pending_skill_reward_index = None
            self.pending_skill_reward_label = ""
            self.handler_state.pop("pending_skill_reward_db_id", None)

    app = _App()
    ctx = _Ctx(
        pending_skill_reward_index=None,
        pending_skill_reward_label="",
        handler_state={},
        operations=[],
    )

    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_cards_settle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "collect_skill_reward_candidates", lambda *_args, **_kwargs: [candidate])
    monkeypatch.setattr(skill_reward_module, "_append_redraw_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "decide_skill_reward", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(skill_reward_module, "sleep", lambda _sec: None)

    result = skill_reward_module.execute_skill_reward_step(app, ctx, position="skill_reward_idle")

    assert result is not None
    assert result.status == "confirmed"
    assert clicks == [card, confirm, confirm]
    assert [operation[0] for operation in ctx.operations] == ["select_skill_reward", "confirm_skill_reward"]


def test_execute_skill_reward_idle_does_not_return_selected_when_confirm_missing(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_MENTAL)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_MENTAL,
        title="勢い任せ",
        selected=False,
        box=card,
        db_id="p_card-01-act-1_023",
    )
    clicks = []
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({ProducerLabels.SKILL_CARD_MENTAL: [card]}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        device=SimpleNamespace(click_element=lambda box: clicks.append(box)),
    )

    class _Ctx(SimpleNamespace):
        def record_operation(self, name, target=None, details=None):
            self.operations.append((name, target, details or {}))

        def mutate_deck_acquire(self, *_args, **_kwargs):
            raise AssertionError("确认失败时不应入牌库")

        def clear_skill_reward_pending(self):
            self.pending_skill_reward_index = None
            self.pending_skill_reward_label = ""
            self.handler_state.pop("pending_skill_reward_db_id", None)

    ctx = _Ctx(
        pending_skill_reward_index=None,
        pending_skill_reward_label="",
        handler_state={},
        operations=[],
    )

    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_cards_settle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "collect_skill_reward_candidates", lambda *_args, **_kwargs: [candidate])
    monkeypatch.setattr(skill_reward_module, "_append_redraw_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "decide_skill_reward", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(skill_reward_module, "sleep", lambda _sec: None)

    result = skill_reward_module.execute_skill_reward_step(app, ctx, position="skill_reward_idle")

    assert result is None
    assert clicks == [card]
    assert [operation[0] for operation in ctx.operations] == ["select_skill_reward"]
    assert ctx.pending_skill_reward_index == 0
    assert ctx.pending_skill_reward_label == "勢い任せ"


def test_execute_skill_reward_selected_single_card_skips_llm_and_confirms(monkeypatch):
    card = _box(260, 1350, 470, 1660, ProducerLabels.SKILL_CARD_INFO)
    candidate = skill_reward_module.SkillRewardCandidate(
        index=0,
        label=ProducerLabels.SKILL_CARD_INFO,
        title="祝福",
        selected=False,
        box=card,
        db_id="p_card-blessing",
    )
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({ProducerLabels.SKILL_CARD_INFO: [card]}),
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        device=SimpleNamespace(click_element=lambda _box: None),
    )

    class _Ctx(SimpleNamespace):
        def record_operation(self, *_args, **_kwargs):
            return None

        def mutate_deck_acquire(self, *_args, **_kwargs):
            return None

        def clear_skill_reward_pending(self):
            self.pending_skill_reward_index = None
            self.pending_skill_reward_label = ""
            self.handler_state.pop("pending_skill_reward_db_id", None)

    ctx = _Ctx(
        pending_skill_reward_index=None,
        pending_skill_reward_label="",
        handler_state={},
    )

    called = {"decide": 0}
    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_cards_settle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_reward_module, "collect_skill_reward_candidates", lambda *_args, **_kwargs: [candidate])
    monkeypatch.setattr(skill_reward_module, "_click_confirm_button", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(skill_reward_module, "_wait_skill_reward_confirm_consumed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(skill_reward_module, "decide_skill_reward", lambda *_args, **_kwargs: called.__setitem__("decide", called["decide"] + 1))

    result = skill_reward_module.execute_skill_reward_step(app, ctx, position="skill_reward_selected")

    assert result is not None
    assert result.status == "confirmed"
    assert called["decide"] == 0
    assert ctx.pending_skill_reward_index is None
    assert ctx.pending_skill_reward_label == ""


def test_build_llm_actions_keeps_unresolved_skill_reward_cards():
    payload = {
        "index": 1,
        "id": "produce_card_unknown:idx_1",
        "db_id": "",
        "name": "ワクワクが止まらない+",
        "type": "skill_reward",
        "label": "unknown",
        "selected": False,
        "recommended": False,
        "available": True,
        "metadata": {"candidate_type": "produce_card", "unresolved": True},
    }

    actions = decision_module._build_llm_actions(
        [payload],
        phase="skill_reward",
        position="skill_reward_idle",
        stage_context={"p_point": 0},
    )

    assert len(actions) == 1
    assert actions[0]["index"] == 1
    assert actions[0]["label"] == "ワクワクが止まらない+"
    assert "暂未匹配主数据库" in actions[0]["description"]
