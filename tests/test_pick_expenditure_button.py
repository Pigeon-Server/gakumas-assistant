"""Tests for expenditure candidate selection via candidate_index in goto__get_expenditure."""
import sys
from types import SimpleNamespace

import pytest


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Yolo import Yolo_Box, Yolo_Results


def _box(x, y, w, h, label):
    """Create a Yolo_Box with edge-based w/h (right, bottom)."""
    return Yolo_Box(x, y, w, h, label, None)


def _results(boxes):
    return Yolo_Results.from_boxes(boxes)


# ---------- coordinates from real dump ----------
# FALSE POSITIVE (親愛度): x=52  y=1044 w=156  h=1187 cx=104 cy=1115
# CORRECT (活動費):        x=49  y=1218 w=158  h=1359 cx=103 cy=1288
# Daily Task anchor:       x=907 y=1210 w=1044 h=1357 cx=975 cy=1283

REAL_FP = _box(52, 1044, 156, 1187, BaseUILabels.HOME_GET_EXPENDITURE)
REAL_CORRECT = _box(49, 1218, 158, 1359, BaseUILabels.HOME_GET_EXPENDITURE)
REAL_DAILY_TASK = _box(907, 1210, 1044, 1357, BaseUILabels.HOME_DAILY_TASK)
REAL_GIFT = _box(906, 1040, 1042, 1182, BaseUILabels.HOME_GIFT_BTN)


class TestCandidateIndexSelection:
    """Verify that filter_by_label + candidate_index selects the right button."""

    def test_two_candidates_index0_is_topmost(self):
        """Index 0 returns the topmost candidate (may be wrong one)."""
        all_boxes = _results([REAL_FP, REAL_CORRECT, REAL_DAILY_TASK, REAL_GIFT])
        candidates = all_boxes.filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        assert len(candidates) == 2
        # sort_boxes puts topmost first
        assert candidates.boxes[0] == REAL_FP

    def test_two_candidates_index1_is_correct(self):
        """Index 1 returns the lower (correct) candidate."""
        all_boxes = _results([REAL_FP, REAL_CORRECT, REAL_DAILY_TASK, REAL_GIFT])
        candidates = all_boxes.filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        assert candidates.boxes[1] == REAL_CORRECT

    def test_single_candidate_index0(self):
        """Only one candidate — index 0 returns it."""
        all_boxes = _results([REAL_CORRECT, REAL_DAILY_TASK])
        candidates = all_boxes.filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        assert len(candidates) == 1
        assert candidates.boxes[0] == REAL_CORRECT

    def test_index_clamped_when_exceeds_count(self):
        """candidate_index >= len(candidates) is clamped to last element."""
        candidates = _results([REAL_FP, REAL_CORRECT])
        idx = min(5, len(candidates) - 1)
        assert candidates.boxes[idx] == REAL_CORRECT

    def test_three_candidates_iterate_all(self):
        """Three FP candidates — can iterate through all of them."""
        c1 = _box(50, 800, 160, 940, BaseUILabels.HOME_GET_EXPENDITURE)
        c2 = _box(50, 1000, 160, 1140, BaseUILabels.HOME_GET_EXPENDITURE)
        c3 = _box(50, 1200, 160, 1340, BaseUILabels.HOME_GET_EXPENDITURE)
        candidates = _results([c1, c2, c3]).filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        assert candidates.boxes[0] == c1
        assert candidates.boxes[1] == c2
        assert candidates.boxes[2] == c3


class TestRetryFlowSimulation:
    """Simulate the full retry flow: click → wrong modal → advance → click next."""

    def test_retry_finds_correct_on_second_attempt(self):
        """First candidate opens wrong modal; second opens correct modal."""
        all_boxes = _results([REAL_FP, REAL_CORRECT, REAL_DAILY_TASK])
        candidates = all_boxes.filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)

        # Attempt 0: click index 0 → wrong
        clicked_0 = candidates.boxes[min(0, len(candidates) - 1)]
        assert clicked_0 == REAL_FP

        # Attempt 1: advance to index 1 → correct
        clicked_1 = candidates.boxes[min(1, len(candidates) - 1)]
        assert clicked_1 == REAL_CORRECT

    def test_single_candidate_retry_clicks_same(self):
        """With one candidate, retrying with index=1 still clicks it (clamped)."""
        candidates = _results([REAL_CORRECT]).filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        assert candidates.boxes[min(0, len(candidates) - 1)] == REAL_CORRECT
        assert candidates.boxes[min(1, len(candidates) - 1)] == REAL_CORRECT

    def test_empty_candidates_raises(self):
        """No candidates → empty list, would raise in goto function."""
        candidates = _results([REAL_DAILY_TASK]).filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        assert len(candidates) == 0


class TestJpegNoiseRobustness:
    """Coordinate jitter from JPEG compression should not change candidate ordering."""

    @pytest.mark.parametrize("jitter", [-8, -4, -2, 0, 2, 4, 8])
    def test_ordering_stable_under_jitter(self, jitter):
        """Candidate ordering remains top-to-bottom despite jitter."""
        fp = _box(52 + jitter, 1044 + jitter, 156 + jitter, 1187 + jitter,
                  BaseUILabels.HOME_GET_EXPENDITURE)
        correct = _box(49 - jitter, 1218 - jitter, 158 - jitter, 1359 - jitter,
                       BaseUILabels.HOME_GET_EXPENDITURE)
        candidates = _results([fp, correct]).filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        # fp is always higher (cy ~1115) than correct (cy ~1288), so index 0 = fp
        assert candidates.boxes[0].cy < candidates.boxes[1].cy
        # index 1 is always the correct (lower) one
        assert candidates.boxes[1] == correct

    @pytest.mark.parametrize("jitter", [-8, -4, -2, 0, 2, 4, 8])
    def test_retry_still_reaches_correct_under_jitter(self, jitter):
        """Even with jitter, advancing candidate_index reaches the correct button."""
        fp = _box(52, 1044 + jitter, 156, 1187 + jitter,
                  BaseUILabels.HOME_GET_EXPENDITURE)
        correct = _box(49, 1218 + jitter, 158, 1359 + jitter,
                       BaseUILabels.HOME_GET_EXPENDITURE)
        candidates = _results([fp, correct]).filter_by_label(BaseUILabels.HOME_GET_EXPENDITURE)
        assert candidates.boxes[min(1, len(candidates) - 1)] == correct
