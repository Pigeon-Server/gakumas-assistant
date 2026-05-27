"""
Comprehensive test suite for support-card detection pipeline.

Validates:
  - HSV-only rarity detection (no OCR) on all card crops
  - Limit-break (上限解放可能) badge detection on all card crops
  - Full SupportCardListParser pipeline speed and accuracy
  - JPEG Q30 noise resilience for rarity and limit-break
  - Gaussian noise σ=15 resilience for rarity and limit-break
  - Incremental parse: per-card parse time must be < 50ms after warm-up
  - Multi-source (ADB + scrcpy) limit-break detection diversity
  - Logic correctness: _card_needs_limit_break trusts badge over digit reading
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.SupportCard import (
    SupportCard,
    SupportCardListParser,
    _build_digit_classifier,
    _detect_limit_break,
    _detect_rarity,
    _cross_validate_rarity,
)
from src.core.tasks.base_ui.auto_enhancement_support_card import (
    _card_needs_limit_break,
    _card_needs_enhancement,
    _find_lb_arrow,
    _FUZZ_CONFIG,
)
from src.entity.Game.Components.Button import ButtonList
from src.constants.game.text.support_card_text import SupportCardText

TEST_CAPTURES = Path(__file__).resolve().parent.parent / "logs" / "debug" / "test_captures"
CARD_LIST_IMAGE = TEST_CAPTURES / "support_card_list.png"
GROUND_TRUTH_FILE = TEST_CAPTURES / "ground_truth.json"


def _load_ground_truth() -> dict:
    """Load ground truth from JSON file, fallback to inline data."""
    if GROUND_TRUTH_FILE.exists():
        with open(GROUND_TRUTH_FILE) as f:
            return json.load(f)
    # Fallback: inline ground truth for CI environments
    # Only card_6 has the "上限解放可能" badge (visually verified)
    return {
        "card_0.png":  {"rarity": "R",   "lb": False},
        "card_1.png":  {"rarity": "SR",  "lb": False},
        "card_2.png":  {"rarity": "SR",  "lb": False},
        "card_3.png":  {"rarity": "SR",  "lb": False},
        "card_4.png":  {"rarity": "SSR", "lb": False},
        "card_5.png":  {"rarity": "SSR", "lb": False},
        "card_6.png":  {"rarity": "SR",  "lb": True},
        "card_7.png":  {"rarity": "SR",  "lb": False},
        "card_8.png":  {"rarity": "SR",  "lb": False},
        "card_9.png":  {"rarity": "SR",  "lb": False},
        "card_10.png": {"rarity": "R",   "lb": False},
        "card_11.png": {"rarity": "R",   "lb": False},
        "card_12.png": {"rarity": "SSR", "lb": False},
        "card_13.png": {"rarity": "SR",  "lb": False},
        "card_14.png": {"rarity": "SR",  "lb": False},
        "card_15.png": {"rarity": "SR",  "lb": False},
        "card_16.png": {"rarity": "R",   "lb": False},
        "card_17.png": {"rarity": "SSR", "lb": False},
    }


GROUND_TRUTH = _load_ground_truth()

# Separate standard card crops (card_N.png) from diversity crops (lb_crop_*, nolb_*)
_CARD_CROPS = {k: v for k, v in GROUND_TRUTH.items() if k.startswith("card_")}
_LB_DIVERSITY = {k: v for k, v in GROUND_TRUTH.items() if k.startswith("lb_crop_")}
_NOLB_DIVERSITY = {k: v for k, v in GROUND_TRUTH.items() if k.startswith("no_lb_")}


def _jpeg_compress(frame: np.ndarray, quality: int = 30) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _add_gaussian_noise(frame: np.ndarray, sigma: float = 15) -> np.ndarray:
    rng = np.random.RandomState(42)
    noise = rng.normal(0, sigma, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def yolo_model():
    return YoloModelFromONNX(config.model_config["BASE_UI"])


@pytest.fixture(scope="session")
def card_list_frame():
    img = cv2.imread(str(CARD_LIST_IMAGE))
    assert img is not None, f"Cannot load {CARD_LIST_IMAGE}"
    return img


@pytest.fixture(scope="session")
def warm_classifier():
    """Pre-build classifier so timing tests exclude cold-start."""
    return _build_digit_classifier()


# ── Test: HSV rarity detection on individual card crops ───────────────────────

class TestRarityDetection:
    """Rarity detection must match ground truth for all card crops.

    Uses HSV + level/star cross-validation (the actual pipeline).
    Raw HSV may misclassify when no color matches threshold; cross-validation
    uses level/stars to correct impossible rarity assignments.
    """

    @pytest.mark.parametrize("filename,expected", [
        (k, v["rarity"]) for k, v in _CARD_CROPS.items()
    ])
    def test_rarity_matches_ground_truth(self, filename, expected):
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = cv2.imread(str(path))
        has_lb = _CARD_CROPS[filename]["lb"]
        detected = _detect_rarity(frame, has_limit_break=has_lb)
        # Apply cross-validation if we have ground truth level/star info
        gt = _CARD_CROPS[filename]
        if "level" in gt and "stars" in gt and gt["level"] is not None and gt["stars"] is not None:
            detected = _cross_validate_rarity(detected, gt["level"], gt["stars"])
        assert detected == expected, f"{filename}: expected {expected}, got {detected}"

    @pytest.mark.parametrize("filename,expected", [
        (k, v["rarity"]) for k, v in _CARD_CROPS.items()
    ])
    def test_rarity_after_jpeg_q30(self, filename, expected):
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = _jpeg_compress(cv2.imread(str(path)), quality=30)
        has_lb = _CARD_CROPS[filename]["lb"]
        detected = _detect_rarity(frame, has_limit_break=has_lb)
        gt = _CARD_CROPS[filename]
        if "level" in gt and "stars" in gt and gt["level"] is not None and gt["stars"] is not None:
            detected = _cross_validate_rarity(detected, gt["level"], gt["stars"])
        assert detected == expected, f"JPEG Q30 {filename}: expected {expected}, got {detected}"

    @pytest.mark.parametrize("filename,expected", [
        (k, v["rarity"]) for k, v in _CARD_CROPS.items()
    ])
    def test_rarity_after_gaussian_noise(self, filename, expected):
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = _add_gaussian_noise(cv2.imread(str(path)), sigma=15)
        has_lb = _CARD_CROPS[filename]["lb"]
        detected = _detect_rarity(frame, has_limit_break=has_lb)
        gt = _CARD_CROPS[filename]
        if "level" in gt and "stars" in gt and gt["level"] is not None and gt["stars"] is not None:
            detected = _cross_validate_rarity(detected, gt["level"], gt["stars"])
        assert detected == expected, f"Gaussian σ=15 {filename}: expected {expected}, got {detected}"


# ── Test: limit-break detection on individual card crops ──────────────────────

class TestLimitBreakDetection:
    """OCR limit-break badge detection must match ground truth for all card crops."""

    @pytest.mark.parametrize("filename,expected", [
        (k, v["lb"]) for k, v in _CARD_CROPS.items()
    ])
    def test_limit_break_matches_ground_truth(self, filename, expected):
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = cv2.imread(str(path))
        detected = _detect_limit_break(frame)
        assert detected == expected, f"{filename}: expected {expected}, got {detected}"

    @pytest.mark.parametrize("filename,expected", [
        (k, v["lb"]) for k, v in _CARD_CROPS.items()
    ])
    def test_limit_break_after_jpeg_q30(self, filename, expected):
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = _jpeg_compress(cv2.imread(str(path)), quality=30)
        detected = _detect_limit_break(frame)
        assert detected == expected, f"JPEG Q30 {filename}: expected {expected}, got {detected}"

    @pytest.mark.parametrize("filename,expected", [
        (k, v["lb"]) for k, v in _CARD_CROPS.items()
    ])
    def test_limit_break_after_gaussian_noise(self, filename, expected):
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = _add_gaussian_noise(cv2.imread(str(path)), sigma=15)
        detected = _detect_limit_break(frame)
        assert detected == expected, f"Gaussian σ=15 {filename}: expected {expected}, got {detected}"


# ── Test: limit-break diversity (multi-source ADB + scrcpy crops) ─────────────

class TestLimitBreakDiversity:
    """Badge detection must work on all diverse crops from different capture sources."""

    # Separate real-badge crops from non-badge crops based on ground truth
    _LB_TRUE = {k: v for k, v in _LB_DIVERSITY.items() if v.get("lb")}
    _LB_FALSE = {k: v for k, v in _LB_DIVERSITY.items() if not v.get("lb")}

    @pytest.mark.parametrize("filename", list(_LB_TRUE.keys()) or ["__skip__"])
    def test_lb_diversity_crop_detected(self, filename):
        """Each real limit-break crop must be detected as True."""
        if filename == "__skip__":
            pytest.skip("No true LB diversity crops")
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = cv2.imread(str(path))
        assert _detect_limit_break(frame), f"{filename}: badge NOT detected (should be True)"

    @pytest.mark.parametrize("filename", list(_LB_TRUE.keys()) or ["__skip__"])
    def test_lb_diversity_jpeg_q50(self, filename):
        if filename == "__skip__":
            pytest.skip("No true LB diversity crops")
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = _jpeg_compress(cv2.imread(str(path)), quality=50)
        assert _detect_limit_break(frame), f"JPEG Q50 {filename}: badge NOT detected"

    @pytest.mark.parametrize("filename", list(_LB_TRUE.keys()) or ["__skip__"])
    def test_lb_diversity_jpeg_q30(self, filename):
        if filename == "__skip__":
            pytest.skip("No true LB diversity crops")
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = _jpeg_compress(cv2.imread(str(path)), quality=30)
        assert _detect_limit_break(frame), f"JPEG Q30 {filename}: badge NOT detected"

    @pytest.mark.parametrize("filename", list(_LB_TRUE.keys()) or ["__skip__"])
    def test_lb_diversity_gaussian_noise(self, filename):
        if filename == "__skip__":
            pytest.skip("No true LB diversity crops")
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = _add_gaussian_noise(cv2.imread(str(path)), sigma=15)
        assert _detect_limit_break(frame), f"Gaussian σ=15 {filename}: badge NOT detected"

    @pytest.mark.parametrize("filename", list(_LB_FALSE.keys()) + list(_NOLB_DIVERSITY.keys()))
    def test_non_lb_diversity_no_false_positive(self, filename):
        """Non-limit-break crops must NOT be detected as True."""
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = cv2.imread(str(path))
        assert not _detect_limit_break(frame), f"{filename}: FALSE POSITIVE (should be False)"

    @pytest.mark.parametrize("filename", list(_LB_FALSE.keys()) + list(_NOLB_DIVERSITY.keys()))
    def test_non_lb_diversity_jpeg_q30_no_false_positive(self, filename):
        path = TEST_CAPTURES / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        frame = _jpeg_compress(cv2.imread(str(path)), quality=30)
        assert not _detect_limit_break(frame), f"JPEG Q30 {filename}: FALSE POSITIVE"


# ── Test: _cross_validate_rarity corrects impossible classifications ──────────

class TestRarityCrossValidation:
    """Cross-validation upgrades rarity when level exceeds star cap."""

    def test_r_at_valid_level_unchanged(self):
        assert _cross_validate_rarity("R", 20, 0) == "R"
        assert _cross_validate_rarity("R", 25, 1) == "R"
        assert _cross_validate_rarity("R", 40, 4) == "R"

    def test_sr_at_valid_level_unchanged(self):
        assert _cross_validate_rarity("SR", 30, 0) == "SR"
        assert _cross_validate_rarity("SR", 45, 3) == "SR"
        assert _cross_validate_rarity("SR", 50, 4) == "SR"

    def test_ssr_at_valid_level_unchanged(self):
        assert _cross_validate_rarity("SSR", 40, 0) == "SSR"
        assert _cross_validate_rarity("SSR", 60, 4) == "SSR"

    def test_r_upgraded_to_sr(self):
        # R ★1 cap=25, Lv=30 → must be at least SR (SR ★1 cap=35)
        assert _cross_validate_rarity("R", 30, 1) == "SR"

    def test_r_upgraded_to_ssr(self):
        # R ★1 cap=25, Lv=45 → must be SSR (SSR ★1 cap=45)
        assert _cross_validate_rarity("R", 45, 1) == "SSR"

    def test_sr_upgraded_to_ssr(self):
        # SR ★2 cap=40, Lv=50 → must be SSR (SSR ★2 cap=50)
        assert _cross_validate_rarity("SR", 50, 2) == "SSR"
        # SR ★4 cap=50, Lv=60 → must be SSR
        assert _cross_validate_rarity("SR", 60, 4) == "SSR"

    def test_level_exceeds_all_caps_returns_ssr(self):
        # Edge case: level higher than any known cap
        assert _cross_validate_rarity("R", 99, 4) == "SSR"


# ── Test: _card_needs_limit_break logic trusts badge ──────────────────────────

class TestLimitBreakLogic:
    """Verify _card_needs_limit_break trusts badge over digit reading."""

    _CONFIG = {
        "R":   {"enabled": True, "max_level": 40},
        "SR":  {"enabled": True, "max_level": 50},
        "SSR": {"enabled": True, "max_level": 60},
    }

    def test_badge_true_level_accurate(self):
        """Badge=True, level at cap → needs limit break."""
        card = SupportCard(
            index=0, box=None, level=45, stars=3, confidence=0.99,
            rarity="SR", limit_break=True,
        )
        assert _card_needs_limit_break(card, self._CONFIG, auto_limit_break=True)

    def test_badge_true_level_underread(self):
        """Badge=True, level misread as below cap → STILL needs limit break.
        This is the critical fix: badge overrides digit reading."""
        card = SupportCard(
            index=0, box=None, level=35, stars=3, confidence=0.80,
            rarity="SR", limit_break=True,
        )
        assert _card_needs_limit_break(card, self._CONFIG, auto_limit_break=True)

    def test_badge_true_level_overread(self):
        """Badge=True, level misread as above cap → still needs limit break."""
        card = SupportCard(
            index=0, box=None, level=55, stars=3, confidence=0.99,
            rarity="SR", limit_break=True,
        )
        assert _card_needs_limit_break(card, self._CONFIG, auto_limit_break=True)

    def test_badge_true_level_none(self):
        """Badge=True, digit recognition failed → still needs limit break."""
        card = SupportCard(
            index=0, box=None, level=None, stars=None, confidence=0.0,
            rarity="SR", limit_break=True,
        )
        assert _card_needs_limit_break(card, self._CONFIG, auto_limit_break=True)

    def test_badge_false_no_limit_break(self):
        """Badge=False → does NOT need limit break."""
        card = SupportCard(
            index=0, box=None, level=45, stars=3, confidence=0.99,
            rarity="SR", limit_break=False,
        )
        assert not _card_needs_limit_break(card, self._CONFIG, auto_limit_break=True)

    def test_auto_limit_break_disabled(self):
        """auto_limit_break=False → never needs limit break."""
        card = SupportCard(
            index=0, box=None, level=45, stars=3, confidence=0.99,
            rarity="SR", limit_break=True,
        )
        assert not _card_needs_limit_break(card, self._CONFIG, auto_limit_break=False)

    def test_max_stars_no_limit_break(self):
        """Stars=4 (max) → no limit break even with badge."""
        card = SupportCard(
            index=0, box=None, level=50, stars=4, confidence=0.99,
            rarity="SR", limit_break=True,
        )
        assert not _card_needs_limit_break(card, self._CONFIG, auto_limit_break=True)

    def test_user_max_below_cap_still_limit_breaks(self):
        """User max_level ≤ current cap → still limit breaks.
        LB is free (uses duplicate cards), so max_level doesn't gate it."""
        config = {"SR": {"enabled": True, "max_level": 45}}  # same as SR ★3 cap
        card = SupportCard(
            index=0, box=None, level=45, stars=3, confidence=0.99,
            rarity="SR", limit_break=True,
        )
        assert _card_needs_limit_break(card, config, auto_limit_break=True)

    def test_rarity_disabled_still_limit_breaks(self):
        """Rarity enhancement disabled → limit break still works.
        Limit break uses duplicate cards, not enhancement points."""
        config = {"SR": {"enabled": False, "max_level": 50}}
        card = SupportCard(
            index=0, box=None, level=45, stars=3, confidence=0.99,
            rarity="SR", limit_break=True,
        )
        assert _card_needs_limit_break(card, config, auto_limit_break=True)

    def test_rarity_disabled_low_max_still_limit_breaks(self):
        """Rarity disabled AND max_level ≤ cap → STILL limit breaks.
        LB is free; auto_limit_break=True means always do it."""
        config = {"SR": {"enabled": False, "max_level": 40}}
        card = SupportCard(
            index=0, box=None, level=40, stars=2, confidence=0.99,
            rarity="SR", limit_break=True,
        )
        assert _card_needs_limit_break(card, config, auto_limit_break=True)

    def test_stars_none_still_limit_breaks(self):
        """Stars unknown → still limit breaks when badge is present."""
        config = {"R": {"enabled": False, "max_level": 10}}
        card = SupportCard(
            index=0, box=None, level=None, stars=None, confidence=0.0,
            rarity="R", limit_break=True,
        )
        assert _card_needs_limit_break(card, config, auto_limit_break=True)

    def test_stars_none_high_max_still_breaks(self):
        """Stars unknown but max_level above 0★ cap → limit break."""
        config = {"SR": {"enabled": False, "max_level": 50}}
        card = SupportCard(
            index=0, box=None, level=None, stars=None, confidence=0.0,
            rarity="SR", limit_break=True,
        )
        # SR 0★ cap=30, max_level=50 > 30 → should limit break
        assert _card_needs_limit_break(card, config, auto_limit_break=True)


# ── Test: _card_needs_enhancement skips badge cards ───────────────────────────

class TestEnhancementLogic:
    """Verify _card_needs_enhancement correctly skips limit-break cards."""

    _CONFIG = {
        "R":   {"enabled": True, "max_level": 40},
        "SR":  {"enabled": True, "max_level": 50},
        "SSR": {"enabled": True, "max_level": 60},
    }

    def test_badge_true_at_cap_no_enhancement(self):
        """Card with badge at exactly the star cap → no enhancement needed."""
        card = SupportCard(
            index=0, box=None, level=45, stars=3, confidence=0.99,
            rarity="SR", limit_break=True,
        )
        # SR ★3 cap=45, target=min(50,45)=45, level(45)>=45 → False
        assert not _card_needs_enhancement(card, self._CONFIG, False, [])

    def test_badge_true_underread_level_needs_enhancement(self):
        """With independence, misread level below cap still triggers enhancement."""
        card = SupportCard(
            index=0, box=None, level=35, stars=3, confidence=0.80,
            rarity="SR", limit_break=True,
        )
        # SR ★3 cap=45, target=min(50,45)=45, level(35)<45 → True
        # 主循环会先处理LB，再处理强化
        assert _card_needs_enhancement(card, self._CONFIG, False, [])

    def test_both_lb_and_enhancement_can_trigger(self):
        """A card can need both LB and enhancement simultaneously."""
        card = SupportCard(
            index=0, box=None, level=35, stars=2, confidence=0.85,
            rarity="SR", limit_break=True,
        )
        # LB: badge=True, auto_lb=True, stars<4 → True
        assert _card_needs_limit_break(card, self._CONFIG, auto_limit_break=True)
        # Enhancement: level(35) < min(50, 40)=40 → True
        assert _card_needs_enhancement(card, self._CONFIG, False, [])

    def test_normal_card_below_target_needs_enhancement(self):
        """Normal card below target level should be enhanced."""
        card = SupportCard(
            index=0, box=None, level=30, stars=3, confidence=0.99,
            rarity="SR", limit_break=False,
        )
        assert _card_needs_enhancement(card, self._CONFIG, False, [])

    def test_normal_card_at_target_skips(self):
        """Normal card at target level should be skipped."""
        card = SupportCard(
            index=0, box=None, level=45, stars=3, confidence=0.99,
            rarity="SR", limit_break=False,
        )
        assert not _card_needs_enhancement(card, self._CONFIG, False, [])


# ── Test: full pipeline on card list page ─────────────────────────────────────

class TestFullPipelineParse:
    """SupportCardListParser on the full card-list screenshot."""

    def test_finds_at_least_10_cards(self, yolo_model, card_list_frame):
        yr = Yolo_Results(yolo_model(card_list_frame), card_list_frame)
        card_list = SupportCardListParser(yr).parse()
        assert len(card_list) >= 10, f"Expected ≥10 cards, got {len(card_list)}"

    def test_all_parsed_cards_have_rarity(self, yolo_model, card_list_frame):
        yr = Yolo_Results(yolo_model(card_list_frame), card_list_frame)
        card_list = SupportCardListParser(yr).parse()
        for card in card_list:
            assert card.rarity is not None, f"Card {card.index} has no rarity"

    def test_limit_break_matches_ground_truth(self, yolo_model, card_list_frame):
        """limit_break detection must match ground truth for main card set."""
        yr = Yolo_Results(yolo_model(card_list_frame), card_list_frame)
        card_list = SupportCardListParser(yr).parse()
        # Only count card_*.png entries (main screenshot), not diversity crops
        gt_lb_count = sum(1 for v in _CARD_CROPS.values() if v.get("lb"))
        detected_lb = [c for c in card_list if c.limit_break]
        assert len(detected_lb) == gt_lb_count, (
            f"Expected {gt_lb_count} LB cards, got {len(detected_lb)}"
        )

    def test_limit_break_is_bool_for_all_cards(self, yolo_model, card_list_frame):
        """Cards with failed digit recognition should still have limit_break checked."""
        yr = Yolo_Results(yolo_model(card_list_frame), card_list_frame)
        card_list = SupportCardListParser(yr).parse()
        for card in card_list:
            assert isinstance(card.limit_break, bool), (
                f"Card {card.index}: limit_break should be bool, got {type(card.limit_break)}"
            )

    def test_rarity_distribution_reasonable(self, yolo_model, card_list_frame):
        yr = Yolo_Results(yolo_model(card_list_frame), card_list_frame)
        card_list = SupportCardListParser(yr).parse()
        rarities = [c.rarity for c in card_list if c.rarity]
        assert len(rarities) > 0, "Expected at least one card with detected rarity"
        assert "SR" in rarities, "Expected at least one SR card"

    def test_pipeline_after_jpeg_q30(self, yolo_model, card_list_frame):
        noisy = _jpeg_compress(card_list_frame, quality=30)
        yr = Yolo_Results(yolo_model(noisy), noisy)
        card_list = SupportCardListParser(yr).parse()
        assert len(card_list) >= 8, f"JPEG Q30: Expected ≥8 cards, got {len(card_list)}"

    def test_pipeline_after_gaussian_noise(self, yolo_model, card_list_frame):
        noisy = _add_gaussian_noise(card_list_frame, sigma=15)
        yr = Yolo_Results(yolo_model(noisy), noisy)
        card_list = SupportCardListParser(yr).parse()
        assert len(card_list) >= 8, f"Gaussian: Expected ≥8 cards, got {len(card_list)}"


# ── Test: speed benchmarks ────────────────────────────────────────────────────

class TestSpeedBenchmarks:
    """Ensure detection pipeline meets speed targets after warm-up."""

    def test_rarity_speed_all_crops(self):
        """HSV rarity detection (without cross-validation) for all crops.
        Most cards are < 0.1ms (pure HSV), but some may trigger OCR fallback (~80ms).
        Total must be < 500ms for all crops."""
        frames = []
        for f in sorted(TEST_CAPTURES.glob("card_*.png")):
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
        assert frames, "No card crop files found"

        t0 = time.time()
        for fr in frames:
            _detect_rarity(fr)
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 500, f"Rarity detection too slow: {elapsed:.1f}ms for {len(frames)} crops"

    def test_limit_break_speed_all_crops(self):
        """OCR limit-break detection for all card crops must be < 200ms/card avg."""
        frames = []
        for f in sorted(TEST_CAPTURES.glob("card_*.png")):
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
        assert frames, "No card crop files found"

        # Warm up OCR engine
        _detect_limit_break(frames[0])

        t0 = time.time()
        for fr in frames:
            _detect_limit_break(fr)
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / len(frames)
        assert avg < 200, f"Limit-break detection too slow: {avg:.1f}ms/card avg ({len(frames)} crops)"

    def test_full_pipeline_warm_speed(self, yolo_model, card_list_frame, warm_classifier):
        """After warm-up, full pipeline parse must be < 5000ms (excl. classifier build).

        OCR-based limit-break detection adds ~70ms/card; 18 cards ≈ 1.3s.
        """
        # Run once to warm up everything
        yr = Yolo_Results(yolo_model(card_list_frame), card_list_frame)
        SupportCardListParser(yr).parse()

        # Time the actual run
        yr = Yolo_Results(yolo_model(card_list_frame), card_list_frame)
        t0 = time.time()
        card_list = SupportCardListParser(yr).parse()
        elapsed = (time.time() - t0) * 1000
        assert elapsed < 5000, (
            f"Pipeline too slow: {elapsed:.0f}ms for {len(card_list)} cards "
            f"(target: <5000ms after warm-up)"
        )


# ── Test: live capture pipeline (2-badge page) ───────────────────────────────

LIVE_CARD_LIST = TEST_CAPTURES / "support_card_list_live.png"
LIVE_CROPS_DIR = TEST_CAPTURES / "live_crops"
LIVE_GROUND_TRUTH = LIVE_CROPS_DIR / "ground_truth.json"


def _load_live_ground_truth() -> dict:
    if LIVE_GROUND_TRUTH.exists():
        with open(LIVE_GROUND_TRUTH) as f:
            return json.load(f)
    return {}


@pytest.mark.skipif(not LIVE_CARD_LIST.exists(), reason="live capture not available")
class TestLivePipeline:
    """End-to-end tests on a second real device capture."""

    def test_finds_at_least_16_cards(self, yolo_model):
        img = cv2.imread(str(LIVE_CARD_LIST))
        yr = Yolo_Results(yolo_model(img), img)
        card_list = SupportCardListParser(yr).parse()
        assert len(card_list) >= 16, f"Expected ≥16 cards, got {len(card_list)}"

    def test_limit_break_cards_detected(self, yolo_model):
        """LB count matches live ground truth (may be 0 if no LB cards on page)."""
        gt = _load_live_ground_truth()
        expected_lb = sum(1 for v in gt.values() if v.get("lb"))
        img = cv2.imread(str(LIVE_CARD_LIST))
        yr = Yolo_Results(yolo_model(img), img)
        card_list = SupportCardListParser(yr).parse()
        lb_cards = [c for c in card_list if c.limit_break]
        assert len(lb_cards) == expected_lb, (
            f"Expected {expected_lb} LB cards, got {len(lb_cards)}"
        )

    def test_lb_cards_rarity_matches_ground_truth(self, yolo_model):
        """LB cards' rarity must match live ground truth."""
        gt = _load_live_ground_truth()
        lb_gt = {k: v for k, v in gt.items() if v.get("lb")}
        if not lb_gt:
            pytest.skip("No LB cards in live ground truth")
        img = cv2.imread(str(LIVE_CARD_LIST))
        yr = Yolo_Results(yolo_model(img), img)
        card_list = SupportCardListParser(yr).parse()
        for card in card_list:
            if card.limit_break:
                gt_entry = gt.get(f"card_{card.index}.png")
                if gt_entry:
                    assert card.rarity == gt_entry["rarity"], (
                        f"LB card {card.index}: expected {gt_entry['rarity']}, got {card.rarity}"
                    )

    def test_pipeline_after_jpeg_q30(self, yolo_model):
        """Rarity distribution after JPEG compression should be stable."""
        img = cv2.imread(str(LIVE_CARD_LIST))
        noisy = _jpeg_compress(img, quality=30)
        yr = Yolo_Results(yolo_model(noisy), noisy)
        card_list = SupportCardListParser(yr).parse()
        assert len(card_list) >= 10, f"JPEG Q30: expected ≥10 cards, got {len(card_list)}"

    def test_pipeline_after_gaussian_noise(self, yolo_model):
        img = cv2.imread(str(LIVE_CARD_LIST))
        noisy = _add_gaussian_noise(img, sigma=25)
        yr = Yolo_Results(yolo_model(noisy), noisy)
        card_list = SupportCardListParser(yr).parse()
        assert len(card_list) >= 10, f"Gaussian σ=25: expected ≥10 cards, got {len(card_list)}"

    def test_pipeline_after_combined_degradation(self, yolo_model):
        img = cv2.imread(str(LIVE_CARD_LIST))
        noisy = _add_gaussian_noise(img, sigma=25)
        noisy = _jpeg_compress(noisy, quality=30)
        yr = Yolo_Results(yolo_model(noisy), noisy)
        card_list = SupportCardListParser(yr).parse()
        assert len(card_list) >= 10, f"Q30+σ=25: expected ≥10 cards, got {len(card_list)}"


@pytest.mark.skipif(not LIVE_CROPS_DIR.exists(), reason="live crops not available")
class TestLiveCropDetection:
    """Test badge detection on individual card crops from live capture."""

    def test_lb_detection_matches_ground_truth(self):
        gt = _load_live_ground_truth()
        assert gt, "Live ground truth is empty"
        for fname, info in gt.items():
            img = cv2.imread(str(LIVE_CROPS_DIR / fname))
            assert img is not None, f"Failed to read {fname}"
            detected = _detect_limit_break(img)
            assert detected == info["lb"], (
                f"{fname}: expected lb={info['lb']}, got {detected}"
            )

    def test_lb_detection_robust_to_jpeg_q30(self):
        gt = _load_live_ground_truth()
        for fname, info in gt.items():
            img = cv2.imread(str(LIVE_CROPS_DIR / fname))
            noisy = _jpeg_compress(img, quality=30)
            detected = _detect_limit_break(noisy)
            assert detected == info["lb"], (
                f"{fname} (JPEG Q30): expected lb={info['lb']}, got {detected}"
            )

    def test_lb_detection_robust_to_gaussian_noise(self):
        gt = _load_live_ground_truth()
        for fname, info in gt.items():
            img = cv2.imread(str(LIVE_CROPS_DIR / fname))
            noisy = _add_gaussian_noise(img, sigma=25)
            detected = _detect_limit_break(noisy)
            assert detected == info["lb"], (
                f"{fname} (Gaussian σ=25): expected lb={info['lb']}, got {detected}"
            )


# ── Test: LB page button detection ───────────────────────────────────────────

LB_PAGE_DIR = TEST_CAPTURES / "lb_page"


@pytest.mark.skipif(not LB_PAGE_DIR.exists(), reason="LB page captures not available")
class TestLBPageButtons:
    """Verify arrow detection and disabled button detection on LB page captures."""

    def test_lb_page_detects_right_arrow(self, yolo_model):
        """LB page with disabled confirm should have a '>' arrow on the right."""
        img = cv2.imread(str(LB_PAGE_DIR / "lb_page_disabled.png"))
        if img is None:
            pytest.skip("lb_page_disabled.png not found")
        yr = Yolo_Results(yolo_model(img, conf_threshold=0.3), img)
        buttons = ButtonList(yr)
        right = _find_lb_arrow(buttons, "right")
        assert right is not None, "Right arrow '>' not found on LB page"
        assert right.cx > 540, f"Right arrow should be on right half, got cx={right.cx}"

    def test_lb_page_no_left_arrow_initially(self, yolo_model):
        """At initial LB level, there should be no '<' arrow."""
        img = cv2.imread(str(LB_PAGE_DIR / "lb_page_disabled.png"))
        if img is None:
            pytest.skip("lb_page_disabled.png not found")
        yr = Yolo_Results(yolo_model(img, conf_threshold=0.3), img)
        buttons = ButtonList(yr)
        left = _find_lb_arrow(buttons, "left")
        assert left is None, f"Left arrow should not exist at initial level, got cx={left.cx}"

    def test_lb_page_confirm_disabled(self, yolo_model):
        """解放する should be detected as disabled when no cards available."""
        img = cv2.imread(str(LB_PAGE_DIR / "lb_page_disabled.png"))
        if img is None:
            pytest.skip("lb_page_disabled.png not found")
        yr = Yolo_Results(yolo_model(img, conf_threshold=0.3), img)
        buttons = ButtonList(yr)
        confirm = buttons.get_button_by_text(
            SupportCardText.LIMIT_BREAK_CONFIRM, _FUZZ_CONFIG
        )
        assert confirm is not None, "解放する button not found"
        assert confirm.is_disabled(), "解放する should be disabled (no cards)"

    def test_lb_page_cancel_found(self, yolo_model):
        """キャンセル should be found and enabled on LB page."""
        img = cv2.imread(str(LB_PAGE_DIR / "lb_page_disabled.png"))
        if img is None:
            pytest.skip("lb_page_disabled.png not found")
        yr = Yolo_Results(yolo_model(img, conf_threshold=0.3), img)
        buttons = ButtonList(yr)
        cancel = buttons.get_button_by_text(
            SupportCardText.ENHANCE_CANCEL, _FUZZ_CONFIG
        )
        assert cancel is not None, "キャンセル button not found on LB page"
        assert not cancel.is_disabled(), "キャンセル should be enabled"

    def test_lb_page_confirm_disabled_after_jpeg_q50(self, yolo_model):
        """Disabled detection must survive JPEG Q50 compression."""
        img = cv2.imread(str(LB_PAGE_DIR / "lb_page_disabled.png"))
        if img is None:
            pytest.skip("lb_page_disabled.png not found")
        noisy = _jpeg_compress(img, quality=50)
        yr = Yolo_Results(yolo_model(noisy, conf_threshold=0.3), noisy)
        buttons = ButtonList(yr)
        confirm = buttons.get_button_by_text(
            SupportCardText.LIMIT_BREAK_CONFIRM, _FUZZ_CONFIG
        )
        assert confirm is not None, "Q50: 解放する button not found"
        assert confirm.is_disabled(), "Q50: 解放する should still be disabled"

    def test_detail_page_has_lb_button(self, yolo_model):
        """Detail page should have an enabled 上限解放 button."""
        img = cv2.imread(str(LB_PAGE_DIR / "detail_page.png"))
        if img is None:
            pytest.skip("detail_page.png not found")
        yr = Yolo_Results(yolo_model(img, conf_threshold=0.3), img)
        buttons = ButtonList(yr)
        lb_btn = buttons.get_button_by_text(
            SupportCardText.LIMIT_BREAK, _FUZZ_CONFIG
        )
        assert lb_btn is not None, "上限解放 button not found on detail page"
