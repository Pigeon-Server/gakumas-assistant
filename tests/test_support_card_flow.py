"""
Comprehensive test suite for the support card enhancement flow.

Tests ALL detection steps using real device captures (5 cards × 5 steps).
Validates: YOLO detection, OCR button text, chevron glyph detection,
card list parsing, and disabled-button HSV analysis.

JPEG noise and Gaussian noise tolerance tests are included to ensure
no strong template matching is used.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.SupportCard import SupportCardListParser
from src.constants.game.text.support_card_text import SupportCardText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.base_ui.auto_enhancement_support_card import (
    _get_level_cap,
    _detect_chevron_count,
)
from src.utils.string_tools import MatchConfig

_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)
CAPTURE_DIR = Path(__file__).resolve().parent.parent / "logs" / "debug" / "test_captures" / "support_card"
METADATA_PATH = CAPTURE_DIR / "metadata.json"


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model():
    model_path = Path(__file__).resolve().parent.parent / "model" / "base_ui.onnx"
    return YoloModelFromONNX(str(model_path))


@pytest.fixture(scope="session")
def metadata():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def captures(metadata):
    return metadata["captures"]


def _load_image(filename):
    path = CAPTURE_DIR / filename
    img = cv2.imread(str(path))
    assert img is not None, f"Failed to load {path}"
    return img


def _detect(model, frame):
    return Yolo_Results(model(frame), frame)


def _jpeg_compress(frame, quality=30):
    """Simulate JPEG compression noise at given quality."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _add_gaussian_noise(frame, sigma=15):
    """Add Gaussian noise to simulate capture variability."""
    noise = np.random.normal(0, sigma, frame.shape).astype(np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Card list page — SUPPORT_CARD labels + SupportCardListParser
# ──────────────────────────────────────────────────────────────────────────────

class TestStep1CardList:
    """Validate card list page detection across 5 card captures."""

    @pytest.mark.parametrize("card_idx", range(5))
    def test_support_card_labels_detected(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step1_list"]["file"])
        yr = _detect(model, frame)
        sc_boxes = list(yr.filter_by_label(BaseUILabels.SUPPORT_CARD).boxes)
        assert len(sc_boxes) >= 3, f"Expected ≥3 Support Card labels, got {len(sc_boxes)}"

    @pytest.mark.parametrize("card_idx", range(5))
    def test_card_list_parser_finds_cards(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step1_list"]["file"])
        yr = _detect(model, frame)
        card_list = SupportCardListParser(yr).parse()
        assert len(card_list) >= 3, f"Expected ≥3 parsed cards, got {len(card_list)}"

    @pytest.mark.parametrize("card_idx", range(5))
    def test_parsed_cards_have_rarity_and_level(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step1_list"]["file"])
        yr = _detect(model, frame)
        card_list = SupportCardListParser(yr).parse()
        valid = [c for c in card_list if c.rarity and c.level is not None]
        assert len(valid) >= 3, f"Expected ≥3 cards with rarity+level, got {len(valid)}"


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: After click — detail page OR selected on list
# ──────────────────────────────────────────────────────────────────────────────

class TestStep2AfterClick:
    """After clicking a card, we should be on either the detail page or list (selected)."""

    @pytest.mark.parametrize("card_idx", range(5))
    def test_on_detail_or_has_view_detail(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step2_after_click"]["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)

        on_detail = (
            btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ) is not None
            or btns.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ) is not None
        )
        has_view_detail = btns.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ) is not None

        assert on_detail or has_view_detail, (
            f"card_{card_idx}: Expected detail page (Lv強化/上限解放) or 詳細を見る, "
            f"got neither. Buttons: {[b.text for b in btns.buttons]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Detail page — Lv強化, 上限解放, Back Button
# ──────────────────────────────────────────────────────────────────────────────

class TestStep3DetailPage:
    """Detail page must have Lv強化 and 上限解放 buttons and a Back button."""

    @pytest.mark.parametrize("card_idx", range(5))
    def test_lv_enhance_button_found(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step3_detail"]["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        btn = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
        assert btn is not None, (
            f"card_{card_idx}: Lv強化 not found. Buttons: {[b.text for b in btns.buttons]}"
        )

    @pytest.mark.parametrize("card_idx", range(5))
    def test_limit_break_button_found(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step3_detail"]["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        btn = btns.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
        assert btn is not None, (
            f"card_{card_idx}: 上限解放 not found. Buttons: {[b.text for b in btns.buttons]}"
        )

    @pytest.mark.parametrize("card_idx", range(5))
    def test_back_button_exists(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step3_detail"]["file"])
        yr = _detect(model, frame)
        assert yr.exists_label(BaseUILabels.BACK_BTN), f"card_{card_idx}: Back button not found"


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Enhancement page — chevrons, confirm, cancel
# ──────────────────────────────────────────────────────────────────────────────

class TestStep4EnhancePage:
    """Enhancement page must have > and >> chevron buttons, confirm, and cancel."""

    @pytest.mark.parametrize("card_idx", range(5))
    def test_chevron_single_detected(self, model, captures, card_idx):
        entry = captures[card_idx]
        step4 = entry["steps"]["step4_enhance"]
        if step4.get("skipped"):
            pytest.skip(step4["reason"])
        frame = _load_image(step4["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        single_chevrons = [b for b in btns.buttons if _detect_chevron_count(b.frame) == 1]
        assert len(single_chevrons) >= 1, (
            f"card_{card_idx}: No single chevron (>) button found"
        )

    @pytest.mark.parametrize("card_idx", range(5))
    def test_chevron_double_detected(self, model, captures, card_idx):
        entry = captures[card_idx]
        step4 = entry["steps"]["step4_enhance"]
        if step4.get("skipped"):
            pytest.skip(step4["reason"])
        frame = _load_image(step4["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        double_chevrons = [b for b in btns.buttons if _detect_chevron_count(b.frame) == 2]
        assert len(double_chevrons) >= 1, (
            f"card_{card_idx}: No double chevron (>>) button found"
        )

    @pytest.mark.parametrize("card_idx", range(5))
    def test_cancel_button_found(self, model, captures, card_idx):
        entry = captures[card_idx]
        step4 = entry["steps"]["step4_enhance"]
        if step4.get("skipped"):
            pytest.skip(step4["reason"])
        frame = _load_image(step4["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        cancel = btns.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ)
        assert cancel is not None, (
            f"card_{card_idx}: キャンセル not found. Buttons: {[b.text for b in btns.buttons]}"
        )

    @pytest.mark.parametrize("card_idx", range(5))
    def test_confirm_button_found(self, model, captures, card_idx):
        entry = captures[card_idx]
        step4 = entry["steps"]["step4_enhance"]
        if step4.get("skipped"):
            pytest.skip(step4["reason"])
        frame = _load_image(step4["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        confirm = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
        assert confirm is not None, (
            f"card_{card_idx}: Lv強化 confirm not found. Buttons: {[b.text for b in btns.buttons]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Back to list — SUPPORT_CARD visible again
# ──────────────────────────────────────────────────────────────────────────────

class TestStep5BackToList:
    """After the full flow, we should be back on the card list page."""

    @pytest.mark.parametrize("card_idx", range(5))
    def test_support_card_labels_after_return(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step5_back_to_list"]["file"])
        yr = _detect(model, frame)
        sc_boxes = list(yr.filter_by_label(BaseUILabels.SUPPORT_CARD).boxes)
        assert len(sc_boxes) >= 3, (
            f"card_{card_idx}: Expected ≥3 Support Card labels after return, got {len(sc_boxes)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Chevron button crop tests — individual button frames
# ──────────────────────────────────────────────────────────────────────────────

class TestChevronButtonCrops:
    """Validate chevron glyph detection on all captured button crops."""

    def _get_button_crop_files(self):
        """Return list of (file, expected_chevrons) for all button crops."""
        results = []
        for card_idx in range(5):
            for btn_idx in range(4):
                path = CAPTURE_DIR / f"card_{card_idx}_btn{btn_idx}.png"
                if path.exists():
                    results.append((path, card_idx, btn_idx))
        return results

    def test_single_chevron_buttons_have_count_1(self):
        """btn0 files should be single chevron (>)."""
        for card_idx in range(5):
            path = CAPTURE_DIR / f"card_{card_idx}_btn0.png"
            if not path.exists():
                continue
            frame = cv2.imread(str(path))
            count = _detect_chevron_count(frame)
            assert count == 1, f"card_{card_idx}_btn0: expected 1, got {count}"

    def test_double_chevron_buttons_have_count_2(self):
        """btn1 files should be double chevron (>>)."""
        for card_idx in range(5):
            path = CAPTURE_DIR / f"card_{card_idx}_btn1.png"
            if not path.exists():
                continue
            frame = cv2.imread(str(path))
            count = _detect_chevron_count(frame)
            assert count == 2, f"card_{card_idx}_btn1: expected 2, got {count}"

    def test_text_buttons_have_count_0(self):
        """btn2 and btn3 (cancel/confirm with text) should have chevron count 0."""
        for card_idx in range(5):
            for btn_idx in (2, 3):
                path = CAPTURE_DIR / f"card_{card_idx}_btn{btn_idx}.png"
                if not path.exists():
                    continue
                frame = cv2.imread(str(path))
                count = _detect_chevron_count(frame)
                assert count == 0, f"card_{card_idx}_btn{btn_idx}: expected 0, got {count}"


# ──────────────────────────────────────────────────────────────────────────────
# Level cap validation
# ──────────────────────────────────────────────────────────────────────────────

class TestLevelCaps:
    """Validate level cap lookup matches game data."""

    @pytest.mark.parametrize("rarity,stars,expected", [
        ("R", 0, 20), ("R", 1, 25), ("R", 2, 30), ("R", 3, 35), ("R", 4, 40),
        ("SR", 0, 30), ("SR", 1, 35), ("SR", 2, 40), ("SR", 3, 45), ("SR", 4, 50),
        ("SSR", 0, 40), ("SSR", 1, 45), ("SSR", 2, 50), ("SSR", 3, 55), ("SSR", 4, 60),
    ])
    def test_level_cap_values(self, rarity, stars, expected):
        assert _get_level_cap(rarity, stars) == expected

    def test_captured_cards_have_correct_caps(self, captures):
        for entry in captures:
            card = entry["card"]
            expected_cap = _get_level_cap(card["rarity"], card["stars"])
            assert card["cap"] == expected_cap, (
                f"{entry['tag']}: cap={card['cap']} but expected {expected_cap}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# JPEG noise tolerance — all detection must survive Q30 compression
# ──────────────────────────────────────────────────────────────────────────────

class TestJPEGNoiseTolerance:
    """All key detections must work after JPEG Q30 compression."""

    @pytest.mark.parametrize("card_idx", range(5))
    def test_card_list_detection_q30(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step1_list"]["file"])
        noisy = _jpeg_compress(frame, quality=30)
        yr = _detect(model, noisy)
        sc = list(yr.filter_by_label(BaseUILabels.SUPPORT_CARD).boxes)
        assert len(sc) >= 3, f"Q30: Expected ≥3 SC labels, got {len(sc)}"

    @pytest.mark.parametrize("card_idx", range(5))
    def test_detail_buttons_q30(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step3_detail"]["file"])
        noisy = _jpeg_compress(frame, quality=30)
        yr = _detect(model, noisy)
        btns = ButtonList(yr)
        lv = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
        assert lv is not None, f"Q30: Lv強化 not found after JPEG Q30"

    @pytest.mark.parametrize("card_idx", range(5))
    def test_chevron_detection_q30(self, model, captures, card_idx):
        entry = captures[card_idx]
        step4 = entry["steps"]["step4_enhance"]
        if step4.get("skipped"):
            pytest.skip(step4["reason"])
        frame = _load_image(step4["file"])
        noisy = _jpeg_compress(frame, quality=30)
        yr = _detect(model, noisy)
        btns = ButtonList(yr)
        double = [b for b in btns.buttons if _detect_chevron_count(b.frame) == 2]
        assert len(double) >= 1, f"Q30: >> chevron not detected after JPEG Q30"

    @pytest.mark.parametrize("card_idx", range(5))
    def test_chevron_crop_q30(self, captures, card_idx):
        """Test chevron detection on button crops after JPEG Q30."""
        for btn_idx, expected in [(0, 1), (1, 2)]:
            path = CAPTURE_DIR / f"card_{card_idx}_btn{btn_idx}.png"
            if not path.exists():
                continue
            frame = cv2.imread(str(path))
            noisy = _jpeg_compress(frame, quality=30)
            count = _detect_chevron_count(noisy)
            assert count == expected, (
                f"Q30 card_{card_idx}_btn{btn_idx}: expected {expected}, got {count}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian noise tolerance
# ──────────────────────────────────────────────────────────────────────────────

class TestGaussianNoiseTolerance:
    """All key detections must work after Gaussian noise (sigma=10)."""

    @pytest.mark.parametrize("card_idx", range(5))
    def test_card_list_detection_noisy(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step1_list"]["file"])
        noisy = _add_gaussian_noise(frame, sigma=10)
        yr = _detect(model, noisy)
        sc = list(yr.filter_by_label(BaseUILabels.SUPPORT_CARD).boxes)
        assert len(sc) >= 3, f"Gaussian: Expected ≥3 SC labels, got {len(sc)}"

    @pytest.mark.parametrize("card_idx", range(5))
    def test_detail_buttons_noisy(self, model, captures, card_idx):
        entry = captures[card_idx]
        frame = _load_image(entry["steps"]["step3_detail"]["file"])
        noisy = _add_gaussian_noise(frame, sigma=10)
        yr = _detect(model, noisy)
        btns = ButtonList(yr)
        lv = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
        assert lv is not None, f"Gaussian: Lv強化 not found after noise sigma=10"

    @pytest.mark.parametrize("card_idx", range(5))
    def test_enhance_page_buttons_noisy(self, model, captures, card_idx):
        """Verify YOLO still detects buttons on noisy enhancement page."""
        entry = captures[card_idx]
        step4 = entry["steps"]["step4_enhance"]
        if step4.get("skipped"):
            pytest.skip(step4["reason"])
        frame = _load_image(step4["file"])
        noisy = _add_gaussian_noise(frame, sigma=10)
        yr = _detect(model, noisy)
        btns = ButtonList(yr)
        assert len(btns) >= 1, f"Gaussian: Expected ≥1 buttons, got {len(btns)}"

    @pytest.mark.parametrize("card_idx", range(5))
    def test_chevron_crop_noisy(self, captures, card_idx):
        """Test chevron detection on button crops after Gaussian noise."""
        for btn_idx, expected in [(0, 1), (1, 2)]:
            path = CAPTURE_DIR / f"card_{card_idx}_btn{btn_idx}.png"
            if not path.exists():
                continue
            frame = cv2.imread(str(path))
            noisy = _add_gaussian_noise(frame, sigma=10)
            count = _detect_chevron_count(noisy)
            assert count == expected, (
                f"Noisy card_{card_idx}_btn{btn_idx}: expected {expected}, got {count}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Disabled button state detection
# ──────────────────────────────────────────────────────────────────────────────

class TestDisabledButtonState:
    """Test that disabled button detection works correctly on captured images."""

    @pytest.mark.parametrize("card_idx", range(5))
    def test_detail_lv_enhance_not_disabled(self, model, captures, card_idx):
        """For non-max cards, Lv強化 should NOT be disabled."""
        entry = captures[card_idx]
        if entry["card"]["level"] >= entry["card"]["cap"]:
            pytest.skip("Max-level card, Lv強化 expected disabled")
        frame = _load_image(entry["steps"]["step3_detail"]["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        lv = btns.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
        assert lv is not None and not lv.is_disabled(), (
            f"card_{card_idx}: Lv強化 should not be disabled for non-max card"
        )

    @pytest.mark.parametrize("card_idx", [1, 2, 3])
    def test_limit_break_disabled_for_max_stars(self, model, captures, card_idx):
        """Cards with ★4 (max stars) should have 上限解放 disabled."""
        entry = captures[card_idx]
        if entry["card"]["stars"] != 4:
            pytest.skip("Not a 4-star card")
        frame = _load_image(entry["steps"]["step3_detail"]["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        lb = btns.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
        assert lb is not None, f"card_{card_idx}: 上限解放 not found"
        assert lb.is_disabled(), f"card_{card_idx}: 上限解放 should be disabled for ★4 card"

    @pytest.mark.parametrize("card_idx", [0, 4])
    def test_limit_break_enabled_for_non_max_stars(self, model, captures, card_idx):
        """Cards with <4 stars should have 上限解放 enabled."""
        entry = captures[card_idx]
        if entry["card"]["stars"] >= 4:
            pytest.skip("Already at max stars")
        frame = _load_image(entry["steps"]["step3_detail"]["file"])
        yr = _detect(model, frame)
        btns = ButtonList(yr)
        lb = btns.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
        assert lb is not None, f"card_{card_idx}: 上限解放 not found"
        assert not lb.is_disabled(), f"card_{card_idx}: 上限解放 should be enabled for <★4 card"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
