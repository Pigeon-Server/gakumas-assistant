"""
竞技场(Contest) + 支援卡详情页按钮检测的综合测试。

覆盖：
  - ContestItem OCR 解析（含 "総合力合計" 锚点模糊匹配）
  - 支援卡详情页按钮检测（Lv強化 / 上限解放）
  - JPEG 压缩噪点鲁棒性测试
  - 多张不同截图的跨卡一致性测试

运行方式：
  cd /path/to/gakumas
  python -m pytest tests/test_contest_and_support_card_robustness.py -v
"""
import os
import glob

import cv2
import numpy as np
import pytest

import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.entity.Yolo import Yolo_Results, Yolo_Box
from src.entity.Game.Components.Contest import ContestItem, ContestList
from src.entity.Game.Components.Button import ButtonList
from src.constants.game.text.support_card_text import SupportCardText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.utils.string_tools import MatchConfig, string_match

# ── Paths ─────────────────────────────────────────────────────────────────────
CONTEST_DIR = os.path.join("logs", "debug", "test_captures", "contest")
DETAIL_DIR = os.path.join("logs", "debug", "test_captures", "support_card_detail")
_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def yolo_model():
    return YoloModelFromONNX(config.model_config["BASE_UI"])


@pytest.fixture(scope="session")
def ocr_service():
    return OCRService()


def _detect(model, frame):
    raw = model(frame, conf_threshold=0.5, iou_threshold=0.5)
    return Yolo_Results(raw, frame)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_jpeg_noise(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode → decode as JPEG at given quality to simulate compression noise."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _add_gaussian_noise(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Add Gaussian noise to simulate sensor/capture noise."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def _glob_images(directory: str, pattern: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        pytest.skip(f"No images matching {pattern} in {directory}")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Contest Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestContestItemParsing:
    """Test ContestItem OCR parsing from captured contest item ROIs."""

    @pytest.fixture(scope="class")
    def contest_item_images(self):
        """Load all captured contest item images."""
        paths = _glob_images(CONTEST_DIR, "cap*_item*.png")
        return [(p, cv2.imread(p)) for p in paths]

    def test_all_items_parseable(self, contest_item_images, ocr_service):
        """Every captured contest item must parse successfully."""
        failures = []
        for path, img in contest_item_images:
            try:
                ci = ContestItem(0, 0, img.shape[1], img.shape[0], "test", img)
                assert ci.combat_power is not None, f"combat_power is None"
                assert ci.combat_power > 0, f"combat_power={ci.combat_power}"
            except Exception as e:
                failures.append(f"{os.path.basename(path)}: {e}")
        assert not failures, f"Failed items:\n" + "\n".join(failures)

    def test_combat_power_values_consistent(self, contest_item_images, ocr_service):
        """Same index item across captures should have same combat_power."""
        by_item_idx: dict[str, list[int]] = {}
        for path, img in contest_item_images:
            # extract item index from filename e.g. "cap00_item1.png" → "item1"
            basename = os.path.basename(path)
            item_key = basename.split("_")[-1].replace(".png", "")
            try:
                ci = ContestItem(0, 0, img.shape[1], img.shape[0], "test", img)
                if ci.combat_power is not None:
                    by_item_idx.setdefault(item_key, []).append(ci.combat_power)
            except Exception:
                pass

        for item_key, powers in by_item_idx.items():
            unique = set(powers)
            assert len(unique) == 1, (
                f"{item_key}: inconsistent combat_power across captures: {powers}"
            )

    @pytest.mark.parametrize("jpeg_quality", [95, 80, 60, 40])
    def test_jpeg_noise_robustness(self, contest_item_images, jpeg_quality):
        """Contest items must parse correctly after JPEG re-encoding at various qualities."""
        failures = []
        for path, img in contest_item_images:
            noisy = _add_jpeg_noise(img, jpeg_quality)
            try:
                ci = ContestItem(0, 0, noisy.shape[1], noisy.shape[0], "test", noisy)
                assert ci.combat_power is not None
            except Exception as e:
                failures.append(f"{os.path.basename(path)} @q={jpeg_quality}: {e}")
        assert not failures, f"JPEG q={jpeg_quality} failures:\n" + "\n".join(failures)

    @pytest.mark.parametrize("sigma", [5, 10, 15])
    def test_gaussian_noise_robustness(self, contest_item_images, sigma):
        """Contest items must parse correctly with added Gaussian noise."""
        np.random.seed(42)
        failures = []
        for path, img in contest_item_images:
            noisy = _add_gaussian_noise(img, sigma=sigma)
            try:
                ci = ContestItem(0, 0, noisy.shape[1], noisy.shape[0], "test", noisy)
                assert ci.combat_power is not None
            except Exception as e:
                failures.append(f"{os.path.basename(path)} @sigma={sigma}: {e}")
        assert not failures, f"Gaussian sigma={sigma} failures:\n" + "\n".join(failures)


class TestContestListFromFullFrame:
    """Test ContestList construction from full-frame screenshots."""

    @pytest.fixture(scope="class")
    def contest_full_frames(self, yolo_model):
        paths = _glob_images(CONTEST_DIR, "cap*_full.png")
        results = []
        for p in paths:
            frame = cv2.imread(p)
            yr = _detect(yolo_model, frame)
            results.append((p, frame, yr))
        return results

    def test_detect_three_items(self, contest_full_frames):
        """Each full frame should yield exactly 3 contest items."""
        for path, frame, yr in contest_full_frames:
            # Check if we have the necessary YOLO labels
            buttons = yr.filter_by_label(BaseUILabels.BUTTON)
            back_btns = yr.filter_by_label(BaseUILabels.BACK_BTN)
            if not buttons or not back_btns:
                pytest.skip(f"Frame {path} is not on contest page")

            try:
                cl = ContestList(yr, frame)
                assert len(cl) == 3, f"{os.path.basename(path)}: expected 3 items, got {len(cl)}"
            except Exception as e:
                pytest.fail(f"{os.path.basename(path)}: ContestList failed: {e}")

    def test_all_items_have_valid_data(self, contest_full_frames):
        """All contest items from full frames must have valid combat_power and username."""
        for path, frame, yr in contest_full_frames:
            buttons = yr.filter_by_label(BaseUILabels.BUTTON)
            back_btns = yr.filter_by_label(BaseUILabels.BACK_BTN)
            if not buttons or not back_btns:
                continue
            try:
                cl = ContestList(yr, frame)
                for ci in cl:
                    assert ci.combat_power is not None and ci.combat_power > 0
                    assert ci.username and len(ci.username) > 0
            except Exception as e:
                pytest.fail(f"{os.path.basename(path)}: {e}")

    def test_fallback_when_only_footer_button_detected(self, contest_full_frames):
        """Even if only a footer button is detected, ContestList should still find 3 items."""
        for path, frame, yr in contest_full_frames:
            back_btn = yr.filter_by_label(BaseUILabels.BACK_BTN).first()
            if back_btn is None:
                continue

            h, w = frame.shape[:2]
            button_top = int(h * 0.90)
            button_bottom = int(h * 0.94)
            button_left = int(w * 0.28)
            button_right = int(w * 0.72)
            synthetic_button = Yolo_Box(
                button_left,
                button_top,
                button_right,
                button_bottom,
                BaseUILabels.BUTTON,
                frame[button_top:button_bottom, button_left:button_right],
            )
            synthetic_results = Yolo_Results.from_boxes([synthetic_button, back_btn])

            cl = ContestList(synthetic_results, frame)
            assert cl.contest_area is not None and cl.contest_area.size > 0, (
                f"{os.path.basename(path)}: contest_area should not be empty"
            )
            assert cl.contest_area.shape[0] > int(h * 0.2), (
                f"{os.path.basename(path)}: contest_area height too small: {cl.contest_area.shape[0]}"
            )
            assert len(cl) == 3, f"{os.path.basename(path)}: expected 3 items, got {len(cl)}"

    def test_no_button_does_not_break_debug_artifacts(self, contest_full_frames):
        """ContestList should keep initialized attributes even when BUTTON labels are missing."""
        for path, frame, yr in contest_full_frames:
            back_btn = yr.filter_by_label(BaseUILabels.BACK_BTN).first()
            if back_btn is None:
                continue

            synthetic_results = Yolo_Results.from_boxes([back_btn])
            cl = ContestList(synthetic_results, frame)
            assert hasattr(cl, "contest_area"), f"{os.path.basename(path)}: contest_area missing"
            assert cl.contest_area is not None


class TestContestAnchorFuzzyMatch:
    """Verify that the anchor text matching handles known OCR confusions."""

    @pytest.mark.parametrize("ocr_text,should_match", [
        ("総合力合計", True),    # exact
        ("総合カ合計", True),    # 力→カ (most common confusion)
        ("総合力合計:", True),   # trailing colon
        ("総合カ合計:", True),   # trailing colon + confusion
        ("総 合力合計", True),   # space break still matches at fuzz ~91%
        ("合計力総合", False),   # reversed order - should not match
        ("ダメなプロデューサ", False),  # clearly unrelated
    ])
    def test_anchor_match(self, ocr_text, should_match):
        from src.entity.Game.Components.Contest import _ANCHOR_TEXT, _ANCHOR_MATCH_CONFIG
        result = string_match(ocr_text, _ANCHOR_TEXT, _ANCHOR_MATCH_CONFIG)
        assert bool(result) == should_match, (
            f"string_match({ocr_text!r}, {_ANCHOR_TEXT!r}) = {bool(result)}, expected {should_match}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Support Card Detail Page Button Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSupportCardDetailButtons:
    """Test button detection on captured support card detail pages."""

    @pytest.fixture(scope="class")
    def detail_images(self, yolo_model):
        """Load all captured detail page images."""
        paths = _glob_images(DETAIL_DIR, "card*_detail_stable.png")
        results = []
        for p in paths:
            frame = cv2.imread(p)
            yr = _detect(yolo_model, frame)
            results.append((p, frame, yr))
        return results

    def test_lv_enhance_button_found(self, detail_images):
        """Lv強化 button should be found on all detail pages."""
        failures = []
        for path, frame, yr in detail_images:
            buttons = ButtonList(yr)
            btn = buttons.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
            if btn is None:
                failures.append(os.path.basename(path))
        assert not failures, f"Lv強化 not found on: {failures}"

    def test_limit_break_button_found(self, detail_images):
        """上限解放 button should be found on all detail pages."""
        failures = []
        for path, frame, yr in detail_images:
            buttons = ButtonList(yr)
            btn = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
            if btn is None:
                failures.append(os.path.basename(path))
        assert not failures, f"上限解放 not found on: {failures}"

    def test_limit_break_enabled_for_non_max_stars(self, detail_images):
        """上限解放 should be ENABLED for cards not at 4★ max."""
        # Card index 4 has stars=3 so should be enabled
        for path, frame, yr in detail_images:
            if "card04" in os.path.basename(path):
                buttons = ButtonList(yr)
                btn = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
                assert btn is not None, "上限解放 button not found on card04"
                assert not btn.is_disabled(), "上限解放 should be enabled for non-4★ card"

    @pytest.mark.parametrize("jpeg_quality", [95, 80, 60, 40])
    def test_buttons_survive_jpeg_noise(self, detail_images, yolo_model, jpeg_quality):
        """Buttons must still be found after JPEG re-encoding."""
        failures = []
        for path, frame, yr in detail_images:
            noisy = _add_jpeg_noise(frame, jpeg_quality)
            yr_noisy = _detect(yolo_model, noisy)
            buttons = ButtonList(yr_noisy)

            lv_btn = buttons.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
            lb_btn = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
            if lv_btn is None:
                failures.append(f"{os.path.basename(path)} @q={jpeg_quality}: Lv強化 missing")
            if lb_btn is None:
                failures.append(f"{os.path.basename(path)} @q={jpeg_quality}: 上限解放 missing")
        assert not failures, f"JPEG noise failures:\n" + "\n".join(failures)

    @pytest.mark.parametrize("sigma", [5, 10])
    def test_buttons_survive_gaussian_noise(self, detail_images, yolo_model, sigma):
        """Buttons must still be found after Gaussian noise (sigma<=10)."""
        np.random.seed(42)
        failures = []
        for path, frame, yr in detail_images:
            noisy = _add_gaussian_noise(frame, sigma=sigma)
            yr_noisy = _detect(yolo_model, noisy)
            buttons = ButtonList(yr_noisy)

            lv_btn = buttons.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ)
            lb_btn = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ)
            if lv_btn is None:
                failures.append(f"{os.path.basename(path)} @sigma={sigma}: Lv強化 missing")
            if lb_btn is None:
                failures.append(f"{os.path.basename(path)} @sigma={sigma}: 上限解放 missing")
        assert not failures, f"Gaussian noise failures:\n" + "\n".join(failures)

    def test_buttons_recovered_after_heavy_gaussian_retry(self, detail_images, yolo_model):
        """With heavy noise (sigma=15), retry on clean image must recover all buttons."""
        np.random.seed(42)
        for path, frame, yr in detail_images:
            noisy = _add_gaussian_noise(frame, sigma=15)
            yr_noisy = _detect(yolo_model, noisy)
            buttons_noisy = ButtonList(yr_noisy)

            # Clean retry
            yr_clean = _detect(yolo_model, frame)
            buttons_clean = ButtonList(yr_clean)

            lv_found = (
                buttons_noisy.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ) is not None
                or buttons_clean.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ) is not None
            )
            lb_found = (
                buttons_noisy.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ) is not None
                or buttons_clean.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ) is not None
            )
            assert lv_found, f"{os.path.basename(path)}: Lv強化 not recovered"
            assert lb_found, f"{os.path.basename(path)}: 上限解放 not recovered"


class TestSupportCardDetailButtonsRetry:
    """
    Verify that re-screenshot + re-detect recovers from a single-frame miss.
    Simulates by adding heavy noise on attempt 1, clean on attempt 2.
    """

    @pytest.fixture(scope="class")
    def detail_clean_images(self, yolo_model):
        paths = _glob_images(DETAIL_DIR, "card*_detail_stable.png")
        results = []
        for p in paths:
            frame = cv2.imread(p)
            results.append((p, frame))
        return results

    def test_retry_recovers_after_heavy_noise(self, detail_clean_images, yolo_model):
        """Even if first attempt (heavy noise) fails, clean retry must succeed."""
        for path, clean_frame in detail_clean_images:
            # Attempt 1: very heavy noise (quality=20)
            noisy = _add_jpeg_noise(clean_frame, quality=20)
            yr_noisy = _detect(yolo_model, noisy)
            buttons_noisy = ButtonList(yr_noisy)
            # Attempt 2: clean
            yr_clean = _detect(yolo_model, clean_frame)
            buttons_clean = ButtonList(yr_clean)

            # At least one attempt must find both buttons
            lv_found = (
                buttons_noisy.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ) is not None
                or buttons_clean.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ) is not None
            )
            lb_found = (
                buttons_noisy.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ) is not None
                or buttons_clean.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ) is not None
            )
            assert lv_found, f"{os.path.basename(path)}: Lv強化 not recovered"
            assert lb_found, f"{os.path.basename(path)}: 上限解放 not recovered"
