import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.core.services.idol_card_ui import (
    compute_selected_idol_card_similarity,
    extract_idol_card_ocr_region,
    extract_selected_idol_card_image,
    get_next_idol_card_candidate_box,
    has_selected_idol_card_changed,
    is_same_selected_idol_card,
)
from src.entity.Yolo import Yolo_Results
from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCE_FLOW_DIR = PROJECT_ROOT / "tests" / "_artifacts" / "produce_flow"
IDOL_CARD_DIR = PROJECT_ROOT / "tests" / "_artifacts" / "idol_card_learning"
METADATA_PATH = IDOL_CARD_DIR / "metadata.json"

SELECTION_PAGE_SAMPLES = sorted(PRODUCE_FLOW_DIR.glob("stepC_idol*.png"))
SELECTION_STEP_SAMPLES = sorted(PRODUCE_FLOW_DIR.glob("stepC_idol_selection*.png"))
LIVE_SAME_CARD_SAMPLES = sorted(IDOL_CARD_DIR.glob("live_same_card_*.png"))
LIVE_CLICK_ADVANCE_BEFORE = IDOL_CARD_DIR / "live_click_advance_before.png"
LIVE_CLICK_ADVANCE_AFTER = IDOL_CARD_DIR / "live_click_advance_after.png"


@pytest.fixture(scope="session")
def yolo_model():
    return YoloModelFromONNX(config.model_config["BASE_UI"])


@pytest.fixture(scope="session")
def ocr_service():
    return OCRService()


@pytest.fixture(scope="session")
def idol_card_db():
    return GakumasDatabase_IdolCardDataUtils()


@pytest.fixture(scope="session")
def live_metadata():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None and image.size > 0, f"Failed to load {path}"
    return image


def _detect(model, frame: np.ndarray) -> Yolo_Results:
    raw = model(frame, conf_threshold=0.5, iou_threshold=0.5)
    return Yolo_Results(raw, frame)


def _match_idol_card_id(frame: np.ndarray, ocr_service: OCRService, idol_card_db: GakumasDatabase_IdolCardDataUtils):
    roi = extract_idol_card_ocr_region(frame)
    assert roi is not None and roi.size > 0
    ocr_result = ocr_service.ocr(roi)
    texts = [item.text.strip() for item in getattr(ocr_result, "results", []) if item.text.strip()]
    for text in texts:
        status, db_result = idol_card_db.search(text)
        if status and db_result:
            return db_result.id, texts
    return None, texts


def _jpeg_compress(frame: np.ndarray, quality: int) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _gaussian_noise(frame: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class TestIdolSelectionDetection:
    @pytest.mark.parametrize("path", SELECTION_PAGE_SAMPLES, ids=lambda p: p.name)
    def test_selection_page_detects_selected_and_candidate(self, yolo_model, path: Path):
        frame = _load_image(path)
        yr = _detect(yolo_model, frame)
        assert yr.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED), f"{path.name}: missing PRODUCT_CARD_SELECTED"
        assert get_next_idol_card_candidate_box(yr) is not None, f"{path.name}: missing adjacent candidate card"

    @pytest.mark.parametrize("path", SELECTION_PAGE_SAMPLES, ids=lambda p: p.name)
    def test_selected_crop_is_trimmed_and_nonempty(self, yolo_model, path: Path):
        frame = _load_image(path)
        yr = _detect(yolo_model, frame)
        selected_box = yr.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
        assert selected_box is not None and selected_box.frame is not None

        cropped = extract_selected_idol_card_image(yr)
        assert cropped is not None and cropped.size > 0, f"{path.name}: selected crop should not be empty"
        assert cropped.shape[0] < selected_box.frame.shape[0], f"{path.name}: selected crop should trim top/bottom chrome"
        assert cropped.shape[1] < selected_box.frame.shape[1], f"{path.name}: selected crop should trim left/right chrome"
        assert cropped.shape[0] >= int(selected_box.frame.shape[0] * 0.70), f"{path.name}: selected crop trimmed too aggressively"
        assert cropped.shape[1] >= int(selected_box.frame.shape[1] * 0.70), f"{path.name}: selected crop trimmed too aggressively"


class TestIdolSelectionOCR:
    def test_same_step_selection_images_match_same_card(self, ocr_service, idol_card_db):
        matched_ids = []
        for path in SELECTION_STEP_SAMPLES:
            frame = _load_image(path)
            matched_id, texts = _match_idol_card_id(frame, ocr_service, idol_card_db)
            assert matched_id is not None, f"{path.name}: OCR failed to match idol card. Texts: {texts}"
            matched_ids.append(matched_id)
        assert len(set(matched_ids)) == 1, f"Expected same idol card across selection captures, got {matched_ids}"

    def test_live_same_card_images_match_expected_card(self, ocr_service, idol_card_db, live_metadata):
        expected_id = live_metadata["same_card_expected_id"]
        for path in LIVE_SAME_CARD_SAMPLES:
            frame = _load_image(path)
            matched_id, texts = _match_idol_card_id(frame, ocr_service, idol_card_db)
            assert matched_id == expected_id, (
                f"{path.name}: expected {expected_id}, got {matched_id}. Texts: {texts}"
            )

    def test_click_advance_changes_recognized_idol_card(self, ocr_service, idol_card_db, live_metadata):
        before_frame = _load_image(LIVE_CLICK_ADVANCE_BEFORE)
        after_frame = _load_image(LIVE_CLICK_ADVANCE_AFTER)

        before_id, before_texts = _match_idol_card_id(before_frame, ocr_service, idol_card_db)
        after_id, after_texts = _match_idol_card_id(after_frame, ocr_service, idol_card_db)

        assert before_id == live_metadata["click_advance_before_id"], before_texts
        assert after_id == live_metadata["click_advance_after_id"], after_texts
        assert before_id != after_id, "Clicking the adjacent candidate should advance to a different idol card"


class TestIdolSelectionNoiseRobustness:
    @pytest.mark.parametrize("jpeg_quality", [95, 80, 60, 40])
    def test_selection_detection_survives_jpeg_noise(self, yolo_model, ocr_service, idol_card_db, jpeg_quality: int):
        failures = []
        for path in SELECTION_STEP_SAMPLES:
            noisy = _jpeg_compress(_load_image(path), quality=jpeg_quality)
            yr = _detect(yolo_model, noisy)
            matched_id, texts = _match_idol_card_id(noisy, ocr_service, idol_card_db)
            if not yr.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED):
                failures.append(f"{path.name} @q={jpeg_quality}: missing PRODUCT_CARD_SELECTED")
            if get_next_idol_card_candidate_box(yr) is None:
                failures.append(f"{path.name} @q={jpeg_quality}: missing PRODUCT_CARD_CANDIDATE")
            if matched_id is None:
                failures.append(f"{path.name} @q={jpeg_quality}: OCR match failed ({texts})")
        assert not failures, "JPEG noise failures:\n" + "\n".join(failures)

    @pytest.mark.parametrize("sigma", [5, 10, 15])
    def test_selection_detection_survives_gaussian_noise(self, yolo_model, ocr_service, idol_card_db, sigma: int):
        np.random.seed(42)
        failures = []
        for path in SELECTION_STEP_SAMPLES:
            noisy = _gaussian_noise(_load_image(path), sigma=sigma)
            yr = _detect(yolo_model, noisy)
            matched_id, texts = _match_idol_card_id(noisy, ocr_service, idol_card_db)
            if not yr.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED):
                failures.append(f"{path.name} @sigma={sigma}: missing PRODUCT_CARD_SELECTED")
            if get_next_idol_card_candidate_box(yr) is None:
                failures.append(f"{path.name} @sigma={sigma}: missing PRODUCT_CARD_CANDIDATE")
            if matched_id is None:
                failures.append(f"{path.name} @sigma={sigma}: OCR match failed ({texts})")
        assert not failures, "Gaussian noise failures:\n" + "\n".join(failures)

    def test_clean_retry_recovers_after_heavy_jpeg_noise(self, yolo_model, ocr_service, idol_card_db):
        for path in SELECTION_STEP_SAMPLES:
            clean = _load_image(path)
            noisy = _jpeg_compress(clean, quality=20)

            noisy_results = _detect(yolo_model, noisy)
            clean_results = _detect(yolo_model, clean)
            noisy_match, _ = _match_idol_card_id(noisy, ocr_service, idol_card_db)
            clean_match, _ = _match_idol_card_id(clean, ocr_service, idol_card_db)

            selected_found = (
                noisy_results.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED)
                or clean_results.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED)
            )
            candidate_found = (
                get_next_idol_card_candidate_box(noisy_results) is not None
                or get_next_idol_card_candidate_box(clean_results) is not None
            )
            matched_found = noisy_match is not None or clean_match is not None

            assert selected_found, f"{path.name}: PRODUCT_CARD_SELECTED not recovered after retry"
            assert candidate_found, f"{path.name}: PRODUCT_CARD_CANDIDATE not recovered after retry"
            assert matched_found, f"{path.name}: idol card OCR match not recovered after retry"


class TestSelectedCardSimilarity:
    def test_same_card_images_stay_above_repeat_threshold(self, yolo_model):
        card_images = []
        for path in LIVE_SAME_CARD_SAMPLES:
            yr = _detect(yolo_model, _load_image(path))
            card_image = extract_selected_idol_card_image(yr)
            assert card_image is not None and card_image.size > 0
            card_images.append((path.name, card_image))

        for (name_a, image_a), (name_b, image_b) in combinations(card_images, 2):
            score = compute_selected_idol_card_similarity(image_a, image_b)
            assert is_same_selected_idol_card(image_a, image_b), (
                f"{name_a} vs {name_b}: expected same selected idol card, similarity={score:.4f}"
            )
            assert not has_selected_idol_card_changed(image_a, image_b), (
                f"{name_a} vs {name_b}: change detector should stay false for same card, similarity={score:.4f}"
            )

    def test_click_advance_pair_is_detected_as_changed(self, yolo_model):
        before_results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_BEFORE))
        after_results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_AFTER))
        before_card = extract_selected_idol_card_image(before_results)
        after_card = extract_selected_idol_card_image(after_results)

        assert before_card is not None and after_card is not None
        score = compute_selected_idol_card_similarity(before_card, after_card)
        assert has_selected_idol_card_changed(before_card, after_card), (
            f"Expected candidate click to change selected card, similarity={score:.4f}"
        )
        assert not is_same_selected_idol_card(before_card, after_card), (
            f"Expected candidate click pair to be different cards, similarity={score:.4f}"
        )
