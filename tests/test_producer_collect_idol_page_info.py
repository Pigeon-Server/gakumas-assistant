from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from src.constants.game.text.produce_text import ProduceText
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Game.Components.Modal import ModalParser
from src.entity.Yolo import Yolo_Results
from src.core.tasks.producer_challenge.steps.diagnostics.collect_idol_page_info import (
    _extract_recommended_effect_anchor_lines,
    _extract_recommended_effect_lines_from_frame,
    _locate_recommended_effect_region,
    _locate_tooltip_search_region,
    _parse_tasks_from_body_frame,
    _read_exam_criteria_from_body,
)

DIAGNOSTIC_IMAGE_DIR = Path(__file__).resolve().parent / "producer_diagnostic_images"
IDOL_PAGE_SAMPLES = [
    "idol_page_live_a.png",
    "idol_page_live_b.png",
]
TOOLTIP_SAMPLE_PAIRS = [
    ("idol_page_live_a.png", "idol_tooltip_live_a.png"),
    ("idol_page_live_b.png", "idol_tooltip_live_b.png"),
]
TRAINING_INFO_SAMPLES = [
    "training_info_live_a.png",
]


def _load_image(image_name: str) -> np.ndarray:
    image = cv2.imread(str(DIAGNOSTIC_IMAGE_DIR / image_name))
    assert image is not None, f"无法读取样本: {image_name}"
    return image


def _build_results(image: np.ndarray, base_ui_model: YoloModelFromONNX) -> Yolo_Results:
    return Yolo_Results(base_ui_model(image, conf_threshold=0.7), image)


def _jpeg_compress(frame: np.ndarray, quality: int = 30) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok is True
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _add_gaussian_noise(frame: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    rng = np.random.RandomState(42)
    noise = rng.normal(0, sigma, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def base_ui_model():
    return YoloModelFromONNX(config.model_config["BASE_UI"])


@pytest.mark.parametrize("image_name", IDOL_PAGE_SAMPLES)
def test_recommended_effect_anchor_lines_are_found_on_live_idol_pages(image_name, base_ui_model):
    frame = _load_image(image_name)
    results = _build_results(frame, base_ui_model)

    recommend_region = _locate_recommended_effect_region(results, frame.shape)
    anchor_lines = _extract_recommended_effect_anchor_lines(frame, recommend_region)
    anchor_texts = [line.text for line in anchor_lines]

    assert len(anchor_lines) >= 2
    assert any("通常" in text for text in anchor_texts)
    assert any("合計" in text or "+60" in text for text in anchor_texts)


@pytest.mark.parametrize("image_name", IDOL_PAGE_SAMPLES)
def test_idol_pages_without_tooltip_do_not_false_positive(image_name, base_ui_model):
    frame = _load_image(image_name)
    results = _build_results(frame, base_ui_model)

    recommend_region = _locate_recommended_effect_region(results, frame.shape)
    tooltip_region = _locate_tooltip_search_region(results, recommend_region, frame.shape)
    tooltip_box, effects = _extract_recommended_effect_lines_from_frame(frame, tooltip_region)

    assert tooltip_box is None
    assert effects == []


@pytest.mark.parametrize(("idol_image_name", "tooltip_image_name"), TOOLTIP_SAMPLE_PAIRS)
def test_recommended_effect_tooltips_are_parseable_from_live_samples(
    idol_image_name,
    tooltip_image_name,
    base_ui_model,
):
    idol_frame = _load_image(idol_image_name)
    tooltip_frame = _load_image(tooltip_image_name)
    results = _build_results(idol_frame, base_ui_model)

    recommend_region = _locate_recommended_effect_region(results, idol_frame.shape)
    tooltip_region = _locate_tooltip_search_region(results, recommend_region, idol_frame.shape)
    tooltip_box, effects = _extract_recommended_effect_lines_from_frame(tooltip_frame, tooltip_region)
    combined = " ".join(effects)

    assert tooltip_box is not None
    assert len(effects) >= 5
    assert ProduceText.P_POINT in combined
    assert "ライバルのスコア" in combined
    assert "授業" in combined


def test_recommended_effect_tooltips_survive_jpeg_and_gaussian_noise(base_ui_model):
    failures: list[str] = []
    for idol_image_name, tooltip_image_name in TOOLTIP_SAMPLE_PAIRS:
        clean_idol = _load_image(idol_image_name)
        clean_tooltip = _load_image(tooltip_image_name)
        for noisy_idol, noisy_tooltip, label in [
            (_jpeg_compress(clean_idol, quality=30), _jpeg_compress(clean_tooltip, quality=30), "jpeg-q30"),
            (_add_gaussian_noise(clean_idol, sigma=10), _add_gaussian_noise(clean_tooltip, sigma=10), "gaussian-10"),
        ]:
            results = _build_results(noisy_idol, base_ui_model)
            recommend_region = _locate_recommended_effect_region(results, noisy_idol.shape)
            tooltip_region = _locate_tooltip_search_region(results, recommend_region, noisy_idol.shape)
            _tooltip_box, effects = _extract_recommended_effect_lines_from_frame(noisy_tooltip, tooltip_region)
            combined = " ".join(effects)
            if len(effects) < 5 or ProduceText.P_POINT not in combined or "ライバルのスコア" not in combined:
                failures.append(f"{label}:{idol_image_name}->{tooltip_image_name}: {effects}")

    assert not failures, "推荐效果抗噪失败:\n" + "\n".join(failures)


@pytest.mark.parametrize("image_name", TRAINING_INFO_SAMPLES)
def test_training_info_modal_samples_parse_exam_and_tasks(image_name, base_ui_model):
    frame = _load_image(image_name)
    modal = ModalParser(_build_results(frame, base_ui_model), no_body=True).parse()

    assert modal is not None
    assert modal.modal_title == ProduceText.TRAINING_INFO
    assert modal.body_box is not None

    exam = _read_exam_criteria_from_body(modal.body_box.frame)
    tasks, _task_region = _parse_tasks_from_body_frame(modal.body_box.frame, set())
    task_conditions = [task["condition"] for task in tasks]

    assert exam == {
        "target_score": 450,
        "priority": ["vocal", "dance", "visual"],
    }
    assert task_conditions == ["ダンス300以上", "ビジュアル450以上"]
    assert tasks[0]["type"] == "生活改善"
    assert tasks[0]["reward"] == "Pポイント+50"
    assert tasks[1]["type"] == "才気煥発"
    assert ProduceText.DRINK in tasks[1]["reward"]


@pytest.mark.parametrize(
    ("transform_name", "transform_fn"),
    [
        ("jpeg-q30", lambda image: _jpeg_compress(image, quality=30)),
        ("gaussian-10", lambda image: _add_gaussian_noise(image, sigma=10)),
    ],
)
def test_training_info_modal_keeps_working_after_noise(transform_name, transform_fn, base_ui_model):
    clean_frame = _load_image("training_info_live_a.png")
    noisy_frame = transform_fn(clean_frame)
    modal = ModalParser(_build_results(noisy_frame, base_ui_model), no_body=True).parse()

    assert modal is not None, transform_name
    assert modal.body_box is not None, transform_name

    exam = _read_exam_criteria_from_body(modal.body_box.frame)
    tasks, _task_region = _parse_tasks_from_body_frame(modal.body_box.frame, set())

    assert exam["target_score"] == 450, transform_name
    assert exam["priority"] == ["vocal", "dance", "visual"], transform_name
    assert [task["condition"] for task in tasks] == ["ダンス300以上", "ビジュアル450以上"], transform_name
