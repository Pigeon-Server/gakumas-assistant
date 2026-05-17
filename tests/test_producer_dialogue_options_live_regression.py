from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.constants.yolo.model_type import YoloModelType
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.tasks.producer_challenge.ui.gameplay_state import classify_gameplay_state
from src.entity.Yolo import Yolo_Results

CAPTURE_DIR = (
    PROJECT_ROOT
    / "tests"
    / "produce_gameplay_captures"
    / "dialogue_options_live"
)


def _collect_images() -> list[Path]:
    if not CAPTURE_DIR.exists():
        return []
    return sorted(
        p
        for p in CAPTURE_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg"} and "annotated" not in p.name
    )


_IMAGES = _collect_images()


@pytest.fixture(scope="module")
def producer_model() -> YoloModelFromONNX:
    return YoloModelFromONNX(config.model_config[YoloModelType.PRODUCER])


def _run_state(model: YoloModelFromONNX, frame) -> tuple[str, str, int]:
    yr = Yolo_Results(model(frame, conf_threshold=0.5), frame)
    phase, position = classify_gameplay_state(yr)
    option_count = len(list(yr.filter_by_label(ProducerLabels.UNIVERSAL_OPTIONS)))
    return str(phase), position, option_count


def _inject_jpg_noise(frame, quality: int = 65):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return frame
    noisy = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return noisy if noisy is not None else frame


@pytest.mark.skipif(not _IMAGES, reason="dialogue_options_live 样本不存在")
@pytest.mark.parametrize("image_path", _IMAGES)
def test_dialogue_options_live_classification_stable(
    producer_model: YoloModelFromONNX,
    image_path: Path,
):
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"读取失败: {image_path}"
    phase, position, option_count = _run_state(producer_model, frame)
    assert phase == "GameplayPhase.DIALOGUE", (
        f"{image_path.name} phase 异常: {phase}"
    )
    assert position == "dialogue_options", (
        f"{image_path.name} position 异常: {position}"
    )
    assert option_count >= 3, (
        f"{image_path.name} 选项数不足: {option_count}"
    )


@pytest.mark.skipif(not _IMAGES, reason="dialogue_options_live 样本不存在")
@pytest.mark.parametrize(
    "image_path",
    [p for p in _IMAGES if p.suffix.lower() == ".png"],
)
def test_dialogue_options_live_jpg_noise_resilient(
    producer_model: YoloModelFromONNX,
    image_path: Path,
):
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"读取失败: {image_path}"
    noisy = _inject_jpg_noise(frame, quality=65)
    phase, position, option_count = _run_state(producer_model, noisy)
    assert phase == "GameplayPhase.DIALOGUE", (
        f"{image_path.name} JPG 噪点后 phase 异常: {phase}"
    )
    assert position == "dialogue_options", (
        f"{image_path.name} JPG 噪点后 position 异常: {position}"
    )
    assert option_count >= 3, (
        f"{image_path.name} JPG 噪点后选项数不足: {option_count}"
    )
