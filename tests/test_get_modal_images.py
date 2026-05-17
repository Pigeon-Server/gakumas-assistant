from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Game.Components.Modal import ModalParser
from src.entity.Yolo import Yolo_Results
from src.utils.game_tools import get_modal

DEFAULT_MODAL_DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "_artifacts" / "modal_debug"
MODAL_IMAGE_NAMES = sorted(path.name for path in (Path(__file__).resolve().parent / "modal_test_images").glob("*.PNG"))


@pytest.fixture(scope="module")
def base_ui_model():
    return YoloModelFromONNX(config.model_config["BASE_UI"])


def _build_modal_parser(image_name: str, base_ui_model: YoloModelFromONNX) -> ModalParser:
    image = cv2.imread(f"tests/modal_test_images/{image_name}")
    results = base_ui_model(image, conf_threshold=0.7)
    return ModalParser(Yolo_Results(results, image), no_body=True)


def _detect_modal(image_name: str, base_ui_model: YoloModelFromONNX):
    return _build_modal_parser(image_name, base_ui_model).parse()


def _detect_modal_from_image(image: np.ndarray, base_ui_model: YoloModelFromONNX):
    results = base_ui_model(image, conf_threshold=0.7)
    return ModalParser(Yolo_Results(results, image), no_body=True).parse()


def _compress_to_jpeg(image: np.ndarray, quality: int = 45) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok is True
    jpeg_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert jpeg_image is not None
    return jpeg_image


def render_modal_debug_image(
        image_name: str,
        base_ui_model: YoloModelFromONNX,
        output_dir: Path = DEFAULT_MODAL_DEBUG_OUTPUT_DIR,
) -> Path:
    parser = _build_modal_parser(image_name, base_ui_model)
    modal = parser.parse()
    assert modal is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_name).stem}_debug.png"
    assert cv2.imwrite(str(output_path), parser.draw_debug(modal))
    return output_path


@pytest.mark.parametrize(
    ("image_name", "expected_title"),
    [
        ("IMG_7512.PNG", "購入確認"),
        ("IMG_7530.PNG", "再生確認"),
        ("IMG_7540.PNG", "使用確認"),
    ],
)
def test_get_modal_recovers_title_when_modal_header_label_is_missing(image_name, expected_title, base_ui_model):
    modal = _detect_modal(image_name, base_ui_model)

    assert modal is not None
    assert modal.modal_title == expected_title


@pytest.mark.parametrize("image_name", MODAL_IMAGE_NAMES)
def test_all_modal_samples_remain_parseable_after_jpeg_compression(image_name, base_ui_model):
    image = cv2.imread(f"tests/modal_test_images/{image_name}")
    assert image is not None

    original_modal = _detect_modal_from_image(image, base_ui_model)
    assert original_modal is not None

    jpeg_image = _compress_to_jpeg(image, quality=45)
    compressed_modal = _detect_modal_from_image(jpeg_image, base_ui_model)

    assert compressed_modal is not None
    assert compressed_modal.modal_title == original_modal.modal_title


@pytest.mark.parametrize(
    ("image_name", "expected_title"),
    [
        ("IMG_7502.PNG", "受取完了"),
        ("IMG_7546.PNG", "アイドル詳細情報"),
    ],
)
def test_get_modal_keeps_single_action_modal_shape(image_name, expected_title, base_ui_model):
    modal = _detect_modal(image_name, base_ui_model)

    assert modal is not None
    assert modal.modal_title == expected_title
    assert modal.panel_box is not None
    assert modal.cancel_button is not None
    assert modal.confirm_button is None


def test_modal_parser_draws_debug_overlay(tmp_path, base_ui_model):
    parser = _build_modal_parser("IMG_7512.PNG", base_ui_model)
    modal = parser.parse()

    assert modal is not None
    assert modal.panel_box is not None

    debug_image = parser.draw_debug(modal)
    assert np.any(debug_image != parser.frame)

    output_path = tmp_path / "IMG_7512_debug.png"
    assert cv2.imwrite(str(output_path), debug_image)
    assert output_path.exists()


def test_get_modal_wrapper_uses_modal_component(base_ui_model):
    image = cv2.imread("tests/modal_test_images/IMG_7512.PNG")
    results = base_ui_model(image, conf_threshold=0.7)
    modal = get_modal(Yolo_Results(results, image), no_body=True)

    assert modal is not None
    assert modal.modal_title == "購入確認"


@pytest.mark.parametrize(
    ("image_name", "expected_title"),
    [
        ("IMG_7512.PNG", "購入確認"),
        ("IMG_7530.PNG", "再生確認"),
        ("IMG_7540.PNG", "使用確認"),
        ("IMG_7502.PNG", "受取完了"),
    ],
)
def test_get_modal_keeps_working_after_jpeg_compression(image_name, expected_title, base_ui_model):
    image = cv2.imread(f"tests/modal_test_images/{image_name}")
    assert image is not None

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    assert ok is True
    jpeg_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert jpeg_image is not None

    modal = _detect_modal_from_image(jpeg_image, base_ui_model)

    assert modal is not None
    assert modal.modal_title == expected_title


@pytest.mark.parametrize("image_name", ["IMG_7512.PNG", "IMG_7540.PNG", "IMG_7502.PNG"])
def test_modal_parser_builds_panel_layout(image_name, base_ui_model):
    modal = _detect_modal(image_name, base_ui_model)

    assert modal is not None
    assert modal.panel_box is not None
    assert modal.header_box is not None
    assert modal.body_box is not None
    assert modal.panel_box.x <= modal.header_box.x <= modal.header_box.w <= modal.panel_box.w
    assert modal.panel_box.x <= modal.body_box.x <= modal.body_box.w <= modal.panel_box.w
    assert modal.panel_box.y <= modal.header_box.y < modal.header_box.h <= modal.body_box.y
