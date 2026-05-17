from __future__ import annotations

from pathlib import Path

import cv2
import pytest

import config
from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.model_type import YoloModelType
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.tasks.producer_challenge.ui import classify_gameplay_phase, classify_pipeline_position
from src.entity.Yolo import Yolo_Results


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ROOT = PROJECT_ROOT / "tests" / "produce_gameplay_captures"

_producer_model = None


def _get_producer_model():
    global _producer_model
    if _producer_model is None:
        _producer_model = YoloModelFromONNX(config.model_config[YoloModelType.PRODUCER])
    return _producer_model


def _infer_capture(folder: str) -> Yolo_Results:
    image_paths = sorted((CAPTURE_ROOT / folder).glob("*.png"))
    assert image_paths, f"未找到截图: {folder}"
    image = cv2.imread(str(image_paths[0]))
    assert image is not None and image.size > 0, f"无法读取截图: {image_paths[0]}"
    raw = _get_producer_model()(image, conf_threshold=0.5)
    return Yolo_Results(raw, image)


@pytest.mark.parametrize(
    ("folder", "expected_phase", "expected_position"),
    [
        ("post_exam_loading_1", GameplayPhase.RESULT, GameplayPosition.RESULT_EXAM_FAILURE),
        ("exam_result_summary_showcase", GameplayPhase.RESULT, GameplayPosition.RESULT_EXAM_SUMMARY_SHOWCASE),
        ("exam_result_ranking_summary", GameplayPhase.RESULT, GameplayPosition.RESULT_EXAM_RANKING_SUMMARY),
        ("skill_reward_showcase", GameplayPhase.SKILL_REWARD, GameplayPosition.SKILL_REWARD_SHOWCASE),
        ("fail_run_tail_fastforward", GameplayPhase.RESULT, GameplayPosition.RESULT_FINAL_EVALUATION),
        ("fail_run_final_stable", GameplayPhase.RESULT, GameplayPosition.RESULT_MEMORY_PAGE),
        ("fail_run_home_or_next", GameplayPhase.RESULT, GameplayPosition.RESULT_REWARD_SUMMARY),
    ],
)
def test_result_chain_phase_and_position_from_real_images(folder: str, expected_phase: str, expected_position: str):
    results = _infer_capture(folder)
    actual_phase = classify_gameplay_phase(results)
    actual_position = classify_pipeline_position(results)
    assert actual_phase == expected_phase, (folder, actual_phase)
    assert actual_position == expected_position, (folder, actual_position)


@pytest.mark.parametrize(
    ("folder", "modal_title", "expected_position"),
    [
        ("exam_after_skip_tap", ProduceText.END_TURN_CONFIRM, GameplayPosition.EXAM_END_TURN_CONFIRM_MODAL),
        ("exam_result_after_next_1", ProduceText.EXAM_RESULT_RETRY_CONFIRM, GameplayPosition.EXAM_RETRY_CONFIRM_MODAL),
        ("memory_regen_action_1", ProduceText.MEMORY_REGEN_CONFIRM, GameplayPosition.MEMORY_REGEN_CONFIRM_MODAL),
        ("fail_run_closeout_1", ProduceText.MEMORY_CONFIRM, GameplayPosition.MEMORY_CONFIRM_MODAL),
    ],
)
def test_modal_positions_from_real_images(folder: str, modal_title: str, expected_position: str):
    results = _infer_capture(folder)
    actual_phase = classify_gameplay_phase(results)
    actual_position = classify_pipeline_position(results, modal_title=modal_title)
    assert actual_phase == GameplayPhase.MODAL, (folder, actual_phase)
    assert actual_position == expected_position, (folder, actual_position)
