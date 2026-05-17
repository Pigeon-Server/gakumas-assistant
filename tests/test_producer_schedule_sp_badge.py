from types import SimpleNamespace

import cv2
import numpy as np

from src.core.tasks.producer_challenge.gameplay.decision_support.enrichment import detect_sp_badge


def _badge_frame() -> np.ndarray:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    hsv = np.zeros((56, 86, 3), dtype=np.uint8)
    for x in range(hsv.shape[1]):
        hue = int(160 - (50 * x / max(1, hsv.shape[1] - 1)))
        hsv[:, x, 0] = np.clip(hue, 0, 179)
        hsv[:, x, 1] = 220
        hsv[:, x, 2] = 220
    badge = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame[0 : badge.shape[0], 0 : badge.shape[1]] = badge
    cv2.putText(
        frame,
        "SP",
        (14, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def test_detect_sp_badge_matches_gradient_badge():
    box = SimpleNamespace(frame=_badge_frame())
    assert detect_sp_badge(box)


def test_detect_sp_badge_rejects_neutral_patch():
    frame = np.full((180, 320, 3), 190, dtype=np.uint8)
    box = SimpleNamespace(frame=frame)
    assert detect_sp_badge(box) is False
