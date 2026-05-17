import sys
from types import SimpleNamespace

import numpy as np


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.device.Android.adapters.scrcpy_adapter import ScrcpyAdapter


class _DecodedFrameStub:
    def __init__(self, rgb_image: np.ndarray):
        self._rgb_image = rgb_image

    def to_ndarray(self, format: str):
        assert format == "rgb24"
        return self._rgb_image.copy()


def test_frame_to_bgr_matches_opencv_channel_order():
    rgb_image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [12, 34, 56]],
        ],
        dtype=np.uint8,
    )

    bgr_image = ScrcpyAdapter._frame_to_bgr(_DecodedFrameStub(rgb_image))

    assert tuple(bgr_image[0, 0]) == (0, 0, 255)
    assert tuple(bgr_image[0, 1]) == (0, 255, 0)
    assert tuple(bgr_image[1, 0]) == (255, 0, 0)
    assert tuple(bgr_image[1, 1]) == (56, 34, 12)
