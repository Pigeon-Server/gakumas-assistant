import sys
from types import SimpleNamespace

import numpy as np


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.inference.ONNX import (
    ONNXYoloClassifyResult,
    ONNXYoloModelMeta,
    ONNXYoloResult,
)


def test_yolo_result_plot_returns_annotated_image():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    meta = ONNXYoloModelMeta(
        imgsz=(64, 64),
        names={0: "button"},
        colors={0: (0, 255, 0)},
    )
    result = ONNXYoloResult(
        boxes=np.array([[10, 12, 20, 18]], dtype=np.float32),
        scores=np.array([0.95], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int64),
        model_mata=meta,
        image=image,
    )

    plotted = result.plot()

    assert plotted.shape == image.shape
    assert np.any(plotted != image)


def test_classify_result_plot_returns_overlay_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    meta = ONNXYoloModelMeta(
        imgsz=(32, 32),
        names={0: "home"},
        colors={},
    )
    result = ONNXYoloClassifyResult(
        class_id=0,
        score=0.88,
        probs=np.array([0.88], dtype=np.float32),
        model_meta=meta,
        image=image,
    )

    plotted = result.plot()

    assert plotted.shape == image.shape
    assert np.any(plotted != image)
