from types import SimpleNamespace

import numpy as np

from src.entity.Yolo import Yolo_Results
from src.utils.debug_tools import DebugTools


class _FakeYoloResults(list):
    def __init__(self, boxes):
        super().__init__(boxes)
        self.class_ids = [0 for _ in boxes]
        self.model_mata = SimpleNamespace(names={0: "dummy"})


def test_yolo_results_clamp_boxes_to_frame_bounds():
    frame = np.zeros((100, 50, 3), dtype=np.uint8)
    results = Yolo_Results(_FakeYoloResults([[40, 80, 20, 30]]), frame)

    box = results.first()
    assert box is not None
    assert box.x == 40
    assert box.y == 80
    assert box.w == 50
    assert box.h == 100
    assert box.frame.shape == (20, 10, 3)


def test_debug_tools_draw_boxes_handles_out_of_bounds_boxes():
    debugger = DebugTools()
    debugger.clear_all()
    debugger.add_box(-10, 5, 80, 30, color=(0, 255, 0), duration=5.0)

    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    first = debugger.draw_boxes(frame.copy())
    second = debugger.draw_boxes(frame.copy())

    assert first.shape == frame.shape
    assert second.shape == frame.shape
    assert np.any(second != frame)

    debugger.clear_all()
