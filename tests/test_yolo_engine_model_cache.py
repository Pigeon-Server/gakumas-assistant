import sys
from types import SimpleNamespace

import numpy as np


class _LoggerStub:
    def __getattr__(self, name):
        if name == "catch":
            return lambda func=None, *args, **kwargs: func
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.yolo.model_type import YoloModelType
import src.core.inference.yolo_engine as yolo_engine_module
from src.core.inference.yolo_engine import YoloInferenceEngine


class _FakeDevice:
    def capture(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)


def test_load_model_keeps_previous_onnx_session_alive_when_switching(monkeypatch):
    created: list["_FakeYoloModel"] = []
    destroyed: list[str] = []

    class _FakeYoloModel:
        def __init__(self, model_path: str):
            self.model_path = model_path
            self._model_meta = SimpleNamespace(names={})
            created.append(self)

        def __del__(self):
            destroyed.append(self.model_path)

    monkeypatch.setattr(yolo_engine_module, "YoloModelFromONNX", _FakeYoloModel)

    engine = YoloInferenceEngine(_FakeDevice())
    base_model = engine._engine
    engine.load_model(YoloModelType.PRODUCER)
    producer_model = engine._engine
    engine.load_model(YoloModelType.BASE_UI)

    assert base_model is engine._engine
    assert producer_model is engine._model_cache[YoloModelType.PRODUCER]
    assert len(created) == 2
    assert destroyed == []
