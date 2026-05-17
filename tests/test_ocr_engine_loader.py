import sys
from types import SimpleNamespace

import numpy as np


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.inference.ocr_backends.base import OCRBackendResult
from src.core.inference import ocr_engine
from src.entity.Base import SingletonMeta


class _FailingBackend:
    name = "vision"
    requires_dml_lock = False

    def infer(self, img, use_cls: bool = False):
        raise RuntimeError("vision inference failed")


class _RapidFallbackBackend:
    name = "rapidocr"
    requires_dml_lock = True

    def __init__(self):
        self.calls = 0

    def infer(self, img, use_cls: bool = False):
        self.calls += 1
        return OCRBackendResult()


def test_ocr_loader_falls_back_to_rapidocr_when_backend_infer_fails(monkeypatch):
    fallback_backend = _RapidFallbackBackend()

    monkeypatch.setattr(ocr_engine, "create_ocr_backend", lambda: _FailingBackend())
    monkeypatch.setattr(ocr_engine, "RapidOCRBackend", lambda: fallback_backend)
    SingletonMeta._instances.pop(ocr_engine.OCRLoader, None)

    try:
        loader = ocr_engine.OCRLoader()
        result = loader(np.zeros((8, 8, 3), dtype=np.uint8), use_cls=False)

        assert result.boxes == []
        assert loader.backend_name == "rapidocr"
        assert fallback_backend.calls == 1
    finally:
        SingletonMeta._instances.pop(ocr_engine.OCRLoader, None)
