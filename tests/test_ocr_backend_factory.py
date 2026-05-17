import sys
from types import SimpleNamespace


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.constants.ocr.backend import OCRBackendType
from src.core.inference.ocr_backends import factory


class _FakeBackend:
    def __init__(self, name: str):
        self.name = name


def test_resolve_auto_backend_prefers_vision_on_macos():
    assert factory.resolve_ocr_backend_candidates(
        OCRBackendType.AUTO,
        current_platform="Darwin",
    ) == [OCRBackendType.VISION, OCRBackendType.RAPIDOCR]


def test_resolve_auto_backend_prefers_rapidocr_on_non_macos():
    assert factory.resolve_ocr_backend_candidates(
        OCRBackendType.AUTO,
        current_platform="Linux",
    ) == [OCRBackendType.RAPIDOCR]


def test_create_ocr_backend_falls_back_from_vision_to_rapidocr(monkeypatch):
    attempts = []

    def fake_build_backend(name: str):
        attempts.append(name)
        if name == OCRBackendType.VISION:
            raise RuntimeError("vision unavailable")
        return _FakeBackend(name)

    monkeypatch.setattr(factory, "_build_backend", fake_build_backend)

    backend = factory.create_ocr_backend(OCRBackendType.VISION)

    assert backend.name == OCRBackendType.RAPIDOCR
    assert attempts == [OCRBackendType.VISION, OCRBackendType.RAPIDOCR]


def test_get_requested_ocr_backend_prefers_env_override(monkeypatch):
    monkeypatch.setenv("GAKUMAS_OCR_BACKEND", OCRBackendType.VISION)
    monkeypatch.setattr(factory, "_load_configured_ocr_backend", lambda: OCRBackendType.RAPIDOCR)

    assert factory.get_requested_ocr_backend() == OCRBackendType.VISION
