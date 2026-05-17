import hashlib
import sys
from types import SimpleNamespace

import numpy as np
from pathlib import Path


class _LoggerStub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("src.utils.logger", SimpleNamespace(logger=_LoggerStub()))

from src.core.inference.ONNX import ONNXExportMeta, ONNXYoloModelMeta, YoloModelFromONNX
import src.utils.dml_manager as dml_manager_module
from src.utils.dml_manager import DMLManager


class _FakeSession:
    def __init__(self, providers, outputs=None, error: Exception | None = None):
        self._providers = list(providers)
        self._outputs = outputs if outputs is not None else []
        self._error = error
        self.calls = 0

    def run(self, *_args, **_kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._outputs

    def get_providers(self):
        return list(self._providers)


def test_yolo_model_parses_standalone_ultralytics_output():
    model = YoloModelFromONNX.__new__(YoloModelFromONNX)
    model._model_meta = ONNXYoloModelMeta(
        imgsz=(640, 640),
        names={0: "button", 1: "modal"},
        colors={},
    )
    model._export_meta = ONNXExportMeta(version="8.4.38", args={"nms": False}, end2end=False)
    model._output_layout = "standalone"

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    outputs = [
        np.array(
            [[[50, 60, 150, 200, 0.9, 1], [20, 30, 30, 40, 0.2, 0]]],
            dtype=np.float32,
        )
    ]

    result = model._postprocess(
        image,
        outputs,
        conf_threshold=0.5,
        iou_threshold=0.5,
        ratio=1.0,
        dw=0.0,
        dh=0.0,
    )

    assert result.class_ids.tolist() == [1]
    assert np.allclose(result.scores, np.array([0.9], dtype=np.float32))
    assert np.allclose(result.boxes, np.array([[50, 60, 100, 140]], dtype=np.float32))


def test_dml_manager_retries_with_cpu_when_accelerated_runtime_fails(monkeypatch):
    accelerated = _FakeSession(
        ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        error=RuntimeError("CoreMLExecutionProvider failed at runtime"),
    )
    cpu = _FakeSession(["CPUExecutionProvider"], outputs=[np.array([1], dtype=np.int64)])

    DMLManager._session_model_paths.clear()
    DMLManager._session_provider_names.clear()
    DMLManager._session_replacements.clear()
    DMLManager._remember_session(accelerated, "model/base_ui.onnx", accelerated.get_providers())
    monkeypatch.setattr(
        DMLManager,
        "_create_cpu_session",
        classmethod(lambda cls, model_path: cpu),
    )

    outputs = DMLManager.run(accelerated, {"images": np.zeros((1, 3, 8, 8), dtype=np.float32)})
    outputs_retry = DMLManager.run(accelerated, {"images": np.zeros((1, 3, 8, 8), dtype=np.float32)})

    assert outputs[0].tolist() == [1]
    assert outputs_retry[0].tolist() == [1]
    assert accelerated.calls == 1
    assert cpu.calls == 2


def test_dml_manager_prunes_stale_coreml_cache_for_same_model_path(tmp_path):
    cache_root = tmp_path / "onnxruntime"
    model_scope = "abc123def456"
    stale_key = f"{model_scope}-oldcache000001"
    new_key = f"{model_scope}-newcache000001"
    model_path = tmp_path / "base_ui.onnx"
    model_path.write_bytes(b"fake-model")

    stale_dir = cache_root / "coreml-cache" / stale_key
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "compiled.bin").write_bytes(b"old")
    unrelated_dir = cache_root / "coreml-cache" / "othermodel-keepcache01"
    unrelated_dir.mkdir(parents=True, exist_ok=True)

    DMLManager._prune_stale_coreml_cache(cache_root, str(model_path), new_key)

    assert not stale_dir.exists()
    assert unrelated_dir.exists()


def test_dml_manager_prunes_legacy_top_level_coreml_cache_dirs(tmp_path):
    cache_root = tmp_path / "onnxruntime"
    model_path = tmp_path / "base_ui.onnx"
    model_path.write_bytes(b"fake-model")

    legacy_dir = cache_root / "coreml-cache" / "975788359513881677"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    managed_dir = cache_root / "coreml-cache" / "abcdef0123456789abcdef01"
    managed_dir.mkdir(parents=True, exist_ok=True)
    index_dir = cache_root / "coreml-cache-index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "other-model.txt").write_text(managed_dir.name, encoding="utf-8")

    DMLManager._prune_stale_coreml_cache(cache_root, str(model_path), "111111111111111111111111")

    assert not legacy_dir.exists()
    assert managed_dir.exists()


def test_dml_manager_cache_key_changes_when_external_data_changes(tmp_path):
    model_path = tmp_path / "clip_visual.onnx"
    external_data_path = tmp_path / "clip_visual.onnx.data"
    model_path.write_bytes(b"onnx-header")
    external_data_path.write_bytes(b"weights-v1")
    provider_options = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}

    key_before = DMLManager._build_model_cache_key(str(model_path), provider_options)
    external_data_path.write_bytes(b"weights-v2")
    key_after = DMLManager._build_model_cache_key(str(model_path), provider_options)

    assert key_before != key_after


def test_create_dml_session_retries_accelerated_session_after_clearing_coreml_cache(monkeypatch, tmp_path):
    model_path = tmp_path / "base_ui.onnx"
    model_path.write_bytes(b"fake-model")
    cache_dir = tmp_path / "onnxruntime" / "coreml-cache" / "k"
    cache_dir.mkdir(parents=True, exist_ok=True)
    providers = [
        (
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "MLProgram",
                "ModelCacheDirectory": str(cache_dir),
            },
        ),
        "CPUExecutionProvider",
    ]
    calls = {"count": 0, "cleared": 0}

    monkeypatch.setattr(DMLManager, "_build_dim_overrides_for_model", classmethod(lambda cls, _model_path: {}))
    monkeypatch.setattr(DMLManager, "_build_session_options", classmethod(lambda cls, _extra: object()))
    monkeypatch.setattr(DMLManager, "_build_provider_config", classmethod(lambda cls, _model_path, _extra: providers))
    monkeypatch.setattr(DMLManager, "_clear_coreml_cache_dir", staticmethod(lambda _cache_dir: calls.__setitem__("cleared", calls["cleared"] + 1)))

    def _fake_inference_session(_model_path, sess_options=None, providers=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Failed to look up root model")
        return _FakeSession(["CoreMLExecutionProvider", "CPUExecutionProvider"], outputs=[np.array([1], dtype=np.int64)])

    monkeypatch.setattr(dml_manager_module.ort, "InferenceSession", _fake_inference_session)

    session = DMLManager.create_dml_session(str(model_path))

    assert isinstance(session, _FakeSession)
    assert calls["count"] == 2
    assert calls["cleared"] == 1
