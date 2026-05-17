import pickle
from types import SimpleNamespace

import numpy as np

from src.utils.clip_tools import CLIPTools


class _DummyEngine:
    def forward(self, image):  # noqa: ARG002
        return np.array([1.0, 0.0], dtype=np.float32)


class _DummyClip(CLIPTools):
    def _save_payload(self, image, features, payload):  # noqa: ARG002
        return payload

    def _load_payload(self, payload_ref):
        return payload_ref


class _MemoryQuery(list):
    def where(self, *_args, **_kwargs):
        return self


class _DeleteQuery:
    def __init__(self, tracker):
        self._tracker = tracker

    def where(self, _condition):
        return self

    def execute(self):
        self._tracker["delete_executed"] += 1
        return self._tracker["delete_count"]


class _FieldStub:
    def in_(self, values):
        return tuple(values)


def test_clip_retrieve_skips_stale_payload(monkeypatch):
    clip = _DummyClip(_DummyEngine(), "dummy_clip")
    stale_memory = SimpleNamespace(
        uuid="stale",
        features=pickle.dumps(np.array([1.0, 0.0], dtype=np.float32)),
        load_payload=lambda: (_ for _ in ()).throw(RuntimeError("payload missing")),
    )
    valid_memory = SimpleNamespace(
        uuid="valid",
        features=pickle.dumps(np.array([0.95, 0.05], dtype=np.float32)),
        load_payload=lambda: "payload-ok",
    )
    tracker = {"delete_executed": 0, "delete_count": 1}

    monkeypatch.setattr(
        "src.utils.clip_tools.CLIPMemory",
        SimpleNamespace(
            clip_name="clip_name",
            uuid=_FieldStub(),
            select=lambda: _MemoryQuery([stale_memory, valid_memory]),
            delete=lambda: _DeleteQuery(tracker),
        ),
    )

    result = clip.retrieve(np.zeros((8, 8, 3), dtype=np.uint8), similarity_threshold=0.5)

    assert result is not None
    assert result.payload == "payload-ok"
    assert result.similarity > 0.5
    assert tracker["delete_executed"] == 1
