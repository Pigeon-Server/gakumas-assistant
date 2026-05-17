from types import SimpleNamespace

import numpy as np

from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.gameplay import exam_wheel as exam_wheel_module


def _build_frame() -> np.ndarray:
    return np.zeros((2340, 1080, 3), dtype=np.uint8)


def _build_compact_frame() -> np.ndarray:
    return np.zeros((120, 240, 3), dtype=np.uint8)


def test_ocr_wheel_area_reads_bonus_from_right_region(monkeypatch):
    monkeypatch.setattr(exam_wheel_module, "_debugger", SimpleNamespace(add_box=lambda *args, **kwargs: None))
    texts = iter(["ダンス701%", "残りターン7"])
    monkeypatch.setattr(exam_wheel_module, "ocr_text", lambda _img: next(texts))

    param_code, bonus_pct, ocr_turns, extra_turns = exam_wheel_module._ocr_wheel_area(
        _build_frame(),
        cx=180,
        cy=200,
        inner_r=70,
    )

    assert param_code == "Da"
    assert bonus_pct == 701
    assert ocr_turns == 7
    assert extra_turns == 0


def test_ocr_wheel_area_strips_turn_prefix_from_bonus(monkeypatch):
    monkeypatch.setattr(exam_wheel_module, "_debugger", SimpleNamespace(add_box=lambda *args, **kwargs: None))
    texts = iter(["ビジュアル9701%", "残りターン9"])
    monkeypatch.setattr(exam_wheel_module, "ocr_text", lambda _img: next(texts))

    param_code, bonus_pct, ocr_turns, extra_turns = exam_wheel_module._ocr_wheel_area(
        _build_frame(),
        cx=180,
        cy=200,
        inner_r=70,
    )

    assert param_code == "Vi"
    assert ocr_turns == 9
    assert bonus_pct == 701
    assert extra_turns == 0


def test_ocr_wheel_area_parses_extra_turns_from_plus_suffix(monkeypatch):
    monkeypatch.setattr(exam_wheel_module, "_debugger", SimpleNamespace(add_box=lambda *args, **kwargs: None))
    texts = iter(["ダンス3053%+3", "残りターン1"])
    monkeypatch.setattr(exam_wheel_module, "ocr_text", lambda _img: next(texts))

    param_code, bonus_pct, ocr_turns, extra_turns = exam_wheel_module._ocr_wheel_area(
        _build_frame(),
        cx=180,
        cy=200,
        inner_r=70,
    )

    assert param_code == "Da"
    assert bonus_pct == 3053
    assert ocr_turns == 1
    assert extra_turns == 3


def test_compact_wheel_fallback_prefers_prefix_turn_over_noisy_left(monkeypatch):
    monkeypatch.setattr(exam_wheel_module, "_debugger", SimpleNamespace(add_box=lambda *args, **kwargs: None))
    # 顺序: right / left / full
    texts = iter(["ビジュアル1745%", "161", "ビジュアル991745%"])
    monkeypatch.setattr(exam_wheel_module, "ocr_text", lambda _img: next(texts))

    result = exam_wheel_module._extract_compact_wheel_text_info(_build_compact_frame())

    assert result is not None
    assert result["current_param"] == "visual"
    assert result["current_bonus_pct"] == 1745
    assert result["remaining_turns"] == 9


def test_compact_wheel_fallback_extracts_turn_from_full_prefix(monkeypatch):
    monkeypatch.setattr(exam_wheel_module, "_debugger", SimpleNamespace(add_box=lambda *args, **kwargs: None))
    # 顺序: right / left / full
    texts = iter(["ダンス3053%+3", "", "ダンス13053%+3"])
    monkeypatch.setattr(exam_wheel_module, "ocr_text", lambda _img: next(texts))

    result = exam_wheel_module._extract_compact_wheel_text_info(_build_compact_frame())

    assert result is not None
    assert result["current_param"] == "dance"
    assert result["current_bonus_pct"] == 3053
    assert result["remaining_turns"] == 4
    assert result["additional_turns"] == 3


def test_compact_wheel_fallback_accepts_turns_above_20(monkeypatch):
    monkeypatch.setattr(exam_wheel_module, "_debugger", SimpleNamespace(add_box=lambda *args, **kwargs: None))
    # 顺序: right / left / full
    texts = iter(["ダンス2517%", "25:2", "ダンス25.2517%"])
    monkeypatch.setattr(exam_wheel_module, "ocr_text", lambda _img: next(texts))

    result = exam_wheel_module._extract_compact_wheel_text_info(_build_compact_frame())

    assert result is not None
    assert result["current_param"] == "dance"
    assert result["current_bonus_pct"] == 2517
    assert result["remaining_turns"] == 25


def test_large_frame_wheel_hint_fallback_parses_double_ring_turns(monkeypatch):
    monkeypatch.setattr(exam_wheel_module, "_debugger", SimpleNamespace(add_box=lambda *args, **kwargs: None))
    # 顺序: center / right / left / full
    texts = iter(["25", "ダンス2517%+3", "1ターン後", "ダンス2517%+3"])
    monkeypatch.setattr(exam_wheel_module, "ocr_text", lambda _img: next(texts))

    result = exam_wheel_module._extract_compact_wheel_text_info(
        _build_frame(),
        allow_large=True,
        wheel_hint=(120, 80, 60, 24),
    )

    assert result is not None
    assert result["current_param"] == "dance"
    assert result["current_bonus_pct"] == 2517
    assert result["additional_turns"] == 3
    assert result["remaining_turns"] == 28


def test_extract_exam_wheel_info_prefers_bonus_anchor_roi(monkeypatch):
    monkeypatch.setattr(exam_wheel_module, "_debugger", SimpleNamespace(add_box=lambda *args, **kwargs: None))
    frame = _build_frame()
    called_shapes = []

    def _fake_find_wheel_circle(crop):
        called_shapes.append(crop.shape[:2])
        return (40, 50, 20, 16)

    monkeypatch.setattr(exam_wheel_module, "_find_wheel_circle", _fake_find_wheel_circle)
    monkeypatch.setattr(
        exam_wheel_module,
        "_detect_segments",
        lambda _hsv, _cx, _cy, _ring_r: [(0, 119, "Da"), (120, 239, "Vi"), (240, 359, "Vo")],
    )
    monkeypatch.setattr(exam_wheel_module, "_find_pointer_angle", lambda *_args, **_kwargs: 10.0)
    monkeypatch.setattr(exam_wheel_module, "_ocr_wheel_area", lambda *_args, **_kwargs: ("Da", 2517, 3, 0))

    class _Results:
        def filter_by_label(self, label):
            if label != ProducerLabels.PC_BONUS_INDICATOR:
                return []
            # 用左上 bonus 锚点裁 ROI，避免全屏误检到排名头像等圆形区域。
            return [SimpleNamespace(x=20, y=20, w=230, h=170, confidence=0.9)]

    info = exam_wheel_module.extract_exam_wheel_info(frame, yolo_results=_Results())

    assert called_shapes
    assert called_shapes[0][0] < frame.shape[0]
    assert called_shapes[0][1] < frame.shape[1]
    assert info is not None
    assert info["current_param"] == "dance"
    assert info["current_bonus_pct"] == 2517
