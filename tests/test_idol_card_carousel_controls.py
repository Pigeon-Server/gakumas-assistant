from pathlib import Path
import cv2
import numpy as np
import pytest

import config
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ONNX import YoloModelFromONNX
import src.core.services.idol_card_ui as idol_card_ui
import src.core.tasks.base_ui.learn_idol_card_clip as learn_idol_card_clip
from src.core.services.idol_card_ui import (
    advance_to_adjacent_idol_card,
    compute_idol_card_carousel_similarity,
    extract_selected_idol_card_image,
    extract_idol_card_carousel_region,
    get_adjacent_idol_card_click_point,
    get_idol_card_carousel_swipe_points,
    get_idol_card_carousel_window_step,
    get_idol_card_carousel_windows,
    get_relative_idol_card_window,
    has_selected_idol_card_changed,
    wait_for_idol_card_carousel_stable,
    wait_for_selected_idol_card_change,
)
from src.entity.Yolo import Yolo_Results, Yolo_Box
from src.utils.opencv_tools import compute_ssim_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IDOL_CARD_DIR = PROJECT_ROOT / "tests" / "_artifacts" / "idol_card_learning"
PRODUCE_FLOW_DIR = PROJECT_ROOT / "tests" / "_artifacts" / "produce_flow"
LIVE_CLICK_ADVANCE_BEFORE = IDOL_CARD_DIR / "live_click_advance_before.png"
LIVE_CLICK_ADVANCE_AFTER = IDOL_CARD_DIR / "live_click_advance_after.png"
CURRENT_WINDOWING_BASE = IDOL_CARD_DIR / "current_windowing_base.png"
CAROUSEL_STEP_SAMPLES = sorted(PRODUCE_FLOW_DIR.glob("stepC_idol*.png"))


@pytest.fixture(scope="session")
def yolo_model():
    return YoloModelFromONNX(config.model_config["BASE_UI"])


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None and image.size > 0, f"Failed to load {path}"
    return image


def _detect(model, frame: np.ndarray) -> Yolo_Results:
    raw = model(frame, conf_threshold=0.5, iou_threshold=0.5)
    return Yolo_Results(raw, frame)


def _jpeg_compress(frame: np.ndarray, quality: int) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _resize(frame: np.ndarray, scale: float) -> np.ndarray:
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, dsize=None, fx=scale, fy=scale, interpolation=interpolation)


def _has_selected_window(results: Yolo_Results) -> bool:
    offsets = [window.offset for window in get_idol_card_carousel_windows(results, max_offset=8)]
    return offsets == [0]


def _next_click_is_valid(results: Yolo_Results) -> bool:
    point = get_adjacent_idol_card_click_point(results, direction="next")
    selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
    frame = getattr(results, "frame", None)
    if point is None or selected is None or frame is None:
        return False
    return (
        selected.w <= point[0] < frame.shape[1]
        and 0 <= point[1] < frame.shape[0]
    )


class _FrameSequenceApp:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = [frame.copy() for frame in frames]
        self._index = 0

    @property
    def latest_frame(self):
        index = min(self._index, len(self._frames) - 1)
        frame = self._frames[index]
        self._index += 1
        return frame

    @property
    def latest_results(self):
        return None


class _StaticResultsApp:
    def __init__(self, results: Yolo_Results):
        self._results = results

    @property
    def latest_frame(self):
        return self._results.frame

    @property
    def latest_results(self):
        return self._results


class _RecordingDevice:
    def __init__(self):
        self.actions: list[tuple] = []

    def click(self, x, y, el_label=""):
        self.actions.append(("click", x, y, el_label))

    def swipe(
            self,
            start_x,
            start_y,
            end_x,
            end_y,
            duration=0.8,
            offset_x=10,
            offset_y=10,
            safe_margin=50,
    ):
        self.actions.append(("swipe", start_x, start_y, end_x, end_y, duration, offset_x, offset_y, safe_margin))


class _StaticResultsDeviceApp(_StaticResultsApp):
    def __init__(self, results: Yolo_Results):
        super().__init__(results)
        self.device = _RecordingDevice()


def _build_synthetic_carousel_frame(width: int = 480, height: int = 900) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x_gradient = np.linspace(0, 255, width, dtype=np.uint8)
    y_gradient = np.linspace(0, 255, height, dtype=np.uint8)
    frame[..., 0] = np.tile(x_gradient, (height, 1))
    frame[..., 1] = np.tile(y_gradient[:, None], (1, width))
    frame[..., 2] = np.tile(np.flip(x_gradient), (height, 1))

    strip_y1 = int(height * 0.72)
    strip = frame[strip_y1:, :]
    stripe_pattern = ((np.arange(width) // 12) % 2) * 90
    strip[..., 1] = np.tile(stripe_pattern.astype(np.uint8), (strip.shape[0], 1))
    return frame


def _make_selected_results(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Yolo_Results:
    box = Yolo_Box(x1, y1, x2, y2, BaseUILabels.PRODUCT_CARD_SELECTED, frame[y1:y2, x1:x2])
    results = Yolo_Results.from_boxes([box])
    results.frame = frame
    return results


def _make_carousel_results(
        frame: np.ndarray,
        selected_bounds: tuple[int, int, int, int],
        candidate_bounds: list[tuple[int, int, int, int]],
) -> Yolo_Results:
    sx1, sy1, sx2, sy2 = selected_bounds
    boxes = [
        Yolo_Box(
            sx1,
            sy1,
            sx2,
            sy2,
            BaseUILabels.PRODUCT_CARD_SELECTED,
            frame[sy1:sy2, sx1:sx2],
        )
    ]
    for x1, y1, x2, y2 in candidate_bounds:
        boxes.append(
            Yolo_Box(
                x1,
                y1,
                x2,
                y2,
                BaseUILabels.PRODUCT_CARD_CANDIDATE,
                frame[y1:y2, x1:x2],
            )
        )
    results = Yolo_Results.from_boxes(boxes)
    results.frame = frame
    return results


class TestAdjacentClickPoints:
    def test_next_click_point_uses_half_selected_width_offset(self, yolo_model):
        results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_BEFORE))
        selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
        assert selected is not None

        point = get_adjacent_idol_card_click_point(results, direction="next")
        assert point is not None
        selected_width = selected.w - selected.x
        assert point[0] == selected.w + selected_width // 2
        assert point[1] == selected.cy

    def test_prev_click_point_uses_half_selected_width_offset(self, yolo_model):
        results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_AFTER))
        selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
        assert selected is not None

        point = get_adjacent_idol_card_click_point(results, direction="prev")
        assert point is not None
        selected_width = selected.w - selected.x
        assert point[0] == selected.x - selected_width // 2
        assert point[1] == selected.cy

    def test_adjacent_click_point_returns_none_when_selected_hits_view_edge(self):
        frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        selected = Yolo_Box(20, 1740, 260, 2160, BaseUILabels.PRODUCT_CARD_SELECTED, frame[1740:2160, 20:260])
        results = Yolo_Results.from_boxes([selected])
        results.frame = frame

        assert get_adjacent_idol_card_click_point(results, direction="prev") is None

        selected = Yolo_Box(860, 1740, 1060, 2160, BaseUILabels.PRODUCT_CARD_SELECTED, frame[1740:2160, 860:1060])
        results = Yolo_Results.from_boxes([selected])
        results.frame = frame

        assert get_adjacent_idol_card_click_point(results, direction="next") is None

    def test_prev_click_point_returns_none_when_no_left_candidate(self, yolo_model):
        results = _detect(yolo_model, _load_image(CURRENT_WINDOWING_BASE))
        assert get_adjacent_idol_card_click_point(results, direction="prev") is None

    def test_next_click_point_returns_none_when_no_right_candidate(self):
        frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        results = _make_carousel_results(
            frame,
            selected_bounds=(260, 1740, 520, 2160),
            candidate_bounds=[(40, 1800, 250, 2110)],
        )

        assert get_adjacent_idol_card_click_point(results, direction="next") is None


class TestCarouselSwipePoints:
    def test_next_swipe_points_cross_selected_card(self, yolo_model):
        results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_BEFORE))
        selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
        assert selected is not None

        swipe_points = get_idol_card_carousel_swipe_points(results, direction="next")
        assert swipe_points is not None
        start_x, start_y, end_x, end_y = swipe_points

        assert start_x > end_x
        assert end_x < selected.cx < start_x
        assert selected.y <= start_y <= selected.h
        assert start_y == end_y

    def test_prev_swipe_points_cross_selected_card(self, yolo_model):
        results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_AFTER))
        selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
        assert selected is not None

        swipe_points = get_idol_card_carousel_swipe_points(results, direction="prev")
        assert swipe_points is not None
        start_x, start_y, end_x, end_y = swipe_points

        assert start_x < end_x
        assert start_x < selected.cx < end_x
        assert selected.y <= start_y <= selected.h
        assert start_y == end_y

    def test_prev_swipe_points_return_none_when_no_left_candidate(self, yolo_model):
        results = _detect(yolo_model, _load_image(CURRENT_WINDOWING_BASE))
        assert get_idol_card_carousel_swipe_points(results, direction="prev") is None

    def test_next_swipe_points_return_none_when_no_right_candidate(self):
        frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        results = _make_carousel_results(
            frame,
            selected_bounds=(260, 1740, 520, 2160),
            candidate_bounds=[(40, 1800, 250, 2110)],
        )

        assert get_idol_card_carousel_swipe_points(results, direction="next") is None


class TestCarouselStripStability:
    def test_carousel_similarity_stays_high_when_only_main_art_changes(self):
        base = _load_image(LIVE_CLICK_ADVANCE_BEFORE)
        variant = base.copy()
        cutoff = int(base.shape[0] * 0.68)
        variant[:cutoff, :] = np.random.RandomState(42).randint(
            0, 255, variant[:cutoff, :].shape, dtype=np.uint8
        )
        strip_a = extract_idol_card_carousel_region(base)
        strip_b = extract_idol_card_carousel_region(variant)
        assert strip_a is not None and strip_b is not None

        full_frame_similarity = compute_ssim_score(base, variant)
        strip_similarity = compute_idol_card_carousel_similarity(strip_a, strip_b)

        assert full_frame_similarity < 0.8
        assert strip_similarity > 0.98

    def test_carousel_similarity_drops_when_selected_card_changes(self):
        before = _load_image(LIVE_CLICK_ADVANCE_BEFORE)
        after = _load_image(LIVE_CLICK_ADVANCE_AFTER)
        similarity = compute_idol_card_carousel_similarity(
            extract_idol_card_carousel_region(before),
            extract_idol_card_carousel_region(after),
        )
        assert similarity < 0.95

    def test_wait_for_carousel_stable_ignores_main_art_animation(self):
        base = _load_image(LIVE_CLICK_ADVANCE_BEFORE)
        frames = []
        for seed in (1, 2, 3, 4):
            frame = base.copy()
            cutoff = int(base.shape[0] * 0.68)
            frame[:cutoff, :] = np.random.RandomState(seed).randint(
                0, 255, frame[:cutoff, :].shape, dtype=np.uint8
            )
            frames.append(frame)

        app = _FrameSequenceApp(frames)
        assert wait_for_idol_card_carousel_stable(app, threshold=0.98, stable_count=2, timeout=1.0)


class TestSelectedChangeDetection:
    def test_wait_for_selected_change_ignores_same_strip_box_jitter(self):
        frame = _build_synthetic_carousel_frame()
        previous_results = _make_selected_results(frame, 140, 560, 300, 820)
        current_results = _make_selected_results(frame, 176, 560, 336, 820)

        previous_image = extract_selected_idol_card_image(previous_results)
        current_image = extract_selected_idol_card_image(current_results)
        previous_strip = extract_idol_card_carousel_region(frame, previous_results)

        assert has_selected_idol_card_changed(previous_image, current_image)

        app = _StaticResultsApp(current_results)
        assert wait_for_selected_idol_card_change(
            app,
            previous_image,
            previous_strip=previous_strip,
            timeout=0.25,
            stable_count=1,
        ) is None

    def test_wait_for_selected_change_requires_strip_shift(self):
        previous_frame = _build_synthetic_carousel_frame()
        current_frame = np.roll(previous_frame.copy(), shift=72, axis=1)
        previous_results = _make_selected_results(previous_frame, 140, 560, 300, 820)
        current_results = _make_selected_results(current_frame, 176, 560, 336, 820)

        previous_image = extract_selected_idol_card_image(previous_results)
        current_image = extract_selected_idol_card_image(current_results)
        previous_strip = extract_idol_card_carousel_region(previous_frame, previous_results)

        assert has_selected_idol_card_changed(previous_image, current_image)

        app = _StaticResultsApp(current_results)
        changed = wait_for_selected_idol_card_change(
            app,
            previous_image,
            previous_strip=previous_strip,
            timeout=0.25,
            stable_count=1,
        )
        assert changed is not None and changed.size > 0


class TestCarouselWindows:
    def test_window_step_is_half_selected_width(self, yolo_model):
        results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_BEFORE))
        selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
        assert selected is not None
        assert get_idol_card_carousel_window_step(results) == (selected.w - selected.x) // 2

    def test_windows_include_selected_and_immediate_neighbors(self, yolo_model):
        results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_BEFORE))
        windows = get_idol_card_carousel_windows(results, max_offset=2)
        offsets = [window.offset for window in windows]
        assert offsets == sorted(offsets)
        assert offsets == [0]

    def test_next_click_point_stays_to_the_right_of_selected_box(self, yolo_model):
        results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_BEFORE))
        point = get_adjacent_idol_card_click_point(results, direction="next")
        selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
        assert point is not None
        assert selected is not None
        assert point[0] >= selected.w
        assert point[1] == selected.cy

    def test_current_sample_has_no_negative_offsets(self, yolo_model):
        results = _detect(yolo_model, _load_image(CURRENT_WINDOWING_BASE))
        offsets = [window.offset for window in get_idol_card_carousel_windows(results, max_offset=5)]
        assert offsets
        assert min(offsets) == 0

    def test_current_sample_returns_selected_only_window(self, yolo_model):
        results = _detect(yolo_model, _load_image(CURRENT_WINDOWING_BASE))
        offsets = [window.offset for window in get_idol_card_carousel_windows(results, max_offset=8)]
        assert offsets == [0]

    def test_selected_learning_window_tightens_visible_selected_crop(self, yolo_model):
        results = _detect(yolo_model, _load_image(CURRENT_WINDOWING_BASE))
        selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
        assert selected is not None
        selected_window = get_relative_idol_card_window(results, 0)
        assert selected_window is not None
        assert get_relative_idol_card_window(results, 1) is None
        assert selected_window.y1 == selected.y
        assert selected_window.y2 == selected.h
        assert selected_window.image.shape[1] == (selected.w - selected.x)
        assert selected_window.image.shape[0] == (selected.h - selected.y)
        assert selected_window.trimmed_image.shape[0] < selected_window.image.shape[0]
        assert selected_window.trimmed_image.shape[1] < selected_window.image.shape[1]

    @pytest.mark.parametrize("quality", [95, 80, 60, 40])
    def test_current_sample_selected_window_survives_jpeg_noise(self, yolo_model, quality: int):
        frame = _jpeg_compress(_load_image(CURRENT_WINDOWING_BASE), quality)
        results = _detect(yolo_model, frame)
        offsets = [window.offset for window in get_idol_card_carousel_windows(results, max_offset=8)]
        assert offsets == [0], f"Q{quality}: offsets={offsets}"

    @pytest.mark.parametrize("path", [LIVE_CLICK_ADVANCE_BEFORE, LIVE_CLICK_ADVANCE_AFTER], ids=lambda p: p.name)
    @pytest.mark.parametrize("quality", [95, 80, 60, 40])
    def test_next_click_point_stays_valid_under_jpeg_noise(self, yolo_model, path: Path, quality: int):
        frame = _jpeg_compress(_load_image(path), quality)
        results = _detect(yolo_model, frame)
        assert _next_click_is_valid(results), f"{path.name} Q{quality}: invalid next click point"

    @pytest.mark.parametrize("path", CAROUSEL_STEP_SAMPLES, ids=lambda p: p.name)
    @pytest.mark.parametrize("scale", [0.85, 1.0, 1.15])
    def test_step_samples_keep_selected_window_across_scaled_resolutions(
            self,
            yolo_model,
            path: Path,
            scale: float,
    ):
        frame = _resize(_load_image(path), scale)
        results = _detect(yolo_model, frame)
        assert _has_selected_window(results), f"{path.name} scale={scale}: missing selected window"
        assert _next_click_is_valid(results), f"{path.name} scale={scale}: invalid next click point"

    @pytest.mark.parametrize("quality", [95, 80, 60, 40])
    def test_step_samples_keep_selected_window_under_jpeg_noise(self, yolo_model, quality: int):
        failures = []
        for path in CAROUSEL_STEP_SAMPLES:
            frame = _jpeg_compress(_load_image(path), quality)
            results = _detect(yolo_model, frame)
            if not _has_selected_window(results):
                failures.append(f"{path.name} @Q{quality}: missing selected window")
                continue
            if not _next_click_is_valid(results):
                failures.append(f"{path.name} @Q{quality}: invalid next click point")
        assert not failures, "JPEG carousel failures:\n" + "\n".join(failures)

    def test_step_samples_recover_after_clean_retry_when_heavy_jpeg_noise_breaks_selected_window(self, yolo_model):
        failures = []
        for path in CAROUSEL_STEP_SAMPLES:
            clean = _load_image(path)
            noisy = _jpeg_compress(clean, 20)
            noisy_results = _detect(yolo_model, noisy)
            clean_results = _detect(yolo_model, clean)

            noisy_ok = _has_selected_window(noisy_results) and _next_click_is_valid(noisy_results)
            clean_ok = _has_selected_window(clean_results) and _next_click_is_valid(clean_results)
            if not (noisy_ok or clean_ok):
                failures.append(f"{path.name}: heavy JPEG failed and clean retry did not recover")

        assert not failures, "Retry recovery failures:\n" + "\n".join(failures)


class TestCarouselAdvanceFallback:
    def test_advance_falls_back_to_swipe_when_click_does_not_confirm_change(self, monkeypatch, yolo_model):
        results = _detect(yolo_model, _load_image(LIVE_CLICK_ADVANCE_BEFORE))
        app = _StaticResultsDeviceApp(results)
        wait_results = [None, extract_selected_idol_card_image(results)]

        monkeypatch.setattr(idol_card_ui, "wait_for_idol_card_carousel_stable", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(
            idol_card_ui,
            "wait_for_selected_idol_card_change",
            lambda *_args, **_kwargs: wait_results.pop(0),
        )

        assert advance_to_adjacent_idol_card(app, direction="next", retries=1, timeout=0.5)
        assert [action[0] for action in app.device.actions] == ["click", "swipe"]


class TestLearnIdolCardClipRewind:
    def test_rewind_to_head_prefers_wide_swipes(self, monkeypatch):
        calls: list[dict[str, object]] = []

        monkeypatch.setattr(learn_idol_card_clip, "wait_for_idol_card_carousel_stable", lambda *_args, **_kwargs: True)

        def _fake_retreat(app, retries=3, timeout=3.0, prefer_swipe=False, swipe_distance_ratio=0.0):
            calls.append({
                "app": app,
                "retries": retries,
                "timeout": timeout,
                "prefer_swipe": prefer_swipe,
                "swipe_distance_ratio": swipe_distance_ratio,
            })
            return len(calls) < 3

        monkeypatch.setattr(learn_idol_card_clip, "retreat_to_previous_idol_card", _fake_retreat)

        rewound = learn_idol_card_clip._rewind_to_head(object(), max_steps=10)

        assert rewound == 2
        assert calls
        assert all(call["prefer_swipe"] is True for call in calls)
        assert all(call["swipe_distance_ratio"] >= 0.6 for call in calls)
