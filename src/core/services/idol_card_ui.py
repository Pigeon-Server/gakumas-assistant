from dataclasses import dataclass
from time import sleep, time
from typing import TYPE_CHECKING, Literal, Optional

import cv2
import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.logger import logger
from src.utils.opencv_tools import compute_ssim_score

if TYPE_CHECKING:
    from src.main import AppProcessor

_SELECTED_CARD_X_PADDING_RATIO = 0.08
_SELECTED_CARD_Y_PADDING_RATIO = 0.11
_OCR_HEIGHT_RATIO = 0.40
_SIMILARITY_IMAGE_SIZE = (160, 240)
_STRIP_SIMILARITY_IMAGE_SIZE = (640, 180)
_CHANGED_CARD_SIMILARITY_THRESHOLD = 0.80
_CHANGED_CAROUSEL_SIMILARITY_THRESHOLD = 0.975
_SAME_CARD_SIMILARITY_THRESHOLD = 0.88
_CAROUSEL_STABLE_THRESHOLD = 0.985
_CAROUSEL_TOP_RATIO = 0.74
_CAROUSEL_BOTTOM_RATIO = 0.93
_IDENTITY_REGION_WIDTH_RATIO = 0.42
_IDENTITY_REGION_HEIGHT_RATIO = 0.24
_DEFAULT_CAROUSEL_SWIPE_DISTANCE_RATIO = 0.34
_DEFAULT_CAROUSEL_SWIPE_DURATION = 0.22


@dataclass
class IdolCardCarouselState:
    selected_box: Optional[Yolo_Box]
    selected_image: Optional[np.ndarray]
    strip_image: Optional[np.ndarray]


@dataclass
class IdolCardCarouselWindow:
    offset: int
    x1: int
    y1: int
    x2: int
    y2: int
    image: np.ndarray
    trimmed_image: np.ndarray

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


def _box_width(box: Optional[Yolo_Box]) -> int:
    if box is None:
        return 0
    return max(0, int(box.w - box.x))


def _box_height(box: Optional[Yolo_Box]) -> int:
    if box is None:
        return 0
    return max(0, int(box.h - box.y))


def trim_idol_card_component(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Trim selected/candidate card chrome so CLIP focuses on the card art itself."""
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    x_padding = max(1, int(round(width * _SELECTED_CARD_X_PADDING_RATIO)))
    y_padding = max(1, int(round(height * _SELECTED_CARD_Y_PADDING_RATIO)))
    if width <= x_padding * 2 or height <= y_padding * 2:
        return image.copy()
    return image[y_padding:height - y_padding, x_padding:width - x_padding].copy()


def trim_idol_card_learning_component(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Trim visible card slices for CLIP learning.

    Compared with the raw carousel slice, this removes boundary bleed from
    neighboring cards and the top/bottom chrome that is not useful for CLIP.
    """
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    x_padding = max(2, int(round(width * 0.08)))
    y_padding = max(2, int(round(height * 0.06)))
    if width <= x_padding * 2 or height <= y_padding * 2:
        return image.copy()
    return image[y_padding:height - y_padding, x_padding:width - x_padding].copy()


def get_selected_idol_card_box(results: Optional[Yolo_Results]) -> Optional[Yolo_Box]:
    if not results:
        return None
    selected = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED)
    return selected.first() if selected else None


def get_next_idol_card_candidate_box(results: Optional[Yolo_Results]) -> Optional[Yolo_Box]:
    return get_adjacent_idol_card_candidate_box(results, direction="next")


def get_adjacent_idol_card_candidate_box(
        results: Optional[Yolo_Results],
        direction: Literal["next", "prev"] = "next",
) -> Optional[Yolo_Box]:
    if not results:
        return None
    candidates = list(results.filter_by_label(BaseUILabels.PRODUCT_CARD_CANDIDATE).boxes)
    if not candidates:
        return None

    selected = get_selected_idol_card_box(results)
    if selected is None:
        return candidates[0] if direction == "next" else None

    if direction == "next":
        side_candidates = [box for box in candidates if box.cx > selected.cx]
        if side_candidates:
            return min(side_candidates, key=lambda box: box.cx - selected.cx)
        return None

    side_candidates = [box for box in candidates if box.cx < selected.cx]
    if side_candidates:
        return min(side_candidates, key=lambda box: selected.cx - box.cx)
    return None


def extract_selected_idol_card_image(results: Optional[Yolo_Results]) -> Optional[np.ndarray]:
    selected = get_selected_idol_card_box(results)
    if selected is None:
        return None
    return trim_idol_card_component(selected.frame)


def extract_idol_card_ocr_region(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None
    height = max(1, int(frame.shape[0] * _OCR_HEIGHT_RATIO))
    return frame[:height, :].copy()


def extract_idol_card_identity_region(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    region_height = max(1, int(height * _IDENTITY_REGION_HEIGHT_RATIO))
    region_width = max(1, int(width * _IDENTITY_REGION_WIDTH_RATIO))
    return frame[:region_height, :region_width].copy()


def extract_idol_card_carousel_region(
        frame: Optional[np.ndarray],
        results: Optional[Yolo_Results] = None,
) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    selected = get_selected_idol_card_box(results)
    if selected is None:
        top = int(height * _CAROUSEL_TOP_RATIO)
        bottom = int(height * _CAROUSEL_BOTTOM_RATIO)
        return frame[top:bottom, :].copy()

    selected_height = _box_height(selected)
    top = max(0, int(selected.y - selected_height * 0.08))
    bottom = min(height, int(selected.h + selected_height * 0.04))
    return frame[top:bottom, :width].copy()


def build_idol_card_carousel_state(
        results: Optional[Yolo_Results],
        frame: Optional[np.ndarray] = None,
) -> IdolCardCarouselState:
    base_frame = frame
    if base_frame is None and results is not None:
        base_frame = getattr(results, "frame", None)
    return IdolCardCarouselState(
        selected_box=get_selected_idol_card_box(results),
        selected_image=extract_selected_idol_card_image(results),
        strip_image=extract_idol_card_carousel_region(base_frame, results),
    )


def _normalize_card_image(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    normalized = image
    if normalized.ndim == 3:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    return cv2.resize(normalized, _SIMILARITY_IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def _normalize_strip_image(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    normalized = image
    if normalized.ndim == 3:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    return cv2.resize(normalized, _STRIP_SIMILARITY_IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def compute_selected_idol_card_similarity(
        first: Optional[np.ndarray],
        second: Optional[np.ndarray],
) -> float:
    normalized_first = _normalize_card_image(first)
    normalized_second = _normalize_card_image(second)
    if normalized_first is None or normalized_second is None:
        return 0.0
    return compute_ssim_score(normalized_first, normalized_second)


def compute_idol_card_carousel_similarity(
        first: Optional[np.ndarray],
        second: Optional[np.ndarray],
) -> float:
    normalized_first = _normalize_strip_image(first)
    normalized_second = _normalize_strip_image(second)
    if normalized_first is None or normalized_second is None:
        return 0.0
    return compute_ssim_score(normalized_first, normalized_second)


def _build_carousel_border_profile(frame: np.ndarray, y1: int, y2: int) -> np.ndarray:
    roi = frame[max(0, y1):max(0, y2), :]
    if roi.size == 0:
        return np.zeros((frame.shape[1],), dtype=np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    profile = np.mean(np.abs(sobel), axis=0)
    if profile.size == 0:
        return profile
    return cv2.GaussianBlur(profile.reshape(1, -1), (9, 1), 0).reshape(-1)


def _search_border_peak(
        profile: np.ndarray,
        start_x: int,
        end_x: int,
) -> Optional[int]:
    if profile.size == 0:
        return None
    start_x = max(0, int(start_x))
    end_x = min(int(end_x), profile.shape[0])
    if end_x - start_x < 3:
        return None
    region = profile[start_x:end_x]
    return int(start_x + int(np.argmax(region)))


def _find_profile_peaks(
        profile: np.ndarray,
        percentile: float = 90.0,
        min_distance: int = 20,
) -> list[int]:
    if profile.size == 0:
        return []
    threshold = float(np.percentile(profile, percentile))
    peaks: list[int] = []
    for index in range(1, len(profile) - 1):
        if profile[index] < threshold:
            continue
        if profile[index] < profile[index - 1] or profile[index] < profile[index + 1]:
            continue
        if not peaks or index - peaks[-1] > min_distance:
            peaks.append(index)
        elif profile[index] > profile[peaks[-1]]:
            peaks[-1] = index
    return peaks


def _pick_peak_near(
        peaks: list[int],
        expected: int,
        tolerance: int,
) -> Optional[int]:
    candidates = [peak for peak in peaks if abs(peak - expected) <= tolerance]
    if not candidates:
        return None
    return min(candidates, key=lambda peak: abs(peak - expected))


def _is_valid_carousel_window(
        image: np.ndarray,
        offset: int,
) -> bool:
    if image is None or image.size == 0:
        return False
    if offset == 0:
        return True

    height = image.shape[0]
    band = image[int(height * 0.55):, :]
    if band.size == 0:
        return False

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_score = float(np.mean(np.abs(sobel_x)) + np.mean(np.abs(sobel_y)))
    std_score = float(np.std(gray))
    return edge_score >= 60.0 or std_score >= 25.0


def _refine_carousel_window_vertical_bounds(
        image: np.ndarray,
        offset: int,
) -> tuple[int, int]:
    """Refine a rough carousel crop to the card's visible top/bottom edges."""
    if image is None or image.size == 0:
        return 0, 0

    height, width = image.shape[:2]
    if height < 24 or width < 24:
        return 0, height

    inner_x1 = max(1, int(round(width * 0.10)))
    inner_x2 = min(width, max(inner_x1 + 1, int(round(width * 0.90))))
    band = image[:, inner_x1:inner_x2]
    if band.size == 0:
        return 0, height

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_sobel_y = np.abs(sobel_y)
    mean_profile = np.mean(abs_sobel_y, axis=1)
    wide_profile = np.percentile(abs_sobel_y, 75, axis=1)
    profile = (mean_profile * 0.65 + wide_profile * 0.35).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).reshape(-1)
    if profile.size == 0:
        return 0, height

    top_search_end = max(8, int(height * 0.30))
    top_idx = int(np.argmax(profile[:top_search_end]))
    top_margin = 2 if offset == 0 else 1
    refined_top = max(0, top_idx - top_margin)

    bottom_search_start = max(refined_top + int(height * 0.45), int(height * 0.68))
    bottom_search_start = min(bottom_search_start, height - 2)
    bottom_idx = bottom_search_start + int(np.argmax(profile[bottom_search_start:]))
    refined_bottom = min(height, bottom_idx + 2)

    min_height = int(height * (0.78 if offset == 0 else 0.72))
    if refined_bottom - refined_top < min_height:
        return 0, height
    return refined_top, refined_bottom


def _detect_carousel_card_vertical_bounds(
        frame: np.ndarray,
        selected: Yolo_Box,
) -> tuple[int, int, int, int]:
    """Detect vertical bounds for carousel strip cards and the selected card.

    Returns ``(strip_y1, strip_y2, sel_y1, sel_y2)`` – all in frame
    coordinates. The strip bounds are derived from horizontal edge peaks
    across the selected + adjacent card region. The selected card uses the
    same vertical band so the learning/debug window stays tight to the
    visible card face instead of including the selected-state glow above it.
    """
    selected_width = _box_width(selected)
    selected_height = _box_height(selected)
    fallback = (int(selected.y), int(selected.h), int(selected.y), int(selected.h))
    x1 = max(0, int(selected.x - selected_width * 0.5))
    x2 = min(frame.shape[1], int(selected.w + selected_width * 0.5))
    roi = frame[int(selected.y):int(selected.h), x1:x2]
    if roi.size == 0:
        return fallback

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    profile = np.mean(np.abs(sobel_y), axis=1)
    if profile.size == 0:
        return fallback
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).reshape(-1)
    peaks = _find_profile_peaks(profile, percentile=88.0, min_distance=10)

    tolerance = max(12, int(selected_height * 0.12))

    # --- Strip top: strong horizontal edge near 17% (top of smaller cards) ---
    expected_strip_top = int(selected_height * 0.17)
    strip_top_peak = _pick_peak_near(peaks, expected_strip_top, tolerance)
    strip_top_idx = strip_top_peak if strip_top_peak is not None else expected_strip_top

    # --- Selected card top: keep a small headroom above the strip so the
    # selected card's top edge is not clipped, but avoid the larger
    # selected-state glow area that previously leaked into the crop.
    sel_top_idx = max(0, strip_top_idx - int(selected_height * 0.04))

    # --- Bottom: last detectable peak + small margin for card frame ---
    bottom_peaks = [p for p in peaks if p > selected_height * 0.75]
    if bottom_peaks:
        last_bottom = bottom_peaks[-1]
        bottom_idx = last_bottom + max(2, int(selected_height * 0.01))
    else:
        expected_bottom = int(selected_height * 0.84)
        bot_peak = _pick_peak_near(peaks, expected_bottom, tolerance)
        bottom_idx = (bot_peak if bot_peak is not None else expected_bottom) + max(2, int(selected_height * 0.01))
    bottom_idx = min(bottom_idx, selected_height)

    strip_y1 = int(selected.y) + strip_top_idx
    y2 = int(selected.y) + bottom_idx
    sel_y1 = int(selected.y) + sel_top_idx

    if y2 - strip_y1 < int(selected_height * 0.45):
        return fallback
    return (
        max(0, strip_y1),
        min(frame.shape[0], y2),
        max(0, sel_y1),
        min(frame.shape[0], y2),
    )


def get_idol_card_carousel_window_step(results: Optional[Yolo_Results]) -> int:
    selected = get_selected_idol_card_box(results)
    selected_width = _box_width(selected)
    if selected_width <= 0:
        return 0
    return max(1, selected_width // 2)


def get_idol_card_carousel_windows(
        results: Optional[Yolo_Results],
        max_offset: Optional[int] = None,
) -> list[IdolCardCarouselWindow]:
    # Neighbor-window slicing is temporarily disabled. The current rollout
    # falls back to the selected card only because the heuristic horizontal
    # splitting is not stable enough on live data yet.
    del max_offset
    if not results:
        return []

    selected = get_selected_idol_card_box(results)
    frame = getattr(results, "frame", None)
    if selected is None or frame is None or frame.size == 0:
        return []
    crop = selected.frame.copy() if selected.frame is not None else frame[int(selected.y):int(selected.h), int(selected.x):int(selected.w)].copy()
    if crop.size == 0:
        return []

    trimmed = trim_idol_card_component(crop)
    if trimmed is None or trimmed.size == 0:
        return []

    return [IdolCardCarouselWindow(
        offset=0,
        x1=int(selected.x),
        y1=int(selected.y),
        x2=int(selected.w),
        y2=int(selected.h),
        image=crop,
        trimmed_image=trimmed,
    )]


def get_relative_idol_card_window(
        results: Optional[Yolo_Results],
        offset: int,
) -> Optional[IdolCardCarouselWindow]:
    for window in get_idol_card_carousel_windows(results, max_offset=max(3, abs(offset) + 1)):
        if window.offset == offset:
            return window
    return None


def has_selected_idol_card_changed(
        first: Optional[np.ndarray],
        second: Optional[np.ndarray],
        threshold: float = _CHANGED_CARD_SIMILARITY_THRESHOLD,
) -> bool:
    return compute_selected_idol_card_similarity(first, second) < threshold


def has_idol_card_carousel_changed(
        first: Optional[np.ndarray],
        second: Optional[np.ndarray],
        threshold: float = _CHANGED_CAROUSEL_SIMILARITY_THRESHOLD,
) -> bool:
    return compute_idol_card_carousel_similarity(first, second) < threshold


def is_same_selected_idol_card(
        first: Optional[np.ndarray],
        second: Optional[np.ndarray],
        threshold: float = _SAME_CARD_SIMILARITY_THRESHOLD,
) -> bool:
    return compute_selected_idol_card_similarity(first, second) >= threshold


def wait_for_selected_idol_card(app: "AppProcessor", timeout: float = 5.0) -> Optional[np.ndarray]:
    start_time = time()
    while time() - start_time <= timeout:
        image = extract_selected_idol_card_image(getattr(app, "latest_results", None))
        if image is not None and image.size > 0:
            return image
        sleep(0.1)
    return None


def wait_for_selected_idol_card_change(
        app: "AppProcessor",
        previous_image: Optional[np.ndarray],
        previous_strip: Optional[np.ndarray] = None,
        timeout: float = 3.0,
        stable_count: int = 2,
) -> Optional[np.ndarray]:
    image_unavailable = previous_image is None or previous_image.size == 0
    strip_unavailable = previous_strip is None or previous_strip.size == 0
    if image_unavailable and strip_unavailable:
        return wait_for_selected_idol_card(app, timeout=timeout)

    start_time = time()
    changed_hits = 0
    while time() - start_time <= timeout:
        current_results = getattr(app, "latest_results", None)
        current_image = extract_selected_idol_card_image(current_results)
        current_frame = getattr(current_results, "frame", None)
        if current_frame is None or current_frame.size == 0:
            current_frame = getattr(app, "latest_frame", None)
        current_strip = extract_idol_card_carousel_region(current_frame, current_results)

        if not image_unavailable and (current_image is None or current_image.size == 0):
            sleep(0.1)
            continue
        if not strip_unavailable and (current_strip is None or current_strip.size == 0):
            sleep(0.1)
            continue

        image_changed = True
        if not image_unavailable:
            image_changed = has_selected_idol_card_changed(previous_image, current_image)

        strip_changed = True
        if not strip_unavailable:
            strip_changed = has_idol_card_carousel_changed(previous_strip, current_strip)

        if image_changed and strip_changed:
            changed_hits += 1
            if changed_hits >= stable_count:
                return current_image
        else:
            changed_hits = 0

        sleep(0.1)
    return None


def wait_for_idol_card_carousel_stable(
        app: "AppProcessor",
        threshold: float = _CAROUSEL_STABLE_THRESHOLD,
        stable_count: int = 2,
        timeout: float = 3.0,
) -> bool:
    start_time = time()
    previous_strip = None
    stable_hits = 0

    while time() - start_time <= timeout:
        state = build_idol_card_carousel_state(
            getattr(app, "latest_results", None),
            getattr(app, "latest_frame", None),
        )
        current_strip = state.strip_image
        if current_strip is None or current_strip.size == 0:
            sleep(0.05)
            continue

        if previous_strip is None:
            previous_strip = current_strip.copy()
            sleep(0.05)
            continue

        similarity = compute_idol_card_carousel_similarity(previous_strip, current_strip)
        if similarity >= threshold:
            stable_hits += 1
            if stable_hits >= stable_count:
                return True
        else:
            stable_hits = 0

        previous_strip = current_strip.copy()
        sleep(0.05)

    return False


def get_adjacent_idol_card_click_point(
        results: Optional[Yolo_Results],
        direction: Literal["next", "prev"] = "next",
) -> Optional[tuple[int, int]]:
    if not results:
        return None

    selected = get_selected_idol_card_box(results)
    frame = getattr(results, "frame", None)
    if selected is None or frame is None or frame.size == 0:
        return None
    if get_adjacent_idol_card_candidate_box(results, direction=direction) is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    offset = get_idol_card_carousel_window_step(results)
    if offset <= 0:
        return None

    if direction == "next":
        target_x = int(selected.w + offset)
    else:
        target_x = int(selected.x - offset)

    if target_x < 0 or target_x >= frame_width:
        return None

    target_y = max(0, min(frame_height - 1, int(selected.cy)))
    return target_x, target_y


def get_idol_card_carousel_swipe_points(
        results: Optional[Yolo_Results],
        direction: Literal["next", "prev"] = "next",
        distance_ratio: float = _DEFAULT_CAROUSEL_SWIPE_DISTANCE_RATIO,
) -> Optional[tuple[int, int, int, int]]:
    if not results:
        return None

    frame = getattr(results, "frame", None)
    if frame is None or frame.size == 0:
        return None
    if get_adjacent_idol_card_candidate_box(results, direction=direction) is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        return None

    selected = get_selected_idol_card_box(results)
    selected_width = _box_width(selected)
    selected_height = _box_height(selected)
    center_x = int(selected.cx) if selected is not None else frame_width // 2
    target_y = int(selected.y + selected_height * 0.76) if selected is not None else int(frame_height * 0.84)
    target_y = max(0, min(frame_height - 1, target_y))

    safe_margin = max(40, int(frame_width * 0.08))
    travel = max(int(frame_width * distance_ratio), selected_width)
    travel = min(travel, max(1, frame_width - safe_margin * 2))
    if travel <= 0:
        return None

    half_travel = max(1, travel // 2)
    if direction == "next":
        start_x = center_x + half_travel
        end_x = center_x - half_travel
    else:
        start_x = center_x - half_travel
        end_x = center_x + half_travel

    start_x = max(safe_margin, min(frame_width - safe_margin, start_x))
    end_x = max(safe_margin, min(frame_width - safe_margin, end_x))
    if direction == "next" and start_x <= end_x:
        return None
    if direction == "prev" and start_x >= end_x:
        return None
    return int(start_x), int(target_y), int(end_x), int(target_y)


def swipe_idol_card_carousel(
        app: "AppProcessor",
        direction: Literal["next", "prev"] = "next",
        distance_ratio: float = _DEFAULT_CAROUSEL_SWIPE_DISTANCE_RATIO,
        duration: float = _DEFAULT_CAROUSEL_SWIPE_DURATION,
) -> bool:
    swipe_points = get_idol_card_carousel_swipe_points(
        getattr(app, "latest_results", None),
        direction=direction,
        distance_ratio=distance_ratio,
    )
    if swipe_points is None:
        logger.debug(f"No idol card swipe points for direction={direction}")
        return False

    start_x, start_y, end_x, end_y = swipe_points
    logger.debug(
        f"Advance idol card by swiping {direction} "
        f"from ({start_x}, {start_y}) to ({end_x}, {end_y})"
    )
    app.device.swipe(start_x, start_y, end_x, end_y, duration=duration, offset_y=0)
    return True


def advance_to_adjacent_idol_card(
        app: "AppProcessor",
        direction: Literal["next", "prev"] = "next",
        retries: int = 3,
        timeout: float = 3.0,
        prefer_swipe: bool = False,
        swipe_distance_ratio: float = _DEFAULT_CAROUSEL_SWIPE_DISTANCE_RATIO,
) -> bool:
    for attempt in range(1, retries + 1):
        wait_for_idol_card_carousel_stable(app, stable_count=2, timeout=min(1.2, timeout))

        results = getattr(app, "latest_results", None)
        previous_image = extract_selected_idol_card_image(results)
        previous_strip = extract_idol_card_carousel_region(getattr(results, "frame", None), results)

        action_order = ("swipe",) if prefer_swipe else ("click", "swipe")
        for action_name in action_order:
            if action_name == "click":
                click_point = get_adjacent_idol_card_click_point(results, direction=direction)
                if click_point is None:
                    logger.debug(f"No adjacent idol card point for direction={direction}")
                    continue
                logger.debug(
                    f"Advance idol card by clicking {direction} neighbor "
                    f"({attempt}/{retries}) at {click_point}"
                )
                app.device.click(click_point[0], click_point[1], f"idol-card-{direction}")
            else:
                if not swipe_idol_card_carousel(
                        app,
                        direction=direction,
                        distance_ratio=swipe_distance_ratio,
                ):
                    continue

            if wait_for_selected_idol_card_change(
                    app,
                    previous_image,
                    previous_strip=previous_strip,
                    timeout=timeout,
            ) is not None:
                wait_for_idol_card_carousel_stable(app, stable_count=2, timeout=min(1.5, timeout))
                return True

            sleep(0.15)
    return False


def advance_to_next_idol_card(
        app: "AppProcessor",
        previous_image: Optional[np.ndarray],
        retries: int = 3,
        timeout: float = 3.0,
        prefer_swipe: bool = False,
        swipe_distance_ratio: float = _DEFAULT_CAROUSEL_SWIPE_DISTANCE_RATIO,
) -> bool:
    del previous_image
    return advance_to_adjacent_idol_card(
        app,
        direction="next",
        retries=retries,
        timeout=timeout,
        prefer_swipe=prefer_swipe,
        swipe_distance_ratio=swipe_distance_ratio,
    )


def retreat_to_previous_idol_card(
        app: "AppProcessor",
        retries: int = 3,
        timeout: float = 3.0,
        prefer_swipe: bool = False,
        swipe_distance_ratio: float = _DEFAULT_CAROUSEL_SWIPE_DISTANCE_RATIO,
) -> bool:
    return advance_to_adjacent_idol_card(
        app,
        direction="prev",
        retries=retries,
        timeout=timeout,
        prefer_swipe=prefer_swipe,
        swipe_distance_ratio=swipe_distance_ratio,
    )
