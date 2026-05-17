import re
from collections import Counter
from dataclasses import dataclass, field
from time import sleep
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from rapidfuzz import fuzz

from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCRService
from src.core.services.idol_card_ui import (
    advance_to_adjacent_idol_card,
    extract_idol_card_identity_region,
    extract_idol_card_ocr_region,
    get_idol_card_carousel_windows,
    retreat_to_previous_idol_card,
    wait_for_idol_card_carousel_stable,
)
from src.entity.Game.Components.Button import ButtonList
from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils
from src.utils.logger import logger
from src.utils.opencv_tools import check_frame_change, compute_ssim_score
from src.utils.string_tools import MatchConfig, fullwidth_to_halfwidth, normalize_ocr_jp, string_match
from src.utils.ui_message_tools import UIMessage

if TYPE_CHECKING:
    from src.entity.Game.Database.IdolCard import IdolCard
    from src.main import AppProcessor

ocr_service = OCRService()
message_tools = UIMessage()
idol_card_db = GakumasDatabase_IdolCardDataUtils()

_MAX_REWIND_STEPS = 200
_MAX_FORWARD_STEPS = 400
_TRACK_DUPLICATE_THRESHOLD = 0.995
_IDOL_LIST_BUTTON_MATCH = MatchConfig(use_fuzz=True, fuzz_threshold=70, use_contains=True)
_IDOL_LIST_MAX_SCROLLS = 30
_IDOL_LIST_GRID_REGION = (0.30, 0.88, 0.02, 0.98)
_REWIND_SWIPE_DISTANCE_RATIO = 0.62
_IDOL_CARD_MATCH_THRESHOLD = 72.0
_IDOL_CARD_CHARACTER_MATCH_THRESHOLD = 88.0
_IDOL_CARD_MATCH_MARGIN = 5.0
_IDOL_CARD_IGNORED_OCR_TEXTS = {"n", "r", "sr", "ssr", "ur", "sp"}


@dataclass
class _IdolListThumbnailBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


@dataclass
class _CarouselTrack:
    temp_id: int
    resolved_card: Optional["IdolCard"] = None
    images: list[np.ndarray] = field(default_factory=list)


def _normalize_track_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.resize(image, (160, 96)))


def _append_track_image(track: _CarouselTrack, image: Optional[np.ndarray]) -> bool:
    if image is None or image.size == 0:
        return False
    candidate = image.copy()
    if track.images:
        previous = _normalize_track_image(track.images[-1])
        current = _normalize_track_image(candidate)
        if compute_ssim_score(previous, current) >= _TRACK_DUPLICATE_THRESHOLD:
            return False
    track.images.append(candidate)
    return True


def _finalize_track(app: "AppProcessor", track: _CarouselTrack) -> int:
    if track.resolved_card is None or not track.images:
        return 0
    learned = 0
    for image in track.images:
        if app.clip_manager.idol_card_clip.add_variant_to_memory(image, track.resolved_card):
            learned += 1
    track.images.clear()
    return learned


def _normalize_idol_card_lookup_text(text: str) -> str:
    normalized = normalize_ocr_jp(fullwidth_to_halfwidth(text or "").strip()).lower()
    return re.sub(r"[^0-9a-zぁ-ゖァ-ヴー一-龯々ヶ]+", "", normalized)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _score_lookup_text_match(text: str, candidate: str) -> float:
    if not text or not candidate:
        return 0.0
    if text == candidate:
        return 100.0

    shorter, longer = (text, candidate) if len(text) <= len(candidate) else (candidate, text)
    if len(shorter) >= 2 and shorter in longer:
        coverage = len(shorter) / max(1, len(longer))
        if coverage >= 0.55:
            return 90.0 + coverage * 8.0

    return float(fuzz.ratio(text, candidate))


def _get_idol_card_name_variants(card: "IdolCard") -> list[str]:
    names = [card.name]
    localization = getattr(card, "localization", None)
    if localization is not None and getattr(localization, "name", None):
        names.append(localization.name)
    return _dedupe_preserve_order(
        normalized
        for normalized in (_normalize_idol_card_lookup_text(name) for name in names if name)
        if normalized
    )


def _get_idol_card_character_variants(card: "IdolCard") -> list[str]:
    character = getattr(card, "characterCls", None)
    if character is None:
        return []

    names = [f"{character.lastName}{character.firstName}"]
    localization = getattr(character, "localization", None)
    if localization is not None:
        localized_name = f"{getattr(localization, 'lastName', '')}{getattr(localization, 'firstName', '')}"
        if localized_name.strip():
            names.append(localized_name)

    return _dedupe_preserve_order(
        normalized
        for normalized in (_normalize_idol_card_lookup_text(name) for name in names if name)
        if normalized
    )


def _resolve_idol_card_from_texts(texts: list[str]) -> Optional["IdolCard"]:
    normalized_texts = _dedupe_preserve_order(
        normalized
        for normalized in (_normalize_idol_card_lookup_text(text) for text in texts)
        if len(normalized) >= 2 and normalized not in _IDOL_CARD_IGNORED_OCR_TEXTS
    )
    if not normalized_texts:
        return None

    scored_cards: list[tuple[float, float, float, "IdolCard"]] = []
    for card in idol_card_db.get_all_item():
        name_variants = _get_idol_card_name_variants(card)
        if not name_variants:
            continue

        title_score = max(
            _score_lookup_text_match(text, name)
            for text in normalized_texts
            for name in name_variants
        )
        if title_score < _IDOL_CARD_MATCH_THRESHOLD:
            continue

        character_variants = _get_idol_card_character_variants(card)
        character_score = 0.0
        if character_variants:
            character_score = max(
                _score_lookup_text_match(text, name)
                for text in normalized_texts
                for name in character_variants
            )

        total_score = title_score
        if character_score >= _IDOL_CARD_CHARACTER_MATCH_THRESHOLD:
            total_score += 20.0
        elif character_score > 0:
            total_score += character_score * 0.1

        scored_cards.append((total_score, title_score, character_score, card))

    if not scored_cards:
        return None

    scored_cards.sort(key=lambda item: (item[0], item[1], item[2], item[3].id), reverse=True)
    best_total, _best_title, _best_character, best_card = scored_cards[0]
    if len(scored_cards) == 1:
        return best_card

    runner_up_total = next(
        (score[0] for score in scored_cards[1:] if score[3].id != best_card.id),
        None,
    )
    if runner_up_total is not None and best_total - runner_up_total < _IDOL_CARD_MATCH_MARGIN:
        return None
    return best_card


def _collect_region_ocr_texts(ocr_result) -> list[str]:
    raw_results = sorted(
        [item for item in getattr(ocr_result, "results", []) if item.text.strip()],
        key=lambda item: (item.cy, item.x),
    )
    texts = [item.text.strip() for item in raw_results]
    if not raw_results:
        return texts

    merged_results = ocr_result.__class__(raw_results).auto_merge_lines(cy_range=10, width_gap=40)
    texts.extend(item.text.strip() for item in getattr(merged_results, "results", []) if item.text.strip())
    return _dedupe_preserve_order(texts)


def _ocr_match_current_idol_card(app: "AppProcessor") -> tuple[Optional["IdolCard"], list[str]]:
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return None, []

    texts: list[str] = []
    for region in (
            extract_idol_card_identity_region(frame),
            extract_idol_card_ocr_region(frame),
    ):
        if region is None or region.size == 0:
            continue

        ocr_result = ocr_service.ocr(region)
        region_texts = _collect_region_ocr_texts(ocr_result)
        texts.extend(region_texts)
        if card := _resolve_idol_card_from_texts(texts):
            return card, texts

    return _resolve_idol_card_from_texts(texts), texts


_GRID_TOP_SAME_SIDE_X_RATIO = 0.15


def _ocr_match_grid_selected_card(app: "AppProcessor") -> tuple[Optional["IdolCard"], list[str]]:
    """网格视图专用 OCR：取网格上方区域最顶部同侧两行（角色名 + 偶像卡名）。"""
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return None, []

    h, w = frame.shape[:2]
    grid_top = max(1, int(h * _IDOL_LIST_GRID_REGION[0]))
    top_region = frame[:grid_top, :].copy()

    ocr_result = ocr_service.ocr(top_region)
    raw_results = sorted(
        [item for item in getattr(ocr_result, "results", []) if item.text.strip()],
        key=lambda item: (item.cy, item.x),
    )
    if not raw_results:
        return None, []

    first = raw_results[0]
    texts = [first.text.strip()]
    x_threshold = w * _GRID_TOP_SAME_SIDE_X_RATIO
    for item in raw_results[1:]:
        if abs(item.x - first.x) < x_threshold:
            texts.append(item.text.strip())
            break

    card = _resolve_idol_card_from_texts(texts)
    if card is not None:
        return card, texts

    all_texts = _dedupe_preserve_order(
        [item.text.strip() for item in raw_results]
    )
    return _resolve_idol_card_from_texts(all_texts), all_texts


def _try_clip_identify(app: "AppProcessor", card_image: np.ndarray) -> Optional["IdolCard"]:
    try:
        return app.clip_manager.idol_card_clip.retrieve(card_image)
    except Exception as exc:
        logger.debug(f"IdolCard CLIP identify failed: {exc}")
        return None


def _resolve_current_selected_idol_card(
        app: "AppProcessor",
        selected_image: Optional[np.ndarray],
) -> tuple[Optional["IdolCard"], list[str]]:
    clip_card = None
    if selected_image is not None and selected_image.size > 0:
        clip_card = _try_clip_identify(app, selected_image)
    ocr_card, texts = _ocr_match_current_idol_card(app)

    if ocr_card is None:
        return clip_card, texts

    if clip_card is not None and clip_card.id != ocr_card.id:
        logger.warning(
            f"IdolCard CLIP mismatch: clip={clip_card.id}, ocr={ocr_card.id}, texts={texts}"
        )
    return ocr_card, texts


def _rewind_to_head(app: "AppProcessor", max_steps: int = _MAX_REWIND_STEPS) -> int:
    rewound = 0
    for _ in range(max_steps):
        wait_for_idol_card_carousel_stable(app, stable_count=2, timeout=2.0)
        if not retreat_to_previous_idol_card(
                app,
                retries=1,
                timeout=1.8,
                prefer_swipe=True,
                swipe_distance_ratio=_REWIND_SWIPE_DISTANCE_RATIO,
        ):
            return rewound
        rewound += 1
    logger.warning(f"Idol card rewind hit max_steps={max_steps}")
    return rewound


def _build_visible_window_map(app: "AppProcessor") -> dict[int, np.ndarray]:
    window_map: dict[int, np.ndarray] = {}
    for window in get_idol_card_carousel_windows(getattr(app, "latest_results", None), max_offset=8):
        window_map[window.offset] = window.trimmed_image.copy()
    return window_map


def _sync_tracks_with_visible_windows(
        tracks_by_slot: dict[int, _CarouselTrack],
        visible_windows: dict[int, np.ndarray],
        next_track_id: int,
) -> int:
    for offset in sorted(visible_windows):
        if offset in tracks_by_slot:
            _append_track_image(tracks_by_slot[offset], visible_windows[offset])
            continue
        if offset < 0:
            continue
        if offset not in tracks_by_slot:
            tracks_by_slot[offset] = _CarouselTrack(temp_id=next_track_id)
            next_track_id += 1
        _append_track_image(tracks_by_slot[offset], visible_windows[offset])
    return next_track_id


def _finalize_invisible_tracks(
        app: "AppProcessor",
        tracks_by_slot: dict[int, _CarouselTrack],
        visible_offsets: set[int],
        pending_tracks: list[_CarouselTrack],
) -> tuple[dict[int, _CarouselTrack], int]:
    learned_images = 0
    kept: dict[int, _CarouselTrack] = {}
    for offset, track in tracks_by_slot.items():
        if offset in visible_offsets:
            kept[offset] = track
            continue
        if track.resolved_card is not None:
            learned_images += _finalize_track(app, track)
        else:
            pending_tracks.append(track)
            logger.debug(f"Queue unresolved idol card track tmp-{track.temp_id} for late CLIP resolve")
    return kept, learned_images


def _late_resolve_track_with_clip(app: "AppProcessor", track: _CarouselTrack) -> Optional["IdolCard"]:
    votes: Counter[str] = Counter()
    first_card_by_id: dict[str, "IdolCard"] = {}
    for image in track.images:
        card = _try_clip_identify(app, image)
        if card is None:
            continue
        votes[card.id] += 1
        first_card_by_id.setdefault(card.id, card)

    if not votes:
        return None
    best_id, _best_votes = votes.most_common(1)[0]
    return first_card_by_id.get(best_id)


def _resolve_pending_tracks(app: "AppProcessor", tracks: list[_CarouselTrack]) -> tuple[int, int, int]:
    learned_images = 0
    resolved_tracks = 0
    dropped_tracks = 0
    for track in tracks:
        if track.resolved_card is None:
            track.resolved_card = _late_resolve_track_with_clip(app, track)
        if track.resolved_card is None:
            dropped_tracks += 1
            logger.warning(f"Dropping unresolved idol card track tmp-{track.temp_id}")
            continue
        resolved_tracks += 1
        learned_images += _finalize_track(app, track)
        logger.success(
            f"[IdolCard CLIP] Late-resolved tmp-{track.temp_id} -> "
            f"{track.resolved_card.id} ({track.resolved_card.name})"
        )
    return learned_images, resolved_tracks, dropped_tracks


def _shift_tracks_after_next(
        tracks_by_slot: dict[int, _CarouselTrack],
) -> dict[int, _CarouselTrack]:
    shifted: dict[int, _CarouselTrack] = {}
    for offset, track in tracks_by_slot.items():
        shifted[offset - 1] = track
    return shifted


def _cluster_positions(values: list[int], tolerance: int) -> list[int]:
    if not values:
        return []
    sorted_values = sorted(values)
    clusters: list[list[int]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    centers = [int(round(float(np.median(cluster)))) for cluster in clusters]
    return sorted(set(centers))


def _extract_idol_list_grid_region(frame: Optional[np.ndarray]) -> np.ndarray:
    if frame is None or frame.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    height, width = frame.shape[:2]
    top_ratio, bottom_ratio, left_ratio, right_ratio = _IDOL_LIST_GRID_REGION
    y1 = max(0, int(height * top_ratio))
    y2 = min(height, int(height * bottom_ratio))
    x1 = max(0, int(width * left_ratio))
    x2 = min(width, int(width * right_ratio))
    return frame[y1:y2, x1:x2].copy()


def _detect_idol_list_thumbnail_boxes(frame: np.ndarray) -> list[_IdolListThumbnailBox]:
    if frame is None or frame.size == 0:
        return []

    grid = _extract_idol_list_grid_region(frame)
    if grid.size == 0:
        return []

    grid_height, grid_width = grid.shape[:2]
    gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < grid_width * grid_height * 0.01:
            continue
        if w < grid_width * 0.10 or h < grid_height * 0.12:
            continue
        if w > grid_width * 0.30 or h > grid_height * 0.40:
            continue
        aspect = w / max(1, h)
        if not 0.58 <= aspect <= 0.90:
            continue
        candidates.append((x, y, w, h))

    if len(candidates) < 4:
        return []

    x_clusters = _cluster_positions([x for x, _, _, _ in candidates], tolerance=max(12, int(grid_width * 0.03)))
    y_clusters = _cluster_positions([y for _, y, _, _ in candidates], tolerance=max(12, int(grid_height * 0.04)))
    median_w = int(round(float(np.median([w for _, _, w, _ in candidates]))))
    median_h = int(round(float(np.median([h for _, _, _, h in candidates]))))
    if median_w <= 0 or median_h <= 0:
        return []

    frame_height, frame_width = frame.shape[:2]
    top_ratio, _, left_ratio, _ = _IDOL_LIST_GRID_REGION
    roi_x1 = int(frame_width * left_ratio)
    roi_y1 = int(frame_height * top_ratio)

    boxes: list[_IdolListThumbnailBox] = []
    for y in y_clusters:
        for x in x_clusters:
            x1 = roi_x1 + x
            y1 = roi_y1 + y
            x2 = min(frame_width, x1 + median_w)
            y2 = min(frame_height, y1 + median_h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(_IdolListThumbnailBox(x1=x1, y1=y1, x2=x2, y2=y2))

    boxes.sort(key=lambda item: (item.y1, item.x1))
    return boxes


def _get_idol_list_swipe_range(
        boxes: list[_IdolListThumbnailBox],
        height: int,
) -> tuple[int, int]:
    if not boxes:
        start_y = int(height * 0.78)
        end_y = int(height * 0.46)
        return start_y, end_y

    start_y = max(box.cy for box in boxes)
    end_y = min(box.cy for box in boxes)
    if start_y <= end_y:
        start_y = int(height * 0.78)
        end_y = int(height * 0.46)
    return start_y, end_y


def _enter_idol_list_page(app: "AppProcessor") -> bool:
    list_button = None
    for _ in range(5):
        buttons = ButtonList(app.latest_results)
        list_button = next(
            (
                button
                for button in buttons
                if string_match(button.text, ButtonText.PAGE__IDOL.IDOL_LIST, _IDOL_LIST_BUTTON_MATCH)
            ),
            None,
        )
        if list_button is None:
            logger.warning(f"find{ButtonText.PAGE__IDOL.IDOL_LIST} button not found, retrying......")
            # logger.warning(f"{ButtonText.PAGE__IDOL.IDOL_LIST} button not found, skipping idol-list learning pass")
            sleep(0.5)
            continue
        if not app.game_utils.click_element_and_wait_trigger(list_button, retries=3, timeout=2.5):
            logger.warning(f"Failed to open {ButtonText.PAGE__IDOL.IDOL_LIST}")
            sleep(0.5)
            continue
        break

    if list_button is None:
        logger.warning(f"{ButtonText.PAGE__IDOL.IDOL_LIST} button not found, skipping idol-list learning pass")
        return False

    sleep(0.8)
    app.game_utils.wait_frame_stable(stable_count=2)
    return True


def _exit_idol_list_page(app: "AppProcessor") -> None:
    try:
        app.game_utils.click_button(ButtonText.CONFIRM, match_config=MatchConfig(use_fuzz=True, fuzz_threshold=70))
        sleep(0.8)
        app.game_utils.wait_frame_stable(stable_count=2)
        return
    except Exception as exc:
        logger.debug(f"Exit idol list via confirm failed: {exc}")

    try:
        app.game_utils.back_next_page()
        sleep(0.8)
        app.game_utils.wait_frame_stable(stable_count=2)
    except Exception as exc:
        logger.warning(f"Exit idol list failed: {exc}")


def _scroll_idol_list(
        app: "AppProcessor",
        boxes: Optional[list[_IdolListThumbnailBox]] = None,
) -> None:
    width, height = app.device.get_window_size()
    start_y, end_y = _get_idol_list_swipe_range(boxes or [], height)
    app.device.swipe(width // 2, start_y, width // 2, end_y, offset_y=0)
    sleep(1.0)
    app.game_utils.wait_frame_stable(stable_count=2)


def _learn_idol_card_variants_from_list(app: "AppProcessor") -> tuple[int, int]:
    rewound = _rewind_to_head(app, max_steps=30)
    if rewound > 0:
        logger.info(f"[IdolCard CLIP] Rewound selected idol before list pass: {rewound} swipes")

    if not _enter_idol_list_page(app):
        return 0, 0

    learned = 0
    failed = 0
    previous_grid: Optional[np.ndarray] = None
    previous_selected_card, _ = _ocr_match_grid_selected_card(app)
    previous_selected_id = previous_selected_card.id if previous_selected_card is not None else None

    for scroll_index in range(_IDOL_LIST_MAX_SCROLLS):
        frame = getattr(app, "latest_frame", None)
        if frame is None or frame.size == 0:
            break

        grid_boxes = _detect_idol_list_thumbnail_boxes(frame)
        if not grid_boxes:
            logger.warning(f"[IdolCard CLIP] No idol thumbnails detected on list page (scroll {scroll_index})")
            break

        page_learned = 0
        page_frame = frame.copy()
        for thumb_box in grid_boxes:
            thumb_image = page_frame[thumb_box.y1:thumb_box.y2, thumb_box.x1:thumb_box.x2].copy()
            if thumb_image.size == 0:
                continue

            app.device.click(thumb_box.cx, thumb_box.cy, "idol-list-thumbnail")
            sleep(0.35)
            app.game_utils.wait_frame_stable(stable_count=2)

            current_card, texts = _ocr_match_grid_selected_card(app)
            current_frame = getattr(app, "latest_frame", None)
            region_changed = False
            if current_frame is not None and current_frame.size > 0:
                after_crop = current_frame[thumb_box.y1:thumb_box.y2, thumb_box.x1:thumb_box.x2].copy()
                if after_crop.shape == thumb_image.shape:
                    region_changed = compute_ssim_score(thumb_image, after_crop) < 0.995

            if current_card is None:
                failed += 1
                logger.debug(f"[IdolCard CLIP] List OCR failed, texts={texts}")
                continue

            # If selection definitely did not change, do not risk learning a wrong label.
            if current_card.id == previous_selected_id and not region_changed:
                continue

            try:
                if app.clip_manager.idol_card_clip.add_variant_to_memory(thumb_image, current_card, augment=False):
                    learned += 1
                    page_learned += 1
                    logger.debug(f"[IdolCard CLIP] Learned list variant: {current_card.id}")
            except Exception as exc:
                failed += 1
                logger.warning(f"[IdolCard CLIP] List variant learn failed: {exc}")

            previous_selected_id = current_card.id

        current_grid = _extract_idol_list_grid_region(getattr(app, "latest_frame", None))
        if previous_grid is not None and check_frame_change(previous_grid, current_grid):
            logger.info("[IdolCard CLIP] Reached end of idol list")
            break
        previous_grid = current_grid.copy()

        if page_learned == 0 and scroll_index > 0:
            logger.info("[IdolCard CLIP] No new list variants learned on current page, stopping early")
            break

        _scroll_idol_list(app, boxes=grid_boxes)

    _exit_idol_list_page(app)
    return learned, failed


def action__learn_idol_card_clip(app: "AppProcessor") -> bool:
    if app.clip_manager is None or not hasattr(app.clip_manager, "idol_card_clip"):
        raise RuntimeError("CLIP 服务未初始化，无法学习偶像卡 CLIP")

    if not getattr(app, "latest_results", None) or not app.latest_results.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED):
        raise RuntimeError("当前不在偶像卡养成页面，未检测到 PRODUCT_CARD_SELECTED")

    message_tools.info("开始偶像卡 CLIP 学习，将按底部卡条逐张遍历", 5)
    wait_for_idol_card_carousel_stable(app, stable_count=2, timeout=2.5)

    rewound = _rewind_to_head(app)
    logger.info(f"Rewound idol card carousel to head in {rewound} swipes")

    resolved = 0
    failed = 0
    learned_selected = 0
    learned_variant = 0
    dropped_unresolved = 0
    tracks_by_slot: dict[int, _CarouselTrack] = {}
    pending_unresolved_tracks: list[_CarouselTrack] = []
    next_track_id = 1

    for step_index in range(_MAX_FORWARD_STEPS):
        wait_for_idol_card_carousel_stable(app, stable_count=2, timeout=2.0)
        visible_windows = _build_visible_window_map(app)
        visible_offsets = set(visible_windows)
        tracks_by_slot, learned_now = _finalize_invisible_tracks(
            app,
            tracks_by_slot,
            visible_offsets,
            pending_unresolved_tracks,
        )
        learned_variant += learned_now

        next_track_id = _sync_tracks_with_visible_windows(tracks_by_slot, visible_windows, next_track_id)

        selected_image = visible_windows.get(0)
        current_card, texts = _resolve_current_selected_idol_card(app, selected_image)
        selected_track = tracks_by_slot.get(0)
        if selected_track is None:
            selected_track = _CarouselTrack(temp_id=next_track_id)
            next_track_id += 1
            tracks_by_slot[0] = selected_track
            if selected_image is not None:
                _append_track_image(selected_track, selected_image)

        if current_card is None:
            failed += 1
            logger.warning(f"[IdolCard CLIP] Failed to resolve selected idol card, texts={texts}")
        else:
            if selected_image is not None and selected_image.size > 0:
                try:
                    if app.clip_manager.idol_card_clip.add_to_memory(selected_image, current_card):
                        learned_selected += 1
                except Exception as exc:
                    logger.warning(f"IdolCard CLIP selected learn failed: {exc}")

            if selected_track.resolved_card is None:
                selected_track.resolved_card = current_card
                resolved += 1
                logger.success(f"[IdolCard CLIP] Resolved tmp-{selected_track.temp_id} -> {current_card.id} ({current_card.name})")
            elif selected_track.resolved_card.id != current_card.id:
                logger.warning(
                    f"Selected track tmp-{selected_track.temp_id} changed id unexpectedly: "
                    f"{selected_track.resolved_card.id} -> {current_card.id}"
                )

        if not advance_to_adjacent_idol_card(app, direction="next", retries=1, timeout=2.2):
            logger.info(f"Reached idol card carousel tail after {step_index + 1} steps")
            break
        tracks_by_slot = _shift_tracks_after_next(tracks_by_slot)

    final_learned = 0
    for track in tracks_by_slot.values():
        if track.resolved_card is not None:
            final_learned += _finalize_track(app, track)
        else:
            pending_unresolved_tracks.append(track)
    learned_variant += final_learned

    list_learned = 0
    list_failed = 0
    try:
        list_learned, list_failed = _learn_idol_card_variants_from_list(app)
        learned_variant += list_learned
        failed += list_failed
    except Exception as exc:
        logger.warning(f"[IdolCard CLIP] Idol list learning pass failed: {exc}")

    late_learned, late_resolved, late_dropped = _resolve_pending_tracks(app, pending_unresolved_tracks)
    learned_variant += late_learned
    resolved += late_resolved
    dropped_unresolved += late_dropped

    summary = (
        f"偶像卡 CLIP 学习完成: 已解析 {resolved} 张, "
        f"选中图像 {learned_selected} 张, 学习图像 {learned_variant} 张, "
        f"列表变体 {list_learned} 张, 解析失败 {failed} 张, 未解析轨迹 {dropped_unresolved} 条"
    )
    message_tools.success(summary, 10)
    logger.success(summary)
    return True
