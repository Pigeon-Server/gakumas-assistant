"""Step 1: 从主页导航到培育（プロデュース）剧本选择页面。"""

import cv2
import numpy as np
from time import sleep, time
from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.game.text.produce_text import ProduceText
from src.core.exceptions.TaskException import TaskUserMessage
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.shared.common import (
    normalize_lookup_text,
    normalize_text,
    ocr_text,
)
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_modal_action_with_retry,
    detect_gameplay_state,
    find_button,
    go_back_in_gameplay,
    wait_frame_stable,
)
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Yolo import Yolo_Box
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_SCENARIO_LABELS = (
    BaseUILabels.PRODUCER_REGULAR,
    BaseUILabels.PRODUCER_PRO,
    BaseUILabels.PRODUCER_MASTER,
    BaseUILabels.PRODUCER_NIA,
)
_MAIN_HOME_SIGNAL_LABELS = (
    BaseUILabels.HOME_PRODUCE_BTN,
    BaseUILabels.HOME_SHOP_BTN,
    BaseUILabels.HOME_GIFT_BTN,
    BaseUILabels.HOME_DAILY_TASK,
    BaseUILabels.HOME_ACHIEVEMENT,
    BaseUILabels.HOME_DISPATCH_WORK,
    BaseUILabels.HOME_GET_EXPENDITURE,
)
_GAMEPLAY_SIGNAL_LABELS = (
    ProducerLabels.PC_SKIP,
    ProducerLabels.PC_TRAINING_REMAINING,
    ProducerLabels.SKILL_CARD_ACTIVE,
    ProducerLabels.SKILL_CARD_MENTAL,
    ProducerLabels.SKILL_CARD_TRAP,
    ProducerLabels.P_DRINK,
)
_GAMEPLAY_STRONG_SIGNAL_LABELS = (
    ProducerLabels.PC_SKIP,
    ProducerLabels.PC_TRAINING_REMAINING,
    ProducerLabels.SKILL_CARD_ACTIVE,
    ProducerLabels.SKILL_CARD_MENTAL,
    ProducerLabels.SKILL_CARD_TRAP,
    ProducerLabels.P_DRINK,
    ProducerLabels.PC_MENU,
)
_GAMEPLAY_MENU_MARKERS = (
    ProduceText.GAMEPLAY_MENU_SUSPEND,
    ProduceText.GAMEPLAY_MENU_SETTINGS,
    ButtonText.RETIRE,
)
_RESUMABLE_GAMEPLAY_POSITIONS = {
    GameplayPosition.LESSON_SUMMARY_SHOWCASE,
    GameplayPosition.RESULT,
    GameplayPosition.RESULT_EXAM_FAILURE,
    GameplayPosition.RESULT_EXAM_SUMMARY_SHOWCASE,
    GameplayPosition.RESULT_EXAM_RANKING_SUMMARY,
    GameplayPosition.RESULT_MEMORY_GENERATION,
    GameplayPosition.RESULT_MEMORY_PAGE,
    GameplayPosition.RESULT_FINAL_EVALUATION,
    GameplayPosition.RESULT_REWARD_SUMMARY,
    GameplayPosition.RESULT_ACHIEVEMENT_PROGRESS,
    GameplayPosition.RESULT_EVENT_REWARD_PROGRESS,
}
_ocr_service = OCRService()
_LESSON_SUMMARY_BUBBLE_WHITE_LOWER = (0, 0, 235)
_LESSON_SUMMARY_BUBBLE_WHITE_UPPER = (179, 48, 255)


def _is_on_scenario_page(app: "AppProcessor") -> bool:
    """判断是否在剧本选择页面"""
    return any(
        app.latest_results.exists_label(lbl) for lbl in _SCENARIO_LABELS
    ) and app.game_utils.update_current_location() == GamePageTypes.HOME_TAB.PRODUCER


def _is_produce_resume_modal(app: "AppProcessor", modal=None) -> bool:
    """判断是否命中了“プロデュース再開”弹窗。"""
    current_modal = modal or app.game_utils.try_get_modal(no_body=True)
    if current_modal is not None:
        modal_title = str(getattr(current_modal, "modal_title", "") or "")
        if ProduceText.PRODUCE_RESUME in modal_title:
            return True

    # 标题 OCR 偶发丢失时，退回到按钮组合判定。
    return bool(
        find_button(app, ButtonText.RETIRE, fuzz_threshold=60)
        and find_button(app, ButtonText.CANCEL, fuzz_threshold=60)
        and find_button(app, ButtonText.PRODUCE_RESUME, fuzz_threshold=60)
    )


def _is_retire_confirmation_title(title: str | None) -> bool:
    """判断给定弹窗标题是否属于放弃当前培育的确认弹窗。"""
    normalized = normalize_text(title)
    if not normalized:
        return False
    return any(
        token in normalized
        for token in (
            normalize_text(ProduceText.PRODUCE_RETIRE_CONFIRM),
            normalize_text(ModalText.TITLE.DESTROYING_PRODUCTION_DATA),
            normalize_text(ButtonText.RETIRE),
        )
    )


def _is_destroying_production_data_modal(modal) -> bool:
    """判断是否命中了“プロデュースデータの破棄”提示弹窗。"""
    if modal is None:
        return False
    normalized_title = normalize_text(getattr(modal, "modal_title", None))
    if not normalized_title:
        return False
    return normalize_text(ModalText.TITLE.DESTROYING_PRODUCTION_DATA) in normalized_title


def _is_menu_overlay_modal(modal) -> bool:
    """判断是否命中了主页/标题侧的“メニュー”残留浮层。"""
    if modal is None:
        return False
    normalized_title = normalize_text(getattr(modal, "modal_title", None))
    return normalized_title == normalize_text("メニュー")


def _build_frame_box(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    label: str,
) -> Yolo_Box | None:
    """把绝对像素矩形安全裁剪成带截图内容的 Yolo_Box。

    Args:
        frame: 当前整帧图像。
        x1: 左上角 x 坐标。
        y1: 左上角 y 坐标。
        x2: 右下角 x 坐标。
        y2: 右下角 y 坐标。
        label: 调试用标签名称。

    Returns:
        Yolo_Box | None: 裁剪成功时返回带 frame 切片的检测框；坐标越界或区域为空时返回 None。
    """
    if frame is None or getattr(frame, "size", 0) <= 0:
        return None
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(frame_width - 1, int(x1)))
    y1 = max(0, min(frame_height - 1, int(y1)))
    x2 = max(x1 + 1, min(frame_width, int(x2)))
    y2 = max(y1 + 1, min(frame_height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return Yolo_Box(x1, y1, x2, y2, label, frame[y1:y2, x1:x2].copy())


def _collect_ocr_candidates(
    frame,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> list[tuple[str, Yolo_Box, float]]:
    """在指定相对区域内执行 OCR，并返回文本、框和置信度候选列表。

    Args:
        frame: 当前整帧图像。
        left_ratio: ROI 左边界相对宽度比例。
        top_ratio: ROI 上边界相对高度比例。
        right_ratio: ROI 右边界相对宽度比例。
        bottom_ratio: ROI 下边界相对高度比例。

    Returns:
        list[tuple[str, Yolo_Box, float]]: 每个候选包含识别文本、对应的绝对坐标框和 OCR 置信度，
        供菜单识别、恢复弹窗信息提取等流程复用。
    """
    if frame is None or getattr(frame, "size", 0) <= 0:
        return []

    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(frame_width - 1, int(frame_width * left_ratio)))
    y1 = max(0, min(frame_height - 1, int(frame_height * top_ratio)))
    x2 = max(x1 + 1, min(frame_width, int(frame_width * right_ratio)))
    y2 = max(y1 + 1, min(frame_height, int(frame_height * bottom_ratio)))
    roi = frame[y1:y2, x1:x2]
    if roi.size <= 0:
        return []

    candidates: list[tuple[str, Yolo_Box, float]] = []
    for result in _ocr_service.ocr(roi):
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            continue
        width = max(1, int(getattr(result, "w", 0) or 0))
        height = max(1, int(getattr(result, "h", 0) or 0))
        abs_x1 = x1 + int(getattr(result, "x", 0) or 0)
        abs_y1 = y1 + int(getattr(result, "y", 0) or 0)
        abs_x2 = abs_x1 + width
        abs_y2 = abs_y1 + height
        box = _build_frame_box(frame, abs_x1, abs_y1, abs_x2, abs_y2, label=f"ocr:{text}")
        if box is None:
            continue
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        candidates.append((text, box, confidence))
    return candidates


def _pick_ocr_candidate(
    candidates: list[tuple[str, Yolo_Box, float]],
    token: str,
) -> Yolo_Box | None:
    """从 OCR 候选中挑出最像目标 token 的文本框。"""
    normalized_token = normalize_text(token)
    matched: list[tuple[str, Yolo_Box, float]] = []
    for text, box, confidence in candidates:
        normalized_text = normalize_text(text)
        if not normalized_text:
            continue
        if normalized_token in normalized_text or normalized_text == normalized_token:
            matched.append((text, box, confidence))
    if not matched:
        return None
    matched.sort(
        key=lambda item: (
            (item[1].w - item[1].x) * (item[1].h - item[1].y),
            len(normalize_text(item[0])),
            -item[2],
            -item[1].cx,
        )
    )
    return matched[0][1]


def _find_gameplay_menu_button(frame) -> Yolo_Box | None:
    """在局内画面右下区域查找打开菜单的圆形按钮。"""
    if frame is None or getattr(frame, "size", 0) <= 0:
        return None

    frame_height, frame_width = frame.shape[:2]
    left = max(0, int(frame_width * 0.62))
    top = max(0, int(frame_height * 0.76))
    roi = frame[top:frame_height, left:frame_width]
    if roi.size <= 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    min_radius = max(22, int(min(frame_width, frame_height) * 0.02))
    max_radius = max(min_radius + 10, int(min(frame_width, frame_height) * 0.06))
    min_dist = max(40, int(frame_width * 0.04))

    candidates: list[tuple[int, int, int]] = []
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=22,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        for cx, cy, radius in np.round(circles[0]).astype(int):
            abs_cx = left + int(cx)
            abs_cy = top + int(cy)
            radius = int(radius)
            if abs_cx < int(frame_width * 0.72) or abs_cy < int(frame_height * 0.82):
                continue
            if any(
                abs(abs_cx - existing_x) <= max(radius, existing_radius) // 2
                and abs(abs_cy - existing_y) <= max(radius, existing_radius) // 2
                for existing_x, existing_y, existing_radius in candidates
            ):
                continue
            candidates.append((abs_cx, abs_cy, radius))

    if not candidates:
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(900, int(frame_width * frame_height * 0.00025))
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < min_area or width < min_radius * 2 or height < min_radius * 2:
                continue
            aspect_ratio = width / max(height, 1)
            if aspect_ratio < 0.75 or aspect_ratio > 1.25:
                continue
            abs_cx = left + x + width // 2
            abs_cy = top + y + height // 2
            if abs_cx < int(frame_width * 0.72) or abs_cy < int(frame_height * 0.82):
                continue
            candidates.append((abs_cx, abs_cy, max(width, height) // 2))

    if not candidates:
        return None

    preferred_candidates = [
        item
        for item in candidates
        if item[2] >= max(45, int(min(frame_width, frame_height) * 0.04))
        and item[2] <= max(70, int(min(frame_width, frame_height) * 0.065))
        and item[1] >= int(frame_height * 0.90)
    ]
    target_cx, target_cy, target_radius = max(
        preferred_candidates or candidates,
        key=lambda item: (item[1], item[0], item[2]),
    )
    target_box = _build_frame_box(
        frame,
        target_cx - target_radius,
        target_cy - target_radius,
        target_cx + target_radius,
        target_cy + target_radius,
        label="producer_gameplay_menu_button",
    )
    if target_box is not None:
        DebugTools().add_box(
            int(target_box.x),
            int(target_box.y),
            int(target_box.w),
            int(target_box.h),
            label="gameplay_menu",
            color=(255, 0, 255),
            alpha=0.12,
            duration=2.5,
            font_size=18,
        )
    return target_box


def _has_gameplay_retire_menu(frame) -> bool:
    """判断当前局内底部菜单是否已经展开到可见リタイア文本。"""
    candidates = _collect_ocr_candidates(
        frame,
        left_ratio=0.0,
        top_ratio=0.78,
        right_ratio=1.0,
        bottom_ratio=0.92,
    )
    matched_boxes: list[Yolo_Box] = []
    for token in _GAMEPLAY_MENU_MARKERS:
        candidate = _pick_ocr_candidate(candidates, token)
        if candidate is None:
            continue
        matched_boxes.append(candidate)
    if len(matched_boxes) < 2:
        return False
    for box in matched_boxes:
        DebugTools().add_box(
            int(box.x),
            int(box.y),
            int(box.w),
            int(box.h),
            label="gameplay_menu_text",
            color=(0, 200, 255),
            alpha=0.12,
            duration=2.5,
            font_size=18,
        )
    return True


def _find_gameplay_retire_menu_entry(frame) -> Yolo_Box | None:
    """在已展开的局内菜单中定位“リタイア”入口文本框。"""
    right_half_candidates = _collect_ocr_candidates(
        frame,
        left_ratio=0.5,
        top_ratio=0.78,
        right_ratio=1.0,
        bottom_ratio=0.92,
    )
    retire_box = _pick_ocr_candidate(right_half_candidates, ButtonText.RETIRE)
    if retire_box is None:
        full_width_candidates = _collect_ocr_candidates(
            frame,
            left_ratio=0.0,
            top_ratio=0.78,
            right_ratio=1.0,
            bottom_ratio=0.92,
        )
        retire_box = _pick_ocr_candidate(full_width_candidates, ButtonText.RETIRE)
    if retire_box is not None:
        DebugTools().add_box(
            int(retire_box.x),
            int(retire_box.y),
            int(retire_box.w),
            int(retire_box.h),
            label="gameplay_retire",
            color=(255, 80, 80),
            alpha=0.14,
            duration=2.5,
            font_size=18,
        )
    return retire_box


def _detect_resume_lesson_summary_bubble(frame) -> tuple[int, int, int, int] | None:
    """检测底部参数展示白色气泡，作为 lesson summary 的视觉兜底特征。"""
    if frame is None or getattr(frame, "size", 0) <= 0:
        return None

    frame_h, frame_w = frame.shape[:2]
    roi_top = int(frame_h * 0.74)
    roi_bottom = int(frame_h * 0.98)
    if roi_bottom <= roi_top:
        return None

    roi = frame[roi_top:roi_bottom, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _LESSON_SUMMARY_BUBBLE_WHITE_LOWER, _LESSON_SUMMARY_BUBBLE_WHITE_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    best_rect: tuple[int, int, int, int] | None = None
    best_score = -1.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        if w_rect <= 0 or h_rect <= 0:
            continue
        abs_y = roi_top + y
        if w_rect < int(frame_w * 0.68):
            continue
        if h_rect < int(frame_h * 0.06) or h_rect > int(frame_h * 0.18):
            continue
        if abs_y < int(frame_h * 0.76):
            continue
        if (abs_y + h_rect) > int(frame_h * 0.98):
            continue
        center_x = x + w_rect / 2.0
        center_penalty = abs(center_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
        score = float(w_rect * h_rect) - center_penalty * float(frame_w * frame_h * 0.08)
        if score > best_score:
            best_score = score
            best_rect = (x, abs_y, x + w_rect, abs_y + h_rect)
    return best_rect


def _looks_like_resume_lesson_summary_showcase(app: "AppProcessor") -> bool:
    """在恢复链中识别 lesson 结束后的参数展示页。

    真机恢复时，这一页偶发会在 producer 模型下出现“零检测框”，导致常规
    `detect_gameplay_state()` 无法命中。这里直接对底部 ROI 做局部 OCR 兜底，
    仅在缺少常规 gameplay 控件时认定为可恢复的 lesson summary 展示页。
    """
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return False

    results = getattr(app, "latest_results", None)
    frame_h, frame_w = frame.shape[:2]

    if results is not None:
        has_blocking_controls = any(
            results.exists_label(label)
            for label in (
                ProducerLabels.PC_ACTION,
                ProducerLabels.PC_RECOMMEND_ACTION,
                ProducerLabels.PC_SKIP,
                ProducerLabels.PC_BONUS_INDICATOR,
                ProducerLabels.SKILL_CARD_ACTIVE,
                ProducerLabels.SKILL_CARD_MENTAL,
                ProducerLabels.SKILL_CARD_TRAP,
                ProducerLabels.SKILL_CARD_INFO,
                ProducerLabels.UNIVERSAL_OPTIONS,
                ProducerLabels.CONFIRM_BUTTON,
                ProducerLabels.DISABLE_BUTTON,
                ProducerLabels.CANCEL_BUTTON,
                ProducerLabels.P_DRINK,
                BaseUILabels.BUTTON,
                BaseUILabels.CURRENT_LOCATION,
                BaseUILabels.SKIP_BUTTON,
            )
        )
        if has_blocking_controls:
            return False
        param_label_count = sum(
            1
            for label in (
                ProducerLabels.PARAM_VOCAL,
                ProducerLabels.PARAM_DANCE,
                ProducerLabels.PARAM_VISUAL,
            )
            if results.exists_label(label)
        )
        detected_labels = {
            str(getattr(box, "label", "") or "")
            for box in getattr(results, "boxes", []) or []
        }
        only_param_labels = bool(detected_labels) and detected_labels.issubset({
            ProducerLabels.PARAM_VOCAL,
            ProducerLabels.PARAM_DANCE,
            ProducerLabels.PARAM_VISUAL,
        })
    else:
        param_label_count = 0
        only_param_labels = False

    if param_label_count >= 2 and only_param_labels:
        DebugTools().add_box(
            0,
            int(frame_h * 0.56),
            frame_w - 1,
            frame_h - 1,
            label="resume_lesson_summary:param_only_layout",
            color=(0, 200, 255),
            alpha=0.08,
            duration=2.5,
            font_size=18,
        )
        logger.debug("navigate_to_produce: 恢复链命中 lesson summary 参数布局兜底")
        return True

    roi_left = int(frame_w * 0.04)
    roi_top = int(frame_h * 0.78)
    roi_right = int(frame_w * 0.96)
    roi_bottom = int(frame_h * 0.98)
    if roi_right <= roi_left or roi_bottom <= roi_top:
        return False

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    roi_text = normalize_lookup_text(ocr_text(roi))
    short_attr_tokens = (
        ProduceText.VOCAL_SHORT,
        ProduceText.DANCE_SHORT,
        ProduceText.VISUAL_SHORT,
    )
    long_attr_tokens = (
        ProduceText.VOCAL,
        ProduceText.DANCE,
        ProduceText.VISUAL,
    )
    has_short_attr = any(normalize_lookup_text(token) in roi_text for token in short_attr_tokens)
    has_long_attr = any(normalize_lookup_text(token) in roi_text for token in long_attr_tokens)
    has_increased = normalize_lookup_text(ProduceText.INCREASED) in roi_text
    ocr_matched = bool(roi_text and has_increased and (has_short_attr or has_long_attr or param_label_count >= 1))

    bubble_rect = _detect_resume_lesson_summary_bubble(frame)
    visual_matched = bool(bubble_rect is not None and param_label_count >= 1)
    if not ocr_matched and not visual_matched:
        return False

    debugger = DebugTools()
    if ocr_matched:
        debugger.add_box(
            roi_left,
            roi_top,
            roi_right,
            roi_bottom,
            label="resume_lesson_summary:ocr_roi",
            color=(0, 220, 120),
            alpha=0.12,
            duration=2.5,
            font_size=18,
        )
        logger.debug(
            "navigate_to_produce: 恢复链命中 lesson summary OCR 兜底 text={!r}",
            roi_text,
        )
    if bubble_rect is not None:
        debugger.add_box(
            bubble_rect[0],
            bubble_rect[1],
            bubble_rect[2],
            bubble_rect[3],
            label="resume_lesson_summary:bubble",
            color=(0, 200, 255),
            alpha=0.10,
            duration=2.5,
            font_size=18,
        )
        if not ocr_matched:
            logger.debug("navigate_to_produce: 恢复链命中 lesson summary 白色气泡兜底")
    return True


def _looks_like_resume_lesson_p_drink_showcase(app: "AppProcessor") -> bool:
    """在恢复链中识别 lesson 收尾后的 P 饮料获得展示页。"""
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return False

    results = getattr(app, "latest_results", None)
    frame_h, frame_w = frame.shape[:2]
    if results is not None:
        has_blocking_controls = any(
            results.exists_label(label)
            for label in (
                ProducerLabels.PC_ACTION,
                ProducerLabels.PC_RECOMMEND_ACTION,
                ProducerLabels.PC_PROGRESS,
                ProducerLabels.PC_STAMINA,
                ProducerLabels.PC_P_POINT,
                ProducerLabels.PC_TRAINING_SCORE,
                ProducerLabels.PC_TRAINING_REMAINING,
                ProducerLabels.PC_SKIP,
                ProducerLabels.PC_BONUS_INDICATOR,
                ProducerLabels.SKILL_CARD_ACTIVE,
                ProducerLabels.SKILL_CARD_MENTAL,
                ProducerLabels.SKILL_CARD_TRAP,
                ProducerLabels.SKILL_CARD_INFO,
                ProducerLabels.UNIVERSAL_OPTIONS,
                ProducerLabels.CONFIRM_BUTTON,
                ProducerLabels.DISABLE_BUTTON,
                ProducerLabels.CANCEL_BUTTON,
                ProducerLabels.P_DRINK,
                BaseUILabels.BUTTON,
                BaseUILabels.CURRENT_LOCATION,
                BaseUILabels.SKIP_BUTTON,
            )
        )
        if has_blocking_controls:
            return False

    roi_left = int(frame_w * 0.06)
    roi_top = int(frame_h * 0.55)
    roi_right = int(frame_w * 0.94)
    roi_bottom = int(frame_h * 0.92)
    if roi_right <= roi_left or roi_bottom <= roi_top:
        return False

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    roi_text = normalize_lookup_text(ocr_text(roi))
    has_drink_hint = (
        normalize_lookup_text(ProduceText.DRINK) in roi_text
        or normalize_lookup_text(ProduceText.ACQUIRE) in roi_text
    )
    has_gain_hint = (
        normalize_lookup_text(ProduceText.PARAMETER) in roi_text
        or normalize_lookup_text(ProduceText.P_POINT) in roi_text
        or normalize_lookup_text("+10") in roi_text
    )
    if not (roi_text and has_drink_hint and has_gain_hint):
        return False

    roi_rgb = roi.astype("int16")
    bright_mask = (
        (roi_rgb[:, :, 0] >= 185)
        & (roi_rgb[:, :, 1] >= 185)
        & (roi_rgb[:, :, 2] >= 185)
    )
    near_gray_mask = (roi_rgb.max(axis=2) - roi_rgb.min(axis=2)) <= 45
    white_ratio = float((bright_mask & near_gray_mask).mean())
    if white_ratio < 0.18:
        return False

    DebugTools().add_box(
        roi_left,
        roi_top,
        roi_right,
        roi_bottom,
        label="resume_lesson_p_drink_showcase:roi",
        color=(80, 220, 255),
        alpha=0.10,
        duration=2.5,
        font_size=18,
    )
    logger.debug(
        "navigate_to_produce: 恢复链命中 lesson p_drink showcase OCR 兜底 text={!r}",
        roi_text,
    )
    return True


def _wait_for_gameplay_retire_menu(app: "AppProcessor", *, timeout: float = 3.0) -> bool:
    """等待局内底部菜单展开到可识别出退出文本。"""
    end_time = time() + timeout
    while time() < end_time:
        frame = getattr(app, "latest_frame", None)
        if _has_gameplay_retire_menu(frame):
            return True
        sleep(0.3)
    return False


def _wait_for_modal_relaxed(
    app: "AppProcessor",
    *,
    timeout: float = 4.0,
):
    """以较宽松的头部要求轮询弹窗，兼容退出确认等过渡态弹窗。"""
    end_time = time() + timeout
    while time() < end_time:
        modal = app.game_utils.try_get_modal(no_body=True, require_header=False)
        if modal is not None:
            return modal
        sleep(0.3)
    return None


def _looks_like_active_gameplay(app: "AppProcessor") -> bool:
    """综合 YOLO、页面位置与菜单特征判断当前是否已在局内 gameplay。"""
    if _is_on_scenario_page(app):
        return False
    if app.latest_results and app.latest_results.exists_label(BaseUILabels.HOME_PRODUCE_BTN):
        return False

    current_location = app.game_utils.update_current_location()
    if current_location == GamePageTypes.MAIN_MENU__HOME:
        return False

    results = getattr(app, "latest_results", None)
    if results is not None:
        for label in _GAMEPLAY_STRONG_SIGNAL_LABELS:
            if results.exists_label(label):
                return True

    frame = getattr(app, "latest_frame", None)
    if frame is None:
        return False

    if _has_gameplay_retire_menu(frame):
        return True

    menu_button = _find_gameplay_menu_button(frame)
    if menu_button is None:
        return False

    # 仅检测到右下角圆形按钮时，不能直接判定为局内。
    # 主页等界面也可能存在相似圆形入口，必须结合位置与其它信号收紧判断。
    frame_height, frame_width = frame.shape[:2]
    near_bottom_right = (
        menu_button.cx >= int(frame_width * 0.86)
        and menu_button.cy >= int(frame_height * 0.88)
    )
    return near_bottom_right and current_location not in {
        GamePageTypes.UNKNOWN,
        GamePageTypes.MAIN_MENU__HOME,
    }


def _retire_active_gameplay_produce(app: "AppProcessor") -> bool:
    """执行结束培育`retire_active_gameplay_produce`。"""
    if not _looks_like_active_gameplay(app) and not _has_gameplay_retire_menu(getattr(app, "latest_frame", None)):
        return False

    frame = getattr(app, "latest_frame", None)
    if not _has_gameplay_retire_menu(frame):
        menu_button = _find_gameplay_menu_button(frame)
        if menu_button is None:
            return False
        logger.info("navigate_to_produce: 常规返回失败，尝试从局内右下菜单退出旧局")
        if not app.game_utils.click_element_and_wait_trigger(menu_button, retries=2, timeout=2.5):
            return False
        wait_frame_stable(app, timeout=2.0)
        if not _wait_for_gameplay_retire_menu(app, timeout=3.0):
            raise TimeoutError("点击局内菜单按钮后，未识别到退出菜单文本")
        frame = getattr(app, "latest_frame", None)

    retire_entry = _find_gameplay_retire_menu_entry(frame)
    if retire_entry is None:
        raise TimeoutError("已识别到局内菜单，但未识别到リタイア入口")

    logger.info("navigate_to_produce: 点击局内菜单中的リタイア，准备重新开局")
    if not app.game_utils.click_element_and_wait_trigger(retire_entry, retries=2, timeout=3.0):
        raise TimeoutError("点击局内菜单リタイア后未触发界面变化")

    sleep(0.5)
    confirm_modal = _wait_for_modal_relaxed(app, timeout=4.0)
    if confirm_modal is None:
        wait_frame_stable(app, timeout=2.0)
        confirm_modal = _wait_for_modal_relaxed(app, timeout=3.0)
    if confirm_modal is None:
        raise TimeoutError("点击局内菜单リタイア后未出现确认弹窗")

    logger.info(f"navigate_to_produce: 确认退出当前培育 {confirm_modal.modal_title!r}")
    if (
        ModalText.TITLE.DESTROYING_PRODUCTION_DATA not in str(confirm_modal.modal_title or "")
        and str(confirm_modal.modal_title or "").strip()
    ):
        logger.debug(f"navigate_to_produce: 局内退出弹窗标题偏差={confirm_modal.modal_title!r}")

    if not click_modal_action_with_retry(
        app,
        confirm_modal,
        prefer_confirm=True,
        retries=2,
        timeout=4.0,
        action_name="navigate_to_produce gameplay retire confirm",
    ):
        raise TimeoutError("未能确认局内リタイア弹窗")

    app.game_utils.wait_loading()
    wait_frame_stable(app, timeout=3.0)
    return True


def open_produce_entry_from_home(app: "AppProcessor", *, timeout: float = 10.0, location_timeout: float = 15.0) -> None:
    """在主页点击 Produce 入口，并等待入口动画/过场收敛。"""
    if not app.game_utils.wait_for_label(BaseUILabels.HOME_PRODUCE_BTN, timeout=timeout):
        raise TimeoutError("等待 Home: Produce Button 超时")
    app.game_utils.click_on_label(BaseUILabels.HOME_PRODUCE_BTN)
    app.game_utils.wait_loading()


def _extract_resume_modal_info(app: "AppProcessor") -> dict:
    """从「プロデュース再開」弹窗中提取中断培育信息。

    提取内容：剧本名、难度、当前周数/总周数、偶像名、三维参数及百分比。
    """
    import re

    frame = getattr(app.latest_results, "frame", None)
    if frame is None:
        return {}

    info: dict = {}
    fh, fw = frame.shape[:2]

    # OCR 整个模态区域（约 y=30%~95%）
    candidates = _collect_ocr_candidates(
        frame,
        left_ratio=0.0,
        top_ratio=0.30,
        right_ratio=1.0,
        bottom_ratio=0.95,
    )
    texts = [(t, box, conf) for t, box, conf in candidates if t.strip()]

    # 剧本名（含「」的文本，如 定期公演『初』）
    for t, _, _ in texts:
        if "『" in t and "』" in t:
            info["challenge_name"] = t.strip()
            break

    # 难度（マスター/レギュラー/プロ/レジェンド）。
    difficulty_map = ProduceText.DIFFICULTY_LABEL_MAP
    for t, _, _ in texts:
        for jp, en in difficulty_map.items():
            if jp in t:
                info["difficulty"] = en
                info["difficulty_jp"] = jp
                break

    # 周数（8/18週目）
    for t, _, _ in texts:
        m = re.search(rf"(\d+)\s*/\s*(\d+)\s*{ProduceText.WEEK}", t)
        if m:
            info["current_week"] = int(m.group(1))
            info["total_weeks"] = int(m.group(2))
            break

    # 偶像名（含 ［ ］ 的文本）
    for t, _, _ in texts:
        if "［" in t or "］" in t or "[" in t or "]" in t:
            info["idol_name"] = t.strip()
            break

    # 参数百分比（XX.X%）
    pct_values = []
    for t, _, _ in texts:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", t):
            pct_values.append(float(m.group(1)))
    if len(pct_values) >= 3:
        info["vocal_pct"] = pct_values[0]
        info["dance_pct"] = pct_values[1]
        info["visual_pct"] = pct_values[2]

    # 参数绝对值（三个 /1800 前面的数字）
    param_values = []
    for t, _, _ in texts:
        for m in re.finditer(r"(\d{2,4})\s*/\s*1[0-9]{3}", t):
            param_values.append(int(m.group(1)))
    if len(param_values) >= 3:
        info["vocal_value"] = param_values[0]
        info["dance_value"] = param_values[1]
        info["visual_value"] = param_values[2]

    logger.info(f"[恢复培育] 弹窗信息: {info}")
    return info


def resume_resumable_produce(app: "AppProcessor", *, timeout: float = 8.0) -> bool:
    """处理主页点 Produce 后出现的“继续上次培育”弹窗，选择恢复旧局。"""
    end_time = time() + timeout
    while time() < end_time:
        modal = app.game_utils.try_get_modal(no_body=True)
        if not _is_produce_resume_modal(app, modal):
            sleep(0.4)
            continue

        logger.info("resume_produce: 检测到未完成培育，点击再開する恢复旧局")
        resume_button = find_button(app, ButtonText.PRODUCE_RESUME, fuzz_threshold=60)
        if resume_button is None:
            raise TimeoutError("检测到培育再开弹窗，但未识别到再開する按钮")

        if not app.game_utils.click_element_and_wait_trigger(resume_button, timeout=3.0):
            raise TimeoutError("点击再開する后未触发界面变化")

        app.game_utils.wait_loading()
        wait_frame_stable(app, timeout=3.0)
        return True

    return False


class NavigateToProduceStep(ProduceStep):
    """从主页或残留局面进入培育入口，并把流程带到稳定可接管的起点页面。"""

    step_name = "navigate_to_produce"

    @staticmethod
    def _looks_like_main_home(app: "AppProcessor") -> bool:
        """用稳定的主页 UI 标签兜底识别主主页，避免被位置 OCR 误导。

        真机上主页弹窗关闭后的短时间内，`update_current_location()` 偶发会把顶部
        文本误读成 `PASS_REWARD` 一类页面，但此时底部 Home Tab 和主页 Produce 入口
        不一定会同帧稳定返回。这里优先用 YOLO 标签做兜底，避免恢复流程走错分支。
        """
        results = getattr(app, "latest_results", None)
        if results is None:
            return False
        has_home_tab = results.exists_label(BaseUILabels.TAB_HOME)
        has_home_signal = any(results.exists_label(label) for label in _MAIN_HOME_SIGNAL_LABELS)
        return bool(has_home_tab and has_home_signal)

    @classmethod
    def _wait_for_main_home_signal(
        cls,
        app: "AppProcessor",
        *,
        timeout: float = 2.0,
        interval: float = 0.25,
    ) -> bool:
        """短轮询主页信号，兼容弹窗关闭后主页标签晚几帧才稳定回来的情况。"""
        end_time = time() + timeout
        while time() < end_time:
            if cls._looks_like_main_home(app):
                return True
            sleep(interval)
        return cls._looks_like_main_home(app)

    @staticmethod
    def _enter_from_home_entry(app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """统一走主页入口进入培育，并复用已有的恢复弹窗判定。

        这里不重新发明主页识别流程，只做两件事：
        1. 进入主页入口前先切回 BASE_UI，确保 `HOME_PRODUCE_BTN` 使用对的模型；
        2. 点击入口后立即复用已有弹窗逻辑，判断是恢复旧局还是进入新培育。
        """
        from src.constants.yolo.model_type import YoloModelType

        app.switch_yolo_model(YoloModelType.BASE_UI, settle_seconds=1.0)
        open_produce_entry_from_home(app)
        wait_frame_stable(app, timeout=2.0)

        if NavigateToProduceStep._try_resume_interrupted(app, ctx):
            return True
        if NavigateToProduceStep._confirm_destroying_production_data_modal(app, ctx):
            return True
        return True

    @staticmethod
    def _recover_from_start_game(app: "AppProcessor") -> None:
        """从标题启动页恢复到主页，避免把启动页误判成 producer 页面。

        这里直接复用现有的启动页处理逻辑：
        1. 点击「Tap / Click Continue」
        2. 等待加载结束
        3. 处理启动阶段可能出现的弹窗，并确保回到主页
        """
        from src.core.tasks.base_ui.start_game import (
            action__click_start_game,
            action__wait_enter_home,
        )

        logger.info("navigate_to_produce: 当前位于 START_GAME，先恢复到主页再继续培育导航")
        action__click_start_game(app)
        app.game_utils.wait_loading()
        action__wait_enter_home(app)
        wait_frame_stable(app, timeout=3.0)

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """把当前会话导航到剧本选择页，或在恢复模式下直接接管旧局。

        Args:
            app: 当前应用处理器，用于点击入口、识别弹窗以及处理局内恢复链。
            ctx: 培育上下文；当 `resume_interrupted` 为 True 时，会优先尝试恢复旧局或直接接管当前 gameplay。

        Returns:
            bool: 成功进入剧本选择页，或在恢复模式下成功接管中断培育时返回 True。

        Raises:
            TimeoutError: 无法进入剧本选择页、无法确认退出弹窗或恢复旧局时抛出。

        Notes:
            该步骤会根据当前界面状态自动选择三条路径：
            1. 已在剧本页则直接返回；
            2. 恢复模式下命中旧局则直接恢复或接管；
            3. 其余情况先清理残留弹窗、必要时退出旧局，再直接接管当前 producer 页面。
        """
        if _is_on_scenario_page(app):
            logger.debug("已经在培育剧本选择页面")
            return True

        is_resume = ctx.resume_interrupted

        self._dismiss_residual_modal(app)

        if self._wait_for_main_home_signal(app):
            logger.debug("navigate_to_produce: 通过 BASE_UI 主页信号确认当前在主页，优先走主页入口点击流程")
            if self._try_resume_interrupted(app, ctx):
                return True
            return self._enter_from_home_entry(app, ctx)

        current_location = app.game_utils.update_current_location()
        if current_location == GamePageTypes.START_GAME:
            self._recover_from_start_game(app)
            if self._wait_for_main_home_signal(app):
                logger.debug("navigate_to_produce: START_GAME 恢复后命中主页信号，继续走主页入口")
                if self._try_resume_interrupted(app, ctx):
                    return True
                return self._enter_from_home_entry(app, ctx)
            current_location = app.game_utils.update_current_location()
        producer_sub_pages = {
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__IDOL_SELECTION,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__SUPPORT_SELECTION,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_SELECTION,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__START_CONFIRMATION,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__FORMATION_DETAIL,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_FORMATION_LIST,
            GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__SUPPORT_FORMATION_LIST,
        }
        if current_location in producer_sub_pages:
            logger.debug(f"navigate_to_produce: 已在 producer 子页 {current_location}，直接交给后续步骤")
            ctx.resumed_from_interrupt = True
            if current_location == GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__IDOL_SELECTION:
                ctx.resume_pipeline_step = "select_idol_card"
            elif current_location == GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__SUPPORT_SELECTION:
                ctx.resume_pipeline_step = "select_support_cards"
            elif current_location == GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_SELECTION:
                ctx.resume_pipeline_step = "select_memories"
            elif current_location == GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__START_CONFIRMATION:
                ctx.resume_pipeline_step = "confirm_and_start"
            elif current_location == GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__FORMATION_DETAIL:
                ctx.resume_pipeline_step = "collect_formation_details"
            elif current_location in {
                GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__MEMORY_FORMATION_LIST,
                GamePageTypes.HOME_TAB.PRODUCER_SUB_PAGE.PRODUCER__SUPPORT_FORMATION_LIST,
            }:
                ctx.resume_pipeline_step = "select_memories"
            return True
        if current_location == GamePageTypes.HOME_TAB.PRODUCER and self._looks_like_producer_entry_page(app):
            logger.debug("navigate_to_produce: 已在 producer 专用选择页，直接交给后续步骤")
            ctx.resumed_from_interrupt = True
            ctx.resume_pipeline_step = "select_scenario"
            return True
        if current_location == GamePageTypes.MAIN_MENU__HOME:
            logger.debug("navigate_to_produce: 当前在主页，优先走主页入口点击流程")
            if self._try_resume_interrupted(app, ctx):
                return True
            return self._enter_from_home_entry(app, ctx)

        # 恢复中断模式：检测到再开弹窗时恢复旧局
        if is_resume:
            if self._try_resume_interrupted(app, ctx):
                return True
            if self._wait_for_main_home_signal(app):
                logger.debug("navigate_to_produce: resume 模式当前仍像主页，优先走主页入口点击流程")
                return self._enter_from_home_entry(app, ctx)
            # 没有中断弹窗，但屏幕上可能仍停在 producer 专用页面。
            # 继续通过 producer 模型直接判断，不回主页重进。
            logger.debug("navigate_to_produce: resume 模式未找到中断弹窗，继续直接恢复当前 producer 页面")
            if self._try_detect_existing_gameplay(app, ctx):
                return True
            self._dismiss_residual_modal(app)
            if self._wait_for_main_home_signal(app):
                logger.debug("navigate_to_produce: 清理后重新判定为主页，改走主页入口点击流程")
                return self._enter_from_home_entry(app, ctx)

        if not self._retire_resumable_produce(app):
            self._dismiss_residual_modal(app)
            if self._wait_for_main_home_signal(app):
                logger.debug("navigate_to_produce: 关闭可恢复旧局弹窗后回到主页，改走主页入口点击流程")
                return self._enter_from_home_entry(app, ctx)

        # 再次确认当前是否已经在 producer 局内或剧本页。
        if self._try_detect_existing_gameplay(app, ctx):
            return True

        # 如果仍然停留在可恢复的 producer 页面，继续专用恢复链，不回主页。
        if app.latest_results and app.latest_results.exists_label(BaseUILabels.MODAL_HEADER):
            modal_pre_recover = app.game_utils.try_get_modal(no_body=True, require_header=False)
            if modal_pre_recover is not None and _is_produce_resume_modal(app, modal_pre_recover):
                logger.info("navigate_to_produce: 命中培育再开弹窗，先关闭")
                if self._retire_resumable_produce(app):
                    sleep(0.5)
                elif not self._force_close_resume_modal(app):
                    logger.warning("navigate_to_produce: 无法关闭培育再开弹窗，继续恢复尝试")

        # 直接走 producer 恢复链，不再回主页。
        if go_back_in_gameplay(app):
            sleep(0.8)
            self._dismiss_residual_modal(app)
            if self._try_detect_existing_gameplay(app, ctx):
                return True
            if self._try_resume_interrupted(app, ctx):
                return True
            if not _retire_active_gameplay_produce(app):
                raise TimeoutError("无法在当前 producer 页面完成恢复")
            self._dismiss_residual_modal(app)
            if self._try_detect_existing_gameplay(app, ctx):
                return True
        else:
            if not _retire_active_gameplay_produce(app):
                raise TimeoutError("无法在当前 producer 页面完成恢复")
            self._dismiss_residual_modal(app)
            if self._try_detect_existing_gameplay(app, ctx):
                return True

        raise TimeoutError("导航到培育剧本选择页超时")

    @staticmethod
    def _looks_like_producer_entry_page(app: "AppProcessor") -> bool:
        """判断是否已在 producer 的剧本/难度/选卡入口页。"""
        results = getattr(app, "latest_results", None)
        if results is None:
            return False
        entry_labels = _SCENARIO_LABELS + (
            BaseUILabels.PRODUCER_NIA,
            BaseUILabels.PRODUCE_CARD_VOCAL,
            BaseUILabels.PRODUCE_CARD_DANCE,
            BaseUILabels.PRODUCE_CARD_VISUAL,
        )
        return any(results.exists_label(lbl) for lbl in entry_labels)

    @staticmethod
    def _try_resume_interrupted(app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """检测并恢复中断的培育。

        如果检测到「プロデュース再開」弹窗，提取信息后点击「再開する」恢复旧局。
        成功恢复后设置 ctx.resumed_from_interrupt = True，并把 resume_pipeline_step
        指向 gameplay 主循环，让流水线从当前局面接着跑。
        """
        modal = app.game_utils.try_get_modal(no_body=True)
        if not _is_produce_resume_modal(app, modal):
            return False

        # 提取弹窗中的培育信息
        resume_info = _extract_resume_modal_info(app)
        ctx.resume_info = resume_info

        # 回填 context 中的周数信息
        if "current_week" in resume_info:
            ctx.current_week = resume_info["current_week"]
        if "difficulty" in resume_info:
            logger.info(f"[恢复培育] 中断难度={resume_info.get('difficulty_jp', resume_info['difficulty'])}")

        # 点击“再開する”恢复旧局。
        if not resume_resumable_produce(app, timeout=8.0):
            logger.warning("[恢复培育] 点击再開する失败，回退到正常流程")
            return False

        ctx.resumed_from_interrupt = True
        ctx.resume_pipeline_step = "produce_gameplay_loop"
        logger.success(
            f"[恢复培育] 成功恢复中断培育: "
            f"week={resume_info.get('current_week', '?')}/{resume_info.get('total_weeks', '?')}, "
            f"idol={resume_info.get('idol_name', '?')}"
        )
        return True

    @staticmethod
    def _confirm_destroying_production_data_modal(app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """确认跨设备旧局提示弹窗，继续进入新的培育入口。"""
        modal = app.game_utils.try_get_modal(no_body=True)
        if not _is_destroying_production_data_modal(modal):
            return False

        if not getattr(ctx, "allow_destroy_production_data", True):
            logger.warning("navigate_to_produce: 用户配置禁止确认「プロデュースデータの破棄」，取消后返回主页")
            if not click_modal_action_with_retry(
                app,
                modal,
                prefer_confirm=False,
                retries=2,
                timeout=4.0,
                action_name="navigate_to_produce destroy production data cancel",
            ):
                raise TimeoutError("未能取消「プロデュースデータの破棄」弹窗")
            wait_frame_stable(app, timeout=2.0)
            try:
                app.game_utils.go_home()
            except Exception as exc:
                raise TaskUserMessage("用户配置禁止确认「プロデュースデータの破棄」，且取消后返回主页失败") from exc
            raise TaskUserMessage("用户配置禁止确认「プロデュースデータの破棄」，已取消并返回主页")

        if getattr(ctx, "resume_interrupted", False):
            logger.warning(
                "[恢复培育] 命中「プロデュースデータの破棄」提示；当前设备无法直接恢复另一端旧局，将确认后改为重新开始"
            )
        else:
            logger.info("navigate_to_produce: 检测到跨设备进行中培育提示，确认后继续进入剧本页")

        if not click_modal_action_with_retry(
            app,
            modal,
            prefer_confirm=True,
            retries=2,
            timeout=4.0,
            action_name="navigate_to_produce destroy production data confirm",
        ):
            raise TimeoutError("未能确认「プロデュースデータの破棄」弹窗")

        app.game_utils.wait_loading()
        wait_frame_stable(app, timeout=3.0)
        return True

    @staticmethod
    def _try_detect_existing_gameplay(app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """resume 模式下 go_home 失败时，检测是否已在 gameplay 中。

        临时切换到 PRODUCER 模型检测画面阶段。如果检测到有效 gameplay 阶段，
        标记为恢复模式并直接返回成功，跳过后续导航步骤。
        """
        from src.constants.game.producer_gameplay import GameplayPhase
        from src.constants.yolo.model_type import YoloModelType

        original_model = app.yolo_engine.model_type
        try:
            app.switch_yolo_model(YoloModelType.PRODUCER, settle_seconds=2.0)

            for _ in range(5):
                # 这里必须基于同一帧快照同时判定 phase / position，避免跨帧读到
                # 「phase=unknown 但 position=transition_empty」的撕裂状态，从而把主页
                # “培育中”入口卡片误当成 gameplay 局内恢复锚点。
                phase, position = detect_gameplay_state(app, ctx)
                results = getattr(app, "latest_results", None)
                has_lesson_summary_signature = bool(
                    results is not None
                    and results.exists_label(ProducerLabels.PC_ACTION_INFO)
                    and any(
                        results.exists_label(label)
                        for label in (
                            ProducerLabels.PARAM_VOCAL,
                            ProducerLabels.PARAM_DANCE,
                            ProducerLabels.PARAM_VISUAL,
                        )
                    )
                )

                # MODAL / STARTUP_MODALS 也属于 gameplay 内的合法阶段
                # （例如休息确认、饮料确认等模态都在培育局内）
                if phase not in {GameplayPhase.UNKNOWN, ""}:
                    ctx.resumed_from_interrupt = True
                    ctx.resume_pipeline_step = "produce_gameplay_loop"
                    logger.success(
                        f"[恢复培育] 已在 gameplay 中，检测到阶段: {phase}，位置: {position}，跳过导航"
                    )
                    return True

                if position in _RESUMABLE_GAMEPLAY_POSITIONS:
                    ctx.resumed_from_interrupt = True
                    ctx.resume_pipeline_step = "produce_gameplay_loop"
                    logger.success(
                        f"[恢复培育] 已在 gameplay 中，检测到位置: {position}，跳过导航"
                    )
                    return True

                if has_lesson_summary_signature:
                    ctx.resumed_from_interrupt = True
                    ctx.resume_pipeline_step = "produce_gameplay_loop"
                    logger.success(
                        "[恢复培育] 命中 lesson-summary 展示页特征，直接交给 gameplay loop 接管"
                    )
                    return True

                if _looks_like_resume_lesson_summary_showcase(app):
                    ctx.resumed_from_interrupt = True
                    ctx.resume_pipeline_step = "produce_gameplay_loop"
                    logger.success(
                        "[恢复培育] 命中 lesson-summary 展示页 OCR 兜底，直接交给 gameplay loop 接管"
                    )
                    return True

                if _looks_like_resume_lesson_p_drink_showcase(app):
                    ctx.resumed_from_interrupt = True
                    ctx.resume_pipeline_step = "produce_gameplay_loop"
                    logger.success(
                        "[恢复培育] 命中 lesson p_drink 展示页 OCR 兜底，直接交给 gameplay loop 接管"
                    )
                    return True

                sleep(1.0)

            # 未检测到 gameplay，恢复原模型
            logger.info("[恢复培育] 未检测到 gameplay 阶段，恢复原模型继续导航")
            app.switch_yolo_model(original_model, settle_seconds=1.0)
            return False
        except Exception:
            app.switch_yolo_model(original_model, settle_seconds=1.0)
            return False

    def _dismiss_residual_modal(self, app: "AppProcessor") -> None:
        """清理任务起跑前残留的弹窗。

        真机断点恢复时，可能停在启动弹窗、提示弹窗或确认弹窗上。
        这些弹窗会阻塞后续的 producer 模型识别，因此先尽量关闭。
        注意：主页上 require_header=False 容易把 UI 元素误判为弹窗，
        导致点到通行证等按钮，因此主页上跳过此流程。
        """
        # 检查是否在主页且无任何弹窗——只有在真正确认无弹窗时才跳过清理。
        # 注意：培育再开弹窗覆盖主页时，HOME_PRODUCE_BTN 仍可能被检测到，
        # 因此先尝试检测弹窗，只有确认无弹窗时才跳过。
        if app.latest_results and app.latest_results.exists_label(BaseUILabels.HOME_PRODUCE_BTN):
            modal_check = app.game_utils.try_get_modal(no_body=True, require_header=False)
            if modal_check is None:
                logger.debug("navigate_to_produce: 当前在主页且无残留弹窗，跳过清理")
                return

        for attempt in range(3):
            modal = app.game_utils.try_get_modal(no_body=True, require_header=False)
            if modal is None:
                return
            if _is_menu_overlay_modal(modal):
                logger.info("navigate_to_produce: 检测到菜单残留弹窗，优先点击閉じる关闭")
                close_button = find_button(app, ButtonText.CLOSE, fuzz_threshold=70)
                if close_button is not None and app.game_utils.click_element_and_wait_trigger(close_button, timeout=3.0):
                    sleep(0.5)
                    continue
                logger.warning("navigate_to_produce: 菜单残留弹窗未识别到閉じる按钮，回退到通用弹窗点击")
            if _is_produce_resume_modal(app, modal):
                logger.debug("navigate_to_produce: 起跑前命中培育再开弹窗，交由专用 retire 流处理")
                # 恢复模式下也应关闭弹窗（不恢复旧局，只是退到主页新建），避免阻塞导航
                if self._retire_resumable_produce(app):
                    return
                # retire 失败，尝试直接点击弹窗中的リタイア按钮兜底
                if self._force_close_resume_modal(app):
                    return
                # 仍失败则继续循环重试
            prefer_confirm = _is_retire_confirmation_title(getattr(modal, "modal_title", None))
            logger.info(f"navigate_to_produce: 清理残留弹窗 {attempt + 1}: {modal.modal_title!r}")
            if click_modal_action_with_retry(
                app,
                modal,
                prefer_confirm=prefer_confirm,
                retries=2,
                timeout=3.0,
                action_name="navigate_to_produce residual modal",
            ):
                sleep(0.5)
                continue
            break

    @staticmethod
    def _retire_resumable_produce(app: "AppProcessor") -> bool:
        """处理主页点 Produce 后出现的“继续上次培育”弹窗。

        这个 step 的目标是进入剧本/难度选择页，而不是恢复旧局。
        因此只要命中再开弹窗，就先执行 `リタイア -> 确认`，再由调用方重新进入。
        """
        modal = app.game_utils.try_get_modal(no_body=True)
        if not _is_produce_resume_modal(app, modal):
            return False

        logger.info("navigate_to_produce: 检测到未完成培育，先执行リタイア后重新进入")
        retire_button = find_button(app, ButtonText.RETIRE, fuzz_threshold=60)
        if retire_button is None:
            logger.warning("navigate_to_produce: 培育再开弹窗存在，但未识别到リタイア按钮，返回 False 交给 _dismiss_residual_modal 处理")
            return False

        if not app.game_utils.click_element_and_wait_trigger(retire_button, timeout=3.0):
            raise TimeoutError("点击リタイア后未触发界面变化")

        sleep(0.5)
        confirm_modal = _wait_for_modal_relaxed(app, timeout=3.0)
        if confirm_modal is not None:
            if not click_modal_action_with_retry(
                app,
                confirm_modal,
                prefer_confirm=True,
                retries=2,
                timeout=4.0,
                action_name="navigate_to_produce retire confirm",
            ):
                raise TimeoutError("未能确认リタイア弹窗")
            sleep(0.5)

        app.game_utils.wait_loading()
        wait_frame_stable(app, timeout=3.0)
        return True

    @staticmethod
    def _force_close_resume_modal(app: "AppProcessor") -> bool:
        """通过 YOLO 结果直接定位并点击培育再开弹窗的リタイア按钮（不依赖 OCR/文本识别）。

        当 PRODUCER 模型检测失败导致 OCR 无法识别按钮文本时，直接基于按钮位置和文本标签点击。
        培育再开弹窗中，リタイア 按钮位于弹窗右侧，通常是下半屏中面积最大的按钮。
        """
        results = app.latest_results
        if results is None:
            return False

        # 收集所有通用按钮标签
        all_labels = ("Universal Confirm button", "Universal Cancel button", "Universal button")
        all_boxes = []
        for label in all_labels:
            all_boxes.extend(list(results.filter_by_label(label)))

        # 找到下半屏最大按钮（リタイア 通常在弹窗右侧下半区）
        candidates = [b for b in all_boxes if b.cy > results.frame_shape[0] * 0.5]
        if not candidates:
            logger.debug("navigate_to_produce: 降级方案未找到下半屏候选按钮")
            return False

        box = max(candidates, key=lambda b: b.w * b.h)
        logger.debug(f"navigate_to_produce: 降级点击 {box.text!r} @ ({box.cx},{box.cy})")
        app.device.click_element(box)
        sleep(0.5)

        confirm_modal = _wait_for_modal_relaxed(app, timeout=3.0)
        if confirm_modal is not None:
            click_modal_action_with_retry(
                app, confirm_modal, prefer_confirm=True, retries=2, timeout=4.0,
                action_name="navigate_to_produce force close confirm",
            )
            app.game_utils.wait_loading()
            wait_frame_stable(app, timeout=3.0)
        return True
