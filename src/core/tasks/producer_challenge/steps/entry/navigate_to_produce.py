"""Step 1: 从主页导航到培育（プロデュース）剧本选择页面。"""

import cv2
import numpy as np
from time import sleep, time
from typing import TYPE_CHECKING

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.shared.common import normalize_text
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_modal_action_with_retry,
    detect_gameplay_phase,
    find_button,
    go_back_in_gameplay,
    wait_frame_stable,
)
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
_GAMEPLAY_SIGNAL_LABELS = (
    ProducerLabels.PC_SKIP,
    ProducerLabels.PC_TRAINING_REMAINING,
    ProducerLabels.SKILL_CARD_ACTIVE,
    ProducerLabels.SKILL_CARD_MENTAL,
    ProducerLabels.SKILL_CARD_TRAP,
    ProducerLabels.P_DRINK,
)
_GAMEPLAY_MENU_MARKERS = (
    ProduceText.GAMEPLAY_MENU_SUSPEND,
    ProduceText.GAMEPLAY_MENU_SETTINGS,
    ButtonText.RETIRE,
)
_ocr_service = OCRService()


def _is_on_scenario_page(app: "AppProcessor") -> bool:
    """当前画面能检测到任意难度标签 → 说明在剧本选择页。"""
    return any(
        app.latest_results.exists_label(lbl) for lbl in _SCENARIO_LABELS
    )


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


def _build_frame_box(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    label: str,
) -> Yolo_Box | None:
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


def _wait_for_gameplay_retire_menu(app: "AppProcessor", *, timeout: float = 3.0) -> bool:
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
    end_time = time() + timeout
    while time() < end_time:
        modal = app.game_utils.try_get_modal(no_body=True, require_header=False)
        if modal is not None:
            return modal
        sleep(0.3)
    return None


def _looks_like_active_gameplay(app: "AppProcessor") -> bool:
    results = getattr(app, "latest_results", None)
    if results is not None:
        for label in _GAMEPLAY_SIGNAL_LABELS:
            if results.exists_label(label):
                return True
    return _find_gameplay_menu_button(getattr(app, "latest_frame", None)) is not None


def _retire_active_gameplay_produce(app: "AppProcessor") -> bool:
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


def open_produce_entry_from_home(app: "AppProcessor", *, timeout: float = 10.0) -> None:
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

    # 难度（マスター/レギュラー/プロ/レジェンド）
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
    """从主页进入培育入口，并把流程带到后续步骤可接管的稳定页面。"""

    step_name = "navigate_to_produce"

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        if _is_on_scenario_page(app):
            logger.debug("已经在培育剧本选择页面")
            return True

        is_resume = getattr(ctx, "resume_interrupted", False)

        # ── 恢复模式：优先检测是否已在 gameplay 局内，直接接管 ──
        if is_resume:
            if self._try_detect_existing_gameplay(app, ctx):
                return True

        self._dismiss_residual_modal(app)

        # ── 恢复中断模式：检测到再开弹窗时恢复旧局 ──
        if is_resume:
            if self._try_resume_interrupted(app, ctx):
                return True
            # 没有中断弹窗，走正常新建流程
            logger.info("navigate_to_produce: resume_interrupted=True 但未检测到中断弹窗，走正常流程")

        if not self._retire_resumable_produce(app):
            self._dismiss_residual_modal(app)

        # 先走通用返回；若当前其实卡在 producer 局内，则退回专用退局链。
        try:
            app.game_utils.go_home()
        except RuntimeError:
            logger.warning("navigate_to_produce: 常规 go_home 失败，尝试局内旧局退出链")
            if go_back_in_gameplay(app):
                sleep(0.8)
                self._dismiss_residual_modal(app)
                try:
                    app.game_utils.go_home()
                except RuntimeError:
                    if not _retire_active_gameplay_produce(app):
                        raise
                    self._dismiss_residual_modal(app)
                    app.game_utils.go_home()
            else:
                if not _retire_active_gameplay_produce(app):
                    raise
                self._dismiss_residual_modal(app)
                app.game_utils.go_home()
        app.game_utils.wait_loading()

        # 点击培育按钮
        open_produce_entry_from_home(app, timeout=10)

        # ── 恢复中断模式：点击培育后检测到弹窗 ──
        if getattr(ctx, "resume_interrupted", False):
            sleep(1.5)
            if self._try_resume_interrupted(app, ctx):
                return True

        # 等待剧本页面出现
        for _ in range(20):
            if _is_on_scenario_page(app):
                logger.debug("成功进入剧本选择页")
                return True
            # 恢复模式下再次检测弹窗
            if getattr(ctx, "resume_interrupted", False):
                if self._try_resume_interrupted(app, ctx):
                    return True
            elif self._retire_resumable_produce(app):
                # 清掉旧局后会回到主页，需要重新点一次 Produce 进入新流程。
                if app.game_utils.wait_for_label(BaseUILabels.HOME_PRODUCE_BTN, timeout=8):
                    open_produce_entry_from_home(app, timeout=8)
                continue
            sleep(1)

        raise TimeoutError("导航到培育剧本选择页超时")

    @staticmethod
    def _try_resume_interrupted(app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """检测并恢复中断的培育。

        如果检测到「プロデュース再開」弹窗，提取信息后点击「再開する」恢复旧局。
        成功恢复后设置 ctx.resumed_from_interrupt = True，pipeline 会跳过编成步骤。
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

        # 点击「再開する」恢复旧局
        if not resume_resumable_produce(app, timeout=8.0):
            logger.warning("[恢复培育] 点击再開する失败，回退到正常流程")
            return False

        ctx.resumed_from_interrupt = True
        logger.success(
            f"[恢复培育] 成功恢复中断培育: "
            f"week={resume_info.get('current_week', '?')}/{resume_info.get('total_weeks', '?')}, "
            f"idol={resume_info.get('idol_name', '?')}"
        )
        return True

    @staticmethod
    def _try_detect_existing_gameplay(app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """resume 模式下 go_home 失败时，检测是否已在 gameplay 中。

        临时切换到 PRODUCER 模型检测画面阶段。如果检测到有效 gameplay 阶段，
        标记为恢复模式并直接返回成功，跳过后续导航步骤。
        """
        from src.constants.game.producer_gameplay import GameplayPhase
        from src.constants.yolo.model_type import YoloModelType

        original_model = getattr(app.yolo_engine, "model_type", YoloModelType.BASE_UI)
        try:
            app.yolo_engine.load_model(YoloModelType.PRODUCER)
            sleep(2.0)  # 等待首帧推理稳定

            for _ in range(5):
                phase = detect_gameplay_phase(app, ctx)
                # MODAL / STARTUP_MODALS 也属于 gameplay 内的合法阶段
                # （例如休み確認、饮料确认等模态都在培育局内）
                if phase not in {GameplayPhase.UNKNOWN, ""}:
                    ctx.resumed_from_interrupt = True
                    logger.success(
                        f"[恢复培育] 已在 gameplay 中，检测到阶段: {phase}，跳过导航"
                    )
                    return True
                sleep(1.0)

            # 未检测到 gameplay，恢复原模型
            logger.info("[恢复培育] 未检测到 gameplay 阶段，恢复原模型继续导航")
            app.yolo_engine.load_model(original_model)
            sleep(1.0)
            return False
        except Exception:
            app.yolo_engine.load_model(original_model)
            sleep(1.0)
            return False

    @staticmethod
    def _dismiss_residual_modal(app: "AppProcessor") -> None:
        """清理任务起跑前残留的弹窗。

        真机断点恢复时，可能停在启动弹窗、提示弹窗或确认弹窗上。
        这些弹窗会让 `go_home()` 的返回路径失效，因此先尽量关闭。
        注意：主页上 require_header=False 容易把 UI 元素误判为弹窗，
        导致点到通行证等按钮，因此主页上跳过此流程。
        """
        # 主页上不执行弹窗清理，避免误点通行证等 UI 元素
        if app.latest_results and app.latest_results.exists_label(BaseUILabels.HOME_PRODUCE_BTN):
            logger.debug("navigate_to_produce: 当前在主页，跳过残留弹窗清理")
            return

        for attempt in range(3):
            modal = app.game_utils.try_get_modal(no_body=True, require_header=False)
            if modal is None:
                return
            if _is_produce_resume_modal(app, modal):
                logger.debug("navigate_to_produce: 起跑前命中培育再开弹窗，交由专用 retire 流处理")
                return
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
            raise TimeoutError("检测到培育再开弹窗，但未识别到リタイア按钮")

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
