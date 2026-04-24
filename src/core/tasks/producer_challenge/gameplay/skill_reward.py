"""技能卡奖励选择 handler。

技能卡奖励画面出现在:
  - 活動支給（活动支给）
  - レッスン完成后
  - 各种事件奖励

画面显示 1-3 张可选技能卡，选中后确认按钮激活。
部分场景可选「再抽選」（re-draw）刷新候选卡。

交互模式（经 ADB 实测确认）:
   - 第一次点击卡片：高亮选中，确认按钮变为可用，信息面板显示卡名/效果。
   - 第二次点击确认按钮（受け取る）：接受卡片并推进。
   - 点击「再抽選」按钮：消耗一次再抽選机会，刷新候选卡。

卡片识别优先级:
  1. CLIP 图像记忆（高置信度、无交互延迟）
  2. 点击卡片 → 信息面板 OCR → 主数据库匹配 → 动态学习 CLIP 记忆
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from time import sleep
from typing import TYPE_CHECKING, Any, List, Sequence

import cv2

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.constants.game.text.produce_text import ProduceText
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.catalog import match_card_and_item_entries
from src.core.tasks.producer_challenge.shared.common import (
    detect_bottom_white_modal_region,
    invoke_decision_strategy,
    normalize_text,
    ocr_text,
    resolve_candidate_index,
)
from src.core.tasks.producer_challenge.gameplay.decision import (
    build_decision_state,
    hydrate_card_candidates,
    _learn_card_clip_from_db_id,
)
from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayHandler,
    HandlerResult,
)
from src.utils.logger import logger
from src.utils.runtime_paths import resolve_data_str
from src.utils.string_tools import fullwidth_to_halfwidth, normalize_ocr_jp

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_REWARD_CARD_LABELS = (
    BaseUILabels.SKILL_CARD_ACTIVE,
    BaseUILabels.SKILL_CARD_MENTAL,
    BaseUILabels.SKILL_CARD_TRAP,
    ProducerLabels.SKILL_CARD_INFO,
)

# 再抽選剩余次数 OCR 匹配正则
_REDRAW_REMAINING_RE = re.compile(ProduceText.REMAINING_COUNT_PATTERN)
_SKILL_REWARD_INFO_PANEL_LABELS = (
    ProducerLabels.SKILL_CARD_INFO,
    ProducerLabels.PC_ACTION_INFO,
)
_SKILL_REWARD_PANEL_OCR = OCRService()
_SKILL_REWARD_NAME_NOISE_RE = re.compile(r'^[\|｜\[\]「」【】\s]+|[\|｜\[\]「」【】\s]+$')
_SKILL_REWARD_EFFECT_PREFIXES = ("↑", "↓", "→", "←", "↗", "↘", "♥", "❤", "♡")
_SKILL_REWARD_EFFECT_LINE_RE = re.compile(
    r"(元気|好印象|やる気|体力|集中|好調|スコア|パラメータ).*[+\-−]\d+"
)
_SKILL_REWARD_JP_CHAR_RE = re.compile(r"[ぁ-んァ-ヶー一-龯]")
_SKILL_REWARD_TITLE_NOISE_TOKENS = (
    "受け取る",
    "スキルカード",
    "選んで",
    "ください",
    "獲得ガイド",
    "おすすめ",
    "NEW",
)
_SKILL_REWARD_SETTLE_LABELS = (
    BaseUILabels.SKILL_CARD_ACTIVE,
    BaseUILabels.SKILL_CARD_MENTAL,
    BaseUILabels.SKILL_CARD_TRAP,
)
_SKILL_REWARD_SETTLE_POLL_SLEEP = 0.12
_SKILL_REWARD_SETTLE_MAX_POLLS = 6
_SKILL_REWARD_SETTLE_STABLE_COUNT_STREAK = 2
_SKILL_REWARD_SETTLE_STABLE_BASELINE_STREAK = 2
_SKILL_REWARD_BASELINE_TOLERANCE_RATIO = 0.016
_SKILL_REWARD_BASELINE_TOLERANCE_MIN = 14
_SKILL_REWARD_BASELINE_TOLERANCE_MAX = 52
_SKILL_REWARD_WHITE_HSV_LOWER = (0, 0, 233)
_SKILL_REWARD_WHITE_HSV_UPPER = (179, 61, 255)
_SKILL_REWARD_DUPLICATE_TOLERANCE = 64
_SKILL_REWARD_LABEL_PRIORITY: dict[str, int] = {
    BaseUILabels.SKILL_CARD_ACTIVE: 0,
    BaseUILabels.SKILL_CARD_MENTAL: 0,
    BaseUILabels.SKILL_CARD_TRAP: 0,
    ProducerLabels.SKILL_CARD_INFO: 1,
}


def _normalize_skill_reward_candidate_title(text: str) -> str:
    """标准化技能卡候选标题，去除常见噪声符号。"""
    normalized = normalize_ocr_jp(fullwidth_to_halfwidth(str(text or "")))
    normalized = _SKILL_REWARD_NAME_NOISE_RE.sub("", normalized).strip()
    return normalized


def _looks_like_skill_reward_effect_text(text: str) -> bool:
    """判断文本是否像技能奖励效果描述。

    Args:
        text: 待处理文本，通常来源于 OCR 或配置。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    normalized = fullwidth_to_halfwidth(str(text or "")).strip()
    if not normalized:
        return False
    if normalized.startswith(_SKILL_REWARD_EFFECT_PREFIXES):
        return True
    return bool(_SKILL_REWARD_EFFECT_LINE_RE.search(normalized))


def _looks_like_plausible_skill_reward_name(text: str) -> bool:
    """判断标题是否像“可用于决策”的卡名，过滤明显乱码。"""
    normalized = _normalize_skill_reward_candidate_title(text)
    if len(normalized) < 2:
        return False
    if _looks_like_skill_reward_effect_text(normalized):
        return False
    if any(token in normalized for token in _SKILL_REWARD_TITLE_NOISE_TOKENS):
        return False
    return bool(_SKILL_REWARD_JP_CHAR_RE.search(normalized))


def _group_skill_reward_rows(
    boxes: Sequence[tuple[str, Any]],
    *,
    tolerance: int,
) -> list[list[tuple[str, Any]]]:
    """处理group、skill、奖励、rows并返回结果。

    Args:
        boxes: 检测框集合。
        tolerance: 用于提供tolerance相关输入。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    groups: list[list[tuple[str, Any]]] = []
    for item in sorted(boxes, key=lambda pair: int(getattr(pair[1], "cy", 0))):
        cy = int(getattr(item[1], "cy", 0))
        placed = False
        for group in groups:
            group_cy = int(sum(int(getattr(box, "cy", 0)) for _, box in group) / max(len(group), 1))
            if abs(cy - group_cy) <= tolerance:
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return groups


def _filter_skill_reward_boxes_by_row(
    app: "AppProcessor",
    boxes: list[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    """候选数量异常偏多时，只保留最可信的一行奖励卡。"""
    if len(boxes) <= 4:
        return boxes

    heights = [
        max(1, int(getattr(box, "h", 0) - getattr(box, "y", 0)))
        for _, box in boxes
    ]
    median_height = sorted(heights)[len(heights) // 2] if heights else 180
    row_tolerance = max(32, min(96, int(median_height * 0.42)))
    row_groups = _group_skill_reward_rows(boxes, tolerance=row_tolerance)
    if len(row_groups) <= 1:
        return boxes

    best_group = max(
        row_groups,
        key=lambda group: (
            len(group),
            int(sum(int(getattr(box, "cy", 0)) for _, box in group) / max(len(group), 1)),
        ),
    )
    if len(best_group) >= len(boxes):
        return boxes

    keep_set = {id(box) for _, box in best_group}
    filtered = [pair for pair in boxes if id(pair[1]) in keep_set]
    debugger = getattr(app, "debug_tools", None)
    if debugger is not None:
        for _, box in boxes:
            is_kept = id(box) in keep_set
            debugger.add_box(
                int(getattr(box, "x", 0)),
                int(getattr(box, "y", 0)),
                int(getattr(box, "w", 0)),
                int(getattr(box, "h", 0)),
                label="skill_reward_row_keep" if is_kept else "skill_reward_row_drop",
                color=(90, 210, 120) if is_kept else (255, 120, 120),
                alpha=0.08,
                duration=2.0,
                font_size=12,
            )
    logger.debug(
        "skill_reward: 候选行过滤 {} -> {} (rows={})",
        len(boxes),
        len(filtered),
        [len(group) for group in row_groups],
    )
    return filtered


def _detect_skill_reward_white_panel_fallback(
    frame: Any,
    *,
    card_boxes: Sequence[Any],
    debug_tools: Any = None,
) -> tuple[int, int, int, int] | None:
    """当通用白底检测失败时，按技能奖励场景做定向兜底。"""
    if frame is None or getattr(frame, "size", 0) <= 0 or not card_boxes:
        return None
    frame_h, frame_w = frame.shape[:2]
    valid_boxes = []
    for box in card_boxes:
        x1 = int(getattr(box, "x", 0))
        y1 = int(getattr(box, "y", 0))
        x2 = int(getattr(box, "w", 0))
        y2 = int(getattr(box, "h", 0))
        if x2 > x1 and y2 > y1:
            valid_boxes.append((x1, y1, x2, y2))
    if not valid_boxes:
        return None
    row_left = min(x1 for x1, _, _, _ in valid_boxes)
    row_right = max(x2 for _, _, x2, _ in valid_boxes)
    row_top = min(y1 for _, y1, _, _ in valid_boxes)
    row_h = max(1, max(y2 for _, _, _, y2 in valid_boxes) - row_top)
    row_w = max(1, row_right - row_left)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _SKILL_REWARD_WHITE_HSV_LOWER, _SKILL_REWARD_WHITE_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    best_rect: tuple[int, int, int, int] | None = None
    best_score = -1e9
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        if w_rect <= 0 or h_rect <= 0:
            continue
        x2 = x + w_rect
        y2 = y + h_rect
        if y >= row_top - int(row_h * 0.6):
            continue
        if y2 <= row_top + int(row_h * 0.2):
            continue
        if w_rect < frame_w * 0.58:
            continue
        if h_rect < row_h * 1.4:
            continue
        # 必须覆盖卡片行的大部分横向范围，避免误命中侧边高亮块。
        overlap_w = max(0, min(x2, row_right) - max(x, row_left))
        if overlap_w < row_w * 0.72:
            continue
        center_x = (x + x2) / 2.0
        center_penalty = abs(center_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
        vertical_bonus = max(0.0, (row_top - y) / max(row_h * 2.0, 1.0))
        score = w_rect * 0.7 + h_rect * 0.3 + vertical_bonus * 80.0 - center_penalty * 120.0
        if score > best_score:
            best_score = score
            best_rect = (
                max(0, x),
                max(0, y),
                min(frame_w, x2),
                min(frame_h, y2),
            )

    if best_rect is not None and debug_tools is not None:
        debug_tools.add_box(
            best_rect[0],
            best_rect[1],
            best_rect[2],
            best_rect[3],
            label="skill_reward_white_panel_hsv_fallback",
            color=(255, 180, 80),
            alpha=0.10,
            duration=2.5,
            font_size=13,
        )
    return best_rect


def _is_same_reward_card_box(a: Any, b: Any, *, tolerance: int) -> bool:
    """判断same、奖励、卡牌、box是否成立。

    Args:
        a: 用于提供a相关输入。
        b: 用于提供b相关输入。
        tolerance: 用于提供tolerance相关输入。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    ax1, ay1 = int(getattr(a, "x", 0)), int(getattr(a, "y", 0))
    ax2, ay2 = int(getattr(a, "w", 0)), int(getattr(a, "h", 0))
    bx1, by1 = int(getattr(b, "x", 0)), int(getattr(b, "y", 0))
    bx2, by2 = int(getattr(b, "w", 0)), int(getattr(b, "h", 0))
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return False

    acx, acy = int(getattr(a, "cx", (ax1 + ax2) // 2)), int(getattr(a, "cy", (ay1 + ay2) // 2))
    bcx, bcy = int(getattr(b, "cx", (bx1 + bx2) // 2)), int(getattr(b, "cy", (by1 + by2) // 2))
    if abs(acx - bcx) <= tolerance and abs(acy - bcy) <= tolerance:
        return True

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return False
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    iou = inter_area / float(area_a + area_b - inter_area)
    return iou >= 0.45


def _dedup_skill_reward_boxes(
    app: "AppProcessor",
    boxes: list[tuple[str, Any]],
    *,
    tolerance: int = _SKILL_REWARD_DUPLICATE_TOLERANCE,
) -> list[tuple[str, Any]]:
    """按空间位置去重奖励卡检测框，避免同一卡被重复计入合法动作。"""
    deduped: list[tuple[str, Any]] = []
    for label, box in boxes:
        replaced = False
        for idx, (kept_label, kept_box) in enumerate(deduped):
            if not _is_same_reward_card_box(box, kept_box, tolerance=tolerance):
                continue
            kept_priority = _SKILL_REWARD_LABEL_PRIORITY.get(kept_label, 99)
            current_priority = _SKILL_REWARD_LABEL_PRIORITY.get(label, 99)
            if current_priority < kept_priority:
                deduped[idx] = (label, box)
            replaced = True
            break
        if not replaced:
            deduped.append((label, box))

    debugger = getattr(app, "debug_tools", None)
    if debugger is not None:
        for idx, (_label, box) in enumerate(deduped):
            debugger.add_box(
                int(getattr(box, "x", 0)),
                int(getattr(box, "y", 0)),
                int(getattr(box, "w", 0)),
                int(getattr(box, "h", 0)),
                label=f"skill_reward_candidate_{idx}",
                color=(90, 200, 255),
                alpha=0.08,
                duration=1.8,
                font_size=12,
            )
    return deduped


# ────────────────────────────────────────────────────────────
# 数据类型
# ────────────────────────────────────────────────────────────

@dataclass
class SkillRewardCandidate:
    """定义 SkillRewardCandidate 的结构化数据。

    Attributes:
        index: 候选项在当前列表中的序号（通常从上到下或从左到右）。
        label: 用于界面展示或日志输出的短标签文本。
        title: 候选项主标题文本，通常来自 OCR 或预设文案。
        selected: 是否为当前已选中项（True 表示已选中）。
        box: 候选项对应的检测框，用于点击、裁剪和可视化调试。
        action_id: 标准化动作标识，用于在决策层与执行层之间关联同一操作。
        db_id: 数据库中的实体 ID；为空通常表示当前候选项尚未完成实体识别。
        source: 候选项来源标记（如 OCR、DB、fallback），便于排查识别链路。
        confidence: 当前识别或匹配结果的置信度，数值越高代表结果越可靠。
        metadata: 扩展元数据，保存额外识别信息与决策辅助字段。
    """
    index: int
    label: str
    title: str
    selected: bool
    box: Any = field(repr=False, default=None)
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRewardStepResult:
    """定义 SkillRewardStepResult 的结构化数据。

    Attributes:
        status: 步骤执行状态（如 selected/confirmed/skipped）。
        candidate: 本步骤最终选中的候选项对象。
    """
    status: str  # 状态值："selected" | "confirmed" | "redrawn"
    candidate: SkillRewardCandidate | None = None


# ────────────────────────────────────────────────────────────
# 信息面板 OCR — 从选中卡片的详情面板读取卡名
# ────────────────────────────────────────────────────────────

def _extract_card_name_from_info_panel(
    app: "AppProcessor",
    card_boxes: list[Any],
) -> str:
    """从技能卡信息面板 OCR 读取卡名。

    信息面板位于卡片缩略图上方，显示当前选中/高亮卡片的名称和效果。
    卡名在面板顶部，使用较大字体居中显示。

    Args:
        app: 设备接口
        card_boxes: YOLO 检测到的卡片 box 列表（用于定位面板区域）

    Returns:
        OCR 识别出的卡名文本（可能为空）
    """
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return ""
    h = frame.shape[0]
    debugger = getattr(app, "debug_tools", None)
    panel_rect = detect_bottom_white_modal_region(
        frame,
        row_boxes=card_boxes,
        debug_tools=debugger,
        debug_label="skill_reward_white_panel",
    )
    if panel_rect is not None:
        # 面板上沿应明显高于卡片行，否则大概率是误截到卡片行本身。
        row_top = min(int(getattr(box, "y", h)) for box in card_boxes) if card_boxes else h
        row_h = max(
            1,
            max(int(getattr(box, "h", row_top)) for box in card_boxes) - row_top,
        ) if card_boxes else max(1, int(h * 0.12))
        max_panel_top = row_top - max(30, int(row_h * 0.55))
        if panel_rect[1] > max_panel_top:
            panel_rect = None
    if panel_rect is None:
        panel_rect = _detect_skill_reward_white_panel_fallback(
            frame,
            card_boxes=card_boxes,
            debug_tools=debugger,
        )

    if panel_rect is not None:
        px1, py1, px2, py2 = panel_rect
        if px2 > px1 + 20 and py2 > py1 + 20:
            panel_crop = frame[py1:py2, px1:px2]
            ocr_result_list = _SKILL_REWARD_PANEL_OCR.ocr(panel_crop)
            merged_lines = (
                list(
                    ocr_result_list.auto_merge_lines(
                        cy_range=max(8, int(panel_crop.shape[0] * 0.02)),
                        width_gap=max(12, int(panel_crop.shape[1] * 0.04)),
                    )
                )
                if hasattr(ocr_result_list, "auto_merge_lines")
                else list(ocr_result_list)
            )
            if merged_lines:
                panel_h = max(1, py2 - py1)
                panel_w = max(1, px2 - px1)
                title_bottom = int(panel_h * 0.42)
                if card_boxes:
                    card_top = min(int(getattr(box, "y", h)) for box in card_boxes)
                    boundary = int(card_top - py1 - panel_h * 0.05)
                    if boundary > int(panel_h * 0.15):
                        title_bottom = min(title_bottom, boundary)
                best_line: tuple[float, str, tuple[int, int, int, int]] | None = None
                for line in merged_lines:
                    line_text = _normalize_skill_reward_candidate_title(str(getattr(line, "text", "") or ""))
                    if not line_text:
                        continue
                    line_y = int(getattr(line, "y", 0))
                    line_h = max(1, int(getattr(line, "h", 0)))
                    line_cy = int(getattr(line, "cy", line_y + line_h // 2))
                    if line_cy < 0 or line_cy > title_bottom:
                        continue
                    if not _looks_like_plausible_skill_reward_name(line_text):
                        continue
                    line_x = int(getattr(line, "x", 0))
                    line_w = max(1, int(getattr(line, "w", 0)))
                    line_cx = int(getattr(line, "cx", line_x + line_w // 2))
                    center_bias = 1.0 - min(1.0, abs(line_cx - panel_w / 2.0) / max(panel_w / 2.0, 1.0))
                    width_score = min(1.0, line_w / max(panel_w * 0.35, 1.0))
                    top_score = 1.0 - min(1.0, line_cy / max(title_bottom, 1))
                    score = center_bias * 3.0 + width_score * 2.0 + top_score
                    if best_line is None or score > best_line[0]:
                        best_line = (
                            score,
                            line_text,
                            (px1 + line_x, py1 + line_y, px1 + line_x + line_w, py1 + line_y + line_h),
                        )
                if best_line is not None:
                    _, card_name, (bx1, by1, bx2, by2) = best_line
                    if debugger is not None:
                        debugger.add_box(
                            bx1,
                            by1,
                            bx2,
                            by2,
                            label=f"skill_reward_title:{card_name[:24]}",
                            color=(80, 220, 120),
                            alpha=0.16,
                            duration=2.5,
                            font_size=14,
                        )
                    logger.debug(
                        "skill_reward: 信息面板 OCR 卡名={!r} (panel y={}..{}, x={}..{})",
                        card_name, py1, py2, px1, px2,
                    )
                    return card_name
    return ""


def _save_unresolved_skill_reward_probe(
    app: "AppProcessor",
    candidate: "SkillRewardCandidate",
    *,
    card_name: str,
) -> None:
    """保存技能卡奖励未匹配样本，便于后续人工校验和补录。"""
    card_frame = getattr(candidate.box, "frame", None)
    if card_frame is None or getattr(card_frame, "size", 0) <= 0:
        return
    collect_dir = resolve_data_str("CLIP", "unresolved_skill_reward")
    os.makedirs(collect_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"reward_{timestamp}_idx{int(candidate.index)}"
    card_path = os.path.join(collect_dir, f"{stem}_card.png")
    cv2.imwrite(card_path, card_frame)

    panel_path = ""
    panel_box = None
    results = getattr(app, "latest_results", None)
    if results is not None:
        panel_boxes: list[Any] = []
        for label in _SKILL_REWARD_INFO_PANEL_LABELS:
            panel_boxes.extend(list(results.filter_by_label(label)))
        if panel_boxes:
            panel_box = max(
                panel_boxes,
                key=lambda box: max(0, int(getattr(box, "w", 0) - getattr(box, "x", 0)))
                * max(0, int(getattr(box, "h", 0) - getattr(box, "y", 0))),
            )
    frame = getattr(app, "latest_frame", None)
    if panel_box is not None and frame is not None and getattr(frame, "size", 0) > 0:
        fh, fw = frame.shape[:2]
        x1 = max(0, int(getattr(panel_box, "x", 0)))
        y1 = max(0, int(getattr(panel_box, "y", 0)))
        x2 = min(fw, int(getattr(panel_box, "w", 0)))
        y2 = min(fh, int(getattr(panel_box, "h", 0)))
        panel_crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
        if panel_crop is not None and getattr(panel_crop, "size", 0) > 0:
            panel_path = os.path.join(collect_dir, f"{stem}_panel.png")
            cv2.imwrite(panel_path, panel_crop)

    raw_candidate_title = str(
        (getattr(candidate, "metadata", {}) or {}).get("raw_candidate_title")
        or candidate.title
        or ""
    )
    metadata = {
        "index": int(candidate.index),
        "raw_candidate_title": raw_candidate_title,
        "ocr_card_name": str(card_name or ""),
        "action_id": str(candidate.action_id or ""),
        "db_id": str(candidate.db_id or ""),
        "card_image": card_path,
        "panel_image": panel_path,
    }
    metadata_path = os.path.join(collect_dir, f"{stem}.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    logger.info(
        "skill_reward: 未匹配样本已保存 idx={} name={!r} card={} panel={}",
        candidate.index,
        card_name,
        card_path,
        panel_path or "-",
    )


def _match_skill_reward_card_entry(card_name: str, card_db: Any) -> Any | None:
    """根据信息面板 OCR 名称匹配技能卡条目。"""
    query = str(card_name or "").strip()
    if not query:
        return None

    variants: list[str] = [query]
    normalized = normalize_ocr_jp(fullwidth_to_halfwidth(query))
    stripped = normalized.lstrip("".join(_SKILL_REWARD_EFFECT_PREFIXES)).strip()
    for text in (normalized, stripped):
        if text and text not in variants:
            variants.append(text)

    for text in variants:
        found, card_entry = card_db.search_by_name(text)
        if found and card_entry is not None:
            return card_entry

    matches = [
        entry
        for entry in match_card_and_item_entries(variants, threshold=72)
        if str(entry.get("kind") or "") == "produce_card"
    ]
    if not matches:
        return None
    best = max(matches, key=lambda entry: float(entry.get("score") or 0.0))
    raw_id = str(best.get("id") or "")
    if not raw_id:
        return None
    same_id_cards = list(card_db.get_all_by_raw_id(raw_id))
    if not same_id_cards:
        return None
    if "+" in query or "＋" in query:
        return max(same_id_cards, key=lambda card: int(getattr(card, "upgradeCount", 0) or 0))
    return min(same_id_cards, key=lambda card: int(getattr(card, "upgradeCount", 0) or 0))


# ────────────────────────────────────────────────────────────
# 再抽選按钮检测
# ────────────────────────────────────────────────────────────

def _detect_redraw_info(
    app: "AppProcessor",
) -> tuple[Any | None, int]:
    """检测再抽選按钮及剩余次数。

    再抽選按钮为 YOLO 检测到的 Universal button，位于受け取る（Confirm）右侧。
    按钮上方/内部有「あとN回」文本标识剩余次数。

    Returns:
        (redraw_box, remaining_count) — 无按钮时 (None, 0)
    """
    confirm_boxes = app.latest_results.filter_by_label(ProducerLabels.CONFIRM_BUTTON)
    generic_buttons = app.latest_results.filter_by_label(BaseUILabels.BUTTON)

    # 确认按钮中心 x（用于区分左侧确认 vs 右侧再抽選）
    confirm_cx = 0
    if confirm_boxes:
        confirm_cx = confirm_boxes.first().cx

    # 再抽選按钮: 位于确认按钮右侧的 Universal button
    redraw_box = None
    for btn in generic_buttons:
        if confirm_cx and btn.cx > confirm_cx:
            redraw_box = btn
            break
    # 如果没有 Confirm button 作参照，尝试找最右侧的 button
    if redraw_box is None and generic_buttons and not confirm_boxes:
        sorted_btns = sorted(generic_buttons, key=lambda b: b.cx, reverse=True)
        redraw_box = sorted_btns[0] if sorted_btns else None

    if redraw_box is None:
        return None, 0

    # 在 OCR 按钮区域读取“あとN回”。
    btn_text = ocr_text(redraw_box.frame)
    remaining = 0
    m = _REDRAW_REMAINING_RE.search(btn_text)
    if m:
        remaining = int(m.group(1))
    else:
        # 按钮上方区域可能有剩余次数徽章
        frame = getattr(app, "latest_frame", None)
        if frame is not None:
            badge_top = max(0, getattr(redraw_box, "y", 0) - 60)
            badge_bottom = getattr(redraw_box, "y", 0) + 20
            badge_left = max(0, getattr(redraw_box, "x", 0) - 10)
            badge_right = min(frame.shape[1], getattr(redraw_box, "w", 0) + 30)
            badge_region = frame[badge_top:badge_bottom, badge_left:badge_right]
            if badge_region.size > 0:
                badge_text = ocr_text(badge_region)
                m2 = _REDRAW_REMAINING_RE.search(badge_text)
                if m2:
                    remaining = int(m2.group(1))

    # 确认按钮文本包含「再抽選」才认定（防误判）
    full_text = btn_text
    if (
        remaining > 0
        or ProduceText.REDRAW in full_text
        or ProduceText.REDRAW_SHORT in full_text
    ):
        logger.debug(
            "skill_reward: 检测到再抽選按钮 (剩余{}次, OCR={!r})",
            remaining, btn_text,
        )
        return redraw_box, remaining

    return None, 0


# ────────────────────────────────────────────────────────────
# 卡片探査 — CLIP 未命中时点击卡片读取信息面板
# ────────────────────────────────────────────────────────────

def _probe_unresolved_cards(
    app: "AppProcessor",
    candidates: list[SkillRewardCandidate],
) -> None:
    """对 CLIP 未识别的卡片执行信息面板探査。

    依次点击 CLIP 未命中的卡片，等待信息面板更新后 OCR 卡名，
    匹配主数据库并动态学习 CLIP 记忆。

    探査完成后不改变画面选中状态（最后点击的卡是最终高亮卡）。
    """
    # 找出未解析的候选项（排除再抽選等非卡片候选）
    unresolved = [
        c for c in candidates
        if not c.db_id
        and not c.metadata.get("is_redraw")
        and c.box is not None
    ]
    if not unresolved:
        return

    from src.utils.game_database_tools import GakumasDatabase_ProduceCardDataUtils
    card_db = GakumasDatabase_ProduceCardDataUtils()

    # 收集所有卡片 box（用于面板区域定位）
    card_boxes = [c.box for c in candidates if c.box and not c.metadata.get("is_redraw")]

    for candidate in unresolved:
        # 点击卡片触发信息面板显示
        app.device.click_element(candidate.box)
        sleep(0.8)

        # 等待帧刷新后 OCR，失败则重试一次
        card_name = ""
        for attempt in range(2):
            sleep(0.3)
            card_name = _extract_card_name_from_info_panel(app, card_boxes)
            if card_name:
                break
            logger.debug(
                "skill_reward: 探査卡片 #{} OCR 第{}次为空，重试",
                candidate.index, attempt + 1,
            )

        if not card_name:
            logger.debug("skill_reward: 探査卡片 #{} 信息面板 OCR 为空", candidate.index)
            # 信息面板未读出卡名时，清空缩略图噪声标题，避免把乱码送给 LLM。
            if not _looks_like_plausible_skill_reward_name(candidate.title):
                candidate.title = ""
            _save_unresolved_skill_reward_probe(app, candidate, card_name="")
            continue

        # 匹配主数据库
        card_entry = _match_skill_reward_card_entry(card_name, card_db)
        if card_entry is None:
            logger.debug(
                "skill_reward: 探査卡片 #{} 名称 {!r} 未匹配数据库",
                candidate.index, card_name,
            )
            # 即使 DB 未匹配，也更新 title（比缩略图 OCR 更可靠）
            candidate.title = card_name
            _save_unresolved_skill_reward_probe(app, candidate, card_name=card_name)
            continue

        # 匹配成功 → 更新候选项元数据
        card_id = str(card_entry.id)
        upgrade_count = int(getattr(card_entry, "upgradeCount", 0) or 0)
        from src.core.tasks.producer_challenge.gameplay.decision import (
            _enrich_card_metadata,
            _apply_resolution,
            CandidateResolution,
        )
        metadata = _enrich_card_metadata(card_id, upgrade_count=upgrade_count)
        display_name = metadata.get("display_name") or card_name
        resolution = CandidateResolution(
            action_id=f"produce_card:{card_id}:{upgrade_count}",
            candidate_type="produce_card",
            db_id=card_id,
            display_name=str(display_name),
            source="info_panel_ocr",
            confidence=0.85,
            metadata=metadata,
        )
        _apply_resolution(candidate, resolution)

        # 动态学习 CLIP 记忆（使用卡片缩略图）
        card_frame = getattr(candidate.box, "frame", None)
        if card_frame is not None:
            _learn_card_clip_from_db_id(app, card_frame, card_id, upgrade_count=upgrade_count)
            logger.info(
                "skill_reward: 探査卡片 #{} {!r} → DB匹配 {} + CLIP学习完成",
                candidate.index, card_name, card_id,
            )


# ────────────────────────────────────────────────────────────
# 采集 / 决策 / 执行
# ────────────────────────────────────────────────────────────

def _collect_skill_reward_card_center_ys(results: Any) -> list[int]:
    """收集skill、奖励、卡牌、center、ys并返回结果。

    Args:
        results: 用于提供results相关输入。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    if results is None:
        return []
    centers: list[int] = []
    for label in _SKILL_REWARD_SETTLE_LABELS:
        for box in results.filter_by_label(label):
            cy = getattr(box, "cy", None)
            if isinstance(cy, (int, float)):
                centers.append(int(cy))
    centers.sort()
    return centers


def _resolve_skill_reward_baseline_tolerance(app: "AppProcessor") -> tuple[int, int]:
    """解析并确定`skill_reward_baseline_tolerance`。"""
    frame = getattr(app, "latest_frame", None)
    frame_h = 0
    if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 1:
        frame_h = int(frame.shape[0] or 0)
    if frame_h <= 0:
        tolerance = 24
    else:
        tolerance = int(frame_h * _SKILL_REWARD_BASELINE_TOLERANCE_RATIO)
        tolerance = max(
            _SKILL_REWARD_BASELINE_TOLERANCE_MIN,
            min(_SKILL_REWARD_BASELINE_TOLERANCE_MAX, tolerance),
        )
    stable_delta = max(8, tolerance // 3)
    return tolerance, stable_delta


def _is_skill_reward_card_baseline_settled(
    center_ys: list[int],
    *,
    tolerance: int,
) -> tuple[bool, int | None]:
    """判断skill、奖励、卡牌、基线、settled是否成立。

    Args:
        center_ys: 用于提供center、ys相关输入。
        tolerance: 用于提供tolerance相关输入。

    Returns:
        tuple[bool, int | None]: 返回值类型见注解。
    """
    if not center_ys:
        return False, None
    baseline = int(center_ys[len(center_ys) // 2])
    if len(center_ys) <= 1:
        return True, baseline
    outliers = [y for y in center_ys if abs(y - baseline) > tolerance]
    if not outliers:
        return True, baseline
    if len(outliers) == 1:
        # 选中卡片会轻微上浮，允许单个上浮离群点。
        floating_offset = baseline - outliers[0]
        if floating_offset > 0 and floating_offset <= tolerance * 2:
            return True, baseline
    return False, baseline


def _wait_skill_reward_cards_settle(
    app: "AppProcessor",
    *,
    position: str,
) -> None:
    """等待skill、奖励、卡牌、settle并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    if position not in {"skill_reward_idle", "skill_reward_selected"}:
        return
    centers = _collect_skill_reward_card_center_ys(getattr(app, "latest_results", None))
    if len(centers) < 2:
        return

    observed_max_count = len(centers)
    last_count = None
    stable_count_streak = 0
    stable_baseline_streak = 0
    last_baseline = None

    for poll_idx in range(_SKILL_REWARD_SETTLE_MAX_POLLS):
        centers = _collect_skill_reward_card_center_ys(getattr(app, "latest_results", None))
        if len(centers) < 2:
            return

        tolerance, stable_delta = _resolve_skill_reward_baseline_tolerance(app)
        settled, baseline = _is_skill_reward_card_baseline_settled(centers, tolerance=tolerance)
        current_count = len(centers)
        observed_max_count = max(observed_max_count, current_count)
        if last_count is not None and current_count == last_count:
            stable_count_streak += 1
        else:
            stable_count_streak = 0
        last_count = current_count

        if settled and baseline is not None and current_count == observed_max_count:
            if last_baseline is not None and abs(baseline - last_baseline) <= stable_delta:
                stable_baseline_streak += 1
            else:
                stable_baseline_streak = 0
            last_baseline = baseline
        else:
            stable_baseline_streak = 0
            last_baseline = baseline if settled else None

        if (
            stable_count_streak >= _SKILL_REWARD_SETTLE_STABLE_COUNT_STREAK
            and stable_baseline_streak >= _SKILL_REWARD_SETTLE_STABLE_BASELINE_STREAK
        ):
            debugger = getattr(app, "debug_tools", None)
            frame = getattr(app, "latest_frame", None)
            if debugger is not None and frame is not None and baseline is not None:
                frame_h, frame_w = frame.shape[:2]
                debugger.add_box(
                    0,
                    max(0, baseline - 2),
                    max(1, frame_w - 1),
                    min(frame_h - 1, baseline + 2),
                    label=f"skill_reward_baseline:{baseline}",
                    color=(90, 210, 120),
                    alpha=0.10,
                    duration=2.0,
                    font_size=12,
                )
            return
        if poll_idx + 1 >= _SKILL_REWARD_SETTLE_MAX_POLLS:
            break
        sleep(_SKILL_REWARD_SETTLE_POLL_SLEEP)


def collect_skill_reward_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> List[SkillRewardCandidate]:
    """采集屏幕上的技能卡奖励选项，按左到右排序。"""
    boxes: list[tuple[str, Any]] = []
    for label in _REWARD_CARD_LABELS:
        for box in app.latest_results.filter_by_label(label):
            boxes.append((label, box))
    boxes = _dedup_skill_reward_boxes(app, boxes)
    boxes = _filter_skill_reward_boxes_by_row(app, boxes)
    boxes.sort(key=lambda pair: pair[1].cx)

    pending = ctx.pending_skill_reward_index if position == "skill_reward_selected" else None
    candidates: list[SkillRewardCandidate] = []
    for idx, (label, box) in enumerate(boxes):
        raw_title = str(ocr_text(box.frame) or "")
        normalized_title = _normalize_skill_reward_candidate_title(raw_title)
        title = normalized_title if _looks_like_plausible_skill_reward_name(normalized_title) else ""
        candidates.append(
            SkillRewardCandidate(
                index=idx,
                label=label,
                title=title,
                selected=pending == idx,
                box=box,
                metadata={"raw_candidate_title": normalized_title or raw_title},
            )
        )
    # CLIP 识别 + OCR fallback（不含信息面板探査）
    hydrate_card_candidates(app, candidates)
    return candidates


def _append_redraw_candidate(
    app: "AppProcessor",
    candidates: list[SkillRewardCandidate],
) -> tuple[Any | None, int]:
    """检测再抽選并追加为特殊候选项。

    Returns:
        (redraw_box, remaining_count) — 无按钮时 (None, 0)
    """
    redraw_box, remaining = _detect_redraw_info(app)
    if redraw_box is None or remaining <= 0:
        return None, 0

    redraw_index = len(candidates)
    candidates.append(SkillRewardCandidate(
        index=redraw_index,
        label="redraw",
        title=(
            f"{ProduceText.REDRAW}"
            f"{ProduceText.REDRAW_REMAINING_DISPLAY_TEMPLATE.format(remaining=remaining)}"
        ),
        selected=False,
        box=redraw_box,
        action_id="skill_reward:redraw",
        db_id="",
        source="ui_detection",
        confidence=1.0,
        metadata={
            "is_redraw": True,
            "redraw_remaining": remaining,
            "candidate_type": "skill_reward_redraw",
        },
    ))
    return redraw_box, remaining


def decide_skill_reward(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: List[SkillRewardCandidate],
    *,
    position: str,
) -> int:
    """决策skill、奖励并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        candidates: 候选项列表，供策略或规则选择目标动作。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        int: 计算得到的数值结果。
    """
    decision_state = build_decision_state(
        app,
        ctx,
        phase="skill_reward",
        position=position,
        candidates=candidates,
        reason="skill_reward_decision",
    )
    decision = invoke_decision_strategy(
        ctx.skill_reward_strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        return resolve_candidate_index(decision, candidates)

    if (
        ctx.pending_skill_reward_index is not None
        and 0 <= ctx.pending_skill_reward_index < len(candidates)
    ):
        return ctx.pending_skill_reward_index

    return 0


def _infer_selected_skill_reward_index(candidates: Sequence[SkillRewardCandidate]) -> int | None:
    """根据卡片纵向位移推断当前已选中的奖励卡索引。"""
    visual_cards = [
        (idx, int(getattr(candidate.box, "cy", 0)))
        for idx, candidate in enumerate(candidates)
        if candidate.box is not None and not candidate.metadata.get("is_redraw")
    ]
    if len(visual_cards) < 2:
        return None
    visual_cards.sort(key=lambda item: item[1])
    first_idx, first_cy = visual_cards[0]
    _, second_cy = visual_cards[1]
    if (second_cy - first_cy) >= 24:
        return first_idx
    return None


def _click_confirm_button(app: "AppProcessor") -> bool:
    """点击confirm、按钮并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    confirm_boxes = app.latest_results.filter_by_label(ProducerLabels.CONFIRM_BUTTON)
    if confirm_boxes:
        app.device.click_element(confirm_boxes.first())
        return True
    # 回退：Confirm 漏检时，仅点击文本明确为“受け取る”的按钮，避免误点“再抽選”。
    buttons = app.latest_results.filter_by_label(BaseUILabels.BUTTON)
    if buttons:
        receive_candidates: list[Any] = []
        for button in buttons:
            text = normalize_text(ocr_text(getattr(button, "frame", None)))
            if not text:
                continue
            is_redraw = normalize_text(ProduceText.REDRAW) in text or normalize_text(ProduceText.REDRAW_SHORT) in text
            is_receive = normalize_text(ProduceText.RECEIVE) in text
            if is_receive and not is_redraw:
                receive_candidates.append(button)
        if receive_candidates:
            app.device.click_element(max(receive_candidates, key=lambda b: int(getattr(b, "cy", 0))))
            return True
    return False


def execute_skill_reward_step(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> SkillRewardStepResult | None:
    """执行一步技能卡奖励交互。

    - skill_reward_selected: 点击确认按钮（第 2 步）
    - skill_reward_idle: 选择一张卡（第 1 步），可选再抽選
    """
    _wait_skill_reward_cards_settle(app, position=position)

    if position == "skill_reward_selected":
        # selected 语义是“已选中待确认”，此阶段不应再次触发 LLM 选卡。
        if ctx.pending_skill_reward_index is None:
            logger.debug("skill_reward: 无待确认卡片，尝试从当前画面推断已选中目标")
            candidates = collect_skill_reward_candidates(app, ctx, position=position)
            if candidates:
                target: SkillRewardCandidate | None = None
                reward_candidates = [c for c in candidates if not c.metadata.get("is_redraw")]
                if len(reward_candidates) == 1:
                    target = reward_candidates[0]
                    logger.debug("skill_reward: selected 场景单卡已高亮，直接确认")
                else:
                    inferred_index = _infer_selected_skill_reward_index(candidates)
                    if inferred_index is not None:
                        target = candidates[inferred_index]
                        logger.debug("skill_reward: 根据纵向位移推断已选中卡片 idx={}", target.index)

                if target is None:
                    logger.warning(
                        "skill_reward: selected 场景无法可靠推断已选中卡片，直接确认当前选中态"
                    )
                else:
                    ctx.pending_skill_reward_index = target.index
                    ctx.pending_skill_reward_label = target.title or target.label or target.action_id
                    ctx.handler_state["pending_skill_reward_db_id"] = target.db_id or ""

        if not _click_confirm_button(app):
            return None
        logger.debug(f"skill_reward: 确认选择 index={ctx.pending_skill_reward_index}")
        acquired_db_id = str(ctx.handler_state.get("pending_skill_reward_db_id") or "")
        ctx.record_operation(
            "confirm_skill_reward",
            target=ctx.pending_skill_reward_label or "skill_reward",
            details={"index": ctx.pending_skill_reward_index, "db_id": acquired_db_id},
        )
        if acquired_db_id:
            ctx.mutate_deck_acquire(
                acquired_db_id,
                kind="produce_card",
                name=ctx.pending_skill_reward_label or "",
                source="skill_reward",
            )
        ctx.clear_skill_reward_pending()
        # 确认领取后会有展示/过渡动画，设置重试容忍
        ctx.handler_state["unknown_retry_override"] = {
            "reason": "skill_reward_confirmed_transition",
            "retry_limit": 15,
            "retry_sleep": 1.0,
        }
        return SkillRewardStepResult(status="confirmed")

    # ── skill_reward_idle: 选卡 / 再抽選 ──
    candidates = collect_skill_reward_candidates(app, ctx, position=position)
    if not candidates:
        return None

    # CLIP 未命中的卡片 → 信息面板探査（点击读取卡名 + DB匹配 + CLIP学习）
    has_unresolved = any(
        not c.db_id and not c.metadata.get("is_redraw")
        for c in candidates
    )
    if has_unresolved:
        _probe_unresolved_cards(app, candidates)
        # 探査完成后等待 YOLO 引擎更新帧（探査过程中点击了卡片）
        sleep(0.3)

    # 检测再抽選按钮并追加为候选项
    _append_redraw_candidate(app, candidates)

    target_index = decide_skill_reward(app, ctx, candidates, position=position)
    target = candidates[target_index]

    # ── 再抽選: 点击再抽選按钮刷新候选卡 ──
    if target.metadata.get("is_redraw"):
        app.device.click_element(target.box)
        remaining = target.metadata.get("redraw_remaining", 0)
        ctx.record_operation(
            "skill_reward_redraw",
            target=ProduceText.REDRAW,
            details={"remaining_after": max(0, remaining - 1)},
        )
        logger.info("skill_reward: 执行再抽選 (剩余{}回→{}回)", remaining, max(0, remaining - 1))
        # 清除 pending，下次循环重新采集新卡
        ctx.clear_skill_reward_pending()
        return SkillRewardStepResult(status="redrawn", candidate=target)

    # ── 普通选卡: 点击卡片高亮选中 ──
    app.device.click_element(target.box)
    ctx.pending_skill_reward_index = target.index
    ctx.pending_skill_reward_label = target.title or target.label or target.action_id
    ctx.handler_state["pending_skill_reward_db_id"] = target.db_id or ""
    ctx.record_operation(
        "select_skill_reward",
        target=ctx.pending_skill_reward_label,
        details={
            "index": target.index,
            "label": target.label,
            "action_id": target.action_id,
            "db_id": target.db_id,
        },
    )
    logger.debug(f"skill_reward: selected {target.index} {target.title!r}")
    return SkillRewardStepResult(status="selected", candidate=target)


# ────────────────────────────────────────────────────────────
# 处理器
# ────────────────────────────────────────────────────────────

class SkillRewardHandler(GameplayHandler):
    """技能卡奖励选择画面处理。"""

    phase_tag = "skill_reward"
    priority = 50

    def can_handle(self, app, ctx, phase, position):
        """判断当前画面是否应由该处理器接管。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        return phase == "skill_reward"

    def handle(self, app, ctx, phase, position):
        # 展示画面（单卡获得/强化演出 / 记忆效果）：点击空白区域推进
        """执行处理器主逻辑并返回处理结果。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            返回执行结果对象，具体类型见函数注解。
        """
        if position == "skill_reward_showcase":
            from src.core.tasks.producer_challenge.shared.common import click_relative_point
            click_relative_point(app, x_ratio=0.5, y_ratio=0.88, label="skill_reward_showcase_advance")
            logger.info("skill_reward: 展示画面，点击空白推进")
            # 展示消失后常伴随切页动画，给更长的 unknown 重试窗口
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "skill_reward_showcase_transition",
                "retry_limit": int(ctx.handler_state.get("skill_reward_transition_unknown_retry_limit", 15)),
                "retry_sleep": float(ctx.handler_state.get("skill_reward_transition_unknown_retry_sleep", 1.0)),
            }
            return HandlerResult.ok("skill_reward showcase advance", sleep_after=1.0)
        # 连续 idle 状态选择卡片但无法进入 selected → 可能是展示画面，点击空白推进
        if position == "skill_reward_idle":
            streak = ctx.handler_state.get("skill_reward_idle_streak", 0) + 1
            ctx.handler_state["skill_reward_idle_streak"] = streak
            ctx.handler_state["skill_reward_selected_streak"] = 0
            if streak >= 4:
                logger.info(f"skill_reward: 连续{streak}次 idle，判定为展示画面，点击空白推进")
                ctx.handler_state["skill_reward_idle_streak"] = 0
                from src.core.tasks.producer_challenge.shared.common import click_relative_point
                # 点击对话框区域（卡片下方），避免点击卡片本身触发详情
                click_relative_point(app, x_ratio=0.5, y_ratio=0.88, label="skill_reward_advance")
                return HandlerResult.ok("skill_reward advance (display)", sleep_after=1.0)
        elif position == "skill_reward_selected":
            streak = ctx.handler_state.get("skill_reward_selected_streak", 0) + 1
            ctx.handler_state["skill_reward_selected_streak"] = streak
            ctx.handler_state["skill_reward_idle_streak"] = 0
            # 连续3次确认都无进展 → 强制重新选卡
            if streak >= 3:
                logger.info(f"skill_reward: 连续{streak}次 selected 无进展，强制重新选卡")
                ctx.handler_state["skill_reward_selected_streak"] = 0
                ctx.pending_skill_reward_index = None  # 重置以触发重新选卡
        else:
            ctx.handler_state["skill_reward_idle_streak"] = 0
            ctx.handler_state["skill_reward_selected_streak"] = 0

        result = execute_skill_reward_step(app, ctx, position=position)
        if result is None:
            return HandlerResult.no_action("no skill_reward elements")

        sleep_time = 1.2 if result.status == "redrawn" else 0.8
        return HandlerResult.ok(f"skill_reward {result.status}", sleep_after=sleep_time)
