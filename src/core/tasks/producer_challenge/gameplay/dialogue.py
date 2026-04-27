"""対話 / コミュ handler。

对话画面包括:
  - 2-3 个可选项（Universal Options）
  - 快进按钮（Fast Forward）
  - 可点击推进的剧情文本

交互模式（经 ADB 实测确认）:
  - 选项需要双击：第一次点击高亮选中，第二次点击确认。
  - 快进：单击切换自动推进。
  - 纯剧情文本：点击任意位置继续。

外出选项探查:
  外出画面会显示 2-3 个选项（如「いちごミルク -100P」「キャラメル -50P」），
  选项名是剧情台词（不在 DB 中），但:
    1. 选项框内 OCR 可提取选项名 + P 点消耗
    2. YOLO「Action Info」区域包含当前高亮选项的效果描述
    3. 通过逐个点击选项（探查），可采集所有效果描述
    4. 将 P 点成本 + 效果描述注入候选项 metadata，提供给 LLM 决策
"""

from __future__ import annotations

import cv2
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.shared.common import (
    click_relative_point,
    invoke_decision_strategy,
    ocr_text,
    probe_fast_forward_enabled_state,
    resolve_candidate_index,
)
from src.core.tasks.producer_challenge.gameplay.decision import (
    build_decision_state,
    hydrate_dialogue_candidates,
    hydrate_outing_candidates,
)
from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayHandler,
    HandlerResult,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


# ────────────────────────────────────────────────────────────
# 数据类型
# ────────────────────────────────────────────────────────────

@dataclass
class DialogueOptionCandidate:
    """对话场景中的一个可选选项。"""
    index: int
    title: str
    selected: bool
    box: Any = field(repr=False, default=None)
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueStepResult:
    """定义 DialogueStepResult 的结构化数据。

    Attributes:
        status: 步骤执行状态（如 selected/confirmed/skipped）。
        candidate: 本步骤最终选中的候选项对象。
    """
    status: str  # 状态值："selected" | "confirmed" | "fast_forward" | "advanced"
    candidate: DialogueOptionCandidate | None = None


# ────────────────────────────────────────────────────────────
# 外出探査 — 定数 / 正規表現
# ────────────────────────────────────────────────────────────

# P 点消耗模式："P-100"、"PP-100"、"P 50"、"-100"（P 被 OCR 截断）等
# ^[-ー] 用来捕获选项开头的负号（P 被拆成单独一行时）
_P_COST_RE = re.compile(r"(?:P{1,2}\s*[-ー]?\s*|^[-ー]\s*)(\d+)", re.MULTILINE)

# 外出探查：点击后等待 UI 刷新
_OUTING_PROBE_TAP_WAIT = 0.5
# 外出探査: 推理等待（Yolo再検出）
_OUTING_PROBE_INFER_WAIT = 0.3

# Debug 可視化
from src.utils.debug_tools import DebugTools
_debugger = DebugTools()
_ACTION_INFO_OCR = OCRService()
_ACTION_INFO_HINT_KEYWORDS = (
    ProduceText.ACTION_INFO_EFFECT,
    ProduceText.STAMINA,
    ProduceText.P_POINT,
    ProduceText.PARAMETER_UP,
    ProduceText.STAMINA_RECOVERY,
    ProduceText.SKILL_CARD,
    ProduceText.SKILL_CARD_REMOVE,
    ProduceText.INCREASE,
    ProduceText.ACQUIRE,
    ProduceText.RECOVERY,
    ProduceText.ENHANCE,
    ProduceText.RANDOM_KEYWORD,
    ProduceText.CHANGE,
)
_DIALOGUE_FAST_FORWARD_ENABLED_KEY = "dialogue_fast_forward_enabled"
_DIALOGUE_FAST_FORWARD_LAST_CLICK_TS_KEY = "dialogue_fast_forward_last_click_ts"


# ────────────────────────────────────────────────────────────
# 外出探査 — 工具函数
# ────────────────────────────────────────────────────────────

def _is_outing_context(app: "AppProcessor", position: str) -> bool:
    """判断当前是否為おでかけ（外出）画面。

    条件:
      - position 为 schedule_event_options（行程事件対話選項）
      - YOLO 検出到 Action Info 区域
    """
    if position not in {"schedule_event_options", "dialogue_options"}:
        return False
    info_boxes = app.latest_results.filter_by_label(ProducerLabels.PC_ACTION_INFO)
    if not info_boxes:
        return False
    # 仅在出现 P 点消耗模式时判定为外出，避免把普通周事件误判成外出。
    option_boxes = list(app.latest_results.filter_by_label(ProducerLabels.UNIVERSAL_OPTIONS))
    return any(_extract_p_cost(ocr_text(getattr(box, "frame", None))) is not None for box in option_boxes)


def _is_dialogue_option_info_context(app: "AppProcessor", position: str) -> bool:
    """判断当前是否可通过 Action Info 面板读取周事件选项效果。"""
    if position not in {"schedule_event_options", "dialogue_options"}:
        return False
    results = getattr(app, "latest_results", None)
    if results is None:
        return False
    if results.filter_by_label(ProducerLabels.PC_ACTION_INFO):
        return True
    if not results.filter_by_label(ProducerLabels.PC_PROGRESS):
        return False
    option_boxes = list(results.filter_by_label(ProducerLabels.UNIVERSAL_OPTIONS))
    if len(option_boxes) < 2:
        return False
    # 普通周事件常见“需要先点选项，信息框才刷新/出现”的场景：
    # 这里不能要求当前帧已读到效果文本，否则会直接跳过探査。
    return True


def _extract_p_cost(text: str) -> int | None:
    """从选项 OCR 文本中提取 P 点消耗。

    例:
      "PP-100 いちごミルク" → 100
      "P 50 キャラメル" → 50
      "激あまアンコ" → None (免费)
    """
    match = _P_COST_RE.search(text or "")
    return int(match.group(1)) if match else None


def _extract_action_info_description(app: "AppProcessor") -> str:
    """OCR 读取 Action Info 区域的効果描述文本。

    Action Info 区域由 YOLO 検出，包含当前高亮選項的効果描述。
    """
    def _looks_like_effect_text(raw: str) -> bool:
        """判断效果、text是否成立。

        Args:
            raw: 用于提供raw相关输入。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        normalized = re.sub(r"\s+", "", str(raw or ""))
        if not normalized:
            return False
        if any(keyword in normalized for keyword in _ACTION_INFO_HINT_KEYWORDS):
            return True
        return bool(re.search(r"[+＋\-ー]\d+", normalized))

    info_boxes = app.latest_results.filter_by_label(ProducerLabels.PC_ACTION_INFO)
    if info_boxes:
        info_box = info_boxes.first()
        frame = getattr(info_box, "frame", None)
        if frame is not None and getattr(frame, "size", 0) > 0:
            text = ocr_text(frame).strip()
            # Debug 可視化: Action Info 区域 + OCR 結果
            _debugger.add_box(
                info_box.x, info_box.y, info_box.w, info_box.h,
                color=(0, 200, 255), thickness=2, duration=3,
                label=f"ActionInfo: {text[:30]}",
            )
            if _looks_like_effect_text(text):
                return text

    def _detect_action_info_white_panel(
        frame: Any,
        option_boxes: list[Any],
    ) -> tuple[int, int, int, int] | None:
        """基于选项行几何关系检测白色 Action Info 面板。

        约束策略：
        - 用最上方选项 + 选项间平均间距估计面板中心。
        - 只接受中心偏移在阈值内的白色矩形，避免漂移到无关白块。
        """
        if frame is None or getattr(frame, "size", 0) <= 0 or not option_boxes:
            return None
        frame_h, frame_w = frame.shape[:2]
        valid_boxes: list[Any] = []
        for box in option_boxes:
            x1 = int(getattr(box, "x", 0))
            y1 = int(getattr(box, "y", 0))
            x2 = int(getattr(box, "w", 0))
            y2 = int(getattr(box, "h", 0))
            if x2 <= x1 or y2 <= y1:
                continue
            valid_boxes.append(box)
        if not valid_boxes:
            return None

        ordered = sorted(valid_boxes, key=lambda b: int(getattr(b, "y", 0)))
        top_box = ordered[0]
        top_y = int(getattr(top_box, "y", 0))
        top_h = max(1, int(getattr(top_box, "h", 0) - getattr(top_box, "y", 0)))
        top_cy = int((int(getattr(top_box, "y", 0)) + int(getattr(top_box, "h", 0))) / 2)

        centers = [
            int((int(getattr(box, "y", 0)) + int(getattr(box, "h", 0))) / 2)
            for box in ordered
        ]
        gaps = [max(0, centers[i + 1] - centers[i]) for i in range(len(centers) - 1)]
        avg_gap = int(sum(gaps) / len(gaps)) if gaps else top_h

        row_left = min(int(getattr(box, "x", 0)) for box in ordered)
        row_right = max(int(getattr(box, "w", 0)) for box in ordered)
        row_center_x = int((row_left + row_right) / 2)
        row_width = max(1, row_right - row_left)

        expected_cx = row_center_x
        expected_cy = max(0, top_cy - int(max(avg_gap * 1.35, top_h * 1.55)))
        tol_x = int(max(frame_w * 0.18, row_width * 0.38))
        tol_y = int(max(frame_h * 0.08, avg_gap * 0.95, top_h * 1.25))

        _debugger.add_box(
            max(0, expected_cx - tol_x),
            max(0, expected_cy - tol_y),
            min(frame_w, expected_cx + tol_x),
            min(frame_h, expected_cy + tol_y),
            color=(170, 220, 255),
            thickness=1,
            duration=3,
            label="ActionInfoWhiteAnchor",
        )

        # 只在选项上方区域搜索，避免把底部卡片等白块误识别成信息框。
        roi_bottom = max(0, top_y - max(4, int(frame_h * 0.008)))
        if roi_bottom <= 0:
            return None
        roi = frame[:roi_bottom, :]
        if roi.size <= 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 168), (180, 56, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        best_rect: tuple[int, int, int, int] | None = None
        best_score = -1e9
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            if w_rect <= 0 or h_rect <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + w_rect, y + h_rect

            area = float(w_rect * h_rect)
            if area < float(frame_w * frame_h) * 0.012:
                continue
            if w_rect < row_width * 0.54:
                continue
            if y2 > roi_bottom:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            dx = abs(cx - expected_cx)
            dy = abs(cy - expected_cy)
            if dx > tol_x or dy > tol_y:
                continue

            # 优先选择更靠近估计中心、且横向覆盖更接近选项行宽度的矩形。
            width_ratio = min(1.8, max(0.0, w_rect / max(float(row_width), 1.0)))
            score = (
                -dx * 1.3
                -dy * 1.9
                + width_ratio * 180.0
                + min(h_rect, int(frame_h * 0.2)) * 0.18
            )
            if score > best_score:
                best_score = score
                best_rect = (x1, y1, x2, y2)

        if best_rect is not None:
            _debugger.add_box(
                best_rect[0], best_rect[1], best_rect[2], best_rect[3],
                color=(90, 220, 255), thickness=2, duration=3,
                label="ActionInfoWhitePanel",
            )
        return best_rect

    # 兜底1：当 YOLO 漏检 Action Info 时，优先尝试“白色信息面板”几何检测。
    results = getattr(app, "latest_results", None)
    frame = getattr(app, "latest_frame", None)
    option_boxes = list(results.filter_by_label(ProducerLabels.UNIVERSAL_OPTIONS)) if results else []
    if frame is None or getattr(frame, "size", 0) <= 0 or not option_boxes:
        return ""
    white_panel = _detect_action_info_white_panel(frame, option_boxes)
    if white_panel is not None:
        x1, y1, x2, y2 = white_panel
        probe = frame[y1:y2, x1:x2].copy()
        if probe.size > 0:
            try:
                ocr_results = _ACTION_INFO_OCR.ocr(probe)
                merged = ocr_results.auto_merge_lines(
                    cy_range=max(4, int(probe.shape[0] * 0.02)),
                    width_gap=max(10, int(probe.shape[1] * 0.03)),
                )
                lines = [
                    str(getattr(line, "text", "") or "").strip()
                    for line in merged
                    if str(getattr(line, "text", "") or "").strip()
                ]
                text = "；".join(lines).strip()
                if not text:
                    text = ocr_text(probe).strip()
            except Exception:  # noqa: BLE001
                text = ocr_text(probe).strip()
            _debugger.add_box(
                x1, y1, x2, y2,
                color=(255, 205, 100), thickness=2, duration=3,
                label=f"ActionInfoWhiteOCR: {text[:30]}",
            )
            if _looks_like_effect_text(text):
                return text

    # 兜底2：按“选项上方区域”做 OCR。
    height, width = frame.shape[:2]
    top = min(int(getattr(box, "y", 0)) for box in option_boxes)
    left = min(int(getattr(box, "x", 0)) for box in option_boxes)
    right = max(int(getattr(box, "w", 0)) for box in option_boxes)
    y2 = max(0, top - max(4, int(height * 0.01)))
    panel_h = max(int(height * 0.16), min(int(height * 0.32), int(top * 0.55)))
    y1 = max(0, y2 - panel_h)
    x_pad = max(8, int(width * 0.06))
    x1 = max(0, left - x_pad)
    x2 = min(width, right + x_pad)
    if y2 <= y1 or x2 <= x1:
        return ""
    probe = frame[y1:y2, x1:x2].copy()
    if probe.size <= 0:
        return ""
    try:
        ocr_results = _ACTION_INFO_OCR.ocr(probe)
        merged = ocr_results.auto_merge_lines(
            cy_range=max(4, int(probe.shape[0] * 0.02)),
            width_gap=max(10, int(probe.shape[1] * 0.03)),
        )
        lines = [
            str(getattr(line, "text", "") or "").strip()
            for line in merged
            if str(getattr(line, "text", "") or "").strip()
        ]
        text = "；".join(lines).strip()
        if not text:
            text = ocr_text(probe).strip()
    except Exception:  # noqa: BLE001
        text = ocr_text(probe).strip()

    _debugger.add_box(
        x1, y1, x2, y2,
        color=(255, 200, 80), thickness=2, duration=3,
        label=f"ActionInfoFallback: {text[:30]}",
    )
    if _looks_like_effect_text(text):
        return text
    return ""


def _read_action_info_after_option_click(
    app: "AppProcessor",
    *,
    previous_desc: str = "",
    max_attempts: int = 5,
) -> str:
    """选项点击后等待帧刷新，再读取 Action Info，避免读到旧帧。"""
    for attempt in range(max(1, max_attempts)):
        game_utils = getattr(app, "game_utils", None)
        if game_utils is not None and hasattr(game_utils, "wait_frame_stable"):
            try:
                game_utils.wait_frame_stable(stable_count=2, timeout=1.2)
            except TypeError:
                game_utils.wait_frame_stable(stable_count=2)
            except Exception:
                pass
        time.sleep(0.12)
        desc = _extract_action_info_description(app)
        if not desc:
            continue
        # 首次读取到有效文本直接接受；若给了上一次文本，优先等待文本变化。
        if not previous_desc or desc != previous_desc or attempt >= 1:
            return desc
    return ""


def _probe_outing_options(
    app: "AppProcessor",
    candidates: List[DialogueOptionCandidate],
) -> None:
    """逐個点击おでかけ選項，采集 Action Info 効果描述 + P 点成本。

    流程:
      1. 解析每個選項的 P 点消耗（从 title OCR 中提取）
      2. 逐個点击選項 → 等待 UI 刷新 → OCR Action Info 区域
      3. 将 P 成本 + 効果描述写入候選項 metadata

    注意:
      - 第一個選項通常已預選（Action Info 已顯示其效果），先読取
      - 点击只切換高亮（不確認），安全
    """
    if not candidates:
        return

    logger.info("dialogue: おでかけ探査開始 — {} 個選項", len(candidates))

    last_desc = ""
    for candidate in candidates:
        # 提取 P 点消耗
        p_cost = _extract_p_cost(candidate.title)
        if p_cost is not None:
            candidate.metadata["p_cost"] = p_cost

        try:
            # 点击切換高亮
            app.device.click_element(candidate.box)
            time.sleep(_OUTING_PROBE_TAP_WAIT)
            time.sleep(_OUTING_PROBE_INFER_WAIT)

            # 点击后等待刷新，再读取 Action Info，避免旧帧误读。
            desc = _read_action_info_after_option_click(app, previous_desc=last_desc)
            if desc:
                candidate.metadata["outing_effect"] = desc
                last_desc = desc
                logger.debug(
                    "dialogue: おでかけ選項 #{} 効果: {}",
                    candidate.index, desc[:60],
                )
            else:
                logger.debug(
                    "dialogue: おでかけ選項 #{} Action Info 未検出",
                    candidate.index,
                )

            # Debug 可視化: 選項框 + P 成本
            cost_label = f"-{p_cost}P" if p_cost is not None else "Free"
            _debugger.add_box(
                candidate.box.x, candidate.box.y,
                candidate.box.w, candidate.box.h,
                color=(255, 165, 0), thickness=2, duration=3,
                label=f"Outing#{candidate.index} {cost_label}",
            )
        except Exception as exc:
            logger.warning(
                "dialogue: おでかけ選項 #{} 探査異常: {}",
                candidate.index, exc,
            )

    # 生成探査結果摘要
    probed = [c for c in candidates if c.metadata.get("outing_effect")]
    logger.info(
        "dialogue: おでかけ探査完了 — {}/{} 個取得効果描述",
        len(probed), len(candidates),
    )


def _probe_dialogue_option_effects(
    app: "AppProcessor",
    candidates: List[DialogueOptionCandidate],
) -> None:
    """逐个点击周事件选项，读取 Action Info 面板效果描述。"""
    if not candidates:
        return
    logger.info("dialogue: 周事件选项探査開始 — {} 個選項", len(candidates))

    last_desc = ""
    has_action_info_label = bool(
        app.latest_results.filter_by_label(ProducerLabels.PC_ACTION_INFO)
    )
    for idx, candidate in enumerate(candidates):
        try:
            app.device.click_element(candidate.box)
            time.sleep(_OUTING_PROBE_TAP_WAIT)
            time.sleep(_OUTING_PROBE_INFER_WAIT)
            # 必须在点击后等待刷新，否则容易读到点击前的旧帧。
            desc = _read_action_info_after_option_click(
                app,
                previous_desc=last_desc,
                max_attempts=5 if has_action_info_label else 6,
            )
            if desc:
                candidate.metadata["option_effect"] = desc
                last_desc = desc
        except Exception as exc:  # noqa: BLE001
            logger.warning("dialogue: 周事件选项 #{} 探査异常: {}", candidate.index, exc)

    probed = [candidate for candidate in candidates if candidate.metadata.get("option_effect")]
    logger.info(
        "dialogue: 周事件选项探査完了 — {}/{} 個取得効果描述",
        len(probed),
        len(candidates),
    )


def _enrich_outing_descriptions(candidates: List[DialogueOptionCandidate]) -> None:
    """将おでかけ探査结果注入候選項的 title / metadata，供 LLM 決策。

    增強后的 candidate:
      - metadata["p_cost"]: int — P 点消耗
      - metadata["outing_effect"]: str — 効果描述
      - metadata["description"]: str — 組合描述（用於 LLM prompt）
    """
    for candidate in candidates:
        parts: list[str] = []
        p_cost = candidate.metadata.get("p_cost")
        if p_cost is not None:
            parts.append(f"消耗{p_cost}Pポイント")
        else:
            parts.append("免费")
        effect = candidate.metadata.get("outing_effect", "")
        if effect:
            parts.append(f"効果: {effect}")
        if parts:
            candidate.metadata["description"] = " | ".join(parts)


def _enrich_dialogue_option_descriptions(candidates: List[DialogueOptionCandidate]) -> None:
    """将周事件选项效果描述注入 metadata.description，供 LLM 决策使用。"""
    for candidate in candidates:
        p_cost = candidate.metadata.get("p_cost")
        option_effect = str(
            candidate.metadata.get("dialogue_db_description")
            or candidate.metadata.get("outing_db_description")
            or candidate.metadata.get("outing_effect")
            or candidate.metadata.get("option_effect")
            or ""
        ).strip()
        parts: list[str] = []
        if p_cost is not None:
            parts.append(f"消耗{int(p_cost)}Pポイント")
        if option_effect:
            parts.append(f"効果: {option_effect}")
        if parts:
            candidate.metadata["description"] = " | ".join(parts)


# ────────────────────────────────────────────────────────────
# 采集 / 决策 / 执行
# ────────────────────────────────────────────────────────────

def collect_dialogue_option_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> List[DialogueOptionCandidate]:
    """采集屏幕上的对话选项，按从上到下排序。"""
    options = sorted(
        app.latest_results.filter_by_label(ProducerLabels.UNIVERSAL_OPTIONS),
        key=lambda o: o.cy,
    )
    pending = ctx.pending_dialogue_option_index if position == "dialogue_options" else None
    candidates = [
        DialogueOptionCandidate(
            index=idx,
            title=ocr_text(box.frame),
            selected=pending == idx,
            box=box,
        )
        for idx, box in enumerate(options)
    ]
    hydrate_dialogue_candidates(candidates)
    return candidates


def decide_dialogue_option(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: List[DialogueOptionCandidate],
    *,
    position: str,
) -> int:
    """选择哪个对话选项（策略回调或默认选第一个）。"""
    decision_state = build_decision_state(
        app,
        ctx,
        phase="dialogue",
        position=position,
        candidates=candidates,
        reason="dialogue_decision",
    )
    decision = invoke_decision_strategy(
        ctx.dialogue_strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        return resolve_candidate_index(decision, candidates)

    if (
        ctx.pending_dialogue_option_index is not None
        and 0 <= ctx.pending_dialogue_option_index < len(candidates)
    ):
        return ctx.pending_dialogue_option_index

    return 0


def _get_dialogue_stuck_count(ctx: "ProduceContext") -> int:
    """获取对话卡住计数器（同一选项连续确认但画面未变化）。"""
    return ctx.handler_state.get("dialogue_stuck_count", 0)


def _update_dialogue_stuck(ctx: "ProduceContext", option_index: int) -> int:
    """更新对话卡住状态，返回当前卡住次数。

    如果连续确认同一选项，计数递增；否则重置。
    """
    last = ctx.handler_state.get("dialogue_stuck_last_option", -1)
    if option_index == last:
        count = ctx.handler_state.get("dialogue_stuck_count", 0) + 1
    else:
        count = 0
    ctx.handler_state["dialogue_stuck_count"] = count
    ctx.handler_state["dialogue_stuck_last_option"] = option_index
    return count


def _reset_dialogue_stuck(ctx: "ProduceContext") -> None:
    """重置对话、stuck并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    ctx.handler_state.pop("dialogue_stuck_count", None)
    ctx.handler_state.pop("dialogue_stuck_last_option", None)
    ctx.handler_state.pop("dialogue_skip_indices", None)


def _get_fast_forward_buttons(app: "AppProcessor") -> list[Any]:
    """获取fast、forward、buttons并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    results = getattr(app, "latest_results", None)
    if results is None:
        return []
    buttons = list(results.filter_by_label(ProducerLabels.PLOT_FAST_FORWARD_BUTTON))
    if buttons:
        return buttons
    return list(results.filter_by_label(BaseUILabels.PLOT_FAST_FORWARD_BUTTON))


def _try_enable_story_fast_forward(
    app: "AppProcessor",
    ctx: "ProduceContext",
) -> bool:
    """剧情页按颜色判定快进状态，仅在未开启时点击。"""
    ff_buttons = _get_fast_forward_buttons(app)
    if not ff_buttons:
        ctx.handler_state.pop(_DIALOGUE_FAST_FORWARD_ENABLED_KEY, None)
        ctx.handler_state.pop(_DIALOGUE_FAST_FORWARD_LAST_CLICK_TS_KEY, None)
        return False

    ff_box = ff_buttons[0]
    enabled_state, orange_ratio = probe_fast_forward_enabled_state(
        ff_box,
        debug_tools=getattr(app, "debug_tools", None),
        debug_label="dialogue_fast_forward",
    )
    if enabled_state is True:
        ctx.handler_state[_DIALOGUE_FAST_FORWARD_ENABLED_KEY] = True
        return False
    if (
        ctx.handler_state.get(_DIALOGUE_FAST_FORWARD_ENABLED_KEY)
        and enabled_state is not False
    ):
        # 无法稳定判定时，若此前已开启则保持不点，避免开关抖动。
        return False

    now = time.monotonic()
    last_click = float(ctx.handler_state.get(_DIALOGUE_FAST_FORWARD_LAST_CLICK_TS_KEY, 0.0) or 0.0)
    if now - last_click < 0.9:
        return False
    app.device.click_element(ff_box)
    ctx.handler_state[_DIALOGUE_FAST_FORWARD_ENABLED_KEY] = True
    ctx.handler_state[_DIALOGUE_FAST_FORWARD_LAST_CLICK_TS_KEY] = now
    logger.debug("dialogue: 开启快进（orange_ratio={:.3f}）", orange_ratio)
    return True


def _set_dialogue_transition_retry_override(
    ctx: "ProduceContext",
    *,
    reason: str,
) -> None:
    """设置`dialogue_transition_retry_override`。"""
    ctx.handler_state["unknown_retry_override"] = {
        "reason": reason,
        "retry_limit": int(
            ctx.handler_state.get("dialogue_transition_unknown_retry_limit", 8) or 8
        ),
        "retry_sleep": float(
            ctx.handler_state.get("dialogue_transition_unknown_retry_sleep", 0.7) or 0.7
        ),
    }


def execute_dialogue_step(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> DialogueStepResult | None:
    """执行一步对话交互。

    - dialogue_options + 有待确认: 确认已选中选项（第 2 次点击）
    - dialogue_options + 无待确认: 选中一个选项（第 1 次点击）
    - dialogue_continue: 快进或点击推进

    卡住检测:
      当某个选项被连续确认 STUCK_THRESHOLD 次后（例如 Pポイント不足
      导致选项无法执行），自动跳过该选项尝试下一个。
    """
    STUCK_THRESHOLD = 3  # 同一选项连续确认N次视为卡住

    candidates = collect_dialogue_option_candidates(app, ctx, position=position)

    if candidates:
        # ── 刚确认过选项，等待画面切换，不要重新调 LLM ──
        just_confirmed = ctx.handler_state.get("dialogue_just_confirmed")
        if just_confirmed is not None and ctx.pending_dialogue_option_index is None:
            grace = ctx.handler_state.get("dialogue_confirm_grace", 0) + 1
            if grace <= 3:
                ctx.handler_state["dialogue_confirm_grace"] = grace
                logger.debug(
                    "dialogue: 确认后等待画面切换 ({}/3)，跳过重复推理", grace,
                )
                return DialogueStepResult(status="waiting_transition")
            # 超过等待次数，认为确认未生效，清除状态重新选择
            logger.warning("dialogue: 确认后画面未切换，重置状态重新选择")
            ctx.handler_state.pop("dialogue_just_confirmed", None)
            ctx.handler_state.pop("dialogue_confirm_grace", None)

        # ── 外出探査: 在第一次選択前采集效果描述 ──
        if (
            ctx.pending_dialogue_option_index is None
            and _is_dialogue_option_info_context(app, position)
        ):
            if _is_outing_context(app, position):
                _probe_outing_options(app, candidates)
                _enrich_outing_descriptions(candidates)
                # 外出 DB 匹配：效果描述 + P 成本 → 稳定 DB ID。
                hydrate_outing_candidates(candidates)
            else:
                _probe_dialogue_option_effects(app, candidates)
            _enrich_dialogue_option_descriptions(candidates)
            # 周事件选项在拿到效果描述后再做一次主库匹配，避免只透传 OCR 文本。
            hydrate_dialogue_candidates(candidates)

        # ── 第二次点击: 确认已选中选项 ──
        if ctx.pending_dialogue_option_index is not None:
            target_index = ctx.pending_dialogue_option_index
            if 0 <= target_index < len(candidates):
                target = candidates[target_index]
                # 检测卡住: 连续确认同一选项
                stuck_count = _update_dialogue_stuck(ctx, target_index)
                if stuck_count >= STUCK_THRESHOLD:
                    # 该选项可能无法执行（如P点不足），加入跳过列表
                    skip_set: set = ctx.handler_state.setdefault("dialogue_skip_indices", set())
                    skip_set.add(target_index)
                    logger.warning(
                        f"dialogue: 选项 {target_index} {target.title!r} "
                        f"连续确认 {stuck_count} 次未生效，跳过此选项"
                    )
                    ctx.clear_dialogue_pending()
                    # 不 return — 直接 fall through 到下面选择新选项
                else:
                    app.device.click_element(target.box)
                    ctx.record_operation(
                        "confirm_dialogue_option",
                        target=target.title or f"option_{target.index + 1}",
                        details={"index": target.index},
                    )
                    ctx.dialogue_choices_made += 1
                    ctx.clear_dialogue_pending()
                    # 标记刚确认，防止画面未切换时重复推理
                    ctx.handler_state["dialogue_just_confirmed"] = target.index
                    ctx.handler_state.pop("dialogue_confirm_grace", None)
                    return DialogueStepResult(status="confirmed", candidate=target)
            else:
                # 待确认索引超出范围 — 重置并重新选择
                ctx.clear_dialogue_pending()

        # ── 第一次点击: 选中 ──
        skip_set = ctx.handler_state.get("dialogue_skip_indices", set())
        available = [c for c in candidates if c.index not in skip_set]
        if not available:
            # 所有选项都被跳过 — 清除跳过列表，从最后一个选项开始
            logger.warning("dialogue: 所有选项均被跳过，重置跳过列表并选择最后一个选项")
            _reset_dialogue_stuck(ctx)
            available = candidates

        target_index = decide_dialogue_option(app, ctx, available, position=position)
        target = available[target_index]
        app.device.click_element(target.box)
        ctx.pending_dialogue_option_index = target.index
        ctx.record_operation(
            "select_dialogue_option",
            target=target.title or f"option_{target.index + 1}",
            details={
                "index": target.index,
                "action_id": target.action_id,
                "db_id": target.db_id,
            },
        )
        logger.debug(f"dialogue: selected option {target.index} {target.title!r}")
        return DialogueStepResult(status="selected", candidate=target)

    # ── 没有选项可见 — 快进或点击推进 ──
    # 选项消失表示对话已推进，重置卡住状态和确认等待状态
    _reset_dialogue_stuck(ctx)
    ctx.handler_state.pop("dialogue_just_confirmed", None)
    ctx.handler_state.pop("dialogue_confirm_grace", None)

    # Skip 按钮（外出剧情等未读交流）— 直接跳过
    skip_buttons = app.latest_results.filter_by_label(BaseUILabels.SKIP_BUTTON)
    if skip_buttons:
        app.device.click_element(skip_buttons.first())
        logger.debug("dialogue: skip button")
        return DialogueStepResult(status="skipped")

    if _try_enable_story_fast_forward(app, ctx):
        return DialogueStepResult(status="fast_forward")

    # 快进已开启时点击正文推进。
    if _get_fast_forward_buttons(app):
        click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="dialogue-advance")
        return DialogueStepResult(status="advanced")
    click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="dialogue-advance")
    return DialogueStepResult(status="advanced")


# ────────────────────────────────────────────────────────────
# 处理器
# ────────────────────────────────────────────────────────────

class DialogueHandler(GameplayHandler):
    """对话 / コミュ画面处理。"""

    phase_tag = "dialogue"
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
        return phase == "dialogue"

    def handle(self, app, ctx, phase, position):
        """执行处理器主逻辑并返回处理结果。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            返回执行结果对象，具体类型见函数注解。
        """
        result = execute_dialogue_step(app, ctx, position=position)
        if result is None:
            return HandlerResult.no_action("no dialogue elements")
        if result.status in {"selected", "confirmed", "fast_forward", "skipped", "advanced"}:
            _set_dialogue_transition_retry_override(
                ctx,
                reason=f"dialogue_{result.status}",
            )
        return HandlerResult.ok(f"dialogue {result.status}", sleep_after=0.6)
