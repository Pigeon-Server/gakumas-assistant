"""P飲料选择 handler。

P 饮料可在以下场景选择:
  - 周间奖励发放（独立的 p_drink 阶段）
  - レッスン/試験内底栏（由 lesson handler 单独处理）

交互模式（经 ADB 实测确认）:
  - 第一次点击饮料：橙色选框高亮，底部受け取る按钮变为可用。
  - 第二次点击确认按钮：接受饮料并推进。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.shared.common import (
    detect_bottom_white_modal_region,
    invoke_decision_strategy,
    ocr_text,
    resolve_candidate_index,
)
from src.core.tasks.producer_challenge.gameplay.decision import (
    _apply_resolution,
    build_decision_state,
    hydrate_p_drink_candidates,
    resolve_produce_drink_identity,
    score_produce_drink_metadata,
)
from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayHandler,
    HandlerResult,
)
from src.core.inference.ocr_engine import OCRService
from src.utils.logger import logger
from src.utils.string_tools import fullwidth_to_halfwidth, normalize_ocr_jp

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_ocr = OCRService()


# ────────────────────────────────────────────────────────────
# 数据类型
# ────────────────────────────────────────────────────────────

@dataclass
class PDrinkCandidate:
    """定义 PDrinkCandidate 的结构化数据。

    Attributes:
        index: 候选项在当前列表中的序号（通常从上到下或从左到右）。
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
    title: str
    selected: bool
    box: Any = field(repr=False, default=None)
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PDrinkStepResult:
    """定义 PDrinkStepResult 的结构化数据。

    Attributes:
        status: 步骤执行状态（如 selected/confirmed/skipped）。
        candidate: 本步骤最终选中的候选项对象。
    """
    status: str  # 状态值："selected" | "confirmed"
    candidate: PDrinkCandidate | None = None


@dataclass
class PDrinkLimitActionCandidate:
    """定义 PDrinkLimitActionCandidate 的结构化数据。

    Attributes:
        index: 候选项在当前列表中的序号（通常从上到下或从左到右）。
        title: 候选项主标题文本，通常来自 OCR 或预设文案。
        kind: 候选项类型标签，用于规则筛选与优先级判断。
        box: 候选项对应的检测框，用于点击、裁剪和可视化调试。
        action_id: 标准化动作标识，用于在决策层与执行层之间关联同一操作。
        db_id: 数据库中的实体 ID；为空通常表示当前候选项尚未完成实体识别。
        source: 候选项来源标记（如 OCR、DB、fallback），便于排查识别链路。
        confidence: 当前识别或匹配结果的置信度，数值越高代表结果越可靠。
        metadata: 扩展元数据，保存额外识别信息与决策辅助字段。
    """
    index: int
    title: str
    kind: str
    box: Any = field(repr=False, default=None)
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_pending_p_drink_payload(candidate: PDrinkCandidate) -> dict[str, Any]:
    """规范化`pending_p_drink_payload`。"""
    metadata = dict(candidate.metadata or {})
    return {
        "action_id": candidate.action_id,
        "db_id": candidate.db_id,
        "title": candidate.title,
        "display_name": str(metadata.get("display_name") or candidate.title or ""),
        "description": str(metadata.get("description") or ""),
        "effect_types": list(metadata.get("effect_types", []) or []),
        "rarity": str(metadata.get("rarity") or ""),
    }


def _remember_pending_new_p_drink(
    ctx: "ProduceContext",
    candidate: PDrinkCandidate,
) -> None:
    """记录`pending_new_p_drink`到上下文状态。"""
    ctx.handler_state[_PENDING_NEW_P_DRINK_STATE_KEY] = _normalize_pending_p_drink_payload(candidate)


def _get_pending_new_p_drink(ctx: "ProduceContext" | None) -> dict[str, Any]:
    """获取pending、new、p、饮料并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        dict: 结构化结果字典。
    """
    if ctx is None:
        return {}
    payload = ctx.handler_state.get(_PENDING_NEW_P_DRINK_STATE_KEY, {})
    return dict(payload or {})


def _drink_display_name(payload: dict[str, Any]) -> str:
    """根据候选数据生成人类可读的饮料显示名。"""
    return str(
        payload.get("display_name")
        or payload.get("title")
        or payload.get("name")
        or payload.get("db_id")
        or ""
    ).strip()


def _score_drink_payload(ctx: "ProduceContext", payload: dict[str, Any]) -> float:
    """计算`drink_payload`得分。"""
    return score_produce_drink_metadata(
        payload,
        phase="lesson",
        stamina=int(ctx.hud_stamina or 0),
        max_stamina=int(ctx.hud_max_stamina or 0),
        remaining_turns=int(ctx.parameter_state.get("remaining_turns") or 0),
    )


def _select_p_drink_limit_preference(
    ctx: "ProduceContext",
    candidates: list[PDrinkLimitActionCandidate],
) -> tuple[PDrinkLimitActionCandidate, str, float]:
    """选择p、饮料、limit、preference并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        tuple[PDrinkLimitActionCandidate, str, float]: 返回值类型见注解，语义由函数用途决定。
    """
    skip_candidate = next(
        (candidate for candidate in candidates if candidate.kind == "skip_new_drink"),
        candidates[0],
    )
    pending_new = _get_pending_new_p_drink(ctx)
    if not pending_new:
        return skip_candidate, "当前没有保留下来的新饮料信息，先保守放弃新饮料。", 0.0

    discard_candidates = [
        candidate for candidate in candidates
        if candidate.kind == "discard_existing_drink"
    ]
    if not discard_candidates:
        return skip_candidate, "当前没有可替换的旧饮料槽位，先保守放弃新饮料。", 0.0

    new_score = _score_drink_payload(ctx, pending_new)
    scored_existing = [
        (_score_drink_payload(ctx, dict(candidate.metadata or {})), candidate)
        for candidate in discard_candidates
    ]
    worst_score, worst_candidate = min(scored_existing, key=lambda item: item[0])
    score_gap = float(new_score - worst_score)
    new_name = _drink_display_name(pending_new) or "新饮料"
    worst_name = _drink_display_name(dict(worst_candidate.metadata or {})) or worst_candidate.title
    if new_score > 0 and score_gap >= 8.0:
        return (
            worst_candidate,
            f"新饮料「{new_name}」的综合价值({new_score:.1f})明显高于当前最弱旧饮料「{worst_name}」({worst_score:.1f})，优先替换旧饮料。",
            score_gap,
        )
    return (
        skip_candidate,
        f"新饮料「{new_name}」的综合价值({new_score:.1f})没有明显高于当前最弱旧饮料「{worst_name}」({worst_score:.1f})，先保守放弃新饮料。",
        score_gap,
    )


def _annotate_p_drink_limit_preference(
    decision_state: dict[str, Any],
    *,
    preferred_index: int,
    reason: str,
) -> None:
    """补充标注p、饮料、limit、preference并返回结果。

    Args:
        decision_state: 决策快照，包含上下文、候选项与当前理由。
        preferred_index: 用于提供preferred、index相关输入。
        reason: 用于提供reason相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    label = f"候选 {preferred_index}"
    for payload in decision_state.get("candidates", []) or []:
        if int(payload.get("index", -1)) == preferred_index:
            payload["recommended"] = True
            label = str(payload.get("label") or payload.get("title") or label)
            break
    for payload in decision_state.get("llm_actions", []) or []:
        if int(payload.get("index", -1)) == preferred_index:
            payload["recommended"] = True

    stage_context = dict(decision_state.get("stage_context", {}) or {})
    stage_context["system_recommendation"] = f"系统当前推荐优先考虑：{label}。{reason}"
    decision_state["stage_context"] = stage_context
    llm_snapshot = decision_state.get("llm_snapshot")
    if isinstance(llm_snapshot, dict):
        llm_snapshot["stage_context"] = stage_context


# ────────────────────────────────────────────────────────────
# 头部 OCR / 逐个扫描
# ────────────────────────────────────────────────────────────

# 说明文本关键词——出现时表示面板尚无选中饮料
_INSTRUCTION_KEYWORDS = (
    ProduceText.SELECT_PROMPT,
    ProduceText.P_DRINK_SELECT,
)
_P_DRINK_HEADER_NOISE_TOKENS = tuple(dict.fromkeys((
    ProduceText.P_DRINK_SELECT,
    ProduceText.SELECT_PROMPT,
    ProduceText.RECEIVE,
    ProduceText.P_DRINK_DETAIL,
    ProduceText.P_DRINK_DISCARD,
)))
_P_DRINK_NAME_NOISE_RE = re.compile(r'^[\|｜\[\]「」【】\s]+|[\|｜\[\]「」【】\s]+$')
_P_DRINK_EFFECT_PREFIXES = ("↑", "↓", "→", "←", "↗", "↘", "♥", "❤", "♡")
_P_DRINK_EFFECT_LINE_RE = re.compile(
    rf"({'|'.join(map(re.escape, ProduceText.STATUS_VALUE_TOKENS))}|{re.escape(ProduceText.YARUKI)}|{re.escape(ProduceText.SCORE)}|{re.escape(ProduceText.STAMINA)}).*[+\-−]\d+"
)
_P_DRINK_JP_CHAR_RE = re.compile(r"[ぁ-んァ-ヶー一-龯]")
_LIMIT_KEYWORDS = (
    ProduceText.P_DRINK_LIMIT_NO_RECEIVE,
    ProduceText.POSSESSION_LIMIT,
)
_PENDING_NEW_P_DRINK_STATE_KEY = "pending_new_p_drink"


def _ocr_p_drink_header(
    frame,
    *,
    app: "AppProcessor" | None = None,
    drink_boxes: list[Any] | None = None,
) -> tuple[str, str]:
    """OCR P饮料面板的头部区域，获取当前选中饮料的名称和效果。

    Returns:
        (name, effect) 饮料名称和效果描述。无选中时返回 ("", "")。
    """
    def _normalize_header_text(text: str) -> str:
        """标准化header、text并返回结果。

        Args:
            text: 待处理文本，通常来源于 OCR 或配置。

        Returns:
            str: 处理后的文本结果。
        """
        cleaned = normalize_ocr_jp(fullwidth_to_halfwidth(str(text or "")))
        cleaned = _P_DRINK_NAME_NOISE_RE.sub("", cleaned).strip()
        return cleaned

    def _looks_like_effect_line(text: str) -> bool:
        """判断文本是否像效果描述行。

        Args:
            text: 待处理文本，通常来源于 OCR 或配置。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        normalized = fullwidth_to_halfwidth(str(text or "")).strip()
        if not normalized:
            return False
        if normalized.startswith(_P_DRINK_EFFECT_PREFIXES):
            return True
        return bool(_P_DRINK_EFFECT_LINE_RE.search(normalized))

    def _is_plausible_drink_name(text: str) -> bool:
        """判断文本是否可作为饮料名称。

        Args:
            text: 待处理文本，通常来源于 OCR 或配置。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        normalized = _normalize_header_text(text)
        if len(normalized) < 2:
            return False
        if _looks_like_effect_line(normalized):
            return False
        if any(token in normalized for token in _P_DRINK_HEADER_NOISE_TOKENS):
            return False
        return bool(_P_DRINK_JP_CHAR_RE.search(normalized))

    if frame is None and app is not None:
        frame = getattr(app, "latest_frame", None)
    if frame is None:
        return "", ""

    h = frame.shape[0]
    debugger = getattr(app, "debug_tools", None) if app is not None else None
    if drink_boxes is None and app is not None:
        drink_boxes = _detect_central_drinks(app)
    drink_boxes = list(drink_boxes or [])

    panel_rect = detect_bottom_white_modal_region(
        frame,
        row_boxes=drink_boxes,
        debug_tools=debugger,
        debug_label="p_drink_white_panel",
    )
    if panel_rect is not None:
        px1, py1, px2, py2 = panel_rect
        if px2 > px1 + 20 and py2 > py1 + 20:
            panel_crop = frame[py1:py2, px1:px2]
            ocr_result_list = _ocr.ocr(panel_crop)
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
                if drink_boxes:
                    drink_top = min(int(getattr(box, "y", h)) for box in drink_boxes)
                    boundary = int(drink_top - py1 - panel_h * 0.05)
                    if boundary > int(panel_h * 0.15):
                        title_bottom = min(title_bottom, boundary)

                normalized_lines = [
                    _normalize_header_text(str(getattr(line, "text", "") or ""))
                    for line in merged_lines
                ]
                full_text = " ".join(normalized_lines)
                if any(kw in full_text for kw in _INSTRUCTION_KEYWORDS):
                    return "", ""

                best_name: tuple[float, str, int, tuple[int, int, int, int]] | None = None
                for line in merged_lines:
                    line_text = _normalize_header_text(str(getattr(line, "text", "") or ""))
                    if not line_text:
                        continue
                    line_y = int(getattr(line, "y", 0))
                    line_h = max(1, int(getattr(line, "h", 0)))
                    line_cy = int(getattr(line, "cy", line_y + line_h // 2))
                    if line_cy < 0 or line_cy > title_bottom:
                        continue
                    if not _is_plausible_drink_name(line_text):
                        continue
                    line_x = int(getattr(line, "x", 0))
                    line_w = max(1, int(getattr(line, "w", 0)))
                    line_cx = int(getattr(line, "cx", line_x + line_w // 2))
                    center_bias = 1.0 - min(1.0, abs(line_cx - panel_w / 2.0) / max(panel_w / 2.0, 1.0))
                    width_score = min(1.0, line_w / max(panel_w * 0.35, 1.0))
                    top_score = 1.0 - min(1.0, line_cy / max(title_bottom, 1))
                    score = center_bias * 3.0 + width_score * 2.0 + top_score
                    if best_name is None or score > best_name[0]:
                        best_name = (
                            score,
                            line_text,
                            line_cy,
                            (px1 + line_x, py1 + line_y, px1 + line_x + line_w, py1 + line_y + line_h),
                        )
                if best_name is not None:
                    _, drink_name, name_cy, (bx1, by1, bx2, by2) = best_name
                    effect_text = ""
                    for line in merged_lines:
                        line_text = _normalize_header_text(str(getattr(line, "text", "") or ""))
                        if not line_text:
                            continue
                        line_y = int(getattr(line, "y", 0))
                        line_h = max(1, int(getattr(line, "h", 0)))
                        line_cy = int(getattr(line, "cy", line_y + line_h // 2))
                        if line_cy <= name_cy:
                            continue
                        if _looks_like_effect_line(line_text):
                            effect_text = line_text
                            break

                    if debugger is not None:
                        debugger.add_box(
                            bx1,
                            by1,
                            bx2,
                            by2,
                            label=f"p_drink_title:{drink_name[:24]}",
                            color=(80, 220, 120),
                            alpha=0.16,
                            duration=2.5,
                            font_size=14,
                        )
                    logger.debug(
                        "p_drink header OCR: 信息面板名称={!r}, effect={!r} (panel y={}..{}, x={}..{})",
                        drink_name,
                        effect_text,
                        py1,
                        py2,
                        px1,
                        px2,
                    )
                    return drink_name, effect_text
    return "", ""


def _detect_central_drinks(app: "AppProcessor"):
    """检测中央区域（非底栏）的 P Drink 图标，按 x 坐标排序。"""
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return []
    frame_height = frame.shape[0]
    return sorted(
        (d for d in app.latest_results.filter_by_label(BaseUILabels.P_DRINK)
         if d.cy < frame_height * 0.85),
        key=lambda item: item.cx,
    )


def _scan_drink_names(
    app: "AppProcessor",
    drink_boxes,
    *,
    settle_time: float = 0.6,
) -> list[str]:
    """逐个点击饮料图标，从头部 OCR 获取每个饮料的名称。

    扫描完成后最后一个饮料处于选中态，调用方需要再次点击目标饮料。
    """
    names: list[str] = []
    for idx, box in enumerate(drink_boxes):
        app.device.click_element(box)
        # 第一个饮料需要更长等待（面板可能还在打开）
        wait = settle_time * 1.5 if idx == 0 else settle_time
        time.sleep(wait)
        frame = app.latest_frame
        name, effect = _ocr_p_drink_header(frame, app=app, drink_boxes=drink_boxes)
        # OCR 失败时重试一次（可能动画还没刷新完）
        if not name:
            time.sleep(0.4)
            frame = app.latest_frame
            name, effect = _ocr_p_drink_header(frame, app=app, drink_boxes=drink_boxes)
        names.append(name)
        logger.debug(f"p_drink: 扫描第{idx}个饮料: {name!r} (效果: {effect!r})")
    return names


def _ocr_limit_controls(frame) -> list[dict[str, Any]]:
    """OCR 上限页底部控件区，返回带全局坐标的文本结果。"""
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    x1 = int(w * 0.18)
    x2 = int(w * 0.82)
    y1 = int(h * 0.70)
    y2 = int(h * 0.92)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for item in _ocr.ocr(crop):
        rows.append({
            "text": str(getattr(item, "text", "") or ""),
            "x": x1 + int(getattr(item, "x", 0)),
            "y": y1 + int(getattr(item, "y", 0)),
            "w": int(getattr(item, "w", 0)),
            "h": int(getattr(item, "h", 0)),
        })
    return rows


def _find_limit_row(rows: list[dict[str, Any]], *tokens: str) -> dict[str, Any] | None:
    """查找limit、row并返回结果。

    Args:
        rows: 用于提供rows相关输入。
        *tokens: 用于提供tokens相关输入。

    Returns:
        dict: 结构化结果字典。
    """
    for row in rows:
        text = str(row.get("text", "") or "")
        if text and any(token in text for token in tokens):
            return row
    return None


# ────────────────────────────────────────────────────────────
# 饮料丢弃链（P_DRINK_DETAIL 模态 → “捨てる” → “廃棄確認” → “はい”）。
# ────────────────────────────────────────────────────────────

_DISCARD_CHAIN_MODAL_WAIT = 2.0     # 等待模态出现的超时秒数
_DISCARD_CHAIN_SETTLE = 0.8         # 各步骤之间的等待秒数
_DISCARD_CHAIN_POLL_INTERVAL = 0.3  # 轮询间隔


def _wait_for_modal_header(app: "AppProcessor", *, timeout: float = _DISCARD_CHAIN_MODAL_WAIT) -> bool:
    """等待 MODAL_HEADER 标签出现（轮询 latest_results）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        results = app.latest_results
        if results is not None and results.exists_label(BaseUILabels.MODAL_HEADER):
            return True
        time.sleep(_DISCARD_CHAIN_POLL_INTERVAL)
    return False


def _find_discard_button_position(app: "AppProcessor") -> tuple[int, int] | None:
    """在 Pドリンク詳細 模态中定位「捨てる」文本按钮。

    模态布局:
      ┌────────────────────────────────────────┐
      │      Pドリンク詳細           (header)  │
      │  [icon]  ジンジャーエール      捨てる   │
      │          効果説明...                    │
      │  キャンセル          使う               │
      └────────────────────────────────────────┘

    策略: OCR 模态头下方区域，找到包含「捨てる」的文本块，返回其中心坐标。
    """
    results = app.latest_results
    frame = app.latest_frame
    if results is None or frame is None:
        return None

    # 定位模态头
    modal_headers = list(results.filter_by_label(BaseUILabels.MODAL_HEADER))
    if not modal_headers:
        return None
    header = modal_headers[0]

    # 定位取消按钮（模态底部参考线）
    cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
    fh, fw = frame.shape[:2]

    # OCR 区域: 模态头下方 → 取消按钮上方
    region_y1 = int(header.h)  # header 底部 y
    region_y2 = int(cancel_boxes[0].y) if cancel_boxes else int(fh * 0.8)
    region_x1 = max(0, int(header.x) - 10)
    region_x2 = min(fw, int(header.w) + 10)

    if region_y2 <= region_y1 + 20 or region_x2 <= region_x1 + 20:
        return None

    modal_crop = frame[region_y1:region_y2, region_x1:region_x2]
    ocr_results = _ocr.ocr(modal_crop)
    if not ocr_results:
        return None

    # 查找包含“捨てる”的文本块。
    for item in ocr_results:
        text = str(getattr(item, "text", "") or "").strip()
        if ProduceText.P_DRINK_DISCARD in text:
            # 计算绝对坐标（OCR 结果相对于裁剪区域）
            item_cx = region_x1 + int(getattr(item, "x", 0)) + int(getattr(item, "w", 0)) // 2
            item_cy = region_y1 + int(getattr(item, "y", 0)) + int(getattr(item, "h", 0)) // 2
            logger.debug(
                "p_drink: 丢弃链 - 找到「捨てる」按钮 at ({}, {})",
                item_cx, item_cy,
            )
            return (item_cx, item_cy)

    return None


def _click_cancel_in_modal(app: "AppProcessor") -> None:
    """点击模态中的キャンセル按钮关闭（安全退出）。"""
    results = app.latest_results
    cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
    if cancel_boxes:
        app.device.click_element(cancel_boxes[0])
        logger.debug("p_drink: 丢弃链 - 点击 キャンセル 关闭模态")
    else:
        # 回退: 点击模态外部区域关闭
        frame = app.latest_frame
        if frame is None or frame.size == 0:
            logger.warning("p_drink: 丢弃链 - 无法获取画面，跳过关闭操作")
            time.sleep(_DISCARD_CHAIN_SETTLE)
            return
        fh = frame.shape[0]
        fw = frame.shape[1]
        app.device.click(fw // 2, int(fh * 0.1), el_label="p_drink_discard_cancel_fallback")
        logger.warning("p_drink: 丢弃链 - 未找到取消按钮，点击空白区域尝试关闭")
    time.sleep(_DISCARD_CHAIN_SETTLE)


def _ocr_drink_detail_name(app: "AppProcessor") -> str:
    """从 Pドリンク詳細 模态中 OCR 饮料名称。

    模态布局:
      ┌────────────────────────────────────────┐
      │      Pドリンク詳細           (header)  │
      │  [icon]  ジンジャーエール      捨てる   │
      │          効果説明...                    │
      │  キャンセル          使う               │
      └────────────────────────────────────────┘

    名称在 header 下方、icon 右侧。OCR 该区域第一段文本即为饮料名。
    """
    results = app.latest_results
    frame = app.latest_frame
    if results is None or frame is None:
        return ""

    modal_headers = list(results.filter_by_label(BaseUILabels.MODAL_HEADER))
    if not modal_headers:
        return ""
    header = modal_headers[0]

    cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
    fh, fw = frame.shape[:2]

    # OCR 区域：模态头下方到取消按钮上方，横向取模态左侧 2/3（排除右侧“捨てる”）。
    region_y1 = int(header.h)
    region_y2 = int(cancel_boxes[0].y) if cancel_boxes else int(fh * 0.8)
    region_x1 = max(0, int(header.x) - 10)
    region_x2 = int(region_x1 + (int(header.w) - region_x1) * 0.65)

    if region_y2 <= region_y1 + 20 or region_x2 <= region_x1 + 20:
        return ""

    # 只取上部 1/3 高度（名称在最上方）
    name_height = (region_y2 - region_y1) // 3
    modal_crop = frame[region_y1:region_y1 + name_height, region_x1:region_x2]
    ocr_results = _ocr.ocr(modal_crop)
    if not ocr_results:
        return ""

    # 第一段非空文本视为饮料名（排除“捨てる”等控件文本）。
    for item in ocr_results:
        text = str(getattr(item, "text", "") or "").strip()
        if text and ProduceText.P_DRINK_DISCARD not in text and len(text) >= 2:
            return text
    return ""


def _scan_and_learn_inventory_drinks(
    app: "AppProcessor",
    candidates: list["PDrinkLimitActionCandidate"],
) -> None:
    """对 CLIP 未识别的底栏库存饮料，逐个点击打开详情模态读取名称并触发自动学习。

    流程（对每个 unresolved 候选）:
      1. 点击底栏饮料图标
      2. 等待 Pドリンク詳細 模态出现
      3. OCR 读取饮料名称
      4. 用名称重新 resolve → 触发 DB 匹配 + CLIP 自动学习
      5. 点击キャンセル关闭模态
    """
    unresolved = [c for c in candidates if c.kind == "discard_existing_drink" and not c.db_id]
    if not unresolved:
        return
    if not hasattr(app, "device"):
        logger.debug("p_drink: 当前上下文无 device，跳过底栏饮料补扫描")
        return

    logger.info(f"p_drink: 底栏有 {len(unresolved)} 瓶饮料未识别，开始逐个点击扫描")

    for candidate in unresolved:
        if candidate.box is None:
            continue

        # 1. 点击底栏饮料
        app.device.click_element(candidate.box)
        time.sleep(_DISCARD_CHAIN_SETTLE)

        # 2. 等待模态出现
        if not _wait_for_modal_header(app):
            logger.warning(f"p_drink: 点击底栏第{candidate.metadata.get('slot_index', '?')}瓶饮料后未出现模态")
            continue

        time.sleep(0.3)

        # 3. OCR 读取饮料名称
        drink_name = _ocr_drink_detail_name(app)
        logger.info(f"p_drink: 底栏饮料 OCR 识别为「{drink_name}」")

        # 4. 用名称重新 resolve（触发 CLIP 自动学习）
        if drink_name:
            resolution = resolve_produce_drink_identity(
                drink_name,
                app=app,
                box=candidate.box,
                index=candidate.index,
            )
            if resolution.db_id:
                candidate.db_id = resolution.db_id
                candidate.source = resolution.source
                candidate.confidence = resolution.confidence
                candidate.metadata.update(resolution.metadata)
                new_drink_name = candidate.metadata.get("new_drink", {}).get("display_name") or "新饮料"
                candidate.title = f"丢弃「{resolution.display_name}」并保留新饮料「{new_drink_name}」"
                logger.info(f"p_drink: 底栏饮料已识别 → {resolution.display_name} (db_id={resolution.db_id})")

        # 5. 关闭模态
        _click_cancel_in_modal(app)


def _execute_drink_discard_chain(
    app: "AppProcessor",
    ctx: "ProduceContext",
    target: "PDrinkLimitActionCandidate",
) -> bool:
    """执行完整的饮料丢弃链。

    流程:
      1. 点击底栏库存饮料图标
      2. 等待「Pドリンク詳細」模态出现
      3. OCR 定位「捨てる」按钮并点击
      4. 等待「廃棄確認」模态出现
      5. 点击确认按钮（はい）
      6. 等待丢弃完成

    Returns:
        True 表示丢弃成功，False 表示链中某步骤失败（已安全回退）。
    """
    drink_name = target.metadata.get("display_name") or target.title or "未知饮料"
    logger.info("p_drink: 开始丢弃链 - 丢弃「{}」(slot={})",
                drink_name, target.metadata.get("slot_index", "?"))

    # 1. 点击底栏库存饮料
    app.device.click_element(target.box)
    time.sleep(_DISCARD_CHAIN_SETTLE)

    # 2. 等待“Pドリンク詳細”模态。
    if not _wait_for_modal_header(app):
        logger.warning("p_drink: 丢弃链 - 点击库存饮料后未检测到模态")
        return False

    # 额外等待确保模态渲染完成
    time.sleep(0.3)

    # 3. OCR 定位“捨てる”并点击。
    discard_pos = _find_discard_button_position(app)
    if discard_pos is None:
        logger.warning("p_drink: 丢弃链 - 未找到「捨てる」按钮，取消操作")
        _click_cancel_in_modal(app)
        return False

    app.device.click(discard_pos[0], discard_pos[1], el_label="p_drink_discard_button")
    time.sleep(_DISCARD_CHAIN_SETTLE)

    # 4. 等待「廃棄確認」模态
    if not _wait_for_modal_header(app, timeout=2.0):
        logger.warning("p_drink: 丢弃链 - 点击捨てる后未检测到廃棄確認模态")
        # 可能模态已被关闭，尝试安全退出
        _click_cancel_in_modal(app)
        return False

    time.sleep(0.3)

    # 5. 点击确认按钮（“はい”）。
    results = app.latest_results
    confirm_boxes = list(results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
    if confirm_boxes:
        # 选择最右侧按钮（“はい”通常在右侧）。
        target_btn = max(confirm_boxes, key=lambda b: b.cx)
        app.device.click_element(target_btn)
        logger.info("p_drink: 丢弃链 - 确认丢弃「{}」", drink_name)
    else:
        # 回退: 尝试通用底部按钮点击
        logger.warning("p_drink: 丢弃链 - 未找到确认按钮，尝试通用点击")
        _click_any_bottom_button(app)

    time.sleep(1.0)

    # 6. 验证丢弃完成（模态是否消失）
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        results = app.latest_results
        if results is not None and not results.exists_label(BaseUILabels.MODAL_HEADER):
            logger.info("p_drink: 丢弃链完成 - 「{}」已丢弃", drink_name)
            return True
        time.sleep(_DISCARD_CHAIN_POLL_INTERVAL)

    # 模态仍在，可能需要额外点击
    logger.warning("p_drink: 丢弃链 - 确认后模态仍存在，尝试额外点击")
    _click_any_bottom_button(app)
    time.sleep(1.0)
    return True


def _is_p_drink_limit_page(app: "AppProcessor") -> bool:
    """判断当前页面是否为 P 饮料上限处理页。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    rows = _ocr_limit_controls(app.latest_frame)
    return any(
        any(keyword in str(row.get("text", "") or "") for keyword in _LIMIT_KEYWORDS)
        for row in rows
    )


def _detect_bottom_inventory_drinks(app: "AppProcessor"):
    """检测bottom、inventory、drinks并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        返回处理结果，具体类型见返回注解。
    """
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return []
    frame_height = frame.shape[0]
    return sorted(
        (d for d in app.latest_results.filter_by_label(ProducerLabels.P_DRINK)
         if d.cy >= frame_height * 0.88),
        key=lambda item: item.cx,
    )


def collect_p_drink_limit_action_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext" | None = None,
) -> list[PDrinkLimitActionCandidate]:
    """收集p、饮料、limit、操作、候选项并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    rows = _ocr_limit_controls(app.latest_frame)
    skip_row = _find_limit_row(rows, ProduceText.P_DRINK_LIMIT_NO_RECEIVE)
    confirm_row = _find_limit_row(rows, *ProduceText.P_DRINK_LIMIT_SKIP_ALTS[1:])
    candidates: list[PDrinkLimitActionCandidate] = []
    pending_new = _get_pending_new_p_drink(ctx)
    new_drink_name = _drink_display_name(pending_new) or "新饮料"

    if skip_row is not None:
        checkbox_click_x = max(0, int(skip_row["x"] - max(28, skip_row["h"] * 0.8)))
        checkbox_click_y = int(skip_row["y"] + max(1, skip_row["h"]) / 2)
        candidates.append(
            PDrinkLimitActionCandidate(
                index=0,
                title=f"放弃新饮料「{new_drink_name}」",
                kind="skip_new_drink",
                action_id="p_drink_limit_skip_new",
                source="ocr",
                confidence=0.85,
                metadata={
                    "checkbox_x": checkbox_click_x,
                    "checkbox_y": checkbox_click_y,
                    "button_x": int(confirm_row["x"] + max(1, confirm_row["w"]) / 2) if confirm_row else 0,
                    "button_y": int(confirm_row["y"] + max(1, confirm_row["h"]) / 2) if confirm_row else 0,
                    "candidate_type": "p_drink_limit",
                    "p_drink_limit_kind": "skip_new_drink",
                    "new_drink": dict(pending_new),
                },
            )
        )

    for slot_index, box in enumerate(_detect_bottom_inventory_drinks(app), start=1):
        candidate = PDrinkLimitActionCandidate(
            index=len(candidates),
            title=f"丢弃底栏第{slot_index}瓶饮料并保留新饮料「{new_drink_name}」",
            kind="discard_existing_drink",
            box=box,
            action_id=f"p_drink_limit_discard_slot_{slot_index}",
            source="yolo",
            confidence=1.0,
            metadata={
                "slot_index": slot_index,
                "candidate_type": "p_drink_limit",
                "p_drink_limit_kind": "discard_existing_drink",
                "new_drink": dict(pending_new),
            },
        )
        resolution = resolve_produce_drink_identity(
            "",
            app=app,
            box=box,
            index=candidate.index,
        )
        if resolution.db_id:
            candidate.db_id = resolution.db_id
            candidate.source = resolution.source
            candidate.confidence = resolution.confidence
            candidate.metadata.update(resolution.metadata)
            candidate.title = f"丢弃「{resolution.display_name}」并保留新饮料「{new_drink_name}」"
        candidates.append(candidate)

    # CLIP 未识别的底栏饮料：逐个点击打开详情模态读取名称并触发自动学习
    _scan_and_learn_inventory_drinks(app, candidates)

    return candidates


def decide_p_drink_limit_action(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: list[PDrinkLimitActionCandidate],
) -> PDrinkLimitActionCandidate:
    """决策p、饮料、limit、操作并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        candidates: 候选项列表，供策略或规则选择目标动作。

    Returns:
        PDrinkLimitActionCandidate: 返回值类型见注解，语义由函数用途决定。
    """
    decision_state = build_decision_state(
        app,
        ctx,
        phase="p_drink",
        position="p_drink_limit",
        candidates=candidates,
        reason="p_drink_limit_decision",
    )
    preferred_candidate, preferred_reason, score_gap = _select_p_drink_limit_preference(ctx, candidates)
    _annotate_p_drink_limit_preference(
        decision_state,
        preferred_index=preferred_candidate.index,
        reason=preferred_reason,
    )
    decision = invoke_decision_strategy(
        ctx.p_drink_strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        target_index = resolve_candidate_index(decision, candidates)
        target = candidates[target_index]
        if target.action_id != preferred_candidate.action_id and abs(score_gap) >= 15.0:
            logger.info(
                "p_drink_limit: 覆盖原决策 {} -> {} ({})",
                target.action_id,
                preferred_candidate.action_id,
                preferred_reason,
            )
            # 记录覆盖决策
            from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper
            DecisionDumper.get_instance().update_last_resolved(
                resolved_index=preferred_candidate.index,
                resolved_name=getattr(preferred_candidate, "title", ""),
                fallback_used=True,
                fallback_reason=f"系统偏好覆盖({preferred_reason})",
            )
            return preferred_candidate
        return target

    # 无 LLM 决策，使用系统偏好
    from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper
    DecisionDumper.get_instance().update_last_resolved(
        resolved_index=preferred_candidate.index,
        resolved_name=getattr(preferred_candidate, "title", ""),
        fallback_used=True,
        fallback_reason=f"无决策,系统偏好({preferred_reason})",
    )
    return preferred_candidate


def _click_absolute(app: "AppProcessor", x: int, y: int, *, label: str) -> None:
    """点击absolute并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        x: 用于提供x相关输入。
        y: 用于提供y相关输入。
        label: 用于提供label相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    app.device.click(int(x), int(y), el_label=label)


def _handle_p_drink_limit_page(
    app: "AppProcessor",
    ctx: "ProduceContext",
) -> PDrinkStepResult | None:
    """处理handle、p、饮料、limit、page并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        PDrinkStepResult | None: 返回值类型见注解。
    """
    candidates = collect_p_drink_limit_action_candidates(app, ctx)
    if not candidates:
        logger.warning("p_drink: 所持上限页未收集到任何候选动作")
        return None

    target = decide_p_drink_limit_action(app, ctx, candidates)
    logger.info(f"p_drink: 所持上限页选择 {target.action_id} ({target.title})")

    if target.kind == "skip_new_drink":
        attempts = int(ctx.handler_state.get("p_drink_limit_skip_attempts", 0) or 0)
        if attempts % 2 == 0:
            checkbox_x = int(target.metadata.get("checkbox_x", 0))
            checkbox_y = int(target.metadata.get("checkbox_y", 0))
            if checkbox_x > 0 and checkbox_y > 0:
                _click_absolute(app, checkbox_x, checkbox_y, label="p_drink_limit_checkbox")
                time.sleep(0.8)

        button_x = int(target.metadata.get("button_x", 0))
        button_y = int(target.metadata.get("button_y", 0))
        if button_x > 0 and button_y > 0:
            _click_absolute(app, button_x, button_y, label="p_drink_limit_confirm")
        else:
            _click_any_bottom_button(app)

        ctx.handler_state["p_drink_limit_skip_attempts"] = attempts + 1
        ctx.record_operation(
            "p_drink_limit_skip_new",
            target=target.action_id,
            details={"attempt": attempts + 1},
        )
        time.sleep(1.0)

        # 处理可能弹出的“報酬スキップ”确认弹窗。
        if _handle_reward_skip_confirmation(app):
            return PDrinkStepResult(status="confirmed")
        return PDrinkStepResult(status="selected")

    if target.kind == "discard_existing_drink" and target.box is not None:
        # 执行完整丢弃链：点击库存饮料 → “捨てる” → “廃棄確認” → “はい”。
        success = _execute_drink_discard_chain(app, ctx, target)
        ctx.handler_state["p_drink_limit_skip_attempts"] = 0
        ctx.record_operation(
            "p_drink_limit_discard_existing",
            target=target.action_id,
            details={
                "slot_index": target.metadata.get("slot_index", 0),
                "discard_success": success,
            },
        )
        if success:
            # 丢弃成功后必须点击“受け取る”接收新饮料，不能只丢不拿。
            logger.info("p_drink: 丢弃成功，点击「受け取る」接收新饮料")
            time.sleep(0.5)
            _click_any_bottom_button(app)
            time.sleep(1.0)
            # 同步牌组变更: 移除旧饮料 + 获取新饮料
            discarded_db_id = str(target.db_id or "")
            if discarded_db_id:
                ctx.mutate_deck_remove(discarded_db_id, kind="produce_drink")
            pending_new = _get_pending_new_p_drink(ctx)
            new_db_id = str(pending_new.get("db_id") or "")
            if new_db_id:
                ctx.mutate_deck_acquire(
                    new_db_id,
                    kind="produce_drink",
                    name=_drink_display_name(pending_new),
                    source="p_drink_limit_replace",
                )
            return PDrinkStepResult(status="confirmed")
        # 丢弃失败则回退，下次迭代重试
        logger.warning("p_drink: 丢弃链失败，等待下次迭代重试")
        return None

    return None


# ────────────────────────────────────────────────────────────
# 采集 / 决策 / 执行
# ────────────────────────────────────────────────────────────

def collect_p_drink_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
    scanned_names: list[str] | None = None,
) -> List[PDrinkCandidate]:
    """采集屏幕上的 P 饮料项（中央区域，非底栏）。

    Args:
        scanned_names: 预扫描的饮料名称列表，与 drink_boxes 一一对应。
                       为 None 时回退到头部 OCR 读取当前选中饮料名。
    """
    drinks = _detect_central_drinks(app)
    pending = ctx.pending_p_drink_index if position == "p_drink_selected" else None

    # 如果没有预扫描名称，尝试从头部 OCR 获取当前选中饮料名
    if scanned_names is None:
        header_name, _ = _ocr_p_drink_header(app.latest_frame, app=app, drink_boxes=drinks)
        scanned_names = []
        for idx, _box in enumerate(drinks):
            if pending == idx and header_name:
                scanned_names.append(header_name)
            else:
                scanned_names.append("")

    candidates = [
        PDrinkCandidate(
            index=idx,
            title=(scanned_names[idx] if idx < len(scanned_names) else ""),
            selected=pending == idx,
            box=box,
        )
        for idx, box in enumerate(drinks)
    ]
    hydrate_p_drink_candidates(app, candidates)

    # post-hydration 回扫: 对仍然未识别的饮料，逐个点击重新 OCR
    unresolved = [c for c in candidates if not c.db_id and not c.title]
    if unresolved:
        logger.info(f"p_drink: {len(unresolved)} 个饮料未识别，尝试逐个点击回扫")
        for c in unresolved:
            app.device.click_element(c.box)
            time.sleep(0.8)
            frame = app.latest_frame
            name, effect = _ocr_p_drink_header(frame, app=app, drink_boxes=drinks)
            if not name:
                time.sleep(0.5)
                frame = app.latest_frame
                name, effect = _ocr_p_drink_header(frame, app=app, drink_boxes=drinks)
            if name:
                logger.info(f"p_drink: 回扫第{c.index}个饮料成功: {name!r}")
                c.title = name
                # 重新尝试解析
                resolution = resolve_produce_drink_identity(
                    name, app=app, box=c.box, index=c.index,
                )
                _apply_resolution(c, resolution)

    # 兜底: 确保未识别的饮料至少有可读的显示名
    for c in candidates:
        if not c.title:
            c.title = f"未知P饮料{c.index + 1}"

    return candidates


def decide_p_drink(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: List[PDrinkCandidate],
    *,
    position: str,
) -> int:
    """决策p、饮料并返回结果。

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
        phase="p_drink",
        position=position,
        candidates=candidates,
        reason="p_drink_decision",
    )
    decision = invoke_decision_strategy(
        ctx.p_drink_strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        return resolve_candidate_index(decision, candidates)

    # 无 LLM 决策，使用本地兜底
    from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper
    fallback_index: int
    fallback_reason: str
    if (
        ctx.pending_p_drink_index is not None
        and 0 <= ctx.pending_p_drink_index < len(candidates)
    ):
        fallback_index = ctx.pending_p_drink_index
        fallback_reason = "pending 索引"
    else:
        fallback_index = 0
        fallback_reason = "默认首选"

    resolved_name = ""
    if 0 <= fallback_index < len(candidates):
        resolved_name = getattr(candidates[fallback_index], "title", "")
    DecisionDumper.get_instance().update_last_resolved(
        resolved_index=fallback_index,
        resolved_name=resolved_name,
        fallback_used=True,
        fallback_reason=fallback_reason,
    )
    return fallback_index


def _handle_reward_skip_confirmation(app: "AppProcessor", timeout: float = 2.0) -> bool:
    """处理「報酬スキップ」确认弹窗。

    当用户在P饮料上限页面放弃新饮料时，游戏可能弹出确认对话框:
      「対象のPドリンクを獲得せず次へ進みますか？」
    需要点击右侧的「決定」按钮确认。

    Returns:
        True 表示检测到并处理了弹窗，False 表示未出现弹窗。
    """
    if not hasattr(app, "latest_results"):
        logger.debug("p_drink: 当前上下文无 latest_results，跳过報酬スキップ确认")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        results = getattr(app, "latest_results", None)
        if results is None:
            time.sleep(0.3)
            continue

        # “報酬スキップ”弹窗会叠加在上限页之上，导致出现 ≥2 个 MODAL_HEADER。
        modal_headers = list(results.filter_by_label(BaseUILabels.MODAL_HEADER))
        if len(modal_headers) >= 2:
            buttons: list = []
            for label in (ProducerLabels.CONFIRM_BUTTON, ProducerLabels.DISABLE_BUTTON, BaseUILabels.BUTTON):
                for box in results.filter_by_label(label):
                    buttons.append(box)
            if buttons:
                # 「決定」在右侧，选最右按钮
                target_btn = max(buttons, key=lambda b: b.cx)
                app.device.click_element(target_btn)
                logger.info("p_drink: 報酬スキップ確認 → 点击「決定」")
                time.sleep(1.0)
                return True

        time.sleep(0.3)
    return False


def _click_any_bottom_button(app: "AppProcessor") -> bool:
    """点击P饮料面板底部的按钮（不区分 Confirm/Disable 标签）。

    YOLO 可能将活跃的橙色「受け取らない」按钮误分类为 Disable Button，
    因此需要同时检查 Confirm 和 Disable 标签。
    """
    results = app.latest_results
    # 收集所有可能的按钮
    candidates = []
    for label in (ProducerLabels.CONFIRM_BUTTON, ProducerLabels.DISABLE_BUTTON, BaseUILabels.BUTTON):
        for box in results.filter_by_label(label):
            candidates.append(box)
    if candidates:
        # 点击最靠下的按钮
        target = max(candidates, key=lambda b: b.cy)
        app.device.click_element(target)
        return True
    return False


def _frame_has_observable_content(frame) -> bool:
    """判断当前帧是否有可观测的真实图像内容。

    静态回归测试里 latest_frame 是全零占位图，无法验证点击后的页面推进。
    """
    return frame is not None and getattr(frame, "size", 0) > 0 and bool(frame.any())


def _verify_p_drink_advanced(app: "AppProcessor", timeout: float = 1.5) -> bool:
    """验证 P 饮料确认后画面是否推进。

    检查是否仍停留在 P_DRINK 页面（仍能看到中央区域的 P Drink 标签）。
    """
    deadline = time.monotonic() + timeout
    time.sleep(0.6)
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return False
    frame_height = frame.shape[0]
    while time.monotonic() < deadline:
        results = app.latest_results
        # 检查中央 P Drink 是否消失（底栏饮料不算）
        central_drinks = [
            d for d in results.filter_by_label(ProducerLabels.P_DRINK)
            if d.cy < frame_height * 0.85
        ]
        if not central_drinks:
            return True
        # 同时检查是否弹出 modal（“報酬スキップ”确认）。
        if results.exists_label(ProducerLabels.MODAL_HEADER):
            return True
        time.sleep(0.3)
    return False


def _record_p_drink_confirmed(ctx: "ProduceContext") -> None:
    """记录`p_drink_confirmed`到上下文状态。"""
    pending_drink = _get_pending_new_p_drink(ctx)
    drink_db_id = str(pending_drink.get("db_id") or "")
    ctx.record_operation(
        "confirm_p_drink",
        target=ctx.pending_p_drink_label or "p_drink",
        details={"index": ctx.pending_p_drink_index, "db_id": drink_db_id},
    )
    if drink_db_id:
        ctx.mutate_deck_acquire(
            drink_db_id,
            kind="produce_drink",
            name=ctx.pending_p_drink_label or pending_drink.get("display_name", ""),
            source="p_drink",
        )
    ctx.clear_p_drink_pending()


def _try_skip_p_drink(app: "AppProcessor", *, checkbox_already_checked: bool = False) -> bool:
    """P饮料所持上限时，点击「受け取らない」按钮跳过领取。

    流程：勾选「受け取らない」复选框 → 点击按钮 → 处理子弹窗。
    如果 checkbox_already_checked=True，则跳过复选框点击步骤，直接点按钮。
    """

    if not checkbox_already_checked:
        # 查找“受け取らない”复选框。
        checkbox_boxes = list(app.latest_results.filter_by_label(BaseUILabels.CHECKBOX))
        if not checkbox_boxes:
            logger.debug("p_drink: 未找到复选框，无法跳过")
            return False

        logger.info("p_drink: P饮料所持上限，点击「受け取らない」跳过领取")
        app.device.click_element(checkbox_boxes[0])
        time.sleep(1.2)

    # 点击底部按钮（可能是 Confirm 或被误分类为 Disable 的橙色按钮）
    _click_any_bottom_button(app)
    time.sleep(1.5)

    # 处理“報酬スキップ”确认子弹窗。
    results = app.latest_results
    modal_headers = list(results.filter_by_label(ProducerLabels.MODAL_HEADER))
    if modal_headers:
        logger.info("p_drink: 检测到\"報酬スキップ确认\"弹窗，点击确认")
        _click_any_bottom_button(app)
        time.sleep(1.0)
    return True


def _is_p_drink_limit_by_controls(app: "AppProcessor") -> bool:
    """通过控件组合判断所持上限页（OCR 失败时兜底）。"""
    results = getattr(app, "latest_results", None)
    if results is None:
        return False
    has_disable = results.exists_label(ProducerLabels.DISABLE_BUTTON)
    has_checkbox = results.exists_label(BaseUILabels.CHECKBOX)
    return bool(has_disable and has_checkbox)


def _should_handle_p_drink_limit_page(app: "AppProcessor") -> bool:
    """判断当前帧是否需要进入 P 饮料上限处理流程。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    return _is_p_drink_limit_page(app) or _is_p_drink_limit_by_controls(app)


def execute_p_drink_step(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> PDrinkStepResult | None:
    """执行一步 P 饮料交互。

    - p_drink_selected: 点击确认按钮（第 2 步），必要时转入所持上限决策
    - p_drink_idle: 选择一个饮料（第 1 步），检测所持上限并走决策流程
    """
    if position == "p_drink_selected":
        if _should_handle_p_drink_limit_page(app):
            return _handle_p_drink_limit_page(app, ctx)

        _click_any_bottom_button(app)

        # 验证画面是否推进
        if _verify_p_drink_advanced(app):
            _record_p_drink_confirmed(ctx)
            return PDrinkStepResult(status="confirmed")

        if not _frame_has_observable_content(getattr(app, "latest_frame", None)):
            logger.debug("p_drink: 当前为静态回归帧，无法观察推进，按确认成功处理")
            _record_p_drink_confirmed(ctx)
            return PDrinkStepResult(status="confirmed")

        # 画面未推进 → 可能是P饮料所持上限，尝试跳过
        logger.warning("p_drink: 确认按钮点击后画面未推进，尝试走所持上限决策流程")
        if _should_handle_p_drink_limit_page(app):
            return _handle_p_drink_limit_page(app, ctx)

        return None

    if _should_handle_p_drink_limit_page(app):
        return _handle_p_drink_limit_page(app, ctx)

    # 检测中央区域的饮料图标
    drink_boxes = _detect_central_drinks(app)
    if not drink_boxes:
        return None

    # 逐个点击饮料图标以获取名称（点击后头部显示选中饮料名）
    logger.info(f"p_drink: 开始扫描 {len(drink_boxes)} 个饮料名称")
    scanned_names = _scan_drink_names(app, drink_boxes)

    # 扫描完成后，需要重新检测当前帧的饮料位置（扫描期间帧可能更新）
    candidates = collect_p_drink_candidates(
        app, ctx, position=position, scanned_names=scanned_names,
    )
    if not candidates:
        return None

    target_index = decide_p_drink(app, ctx, candidates, position=position)
    target = candidates[target_index]
    app.device.click_element(target.box)
    ctx.pending_p_drink_index = target.index
    ctx.pending_p_drink_label = target.title or target.action_id or f"p_drink_{target.index + 1}"
    _remember_pending_new_p_drink(ctx, target)
    ctx.record_operation(
        "select_p_drink",
        target=ctx.pending_p_drink_label,
        details={
            "index": target.index,
            "action_id": target.action_id,
            "db_id": target.db_id,
        },
    )
    logger.debug(f"p_drink: selected {target.index} {target.title!r}")
    return PDrinkStepResult(status="selected", candidate=target)


# ────────────────────────────────────────────────────────────
# 处理器
# ────────────────────────────────────────────────────────────

class PDrinkHandler(GameplayHandler):
    """P饮料选择画面处理。"""

    phase_tag = "p_drink"
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
        return phase == "p_drink"

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
        result = execute_p_drink_step(app, ctx, position=position)
        if result is None:
            return HandlerResult.no_action("no p_drink elements")
        sleep_time = 1.0 if result.status in ("confirmed", "skipped") else 0.8
        return HandlerResult.ok(f"p_drink {result.status}", sleep_after=sleep_time)
