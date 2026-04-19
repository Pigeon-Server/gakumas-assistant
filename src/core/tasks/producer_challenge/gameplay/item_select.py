"""Pアイテム選択 handler。

「受け取るPアイテムを選んでください。」画面を処理する。
画面に 1~3 個の Special Item が表示され、選択後に
「受け取る」ボタンが有効化される。

交互模式:
  1. Special Item アイコンをタップ → 選択ハイライト → ボタン有効化
  2. 受け取るボタンをタップ → アイテム取得、次の画面へ

CLIP 未识别时的探查流程:
  1. CLIP + 上下文 OCR 均无法识别 → 标记为 unresolved
  2. 逐个点击未识别物品 → 面板显示物品名称和效果描述
  3. OCR 提取物品名 → 数据库匹配 → CLIP 自动学习
  4. 所有物品识别完毕后再进行 LLM 决策
"""

from __future__ import annotations

import re
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.shared.common import (
    detect_bottom_white_modal_region,
    invoke_decision_strategy,
    ocr_text,
    resolve_candidate_index,
)
from src.core.tasks.producer_challenge.gameplay.decision import (
    build_decision_state,
    resolve_produce_item_identity,
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

_ITEM_SELECT_SCREEN_OCR = OCRService()
_ITEM_SELECT_NOISE_TOKENS = ProduceText.ITEM_SELECT_NOISE_TOKENS
_ITEM_SELECT_TITLE_NOISE_TOKENS = tuple(dict.fromkeys((
    *_ITEM_SELECT_NOISE_TOKENS,
    ProduceText.GUIDE,
    ProduceText.RECOMMEND,
)))
_ITEM_SELECT_NAME_NOISE_RE = re.compile(r'^[\|｜\[\]「」【】\s]+|[\|｜\[\]「」【】\s]+$')
_ITEM_SELECT_EFFECT_PREFIXES = ("↑", "↓", "→", "←", "↗", "↘", "♥", "❤", "♡")
_ITEM_SELECT_EFFECT_LINE_RE = re.compile(
    rf"({'|'.join(map(re.escape, ProduceText.STATUS_VALUE_TOKENS))}|{re.escape(ProduceText.YARUKI)}|{re.escape(ProduceText.SCORE)}|{re.escape(ProduceText.STAMINA)}).*[+\-−]\d+"
)
_ITEM_SELECT_JP_CHAR_RE = re.compile(r"[ぁ-んァ-ヶー一-龯]")

# ── 探查交互等待时间 ──
_PROBE_TAP_WAIT = 0.4       # 点击后等待 UI 刷新
_PROBE_INFERENCE_WAIT = 0.3  # 等待 YOLO 推理完成
_MAX_PROBE_RETRIES = 2       # 同一物品最多探查次数，超过后接受未识别状态

# ── 探查结果缓存 ──
_ITEM_CACHE_KEY = "_item_select_resolved_cache"
_ITEM_PROBE_COUNT_KEY = "_item_select_probe_count"
_CACHE_POS_TOLERANCE = 30    # 像素容差


@dataclass
class ItemSelectCandidate:
    index: int
    title: str
    selected: bool
    box: object = field(repr=False, default=None)
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


def _normalize_item_select_text(text: str | None) -> str:
    return normalize_ocr_jp(str(text or "")).strip()


def _collect_item_select_lookup_texts(app: "AppProcessor", items: list) -> list[list[str]]:
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0 or not items:
        return [[] for _ in items]

    height, width = frame.shape[:2]
    merged_lines = _ITEM_SELECT_SCREEN_OCR.ocr(frame).auto_merge_lines(
        cy_range=max(6, int(height * 0.003)),
        width_gap=max(20, int(width * 0.02)),
    )
    content_lines: list[tuple[object, str]] = []
    for line in merged_lines:
        text = _normalize_item_select_text(getattr(line, "text", ""))
        if not text or len(text) < 2:
            continue
        if not (height * 0.22 <= line.cy <= height * 0.82):
            continue
        if any(token in text for token in _ITEM_SELECT_NOISE_TOKENS):
            continue
        content_lines.append((line, text))

    if not content_lines:
        return [[] for _ in items]

    centers = [int(item.cx) for item in items]
    lookup_texts: list[list[str]] = []
    for idx, box in enumerate(items):
        left_boundary = 0 if idx == 0 else int((centers[idx - 1] + centers[idx]) / 2)
        right_boundary = width if idx == len(items) - 1 else int((centers[idx] + centers[idx + 1]) / 2)
        box_height = max(1, int(getattr(box, "h", 0) - getattr(box, "y", 0)))
        candidate_rows: list[tuple[float, str]] = []

        for line, text in content_lines:
            if not (left_boundary <= line.cx <= right_boundary):
                continue
            if not (int(box.y - box_height * 0.10) <= line.cy <= int(box.h + box_height * 0.95)):
                continue
            vertical_gap = 0 if box.y <= line.cy <= box.h else min(abs(line.cy - box.y), abs(line.cy - box.h))
            horizontal_gap = abs(line.cx - box.cx)
            score = float(vertical_gap) * 2.0 + float(horizontal_gap) * 0.35
            candidate_rows.append((score, text))

        candidate_rows.sort(key=lambda item: item[0])
        deduped: list[str] = []
        for _score, text in candidate_rows:
            if text not in deduped:
                deduped.append(text)
        lookup_texts.append(deduped)

    return lookup_texts


def _item_pos_key(box) -> tuple[int, int]:
    """将 box 中心坐标量化为缓存 key。"""
    cx = int(round(getattr(box, "cx", 0) / _CACHE_POS_TOLERANCE) * _CACHE_POS_TOLERANCE)
    cy = int(round(getattr(box, "cy", 0) / _CACHE_POS_TOLERANCE) * _CACHE_POS_TOLERANCE)
    return (cx, cy)


def _apply_item_cache(ctx: "ProduceContext", candidates: list[ItemSelectCandidate]) -> None:
    """从 handler_state 缓存中恢复之前探查识别的结果。"""
    cache: dict = ctx.handler_state.get(_ITEM_CACHE_KEY, {})
    if not cache:
        return
    for cand in candidates:
        if cand.db_id:
            continue
        key = _item_pos_key(cand.box)
        cached = cache.get(key)
        if cached is None:
            continue
        cand.db_id = cached.get("db_id", "")
        cand.action_id = cached.get("action_id", "")
        cand.title = cached.get("title", "") or cand.title
        cand.source = cached.get("source", "cache")
        cand.confidence = cached.get("confidence", 0.0)
        if cached.get("metadata"):
            cand.metadata.update(cached["metadata"])
        logger.debug("item_select: 从缓存恢复 #{} → db_id={}", cand.index, cand.db_id)


def _save_item_cache(ctx: "ProduceContext", candidates: list[ItemSelectCandidate]) -> None:
    """将已识别（或已达最大探查次数）的候选项写入缓存。"""
    cache: dict = ctx.handler_state.setdefault(_ITEM_CACHE_KEY, {})
    for cand in candidates:
        key = _item_pos_key(cand.box)
        if key in cache:
            continue
        # 只缓存有 db_id 或已标记为 probed_max 的
        if not cand.db_id and not cand.metadata.get("probed_max"):
            continue
        cache[key] = {
            "db_id": cand.db_id,
            "action_id": cand.action_id,
            "title": cand.title,
            "source": cand.source,
            "confidence": cand.confidence,
            "metadata": dict(cand.metadata) if cand.metadata else {},
        }


def _should_skip_probe(ctx: "ProduceContext", candidate: ItemSelectCandidate) -> bool:
    """检查是否已达最大探查次数，应跳过此候选项的探查。"""
    probe_counts: dict = ctx.handler_state.setdefault(_ITEM_PROBE_COUNT_KEY, {})
    key = _item_pos_key(candidate.box)
    count = probe_counts.get(key, 0)
    return count >= _MAX_PROBE_RETRIES


def _increment_probe_count(ctx: "ProduceContext", candidate: ItemSelectCandidate) -> int:
    """递增探查计数，返回新的计数值。"""
    probe_counts: dict = ctx.handler_state.setdefault(_ITEM_PROBE_COUNT_KEY, {})
    key = _item_pos_key(candidate.box)
    count = probe_counts.get(key, 0) + 1
    probe_counts[key] = count
    return count


def collect_item_select_candidates(app: "AppProcessor", *, selected: bool = False) -> list[ItemSelectCandidate]:
    items = sorted(app.latest_results.filter_by_label(ProducerLabels.SPECIAL_ITEM), key=lambda b: b.cx)
    lookup_text_groups = _collect_item_select_lookup_texts(app, items)
    candidates: list[ItemSelectCandidate] = []
    for idx, box in enumerate(items):
        direct_title = _normalize_item_select_text(ocr_text(box.frame))
        lookup_texts = list(lookup_text_groups[idx]) if idx < len(lookup_text_groups) else []
        candidate = ItemSelectCandidate(
            index=idx,
            title=direct_title or (lookup_texts[0] if lookup_texts else ""),
            selected=selected,
            box=box,
        )
        resolution = resolve_produce_item_identity(
            candidate.title,
            app=app,
            box=box,
            index=idx,
            lookup_texts=lookup_texts,
        )
        candidate.action_id = resolution.action_id
        candidate.db_id = resolution.db_id
        candidate.source = resolution.source
        candidate.confidence = resolution.confidence
        candidate.title = resolution.display_name or candidate.title
        candidate.metadata = {
            **dict(resolution.metadata),
            "lookup_texts": lookup_texts,
        }
        candidates.append(candidate)
    if candidates:
        logger.debug(
            "item_select: 候选项 {}",
            [
                {
                    "index": candidate.index,
                    "title": candidate.title,
                    "db_id": candidate.db_id,
                    "action_id": candidate.action_id,
                    "source": candidate.source,
                    "lookup_texts": candidate.metadata.get("lookup_texts", []),
                }
                for candidate in candidates
            ],
        )
    return candidates


def decide_item_select(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: list[ItemSelectCandidate],
    *,
    position: str,
) -> int:
    decision_state = build_decision_state(
        app,
        ctx,
        phase="item_select",
        position=position,
        candidates=candidates,
        reason="item_select_decision",
    )
    decision = invoke_decision_strategy(
        ctx.item_select_strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        return resolve_candidate_index(decision, candidates)

    # 无 LLM 决策，使用默认兜底
    from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper
    DecisionDumper.get_instance().update_last_resolved(
        resolved_index=0,
        resolved_name=getattr(candidates[0], "title", "") if candidates else "",
        fallback_used=True,
        fallback_reason="默认首选",
    )
    return 0


# ── P物品探查（点击识别）──────────────────────────────────────────


def _extract_selected_item_name(app: "AppProcessor") -> str | None:
    """从已选中的P物品面板中 OCR 提取物品名称。

    选中物品后，面板上方会显示物品名称和效果描述。
    利用 YOLO 检测到的 Special Item 位置作为锚点，
    裁切物品图标上方的区域进行 OCR，取第一行有效文本作为物品名。

    面板布局（选中状态）:
      ┌─────────────────────────────┐
      │    はつぼし時計              │  ← 物品名（OCR 目标）
      │                             │
      │  ターン開始時、全力値が...   │  ← 效果描述（过滤）
      │  ──────────                 │
      │  [item1] [item2] [item3]    │  ← Special Item（YOLO 锚点）
      │  [     受け取る     ]       │
      └─────────────────────────────┘
    """
    frame = getattr(app, "latest_frame", None)
    results = getattr(app, "latest_results", None)
    if frame is None or getattr(frame, "size", 0) <= 0 or results is None:
        return None

    def _normalize_item_panel_name(text: str) -> str:
        cleaned = normalize_ocr_jp(fullwidth_to_halfwidth(str(text or "")))
        cleaned = _ITEM_SELECT_NAME_NOISE_RE.sub("", cleaned).strip()
        return cleaned

    def _looks_like_effect_text(text: str) -> bool:
        normalized = fullwidth_to_halfwidth(str(text or "")).strip()
        if not normalized:
            return False
        return normalized.startswith(_ITEM_SELECT_EFFECT_PREFIXES)

    def _is_plausible_item_name(text: str) -> bool:
        normalized = _normalize_item_panel_name(text)
        if len(normalized) < 2:
            return False
        if _looks_like_effect_text(normalized):
            return False
        if _ITEM_SELECT_EFFECT_LINE_RE.search(normalized):
            return False
        if any(token in normalized for token in _ITEM_SELECT_TITLE_NOISE_TOKENS):
            return False
        return bool(_ITEM_SELECT_JP_CHAR_RE.search(normalized))

    items = sorted(results.filter_by_label(ProducerLabels.SPECIAL_ITEM), key=lambda b: b.cx)
    if not items:
        return None

    h = frame.shape[0]
    debugger = getattr(app, "debug_tools", None)

    panel_rect = detect_bottom_white_modal_region(
        frame,
        row_boxes=items,
        debug_tools=debugger,
        debug_label="item_select_white_panel",
    )
    if panel_rect is not None:
        px1, py1, px2, py2 = panel_rect
        if px2 > px1 + 20 and py2 > py1 + 20:
            panel_crop = frame[py1:py2, px1:px2]
            ocr_result_list = _ITEM_SELECT_SCREEN_OCR.ocr(panel_crop)
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
                item_top = min(int(getattr(box, "y", h)) for box in items)
                boundary = int(item_top - py1 - panel_h * 0.05)
                if boundary > int(panel_h * 0.15):
                    title_bottom = min(title_bottom, boundary)
                best_line: tuple[float, str, tuple[int, int, int, int]] | None = None
                for line in merged_lines:
                    line_text = _normalize_item_panel_name(str(getattr(line, "text", "") or ""))
                    if not line_text:
                        continue
                    line_y = int(getattr(line, "y", 0))
                    line_h = max(1, int(getattr(line, "h", 0)))
                    line_cy = int(getattr(line, "cy", line_y + line_h // 2))
                    if line_cy < 0 or line_cy > title_bottom:
                        continue
                    if not _is_plausible_item_name(line_text):
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
                    _, item_name, (bx1, by1, bx2, by2) = best_line
                    if debugger is not None:
                        debugger.add_box(
                            bx1,
                            by1,
                            bx2,
                            by2,
                            label=f"item_select_title:{item_name[:24]}",
                            color=(80, 220, 120),
                            alpha=0.16,
                            duration=2.5,
                            font_size=14,
                        )
                    logger.debug(
                        "item_select: 信息面板 OCR 物品名={!r} (panel y={}..{}, x={}..{})",
                        item_name, py1, py2, px1, px2,
                    )
                    return item_name

    return None


def _auto_collect_unresolved_item_image(box, index: int) -> None:
    """CLIP 识别失败时自动采集未识别的P物品图像，用于后续人工标注和学习。"""
    frame = getattr(box, "frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return
    try:
        import cv2

        collect_dir = resolve_data_str("CLIP", "unresolved_item")
        os.makedirs(collect_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(collect_dir, f"item_{ts}_{index}.png")
        cv2.imwrite(path, frame)
        logger.info(f"[CLIP] 未识别P物品已采集至: {path}")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[CLIP] P物品自动采集失败: {exc}")


def _probe_unresolved_items(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: list[ItemSelectCandidate],
) -> int | None:
    """对未识别的P物品逐个点击，从选中面板 OCR 提取名称并匹配数据库。

    流程：
      1. 筛选出 db_id 为空且未达最大探查次数的候选项
      2. 逐个点击 → 等待 UI 刷新 → OCR 提取物品名
      3. 用 OCR 名称调用 resolve_produce_item_identity（自动匹配 DB + CLIP 学习）
      4. 更新候选项的 db_id / title / metadata

    注意：点击物品只切换选中状态，不会触发领取（需要按「受け取る」确认）。

    Returns:
        最后一个被点击的候选项 index（用于避免重复点击），无探查时返回 None。
    """
    unresolved = [
        c for c in candidates
        if not c.db_id and not _should_skip_probe(ctx, c)
    ]
    if not unresolved:
        return None

    logger.info(
        "item_select: {} 个P物品未识别，开始逐个点击探查",
        len(unresolved),
    )

    last_probed_index: int | None = None

    for candidate in unresolved:
        # 递增探查计数
        probe_count = _increment_probe_count(ctx, candidate)

        try:
            # 自动采集未识别的物品图像
            _auto_collect_unresolved_item_image(candidate.box, candidate.index)

            # 点击物品切换到选中状态（面板显示物品名和描述）
            app.device.click_element(candidate.box)
            time.sleep(_PROBE_TAP_WAIT)
            time.sleep(_PROBE_INFERENCE_WAIT)

            results = app.latest_results
            frame = app.latest_frame
            if results is None or frame is None:
                continue

            # 从选中面板 OCR 提取物品名
            name = _extract_selected_item_name(app)
            if not name:
                logger.debug(
                    "item_select: P物品 #{} 点击后未提取到名称",
                    candidate.index,
                )
                continue

            # 用 OCR 名称重新走解析管线（CLIP 学习 + DB 匹配）
            resolution = resolve_produce_item_identity(
                name,
                app=app,
                box=candidate.box,
                index=candidate.index,
            )

            # 更新候选项
            candidate.action_id = resolution.action_id
            candidate.db_id = resolution.db_id
            candidate.source = resolution.source
            candidate.confidence = resolution.confidence
            candidate.title = resolution.display_name or name
            candidate.metadata.update(resolution.metadata)

            last_probed_index = candidate.index

            if candidate.db_id:
                logger.info(
                    "item_select: P物品 #{} 探查成功: \"{}\" → db_id={}",
                    candidate.index, name, candidate.db_id,
                )
            else:
                if probe_count >= _MAX_PROBE_RETRIES:
                    # 达到最大探查次数，标记为已尽力，不再重复探查
                    candidate.metadata["probed_max"] = True
                    logger.warning(
                        "item_select: P物品 #{} 已达最大探查次数({})，OCR=\"{}\" 仍未匹配，跳过后续探查",
                        candidate.index, _MAX_PROBE_RETRIES, name,
                    )
                else:
                    logger.warning(
                        "item_select: P物品 #{} OCR=\"{}\" 但数据库未匹配 (探查 {}/{})",
                        candidate.index, name, probe_count, _MAX_PROBE_RETRIES,
                    )
        except Exception as exc:
            logger.warning(
                "item_select: P物品 #{} 探查异常: {}",
                candidate.index, exc,
            )

    return last_probed_index


def _click_receive_button(app: "AppProcessor") -> bool:
    """点击激活的「受け取る」按钮。"""
    # 优先找 Confirm Button
    confirm = app.latest_results.filter_by_label(ProducerLabels.CONFIRM_BUTTON)
    if confirm:
        app.device.click_element(confirm.first())
        return True
    # 其次找 Universal button
    buttons = list(app.latest_results.filter_by_label(BaseUILabels.BUTTON))
    if buttons:
        # 取最靠下的按钮（受け取る通常在底部）
        btn = max(buttons, key=lambda b: b.cy)
        app.device.click_element(btn)
        return True
    return False


class ItemSelectHandler(GameplayHandler):
    """Pアイテム選択画面処理。

    idle 状態: Special Item は検出されるがボタンが無効（Disable）→ アイテムを選択
    selected 状態: ボタンが有効 → 受け取るをタップ
    """

    phase_tag = GameplayPhase.ITEM_SELECT
    priority = 50  # 与 LESSON / SCHEDULE 等同级

    def can_handle(self, app, ctx, phase, position):
        return phase == GameplayPhase.ITEM_SELECT

    def handle(self, app, ctx, phase, position):
        if position == GameplayPosition.ITEM_SELECT_SELECTED:
            # 探查模式下不点击受け取る，等探查完成后再决策
            if ctx.handler_state.get("_item_probe_active"):
                return HandlerResult.no_action("item_select: 探查中，跳过确认")
            # 已选择物品，点击受け取る
            if _click_receive_button(app):
                ctx.handler_state["item_select_idle_streak"] = 0
                # 同步已选 P-item 到 deck_mutations
                pending = ctx.handler_state.pop("pending_item_select_db_id", None)
                pending_name = ctx.handler_state.pop("pending_item_select_name", None)
                if pending:
                    ctx.mutate_deck_acquire(
                        pending,
                        kind="produce_item",
                        name=pending_name or "",
                        source="item_select",
                    )
                return HandlerResult.ok("item_select: 确认受取", sleep_after=1.0)
            return HandlerResult.no_action("item_select: 无法找到确认按钮")

        # idle — 选择一个物品
        streak = ctx.handler_state.get("item_select_idle_streak", 0) + 1
        ctx.handler_state["item_select_idle_streak"] = streak

        candidates = collect_item_select_candidates(app)
        if candidates:
            # 从缓存恢复之前探查的识别结果
            _apply_item_cache(ctx, candidates)

            # 探查未识别的P物品（跳过已达最大探查次数的）
            has_unresolved = any(
                not c.db_id and not _should_skip_probe(ctx, c) for c in candidates
            )
            last_probed_index = None
            if has_unresolved:
                ctx.handler_state["_item_probe_active"] = True
                try:
                    last_probed_index = _probe_unresolved_items(app, ctx, candidates)
                finally:
                    ctx.handler_state["_item_probe_active"] = False

            # 将探查结果写入缓存
            _save_item_cache(ctx, candidates)

            target_index = decide_item_select(app, ctx, candidates, position=position)
            target = candidates[target_index]

            # 暂存选中目标的 db_id/名称，确认时同步到上下文
            if target.db_id:
                ctx.handler_state["pending_item_select_db_id"] = target.db_id
                ctx.handler_state["pending_item_select_name"] = target.title or ""

            # 如果目标物品在探查中已被选中，无需再次点击（避免取消选中）
            if last_probed_index is not None and target.index == last_probed_index:
                logger.debug(
                    "item_select: 目标物品 #{} 在探查中已选中，跳过点击",
                    target.index,
                )
                return HandlerResult.ok("item_select: 探查后已选中目标", sleep_after=0.5)

            app.device.click_element(target.box)
            logger.debug(
                "item_select: 选择物品 {} {}",
                target.index,
                target.title or target.action_id or target.db_id,
            )
            return HandlerResult.ok("item_select: 选择物品", sleep_after=0.8)

        # 无 Special Item 检测到 → 可能是过渡帧
        if streak >= 5:
            logger.warning("item_select: 连续无法选择物品，尝试点击屏幕推进")
            from src.core.tasks.producer_challenge.shared.common import click_relative_point
            click_relative_point(app, x_ratio=0.5, y_ratio=0.7, label="item_select_advance")
            ctx.handler_state["item_select_idle_streak"] = 0
            return HandlerResult.ok("item_select: 强制推进", sleep_after=1.0)

        return HandlerResult.no_action("item_select: 等待 Special Item 出现")
