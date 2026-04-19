from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Set

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.shared.common import click_relative_point
from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper
from src.utils.logger import logger
from src.utils.string_tools import fullwidth_to_halfwidth

from src.core.tasks.producer_challenge.shared.common import (
    invoke_decision_strategy,
    ocr_text,
    resolve_candidate_index,
)
from .decision import (
    _apply_resolution,
    _learn_card_clip_from_db_id,
    _learn_drink_clip_from_db_id,
    build_decision_state,
    hydrate_card_candidates,
    is_end_turn_action_id,
    is_produce_drink_action_id,
    resolve_produce_card_identity,
    resolve_produce_drink_identity,
    score_produce_drink_metadata,
)

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


_CARD_LABEL_PRIORITY = (
    ProducerLabels.SKILL_CARD_ACTIVE,
    ProducerLabels.SKILL_CARD_MENTAL,
    ProducerLabels.SKILL_CARD_TRAP,
)
_BATTLE_BLOCKED_CARD_STATE_KEY = "battle_blocked_cards"
_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY = "battle_last_attempted_card"
_PENDING_LESSON_CARD_POINT_STATE_KEY = "pending_lesson_click_point"
_PENDING_LESSON_CARD_ACTION_ID_STATE_KEY = "pending_lesson_action_id"
_PENDING_LESSON_CARD_DB_ID_STATE_KEY = "pending_lesson_db_id"
_CARD_DOUBLE_TAP_INTERVAL = 0.2
_VERIFY_CARD_PLAYED_POLL_SLEEP = 0.16
_VERIFY_CARD_PLAYED_STABLE_CLEAR_POLLS = 3
_BATTLE_DEAL_SETTLE_SAMPLE_SLEEP = 0.18
_BATTLE_DEAL_SETTLE_MAX_POLLS = 6
_BATTLE_DEAL_SETTLE_STABLE_COUNT_STREAK = 2
_BATTLE_DEAL_SETTLE_STABLE_BASELINE_STREAK = 2
_BATTLE_CARD_BASELINE_TOLERANCE_RATIO = 0.018
_BATTLE_CARD_BASELINE_TOLERANCE_MIN = 18
_BATTLE_CARD_BASELINE_TOLERANCE_MAX = 56
_CRITICAL_BATTLE_STAMINA_RATIO = 0.18
_LOW_BATTLE_STAMINA_RATIO = 0.32
_END_TURN_HOTSPOT_X_RATIO = 0.4
_EMPTY_HAND_NOTICE_X1_RATIO = 0.18
_EMPTY_HAND_NOTICE_X2_RATIO = 0.84
_EMPTY_HAND_NOTICE_Y1_RATIO = 0.68
_EMPTY_HAND_NOTICE_Y2_RATIO = 0.86
_EMPTY_HAND_BLANK_SLOT_Y_RATIO = 0.84
_EMPTY_HAND_STRONG_BLANK_SLOT_COUNT = 3
_BATTLE_PLAN_TOKENS = {
    ProduceText.PLAN_SENSE: ProduceText.BATTLE_SENSE_TOKENS,
    ProduceText.PLAN_LOGIC: ProduceText.BATTLE_LOGIC_TOKENS,
    ProduceText.PLAN_ANOMALY: ProduceText.BATTLE_ANOMALY_TOKENS,
}
_BATTLE_EXTRA_PLAY_TOKENS = (
    ProduceText.SKILL_CARD_USE_COUNT_UP,
    ProduceText.SKILL_CARD_USE_COUNT_UP_SHORT,
)
_BATTLE_RECOVERY_TOKENS = ProduceText.BATTLE_RECOVERY_TOKENS
_BATTLE_SETUP_TOKENS = ProduceText.BATTLE_SETUP_TOKENS
_BATTLE_IMMEDIATE_OUTPUT_TOKENS = (
    *ProduceText.BATTLE_IMMEDIATE_OUTPUT_TOKENS,
    "打分",
    "固定打分",
)

# 空白区域坐标（用于取消卡片选中）
_DESELECT_TAP_Y = 800

# ── 信息面板探查常量（用于未识别卡片的单击读取） ──
_CARD_INFO_PANEL_TAP_WAIT = 0.5   # 点击卡片后等待信息面板出现的秒数
_CARD_INFO_PANEL_INFER_WAIT = 0.4  # 等待 YOLO 推理完成的秒数
_CARD_INFO_PANEL_DESELECT_WAIT = 0.4  # 取消选中后等待恢复的秒数
# 信息面板 YOLO 标签（展示卡片详细信息的弹出面板）
_CARD_INFO_PANEL_LABELS = (
    ProducerLabels.SKILL_CARD_INFO,
    ProducerLabels.PC_ACTION_INFO,
)

# ── 饮料模态探查常量（用于未识别 P 饮料的单击读取） ──
# 纯状态驱动轮询：每次短暂 sleep 后检查 YOLO 结果，检测到目标立刻退出
_DRINK_MODAL_POLL_SLEEP = 0.3   # 轮询间歇（仅防忙等，不作为计时依据）
_DRINK_MODAL_MAX_POLLS = 20     # 最大轮询次数（足够覆盖慢设备，快设备会提前退出）
_DRINK_MODAL_STABLE_POLLS = 2   # 目标状态连续命中次数（防止旧帧/残帧误触发）
_DRINK_MODAL_NAME_MIN_CONF = 0.35
_DRINK_MODAL_NAME_MIN_SCORE = 1.1
_DRINK_MODAL_DB_MATCH_MIN_CONF = 0.8
# 模态头 OCR 排除的文本（标题、按钮等）
_DRINK_MODAL_HEADER_TEXT = ProduceText.P_DRINK_DETAIL
_DRINK_MODAL_EXCLUDE_TEXTS = (
    ProduceText.P_DRINK_DETAIL,
    ProduceText.P_DRINK_DISCARD,
    ButtonText.CANCEL,
    ProduceText.P_DRINK_USE,
)

# ── 饮料模态识别结果缓存 ──
# 避免每次循环重复弹模态识别同一瓶饮料
_DRINK_CACHE_KEY = "_lesson_drink_resolved_cache"
_DRINK_CACHE_SCOPE_KEY = "_lesson_drink_cache_scope"
_DRINK_CACHE_POS_TOLERANCE = 30  # 像素容差
_DRINK_MAX_PROBE = 2  # 同一饮料最大模态探查次数
_DRINK_PROBE_COUNT_KEY = "_lesson_drink_probe_count"


@dataclass
class LessonCardCandidate:
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
class LessonStepResult:
    status: str
    candidate: LessonCardCandidate


def _resolve_box_horizontal_bounds(box: Any) -> tuple[int | None, int | None]:
    left = getattr(box, "x", None)
    right = getattr(box, "w", None)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and right > left:
        return int(left), int(right)
    return None, None


def _battle_turn_marker(ctx: "ProduceContext", phase: str) -> tuple[str, int, int]:
    return (
        str(phase or ""),
        int(ctx.current_week or 0),
        int(ctx.parameter_state.get("remaining_turns") or -1),
    )


def _candidate_block_keys(candidate: LessonCardCandidate) -> set[str]:
    return {
        str(value)
        for value in (
            candidate.action_id,
            candidate.db_id,
            candidate.title,
            candidate.label,
        )
        if str(value or "").strip()
    }


def _current_blocked_card_indices(
    ctx: "ProduceContext",
    candidates: List[LessonCardCandidate],
    *,
    phase: str,
) -> Set[int]:
    blocked_state = dict(ctx.handler_state.get(_BATTLE_BLOCKED_CARD_STATE_KEY, {}) or {})
    if blocked_state.get("turn_marker") != _battle_turn_marker(ctx, phase):
        ctx.handler_state.pop(_BATTLE_BLOCKED_CARD_STATE_KEY, None)
        return set()
    blocked_keys = {
        str(key)
        for key in blocked_state.get("keys", [])
        if str(key or "").strip()
    }
    if not blocked_keys:
        return set()
    return {
        candidate.index
        for candidate in candidates
        if _candidate_block_keys(candidate) & blocked_keys
    }


def _remember_last_attempted_card(
    ctx: "ProduceContext",
    candidate: LessonCardCandidate,
    *,
    phase: str,
) -> None:
    ctx.handler_state[_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY] = {
        "turn_marker": _battle_turn_marker(ctx, phase),
        "title": candidate.title or candidate.label,
        "action_id": candidate.action_id,
        "db_id": candidate.db_id,
    }


def _set_pending_lesson_target(
    ctx: "ProduceContext",
    candidate: LessonCardCandidate,
) -> None:
    ctx.pending_lesson_card_index = candidate.index
    ctx.pending_lesson_card_label = candidate.title or candidate.label or candidate.action_id
    ctx.handler_state[_PENDING_LESSON_CARD_ACTION_ID_STATE_KEY] = candidate.action_id
    ctx.handler_state[_PENDING_LESSON_CARD_DB_ID_STATE_KEY] = candidate.db_id
    if candidate.box is not None and hasattr(candidate.box, "get_COL"):
        x, y = candidate.box.get_COL()
        ctx.handler_state[_PENDING_LESSON_CARD_POINT_STATE_KEY] = (int(x), int(y))


def _build_pending_lesson_candidate(ctx: "ProduceContext") -> LessonCardCandidate:
    return LessonCardCandidate(
        index=int(ctx.pending_lesson_card_index or -1),
        label="pending_lesson_card",
        title=ctx.pending_lesson_card_label,
        selected=True,
        box=None,
        action_id=str(ctx.handler_state.get(_PENDING_LESSON_CARD_ACTION_ID_STATE_KEY, "") or ""),
        db_id=str(ctx.handler_state.get(_PENDING_LESSON_CARD_DB_ID_STATE_KEY, "") or ""),
    )


def _find_lesson_candidate_by_index(
    candidates: List[LessonCardCandidate],
    candidate_index: int | None,
) -> LessonCardCandidate | None:
    if candidate_index is None:
        return None
    for candidate in candidates:
        if int(getattr(candidate, "index", -1)) == int(candidate_index):
            return candidate
    if 0 <= int(candidate_index) < len(candidates):
        return candidates[int(candidate_index)]
    return None


def _tap_pending_lesson_card(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    fallback_box: Any = None,
    tap_label: str = "pending_lesson_card",
) -> bool:
    point = ctx.handler_state.get(_PENDING_LESSON_CARD_POINT_STATE_KEY)
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        app.device.click(int(point[0]), int(point[1]), tap_label)
        return True
    if fallback_box is not None:
        app.device.click_element(fallback_box)
        return True
    return False


def _normalize_battle_notice_text(text: str | None) -> str:
    return "".join(fullwidth_to_halfwidth(str(text or "")).split())


def _looks_like_empty_hand_notice(text: str | None) -> bool:
    normalized = _normalize_battle_notice_text(text)
    if not normalized:
        return False
    return (
        ProduceText.HAND in normalized
        and ProduceText.SKILL_CARD in normalized
        and any(token in normalized for token in ProduceText.ZERO_CARDS_OCR_VARIANTS)
    )


def _ocr_battle_empty_hand_notice(
    app: "AppProcessor",
    *,
    blank_slots: list[Any],
) -> str:
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return ""
    frame_height, frame_width = frame.shape[:2]
    x1 = int(frame_width * _EMPTY_HAND_NOTICE_X1_RATIO)
    x2 = int(frame_width * _EMPTY_HAND_NOTICE_X2_RATIO)
    y1 = int(frame_height * _EMPTY_HAND_NOTICE_Y1_RATIO)
    y2 = int(frame_height * _EMPTY_HAND_NOTICE_Y2_RATIO)
    if blank_slots:
        blank_top = min(int(getattr(box, "y", 0) or 0) for box in blank_slots)
        avg_height = max(
            int(
                sum(
                    max(int(getattr(box, "h", 0) or 0) - int(getattr(box, "y", 0) or 0), 1)
                    for box in blank_slots
                )
                / max(len(blank_slots), 1)
            ),
            1,
        )
        y1 = max(y1, blank_top - int(avg_height * 4.8))
        y2 = min(y2, blank_top - int(avg_height * 1.3))
    if y2 <= y1 or x2 <= x1:
        return ""
    crop = frame[y1:y2, x1:x2]
    if crop.size <= 0:
        return ""
    debugger = getattr(app, "debug_tools", None)
    if debugger is not None:
        debugger.add_box(
            x1,
            y1,
            x2,
            y2,
            label="battle_empty_hand_notice",
            color=(255, 180, 80),
            alpha=0.12,
            duration=2.5,
            font_size=16,
        )
    return ocr_text(crop)


def _is_battle_empty_hand_observed(app: "AppProcessor") -> bool:
    results = getattr(app, "latest_results", None)
    frame = getattr(app, "latest_frame", None)
    if results is None or frame is None or getattr(frame, "size", 0) <= 0:
        return False
    if any(results.filter_by_label(label) for label in _CARD_LABEL_PRIORITY):
        return False
    frame_height = frame.shape[0]
    blank_slots = [
        box
        for box in results.filter_by_label(BaseUILabels.BLANK_SLOT)
        if int(getattr(box, "cy", 0) or 0) >= int(frame_height * _EMPTY_HAND_BLANK_SLOT_Y_RATIO)
    ]
    notice_text = _ocr_battle_empty_hand_notice(app, blank_slots=blank_slots)
    if _looks_like_empty_hand_notice(notice_text):
        logger.info("lesson: OCR 识别到无手牌提示 {!r}", notice_text)
        return True
    if len(blank_slots) >= _EMPTY_HAND_STRONG_BLANK_SLOT_COUNT:
        logger.info("lesson: 识别到 {} 个 Blank Slot，判定当前无手牌", len(blank_slots))
        return True
    return False


def _count_visible_battle_cards(results: Any) -> int:
    if results is None:
        return 0
    return sum(len(results.filter_by_label(label)) for label in _CARD_LABEL_PRIORITY)


def _collect_visible_battle_card_center_ys(results: Any) -> list[int]:
    if results is None:
        return []
    centers: list[int] = []
    for label in _CARD_LABEL_PRIORITY:
        for box in results.filter_by_label(label):
            cy = getattr(box, "cy", None)
            if isinstance(cy, (int, float)):
                centers.append(int(cy))
    centers.sort()
    return centers


def _resolve_battle_card_baseline_tolerance(app: "AppProcessor") -> tuple[int, int]:
    frame = getattr(app, "latest_frame", None)
    frame_height = 0
    if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 1:
        frame_height = int(frame.shape[0] or 0)
    if frame_height <= 0:
        tolerance = 30
    else:
        tolerance = int(frame_height * _BATTLE_CARD_BASELINE_TOLERANCE_RATIO)
        tolerance = max(
            _BATTLE_CARD_BASELINE_TOLERANCE_MIN,
            min(_BATTLE_CARD_BASELINE_TOLERANCE_MAX, tolerance),
        )
    stable_delta = max(8, tolerance // 3)
    return tolerance, stable_delta


def _is_battle_card_baseline_settled(
    center_ys: list[int],
    *,
    tolerance: int,
) -> tuple[bool, int | None]:
    if not center_ys:
        return False, None
    baseline = int(center_ys[len(center_ys) // 2])
    if len(center_ys) <= 1:
        return True, baseline
    outliers = [y for y in center_ys if abs(y - baseline) > tolerance]
    if not outliers:
        return True, baseline
    if len(outliers) == 1:
        floating_offset = baseline - outliers[0]
        if floating_offset > 0 and floating_offset <= tolerance * 2:
            return True, baseline
    return False, baseline


def _wait_battle_card_deal_settle(
    app: "AppProcessor",
    *,
    phase: str,
    position: str,
    pending_index: int | None,
) -> None:
    """在 idle 首次读手牌前等待到手牌中心点基线稳定，避免发牌动画中途识别。"""
    if pending_index is not None or not position.endswith("_idle"):
        return
    base_centers = _collect_visible_battle_card_center_ys(getattr(app, "latest_results", None))
    if not base_centers:
        return
    observed_max_count = len(base_centers)
    last_count = None
    stable_count_streak = 0
    stable_baseline_streak = 0
    last_baseline = None
    for poll_idx in range(_BATTLE_DEAL_SETTLE_MAX_POLLS):
        center_ys = _collect_visible_battle_card_center_ys(getattr(app, "latest_results", None))
        if not center_ys:
            return
        tolerance, stable_delta = _resolve_battle_card_baseline_tolerance(app)
        settled, baseline = _is_battle_card_baseline_settled(center_ys, tolerance=tolerance)
        current_count = len(center_ys)
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
            stable_count_streak >= _BATTLE_DEAL_SETTLE_STABLE_COUNT_STREAK
            and stable_baseline_streak >= _BATTLE_DEAL_SETTLE_STABLE_BASELINE_STREAK
        ):
            if poll_idx > 0:
                logger.debug(
                    "{}: 发牌基线已稳定 count={} baseline={} tol={}",
                    phase,
                    current_count,
                    last_baseline,
                    tolerance,
                )
            return
        if poll_idx + 1 >= _BATTLE_DEAL_SETTLE_MAX_POLLS:
            break
        time.sleep(_BATTLE_DEAL_SETTLE_SAMPLE_SLEEP)


def _confirm_selected_lesson_card(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidate: LessonCardCandidate,
    *,
    phase: str,
) -> bool:
    _remember_last_attempted_card(ctx, candidate, phase=phase)
    if not _tap_pending_lesson_card(
        app,
        ctx,
        fallback_box=candidate.box,
        tap_label="confirm_lesson_card",
    ):
        logger.warning("lesson: 缺少待确认卡片坐标，无法补发第二次点击")
        return False
    return _verify_card_played(app)


def _try_use_lesson_card_double_tap(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidate: LessonCardCandidate,
    *,
    phase: str,
) -> bool:
    app.device.click_element(candidate.box)
    _set_pending_lesson_target(ctx, candidate)
    time.sleep(_CARD_DOUBLE_TAP_INTERVAL)
    return _confirm_selected_lesson_card(app, ctx, candidate, phase=phase)


def _extract_battle_info_panel_name(results: Any, frame: Any) -> str | None:
    """从考试/课程信息面板中提取卡名。

    方案: OCR 整个面板区域，按 y 排序取最顶行文本即为卡名。
    排除面板右边距近且宽度小的短字符（体力消耗徽章如 "-2"）。

    点击卡片后弹出的信息面板布局:
      ┌─────────────────────────────────────┐
      │    カード名                    ❤-2  │  ← 第一行 = 卡名（排除右侧费用）
      │         ──────────   Mental         │
      │         効果说明 ...                 │
      └─────────────────────────────────────┘

    Returns:
        标准化后的卡名字符串，未检测到面板时返回 None。
    """
    from src.utils.string_tools import normalize_ocr_jp
    from src.core.inference.ocr_engine import OCRService
    import re

    # 1. 通过 YOLO 找到信息面板
    panel = None
    for label in _CARD_INFO_PANEL_LABELS:
        panels = list(results.filter_by_label(label))
        if panels:
            panel = panels[0]
            break
    if panel is None:
        return None

    # 2. 裁切面板区域（Yolo_Box: x=x1, y=y1, w=x2, h=y2）
    px1, py1, px2, py2 = int(panel.x), int(panel.y), int(panel.w), int(panel.h)
    fh, fw = frame.shape[:2]
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(fw, px2), min(fh, py2)
    if px2 <= px1 + 20 or py2 <= py1 + 20:
        return None
    panel_crop = frame[py1:py2, px1:px2]

    # 3. OCR 整个面板，获取带位置信息的结果
    ocr_svc = OCRService()
    ocr_results = ocr_svc.ocr(panel_crop)
    if not ocr_results or len(ocr_results) == 0:
        return None

    # 4. 按 y 排序，找最顶行文本
    sorted_items = sorted(ocr_results, key=lambda r: r.y)
    panel_w = px2 - px1
    first_y = sorted_items[0].y
    # 同一行判定: y 差值 < 行高的一半（取第一个结果的高度作为参考）
    line_threshold = max(sorted_items[0].h * 0.6, 15)

    first_line_parts = []
    for item in sorted_items:
        if item.y - first_y > line_threshold:
            break
        # 排除靠右边的短字符（体力消耗徽章，如 "-2"）
        right_edge = item.x + item.w
        right_margin = panel_w - right_edge
        if right_margin < panel_w * 0.06 and item.w < panel_w * 0.1:
            logger.debug(
                "battle: 信息面板排除右侧短字符: text=\"{}\" right_margin={}",
                item.text, right_margin,
            )
            continue
        first_line_parts.append(item.text)

    if not first_line_parts:
        return None
    raw_name = ''.join(first_line_parts).strip()
    if not raw_name:
        return None

    # 5. 标准化 OCR: 全角→半角 + 日文形近字修正 + 去杂字符
    cleaned = fullwidth_to_halfwidth(raw_name)
    cleaned = normalize_ocr_jp(cleaned)
    cleaned = re.sub(r'^[\|｜\[\]「」【】\s]+|[\|｜\[\]「」【】\s]+$', '', cleaned).strip()

    if cleaned and cleaned != raw_name:
        logger.debug(
            "battle: 信息面板 OCR 原始=\"{}\" → 标准化=\"{}\"",
            raw_name, cleaned,
        )
    return cleaned or None


def _resolve_unidentified_cards_via_info_panel(
    app: "AppProcessor",
    candidates: list[LessonCardCandidate],
) -> None:
    """对未识别的手牌卡片逐个单击，读取信息面板提取卡名并匹配数据库。

    工作流:
      1. 找出 CLIP + OCR 均未能识别的卡片（db_id 为空）
      2. 对每张未识别卡单击一次 → 弹出信息面板
      3. OCR 面板中的卡名 → 走数据库匹配
      4. 匹配成功则 CLIP 学习（记忆图像 → 下次可直接识别）
      5. 匹配后取消选中（点击空白区域），避免二次点击出牌

    安全机制:
      - 仅单击一次（点击后只显示信息面板，不出牌）
      - 读取完毕后立即取消选中
      - 异常时也保证取消选中
    """
    # 仅处理技能卡类型的候选（排除饮料、SKIP 等）
    unresolved = [
        c for c in candidates
        if not c.db_id
        and c.label in _CARD_LABEL_PRIORITY
        and c.box is not None
    ]
    if not unresolved:
        return

    logger.info(
        "battle: {} 张手牌未识别，开始逐个点击读取信息面板",
        len(unresolved),
    )

    for candidate in unresolved:
        try:
            # 单击卡片 → 弹出信息面板（不会出牌）
            app.device.click_element(candidate.box)
            time.sleep(_CARD_INFO_PANEL_TAP_WAIT)
            time.sleep(_CARD_INFO_PANEL_INFER_WAIT)

            results = app.latest_results
            frame = app.latest_frame
            if results is None or frame is None:
                _deselect_card(app)
                time.sleep(_CARD_INFO_PANEL_DESELECT_WAIT)
                continue

            # 提取信息面板中的卡名
            panel_name = _extract_battle_info_panel_name(results, frame)
            if panel_name is None:
                logger.debug(
                    "battle: 卡片 #{} 点击后未检测到信息面板",
                    candidate.index,
                )
                _deselect_card(app)
                time.sleep(_CARD_INFO_PANEL_DESELECT_WAIT)
                continue

            # 用 OCR 提取的卡名重新走解析管线（CLIP + 数据库匹配）
            resolution = resolve_produce_card_identity(
                app,
                title=panel_name,
                box=candidate.box,
                index=candidate.index,
            )
            _apply_resolution(candidate, resolution)

            if candidate.db_id:
                # 匹配成功: CLIP 学习（用点击前的卡片图标图像）
                card_image = getattr(candidate.box, "frame", None)
                if card_image is not None:
                    _learn_card_clip_from_db_id(
                        app, card_image, candidate.db_id,
                        upgrade_count=int((candidate.metadata or {}).get("upgrade_count") or 0),
                    )
                logger.info(
                    "battle: 卡片 #{} 通过信息面板识别成功: \"{}\" → db_id={}",
                    candidate.index, panel_name, candidate.db_id,
                )
            else:
                logger.warning(
                    "battle: 卡片 #{} 信息面板 OCR=\"{}\" 但数据库未匹配",
                    candidate.index, panel_name,
                )

        except Exception as exc:
            logger.warning(
                "battle: 卡片 #{} 信息面板识别异常: {}",
                candidate.index, exc,
            )
        finally:
            # 必须取消选中，避免残留选中态导致下一次操作误出牌
            _deselect_card(app)
            time.sleep(_CARD_INFO_PANEL_DESELECT_WAIT)


# ─────────────────────────────────────────────────────────────────
# 饮料模态探查：点击未识别的 P 饮料 → 打开详情模态 → OCR 提取饮料名
# ─────────────────────────────────────────────────────────────────

def _looks_like_drink_effect_line(text: str) -> bool:
    normalized = fullwidth_to_halfwidth(str(text or "")).strip()
    if not normalized:
        return True
    if any(symbol in normalized for symbol in ("+", "＋", "%", "％")) and any(ch.isdigit() for ch in normalized):
        return True
    effect_tokens = (
        ProduceText.GOOD_IMPRESSION,
        ProduceText.GENKI,
        ProduceText.YARUKI,
        ProduceText.STAMINA,
        ProduceText.PARAMETER,
        ProduceText.SCORE,
        ProduceText.NOT_MULTIPLE,
    )
    return any(token in normalized for token in effect_tokens)


def _score_drink_modal_name_candidate(
    *,
    text: str,
    item: Any,
    crop_w: int,
    crop_h: int,
) -> float:
    import re

    center_y_ratio = (float(item.y) + float(item.h) * 0.5) / max(float(crop_h), 1.0)
    center_x_ratio = (float(item.x) + float(item.w) * 0.5) / max(float(crop_w), 1.0)
    conf = float(getattr(item, "confidence", 1.0) or 1.0)

    score = 0.0
    if center_y_ratio <= 0.58:
        score += 2.4
    else:
        score -= 1.2
    if 0.12 <= center_x_ratio <= 0.76:
        score += 1.1
    else:
        score -= 0.5
    if re.search(r"[ァ-ヶぁ-ん一-龯]", text):
        score += 1.3
    else:
        score -= 0.8
    if any(ch.isdigit() for ch in text) or any(symbol in text for symbol in ("+", "＋", "%", "％")):
        score -= 2.0
    if _looks_like_drink_effect_line(text):
        score -= 2.2
    if len(text) > 14:
        score -= 1.0
    score += min(max(conf, 0.0), 1.0) * 0.8
    return score


def _extract_drink_modal_name_candidates(
    results: Any,
    frame: Any,
    *,
    debugger: Any = None,
) -> list[str]:
    """从 Pドリンク詳細 模态中提取候选饮料名（按置信排序）。"""
    from src.utils.string_tools import normalize_ocr_jp
    from src.core.inference.ocr_engine import OCRService
    import re

    modal_headers = list(results.filter_by_label(BaseUILabels.MODAL_HEADER))
    if not modal_headers:
        return []
    header = modal_headers[0]

    cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
    fh, fw = frame.shape[:2]
    region_y1 = int(header.h)
    region_y2 = int(cancel_boxes[0].y) if cancel_boxes else int(fh * 0.8)
    region_x1 = max(0, int(header.x) - 10)
    region_x2 = min(fw, int(header.w) + 10)
    if region_y2 <= region_y1 + 20 or region_x2 <= region_x1 + 20:
        return []

    if debugger is not None:
        debugger.add_box(
            region_x1,
            region_y1,
            region_x2,
            region_y2,
            label="battle_drink_modal_ocr",
            color=(120, 190, 255),
            alpha=0.1,
            duration=2.5,
            font_size=16,
        )

    modal_crop = frame[region_y1:region_y2, region_x1:region_x2]
    ocr_results = OCRService().ocr(modal_crop)
    if not ocr_results or len(ocr_results) == 0:
        return []

    crop_w = region_x2 - region_x1
    crop_h = region_y2 - region_y1
    sorted_items = sorted(ocr_results, key=lambda r: r.y)
    ranked: list[tuple[float, str, Any]] = []

    for item in sorted_items:
        text = str(item.text or "").strip()
        if not text or len(text) < 2:
            continue
        item_conf = float(getattr(item, "confidence", 1.0) or 1.0)
        if item_conf < _DRINK_MODAL_NAME_MIN_CONF:
            logger.debug("battle: 饮料模态排除低置信度文本: \"{}\" conf={:.2f}", text, item_conf)
            continue
        if any(exc in text for exc in _DRINK_MODAL_EXCLUDE_TEXTS):
            logger.debug("battle: 饮料模态排除 UI 文本: \"{}\"", text)
            continue
        right_edge = item.x + item.w
        right_margin = crop_w - right_edge
        if right_margin < crop_w * 0.06 and item.w < crop_w * 0.15:
            logger.debug(
                "battle: 饮料模态排除右侧短文本: \"{}\" right_margin={}",
                text, right_margin,
            )
            continue

        cleaned = fullwidth_to_halfwidth(text)
        cleaned = normalize_ocr_jp(cleaned)
        cleaned = re.sub(
            r'^[\|｜\[\]「」【】\s]+|[\|｜\[\]「」【】\s]+$',
            "",
            cleaned,
        ).strip()
        if not cleaned:
            continue
        if _looks_like_drink_effect_line(cleaned):
            continue

        candidate_score = _score_drink_modal_name_candidate(
            text=cleaned,
            item=item,
            crop_w=crop_w,
            crop_h=crop_h,
        )
        if candidate_score < _DRINK_MODAL_NAME_MIN_SCORE:
            continue
        ranked.append((candidate_score, cleaned, item))
        if cleaned != text:
            logger.debug(
                "battle: 饮料模态 OCR 原始=\"{}\" → 标准化=\"{}\"",
                text, cleaned,
            )

    if not ranked:
        return []

    ranked.sort(key=lambda entry: entry[0], reverse=True)
    names: list[str] = []
    seen: set[str] = set()
    for score, name, item in ranked:
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if debugger is not None and len(names) <= 2:
            debugger.add_box(
                region_x1 + int(item.x),
                region_y1 + int(item.y),
                region_x1 + int(item.x + item.w),
                region_y1 + int(item.y + item.h),
                label=f"drink_name:{name}({score:.2f})",
                color=(120, 255, 160),
                alpha=0.16,
                duration=2.5,
                font_size=16,
            )
    return names


def _cancel_drink_modal(app: "AppProcessor") -> bool:
    """关闭饮料详情模态——轮询点击キャンセル直到模态消失。

    纯状态驱动：每次轮询检查 YOLO 结果，检测到キャンセル就点，
    确认 MODAL_HEADER 消失即返回。不依赖固定时间或帧率。
    """
    clicked = False
    hidden_streak = 0
    for _ in range(_DRINK_MODAL_MAX_POLLS):
        time.sleep(_DRINK_MODAL_POLL_SLEEP)
        results = app.latest_results
        has_modal = _is_drink_modal_visible(results)
        if not has_modal:
            hidden_streak += 1
            if hidden_streak >= _DRINK_MODAL_STABLE_POLLS:
                if clicked:
                    logger.debug("battle: 饮料模态已确认关闭")
                return True
            continue
        hidden_streak = 0
        cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
        if cancel_boxes:
            app.device.click_element(cancel_boxes[0])
            logger.debug("battle: 饮料模态点击 キャンセル 关闭")
            clicked = True
    logger.warning("battle: 饮料模态关闭轮询 {} 次仍未确认，可能残留", _DRINK_MODAL_MAX_POLLS)
    return False


def _is_drink_modal_visible(
    results: Any,
    *,
    require_action_button: bool = False,
) -> bool:
    """判断当前是否处于饮料详情模态。"""
    if results is None:
        return False
    headers = list(results.filter_by_label(BaseUILabels.MODAL_HEADER))
    if not headers:
        return False
    if not require_action_button:
        return True
    cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
    confirm_boxes = list(results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
    return bool(cancel_boxes or confirm_boxes)


def _wait_drink_modal_visibility(
    app: "AppProcessor",
    *,
    expected_visible: bool,
    require_action_button: bool = False,
    reason: str = "",
) -> bool:
    """等待饮料模态到达目标可见状态，并要求连续稳定若干帧。"""
    stable_streak = 0
    for _ in range(_DRINK_MODAL_MAX_POLLS):
        time.sleep(_DRINK_MODAL_POLL_SLEEP)
        results = app.latest_results
        visible = _is_drink_modal_visible(
            results,
            require_action_button=require_action_button if expected_visible else False,
        )
        if visible == expected_visible:
            stable_streak += 1
            if stable_streak >= _DRINK_MODAL_STABLE_POLLS:
                return True
        else:
            stable_streak = 0
    logger.debug(
        "battle: 等待饮料模态状态超时 expected_visible={} require_action_button={} reason={}",
        expected_visible,
        require_action_button,
        reason,
    )
    return False


def _confirm_drink_usage_modal(app: "AppProcessor") -> bool:
    """等待饮料详情模态出现，然后点击「使う」确认使用。

    纯状态驱动：轮询检查 YOLO 结果，等待模态出现后点击确认按钮，
    确认 MODAL_HEADER 消失即返回。不依赖固定时间或帧率。
    """
    # 第一阶段：等待模态出现
    if not _wait_drink_modal_visibility(
        app,
        expected_visible=True,
        require_action_button=True,
        reason="drink_use_wait_open",
    ):
        logger.warning("battle: 饮料使用模态等待超时，未检测到模态")
        return False

    # 第二阶段：点击确认按钮并等待模态关闭
    clicked = False
    hidden_streak = 0
    for _ in range(_DRINK_MODAL_MAX_POLLS):
        time.sleep(_DRINK_MODAL_POLL_SLEEP)
        results = app.latest_results
        has_modal = _is_drink_modal_visible(results)
        if not has_modal:
            hidden_streak += 1
            if hidden_streak >= _DRINK_MODAL_STABLE_POLLS:
                if clicked:
                    logger.debug("battle: 饮料使用模态已确认关闭")
                return True
            continue
        hidden_streak = 0
        confirm_boxes = list(results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
        if confirm_boxes:
            app.device.click_element(confirm_boxes[0])
            logger.debug("battle: 饮料使用模态点击确认（使う）")
            clicked = True
    logger.warning("battle: 饮料使用确认轮询 {} 次仍未关闭，可能残留", _DRINK_MODAL_MAX_POLLS)
    return clicked


def _drink_pos_key(box: Any) -> tuple[int, int]:
    """将饮料 box 中心坐标量化为缓存 key。"""
    cx = int(round(getattr(box, "cx", 0) / _DRINK_CACHE_POS_TOLERANCE) * _DRINK_CACHE_POS_TOLERANCE)
    cy = int(round(getattr(box, "cy", 0) / _DRINK_CACHE_POS_TOLERANCE) * _DRINK_CACHE_POS_TOLERANCE)
    return (cx, cy)


def _drink_cache_scope(ctx: "ProduceContext", phase: str) -> tuple[str, int]:
    return (str(phase or ""), int(ctx.current_week or -1))


def _ensure_drink_cache_scope(ctx: "ProduceContext", *, phase: str) -> None:
    scope = _drink_cache_scope(ctx, phase)
    previous_scope = ctx.handler_state.get(_DRINK_CACHE_SCOPE_KEY)
    if previous_scope == scope:
        return
    ctx.handler_state[_DRINK_CACHE_SCOPE_KEY] = scope
    ctx.handler_state[_DRINK_CACHE_KEY] = {}
    ctx.handler_state[_DRINK_PROBE_COUNT_KEY] = {}


def _apply_drink_cache(
    ctx: "ProduceContext",
    candidates: list[LessonCardCandidate],
    *,
    phase: str,
) -> None:
    """从 handler_state 缓存中恢复之前模态识别的饮料结果。"""
    _ensure_drink_cache_scope(ctx, phase=phase)
    cache: dict = ctx.handler_state.get(_DRINK_CACHE_KEY, {})
    if not cache:
        return
    for cand in candidates:
        if cand.db_id or cand.label != ProducerLabels.P_DRINK:
            continue
        key = _drink_pos_key(cand.box)
        cached = cache.get(key)
        if cached is None:
            continue
        cand.db_id = cached.get("db_id", "")
        cand.action_id = cached.get("action_id", "") or cand.action_id
        cand.title = cached.get("title", "") or cand.title
        cand.source = cached.get("source", "cache")
        cand.confidence = cached.get("confidence", 0.0)
        if cached.get("metadata"):
            cand.metadata.update(cached["metadata"])
        logger.debug("lesson: 从缓存恢复饮料 #{} → db_id={}", cand.index, cand.db_id)


def _save_drink_cache(
    ctx: "ProduceContext",
    candidates: list[LessonCardCandidate],
    *,
    phase: str,
) -> None:
    """将已识别的饮料写入缓存。"""
    _ensure_drink_cache_scope(ctx, phase=phase)
    cache: dict = ctx.handler_state.setdefault(_DRINK_CACHE_KEY, {})
    for cand in candidates:
        if not cand.db_id or cand.label != ProducerLabels.P_DRINK:
            continue
        key = _drink_pos_key(cand.box)
        if key in cache:
            continue
        cache[key] = {
            "db_id": cand.db_id,
            "action_id": cand.action_id,
            "title": cand.title,
            "source": cand.source,
            "confidence": cand.confidence,
            "metadata": dict(cand.metadata) if cand.metadata else {},
        }
        logger.debug("lesson: 缓存饮料 pos={} → db_id={}, title={}", key, cand.db_id, cand.title)


def _should_skip_drink_probe(ctx: "ProduceContext", box: Any) -> bool:
    """检查饮料是否已达最大探查次数。"""
    counts: dict = ctx.handler_state.get(_DRINK_PROBE_COUNT_KEY, {})
    return counts.get(_drink_pos_key(box), 0) >= _DRINK_MAX_PROBE


def _increment_drink_probe(ctx: "ProduceContext", box: Any) -> int:
    """递增饮料探查计数，返回新值。"""
    counts: dict = ctx.handler_state.setdefault(_DRINK_PROBE_COUNT_KEY, {})
    key = _drink_pos_key(box)
    count = counts.get(key, 0) + 1
    counts[key] = count
    return count


def _resolve_unidentified_drinks_via_modal(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: list[LessonCardCandidate],
) -> None:
    """对未识别的 P 饮料逐个单击，打开详情模态提取饮料名并匹配数据库。

    工作流:
      1. 找出 CLIP + OCR 均未能识别的饮料（db_id 为空，且未达最大探查次数）
      2. 对每个未识别饮料单击 → 轮询等待「Pドリンク詳細」模态出现
      3. OCR 模态中的饮料名 → 走数据库匹配
      4. 匹配成功则 CLIP 学习（记忆图像 → 下次可直接识别）
      5. 点击「キャンセル」关闭模态（绝不点击「使う」使用饮料）

    安全机制:
      - 使用轮询确认模态出现/关闭，避免时序问题导致模态残留
      - 模态中仅点击「キャンセル」取消，不点击「使う」（使用）
      - 异常时也通过轮询保证关闭模态
    """
    unresolved = [
        c for c in candidates
        if not c.db_id
        and c.label == ProducerLabels.P_DRINK
        and c.box is not None
        and not _should_skip_drink_probe(ctx, c.box)
    ]
    if not unresolved:
        return

    logger.info(
        "battle: {} 个 P 饮料未识别，开始逐个点击读取模态",
        len(unresolved),
    )

    for candidate in unresolved:
        probe_count = _increment_drink_probe(ctx, candidate.box)
        try:
            # 先确认前一个模态已稳定关闭，避免把旧模态内容误当成当前饮料。
            _wait_drink_modal_visibility(
                app,
                expected_visible=False,
                reason=f"drink_probe_pre_close#{candidate.index}",
            )
            # 单击饮料 → 轮询等待「Pドリンク詳細」模态出现
            app.device.click_element(candidate.box)

            # ── 轮询等待模态出现（状态驱动，不依赖固定时间/帧率） ──
            if not _wait_drink_modal_visibility(
                app,
                expected_visible=True,
                require_action_button=True,
                reason=f"drink_probe_wait_open#{candidate.index}",
            ):
                logger.debug(
                    "battle: P 饮料 #{} 点击后 {} 次轮询未检测到模态，跳过",
                    candidate.index, _DRINK_MODAL_MAX_POLLS,
                )
                # 安全起见仍然尝试关闭（可能模态延迟出现）
                _cancel_drink_modal(app)
                continue

            results = app.latest_results
            frame = app.latest_frame
            if results is None or frame is None:
                _cancel_drink_modal(app)
                continue

            # 提取模态中的饮料名候选（按置信度排序）
            modal_names = _extract_drink_modal_name_candidates(
                results,
                frame,
                debugger=getattr(app, "debug_tools", None),
            )
            if not modal_names:
                logger.debug(
                    "battle: P 饮料 #{} 模态 OCR 未提取到饮料名",
                    candidate.index,
                )
                _cancel_drink_modal(app)
                continue

            # 用候选名逐个重走解析管线，只有匹配置信足够高才接受
            resolved_name = ""
            resolved_resolution = None
            for modal_name in modal_names:
                resolution = resolve_produce_drink_identity(
                    modal_name,
                    app=app,
                    box=candidate.box,
                    index=candidate.index,
                    min_ocr_confidence=_DRINK_MODAL_DB_MATCH_MIN_CONF,
                )
                if resolution.db_id:
                    resolved_name = modal_name
                    resolved_resolution = resolution
                    break

            if resolved_resolution is not None:
                _apply_resolution(candidate, resolved_resolution)
                drink_image = getattr(candidate.box, "frame", None)
                if drink_image is not None:
                    _learn_drink_clip_from_db_id(app, drink_image, candidate.db_id)
                logger.info(
                    "battle: P 饮料 #{} 通过模态识别成功: \"{}\" → db_id={}",
                    candidate.index, resolved_name, candidate.db_id,
                )
            else:
                modal_name = modal_names[0]
                if probe_count >= _DRINK_MAX_PROBE:
                    logger.warning(
                        "battle: P 饮料 #{} 已达最大探查次数({})，OCR=\"{}\" 仍未匹配，跳过后续探查",
                        candidate.index, _DRINK_MAX_PROBE, modal_name,
                    )
                else:
                    logger.warning(
                        "battle: P 饮料 #{} 模态 OCR=\"{}\" 但数据库未匹配 (探查 {}/{})",
                        candidate.index, modal_name, probe_count, _DRINK_MAX_PROBE,
                    )

        except Exception as exc:
            logger.warning(
                "battle: P 饮料 #{} 模态识别异常: {}",
                candidate.index, exc,
            )
        finally:
            # 必须关闭模态，避免残留模态阻塞后续操作
            _cancel_drink_modal(app)


def collect_lesson_card_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    phase: str,
    position: str,
) -> List[LessonCardCandidate]:
    """收集当前手牌中的技能卡与底栏 P 饮料候选列表。

    按 Active > Mental > Trap 优先级排列，同类别按 x 坐标左→右排列。
    """
    cards: list[LessonCardCandidate] = []
    pending_index = ctx.pending_lesson_card_index if position == "lesson_selected" else None

    current_index = 0
    for label in _CARD_LABEL_PRIORITY:
        boxes = sorted(app.latest_results.filter_by_label(label), key=lambda item: item.cx)
        for box in boxes:
            cards.append(
                LessonCardCandidate(
                    index=current_index,
                    label=label,
                    title=ocr_text(box.frame),
                    selected=pending_index == current_index,
                    box=box,
                )
            )
            current_index += 1
    hydrate_card_candidates(app, cards)
    # 对 CLIP + OCR 均未识别的卡片，单击打开信息面板读取卡名后匹配数据库
    _resolve_unidentified_cards_via_info_panel(app, cards)
    # 重新回到 lesson / exam 画面时，先用当前帧 HUD 与手牌把 battle 上下文刷到最新，
    # 避免底栏饮料评分和后续决策继续沿用上一帧/上一回合的旧体力、回合数与资源值。
    collect_sync_state = build_decision_state(
        app,
        ctx,
        phase=phase,
        position=position,
        candidates=cards,
        reason=f"{phase}_collect_sync",
    )
    collect_snapshot = dict(collect_sync_state.get("llm_snapshot", {}) or {})
    logger.debug(
        "lesson: 重入画面同步上下文 stamina={}/{} remaining={} resources={}",
        int(collect_snapshot.get("stamina") or ctx.hud_stamina or 0),
        int(collect_snapshot.get("max_stamina") or ctx.hud_max_stamina or 0),
        int(collect_snapshot.get("remaining") or ctx.parameter_state.get("remaining_turns") or 0),
        {
            "block": dict(collect_snapshot.get("resources", {}) or {}).get("block", ""),
            "review": dict(collect_snapshot.get("resources", {}) or {}).get("review", ""),
            "aggressive": dict(collect_snapshot.get("resources", {}) or {}).get("aggressive", ""),
            "parameter_buff": dict(collect_snapshot.get("resources", {}) or {}).get("parameter_buff", ""),
        },
    )
    if position.endswith("_idle"):
        drink_candidates = _collect_battle_drink_candidates(
            app,
            ctx,
            phase=phase,
            start_index=current_index,
        )
        cards.extend(drink_candidates)
        current_index += len(drink_candidates)
    cards.extend(
        _collect_battle_end_turn_candidates(
            app,
            phase=phase,
            start_index=current_index,
        )
    )
    return cards


def _collect_battle_drink_candidates(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    phase: str,
    start_index: int,
) -> List[LessonCardCandidate]:
    _ensure_drink_cache_scope(ctx, phase=phase)
    frame = getattr(app, "latest_frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return []
    frame_height = frame.shape[0]
    drink_boxes = sorted(
        (
            box
            for box in app.latest_results.filter_by_label(ProducerLabels.P_DRINK)
            if box.cy >= frame_height * 0.88
        ),
        key=lambda item: item.cx,
    )
    if not drink_boxes:
        return []

    stamina = int(ctx.hud_stamina or 0)
    max_stamina = int(ctx.hud_max_stamina or 0)
    remaining_turns = int(ctx.parameter_state.get("remaining_turns") or 0)
    candidates: list[LessonCardCandidate] = []
    for offset, box in enumerate(drink_boxes):
        index = start_index + offset
        raw_title = ocr_text(box.frame)
        resolution = resolve_produce_drink_identity(
            raw_title,
            app=app,
            box=box,
            index=index,
            allow_ocr_fallback=False,
        )
        metadata = dict(resolution.metadata or {})
        metadata["candidate_type"] = "battle_p_drink"
        metadata["battle_drink_slot"] = offset + 1
        metadata["drink_score"] = score_produce_drink_metadata(
            metadata,
            phase=phase,
            stamina=stamina,
            max_stamina=max_stamina,
            remaining_turns=remaining_turns,
        )
        candidates.append(
            LessonCardCandidate(
                index=index,
                label="P Drink",
                title=resolution.display_name or raw_title or f"P饮料{offset + 1}",
                selected=False,
                box=box,
                action_id=resolution.action_id or f"produce_drink:unknown:{offset + 1}",
                db_id=resolution.db_id,
                source=resolution.source,
                confidence=resolution.confidence,
                metadata=metadata,
            )
        )
    # 从缓存恢复之前模态识别的饮料结果
    _apply_drink_cache(ctx, candidates, phase=phase)
    # 对 CLIP + OCR + 缓存均未识别的饮料，点击打开模态读取饮料名后匹配数据库
    _resolve_unidentified_drinks_via_modal(app, ctx, candidates)
    # 将新识别的结果写入缓存
    _save_drink_cache(ctx, candidates, phase=phase)
    return candidates


def _collect_battle_end_turn_candidates(
    app: "AppProcessor",
    *,
    phase: str,
    start_index: int,
) -> List[LessonCardCandidate]:
    skip_boxes = _get_battle_end_turn_boxes(getattr(app, "latest_results", None))
    if not skip_boxes:
        return []
    label = "SKIP" if str(phase or "") == "lesson" else "结束回合"
    description = (
        "放弃当前回合剩余出牌，直接推进到下一回合。"
        if str(phase or "") == "exam"
        else "执行 SKIP，放弃当前回合剩余出牌并直接进入下一回合。"
    )
    return [
        LessonCardCandidate(
            index=start_index,
            label=label,
            title=label,
            selected=False,
            box=skip_boxes[0],
            action_id="end_turn",
            source="yolo",
            confidence=float(getattr(skip_boxes[0], "confidence", 0.0) or 0.0),
            metadata={
                "candidate_type": "end_turn",
                "description": description,
                "available": True,
            },
        )
    ]


def _select_forced_battle_drink_index(
    decision_state: dict[str, Any],
    *,
    skip_indices: Set[int],
) -> int | None:
    legal_indices = {
        int(index)
        for index in decision_state.get("legal_actions", [])
        if int(index) not in skip_indices
    }
    if not legal_indices:
        return None

    payloads = list(decision_state.get("candidates", []) or [])
    available_payloads = [
        payload
        for payload in payloads
        if int(payload.get("index", -1)) in legal_indices
    ]
    drink_payloads = [
        payload
        for payload in available_payloads
        if is_produce_drink_action_id(payload.get("id"))
    ]
    if not drink_payloads:
        return None
    card_payloads = [
        payload
        for payload in available_payloads
        if str(payload.get("id") or "").startswith("produce_card:")
    ]

    def drink_score(payload: dict[str, Any]) -> float:
        metadata = dict(payload.get("metadata", {}) or {})
        return float(metadata.get("drink_score") or 0.0)

    best_drink = max(drink_payloads, key=drink_score)
    best_score = drink_score(best_drink)
    snapshot = dict(decision_state.get("llm_snapshot", {}) or {})
    stamina = int(snapshot.get("stamina") or 0)
    max_stamina = int(snapshot.get("max_stamina") or 0)
    stamina_ratio = float(stamina) / max(max_stamina, 1) if max_stamina > 0 else 1.0
    play_limit_remaining = int(snapshot.get("play_limit_remaining") or 1)
    description = str(best_drink.get("description") or best_drink.get("label") or "")
    recovery_drink = any(
        token in description
        for token in (
            *ProduceText.BATTLE_RECOVERY_TOKENS,
            ProduceText.BLOCK,
        )
    )

    if play_limit_remaining <= 0 and best_score > 0:
        return int(best_drink["index"])
    if not card_payloads and best_score > 0:
        return int(best_drink["index"])
    if (
        (stamina <= 3 or stamina_ratio <= _CRITICAL_BATTLE_STAMINA_RATIO)
        and recovery_drink
        and best_score >= 16.0
    ):
        return int(best_drink["index"])
    if (
        (stamina <= 5 or stamina_ratio <= _LOW_BATTLE_STAMINA_RATIO)
        and recovery_drink
        and best_score >= 24.0
    ):
        return int(best_drink["index"])
    return None


def _battle_payload_text(payload: dict[str, Any]) -> str:
    metadata = dict(payload.get("metadata", {}) or {})
    effect_types = " / ".join(str(value or "") for value in metadata.get("effect_types", []) or [])
    return fullwidth_to_halfwidth(
        "；".join(
            value
            for value in (
                str(payload.get("description") or ""),
                str(metadata.get("description") or ""),
                effect_types,
            )
            if str(value or "").strip()
        )
    )


def _battle_has_immediate_output(text: str) -> bool:
    normalized = str(text or "")
    if any(token in normalized for token in _BATTLE_IMMEDIATE_OUTPUT_TOKENS):
        return True
    if ProduceText.PARAMETER_UP_INCREASE in normalized:
        normalized = normalized.replace(ProduceText.PARAMETER_UP_INCREASE, "")
    return (
        ProduceText.PARAMETER in normalized
        and ProduceText.INCREASE in normalized
    )


def _score_battle_payload(
    payload: dict[str, Any],
    *,
    llm_snapshot: dict[str, Any],
    phase: str,
) -> tuple[float, list[str]]:
    metadata = dict(payload.get("metadata", {}) or {})
    text = _battle_payload_text(payload)
    remaining_turns = int(llm_snapshot.get("remaining") or 0)
    stamina = int(llm_snapshot.get("stamina") or 0)
    max_stamina = int(llm_snapshot.get("max_stamina") or 0)
    stamina_ratio = float(stamina) / max(max_stamina, 1) if max_stamina > 0 else 1.0
    exam_ranking = int(llm_snapshot.get("exam_ranking") or 0)
    current_score = int(llm_snapshot.get("score") or 0)
    target_score = int(llm_snapshot.get("target") or 0)
    current_plan = str(llm_snapshot.get("idol_plan_label") or "")
    action_id = str(payload.get("id") or "")
    score = 0.0
    reasons: list[str] = []

    if is_produce_drink_action_id(action_id):
        score += float(metadata.get("drink_score") or 0.0)
        reasons.append("当前是可立即使用的 P 饮料")
    else:
        if _battle_has_immediate_output(text):
            score += 20.0
            reasons.append("能立刻兑现当前回合收益")

    if any(token in text for token in _BATTLE_EXTRA_PLAY_TOKENS):
        score += 26.0
        reasons.append("能追加本回合出牌次数")

    if any(token in text for token in _BATTLE_RECOVERY_TOKENS):
        if stamina <= 4 or stamina_ratio <= _CRITICAL_BATTLE_STAMINA_RATIO:
            score += 24.0
            reasons.append("当前体力偏危险，先补元気/体力更稳")
        elif stamina <= 7 or stamina_ratio <= _LOW_BATTLE_STAMINA_RATIO:
            score += 14.0
            reasons.append("当前体力偏低，续航收益更高")

    plan_tokens = _BATTLE_PLAN_TOKENS.get(current_plan, ())
    matched_plan_tokens = [token for token in plan_tokens if token in text]
    if matched_plan_tokens:
        score += 16.0
        reasons.append(f"契合当前{current_plan}流派核心资源")

    if remaining_turns > 0 and remaining_turns <= 2:
        if _battle_has_immediate_output(text):
            score += 14.0
            reasons.append("剩余回合少，优先立即兑现")
        elif any(token in text for token in _BATTLE_SETUP_TOKENS):
            score -= 12.0
            reasons.append("剩余回合少，纯铺垫价值下降")

    if phase == "exam":
        unsafe_ranking = exam_ranking > 3
        score_gap = target_score - current_score if target_score > 0 else 0
        if unsafe_ranking or (remaining_turns <= 2 and score_gap > 0):
            if _battle_has_immediate_output(text):
                score += 12.0
                reasons.append("考试压力高，立即抢分更重要")
            elif any(token in text for token in _BATTLE_SETUP_TOKENS):
                score -= 10.0
                reasons.append("考试压力高，纯铺垫过慢")

    return score, reasons


def _annotate_battle_preference(
    decision_state: dict[str, Any],
    *,
    preferred_index: int,
    reason: str,
) -> None:
    label = f"候选 {preferred_index}"
    for payload in decision_state.get("candidates", []) or []:
        if int(payload.get("index", -1)) == preferred_index:
            payload["recommended"] = True
            label = str(payload.get("name") or payload.get("label") or label)
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


def _select_battle_preference(
    decision_state: dict[str, Any],
    *,
    preferred_indices: Set[int],
    retryable_indices: Set[int],
    end_turn_indices: Set[int],
    phase: str,
) -> tuple[int | None, float, str]:
    llm_snapshot = dict(decision_state.get("llm_snapshot", {}) or {})
    best_index: int | None = None
    best_score = float("-inf")
    best_reason = ""
    considered_indices = preferred_indices or retryable_indices
    for payload in decision_state.get("candidates", []) or []:
        payload_index = int(payload.get("index", -1))
        if payload_index not in considered_indices or payload_index in end_turn_indices:
            continue
        if not bool(payload.get("available", True)):
            continue
        score, reasons = _score_battle_payload(
            payload,
            llm_snapshot=llm_snapshot,
            phase=phase,
        )
        if score > best_score:
            best_index = payload_index
            best_score = score
            best_reason = "；".join(reasons[:3]) or "综合当前回合收益与流派联动后更优。"
    return best_index, best_score, best_reason


def decide_lesson_card(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: List[LessonCardCandidate],
    *,
    phase: str,
    position: str,
    skip_indices: Set[int] | None = None,
) -> int | None:
    """决定要打出哪张卡片，支持跳过不可用的卡片索引。"""
    strategy = ctx.exam_strategy if phase == "exam" and ctx.exam_strategy is not None else ctx.lesson_strategy
    blocked_indices = _current_blocked_card_indices(ctx, candidates, phase=phase)
    soft_skip_indices = set(skip_indices or set()) - blocked_indices
    decision_state = build_decision_state(
        app,
        ctx,
        phase=phase,
        position=position,
        candidates=candidates,
        reason=f"{phase}_decision",
    )
    unavailable_indices = {
        int(payload.get("index", -1))
        for payload in decision_state.get("candidates", [])
        if not bool(payload.get("available", True))
    }
    hard_skip_indices = blocked_indices | unavailable_indices
    merged_skip_indices = hard_skip_indices | soft_skip_indices
    legal_indices = {
        int(index)
        for index in decision_state.get("legal_actions", []) or []
        if int(index) >= 0
    }
    end_turn_indices = {
        int(payload.get("index", -1))
        for payload in decision_state.get("candidates", []) or []
        if is_end_turn_action_id(payload.get("id"))
    }
    preferred_indices = {
        index
        for index in legal_indices
        if index not in merged_skip_indices
    }
    retryable_indices = {
        index
        for index in legal_indices
        if index not in hard_skip_indices
    }
    preferred_non_end_turn_indices = preferred_indices - end_turn_indices
    retryable_non_end_turn_indices = retryable_indices - end_turn_indices
    logger.debug(
        "lesson: 决策候选={} | legal={} | tried={} | blocked={} | unavailable={} | skip={}",
        [
            dict(filter(lambda item: item[1] not in ("", None), {
                "index": int(payload.get("index", -1)),
                "label": payload.get("name") or payload.get("label") or payload.get("id"),
                "available": bool(payload.get("available", True)),
                "reason": str(
                    payload.get("unavailable_reason")
                    or dict(payload.get("metadata", {}) or {}).get("unavailable_reason")
                    or ""
                ).strip(),
            }.items()))
            for payload in decision_state.get("candidates", [])
        ],
        list(decision_state.get("legal_actions", []) or []),
        sorted(soft_skip_indices),
        sorted(blocked_indices),
        sorted(unavailable_indices),
        sorted(merged_skip_indices),
    )
    forced_drink_index = _select_forced_battle_drink_index(
        decision_state,
        skip_indices=merged_skip_indices,
    )
    if forced_drink_index is not None:
        logger.info("lesson: 本地兜底改为使用底栏饮料 [{}]", forced_drink_index)
        return forced_drink_index
    preferred_index, preferred_score, preferred_reason = _select_battle_preference(
        decision_state,
        preferred_indices=preferred_indices,
        retryable_indices=retryable_indices,
        end_turn_indices=end_turn_indices,
        phase=phase,
    )
    if preferred_index is not None:
        _annotate_battle_preference(
            decision_state,
            preferred_index=preferred_index,
            reason=preferred_reason,
        )
        logger.debug(
            "lesson: 系统推荐索引={} score={:.1f} reason={}",
            preferred_index,
            preferred_score,
            preferred_reason,
        )
    decision = invoke_decision_strategy(
        strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    if decision is not None:
        idx = resolve_candidate_index(decision, candidates)
        logger.debug(
            "lesson: 原始决策={} -> resolved_index={} | skip={}",
            decision,
            idx,
            sorted(merged_skip_indices),
        )
        if idx in preferred_indices:
            if preferred_index is not None and idx != preferred_index and preferred_score >= 24.0:
                # 仅记录参考，不再覆盖 LLM 决策
                logger.info(
                    "lesson: LLM 决策={} vs 系统推荐={} (score={:.1f}, {}) → 信任 LLM",
                    idx,
                    preferred_index,
                    preferred_score,
                    preferred_reason,
                )
            return idx
        if idx in retryable_indices and not preferred_indices:
            logger.info("lesson: 决策索引 {} 已尝试过，但当前只剩该候选可重试", idx)
            return idx
        logger.warning("lesson: 决策索引 {} 已被跳过，回退到本地兜底", idx)

    # ── 本地兜底逻辑（LLM 无决策或决策不可用时）──
    fallback_index: int | None = None
    fallback_reason = ""

    if not preferred_non_end_turn_indices and not retryable_non_end_turn_indices:
        for payload in decision_state.get("candidates", []) or []:
            payload_index = int(payload.get("index", -1))
            if payload_index in preferred_indices and is_end_turn_action_id(payload.get("id")):
                fallback_index = payload_index
                fallback_reason = "无可打出卡片，SKIP"
                break
        if fallback_index is None:
            for payload in decision_state.get("candidates", []) or []:
                payload_index = int(payload.get("index", -1))
                if payload_index in retryable_indices and is_end_turn_action_id(payload.get("id")):
                    fallback_index = payload_index
                    fallback_reason = "仅剩 SKIP 可重试"
                    break

    if fallback_index is None:
        # 决策策略返回的卡片不可用，或无决策 → 按优先级顺序尝试
        if ctx.pending_lesson_card_index is not None and 0 <= ctx.pending_lesson_card_index < len(candidates):
            if ctx.pending_lesson_card_index in preferred_indices:
                fallback_index = ctx.pending_lesson_card_index
                fallback_reason = "pending 索引"
            elif ctx.pending_lesson_card_index in retryable_indices and not preferred_indices:
                fallback_index = ctx.pending_lesson_card_index
                fallback_reason = "pending 索引(重试)"

    if fallback_index is None:
        for payload in decision_state.get("candidates", []):
            payload_index = int(payload.get("index", -1))
            if bool(payload.get("recommended")) and payload_index in preferred_indices:
                fallback_index = payload_index
                fallback_reason = "推荐候选"
                break
            if (
                bool(payload.get("recommended"))
                and payload_index in retryable_indices
                and not preferred_indices
            ):
                fallback_index = payload_index
                fallback_reason = "推荐候选(重试)"
                break

    if fallback_index is None:
        # 回退：选第一个不在跳过列表中的可用动作（技能卡或饮料）
        for c in candidates:
            if c.index in preferred_indices:
                fallback_index = c.index
                fallback_reason = "第一个可用候选"
                break

    if fallback_index is None:
        for c in candidates:
            if c.index in retryable_indices:
                fallback_index = c.index
                fallback_reason = "已尝试候选(重试)"
                break

    if fallback_index is None:
        fallback_index = -1
        fallback_reason = "所有候选已跳过"

    # 更新 dump 记录的最终执行结果
    if fallback_reason:
        _dumper = DecisionDumper.get_instance()
        resolved_name = ""
        if 0 <= fallback_index < len(candidates):
            resolved_name = getattr(candidates[fallback_index], "title", "")
        _dumper.update_last_resolved(
            resolved_index=fallback_index,
            resolved_name=resolved_name,
            fallback_used=True,
            fallback_reason=fallback_reason,
        )
        logger.info("lesson: 兜底决策 idx={} reason={}", fallback_index, fallback_reason)

    return fallback_index


def _verify_card_played(app: "AppProcessor", timeout: float = 1.5) -> bool:
    """验证卡片是否成功打出。

    检查 Skill Card Info 面板是否消失，且没有转入确认弹窗；
    只有真正离开卡牌信息面板时才认为出牌成功。
    """
    deadline = time.monotonic() + timeout
    stable_clear_polls = 0
    time.sleep(0.35)
    while time.monotonic() < deadline:
        results = getattr(app, "latest_results", None)
        if results is None:
            stable_clear_polls = 0
            time.sleep(_VERIFY_CARD_PLAYED_POLL_SLEEP)
            continue
        if results.exists_label(ProducerLabels.MODAL_HEADER):
            return False
        if (
            results.exists_label(ProducerLabels.CONFIRM_BUTTON)
            or results.exists_label(ProducerLabels.CANCEL_BUTTON)
        ):
            return False
        info_visible = (
            results.exists_label(ProducerLabels.SKILL_CARD_INFO)
            or results.exists_label(ProducerLabels.PC_ACTION_INFO)
            or results.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED)
        )
        if info_visible:
            stable_clear_polls = 0
            time.sleep(_VERIFY_CARD_PLAYED_POLL_SLEEP)
            continue
        stable_clear_polls += 1
        if stable_clear_polls >= _VERIFY_CARD_PLAYED_STABLE_CLEAR_POLLS:
            return True
        time.sleep(_VERIFY_CARD_PLAYED_POLL_SLEEP)
    return False


def _deselect_card(app: "AppProcessor") -> None:
    """点击空白区域取消卡片选中。"""
    # 使用屏幕中部偏上区域（角色立绘区，不会触发任何UI元素）
    screen_width = 1080  # 标准竖屏宽度
    app.device.click(screen_width // 2, _DESELECT_TAP_Y, el_label="deselect_card")
    time.sleep(0.5)


def _get_battle_end_turn_boxes(results: Any) -> List[Any]:
    if results is None or not hasattr(results, "filter_by_label"):
        return []
    return sorted(
        list(results.filter_by_label(ProducerLabels.PC_SKIP)),
        key=lambda item: (item.cy, item.cx),
    )


def _click_battle_end_turn(
    app: "AppProcessor",
    *,
    fallback_box: Any = None,
) -> bool:
    target_box = fallback_box
    skip_boxes = _get_battle_end_turn_boxes(getattr(app, "latest_results", None))
    if skip_boxes:
        target_box = skip_boxes[0]
    if target_box is None:
        return False
    left, right = _resolve_box_horizontal_bounds(target_box)
    if left is None or right is None:
        app.device.click_element(target_box)
        return True
    hotspot_offset = int(max((right - left) * _END_TURN_HOTSPOT_X_RATIO, 50))
    tap_x = int(getattr(target_box, "cx", 0) - hotspot_offset)
    tap_x = max(left + 8, min(right - 8, tap_x))
    tap_y = int(getattr(target_box, "cy", 0))
    logger.debug(
        "lesson: 点击结束回合按钮 ({}, {}) box=({}, {}, {}, {})",
        tap_x,
        tap_y,
        left,
        int(getattr(target_box, "y", 0)),
        right,
        int(getattr(target_box, "h", 0)),
    )
    app.device.click(tap_x, tap_y, el_label="battle_end_turn")
    return True


def _try_resolve_empty_hand_action(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    phase: str,
    position: str,
):
    """空手牌时，把底栏饮料与 SKIP 重新组为候选，交给现有决策链判断。"""
    from src.core.tasks.producer_challenge.gameplay.handler_base import HandlerResult

    phase_tag = str(phase or "lesson")
    decision_position = position if str(position or "").endswith("_idle") else f"{phase}_idle"
    build_decision_state(
        app,
        ctx,
        phase=phase,
        position=decision_position,
        candidates=[],
        reason=f"{phase}_empty_hand_sync",
    )
    fallback_candidates = _collect_battle_drink_candidates(
        app,
        ctx,
        phase=phase,
        start_index=0,
    )
    fallback_candidates.extend(
        _collect_battle_end_turn_candidates(
            app,
            phase=phase,
            start_index=len(fallback_candidates),
        )
    )
    if not fallback_candidates:
        return None

    chosen_index = decide_lesson_card(
        app,
        ctx,
        fallback_candidates,
        phase=phase,
        position=decision_position,
    )
    target = _find_lesson_candidate_by_index(fallback_candidates, chosen_index)
    if target is None:
        return None
    if is_produce_drink_action_id(target.action_id):
        logger.info("{}: 空手牌 fallback 选择底栏饮料 [{}] {!r}", phase_tag, target.index, target.title)
        ctx.clear_lesson_pending()
        ctx.pending_p_drink_index = target.index
        ctx.pending_p_drink_label = target.title or target.action_id or f"p_drink_{target.index + 1}"
        app.device.click_element(target.box)
        # 等待饮料详情模态出现并点击「使う」确认
        if _confirm_drink_usage_modal(app):
            logger.info("{}: 饮料 {!r} 使用确认成功", phase_tag, target.title)
            ctx.record_operation(
                "use_p_drink_in_lesson",
                target=target.title or target.label,
                details={
                    "index": target.index,
                    "label": target.label,
                    "action_id": target.action_id,
                    "db_id": target.db_id,
                    "reason": "empty_hand_fallback",
                },
            )
            drink_idx = getattr(target, "index", None)
            if drink_idx is not None:
                ctx.consume_recognized_drink(drink_idx)
            return HandlerResult.ok(
                f"{phase_tag}: 空手牌 fallback 饮料使用成功 {target.title!r}",
                sleep_after=1.0,
            )
        else:
            # 模态确认失败，尝试关闭残留模态
            logger.warning("{}: 饮料使用确认失败，尝试关闭残留模态", phase_tag)
            _cancel_drink_modal(app)
            return HandlerResult.ok(
                f"{phase_tag}: 空手牌 fallback 饮料使用未确认 {target.title!r}",
                sleep_after=0.8,
            )
    if is_end_turn_action_id(target.action_id) and _click_battle_end_turn(app, fallback_box=target.box):
        logger.info("{}: 空手牌 fallback 选择 SKIP/结束回合", phase_tag)
        ctx.clear_lesson_pending()
        ctx.record_operation(
            "end_turn",
            target=target.title or target.label,
            details={
                "index": target.index,
                "label": target.label,
                "action_id": target.action_id,
                "db_id": target.db_id,
                "reason": "empty_hand_fallback",
            },
        )
        return HandlerResult.ok(f"{phase_tag}: skip (empty_hand_fallback)", sleep_after=1.0)
    return None


def execute_lesson_step(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
    phase: str = "lesson",
) -> LessonStepResult | None:
    """执行一次 lesson 出牌步骤。

    注意：手牌没有真正的“选中态”业务语义，正确交互是双击同一张牌直接出牌。
    这里把 lesson_selected / exam_selected 仅视为“第一击后信息面板仍停留在场上”的恢复态。
    """
    _wait_battle_card_deal_settle(
        app,
        phase=phase,
        position=position,
        pending_index=ctx.pending_lesson_card_index,
    )
    candidates = collect_lesson_card_candidates(app, ctx, phase=phase, position=position)
    empty_hand_observed = _is_battle_empty_hand_observed(app)
    ctx.observability_state = {
        **ctx.observability_state,
        "empty_hand_observed": empty_hand_observed,
    }
    if empty_hand_observed:
        logger.info("{}: 显式识别到无手牌（0枚），改走 empty_hand fallback", phase)
        return None
    # 动画延迟: 没有检测到空手牌提示，但也没检测到卡牌 → 等一下重试
    if not candidates:
        _anim_retries = 2
        for _anim_i in range(_anim_retries):
            time.sleep(0.5)
            # YOLO 引擎后台线程会持续推理，sleep 后 app.latest_results 自动更新
            candidates = collect_lesson_card_candidates(app, ctx, phase=phase, position=position)
            if candidates:
                logger.info("{}: 动画延迟重试 #{} 检测到 {} 张卡", phase, _anim_i + 1, len(candidates))
                break
            # 再次检查是否真的空手牌
            if _is_battle_empty_hand_observed(app):
                logger.info("{}: 动画延迟重试 #{} 确认空手牌", phase, _anim_i + 1)
                return None
        if not candidates:
            logger.debug("{}: 动画延迟重试 {} 次后仍无卡牌", phase, _anim_retries)
            return None

    is_idle = position.endswith("_idle")
    tried_indices: Set[int] = set()
    pending_index = ctx.pending_lesson_card_index
    blocked_indices = _current_blocked_card_indices(ctx, candidates, phase=phase)

    if not is_idle and pending_index is not None and pending_index not in blocked_indices:
        target = _find_lesson_candidate_by_index(candidates, pending_index) or _build_pending_lesson_candidate(ctx)
        logger.debug(
            f"lesson: 补发第二次点击 [{pending_index}] {target.label} {target.title!r}"
        )
        if _confirm_selected_lesson_card(app, ctx, target, phase=phase):
            logger.info(f"lesson: 卡片打出成功 [{pending_index}] {target.title!r}")
            ctx.pending_lesson_card_index = None
            ctx.pending_lesson_card_label = ""
            ctx.handler_state.pop(_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY, None)
            ctx.record_operation(
                "use_lesson_card",
                target=target.title or target.label,
                details={
                    "index": target.index,
                    "label": target.label,
                    "action_id": target.action_id,
                    "db_id": target.db_id,
                },
            )
            return LessonStepResult(status="used", candidate=target)
        logger.warning(
            f"lesson: 卡片 [{pending_index}] {target.title!r} 补发第二击后仍未出牌，"
            "取消残留面板并重新双击下一张"
        )
        if 0 <= pending_index < len(candidates):
            tried_indices.add(pending_index)
        _deselect_card(app)
        time.sleep(0.3)
        candidates = collect_lesson_card_candidates(app, ctx, phase=phase, position="lesson_idle")
        if not candidates:
            logger.warning("lesson: 取消选中后无法检测到手牌")
            return None

    retry_budget = max(len(candidates), 1)
    for attempt in range(retry_budget):
        target_index = decide_lesson_card(
            app,
            ctx,
            candidates,
            phase=phase,
            position="lesson_idle",
            skip_indices=(tried_indices | blocked_indices) if (tried_indices or blocked_indices) else None,
        )
        if target_index is None or int(target_index) < 0:
            logger.info("lesson: 当前无可执行候选，刷新候选后重试")
            _deselect_card(app)
            time.sleep(0.3)
            candidates = collect_lesson_card_candidates(app, ctx, phase=phase, position="lesson_idle")
            if not candidates:
                logger.warning("lesson: 刷新候选后无法检测到手牌")
                return None
            continue
        target = _find_lesson_candidate_by_index(candidates, target_index)
        if target is None:
            fallback_pos = max(0, min(int(target_index), len(candidates) - 1))
            target = candidates[fallback_pos]
        resolved_index = int(target.index)
        action_name = (
            "select_battle_p_drink"
            if is_produce_drink_action_id(target.action_id)
            else "select_lesson_card"
        )
        ctx.record_operation(
            action_name,
            target=target.title or target.label,
            details={
                "index": target.index,
                "label": target.label,
                "action_id": target.action_id,
                "db_id": target.db_id,
            },
        )
        logger.debug(
            f"lesson: 双击出牌 [{resolved_index}] {target.label} {target.title!r}"
            f" (尝试 {attempt + 1}/{retry_budget})"
        )
        if is_end_turn_action_id(target.action_id):
            logger.info("lesson: 改为执行 {} [{}]", target.title or target.label or "SKIP", resolved_index)
            ctx.clear_lesson_pending()
            ctx.handler_state.pop(_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY, None)
            if not is_idle:
                _deselect_card(app)
                time.sleep(0.3)
            if not _click_battle_end_turn(app, fallback_box=target.box):
                logger.warning("lesson: 缺少 SKIP/结束回合按钮，无法执行 end_turn")
                return None
            ctx.record_operation(
                "end_turn",
                target=target.title or target.label,
                details={
                    "index": target.index,
                    "label": target.label,
                    "action_id": target.action_id,
                    "db_id": target.db_id,
                },
            )
            return LessonStepResult(status="end_turn", candidate=target)
        if is_produce_drink_action_id(target.action_id):
            logger.info("lesson: 改为使用底栏饮料 [{}] {!r}", resolved_index, target.title)
            ctx.clear_lesson_pending()
            ctx.pending_p_drink_index = target.index
            ctx.pending_p_drink_label = target.title or target.action_id or f"p_drink_{target.index + 1}"
            app.device.click_element(target.box)
            # 等待饮料详情模态出现并点击「使う」确认
            if _confirm_drink_usage_modal(app):
                logger.info("lesson: 饮料 {!r} 使用确认成功", target.title)
                ctx.record_operation(
                    "use_p_drink_in_lesson",
                    target=target.title or target.label,
                    details={
                        "index": target.index,
                        "label": target.label,
                        "action_id": target.action_id,
                        "db_id": target.db_id,
                    },
                )
                return LessonStepResult(status="used", candidate=target)
            else:
                # 模态确认失败，尝试关闭残留模态
                logger.warning("lesson: 饮料使用确认失败，尝试关闭残留模态")
                _cancel_drink_modal(app)
                return LessonStepResult(status="selected", candidate=target)
        if _try_use_lesson_card_double_tap(app, ctx, target, phase=phase):
            logger.info(f"lesson: 卡片打出成功 [{resolved_index}] {target.title!r}")
            ctx.pending_lesson_card_index = None
            ctx.pending_lesson_card_label = ""
            ctx.handler_state.pop(_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY, None)
            ctx.record_operation(
                "use_lesson_card",
                target=target.title or target.label,
                details={
                    "index": target.index,
                    "label": target.label,
                    "action_id": target.action_id,
                    "db_id": target.db_id,
                },
            )
            return LessonStepResult(status="used", candidate=target)

        logger.warning(
            f"lesson: 卡片 [{resolved_index}] {target.title!r} 双击后仍未出牌，"
            "取消残留面板并尝试下一张"
        )
        tried_indices.add(resolved_index)
        _deselect_card(app)
        time.sleep(0.3)
        candidates = collect_lesson_card_candidates(app, ctx, phase=phase, position="lesson_idle")
        if not candidates:
            logger.warning("lesson: 取消选中后无法检测到手牌")
            return None

    logger.warning("lesson: 所有手牌均无法打出")
    ctx.pending_lesson_card_index = None
    ctx.pending_lesson_card_label = ""
    ctx.handler_state.pop(_BATTLE_LAST_ATTEMPTED_CARD_STATE_KEY, None)
    return LessonStepResult(status="all_unplayable", candidate=candidates[0])


# ────────────────────────────────────────────────────────────
# Handler
# ────────────────────────────────────────────────────────────

class LessonHandler:
    """レッスン出牌的 gameplay handler 包装。

    委托给 execute_lesson_step()，并跟踪已打出的回合数。
    通过鸭子类型作为 GameplayHandler 被 dispatcher 导入。
    """

    phase_tag = "lesson"
    priority = 50

    def can_handle(self, app, ctx, phase, position):
        return phase == "lesson"

    def handle(self, app, ctx, phase, position):
        from src.core.tasks.producer_challenge.gameplay.handler_base import HandlerResult

        if position == GameplayPosition.LESSON_SUMMARY_SHOWCASE:
            # lesson 结束后会先弹一个参数上升说明页，点上方安全区域继续即可；
            # 这里避免误触底部后续可能出现的奖励/按钮区域。
            click_relative_point(
                app,
                x_ratio=0.5,
                y_ratio=0.35,
                label="lesson-summary-showcase",
            )
            ctx.record_operation(
                "advance_lesson_summary_showcase",
                target="lesson_summary",
                position=position,
            )
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "lesson_summary_showcase",
                "retry_limit": int(
                    ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
                ),
                "retry_sleep": float(
                    ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
                ),
            }
            return HandlerResult.ok("lesson: 参数展示页推进", sleep_after=0.8)

        result = execute_lesson_step(app, ctx, position=position, phase=phase)
        if result is None:
            logger.info("lesson: 手牌为空（0枚），改为重新决策饮料 / SKIP")
            empty_hand_result = _try_resolve_empty_hand_action(
                app,
                ctx,
                phase=phase,
                position=position,
            )
            if empty_hand_result is not None:
                return empty_hand_result
            # 无跳过按钮时，点击画面中央推进动画
            click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="lesson-empty-hand-advance")
            return HandlerResult.ok("lesson: 空手牌等待推进", sleep_after=1.0)
        if result.status == "used":
            if is_produce_drink_action_id(result.candidate.action_id):
                # 饮料使用成功，不计入出牌回合
                ctx.handler_state["lesson_idle_streak"] = 0
                drink_idx = getattr(result.candidate, "index", None)
                if drink_idx is not None:
                    ctx.consume_recognized_drink(drink_idx)
                ctx.handler_state["unknown_retry_override"] = {
                    "reason": "lesson_drink_used",
                    "retry_limit": int(
                        ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
                    ),
                    "retry_sleep": float(
                        ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
                    ),
                }
                return HandlerResult.ok(
                    f"lesson: 饮料使用成功 {result.candidate.title!r}",
                    sleep_after=1.0,
                )
            ctx.lesson_turns_played += 1
            ctx.handler_state["lesson_idle_streak"] = 0
            # 打出卡片后可能触发 lesson 结束过渡动画，需要更多重试等待
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "lesson_card_used",
                "retry_limit": int(
                    ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
                ),
                "retry_sleep": float(
                    ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
                ),
            }
            return HandlerResult.ok(
                f"lesson: 打出 {result.candidate.title!r}",
                sleep_after=1.0,
            )
        if result.status == "end_turn":
            ctx.handler_state["lesson_idle_streak"] = 0
            # 结束回合后也可能进入过渡动画
            ctx.handler_state["unknown_retry_override"] = {
                "reason": "lesson_end_turn",
                "retry_limit": int(
                    ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
                ),
                "retry_sleep": float(
                    ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
                ),
            }
            return HandlerResult.ok(
                f"lesson: 结束回合 {result.candidate.title!r}",
                sleep_after=0.8,
            )
        if result.status == "all_unplayable":
            # 所有卡片不可用 → 尝试点击跳过按钮
            ctx.handler_state["lesson_idle_streak"] = 0
            if _click_battle_end_turn(app):
                logger.info("lesson: 所有手牌不可用，点击スキップ跳过回合")
                return HandlerResult.ok("lesson: skip (all_unplayable)", sleep_after=1.0)
            logger.warning("lesson: 所有手牌不可用，无跳过按钮，等待")
            return HandlerResult.ok("lesson: all_unplayable", sleep_after=1.0)
        # status="selected" 且为饮料：模态确认失败，已尝试关闭残留模态
        if is_produce_drink_action_id(result.candidate.action_id):
            ctx.handler_state["lesson_idle_streak"] = 0
            return HandlerResult.ok(
                f"lesson: 饮料使用未确认 {result.candidate.title!r}",
                sleep_after=0.8,
            )

        # status == "selected" — 跟踪连续 idle→selected 未进入 lesson_selected 的次数
        if position.endswith("_idle"):
            streak = ctx.handler_state.get("lesson_idle_streak", 0) + 1
            ctx.handler_state["lesson_idle_streak"] = streak
            if streak >= 4:
                # 连续多次在 idle 状态选择卡片但无法进入 selected → 尝试跳过
                logger.warning(f"lesson: 连续{streak}次无法选中卡片，尝试跳过")
                ctx.handler_state["lesson_idle_streak"] = 0
                if _click_battle_end_turn(app):
                    return HandlerResult.ok("lesson: skip (idle_stuck)", sleep_after=1.0)
        else:
            ctx.handler_state["lesson_idle_streak"] = 0

        return HandlerResult.ok(f"lesson: 选中 {result.candidate.title!r}", sleep_after=0.8)

    def __repr__(self):
        return f"<LessonHandler phase={self.phase_tag!r} priority={self.priority}>"
