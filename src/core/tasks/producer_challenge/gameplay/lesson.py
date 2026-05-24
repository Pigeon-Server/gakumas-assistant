from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, List, Set

import cv2

from src.constants.game.producer_gameplay import GameplayPhase, GameplayPosition
from src.constants.game.text.produce_text import ProduceText
from src.constants.game.text.button_text import ButtonText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.shared.common import click_relative_point
from src.core.tasks.producer_challenge.ui import detect_gameplay_state
from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper
from src.utils.logger import logger
from src.utils.string_tools import fullwidth_to_halfwidth

from src.core.tasks.producer_challenge.shared.common import (
    invoke_decision_strategy,
    normalize_lookup_text,
    normalize_text,
    ocr_text,
    detect_bottom_white_modal_region,
    resolve_candidate_index,
)
from .decision import (
    _apply_resolution,
    _enrich_drink_metadata,
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


_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
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
_BATTLE_PLAY_ANIMATION_OFFSET_RATIO = 0.12
_BATTLE_PLAY_ANIMATION_OFFSET_MIN = 140
_BATTLE_PLAY_ANIMATION_WAIT_MAX_POLLS = 15   # 等待时间延长到 3 秒（15 * 0.2）
_BATTLE_PLAY_ANIMATION_WAIT_SLEEP = 0.2
_BATTLE_PLAY_ANIMATION_ENSURE_SETTLE_POLLS = 2  # 检测到动画结束后额外等待稳定
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
_BATTLE_IMMEDIATE_SCORE_PATTERNS = (
    re.compile(r"(?:固定)?打分\s*[+＋]\s*(\d+)"),
    re.compile(r"スコア\s*[+＋]\s*(\d+)"),
)
_BATTLE_PARAM_TOKEN_MAP = {
    "vocal": ProduceText.VOCAL,
    "dance": ProduceText.DANCE,
    "visual": ProduceText.VISUAL,
}
_BATTLE_HIGH_BONUS_MULTIPLIER = 1.5
_BATTLE_VERY_HIGH_BONUS_MULTIPLIER = 2.0
_BATTLE_CLEAR_PRESSURE_THRESHOLD = 45
_BATTLE_PERFECT_PRESSURE_THRESHOLD = 35
_BATTLE_FINISHING_SCORE_MARGIN = 6
_BATTLE_COLOR_FOCUS_BONUS = 8.0
_BATTLE_WHEEL_FOCUS_BONUS = 12.0
_BATTLE_HIGH_BONUS_FOCUS_BONUS = 10.0
_BATTLE_FINISHING_IMMEDIATE_BONUS = 18.0
_BATTLE_FINISHING_SETUP_PENALTY = 16.0
_BATTLE_PERFECT_PUSH_BONUS = 10.0
_BATTLE_PRESSURE_IMMEDIATE_BONUS = 12.0
_BATTLE_PRESSURE_SETUP_PENALTY = 10.0

# 取消卡片选中时点击空白区域的 Y 轴比例（屏幕高度的 83% 处）
_DESELECT_TAP_Y_RATIO = 0.83

# ── 信息面板探查常量（用于未识别卡片的单击读取） ──
_CARD_INFO_PANEL_TAP_WAIT = 0.5   # 点击卡片后等待信息面板出现的秒数
_CARD_INFO_PANEL_INFER_WAIT = 0.4  # 等待 YOLO 推理完成的秒数
_CARD_INFO_PANEL_DESELECT_WAIT = 0.4  # 取消选中后等待恢复的秒数
# 信息面板状态驱动轮询（与饮料模态一致）
_CARD_INFO_PANEL_POLL_SLEEP = 0.3   # 轮询间歇
_CARD_INFO_PANEL_MAX_POLLS = 20     # 最大轮询次数
_CARD_INFO_PANEL_STABLE_POLLS = 2   # 目标状态连续命中次数
# 信息面板 YOLO 标签（展示卡片详细信息的弹出面板）
_CARD_INFO_PANEL_LABELS = (
    ProducerLabels.SKILL_CARD_INFO,
    ProducerLabels.PC_ACTION_INFO,
)
_CARD_INFO_PANEL_HUD_LABELS = (
    ProducerLabels.PC_PROGRESS,
    ProducerLabels.PC_STAMINA,
    ProducerLabels.PC_P_POINT,
    ProducerLabels.PC_TARGET,
)
_CARD_INFO_PANEL_MIN_WIDTH_RATIO = 0.52
_CARD_INFO_PANEL_MIN_HEIGHT_RATIO = 0.22
_CARD_INFO_PANEL_MIN_TOP_RATIO = 0.20
_CARD_INFO_PANEL_MAX_TOP_RATIO = 0.66
_CARD_INFO_PANEL_MIN_BOTTOM_RATIO = 0.74
_CARD_INFO_PANEL_TITLE_BAND_TOP_RATIO = 0.05
_CARD_INFO_PANEL_TITLE_BAND_BOTTOM_RATIO = 0.28
_CARD_INFO_PANEL_TITLE_BAND_LEFT_RATIO = 0.12
_CARD_INFO_PANEL_TITLE_BAND_RIGHT_RATIO = 0.82
_CARD_INFO_PANEL_NAME_NOISE_RE = re.compile(r'^[\|｜\[\]「」【】\s]+|[\|｜\[\]「」【】\s]+$')
_CARD_INFO_PANEL_EFFECT_PREFIXES = ("↑", "↓", "→", "←", "↗", "↘", "♥", "❤", "♡")
_CARD_INFO_PANEL_EFFECT_LINE_RE = re.compile(
    r"(元気|好印象|やる気|体力|集中|好調|絶好調|スコア|パラメータ).*[+\-−]\d+"
)
_CARD_INFO_PANEL_JP_CHAR_RE = re.compile(r"[ぁ-んァ-ヶー一-龯]")
_CARD_INFO_PANEL_TITLE_NOISE_TOKENS = (
    ProduceText.GUIDE,
    "受け取る",
    "スキルカード",
    "選んで",
    "ください",
    "おすすめ",
    "new",
)

# ── 饮料模态探查常量（用于未识别 P 饮料的单击读取） ──
# 纯状态驱动轮询：每次短暂 sleep 后检查 YOLO 结果，检测到目标立刻退出
_DRINK_MODAL_POLL_SLEEP = 0.3   # 轮询间歇（仅防忙等，不作为计时依据）
_DRINK_MODAL_MAX_POLLS = 20     # 最大轮询次数（足够覆盖慢设备，快设备会提前退出）
_DRINK_MODAL_STABLE_POLLS = 2   # 目标状态连续命中次数（防止旧帧/残帧误触发）
_DRINK_MODAL_NAME_MIN_CONF = 0.35
_DRINK_MODAL_NAME_MIN_SCORE = 1.1
_DRINK_MODAL_DB_MATCH_MIN_CONF = 0.8
_DRINK_MODAL_INSTRUCTION_KEYWORDS = (
    ProduceText.SELECT_PROMPT,
    ProduceText.P_DRINK_SELECT,
)
_DRINK_MODAL_NAME_NOISE_RE = re.compile(r'^[\|｜\[\]「」【】\s]+|[\|｜\[\]「」【】\s]+$')
_DRINK_MODAL_EFFECT_PREFIXES = ("↑", "↓", "→", "←", "↗", "↘", "♥", "❤", "♡")
_DRINK_MODAL_EFFECT_LINE_RE = re.compile(
    rf"({'|'.join(map(re.escape, ProduceText.STATUS_VALUE_TOKENS))}|{re.escape(ProduceText.YARUKI)}|{re.escape(ProduceText.SCORE)}|{re.escape(ProduceText.STAMINA)}).*[+\-−]\d+"
)
_DRINK_MODAL_JP_CHAR_RE = re.compile(r"[ぁ-んァ-ヶー一-龯]")
# 模态头 OCR 排除的文本（标题、按钮等）
_DRINK_MODAL_HEADER_TEXT = ProduceText.P_DRINK_DETAIL
_DRINK_MODAL_HEADER_ALTS = tuple(
    dict.fromkeys(
        normalize_lookup_text(text)
        for text in ProduceText.P_DRINK_DETAIL_ALTS
        if normalize_lookup_text(text)
    )
)
_DRINK_MODAL_EXCLUDE_TEXTS = (
    ProduceText.P_DRINK_DETAIL,
    *ProduceText.P_DRINK_DETAIL_ALTS,
    ProduceText.P_DRINK_DISCARD,
    ButtonText.CANCEL,
    ProduceText.P_DRINK_USE,
)
_DRINK_MODAL_NAME_NOISE_TOKENS = tuple(dict.fromkeys((
    ProduceText.P_DRINK_SELECT,
    ProduceText.SELECT_PROMPT,
    ProduceText.RECEIVE,
    ProduceText.P_DRINK_DETAIL,
    *ProduceText.P_DRINK_DETAIL_ALTS,
    ProduceText.P_DRINK_DISCARD,
)))
_DRINK_UNAVAILABLE_TITLE_TEMPLATE = "未识别P饮料#{slot}"

# ── 饮料模态识别结果缓存 ──
# 避免每次循环重复弹模态识别同一瓶饮料
_DRINK_CACHE_KEY = "_lesson_drink_resolved_cache"
_DRINK_CACHE_SCOPE_KEY = "_lesson_drink_cache_scope"
_DRINK_CACHE_POS_TOLERANCE = 30  # 像素容差
_DRINK_MAX_PROBE = 2  # 同一饮料最大模态探查次数
_DRINK_PROBE_COUNT_KEY = "_lesson_drink_probe_count"


@dataclass
class LessonCardCandidate:
    """定义 LessonCardCandidate 的结构化数据。

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
class LessonStepResult:
    """定义 LessonStepResult 的结构化数据。

    Attributes:
        status: 步骤执行状态（如 selected/confirmed/skipped）。
        candidate: 本步骤最终选中的候选项对象；阶段漂移等中断场景下可为空。
    """
    status: str
    candidate: LessonCardCandidate | None = None


def _serialize_yolo_results(results: Any) -> list[dict[str, Any]]:
    """将当前 YOLO 结果序列化，供 phase drift 现场落盘使用。"""
    if results is None:
        return []
    payload: list[dict[str, Any]] = []
    for item in results:
        payload.append(
            {
                "label": getattr(item, "label", ""),
                "confidence": float(getattr(item, "confidence", 0.0)),
                "box": [
                    int(getattr(item, "x", 0) or 0),
                    int(getattr(item, "y", 0) or 0),
                    int(getattr(item, "w", 0) or 0),
                    int(getattr(item, "h", 0) or 0),
                ],
            }
        )
    return payload


def _draw_yolo_results(frame: Any, results: Any) -> Any:
    """绘制当前 YOLO 检测框，生成 phase drift 调试标注图。"""
    canvas = frame.copy()
    if results is None:
        return canvas
    for item in results:
        x = int(getattr(item, "x", 0) or 0)
        y = int(getattr(item, "y", 0) or 0)
        w = int(getattr(item, "w", 0) or 0)
        h = int(getattr(item, "h", 0) or 0)
        label = str(getattr(item, "label", "") or "?")
        confidence = float(getattr(item, "confidence", 0.0))
        cv2.rectangle(canvas, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{label} {confidence:.2f}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return canvas


def _dump_phase_drift_probe(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    expected_phase: str,
    runtime_phase: str,
    runtime_position: str,
    source: str,
) -> None:
    """在 lesson 阶段漂移时自动落盘当前现场，便于后续按图定位问题。"""
    frame = getattr(app, "latest_frame", None)
    results = getattr(app, "latest_results", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        logger.warning(
            "lesson: phase drift 现场抓图跳过，当前帧为空 source={} runtime={}/{}",
            source,
            runtime_phase,
            runtime_position,
        )
        return
    out_dir = os.path.join(_ROOT_DIR, "out", "debug_captures")
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    stem = (
        f"lesson_phase_drift_{source}_{str(expected_phase).lower()}_to_"
        f"{str(runtime_phase).lower()}_{ts}"
    )
    raw_path = os.path.join(out_dir, f"{stem}.png")
    annotated_path = os.path.join(out_dir, f"{stem}_annotated.jpg")
    meta_path = os.path.join(out_dir, f"{stem}_meta.json")
    cv2.imwrite(raw_path, frame)
    cv2.imwrite(annotated_path, _draw_yolo_results(frame, results))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "captured_at": ts,
                "source": source,
                "expected_phase": str(expected_phase or ""),
                "runtime_phase": str(runtime_phase or ""),
                "runtime_position": str(runtime_position or ""),
                "ctx_phase": str(getattr(ctx, "gameplay_phase", "") or ""),
                "ctx_position": str(getattr(ctx, "gameplay_position", "") or ""),
                "detections": _serialize_yolo_results(results),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(
        "lesson: phase drift 现场已保存 source={} raw={} annotated={} meta={}",
        source,
        raw_path,
        annotated_path,
        meta_path,
    )


def _resolve_box_horizontal_bounds(box: Any) -> tuple[int | None, int | None]:
    """解析检测框的水平边界坐标。

    从 box 对象中读取 x（左边界）和 w（右边界），用于后续计算点击热区偏移。

    Args:
        box: YOLO 检测框对象，需具备 x 和 w 属性。

    Returns:
        (left, right) 元组；无法解析时返回 (None, None)。
    """
    left = getattr(box, "x", None)
    right = getattr(box, "w", None)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and right > left:
        return int(left), int(right)
    return None, None


def _battle_turn_marker(ctx: "ProduceContext", phase: str) -> tuple[str, int, int]:
    """生成当前回合的唯一标记，用于判断 blocked 卡片缓存是否过期。

    当 phase、周数或剩余回合数任一发生变化时，标记即失效，之前记录的
    失败卡片将自动解除封锁。

    Args:
        ctx: 培育上下文对象，提供当前周数和回合参数。
        phase: 当前 gameplay 阶段标识（lesson 或 exam）。

    Returns:
        (phase, current_week, remaining_turns) 三元组。
    """
    return (
        str(phase or ""),
        int(ctx.current_week or 0),
        int(ctx.parameter_state.get("remaining_turns") or -1),
    )


def _candidate_block_keys(candidate: LessonCardCandidate) -> set[str]:
    """提取候选卡片的唯一标识键集合，用于 blocked 卡片匹配。

    从 action_id、db_id、title、label 中过滤出非空值，当前后两帧的
    候选列表发生变化时，仍能通过任意一个标识匹配到同一张卡片。

    Args:
        candidate: 单个候选项对象。

    Returns:
        去重后的非空标识字符串集合。
    """
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
    """获取当前回合中已被封锁的卡片索引集合。

    当回合标记（phase/周数/剩余回合数）发生变化时自动清空封锁记录，
    确保进入新回合后所有卡片重新可用。

    Args:
        ctx: 培育上下文对象，保存 handler_state 中的封锁记录。
        candidates: 当前手牌候选项列表。
        phase: 当前 gameplay 阶段标识。

    Returns:
        被封锁卡片的 index 集合。
    """
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
    """记录上次尝试打出的卡片信息，用于后续 blocked 追踪。

    在双击出牌前调用，若出牌失败则将该卡片加入封锁列表，避免无限重试同一张卡。
    """
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
    """将候选卡片标记为"待确认"目标，记录到 ctx 和 handler_state 中。

    第一次点击卡片后调用，保存索引、名称、action_id、db_id 和点击坐标，
    供第二次点击（双击确认出牌）时恢复使用。
    """
    ctx.pending_lesson_card_index = candidate.index
    ctx.pending_lesson_card_label = candidate.title or candidate.label or candidate.action_id
    ctx.handler_state[_PENDING_LESSON_CARD_ACTION_ID_STATE_KEY] = candidate.action_id
    ctx.handler_state[_PENDING_LESSON_CARD_DB_ID_STATE_KEY] = candidate.db_id
    if candidate.box is not None and hasattr(candidate.box, "get_COL"):
        x, y = candidate.box.get_COL()
        ctx.handler_state[_PENDING_LESSON_CARD_POINT_STATE_KEY] = (int(x), int(y))


def _build_pending_lesson_candidate(ctx: "ProduceContext") -> LessonCardCandidate:
    """从 ctx 的 pending 状态重建 LessonCardCandidate 对象。

    用于第一次点击后画面状态发生变化，原始候选列表已失效时，从
    保存的 pending 字段恢复卡片信息以进行第二次点击确认。

    Args:
        ctx: 培育上下文对象，包含 pending_lesson_card_index/label 等字段。

    Returns:
        重建的 LessonCardCandidate 对象（box 为 None）。
    """
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
    """根据索引在候选列表中查找对应的卡片候选项。

    优先按 candidate.index 字段精确匹配；若匹配失败且索引在有效范围内，
    则退化为按列表位置取元素。

    Args:
        candidates: 候选项列表。
        candidate_index: 目标索引值；为 None 时直接返回 None。

    Returns:
        匹配到的候选项对象，未找到则返回 None。
    """
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
    """点击待确认的卡片，完成双击出牌的第二击。

    优先使用 handler_state 中保存的坐标点；若无则使用 fallback_box
    进行检测框中心点击。

    Args:
        app: 应用处理器实例。
        ctx: 培育上下文对象，包含 pending 卡片坐标。
        fallback_box: 坐标缺失时的备用检测框。
        tap_label: 点击操作的调试标签。

    Returns:
        True 表示点击成功发出，False 表示既无坐标也无 fallback_box。
    """
    point = ctx.handler_state.get(_PENDING_LESSON_CARD_POINT_STATE_KEY)
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        app.device.click(int(point[0]), int(point[1]), tap_label)
        return True
    if fallback_box is not None:
        app.device.click_element(fallback_box)
        return True
    return False


def _normalize_battle_notice_text(text: str | None) -> str:
    """规范化战斗提示文本：全角转半角并去除所有空白字符。

    用于统一 OCR 结果的格式，方便后续关键词匹配。
    """
    return "".join(fullwidth_to_halfwidth(str(text or "")).split())


def _looks_like_empty_hand_notice(text: str | None) -> bool:
    """判断 OCR 文本是否为"手牌为空（0枚）"的提示消息。

    同时检查文本中是否包含"手札"、"スキルカード"和"0"等关键词变体。
    """
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
    """对屏幕中"手牌为空"提示区域进行 OCR 识别。

    根据 blank_slots 的位置动态计算提示文本的裁剪区域（位于空白槽位上方），
    并在 debug_tools 中绘制橙色调试框。

    Args:
        app: 应用处理器实例，提供最新帧和 debug_tools。
        blank_slots: 底部空白槽位检测框列表，用于精确定位提示区域。

    Returns:
        OCR 识别出的文本字符串；无法识别时返回空字符串。
    """
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
    """综合判断当前画面是否为"空手牌"状态。

    检测条件（满足任一即判定为空手牌）：
    1. 屏幕上无任何技能卡检测框，且 OCR 识别到"0枚"提示文本。
    2. 底部空白槽位数量 >= 3（强信号，无需 OCR 确认）。
    """
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
    """统计屏幕上可见的技能卡总数（Active + Mental + Trap）。"""
    if results is None:
        return 0
    return sum(len(results.filter_by_label(label)) for label in _CARD_LABEL_PRIORITY)


def _collect_visible_battle_card_center_ys(results: Any) -> list[int]:
    """收集所有可见技能卡的垂直中心点 Y 坐标，按升序排列。

    用于后续的发牌动画检测：通过中心点基线稳定性判断手牌是否已就位。
    """
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
    """根据屏幕高度计算手牌基线容差值。

    容差用于判断卡牌中心点是否已稳定在基线附近（发牌动画结束标志）。
    取值范围被限制在 [18, 56] 像素之间。

    Returns:
        (tolerance, stable_delta) — 容差值和基线稳定判定阈值。
    """
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
    """判断手牌中心点基线是否已稳定（发牌动画结束判定）。

    以中位数 Y 作为基线，检查所有卡是否都在容差范围内。允许一张卡
    处于"浮空过渡态"（正在落位），只要偏移量不超过 2 倍容差。

    Args:
        center_ys: 升序排列的手牌中心 Y 坐标列表。
        tolerance: 基线容差像素值。

    Returns:
        (是否稳定, 基线 Y 值)。
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


def _is_battle_play_animation_frame(
    app: "AppProcessor",
    center_ys: list[int],
) -> tuple[bool, int, int]:
    """判定当前手牌布局是否处于“出牌动画浮空态”。

    Returns:
        (is_animation_frame, baseline_y, floating_count)
    """
    if len(center_ys) <= 1:
        return False, 0, 0

    baseline = int(center_ys[len(center_ys) // 2])
    frame = getattr(app, "latest_frame", None)
    frame_height = int(frame.shape[0]) if frame is not None and hasattr(frame, "shape") else 0
    tolerance, _ = _resolve_battle_card_baseline_tolerance(app)
    floating_threshold = max(
        _BATTLE_PLAY_ANIMATION_OFFSET_MIN,
        tolerance * 3,
        int(frame_height * _BATTLE_PLAY_ANIMATION_OFFSET_RATIO) if frame_height > 0 else 0,
    )
    floating_count = sum(
        1
        for y in center_ys
        if baseline - int(y) > floating_threshold
    )
    # 至少有一张卡贴近基线，才视作“手牌区 + 浮空动画卡”。
    has_hand_card_near_baseline = any(abs(int(y) - baseline) <= tolerance for y in center_ys)
    return bool(floating_count > 0 and has_hand_card_near_baseline), baseline, floating_count


def _wait_battle_play_animation_end(
    app: "AppProcessor",
    *,
    phase: str,
    position: str,
    pending_index: int | None,
) -> None:
    """在 idle 态等待出牌动画结束，再进入候选收集与决策。

    策略：
      1. 检测到动画帧后继续轮询，直到连续 N 次确认无动画（防残帧误触）
      2. 超时则放弃等待（防止卡死），但不会在动画进行中出牌
    """
    if pending_index is not None or not str(position or "").endswith("_idle"):
        return

    settled_streak = 0
    for poll_idx in range(_BATTLE_PLAY_ANIMATION_WAIT_MAX_POLLS):
        center_ys = _collect_visible_battle_card_center_ys(getattr(app, "latest_results", None))
        is_animation_frame, baseline, floating_count = _is_battle_play_animation_frame(
            app,
            center_ys,
        )
        if not is_animation_frame:
            settled_streak += 1
            if settled_streak >= _BATTLE_PLAY_ANIMATION_ENSURE_SETTLE_POLLS:
                if poll_idx > 0:
                    logger.debug(
                        "{}: 出牌动画已稳定结束，恢复决策 (baseline={}, centers={})",
                        phase,
                        baseline,
                        center_ys,
                    )
                return
        else:
            settled_streak = 0

        debugger = getattr(app, "debug_tools", None)
        results = getattr(app, "latest_results", None)
        if debugger is not None and results is not None:
            for label in _CARD_LABEL_PRIORITY:
                for box in list(results.filter_by_label(label)):
                    cy = getattr(box, "cy", None)
                    if isinstance(cy, (int, float)) and baseline - int(cy) > _BATTLE_PLAY_ANIMATION_OFFSET_MIN:
                        debugger.add_box(
                            int(getattr(box, "x", 0)),
                            int(getattr(box, "y", 0)),
                            int(getattr(box, "w", 0)),
                            int(getattr(box, "h", 0)),
                            label="battle_play_animation_waiting",
                            color=(255, 90, 90),
                            alpha=0.14,
                            duration=2.5,
                            font_size=16,
                        )
        if poll_idx + 1 >= _BATTLE_PLAY_ANIMATION_WAIT_MAX_POLLS:
            logger.debug(
                "{}: 出牌动画等待超时，继续流程 (floating_count={}, baseline={})",
                phase,
                floating_count,
                baseline,
            )
            return
        time.sleep(_BATTLE_PLAY_ANIMATION_WAIT_SLEEP)


def _confirm_selected_lesson_card(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidate: LessonCardCandidate,
    *,
    phase: str,
) -> bool:
    """确认已选中的卡片——执行双击出牌的第二击。

    先记录本次尝试的卡片信息（用于失败时封锁），然后点击卡片坐标，
    最后轮询验证卡片是否成功打出（信息面板消失）。

    Args:
        app: 应用处理器实例。
        ctx: 培育上下文对象。
        candidate: 待确认的卡片候选项。
        phase: 当前 gameplay 阶段标识。

    Returns:
        True 表示出牌验证成功。
    """
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
    """尝试通过双击使用技能卡（第一击 + 间隔 + 第二击确认）。

    第一次点击后设置 pending 目标，等待固定间隔后执行第二击确认。
    若第一击后弹出的是信息面板而非出牌，则进入 pending 恢复流程。

    Args:
        app: 应用处理器实例。
        ctx: 培育上下文对象。
        candidate: 待打出的卡片候选项。
        phase: 当前 gameplay 阶段标识。

    Returns:
        True 表示双击出牌成功。
    """
    app.device.click_element(candidate.box)
    _set_pending_lesson_target(ctx, candidate)
    time.sleep(_CARD_DOUBLE_TAP_INTERVAL)
    return _confirm_selected_lesson_card(app, ctx, candidate, phase=phase)


def _extract_battle_info_panel_name(
    results: Any,
    frame: Any,
    *,
    debugger: Any = None,
) -> str | None:
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

    # 1. 优先使用 YOLO 面板框，缺失时退化到版式检测。
    panel = None
    for label in _CARD_INFO_PANEL_LABELS:
        panels = list(results.filter_by_label(label))
        if panels:
            panel = panels[0]
            break

    if panel is not None:
        px1, py1, px2, py2 = int(panel.x), int(panel.y), int(panel.w), int(panel.h)
    else:
        panel_rect = _detect_battle_info_panel_rect(
            results,
            frame,
            debugger=debugger,
        )
        if panel_rect is None:
            return None
        px1, py1, px2, py2 = panel_rect

    # 2. 裁切面板区域（Yolo_Box: x=x1, y=y1, w=x2, h=y2）
    fh, fw = frame.shape[:2]
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(fw, px2), min(fh, py2)
    if px2 <= px1 + 20 or py2 <= py1 + 20:
        return None
    if debugger is not None:
        debugger.add_box(
            px1,
            py1,
            px2,
            py2,
            label="battle_card_info_panel",
            color=(100, 220, 255),
            alpha=0.1,
            duration=2.5,
            font_size=16,
        )
    panel_crop = frame[py1:py2, px1:px2]

    # 3. OCR 整个面板，获取带位置信息的结果
    ocr_svc = OCRService()
    ocr_results = ocr_svc.ocr(panel_crop)
    if not ocr_results or len(ocr_results) == 0:
        ocr_results = []

    # 4. 按 y 排序，在上半区找“最像卡名”的一行
    sorted_items = sorted(ocr_results, key=lambda r: r.y)
    panel_w = px2 - px1
    panel_h = py2 - py1
    title_bottom = max(1, int(panel_h * 0.36))
    line_threshold = max(sorted_items[0].h * 0.6, 15)
    current_line_y: int | None = None
    current_line: list[Any] = []
    line_groups: list[list[Any]] = []
    for item in sorted_items:
        item_y = int(getattr(item, "y", 0))
        item_h = max(1, int(getattr(item, "h", 0)))
        item_cy = int(getattr(item, "cy", item_y + item_h // 2))
        if item_cy > title_bottom:
            continue
        if current_line_y is None or item_y - current_line_y <= line_threshold:
            current_line.append(item)
            if current_line_y is None:
                current_line_y = item_y
            continue
        if current_line:
            line_groups.append(current_line)
        current_line = [item]
        current_line_y = item_y
    if current_line:
        line_groups.append(current_line)

    best_line: tuple[float, str, tuple[int, int, int, int]] | None = None
    for group in line_groups:
        ordered_group = sorted(group, key=lambda r: int(getattr(r, "x", 0)))
        parts: list[str] = []
        line_x1 = min(int(getattr(item, "x", 0)) for item in ordered_group)
        line_y1 = min(int(getattr(item, "y", 0)) for item in ordered_group)
        line_x2 = max(int(getattr(item, "x", 0)) + max(1, int(getattr(item, "w", 0))) for item in ordered_group)
        line_y2 = max(int(getattr(item, "y", 0)) + max(1, int(getattr(item, "h", 0))) for item in ordered_group)
        for item in ordered_group:
            right_edge = int(getattr(item, "x", 0)) + max(1, int(getattr(item, "w", 0)))
            right_margin = panel_w - right_edge
            if right_margin < panel_w * 0.06 and int(getattr(item, "w", 0)) < panel_w * 0.1:
                logger.debug(
                    "battle: 信息面板排除右侧短字符: text=\"{}\" right_margin={}",
                    item.text, right_margin,
                )
                continue
            parts.append(str(getattr(item, "text", "") or ""))
        if not parts:
            continue
        raw_line = "".join(parts).strip()
        if not _looks_like_plausible_battle_info_panel_name(raw_line):
            continue
        line_w = max(1, line_x2 - line_x1)
        line_cx = (line_x1 + line_x2) / 2.0
        center_bias = 1.0 - min(1.0, abs(line_cx - panel_w / 2.0) / max(panel_w / 2.0, 1.0))
        width_score = min(1.0, line_w / max(panel_w * 0.22, 1.0))
        top_score = 1.0 - min(1.0, line_y1 / max(title_bottom, 1))
        score = center_bias * 2.4 + width_score * 1.8 + top_score
        if best_line is None or score > best_line[0]:
            best_line = (score, raw_line, (line_x1, line_y1, line_x2, line_y2))

    if best_line is None:
        title_y1 = max(0, int(panel_h * _CARD_INFO_PANEL_TITLE_BAND_TOP_RATIO))
        title_y2 = min(panel_h, max(title_y1 + 1, int(panel_h * _CARD_INFO_PANEL_TITLE_BAND_BOTTOM_RATIO)))
        title_x1 = max(0, int(panel_w * _CARD_INFO_PANEL_TITLE_BAND_LEFT_RATIO))
        title_x2 = min(panel_w, max(title_x1 + 1, int(panel_w * _CARD_INFO_PANEL_TITLE_BAND_RIGHT_RATIO)))
        title_crop = panel_crop[title_y1:title_y2, title_x1:title_x2]
        title_ocr_results = ocr_svc.ocr(title_crop)
        title_text = "".join(str(getattr(item, "text", "") or "") for item in title_ocr_results).strip()
        if not _looks_like_plausible_battle_info_panel_name(title_text):
            return None
        best_line = (
            0.0,
            title_text,
            (title_x1, title_y1, title_x2, title_y2),
        )
    raw_name = best_line[1].strip()
    if not raw_name:
        return None

    # 5. 标准化 OCR: 全角→半角 + 日文形近字修正 + 去杂字符
    cleaned = fullwidth_to_halfwidth(raw_name)
    cleaned = normalize_ocr_jp(cleaned)
    cleaned = _CARD_INFO_PANEL_NAME_NOISE_RE.sub("", cleaned).strip()

    if cleaned and cleaned != raw_name:
        logger.debug(
            "battle: 信息面板 OCR 原始=\"{}\" → 标准化=\"{}\"",
            raw_name, cleaned,
        )
    if debugger is not None:
        _, _, (line_x1, line_y1, line_x2, line_y2) = best_line
        debugger.add_box(
            px1 + line_x1,
            py1 + line_y1,
            px1 + line_x2,
            py1 + line_y2,
            label=f"battle_card_name:{cleaned or raw_name}",
            color=(120, 255, 160),
            alpha=0.14,
            duration=2.5,
            font_size=16,
        )
    return cleaned or None


def _looks_like_battle_info_panel_effect_text(text: str) -> bool:
    """判断文本是否像信息面板内的效果说明。"""
    normalized = fullwidth_to_halfwidth(str(text or "")).strip()
    if not normalized:
        return False
    if normalized.startswith(_CARD_INFO_PANEL_EFFECT_PREFIXES):
        return True
    return bool(_CARD_INFO_PANEL_EFFECT_LINE_RE.search(normalized))


def _looks_like_plausible_battle_info_panel_name(text: str) -> bool:
    """判断文本是否像 lesson 信息面板中的卡名。"""
    normalized = normalize_lookup_text(text)
    if len(normalized) < 2:
        return False
    if _looks_like_battle_info_panel_effect_text(text):
        return False
    if any(token in normalized for token in _CARD_INFO_PANEL_TITLE_NOISE_TOKENS):
        return False
    return bool(_CARD_INFO_PANEL_JP_CHAR_RE.search(str(text or "")))


def _collect_battle_info_panel_anchor_boxes(results: Any) -> list[Any]:
    """收集信息面板白框检测所需的手牌锚点框。"""
    if results is None or not hasattr(results, "filter_by_label"):
        return []
    anchors: list[Any] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for label in _CARD_LABEL_PRIORITY:
        for box in list(results.filter_by_label(label)):
            key = (
                int(getattr(box, "x", 0) or 0),
                int(getattr(box, "y", 0) or 0),
                int(getattr(box, "w", 0) or 0),
                int(getattr(box, "h", 0) or 0),
                str(getattr(box, "label", "") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            anchors.append(box)
    return anchors


def _detect_battle_info_panel_rect(
    results: Any,
    frame: Any,
    *,
    debugger: Any = None,
) -> tuple[int, int, int, int] | None:
    """在 YOLO 无面板标签时，基于 HUD 与白色面板版式推断信息面板区域。"""
    if (
        frame is None
        or getattr(frame, "size", 0) <= 0
        or results is None
        or not hasattr(results, "filter_by_label")
    ):
        return None

    hud_hits = sum(1 for label in _CARD_INFO_PANEL_HUD_LABELS if list(results.filter_by_label(label)))
    if hud_hits < 2:
        return None

    anchor_boxes = _collect_battle_info_panel_anchor_boxes(results)
    if not anchor_boxes:
        return None

    panel_rect = detect_bottom_white_modal_region(
        frame,
        row_boxes=anchor_boxes,
        debug_tools=debugger,
        debug_label="battle_card_info_panel_fallback",
    )
    if panel_rect is None:
        return None

    x1, y1, x2, y2 = panel_rect
    frame_h, frame_w = frame.shape[:2]
    panel_w = max(1, x2 - x1)
    panel_h = max(1, y2 - y1)
    if panel_w < int(frame_w * _CARD_INFO_PANEL_MIN_WIDTH_RATIO):
        return None
    if panel_h < int(frame_h * _CARD_INFO_PANEL_MIN_HEIGHT_RATIO):
        return None
    if y1 < int(frame_h * _CARD_INFO_PANEL_MIN_TOP_RATIO):
        return None
    if y1 > int(frame_h * _CARD_INFO_PANEL_MAX_TOP_RATIO):
        return None
    if y2 < int(frame_h * _CARD_INFO_PANEL_MIN_BOTTOM_RATIO):
        return None
    return panel_rect


def _is_info_panel_visible(
    results: Any,
    *,
    frame: Any = None,
    debugger: Any = None,
) -> bool:
    """检查当前画面是否存在卡片信息面板。"""
    if results is None or not hasattr(results, "filter_by_label"):
        return False
    for label in _CARD_INFO_PANEL_LABELS:
        if list(results.filter_by_label(label)):
            return True
    if frame is not None:
        return _detect_battle_info_panel_rect(results, frame, debugger=debugger) is not None
    return False


def _wait_info_panel_visibility(
    app: "AppProcessor",
    *,
    expected_visible: bool,
    reason: str = "",
) -> bool:
    """等待信息面板到达目标可见状态。

    关闭等待仍要求连续稳定若干帧，避免旧帧短闪造成误判；
    打开等待则在首帧明确命中后立即返回，把后续真假校验交给
    面板 OCR 提取逻辑，降低 YOLO 闪断导致的误超时。
    """
    stable_streak = 0
    target_stable_polls = _CARD_INFO_PANEL_STABLE_POLLS if not expected_visible else 1
    for _ in range(_CARD_INFO_PANEL_MAX_POLLS):
        time.sleep(_CARD_INFO_PANEL_POLL_SLEEP)
        results = app.latest_results
        if expected_visible:
            visible = _is_info_panel_visible(
                results,
                frame=getattr(app, "latest_frame", None),
                debugger=getattr(app, "debug_tools", None),
            )
        else:
            visible = _is_info_panel_visible(results)
        if visible == expected_visible:
            stable_streak += 1
            if stable_streak >= target_stable_polls:
                return True
        else:
            stable_streak = 0
    logger.debug(
        "battle: 等待信息面板状态超时 expected_visible={} reason={}",
        expected_visible,
        reason,
    )
    return False


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
      - 状态驱动轮询：等待面板出现，不依赖固定时间
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
            # 正式点击前：等待上一轮残留面板关闭（防止残留信息影响本次读数）
            _wait_info_panel_visibility(
                app,
                expected_visible=False,
                reason=f"card_probe_pre_close#{candidate.index}",
            )
            # 单击卡片 → 弹出信息面板（不会出牌）
            app.device.click_element(candidate.box)

            # 状态驱动：等待信息面板出现，检测到目标后仍要求连续命中以过滤残帧
            if not _wait_info_panel_visibility(
                app,
                expected_visible=True,
                reason=f"card_probe_wait_open#{candidate.index}",
            ):
                logger.debug(
                    "battle: 卡片 #{} 点击后未检测到信息面板（轮询超时）",
                    candidate.index,
                )
                _deselect_card(app)
                time.sleep(_CARD_INFO_PANEL_DESELECT_WAIT)
                continue

            results = app.latest_results
            frame = app.latest_frame
            if results is None or frame is None:
                _deselect_card(app)
                time.sleep(_CARD_INFO_PANEL_DESELECT_WAIT)
                continue

            # 提取信息面板中的卡名
            panel_name = _extract_battle_info_panel_name(
                results,
                frame,
                debugger=getattr(app, "debug_tools", None),
            )
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
    """判断 OCR 文本是否像饮料效果描述行（而非饮料名称）。

    空文本、含数字/符号的文本、包含效果关键词（好感/元气/意欲/体力等）
    的文本均判定为效果描述行，应从名称候选中排除。
    """
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


def _normalize_drink_modal_name_text(text: str) -> str:
    """标准化饮料模态中的名称文本。"""
    from src.utils.string_tools import normalize_ocr_jp

    cleaned = normalize_ocr_jp(fullwidth_to_halfwidth(str(text or "")))
    cleaned = _DRINK_MODAL_NAME_NOISE_RE.sub("", cleaned).strip()
    return cleaned


def _looks_like_drink_modal_effect_line(text: str) -> bool:
    """判断文本是否像饮料模态中的效果描述。"""
    normalized = fullwidth_to_halfwidth(str(text or "")).strip()
    if not normalized:
        return False
    if normalized.startswith(_DRINK_MODAL_EFFECT_PREFIXES):
        return True
    return bool(_DRINK_MODAL_EFFECT_LINE_RE.search(normalized))


def _is_plausible_drink_modal_name(text: str) -> bool:
    """判断文本是否可作为饮料模态中的名称标题。"""
    normalized = _normalize_drink_modal_name_text(text)
    if len(normalized) < 2:
        return False
    if _looks_like_drink_modal_effect_line(normalized):
        return False
    if _looks_like_drink_effect_line(normalized):
        return False
    if any(token in normalized for token in _DRINK_MODAL_NAME_NOISE_TOKENS):
        return False
    return bool(_DRINK_MODAL_JP_CHAR_RE.search(normalized))


def _score_drink_modal_name_candidate(
    *,
    text: str,
    item: Any,
    crop_w: int,
    crop_h: int,
) -> float:
    """对饮料模态中的 OCR 文本进行评分，判断其是否为饮料名称。

    评分依据：位置（上半部分加分、横向居中加分）、字符类型（日文加分）、
    排除特征（含数字/符号减分、效果描述行减分、过长文本减分）。
    """
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
    from src.core.inference.ocr_engine import OCRService

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
    ocr_svc = OCRService()
    panel_rect = detect_bottom_white_modal_region(
        frame,
        row_boxes=cancel_boxes or [header],
        debug_tools=debugger,
        debug_label="battle_drink_white_panel",
    )

    ranked: list[tuple[float, str, Any]] = []
    if panel_rect is not None:
        px1, py1, px2, py2 = panel_rect
        if px2 > px1 + 20 and py2 > py1 + 20:
            panel_crop = frame[py1:py2, px1:px2]
            ocr_result_list = ocr_svc.ocr(panel_crop)
            merged_lines = (
                list(
                    ocr_result_list.auto_merge_lines(
                        cy_range=max(8, int(panel_crop.shape[0] * 0.02)),
                        width_gap=max(12, int(panel_crop.shape[1] * 0.04)),
                    )
                )
                if hasattr(ocr_result_list, "auto_merge_lines")
                else list(ocr_result_list or [])
            )
            if merged_lines:
                panel_h = max(1, py2 - py1)
                panel_w = max(1, px2 - px1)
                title_bottom = int(panel_h * 0.42)
                boundary = int(region_y2 - py1 - panel_h * 0.05)
                if boundary > int(panel_h * 0.15):
                    title_bottom = min(title_bottom, boundary)

                normalized_lines = [
                    _normalize_drink_modal_name_text(str(getattr(line, "text", "") or ""))
                    for line in merged_lines
                ]
                full_text = " ".join(normalized_lines)
                if any(kw in full_text for kw in _DRINK_MODAL_INSTRUCTION_KEYWORDS):
                    return []

                for line in merged_lines:
                    raw_text = str(getattr(line, "text", "") or "").strip()
                    line_text = _normalize_drink_modal_name_text(raw_text)
                    if not line_text:
                        continue
                    line_y = int(getattr(line, "y", 0))
                    line_h = max(1, int(getattr(line, "h", 0)))
                    line_cy = int(getattr(line, "cy", line_y + line_h // 2))
                    if line_cy < 0 or line_cy > title_bottom:
                        continue
                    if not _is_plausible_drink_modal_name(line_text):
                        continue
                    candidate_score = _score_drink_modal_name_candidate(
                        text=line_text,
                        item=line,
                        crop_w=panel_w,
                        crop_h=panel_h,
                    )
                    if candidate_score < _DRINK_MODAL_NAME_MIN_SCORE:
                        continue
                    ranked.append((candidate_score, line_text, SimpleNamespace(
                        x=px1 + int(getattr(line, "x", 0)),
                        y=py1 + int(getattr(line, "y", 0)),
                        w=max(1, int(getattr(line, "w", 0))),
                        h=max(1, int(getattr(line, "h", 0))),
                    )))
                    if line_text != raw_text:
                        logger.debug(
                            "battle: 饮料模态 OCR 原始=\"{}\" → 标准化=\"{}\"",
                            raw_text, line_text,
                        )

    if not ranked:
        ocr_results = ocr_svc.ocr(modal_crop)
        if not ocr_results or len(ocr_results) == 0:
            return []

        crop_w = region_x2 - region_x1
        crop_h = region_y2 - region_y1
        sorted_items = sorted(ocr_results, key=lambda r: r.y)

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

            cleaned = _normalize_drink_modal_name_text(text)
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
            ranked.append((candidate_score, cleaned, SimpleNamespace(
                x=region_x1 + int(getattr(item, "x", 0)),
                y=region_y1 + int(getattr(item, "y", 0)),
                w=max(1, int(getattr(item, "w", 0))),
                h=max(1, int(getattr(item, "h", 0))),
            )))
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
                int(item.x),
                int(item.y),
                int(item.x + item.w),
                int(item.y + item.h),
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


def _extract_drink_modal_header_text(results: Any) -> str:
    """读取饮料模态标题文本并返回归一化前结果。"""
    if results is None:
        return ""
    headers = list(results.filter_by_label(BaseUILabels.MODAL_HEADER))
    if not headers:
        return ""
    header = headers[0]
    return ocr_text(getattr(header, "frame", None))


def _has_drink_modal_header(results: Any) -> bool:
    """判断当前结果中是否存在任意模态标题框。"""
    if results is None:
        return False
    return bool(list(results.filter_by_label(BaseUILabels.MODAL_HEADER)))


def _has_drink_modal_action_button(results: Any) -> bool:
    """判断当前模态中是否存在取消或确认按钮。"""
    if results is None:
        return False
    cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
    confirm_boxes = list(results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
    return bool(cancel_boxes or confirm_boxes)


def _is_drink_modal_title(text: str) -> bool:
    """判断 OCR 文本是否可视为 P 饮料详情模态标题。"""
    normalized = normalize_lookup_text(text)
    if not normalized:
        return False
    return any(token and token in normalized for token in _DRINK_MODAL_HEADER_ALTS)


def _is_drink_modal_visible(
    results: Any,
    *,
    require_action_button: bool = False,
    allow_header_fallback: bool = False,
) -> bool:
    """判断饮料详情模态（Pドリンク詳細）是否可见。

    Args:
        results: YOLO 检测结果对象。
        require_action_button: 是否要求同时检测到 キャンセル 或 确认按钮。
        allow_header_fallback: 当标题 OCR 脏帧时，是否允许用“标题框 + 操作按钮”
            的弱信号兜底判定模态已打开。

    Returns:
        True 表示模态可见。
    """
    if not _has_drink_modal_header(results):
        return False
    has_action_button = _has_drink_modal_action_button(results)
    if require_action_button and not has_action_button:
        return False
    header_text = _extract_drink_modal_header_text(results)
    if _is_drink_modal_title(header_text):
        return True
    if allow_header_fallback:
        return True
    return False


def _mark_drink_candidate_unavailable(
    candidate: LessonCardCandidate,
    *,
    slot_index: int,
    reason: str,
    failure_flag: str,
    raw_title: str = "",
) -> None:
    """将未识别饮料降级为不可用候选，避免进入决策。"""
    candidate.db_id = ""
    candidate.action_id = candidate.action_id or f"produce_drink_unknown:{slot_index}"
    candidate.title = _DRINK_UNAVAILABLE_TITLE_TEMPLATE.format(slot=slot_index)
    candidate.source = "unresolved"
    candidate.confidence = 0.0
    candidate.metadata["available"] = False
    candidate.metadata["unavailable_reason"] = reason
    candidate.metadata[failure_flag] = True
    candidate.metadata["identity_unresolved"] = True
    candidate.metadata["raw_ocr_title"] = str(raw_title or candidate.metadata.get("raw_ocr_title") or "")
    candidate.metadata["canonical_name"] = candidate.title
    candidate.metadata["db_id"] = ""
    candidate.metadata.pop("drink_score", None)
    candidate.metadata.pop("matched_text", None)
    candidate.metadata.pop("name", None)
    logger.debug(
        "battle: 饮料候选 #{} 标记为不可用 reason={} flag={} raw_title={!r}",
        candidate.index,
        reason,
        failure_flag,
        candidate.metadata.get("raw_ocr_title", ""),
    )


def _finalize_resolved_drink_candidate(candidate: LessonCardCandidate) -> None:
    """将已识别饮料候选统一归一到 db_id 与结构化 metadata。"""
    if not candidate.db_id:
        return
    metadata = dict(candidate.metadata or {})
    enriched = _enrich_drink_metadata(candidate.db_id)
    metadata.update(enriched)
    metadata["available"] = True
    metadata.pop("unavailable_reason", None)
    metadata.pop("probe_failed", None)
    metadata.pop("modal_open_timeout", None)
    metadata.pop("modal_title_mismatch", None)
    metadata.pop("identity_unresolved", None)
    metadata["db_id"] = candidate.db_id
    candidate.metadata = metadata
    candidate.title = str(metadata.get("raw_name") or candidate.title or "")
    candidate.action_id = candidate.action_id or f"produce_drink:{candidate.db_id}"
    candidate.source = candidate.source or "ocr"


def _resolve_drink_modal_failure_reason(results: Any) -> tuple[str, str]:
    """根据当前帧状态归纳饮料模态等待失败原因。"""
    if results is None:
        return "等待饮料详情模态时未拿到检测结果", "probe_failed"
    if not _has_drink_modal_header(results):
        return "点击饮料后未检测到饮料详情模态标题", "modal_open_timeout"
    header_text = _extract_drink_modal_header_text(results)
    if not _is_drink_modal_title(header_text):
        if _has_drink_modal_action_button(results):
            return (
                f"检测到模态框与操作按钮，但标题OCR未稳定命中：{header_text or '空标题'}",
                "modal_open_timeout",
            )
        return f"检测到模态标题但不是P饮料详情：{header_text or '空标题'}", "modal_title_mismatch"
    if not _has_drink_modal_action_button(results):
        return "检测到饮料详情模态标题，但未检测到可操作按钮", "modal_open_timeout"
    return "饮料详情模态状态不稳定，未通过连续帧确认", "modal_open_timeout"


def _is_drink_candidate_resolved(candidate: LessonCardCandidate) -> bool:
    """判断饮料候选是否已经得到可信 db_id。"""
    return bool(str(candidate.db_id or "").strip()) and not bool(candidate.metadata.get("identity_unresolved", False))


def _refresh_battle_drink_candidate_score(
    candidate: LessonCardCandidate,
    *,
    phase: str,
    stamina: int,
    max_stamina: int,
    remaining_turns: int,
) -> None:
    """基于最新 metadata 刷新饮料评分。"""
    metadata = dict(candidate.metadata or {})
    if not _is_drink_candidate_resolved(candidate):
        metadata.pop("drink_score", None)
        candidate.metadata = metadata
        return
    metadata["drink_score"] = score_produce_drink_metadata(
        metadata,
        phase=phase,
        stamina=stamina,
        max_stamina=max_stamina,
        remaining_turns=remaining_turns,
    )
    candidate.metadata = metadata


def _build_drink_candidate_title(resolution_title: str, raw_title: str, slot_index: int) -> str:
    """生成饮料候选显示标题。"""
    normalized_raw_title = normalize_text(raw_title)
    if normalized_raw_title and len(normalized_raw_title) >= 2 and normalized_raw_title != normalize_text(ProduceText.P_DRINK):
        return resolution_title or raw_title
    return resolution_title or _DRINK_UNAVAILABLE_TITLE_TEMPLATE.format(slot=slot_index)


def _ocr_battle_drink_modal_name(app: "AppProcessor") -> str:
    """复用 lesson 详情模态名称候选，提取当前最佳饮料名。"""
    results = getattr(app, "latest_results", None)
    frame = getattr(app, "latest_frame", None)
    if results is None or frame is None:
        return ""
    names = _extract_drink_modal_name_candidates(
        results,
        frame,
        debugger=getattr(app, "debug_tools", None),
    )
    return names[0] if names else ""


def _sync_drink_cache_entry_from_candidate(cache_entry: dict[str, Any], candidate: LessonCardCandidate) -> None:
    """缓存载荷统一同步为以 db_id 为核心的结构。"""
    cache_entry["db_id"] = candidate.db_id
    cache_entry["action_id"] = candidate.action_id
    cache_entry["title"] = candidate.title
    cache_entry["source"] = candidate.source
    cache_entry["confidence"] = candidate.confidence
    cache_entry["metadata"] = dict(candidate.metadata or {})
    cache_entry["metadata"]["db_id"] = candidate.db_id
    cache_entry["metadata"].pop("raw_ocr_title", None)
    cache_entry["metadata"].pop("identity_unresolved", None)
    cache_entry["metadata"]["available"] = True
    cache_entry["metadata"].pop("unavailable_reason", None)
    cache_entry["metadata"].pop("probe_failed", None)
    cache_entry["metadata"].pop("modal_open_timeout", None)
    cache_entry["metadata"].pop("modal_title_mismatch", None)
    cache_entry["metadata"]["canonical_name"] = candidate.title
    cache_entry["matched_db_id"] = candidate.db_id
    cache_entry["matched_name"] = candidate.title
    cache_entry["matched_source"] = candidate.source


def _clear_lesson_drink_cache(ctx: "ProduceContext") -> None:
    """清空 lesson 饮料识别缓存与探查计数。"""
    ctx.handler_state[_DRINK_CACHE_KEY] = {}
    ctx.handler_state[_DRINK_PROBE_COUNT_KEY] = {}
    ctx.handler_state.pop(_DRINK_CACHE_SCOPE_KEY, None)


def _clear_lesson_drink_cache_after_use(ctx: "ProduceContext") -> None:
    """饮料实际使用后清空缓存，避免槽位复用旧 db_id。"""
    _clear_lesson_drink_cache(ctx)
    logger.debug("lesson: 饮料使用后已清空底栏饮料缓存")


def _extract_drink_slot_index(candidate: LessonCardCandidate) -> int:
    """提取饮料槽位序号。"""
    return int(candidate.metadata.get("battle_drink_slot") or max(candidate.index + 1, 1))


def _resolve_drink_candidate_display_name(candidate: LessonCardCandidate) -> str:
    """返回以 db_id 结果为准的显示名。"""
    if candidate.db_id:
        return str(candidate.metadata.get("raw_name") or candidate.title or "")
    return _DRINK_UNAVAILABLE_TITLE_TEMPLATE.format(slot=_extract_drink_slot_index(candidate))


def _should_disable_unresolved_drink(candidate: LessonCardCandidate) -> bool:
    """未识别饮料是否必须禁入决策。"""
    return not _is_drink_candidate_resolved(candidate)


def _annotate_drink_candidate_post_resolution(candidate: LessonCardCandidate) -> None:
    """统一饮料候选解析后的元数据。"""
    if _should_disable_unresolved_drink(candidate):
        candidate.metadata["available"] = False
        candidate.metadata.setdefault("identity_unresolved", True)
        candidate.metadata.setdefault("unavailable_reason", "未识别出饮料 db_id，不能参与决策")
        candidate.metadata["db_id"] = ""
        candidate.title = _resolve_drink_candidate_display_name(candidate)
        return
    _finalize_resolved_drink_candidate(candidate)
    candidate.title = _resolve_drink_candidate_display_name(candidate)
    candidate.metadata["db_id"] = candidate.db_id
    candidate.metadata["available"] = True
    candidate.metadata.pop("raw_ocr_title", None)
    candidate.metadata.pop("identity_unresolved", None)
    candidate.metadata.pop("unavailable_reason", None)
    candidate.metadata.pop("probe_failed", None)
    candidate.metadata.pop("modal_open_timeout", None)
    candidate.metadata.pop("modal_title_mismatch", None)


def _is_available_battle_drink_candidate(candidate: LessonCardCandidate) -> bool:
    """当前饮料候选是否可参与 battle 决策。"""
    return bool(candidate.metadata.get("available", True)) and _is_drink_candidate_resolved(candidate)


def _normalize_drink_candidate_resolution(candidate: LessonCardCandidate) -> None:
    """统一饮料候选识别结果，保证内部状态以 db_id 为主。"""
    _annotate_drink_candidate_post_resolution(candidate)
    candidate.title = _resolve_drink_candidate_display_name(candidate)
    if not candidate.db_id:
        candidate.action_id = candidate.action_id or f"produce_drink_unknown:{_extract_drink_slot_index(candidate)}"
        candidate.source = candidate.source or "unresolved"
        candidate.confidence = 0.0
        return
    candidate.action_id = f"produce_drink:{candidate.db_id}"
    candidate.metadata["candidate_type"] = "battle_p_drink"
    candidate.metadata["battle_drink_slot"] = _extract_drink_slot_index(candidate)
    candidate.metadata["db_id"] = candidate.db_id
    candidate.source = candidate.source or "ocr"
    candidate.metadata["canonical_name"] = candidate.title


def _is_drink_payload_available(payload: dict[str, Any]) -> bool:
    """判断 battle 饮料 payload 是否可执行。"""
    if not is_produce_drink_action_id(payload.get("id")):
        return True
    if not str(payload.get("db_id") or "").strip():
        return False
    metadata = dict(payload.get("metadata", {}) or {})
    if not bool(metadata.get("available", True)):
        return False
    if bool(metadata.get("identity_unresolved", False)):
        return False
    if bool(metadata.get("probe_failed", False)):
        return False
    if bool(metadata.get("modal_open_timeout", False)):
        return False
    if bool(metadata.get("modal_title_mismatch", False)):
        return False
    return True




def _wait_drink_modal_visibility(
    app: "AppProcessor",
    *,
    expected_visible: bool,
    require_action_button: bool = False,
    reason: str = "",
) -> bool:
    """等待饮料模态到达目标可见状态。

    关闭等待要求连续稳定隐藏；打开等待则允许“标题框 + 操作按钮”的弱信号
    先通过，后续再用标题 OCR 与正文名称做二次确认，降低标题脏帧造成的误伤。
    """
    stable_streak = 0
    target_stable_polls = _DRINK_MODAL_STABLE_POLLS if not expected_visible else 1
    for _ in range(_DRINK_MODAL_MAX_POLLS):
        time.sleep(_DRINK_MODAL_POLL_SLEEP)
        results = app.latest_results
        visible = _is_drink_modal_visible(
            results,
            require_action_button=require_action_button if expected_visible else False,
            allow_header_fallback=expected_visible,
        )
        if visible == expected_visible:
            stable_streak += 1
            if stable_streak >= target_stable_polls:
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
    """生成饮料识别缓存的作用域标记。

    缓存按 (phase, current_week) 分组，切换阶段或周数时自动失效。
    """
    return (str(phase or ""), int(ctx.current_week or -1))


def _ensure_drink_cache_scope(ctx: "ProduceContext", *, phase: str) -> None:
    """确保饮料缓存作用域与当前 (phase, week) 一致。

    当作用域发生变化时清空旧缓存和探查计数器，避免跨周/跨阶段误用。
    """
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
        if cand.label != ProducerLabels.P_DRINK:
            continue
        key = _drink_pos_key(cand.box)
        cached = cache.get(key)
        if cached is None:
            continue
        cached_db_id = str(cached.get("db_id") or "").strip()
        if not cached_db_id:
            continue
        cand.db_id = cached_db_id
        cand.action_id = str(cached.get("action_id") or f"produce_drink:{cached_db_id}")
        cand.title = str(cached.get("title") or cand.title or cached_db_id)
        cand.source = str(cached.get("source") or "cache")
        cand.confidence = float(cached.get("confidence") or 0.0)
        if cached.get("metadata"):
            cand.metadata.update(dict(cached["metadata"] or {}))
        _normalize_drink_candidate_resolution(cand)
        logger.debug("lesson: 从缓存恢复饮料 #{} → db_id={}", cand.index, cand.db_id)


def _save_drink_cache(
    ctx: "ProduceContext",
    candidates: list[LessonCardCandidate],
    *,
    phase: str,
) -> None:
    """将已识别的饮料结果写入 handler_state 缓存。

    以量化后的中心坐标 (cx, cy) 为 key，保存 db_id、action_id、title 等信息，
    下次同一位置的饮料可直接从缓存恢复，无需重复弹模态。
    """
    _ensure_drink_cache_scope(ctx, phase=phase)
    cache: dict = ctx.handler_state.setdefault(_DRINK_CACHE_KEY, {})
    for cand in candidates:
        if cand.label != ProducerLabels.P_DRINK or not _is_drink_candidate_resolved(cand):
            continue
        key = _drink_pos_key(cand.box)
        cache_entry = cache.setdefault(key, {})
        _sync_drink_cache_entry_from_candidate(cache_entry, cand)
        logger.debug("lesson: 缓存饮料 pos={} → db_id={}, title={}", key, cand.db_id, cand.title)


def _should_skip_drink_probe(ctx: "ProduceContext", box: Any) -> bool:
    """判断当前帧是否应跳过饮料探测流程。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        box: 单个检测框对象。

    Returns:
        bool: True 表示该位置已探测次数达到上限，应跳过。
    """
    counts: dict = ctx.handler_state.get(_DRINK_PROBE_COUNT_KEY, {})
    return counts.get(_drink_pos_key(box), 0) >= _DRINK_MAX_PROBE


def _increment_drink_probe(ctx: "ProduceContext", box: Any) -> int:
    """增加饮料模态探查计数器并返回新值。

    同一饮料的探查次数达到上限后将被跳过，避免无限循环弹模态。
    """
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
    """对未识别的 P 饮料逐个单击，打开详情模态提取饮料名并匹配数据库。"""
    unresolved = [
        c for c in candidates
        if c.label == ProducerLabels.P_DRINK
        and c.box is not None
        and not _is_drink_candidate_resolved(c)
        and not _should_skip_drink_probe(ctx, c.box)
    ]
    if not unresolved:
        return

    logger.info(
        "battle: {} 个 P 饮料未识别，开始逐个点击读取模态",
        len(unresolved),
    )

    for candidate in unresolved:
        slot_index = _extract_drink_slot_index(candidate)
        probe_count = _increment_drink_probe(ctx, candidate.box)
        try:
            _wait_drink_modal_visibility(
                app,
                expected_visible=False,
                reason=f"drink_probe_pre_close#{candidate.index}",
            )
            if getattr(app, "debug_tools", None) is not None and candidate.box is not None:
                app.debug_tools.add_box(
                    int(candidate.box.x),
                    int(candidate.box.y),
                    int(candidate.box.w),
                    int(candidate.box.h),
                    label=f"lesson_drink_probe_target:{slot_index}",
                    color=(255, 180, 60),
                    alpha=0.14,
                    duration=2.5,
                    font_size=14,
                )
            app.device.click_element(candidate.box)

            if not _wait_drink_modal_visibility(
                app,
                expected_visible=True,
                require_action_button=True,
                reason=f"drink_probe_wait_open#{candidate.index}",
            ):
                failure_reason, failure_flag = _resolve_drink_modal_failure_reason(app.latest_results)
                logger.debug(
                    "battle: P 饮料 #{} 点击后未稳定进入模态 reason={}",
                    candidate.index,
                    failure_reason,
                )
                _mark_drink_candidate_unavailable(
                    candidate,
                    slot_index=slot_index,
                    reason=failure_reason,
                    failure_flag=failure_flag,
                    raw_title=str(candidate.metadata.get("raw_ocr_title") or candidate.title or ""),
                )
                _cancel_drink_modal(app)
                continue

            results = app.latest_results
            frame = app.latest_frame
            if results is None or frame is None:
                _mark_drink_candidate_unavailable(
                    candidate,
                    slot_index=slot_index,
                    reason="饮料详情模态打开后未获取到最新画面",
                    failure_flag="probe_failed",
                    raw_title=str(candidate.metadata.get("raw_ocr_title") or candidate.title or ""),
                )
                _cancel_drink_modal(app)
                continue

            header_text = _extract_drink_modal_header_text(results)
            if not _is_drink_modal_title(header_text):
                logger.debug(
                    "battle: P 饮料 #{} 模态标题 OCR 未稳定命中，改用正文名称继续识别 header={!r}",
                    candidate.index,
                    header_text,
                )
            modal_names = _extract_drink_modal_name_candidates(
                results,
                frame,
                debugger=getattr(app, "debug_tools", None),
            )
            modal_name = modal_names[0] if modal_names else _ocr_battle_drink_modal_name(app)
            if not modal_name:
                failure_flag = "identity_unresolved"
                failure_reason = "饮料详情模态 OCR 未提取到饮料名"
                if not _is_drink_modal_title(header_text):
                    failure_flag = "modal_title_mismatch"
                    failure_reason = f"模态标题OCR异常且正文未提取到饮料名：{header_text or '空标题'}"
                _mark_drink_candidate_unavailable(
                    candidate,
                    slot_index=slot_index,
                    reason=failure_reason,
                    failure_flag=failure_flag,
                    raw_title=str(candidate.metadata.get("raw_ocr_title") or candidate.title or ""),
                )
                _cancel_drink_modal(app)
                continue

            resolution = resolve_produce_drink_identity(
                modal_name,
                app=app,
                box=candidate.box,
                index=candidate.index,
                min_ocr_confidence=_DRINK_MODAL_DB_MATCH_MIN_CONF,
            )
            if resolution.db_id:
                _apply_resolution(candidate, resolution)
                _normalize_drink_candidate_resolution(candidate)
                drink_image = getattr(candidate.box, "frame", None)
                if drink_image is not None:
                    _learn_drink_clip_from_db_id(app, drink_image, candidate.db_id)
                logger.info(
                    "battle: P 饮料 #{} 通过模态识别成功: db_id={} name={!r}",
                    candidate.index,
                    candidate.db_id,
                    modal_name,
                )
            else:
                reason = (
                    f"饮料详情模态 OCR 已读到名称但未匹配到 db_id：{modal_name}"
                    if probe_count < _DRINK_MAX_PROBE
                    else f"饮料详情模态 OCR 多次未匹配到 db_id：{modal_name}"
                )
                _mark_drink_candidate_unavailable(
                    candidate,
                    slot_index=slot_index,
                    reason=reason,
                    failure_flag="identity_unresolved",
                    raw_title=modal_name,
                )
                logger.warning(
                    "battle: P 饮料 #{} 模态 OCR={!r} 未匹配 db_id (探查 {}/{})",
                    candidate.index,
                    modal_name,
                    probe_count,
                    _DRINK_MAX_PROBE,
                )

        except Exception as exc:
            _mark_drink_candidate_unavailable(
                candidate,
                slot_index=slot_index,
                reason=f"P饮料模态识别异常：{exc}",
                failure_flag="probe_failed",
                raw_title=str(candidate.metadata.get("raw_ocr_title") or candidate.title or ""),
            )
            logger.warning(
                "battle: P 饮料 #{} 模态识别异常: {}",
                candidate.index, exc,
            )
        finally:
            _cancel_drink_modal(app)
            _normalize_drink_candidate_resolution(candidate)


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
        drink_candidates = _filter_unavailable_battle_drinks(drink_candidates)
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
    """收集底部栏位中的 P 饮料候选项。

    仅取屏幕底部 88% 以下的 P_DRINK 检测框，按 x 坐标左→右排序。
    对每个饮料执行：OCR 识别 → 数据库匹配 → 评分 → 缓存恢复 → 模态探查。

    Args:
        app: 应用处理器实例。
        ctx: 培育上下文对象。
        phase: 当前 gameplay 阶段标识。
        start_index: 候选项起始序号（接在手牌之后）。

    Returns:
        P 饮料候选项列表。
    """
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
        slot_index = offset + 1
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
        metadata["battle_drink_slot"] = slot_index
        metadata["raw_ocr_title"] = raw_title
        candidate = LessonCardCandidate(
            index=index,
            label=ProducerLabels.P_DRINK,
            title=_build_drink_candidate_title(resolution.display_name or "", raw_title, slot_index),
            selected=False,
            box=box,
            action_id=resolution.action_id or f"produce_drink_unknown:{slot_index}",
            db_id=resolution.db_id,
            source=resolution.source,
            confidence=resolution.confidence,
            metadata=metadata,
        )
        _normalize_drink_candidate_resolution(candidate)
        _refresh_battle_drink_candidate_score(
            candidate,
            phase=phase,
            stamina=stamina,
            max_stamina=max_stamina,
            remaining_turns=remaining_turns,
        )
        candidates.append(candidate)
    _apply_drink_cache(ctx, candidates, phase=phase)
    _resolve_unidentified_drinks_via_modal(app, ctx, candidates)
    for candidate in candidates:
        _normalize_drink_candidate_resolution(candidate)
        _refresh_battle_drink_candidate_score(
            candidate,
            phase=phase,
            stamina=stamina,
            max_stamina=max_stamina,
            remaining_turns=remaining_turns,
        )
    _save_drink_cache(ctx, candidates, phase=phase)
    return candidates


def _filter_unavailable_battle_drinks(candidates: list[LessonCardCandidate]) -> list[LessonCardCandidate]:
    """过滤掉不应参与决策的底栏饮料候选。"""
    filtered: list[LessonCardCandidate] = []
    for candidate in candidates:
        if candidate.label == ProducerLabels.P_DRINK and not _is_available_battle_drink_candidate(candidate):
            continue
        filtered.append(candidate)
    return filtered


def _collect_battle_end_turn_candidates(
    app: "AppProcessor",
    *,
    phase: str,
    start_index: int,
) -> List[LessonCardCandidate]:
    """收集"结束回合/SKIP"按钮候选项。

    通过 YOLO 检测 PC_SKIP 标签，若检测到则构建为候选项，供决策链
    在手牌用尽或体力不足时选择 SKIP 进入下一回合。

    Args:
        app: 应用处理器实例。
        phase: 当前 gameplay 阶段标识（决定 label 显示为 "SKIP" 还是 "结束回合"）。
        start_index: 候选项起始序号。

    Returns:
        包含 SKIP 候选项的列表（0 或 1 个元素）。
    """
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
    """在特定条件下强制选择底栏 P 饮料，优先于 LLM 决策。

    强制使用条件：
    - play_limit <= 0（无法出牌）且有可用饮料
    - 手牌已全部用尽，仅剩饮料可用
    - 体力极低（<=3 或 <=18%）且有高评分恢复饮料
    - 体力偏低（<=5 或 <=32%）且有足够评分的恢复饮料

    Args:
        decision_state: 决策快照，包含候选项、legal_actions 和 llm_snapshot。
        skip_indices: 需要跳过的索引集合。

    Returns:
        强制选择的饮料索引，不满足条件时返回 None。
    """
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
        """获取饮料候选项的评分值（来自 metadata.drink_score）。"""
        metadata = dict(payload.get("metadata", {}) or {})
        return float(metadata.get("drink_score") or 0.0)

    best_drink = max(drink_payloads, key=drink_score)
    best_score = drink_score(best_drink)
    snapshot = dict(decision_state.get("llm_snapshot", {}) or {})
    stamina = int(snapshot.get("stamina") or 0)
    max_stamina = int(snapshot.get("max_stamina") or 0)
    stamina_ratio = float(stamina) / max(max_stamina, 1) if max_stamina > 0 else 1.0
    play_limit_raw = snapshot.get("play_limit_remaining")
    if play_limit_raw is None:
        play_limit_remaining = 1
    else:
        normalized_play_limit = fullwidth_to_halfwidth(str(play_limit_raw)).strip()
        match = re.search(r"\d+", normalized_play_limit)
        play_limit_remaining = int(match.group()) if match else 1
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
    """拼接候选项的完整描述文本（description + effect_types）。

    合并 payload.description、metadata.description 和 metadata.effect_types，
    用分号分隔，全角转半角后返回，供后续关键词匹配使用。
    """
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
    """判断卡片效果文本是否包含"即时输出"特征（打分/参数上升兑现）。

    检测关键词：打分、固定打分、パラ↑↑ 等。即时输出卡片在当前回合
    就能直接转化为分数收益，优先级较高。
    """
    normalized = str(text or "")
    if any(token in normalized for token in _BATTLE_IMMEDIATE_OUTPUT_TOKENS):
        return True
    if ProduceText.PARAMETER_UP_INCREASE in normalized:
        normalized = normalized.replace(ProduceText.PARAMETER_UP_INCREASE, "")
    return (
        ProduceText.PARAMETER in normalized
        and ProduceText.INCREASE in normalized
    )


def _extract_battle_immediate_score_points(text: str) -> int:
    """从效果文本中提取即时打分的数值总和。

    匹配"打分 +N"或"スコア +N"模式，支持中日文混合格式。
    """
    normalized = fullwidth_to_halfwidth(str(text or ""))
    if not normalized:
        return 0
    total = 0
    for pattern in _BATTLE_IMMEDIATE_SCORE_PATTERNS:
        for match in pattern.finditer(normalized):
            try:
                total += int(match.group(1) or 0)
            except (TypeError, ValueError):
                continue
    return max(total, 0)


def _extract_battle_multiplier(snapshot: dict[str, Any]) -> float:
    """解析当前倍率文本，返回数值倍率。"""
    raw_value = str(snapshot.get("score_bonus_multiplier") or "").strip()
    if not raw_value:
        return 1.0
    normalized = fullwidth_to_halfwidth(raw_value).lower().replace("x", "")
    try:
        multiplier = float(normalized)
    except ValueError:
        return 1.0
    return multiplier if multiplier > 0 else 1.0



def _battle_focus_token(llm_snapshot: dict[str, Any], *, phase: str) -> str:
    """提取当前回合更值得兑现的参数 token。"""
    if phase == "exam":
        wheel_info = dict(llm_snapshot.get("exam_wheel") or {})
        current_param = str(wheel_info.get("current_param") or "").lower()
        token = _BATTLE_PARAM_TOKEN_MAP.get(current_param, "")
        if token:
            return token
    return str(llm_snapshot.get("turn_color_display_label") or llm_snapshot.get("turn_color_label") or "")



def _score_battle_payload(
    payload: dict[str, Any],
    *,
    llm_snapshot: dict[str, Any],
    phase: str,
) -> tuple[float, list[str]]:
    """根据当前游戏状态为候选项计算优先级得分和理由列表。

    评分维度：
    - 即时输出（打分/参数兑现）：+20~38
    - 追加出牌次数：+26
    - 体力恢复：按体力危险程度 +14~24
    - 契合当前流派资源：+16
    - 剩余回合少时优先即时兑现/惩罚纯铺垫：±10~14
    - 考试排名压力下的抢分偏好：±10~12
    """
    metadata = dict(payload.get("metadata", {}) or {})
    text = _battle_payload_text(payload)
    immediate_score_points = _extract_battle_immediate_score_points(text)
    remaining_turns = int(llm_snapshot.get("remaining") or 0)
    stamina = int(llm_snapshot.get("stamina") or 0)
    max_stamina = int(llm_snapshot.get("max_stamina") or 0)
    stamina_ratio = float(stamina) / max(max_stamina, 1) if max_stamina > 0 else 1.0
    exam_ranking = int(llm_snapshot.get("exam_ranking") or 0)
    current_score = int(llm_snapshot.get("score") or 0)
    target_score = int(llm_snapshot.get("target") or 0)
    clear_achieved = llm_snapshot.get("clear_achieved")
    remaining_to_clear = int(llm_snapshot.get("remaining_to_clear") or 0)
    remaining_to_perfect = int(llm_snapshot.get("remaining_to_perfect") or 0)
    current_plan = str(llm_snapshot.get("idol_plan_label") or "")
    action_id = str(payload.get("id") or "")
    current_multiplier = _extract_battle_multiplier(llm_snapshot)
    focus_token = _battle_focus_token(llm_snapshot, phase=phase)
    wheel_info = dict(llm_snapshot.get("exam_wheel") or {}) if phase == "exam" else {}
    wheel_bonus = int(wheel_info.get("bonus_pct") or 0)
    score = 0.0
    reasons: list[str] = []

    if is_produce_drink_action_id(action_id):
        score += float(metadata.get("drink_score") or 0.0)
        reasons.append("当前是可立即使用的 P 饮料")
    else:
        if _battle_has_immediate_output(text):
            score += 20.0
            reasons.append("能立刻兑现当前回合收益")
            if immediate_score_points > 0:
                score += min(float(immediate_score_points) * 0.25, 18.0)
                reasons.append(f"即时打分面值更高({immediate_score_points})")

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

    if focus_token and focus_token in text:
        score += _BATTLE_COLOR_FOCUS_BONUS
        reasons.append(f"契合当前{focus_token}输出窗口")

    if phase == "exam" and focus_token and focus_token in text and wheel_bonus > 0:
        score += _BATTLE_WHEEL_FOCUS_BONUS
        reasons.append("命中当前考试轮盘参数窗口")
    elif phase == "exam" and focus_token and focus_token in text and current_multiplier >= _BATTLE_HIGH_BONUS_MULTIPLIER:
        score += _BATTLE_HIGH_BONUS_FOCUS_BONUS
        reasons.append("当前倍率窗口较高，优先兑现同色输出")

    if current_multiplier >= _BATTLE_VERY_HIGH_BONUS_MULTIPLIER and _battle_has_immediate_output(text):
        score += 10.0
        reasons.append("当前倍率很高，立即输出收益更大")

    under_clear_pressure = bool(
        remaining_turns > 0
        and remaining_turns <= 2
        and clear_achieved is False
        and remaining_to_clear > 0
        and remaining_to_clear <= _BATTLE_CLEAR_PRESSURE_THRESHOLD
    )
    perfect_push_window = bool(
        remaining_turns > 0
        and remaining_turns <= 2
        and clear_achieved is True
        and remaining_to_perfect > 0
        and remaining_to_perfect <= _BATTLE_PERFECT_PRESSURE_THRESHOLD
    )
    if under_clear_pressure:
        if immediate_score_points > 0 and immediate_score_points + _BATTLE_FINISHING_SCORE_MARGIN >= remaining_to_clear:
            score += _BATTLE_FINISHING_IMMEDIATE_BONUS
            reasons.append("这一击有机会直接过线，优先兑现")
        elif _battle_has_immediate_output(text):
            score += _BATTLE_PRESSURE_IMMEDIATE_BONUS
            reasons.append("CLEAR 压力下优先立即得分")
        elif any(token in text for token in _BATTLE_SETUP_TOKENS):
            score -= _BATTLE_PRESSURE_SETUP_PENALTY
            reasons.append("CLEAR 压力下继续铺垫偏慢")
    if perfect_push_window:
        if _battle_has_immediate_output(text):
            score += _BATTLE_PERFECT_PUSH_BONUS
            reasons.append("已过 CLEAR，当前更适合冲 PERFECT")
        elif any(token in text for token in _BATTLE_SETUP_TOKENS):
            score -= 8.0
            reasons.append("PERFECT 收尾阶段，纯铺垫价值下降")

    if remaining_turns > 0 and remaining_turns <= 2:
        if _battle_has_immediate_output(text):
            score += 14.0
            reasons.append("剩余回合少，优先立即兑现")
        elif any(token in text for token in _BATTLE_SETUP_TOKENS):
            score -= 12.0
            reasons.append("剩余回合少，纯铺垫价值下降")
    if phase == "lesson" and remaining_turns > 0 and remaining_turns <= 1:
        if immediate_score_points > 0:
            score += min(float(immediate_score_points) * 0.6, 24.0)
            reasons.append("临近收尾，优先更高即时打分")
        elif any(token in text for token in _BATTLE_SETUP_TOKENS):
            score -= _BATTLE_FINISHING_SETUP_PENALTY
            reasons.append("最后回合，纯铺垫基本无法兑现")

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
    """记录本地兜底偏好，不进入 LLM 候选、prompt 或 stage_context。"""
    label = f"候选 {preferred_index}"
    for payload in decision_state.get("candidates", []) or []:
        if int(payload.get("index", -1)) == preferred_index:
            label = str(payload.get("name") or payload.get("label") or label)
            break
    decision_state["local_preference"] = {
        "index": int(preferred_index),
        "label": label,
        "reason": str(reason or ""),
    }


def _select_battle_preference(
    decision_state: dict[str, Any],
    *,
    preferred_indices: Set[int],
    retryable_indices: Set[int],
    end_turn_indices: Set[int],
    phase: str,
) -> tuple[int | None, float, str]:
    """从优先候选或可重试候选中选出得分最高的卡片（排除 end_turn）。

    对每个候选调用 _score_battle_payload 计算得分，返回最高分的索引、
    得分值和理由摘要。

    Returns:
        (最佳索引, 得分, 理由文本)；无候选时索引为 None。
    """
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
        local_preference = dict(decision_state.get("local_preference", {}) or {})
        preferred_payload_index = int(local_preference.get("index", -1))
        if preferred_payload_index in preferred_indices:
            fallback_index = preferred_payload_index
            fallback_reason = "本地偏好候选"
        elif preferred_payload_index in retryable_indices and not preferred_indices:
            fallback_index = preferred_payload_index
            fallback_reason = "本地偏好候选(重试)"

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
    """点击屏幕空白区域取消当前卡片选中状态。

    通过 app.latest_frame.shape 动态获取屏幕宽高，点击位置为屏幕中下部（X=50%, Y=83%），
    位于角色立绘区，不会触发任何 UI 元素。
    """
    frame = app.latest_frame
    if frame is None or frame.size == 0:
        logger.warning("deselect_card: 无法获取画面尺寸，跳过取消选中操作")
        return
    height, width = frame.shape[:2]
    tap_x = width // 2
    tap_y = int(height * _DESELECT_TAP_Y_RATIO)
    app.device.click(tap_x, tap_y, el_label="deselect_card")
    time.sleep(0.5)


def _get_battle_end_turn_boxes(results: Any) -> List[Any]:
    """获取"结束回合/SKIP"按钮的检测框列表。

    按 (cy, cx) 排序，优先取屏幕中位置最靠前的按钮。
    """
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
    """点击"结束回合/SKIP"按钮。

    优先使用最新帧检测到的 PC_SKIP 按钮，否则使用 fallback_box。
    点击位置偏向按钮左侧 40% 处，避免误触右侧图标。
    """
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
    fallback_candidates = _filter_unavailable_battle_drinks(
        _collect_battle_drink_candidates(
            app,
            ctx,
            phase=phase,
            start_index=0,
        )
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
        # 等待饮料详情模态出现并点击“使う”确认。
        if _confirm_drink_usage_modal(app):
            logger.info("{}: 饮料 {!r} 使用确认成功", phase_tag, target.title)
            _clear_lesson_drink_cache_after_use(ctx)
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
    _wait_battle_play_animation_end(
        app,
        phase=phase,
        position=position,
        pending_index=ctx.pending_lesson_card_index,
    )
    runtime_phase, runtime_position = detect_gameplay_state(app, ctx)
    if runtime_phase not in {phase, GameplayPhase.EXAM, GameplayPhase.UNKNOWN}:
        _dump_phase_drift_probe(
            app,
            ctx,
            expected_phase=phase,
            runtime_phase=runtime_phase,
            runtime_position=runtime_position,
            source="execute_entry",
        )
        logger.info(
            "lesson: 执行中检测到阶段漂移 phase={} position={}，停止 lesson 步骤并交回主循环",
            runtime_phase,
            runtime_position,
        )
        ctx.set_phase(runtime_phase)
        ctx.set_position(runtime_position)
        return LessonStepResult(status="phase_drift")
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
            runtime_phase, runtime_position = detect_gameplay_state(app, ctx)
            if runtime_phase not in {phase, GameplayPhase.EXAM, GameplayPhase.UNKNOWN}:
                _dump_phase_drift_probe(
                    app,
                    ctx,
                    expected_phase=phase,
                    runtime_phase=runtime_phase,
                    runtime_position=runtime_position,
                    source="refresh_candidates",
                )
                logger.info(
                    "lesson: 刷新候选前检测到阶段漂移 phase={} position={}，停止 lesson 步骤并交回主循环",
                    runtime_phase,
                    runtime_position,
                )
                ctx.set_phase(runtime_phase)
                ctx.set_position(runtime_position)
                return LessonStepResult(status="phase_drift")
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
            # 等待饮料详情模态出现并点击“使う”确认。
            if _confirm_drink_usage_modal(app):
                logger.info("lesson: 饮料 {!r} 使用确认成功", target.title)
                _clear_lesson_drink_cache_after_use(ctx)
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
        runtime_phase, runtime_position = detect_gameplay_state(app, ctx)
        if runtime_phase not in {phase, GameplayPhase.EXAM, GameplayPhase.UNKNOWN}:
            _dump_phase_drift_probe(
                app,
                ctx,
                expected_phase=phase,
                runtime_phase=runtime_phase,
                runtime_position=runtime_position,
                source="retry_before_recollect",
            )
            logger.info(
                "lesson: 重试前检测到阶段漂移 phase={} position={}，停止 lesson 步骤并交回主循环",
                runtime_phase,
                runtime_position,
            )
            ctx.set_phase(runtime_phase)
            ctx.set_position(runtime_position)
            return LessonStepResult(status="phase_drift")
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
# 处理器
# ────────────────────────────────────────────────────────────

class LessonHandler:
    """Lesson/Exam 阶段出牌的 gameplay handler 包装。

    核心委托给 execute_lesson_step()，处理空手牌时走 fallback 逻辑
    选择底栏饮料或 SKIP。LESSON_SUMMARY_SHOWCASE 位置特殊处理为
    点击安全区域推进。
    """

    phase_tag = "lesson"
    priority = 50

    def can_handle(self, app, ctx, phase, position):
        """仅接管 phase == "lesson" 的画面（exam 阶段由 ExamHandler 处理）。"""
        return phase == "lesson"

    @staticmethod
    def _detect_phase_drift(app, ctx, *, expected_phase: str) -> tuple[str, str] | None:
        """在 lesson 处理前复核当前实时画面，避免中途切页后继续沿用旧 phase。

        真机上 lesson 出牌后可能立刻切到 skill_reward / p_drink / result 等相邻阶段。
        如果 handler 仍按进入时的 lesson phase 继续执行，就会把奖励页误当手牌页处理。
        这里做一次轻量复核，发现当前帧已经漂移到非 lesson/exam 的新阶段时，
        立刻把控制权交回主循环重新分发。
        """
        current_phase, current_position = detect_gameplay_state(app, ctx)
        if current_phase in {expected_phase, GameplayPhase.EXAM, GameplayPhase.UNKNOWN}:
            return None
        _dump_phase_drift_probe(
            app,
            ctx,
            expected_phase=expected_phase,
            runtime_phase=current_phase,
            runtime_position=current_position,
            source="handler_entry",
        )
        logger.info(
            "lesson: 处理前检测到阶段漂移 phase={} position={}，停止 lesson 处理并交回主循环",
            current_phase,
            current_position,
        )
        return current_phase, current_position

    def handle(self, app, ctx, phase, position):
        """执行 lesson 出牌逻辑。

        特殊位置 LESSON_SUMMARY_SHOWCASE 直接点击推进；其余位置委托
        execute_lesson_step，返回 None 时走空手牌 fallback。
        """
        from src.core.tasks.producer_challenge.gameplay.handler_base import HandlerResult

        drift_state = self._detect_phase_drift(app, ctx, expected_phase=phase)
        if drift_state is not None:
            drift_phase, drift_position = drift_state
            ctx.set_phase(drift_phase)
            ctx.set_position(drift_position)
            return HandlerResult.no_action(
                f"lesson: phase drift -> {drift_phase}/{drift_position}",
                sleep_after=0.0,
            )

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
        if result.status == "phase_drift":
            return HandlerResult.no_action(
                f"lesson: phase drift -> {ctx.gameplay_phase}/{ctx.gameplay_position}",
                sleep_after=0.0,
            )
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
        """处理repr并返回结果。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
        return f"<LessonHandler phase={self.phase_tag!r} priority={self.priority}>"
