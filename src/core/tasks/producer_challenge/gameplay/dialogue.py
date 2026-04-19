"""対話 / コミュ handler。

対話画面包括:
  - 2-3 个可選選項 (Universal Options)
  - 快進按鈕 (Fast Forward)
  - 可点击推進的劇情文本

交互模式（経 ADB 実測確認）:
  - 選項需要双击: 第一次点击高亮選中，第二次点击確認。
  - 快進: 単击切換自動推進。
  - 純劇情文本: 点击任意位置継続。

おでかけ（外出）選項探査:
  外出画面会顯示 2-3 個選項（如「いちごミルク -100P」「キャラメル -50P」），
  選項名是劇情台詞（不在 DB 中），但:
    1. 選項框內 OCR 可提取選項名 + P 点消耗
    2. YOLO「Action Info」區域包含当前高亮選項的効果描述
    3. 通過逐個点击選項（探査），可采集所有効果描述
    4. 将 P 点成本 + 効果描述注入候選項 metadata，提供給 LLM 決策
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
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
    status: str  # "selected" | "confirmed" | "fast_forward" | "advanced"
    candidate: DialogueOptionCandidate | None = None


# ────────────────────────────────────────────────────────────
# おでかけ探査 — 定数 / 正規表現
# ────────────────────────────────────────────────────────────

# P 点消耗パターン: "P-100", "PP-100", "P 50", "-100"(P被OCR截断) 等
# ^[-ー] は選項先頭のマイナス記号（P が別行に分離した場合）を拾う
_P_COST_RE = re.compile(r"(?:P{1,2}\s*[-ー]?\s*|^[-ー]\s*)(\d+)", re.MULTILINE)

# おでかけ探査: 点击後 UI 刷新等待
_OUTING_PROBE_TAP_WAIT = 0.5
# おでかけ探査: 推理等待（Yolo再検出）
_OUTING_PROBE_INFER_WAIT = 0.3

# Debug 可視化
from src.utils.debug_tools import DebugTools
_debugger = DebugTools()
_DIALOGUE_FAST_FORWARD_ENABLED_KEY = "dialogue_fast_forward_enabled"
_DIALOGUE_FAST_FORWARD_LAST_CLICK_TS_KEY = "dialogue_fast_forward_last_click_ts"


# ────────────────────────────────────────────────────────────
# おでかけ探査 — 工具函数
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
    # 仅在出现 P 点消耗模式时判定为おでかけ，避免把普通周事件误判成外出。
    option_boxes = list(app.latest_results.filter_by_label(ProducerLabels.UNIVERSAL_OPTIONS))
    return any(_extract_p_cost(ocr_text(getattr(box, "frame", None))) is not None for box in option_boxes)


def _is_dialogue_option_info_context(app: "AppProcessor", position: str) -> bool:
    """判断当前是否可通过 Action Info 面板读取周事件选项效果。"""
    if position not in {"schedule_event_options", "dialogue_options"}:
        return False
    info_boxes = app.latest_results.filter_by_label(ProducerLabels.PC_ACTION_INFO)
    return bool(info_boxes)


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
    info_boxes = app.latest_results.filter_by_label(ProducerLabels.PC_ACTION_INFO)
    if not info_boxes:
        return ""
    info_box = info_boxes.first()
    frame = getattr(info_box, "frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return ""
    text = ocr_text(frame)

    # Debug 可視化: Action Info 区域 + OCR 結果
    _debugger.add_box(
        info_box.x, info_box.y, info_box.w, info_box.h,
        color=(0, 200, 255), thickness=2, duration=3,
        label=f"ActionInfo: {text[:30]}",
    )
    return text.strip()


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

    # 第一個選項通常已預選 → 先直接読取 Action Info
    first_desc = _extract_action_info_description(app)
    if first_desc:
        candidates[0].metadata["outing_effect"] = first_desc
        logger.debug(
            "dialogue: おでかけ選項 #0 効果(預選): {}",
            first_desc[:60],
        )

    # 逐個点击残りの選項
    for candidate in candidates:
        # 提取 P 点消耗
        p_cost = _extract_p_cost(candidate.title)
        if p_cost is not None:
            candidate.metadata["p_cost"] = p_cost

        # 已取得効果描述的（第一個預選項）跳過
        if candidate.metadata.get("outing_effect"):
            continue

        try:
            # 点击切換高亮
            app.device.click_element(candidate.box)
            time.sleep(_OUTING_PROBE_TAP_WAIT)
            time.sleep(_OUTING_PROBE_INFER_WAIT)

            # 読取 Action Info
            desc = _extract_action_info_description(app)
            if desc:
                candidate.metadata["outing_effect"] = desc
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

    first_desc = _extract_action_info_description(app)
    if first_desc:
        candidates[0].metadata["option_effect"] = first_desc

    for candidate in candidates:
        if candidate.metadata.get("option_effect"):
            continue
        try:
            app.device.click_element(candidate.box)
            time.sleep(_OUTING_PROBE_TAP_WAIT)
            time.sleep(_OUTING_PROBE_INFER_WAIT)
            desc = _extract_action_info_description(app)
            if desc:
                candidate.metadata["option_effect"] = desc
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
    """重置对话卡住计数。"""
    ctx.handler_state.pop("dialogue_stuck_count", None)
    ctx.handler_state.pop("dialogue_stuck_last_option", None)
    ctx.handler_state.pop("dialogue_skip_indices", None)


def _get_fast_forward_buttons(app: "AppProcessor") -> list[Any]:
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

        # ── おでかけ探査: 在第一次選択前采集效果描述 ──
        if (
            ctx.pending_dialogue_option_index is None
            and _is_dialogue_option_info_context(app, position)
            and not any(
                c.metadata.get("description")
                or c.metadata.get("outing_effect")
                or c.metadata.get("option_effect")
                for c in candidates
            )
        ):
            if _is_outing_context(app, position):
                _probe_outing_options(app, candidates)
                _enrich_outing_descriptions(candidates)
                # おでかけ DB マッチング: 効果描述 + P 成本 → 安定的 DB ID
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

    # Skip 按钮（おでかけ剧情等未读コミュ）— 直接跳过
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
# Handler
# ────────────────────────────────────────────────────────────

class DialogueHandler(GameplayHandler):
    """对话 / コミュ画面处理。"""

    phase_tag = "dialogue"
    priority = 50

    def can_handle(self, app, ctx, phase, position):
        return phase == "dialogue"

    def handle(self, app, ctx, phase, position):
        result = execute_dialogue_step(app, ctx, position=position)
        if result is None:
            return HandlerResult.no_action("no dialogue elements")
        if result.status in {"selected", "confirmed", "fast_forward", "skipped", "advanced"}:
            _set_dialogue_transition_retry_override(
                ctx,
                reason=f"dialogue_{result.status}",
            )
        return HandlerResult.ok(f"dialogue {result.status}", sleep_after=0.6)
