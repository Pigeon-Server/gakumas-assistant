"""Step 11: 培育主循环 — 基于 handler 调度器的可扩展 gameplay loop。

这是培育过程中最核心的循环步骤，处理从第一周行程选择到
生产结束之间的所有游戏交互。

当前支持的画面类型（handler 可通过 dispatcher 动态注册/替换）:
  - schedule:     周行程选择画面 → 选择/确认行程
  - dialogue:     对话/剧情选项 → 双击选项/快进
  - lesson:       レッスン → 手牌选择/出牌
  - exam:         試験/オーディション → 手牌选择（与 lesson 共用机制）
  - skill_reward: 技能卡奖励选择 → 选卡→确认
  - p_drink:      P饮料选择 → 选饮料→确认
  - consult:      相談交换页 → 交换/強化/削除
  - modal:        弹窗 → 确认关闭
  - effect_chain: 过场展示（奖励/メモリー/角色过场）→ 点击推进
  - result:       结果画面 → 退出循环

扩展方式:
  - 新增 handler: 在 gameplay/ 下创建模块，注册到 dispatcher
  - 替换 handler: dispatcher.unregister() + register()
  - 调整优先级: 修改 handler 的 priority 属性
"""

import re

from time import monotonic, sleep
from typing import TYPE_CHECKING

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.constants.game.text.general_text import GeneralText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.model_type import YoloModelType
from src.core.tasks.base_ui.start_game import action__wait_enter_home
from src.core.tasks.producer_challenge.context import GameplayPhase
from src.core.tasks.producer_challenge.shared.common import (
    click_relative_point,
    normalize_text,
    probe_fast_forward_enabled_state,
)
from src.core.tasks.producer_challenge.gameplay.decision import sync_visible_planning_context
from src.core.tasks.producer_challenge.gameplay import build_default_dispatcher
from src.core.tasks.producer_challenge.gameplay.strategy.llm_strategy import LLMStrategy
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.steps.entry.navigate_to_produce import (
    open_produce_entry_from_home,
    resume_resumable_produce,
)
from src.core.tasks.producer_challenge.ui import (
    collect_frame_text,
    detect_gameplay_state,
    find_button,
)
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.i18n_tools import i18n_text
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


_STORY_JP_TEXT_RE = re.compile(r"[ぁ-んァ-ヶ一-龯]")
_STORY_EPISODE_RE = re.compile(r"\d+\s*話")

# ── 培育收尾：结果链推进直到回到主页 ──
_FINISHING_MAX_ITERATIONS = 120   # 最多轮询次数（约2分钟）
_FINISHING_POLL_INTERVAL = 1.0    # 每次轮询间隔（秒）
_FINISHING_HOME_CONFIRM = 3       # 连续 N 次检测到主页才确认


def _finish_produce_with_base_ui(
    app: "AppProcessor",
    ctx: "ProduceContext",
) -> bool:
    """切换到 BASE_UI 模型，推进结果链展示画面直到回到主页。

    此函数在到达记忆卡面选择等后期通用 UI 阶段后被调用，
    此时游戏画面已经是 BASE_UI 可识别的通用元素（按钮、弹窗等）。

    策略：
      1. 切换到 BASE_UI 模型
      2. 每轮检测是否已到主页（TAB_HOME）
      3. 有按钮就点，没按钮就点屏幕中央推进
      4. 连续多次确认主页后返回 True
    """
    logger.info("produce_finishing: 切换到 BASE_UI 模型，开始推进结果链")
    app.switch_yolo_model(YoloModelType.BASE_UI, settle_seconds=1.0)

    home_count = 0  # 连续检测到主页的次数

    for iteration in range(_FINISHING_MAX_ITERATIONS):
        sleep(_FINISHING_POLL_INTERVAL)
        results = app.latest_results
        if results is None:
            continue

        # 检测主页
        if results.filter_by_label(BaseUILabels.TAB_HOME):
            home_count += 1
            logger.debug(
                "produce_finishing: 检测到 TAB_HOME ({}/{})",
                home_count, _FINISHING_HOME_CONFIRM,
            )
            if home_count >= _FINISHING_HOME_CONFIRM:
                logger.success("produce_finishing: 已确认回到主页")
                return True
            continue
        home_count = 0

        # 处理弹窗（如受取完了、奖励确认等）
        if results.filter_by_label(BaseUILabels.MODAL_HEADER):
            modal = app.game_utils.try_get_modal(no_body=True)
            if modal is not None:
                from src.core.tasks.base_ui.start_game import _handle__modal_boxes
                _handle__modal_boxes(app)
                logger.debug("produce_finishing: 处理弹窗 (iter={})", iteration)
                continue

        # 点击可见按钮推进（如“次へ”“完了する”“確認”）。
        buttons = list(results.filter_by_label(BaseUILabels.BUTTON))
        if buttons:
            target = max(buttons, key=lambda b: b.cy)
            app.device.click_element(target)
            logger.debug("produce_finishing: 点击按钮推进 (iter={})", iteration)
            continue

        # 点击 close/skip/fast-forward/back 按钮
        for label, desc in (
            (BaseUILabels.CLOSE_BUTTON, "关闭"),
            (BaseUILabels.SKIP_BUTTON, "跳过"),
            (BaseUILabels.PLOT_FAST_FORWARD_BUTTON, "快进"),
            (BaseUILabels.BACK_BTN, "返回"),
        ):
            found = list(results.filter_by_label(label))
            if found:
                app.device.click_element(found[0])
                logger.debug("produce_finishing: 点击{}按钮 (iter={})", desc, iteration)
                break
        else:
            # 无任何按钮 → 点击屏幕中央推进
            click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="finishing-tap")
            logger.debug("produce_finishing: 点击屏幕推进 (iter={})", iteration)

    logger.warning("produce_finishing: 达到最大轮询次数仍未回到主页")
    return False


def _try_back_button_recovery(app: "AppProcessor") -> bool:
    """在 UNKNOWN 画面上优先尝试点击返回按钮做轻量恢复。

    适用于误入手牌详情、说明页等 producer 子页面的场景。这类页面通常仍在当前流程里，
    不需要完整回主页重进，只要能点掉 `BACK_BTN` 就能回到主画面继续识别。
    """
    results = app.latest_results
    if results is None or not hasattr(results, "filter_by_label"):
        return False
    back_buttons = list(results.filter_by_label(BaseUILabels.BACK_BTN))
    if back_buttons:
        logger.info("recovery: 检测到 Back Button，点击返回")
        app.device.click_element(back_buttons[0])
        return True
    return False


def _switch_model_for_recovery(
    app: "AppProcessor",
    model_type: str,
    *,
    settle_seconds: float = 1.0,
) -> None:
    """切换 YOLO 模型，并等待首帧稳定，供外页恢复流程复用。"""
    logger.info(f"recovery: 切换 YOLO 模型到 {model_type}")
    app.switch_yolo_model(model_type, settle_seconds=settle_seconds)


def _looks_like_external_recovery_page(frame_text: str) -> bool:
    """识别 producer 之外、但已知可自动清理并恢复的全局页面。"""
    normalized = normalize_text(frame_text)
    if not normalized:
        return False
    return any(
        normalize_text(token) in normalized
        for token in GeneralText.PRODUCE_EXTERNAL_RECOVERY_TOKENS
    )


def _is_unclassified_recovery_location(current_location: str | None) -> bool:
    """只允许在 BASE_UI 仍无法判定明确页面时走 producer 内部补救。"""
    return current_location in {None, GamePageTypes.UNKNOWN, GamePageTypes.LOADING}


def _should_fast_poll_unknown_retry(ctx: "ProduceContext") -> bool:
    """结果链说明页需要更快轮询，避免两次推进之间等待过久。"""
    retry_override = ctx.handler_state.get("unknown_retry_override")
    if not isinstance(retry_override, dict):
        return False
    reason = str(retry_override.get("reason", ""))
    return reason.startswith("result_memory_generate_recovery") or reason.startswith(
        "result_chain_tap_recovery"
    ) or reason.startswith("result_reward_overlay_recovery") or reason.startswith(
        "result_closeout_summary_recovery"
    )


def _looks_like_story_dialogue_recovery_page(results, frame_text: str) -> bool:
    """识别 producer 内部短暂落到 BASE_UI 的剧情消息页。

    这类页面的共同点是：
    - PRODUCER 模型通常 0 框，导致主循环把它当成 unknown；
    - BASE_UI OCR 能读到「メッセージ」等剧情文本，且常伴随 SKIP/快进控件；
    - 实际上仍属于 producer 内剧情过场，应该继续推进，而不是回主页恢复。
    """
    normalized = normalize_text(frame_text)
    if not normalized:
        return False

    has_message_token = normalize_text(ProduceText.MESSAGE) in normalized
    has_episode_marker = bool(_STORY_EPISODE_RE.search(str(frame_text or "")))
    if not (has_message_token or has_episode_marker):
        return False

    jp_char_count = len(_STORY_JP_TEXT_RE.findall(normalized))
    if jp_char_count < 12:
        return False

    has_skip_button = bool(results and results.exists_label(BaseUILabels.SKIP_BUTTON))
    has_fast_forward = bool(results and results.exists_label(BaseUILabels.PLOT_FAST_FORWARD_BUTTON))
    has_skip_text = normalize_text(ProduceText.SKIP) in normalized
    return (has_skip_button or has_fast_forward or has_skip_text) and any(
        token in normalized
        for token in (
            normalize_text(ProduceText.MESSAGE),
            *(
                normalize_text(token)
                for token in ProduceText.STORY_DIALOGUE_RECOVERY_HINT_TOKENS
            ),
        )
    )


def _looks_like_reward_receive_confirmation_page(results, frame_text: str) -> bool:
    """识别奖励领取确认页（如活動支給里的饮料/奖励确认）。"""
    normalized = normalize_text(frame_text)
    if not normalized:
        return False

    has_receive = normalize_text(ProduceText.RECEIVE) in normalized
    has_reward_detail = any(
        normalize_text(token) in normalized
        for token in ProduceText.REWARD_RECEIVE_DETAIL_TOKENS
    )
    return has_receive and has_reward_detail


def _try_story_dialogue_recovery(app: "AppProcessor", ctx: "ProduceContext", frame_text: str) -> bool:
    """把 BASE_UI 复核命中的剧情消息页按 dialogue_continue 推进。"""
    results = app.latest_results
    if not _looks_like_story_dialogue_recovery_page(results, frame_text):
        return False

    ff_state_key = "dialogue_fast_forward_enabled"
    ff_last_click_key = "dialogue_fast_forward_last_click_ts"
    skip_buttons = list(results.filter_by_label(BaseUILabels.SKIP_BUTTON)) if results else []
    fast_forward_buttons = (
        list(results.filter_by_label(BaseUILabels.PLOT_FAST_FORWARD_BUTTON))
        if results
        else []
    )
    if not fast_forward_buttons:
        ctx.handler_state.pop(ff_state_key, None)
        ctx.handler_state.pop(ff_last_click_key, None)

    action = "advance"
    if skip_buttons:
        app.device.click_element(skip_buttons[0])
        action = "skip"
    elif fast_forward_buttons:
        ff_box = fast_forward_buttons[0]
        enabled_state, orange_ratio = probe_fast_forward_enabled_state(
            ff_box,
            debug_tools=getattr(app, "debug_tools", None),
            debug_label="story_recovery_fast_forward",
        )
        if enabled_state is True:
            ctx.handler_state[ff_state_key] = True
            click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="story-dialogue-recovery")
            action = "advance"
        elif (
            ctx.handler_state.get(ff_state_key)
            and enabled_state is not False
        ):
            click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="story-dialogue-recovery")
            action = "advance"
        else:
            now = monotonic()
            last_click = float(ctx.handler_state.get(ff_last_click_key, 0.0) or 0.0)
            if now - last_click >= 0.9:
                app.device.click_element(ff_box)
                ctx.handler_state[ff_state_key] = True
                ctx.handler_state[ff_last_click_key] = now
                logger.debug("recovery: 剧情页开启快进（orange_ratio={:.3f}）", orange_ratio)
                action = "fast_forward"
            else:
                click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="story-dialogue-recovery")
                action = "advance"
    else:
        click_relative_point(app, x_ratio=0.5, y_ratio=0.82, label="story-dialogue-recovery")
        action = "advance"

    logger.info("recovery: BASE_UI 复核命中剧情消息页，按 dialogue_continue 处理 ({})", action)
    ctx.handler_state["unknown_retry_override"] = {
        "reason": f"story_dialogue_recovery_{action}",
        "retry_limit": int(
            ctx.handler_state.get("dialogue_transition_unknown_retry_limit", 8) or 8
        ),
        "retry_sleep": float(
            ctx.handler_state.get("dialogue_transition_unknown_retry_sleep", 0.7) or 0.7
        ),
    }
    return True


def _try_reward_receive_confirmation_recovery(
    app: "AppProcessor",
    ctx: "ProduceContext",
    frame_text: str,
) -> bool:
    """把 BASE_UI 复核命中的奖励领取确认页直接点过。"""
    results = app.latest_results
    if not _looks_like_reward_receive_confirmation_page(results, frame_text):
        return False

    buttons = list(results.filter_by_label(BaseUILabels.BUTTON)) if results else []
    if buttons:
        app.device.click_element(sorted(buttons, key=lambda box: box.cy)[-1])
    else:
        click_relative_point(app, x_ratio=0.5, y_ratio=0.81, label="reward-receive-confirm")

    logger.info("recovery: BASE_UI 复核命中奖励领取确认页，点击受け取る继续")
    ctx.handler_state["unknown_retry_override"] = {
        "reason": "reward_receive_confirmation",
        "retry_limit": int(
            ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
        ),
        "retry_sleep": float(
            ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
        ),
    }
    return True


def _looks_like_live_tap_to_start_recovery_page(frame_text: str) -> bool:
    """识别 live 开始前的竖屏 TAP TO START 提示页。"""
    normalized = normalize_text(frame_text)
    if not normalized:
        return False
    has_landscape_notice = any(
        normalize_text(token) in normalized
        for token in ProduceText.LANDSCAPE_START_NOTICE_OCR_VARIANTS
    )
    has_tap_to_start = any(
        normalize_text(token) in normalized
        for token in ProduceText.TAP_TO_START_OCR_VARIANTS
    )
    return has_landscape_notice or has_tap_to_start


def _try_live_tap_to_start_recovery(
    app: "AppProcessor",
    ctx: "ProduceContext",
    frame_text: str,
) -> bool:
    """把 BASE_UI 复核命中的 TAP TO START 提示页直接点进 live。"""
    if not _looks_like_live_tap_to_start_recovery_page(frame_text):
        return False

    click_relative_point(
        app,
        x_ratio=0.5,
        y_ratio=0.77,
        label="live-tap-to-start-recovery",
    )
    logger.info("recovery: BASE_UI 复核命中 live 开始提示页，点击 TAP TO START 继续")
    ctx.handler_state["unknown_retry_override"] = {
        "reason": "live_tap_to_start_recovery",
        "retry_limit": int(
            ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
        ),
        "retry_sleep": float(
            ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
        ),
    }
    return True


def _looks_like_result_memory_generate_page(frame_text: str) -> bool:
    """识别结果链里的 MEMORY 生成页（圆形「生成」按钮页）。"""
    text = str(frame_text or "")
    normalized = normalize_text(text)
    if not normalized:
        return False
    has_generate = ButtonText.GENERATE in text
    has_memory = any(
        normalize_text(token) in normalized
        for token in ProduceText.MEMORY_OCR_VARIANTS
    )
    if (
        ProduceText.MEMORY_GENERATION_COMPLETE in text
        or ButtonText.REGENERATE in text
    ):
        return False
    return has_generate and (has_memory or len(normalized) <= 16)


def _try_result_memory_generate_recovery(
    app: "AppProcessor",
    ctx: "ProduceContext",
    frame_text: str,
) -> bool:
    """把结果链里误落成 unknown 的 MEMORY 生成页直接点过。"""
    if not _looks_like_result_memory_generate_page(frame_text):
        return False

    click_relative_point(
        app,
        x_ratio=0.5,
        y_ratio=0.77,
        label="result-memory-generate-recovery",
    )
    logger.info("recovery: BASE_UI 复核命中 MEMORY 生成页，点击生成继续结果链")
    ctx.handler_state["unknown_retry_override"] = {
        "reason": "result_memory_generate_recovery",
        "retry_limit": int(
            ctx.handler_state.get("result_transition_unknown_retry_limit", 2) or 2
        ),
        "retry_sleep": float(
            ctx.handler_state.get("result_transition_unknown_retry_sleep", 0.3) or 0.3
        ),
    }
    return True


def _looks_like_result_chain_tap_page(frame_text: str) -> bool:
    """识别结果链里需要 TAP 推进、但 PRODUCER 模型无框的展示页。"""
    text = str(frame_text or "")
    normalized = normalize_text(text)
    if not normalized:
        return False
    has_tap = normalize_text(ProduceText.TAP) in normalized
    is_live_tap = any(
        normalize_text(token) in normalized
        for token in ProduceText.TAP_TO_START_OCR_VARIANTS
    )
    return has_tap and not is_live_tap


def _try_result_chain_tap_recovery(
    app: "AppProcessor",
    ctx: "ProduceContext",
    frame_text: str,
) -> bool:
    """把结果链里无按钮/弱按钮的 TAP 展示页继续点过。"""
    if not _looks_like_result_chain_tap_page(frame_text):
        return False

    next_button = find_button(app, ButtonText.NEXT, fuzz_threshold=45)
    if next_button is not None:
        app.device.click_element(next_button)
        action = "next"
        ctx.handler_state["result_chain_finish_pending"] = True
    else:
        click_relative_point(
            app,
            x_ratio=0.5,
            y_ratio=0.5,
            label="result-chain-tap-recovery",
        )
        action = "tap"

    logger.info("recovery: BASE_UI 复核命中结果链 TAP 展示页，继续推进 ({})", action)
    ctx.handler_state["unknown_retry_override"] = {
        "reason": f"result_chain_tap_recovery_{action}",
        "retry_limit": int(
            ctx.handler_state.get("result_transition_unknown_retry_limit", 2) or 2
        ),
        "retry_sleep": float(
            ctx.handler_state.get("result_transition_unknown_retry_sleep", 0.3) or 0.3
        ),
    }
    return True


def _try_external_page_recovery(app: "AppProcessor", ctx: "ProduceContext") -> bool:
    """当 gameplay 意外掉到游戏外页面时，尝试回主页并恢复未完成培育。"""
    logger.info("recovery: 疑似已掉出 producer gameplay，切到 BASE_UI 复核")
    base_ui_settle_seconds = float(
        ctx.handler_state.get("external_recovery_base_ui_settle", 0.8) or 0.8
    )
    producer_settle_seconds = float(
        ctx.handler_state.get("external_recovery_producer_settle", 0.8) or 0.8
    )
    quick_recovery_producer_settle = float(
        ctx.handler_state.get("quick_recovery_producer_settle", 0.4) or 0.4
    )
    _switch_model_for_recovery(
        app,
        YoloModelType.BASE_UI,
        settle_seconds=base_ui_settle_seconds,
    )
    try:
        if resume_resumable_produce(app, timeout=1.0):
            logger.success("recovery: 直接命中培育再开弹窗，已恢复旧局")
        else:
            current_location = app.game_utils.update_current_location()
            frame_text = collect_frame_text(app.latest_results)
            if (
                current_location == GamePageTypes.MAIN_MENU__HOME
                and ctx.handler_state.pop("result_chain_finish_pending", False)
            ):
                logger.success("recovery: 结果链点完次へ后已回主页，视为本轮培育正常结束")
                ctx.handler_state["result_chain_completed"] = True
                return True
            if _is_unclassified_recovery_location(current_location):
                if _try_result_memory_generate_recovery(app, ctx, frame_text):
                    producer_settle_seconds = quick_recovery_producer_settle
                    return True
                if _try_live_tap_to_start_recovery(app, ctx, frame_text):
                    producer_settle_seconds = quick_recovery_producer_settle
                    return True
                if _try_result_chain_tap_recovery(app, ctx, frame_text):
                    producer_settle_seconds = quick_recovery_producer_settle
                    return True
                if _try_story_dialogue_recovery(app, ctx, frame_text):
                    producer_settle_seconds = quick_recovery_producer_settle
                    return True
                if _try_reward_receive_confirmation_recovery(app, ctx, frame_text):
                    producer_settle_seconds = quick_recovery_producer_settle
                    return True
                if _try_back_button_recovery(app):
                    logger.info("recovery: BASE_UI 复核命中可返回子页面，点击 Back Button 继续")
                    ctx.handler_state["unknown_retry_override"] = {
                        "reason": "external_back_button_recovery",
                        "retry_limit": int(
                            ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
                        ),
                        "retry_sleep": float(
                            ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
                        ),
                    }
                    producer_settle_seconds = quick_recovery_producer_settle
                    return True
            has_external_tokens = _looks_like_external_recovery_page(frame_text)
            has_known_location = not _is_unclassified_recovery_location(current_location)
            logger.info(
                "recovery probe: location={}, external_tokens={}, text={}",
                current_location,
                has_external_tokens,
                frame_text[:60],
            )
            if not (has_known_location or has_external_tokens):
                logger.info("recovery: BASE_UI 复核未命中外页特征，回退给未知页分析")
                return False

            action__wait_enter_home(app)
            open_produce_entry_from_home(app, timeout=10)
            if not resume_resumable_produce(app, timeout=8.0):
                raise RuntimeError(i18n_text("backend.task.externalPageRecoveryFailed", fallback="外页恢复失败：已回到主页并打开 Produce，但未命中未完成培育再开弹窗"))
            logger.success("recovery: 已通过主页入口恢复未完成培育")

        ctx.handler_state["unknown_retry_override"] = {
            "reason": "external_page_recovery",
            "retry_limit": int(
                ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
            ),
            "retry_sleep": float(
                ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
            ),
        }
        return True
    finally:
        _switch_model_for_recovery(
            app,
            YoloModelType.PRODUCER,
            settle_seconds=producer_settle_seconds,
        )


def _stabilize_unknown_state(
    app: "AppProcessor",
    ctx: "ProduceContext",
) -> tuple[str, str]:
    """对单帧 unknown 做无操作复检，避免把瞬时空帧误判成未知页面。

    真机联调时，切页动画、摄像头动态模糊、YOLO 单帧漏检都可能导致一次性
    `phase=unknown`。这时不能立刻继续盲点，但也不该马上把明显已稳定的页面
    当成未知页暂停；因此先被动等待几帧重新取样，只有连续 unknown 才认定为
    真实未知页面。
    """
    phase, position = detect_gameplay_state(app, ctx)
    retry_override = ctx.handler_state.get("unknown_retry_override")
    if phase != GameplayPhase.UNKNOWN:
        if retry_override is not None:
            ctx.handler_state.pop("unknown_retry_override", None)
        return phase, position

    retry_limit = int(ctx.handler_state.get("unknown_retry_limit", 2) or 0)
    retry_sleep = float(ctx.handler_state.get("unknown_retry_sleep", 0.4) or 0.0)
    if isinstance(retry_override, dict):
        retry_limit = int(retry_override.get("retry_limit", retry_limit) or retry_limit)
        retry_sleep = float(retry_override.get("retry_sleep", retry_sleep) or retry_sleep)
        logger.debug(
            "unknown retry override: reason={}, limit={}, sleep={}",
            retry_override.get("reason", "unknown"),
            retry_limit,
            retry_sleep,
        )

    for retry_index in range(1, retry_limit + 1):
        sleep(retry_sleep)
        retried_phase, retried_position = detect_gameplay_state(app, ctx)
        logger.debug(
            "unknown retry {}/{} -> phase={}, position={}",
            retry_index,
            retry_limit,
            retried_phase,
            retried_position,
        )
        if retried_phase != GameplayPhase.UNKNOWN:
            if retry_override is not None:
                ctx.handler_state.pop("unknown_retry_override", None)
            logger.info(
                "unknown recovery: 被动复检后识别为 phase={}, position={}",
                retried_phase,
                retried_position,
            )
            return retried_phase, retried_position

    if retry_override is not None:
        logger.warning(
            "unknown retry override exhausted: reason={}, limit={}, sleep={}",
            retry_override.get("reason", "unknown"),
            retry_limit,
            retry_sleep,
        )
        ctx.handler_state.pop("unknown_retry_override", None)

    return phase, position


class ProduceGameplayLoopStep(ProduceStep):
    """驱动 gameplay 主循环，在各 phase handler 之间持续调度直到退出。"""

    step_name = "produce_gameplay_loop"

    @staticmethod
    def _flush_llm_session(ctx: "ProduceContext") -> None:
        """在主循环退出前尽量写回局内会话摘要。"""
        strategy = getattr(ctx, "schedule_strategy", None)
        if isinstance(strategy, LLMStrategy):
            strategy.flush_session(ctx)

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        logger.info("进入培育主循环")
        ctx.set_phase(GameplayPhase.SCHEDULE)
        ctx.last_pipeline_step = self.step_name

        dispatcher = build_default_dispatcher()
        max_unknown = 20         # 连续 unknown 阈值（触发安全退出）
        total_actions = 0        # 总操作次数

        while total_actions < ctx.max_gameplay_loops:
            # ── 培育收尾：切到 BASE_UI 检测主页 ──
            if ctx.handler_state.get("produce_finishing"):
                if _finish_produce_with_base_ui(app, ctx):
                    logger.success("培育收尾完成，已检测到主页，退出主循环")
                    return True
                raise RuntimeError(i18n_text("backend.task.produceFinishTimeout", fallback="培育收尾失败：推进结果链超时仍未回到主页"))

            loop_sleep = 0.8
            if _should_fast_poll_unknown_retry(ctx):
                loop_sleep = float(
                    ctx.handler_state.get("result_transition_loop_sleep", 0.2) or 0.2
                )
            sleep(loop_sleep)

            phase, position = _stabilize_unknown_state(app, ctx)
            ctx.set_phase(phase)
            ctx.set_position(position)
            logger.debug(f"[Loop {total_actions}] phase={phase}, position={position}")

            if phase in {
                GameplayPhase.SCHEDULE,
                GameplayPhase.CONSULT,
                GameplayPhase.P_DRINK,
                GameplayPhase.SKILL_REWARD,
                GameplayPhase.LESSON,
                GameplayPhase.EXAM,
            }:
                sync_visible_planning_context(
                    app,
                    ctx,
                    phase=phase,
                    position=position,
                    reason="gameplay_loop_visible_hud_sync",
                )

            if phase == GameplayPhase.LOADING:
                ctx.handler_state["unknown_retry_override"] = {
                    "reason": "loading_transition",
                    "retry_limit": int(
                        ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
                    ),
                    "retry_sleep": float(
                        ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
                    ),
                }
                logger.debug("loading: 检测到 NOW LOADING，等待过场结束")
                sleep(1.0)
                total_actions += 1
                continue

            if (
                phase == GameplayPhase.UNKNOWN
                and position == GameplayPosition.TRANSITION_RESUME_TITLE
            ):
                logger.info("recovery: 检测到游戏标题/启动页，点击中间继续恢复 gameplay")
                click_relative_point(
                    app,
                    x_ratio=0.5,
                    y_ratio=0.64,
                    label="resume-title-advance",
                )
                ctx.handler_state["unknown_retry_override"] = {
                    "reason": "resume_title_transition",
                    "retry_limit": int(
                        ctx.handler_state.get("loading_unknown_retry_limit", 15) or 15
                    ),
                    "retry_sleep": float(
                        ctx.handler_state.get("loading_unknown_retry_sleep", 1.0) or 1.0
                    ),
                }
                sleep(1.0)
                total_actions += 1
                continue

            if (
                phase == GameplayPhase.UNKNOWN
                and position in {
                    GameplayPosition.TRANSITION_EMPTY,
                    GameplayPosition.TRANSITION_HUD,
                }
            ):
                if _try_external_page_recovery(app, ctx):
                    if ctx.handler_state.pop("result_chain_completed", False):
                        logger.info("recovery: 结果链已正常结束，退出培育主循环")
                        return True
                    logger.info("recovery: 已从游戏外页面恢复到未完成培育，继续主循环")
                    sleep(1.0)
                    total_actions += 1
                    continue

                rechecked_phase, rechecked_position = _stabilize_unknown_state(app, ctx)
                if rechecked_phase != GameplayPhase.UNKNOWN:
                    logger.info(
                        "recovery: 外页复核期间画面已稳定为 phase={}, position={}，重新进入主循环分发",
                        rechecked_phase,
                        rechecked_position,
                    )
                    ctx.set_phase(rechecked_phase)
                    ctx.set_position(rechecked_position)
                    total_actions += 1
                    continue

            if (
                ctx.handler_state.get("pause_on_unknown")
                and phase == GameplayPhase.UNKNOWN
            ):
                raise RuntimeError(
                    f"培育主循环: 遇到未识别页面，已暂停等待分析 (phase={phase}, position={position})"
                )

            # 连续无法识别画面安全阈值
            if ctx.consecutive_unknowns >= max_unknown:
                logger.error(f"连续 {max_unknown} 次无法识别画面，安全退出循环")
                raise RuntimeError(i18n_text("backend.task.consecutiveUnknownsExceeded", fallback="培育主循环: 连续无法识别画面阈值超出"))

            # 分发到对应 handler
            result = dispatcher.dispatch(app, ctx, phase, position)

            if result.status == "exit":
                logger.info(f"主循环退出: {result.detail}")
                return True

            if result.status == "unhandled":
                logger.warning(f"无 handler 匹配: phase={phase}, position={position}")
                if ctx.handler_state.get("pause_on_unknown"):
                    raise RuntimeError(
                        i18n_text(
                            "backend.task.noHandlerPause",
                            fallback=f"培育主循环: 当前页面无可用 handler，已暂停等待分析 (phase={phase}, position={position})",
                            phase=phase,
                            position=position,
                        )
                    )
                # 优先尝试 Back Button 恢复（可能误入手牌库等子画面）
                if _try_back_button_recovery(app):
                    sleep(1.0)
                else:
                    click_relative_point(app, x_ratio=0.5, y_ratio=0.35, label="unhandled-advance")
                    sleep(1.0)
            elif result.sleep_after > 0:
                sleep(result.sleep_after)

            total_actions += 1

        raise RuntimeError(i18n_text("backend.task.maxGameplayLoopsExceeded", fallback=f"培育主循环: 达到最大循环次数 {ctx.max_gameplay_loops}", max_loops=ctx.max_gameplay_loops))
