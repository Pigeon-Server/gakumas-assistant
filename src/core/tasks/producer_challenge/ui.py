from __future__ import annotations

import re
from statistics import median
from time import sleep, time
from typing import TYPE_CHECKING, Sequence

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.entity.Game.Components.Button import Button, ButtonList
from src.core.tasks.producer_challenge.gameplay.common import (
    invoke_decision_strategy,
    ocr_text,
    resolve_candidate_index,
)
from src.utils.logger import logger
from src.utils.string_tools import MatchConfig, string_match

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.entity.Game.Components.Modal import Modal
    from src.main import AppProcessor

_PRESET_INDEX_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")


def get_buttons(app: "AppProcessor") -> ButtonList:
    """获取buttons并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        ButtonList: 返回值类型见注解，语义由函数用途决定。
    """
    return ButtonList(app.latest_results)


def parse_preset_index(text: str | None) -> tuple[int, int] | None:
    """从按钮文本中解析当前编组页签编号。

    Args:
        text: 按钮或标签上的原始文本，通常形如 `1 / 3`。

    Returns:
        tuple[int, int] | None: 解析成功时返回 `(当前编号, 总数)`；
        当文本为空或不符合编号格式时返回 None。
    """
    normalized = str(text or "").replace(" ", "")
    if not normalized:
        return None
    match = _PRESET_INDEX_PATTERN.search(normalized)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def get_current_preset_index(app: "AppProcessor") -> tuple[int, int] | None:
    """从当前页面按钮中读取正在显示的编组编号。

    Args:
        app: 当前应用处理器，用于访问最新一帧的按钮检测结果。

    Returns:
        tuple[int, int] | None: 识别到形如 `当前/总数` 的编号时返回解析结果；
        若页面上没有可识别的编号按钮则返回 None。
    """
    for button in get_buttons(app):
        if parsed := parse_preset_index(button.text):
            return parsed
    return None


def build_preset_swipe_paths(
    boxes: Sequence,
    *,
    frame_width: int,
) -> list[tuple[int, int, int, int]]:
    """根据卡片区域构建横向切换编组的滑动轨迹。

    Args:
        boxes: 当前页面中可作为编组区域锚点的检测框列表。
        frame_width: 当前画面宽度，用于限制滑动起止点不要超出屏幕边界。

    Returns:
        list[tuple[int, int, int, int]]: 若干条 `(start_x, start_y, end_x, end_y)` 滑动路径。
        每一行候选框会生成一条横滑轨迹，供预设切换时轮流尝试，降低单一路径失效概率。
    """
    if not boxes:
        return []

    left = int(min(box.x for box in boxes))
    right = int(max(box.w for box in boxes))
    span = max(1, right - left)
    margin = max(40, int(span * 0.15))
    start_x = min(frame_width - 40, right - margin)
    end_x = max(40, left + margin)
    if start_x - end_x < 160:
        start_x = max(end_x + 160, int(frame_width * 0.75))
        end_x = int(frame_width * 0.25)

    rows: list[list] = []
    current_row: list = []
    row_anchor_cy: int | None = None
    for box in sorted(boxes, key=lambda item: item.cy):
        if row_anchor_cy is None or abs(box.cy - row_anchor_cy) <= 120:
            if not current_row:
                row_anchor_cy = box.cy
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            row_anchor_cy = box.cy
    if current_row:
        rows.append(current_row)

    return [
        (start_x, int(round(median([box.cy for box in row]))), end_x, int(round(median([box.cy for box in row]))))
        for row in rows
    ]


def get_preset_swipe_paths(
    app: "AppProcessor",
    *,
    card_labels: Sequence[str],
) -> list[tuple[int, int, int, int]]:
    """提取当前页面可用于切换预设的滑动路径。"""
    boxes = list(app.latest_results.filter_by_labels(list(card_labels)))
    if not boxes:
        boxes = list(app.latest_results.filter_by_label(BaseUILabels.BLANK_SLOT))
    if not boxes:
        raise TimeoutError("未识别到可切换编组的卡片区域")

    frame_width = app.latest_frame.shape[1]
    paths = build_preset_swipe_paths(boxes, frame_width=frame_width)
    if not paths:
        raise TimeoutError("未能计算编组横滑路径")
    return paths


def select_preset_by_horizontal_swipe(
    app: "AppProcessor",
    target_index: int,
    *,
    card_labels: Sequence[str],
    description: str,
    max_swipes: int | None = None,
) -> bool:
    """通过横向滑动把当前页面切换到指定编组。

    Args:
        app: 当前应用处理器，用于执行滑动、读取按钮文本和等待页面稳定。
        target_index: 目标编组编号，从 1 开始计数。
        card_labels: 用来定位编组卡片区域的 YOLO 标签集合；若缺失会退回空槽标签。
        description: 当前页面说明文本，会写入日志和异常信息，便于定位是支援卡页还是记忆页。
        max_swipes: 允许尝试的最大滑动次数；不传时按当前编号与目标编号距离自动估算。

    Returns:
        bool: 成功切换到目标编组时返回 True。

    Raises:
        TimeoutError: 无法识别当前编号、找不到滑动区域，或在允许次数内仍未切换成功时抛出。
        ValueError: 目标编号超出当前页面可用编组总数时抛出。

    Notes:
        该函数会根据滑动后的编号变化自动推断“左滑是否让编号递增”，
        并在连续卡住时翻转方向假设，避免不同页面的滑动方向不一致导致死循环。
    """
    current_info = get_current_preset_index(app)
    if current_info is None:
        raise TimeoutError(f"{description}页面未识别到编组编号")

    current_index, total = current_info
    if target_index < 1 or target_index > total:
        raise ValueError(f"{description}预设编号超出范围: {target_index} (1-{total})")
    if current_index == target_index:
        logger.debug(f"{description}已在目标编组 {current_index}/{total}")
        return True

    left_swipe_increases = True
    stuck_attempts = 0
    swipe_limit = max_swipes or max(abs(target_index - current_index) * 2 + 4, 6)

    for attempt in range(1, swipe_limit + 1):
        paths = get_preset_swipe_paths(app, card_labels=card_labels)
        start_x, start_y, end_x, end_y = paths[(attempt - 1) % len(paths)]
        should_increase = target_index > current_index
        swipe_left = should_increase if left_swipe_increases else not should_increase
        if swipe_left:
            inertial_swipe(app, start_x, start_y, end_x, end_y, duration=0.35, settle_timeout=4.5)
        else:
            inertial_swipe(app, end_x, start_y, start_x, end_y, duration=0.35, settle_timeout=4.5)

        updated_info = get_current_preset_index(app)
        if updated_info is None:
            raise TimeoutError(f"{description}页面横滑后未识别到新的编组编号")

        updated_index, updated_total = updated_info
        total = updated_total
        if updated_index == target_index:
            logger.debug(f"{description}切换到目标编组 {updated_index}/{total}")
            return True

        if updated_index != current_index:
            if swipe_left:
                left_swipe_increases = updated_index > current_index
            else:
                left_swipe_increases = updated_index < current_index
            logger.debug(
                f"{description}横滑后编组变化: {current_index}/{total} -> {updated_index}/{total}, "
                f"left_swipe_increases={left_swipe_increases}"
            )
            current_index = updated_index
            stuck_attempts = 0
            continue

        stuck_attempts += 1
        logger.debug(
            f"{description}横滑后编组编号未变化: {current_index}/{total}, "
            f"attempt={attempt}/{swipe_limit}, path_index={(attempt - 1) % len(paths)}"
        )
        if stuck_attempts >= len(paths):
            left_swipe_increases = not left_swipe_increases
            stuck_attempts = 0

    raise TimeoutError(
        f"{description}未能切换到目标编组 {target_index}/{total}，当前仍为 {current_index}/{total}"
    )


def find_button(
    app: "AppProcessor",
    text: str,
    *,
    fuzz_threshold: float = 70,
    use_contains: bool = True,
) -> Button | None:
    """按文本在当前页面按钮列表中查找目标按钮。

    Args:
        app: 当前应用处理器，用于读取最新按钮检测结果。
        text: 需要匹配的按钮文本。
        fuzz_threshold: 模糊匹配阈值，数值越高越严格。
        use_contains: 是否允许使用包含关系辅助命中，适合 OCR 存在少量缺字的场景。

    Returns:
        Button | None: 命中时返回按钮对象，未找到时返回 None。
    """
    return get_buttons(app).get_button_by_text(
        text,
        match_config=MatchConfig(fuzz_threshold=fuzz_threshold, use_contains=use_contains, normalize=True),
    )


def has_button(
    app: "AppProcessor",
    text: str,
    *,
    fuzz_threshold: float = 70,
    use_contains: bool = True,
) -> bool:
    """判断当前页面是否存在指定文本按钮。"""
    return find_button(app, text, fuzz_threshold=fuzz_threshold, use_contains=use_contains) is not None


def wait_frame_stable(app: "AppProcessor", timeout: float = 4.0) -> None:
    """在关键操作后等待画面稳定，降低 OCR 和按钮判定抖动。"""
    app.game_utils.wait_frame_stable(
        threshold=0.985,
        stable_count=2,
        timeout=timeout,
    )


def inertial_swipe(
    app: "AppProcessor",
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    *,
    duration: float = 0.45,
    settle_timeout: float = 4.0,
    hold_end: float = 0.15,
    ease: str | None = "out_quad",
) -> None:
    """执行带惯性抑制的滑动，避免页面惯性过冲。"""
    app.device.swipe(
        start_x, start_y, end_x, end_y,
        duration=duration, offset_y=0,
        hold_end=hold_end, ease=ease,
    )
    sleep(0.1)
    wait_frame_stable(app, timeout=settle_timeout)


def is_final_confirm_page(app: "AppProcessor") -> bool:
    """判断当前是否处于最终确认开跑页面。

    Returns:
        bool: 同时满足“存在编成详情与开跑按钮”且页面上仍可见支援卡/记忆/道具区域时返回 True。
        该判定用于区分最终确认页与前面的记忆编成页、编成详情页等相似页面。
    """
    if has_button(app, ButtonText.AUTO_SELECT, fuzz_threshold=75):
        return False
    if has_button(app, ButtonText.NEXT, fuzz_threshold=75):
        return False
    if has_button(app, ButtonText.RESET, fuzz_threshold=75):
        return False

    has_detail_button = has_button(app, ProduceText.FORMATION_DETAILS, fuzz_threshold=68)
    has_start_button = has_button(app, ButtonText.PRODUCE_START, fuzz_threshold=65)
    has_context = any(
        app.latest_results.exists_label(label)
        for label in (
            BaseUILabels.SUPPORT_CARD,
            BaseUILabels.MEMORY_CARD,
            BaseUILabels.SPECIAL_ITEMS,
        )
    )
    return bool(has_detail_button and has_start_button and has_context)


def wait_for_final_confirm_page(app: "AppProcessor", timeout: float = 15.0) -> bool:
    """轮询等待页面进入最终确认开跑页。

    Args:
        app: 当前应用处理器。
        timeout: 最长等待时间，超时后返回 False。

    Returns:
        bool: 在超时前识别到最终确认页时返回 True，否则返回 False。
    """
    end_time = time() + timeout
    while time() < end_time:
        if is_final_confirm_page(app):
            wait_frame_stable(app, timeout=3.0)
            return True
        sleep(0.4)
    return False


def is_memory_selection_page(app: "AppProcessor") -> bool:
    """判断当前是否处于记忆编成页面。"""
    if has_button(app, ButtonText.PRODUCE_START, fuzz_threshold=65):
        return False
    if not has_button(app, ButtonText.NEXT, fuzz_threshold=75):
        return False
    if not has_button(app, ButtonText.AUTO_SELECT, fuzz_threshold=75):
        return False
    if not has_button(app, ButtonText.RESET, fuzz_threshold=75):
        return False
    if not has_button(app, ProduceText.FORMATION_DETAILS, fuzz_threshold=68):
        return False
    return bool(app.latest_results.exists_label(BaseUILabels.MEMORY_CARD))


def wait_for_memory_selection_page(app: "AppProcessor", timeout: float = 12.0) -> bool:
    """轮询等待页面进入记忆编成页，并在命中后额外等待画面稳定。"""
    end_time = time() + timeout
    while time() < end_time:
        if is_memory_selection_page(app):
            wait_frame_stable(app, timeout=3.0)
            return True
        sleep(0.4)
    return False


def click_modal_action_with_retry(
    app: "AppProcessor",
    modal: "Modal | None" = None,
    *,
    prefer_confirm: bool = True,
    retries: int = 3,
    timeout: float = 5.0,
    action_name: str = "modal action",
) -> bool:
    """在弹窗上重复尝试点击可执行按钮，直到弹窗消失或重试耗尽。

    Args:
        app: 当前应用处理器，用于读取弹窗和执行点击。
        modal: 已经识别出的弹窗对象；不传时函数会在每轮重试前自行重新识别。
        prefer_confirm: 为 True 时优先点击确认按钮，否则优先点击取消按钮。
        retries: 最大尝试次数。
        timeout: 每次点击后等待弹窗过渡完成的超时时间。
        action_name: 当前动作说明，会写入日志，帮助定位失败的是哪一种弹窗处理。

    Returns:
        bool: 成功点击并等待弹窗过渡完成时返回 True；没有弹窗可处理时也返回 True；
        弹窗缺少可点击按钮或多次点击后仍未发生过渡时返回 False。
    """
    current_modal = modal
    for attempt in range(1, retries + 1):
        if current_modal is None:
            current_modal = app.game_utils.try_get_modal(no_body=True)
        if current_modal is None:
            return True

        button = current_modal.confirm_button if prefer_confirm else current_modal.cancel_button
        if button is None:
            button = current_modal.cancel_button or current_modal.confirm_button
        if button is None:
            logger.warning(f"{action_name}: modal {current_modal.modal_title!r} has no actionable button")
            return False

        if app.game_utils.click_modal_button_and_wait_transition(
            button,
            previous_modal_title=current_modal.modal_title,
            timeout=timeout,
            interval=0.2,
        ):
            wait_frame_stable(app, timeout=min(timeout, 3.0))
            return True

        logger.warning(
            f"{action_name}: modal {current_modal.modal_title!r} did not transition "
            f"after attempt {attempt}/{retries}"
        )
        sleep(0.5)
        current_modal = app.game_utils.try_get_modal(no_body=True)

    return False


def click_top_right_action(app: "AppProcessor", *, timeout: float = 6.0) -> bool:
    """点击右上角最靠前的可操作按钮，常用于推进剧情或关闭提示。"""
    buttons = get_buttons(app)
    frame = app.latest_frame
    if frame is None or frame.size == 0:
        return False
    height, width = frame.shape[:2]
    cx_threshold = int(width * 0.66)
    cy_threshold = int(height * 0.12)
    candidates = [button for button in buttons if button.cx >= cx_threshold and button.cy <= cy_threshold]
    candidates.sort(key=lambda button: (button.cy, -button.cx))
    if not candidates:
        return False
    return app.game_utils.click_element_and_wait_trigger(candidates[0], timeout=timeout)


# ──────────────────────────────────────────────────────────
# Gameplay 阶段检测辅助
# ──────────────────────────────────────────────────────────

def classify_gameplay_phase(results, *, ctx: "ProduceContext | None" = None) -> str:
    """根据 YOLO 结果判定当前 gameplay 阶段。"""
    if results is None:
        return "unknown"

    has_action = results.exists_label(ProducerLabels.PC_ACTION)
    has_recommend = results.exists_label(ProducerLabels.PC_RECOMMEND_ACTION)
    has_skill_card = any(
        results.exists_label(label)
        for label in (
            ProducerLabels.SKILL_CARD_ACTIVE,
            ProducerLabels.SKILL_CARD_MENTAL,
            ProducerLabels.SKILL_CARD_TRAP,
        )
    )
    has_skill_card_info = results.exists_label(ProducerLabels.SKILL_CARD_INFO)
    has_training_score = results.exists_label(ProducerLabels.PC_TRAINING_SCORE)
    has_training_remaining = results.exists_label(ProducerLabels.PC_TRAINING_REMAINING)
    has_options = results.exists_label(ProducerLabels.UNIVERSAL_OPTIONS)
    has_p_drink = results.exists_label(ProducerLabels.P_DRINK)
    has_modal_header = results.exists_label(ProducerLabels.MODAL_HEADER)
    has_skip_button = results.exists_label(ProducerLabels.SKIP_BUTTON)
    has_fast_forward = results.exists_label(ProducerLabels.PLOT_FAST_FORWARD_BUTTON)
    has_progress = results.exists_label(ProducerLabels.PC_PROGRESS)
    has_button = results.exists_label(BaseUILabels.BUTTON)
    has_confirm = results.exists_label(ProducerLabels.CONFIRM_BUTTON)
    has_disable = results.exists_label(ProducerLabels.DISABLE_BUTTON)
    has_cancel = results.exists_label(ProducerLabels.CANCEL_BUTTON)
    has_card_exchange = results.exists_label(ProducerLabels.CARD_ITEM_EXCHANGE)
    has_enhancement = results.exists_label(ProducerLabels.PC_SKILL_CARD_ENHANCEMENT)
    has_remove = results.exists_label(ProducerLabels.PC_SKILL_CARD_REMOVE)
    has_bonus_indicator = results.exists_label(ProducerLabels.PC_BONUS_INDICATOR)
    has_pc_skip = results.exists_label(ProducerLabels.PC_SKIP) or results.exists_label(BaseUILabels.SKIP_BUTTON)
    last_consult_position = str(getattr(ctx, "last_stable_position", "") or "")

    if has_modal_header:
        return "modal"
    # 相談交换页：出现兑换卡、強化、削除任一元素
    if (has_card_exchange or has_enhancement or has_remove) and not has_training_score:
        return "consult"
    # 相談子流程（强化/删除预览页）与 skill_reward 外观相似，需要借助上一个稳定位置反解。
    if (
        ctx is not None
        and last_consult_position.startswith("consult")
        and has_skill_card
        and not has_action
        and not has_training_score
        and (has_button or has_confirm or has_disable or has_cancel)
    ):
        return "consult"
    if has_skill_card and (has_bonus_indicator or has_pc_skip) and not has_action and not has_training_score:
        return "exam"
    if has_skill_card and (has_training_score or has_training_remaining):
        return "lesson"
    if (has_skill_card or has_skill_card_info) and not has_action and not has_training_score:
        if has_button or has_confirm or has_disable:
            return "skill_reward"
    if has_options and not has_action:
        return "dialogue"
    if has_fast_forward and not has_action and not has_skill_card:
        return "dialogue"
    if has_action and has_progress:
        return "schedule"
    if has_recommend and has_progress and not has_options:
        return "schedule"
    if has_p_drink and not has_action and not has_skill_card:
        frame = getattr(results, "frame", None)
        if frame is not None:
            frame_height = frame.shape[0]
            p_drink_boxes = results.filter_by_label(ProducerLabels.P_DRINK)
            if any(box.cy < frame_height * 0.85 for box in p_drink_boxes):
                return "p_drink"
    if has_skip_button and not has_action and not has_skill_card:
        return "result"
    return "unknown"


def detect_gameplay_phase(app: "AppProcessor", ctx: "ProduceContext | None" = None) -> str:
    """从 App 中读取当前 gameplay 阶段。"""
    return classify_gameplay_phase(app.latest_results, ctx=ctx)


def classify_pipeline_position(
    results,
    *,
    modal_title: str | None = None,
    final_confirm: bool = False,
    ctx: "ProduceContext | None" = None,
    phase: str | None = None,
) -> str:
    """判定流水线、position并返回结果。

    Args:
        results: 用于提供results相关输入。
        modal_title: 用于提供弹窗、title相关输入。
        final_confirm: 用于提供final、confirm相关输入。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        phase: 已判定的 gameplay phase；为空时会重新识别。

    Returns:
        str: 处理后的文本结果。
    """
    if final_confirm:
        return "final_confirm"

    phase = phase or classify_gameplay_phase(results, ctx=ctx)
    if phase == "modal":
        modal_title = modal_title or ""
        if string_match(modal_title, ProduceText.VOICE_PLAYBACK_CONFIRM, MatchConfig(fuzz_threshold=65, normalize=True)):
            return "startup_modal_voice"
        if string_match(modal_title, ProduceText.COMMU_FAST_FORWARD, MatchConfig(fuzz_threshold=65, normalize=True)):
            return "startup_modal_fast_forward"
        if string_match(modal_title, ProduceText.PRODUCE_SKIP_SETTINGS, MatchConfig(fuzz_threshold=65, normalize=True)):
            return "startup_modal_skip_settings"
        if ProduceText.P_DRINK in modal_title and ProduceText.DETAIL in modal_title:
            return "p_drink_detail"
        if ProduceText.DETAIL in modal_title:
            return "detail_modal"
        last_position = str(getattr(ctx, "last_stable_position", "") or "")
        if last_position.startswith("consult_enhancement"):
            return "consult_enhancement_confirm_modal"
        return "gameplay_modal"

    if results is None:
        return "unknown"

    if phase == "schedule":
        if results.exists_label(ProducerLabels.PC_ACTION_INFO):
            return "schedule_selected"
        if results.exists_label(ProducerLabels.PC_RECOMMEND_ACTION):
            return "schedule_recommend"
        return "schedule_idle"

    if phase == "lesson":
        if results.exists_label(ProducerLabels.SKILL_CARD_INFO):
            return "lesson_selected"
        return "lesson_idle"

    if phase == "dialogue":
        if results.exists_label(ProducerLabels.UNIVERSAL_OPTIONS):
            return "dialogue_options"
        return "dialogue_continue"

    if phase == "p_drink":
        if results.exists_label(ProducerLabels.CONFIRM_BUTTON):
            return "p_drink_selected"
        if results.exists_label(ProducerLabels.DISABLE_BUTTON):
            return "p_drink_idle"
        return "p_drink_idle"

    if phase == "skill_reward":
        if results.exists_label(ProducerLabels.CONFIRM_BUTTON):
            return "skill_reward_selected"
        if results.exists_label(ProducerLabels.DISABLE_BUTTON):
            return "skill_reward_idle"
        if results.exists_label(BaseUILabels.BUTTON):
            return "skill_reward_selected"
        return "skill_reward_idle"

    if phase == "consult":
        has_enhance = results.exists_label(ProducerLabels.PC_SKILL_CARD_ENHANCEMENT)
        has_remove = results.exists_label(ProducerLabels.PC_SKILL_CARD_REMOVE)
        has_exchange = results.exists_label(ProducerLabels.CARD_ITEM_EXCHANGE)
        has_consult_cards = any(
            results.exists_label(label)
            for label in (
                ProducerLabels.SKILL_CARD_ACTIVE,
                ProducerLabels.SKILL_CARD_MENTAL,
                ProducerLabels.SKILL_CARD_TRAP,
            )
        )
        if has_enhance or has_remove or has_exchange:
            return "consult_exchange"
        if has_consult_cards:
            if results.exists_label(ProducerLabels.CONFIRM_BUTTON) or results.exists_label(BaseUILabels.BUTTON):
                return "consult_enhancement_ready"
            return "consult_enhancement_preview"
        return "consult_idle"

    if phase == "exam":
        if results.exists_label(ProducerLabels.SKILL_CARD_INFO):
            return "exam_selected"
        return "exam_idle"

    if phase == "result":
        return "result"

    if not results:
        return "transition_empty"

    has_hud = any(
        results.exists_label(label)
        for label in (
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_TRAINING_SCORE,
            ProducerLabels.PC_TRAINING_REMAINING,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.PC_TARGET,
        )
    )
    if has_hud:
        return "transition_hud"
    return phase


def get_pipeline_position(app: "AppProcessor", ctx: "ProduceContext | None" = None) -> str:
    """获取流水线、position并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        str: 处理后的文本结果。
    """
    modal_title: str | None = None
    results = app.latest_results
    if results and results.exists_label(ProducerLabels.MODAL_HEADER):
        modal = app.game_utils.try_get_modal(no_body=True)
        if modal is not None:
            modal_title = modal.modal_title
    return classify_pipeline_position(
        results,
        modal_title=modal_title,
        final_confirm=is_final_confirm_page(app),
        ctx=ctx,
    )


def click_recommend_action(app: "AppProcessor", ctx: "ProduceContext | None" = None) -> str | None:
    """点击recommend、操作并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        str | None: 返回值类型见注解，语义由函数用途决定。
    """
    if ctx is None:
        return None
    from src.core.tasks.producer_challenge.gameplay.schedule import execute_schedule_step

    result = execute_schedule_step(app, ctx, position=get_pipeline_position(app))
    return result.status if result else None


def handle_skill_card_selection(app: "AppProcessor", ctx: "ProduceContext | None" = None) -> str | None:
    """处理handle、skill、卡牌、selection并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        str | None: 返回值类型见注解。
    """
    if ctx is None:
        return None
    from src.core.tasks.producer_challenge.gameplay.lesson import execute_lesson_step

    result = execute_lesson_step(app, ctx, position=get_pipeline_position(app))
    return result.status if result else None


def _click_preferred_confirmation(app: "AppProcessor") -> bool:
    """点击preferred、confirmation并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        bool: 条件判断结果，True 表示满足。
    """
    confirm_boxes = app.latest_results.filter_by_label(ProducerLabels.CONFIRM_BUTTON)
    if confirm_boxes:
        app.device.click_element(confirm_boxes.first())
        return True

    buttons = app.latest_results.filter_by_label(BaseUILabels.BUTTON)
    if buttons:
        app.device.click_element(max(buttons, key=lambda button: button.cy))
        return True
    return False


def handle_p_drink_select(app: "AppProcessor", ctx: "ProduceContext | None" = None) -> str | None:
    """处理handle、p、饮料、select并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        str | None: 返回值类型见注解。
    """
    position = get_pipeline_position(app)
    if position not in {"p_drink_idle", "p_drink_selected"}:
        return None

    if position == "p_drink_selected":
        if not _click_preferred_confirmation(app):
            return None
        if ctx is not None:
            ctx.record_operation(
                "confirm_p_drink",
                target=ctx.pending_p_drink_label or "p_drink",
                details={"index": ctx.pending_p_drink_index},
            )
            ctx.clear_p_drink_pending()
        sleep(1.0)
        return "confirmed"

    drinks = sorted(app.latest_results.filter_by_label(ProducerLabels.P_DRINK), key=lambda item: item.cx)
    if not drinks:
        return None

    target_index = 0
    if ctx is not None:
        decision = invoke_decision_strategy(ctx.p_drink_strategy, app, ctx, drinks)
        target_index = resolve_candidate_index(decision, drinks, default_index=ctx.pending_p_drink_index or 0)

    target = drinks[target_index]
    app.device.click_element(target)
    if ctx is not None:
        ctx.pending_p_drink_index = target_index
        ctx.pending_p_drink_label = ocr_text(target.frame) or f"p_drink_{target_index + 1}"
        ctx.record_operation(
            "select_p_drink",
            target=ctx.pending_p_drink_label,
            details={"index": target_index},
        )
    sleep(0.8)
    return "selected"


def handle_skill_reward_selection(app: "AppProcessor", ctx: "ProduceContext | None" = None) -> str | None:
    """处理handle、skill、奖励、selection并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        str | None: 返回值类型见注解。
    """
    position = get_pipeline_position(app)
    if position not in {"skill_reward_idle", "skill_reward_selected"}:
        return None

    if position == "skill_reward_selected":
        if not _click_preferred_confirmation(app):
            return None
        if ctx is not None:
            ctx.record_operation(
                "confirm_skill_reward",
                target=ctx.pending_skill_reward_label or "skill_reward",
                details={"index": ctx.pending_skill_reward_index},
            )
            ctx.clear_skill_reward_pending()
        sleep(1.0)
        return "confirmed"

    candidates = []
    for label in (
        ProducerLabels.SKILL_CARD_ACTIVE,
        ProducerLabels.SKILL_CARD_MENTAL,
        ProducerLabels.SKILL_CARD_TRAP,
        ProducerLabels.SKILL_CARD_INFO,
    ):
        candidates.extend(app.latest_results.filter_by_label(label))
    candidates = sorted(candidates, key=lambda item: item.cx)
    if not candidates:
        return None

    target_index = 0
    if ctx is not None:
        decision = invoke_decision_strategy(ctx.skill_reward_strategy, app, ctx, candidates)
        target_index = resolve_candidate_index(decision, candidates, default_index=ctx.pending_skill_reward_index or 0)

    target = candidates[target_index]
    app.device.click_element(target)
    if ctx is not None:
        ctx.pending_skill_reward_index = target_index
        ctx.pending_skill_reward_label = ocr_text(target.frame) or f"skill_reward_{target_index + 1}"
        ctx.record_operation(
            "select_skill_reward",
            target=ctx.pending_skill_reward_label,
            details={"index": target_index},
        )
    sleep(0.8)
    return "selected"


def go_back_in_gameplay(app: "AppProcessor") -> bool:
    """在 gameplay 中尝试返回上一层。"""
    position = get_pipeline_position(app)
    if position in {
        "startup_modal_voice",
        "startup_modal_fast_forward",
        "startup_modal_skip_settings",
        "gameplay_modal",
        "p_drink_detail",
        "detail_modal",
    }:
        modal = app.game_utils.try_get_modal(no_body=True)
        if modal is not None:
            prefer_confirm = position not in {"p_drink_detail", "detail_modal"}
            return click_modal_action_with_retry(
                app,
                modal,
                prefer_confirm=prefer_confirm,
                retries=2,
                timeout=4.0,
                action_name=f"go_back_in_gameplay[{position}]",
            )

    if close_buttons := app.latest_results.filter_by_label(BaseUILabels.CLOSE_BUTTON):
        return app.game_utils.click_element_and_wait_trigger(close_buttons.first(), timeout=3.0)
    if back_buttons := app.latest_results.filter_by_label(BaseUILabels.BACK_BTN):
        return app.game_utils.click_element_and_wait_trigger(back_buttons.first(), timeout=3.0)

    close_button = find_button(app, ButtonText.CLOSE, fuzz_threshold=60)
    if close_button is not None:
        return app.game_utils.click_element_and_wait_trigger(close_button, timeout=3.0)

    cancel_button = (
        find_button(app, ButtonText.CLOSE, fuzz_threshold=60)
        or find_button(app, ButtonText.CANCEL, fuzz_threshold=60)
    )
    if cancel_button is not None:
        return app.game_utils.click_element_and_wait_trigger(cancel_button, timeout=3.0)

    return False


def go_home_from_gameplay(app: "AppProcessor", *, max_try: int = 4) -> bool:
    """尽量从 gameplay 画面返回主页。"""
    for _ in range(max_try):
        if app.latest_results.exists_label(BaseUILabels.TAB_HOME):
            return True

        if home_buttons := app.latest_results.filter_by_label(BaseUILabels.GO_HOME_BTN):
            if app.game_utils.click_element_and_wait_trigger(home_buttons.first(), timeout=3.0):
                sleep(1.0)
                continue

        text_button = (
            find_button(app, ButtonText.SAVE_AND_SUSPEND, fuzz_threshold=60)
            or find_button(app, ButtonText.HOME, fuzz_threshold=60)
            or find_button(app, ButtonText.RETIRE, fuzz_threshold=60)
        )
        if text_button is not None:
            if app.game_utils.click_element_and_wait_trigger(text_button, timeout=3.0):
                sleep(1.0)
                continue

        if go_back_in_gameplay(app):
            sleep(0.8)
            continue

        if click_top_right_action(app, timeout=2.0):
            sleep(1.0)
            continue

        break
    return app.latest_results.exists_label(BaseUILabels.TAB_HOME)
