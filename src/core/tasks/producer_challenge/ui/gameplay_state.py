from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Sequence

from src.constants.game.producer_gameplay import (
    CONSULT_ENHANCEMENT_POSITION_PREFIX,
    CONSULT_POSITION_PREFIX,
    GameplayPhase,
    GameplayPosition,
    P_DRINK_SELECTION_POSITIONS,
)
from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.shared.common import (
    normalize_text,
    ocr_text,
)
from src.utils.debug_tools import DebugTools
from src.utils.string_tools import MatchConfig, string_match

from .common import is_final_confirm_page

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_DIALOGUE_TEXT_CHAR_RE = re.compile(r"[ぁ-んァ-ヶ一-龯]")
_SKILL_REWARD_LAYOUT_DUP_TOLERANCE = 56


def _call_ui_attr(name: str, fallback, *args, **kwargs):
    """动态调用 UI 模块中的函数，支持运行时替换实现。

    优先从 `src.core.tasks.producer_challenge.ui` 模块中按名称获取函数，
    如果不存在或模块未加载则回退到 fallback。该机制主要用于调试时替换
    OCR 等函数而不修改调用方代码。

    Args:
        name: 要在 UI 模块中查找的函数名。
        fallback: 默认实现函数，当 UI 模块中找不到 name 时使用。
        *args: 位置参数，透传给目标函数。
        **kwargs: 关键字参数，透传给目标函数。

    Returns:
        目标函数的返回值，类型取决于被调用的函数。
    """
    ui_module = sys.modules.get("src.core.tasks.producer_challenge.ui")
    if ui_module is not None:
        candidate = getattr(ui_module, name, fallback)
        if candidate is not fallback:
            return candidate(*args, **kwargs)
    return fallback(*args, **kwargs)


def _button_like_boxes(results) -> list:
    """从 YOLO 检测结果中提取所有按钮类检测框（去重后）。

    聚合 BUTTON、CONFIRM_BUTTON、CANCEL_BUTTON、PLOT_FAST_FORWARD_BUTTON、SKIP_BUTTON
    五种标签的检测框，基于坐标+标签进行去重。

    Args:
        results: YOLO 检测结果对象，提供 filter_by_label 方法。为 None 时返回空列表。

    Returns:
        list: 去重后的检测框对象列表。
    """
    if results is None:
        return []
    boxes = []
    for label in (
        BaseUILabels.BUTTON,
        ProducerLabels.CONFIRM_BUTTON,
        ProducerLabels.CANCEL_BUTTON,
        BaseUILabels.PLOT_FAST_FORWARD_BUTTON,
        BaseUILabels.SKIP_BUTTON,
    ):
        boxes.extend(list(results.filter_by_label(label)))
    deduped: list = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for box in boxes:
        key = (int(box.x), int(box.y), int(box.w), int(box.h), str(box.label))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


def collect_button_like_texts(results) -> list[str]:
    """对画面中所有按钮类检测框进行 OCR，收集识别出的文本。

    遍历 _button_like_boxes 返回的检测框，对每个框的 frame 做 OCR 识别，
    同时通过 DebugTools 添加可视化标注框以便调试。

    Args:
        results: YOLO 检测结果对象，提供 filter_by_label 方法。

    Returns:
        list[str]: 各按钮框 OCR 识别出的非空文本列表。
    """
    texts: list[str] = []
    debugger = DebugTools()
    for box in _button_like_boxes(results):
        text = _call_ui_attr("ocr_text", ocr_text, getattr(box, "frame", None))
        if text:
            texts.append(text)
            debugger.add_box(
                int(box.x),
                int(box.y),
                int(box.w),
                int(box.h),
                label=f"button:{text[:20]}",
                color=(0, 200, 255),
                alpha=0.12,
                duration=2.5,
                font_size=18,
            )
    return texts


def collect_frame_text(results) -> str:
    """对整帧画面进行 OCR，返回识别出的全部文本。

    Args:
        results: YOLO 检测结果对象，其 frame 属性为待识别的图像。为 None 时返回空字符串。

    Returns:
        str: 整帧 OCR 识别结果，识别失败时返回空字符串。
    """
    if results is None:
        return ""
    return _call_ui_attr("ocr_text", ocr_text, getattr(results, "frame", None))


def _contains_text(text: str, *tokens: str) -> bool:
    """判断文本中是否包含任意一个目标 token（归一化后比较）。

    将输入文本和所有 token 都通过 normalize_text 归一化（全角转半角、去空白等），
    然后检查是否有任一 token 是归一化后文本的子串。

    Args:
        text: 待检查的源文本，通常来自 OCR 结果。
        *tokens: 一个或多个待匹配的子串 token。

    Returns:
        bool: 包含任一 token 返回 True，文本为空或全部不匹配返回 False。
    """
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(normalize_text(token) in normalized for token in tokens if token)


def _button_text_matches(button_texts: list[str], *tokens: str) -> bool:
    """判断按钮文本列表中是否有任一文本包含任意一个目标 token。

    对 button_texts 中的每个文本调用 _contains_text，只要有一个文本命中任一 token
    即返回 True。

    Args:
        button_texts: 按钮 OCR 文本列表。
        *tokens: 一个或多个待匹配的子串 token。

    Returns:
        bool: 有匹配返回 True，全部不匹配返回 False。
    """
    return any(_contains_text(text, *tokens) for text in button_texts)


def _looks_like_dialogue_text(frame_text: str) -> bool:
    """判断 OCR 文本是否像对话正文（而非 UI 元素或按钮文本）。

    通过统计日文字符（平假名/片假名/汉字）数量并检查是否包含对话常见结尾
    标记（です/ます/。/！等）来判断。字符数 >= 8 且含对话标记，或字符数 >= 14
    时判定为对话文本。

    Args:
        frame_text: 整帧 OCR 识别出的文本。

    Returns:
        bool: 判定为对话正文返回 True，否则返回 False。
    """
    normalized = normalize_text(frame_text)
    if not normalized:
        return False
    char_count = len(_DIALOGUE_TEXT_CHAR_RE.findall(normalized))
    if char_count < 8:
        return False
    if any(token in normalized for token in ("。", "！", "!", "？", "?", "です", "ます", "いただ", "もちろん")):
        return True
    return char_count >= 14


def _looks_like_present_support_selection(frame_text: str, results) -> bool:
    """判断当前页面是否为「活動支給」支援卡选择页。

    综合 OCR 文本和 YOLO 标签进行判断：
    - OCR 包含「活動支給」文本
    - OCR 包含选择提示（「選択」等）
    - OCR 包含 3 个以上的 "+数字" 加成数值
    - YOLO 检测到至少 2 个参数标签（VOCAL/DANCE/VISUAL）
    - OCR 包含审查条件（「審査基準」）

    Args:
        frame_text: 整帧 OCR 识别出的文本。
        results: YOLO 检测结果对象，用于检查参数标签是否存在。

    Returns:
        bool: 满足所有条件返回 True，否则返回 False。
    """
    normalized = normalize_text(frame_text)
    if not normalized:
        return False
    has_present = bool(
        string_match(
            normalized,
            ProduceText.PRESENT_SUPPORT,
            MatchConfig(fuzz_threshold=60, normalize=True),
        )
    )
    has_selection_hint = (
        _contains_text(normalized, ProduceText.PRESENT_SELECTION)
        or ProduceText.PRESENT_SELECTION_SHORT in normalized
    )
    bonus_count = len(re.findall(r"\+\d+", normalized))
    has_param_panel = sum(
        bool(results.exists_label(label))
        for label in (
            ProducerLabels.PARAM_VOCAL,
            ProducerLabels.PARAM_DANCE,
            ProducerLabels.PARAM_VISUAL,
        )
    ) >= 2
    has_exam_criteria = bool(
        string_match(
            normalized,
            ProduceText.EXAM_CRITERIA,
            MatchConfig(fuzz_threshold=65, normalize=True),
        )
    )
    return has_present and has_selection_hint and has_param_panel and has_exam_criteria and bonus_count >= 3


def _looks_like_present_support_showcase(frame_text: str, results) -> bool:
    """识别活動支給奖励链中的资源箱 / 展示页。

    这类页面的共同点是：
    - 顶部仍保留「活動支給」和 Progress HUD
    - 中间是资源箱 / 奖励展示，没有可选行动按钮
    - 点画面上方安全区域即可继续推进到下一层奖励或对话
    """
    normalized = normalize_text(frame_text)
    if not normalized:
        return False
    if _looks_like_present_support_selection(frame_text, results):
        return False

    header_text = ""
    frame = results.frame
    if frame is not None and frame.size > 0:
        height, width = frame.shape[:2]
        header_crop = frame[:int(height * 0.12), :int(width * 0.32)]
        header_text = normalize_text(
            _call_ui_attr("ocr_text", ocr_text, header_crop)
        )

    has_present = bool(
        string_match(
            header_text or normalized,
            ProduceText.PRESENT_SUPPORT,
            MatchConfig(
                fuzz_threshold=40 if header_text else 60,
                normalize=True,
            ),
        )
    )
    has_action_like = any(
        results.exists_label(label)
        for label in (
            ProducerLabels.PC_ACTION,
            ProducerLabels.PC_RECOMMEND_ACTION,
            ProducerLabels.UNIVERSAL_OPTIONS,
            ProducerLabels.PLOT_FAST_FORWARD_BUTTON,
            BaseUILabels.PLOT_FAST_FORWARD_BUTTON,
        )
    )
    has_reward_controls = any(
        results.exists_label(label)
        for label in (
            ProducerLabels.CONFIRM_BUTTON,
            ProducerLabels.DISABLE_BUTTON,
            ProducerLabels.CANCEL_BUTTON,
            BaseUILabels.BUTTON,
        )
    )
    has_card_like = any(
        results.exists_label(label)
        for label in (
            ProducerLabels.SKILL_CARD_ACTIVE,
            ProducerLabels.SKILL_CARD_MENTAL,
            ProducerLabels.SKILL_CARD_TRAP,
            ProducerLabels.SKILL_CARD_INFO,
        )
    )
    has_modal_header = results.exists_label(ProducerLabels.MODAL_HEADER)
    return (
        has_present
        and not has_action_like
        and not has_reward_controls
        and not has_card_like
        and not has_modal_header
    )


def _looks_like_loading_screen(results) -> bool:
    """判断当前画面是否为 loading 加载页面。

    裁剪画面右下角区域（78%-100% 高度，60%-100% 宽度）进行 OCR，
    检查是否包含 LOADING_TOKENS 中的加载相关文本（如 "LOADING"）。

    Args:
        results: YOLO 检测结果对象，其 frame 属性为待识别的图像。

    Returns:
        bool: 判定为 loading 页面返回 True，否则返回 False。
    """
    frame = results.frame
    if frame is None or not hasattr(frame, "shape"):
        return False
    height, width = frame.shape[:2]
    crop = frame[int(height * 0.78):height, int(width * 0.60):width]
    if crop.size <= 0:
        return False
    text = _call_ui_attr("ocr_text", ocr_text, crop).upper().replace(" ", "")
    return any(token in text for token in ProduceText.LOADING_TOKENS)


def _looks_like_exam_prep_screen(results) -> bool:
    """检测考试准备页面（参数加成倍率预览）。

    该页面 YOLO 无法检测（0 个标签），需要纯 OCR 判定：
      - 审查条件区 (55%~64% 高度) 包含「審査基準」或「合格条件」
      - 底部提示区 (83%~90% 高度) 包含「タップして次へ」
    两项至少满足一项即判定为考试准备页面。
    """
    frame = results.frame
    if frame is None or not hasattr(frame, "shape"):
        return False
    h, w = frame.shape[:2]

    from src.utils.string_tools import fullwidth_to_halfwidth

    # 审查条件区 OCR
    criteria_crop = frame[int(h * 0.55):int(h * 0.64), :]
    criteria_text = fullwidth_to_halfwidth(
        _call_ui_attr("ocr_text", ocr_text, criteria_crop)
    )
    if ProduceText.EXAM_CRITERIA in criteria_text or ProduceText.PASS_CONDITION in criteria_text:
        return True

    # 底部提示区 OCR 备选
    tap_crop = frame[int(h * 0.83):int(h * 0.90), :]
    tap_text = fullwidth_to_halfwidth(
        _call_ui_attr("ocr_text", ocr_text, tap_crop)
    )
    return ProduceText.TAP_TO_CONTINUE in tap_text


def _has_center_p_drink_boxes(results) -> bool:
    """判断画面中是否存在位于中上区域的 P 饮料检测框。

    筛选 P_DRINK 标签的检测框，检查是否有任一框的中心 Y 坐标小于画面高度
    的 85%。用于区分行程栏底部的 P 饮料图标和 P 饮料选择弹窗中的选项。

    Args:
        results: YOLO 检测结果对象，提供 filter_by_label 方法。

    Returns:
        bool: 存在中上区域的 P 饮料框返回 True，否则返回 False。
    """
    if results is None:
        return False
    frame = results.frame
    if frame is None or not hasattr(frame, "shape"):
        return False
    frame_height = frame.shape[0]
    p_drink_boxes = results.filter_by_label(ProducerLabels.P_DRINK)
    return any(box.cy < frame_height * 0.85 for box in p_drink_boxes)


def _looks_like_p_drink_receive_confirmation(
    frame_text: str,
    *,
    ctx: "ProduceContext | None" = None,
) -> bool:
    """判断当前界面是否为 P 饮料领取确认页。

    要求上一个稳定位置是 P 饮料选择阶段（P_DRINK_SELECTION_POSITIONS），
    且 OCR 文本中包含「受取」关键字。

    Args:
        frame_text: 整帧 OCR 识别出的文本。
        ctx: 培育上下文对象，用于获取 last_stable_position 判断阶段连续性。

    Returns:
        bool: 判定为 P 饮料领取确认页返回 True，否则返回 False。
    """
    normalized = normalize_text(frame_text)
    if not normalized:
        return False

    last_position = str(getattr(ctx, "last_stable_position", "") or "")
    if last_position not in P_DRINK_SELECTION_POSITIONS:
        return False

    return normalize_text(ProduceText.RECEIVE) in normalized


def _looks_like_resume_title_screen(frame_text: str, results) -> bool:
    """识别意外回到游戏标题 / 启动画面的场景。

    实机上偶发会在剧情/弹窗切换后短暂落到标题 Logo 画面，需要先点一次中间
    才会继续进入 loading；这类页面不属于正常 gameplay，但也不是需要人工分析
    的新页面，因此单独标出来交给主循环做恢复。
    """
    normalized = normalize_text(frame_text)
    if not normalized:
        return False

    title_text = normalized
    frame = results.frame
    if frame is not None and frame.size > 0:
        height, width = frame.shape[:2]
        logo_crop = frame[int(height * 0.40):int(height * 0.63), int(width * 0.05):int(width * 0.95)]
        logo_text = normalize_text(
            _call_ui_attr("ocr_text", ocr_text, logo_crop)
        )
        if logo_text:
            title_text = logo_text

    return bool(
        string_match(
            title_text,
            ProduceText.GAME_TITLE,
            MatchConfig(fuzz_threshold=60, normalize=True),
        )
    )


def _looks_like_lesson_summary_showcase(results) -> bool:
    """识别 lesson 结束后的参数上升展示页。

    这类页面的稳定特征是：
    - 只剩下 `Action Info` 说明框；
    - 同时出现 Vocal / Dance / Visual 参数标签；
    - 没有手牌、Skip、进度 HUD、按钮等常规 gameplay 控件。

    历史采集样本：
    - `tests/produce_gameplay_captures/lesson_stage_done_or_fail/...`
    - `out/debug_captures/capture_1775680213.png`
    """
    if results is None:
        return False

    has_action_info = results.exists_label(ProducerLabels.PC_ACTION_INFO)
    param_label_count = sum(
        1 for label in (
            ProducerLabels.PARAM_VOCAL,
            ProducerLabels.PARAM_DANCE,
            ProducerLabels.PARAM_VISUAL,
        )
        if results.exists_label(label)
    )
    has_other_gameplay_controls = any(
        results.exists_label(label)
        for label in (
            ProducerLabels.PC_ACTION,
            ProducerLabels.PC_RECOMMEND_ACTION,
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.PC_TRAINING_SCORE,
            ProducerLabels.PC_TRAINING_REMAINING,
            ProducerLabels.PC_SKIP,
            ProducerLabels.PC_BONUS_INDICATOR,
            ProducerLabels.SKILL_CARD_ACTIVE,
            ProducerLabels.SKILL_CARD_MENTAL,
            ProducerLabels.SKILL_CARD_TRAP,
            ProducerLabels.SKILL_CARD_INFO,
            ProducerLabels.UNIVERSAL_OPTIONS,
            ProducerLabels.CONFIRM_BUTTON,
            ProducerLabels.DISABLE_BUTTON,
            ProducerLabels.CANCEL_BUTTON,
            ProducerLabels.P_DRINK,
            BaseUILabels.BUTTON,
            BaseUILabels.CURRENT_LOCATION,
            BaseUILabels.SKIP_BUTTON,
        )
    )
    return bool(
        has_action_info
        and param_label_count >= 2
        and not has_other_gameplay_controls
    )


def _looks_like_skill_reward_showcase(results, frame_text: str) -> bool:
    """识别带 HUD 的单卡强化 / 获得展示页。

    这类页面常见于支援事件或奖励结算后，稳定特征是：
    - 顶部仍保留 progress / stamina / P 点等 gameplay HUD；
    - 中间只展示 1 张技能卡；
    - 底部 Action Info 会读到「…を強化しました / 獲得しました」这类结算句子；
    - 没有确认按钮、行动按钮和选项，不属于真正的选卡页。
    """
    if results is None:
        return False

    def _collect_reward_layout_boxes(labels: tuple[str, ...]) -> list[tuple[str, object]]:
        """按指定标签收集检测框，返回 (label, box) 元组列表。

        Args:
            labels: 要查询的 YOLO 标签名称元组。

        Returns:
            list[tuple[str, object]]: 每个匹配检测框与其标签的元组列表。
        """
        collected: list[tuple[str, object]] = []
        for label in labels:
            for box in results.filter_by_label(label):
                collected.append((label, box))
        return collected

    def _box_area(box) -> int:
        """计算检测框的面积（宽高乘积）。

        Args:
            box: 检测框对象，需具有 x, y, w, h 属性。

        Returns:
            int: 检测框面积，宽高为负时返回 0。
        """
        return max(0, int(getattr(box, "w", 0) - getattr(box, "x", 0))) * max(
            0, int(getattr(box, "h", 0) - getattr(box, "y", 0))
        )

    def _dedup_reward_layout_boxes(boxes: list[tuple[str, object]]) -> list[tuple[str, object]]:
        """对技能奖励检测框进行去重，保留优先级更高的框。

        以中心点坐标为基准，在容差范围（56px）内的框视为重复。去重策略：
        - 非 INFO 标签优先替换 INFO 标签（具体类型优先于通用类型）
        - 同类标签时面积更大的优先保留

        Args:
            boxes: (label, box) 元组列表。

        Returns:
            list[tuple[str, object]]: 去重后的元组列表。
        """
        deduped: list[tuple[str, object]] = []
        for label, box in boxes:
            cx = int(getattr(box, "cx", 0))
            cy = int(getattr(box, "cy", 0))
            replaced = False
            for idx, (kept_label, kept_box) in enumerate(deduped):
                kept_cx = int(getattr(kept_box, "cx", 0))
                kept_cy = int(getattr(kept_box, "cy", 0))
                if (
                    abs(cx - kept_cx) <= _SKILL_REWARD_LAYOUT_DUP_TOLERANCE
                    and abs(cy - kept_cy) <= _SKILL_REWARD_LAYOUT_DUP_TOLERANCE
                ):
                    kept_is_info = kept_label == ProducerLabels.SKILL_CARD_INFO
                    current_is_info = label == ProducerLabels.SKILL_CARD_INFO
                    if kept_is_info and not current_is_info:
                        deduped[idx] = (label, box)
                    elif kept_is_info == current_is_info and _box_area(box) > _box_area(kept_box):
                        deduped[idx] = (label, box)
                    replaced = True
                    break
            if not replaced:
                deduped.append((label, box))
        return deduped

    primary_card_boxes = _dedup_reward_layout_boxes(
        _collect_reward_layout_boxes(
            (
                ProducerLabels.SKILL_CARD_ACTIVE,
                ProducerLabels.SKILL_CARD_MENTAL,
                ProducerLabels.SKILL_CARD_TRAP,
            )
        )
    )
    info_boxes = _dedup_reward_layout_boxes(
        _collect_reward_layout_boxes((ProducerLabels.SKILL_CARD_INFO,))
    )
    primary_card_count = len(primary_card_boxes)
    frame = results.frame
    if frame is None:
        return False
    frame_h = int(frame.shape[0])
    frame_w = int(frame.shape[1])
    single_card_center_y = (
        int(getattr(primary_card_boxes[0][1], "cy", frame_h))
        if primary_card_count == 1 and primary_card_boxes
        else frame_h
    )
    single_card_is_high = single_card_center_y <= int(frame_h * 0.70)
    has_info_panel_below = False
    if primary_card_count == 1 and info_boxes:
        single_card_box = primary_card_boxes[0][1]
        single_card_cy = int(getattr(single_card_box, "cy", 0))
        info_below = [
            info_box
            for _, info_box in info_boxes
            if int(getattr(info_box, "cy", 0)) >= single_card_cy + int(frame_h * 0.08)
        ]
        has_info_panel_below = bool(info_below)
        if has_info_panel_below:
            # 识别到「上方单卡 + 下方信息面板」时打调试框，便于追踪该分支是否触发。
            chosen_info = info_below[0]
            DebugTools().add_box(
                int(getattr(single_card_box, "x", 0)),
                int(getattr(single_card_box, "y", 0)),
                int(getattr(single_card_box, "w", frame_w)),
                int(getattr(single_card_box, "h", frame_h)),
                label="skill_reward_showcase:top_card",
                color=(255, 180, 0),
                alpha=0.10,
                duration=2.5,
                font_size=16,
            )
            DebugTools().add_box(
                int(getattr(chosen_info, "x", 0)),
                int(getattr(chosen_info, "y", 0)),
                int(getattr(chosen_info, "w", frame_w)),
                int(getattr(chosen_info, "h", frame_h)),
                label="skill_reward_showcase:info_panel",
                color=(0, 220, 255),
                alpha=0.10,
                duration=2.5,
                font_size=16,
            )
    hud_marker_count = sum(
        1
        for label in (
            ProducerLabels.PC_PROGRESS,
            ProducerLabels.PC_STAMINA,
            ProducerLabels.PC_P_POINT,
            ProducerLabels.PC_TARGET,
        )
        if results.exists_label(label)
    )
    has_action_info = results.exists_label(ProducerLabels.PC_ACTION_INFO)
    has_skill_card_info = results.exists_label(ProducerLabels.SKILL_CARD_INFO)
    normalized = normalize_text(frame_text)
    has_showcase_text = _contains_text(frame_text, *ProduceText.SKILL_REWARD_SHOWCASE_VERBS)
    has_skill_card_detail_text = bool(normalized) and any(
        token in normalized
        for token in map(normalize_text, ProduceText.SKILL_CARD_DETAIL_HINT_TOKENS)
    )
    has_other_controls = any(
        results.exists_label(label)
        for label in (
            ProducerLabels.PC_ACTION,
            ProducerLabels.PC_RECOMMEND_ACTION,
            ProducerLabels.UNIVERSAL_OPTIONS,
            ProducerLabels.CONFIRM_BUTTON,
            ProducerLabels.DISABLE_BUTTON,
            ProducerLabels.CANCEL_BUTTON,
            ProducerLabels.MODAL_HEADER,
            ProducerLabels.PC_TRAINING_SCORE,
            ProducerLabels.PC_TRAINING_REMAINING,
            ProducerLabels.PC_BONUS_INDICATOR,
            ProducerLabels.SPECIAL_ITEM,
            ProducerLabels.SKIP_BUTTON,
            ProducerLabels.PLOT_FAST_FORWARD_BUTTON,
            BaseUILabels.BUTTON,
            BaseUILabels.BACK_BTN,
            BaseUILabels.CURRENT_LOCATION,
            BaseUILabels.SKIP_BUTTON,
            BaseUILabels.PLOT_FAST_FORWARD_BUTTON,
        )
    )
    # 展示页判据：单卡且位置明显高于三选一卡行。
    # 再结合 HUD / ActionInfo / 详情文本约束，避免把普通三选一误判为展示。
    return bool(
        primary_card_count == 1
        and single_card_is_high
        and not has_other_controls
        and (
            (hud_marker_count >= 3 and has_action_info and has_showcase_text)
            or (has_skill_card_info and has_skill_card_detail_text)
            or (has_skill_card_info and has_info_panel_below)
            or (
                hud_marker_count >= 2
                and has_action_info
                and not results.exists_label(ProducerLabels.PC_ACTION)
                and not results.exists_label(ProducerLabels.PC_RECOMMEND_ACTION)
            )
        )
    )


def _looks_like_exam_result_summary_showcase(frame_text: str) -> bool:
    """识别试验 / 审核结束后的名次与分项评价展示页。"""
    normalized = normalize_text(frame_text)
    if not normalized:
        return False
    has_pass_condition = (
        ProduceText.PASS_CONDITION in normalized
        or (
            ProduceText.PASS in normalized
            and ProduceText.PASS_POSITION_SUFFIX in normalized
        )
    )
    has_breakdown = (
        normalized.count("%") >= 2
        or normalized.count(ProduceText.RANKING_POINT_UNIT) >= 2
    )
    has_praise = any(
        token in normalized
        for token in ProduceText.PRAISE_EXCELLENT_OCR_VARIANTS
    )
    return bool(has_pass_condition and (has_breakdown or has_praise))


def _looks_like_exam_result_ranking_summary(
    frame_text: str,
    button_texts: Sequence[str] | None = None,
) -> bool:
    """识别考试结束后的排行榜结果页。

    实机样本 `capture_1775682132.png` 的稳定特征：
    - 画面正文会连续出现多名偶像的 `Pt` 分数；
    - 底部同时存在 `次へ`，并常伴随 `再挑戦`；
    - producer 模型虽然能检到底部按钮，但正文不包含既有结果链关键词，
      需要单独补这一类“排行榜 + Next”页面。
    """
    normalized = normalize_text(frame_text)
    if not normalized:
        return False
    button_texts = list(button_texts or [])
    has_next = _button_text_matches(button_texts, ButtonText.NEXT)
    has_retry = _button_text_matches(button_texts, ButtonText.RETRY)
    pt_count = normalized.count(ProduceText.RANKING_POINT_UNIT)
    ordinal_count = sum(
        1
        for marker in ProduceText.RANKING_ORDINAL_MARKERS
        if marker in normalized
    )
    return bool(
        has_next
        and pt_count >= 3
        and (has_retry or ordinal_count >= 2)
    )


def _looks_like_result_chain(results) -> bool:
    """结果链 / 记忆生成链的 OCR 兜底识别。

    这些页面在 producer 模型下经常只有少量按钮或根本没有可分辨标签，
    单靠 YOLO 不足以稳定分流，因此在低频的 unknown 分支中叠加 OCR 文本。
    """
    # 结果链里常见「Current Location + Skip」组合，静态回归帧下 OCR 很难稳定拿到文本，
    # 先用标签组合做一次无 OCR 的快速分流。
    # 排除带有 gameplay 元素的画面（进度条 / 对话选项 / 技能卡），避免与对话场景混淆。
    if (
        results.exists_label(BaseUILabels.CURRENT_LOCATION)
        and not results.exists_label(ProducerLabels.PLOT_FAST_FORWARD_BUTTON)
        and not results.exists_label(BaseUILabels.PLOT_FAST_FORWARD_BUTTON)
        and not results.exists_label(ProducerLabels.PC_PROGRESS)
        and not results.exists_label(ProducerLabels.UNIVERSAL_OPTIONS)
        and (
        results.exists_label(ProducerLabels.SKIP_BUTTON)
        or results.exists_label(BaseUILabels.SKIP_BUTTON)
        )
    ):
        return True

    button_texts = collect_button_like_texts(results)
    frame_text = _call_ui_attr("collect_frame_text", collect_frame_text, results)

    if _button_text_matches(button_texts, ButtonText.REGENERATE, ButtonText.COMPLETE):
        return True
    if _button_text_matches(button_texts, ButtonText.GENERATE) and _contains_text(frame_text, ProduceText.MEMORY_SELECT):
        return True
    if _looks_like_exam_result_summary_showcase(frame_text):
        return True
    if _looks_like_exam_result_ranking_summary(frame_text, button_texts):
        return True
    if _contains_text(
        frame_text,
        ProduceText.FAILED,
        ProduceText.FINAL_PRODUCE_EVALUATION,
        ProduceText.MEMORY_GENERATION_COMPLETE,
        ProduceText.MEMORY_SELECT,
        ProduceText.ACHIEVEMENT_PROGRESS,
        ProduceText.EVENT_REWARD_PROGRESS,
        ProduceText.EVENT_POINT,
        ProduceText.PRODUCE_RESULT,
        ProduceText.REWARD_ITEMS,
    ):
        return True
    return False


# ──────────────────────────────────────────────────────────
# Gameplay 阶段检测辅助
# ──────────────────────────────────────────────────────────

def classify_gameplay_phase(results, *, ctx: "ProduceContext | None" = None) -> str:
    """根据 PRODUCER YOLO 结果判定当前 gameplay phase。"""
    if results is None:
        return GameplayPhase.UNKNOWN

    # ── 横画面检测（Live演出）──
    # Live 演出中游戏画面会横屏（width > height）。
    # YOLO 按竖屏训练，横屏时可能无法检测到标签。
    frame = results.frame
    if frame is not None and frame.shape[1] > frame.shape[0] * 1.3:
        return GameplayPhase.LIVE_PERFORMANCE

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
    skill_card_count = sum(
        len(results.filter_by_label(label))
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
    has_center_p_drink = _has_center_p_drink_boxes(results)
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
    has_special_item = results.exists_label(ProducerLabels.SPECIAL_ITEM)
    has_current_location = results.exists_label(BaseUILabels.CURRENT_LOCATION)
    has_reward_controls = has_button or has_confirm or has_disable or has_cancel
    has_schedule_actions = has_action or (has_recommend and has_progress)
    last_stable_position = str(getattr(ctx, "last_stable_position", "") or "")
    last_consult_position = last_stable_position
    frame_text = ""

    # 弹窗判定：排除 P_DRINK 面板（其标题也被检测为 Modal Header）
    # 但如果同时有 Cancel 按钮，说明是确认弹窗（如交換確認），应优先判定为 MODAL
    if has_modal_header and (not has_p_drink or has_cancel):
        return GameplayPhase.MODAL
    # 相談交换页：出现兑换卡、強化、削除任一元素
    if (has_card_exchange or has_enhancement or has_remove) and not has_training_score:
        return GameplayPhase.CONSULT
    # 相談子流程（強化/削除预览页）与 skill_reward 外观相似，需要借助上一个稳定位置反解。
    if (
        ctx is not None
        and last_consult_position.startswith(CONSULT_POSITION_PREFIX)
        and has_skill_card
        and not has_action
        and not has_training_score
        and (has_button or has_confirm or has_disable or has_cancel)
    ):
        return GameplayPhase.CONSULT
    # P道具选择：出现 Special Item 且无技能卡、无行动。
    if has_special_item and not has_skill_card and not has_action:
        return GameplayPhase.ITEM_SELECT
    if has_skill_card and (has_bonus_indicator or has_pc_skip) and not has_action and not has_training_score:
        return GameplayPhase.EXAM
    if has_skill_card and (has_training_score or has_training_remaining):
        return GameplayPhase.LESSON
    if _looks_like_lesson_summary_showcase(results):
        return GameplayPhase.LESSON
    # 手牌为空的 LESSON/EXAM：HUD 存在但没有技能卡（0枚状态，回合自动前进）
    if not has_skill_card and (has_training_score or has_training_remaining) and has_pc_skip and not has_action:
        return GameplayPhase.LESSON
    if not has_skill_card and has_bonus_indicator and not has_action and not has_training_score:
        return GameplayPhase.EXAM
    if not frame_text:
        frame_text = _call_ui_attr("collect_frame_text", collect_frame_text, results)
    if (
        has_progress
        and not has_action
        and not has_skill_card
        and not has_training_score
        and _looks_like_present_support_selection(frame_text, results)
    ):
        return GameplayPhase.SCHEDULE
    if (
        has_progress
        and not has_action
        and not has_skill_card
        and not has_training_score
        and _looks_like_present_support_showcase(frame_text, results)
    ):
        return GameplayPhase.SCHEDULE
    if not frame_text:
        frame_text = _call_ui_attr("collect_frame_text", collect_frame_text, results)
    if _looks_like_skill_reward_showcase(results, frame_text):
        return GameplayPhase.SKILL_REWARD
    # 结果链检测：至少需要一个可交互元素（按钮/确认/跳过）才认定为结果画面，
    # 避免过渡帧的 OCR 噪声导致误判
    has_any_interactive = has_button or has_confirm or has_disable or has_cancel or has_skip_button or has_pc_skip
    if has_any_interactive and _looks_like_result_chain(results):
        return GameplayPhase.RESULT
    if not has_any_interactive:
        frame_text = _call_ui_attr("collect_frame_text", collect_frame_text, results)
        if (
            _looks_like_exam_result_summary_showcase(frame_text)
            or _contains_text(frame_text, ProduceText.FINAL_PRODUCE_EVALUATION, ProduceText.FAILED, ProduceText.PRODUCE_RESULT)
        ):
            return GameplayPhase.RESULT

    # 行程中的事件对话同样会带 Progress HUD，不能再被粗暴吞进 schedule。
    if has_options:
        return GameplayPhase.DIALOGUE
    if has_fast_forward and not has_schedule_actions and not has_skill_card and not has_skill_card_info:
        return GameplayPhase.DIALOGUE
    # Current Location + Skip 更像结果链；只有纯 Skip 时才交给 dialogue。
    if has_skip_button and not has_schedule_actions and not has_skill_card and not has_current_location:
        return GameplayPhase.DIALOGUE
    if (
        has_progress
        and not has_schedule_actions
        and not has_skill_card
        and not has_skill_card_info
        and not has_center_p_drink
        and not has_modal_header
        and not has_any_interactive
        and _looks_like_dialogue_text(frame_text)
    ):
        return GameplayPhase.DIALOGUE
    # 外出等事件中展示技能卡的对话画面：
    # 有技能卡 + 有 HUD + 无训练/考试标签 + 无操作按钮 + 有对话文本 → DIALOGUE
    if (
        has_skill_card
        and has_progress
        and not has_schedule_actions
        and not has_training_score
        and not has_training_remaining
        and not has_bonus_indicator
        and not has_any_interactive
        and _looks_like_dialogue_text(frame_text)
    ):
        return GameplayPhase.DIALOGUE

    # 牌组查看器覆盖层：Tab Bar + Cancel + 大量技能卡 + 无 Confirm
    # → 属于弹窗覆盖层（如“所持スキルカード”），不是技能奖励选择。
    has_tab_bar = results.exists_label(BaseUILabels.TAB_BAR)
    if (
        has_tab_bar
        and has_cancel
        and has_skill_card
        and skill_card_count >= 6
        and not has_confirm
        and not has_action
        and not has_training_score
    ):
        return GameplayPhase.MODAL

    if (
        (has_skill_card or has_skill_card_info)
        and not has_action
        and not has_training_score
        and (has_reward_controls or skill_card_count >= 2)
    ):
        return GameplayPhase.SKILL_REWARD

    # P_DRINK 弹窗（cy < 85% 区分行程栏底部的P饮料图标）
    if has_p_drink and not has_action and not has_skill_card:
        if has_center_p_drink:
            return GameplayPhase.P_DRINK
    if (
        ctx is not None
        and not has_action
        and not has_skill_card
        and not has_center_p_drink
        and (has_confirm or has_disable)
        and _looks_like_p_drink_receive_confirmation(frame_text, ctx=ctx)
    ):
        return GameplayPhase.P_DRINK

    # 稳定的行程页至少应当还能看到行动按钮 / 推荐行动，而不是只剩 HUD 残影。
    if has_schedule_actions:
        return GameplayPhase.SCHEDULE

    # 考试准备页面：YOLO 无法检测（0 个标签），使用 OCR 识别「審査基準」/「合格条件」
    if _looks_like_exam_prep_screen(results):
        return GameplayPhase.EXAM

    if _looks_like_loading_screen(results):
        return GameplayPhase.LOADING

    return GameplayPhase.UNKNOWN


def detect_gameplay_phase(app: "AppProcessor", ctx: "ProduceContext | None" = None) -> str:
    """检测当前 gameplay 阶段（Phase）。

    基于最新一帧 YOLO 检测结果，委托给 classify_gameplay_phase 判断当前
    画面属于哪个游戏阶段（如 SCHEDULE、LESSON、DIALOGUE、EXAM 等）。

    Args:
        app: 应用处理器实例，提供 latest_results 中的 YOLO 检测结果。
        ctx: 培育上下文对象，用于获取 last_stable_position 等辅助信息。

    Returns:
        str: GameplayPhase 枚举值的字符串表示。
    """
    return _call_ui_attr(
        "classify_gameplay_phase",
        classify_gameplay_phase,
        app.latest_results,
        ctx=ctx,
    )


def classify_pipeline_position(
    results,
    *,
    modal_title: str | None = None,
    final_confirm: bool = False,
    ctx: "ProduceContext | None" = None,
    phase: str | None = None,
) -> str:
    """判定`pipeline_position`类别。"""
    if final_confirm:
        return GameplayPosition.FINAL_CONFIRM

    phase = phase or _call_ui_attr(
        "classify_gameplay_phase",
        classify_gameplay_phase,
        results,
        ctx=ctx,
    )
    if phase == GameplayPhase.MODAL:
        modal_title = modal_title or ""
        if string_match(modal_title, ProduceText.VOICE_PLAYBACK_CONFIRM, MatchConfig(fuzz_threshold=65, normalize=True)):
            return GameplayPosition.STARTUP_MODAL_VOICE
        if string_match(modal_title, ProduceText.COMMU_FAST_FORWARD, MatchConfig(fuzz_threshold=65, normalize=True)):
            return GameplayPosition.STARTUP_MODAL_FAST_FORWARD
        if string_match(modal_title, ProduceText.PRODUCE_SKIP_SETTINGS, MatchConfig(fuzz_threshold=65, normalize=True)):
            return GameplayPosition.STARTUP_MODAL_SKIP_SETTINGS
        if _contains_text(modal_title, ProduceText.END_TURN_CONFIRM):
            return GameplayPosition.EXAM_END_TURN_CONFIRM_MODAL
        if _contains_text(modal_title, ProduceText.EXAM_RESULT_RETRY_CONFIRM):
            return GameplayPosition.EXAM_RETRY_CONFIRM_MODAL
        if _contains_text(modal_title, ProduceText.MEMORY_CONFIRM):
            return GameplayPosition.MEMORY_CONFIRM_MODAL
        if _contains_text(modal_title, ProduceText.MEMORY_REGEN_CONFIRM):
            return GameplayPosition.MEMORY_REGEN_CONFIRM_MODAL
        if _contains_text(modal_title, ProduceText.UNREAD_COMMU_FAST_FORWARD_CONFIRM):
            return GameplayPosition.FAST_FORWARD_CONFIRM_MODAL
        if ProduceText.P_DRINK in modal_title and ProduceText.DETAIL in modal_title:
            return GameplayPosition.P_DRINK_DETAIL
        if ProduceText.DETAIL in modal_title:
            return GameplayPosition.DETAIL_MODAL
        last_position = str(getattr(ctx, "last_stable_position", "") or "")
        if last_position.startswith(CONSULT_ENHANCEMENT_POSITION_PREFIX):
            return GameplayPosition.CONSULT_ENHANCEMENT_CONFIRM_MODAL
        return GameplayPosition.GAMEPLAY_MODAL

    if results is None:
        return GameplayPosition.UNKNOWN

    if phase == GameplayPhase.SCHEDULE:
        frame_text = _call_ui_attr("collect_frame_text", collect_frame_text, results)
        has_pc_action = results.exists_label(ProducerLabels.PC_ACTION)
        has_ff = results.exists_label(BaseUILabels.PLOT_FAST_FORWARD_BUTTON)
        has_opts = results.exists_label(ProducerLabels.UNIVERSAL_OPTIONS)

        if _looks_like_present_support_selection(frame_text, results):
            return GameplayPosition.SCHEDULE_PRESENT_SUPPORT
        if _looks_like_present_support_showcase(frame_text, results):
            return GameplayPosition.SCHEDULE_PRESENT_SUPPORT_SHOWCASE
        # 行程事件对话选项（外出等）— 有选项且无行程行动按钮
        if has_opts and not has_pc_action:
            return GameplayPosition.SCHEDULE_EVENT_OPTIONS
        # 授業課程選項：PC_ACTION + 快進按鈕 + 無推薦標記 + 少量選項 + 「授業」文本
        # 授業画面与常規周行程的区別：
        # - 常規周行程: PC_ACTION + PC_RECOMMEND_ACTION, 通常 5+ 個選項
        # - 授業選項: PC_ACTION(3個) + 快進按鈕（無効化）, 無推薦標記
        has_recommend = results.exists_label(ProducerLabels.PC_RECOMMEND_ACTION)
        if has_pc_action and has_ff and not has_recommend:
            action_count = len(list(results.filter_by_label(ProducerLabels.PC_ACTION)))
            if action_count <= 3 and _contains_text(frame_text, ProduceText.CLASS):
                if results.exists_label(ProducerLabels.PC_ACTION_INFO):
                    return GameplayPosition.SCHEDULE_LESSON_SELECTED
                return GameplayPosition.SCHEDULE_LESSON_OPTIONS
        if results.exists_label(ProducerLabels.PC_ACTION_INFO):
            return GameplayPosition.SCHEDULE_SELECTED
        if has_recommend:
            return GameplayPosition.SCHEDULE_RECOMMEND
        # 行程事件对话文本（有快进按钮、无选项、无行动）。
        if has_ff and not has_opts and not has_pc_action:
            return GameplayPosition.SCHEDULE_EVENT_DIALOGUE
        return GameplayPosition.SCHEDULE_IDLE

    if phase == GameplayPhase.LESSON:
        if _looks_like_lesson_summary_showcase(results):
            return GameplayPosition.LESSON_SUMMARY_SHOWCASE
        if results.exists_label(ProducerLabels.SKILL_CARD_INFO):
            return GameplayPosition.LESSON_SELECTED
        return GameplayPosition.LESSON_IDLE

    if phase == GameplayPhase.DIALOGUE:
        if results.exists_label(ProducerLabels.UNIVERSAL_OPTIONS):
            return GameplayPosition.DIALOGUE_OPTIONS
        return GameplayPosition.DIALOGUE_CONTINUE

    if phase == GameplayPhase.P_DRINK:
        if results.exists_label(ProducerLabels.CONFIRM_BUTTON):
            return GameplayPosition.P_DRINK_SELECTED
        if results.exists_label(ProducerLabels.DISABLE_BUTTON):
            return GameplayPosition.P_DRINK_IDLE
        return GameplayPosition.P_DRINK_IDLE

    if phase == GameplayPhase.SKILL_REWARD:
        frame_text = _call_ui_attr("collect_frame_text", collect_frame_text, results)
        if _looks_like_skill_reward_showcase(results, frame_text):
            return GameplayPosition.SKILL_REWARD_SHOWCASE
        # 出现“请选择要领取的技能卡”提示时，明确是未选中状态。
        if _contains_text(frame_text, ProduceText.SKILL_REWARD_SELECT):
            return GameplayPosition.SKILL_REWARD_IDLE
        if results.exists_label(ProducerLabels.CONFIRM_BUTTON):
            return GameplayPosition.SKILL_REWARD_SELECTED
        if results.exists_label(ProducerLabels.DISABLE_BUTTON):
            return GameplayPosition.SKILL_REWARD_IDLE
        if results.exists_label(BaseUILabels.BUTTON):
            button_texts = _call_ui_attr("collect_button_like_texts", collect_button_like_texts, results)
            has_receive_button = _button_text_matches(button_texts, ProduceText.RECEIVE)
            has_redraw_button = _button_text_matches(button_texts, ProduceText.REDRAW, ProduceText.REDRAW_SHORT)
            if has_redraw_button and not has_receive_button:
                return GameplayPosition.SKILL_REWARD_IDLE
            if has_receive_button and not has_redraw_button:
                return GameplayPosition.SKILL_REWARD_SELECTED
            # OCR 不稳定时默认保守为 idle，避免误入 selected 后死循环确认。
            return GameplayPosition.SKILL_REWARD_IDLE
        return GameplayPosition.SKILL_REWARD_IDLE

    if phase == GameplayPhase.CONSULT:
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
            return GameplayPosition.CONSULT_EXCHANGE
        if has_consult_cards:
            if results.exists_label(ProducerLabels.CONFIRM_BUTTON) or results.exists_label(BaseUILabels.BUTTON):
                return GameplayPosition.CONSULT_ENHANCEMENT_READY
            return GameplayPosition.CONSULT_ENHANCEMENT_PREVIEW
        return GameplayPosition.CONSULT_IDLE

    if phase == GameplayPhase.ITEM_SELECT:
        # Disable Button → 未选择; Confirm Button → 已选择
        if results.exists_label(ProducerLabels.CONFIRM_BUTTON) or results.exists_label(BaseUILabels.BUTTON):
            return GameplayPosition.ITEM_SELECT_SELECTED
        return GameplayPosition.ITEM_SELECT_IDLE

    if phase == GameplayPhase.EXAM:
        # 考试准备页面：YOLO 检测不到技能卡，通过 OCR 判定
        if _looks_like_exam_prep_screen(results):
            return GameplayPosition.EXAM_PREP
        if results.exists_label(ProducerLabels.SKILL_CARD_INFO):
            return GameplayPosition.EXAM_SELECTED
        return GameplayPosition.EXAM_IDLE

    if phase == GameplayPhase.RESULT:
        frame_text = _call_ui_attr("collect_frame_text", collect_frame_text, results)
        button_texts = collect_button_like_texts(results)
        if _contains_text(frame_text, ProduceText.FAILED):
            return GameplayPosition.RESULT_EXAM_FAILURE
        if _looks_like_exam_result_summary_showcase(frame_text):
            return GameplayPosition.RESULT_EXAM_SUMMARY_SHOWCASE
        if _looks_like_exam_result_ranking_summary(frame_text, button_texts):
            return GameplayPosition.RESULT_EXAM_RANKING_SUMMARY
        if _contains_text(frame_text, ProduceText.FINAL_PRODUCE_EVALUATION):
            return GameplayPosition.RESULT_FINAL_EVALUATION
        if _contains_text(frame_text, ProduceText.MEMORY_SELECT) or _button_text_matches(
            button_texts,
            ButtonText.REGENERATE,
            ProduceText.MEMORY_LIST,
        ):
            return GameplayPosition.RESULT_MEMORY_PAGE
        if _contains_text(frame_text, ProduceText.ACHIEVEMENT_PROGRESS):
            return GameplayPosition.RESULT_ACHIEVEMENT_PROGRESS
        if _contains_text(frame_text, ProduceText.EVENT_REWARD_PROGRESS, ProduceText.EVENT_POINT):
            return GameplayPosition.RESULT_EVENT_REWARD_PROGRESS
        if _button_text_matches(button_texts, ButtonText.NEXT) and _button_text_matches(button_texts, ProduceText.PRODUCE_HISTORY):
            return GameplayPosition.RESULT_REWARD_SUMMARY
        if _contains_text(frame_text, ProduceText.PRODUCE_RESULT):
            return GameplayPosition.RESULT_REWARD_SUMMARY
        if _button_text_matches(button_texts, ButtonText.GENERATE):
            return GameplayPosition.RESULT_MEMORY_GENERATION
        if _button_text_matches(button_texts, ButtonText.COMPLETE):
            return GameplayPosition.RESULT_FINAL_EVALUATION
        return GameplayPosition.RESULT

    if not results:
        return GameplayPosition.TRANSITION_EMPTY

    frame_text = _call_ui_attr("collect_frame_text", collect_frame_text, results)
    if _looks_like_resume_title_screen(frame_text, results):
        return GameplayPosition.TRANSITION_RESUME_TITLE

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
        return GameplayPosition.TRANSITION_HUD
    # 无 HUD 标签但画面非空（例如过场动画、外部界面）→ 统一归为 TRANSITION_EMPTY
    return GameplayPosition.TRANSITION_EMPTY


def classify_gameplay_state(
    results,
    *,
    modal_title: str | None = None,
    final_confirm: bool = False,
    ctx: "ProduceContext | None" = None,
) -> tuple[str, str]:
    """基于同一帧检测结果同时计算 phase 和 position。

    gameplay 联调时，YOLO 线程会持续刷新 latest_results。
    如果 phase / position 分别从两次 latest_results 读取，可能落在不同帧上，
    从而出现 phase=unknown 但 position=dialogue_options 之类的撕裂状态。
    """
    phase = _call_ui_attr(
        "classify_gameplay_phase",
        classify_gameplay_phase,
        results,
        ctx=ctx,
    )
    position = _call_ui_attr(
        "classify_pipeline_position",
        classify_pipeline_position,
        results,
        modal_title=modal_title,
        final_confirm=final_confirm,
        ctx=ctx,
        phase=phase,
    )
    return phase, position


def detect_gameplay_state(
    app: "AppProcessor",
    ctx: "ProduceContext | None" = None,
    *,
    include_final_confirm: bool = False,
) -> tuple[str, str]:
    """从同一帧快照中读取 gameplay 的 phase 与 position。"""
    results = app.latest_results
    modal_title: str | None = None
    if results and results.exists_label(ProducerLabels.MODAL_HEADER):
        modal = app.game_utils.try_get_modal(no_body=True)
        if modal is not None:
            modal_title = modal.modal_title
    final_confirm = is_final_confirm_page(app) if include_final_confirm else False
    return _call_ui_attr(
        "classify_gameplay_state",
        classify_gameplay_state,
        results,
        modal_title=modal_title,
        final_confirm=final_confirm,
        ctx=ctx,
    )


def get_pipeline_position(app: "AppProcessor", ctx: "ProduceContext | None" = None) -> str:
    """获取当前流水线位置（Position）。

    基于最新一帧 YOLO 检测结果，结合弹窗标题和最终确认页状态，委托给
    classify_pipeline_position 判断当前画面在流水线中的精细位置。
    Position 比 Phase 更细粒度，例如 SCHEDULE 阶段可细分为
    SCHEDULE_IDLE、SCHEDULE_RECOMMEND、SCHEDULE_SELECTED 等。

    Args:
        app: 应用处理器实例，提供 latest_results 和 game_utils.try_get_modal。
        ctx: 培育上下文对象，用于获取 last_stable_position 等辅助信息。

    Returns:
        str: GameplayPosition 枚举值的字符串表示。
    """
    modal_title: str | None = None
    results = app.latest_results
    if results and results.exists_label(ProducerLabels.MODAL_HEADER):
        modal = app.game_utils.try_get_modal(no_body=True)
        if modal is not None:
            modal_title = modal.modal_title
    return _call_ui_attr(
        "classify_pipeline_position",
        classify_pipeline_position,
        results,
        modal_title=modal_title,
        final_confirm=is_final_confirm_page(app),
        ctx=ctx,
    )
