from __future__ import annotations

import re
from inspect import signature
from typing import Any, Iterable, Sequence

import cv2

from src.constants.game.text.produce_text import ProduceText
from src.core.inference.ocr_engine import OCRService
from src.utils.i18n_tools import i18n_text
from src.utils.opencv_tools import check_status_detection
from src.utils.string_tools import fullwidth_to_halfwidth, normalize_ocr_jp

_ocr_service = OCRService()
_LOOKUP_CLEANUP_RE = re.compile(r"[\s　・･/／|｜,，.。:：()\[\]{}<>「」『』【】'\"`]+")

_VOCAL_TOKENS = (ProduceText.VOCAL, "vocal", "vo")
_DANCE_TOKENS = (ProduceText.DANCE, "dance", "da")
_VISUAL_TOKENS = (ProduceText.VISUAL, "visual", "vi")


def ocr_text(image) -> str:
    """对图像执行 OCR 识别，返回合并后的全部文本。

    Args:
        image: 待识别的 OpenCV 图像对象（numpy array）。为 None 或空图像时返回空字符串。

    Returns:
        str: 所有 OCR 识别结果的拼接文本，识别失败时返回空字符串。
    """
    if image is None or getattr(image, "size", 0) <= 0:
        return ""
    return "".join(item.text for item in _ocr_service.ocr(image))


def normalize_text(text: str | None) -> str:
    """将文本归一化为紧凑的小写字符串，便于子串匹配和模糊比较。

    处理方式：转为字符串 -> 小写 -> 按空白分割 -> 拼接（去除所有空白字符）。

    Args:
        text: 待归一化的文本，为 None 时视为空字符串。

    Returns:
        str: 归一化后的紧凑小写字符串。
    """
    return "".join(str(text or "").lower().split())


def normalize_lookup_text(text: str | None) -> str:
    """统一 producer 任务中的 OCR/目录查找文本归一化方式。

    处理流水线：全角转半角 -> OCR 日文归一化 -> 去除标点符号和分隔符 -> 小写 -> 去首尾空白。
    适用于需要在不同来源的 OCR 文本之间进行稳定匹配的场景。

    Args:
        text: 待归一化的文本，为 None 或空时返回空字符串。

    Returns:
        str: 归一化后的紧凑文本。
    """
    if not text:
        return ""
    normalized = normalize_ocr_jp(fullwidth_to_halfwidth(str(text)))
    normalized = _LOOKUP_CLEANUP_RE.sub("", normalized)
    return normalized.lower().strip()


def infer_param_kind(text: str | None) -> str:
    """根据 OCR 文本推断属性类型。

    Args:
        text: 从页面标签、按钮或 OCR 结果中提取的原始文本。

    Returns:
        str: `vocal`、`dance`、`visual` 或 `unknown`。
        该结果通常用于记忆属性采集、属性加成识别等需要把文本映射为统一枚举值的场景。
    """
    normalized = normalize_text(text)
    if any(token in normalized for token in _VOCAL_TOKENS):
        return "vocal"
    if any(token in normalized for token in _DANCE_TOKENS):
        return "dance"
    if any(token in normalized for token in _VISUAL_TOKENS):
        return "visual"
    return "unknown"


def get_frame_size(app) -> tuple[int, int]:
    """获取当前画面帧的尺寸。

    从 app.latest_frame 中读取图像的宽高，当 frame 不存在时返回 None。

    Args:
        app: 应用处理器实例，需具有 latest_frame 属性（OpenCV 图像对象）。

    Returns:
        tuple[int, int] | None: (宽度, 高度)，单位为像素；无法获取时返回 None。
    """
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    return int(width), int(height)


def detect_bottom_white_modal_region(
    frame: Any,
    *,
    row_boxes: Sequence[Any],
    debug_tools: Any = None,
    debug_label: str = "white_modal",
) -> tuple[int, int, int, int] | None:
    """基于同一行候选框锚定，检测画面底部的白色弹窗区域。

    处理流程：
    1. 过滤有效检测框，仅保留靠近底部的一行
    2. 计算行的包围盒作为锚定区域
    3. 基于锚定区域扩展出期望矩形
    4. 对画面做 HSV 白色阈值分割 + 形态学处理
    5. 遍历白色轮廓，与锚定区域计算重叠度和 IoU，打分选出最佳匹配

    Args:
        frame: OpenCV 图像对象（BGR）。
        row_boxes: 候选检测框序列（如底部按钮行），用于锚定弹窗区域。
        debug_tools: 可选的调试工具，用于绘制检测框可视化。
        debug_label: 调试标注的前缀标签。

    Returns:
        tuple[int, int, int, int] | None: 最佳匹配的白色弹窗区域 (x1, y1, x2, y2)，未检测到返回 None。
    """
    if frame is None or getattr(frame, "size", 0) <= 0 or not row_boxes:
        return None

    frame_h, frame_w = frame.shape[:2]
    valid_boxes: list[Any] = []
    for box in row_boxes:
        x = int(getattr(box, "x", 0))
        y = int(getattr(box, "y", 0))
        w = int(getattr(box, "w", 0))
        h = int(getattr(box, "h", 0))
        if w <= x or h <= y:
            continue
        valid_boxes.append(box)
    if not valid_boxes:
        return None

    # 只保留靠近底部的一行候选框，避免把信息面板/提示框混入锚定导致模态过大。
    if len(valid_boxes) >= 2:
        max_cy = max(int(getattr(box, "cy", 0) or 0) for box in valid_boxes)
        lane_threshold = max_cy - int(frame_h * 0.12)
        lane_boxes = [box for box in valid_boxes if int(getattr(box, "cy", 0) or 0) >= lane_threshold]
        if lane_boxes:
            valid_boxes = lane_boxes

    xs = [int(getattr(box, "x", 0)) for box in valid_boxes]
    ys = [int(getattr(box, "y", 0)) for box in valid_boxes]
    ws = [int(getattr(box, "w", 0)) for box in valid_boxes]
    hs = [int(getattr(box, "h", 0)) for box in valid_boxes]
    row_left = max(0, min(xs))
    row_top = max(0, min(ys))
    row_right = min(frame_w, max(ws))
    row_bottom = min(frame_h, max(hs))
    # 单目标时扩展横向锚定，避免白色模态被误判为“过大轮廓”
    if len(valid_boxes) <= 1:
        row_left = min(row_left, int(frame_w * 0.18))
        row_right = max(row_right, int(frame_w * 0.82))
    row_w = max(1.0, float(row_right - row_left))
    row_h = max(1.0, float(row_bottom - row_top))
    row_area = row_w * row_h

    expected_rect = (
        max(0, int(row_left - row_w * 0.20)),
        max(0, int(row_top - row_h * 3.9)),
        min(frame_w, int(row_right + row_w * 0.20)),
        min(frame_h, int(row_bottom + row_h * 1.5)),
    )
    if debug_tools is not None:
        debug_tools.add_box(
            expected_rect[0],
            expected_rect[1],
            expected_rect[2],
            expected_rect[3],
            label=f"{debug_label}_anchor",
            color=(180, 200, 255),
            alpha=0.08,
            duration=2.0,
            font_size=13,
        )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 适当收紧饱和度上界，抑制角色高亮/半透明过渡区域对底部白模态的污染。
    white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 52, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    ex_x1, ex_y1, ex_x2, ex_y2 = expected_rect
    ex_area = max(1.0, float(max(1, ex_x2 - ex_x1) * max(1, ex_y2 - ex_y1)))
    area_cap_ratio = 1.85 if len(valid_boxes) >= 2 else 4.5
    panel_top_min = max(ex_y1, int(frame_h * 0.20))

    best_rect: tuple[int, int, int, int] | None = None
    best_score = -1e9
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        if w_rect <= 0 or h_rect <= 0:
            continue

        rx1, ry1, rx2, ry2 = x, y, x + w_rect, y + h_rect
        # 只取轮廓与锚定期望区域的交集，强制聚焦到底部白色模态
        rect = (
            max(rx1, ex_x1),
            max(ry1, ex_y1),
            min(rx2, ex_x2),
            min(ry2, ex_y2),
        )
        w_clip = max(0, rect[2] - rect[0])
        h_clip = max(0, rect[3] - rect[1])
        if w_clip <= 0 or h_clip <= 0:
            continue

        rect_area = float(w_clip * h_clip)
        area_ratio = rect_area / max(float(frame_w * frame_h), 1.0)
        if area_ratio < 0.08:
            continue
        if w_clip < frame_w * 0.42 or h_clip < row_h * 2.0:
            continue
        if rect_area > ex_area * area_cap_ratio:
            continue
        if rect[1] < panel_top_min:
            continue
        if rect[1] > int(row_top + row_h * 0.25):
            continue
        if rect[3] < int(row_bottom - row_h * 0.05):
            continue

        ix1 = max(rect[0], row_left)
        iy1 = max(rect[1], row_top)
        ix2 = min(rect[2], row_right)
        iy2 = min(rect[3], row_bottom)
        overlap_w = max(0, ix2 - ix1)
        overlap_h = max(0, iy2 - iy1)
        overlap_area = float(overlap_w * overlap_h)
        row_overlap = overlap_area / max(row_area, 1.0)
        if row_overlap < 0.16:
            continue

        iex1 = max(rect[0], ex_x1)
        iex2 = min(rect[2], ex_x2)
        iey1 = max(rect[1], ex_y1)
        iey2 = min(rect[3], ex_y2)
        inter = float(max(0, iex2 - iex1) * max(0, iey2 - iey1))
        iou_expected = inter / max(ex_area + rect_area - inter, 1.0)
        center_x = (rect[0] + rect[2]) / 2.0
        center_penalty = abs(center_x - frame_w / 2.0) / max(float(frame_w), 1.0)
        oversize_penalty = max(0.0, rect_area / ex_area - 2.0)
        score = (
            row_overlap * 860.0
            + iou_expected * 640.0
            + area_ratio * 120.0
            - center_penalty * 35.0
            - oversize_penalty * 200.0
        )
        if score > best_score:
            best_score = score
            best_rect = rect

    if best_rect and debug_tools is not None:
        debug_tools.add_box(
            best_rect[0],
            best_rect[1],
            best_rect[2],
            best_rect[3],
            label=debug_label,
            color=(120, 210, 255),
            alpha=0.12,
            duration=2.5,
            font_size=14,
        )
    return best_rect


def click_relative_point(
    app,
    *,
    x_ratio: float,
    y_ratio: float,
    label: str = "",
) -> tuple[int, int]:
    """按画面相对比例坐标点击指定位置。

    将相对坐标（0.0-1.0）转换为实际像素坐标后进行点击，坐标会自动约束在
    画面范围内。

    Args:
        app: 应用处理器实例，提供 latest_frame（获取尺寸）和 device.click。
        x_ratio: X 轴相对位置（0.0=左边缘，1.0=右边缘）。
        y_ratio: Y 轴相对位置（0.0=上边缘，1.0=下边缘）。
        label: 点击操作的标识标签，透传给 device.click。

    Returns:
        tuple[int, int]: 实际点击的像素坐标 (x, y)。
    """
    width, height = get_frame_size(app)
    x = max(0, min(width - 1, int(round(width * x_ratio))))
    y = max(0, min(height - 1, int(round(height * y_ratio))))
    app.device.click(x, y, label)
    return x, y


def probe_fast_forward_enabled_state(
    button_box: Any,
    *,
    debug_tools: Any = None,
    debug_label: str = "fast_forward_state",
) -> tuple[bool | None, float]:
    """判断快进按钮是否已处于橙色开启态。

    Returns:
        (enabled, orange_ratio)
        - enabled=True: 明确识别为已开启（橙色）
        - enabled=False: 明确识别为未开启
        - enabled=None: 无法判定（例如按钮裁剪为空）
    """
    frame = getattr(button_box, "frame", None)
    if frame is None or getattr(frame, "size", 0) <= 0:
        return None, 0.0

    # 快进按钮开启态是高饱和橙色；阈值放宽以提升 JPG 噪声下的稳定性。
    status = check_status_detection(
        frame,
        threshold=0.10,
        upper_color=(24, 255, 255),
        lower_color=(6, 90, 120),
    )
    enabled = bool(getattr(status, "status", False))
    orange_ratio = float(getattr(status, "value", 0.0) or 0.0)

    if debug_tools is not None:
        x1 = int(getattr(button_box, "x", 0))
        y1 = int(getattr(button_box, "y", 0))
        x2 = int(getattr(button_box, "w", x1))
        y2 = int(getattr(button_box, "h", y1))
        debug_tools.add_box(
            x1,
            y1,
            x2,
            y2,
            label=f"{debug_label}:{'on' if enabled else 'off'}:{orange_ratio:.2f}",
            color=(0, 165, 255) if enabled else (160, 160, 160),
            alpha=0.14,
            duration=2.0,
            font_size=13,
        )

    return enabled, orange_ratio


def invoke_decision_strategy(
    strategy,
    app,
    ctx,
    candidates: Sequence[Any],
    *,
    decision_state: Any = None,
) -> Any:
    """按策略函数实际签名调用自动决策回调。

    Args:
        strategy: 外部注入的决策函数，可接受 app、ctx、candidates、decision_state 的不同组合。
        app: 当前应用处理器。
        ctx: 培育上下文。
        candidates: 当前阶段可供选择的候选列表。
        decision_state: 已构建好的决策快照；当策略签名更偏向状态输入时优先传入该对象。

    Returns:
        Any: 策略返回的原始决策结果，不在此处做结构限制。

    Notes:
        该函数会根据参数个数与最后一个参数名自适应调用方式，兼容旧策略接口与新的
        无状态决策接口，避免调用侧在各个 handler 中重复写签名分支判断。
    """
    if strategy is None:
        return None
    parameters: list[Any] = []
    try:
        parameters = list(signature(strategy).parameters.values())
        parameter_count = len(parameters)
    except (TypeError, ValueError):
        parameter_count = 0

    if parameter_count >= 4:
        return strategy(app, ctx, candidates, decision_state)
    if parameter_count >= 3:
        last_param_name = parameters[2].name.lower()
        if decision_state is not None and last_param_name in {
            "state",
            "snapshot",
            "payload",
            "decision_state",
            "game_state",
            "input",
        }:
            return strategy(app, ctx, decision_state)
        return strategy(app, ctx, candidates)
    if parameter_count == 2:
        return strategy(app, ctx)
    if parameter_count == 1:
        param_name = parameters[0].name.lower() if parameters else ""
        if decision_state is not None and param_name in {
            "state",
            "snapshot",
            "payload",
            "decision_state",
            "game_state",
            "input",
        }:
            return strategy(decision_state)
        return strategy(candidates)
    if parameter_count == 0:
        return strategy()
    return strategy(app, ctx)


def resolve_candidate_index(
    decision: Any,
    candidates: Sequence[Any],
    *,
    default_index: int = 0,
) -> int:
    """把策略输出解析为候选列表中的有效索引。

    Args:
        decision: 策略返回值，可以是整数索引、带索引字段的对象、字典，或候选名称/ID 文本。
        candidates: 当前阶段可供选择的候选列表。
        default_index: 当 decision 无法可靠解析时回退使用的索引。

    Returns:
        int: 最终落地的候选索引，保证落在 `candidates` 范围内。

    Raises:
        ValueError: 候选列表为空时抛出，因为此时无法生成任何合法索引。

    Notes:
        解析顺序依次为显式索引、对象属性、字典字段、名称/ID 文本匹配；
        这样可以兼容 LLM、规则策略和 RL 策略返回的不同结构。
    """
    if not candidates:
        raise ValueError(i18n_text("backend.task.candidateListEmpty", fallback="候选列表为空"))

    if isinstance(decision, int):
        if 0 <= decision < len(candidates):
            return decision

    if hasattr(decision, "index"):
        candidate_index = getattr(decision, "index")
        if isinstance(candidate_index, int) and 0 <= candidate_index < len(candidates):
            return candidate_index

    if isinstance(decision, dict):
        for key in ("action_index", "index", "candidate_index"):
            candidate_index = decision.get(key)
            if isinstance(candidate_index, int) and 0 <= candidate_index < len(candidates):
                return candidate_index
        for key in ("candidate_id", "action_id", "db_id", "id", "action_type", "label", "title", "name", "kind"):
            candidate_index = _match_candidate_key(str(decision.get(key) or ""), candidates)
            if candidate_index is not None:
                return candidate_index

    for attr_name in ("action_index", "candidate_index", "choice_index"):
        if hasattr(decision, attr_name):
            candidate_index = getattr(decision, attr_name)
            if isinstance(candidate_index, int) and 0 <= candidate_index < len(candidates):
                return candidate_index
    for attr_name in ("candidate_id", "action_id", "db_id", "id", "action_type", "label", "title", "name", "kind"):
        if hasattr(decision, attr_name):
            candidate_index = _match_candidate_key(str(getattr(decision, attr_name) or ""), candidates)
            if candidate_index is not None:
                return candidate_index

    normalized_decision = normalize_text(decision) if isinstance(decision, str) else ""
    if normalized_decision:
        matched_index = _match_candidate_key(normalized_decision, candidates, already_normalized=True)
        if matched_index is not None:
            return matched_index

    return max(0, min(default_index, len(candidates) - 1))


def _match_candidate_key(
    value: str,
    candidates: Sequence[Any],
    *,
    already_normalized: bool = False,
) -> int | None:
    """按候选对象上的关键字段匹配文本，并返回首个命中索引。

    Args:
        value: 需要匹配的文本，可为动作 ID、名称、标签等。
        candidates: 候选对象列表。
        already_normalized: 为 True 时表示 `value` 已经做过 `normalize_text` 处理，
            可以跳过重复归一化。

    Returns:
        int | None: 命中候选时返回索引，否则返回 None。
    """
    normalized_value = value if already_normalized else normalize_text(value)
    if not normalized_value:
        return None
    for idx, candidate in enumerate(candidates):
        for attr_name in ("action_id", "db_id", "action_type", "title", "kind", "label", "name"):
            candidate_value = normalize_text(getattr(candidate, attr_name, ""))
            if candidate_value and (
                normalized_value == candidate_value
                or normalized_value in candidate_value
                or candidate_value in normalized_value
            ):
                return idx
    return None


def first_matching_index(candidates: Iterable[Any], *, kind: str) -> int | None:
    """返回候选列表中第一个 kind 属性与目标值相同的索引。

    Args:
        candidates: 候选对象可迭代集合，每个对象需具有 kind 属性。
        kind: 目标 kind 值，用于精确匹配。

    Returns:
        int | None: 匹配到的索引，未找到返回 None。
    """
    for idx, candidate in enumerate(candidates):
        if getattr(candidate, "kind", "") == kind:
            return idx
    return None
