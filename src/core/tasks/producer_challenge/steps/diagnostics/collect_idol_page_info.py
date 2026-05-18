"""Step 4b: 采集偶像卡页面信息并点击「次へ」。

在偶像卡选择完毕后、点击「次へ」之前，采集：
  - おすすめ効果（推荐效果）— 基于 YOLO 锚点与 OCR 行定位推荐效果区域，再点击展开提示框
  - 育成情報 — 基于 modal body 布局读取审查基准和育成课题
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    find_button,
    inertial_swipe,
    wait_frame_stable,
)
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.debug_tools import DebugTools
from src.utils.i18n_tools import i18n_text
from src.utils.logger import logger
from src.utils.opencv_tools import compute_ssim_score
from src.utils.string_tools import MatchConfig

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

_ocr = OCRService()
_debugger = DebugTools()

_OCR_SCALE = 2.0
_OCR_MIN_CONFIDENCE = 0.25
_LINE_MERGE_Y_GAP = 40
_RECOMMEND_LINE_MIN_WIDTH_RATIO = 0.08
_TOOLTIP_MIN_TOKEN_COUNT = 6
_TOOLTIP_MIN_LINE_COUNT = 4
_TOOLTIP_BOX_PADDING_X = 24
_TOOLTIP_BOX_PADDING_Y = 20
_SCROLL_COMPARE_TOP_RATIO = 0.56
_SCROLL_COMPARE_BOTTOM_RATIO = 0.94
_SCROLL_START_RATIO = 0.82
_SCROLL_END_RATIO = 0.45
_SCROLL_SETTLE_SSIM_THRESHOLD = 0.985

_PARAM_CANONICAL = {
    **{variant: "vocal" for variant in ProduceText.VOCAL_OCR_VARIANTS},
    **{variant: "dance" for variant in ProduceText.DANCE_OCR_VARIANTS},
    **{variant: "visual" for variant in ProduceText.VISUAL_OCR_VARIANTS},
}
_TASK_PARAM_PATTERN = "|".join(
    re.escape(variant)
    for variant in (
        *ProduceText.VOCAL_OCR_VARIANTS,
        *ProduceText.DANCE_OCR_VARIANTS,
        *ProduceText.VISUAL_OCR_VARIANTS,
    )
)
_TASK_CONDITION_RE = re.compile(
    rf"({_TASK_PARAM_PATTERN})"
    rf"\s*(\d+)\s*({ProduceText.COMPARISON_GE}|{ProduceText.COMPARISON_LE})",
)
_TASK_TYPE_VARIANTS = {
    **{
        variant: ProduceText.TASK_TYPE_PERFORMANCE
        for variant in ProduceText.TASK_TYPE_PERFORMANCE_OCR_VARIANTS
    },
    ProduceText.TASK_TYPE_WEAKNESS: ProduceText.TASK_TYPE_WEAKNESS,
}
_TASK_REWARD_TOKENS = (
    ProduceText.POINT,
    ProduceText.P_POINT,
    ProduceText.DRINK,
    ProduceText.BONUS,
)


@dataclass(frozen=True)
class _Rect:
    """统一的矩形区域描述。

    本文件会在 YOLO 检测框、OCR token 区域和手工推导区域之间来回转换坐标。
    使用 `_Rect` 统一 `x1/y1/x2/y2` 语义，可以避免和 `Yolo_Box` 的 `x/w/y/h`
    命名混淆，减少区域裁剪与调试框绘制时的坐标误用。
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """返回矩形宽度，供区域筛选与相对坐标换算使用。"""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """返回矩形高度，供区域筛选与相对坐标换算使用。"""
        return max(0, self.y2 - self.y1)

    @property
    def cx(self) -> int:
        """返回矩形中心点 x 坐标，便于点击和对齐 OCR 行。"""
        return self.x1 + self.width // 2

    @property
    def cy(self) -> int:
        """返回矩形中心点 y 坐标，便于点击和按行聚类 OCR token。"""
        return self.y1 + self.height // 2

    @classmethod
    def from_yolo_box(cls, box: Yolo_Box) -> "_Rect":
        """根据 YOLO 检测框构建 Rect 对象。"""
        return cls(int(box.x), int(box.y), int(box.w), int(box.h))

    def translate(self, dx: int, dy: int) -> "_Rect":
        """处理translate并返回结果。

        Args:
            dx: 用于提供dx相关输入。
            dy: 用于提供dy相关输入。

        Returns:
            '_Rect': 返回值类型见注解。
        """
        return _Rect(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def expand(self, padding_x: int, padding_y: int, frame_shape: Sequence[int]) -> "_Rect":
        """处理expand并返回结果。

        Args:
            padding_x: 用于提供padding、x相关输入。
            padding_y: 用于提供padding、y相关输入。
            frame_shape: 用于提供frame、shape相关输入。

        Returns:
            '_Rect': 返回值类型见注解。
        """
        frame_h, frame_w = frame_shape[:2]
        return _Rect(
            max(0, self.x1 - padding_x),
            max(0, self.y1 - padding_y),
            min(frame_w, self.x2 + padding_x),
            min(frame_h, self.y2 + padding_y),
        )


@dataclass(frozen=True)
class _OcrToken:
    """单个 OCR token 的标准化表示。

    除了文本本身，还保留识别框与置信度，方便后续按位置聚合成行，或在调试时回溯
    某段文本是从哪个局部区域识别出来的。
    """
    text: str
    rect: _Rect
    confidence: float


@dataclass(frozen=True)
class _OcrLine:
    """由多个 OCR token 聚合而成的一行文本。

    推荐效果与育成信息的解析都更依赖“行”而不是单个词，因此这里会保留整行文本、
    行级包围框以及组成该行的 token 列表，供后续做规则匹配与点击区域推导。
    """
    text: str
    rect: _Rect
    tokens: Tuple[_OcrToken, ...]


def _normalize_text(text: str) -> str:
    """压缩多余空白，统一 OCR 文本的比较格式。"""
    return re.sub(r"\s+", " ", (text or "").strip())


def _safe_confidence(value: Any) -> float:
    """将 OCR 置信度安全转换为 0-1 浮点数。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _union_rect(rects: Iterable[_Rect]) -> _Rect | None:
    """合并多个矩形为最小包围框。

    常用于把一组 OCR token 的局部框聚合成行级区域，后续可直接用于裁剪、点击或绘制调试框。
    空输入返回 `None`，由调用方决定是否跳过该区域。
    """
    rects = list(rects)
    if not rects:
        return None
    return _Rect(
        min(rect.x1 for rect in rects),
        min(rect.y1 for rect in rects),
        max(rect.x2 for rect in rects),
        max(rect.y2 for rect in rects),
    )


def _crop(frame: np.ndarray, rect: _Rect) -> np.ndarray:
    """按给定矩形裁剪图像区域。"""
    return frame[rect.y1:rect.y2, rect.x1:rect.x2]


def _relative_rect(parent: _Rect, left: float, top: float, right: float, bottom: float) -> _Rect:
    """在父区域内按相对比例构造子矩形。"""
    return _Rect(
        parent.x1 + int(parent.width * left),
        parent.y1 + int(parent.height * top),
        parent.x1 + int(parent.width * right),
        parent.y1 + int(parent.height * bottom),
    )


def _add_debug_box(
    rect: _Rect,
    *,
    label: str,
    color: tuple[int, int, int],
    alpha: float = 0.18,
    duration: float = 8.0,
) -> None:
    """把矩形区域绘制到 DebugTools 上，便于核对识别范围。"""
    _debugger.add_box(
        rect.x1,
        rect.y1,
        rect.x2,
        rect.y2,
        label=label,
        color=color,
        alpha=alpha,
        duration=duration,
    )


def _extract_ocr_tokens(
    frame: np.ndarray,
    region: _Rect,
    *,
    scale: float = _OCR_SCALE,
    min_confidence: float = _OCR_MIN_CONFIDENCE,
) -> list[_OcrToken]:
    """提取OCR、tokens并返回结果。

    Args:
        frame: 待识别图像帧。
        region: 用于提供region相关输入。
        scale: 用于提供scale相关输入。
        min_confidence: 用于提供min、confidence相关输入。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    crop = _crop(frame, region)
    if crop.size == 0:
        return []

    ocr_frame = crop
    if scale != 1.0:
        ocr_frame = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    result = _ocr.ocr(ocr_frame)
    tokens: list[_OcrToken] = []
    for item in getattr(result, "results", []):
        text = _normalize_text(item.text)
        confidence = _safe_confidence(getattr(item, "confidence", None))
        if not text or confidence < min_confidence:
            continue
        x1 = int(round(item.x / scale)) + region.x1
        y1 = int(round(item.y / scale)) + region.y1
        width = int(round(item.w / scale))
        height = int(round(item.h / scale))
        x2 = x1 + width
        y2 = y1 + height
        tokens.append(_OcrToken(text=text, rect=_Rect(x1, y1, x2, y2), confidence=confidence))

    return sorted(tokens, key=lambda token: (token.rect.cy, token.rect.x1))


def _tokens_to_lines(tokens: Sequence[_OcrToken], y_gap: int = _LINE_MERGE_Y_GAP) -> list[_OcrLine]:
    """将 OCR token 聚合成按行排列的文本。"""
    if not tokens:
        return []

    grouped: list[list[_OcrToken]] = []
    current: list[_OcrToken] = []
    line_anchor: int | None = None
    for token in sorted(tokens, key=lambda item: (item.rect.cy, item.rect.x1)):
        if line_anchor is None or abs(token.rect.cy - line_anchor) <= y_gap:
            if not current:
                line_anchor = token.rect.cy
            current.append(token)
            continue
        grouped.append(sorted(current, key=lambda item: item.rect.x1))
        current = [token]
        line_anchor = token.rect.cy
    if current:
        grouped.append(sorted(current, key=lambda item: item.rect.x1))

    lines: list[_OcrLine] = []
    for group in grouped:
        text = _normalize_text(" ".join(item.text for item in group))
        rect = _union_rect(item.rect for item in group)
        if text and rect is not None:
            lines.append(_OcrLine(text=text, rect=rect, tokens=tuple(group)))
    return lines


def _dedupe_preserve_order(texts: Iterable[str]) -> list[str]:
    """对`preserve_order`去重。"""
    seen: set[str] = set()
    deduped: list[str] = []
    for text in texts:
        normalized = _normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _get_bottom_button_boxes(results: Yolo_Results | None, frame_height: int) -> list[Yolo_Box]:
    """获取bottom、按钮、boxes并返回结果。

    Args:
        results: 用于提供results相关输入。
        frame_height: 用于提供frame、height相关输入。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    if not results:
        return []
    buttons = results.filter_by_label(BaseUILabels.BUTTON)
    return [box for box in buttons.boxes if box.cy >= frame_height * 0.75]


def _locate_recommended_effect_region(
    results: Yolo_Results | None,
    frame_shape: Sequence[int],
) -> _Rect:
    """用属性条、底部按钮与 carousel 顶边夹出推荐效果可点击区域。"""

    frame_h, frame_w = frame_shape[:2]
    attr_boxes = (
        results.filter_by_labels(
            [
                BaseUILabels.PRODUCE_CARD_VOCAL,
                BaseUILabels.PRODUCE_CARD_DANCE,
                BaseUILabels.PRODUCE_CARD_VISUAL,
            ]
        ).boxes
        if results
        else []
    )
    carousel_boxes = (
        results.filter_by_labels(
            [
                BaseUILabels.PRODUCT_CARD_SELECTED,
                BaseUILabels.PRODUCT_CARD_CANDIDATE,
            ]
        ).boxes
        if results
        else []
    )
    bottom_buttons = _get_bottom_button_boxes(results, frame_h)

    attr_bottom = max((int(box.h) for box in attr_boxes), default=int(frame_h * 0.48))
    lower_anchor = min(
        [int(box.y) for box in carousel_boxes] or [int(box.y) for box in bottom_buttons] or [int(frame_h * 0.78)]
    )
    top = attr_bottom + max(20, int(frame_h * 0.02))
    bottom = lower_anchor - max(28, int(frame_h * 0.03))
    if bottom <= top:
        top = int(frame_h * 0.50)
        bottom = int(frame_h * 0.67)

    return _Rect(int(frame_w * 0.02), top, int(frame_w * 0.98), bottom)


def _locate_tooltip_search_region(
    results: Yolo_Results | None,
    recommend_region: _Rect,
    frame_shape: Sequence[int],
) -> _Rect:
    """提示框会出现在推荐效果区上方，搜索区域由同一组锚点反推。"""

    frame_h, frame_w = frame_shape[:2]
    attr_boxes = (
        results.filter_by_labels(
            [
                BaseUILabels.PRODUCE_CARD_VOCAL,
                BaseUILabels.PRODUCE_CARD_DANCE,
                BaseUILabels.PRODUCE_CARD_VISUAL,
            ]
        ).boxes
        if results
        else []
    )
    bottom_buttons = _get_bottom_button_boxes(results, frame_h)
    carousel_boxes = (
        results.filter_by_labels(
            [
                BaseUILabels.PRODUCT_CARD_SELECTED,
                BaseUILabels.PRODUCT_CARD_CANDIDATE,
            ]
        ).boxes
        if results
        else []
    )

    attr_top = min((int(box.y) for box in attr_boxes), default=recommend_region.y1)
    lower_anchor = min(
        [int(box.y) for box in carousel_boxes] or [int(box.y) for box in bottom_buttons] or [recommend_region.y2]
    )
    tooltip_height = max(int(recommend_region.height * 0.95), int(frame_h * 0.12))
    top = max(0, min(attr_top, recommend_region.y1) - tooltip_height)
    bottom = min(frame_h, lower_anchor - max(18, int(frame_h * 0.04)))
    if bottom <= top:
        bottom = recommend_region.y2

    return _Rect(int(frame_w * 0.04), top, int(frame_w * 0.96), bottom)


def _extract_recommended_effect_anchor_lines(frame: np.ndarray, recommend_region: _Rect) -> list[_OcrLine]:
    """从推荐效果区 OCR 出可点击的文本行，避免再扫固定横线。"""

    tokens = _extract_ocr_tokens(frame, recommend_region, scale=_OCR_SCALE)
    lines = _tokens_to_lines(tokens, y_gap=28)
    if not lines:
        return []

    frame_w = frame.shape[1]
    min_width = max(80, int(frame_w * _RECOMMEND_LINE_MIN_WIDTH_RATIO))
    left_label_limit = recommend_region.x1 + int(recommend_region.width * 0.14)
    filtered: list[_OcrLine] = []
    for line in lines:
        if line.rect.width < min_width:
            continue
        if line.rect.x1 <= left_label_limit and line.rect.width < int(recommend_region.width * 0.18):
            continue
        if any(
            keyword in line.text
            for keyword in (ProduceText.TRAINING_INFO, ProduceText.FORMATION_DETAILS, ButtonText.NEXT)
        ):
            continue
        filtered.append(line)
    return filtered or lines


def _build_recommended_effect_candidate_points(
    anchor_lines: Sequence[_OcrLine],
    recommend_region: _Rect,
) -> list[tuple[int, int]]:
    """根据推荐效果文本行推导一组可点击探测点。

    推荐效果的提示气泡并不总是与文本严格对齐，因此这里会围绕每一行文本的右侧、中心
    以及整组文本的联合区域生成多个候选点，后续按顺序点击探测，直到成功弹出 tooltip。
    """
    rects = [line.rect for line in anchor_lines]
    union = _union_rect(rects) or recommend_region
    raw_points: list[tuple[int, int]] = []
    right_side_limit = recommend_region.x2 - max(24, int(recommend_region.width * 0.04))
    for line in anchor_lines:
        raw_points.extend(
            [
                (
                    min(right_side_limit, int(line.rect.x2 + recommend_region.width * 0.12)),
                    line.rect.cy,
                ),
                (
                    min(right_side_limit, int(line.rect.x2 + recommend_region.width * 0.24)),
                    line.rect.cy,
                ),
                (line.rect.cx, line.rect.cy),
            ]
        )
    raw_points.extend(
        [
            (int(union.x1 + union.width * 0.72), union.cy),
            (union.cx, union.cy),
            (union.cx, int(union.y1 + union.height * 0.62)),
        ]
    )

    deduped: list[tuple[int, int]] = []
    for point in raw_points:
        if any(abs(point[0] - existing[0]) <= 24 and abs(point[1] - existing[1]) <= 24 for existing in deduped):
            continue
        deduped.append(point)
    return deduped


def _extract_recommended_effect_lines_from_frame(
    frame: np.ndarray,
    tooltip_search_region: _Rect,
) -> tuple[_Rect | None, list[str]]:
    """提示框打开后，通过 OCR 密集区域反推出提示框内容框。"""

    tokens = _extract_ocr_tokens(frame, tooltip_search_region, scale=_OCR_SCALE)
    lines = _tokens_to_lines(tokens)
    left_anchor_limit = tooltip_search_region.x1 + int(tooltip_search_region.width * 0.45)
    min_cluster_width = max(120, int(tooltip_search_region.width * 0.12))
    candidate_lines = [
        line
        for line in lines
        if line.rect.x1 <= left_anchor_limit and line.rect.width >= min_cluster_width
    ]
    union = _union_rect(line.rect for line in candidate_lines)
    if (
        len(tokens) < _TOOLTIP_MIN_TOKEN_COUNT
        or len(candidate_lines) < _TOOLTIP_MIN_LINE_COUNT
        or union is None
        or union.height < int(frame.shape[0] * 0.10)
    ):
        return None, []

    tooltip_box = union.expand(_TOOLTIP_BOX_PADDING_X, _TOOLTIP_BOX_PADDING_Y, frame.shape)
    refined_lines = _tokens_to_lines(_extract_ocr_tokens(frame, tooltip_box, scale=_OCR_SCALE))

    texts: list[str] = []
    min_width = max(72, int(tooltip_box.width * 0.08))
    for line in refined_lines:
        text = line.text.strip(" |")
        if not text:
            continue
        visible_chars = re.sub(r"[^0-9A-Za-zぁ-ゖァ-ヴー一-龯々ヶ+％%]", "", text)
        if len(visible_chars) < 2 and not re.search(r"\d", text):
            continue
        if line.rect.width < min_width and not re.search(r"[+＋]\d", text):
            continue
        texts.append(text)

    deduped = _dedupe_preserve_order(texts)
    if len(deduped) < _TOOLTIP_MIN_LINE_COUNT:
        return None, []
    return tooltip_box, deduped


def _extract_training_body_tokens(body_frame: np.ndarray) -> list[_OcrToken]:
    """对育成信息弹窗正文区域做 OCR，并返回标准化 token 列表。"""
    full_rect = _Rect(0, 0, body_frame.shape[1], body_frame.shape[0])
    return _extract_ocr_tokens(body_frame, full_rect, scale=_OCR_SCALE)


def _locate_training_task_region(body_frame: np.ndarray) -> _Rect:
    """根据「育成課題」标题定位任务区，而不是裁固定像素。"""

    height, width = body_frame.shape[:2]
    tokens = _extract_training_body_tokens(body_frame)
    heading = next((token for token in tokens if ProduceText.TRAINING_TASKS in token.text), None)
    top = heading.rect.y2 + max(14, int(height * 0.02)) if heading else int(height * 0.45)
    return _Rect(int(width * 0.06), top, int(width * 0.94), int(height * 0.98))


def _read_exam_criteria_from_body(body_frame: np.ndarray) -> Dict[str, Any]:
    """从育成信息弹窗正文中提取审查基准摘要。

    当前会解析两类信息：
    - `target_score`：审查分数目标。
    - `priority`：按横向位置推导出的属性优先级顺序。

    这些信息会和育成课题一起写入 `ctx.idol_page_info`，供后续诊断和策略分析使用。
    """
    tokens = _extract_training_body_tokens(body_frame)
    task_heading_top = min(
        (token.rect.y1 for token in tokens if ProduceText.TRAINING_TASKS in token.text),
        default=body_frame.shape[0],
    )
    exam_heading_bottom = max(
        (
            token.rect.y2
            for token in tokens
            if ProduceText.FINAL_EXAM in token.text or ProduceText.EXAM_CRITERIA in token.text
        ),
        default=0,
    )
    exam_tokens = [
        token for token in tokens
        if exam_heading_bottom <= token.rect.y1 < task_heading_top
    ]

    exam: Dict[str, Any] = {"target_score": None, "priority": []}
    for token in exam_tokens:
        digits = re.search(r"\d{2,4}", token.text.replace("O", "0").replace("o", "0"))
        if digits is None:
            continue
        value = int(digits.group())
        if 50 <= value <= 9999:
            exam["target_score"] = value
            break

    params: list[tuple[int, str]] = []
    for token in exam_tokens:
        canonical = _PARAM_CANONICAL.get(token.text)
        if canonical:
            params.append((token.rect.cx, canonical))
    params.sort(key=lambda item: item[0])
    exam["priority"] = _dedupe_preserve_order(name for _, name in params)
    return exam


def _parse_tasks_from_body_frame(
    body_frame: np.ndarray,
    seen_conditions: set[str],
) -> tuple[list[Dict[str, Any]], _Rect]:
    """解析`tasks_from_body_frame`。"""
    task_region = _locate_training_task_region(body_frame)
    task_lines = _tokens_to_lines(_extract_ocr_tokens(body_frame, task_region, scale=_OCR_SCALE))

    tasks: list[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in task_lines:
        text = line.text
        if not text or ProduceText.TRAINING_TASKS in text:
            continue

        match = _TASK_CONDITION_RE.search(text)
        if match:
            if current.get("condition"):
                _finalize_task(current, tasks, seen_conditions)
            param_jp = match.group(1)
            current = {
                "condition": text,
                "param": _PARAM_CANONICAL.get(param_jp, param_jp),
                "threshold": int(match.group(2)),
                "comparison": match.group(3),
                "type": "",
                "reward": "",
            }
            continue

        if not current.get("condition"):
            continue

        task_type = _TASK_TYPE_VARIANTS.get(text)
        if task_type:
            current["type"] = task_type
            continue

        if any(token in text for token in _TASK_REWARD_TOKENS):
            current["reward"] = text
            _finalize_task(current, tasks, seen_conditions)
            current = {}
            continue

        if not current.get("type"):
            current["type"] = text

    if current.get("condition"):
        _finalize_task(current, tasks, seen_conditions)

    return tasks, task_region


def _resize_pair_for_ssim(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将两张图缩放到可比较尺寸以计算 SSIM。"""
    height = min(first.shape[0], second.shape[0])
    width = min(first.shape[1], second.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("SSIM 输入区域尺寸无效")
    if first.shape[:2] != (height, width):
        first = cv2.resize(first, (width, height), interpolation=cv2.INTER_AREA)
    if second.shape[:2] != (height, width):
        second = cv2.resize(second, (width, height), interpolation=cv2.INTER_AREA)
    return first, second


def _get_training_info_modal_body(
    app: "AppProcessor",
) -> tuple[Any, _Rect, np.ndarray] | None:
    """获取育成信息弹窗的正文区域与对应截图。

    Returns:
        tuple[Any, _Rect, np.ndarray] | None: 返回原始 modal、正文区域矩形以及正文截图副本；
            若当前没有弹窗、没有正文框，或正文区域无效，则返回 None。
    """
    modal = app.game_utils.try_get_modal(no_body=True)
    if modal is None or modal.body_box is None or modal.body_box.frame is None:
        return None
    body_rect = _Rect.from_yolo_box(modal.body_box)
    body_frame = modal.body_box.frame.copy()
    if body_frame.size == 0 or body_rect.width <= 0 or body_rect.height <= 0:
        return None
    return modal, body_rect, body_frame


def _resolve_tooltip_dismiss_point(results: Yolo_Results | None) -> tuple[int, int] | None:
    """基于当前选中偶像卡推导 tooltip 的安全关闭点。"""

    if not results:
        return None
    selected_box = results.filter_by_label(BaseUILabels.PRODUCT_CARD_SELECTED).first()
    if selected_box is None:
        return None
    selected_rect = _Rect.from_yolo_box(selected_box)
    return (
        int(selected_rect.x1 + selected_rect.width * 0.36),
        int(selected_rect.y1 + selected_rect.height * 0.28),
    )


def _resolve_modal_overlay_dismiss_point(modal: Any, frame_shape: Sequence[int]) -> tuple[int, int] | None:
    """优先点击 modal 面板外侧的蒙层，避免再回落到固定点位。"""

    panel_box = getattr(modal, "panel_box", None)
    if panel_box is None:
        return None
    panel_rect = _Rect.from_yolo_box(panel_box)
    frame_h, _frame_w = frame_shape[:2]
    dismiss_x = max(8, panel_rect.x1 // 2)
    dismiss_y = min(frame_h - 8, int(panel_rect.y1 + panel_rect.height * 0.22))
    return dismiss_x, dismiss_y


def _is_idol_page_layout_visible(results: Yolo_Results | None) -> bool:
    """判断当前检测结果是否仍像偶像卡选择页的布局。

    只要能看到当前已选偶像卡，或同时看到 vocal/dance/visual 三个属性条，
    就认为页面仍停留在偶像卡页，可继续执行推荐效果与育成信息采集。
    """
    if not results:
        return False
    if results.exists_label(BaseUILabels.PRODUCT_CARD_SELECTED):
        return True
    return results.exists_all_labels(
        [
            BaseUILabels.PRODUCE_CARD_VOCAL,
            BaseUILabels.PRODUCE_CARD_DANCE,
            BaseUILabels.PRODUCE_CARD_VISUAL,
        ]
    )


class CollectIdolPageInfoStep(ProduceStep):
    """诊断性采集偶像页信息，辅助调试识别链路与数据校验。"""

    step_name = "collect_idol_page_info"

    def validate(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """确认当前页面仍是可采集信息的偶像卡选择页。"""
        return _is_idol_page_layout_visible(getattr(app, "latest_results", None))

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """采集偶像页的推荐效果与育成信息，然后推进到支援卡编成页。

        这是一个带诊断性质的补充步骤：即便其中某项采集失败，也不会中断主流程，
        而是记录告警后继续点击「次へ」。只有在页面迟迟没有进入支援卡编成页时，
        才会抛出超时异常。
        """
        try:
            self._collect_recommended_effects(app, ctx)
        except Exception as exc:
            logger.warning(f"おすすめ効果采集失败: {exc}")

        try:
            self._collect_training_info(app, ctx)
        except Exception as exc:
            logger.warning(f"育成情報采集失败: {exc}")

        # 点击“次へ”进入支援卡编成。
        app.game_utils.click_button(
            ButtonText.NEXT,
            match_config=MatchConfig(fuzz_threshold=80),
        )
        app.game_utils.wait_loading()

        for _ in range(15):
            if app.latest_results.exists_label(BaseUILabels.SUPPORT_CARD):
                logger.debug("成功进入支援卡编成页")
                return True
            if app.latest_results.exists_label(BaseUILabels.BLANK_SLOT):
                logger.debug("进入支援卡编成页（存在空白槽位）")
                return True
            sleep(1)

        raise TimeoutError("等待支援卡编成页超时")

    def _clear_recommendation_tooltip(self, app: "AppProcessor") -> None:
        """处理clear、recommendation、tooltip并返回结果。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

        Returns:
            None: 仅产生副作用，不返回业务值。
        """
        dismiss_point = _resolve_tooltip_dismiss_point(getattr(app, "latest_results", None))
        if dismiss_point is None:
            logger.debug("未找到选中偶像卡框，跳过 tooltip 清场点击")
            return
        app.device.click(
            dismiss_point[0],
            dismiss_point[1],
            "clear-recommended-effect-tooltip",
        )
        sleep(0.25)

    def _dismiss_unexpected_recommend_overlay(self, app: "AppProcessor") -> bool:
        """若误点进整页详情，先关闭再继续试下一个候选点。"""

        results = getattr(app, "latest_results", None)
        close_btn = find_button(app, ButtonText.CLOSE, fuzz_threshold=70)
        if close_btn is None:
            return False
        has_next = find_button(app, ButtonText.NEXT, fuzz_threshold=75) is not None
        has_modal_header = bool(results and results.exists_label(BaseUILabels.MODAL_HEADER))
        has_close_label = bool(results and results.exists_label(BaseUILabels.CLOSE_BUTTON))
        if has_next and _is_idol_page_layout_visible(results) and not (has_modal_header or has_close_label):
            return False
        app.device.click_element(close_btn)
        sleep(0.5)
        wait_frame_stable(app, timeout=2.0)
        return True

    def _restore_after_recommend_probe(self, app: "AppProcessor") -> bool:
        """把推荐效果探测后遗留的弹层收回偶像页。"""

        cancel_btn = find_button(app, ButtonText.CANCEL, fuzz_threshold=70)
        confirm_btn = find_button(app, ButtonText.CONFIRM, fuzz_threshold=70)
        if cancel_btn is not None and confirm_btn is not None:
            app.device.click_element(cancel_btn)
            sleep(0.5)
            wait_frame_stable(app, timeout=2.5)
            return True
        if self._dismiss_unexpected_recommend_overlay(app):
            return True
        return False

    def _collect_recommended_effects(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
    ) -> None:
        """从推荐效果区 OCR 出点击锚点，再展开提示框读取效果列表。"""

        if not self._restore_after_recommend_probe(app):
            self._clear_recommendation_tooltip(app)
            wait_frame_stable(app, timeout=2.0)
        baseline = app.device.capture()

        recommend_region = _locate_recommended_effect_region(app.latest_results, baseline.shape)
        tooltip_search_region = _locate_tooltip_search_region(
            app.latest_results,
            recommend_region,
            baseline.shape,
        )
        _add_debug_box(recommend_region, label="recommend band", color=(255, 200, 0))
        _add_debug_box(tooltip_search_region, label="tooltip search", color=(0, 200, 255))

        anchor_lines = _extract_recommended_effect_anchor_lines(baseline, recommend_region)
        for index, line in enumerate(anchor_lines, start=1):
            _add_debug_box(
                line.rect,
                label=f"recommend anchor {index}",
                color=(255, 0, 255),
                alpha=0.14,
            )

        candidate_points = _build_recommended_effect_candidate_points(anchor_lines, recommend_region)
        effects: list[str] = []
        tooltip_box: _Rect | None = None
        for index, (click_x, click_y) in enumerate(candidate_points, start=1):
            self._clear_recommendation_tooltip(app)
            app.device.click(click_x, click_y, f"recommended-effect-{index}")
            sleep(0.55)
            wait_frame_stable(app, timeout=2.0)
            if self._dismiss_unexpected_recommend_overlay(app):
                logger.debug(
                    f"おすすめ効果点击落入详情页，已关闭并继续尝试: point=({click_x},{click_y})"
                )
                continue
            frame = app.device.capture()
            tooltip_box, effects = _extract_recommended_effect_lines_from_frame(frame, tooltip_search_region)
            if not effects:
                continue
            logger.debug(
                f"おすすめ効果点击命中: point=({click_x},{click_y}), lines={len(effects)}"
            )
            break

        if tooltip_box is not None:
            _add_debug_box(
                tooltip_box,
                label="tooltip OCR",
                color=(80, 220, 120),
                alpha=0.14,
            )

        self._clear_recommendation_tooltip(app)
        wait_frame_stable(app, timeout=2.0)

        if not effects:
            effects = _dedupe_preserve_order(line.text for line in anchor_lines)
            logger.warning("提示框未成功展开，回退为页面可见推荐效果文本")

        ctx.recommended_effects = effects
        logger.info(f"おすすめ効果采集完成: {len(effects)} 条")
        for index, effect in enumerate(effects, start=1):
            logger.debug(f"  効果{index}: {effect}")

    def _collect_training_info(
        self,
        app: "AppProcessor",
        ctx: "ProduceContext",
    ) -> None:
        """打开育成情報面板，采集审查基准和育成课题。"""

        self._restore_after_recommend_probe(app)
        btn = find_button(app, ProduceText.TRAINING_INFO, fuzz_threshold=65)
        if btn is None:
            logger.warning("未找到育成情報按钮")
            return

        app.device.click_element(btn)
        sleep(1.0)
        wait_frame_stable(app, timeout=3.0)

        modal_payload = _get_training_info_modal_body(app)
        if modal_payload is None:
            logger.warning("育成情報面板未打开")
            return

        modal, body_rect, body_frame = modal_payload
        if getattr(modal, "panel_box", None) is not None:
            _add_debug_box(
                _Rect.from_yolo_box(modal.panel_box),
                label="training panel",
                color=(0, 255, 255),
            )
        if getattr(modal, "header_box", None) is not None:
            _add_debug_box(
                _Rect.from_yolo_box(modal.header_box),
                label="training header",
                color=(0, 165, 255),
            )
        _add_debug_box(body_rect, label="training body", color=(255, 215, 0))

        try:
            exam = self._read_exam_criteria(body_frame)
            ctx.exam_criteria = exam
            logger.info(f"审查基准: {exam}")

            tasks = self._read_training_tasks_with_scroll(app)
            ctx.training_tasks = tasks
            logger.info(f"育成课题: {len(tasks)} 条")
            for task in tasks:
                logger.debug(f"  课题: {task}")
        finally:
            self._close_training_info(app)

    def _read_exam_criteria(self, body_frame: np.ndarray) -> Dict[str, Any]:
        """读取「最終試験の審査基準」: 目标分数 + 参数优先级。"""

        return _read_exam_criteria_from_body(body_frame)

    def _read_training_tasks_with_scroll(
        self,
        app: "AppProcessor",
    ) -> List[Dict[str, Any]]:
        """读取育成课题列表，滚动逻辑全部基于 modal body 的相对区域。"""

        all_tasks: list[Dict[str, Any]] = []
        seen_conditions: set[str] = set()

        for scroll_round in range(6):
            modal_payload = _get_training_info_modal_body(app)
            if modal_payload is None:
                break
            _modal, body_rect, body_frame = modal_payload
            new_tasks, task_local_region = _parse_tasks_from_body_frame(body_frame, seen_conditions)
            all_tasks.extend(new_tasks)

            compare_local = _relative_rect(
                _Rect(0, 0, body_rect.width, body_rect.height),
                0.06,
                _SCROLL_COMPARE_TOP_RATIO,
                0.94,
                _SCROLL_COMPARE_BOTTOM_RATIO,
            )
            if task_local_region.y1 > compare_local.y1:
                compare_local = _Rect(
                    compare_local.x1,
                    task_local_region.y1,
                    compare_local.x2,
                    compare_local.y2,
                )
            before_crop = _crop(body_frame, compare_local)
            global_task_region = task_local_region.translate(body_rect.x1, body_rect.y1)
            global_compare_region = compare_local.translate(body_rect.x1, body_rect.y1)
            _add_debug_box(
                global_task_region,
                label=f"task OCR:{scroll_round}",
                color=(200, 100, 255),
                alpha=0.10,
                duration=3.0,
            )
            _add_debug_box(
                global_compare_region,
                label=f"scroll SSIM:{scroll_round}",
                color=(100, 200, 255),
                alpha=0.14,
                duration=3.0,
            )

            if scroll_round > 0 and not new_tasks:
                break

            scroll_x = body_rect.cx
            start_y = int(body_rect.y1 + body_rect.height * _SCROLL_START_RATIO)
            end_y = int(body_rect.y1 + body_rect.height * _SCROLL_END_RATIO)
            inertial_swipe(
                app,
                scroll_x,
                start_y,
                scroll_x,
                end_y,
                duration=0.35,
                settle_timeout=2.5,
            )
            wait_frame_stable(app, timeout=2.5)

            after_payload = _get_training_info_modal_body(app)
            if after_payload is None:
                break
            _after_modal, after_body_rect, after_body_frame = after_payload
            after_compare_local = _relative_rect(
                _Rect(0, 0, after_body_rect.width, after_body_rect.height),
                0.06,
                _SCROLL_COMPARE_TOP_RATIO,
                0.94,
                _SCROLL_COMPARE_BOTTOM_RATIO,
            )
            after_crop = _crop(after_body_frame, after_compare_local)
            before_crop, after_crop = _resize_pair_for_ssim(before_crop, after_crop)
            ssim = compute_ssim_score(before_crop, after_crop)
            if ssim >= _SCROLL_SETTLE_SSIM_THRESHOLD:
                extra_tasks, _ = _parse_tasks_from_body_frame(after_body_frame, seen_conditions)
                all_tasks.extend(extra_tasks)
                break

        return all_tasks

    def _close_training_info(self, app: "AppProcessor") -> None:
        """关闭training、info并返回结果。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

        Returns:
            None: 仅产生副作用，不返回业务值。
        """

        btn = find_button(app, ButtonText.CLOSE, fuzz_threshold=70)
        if btn is not None:
            app.device.click_element(btn)
            sleep(0.5)
            wait_frame_stable(app, timeout=2.0)
            return
        modal = app.game_utils.try_get_modal(no_body=True)
        dismiss_point = _resolve_modal_overlay_dismiss_point(modal, app.latest_frame.shape) if modal else None
        if dismiss_point is None:
            raise RuntimeError(i18n_text("backend.task.trainingInfoCloseButtonNotFound", fallback="未找到育成情報关闭按钮，也无法推导蒙层关闭点"))
        app.device.click(dismiss_point[0], dismiss_point[1], "close-training-info-fallback")
        sleep(0.5)
        wait_frame_stable(app, timeout=2.0)


def _finalize_task(
    task: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    seen: set[str],
) -> None:
    """处理finalize、task并返回结果。

    Args:
        task: 用于提供task相关输入。
        tasks: 用于提供tasks相关输入。
        seen: 用于提供seen相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    cond = task.get("condition", "")
    if cond and cond not in seen:
        seen.add(cond)
        tasks.append(dict(task))
