"""考试排名检测。

考试战斗画面顶部有一个排名条（leaderboard bar），展示 6 位参赛者的排名：
  - 每个参赛者占一个卡片，显示序号（1st, 2nd, 3rd...）、头像、分数
  - 玩家自己的卡片有白边 + 彩色渐变 + 较大的序号文字
  - 序号文字以英文序数形式显示（1st, 2nd, 3rd, 4th, 5th, 6th）

检测原理：
  - 通过 YOLO 检测的 Bonus Indicator / Stamina 作为锚点定位排名条区域
  - OCR 该区域后，用锚点 y 坐标过滤出序号所在的行
  - 玩家自己的序号文字字体明显更大 → 对应 OCR bounding box 面积最大
  - 通过比较 bounding box 面积，找出最大的序号即为玩家当前排名

提取结果：
  - rank: 玩家当前排名 (1-6)
  - confidence: 置信度 (high/medium/low)
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.inference.ocr_engine import OCRService
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.entity.Yolo import Yolo_Results
    from src.core.tasks.producer_challenge.context import ProduceContext

# ── 序号数字正则（提取文本中第一个数字） ──
_FIRST_DIGIT_RE = re.compile(r"(\d)")

# ── OCR 常见字符混淆映射（如 "1" ↔ "L"/"l"/"I"） ──
_OCR_CHAR_CONFUSION = {"L": "1", "l": "1", "I": "1", "O": "0", "o": "0"}

# ── OCR 数字混淆：提取到的数字不在有效范围时尝试替换 ──
# 0 和 6 形状相似，OCR 经常把游戏字体的 "6" 识别为 "O" → 映射 "0"
_DIGIT_FALLBACK = {0: 6, 9: 6}

# ── 排除非序号文本的模式 ──
_EXCLUDE_PATTERNS = [
    re.compile(r"[%％]"),                                  # 百分比数字
    re.compile(r"^\d{3,}$"),                               # 纯3位以上数字（分数）
    re.compile(r"[ターカルビジュアダンスボ残り張]"),       # 日文 UI 文字
]

# ── 有效排名范围 ──
_MIN_RANK = 1
_MAX_RANK = 6

# ── 面积比阈值 ──
_HIGH_CONF_RATIO = 1.5
_MEDIUM_CONF_RATIO = 1.1

_ocr_service = OCRService()


def _is_ordinal_candidate(text: str) -> bool:
    """判断 OCR 文本是否可能是排名序号（排除百分比、分数、日文标签等）。"""
    for pat in _EXCLUDE_PATTERNS:
        if pat.search(text):
            return False
    return True


def _extract_ordinal_digit(text: str) -> Optional[int]:
    """从 OCR 文本中提取序号数字（1-6），处理常见字符混淆。

    处理两层混淆：
      1. 字符混淆：OCR 把 "1st" 读成 "Lst"（L→1）
      2. 数字混淆：OCR 把 "6th" 读成 "Oth" → "0th"（0→6）
    """
    def _try_extract(t: str) -> Optional[int]:
        """处理try、extract并返回结果。

        Args:
            t: 用于提供t相关输入。

        Returns:
            Optional[int]: 返回值类型见注解。
        """
        m = _FIRST_DIGIT_RE.search(t)
        if not m:
            return None
        d = int(m.group(1))
        if _MIN_RANK <= d <= _MAX_RANK:
            return d
        # 数字不在有效范围，尝试形状相似的数字替换
        return _DIGIT_FALLBACK.get(d)

    # 先直接提取
    result = _try_extract(text)
    if result is not None:
        return result
    # 尝试字符混淆修正后再提取
    corrected = "".join(_OCR_CHAR_CONFUSION.get(c, c) for c in text)
    return _try_extract(corrected)


def _find_anchor_boxes(yolo_results: "Yolo_Results") -> tuple[Optional[dict], Optional[dict]]:
    """从 YOLO 检测结果中找到 Bonus Indicator 和 Stamina 锚点。

    返回 (bonus_anchor, stamina_anchor)，各为 {"y": y1, "h": height} 或 None。
    Yolo_Box 中 x/y = 左上角, w/h = 右下角坐标。
    """
    bonus = stamina = None
    for box in yolo_results.boxes:
        if box.label == ProducerLabels.PC_BONUS_INDICATOR:
            bonus = {"y": int(box.y), "h": int(box.h - box.y)}
        elif box.label == ProducerLabels.PC_STAMINA:
            stamina = {"y": int(box.y), "h": int(box.h - box.y)}
    return bonus, stamina


def extract_exam_ranking(
    frame: np.ndarray,
    yolo_results: "Yolo_Results",
) -> Optional[dict]:
    """从考试战斗全屏帧提取玩家当前排名。

    使用 YOLO 检测的 Bonus Indicator 和 Stamina 作为锚点定位排名条：
      1. 找 YOLO 锚点确定排名条 y 坐标范围
      2. OCR 从画面顶部到锚点下方的区域
      3. 用锚点 y 坐标过滤出序号行的 OCR 结果
      4. 找 bounding box 面积最大的序号 → 即玩家排名

    Args:
        frame: 全屏游戏帧（BGR numpy 数组）
        yolo_results: 当前帧的 YOLO 检测结果

    返回:
      {"rank": 2, "confidence": "high"} 或 None。
    """
    if frame is None or not hasattr(frame, "shape"):
        return None
    if yolo_results is None:
        return None

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return None

    # 1. 找 YOLO 锚点
    bonus, stamina = _find_anchor_boxes(yolo_results)
    if not bonus and not stamina:
        logger.debug("[考试排名] 未检测到 Bonus Indicator 或 Stamina 锚点")
        return None

    # 2. 确定 OCR 区域和序号 y 过滤范围
    if bonus:
        bi_y = bonus["y"]
        bi_h = bonus["h"]
        # 序号在 Bonus Indicator 的 y 坐标附近（上下各一个 BI 高度）
        ordinal_y_min = max(0, bi_y - bi_h)
        ordinal_y_max = bi_y + bi_h
        # OCR 区域: 从顶部到 Stamina 或 BI 下方
        ocr_y_end = stamina["y"] if stamina else bi_y + bi_h * 2
    else:
        # 仅有 Stamina 时：序号在 Stamina 上方的 1/3 到 2/3 区域
        st_y = stamina["y"]
        ordinal_y_min = int(st_y * 0.3)
        ordinal_y_max = int(st_y * 0.7)
        ocr_y_end = st_y

    # 3. 裁剪并 OCR
    ocr_region = frame[0:int(ocr_y_end), :]
    if ocr_region.size == 0:
        return None

    ocr_results = _ocr_service.ocr(ocr_region)
    if not ocr_results or not ocr_results.results:
        logger.debug("[考试排名] OCR 未检测到任何文字")
        return None

    # 4. 用 y 坐标 + 文本模式过滤出序号候选
    ordinals: list[tuple[int, int, str]] = []  # (数字, 面积, 原始文本)

    for r in ocr_results.results:
        # y 范围过滤
        r_center_y = r.y + r.h // 2
        if not (ordinal_y_min <= r_center_y <= ordinal_y_max):
            continue
        # 文本模式过滤
        if not _is_ordinal_candidate(r.text):
            continue
        # 提取序号数字（含 OCR 混淆修正）
        digit = _extract_ordinal_digit(r.text)
        if digit is not None:
            area = r.w * r.h
            ordinals.append((digit, area, r.text))

    if not ordinals:
        logger.debug("[考试排名] 未检测到有效序号 (1-6)")
        return None

    # 5. 按面积降序排列，最大面积 = 玩家序号
    ordinals.sort(key=lambda x: x[1], reverse=True)
    best_digit, best_area, _ = ordinals[0]

    # 6. 计算置信度
    if len(ordinals) >= 2:
        second_area = ordinals[1][1]
        if best_area == second_area:
            # 平局：返回最小序号（保守策略）
            tied = [o for o in ordinals if o[1] == best_area]
            best_digit = min(t[0] for t in tied)
            confidence = "medium"
        elif best_area >= second_area * _HIGH_CONF_RATIO:
            confidence = "high"
        elif best_area >= second_area * _MEDIUM_CONF_RATIO:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "low"

    logger.debug(
        f"[考试排名] 检测结果: 第{best_digit}位, "
        f"置信度={confidence}, 面积={best_area}, "
        f"序号数={len(ordinals)}"
    )

    return {
        "rank": best_digit,
        "confidence": confidence,
    }


def store_exam_ranking(ctx: "ProduceContext", info: dict) -> None:
    """保存考试、排名并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        info: 用于提供info相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    ctx.handler_state["exam_ranking_info"] = info


def get_exam_ranking(ctx: "ProduceContext") -> Optional[dict]:
    """获取考试、排名并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        Optional[dict]: 返回值类型见注解，语义由函数用途决定。
    """
    return ctx.handler_state.get("exam_ranking_info")


def get_exam_ranking_value(ctx: "ProduceContext") -> str:
    """获取排名数字的字符串形式（供 LLM prompt 使用）。

    返回 "1"~"6" 或空字符串。
    """
    info = get_exam_ranking(ctx)
    if info and info.get("rank"):
        return str(info["rank"])
    return ""
