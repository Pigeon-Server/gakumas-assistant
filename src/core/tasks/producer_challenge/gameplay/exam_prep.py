"""考试准备页面加成倍率提取。

考试开始前会显示一个参数加成倍率预览页面，展示：
  - 审查基准 + 合格条件
  - 三个参数 (Vocal/Dance/Visual) 的基值和加成百分比
  - 需要点击画面继续

该页面 YOLO 无法检测（0 个标签），通过 HoughCircles 定位参数图标 +
OCR 提取数值来实现自适应检测和提取，不依赖固定坐标。
提取的加成倍率存储在 `ctx.handler_state["exam_prep_bonuses"]` 中，
供考试阶段 LLM 决策使用，无需每回合重复 OCR 加成指示器。
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from src.core.tasks.producer_challenge.shared.common import ocr_text
from src.utils.logger import logger
from src.utils.string_tools import fullwidth_to_halfwidth

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext

# ── 固定比例 fallback（仅当 HoughCircles 失败时使用）──
_FALLBACK_BONUS_REGION_Y = (0.64, 0.83)


def _find_param_row_anchors(frame: np.ndarray) -> Optional[list[int]]:
    """使用 HoughCircles 定位三个参数图标的 Y 坐标。

    通过检测画面下半部分左侧的三个等距、X对齐、半径一致的圆形图标，
    动态确定 Vocal / Dance / Visual 行位置，不依赖固定坐标比例。

    返回 [vocal_y, dance_y, visual_y]（按 Y 从上到下排序），失败返回 None。
    """
    h, w = frame.shape[:2]

    # 搜索区域：下半部分左侧（图标位于画面左边 ~12% 处）
    y1, y2 = int(h * 0.55), int(h * 0.95)
    x1, x2 = 0, int(w * 0.20)
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 多轮检测：逐步降低累加器阈值直到找到足够多的圆
    circles = None
    for p2 in (30, 25, 20, 15):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=80,
            param2=p2,
            minRadius=12,
            maxRadius=45,
        )
        if circles is not None and len(circles[0]) >= 3:
            break
    else:
        return None

    all_circles = np.round(circles[0]).astype(int)
    candidates = [
        {"center_x": int(cx + x1), "center_y": int(cy + y1), "radius": int(r)}
        for cx, cy, r in all_circles
    ]

    if len(candidates) < 3:
        return None

    # 遍历所有3元组合，按 X对齐 + 等距性 + 半径一致性 综合评分
    best_group = None
    best_score = float("inf")

    for combo in combinations(range(len(candidates)), 3):
        group = sorted(
            [candidates[i] for i in combo], key=lambda c: c["center_y"]
        )
        radii = [c["radius"] for c in group]
        xs = [c["center_x"] for c in group]
        ys = [c["center_y"] for c in group]

        # 半径一致性
        r_std = float(np.std(radii))
        if r_std > 8:
            continue

        # X 对齐
        x_std = float(np.std(xs))
        if x_std > 30:
            continue

        # 等距性
        gap1 = ys[1] - ys[0]
        gap2 = ys[2] - ys[1]
        if gap1 < 30 or gap2 < 30:
            continue
        spacing_ratio = min(gap1, gap2) / max(gap1, gap2)
        if spacing_ratio < 0.6:
            continue

        score = r_std * 2 + x_std * 1 + (1 - spacing_ratio) * 100
        if score < best_score:
            best_score = score
            best_group = group

    if best_group is None:
        return None

    result = [c["center_y"] for c in best_group]
    logger.debug(
        f"[考试准备] HoughCircles 动态锚点: "
        f"Vocal={result[0]} Dance={result[1]} Visual={result[2]}"
    )
    return result


def _extract_pcts_from_crop(crop: np.ndarray) -> Optional[list[int]]:
    """从参数加成区裁切中提取三个有效百分比。

    使用 2-4 位数字正则 `(\\d{2,4})%`，支持三位 (393%) 和四位 (1234%) 加成，
    同时兼容审查条件数字粘连（如 "150393%" → 匹配 "0393" → int=393）。

    成功返回 [vocal_pct, dance_pct, visual_pct]，失败返回 None。
    """
    text = fullwidth_to_halfwidth(ocr_text(crop))
    logger.debug(f"[考试准备] 参数加成区 OCR: {text!r}")
    # (\d{2,4})% 匹配 2~4 位百分比，覆盖高加成场景
    pct_matches = re.findall(r"(\d{2,4})%", text)
    valid = [int(p) for p in pct_matches if 50 <= int(p) <= 9999]
    if len(valid) == 3:
        return valid
    logger.warning(
        f"[考试准备] 加成百分比提取异常: 期望3个，实际{len(valid)}个 "
        f"(原始={pct_matches}, 有效={valid})"
    )
    return None


def extract_exam_prep_bonuses(
    frame: np.ndarray,
) -> Optional[dict]:
    """从考试准备页面帧提取三个参数的加成倍率。

    检测策略：
      1. HoughCircles 动态定位参数图标行 → 自适应裁切 OCR
      2. 失败时回退到固定比例裁切
      3. OCR 失败时使用 CLAHE 增强重试

    返回:
      {
          "vocal_bonus_pct": 393,
          "dance_bonus_pct": 527,
          "visual_bonus_pct": 701,
          "vocal_base": 261,      # 可选
          "dance_base": 337,      # 可选
          "visual_base": 455,     # 可选
          "pass_condition": "3位以上",  # 可选
      }
    提取失败返回 None。
    """
    if frame is None or not hasattr(frame, "shape"):
        return None
    h, w = frame.shape[:2]

    # ── 审查条件区：提取合格条件 ──
    # 条件区在参数图标上方，使用宽范围 OCR + 正则提取
    criteria_crop = frame[int(h * 0.50):int(h * 0.65), :]
    criteria_text = fullwidth_to_halfwidth(ocr_text(criteria_crop))
    pass_match = re.search(r"(\d+)位以上", criteria_text)
    pass_condition = pass_match.group(0) if pass_match else ""

    # ── 参数加成区：动态定位 → OCR ──
    anchors = _find_param_row_anchors(frame)
    if anchors is not None:
        # 基于图标间距计算 OCR 区域（图标上方半个间距 ~ 最下图标下方半个间距）
        spacing = (anchors[2] - anchors[0]) / 2
        pad = int(spacing * 0.5)
        bonus_y1 = max(0, anchors[0] - pad)
        bonus_y2 = min(h, anchors[2] + pad)
        logger.debug(
            f"[考试准备] 动态裁切区域: y=[{bonus_y1},{bonus_y2}] "
            f"(占比 {bonus_y1/h*100:.1f}%~{bonus_y2/h*100:.1f}%)"
        )
    else:
        # HoughCircles 失败，回退到固定比例
        logger.debug("[考试准备] HoughCircles 未定位到图标，使用固定比例 fallback")
        bonus_y1 = int(h * _FALLBACK_BONUS_REGION_Y[0])
        bonus_y2 = int(h * _FALLBACK_BONUS_REGION_Y[1])

    bonus_crop = frame[bonus_y1:bonus_y2, :]
    valid_pcts = _extract_pcts_from_crop(bonus_crop)

    # 首次失败时使用 CLAHE 对比度增强重试（覆盖低饱和/灰度等极端场景）
    if valid_pcts is None:
        gray = cv2.cvtColor(bonus_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        logger.debug("[考试准备] 首次提取失败，使用 CLAHE 增强后重试")
        valid_pcts = _extract_pcts_from_crop(enhanced_bgr)

    if valid_pcts is None:
        return None

    bonus_text = fullwidth_to_halfwidth(ocr_text(bonus_crop))

    # 提取基值（2-4位数字，不紧跟%）
    base_matches = re.findall(r"(\d{2,4})(?![\d%])", bonus_text)
    valid_bases = [int(b) for b in base_matches if 50 <= int(b) <= 5000]

    result = {
        "vocal_bonus_pct": valid_pcts[0],
        "dance_bonus_pct": valid_pcts[1],
        "visual_bonus_pct": valid_pcts[2],
        "pass_condition": pass_condition,
    }

    if len(valid_bases) >= 3:
        result["vocal_base"] = valid_bases[0]
        result["dance_base"] = valid_bases[1]
        result["visual_base"] = valid_bases[2]

    return result


def store_exam_prep_bonuses(ctx: "ProduceContext", bonuses: dict) -> None:
    """将考试准备页面的加成数据存入上下文。"""
    ctx.handler_state["exam_prep_bonuses"] = bonuses
    logger.info(
        f"[考试准备] 加成倍率已存储: "
        f"Vocal={bonuses['vocal_bonus_pct']}%, "
        f"Dance={bonuses['dance_bonus_pct']}%, "
        f"Visual={bonuses['visual_bonus_pct']}%"
        + (f", 合格条件={bonuses['pass_condition']}" if bonuses.get("pass_condition") else "")
    )


def get_exam_prep_bonuses(ctx: "ProduceContext") -> Optional[dict]:
    """从上下文读取已提取的考试加成数据。"""
    return ctx.handler_state.get("exam_prep_bonuses")
