"""考试结果页检测。

考试结束后会显示排名结果页，包含：
  - 顶部徽章显示玩家排名（如 "3位"）
  - 6 行排名条目（1st~6th），每行有序号、头像、名字、分数
  - 玩家行有白色圆角边框 + "ログ" 按钮
  - 底部有 "再挑戦" 和 "次へ" 按钮

检测方法（白色边框 + YOLO 按钮双重验证）：
  1. HSV 检测白色轮廓 → 找宽矩形（aspect > 3）定位玩家行
  2. YOLO 检测 "Universal button" 在玩家行内 → 双重确认
  3. OCR 提取序号、分数、重试次数

提取结果：
  - player_rank: 玩家排名 (1-6)
  - player_score: 玩家分数
  - passed: 是否合格（根据 pass_condition 动态判定，如 "1位以上" / "3位以上"）
  - pass_rank: 生效的合格线排名
  - all_rankings: 所有参赛者排名和分数
  - retry_count: 剩余重试次数（如有）
  - gap_to_above: 与上一名的分差
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.inference.ocr_engine import OCRService
from src.core.tasks.producer_challenge.gameplay.exam_ranking import _extract_ordinal_digit
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.entity.Yolo import Yolo_Box, Yolo_Results
    from src.core.tasks.producer_challenge.context import ProduceContext

# ── 默认合格线：排名 <= 此值为合格（实际由 pass_condition 覆盖）──
_DEFAULT_PASS_RANK = 3

# ── 合格条件解析正则（匹配 "3位以上" / "1位以上" 等）──
_PASS_CONDITION_RE = re.compile(r"(\d+)位以上")

# ── 白色边框 HSV 阈值 ──
_WHITE_HSV_LOW = (0, 0, 200)
_WHITE_HSV_HIGH = (180, 50, 255)

# ── 玩家行白色轮廓筛选条件 ──
_MIN_AREA_RATIO = 0.015       # 轮廓面积占图像面积的最小比例
_MIN_ASPECT_RATIO = 3.0       # 宽高比最低要求（排名行是宽矩形）

# ── OCR 序号最小高度（过滤小文字） ──
_MIN_ORDINAL_H_RATIO = 0.025  # 占图像高度的比例

# ── 分数提取正则 ──
_SCORE_RE = re.compile(r"([\d,]+)\s*[Pp]")

# ── 重试次数正则 ──
_RETRY_RE = re.compile(r"あと(\d+)回")

_ocr_service = OCRService()


def parse_pass_rank(pass_condition: Optional[str]) -> Optional[int]:
    """从合格条件字符串（如 "3位以上"）解析出合格线排名。

    Args:
        pass_condition: exam_prep 提取的合格条件文本，如 "3位以上" / "1位以上"

    Returns:
        合格线排名（int），解析失败返回 None
    """
    if not pass_condition:
        return None
    m = _PASS_CONDITION_RE.search(pass_condition)
    if m:
        return int(m.group(1))
    return None


def _find_player_row_rect(
    hsv: np.ndarray,
    img_h: int,
    img_w: int,
) -> Optional[tuple[int, int, int, int]]:
    """通过白色边框轮廓检测玩家行的矩形区域。

    返回 (x, y, w, h) 或 None。
    """
    white_mask = cv2.inRange(hsv, _WHITE_HSV_LOW, _WHITE_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = img_h * img_w * _MIN_AREA_RATIO
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if aspect >= _MIN_ASPECT_RATIO and area > best_area:
            best = (x, y, w, h)
            best_area = area

    return best


def _has_button_in_rect(
    yolo_results: "Yolo_Results",
    rect: tuple[int, int, int, int],
) -> bool:
    """检查 YOLO 是否在指定矩形区域内检测到按钮（ログ按钮）。"""
    rx, ry, rw, rh = rect
    for box in yolo_results.boxes:
        if box.label != BaseUILabels.BUTTON:
            continue
        # Yolo_Box: x/y = 左上角, w/h = 右下角坐标
        bcx = (box.x + box.w) // 2
        bcy = (box.y + box.h) // 2
        if rx <= bcx <= rx + rw and ry <= bcy <= ry + rh:
            return True
    return False


def _has_bottom_buttons(yolo_results: "Yolo_Results", img_h: int) -> bool:
    """检查底部是否有 再挑戦/次へ 按钮（y > 80% 画面高度）。"""
    bottom_threshold = img_h * 0.80
    has_button = False
    for box in yolo_results.boxes:
        if box.y > bottom_threshold and box.label in (
            BaseUILabels.BUTTON,
            "Universal Confirm button",
        ):
            has_button = True
            break
    return has_button


def extract_exam_result(
    frame: np.ndarray,
    yolo_results: "Yolo_Results",
    pass_rank: Optional[int] = None,
) -> Optional[dict]:
    """从考试结果页提取玩家排名、分数等信息。

    使用白色边框 + YOLO 按钮双重检测确认玩家行：
      1. HSV 找白色宽矩形轮廓 → 玩家行位置
      2. YOLO 检测该区域内有按钮（ログ）→ 双重确认
      3. OCR 提取序号、分数、重试次数

    Args:
        frame: 全屏游戏帧（BGR numpy 数组）
        yolo_results: 当前帧的 YOLO 检测结果
        pass_rank: 合格线排名（如 1 = 仅第1名合格, 3 = 前3名合格）。
                   为 None 时使用默认值 _DEFAULT_PASS_RANK。
                   应从 exam_prep 提取的 pass_condition 解析而来。

    返回:
      {
          "player_rank": 3,
          "player_score": 600,
          "passed": True,
          "pass_rank": 3,
          "all_rankings": [
              {"rank": 1, "score": 1497},
              {"rank": 2, "score": 1002},
              ...
          ],
          "retry_count": 2,
          "gap_to_above": 402,
      }
      检测失败返回 None。
    """
    if frame is None or not hasattr(frame, "shape"):
        return None
    if yolo_results is None:
        return None

    img_h, img_w = frame.shape[:2]
    if img_h == 0 or img_w == 0:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. 白色边框检测 → 玩家行矩形
    player_rect = _find_player_row_rect(hsv, img_h, img_w)
    if player_rect is None:
        logger.debug("[考试结果] 未检测到白色边框的玩家行")
        return None

    # 2. YOLO 按钮双重验证：玩家行内存在“ログ”按钮。
    has_log_btn = _has_button_in_rect(yolo_results, player_rect)
    has_bottom_btns = _has_bottom_buttons(yolo_results, img_h)
    if not has_log_btn and not has_bottom_btns:
        logger.debug("[考试结果] YOLO 未检测到确认按钮，跳过")
        return None

    px, py, pw, ph = player_rect

    # 3. OCR 全图
    ocr_results = _ocr_service.ocr(frame)
    if not ocr_results or not ocr_results.results:
        logger.debug("[考试结果] OCR 未检测到任何文字")
        return None

    # 4. 提取所有排名条目
    min_ordinal_h = int(img_h * _MIN_ORDINAL_H_RATIO)
    ordinal_entries = []

    for r in ocr_results.results:
        if r.h < min_ordinal_h:
            continue
        d = _extract_ordinal_digit(r.text)
        if d is None:
            continue

        # 找同行的分数：取序号下方、最近的 "xxxPt" 文本
        score = None
        best_dist = float("inf")
        for sr in ocr_results.results:
            if sr.x <= r.x:
                continue
            # 分数在序号下方（y 更大）
            dy = sr.y - r.y
            if dy < 0 or dy > r.h * 2:
                continue
            m = _SCORE_RE.search(sr.text)
            if m and dy < best_dist:
                score = int(m.group(1).replace(",", ""))
                best_dist = dy

        # 判断是否在玩家行白色边框内
        in_player_row = (py <= r.y <= py + ph)
        ordinal_entries.append({
            "rank": d,
            "score": score,
            "is_player": in_player_row,
        })

    if not ordinal_entries:
        logger.debug("[考试结果] 未检测到有效序号")
        return None

    # 5. 提取重试次数
    retry_count = None
    for r in ocr_results.results:
        m = _RETRY_RE.search(r.text)
        if m:
            retry_count = int(m.group(1))
            break

    # 6. 确定玩家排名
    player_entry = next((e for e in ordinal_entries if e["is_player"]), None)
    if player_entry is None:
        # 白色边框找到了但没有序号落在其中，尝试用最近的序号
        for e in ordinal_entries:
            logger.debug(
                f"[考试结果] 备选序号: {e['rank']}位, score={e['score']}"
            )
        logger.debug("[考试结果] 未能匹配玩家行中的序号")
        return None

    player_rank = player_entry["rank"]
    player_score = player_entry["score"]

    # 7. 构建排名列表并排序
    all_rankings = sorted(
        [{"rank": e["rank"], "score": e["score"]} for e in ordinal_entries],
        key=lambda x: x["rank"],
    )

    # 8. 计算与上一名的分差
    gap_to_above = None
    if player_rank > 1:
        above = next((r for r in all_rankings if r["rank"] == player_rank - 1), None)
        if above and above["score"] is not None and player_score is not None:
            gap_to_above = above["score"] - player_score

    # 8. 合格判定：使用传入的 pass_rank 或默认值
    effective_pass_rank = pass_rank if pass_rank is not None else _DEFAULT_PASS_RANK
    passed = player_rank <= effective_pass_rank

    logger.info(
        f"[考试结果] 第{player_rank}位 ({player_score}pt), "
        f"合格线={effective_pass_rank}位以上, "
        f"{'合格' if passed else '不合格'}, "
        f"剩余重试={retry_count}, 差距={gap_to_above}"
    )

    return {
        "player_rank": player_rank,
        "player_score": player_score,
        "passed": passed,
        "pass_rank": effective_pass_rank,
        "all_rankings": all_rankings,
        "retry_count": retry_count,
        "gap_to_above": gap_to_above,
    }


def store_exam_result(ctx: "ProduceContext", info: dict) -> None:
    """保存考试、结果并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        info: 用于提供info相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    ctx.handler_state["exam_result_info"] = info


def get_exam_result(ctx: "ProduceContext") -> Optional[dict]:
    """获取考试、结果并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        Optional[dict]: 返回值类型见注解，语义由函数用途决定。
    """
    return ctx.handler_state.get("exam_result_info")


def get_pass_rank_from_context(ctx: "ProduceContext") -> Optional[int]:
    """从上下文中读取合格线排名。

    优先从 exam_prep 提取的 pass_condition（如 "3位以上"）解析，
    解析失败则返回 None（调用方应使用默认值）。
    """
    bonuses = ctx.handler_state.get("exam_prep_bonuses")
    if not bonuses:
        return None
    return parse_pass_rank(bonuses.get("pass_condition"))
