"""考试轮盘队列识别。

考试中左上角有一个圆形轮盘，展示回合队列信息：
  - 各色扇区代表参数类型：粉红=Vocal，蓝=Dance，黄=Visual
  - 白色三角指针指向当前回合
  - 扇区数量 = 剩余回合数（已结束的回合扇区会消失）
  - 轮盘右侧显示当前参数名和加成倍率

提取结果：
  - queue: 逆时针顺序的参数队列（从当前回合开始）
  - remaining_turns: 剩余回合数（= 扇区数量）
  - current_param: 当前回合参数类型
  - current_bonus_pct: 当前回合加成百分比（右侧文字）
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.shared.common import ocr_text
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger
from src.utils.string_tools import fullwidth_to_halfwidth, normalize_ocr_jp

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext

# ── 颜色分类阈值（HSV 空间，高饱和+高亮度过滤分割线） ──
_SEG_S_THRESH = 150  # 饱和度下限（排除暗色分割线）
_SEG_V_THRESH = 150  # 明度下限
_POINTER_S_MAX = 60  # 指针白色：饱和度上限
_POINTER_V_MIN = 200  # 指针白色：明度下限
_POINTER_MIN_AREA = 50  # 指针三角最小面积(px²)
_POINTER_MAX_VERTICES = 5  # 指针轮廓最大顶点数（三角≈3~4）

# 参数名 → 内部 key
_PARAM_KEY = {"Vo": "vocal", "Da": "dance", "Vi": "visual"}
_PARAM_JP = {"Vo": ProduceText.VOCAL, "Da": ProduceText.DANCE, "Vi": ProduceText.VISUAL}
_debugger = DebugTools()
_MAX_REMAINING_TURNS = 40


def _resolve_wheel_roi(
    frame: np.ndarray,
    yolo_results=None,
) -> tuple[np.ndarray, int, int]:
    """优先使用 YOLO 的 Bonus Indicator 锚点裁出轮盘 ROI。"""
    h, w = frame.shape[:2]
    if yolo_results is None or not hasattr(yolo_results, "filter_by_label"):
        return frame, 0, 0
    boxes = list(yolo_results.filter_by_label(ProducerLabels.PC_BONUS_INDICATOR) or [])
    if not boxes:
        return frame, 0, 0
    def _anchor_sort_key(box) -> tuple[int, int, float]:
        x = int(getattr(box, "x", 0) or 0)
        y = int(getattr(box, "y", 0) or 0)
        conf = float(getattr(box, "confidence", 0.0) or 0.0)
        return (x + y, x, -conf)

    target = sorted(boxes, key=_anchor_sort_key)[0]
    x1 = int(getattr(target, "x", 0) or 0)
    y1 = int(getattr(target, "y", 0) or 0)
    x2 = int(getattr(target, "w", x1) or x1)
    y2 = int(getattr(target, "h", y1) or y1)
    if x2 <= x1 or y2 <= y1:
        return frame, 0, 0
    bw = x2 - x1
    bh = y2 - y1
    # Bonus Indicator 锚点右侧通常会覆盖排名卡片，轮盘本体稳定在锚点左侧区域。
    roi_x1 = max(0, x1 - int(bw * 0.18))
    roi_y1 = max(0, y1 - int(bh * 0.35))
    roi_x2 = min(w, x1 + int(bw * 0.62))
    roi_y2 = min(h, y2 + int(bh * 0.35))
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return frame, 0, 0
    _debugger.add_box(
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        label="wheel_bonus_anchor_roi",
        color=(255, 120, 60),
        alpha=0.12,
        duration=2.0,
        font_size=16,
    )
    return roi, roi_x1, roi_y1


def _classify_pixel(h: int, s: int, v: int) -> Optional[str]:
    """将 HSV 像素分类为参数类型（仅高饱和+高亮度有效）。"""
    if s < _SEG_S_THRESH or v < _SEG_V_THRESH:
        return None  # 分割线 / 暗区
    if h >= 150 or h <= 12:
        return "Vo"  # Vocal（粉红）
    if 85 <= h <= 130:
        return "Da"  # Dance（蓝）
    if 12 < h <= 42:
        return "Vi"  # Visual（黄）
    return None


def _find_wheel_circle(frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """在画面左上角用 HoughCircles 定位轮盘，并自动扫描色环半径。

    通过以下步骤确保定位正确：
    1. HoughCircles 找候选圆心
    2. 对每个圆心扫描半径，找高饱和度色环（色段所在位置）
    3. 色环覆盖率最高的候选即为轮盘

    返回 (cx, cy, inner_r, ring_r)：
      - inner_r: HoughCircles 检测的几何圆半径
      - ring_r: 实际色段环的采样半径
    失败返回 None。
    """
    h, w = frame.shape[:2]
    # 优先小 ROI，失败后逐步放宽，兼容“整屏截图”和“局部裁剪图”两种输入。
    roi_scales = [0.25, 0.4, 0.6, 0.8, 1.0]

    def _find_ring_radius(hsv_roi: np.ndarray, cx: int, cy: int, roi_x2: int, roi_y2: int) -> tuple[int, float]:
        """扫描各半径，找到高饱和度色环所在位置。返回 (best_r, coverage)。"""
        best_r, best_cov = 0, 0.0
        # 不要求圆完整落在 ROI 内：即使轮盘贴边/被裁切，也允许基于可见弧段估计半径。
        scan_min = max(8, int(min(roi_x2, roi_y2) * 0.12))
        scan_max = min(
            int(max(roi_x2, roi_y2) * 0.65),
            int(w * 0.20),
            int(h * 0.12),
        )
        if scan_max <= scan_min:
            return 0, 0.0
        for test_r in range(scan_min, scan_max, 2):
            colored = 0
            total = 0
            for deg in range(0, 360, 8):
                rad = np.deg2rad(deg)
                sx = int(cx + test_r * np.cos(rad))
                sy = int(cy + test_r * np.sin(rad))
                if 0 <= sx < hsv_roi.shape[1] and 0 <= sy < hsv_roi.shape[0]:
                    total += 1
                    s_val = int(hsv_roi[sy, sx, 1])
                    v_val = int(hsv_roi[sy, sx, 2])
                    if s_val > 150 and v_val > 150:
                        colored += 1
            if total < 12:
                continue
            cov = colored / max(total, 1)
            if cov > best_cov:
                best_cov = cov
                best_r = test_r
        return best_r, best_cov

    best_result = None
    best_score = 0.0
    best_scale = 0.0
    for scale in roi_scales:
        roi_x2 = max(24, min(w, int(w * scale)))
        roi_y2 = max(24, min(h, int(h * scale)))
        roi = frame[0:roi_y2, 0:roi_x2]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 自适应半径范围：小图时允许更小半径，避免固定阈值漏检。
        roi_min_dim = float(min(roi_x2, roi_y2))
        min_r_floor = int(w * (0.08 if w > 320 else 0.04))
        min_r = max(8, int(roi_min_dim * 0.14), min_r_floor)
        max_r = max(min_r + 4, int(roi_min_dim * 0.48), int(w * 0.10))
        max_r = min(max_r, max(24, int(w * 0.18), int(h * 0.10)))

        # 收集所有候选圆
        all_circles: list[tuple[int, int, int]] = []
        for p2 in (40, 30, 20, 15, 12, 10):
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                param1=100, param2=p2, minRadius=min_r, maxRadius=max_r,
            )
            if circles is not None:
                for c in circles[0]:
                    all_circles.append((int(c[0]), int(c[1]), int(c[2])))
        if not all_circles:
            continue

        # 去重（同一圆心附近多次检测）
        unique: list[tuple[int, int, int]] = []
        for c in all_circles:
            if not any(abs(c[0] - u[0]) < 12 and abs(c[1] - u[1]) < 12 for u in unique):
                unique.append(c)

        # 对每个独立圆心扫描色环
        for cx, cy, r in unique:
            ring_r, ring_cov = _find_ring_radius(hsv_roi, cx, cy, roi_x2, roi_y2)
            if ring_cov > best_score and ring_cov > 0.35:
                best_score = ring_cov
                best_result = (cx, cy, r, ring_r)
                best_scale = scale
        if best_result is not None and best_score >= 0.90 and scale <= 0.40:
            min_reasonable_r = max(18, int(min(roi_x2, roi_y2) * 0.24))
            if int(best_result[2]) >= min_reasonable_r:
                break

    if best_result:
        cx, cy, r, ring_r = best_result
        logger.debug(
            f"[轮盘] 内圆检测: center=({cx},{cy}) inner_r={r} "
            f"ring_r={ring_r} 色环覆盖={best_score:.0%} roi_scale={best_scale:.2f}"
        )
        return best_result

    return None


def _detect_segments(
    hsv: np.ndarray, cx: int, cy: int, ring_r: int,
) -> list[tuple[int, int, str]]:
    """检测轮盘色段。

    在色环半径附近按角度采样，多半径投票分类颜色，
    合并连续同色段并处理 360° wrap-around。

    Args:
        ring_r: 色环采样半径（由 _find_wheel_circle 自动扫描得出）

    返回 [(start_deg, end_deg, color_code), ...] 按角度排序。
    """
    h, w = hsv.shape[:2]
    # 采样半径：围绕色环中心半径 ±8 步长3
    sample_radii = [ring_r + dr for dr in range(-8, 9, 3)]

    # 每 2° 采样
    angle_colors: dict[int, Optional[str]] = {}
    for deg in range(0, 360, 2):
        rad = np.radians(deg)
        votes: dict[str, int] = {}
        for r_s in sample_radii:
            px = int(cx + r_s * np.cos(rad))
            py = int(cy + r_s * np.sin(rad))
            if 0 <= py < h and 0 <= px < w:
                c = _classify_pixel(*hsv[py, px])
                if c:
                    votes[c] = votes.get(c, 0) + 1
        angle_colors[deg] = max(votes, key=votes.get) if votes else None

    # 合并连续同色区间
    raw_segs: list[tuple[int, int, Optional[str]]] = []
    cur_color: Optional[str] = None
    start = 0
    for deg in range(0, 360, 2):
        c = angle_colors[deg]
        if c != cur_color:
            if cur_color is not None:
                raw_segs.append((start, deg - 2, cur_color))
            start = deg
            cur_color = c
    if cur_color is not None:
        raw_segs.append((start, 358, cur_color))

    # 只保留有颜色的段
    colored = [(s, e, c) for s, e, c in raw_segs if c is not None]
    if not colored:
        return []

    # 处理首尾 wrap-around（首段和末段同色 → 合并）
    if len(colored) >= 2 and colored[0][2] == colored[-1][2]:
        merged_start = colored[-1][0]
        merged_end = colored[0][1] + 360  # 跨越 360° 边界
        colored = [(merged_start, merged_end, colored[0][2])] + colored[1:-1]

    # 按段中心角排序
    def _center(s: int, e: int) -> float:
        return ((s + e) / 2) % 360

    colored.sort(key=lambda x: _center(x[0], x[1]))
    return colored


def _find_pointer_angle(
    frame: np.ndarray, hsv: np.ndarray,
    cx: int, cy: int, ring_r: int,
) -> Optional[float]:
    """检测白色三角指针，返回其相对轮盘中心的角度。

    在色环外围搜索白色区域，找到面积最大且顶点≤5的三角形轮廓。
    """
    h, w = frame.shape[:2]

    # 构建环形 mask（色环半径 ±15 范围搜索指针）
    r_inner = ring_r - 5
    r_outer = ring_r + max(30, int(ring_r * 0.5))
    mask = np.zeros((h, w), dtype=np.uint8)
    for deg in range(0, 360, 1):
        rad = np.radians(deg)
        for r_s in range(r_inner, r_outer):
            px = int(cx + r_s * np.cos(rad))
            py = int(cy + r_s * np.sin(rad))
            if 0 <= py < h and 0 <= px < w:
                sv, vv = hsv[py, px][1], hsv[py, px][2]
                if sv < _POINTER_S_MAX and vv > _POINTER_V_MIN:
                    mask[py, px] = 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 找面积最大的三角形轮廓
    best_cnt = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _POINTER_MIN_AREA:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) <= _POINTER_MAX_VERTICES and area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None

    M = cv2.moments(best_cnt)
    if M["m00"] == 0:
        return None
    ptr_x = int(M["m10"] / M["m00"])
    ptr_y = int(M["m01"] / M["m00"])
    angle = float(np.degrees(np.arctan2(ptr_y - cy, ptr_x - cx))) % 360
    logger.debug(f"[轮盘] 指针检测: pos=({ptr_x},{ptr_y}) 角度={angle:.1f}° 面积={best_area:.0f}")
    return angle


def _ocr_wheel_area(
    frame: np.ndarray, cx: int, cy: int, inner_r: int,
) -> tuple[Optional[str], Optional[int], Optional[int], int]:
    """OCR 轮盘区域，提取当前参数名、加成百分比、回合数字。

    整体 OCR 轮盘 + 上方标签 + 右侧文字，通过正则分离各信息：
      - 参数名（ボーカル/ダンス/ビジュアル）
      - 加成百分比（紧跟参数名后的 N 位数字 + %）
      - 回合数字（参数名前的单位数，即中心数字）

    返回 (param_code, bonus_pct, ocr_turns, extra_turns)。
    """
    h, w = frame.shape[:2]
    margin_top = int(inner_r * 1.8)

    # 倍率文本位于轮盘右侧：显式排除轮盘圆盘区域，避免把“剩余回合数字”拼进倍率。
    bonus_x1 = min(w, max(0, cx + int(inner_r * 0.55)))
    bonus_x2 = min(w, cx + inner_r + int(w * 0.3))
    bonus_y1 = max(0, cy - int(inner_r * 0.95))
    bonus_y2 = min(h, cy + int(inner_r * 0.95))
    bonus_crop = frame[bonus_y1:bonus_y2, bonus_x1:bonus_x2]
    if bonus_crop.size == 0:
        return None, None, None, 0
    bonus_scale = max(2, 220 // max(1, bonus_y2 - bonus_y1))
    bonus_big = cv2.resize(
        bonus_crop,
        None,
        fx=bonus_scale,
        fy=bonus_scale,
        interpolation=cv2.INTER_CUBIC,
    )
    bonus_text = fullwidth_to_halfwidth(ocr_text(bonus_big))

    # “残りターン N”位于轮盘上方，单独读取，避免与倍率文本互相污染。
    turns_x1 = max(0, cx - int(inner_r * 1.2))
    turns_x2 = min(w, cx + int(inner_r * 1.2))
    turns_y1 = max(0, cy - inner_r - margin_top)
    turns_y2 = max(turns_y1 + 1, min(h, cy - int(inner_r * 0.4)))
    turns_crop = frame[turns_y1:turns_y2, turns_x1:turns_x2]
    turns_text = ""
    if turns_crop.size > 0:
        turns_scale = max(2, 220 // max(1, turns_y2 - turns_y1))
        turns_big = cv2.resize(
            turns_crop,
            None,
            fx=turns_scale,
            fy=turns_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        turns_text = fullwidth_to_halfwidth(ocr_text(turns_big))

    _debugger.add_box(
        bonus_x1,
        bonus_y1,
        bonus_x2,
        bonus_y2,
        label=f"wheel_bonus_text: {bonus_text[:24]}",
        color=(80, 220, 120),
        alpha=0.15,
        duration=2.0,
        font_size=16,
    )
    _debugger.add_box(
        turns_x1,
        turns_y1,
        turns_x2,
        turns_y2,
        label=f"wheel_turns_text: {turns_text[:24]}",
        color=(255, 200, 0),
        alpha=0.15,
        duration=2.0,
        font_size=16,
    )
    logger.debug(f"[轮盘] 右侧OCR: '{bonus_text}'")
    logger.debug(f"[轮盘] 上方OCR: '{turns_text}'")

    # 匹配参数名
    param_code = None
    param_name_found = ""
    for code, jp_name in _PARAM_JP.items():
        if jp_name in bonus_text:
            param_code = code
            param_name_found = jp_name
            break

    # 从「参数名 + 百分比%」中提取倍率，白色「+N」识别为附加回合
    # 注意: 考试进行中 "ダンス527%" 整体是倍率，不含回合数
    # 目标: 倍率按最高有效百分比输出；+1/+2/+3 仅用于附加回合，不参与倍率。
    ocr_turns = None
    bonus_pct = None
    extra_turns = 0
    if param_name_found:
        percent_values = [
            int(value)
            for value in re.findall(re.escape(param_name_found) + r"\s*(\d{2,6})%", bonus_text)
        ]
        if not percent_values:
            percent_values = [int(value) for value in re.findall(r"(\d{2,6})%", bonus_text)]
        valid_values = [value for value in percent_values if 50 <= value <= 9999]
        if valid_values:
            bonus_pct = max(valid_values)

    extra_turn_matches = [
        int(value)
        for value in re.findall(r"[+＋]\s*(\d{1,2})", f"{bonus_text} {turns_text}")
    ]
    valid_extra_turns = [value for value in extra_turn_matches if 1 <= value <= 20]
    if valid_extra_turns:
        extra_turns = max(valid_extra_turns)

    # 也尝试从 "残りターン" 后提取回合数
    if ocr_turns is None:
        for variant in ProduceText.REMAINING_TURNS_OCR_VARIANTS:
            m3 = re.search(re.escape(variant) + r"\s*(\d{1,2})", turns_text)
            if m3:
                ocr_turns = int(m3.group(1))
                break

    # 防污染兜底：当倍率被拼成“剩余回合+真实倍率”（如 7 + 701 => 9701）时自动剥离前缀。
    if bonus_pct is not None and ocr_turns is not None and bonus_pct >= 1000:
        bonus_str = str(int(bonus_pct))
        turn_prefix = str(int(ocr_turns))
        if bonus_str.startswith(turn_prefix) and len(bonus_str) > len(turn_prefix):
            trimmed = int(bonus_str[len(turn_prefix):])
            if 50 <= trimmed <= 999:
                logger.debug(
                    "[轮盘] 倍率纠偏: 原始={} 识别到前缀回合={} -> {}",
                    bonus_pct,
                    ocr_turns,
                    trimmed,
                )
                bonus_pct = trimmed

    logger.debug(
        f"[轮盘] OCR解析: 参数={param_code} 倍率={bonus_pct} "
        f"OCR回合={ocr_turns} 附加回合={extra_turns}"
    )
    return param_code, bonus_pct, ocr_turns, extra_turns


def _extract_compact_wheel_text_info(
    frame: np.ndarray,
    *,
    allow_large: bool = False,
    wheel_hint: tuple[int, int, int, int] | None = None,
) -> Optional[dict]:
    """轮盘 OCR 回退解析。

    默认仅用于紧凑裁剪图；当提供 wheel_hint 且 allow_large=True 时，
    会先按轮盘位置裁局部 ROI，再走同一套文本回退逻辑。
    """
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None
    # 默认仅用于紧凑裁剪图，避免干扰正常整屏流程。
    if (not allow_large) and (w > 320 or h > 180):
        return None

    source = frame
    source_x = 0
    source_y = 0
    turn_hint_text = ""
    if wheel_hint is not None:
        cx, cy, inner_r, _ = wheel_hint
        roi_x1 = max(0, cx - int(inner_r * 1.6))
        roi_x2 = min(w, cx + int(inner_r * 4.2))
        roi_y1 = max(0, cy - int(inner_r * 1.8))
        roi_y2 = min(h, cy + int(inner_r * 1.8))
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size > 0:
            source = roi
            source_x = roi_x1
            source_y = roi_y1
            sh, sw = source.shape[:2]
            _debugger.add_box(
                roi_x1,
                roi_y1,
                roi_x2,
                roi_y2,
                label="wheel_compact_roi",
                color=(255, 120, 60),
                alpha=0.12,
                duration=2.0,
                font_size=16,
            )
            # 中心回合数字位于轮盘左半内圈，右侧是倍率文本，需显式排除右侧区域。
            center_x1 = max(0, cx - int(inner_r * 0.70))
            center_x2 = min(w, cx + int(inner_r * 0.10))
            center_y1 = max(0, cy - int(inner_r * 0.60))
            center_y2 = min(h, cy + int(inner_r * 0.60))
            center_crop = frame[center_y1:center_y2, center_x1:center_x2]
            if center_crop.size > 0:
                turn_hint_text = normalize_ocr_jp(fullwidth_to_halfwidth(ocr_text(center_crop)))
                _debugger.add_box(
                    center_x1,
                    center_y1,
                    center_x2,
                    center_y2,
                    label=f"wheel_turn_center: {turn_hint_text[:24]}",
                    color=(120, 180, 255),
                    alpha=0.15,
                    duration=2.0,
                    font_size=16,
                )

    sh, sw = source.shape[:2]
    right_x1 = int(sw * 0.46)
    right_crop = source[0:int(sh * 0.96), right_x1:sw]
    left_crop = source[int(sh * 0.12):int(sh * 0.90), 0:int(sw * 0.58)]
    full_crop = source
    if right_crop.size == 0 or left_crop.size == 0:
        return None

    right_text = normalize_ocr_jp(fullwidth_to_halfwidth(ocr_text(right_crop)))
    left_text = normalize_ocr_jp(fullwidth_to_halfwidth(ocr_text(left_crop)))
    full_text = normalize_ocr_jp(fullwidth_to_halfwidth(ocr_text(full_crop)))

    _debugger.add_box(
        source_x + right_x1,
        source_y,
        source_x + sw,
        source_y + int(sh * 0.96),
        label=f"wheel_compact_right: {right_text[:24]}",
        color=(80, 220, 120),
        alpha=0.15,
        duration=2.0,
        font_size=16,
    )
    _debugger.add_box(
        source_x,
        source_y + int(sh * 0.12),
        source_x + int(sw * 0.58),
        source_y + int(sh * 0.90),
        label=f"wheel_compact_left: {left_text[:24]}",
        color=(255, 200, 0),
        alpha=0.15,
        duration=2.0,
        font_size=16,
    )

    param_code = None
    if "ボーカル" in right_text or "ボーカ" in right_text or "ボーカル" in full_text:
        param_code = "Vo"
    elif "ダンス" in right_text or "ダンス" in full_text:
        param_code = "Da"
    elif "ビジュアル" in right_text or "ビジュ" in right_text or "ビジュアル" in full_text:
        param_code = "Vi"

    right_matches = re.findall(r"(\d{2,6})\s*%", right_text)
    full_matches = re.findall(r"(\d{2,6})\s*%", full_text)
    raw_right = max((int(value) for value in right_matches), default=None)
    raw_full = max((int(value) for value in full_matches), default=None)
    extra_turn_candidates = [
        int(value)
        for value in re.findall(r"[+＋]\s*(\d{1,2})", f"{right_text} {full_text}")
    ]
    extra_turns = max((value for value in extra_turn_candidates if 1 <= value <= 20), default=0)

    def _valid_bonus(value: int | None) -> bool:
        return value is not None and 50 <= int(value) <= 9999

    # 左侧 OCR 粗读回合
    left_turn = None
    left_turn_match = re.search(r"(\d{1,2})", left_text)
    if left_turn_match:
        turn_value = int(left_turn_match.group(1))
        if 1 <= turn_value <= _MAX_REMAINING_TURNS:
            left_turn = turn_value

    prefix_turn = None
    if raw_full is not None and raw_right is not None:
        full_digits = str(raw_full)
        right_digits = str(raw_right)
        if full_digits.endswith(right_digits) and len(full_digits) > len(right_digits):
            prefix = full_digits[:-len(right_digits)]
            if prefix:
                if len(prefix) >= 2:
                    head_two = int(prefix[:2])
                    if 1 <= head_two <= _MAX_REMAINING_TURNS:
                        prefix_turn = head_two
                if prefix_turn is None:
                    head_one = int(prefix[:1])
                    if 1 <= head_one <= _MAX_REMAINING_TURNS:
                        prefix_turn = head_one

    center_turn = None
    center_after_param = None
    center_param_match = re.search(
        r"(?:ダンス|ボーカル|ビジュアル|ビジュ)\D*(\d{1,2})",
        turn_hint_text,
    )
    if center_param_match:
        parsed = int(center_param_match.group(1))
        if 1 <= parsed <= _MAX_REMAINING_TURNS:
            center_after_param = parsed
    center_turn_matches = [
        int(value)
        for value in re.findall(r"(?<![.．])(\d{1,2})(?!\d)", turn_hint_text)
    ]
    valid_center_turns = [value for value in center_turn_matches if 1 <= value <= _MAX_REMAINING_TURNS]
    if center_after_param is not None:
        center_turn = center_after_param
    elif valid_center_turns:
        center_turn = max(valid_center_turns)

    remaining_turns = None
    if center_turn is not None:
        remaining_turns = center_turn
    if left_turn is not None and prefix_turn is not None:
        # 两路都可用但冲突时，取更小值（常见污染是“前缀多读一位/左侧多读一位”）。
        if remaining_turns is None:
            remaining_turns = min(left_turn, prefix_turn)
    elif prefix_turn is not None:
        if remaining_turns is None:
            remaining_turns = prefix_turn
    else:
        if remaining_turns is None:
            remaining_turns = left_turn

    bonus_pct = None
    if _valid_bonus(raw_right):
        bonus_pct = raw_right
    elif _valid_bonus(raw_full):
        bonus_pct = raw_full
    else:
        oversized = raw_right if raw_right is not None else raw_full
        if oversized is not None:
            oversized_str = str(oversized)
            if remaining_turns is not None:
                prefix = str(remaining_turns)
                if oversized_str.startswith(prefix) and len(oversized_str) > len(prefix):
                    tail_val = int(oversized_str[len(prefix):])
                    if _valid_bonus(tail_val):
                        bonus_pct = tail_val
            if bonus_pct is None and len(oversized_str) >= 5:
                for split_len in (2, 1):
                    if len(oversized_str) <= split_len:
                        continue
                    turn_part = int(oversized_str[:split_len])
                    bonus_part = int(oversized_str[split_len:])
                    if 1 <= turn_part <= _MAX_REMAINING_TURNS and _valid_bonus(bonus_part):
                        if remaining_turns is None:
                            remaining_turns = turn_part
                        bonus_pct = bonus_part
                        break

    if param_code is None or bonus_pct is None:
        return None
    current_key = _PARAM_KEY.get(param_code)
    if current_key is None:
        return None
    turns = int(remaining_turns or 1) + int(extra_turns or 0)
    return {
        "remaining_turns": turns,
        "current_param": current_key,
        "current_bonus_pct": int(bonus_pct),
        "queue": [current_key],
        "additional_turns": int(extra_turns or 0),
        "confidence": "low",
    }


def extract_exam_wheel_info(frame: np.ndarray, yolo_results=None) -> Optional[dict]:
    """从考试页面单帧提取轮盘队列信息。

    交叉验证逻辑：
      - 色段数量 vs OCR 回合数字 → 不一致时记录警告但以色段为准
      - 指针色段 vs OCR 参数名 → 不一致时以 OCR 为准（文字更可靠）

    返回:
        {
          "remaining_turns": 9,              # 剩余回合数（色段数量）
          "current_param": "visual",         # 当前回合参数
          "current_bonus_pct": 701,          # 当前回合加成百分比
          "additional_turns": 0,             # 白色 +N 附加回合
          "queue": ["visual", "dance", ...], # 逆时针队列（从当前回合开始）
          "confidence": "high",              # 置信度（high/medium/low）
        }
    提取失败返回 None。
    """
    if frame is None or not hasattr(frame, "shape"):
        return None

    # 1. 定位轮盘内圆 + 色环半径（优先用 Bonus Indicator 锚点裁 ROI）
    roi_frame, roi_x, roi_y = _resolve_wheel_roi(frame, yolo_results=yolo_results)
    used_anchor_roi = roi_frame is not frame
    bonus_anchor_frame: np.ndarray | None = None
    if yolo_results is not None and hasattr(yolo_results, "filter_by_label"):
        bonus_boxes = list(yolo_results.filter_by_label(ProducerLabels.PC_BONUS_INDICATOR) or [])
        if bonus_boxes:
            target = sorted(
                bonus_boxes,
                key=lambda box: (
                    int(getattr(box, "x", 0) or 0) + int(getattr(box, "y", 0) or 0),
                    int(getattr(box, "x", 0) or 0),
                ),
            )[0]
            bx1 = max(0, int(getattr(target, "x", 0) or 0))
            by1 = max(0, int(getattr(target, "y", 0) or 0))
            bx2 = min(frame.shape[1], int(getattr(target, "w", bx1) or bx1))
            by2 = min(frame.shape[0], int(getattr(target, "h", by1) or by1))
            if bx2 > bx1 and by2 > by1:
                bonus_anchor_frame = frame[by1:by2, bx1:bx2]

    def _try_compact_fallback(
        wheel_hint: tuple[int, int, int, int] | None = None,
    ) -> Optional[dict]:
        # 先尝试“带轮盘提示”的局部回退，再尝试锚点 ROI / 全图回退。
        candidate_calls: list[tuple[np.ndarray, tuple[int, int, int, int] | None]] = []
        if wheel_hint is not None:
            candidate_calls.append((frame, wheel_hint))
        if bonus_anchor_frame is not None:
            candidate_calls.append((bonus_anchor_frame, None))
        if used_anchor_roi:
            candidate_calls.append((roi_frame, None))
        candidate_calls.append((frame, None))
        for source_frame, hint in candidate_calls:
            compact = _extract_compact_wheel_text_info(
                source_frame,
                allow_large=True,
                wheel_hint=hint,
            )
            if compact is not None:
                return compact
        return None

    circle_local = _find_wheel_circle(roi_frame)
    circle = None
    if circle_local is not None:
        lx, ly, inner_r, ring_r = circle_local
        circle = (lx + roi_x, ly + roi_y, inner_r, ring_r)

    if circle is None:
        compact = _try_compact_fallback()
        if compact is not None:
            logger.info(
                "[轮盘] 紧凑图回退: 剩余{}回合 当前={} {}% 置信={}",
                compact["remaining_turns"],
                compact["current_param"],
                compact["current_bonus_pct"],
                compact["confidence"],
            )
            return compact
        logger.warning("[轮盘] 未检测到轮盘内圆")
        return None
    cx, cy, inner_r, ring_r = circle

    # 2. 检测色段（在色环半径处采样）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    segments = _detect_segments(hsv, cx, cy, ring_r)
    if len(segments) < 2:
        compact = _try_compact_fallback(circle)
        if compact is not None:
            logger.info(
                "[轮盘] 色段不足，紧凑图回退: 剩余{}回合 当前={} {}%",
                compact["remaining_turns"],
                compact["current_param"],
                compact["current_bonus_pct"],
            )
            return compact
        logger.warning(f"[轮盘] 色段不足: {len(segments)}")
        return None

    seg_count = len(segments)

    # 3. 检测指针角度（使用色环半径范围搜索）
    pointer_angle = _find_pointer_angle(frame, hsv, cx, cy, ring_r)
    if pointer_angle is None:
        compact = _try_compact_fallback(circle)
        if compact is not None:
            logger.info(
                "[轮盘] 指针缺失，紧凑图回退: 剩余{}回合 当前={} {}%",
                compact["remaining_turns"],
                compact["current_param"],
                compact["current_bonus_pct"],
            )
            return compact
        logger.warning("[轮盘] 未检测到指针")
        return None

    # 4. 找指针所在的段（当前回合）
    def _center(s: int, e: int) -> float:
        return ((s + e) / 2) % 360

    def _angle_dist(a1: float, a2: float) -> float:
        d = abs(a1 - a2) % 360
        return min(d, 360 - d)

    best_idx = min(
        range(len(segments)),
        key=lambda i: _angle_dist(pointer_angle, _center(segments[i][0], segments[i][1])),
    )

    # 5. 逆时针顺序（角度递减方向）
    queue_codes: list[str] = []
    n = len(segments)
    for offset in range(n):
        idx = (best_idx - offset) % n
        queue_codes.append(segments[idx][2])

    queue_keys = [_PARAM_KEY[c] for c in queue_codes]

    # 6. OCR 整体区域（参数名 + 加成倍率 + 回合数字）
    ocr_param, bonus_pct, ocr_turns, extra_turns = _ocr_wheel_area(frame, cx, cy, inner_r)
    total_turns = seg_count + int(extra_turns or 0)

    # ── 交叉验证 ──
    confidence = "high"

    # 验证1: 色段数 vs OCR 回合数
    if ocr_turns is not None and ocr_turns != total_turns:
        logger.warning(
            f"[轮盘] 色段数+附加={total_turns} 与 OCR回合={ocr_turns} 不一致，"
            f"以色段数为准"
        )
        confidence = "medium"

    # 验证2: 指针段颜色 vs OCR 参数名
    current_code = queue_codes[0]
    current_key = _PARAM_KEY[current_code]
    if ocr_param and ocr_param != current_code:
        logger.warning(
            f"[轮盘] 指针段={current_code} 与 OCR参数={ocr_param} 不一致，"
            f"以 OCR 为准"
        )
        current_key = _PARAM_KEY[ocr_param]
        queue_keys[0] = current_key
        confidence = "medium"

    # 缺少 OCR 信息时降低置信度
    if ocr_param is None or bonus_pct is None:
        confidence = "low"

    # 紧凑截图下，主链路若低置信则优先采用文本回退结果，避免小图几何误检。
    if confidence == "low":
        compact = _try_compact_fallback(circle)
        if compact is not None and compact.get("current_bonus_pct") is not None:
            logger.info(
                "[轮盘] 低置信主链路，采用文本回退: 剩余{}回合 当前={} {}%",
                compact["remaining_turns"],
                compact["current_param"],
                compact["current_bonus_pct"],
            )
            return compact

    result = {
        "remaining_turns": total_turns,
        "current_param": current_key,
        "current_bonus_pct": bonus_pct,
        "queue": queue_keys,
        "additional_turns": int(extra_turns or 0),
        "confidence": confidence,
    }
    logger.info(
        f"[轮盘] 剩余{total_turns}回合 "
        f"当前={current_key}"
        + (f" {bonus_pct}%" if bonus_pct else "")
        + f" 队列={queue_keys}"
        + (f" (OCR回合={ocr_turns})" if ocr_turns is not None else "")
        + (f" (附加={extra_turns})" if extra_turns else "")
        + f" 置信={confidence}"
    )
    return result


def extract_exam_wheel_validated(
    capture_fn,
    max_frames: int = 3,
    min_agreement: int = 2,
) -> Optional[dict]:
    """多帧采集 + 共识校验。

    连续采集 max_frames 帧，对每帧独立提取，要求至少 min_agreement 帧
    在关键字段（remaining_turns + queue）上一致才返回结果。

    Args:
        capture_fn: 无参函数，返回 np.ndarray（ADB 截图等）
        max_frames: 最大采集帧数
        min_agreement: 最小一致帧数

    Returns:
        置信度最高的一致结果，或 None。
    """
    results: list[dict] = []
    for i in range(max_frames):
        frame = capture_fn()
        if frame is None:
            continue
        info = extract_exam_wheel_info(frame)
        if info is not None:
            results.append(info)

    if not results:
        logger.warning("[轮盘] 多帧采集全部失败")
        return None

    if len(results) == 1:
        logger.debug("[轮盘] 仅1帧有效，直接返回")
        return results[0]

    # 以 (remaining_turns, tuple(queue)) 为签名分组
    from collections import Counter

    def _signature(r: dict) -> tuple:
        return (r["remaining_turns"], tuple(r["queue"]))

    sig_counter = Counter(_signature(r) for r in results)
    best_sig, best_count = sig_counter.most_common(1)[0]

    if best_count < min_agreement:
        logger.warning(
            f"[轮盘] 多帧不一致: "
            f"{dict(sig_counter)} (需要{min_agreement}帧一致)"
        )
        # 仍返回出现最多的，但降低置信度
        for r in results:
            if _signature(r) == best_sig:
                r["confidence"] = "low"
                return r

    # 从一致帧中选置信度最高的
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    best_result = max(
        (r for r in results if _signature(r) == best_sig),
        key=lambda r: confidence_rank.get(r.get("confidence", "low"), 0),
    )
    logger.info(
        f"[轮盘] 多帧共识: {best_count}/{len(results)}帧一致 "
        f"置信={best_result['confidence']}"
    )
    return best_result


def store_exam_wheel_info(ctx: "ProduceContext", info: dict) -> None:
    """将轮盘信息存入上下文。"""
    ctx.handler_state["exam_wheel_info"] = info


def get_exam_wheel_info(ctx: "ProduceContext") -> Optional[dict]:
    """从上下文读取轮盘信息。"""
    return ctx.handler_state.get("exam_wheel_info")
