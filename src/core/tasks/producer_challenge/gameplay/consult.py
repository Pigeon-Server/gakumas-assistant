"""相談（咨询交换页）handler。

基于 `producer_plan.md` 中的拆分思路，把相談视为可编排的多子动作流程：
  - 交换商品（技能卡 / P饮料 / 其他道具）
  - 技能卡强化
  - 技能卡删除
  - 退出相談

当前优先补齐「候选项规范化 + 无状态决策桥 + 强化骨架」。
默认兜底策略保持保守：
  - 优先只自动进入一次强化
  - 强化页优先选择目标卡，再确认
  - 不默认执行删除
后续可通过 ``ctx.consult_strategy`` 完全覆盖。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Sequence

import cv2
import numpy as np

from src.constants.game.producer_gameplay import (
    CONSULT_SELECTION_POSITIONS,
    GameplayPhase,
    GameplayPosition,
)
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.general_text import GeneralText
from src.constants.game.text.produce_text import ProduceText
from src.core.tasks.producer_challenge.shared.common import (
    invoke_decision_strategy,
    ocr_text,
    resolve_candidate_index,
)
from src.core.tasks.producer_challenge.gameplay.decision import (
    _apply_resolution,
    build_decision_state,
    hydrate_consult_candidates,
    resolve_produce_card_identity,
    resolve_produce_entity_identity,
)
from src.core.tasks.producer_challenge.gameplay.handler_base import (
    GameplayHandler,
    HandlerResult,
)
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger
from src.utils.string_tools import fullwidth_to_halfwidth, normalize_ocr_jp

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

# 强化/削除页面的缩略图卡片标签（不含 INFO 面板）
_CONSULT_THUMBNAIL_LABELS = (
    ProducerLabels.SKILL_CARD_ACTIVE,
    ProducerLabels.SKILL_CARD_MENTAL,
    ProducerLabels.SKILL_CARD_TRAP,
)
_CONSULT_EXCHANGE_RETRY_THRESHOLD = 2

# OCR 结果前后缀杂字符清理（如 |初星水 → 初星水）
# 注意：不去除前缀数字，存在合法的数字开头名称（如 200%スマイル、873シューター）。
_OCR_ARTIFACT_RE = re.compile(r"^[|｜\[\]【】\s]+|[|｜\[\]【】\s]+$")


def _dedup_boxes_by_position(boxes: list, tolerance: int = 60) -> list:
    """按中心坐标去重，同一位置（距离 < tolerance）的 YOLO 框只保留第一个。"""
    deduped: list = []
    for box in boxes:
        cx, cy = box.cx, box.cy
        is_dup = False
        for kept in deduped:
            if abs(cx - kept.cx) < tolerance and abs(cy - kept.cy) < tolerance:
                is_dup = True
                break
        if not is_dup:
            deduped.append(box)
    return deduped


def _clean_ocr_item_name(raw: str) -> str:
    """标准化 OCR 提取的物品名称。

    应用项目标准的 fullwidth_to_halfwidth + normalize_ocr_jp，
    并去除常见的 OCR 前/后缀杂字符（管道符、括号等）。
    """
    text = fullwidth_to_halfwidth(raw)
    text = normalize_ocr_jp(text)
    text = _OCR_ARTIFACT_RE.sub("", text).strip()
    return text


def _consult_mode_action_prefix(mode: str) -> str:
    """将相談子模式映射到统一动作前缀。

    说明：
    - `enhancement` 与 `remove` 共享同一套“进入子流程 -> 选卡 -> 预览 -> 确认 -> 返回”骨架。
    - 区别只体现在动作名前缀与按钮文案，不额外复制一套状态机。
    """
    return "remove" if mode == "remove" else "enhancement"


def _consult_target_kind_for_mode(mode: str) -> str:
    """处理consult、target、kind、for、mode并返回结果。

    Args:
        mode: 用于提供mode相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    return "remove_target" if mode == "remove" else "enhancement_target"


def _consult_confirm_kind_for_mode(mode: str) -> str:
    """处理consult、confirm、kind、for、mode并返回结果。

    Args:
        mode: 用于提供mode相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    return "confirm_remove" if mode == "remove" else "confirm_enhancement"


def _consult_select_operation_for_mode(mode: str) -> str:
    """处理consult、select、operation、for、mode并返回结果。

    Args:
        mode: 用于提供mode相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    prefix = _consult_mode_action_prefix(mode)
    return f"consult_select_{prefix}_target"


def _consult_confirm_operation_for_mode(mode: str) -> str:
    """处理consult、confirm、operation、for、mode并返回结果。

    Args:
        mode: 用于提供mode相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    prefix = _consult_mode_action_prefix(mode)
    return f"consult_confirm_{prefix}"


@dataclass
class ConsultActionCandidate:
    """相談页面上的一个可执行候选项。"""

    index: int
    kind: str
    title: str
    box: Any = field(repr=False, default=None)
    icon_box: Any = field(repr=False, default=None)  # 内层 YOLO 检测（P Drink / Skill Card），用于 CLIP
    entity_type_hint: str = ""  # "produce_drink" / "produce_card" / ""（实体类型提示）
    selected: bool = False
    action_id: str = ""
    db_id: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _sorted_boxes(boxes) -> list:
    """处理sorted、boxes并返回结果。

    Args:
        boxes: 检测框集合。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    return sorted(boxes, key=lambda item: (item.cy, item.cx))


def _boxes_overlap(a, b) -> bool:
    """判断两个 YOLO Box 是否重叠（坐标格式: x,y=左上角, w,h=右下角）。"""
    return a.x < b.w and b.x < a.w and a.y < b.h and b.y < a.h


# 内层 YOLO 标签 → 实体类型提示的映射
_INNER_LABEL_TYPE_HINT = {
    ProducerLabels.SKILL_CARD_ACTIVE: "produce_card",
    ProducerLabels.SKILL_CARD_MENTAL: "produce_card",
    ProducerLabels.SKILL_CARD_TRAP: "produce_card",
}


def _find_inner_icon_box(exchange_box, results) -> tuple:
    """在 Card: Item Exchange 区域内查找更精确的内层 YOLO 检测。

    Returns:
        (icon_box, entity_type_hint): 内层检测框和类型提示。
        未找到时返回 (None, "")。
    """
    # 优先查找 P Drink 内层检测
    for drink_box in results.filter_by_label(BaseUILabels.P_DRINK):
        if _boxes_overlap(exchange_box, drink_box):
            return drink_box, "produce_drink"
    # 查找 Skill Card 内层检测
    for label, type_hint in _INNER_LABEL_TYPE_HINT.items():
        for card_box in results.filter_by_label(label):
            if _boxes_overlap(exchange_box, card_box):
                return card_box, type_hint
    return None, ""


def _extract_exchange_price(box_frame) -> str:
    """从交换卡底部区域提取 P ポイント价格。

    仅对卡片底部 ~30% 区域做 OCR，提取价格数字。
    SALE 卡显示「原价 → 折后价」，取最后一个有效数字为实际价格。
    """
    if box_frame is None:
        return ""
    h, _w = box_frame.shape[:2]
    # 底部约 30% 区域包含价格信息
    price_region = box_frame[int(h * 0.70):, :]
    price_text = ocr_text(price_region)
    # 提取所有数字序列，取最后一个有效价格
    numbers = re.findall(r"\d+", price_text)
    if not numbers:
        return ""
    # 优先取较大数字（避免 OCR 碎片误识别）。
    valid = [n for n in numbers if int(n) >= 10]
    return valid[-1] if valid else numbers[-1]


# ── 信息面板 OCR 物品名提取 ──────────────────────────────────────

# 信息面板 YOLO 标签（用于展示被选中物品的详细信息）
_INFO_PANEL_LABELS = (
    ProducerLabels.SKILL_CARD_INFO,   # "Skill Card: Info"（技能卡信息面板标签）
    ProducerLabels.PC_ACTION_INFO,    # "Producer Challenge: Action Info"（行动信息面板标签）
)

# 名称区域高度占面板高度的比例（仅取第一行物品名，排除效果说明文字）
_INFO_PANEL_NAME_HEIGHT_RATIO = 0.15


def _extract_info_panel_item_name(results, frame) -> tuple[str, str, Any] | None:
    """从物品信息面板中提取物品名称。

    信息面板布局:
      ┌─────────────────────────────────────┐
      │ [图标]  物品名称                     │  ← 名称区域 (icon.w → panel.w)
      │         ──────────                  │
      │         效果说明 ...                 │
      └─────────────────────────────────────┘

    通过检测面板内部的 YOLO 子检测（P Drink / Skill Card），
    从其右边缘 (w) 到面板右边缘裁切名称区域，OCR 提取物品名。

    Returns:
        (item_name, entity_type_hint, panel_box) 或 None（未检测到面板时）。
    """
    # 查找信息面板
    panel = None
    for label in _INFO_PANEL_LABELS:
        panels = list(results.filter_by_label(label))
        if panels:
            panel = panels[0]
            break
    if panel is None:
        return None

    # 在面板内部查找被识别对象（图标）
    inner, entity_type_hint = _find_inner_icon_box(panel, results)
    if inner is None:
        return None

    # 名称区域: 从图标右边缘 → 面板右边缘，顶部约 15%
    panel_h = panel.h - panel.y
    name_x1 = int(inner.w)
    name_y1 = int(panel.y)
    name_x2 = int(panel.w)
    name_y2 = int(panel.y + panel_h * _INFO_PANEL_NAME_HEIGHT_RATIO)

    if name_x2 <= name_x1 + 10 or name_y2 <= name_y1 + 5:
        return None

    name_crop = frame[name_y1:name_y2, name_x1:name_x2]
    if name_crop is None or name_crop.size == 0:
        return None

    item_name = ocr_text(name_crop).strip()
    if not item_name:
        return None

    # 标准化 OCR 文本：全角→半角 + 日文形近字修正 + 去除杂字符
    raw_name = item_name
    item_name = _clean_ocr_item_name(item_name)
    if not item_name:
        return None

    if raw_name != item_name:
        logger.debug(
            "consult: 信息面板 OCR 原始文本 \"{}\" → 标准化后 \"{}\"",
            raw_name, item_name,
        )

    logger.debug(
        "consult: 信息面板 OCR 提取物品名: \"{}\" (type_hint={})",
        item_name, entity_type_hint,
    )
    return item_name, entity_type_hint, panel


def _match_info_panel_to_exchange_card(
    panel_inner_icon,
    candidates: list[ConsultActionCandidate],
) -> int:
    """将信息面板中的图标与交换卡列表中的图标进行视觉匹配。

    使用直方图相关性比对来判断信息面板中显示的是哪一张交换卡。

    Returns:
        匹配到的候选项索引，未匹配到返回 -1。
    """
    if panel_inner_icon is None:
        return -1
    panel_frame = getattr(panel_inner_icon, "frame", None)
    if panel_frame is None or panel_frame.size == 0:
        return -1

    # 统一尺寸以便比较
    target_size = (64, 64)
    panel_resized = cv2.resize(panel_frame, target_size)
    panel_hsv = cv2.cvtColor(panel_resized, cv2.COLOR_BGR2HSV)
    panel_hist = cv2.calcHist([panel_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(panel_hist, panel_hist, 0, 1, cv2.NORM_MINMAX)

    best_idx = -1
    best_score = -1.0

    for candidate in candidates:
        if candidate.kind != "exchange":
            continue
        icon_box = candidate.icon_box
        if icon_box is None:
            continue
        icon_frame = getattr(icon_box, "frame", None)
        if icon_frame is None or icon_frame.size == 0:
            continue

        icon_resized = cv2.resize(icon_frame, target_size)
        icon_hsv = cv2.cvtColor(icon_resized, cv2.COLOR_BGR2HSV)
        icon_hist = cv2.calcHist([icon_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(icon_hist, icon_hist, 0, 1, cv2.NORM_MINMAX)

        score = cv2.compareHist(panel_hist, icon_hist, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            best_idx = candidate.index

    # 相关性阈值：要求至少 0.5 以上才认为匹配
    if best_score < 0.5:
        logger.debug(
            "consult: 信息面板图标与交换卡匹配失败 (best_score={:.3f})",
            best_score,
        )
        return -1

    logger.debug(
        "consult: 信息面板匹配到交换卡 #{} (score={:.3f})",
        best_idx, best_score,
    )
    return best_idx


def _apply_info_panel_name(
    app: "AppProcessor",
    candidates: list[ConsultActionCandidate],
) -> None:
    """从当前可见的信息面板中提取物品名，并匹配到对应的交换卡。

    这是被动方式 —— 只读取当前画面中已经显示的信息面板，不产生任何交互。
    """
    results = app.latest_results
    frame = app.latest_frame
    if results is None or frame is None:
        return

    info = _extract_info_panel_item_name(results, frame)
    if info is None:
        return

    name, entity_type_hint, panel = info

    # DebugTools: 可视化信息面板和 OCR 提取结果
    debugger = getattr(app, "debug_tools", None) or DebugTools()
    debugger.add_box(
        int(panel.x), int(panel.y), int(panel.w), int(panel.h),
        label=f"InfoPanel: {name}",
        color=(0, 200, 255),  # 青色
        alpha=0.12,
        duration=3.0,
        font_size=20,
    )

    # 找到面板内部图标，用于与交换卡匹配
    inner, _ = _find_inner_icon_box(panel, results)
    match_idx = _match_info_panel_to_exchange_card(inner, candidates)
    if match_idx < 0 or match_idx >= len(candidates):
        return

    matched = candidates[match_idx]
    if matched.kind != "exchange":
        return

    # 设置匹配到的交换卡的 title
    matched.title = name
    if entity_type_hint and not matched.entity_type_hint:
        matched.entity_type_hint = entity_type_hint

    # DebugTools: 标注匹配到的交换卡
    matched_box = matched.box
    if matched_box is not None:
        debugger.add_box(
            int(matched_box.x), int(matched_box.y),
            int(matched_box.w), int(matched_box.h),
            label=f"InfoPanel→#{match_idx}: {name}",
            color=(0, 255, 200),  # 绿青色
            alpha=0.2,
            duration=3.0,
            font_size=16,
        )

    logger.info(
        "consult: 信息面板 OCR 识别到交换卡 #{} 名称: \"{}\"",
        match_idx, name,
    )


# 信息面板交互式识别：点击未识别卡片 → 等待信息面板刷新 → OCR 提取名称
_INFO_PANEL_TAP_WAIT = 0.4  # 点击后等待信息面板刷新的秒数
_INFO_PANEL_INFERENCE_WAIT = 0.3  # 等待 YOLO 推理完成的秒数

# 右边距 OCR 提取卡名时使用的标签（排除 INFO 面板本身，避免循环检测）
_RIGHT_EDGE_CARD_LABELS = (
    ProducerLabels.SKILL_CARD_ACTIVE,
    ProducerLabels.SKILL_CARD_MENTAL,
    ProducerLabels.SKILL_CARD_TRAP,
)


def _extract_card_name_from_right_edge(
    results,
    frame: np.ndarray,
) -> str | None:
    """从信息面板中技能卡图标的右边距提取卡名。

    方案：
      1. 找到画面中最靠上的技能卡 YOLO 框（位于信息面板内）
      2. 从该框的右边缘裁切到屏幕右边缘
      3. OCR 第一行 → 卡名

    相比 ``_extract_info_panel_item_name``，不依赖面板标签检测，更简洁稳健。
    """
    h, w = frame.shape[:2]

    # 收集所有技能卡 YOLO 框
    card_boxes = []
    for label in _RIGHT_EDGE_CARD_LABELS:
        for box in results.filter_by_label(label):
            card_boxes.append(box)
    if not card_boxes:
        return None

    # 取最靠上的技能卡框（信息面板中的「当前」卡图标通常是最上方的）
    top_card = min(card_boxes, key=lambda b: b.cy)

    crop_x1 = int(top_card.w)   # 卡片右边缘
    crop_y1 = int(top_card.y)   # 卡片顶部
    crop_x2 = w                 # 屏幕右边缘
    crop_y2 = int(top_card.h)   # 卡片底部

    if crop_x2 - crop_x1 < 50 or crop_y2 - crop_y1 < 10:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop is None or crop.size == 0:
        return None

    raw_text = ocr_text(crop).strip()
    if not raw_text:
        return None

    # 标准化并取第一行作为卡名
    raw_text = fullwidth_to_halfwidth(raw_text)
    raw_text = normalize_ocr_jp(raw_text)
    first_line = raw_text.split("\n")[0].strip()
    first_line = _OCR_ARTIFACT_RE.sub("", first_line).strip()
    return first_line if first_line else None


def _resolve_unidentified_via_info_panel(
    app: "AppProcessor",
    candidates: list[ConsultActionCandidate],
) -> None:
    """对未识别的交换卡逐个点击，读取信息面板提取物品名称并重新解析。

    仅对 kind == "exchange" 且 db_id 为空的候选项执行。
    点击交换卡只会切换选中状态（显示信息面板），不会触发兑换。
    """
    unresolved = [c for c in candidates if c.kind == "exchange" and not c.db_id]
    if not unresolved:
        return

    logger.info(
        "consult: {} 张交换卡未识别，开始逐个点击读取信息面板",
        len(unresolved),
    )

    for candidate in unresolved:
        try:
            # 点击交换卡以选中（不会触发兑换操作）
            app.device.click_element(candidate.box)
            time.sleep(_INFO_PANEL_TAP_WAIT)

            # 等待 YOLO 引擎处理新帧
            time.sleep(_INFO_PANEL_INFERENCE_WAIT)

            results = app.latest_results
            frame = app.latest_frame
            if results is None or frame is None:
                continue

            # 提取信息面板物品名
            info = _extract_info_panel_item_name(results, frame)
            if info is None:
                logger.debug(
                    "consult: 交换卡 #{} 点击后未检测到信息面板",
                    candidate.index,
                )
                continue

            name, entity_type_hint, _ = info
            candidate.title = name
            if entity_type_hint:
                candidate.entity_type_hint = entity_type_hint

            # 用 OCR 提取的名称重新调用解析管线
            resolution = resolve_produce_entity_identity(
                name,
                app=app,
                box=candidate.box,
                index=candidate.index,
                icon_box=candidate.icon_box,
                entity_type_hint=entity_type_hint or candidate.entity_type_hint,
            )
            _apply_resolution(candidate, resolution)

            if candidate.db_id:
                logger.info(
                    "consult: 交换卡 #{} 通过信息面板识别成功: \"{}\" → db_id={}",
                    candidate.index, name, candidate.db_id,
                )
            else:
                logger.warning(
                    "consult: 交换卡 #{} 信息面板 OCR=\"{}\" 但数据库未匹配",
                    candidate.index, name,
                )
        except Exception as exc:
            logger.warning(
                "consult: 交换卡 #{} 信息面板识别异常: {}",
                candidate.index, exc,
            )


# ── 强化 / 削除页面的主动识别 ──────────────────────────────────────

_SELECTION_TARGET_KINDS = {"enhancement_target", "remove_target"}
# 分行时的 Y 容差（像素）
_GRID_ROW_Y_TOLERANCE = 50


def _filter_grid_cards(all_boxes: list) -> list:
    """
    通过网格结构特征过滤出卡片网格区域的卡片，排除预览/信息面板区域的孤立卡片。

    原理：网格区域的卡片大小相近、按等距排列成行列结构。
    先从多卡行（独立列位置 ≥ 2）确定网格的列 x 坐标和行间距，
    再判断单卡行中的卡片是否对齐到已知网格列且行间距吻合。
    """
    if len(all_boxes) < 3:
        return list(all_boxes)

    # ── 按 cy 分行 ──
    sorted_by_y = sorted(all_boxes, key=lambda b: b.cy)
    rows: list[list] = []
    for box in sorted_by_y:
        placed = False
        for row in rows:
            if abs(box.cy - row[0].cy) <= _GRID_ROW_Y_TOLERANCE:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    def _unique_col_count(row: list) -> int:
        """行内按 cx 去重，统计独立列数（排除同位置双重检测）。"""
        cols = sorted(b.cx for b in row)
        unique = [cols[0]]
        for cx in cols[1:]:
            if cx - unique[-1] > _GRID_ROW_Y_TOLERANCE:
                unique.append(cx)
        return len(unique)

    # ── 识别网格行（独立列数 ≥ 2，排除同位置双重检测的干扰）──
    multi_rows = [r for r in rows if _unique_col_count(r) >= 2]

    if not multi_rows:
        # 没有明显的网格行，返回全部
        return list(all_boxes)

    # ── 从多卡行提取网格列 x 坐标 ──
    grid_col_xs: list[int] = []
    for row in multi_rows:
        for b in row:
            if not any(abs(b.cx - cx) <= _GRID_ROW_Y_TOLERANCE for cx in grid_col_xs):
                grid_col_xs.append(b.cx)
    grid_col_xs.sort()

    # ── 计算网格行间距 ──
    multi_row_ys = sorted(sum(b.cy for b in r) / len(r) for r in multi_rows)
    if len(multi_row_ys) >= 2:
        row_gaps = [multi_row_ys[i + 1] - multi_row_ys[i]
                    for i in range(len(multi_row_ys) - 1)]
        avg_row_gap = sum(row_gaps) / len(row_gaps)
    else:
        avg_row_gap = max((b.h - b.y) for b in multi_rows[0]) * 1.2

    # ── 过滤：多卡行全部保留，单卡行需列对齐 + 行间距吻合 ──
    grid_boxes: list = []
    for row in rows:
        if _unique_col_count(row) >= 2:
            grid_boxes.extend(row)
        else:
            box = row[0]
            col_ok = any(abs(box.cx - cx) <= _GRID_ROW_Y_TOLERANCE
                         for cx in grid_col_xs)
            row_cy = box.cy
            y_dist = min(abs(row_cy - mry) for mry in multi_row_ys)
            row_ok = y_dist <= avg_row_gap * 1.5
            if col_ok and row_ok:
                grid_boxes.extend(row)

    return grid_boxes


# 灰色（不可强化）卡片的亮度阈值：正常卡 V≈230+，灰色卡 V≈120-
_GRAYED_CARD_V_THRESHOLD = 170


def _is_card_grayed_out(box, frame: np.ndarray, *, debugger: DebugTools | None = None) -> bool:
    """
    判断卡片缩略图是否为灰色（不可强化/已满级）。
    通过 HSV 空间的亮度(V)通道判断：灰色卡片整体明显偏暗。
    """
    card_region = frame[box.y:box.h, box.x:box.w]
    if card_region.size == 0:
        return False
    # 取中央 50% 区域，避免边框/选中框干扰
    ch, cw = card_region.shape[:2]
    margin_h, margin_w = ch // 4, cw // 4
    center = card_region[margin_h:ch - margin_h, margin_w:cw - margin_w]
    if center.size == 0:
        return False
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    mean_v = float(hsv[:, :, 2].mean())
    grayed = mean_v < _GRAYED_CARD_V_THRESHOLD
    if debugger is not None:
        debugger.add_box(
            box.x, box.y, box.w, box.h,
            label=f"灰卡V={mean_v:.0f}" if grayed else f"正常V={mean_v:.0f}",
            color=(128, 128, 128) if grayed else (0, 200, 0),
            alpha=0.15,
            duration=3.0,
        )
    return grayed


# ── 已兑换卡片检测 ──
# 已交换的卡片有半透明✓遮罩，导致图标区域亮度(V)下降且饱和度(S)极低
_EXCHANGED_CARD_V_THRESHOLD = 220
_EXCHANGED_CARD_S_THRESHOLD = 50


def _is_card_exchanged(box, frame: np.ndarray, *, debugger: DebugTools | None = None) -> bool:
    """
    判断 CARD_ITEM_EXCHANGE 框对应的卡片是否已交换（交換済）。
    已交换卡片有✓遮罩 + 交換済文字，使图标区域 V 和 S 同时偏低。
    取框的上部 60%（图标区域，排除底部价格栏）的中心 50% 分析。
    """
    # CARD_ITEM_EXCHANGE 框包含底部价格栏，只取上部 60% 为图标区域
    full_h = box.h - box.y
    icon_h = int(full_h * 0.6)
    icon_region = frame[box.y:box.y + icon_h, box.x:box.w]
    if icon_region.size == 0:
        return False
    ih, iw = icon_region.shape[:2]
    margin_h, margin_w = ih // 4, iw // 4
    center = icon_region[margin_h:ih - margin_h, margin_w:iw - margin_w]
    if center.size == 0:
        return False
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    mean_v = float(hsv[:, :, 2].mean())
    mean_s = float(hsv[:, :, 1].mean())
    exchanged = mean_v < _EXCHANGED_CARD_V_THRESHOLD and mean_s < _EXCHANGED_CARD_S_THRESHOLD
    if debugger is not None:
        debugger.add_box(
            box.x, box.y, box.w, box.h,
            label=f"交換済V={mean_v:.0f},S={mean_s:.0f}" if exchanged else f"未交換V={mean_v:.0f},S={mean_s:.0f}",
            color=(200, 100, 0) if exchanged else (0, 200, 100),
            alpha=0.15,
            duration=3.0,
        )
    return exchanged


def _close_modal_if_present(app: "AppProcessor") -> bool:
    """检测并关闭意外弹出的模态窗（如卡片详细弹窗），返回是否关闭了模态窗。"""
    results = app.latest_results
    if results is None:
        return False
    modal_headers = list(results.filter_by_label(BaseUILabels.MODAL_HEADER))
    if not modal_headers:
        return False
    # 寻找关闭/取消按钮
    cancel_boxes = list(results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
    if cancel_boxes:
        app.device.click_element(cancel_boxes[0])
    else:
        close_boxes = list(results.filter_by_label(BaseUILabels.CLOSE_BUTTON))
        if close_boxes:
            app.device.click_element(close_boxes[0])
        else:
            return False
    time.sleep(_INFO_PANEL_TAP_WAIT)
    time.sleep(_INFO_PANEL_INFERENCE_WAIT)
    logger.info("consult: 检测到弹出模态窗，已自动关闭")
    return True


def _resolve_selection_targets_via_probe(
    app: "AppProcessor",
    candidates: list[ConsultActionCandidate],
) -> None:
    """对未识别的强化/削除目标卡逐个点击，读取信息面板提取卡名。

    注意事项:
      - 首次点击后信息面板出现，缩略图格子会整体下移，原始坐标失效
      - 每次点击后必须重新检测缩略图位置
      - 可能意外打开卡片详细弹窗，需要检测并关闭
      - YOLO 可能对同一张卡检测出重叠框，需按位置去重
    """
    unresolved = [
        c for c in candidates
        if c.kind in _SELECTION_TARGET_KINDS and not c.db_id
    ]
    if not unresolved:
        return

    logger.info(
        "consult: {} 张目标卡未识别（CLIP 未命中），开始逐卡点击自动学习",
        len(unresolved),
    )

    # 跟踪已点击的格子位置（按行列索引），避免重复点击
    # 格子列宽约 220px，用 (col_idx, row_idx) 标识
    _GRID_COL_WIDTH = 220
    _GRID_ROW_HEIGHT = 220
    clicked_grid_cells: set[tuple[int, int]] = set()
    identified_db_ids: dict[str, str] = {}  # db_id → 展示名称
    last_info_name: str | None = None
    max_rounds = len(unresolved) + 3  # 安全上限（略多于候选数）

    for round_idx in range(max_rounds):
        # 等待 YOLO 引擎更新
        time.sleep(_INFO_PANEL_TAP_WAIT)
        time.sleep(_INFO_PANEL_INFERENCE_WAIT)

        results = app.latest_results
        frame = app.latest_frame
        if results is None or frame is None:
            break

        # 检查并关闭意外弹出的模态窗
        if _close_modal_if_present(app):
            continue

        # 重新检测缩略图格子中的卡片（通过网格结构排除预览区域）
        all_cards = []
        for label in _CONSULT_THUMBNAIL_LABELS:
            for box in results.filter_by_label(label):
                all_cards.append(box)
        grid_cards = _filter_grid_cards(all_cards)
        # 过滤灰色（不可强化/已满级）卡片
        grid_cards = [b for b in grid_cards if not _is_card_grayed_out(b, frame, debugger=getattr(app, "debug_tools", None))]
        grid_cards = _dedup_boxes_by_position(grid_cards, tolerance=60)
        grid_cards.sort(key=lambda b: (b.cy, b.cx))

        if not grid_cards:
            break

        # 找到下一张未点击的缩略图
        next_box = None
        for box in grid_cards:
            col_idx = round(box.cx / _GRID_COL_WIDTH)
            row_idx = round(box.cy / _GRID_ROW_HEIGHT)
            cell_key = (col_idx, row_idx)
            if cell_key not in clicked_grid_cells:
                clicked_grid_cells.add(cell_key)
                next_box = box
                break

        if next_box is None:
            break  # 所有格子都已点击过

        # 点击缩略图
        app.device.click_element(next_box)
        time.sleep(_INFO_PANEL_TAP_WAIT)
        time.sleep(_INFO_PANEL_INFERENCE_WAIT)

        results = app.latest_results
        frame = app.latest_frame
        if results is None or frame is None:
            continue

        # 检查点击后是否弹出了模态窗
        if _close_modal_if_present(app):
            continue

        # 从信息面板内卡片图标的右边距提取卡名
        name = _extract_card_name_from_right_edge(results, frame)
        if not name:
            logger.debug(
                "consult: 目标卡探查 round={} 右边距 OCR 未提取到卡名",
                round_idx,
            )
            continue

        # 与上次相同则说明点击未生效，跳过
        if name == last_info_name:
            logger.debug(
                "consult: 目标卡探查 round={} 信息面板未更新 (仍为 \"{}\")",
                round_idx, name,
            )
            continue
        last_info_name = name

        # 找到距离最近的未识别候选并更新
        best_candidate = None
        best_dist = float("inf")
        for c in unresolved:
            if c.db_id:
                continue
            dx = c.box.cx - next_box.cx
            dy = c.box.cy - next_box.cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_candidate = c

        if best_candidate is None:
            continue

        best_candidate.title = name

        # 用 OCR 提取的卡名走 DB 匹配 + CLIP 学习
        resolution = resolve_produce_card_identity(
            app,
            title=name,
            box=best_candidate.box,
            index=best_candidate.index,
        )

        # 覆盖 action_id 为相談选择操作格式
        consult_action = (
            "consult_select_remove_target"
            if best_candidate.kind == "remove_target"
            else "consult_select_enhancement_target"
        )
        from src.core.tasks.producer_challenge.gameplay.decision import (
            CandidateResolution,
        )
        resolution = CandidateResolution(
            action_id=f"{consult_action}:{resolution.db_id or best_candidate.index}",
            candidate_type="consult_action",
            db_id=resolution.db_id,
            display_name=str(resolution.metadata.get("raw_name") or name),
            source=resolution.source,
            confidence=resolution.confidence,
            metadata={
                **resolution.metadata,
                "consult_action": consult_action,
            },
        )
        _apply_resolution(best_candidate, resolution)

        if best_candidate.db_id:
            identified_db_ids[best_candidate.db_id] = best_candidate.title
            logger.info(
                "consult: 目标卡 #{} 自动学习成功: \"{}\" → db_id={}",
                best_candidate.index, name, best_candidate.db_id,
            )
        else:
            logger.warning(
                "consult: 目标卡 #{} OCR=\"{}\" 但数据库未匹配",
                best_candidate.index, name,
            )

    # 对于仍有同 db_id 的其他未识别候选，用已知结果填充
    for c in unresolved:
        if c.db_id:
            continue
        # 尝试用位置相近的已识别候选填充（同位置 YOLO 重复检测）
        for ic in candidates:
            if ic.db_id and ic is not c:
                dx = abs(c.box.cx - ic.box.cx)
                dy = abs(c.box.cy - ic.box.cy)
                if dx < 60 and dy < 60:
                    c.title = ic.title
                    c.db_id = ic.db_id
                    c.action_id = ic.action_id
                    c.source = ic.source
                    c.confidence = ic.confidence
                    if ic.metadata:
                        c.metadata = dict(ic.metadata)
                    break


def _consult_subflow_mode(ctx: "ProduceContext") -> str:
    """处理consult、subflow、mode并返回结果。

    Args:
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。

    Returns:
        str: 处理后的文本结果。
    """
    mode = str(ctx.handler_state.get("consult_pending_mode", "") or "")
    if mode in {"enhancement", "remove"}:
        return mode
    last_subaction = str(ctx.handler_state.get("consult_last_subaction", "") or "")
    if "remove" in last_subaction:
        return "remove"
    return "enhancement"


def _consult_exchange_signature(
    candidates: Sequence[ConsultActionCandidate],
) -> tuple[str, ...]:
    """构建当前兑换候选签名，用于判断页面是否变化。"""
    signature: list[str] = []
    for candidate in candidates:
        if candidate.kind == "exit":
            continue
        signature.append(
            str(
                candidate.action_id
                or candidate.db_id
                or candidate.title
                or f"{candidate.kind}:{candidate.index}"
            )
        )
    return tuple(signature)


def _consult_current_p_points(ctx: "ProduceContext") -> int:
    """读取当前可用 P 点，优先使用咨询剩余值。"""
    return int(ctx.consult_remaining_p_points or ctx.hud_p_point or 0)


def _remember_consult_exchange_state(
    ctx: "ProduceContext",
    target: ConsultActionCandidate,
    candidates: Sequence[ConsultActionCandidate],
) -> None:
    """记录一次兑换点击后的页面签名与资源快照。"""
    ctx.handler_state["consult_last_exchange_action_id"] = target.action_id or ""
    ctx.handler_state["consult_last_exchange_db_id"] = target.db_id or ""
    ctx.handler_state["consult_last_exchange_p_points"] = _consult_current_p_points(ctx)
    ctx.handler_state["consult_last_exchange_signature"] = list(
        _consult_exchange_signature(candidates)
    )
    ctx.handler_state["consult_waiting_exchange_result"] = True
    ctx.handler_state["consult_exchange_progressed"] = False
    ctx.handler_state["consult_exchange_retry_count"] = 0


def _resolve_unchanged_exchange_retry(
    ctx: "ProduceContext",
    candidates: Sequence[ConsultActionCandidate],
) -> int | None:
    """当兑换后页面未变化时，决定是否触发重试点击。"""
    if not bool(ctx.handler_state.get("consult_waiting_exchange_result")):
        return None
    if str(ctx.handler_state.get("consult_last_subaction", "") or "") != "exchange":
        return None

    last_action_id = str(ctx.handler_state.get("consult_last_exchange_action_id", "") or "")
    last_db_id = str(ctx.handler_state.get("consult_last_exchange_db_id", "") or "")
    if not last_action_id and not last_db_id:
        return None

    current_signature = tuple(str(value or "") for value in _consult_exchange_signature(candidates))
    previous_signature = tuple(
        str(value or "")
        for value in (ctx.handler_state.get("consult_last_exchange_signature") or [])
    )
    current_p_points = _consult_current_p_points(ctx)
    previous_p_points = int(
        ctx.handler_state.get("consult_last_exchange_p_points") or current_p_points
    )

    if current_signature != previous_signature or current_p_points != previous_p_points:
        ctx.handler_state["consult_exchange_progressed"] = True
        ctx.handler_state["consult_waiting_exchange_result"] = False
        ctx.handler_state["consult_last_exchange_signature"] = list(current_signature)
        ctx.handler_state["consult_last_exchange_p_points"] = current_p_points
        ctx.handler_state.pop("consult_exchange_retry_count", None)
        return None

    retry_count = int(ctx.handler_state.get("consult_exchange_retry_count") or 0)
    if retry_count >= _CONSULT_EXCHANGE_RETRY_THRESHOLD:
        return None

    for idx, candidate in enumerate(candidates):
        if last_db_id and str(candidate.db_id or "") == last_db_id:
            match_index = idx
            break
        if last_action_id and str(candidate.action_id or "") == last_action_id:
            match_index = idx
            break
    else:
        return None

    retry_count += 1
    ctx.handler_state["consult_exchange_retry_count"] = retry_count
    logger.warning(
        "consult: 上次兑换点击后画面未变化，重试同一候选 {!r} ({}/{})",
        candidates[match_index].title or candidates[match_index].action_id,
        retry_count,
        _CONSULT_EXCHANGE_RETRY_THRESHOLD,
    )
    return match_index


# ── 信息面板识别结果缓存 ──────────────────────────────────────────
# 避免每次循环重复点击所有未识别交换卡读取信息面板。
# 缓存 key 使用 box 中心坐标量化值（容差 30px），存储在 ctx.handler_state 中。

_CACHE_POS_TOLERANCE = 30  # 像素容差，同一张卡在不同帧间 YOLO 检测框位置微小抖动

_CONSULT_CACHE_KEY = "_consult_resolved_cache"


def _pos_key(box: Any) -> tuple[int, int]:
    """将 box 中心坐标量化为缓存 key。"""
    cx = int(round(getattr(box, "cx", 0) / _CACHE_POS_TOLERANCE) * _CACHE_POS_TOLERANCE)
    cy = int(round(getattr(box, "cy", 0) / _CACHE_POS_TOLERANCE) * _CACHE_POS_TOLERANCE)
    return (cx, cy)


def _apply_cached_resolutions(
    ctx: "ProduceContext",
    candidates: list[ConsultActionCandidate],
) -> None:
    """从 handler_state 缓存中恢复之前识别的结果到匹配位置的候选项。"""
    cache: dict = ctx.handler_state.get(_CONSULT_CACHE_KEY, {})
    if not cache:
        return

    applied = 0
    for cand in candidates:
        if cand.db_id:
            continue  # 已经有 db_id，跳过
        key = _pos_key(cand.box)
        cached = cache.get(key)
        if cached is None:
            continue
        # 恢复缓存的识别结果
        cand.db_id = cached.get("db_id", "")
        cand.action_id = cached.get("action_id", "")
        cand.title = cached.get("title", "") or cand.title
        cand.entity_type_hint = cached.get("entity_type_hint", "") or cand.entity_type_hint
        cand.source = cached.get("source", "cache")
        cand.confidence = cached.get("confidence", 0.0)
        if cached.get("metadata"):
            cand.metadata.update(cached["metadata"])
        applied += 1

    if applied:
        logger.debug("consult: 从缓存恢复了 {} 个候选项的识别结果", applied)


def _cache_resolved_candidates(
    ctx: "ProduceContext",
    candidates: list[ConsultActionCandidate],
) -> None:
    """将已识别（有 db_id）的候选项写入缓存。"""
    cache: dict = ctx.handler_state.setdefault(_CONSULT_CACHE_KEY, {})

    for cand in candidates:
        if not cand.db_id:
            continue
        key = _pos_key(cand.box)
        if key in cache:
            continue  # 已缓存，跳过
        cache[key] = {
            "db_id": cand.db_id,
            "action_id": cand.action_id,
            "title": cand.title,
            "entity_type_hint": cand.entity_type_hint,
            "source": cand.source,
            "confidence": cand.confidence,
            "metadata": dict(cand.metadata) if cand.metadata else {},
        }
        logger.debug(
            "consult: 缓存候选项 pos={} → db_id={}, title={}",
            key, cand.db_id, cand.title,
        )


def detect_consult_actions(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> List[ConsultActionCandidate]:
    """检测consult、actions并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        list: 结果列表，元素类型见返回注解。
    """
    candidates: list[ConsultActionCandidate] = []

    if position == GameplayPosition.CONSULT_EXCHANGE:
        # 过滤已交换（交換済）的卡片
        frame = app.latest_frame
        for box in _sorted_boxes(app.latest_results.filter_by_label(ProducerLabels.CARD_ITEM_EXCHANGE)):
            if frame is not None and _is_card_exchanged(box, frame, debugger=getattr(app, "debug_tools", None)):
                logger.debug("consult: 跳过已交换卡片 box=({},{},{},{})", box.x, box.y, box.w, box.h)
                continue
            # 查找内层 YOLO 子检测（P Drink / Skill Card），获取更精确的图标裁切
            icon_box, entity_type_hint = _find_inner_icon_box(box, app.latest_results)
            # 提取价格信息（卡片底部的 P 点数数字）。
            price = _extract_exchange_price(box.frame)
            candidates.append(
                ConsultActionCandidate(
                    index=len(candidates),
                    kind="exchange",
                    title="",  # 交换卡无可读文本，由信息面板 OCR 或 CLIP 识别
                    box=box,
                    icon_box=icon_box,
                    entity_type_hint=entity_type_hint,
                    metadata={"price": price} if price else {},
                )
            )

        # ── 被动方式: 从当前可见的信息面板提取物品名（不产生交互） ──
        _apply_info_panel_name(app, candidates)

        for label, kind, fallback_title in (
            (ProducerLabels.PC_SKILL_CARD_ENHANCEMENT, "enhance", GeneralText.ENHANCE),
            (ProducerLabels.PC_SKILL_CARD_REMOVE, "delete", ProduceText.SKILL_CARD_REMOVE),
        ):
            for box in _sorted_boxes(app.latest_results.filter_by_label(label)):
                candidates.append(
                    ConsultActionCandidate(
                        index=len(candidates),
                        kind=kind,
                        title=ocr_text(box.frame) or fallback_title,
                        box=box,
                    )
                )
        # 收集退出按钮（Close Button）
        for box in app.latest_results.filter_by_label(ProducerLabels.CLOSE_BUTTON):
            candidates.append(
                ConsultActionCandidate(
                    index=len(candidates),
                    kind="exit",
                    title=ButtonText.EXIT,
                    box=box,
                )
            )
    elif position in CONSULT_SELECTION_POSITIONS:
        subflow_mode = _consult_subflow_mode(ctx)
        pending_target = str(ctx.handler_state.get("consult_enhancement_target_label", "") or "")
        target_kind = _consult_target_kind_for_mode(subflow_mode)

        # 收集缩略图格子中的技能卡（通过网格结构排除预览区域的卡片）
        all_card_boxes = []
        for label in _CONSULT_THUMBNAIL_LABELS:
            for box in _sorted_boxes(app.latest_results.filter_by_label(label)):
                all_card_boxes.append(box)
        raw_boxes = _filter_grid_cards(all_card_boxes)
        # 过滤灰色（不可强化/已满级）卡片
        frame = app.latest_frame
        if frame is not None:
            raw_boxes = [b for b in raw_boxes if not _is_card_grayed_out(b, frame, debugger=getattr(app, "debug_tools", None))]

        # 按位置去重（YOLO 可能对同一张卡检测出多个框）
        deduped = _dedup_boxes_by_position(raw_boxes, tolerance=60)

        for box in deduped:
            title = _clean_ocr_item_name(ocr_text(box.frame))
            candidates.append(
                ConsultActionCandidate(
                    index=len(candidates),
                    kind=target_kind,
                    title=title,
                    selected=bool(pending_target and pending_target == title),
                    box=box,
                )
            )

        confirm_boxes = list(app.latest_results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
        if not confirm_boxes:
            confirm_boxes = list(app.latest_results.filter_by_label(BaseUILabels.BUTTON))
        if confirm_boxes:
            confirm_box = max(confirm_boxes, key=lambda item: item.cy)
            candidates.append(
                ConsultActionCandidate(
                    index=len(candidates),
                    kind=_consult_confirm_kind_for_mode(subflow_mode),
                    title=ocr_text(confirm_box.frame) or (
                        ProduceText.SKILL_CARD_REMOVE if subflow_mode == "remove" else ProduceText.ENHANCE_CONFIRM
                    ),
                    box=confirm_box,
                )
            )

        cancel_boxes = list(app.latest_results.filter_by_label(ProducerLabels.CANCEL_BUTTON))
        if cancel_boxes:
            cancel_box = max(cancel_boxes, key=lambda item: item.cy)
            candidates.append(
                ConsultActionCandidate(
                    index=len(candidates),
                    kind="exit",
                    title=ocr_text(cancel_box.frame) or ButtonText.EXIT,
                    box=cancel_box,
                )
            )

    hydrate_consult_candidates(app, candidates)

    # ── 从缓存恢复之前通过信息面板识别的结果，避免每次循环重复点击识别 ──
    _apply_cached_resolutions(ctx, candidates)

    # ── 主动方式: 逐个点击未识别卡片，读取信息面板提取名称 ──
    if position == GameplayPosition.CONSULT_EXCHANGE:
        _resolve_unidentified_via_info_panel(app, candidates)
    elif position in CONSULT_SELECTION_POSITIONS:
        _resolve_selection_targets_via_probe(app, candidates)

    # ── 将新识别的结果写入缓存 ──
    _cache_resolved_candidates(ctx, candidates)

    return candidates


def decide_consult_action(
    app: "AppProcessor",
    ctx: "ProduceContext",
    candidates: List[ConsultActionCandidate],
    *,
    position: str,
) -> int:
    """决策consult、操作并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        ctx: 培育上下文对象，保存跨步骤状态与策略配置。
        candidates: 候选项列表，供策略或规则选择目标动作。
        position: 当前阶段下的细分画面位置标识。

    Returns:
        int: 计算得到的数值结果。
    """
    # ── CONSULT 总操作次数硬限制，防止 LLM 无限购买 ──
    _CONSULT_TOTAL_OP_LIMIT = 30
    consult_op_count = ctx.handler_state.get("consult_total_op_count", 0) + 1
    ctx.handler_state["consult_total_op_count"] = consult_op_count
    if consult_op_count > _CONSULT_TOTAL_OP_LIMIT and position == GameplayPosition.CONSULT_EXCHANGE:
        logger.warning(
            "consult: 总操作次数 {} 超过硬限制 {}，强制退出",
            consult_op_count,
            _CONSULT_TOTAL_OP_LIMIT,
        )
        for idx, candidate in enumerate(candidates):
            if candidate.kind == "exit":
                return idx

    if position == GameplayPosition.CONSULT_EXCHANGE:
        retry_index = _resolve_unchanged_exchange_retry(ctx, candidates)
        if retry_index is not None:
            # 标记本次为重试，避免 execute_consult_step 重置计数器
            ctx.handler_state["_consult_is_exchange_retry"] = True
            return retry_index
    # 非重试路径
    ctx.handler_state["_consult_is_exchange_retry"] = False

    decision_state = build_decision_state(
        app,
        ctx,
        phase="consult",
        position=position,
        candidates=candidates,
        reason="consult_decision",
    )
    # ── 已选中目标时自动确认，防止 LLM 反复选卡死循环 ──
    pending_target = ctx.handler_state.get("consult_enhancement_target")
    if pending_target and position != GameplayPosition.CONSULT_EXCHANGE:
        subflow_mode = _consult_subflow_mode(ctx)
        confirm_kind = _consult_confirm_kind_for_mode(subflow_mode)
        for idx, candidate in enumerate(candidates):
            if candidate.kind == confirm_kind:
                logger.info(
                    "consult: 已选中目标 '{}' → 自动确认 {} (idx={})",
                    ctx.handler_state.get("consult_enhancement_target_label", ""),
                    confirm_kind,
                    idx,
                )
                return idx
        # 未找到确认按钮，清除 pending 状态避免死循环
        logger.warning("consult: 已选中目标但未找到确认按钮，清除 pending 状态")
        ctx.handler_state.pop("consult_enhancement_target", None)
        ctx.handler_state.pop("consult_enhancement_target_label", None)

    decision = invoke_decision_strategy(
        ctx.consult_strategy,
        app,
        ctx,
        candidates,
        decision_state=decision_state,
    )
    consult_entered_fresh = False
    if decision is not None:
        decided_index = resolve_candidate_index(decision, candidates)
        decided_candidate = candidates[decided_index]
        if position == GameplayPosition.CONSULT_EXCHANGE:
            has_non_exit_candidate = any(candidate.kind != "exit" for candidate in candidates)
            consult_entered_fresh = not (
                ctx.handler_state.get("consult_last_subaction")
                or ctx.handler_state.get("consult_auto_used_enhancement")
            )
            consult_waiting_without_progress = bool(
                ctx.handler_state.get("consult_waiting_exchange_result")
            ) and not bool(ctx.handler_state.get("consult_exchange_progressed"))
            if (
                decided_candidate.kind == "exit"
                and has_non_exit_candidate
                and (consult_entered_fresh or consult_waiting_without_progress)
            ):
                logger.warning(
                    "consult: 当前相談尚未产生有效收益时 LLM 选择退出，改走本地兜底"
                )
            else:
                return decided_index
        else:
            return decided_index

    # ── 本地兜底逻辑 ──
    fallback_index: int | None = None
    fallback_reason = ""

    if position == GameplayPosition.CONSULT_EXCHANGE:
        # 卡顿检测：如果连续多次停留在 consult_exchange，直接退出
        exchange_stuck = ctx.handler_state.get("consult_exchange_stuck", 0) + 1
        ctx.handler_state["consult_exchange_stuck"] = exchange_stuck
        CONSULT_STUCK_THRESHOLD = 5

        if exchange_stuck <= CONSULT_STUCK_THRESHOLD:
            if not ctx.handler_state.get("consult_auto_used_enhancement"):
                for idx, candidate in enumerate(candidates):
                    if candidate.kind == "enhance":
                        fallback_index = idx
                        fallback_reason = "自动强化"
                        break
        else:
            # 连续卡顿超过阈值，强制标记已使用过强化，后续直接退出
            logger.warning(
                "consult: exchange页面卡顿 {} 次，跳过强化直接退出",
                exchange_stuck,
            )
            ctx.handler_state["consult_auto_used_enhancement"] = True

        if fallback_index is None:
            # 强化完成后，优先退出相談页面
            for idx, candidate in enumerate(candidates):
                if candidate.kind == "exit":
                    fallback_index = idx
                    fallback_reason = "退出相談"
                    break
        if fallback_index is None:
            for idx, candidate in enumerate(candidates):
                if candidate.kind == "exchange":
                    fallback_index = idx
                    fallback_reason = "交换"
                    break
        if fallback_index is None:
            for idx, candidate in enumerate(candidates):
                if candidate.kind == "enhance":
                    fallback_index = idx
                    fallback_reason = "强化"
                    break
        if fallback_index is None:
            fallback_index = 0
            fallback_reason = "默认首选"
    else:
        subflow_mode = _consult_subflow_mode(ctx)
        pending_target = ctx.handler_state.get("consult_enhancement_target")
        target_kind = _consult_target_kind_for_mode(subflow_mode)
        confirm_kind = _consult_confirm_kind_for_mode(subflow_mode)
        if pending_target:
            for idx, candidate in enumerate(candidates):
                if candidate.kind == confirm_kind:
                    fallback_index = idx
                    fallback_reason = f"确认({confirm_kind})"
                    break
        if fallback_index is None:
            for idx, candidate in enumerate(candidates):
                if candidate.kind == target_kind:
                    fallback_index = idx
                    fallback_reason = f"目标({target_kind})"
                    break
        if fallback_index is None:
            for idx, candidate in enumerate(candidates):
                if candidate.kind == confirm_kind:
                    fallback_index = idx
                    fallback_reason = f"确认({confirm_kind})"
                    break
        if fallback_index is None:
            for idx, candidate in enumerate(candidates):
                if candidate.kind == "exit":
                    fallback_index = idx
                    fallback_reason = "退出"
                    break
        if fallback_index is None:
            fallback_index = 0
            fallback_reason = "默认首选"

    # 更新 dump 记录
    from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import DecisionDumper
    _dumper = DecisionDumper.get_instance()
    resolved_name = ""
    if 0 <= fallback_index < len(candidates):
        resolved_name = getattr(candidates[fallback_index], "title", "") or getattr(candidates[fallback_index], "kind", "")
    _dumper.update_last_resolved(
        resolved_index=fallback_index,
        resolved_name=resolved_name,
        fallback_used=True,
        fallback_reason=fallback_reason,
    )
    logger.info("consult: 兜底决策 idx={} reason={}", fallback_index, fallback_reason)
    return fallback_index


def execute_consult_step(
    app: "AppProcessor",
    ctx: "ProduceContext",
    *,
    position: str,
) -> ConsultActionCandidate | None:
    """执行一轮咨询页面的识别、决策与点击操作。"""
    # 非 exchange 页面时重置卡顿计数
    if position != GameplayPosition.CONSULT_EXCHANGE:
        ctx.handler_state.pop("consult_exchange_stuck", None)

    candidates = detect_consult_actions(app, ctx, position=position)
    if not candidates:
        return None

    target_index = decide_consult_action(app, ctx, candidates, position=position)
    target = candidates[target_index]
    app.device.click_element(target.box)

    # ── 兑换卡片需要两步：先点卡片打开信息面板，再点“交換する”完成购买。──
    if target.kind == "exchange" and position == GameplayPosition.CONSULT_EXCHANGE:
        # 在当前帧预先定位“交換する”按钮（Universal Confirm button）。
        confirm_boxes = list(app.latest_results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
        if confirm_boxes:
            # 选择最靠下的 Confirm 按钮（“交換する”通常在底部）。
            confirm_box = max(confirm_boxes, key=lambda item: item.cy)
            time.sleep(0.4)  # 等待信息面板刷新
            app.device.click_element(confirm_box)
            logger.info(
                "consult: 点击'交換する'按钮完成兑换 (target={})",
                target.title or target.action_id or f"#{target.index}",
            )
            time.sleep(0.5)  # 等待兑换动画
        else:
            logger.warning("consult: 未找到'交換する'按钮，兑换可能未完成")

    if target.kind in {"enhancement_target", "remove_target"}:
        pending_mode = "remove" if target.kind == "remove_target" else "enhancement"
        operation_name = _consult_select_operation_for_mode(pending_mode)
        ctx.handler_state["consult_enhancement_target"] = target.db_id or target.action_id or str(target.index)
        ctx.handler_state["consult_enhancement_target_label"] = target.title or target.action_id
        ctx.handler_state["consult_pending_mode"] = pending_mode
        ctx.handler_state["consult_last_subaction"] = operation_name.removeprefix("consult_")
        ctx.record_operation(
            operation_name,
            target=target.title or target.action_id,
            details={
                "index": target.index,
                "action_id": target.action_id,
                "db_id": target.db_id,
            },
        )
        return target

    if target.kind == "confirm_enhancement":
        ctx.handler_state["consult_auto_used_enhancement"] = True
        ctx.handler_state["consult_last_subaction"] = "confirm_enhancement"
        enhance_card_id = str(ctx.handler_state.get("consult_enhancement_target") or target.db_id or "")
        ctx.record_operation(
            _consult_confirm_operation_for_mode("enhancement"),
            target=ctx.handler_state.get("consult_enhancement_target_label", "") or target.title or target.action_id,
            details={
                "index": target.index,
                "action_id": target.action_id,
                "db_id": target.db_id,
            },
        )
        ctx.mutate_deck_enhance(enhance_card_id)
        return target

    if target.kind == "confirm_remove":
        ctx.handler_state["consult_auto_used_remove"] = True
        ctx.handler_state["consult_last_subaction"] = "confirm_remove"
        remove_card_id = str(ctx.handler_state.get("consult_enhancement_target") or target.db_id or "")
        ctx.record_operation(
            _consult_confirm_operation_for_mode("remove"),
            target=ctx.handler_state.get("consult_enhancement_target_label", "") or target.title or target.action_id,
            details={
                "index": target.index,
                "action_id": target.action_id,
                "db_id": target.db_id,
            },
        )
        ctx.mutate_deck_remove(remove_card_id)
        ctx.clear_consult_pending()
        return target

    if target.kind == "exit":
        ctx.clear_consult_pending()
        ctx.record_operation(
            "consult_exit",
            target=target.title or target.action_id or "consult_exit",
            details={
                "index": target.index,
                "action_id": target.action_id,
            },
        )
        # CONSULT退出后会有过渡动画，给足重试时间
        ctx.handler_state["unknown_retry_override"] = {
            "reason": "consult_exit_transition",
            "retry_limit": 15,
            "retry_sleep": 1.0,
        }
        return target

    if target.kind == "enhance":
        ctx.handler_state["consult_pending_mode"] = "enhancement"
        ctx.handler_state["consult_last_subaction"] = "open_enhancement"
    elif target.kind == "delete":
        ctx.handler_state["consult_pending_mode"] = "remove"
        ctx.handler_state["consult_last_subaction"] = "open_remove"
    else:
        ctx.handler_state["consult_last_subaction"] = "exchange"
        # 重试时不重置计数器，否则 retry_count 永远达不到阈值
        if not ctx.handler_state.pop("_consult_is_exchange_retry", False):
            _remember_consult_exchange_state(ctx, target, candidates)

    ctx.record_operation(
        f"consult_{target.kind}",
        target=target.title or target.action_id or f"consult_{target.index + 1}",
        details={
            "index": target.index,
            "kind": target.kind,
            "action_id": target.action_id,
            "db_id": target.db_id,
        },
    )

    # 兑换操作完成后同步牌组变更
    if target.kind == "exchange" and target.db_id:
        hint = target.entity_type_hint or ""
        if "card" in hint:
            ctx.mutate_deck_acquire(
                target.db_id,
                kind="produce_card",
                name=target.title,
                source="consult_exchange",
            )
        elif "drink" in hint:
            ctx.mutate_deck_acquire(
                target.db_id,
                kind="produce_drink",
                name=target.title,
                source="consult_exchange",
            )
        elif "item" in hint:
            ctx.mutate_deck_acquire(
                target.db_id,
                kind="produce_item",
                name=target.title,
                source="consult_exchange",
            )

    return target


class ConsultHandler(GameplayHandler):
    """相談（咨询交换页）处理。"""

    phase_tag = GameplayPhase.CONSULT
    priority = 50

    def can_handle(self, app, ctx, phase, position):
        """判断当前画面是否应由该处理器接管。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            bool: 条件判断结果，True 表示满足。
        """
        return phase == GameplayPhase.CONSULT

    def handle(self, app, ctx, phase, position):
        """执行处理器主逻辑并返回处理结果。

        Args:
            app: 应用处理器实例，提供截图、检测结果与点击/滑动能力。
            ctx: 培育上下文对象，用于读写跨步骤的业务状态。
            phase: 当前识别到的 gameplay 阶段标识。
            position: 当前界面在该阶段下的细分位置标识。

        Returns:
            返回执行结果对象，具体类型见函数注解。
        """
        target = execute_consult_step(app, ctx, position=position)
        if target is None:
            return HandlerResult.no_action("consult: no actionable candidates")

        logger.debug(
            "consult: position={}, kind={}, title={!r}, action_id={}",
            position,
            target.kind,
            target.title,
            target.action_id,
        )
        return HandlerResult.ok(f"consult {target.kind}", sleep_after=0.8)
