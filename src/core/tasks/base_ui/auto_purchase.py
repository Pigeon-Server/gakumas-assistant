import os.path
import re
from collections import Counter
from copy import copy
from dataclasses import dataclass
from time import sleep
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

from src.constants.device.adb import ADBOperation
from src.constants.game.text.general_text import GeneralText
from src.constants.path.data_path import DataPath
from src.constants.path.debug_path import DebugPath
from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.game.text.shop_text import ShopText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.device.Android.app import Android_App
from src.core.services.clip.item import Item
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.Modal import Modal
from src.entity.Game.Components.TabBar import TabBar
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.models import AutoPurchaseExchangeRecord, CLIPMemory
from src.utils.game_database_tools import GakumasDatabase_ItemDataUtils
from src.utils.game_tools import modal_body_extract_item_info
from src.core.inference.ocr_engine import OCRService
from src.utils.opencv_tools import check_color, check_frame_change
from src.utils.string_tools import MatchConfig, string_match
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.main import AppProcessor

ocr_service = OCRService()
item_db = GakumasDatabase_ItemDataUtils()
ITEM_DB_MATCH_CONFIG = MatchConfig(fuzz_threshold=85, use_contains=False, normalize=True)
ITEM_DB_NAME_DESC_MATCH_CONFIG = MatchConfig(fuzz_threshold=85, use_contains=False, normalize=True)
_NUMBER_PATTERN = re.compile(r"\d[\d,]*")
_CURRENCY_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+|\d{4,}")
_STAT_LABEL_MATCH_CONFIG = MatchConfig(fuzz_threshold=72, use_contains=True, normalize=True)


@dataclass(slots=True)
class ExchangeConfirmationStats:
    owned_before: int | None = None
    owned_after: int | None = None
    exchange_limit_before: int | None = None
    exchange_limit_after: int | None = None
    modal_money_before: int | None = None
    modal_money_after: int | None = None
    purchase_quantity: int | None = None


def _iter_ocr_candidate_images(image: Optional[np.ndarray]):
    """生成适合数字/短文本识别的若干图像变体，增强抗 JPG 噪点能力。"""
    if image is None or getattr(image, "size", 0) == 0:
        return

    yield "origin", image

    enlarged = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    yield "enlarged", enlarged

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 5, 30, 30)
    yield "gray", cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _extract_numbers_from_text(text: str, require_grouped: bool = False) -> list[int]:
    """从 OCR 文本中提取整数列表。"""
    if not text:
        return []
    pattern = _CURRENCY_PATTERN if require_grouped else _NUMBER_PATTERN
    values = []
    for raw in pattern.findall(text):
        digits = raw.replace(",", "")
        if not digits.isdigit():
            continue
        values.append(int(digits))
    return values


def _collect_ocr_lines_with_variants(image: Optional[np.ndarray], width_gap: int = 30):
    """对图像的多个变体做 OCR，返回按行合并后的结果。"""
    candidates = []
    for variant_name, candidate_image in _iter_ocr_candidate_images(image):
        ocr_results = ocr_service.ocr(candidate_image)
        if not ocr_results:
            continue
        merged = ocr_results.auto_merge_lines(width_gap=width_gap)
        candidates.append((variant_name, ocr_results, merged))
    return candidates


def _find_line_number_pair(ocr_variants, labels: list[str]):
    """按标签从 OCR 行结果中提取一对整数。"""
    for variant_name, raw_results, merged_lines in ocr_variants:
        label_results = raw_results.search(labels, _STAT_LABEL_MATCH_CONFIG)
        if label_results:
            label_box = label_results.first()
            row_results = [
                result for result in raw_results
                if abs(result.cy - label_box.cy) <= max(18, label_box.h)
            ]
            row_results.sort(key=lambda result: result.x)
            ordered_numbers: list[int] = []
            for result in row_results:
                if result == label_box:
                    continue
                ordered_numbers.extend(_extract_numbers_from_text(result.text))
            if len(ordered_numbers) >= 2:
                return ordered_numbers[0], ordered_numbers[1], label_box, variant_name
        for line in merged_lines:
            if not string_match(line.text, labels, _STAT_LABEL_MATCH_CONFIG):
                continue
            numbers = _extract_numbers_from_text(line.text)
            if len(numbers) >= 2:
                return numbers[0], numbers[1], line, variant_name
    return None, None, None, None


def _find_bottom_currency_pair(ocr_variants):
    """提取交换确认框底部的金钱变化数值。"""
    best = None
    for variant_name, raw_results, merged_lines in ocr_variants:
        money_results = [
            result for result in raw_results
            if _extract_numbers_from_text(result.text, require_grouped=True)
        ]
        if money_results:
            bottom_cy = max(result.cy for result in money_results)
            row_results = [
                result for result in money_results
                if abs(result.cy - bottom_cy) <= 18
            ]
            row_results.sort(key=lambda result: result.x)
            ordered_numbers: list[int] = []
            for result in row_results:
                ordered_numbers.extend(_extract_numbers_from_text(result.text, require_grouped=True))
            if len(ordered_numbers) >= 2:
                return ordered_numbers[0], ordered_numbers[1], row_results[0], variant_name
        for line in merged_lines:
            numbers = _extract_numbers_from_text(line.text, require_grouped=True)
            if len(numbers) < 2:
                continue
            candidate = (numbers[0], numbers[1], line, variant_name)
            if best is None or line.y > best[2].y:
                best = candidate
    if best is None:
        return None, None, None, None
    return best


def _draw_modal_debug_box(app, modal: Modal, line, label: str, color):
    """在原图上可视化交换确认框中的识别行。"""
    if line is None or getattr(modal, "body_box", None) is None or not hasattr(app, "debug_tools"):
        return
    body_box = modal.body_box
    app.debug_tools.add_box(
        int(body_box.x + line.x),
        int(body_box.y + line.y),
        int(body_box.x + line.x + line.w),
        int(body_box.y + line.y + line.h),
        label=label,
        color=color,
    )


def _extract_modal_purchase_quantity(modal_body: Optional[np.ndarray]) -> int | None:
    """从交换确认框的物品卡面角标中读取本次购买数量。"""
    item_image, _item_info = modal_body_extract_item_info(modal_body)
    if item_image is None or getattr(item_image, "size", 0) == 0:
        return None

    best = None
    for _variant_name, raw_results, merged_lines in _collect_ocr_lines_with_variants(item_image, width_gap=20):
        line_text = " ".join(line.text for line in merged_lines)
        candidates = _extract_numbers_from_text(line_text)
        if not candidates:
            continue
        current = max(candidates)
        if 0 < current <= 99 and (best is None or current > best):
            best = current
    return best


def _read_daily_exchange_money_from_box(money_box: Optional[Yolo_Box]) -> int | None:
    """从顶部金钱区域单帧读取金钱值。"""
    if money_box is None or money_box.frame is None or money_box.frame.size == 0:
        return None

    best_value = None
    best_score = None
    for variant_name, _raw_results, merged_lines in _collect_ocr_lines_with_variants(money_box.frame, width_gap=40):
        joined_text = " ".join(line.text for line in merged_lines)
        numbers = _extract_numbers_from_text(joined_text, require_grouped=True)
        if not numbers:
            continue
        value = max(numbers)
        score = (
            1 if "," in joined_text else 0,
            max(len(raw) for raw in _CURRENCY_PATTERN.findall(joined_text)) if _CURRENCY_PATTERN.findall(joined_text) else 0,
            1 if variant_name == "origin" else 0,
        )
        if best_score is None or score > best_score:
            best_value = value
            best_score = score
    return best_value


def _read_daily_exchange_money_multiframe(
        app: "AppProcessor",
        sample_count: int = 3,
        max_attempts: int = 8,
        interval: float = 0.15,
) -> int | None:
    """
    多帧读取每日交换页顶部金钱，使用多数投票降低 OCR 误识别概率。
    """
    seen_results = set()
    samples: list[int] = []
    money_box = None
    attempts = 0
    while len(samples) < sample_count and attempts < max_attempts:
        current = getattr(app, "latest_results", None)
        if current is None or id(current) in seen_results:
            sleep(interval)
            attempts += 1
            continue
        seen_results.add(id(current))
        money_box = current.filter_by_label(BaseUILabels.CURRENCY_MONEY).first()
        value = _read_daily_exchange_money_from_box(money_box)
        if value is not None:
            samples.append(value)
        sleep(interval)
        attempts += 1

    if not samples:
        return None

    value, count = Counter(samples).most_common(1)[0]
    if hasattr(app, "debug_tools") and money_box is not None:
        app.debug_tools.add_box(
            int(money_box.x),
            int(money_box.y),
            int(money_box.w),
            int(money_box.h),
            label=f"顶部マニー:{value}\n样本:{samples}",
            color=(255, 165, 0),
        )
    if count < max(2, sample_count - 1) and len(set(samples)) > 1:
        logger.warning(f"Daily exchange money OCR unstable: {samples}, selected={value}")
    return value


def _extract_exchange_confirmation_stats(app: "AppProcessor", modal: Modal) -> ExchangeConfirmationStats:
    """从交换确认弹窗中提取持有数、可交换次数与金钱变化。"""
    stats = ExchangeConfirmationStats()
    if modal.modal_body is None or getattr(modal.modal_body, "size", 0) == 0:
        return stats

    ocr_variants = _collect_ocr_lines_with_variants(modal.modal_body, width_gap=45)
    item_quantity = _extract_modal_purchase_quantity(modal.modal_body)
    owned_before, owned_after, owned_line, owned_variant = _find_line_number_pair(
        ocr_variants,
        [ShopText.DAILY_EXCHANGE.OWNED_COUNT],
    )
    if owned_before is not None and owned_after is not None:
        stats.owned_before = owned_before
        stats.owned_after = owned_after
        _draw_modal_debug_box(
            app,
            modal,
            owned_line,
            f"所持数:{owned_before}->{owned_after}\n{owned_variant}",
            (255, 0, 0),
        )

    limit_before, limit_after, limit_line, limit_variant = _find_line_number_pair(
        ocr_variants,
        [ShopText.DAILY_EXCHANGE.EXCHANGE_LIMIT],
    )
    if limit_before is not None and limit_after is not None:
        stats.exchange_limit_before = limit_before
        stats.exchange_limit_after = limit_after
        _draw_modal_debug_box(
            app,
            modal,
            limit_line,
            f"交換回数:{limit_before}->{limit_after}\n{limit_variant}",
            (0, 255, 255),
        )

    money_before, money_after, money_line, money_variant = _find_bottom_currency_pair(ocr_variants)
    if money_before is not None and money_after is not None:
        stats.modal_money_before = money_before
        stats.modal_money_after = money_after
        _draw_modal_debug_box(
            app,
            modal,
            money_line,
            f"模态マニー:{money_before}->{money_after}\n{money_variant}",
            (0, 255, 0),
        )

    if item_quantity is not None:
        stats.purchase_quantity = item_quantity

    if (
        stats.purchase_quantity is not None
        and stats.owned_after is not None
        and (
            stats.owned_before is None
            or stats.owned_before > stats.owned_after
            or (stats.owned_after - stats.owned_before) != stats.purchase_quantity
        )
    ):
        corrected_before = stats.owned_after - stats.purchase_quantity
        if corrected_before >= 0:
            logger.warning(
                "Owned-count OCR inconsistent in exchange modal, correcting "
                f"{stats.owned_before}->{stats.owned_after} with quantity={stats.purchase_quantity}"
            )
            stats.owned_before = corrected_before

    if (
        stats.purchase_quantity is None
        and stats.owned_before is not None
        and stats.owned_after is not None
        and stats.owned_after >= stats.owned_before
    ):
        stats.purchase_quantity = stats.owned_after - stats.owned_before

    return stats


def _save_exchange_record(item_data: Item, page_money_before: int | None, stats: ExchangeConfirmationStats):
    """写入自动每日交换记录表。"""
    money_before = page_money_before if page_money_before is not None else stats.modal_money_before
    money_after = stats.modal_money_after
    money_delta = None
    if money_before is not None and money_after is not None:
        money_delta = money_before - money_after
    AutoPurchaseExchangeRecord.create(
        item_id=item_data.id,
        item_name=item_data.name,
        page_money_before=page_money_before,
        modal_money_before=stats.modal_money_before,
        modal_money_after=stats.modal_money_after,
        money_delta=money_delta,
        owned_before=stats.owned_before,
        owned_after=stats.owned_after,
        purchase_quantity=stats.purchase_quantity,
        exchange_limit_before=stats.exchange_limit_before,
        exchange_limit_after=stats.exchange_limit_after,
    )


def _can_confirm_exchange_modal(modal: Modal) -> bool:
    """判断当前交换确认框是否允许执行购买。"""
    modal_body_text = modal.modal_body_text or ""
    return bool(
        modal.confirm_button is not None
        and not modal.confirm_button.is_disabled()
        and not ("不足" in modal_body_text and "AP" in modal_body_text)
    )


def _confirm_and_record_exchange(
        app: "AppProcessor",
        item_data: Item,
        modal: Modal,
        page_money_before: int | None = None,
) -> bool:
    """统一处理交换确认：解析数值、确认点击、记录数据库。"""
    stats = _extract_exchange_confirmation_stats(app, modal)
    confirmed = _can_confirm_exchange_modal(modal)
    handled = _confirm_exchange_modal(app, modal, item_data.name)
    if handled and confirmed:
        _save_exchange_record(item_data, page_money_before, stats)
    return handled

def _calculate_median_size(item_groups: List) -> Tuple[int, int]:
    """计算当前页面物品框的中位数宽高，用于过滤误识别"""
    if not item_groups:
        return 0, 0
    x_list = [g.w - g.x for g in item_groups]
    y_list = [g.h - g.y for g in item_groups]
    return int(np.median(x_list)), int(np.median(y_list))

def _is_valid_item(app, item_box, item_inner, median_x, median_y, index) -> bool:
    """检查物品是否有效（尺寸检查 + 颜色检查）"""
    # 1. 尺寸检查
    if median_x > 0 and median_y > 0:
        if item_box.w - item_box.x < median_x * 0.8 or item_box.h - item_box.y < median_y * 0.8:
            logger.warning(f"Skip over small item {index}")
            app.debug_tools.add_box(
                item_inner.x,
                item_inner.y,
                item_inner.w,
                item_inner.h,
                label=f"{index} 已跳过：框过小"
            )
            return False

    # 2. 颜色检查 (是否已兑换/不可用)
    if check_color_result := check_color(item_inner.frame, (0, 0, 85), (179, 90, 145), threshold=40):
        logger.debug(f"Skip item {index} Reason: 交換済み/无库存")
        app.debug_tools.add_box(
            item_inner.x,
            item_inner.y,
            item_inner.w,
            item_inner.h,
            label=f"{index} 已跳过：无库存\nval: {check_color_result.value}",
            color=(255, 255, 0)
        )
        return False

    return True

def _handle_known_item(app, item_inner, clip_result, commodity_target, index):
    """处理已经在记忆中的物品"""
    logger.debug(f"Item {index} found in memory: {clip_result.name}")
    app.debug_tools.add_box(
        item_inner.x,
        item_inner.y,
        item_inner.w,
        item_inner.h,
        label=f"{index} 记忆中：{clip_result.name}",
        color=(0, 255, 0)
    )

    if string_match(clip_result.name, commodity_target, MatchConfig(fuzz_threshold=80)):
        page_money_before = _read_daily_exchange_money_multiframe(app)
        _purchase_item(app, clip_result, item_inner, page_money_before=page_money_before)
        return True

    logger.debug(f"{clip_result.name} not in target list.")
    return False


def _ocr_modal_item_candidates(image: Optional[np.ndarray], limit: int = 5) -> list[str]:
    """
    对给定图像做 OCR，并按从上到下的顺序提取若干个候选文本。

    用于未知商品兜底识别：优先取更靠上的短文本，避免直接落到说明正文。
    :param image: OCR 输入图像
    :param limit: 最多返回多少个候选文本
    :return: 候选文本列表
    """
    if image is None or getattr(image, "size", 0) == 0:
        return []

    ocr_results = ocr_service.ocr(image)
    if not ocr_results:
        return []

    merged_results = ocr_results.auto_merge_lines(width_gap=30)
    ordered_results = sorted(merged_results, key=lambda res: (res.y, len(res.text)))
    candidates: list[str] = []
    for result in ordered_results:
        text = result.text.strip()
        if len(text) <= 2:
            continue
        if text in candidates:
            continue
        candidates.append(text)
        if len(candidates) >= limit:
            break
    return candidates


def _build_item_modal_lookup_queries(candidates: list[str]) -> list[str]:
    """仅使用第一行或前两行拼接后的文本做数据库查询。"""
    if not candidates:
        return []

    queries: list[str] = []
    first_line = str(candidates[0] or "").strip()
    if first_line:
        queries.append(first_line)

    if len(candidates) >= 2:
        first_two_lines = f"{first_line}{str(candidates[1] or '').strip()}".strip()
        if first_two_lines and first_two_lines not in queries:
            queries.append(first_two_lines)

    return queries


def _search_item_from_lookup_queries(queries: list[str]):
    """
    使用「第一行」或「第一行+第二行」查询主数据库。

    先匹配名称，再匹配名称+说明的拼接文本；未命中则返回失败。
    """
    if not queries:
        return False, None, None

    for query in queries:
        status, db_result = item_db.search(query, ITEM_DB_MATCH_CONFIG)
        if status:
            return True, db_result, query

    combined_map = {}
    for item in item_db.get_all_item():
        name = str(getattr(item, "name", "") or "").strip()
        description = str(getattr(item, "description", "") or "").strip()
        if not name or not description:
            continue
        combined_text = f"{name}{description}"
        if combined_text not in combined_map:
            combined_map[combined_text] = item

    if not combined_map:
        return False, None, queries[0]

    combined_candidates = list(combined_map.keys())
    for query in queries:
        result = string_match(query, combined_candidates, ITEM_DB_NAME_DESC_MATCH_CONFIG)
        if result:
            return True, combined_map[result.result], query

    return False, None, queries[0]


def _resolve_item_from_modal_ocr(item_info: Optional[np.ndarray], modal_body: Optional[np.ndarray]):
    """
    从商品确认弹窗中解析物品名称候选，并尝试在数据库中匹配。

    先使用裁剪出的 item_info 区域，再退回整个 modal body 顶部文本，
    用于处理标题裁剪不完整时的未知物品识别。
    :param item_info: 物品信息裁剪区域
    :param modal_body: 整个模态框正文区域
    :return: (是否命中数据库, 数据库对象, 最佳候选文本)
    """
    # 优先使用 item_info 区域的候选
    info_candidates = _ocr_modal_item_candidates(item_info, limit=3)
    info_queries = _build_item_modal_lookup_queries(info_candidates)

    # 先在 item_info 的第一行 / 前两行组合中查找匹配
    if info_candidates:
        logger.debug(f"Unknown item OCR info candidates: {info_candidates}")
        status, db_result, matched_text = _search_item_from_lookup_queries(info_queries)
        if status:
            logger.debug(f"Unknown item DB matched from info OCR: {matched_text} -> {db_result.name}")
            return True, db_result, matched_text

    # item_info 无结果时再退回到 modal body 候选
    body_candidates = [
        text for text in _ocr_modal_item_candidates(modal_body, limit=5)
        if text not in info_candidates
    ]
    body_queries = _build_item_modal_lookup_queries(body_candidates)
    all_candidates = info_candidates + body_candidates

    if not all_candidates:
        return False, None, None

    if body_candidates:
        logger.debug(f"Unknown item OCR body candidates: {body_candidates}")
        status, db_result, matched_text = _search_item_from_lookup_queries(body_queries)
        if status:
            logger.debug(f"Unknown item DB matched from body OCR: {matched_text} -> {db_result.name}")
            return True, db_result, matched_text

    return False, None, all_candidates[0]

def _handle_unknown_item(app, item_box, item_inner, commodity_target, index):
    """处理未知物品：点击 -> OCR -> 存库 -> 决策"""
    logger.debug(f"Item {index} not in memory, analyzing...")
    page_money_before = _read_daily_exchange_money_multiframe(app)
    app.debug_tools.add_box(
        item_inner.x,
        item_inner.y,
        item_inner.w,
        item_inner.h,
        label=f"{index} 未知物品",
        color=(255, 0, 0)
    )

    # 点击打开模态
    modal = None
    for _ in range(3):
        # 始终点击卡片主体区域，避免 group 外框误触到 TabBar/页签切换区域
        app.device.click_element(item_inner)
        sleep(0.5)
        app.game_utils.wait_loading()
        app.game_utils.wait_frame_stable()
        modal = app.game_utils.wait_for_modal(ModalText.TITLE.EXCHANGE_CONFIRMATION)
        if modal: break
    if not modal:
        return False
    # 提取信息
    try:
        yolo_result_item = copy(item_inner.frame)
        modal_item_image, item_info = modal_body_extract_item_info(modal.modal_body)
        status, db_result, matched_text = _resolve_item_from_modal_ocr(item_info, modal.modal_body)

        if matched_text is None:
            logger.warning("No OCR results found.")
            app.device.click_element(modal.cancel_button)
            return False

        final_name = matched_text

        # 存入记忆库
        if status:
            app.clip_manager.item_clip.add_to_memory(modal_item_image, db_result, 0.99)
            app.clip_manager.item_clip.add_to_memory(yolo_result_item, db_result, 0.99)
            final_name = db_result.name
        else:
            # 记录未知物品用于调试
            _save_debug_unknown_item(item_info, modal_item_image, modal.modal_body, index, final_name)

        # 购买决策
        if string_match(final_name, commodity_target, MatchConfig(fuzz_threshold=80)):
            logger.info(f"Purchase new item {final_name} (id={index})")
            if status and db_result is not None:
                _confirm_and_record_exchange(app, db_result, modal, page_money_before=page_money_before)
            else:
                _confirm_exchange_modal(app, modal, final_name)
        else:
            logger.debug(f"{final_name} not in target, cancel.")
            if modal.cancel_button is not None:
                app.device.click_element(modal.cancel_button)

        app.game_utils.wait_label_exist(BaseUILabels.MODAL_HEADER)
        return True

    except Exception as e:
        logger.error(f"Error processing unknown item: {e}")
        if modal: app.device.click_element(modal.cancel_button)
        return False

def _save_debug_unknown_item(info_img, item_img, body_img, index, name):
    """保存识别失败的物品图片"""
    logger.warning(f"Item '{name}' not found in DB.")
    os.makedirs(DebugPath.UnknownItem, exist_ok=True)
    cv2.imwrite(os.path.join(DebugPath.UnknownItem, f"item_info_{index}.png"), info_img)
    cv2.imwrite(os.path.join(DebugPath.UnknownItem, f"modal_item_{index}.png"), item_img)

def _purchase_item(app: "AppProcessor", item_data, el: Yolo_Box, page_money_before: int | None = None):
    """购买物品"""
    logger.info(f"Purchase {item_data.name}...")
    for _ in range(3):
        app.device.click_element(el)
        app.game_utils.wait_frame_stable()
        modal = app.game_utils.wait_for_modal(ModalText.TITLE.EXCHANGE_CONFIRMATION)
        if modal:
            _confirm_and_record_exchange(app, item_data, modal, page_money_before=page_money_before)
            break
    app.game_utils.wait_label_exist(BaseUILabels.MODAL_HEADER)

def _scroll_page(app, scroll_x, scroll_y, item_commodity):
    """
    处理页面滚动。

    返回 True 表示滚动导致页面发生了变化（还有更多内容）；
    返回 False 表示页面未变化（已到达列表底部）。
    """
    pre_scroll = app.latest_frame.copy()
    if isinstance(app.device, Android_App):
        app.device.swipe(
            scroll_x,
            item_commodity.get_y_max_element().y,
            scroll_x,
            item_commodity.get_y_min_element().y
        )
    else:
        app.device.scrollY(scroll_x, scroll_y, -5)
    app.game_utils.wait_frame_stable(min_stable_duration=0.3)
    return not check_frame_change(pre_scroll, app.latest_frame)


def _dismiss_daily_exchange_blocking_modal_if_present(app: "AppProcessor") -> bool:
    """
    关闭挡住每日交换商品列表的残留弹窗。

    当前已知最常见的是数量调整弹窗；这类弹窗会保留在每日交换页上方，
    让商品列表检测始终为 0。根据当前业务约定，优先点击“最大值にする”继续流程。
    """
    results = getattr(app, "latest_results", None)
    if results is None or not results.exists_label(BaseUILabels.MODAL_HEADER):
        return False

    quantity_selector_labels = [
        BaseUILabels.QUANTITY_SELECTOR,
        BaseUILabels.QUANTITY_SELECTOR_ADDED,
        BaseUILabels.QUANTITY_SELECTOR_REDUCED,
    ]
    has_quantity_selector = bool(results.filter_by_labels(quantity_selector_labels))
    modal = app.game_utils.try_get_modal(no_body=True)
    if modal is None:
        return False

    close_button = modal.cancel_button or modal.confirm_button
    if has_quantity_selector:
        close_button = (
            ButtonList(results).get_button_by_text(
                ButtonText.SHOP.SET_MAX_QUANTITY,
                match_config=MatchConfig(fuzz_threshold=90),
            )
            or close_button
        )

    if close_button is None:
        logger.warning(
            f"Daily exchange blocking modal '{modal.modal_title}' has no actionable button"
        )
        return False

    logger.warning(
        f"Handle blocking modal before detecting daily exchange list: {modal.modal_title}"
    )
    if hasattr(close_button, "x") and hasattr(app, "debug_tools"):
        app.debug_tools.add_box(
            int(close_button.x),
            int(close_button.y),
            int(close_button.w),
            int(close_button.h),
            label="数量弹窗:最大值",
            color=(0, 165, 255),
        )
        app.debug_tools.show()
        app.debug_tools.clear_all()
    if not app.game_utils.click_modal_button_and_wait_transition(
            close_button,
            previous_modal_title=modal.modal_title,
            timeout=5,
            interval=0.2,
    ):
        logger.warning(
            f"Blocking modal '{modal.modal_title}' did not transition after dismiss attempt"
        )
        return False

    app.game_utils.wait_frame_stable(min_stable_duration=0.2)
    return True


def _confirm_exchange_modal(app: "AppProcessor", modal: Modal, item_name: str) -> bool:
    """
    处理交换确认弹窗。

    若资源不足，则关闭弹窗并返回；若确认点击后弹窗未切换，
    则尝试重新解析当前弹窗并兜底关闭，避免残留模态阻塞后续流程。
    """
    previous_title = modal.modal_title
    modal_body_text = modal.modal_body_text or ""
    if modal.confirm_button is None and modal.cancel_button is None:
        logger.warning(f"Exchange modal for '{item_name}' has no actionable button")
        return False

    should_cancel = (
        modal.confirm_button is None
        or modal.confirm_button.is_disabled()
        or ("不足" in modal_body_text and "AP" in modal_body_text)
    )
    action_button = modal.cancel_button if should_cancel else modal.confirm_button
    if action_button is None:
        action_button = modal.cancel_button or modal.confirm_button
    if action_button is None:
        logger.warning(f"Exchange modal for '{item_name}' has no fallback button")
        return False

    if should_cancel:
        logger.warning(f"Skip purchasing '{item_name}' because resources are insufficient")
    elif not app.game_utils.click_modal_button_and_wait_transition(
            action_button,
            previous_modal_title=previous_title,
            timeout=5,
            interval=0.2,
    ):
        logger.warning(
            f"Exchange modal for '{item_name}' did not transition after confirm, trying to close it"
        )
        fallback_modal = app.game_utils.try_get_modal(no_body=False)
        if fallback_modal is not None:
            fallback_button = fallback_modal.cancel_button or fallback_modal.confirm_button
            if fallback_button is not None:
                app.game_utils.click_modal_button_and_wait_transition(
                    fallback_button,
                    previous_modal_title=fallback_modal.modal_title,
                    timeout=5,
                    interval=0.2,
                )
        app.game_utils.wait_frame_stable()
        return True
    else:
        app.game_utils.wait_frame_stable()
        return True

    if not app.game_utils.click_modal_button_and_wait_transition(
            action_button,
            previous_modal_title=previous_title,
            timeout=5,
            interval=0.2,
    ):
        logger.warning(f"Exchange modal for '{item_name}' did not close after cancel")
        return False
    app.game_utils.wait_frame_stable()
    return True


def _wait_exchange_item_groups(
        app: "AppProcessor",
        timeout: float = 5.0,
        interval: float = 0.5,
) -> Tuple[Yolo_Results, List[Yolo_Results]]:
    """
    等待每日交换所商品列表出现并稳定，避免在列表仍在滚动时就开始推理。

    先等待商品列表出现，再等待画面稳定（列表滚动结束），
    最后用稳定后的帧重新检测商品位置，确保坐标准确。
    """
    waited = 0.0
    last_item_commodity = None
    last_item_groups = []
    saw_blocking_modal = False
    while waited <= timeout:
        if _dismiss_daily_exchange_blocking_modal_if_present(app):
            saw_blocking_modal = True
            continue
        item_commodity = app.latest_results.filter_by_labels([BaseUILabels.ITEM, BaseUILabels.CARD_COMMODITY])
        item_groups = item_commodity.find_containing_groups(BaseUILabels.CARD_COMMODITY, [BaseUILabels.ITEM])
        if item_commodity and item_groups:
            # 列表已出现，等待画面稳定（滚动动画结束）
            # min_stable_duration=0.3 确保稳定持续至少 300ms，防止滚动动画中短暂停顿被误判
            app.game_utils.wait_frame_stable(min_stable_duration=0.3)
            # 用稳定后的帧重新检测，获取准确坐标
            item_commodity = app.latest_results.filter_by_labels([BaseUILabels.ITEM, BaseUILabels.CARD_COMMODITY])
            item_groups = item_commodity.find_containing_groups(BaseUILabels.CARD_COMMODITY, [BaseUILabels.ITEM])
            if item_commodity and item_groups:
                return item_commodity, item_groups
        last_item_commodity = item_commodity
        last_item_groups = item_groups
        sleep(interval)
        waited += interval

    current_location = app.game_utils.update_current_location()
    raise RuntimeError(
        "Daily exchange commodity list not detected. "
        f"current_location={current_location}, "
        f"commodity_boxes={len(last_item_commodity) if last_item_commodity else 0}, "
        f"commodity_groups={len(last_item_groups)}, "
        f"saw_blocking_modal={saw_blocking_modal}"
    )

def _exchange_items(app: "AppProcessor", commodity_target: List[str]):
    """
    主循环：交换物品
    """
    logger.info(f"Shopping list: {commodity_target}")
    median_x, median_y = None, None

    # 检查内存是否为空，避免在空内存时进行无效检索
    is_memory_empty = len(CLIPMemory.select()) == 0

    while True:
        # 识别当前页面元素
        item_commodity, item_groups = _wait_exchange_item_groups(app)
        scroll_x, scroll_y = item_commodity.get_COL()

        # 初始化尺寸阈值（仅一次）
        if median_x is None:
            median_x, median_y = _calculate_median_size(item_groups)
            logger.debug(f"Median Size: w={median_x}, h={median_y}")

        # 遍历当前页面的所有物品
        for index, item_box in enumerate(item_groups):
            item_inner = item_box.filter_by_label(BaseUILabels.ITEM).first()

            # 有效性检查
            if not _is_valid_item(app, item_box, item_inner, median_x, median_y, index):
                continue

            # 尝试从记忆中检索
            clip_result = None
            if not is_memory_empty:
                clip_result = app.clip_manager.item_clip.retrieve(item_inner.frame, 0.99)

            if clip_result:
                # 记忆中存在
                _handle_known_item(app, item_inner, clip_result, commodity_target, index)
            else:
                # 记忆中不存在 (OCR + 更新记忆)
                _handle_unknown_item(app, item_box, item_inner, commodity_target, index)

            # 刷新调试显示
            app.debug_tools.show()
            sleep(0.5)

        # 翻页判断
        app.debug_tools.clear_all()
        if not _scroll_page(app, scroll_x, scroll_y, item_commodity):
            logger.info("Page content stable, end of list.")
            break

def _find_visible_weekly_gift_buttons(app: "AppProcessor") -> List:
    # 只返回当前页真正可点击的“免费”礼包按钮。
    buttons = ButtonList(app.latest_results)
    return [
        button for button in buttons
        if button.text
        and string_match(button.text, ButtonText.FREE, MatchConfig(use_fuzz=False))
        and not button.is_disabled()
    ]


def _close_weekly_gift_result_modal(app: "AppProcessor") -> bool:
    """
    关闭周礼包领取后的结果模态框。

    返回 True 表示结果框已识别并完成关闭；
    返回 False 表示没有找到可收尾的结果框。
    """
    result_modal = app.game_utils.wait_for_modal(
        ModalText.TITLE.RECEIPT_COMPLETED,
        timeout=5,
        interval=0.5,
        no_body=True,
        match_config=MatchConfig(fuzz_threshold=90),
    )
    if result_modal is None:
        result_modal = app.game_utils.wait_for_modal(None, timeout=2, interval=0.5, no_body=True)
    if result_modal is None:
        logger.warning("Weekly gift result modal not found")
        return False

    close_button = result_modal.cancel_button or result_modal.confirm_button
    if close_button is None:
        logger.warning("Weekly gift result modal close button not found")
        return False

    app.device.click_element(close_button)
    app.game_utils.wait_label_exist(BaseUILabels.MODAL_HEADER, timeout=5, interval=0.5)
    app.game_utils.wait_frame_stable()
    return True


def _dismiss_weekly_gift_modal_if_present(
        app: "AppProcessor",
        modal_titles: Optional[List[str]] = None,
) -> bool:
    """
    关闭当前仍残留在画面上的周礼包相关模态框。

    当 modal_titles 不为空时，仅在标题命中这些候选值时才执行关闭。
    返回 True 表示检测到模态且完成了关闭/切换等待。
    """
    modal = app.game_utils.try_get_modal(no_body=True)
    if modal is None:
        return False

    if modal_titles and not string_match(modal.modal_title, modal_titles, MatchConfig(fuzz_threshold=90)):
        return False

    close_button = modal.cancel_button or modal.confirm_button
    if close_button is None:
        logger.warning("Weekly gift modal close button not found")
        return False

    logger.debug(f"Dismissing weekly gift modal: {modal.modal_title}")
    closed = app.game_utils.click_modal_button_and_wait_transition(
        close_button,
        previous_modal_title=modal.modal_title,
        timeout=5,
        interval=0.2,
    )
    if closed:
        app.game_utils.wait_frame_stable()
    return closed


def _claim_weekly_gift_button(app: "AppProcessor", button) -> bool:
    """
    执行单个周礼包领取流程。

    流程为：点击礼包 -> 等确认框 -> 点击确认并等待模态切换 -> 关闭结果框。
    返回 True 表示一次礼包领取流程完整结束；
    返回 False 表示中途遇到异常，需要跳过当前礼包。
    """
    app.device.click_element(button)

    confirm_modal = app.game_utils.wait_for_modal(None, timeout=5, interval=0.5, no_body=True)
    if confirm_modal is None:
        logger.warning(f"Weekly gift confirm modal not found for button '{button.text}'")
        return False

    confirm_button = confirm_modal.confirm_button or confirm_modal.cancel_button
    if confirm_button is None:
        logger.warning("Weekly gift confirm modal action button not found")
        _dismiss_weekly_gift_modal_if_present(app)
        return False
    if confirm_button.is_disabled():
        logger.warning(f"Weekly gift button '{button.text}' is disabled after opening modal")
        if modal_close_button := (confirm_modal.cancel_button or confirm_modal.confirm_button):
            app.game_utils.click_modal_button_and_wait_transition(
                modal_close_button,
                previous_modal_title=confirm_modal.modal_title,
                timeout=5,
                interval=0.2,
            )
            app.game_utils.wait_frame_stable()
        return False

    if not app.game_utils.click_modal_button_and_wait_transition(
            confirm_button,
            previous_modal_title=confirm_modal.modal_title,
            timeout=5,
            interval=0.2,
    ):
        logger.warning(f"Weekly gift confirm modal '{confirm_modal.modal_title}' did not transition after confirm.")
        _dismiss_weekly_gift_modal_if_present(app, modal_titles=[confirm_modal.modal_title])
        return False

    if _close_weekly_gift_result_modal(app):
        return True

    _dismiss_weekly_gift_modal_if_present(
        app,
        modal_titles=[ModalText.TITLE.PURCHASE_CONFIRMATION, ModalText.TITLE.CONFIRM],
    )
    return False


def _collect_visible_weekly_gifts(app: "AppProcessor") -> int:
    collected = 0
    skipped_signatures = set()

    while True:
        # 每次弹窗关闭后都重新取按钮，避免继续使用已经失效的 OCR/坐标结果。
        buttons = [
            button for button in _find_visible_weekly_gift_buttons(app)
            if (int(button.cx), int(button.cy), button.text) not in skipped_signatures
        ]
        if not buttons:
            return collected

        button = buttons[0]
        signature = (int(button.cx), int(button.cy), button.text)
        if _claim_weekly_gift_button(app, button):
            collected += 1
            skipped_signatures.clear()
            continue

        skipped_signatures.add(signature)
        app.game_utils.wait_frame_stable()


def _scroll_weekly_gift_page(app: "AppProcessor") -> bool:
    """
    每次向下滚一屏左右，继续处理下一批可见礼包。

    返回 True 表示滚动导致页面发生了变化（还有更多内容）；
    返回 False 表示页面未变化（已到达列表底部）。
    """
    pre_scroll = app.latest_frame.copy()
    height, width = pre_scroll.shape[:2]
    if isinstance(app.device, Android_App):
        center_x = width // 2
        app.device.swipe(
            center_x,
            int(height * 0.82),
            center_x,
            int(height * 0.38),
            0.6,
        )
    else:
        app.device.scrollY(width // 2, height // 2, -20)
    app.game_utils.wait_frame_stable(min_stable_duration=0.3)
    return not check_frame_change(pre_scroll, app.latest_frame)


def action__receive_weekly_gift(app: "AppProcessor"):
    """领取每周礼包"""
    app.game_utils.click_button(ButtonText.SHOP.PACK, match_config=MatchConfig(use_fuzz=False))
    app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.PACK)
    app.game_utils.wait_frame_stable()

    while True:
        # 先清空当前可视区域里的免费礼包，再决定是否继续翻页。
        collected = _collect_visible_weekly_gifts(app)
        logger.debug(f"Collected {collected} weekly gift packs on current page")

        if not _scroll_weekly_gift_page(app):
            logger.info("Page content stable, end of list.")
            break

    while _dismiss_weekly_gift_modal_if_present(app):
        pass
    app.game_utils.back_next_page()
    app.game_utils.wait_loading()
    app.game_utils.update_current_location(GamePageTypes.HOME_TAB.SHOP)


def _reset_daily_exchange_to_top(app: "AppProcessor"):
    """
    刷新列表后回到商店主页再重新进入每日交换所，确保列表从顶部开始。
    Retry back navigation up to 3 times, because the first back may only dismiss
    a result dialog (e.g. after a free list refresh) instead of leaving the page.
    """
    max_back_attempts = 3
    for attempt in range(max_back_attempts):
        app.game_utils.back_next_page()
        app.game_utils.wait_loading()
        app.game_utils.wait_frame_stable()
        try:
            app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.SHOP)
            break
        except TimeoutError:
            buttons = ButtonList(app.latest_results)
            if buttons.get_button_by_text(ButtonText.SHOP.PACK, match_config=MatchConfig(use_fuzz=False)):
                logger.debug("Shop root detected by button layout after back, overriding current location.")
                app.game_utils.update_current_location(GamePageTypes.HOME_TAB.SHOP)
                break
            current = app.game_utils.update_current_location()
            if current and str(current).startswith("SHOP__") and attempt < max_back_attempts - 1:
                logger.warning(
                    f"Still on sub-page '{current}' after back attempt {attempt + 1}/{max_back_attempts}, retrying..."
                )
                continue
            raise
    app.game_utils.wait_frame_stable()
    app.game_utils.click_button(ButtonText.SHOP.DAILY_EXCHANGE, match_config=MatchConfig(fuzz_threshold=90))
    app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.DAILY_EXCHANGE)
    app.game_utils.wait_frame_stable()
    _wait_exchange_item_groups(app)


def _handle_tabbar__manny_exchange(app: "AppProcessor", commodity_target):
    refresh_shop = app.config_service().task__auto_purchase.refresh_shop.value
    use_gem_refresh = app.config_service().task__auto_purchase.use_gem_refresh.value
    height, width = app.latest_frame.shape[:2]
    remaining_attempts: Optional[int] = None
    remaining_attempts_match = re.compile(r".*更新可能回数 {,3}あと(\d+)回.*")
    while True:
        _exchange_items(app, commodity_target)
        if not refresh_shop:
            logger.debug("No need to refresh shop list")
            break
        buttons = ButtonList(app.latest_results)
        update_shop_btn = buttons.get_button_by_text(ButtonText.SHOP.LIST_UPDATE)
        logger.debug("update_shop_btn: {}".format(update_shop_btn))
        if not update_shop_btn:
            logger.warning("Not find update shop list button")
            break
        if string_match(update_shop_btn.text, ButtonText.FREE, MatchConfig(use_fuzz=False)):
            app.device.click_element(update_shop_btn)
            app.game_utils.wait_frame_stable()
            _reset_daily_exchange_to_top(app)
            continue
        if not use_gem_refresh:
            break
        if remaining_attempts is not None and remaining_attempts == 0:
            break
        app.device.click_element(update_shop_btn)
        app.game_utils.wait_frame_stable()
        modal_body: str = ""
        update_confirm_modal: Optional[Modal] = None
        for i in range(3):
            update_confirm_modal = app.game_utils.wait_for_modal(ModalText.TITLE.UPDATE_CONFIRM)
            if not update_confirm_modal:
                logger.error("Not find update confirm modal")
                break
            if update_confirm_modal.confirm_button.is_disabled():
                logger.warning("Unable to update the list: Insufficient gems available")
                break
            modal_body = update_confirm_modal.modal_body_text
            logger.debug("modal_body: {}".format(modal_body))
            if modal_body is not None:
                break
            logger.warning("Modal body is empty and is currently attempting to reacquire.")
            sleep(0.5)
        if remaining_attempts is None:
            m1 = remaining_attempts_match.match(modal_body)
            if not m1:
                break
            remaining_attempts = int(m1.group(1))
            logger.debug(f"remaining attempts: {remaining_attempts}")
        app.device.click_element(update_confirm_modal.confirm_button)
        app.game_utils.wait_loading()
        app.game_utils.wait_frame_stable()
        remaining_attempts -= 1
        _reset_daily_exchange_to_top(app)


def _detect_daily_exchange_tabbar(
        app: "AppProcessor",
        retries: int = 3,
) -> Optional[TabBar]:
    """
    在每日交换页面重试识别 TabBar。

    识别过程可能受动画、压缩噪点或瞬时漏检影响，这里采用多次重试并在每轮稳帧后重取结果；
    当识别到候选框时会打 Debug 框，便于线上问题回放。
    """
    for attempt in range(1, retries + 1):
        app.game_utils.wait_for_label(BaseUILabels.TAB_BAR, timeout=2, interval=0.2)
        tabbar_element = app.latest_results.filter_by_label(BaseUILabels.TAB_BAR).first()
        if tabbar_element is None:
            logger.warning(f"Daily exchange tab bar not detected ({attempt}/{retries})")
            app.game_utils.wait_frame_stable(min_stable_duration=0.2)
            continue

        if hasattr(app, "debug_tools"):
            app.debug_tools.add_box(
                int(tabbar_element.x),
                int(tabbar_element.y),
                int(tabbar_element.w),
                int(tabbar_element.h),
                label=f"每日交换TabBar候选 {attempt}/{retries}",
                color=(255, 105, 180),
            )

        tabbar = TabBar(tabbar_element)
        if tabbar:
            return tabbar
        logger.warning(f"Daily exchange tab bar parsed empty ({attempt}/{retries})")
        app.game_utils.wait_frame_stable(min_stable_duration=0.2)

    return None


def action__daily_exchange(app: "AppProcessor"):
    app.game_utils.click_button(ButtonText.SHOP.DAILY_EXCHANGE, match_config=MatchConfig(fuzz_threshold=90))
    app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.DAILY_EXCHANGE)
    _wait_exchange_item_groups(app)

    commodity_target = []
    for item_id in app.config_service().task__auto_purchase.daily_buy_list.value:
        if (result := item_db.get_by_id(item_id)) is not None:
            commodity_target.append(result.name)

    tabbar = _detect_daily_exchange_tabbar(app)
    if not tabbar:
        logger.warning("Daily exchange tab bar unavailable, fallback to current tab exchange flow")
        _exchange_items(app, commodity_target)
        return True

    for index, tab_item in enumerate(tabbar):
        app.device.click_element(tab_item)
        app.game_utils.wait_frame_stable()
        # if string_match(tab_item.text, "マニー"):
        match index:
            case 0:
                _handle_tabbar__manny_exchange(app, commodity_target)
            case _:
                _exchange_items(app, commodity_target)
    return True
