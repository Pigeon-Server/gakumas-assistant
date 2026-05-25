from time import sleep
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from src.constants.game.text.support_card_text import SupportCardText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.device.Android.app import Android_App
from src.core.inference.ocr_engine import OCRService
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.SupportCard import (
    SupportCard as SupportCardComponent,
    SupportCardListParser,
)
from src.utils.debug_tools import DebugTools
from src.utils.game_database_tools import GakumasDatabase_SupportCardDataUtils
from src.utils.i18n_tools import i18n_text
from src.utils.logger import logger
from src.utils.opencv_tools import check_frame_change
from src.utils.string_tools import MatchConfig
from src.utils.ui_message_tools import UIMessage

if TYPE_CHECKING:
    from src.main import AppProcessor

_FUZZ_CONFIG = MatchConfig(use_fuzz=True, fuzz_threshold=70)

ocr_service = OCRService()
debug_tools = DebugTools()
message_tools = UIMessage()
support_card_db = GakumasDatabase_SupportCardDataUtils()

# ── Level cap data (from SupportCardLevelLimit.yaml) ──────────────────────────
# 每个品级在不同突破星数下的等级上限
# R:   0★=20, 1★=25, 2★=30, 3★=35, 4★=40
# SR:  0★=30, 1★=35, 2★=40, 3★=45, 4★=50
# SSR: 0★=40, 1★=45, 2★=50, 3★=55, 4★=60
LEVEL_CAPS: dict[str, dict[int, int]] = {
    SupportCardText.RARITY_R:   {0: 20, 1: 25, 2: 30, 3: 35, 4: 40},
    SupportCardText.RARITY_SR:  {0: 30, 1: 35, 2: 40, 3: 45, 4: 50},
    SupportCardText.RARITY_SSR: {0: 40, 1: 45, 2: 50, 3: 55, 4: 60},
}

# 品级对应的最高可能等级（4★满突破）
MAX_POSSIBLE_LEVEL: dict[str, int] = {
    SupportCardText.RARITY_R:   40,
    SupportCardText.RARITY_SR:  50,
    SupportCardText.RARITY_SSR: 60,
}


# ── Glyph detection ───────────────────────────────────────────────────────────

def _detect_chevron_count(button_frame: np.ndarray) -> int:
    """通过形状识别检测按钮帧中 chevron (>) 字形的数量。

    利用 Otsu 二值化 + 轮廓多边形近似，识别向右的 V 字形。
    Returns: 0=没有, 1=单箭头(>), 2=双箭头(>>)
    """
    if button_frame is None or button_frame.size == 0:
        return 0
    h, w = button_frame.shape[:2]
    if h < 10 or w < 10:
        return 0
    # chevron 按钮是近正方形的小按钮；宽矩形文本按钮直接排除
    if w / h > 2.0 or h / w > 2.0:
        return 0

    gray = cv2.cvtColor(button_frame, cv2.COLOR_BGR2GRAY)
    # Otsu BINARY_INV: 深色字形变白（前景）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 裁掉边缘（圆形按钮边框会形成噪声轮廓）
    margin = max(2, int(min(h, w) * 0.08))
    roi = binary[margin:h - margin, margin:w - margin]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = h * w * 0.02
    roi_h = roi.shape[0]
    chevron_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = 0.04 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)
        pts = poly.reshape(-1, 2)

        # 字形高度需占 ROI 30% 以上
        shape_h = pts[:, 1].max() - pts[:, 1].min()
        if shape_h < roi_h * 0.3:
            continue

        # chevron 特征: 3~8 顶点 + 最右端点接近垂直中心（箭头尖端朝右）
        n = len(poly)
        rightmost_pt = pts[pts[:, 0].argmax()]
        if 3 <= n <= 8 and abs(rightmost_pt[1] - roi_h / 2.0) < roi_h * 0.3:
            chevron_count += 1

    return min(chevron_count, 2)


# ── Configuration helpers ─────────────────────────────────────────────────────

def _get_level_cap(rarity: str | None, stars: int | None) -> int:
    """根据品级和突破星数获取等级上限。"""
    if rarity is None:
        return 0
    caps = LEVEL_CAPS.get(rarity)
    if caps is None:
        return 0
    star_count = min(max(stars or 0, 0), 4)
    return caps.get(star_count, caps[0])


def _get_enhancement_config(app: "AppProcessor") -> dict:
    config = app.config_service().task__auto_enhancement_support_card
    return {
        SupportCardText.RARITY_R: {
            "enabled": config.enhance_r.value,
            "max_level": config.enhance_r_max_level.value,
        },
        SupportCardText.RARITY_SR: {
            "enabled": config.enhance_sr.value,
            "max_level": config.enhance_sr_max_level.value,
        },
        SupportCardText.RARITY_SSR: {
            "enabled": config.enhance_ssr.value,
            "max_level": config.enhance_ssr_max_level.value,
        },
    }


def _card_needs_enhancement(
    card: SupportCardComponent,
    config: dict,
    whitelist_mode: bool,
    whitelist_ids: list[str],
    clip_db_id: str | None = None,
) -> bool:
    if card.level is None or card.rarity is None:
        return False
    rarity_config = config.get(card.rarity)
    if rarity_config is None or not rarity_config["enabled"]:
        return False
    # 上限解放和等级强化是两个独立状态，互不影响。
    # 即使 badge=True（卡片可做上限解放），只要等级未达目标就需要强化。
    # 主循环会按顺序处理：先上限解放，再强化。
    # 计算该卡的实际等级上限（基于星数）
    actual_cap = _get_level_cap(card.rarity, card.stars)
    # 用户设定目标 vs 实际突破上限，取较小值
    target_level = min(rarity_config["max_level"], actual_cap)
    if card.level >= target_level:
        return False
    if whitelist_mode and clip_db_id and clip_db_id not in whitelist_ids:
        return False
    return True


def _card_needs_limit_break(
    card: SupportCardComponent,
    config: dict,
    auto_limit_break: bool,
) -> bool:
    """判断卡片是否需要上限解放。

    当「上限解放可能」badge 存在时，游戏已经确认：
      1. 卡片等级已达当前星级上限
      2. 有重复卡片可用于上限解放
    因此 badge 是权威信号，优先于可能不准确的数字识别结果。

    上限解放消耗的是重复卡片（不消耗强化 Pt），是"免费"操作。
    用户开启 auto_limit_break=True 即表示希望尽可能解放所有可解放的卡，
    不再受 max_level 限制。仅检查星数上限 (4★ = 最大) 即可。
    """
    if not auto_limit_break:
        return False
    if not card.limit_break:
        return False
    if card.rarity is None:
        return False
    # 如果已知星数，检查是否已到最高星数 (4★)
    if card.stars is not None and card.stars >= 4:
        return False
    return True


# ── CLIP helpers ──────────────────────────────────────────────────────────────

def _try_clip_identify(app: "AppProcessor", card_frame: np.ndarray) -> Optional[str]:
    try:
        db_card = app.clip_manager.support_card_clip.retrieve(card_frame)
        if db_card is not None:
            return db_card.id
    except Exception as e:
        logger.debug(f"CLIP identify failed: {e}")
    return None


def _clip_learn_from_detail(app: "AppProcessor", card_image: np.ndarray):
    """Try to learn the card via OCR on the detail page header."""
    frame = app.latest_frame
    if frame is None or frame.size == 0:
        return
    height = frame.shape[0]
    header = frame[:int(height * 0.20), :]
    ocr_result = ocr_service.ocr(header)
    if not ocr_result or not ocr_result.results:
        return
    for item in ocr_result.results:
        if len(item.text) >= 3:
            status, db_result = support_card_db.search(item.text)
            if status and db_result:
                try:
                    app.clip_manager.support_card_clip.add_to_memory(card_image, db_result)
                    logger.debug(f"[CLIP] Learned: {db_result.name} ({db_result.id})")
                except Exception as e:
                    logger.warning(f"[CLIP] Learn failed: {e}")
                return


# ── Navigation helpers ────────────────────────────────────────────────────────

def _click_view_detail(app: "AppProcessor") -> bool:
    """Click 詳細を見る on the card list page to enter detail page.

    If clicking the card thumbnail went directly to the detail page
    (card was already selected), detect this and return True.
    """
    for attempt in range(3):
        buttons = ButtonList(app.latest_results)

        # Check if we're already on the detail page (card was pre-selected)
        if (buttons.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ_CONFIG) is not None
                or buttons.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ_CONFIG) is not None):
            logger.debug("Already on detail page (card was pre-selected)")
            return True

        detail_btn = buttons.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ_CONFIG)
        if detail_btn is None:
            if attempt < 2:
                sleep(0.5)
                app.game_utils.wait_frame_stable(stable_count=2)
                continue
            logger.warning("Cannot find 詳細を見る button")
            return False
        if app.game_utils.click_element_and_wait_trigger(detail_btn, retries=3, timeout=3.0):
            sleep(0.8)
            app.game_utils.wait_frame_stable(stable_count=2)
            return True
    return False


def _go_back_via_cancel(app: "AppProcessor") -> bool:
    """Go back from enhancement page by clicking キャンセル.
    Returns False if キャンセル is not visible (caller should not navigate further)."""
    for attempt in range(3):
        buttons = ButtonList(app.latest_results)
        cancel_btn = buttons.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ_CONFIG)
        if cancel_btn:
            app.game_utils.click_element_and_wait_trigger(cancel_btn, retries=2, timeout=2.0)
            sleep(0.8)
            app.game_utils.wait_frame_stable(stable_count=2)
            return True
        # 可能画面还没更新
        if attempt < 2:
            sleep(0.5)
            app.game_utils.wait_frame_stable(stable_count=2)
    return False


def _go_back_to_list(app: "AppProcessor") -> bool:
    """Navigate from detail page back to card list via Back Button."""
    for attempt in range(3):
        try:
            app.game_utils.back_next_page()
            return True
        except TimeoutError:
            if attempt < 2:
                sleep(0.5)
                app.game_utils.wait_frame_stable(stable_count=2)
    return False


def _ensure_on_card_list(app: "AppProcessor", max_attempts: int = 5) -> bool:
    """确保当前回到卡片列表页面（通过连续检测 SUPPORT_CARD label 确认）。"""
    for _ in range(max_attempts):
        if app.game_utils.wait_for_label(BaseUILabels.SUPPORT_CARD, timeout=5, continuous=2):
            return True
        # 可能在详情页或强化页，尝试返回
        buttons = ButtonList(app.latest_results)
        cancel_btn = buttons.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ_CONFIG)
        if cancel_btn:
            app.game_utils.click_element_and_wait_trigger(cancel_btn, retries=2, timeout=2.0)
            sleep(0.5)
            continue
        if app.latest_results.exists_label(BaseUILabels.BACK_BTN):
            app.game_utils.click_on_label(BaseUILabels.BACK_BTN)
            sleep(1)
            continue
        # 尝试处理弹窗
        modal = app.game_utils.try_get_modal(no_body=True)
        if modal and modal.confirm_button:
            app.game_utils.click_modal_button_and_wait_transition(modal.confirm_button)
            sleep(0.5)
            continue
        try:
            app.game_utils.back_next_page()
        except TimeoutError as exc:
            logger.debug("支援卡强化: 返回列表页超时，继续尝试恢复: {}", exc)
    return False

def _enter_enhance_page(app: "AppProcessor") -> bool:
    """On the detail page, click Lv強化 to enter the enhancement page.
    Returns False if the button is not found or is disabled (already at max)."""
    for attempt in range(3):
        buttons = ButtonList(app.latest_results)
        btn = buttons.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ_CONFIG)
        if btn is None:
            if attempt < 2:
                sleep(0.5)
                app.game_utils.wait_frame_stable(stable_count=2)
                continue
            logger.warning("Lv強化 button not found on detail page")
            return False
        if btn.is_disabled():
            logger.debug("Lv強化 disabled (card at current max level)")
            return False
        if app.game_utils.click_element_and_wait_trigger(btn, retries=3, timeout=3.0):
            sleep(0.8)
            app.game_utils.wait_frame_stable(stable_count=2)
            return True
    return False


def _click_max_level_button(app: "AppProcessor") -> bool:
    """On the enhancement page, click '>>' to set target to max level.
    Returns True if successfully clicked, False otherwise."""
    for attempt in range(3):
        buttons = ButtonList(app.latest_results)
        max_btn = buttons.get_button_by_text(SupportCardText.MAX_LEVEL_BUTTON, _FUZZ_CONFIG)
        if max_btn and not max_btn.is_disabled():
            if app.game_utils.click_element_and_wait_trigger(max_btn, retries=2, timeout=1.5):
                sleep(0.5)
                return True
        # ">>" OCR 可能识别失败，尝试用 "»" 匹配
        if max_btn is None:
            max_btn = buttons.get_button_by_text("»", _FUZZ_CONFIG)
            if max_btn and not max_btn.is_disabled():
                if app.game_utils.click_element_and_wait_trigger(max_btn, retries=2, timeout=1.5):
                    sleep(0.5)
                    return True
        # 字形识别回退: OCR 无法读取 ">>/>", 用形状检测找到包含 2 个 chevron 的按钮
        if max_btn is None:
            for b in buttons.buttons:
                if _detect_chevron_count(b.frame) == 2 and not b.is_disabled():
                    max_btn = b
                    logger.debug(f">> button found via chevron glyph detection at x={b.x}")
                    if app.game_utils.click_element_and_wait_trigger(
                        max_btn, retries=2, timeout=1.5
                    ):
                        sleep(0.5)
                        return True
                    break
        if attempt < 2:
            sleep(0.3)
            app.game_utils.wait_frame_stable(stable_count=1)
    logger.debug(">> button not found or disabled, proceeding without max level selection")
    return False


def _click_level_down_button(app: "AppProcessor") -> bool:
    """On the enhancement page, click '<' to decrease target level by 1.
    Finds the single-chevron button on the LEFT side of the screen (cx < 400).
    Returns True if successfully clicked, False if not found or disabled."""
    buttons = ButtonList(app.latest_results)
    for b in buttons.buttons:
        if b.cx >= 400:
            continue  # right-side buttons are increase (> / >>)
        if b.is_disabled():
            continue
        if _detect_chevron_count(b.frame) == 1:
            app.device.click(b.cx, b.cy, el_label="<")
            sleep(0.5)
            app.game_utils.wait_frame_stable(stable_count=2)
            return True
    # Fallback: try double-chevron (<<) to jump to minimum
    for b in buttons.buttons:
        if b.cx >= 400 or b.is_disabled():
            continue
        if _detect_chevron_count(b.frame) == 2:
            app.device.click(b.cx, b.cy, el_label="<<")
            sleep(0.5)
            app.game_utils.wait_frame_stable(stable_count=2)
            return True
    logger.debug("'<' level-down button not found or disabled")
    return False


def _check_confirm_enabled(app: "AppProcessor") -> tuple[str, Optional["ButtonList"]]:
    """Check if the Lv強化 confirm button on enhancement page is enabled.
    Returns (status, ButtonList) where status is one of:
        "enabled"   - button found and clickable
        "disabled"  - button found but greyed out (no materials)
        "not_found" - button text not detected (OCR failure or wrong page)
    """
    buttons = ButtonList(app.latest_results)
    confirm = buttons.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ_CONFIG)
    if confirm is None:
        btn_texts = [b.text for b in buttons.buttons]
        logger.debug(f"Lv強化 button not found among detected buttons: {btn_texts}")
        return "not_found", buttons
    if confirm.is_disabled():
        return "disabled", buttons
    return "enabled", buttons


def _confirm_enhancement(app: "AppProcessor") -> bool:
    """On the enhancement page, click Lv強化 (confirm) to perform enhancement.
    Returns False if the confirm button is disabled (no materials or already max)."""
    buttons = ButtonList(app.latest_results)
    confirm = buttons.get_button_by_text(SupportCardText.LV_ENHANCE, _FUZZ_CONFIG)
    if confirm is None:
        logger.warning("Lv強化 confirm button not found on enhancement page")
        return False
    if confirm.is_disabled():
        logger.info("Lv強化 confirm disabled (疑似无サポート強化Pt材料或已达当前星级上限)")
        return False
    app.game_utils.click_element_and_wait_trigger(confirm, retries=3, timeout=3.0)
    sleep(1.5)
    app.game_utils.wait_frame_stable(stable_count=3)
    return True


def _handle_post_enhancement(app: "AppProcessor"):
    """Handle modals/popups after enhancement completes."""
    for _ in range(5):
        modal = app.game_utils.try_get_modal(no_body=True)
        if modal and modal.confirm_button:
            app.game_utils.click_modal_button_and_wait_transition(modal.confirm_button)
            sleep(0.5)
            continue
        break
    app.game_utils.wait_frame_stable(stable_count=2)


def _enhance_single_card(app: "AppProcessor") -> str:
    """Full flow: detail page → enhancement page → enhance → back to detail.
    Returns:
        "success"            - enhancement was performed
        "no_materials"       - サポート強化Pt 不足（全局性的，后续所有卡都会一样）
        "at_cap"             - 已达当前星级上限或 Lv強化 按钮不可用
        "failed"             - 其他失败原因
    """
    if not _enter_enhance_page(app):
        return "at_cap"

    # 先检查默认等级（+1）时确认按钮是否可用（带重试）
    status = "not_found"
    for _attempt in range(3):
        status, _ = _check_confirm_enabled(app)
        if status != "not_found":
            break
        sleep(0.5)
        app.game_utils.wait_frame_stable(stable_count=2)
    if status == "disabled":
        # 连 +1 级都做不了 → 真的没材料
        logger.info("Lv強化 confirm disabled at default level → 资源不足")
        _go_back_via_cancel(app)
        return "no_materials"
    if status == "not_found":
        logger.warning("Lv強化 confirm button not found after retries → 页面异常")
        _go_back_via_cancel(app)
        return "failed"

    # 尝试 ">>" 跳到最大等级
    clicked_max = _click_max_level_button(app)
    if clicked_max:
        status, _ = _check_confirm_enabled(app)
        if status != "enabled":
            # ">>" 后资源不够 → 用 '<' 逐级递减，找到最高可负担等级
            logger.info("Lv強化 confirm disabled at max → 逐级递减寻找可负担等级")
            found_affordable = False
            for step in range(20):  # 安全上限：最多递减 20 级
                if not _click_level_down_button(app):
                    logger.debug(f"'<' not found after {step} steps, giving up step-down")
                    break
                status, _ = _check_confirm_enabled(app)
                if status == "enabled":
                    logger.info(f"Lv強化 re-enabled after {step + 1} level-down steps")
                    found_affordable = True
                    break
            if not found_affordable:
                # 逐级递减仍无法找到可负担等级 → 取消
                logger.info("逐级递减失败，取消强化")
                _go_back_via_cancel(app)
                return "no_materials"

    # Click Lv強化 (confirm)
    if not _confirm_enhancement(app):
        _go_back_via_cancel(app)
        return "no_materials"

    _handle_post_enhancement(app)

    # Go back to detail page (might still be on enhancement result page)
    _go_back_via_cancel(app)

    return "success"


# ── 上限解放 flow ─────────────────────────────────────────────────────────────


def _find_lb_arrow(buttons: ButtonList, direction: str) -> "Button | None":
    """Find left (<) or right (>) arrow button on the LB page by position.

    The LB page has a '<' arrow on the left and '>' arrow on the right.
    Both are detected as Universal Buttons with 1 chevron glyph.
    We distinguish them by horizontal position relative to screen center (540px).
    """
    # screen width = 1080, midpoint = 540
    for b in buttons.buttons:
        if _detect_chevron_count(b.frame) < 1 or b.is_disabled():
            continue
        if direction == "right" and b.cx > 540:
            return b
        if direction == "left" and b.cx < 540:
            return b
    return None


def _maximize_lb_level(app: "AppProcessor") -> bool:
    """On the LB page, click '>' repeatedly to maximize the limit break level.

    Strategy: advance with '>' while 解放する stays enabled.
    If 解放する becomes disabled (枚数が足りません), revert one step with '<'.
    Returns True if 解放する is enabled after maximization, False otherwise.
    """
    # Pre-check: if 解放する is already disabled at initial level, skip arrow clicking
    buttons = ButtonList(app.latest_results)
    confirm = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK_CONFIRM, _FUZZ_CONFIG)
    if confirm is None:
        logger.debug("解放する not found on LB page — not on LB page?")
        return False
    if confirm.is_disabled():
        logger.debug("解放する already disabled at initial level (no cards available)")
        return False

    advanced = 0
    for step in range(4):  # max 4 star increments (0★→4★)
        buttons = ButtonList(app.latest_results)
        right_arrow = _find_lb_arrow(buttons, "right")
        if right_arrow is None:
            logger.debug(f"'>' arrow not found after {advanced} advances (max star level)")
            break

        # Click '>'
        app.device.click(right_arrow.cx, right_arrow.cy, el_label=">")
        sleep(0.8)
        app.game_utils.wait_frame_stable(stable_count=2)
        advanced += 1

        # Check if 解放する is still enabled
        buttons = ButtonList(app.latest_results)
        confirm = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK_CONFIRM, _FUZZ_CONFIG)
        if confirm is None or confirm.is_disabled():
            logger.debug(f"解放する disabled at advance {advanced} — reverting one step")
            left_arrow = _find_lb_arrow(buttons, "left")
            if left_arrow:
                app.device.click(left_arrow.cx, left_arrow.cy, el_label="<")
                sleep(0.8)
                app.game_utils.wait_frame_stable(stable_count=2)
                advanced -= 1
            break

        logger.debug(f"LB level advanced to step {advanced}")

    if advanced > 0:
        logger.info(f"上限解放: selected {advanced + 1}★ target (+{advanced} from initial)")
    return True


def _perform_limit_break(app: "AppProcessor") -> bool:
    """On the detail page, perform 上限解放.
    Returns True if successfully performed."""
    buttons = ButtonList(app.latest_results)
    lb_btn = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK, _FUZZ_CONFIG)
    if lb_btn is None:
        logger.debug("上限解放 button not found on detail page")
        return False
    if lb_btn.is_disabled():
        logger.debug("上限解放 button is disabled (no duplicate card available)")
        return False

    # 点击上限解放进入上限解放页面
    if not app.game_utils.click_element_and_wait_trigger(lb_btn, retries=3, timeout=3.0):
        logger.warning("上限解放 button click did not trigger")
        return False
    sleep(1)
    app.game_utils.wait_frame_stable(stable_count=2)

    # ── 点击 '>' 将上限解放等级设为最大 ──
    if not _maximize_lb_level(app):
        logger.info("上限解放: 解放する disabled — no cards available, cancelling")
        _go_back_via_cancel(app)
        return False

    # 找到确认按钮「解放する」
    confirm = None
    for attempt in range(3):
        buttons = ButtonList(app.latest_results)
        confirm = buttons.get_button_by_text(SupportCardText.LIMIT_BREAK_CONFIRM, _FUZZ_CONFIG)
        if confirm and not confirm.is_disabled():
            break
        if attempt < 2:
            sleep(0.5)
            app.game_utils.wait_frame_stable(stable_count=2)
    else:
        logger.info("上限解放 confirm button not found or disabled after maximization")
        _go_back_via_cancel(app)
        return False

    # 点击确認
    if not app.game_utils.click_element_and_wait_trigger(confirm, retries=3, timeout=3.0):
        logger.warning("解放する click did not trigger — cancelling")
        _go_back_via_cancel(app)
        return False
    sleep(1.5)
    app.game_utils.wait_frame_stable(stable_count=3)

    # 处理弹窗（解放完成提示）
    _handle_post_enhancement(app)

    # 尝试返回详情页：先试キャンセル（如果还在LB页面），再试閉じる
    if not _go_back_via_cancel(app):
        # 可能已经自动回到详情页，尝试用閉じる关闭结果面板
        buttons = ButtonList(app.latest_results)
        close_btn = buttons.get_button_by_text(SupportCardText.ENHANCE_CLOSE, _FUZZ_CONFIG)
        if close_btn:
            app.game_utils.click_element_and_wait_trigger(close_btn, retries=2, timeout=2.0)
            sleep(0.5)
            app.game_utils.wait_frame_stable(stable_count=2)

    return True


# ── サポート変換 flow ─────────────────────────────────────────────────────────

def _perform_convert_batch(app: "AppProcessor") -> int:
    """From the card list page, enter サポート変換 mode, select all, and convert.

    The サポート変換 button is on the card **list** page (not the detail page).
    Clicking it opens a batch card selection screen where the user picks cards
    to convert into 「サポートの証」.  The game automatically filters out cards
    that are in a deck or protected.

    Returns the number of cards converted (0 if nothing was converted).
    """
    # Step 1: Click サポート変換 on the card list page
    buttons = ButtonList(app.latest_results)
    conv_btn = buttons.get_button_by_text(SupportCardText.SUPPORT_CONVERT, _FUZZ_CONFIG)
    if conv_btn is None:
        logger.debug("サポート変換 button not found on card list page")
        return 0
    if conv_btn.is_disabled():
        logger.debug("サポート変換 button is disabled")
        return 0

    if not app.game_utils.click_element_and_wait_trigger(conv_btn, retries=3, timeout=3.0):
        logger.warning("サポート変換 button click did not trigger")
        return 0
    sleep(1)
    app.game_utils.wait_frame_stable(stable_count=2)

    # Step 2: Click 全選択 to select all convertible cards
    for attempt in range(3):
        buttons = ButtonList(app.latest_results)
        select_all = buttons.get_button_by_text(SupportCardText.CONVERT_SELECT_ALL, _FUZZ_CONFIG)
        if select_all and not select_all.is_disabled():
            break
        if attempt < 2:
            sleep(0.5)
            app.game_utils.wait_frame_stable(stable_count=2)
    else:
        logger.info("全選択 button not found or disabled on convert page")
        # Go back via BACK_BTN
        _go_back_to_list_from_convert(app)
        return 0

    app.game_utils.click_element_and_wait_trigger(select_all, retries=2, timeout=2.0)
    sleep(0.5)
    app.game_utils.wait_frame_stable(stable_count=2)

    # Step 3: Click 変換する to confirm
    for attempt in range(3):
        buttons = ButtonList(app.latest_results)
        confirm = buttons.get_button_by_text(SupportCardText.CONVERT_CONFIRM, _FUZZ_CONFIG)
        if confirm and not confirm.is_disabled():
            break
        if attempt < 2:
            sleep(0.5)
            app.game_utils.wait_frame_stable(stable_count=2)
    else:
        logger.info("変換する button not found or still disabled after select all")
        _go_back_to_list_from_convert(app)
        return 0

    app.game_utils.click_element_and_wait_trigger(confirm, retries=3, timeout=3.0)
    sleep(1.5)
    app.game_utils.wait_frame_stable(stable_count=3)

    # Step 4: Handle 変換確認 confirmation modal (キャンセル / 決定)
    for attempt in range(3):
        buttons = ButtonList(app.latest_results)
        decision = buttons.get_button_by_text(SupportCardText.CONVERT_DECISION, _FUZZ_CONFIG)
        if decision and not decision.is_disabled():
            break
        if attempt < 2:
            sleep(0.5)
            app.game_utils.wait_frame_stable(stable_count=2)
    else:
        logger.info("決定 button not found on convert confirmation modal")
        # Try cancel or go back
        cancel = ButtonList(app.latest_results).get_button_by_text(
            SupportCardText.ENHANCE_CANCEL, _FUZZ_CONFIG
        )
        if cancel:
            app.game_utils.click_element_and_wait_trigger(cancel, retries=2, timeout=2.0)
        _go_back_to_list_from_convert(app)
        return 0

    app.game_utils.click_element_and_wait_trigger(decision, retries=3, timeout=3.0)
    sleep(1.5)
    for _ in range(3):
        if app.latest_results.exists_label(BaseUILabels.SUPPORT_CARD):
            sleep(1)
        else:
            break
    app.game_utils.wait_frame_stable(stable_count=3)

    # Step 5: Handle post-convert modal(s) (completion popup)
    _handle_post_enhancement(app)

    # Step 5: We should now be back on the card list or on the convert page
    # Try to go back to list.
    _go_back_to_list_from_convert(app)

    logger.info("サポート変換 batch completed")
    return 1  # at least 1 batch was converted


def _go_back_to_list_from_convert(app: "AppProcessor"):
    """Go back from the サポート変換 page to the card list.

    The convert page uses the BACK_BTN (<<), not キャンセル.
    """
    for attempt in range(3):
        try:
            app.game_utils.back_next_page()
            return
        except TimeoutError:
            if attempt < 2:
                sleep(0.5)
                app.game_utils.wait_frame_stable(stable_count=2)


# ── Main action ───────────────────────────────────────────────────────────────

def action__auto_enhance_support_cards(app: "AppProcessor") -> bool:
    """
    Main action: iterate through support card list and enhance cards.

    Tested ADB flow:
      Card list → click thumbnail (select) → click 詳細を見る → detail page
      → click Lv強化 → enhancement page → click >> (max) → click Lv強化 (confirm)
      → キャンセル → Back Button → card list

    Level caps per rarity and star count (from game data):
      R:   0★=20, 1★=25, 2★=30, 3★=35, 4★=40
      SR:  0★=30, 1★=35, 2★=40, 3★=45, 4★=50
      SSR: 0★=40, 1★=45, 2★=50, 3★=55, 4★=60
    """
    enhancement_config = _get_enhancement_config(app)
    task_config = app.config_service().task__auto_enhancement_support_card
    whitelist_mode = task_config.whitelist_mode.value
    whitelist_ids = task_config.whitelist_card_ids.value or []
    auto_limit_break = task_config.auto_limit_break.value
    auto_convert = task_config.auto_convert.value

    if not any(cfg["enabled"] for cfg in enhancement_config.values()):
        message_tools.info(i18n_text("backend.task.noEnhancementConfigured", fallback="未配置任何需要强化的品级"), 10)
        return True

    width, _ = app.device.get_window_size()
    prev_frame: Optional[np.ndarray] = None
    total_enhanced = 0
    total_limit_breaks = 0
    total_converted = 0
    max_scroll_attempts = 30
    scroll_count = 0
    # 本次运行中已确认无法强化的卡（材料不足或已达硬上限），避免重复进入
    failed_clip_ids: set[str] = set()
    # CLIP 后备：用 YOLO 框位置（15px 网格对齐）标记失败卡，滚屏后清除
    failed_positions: set[tuple[int, int]] = set()
    # 全局材料耗尽标记：一旦确认 サポート強化Pt 不足，跳过所有后续强化
    materials_exhausted = False

    while scroll_count < max_scroll_attempts:
        app.game_utils.wait_frame_stable(stable_count=2)

        # Parse visible cards on list page
        card_list = SupportCardListParser(app.latest_results).parse()
        if not card_list:
            logger.debug("No support cards detected, scrolling...")
            _scroll_card_list(app, width)
            failed_positions.clear()  # positions invalidated after scroll
            scroll_count += 1
            continue

        # Filter cards that need enhancement (pre-check from thumbnails)
        cards_to_enhance: list[SupportCardComponent] = []
        cards_to_limit_break: list[SupportCardComponent] = []
        for card in card_list:
            if card.occluded:
                continue
            pos_key = (round(card.box.cx / 15) * 15, round(card.box.cy / 15) * 15)
            if pos_key in failed_positions:
                continue  # already confirmed unenhanceable this scroll page

            # 检查是否需要上限解放
            if _card_needs_limit_break(card, enhancement_config, auto_limit_break):
                cards_to_limit_break.append(card)

            # 检查是否需要强化（材料未耗尽时才加入）
            if not materials_exhausted and _card_needs_enhancement(
                card, enhancement_config, whitelist_mode, whitelist_ids
            ):
                cards_to_enhance.append(card)

        # Show all detected cards in debug overlay with colour-coded status
        debug_tools.clear_all()
        enhance_set = set(id(c) for c in cards_to_enhance)
        lb_set = set(id(c) for c in cards_to_limit_break)
        for card in card_list:
            _pos = (round(card.box.cx / 15) * 15, round(card.box.cy / 15) * 15)
            actual_cap = _get_level_cap(card.rarity, card.stars) if card.rarity else 0
            stars_str = f"{'★' * (card.stars or 0)}" if card.stars else ""
            if card.occluded:
                color = (0, 100, 255)   # orange: occluded
                label = f"{card.rarity or '?'} Lv? (遮挡)"
            elif _pos in failed_positions:
                color = (0, 0, 200)     # red: given up this page
                label = f"{card.rarity or '?'} Lv{card.level}/{actual_cap} {stars_str} (已放弃)"
            elif id(card) in lb_set and id(card) in enhance_set:
                color = (255, 200, 0)   # gold: needs both LB + enhance
                label = f"{card.rarity} Lv{card.level}/{actual_cap} {stars_str} 解放+✓"
            elif id(card) in lb_set:
                color = (255, 165, 0)   # orange: needs limit break
                label = f"{card.rarity} Lv{card.level}/{actual_cap} {stars_str} 解放"
            elif id(card) in enhance_set:
                color = (0, 255, 0)     # green: will enhance
                label = f"{card.rarity} Lv{card.level}/{actual_cap} {stars_str} ✓"
            else:
                color = (128, 128, 128) # grey: skip
                label = f"{card.rarity or '?'} Lv{card.level}/{actual_cap} {stars_str} (跳过)"
            debug_tools.add_box(
                int(card.box.x), int(card.box.y),
                int(card.box.w), int(card.box.h),
                label=label,
                color=color,
                duration=300,
            )

        # 合并需要处理的卡片（去重，优先上限解放再强化）
        # ── Incremental: pick only the FIRST actionable card, process it,
        #    then rescan the page.  This avoids relying on stale positions
        #    from a batch scan and keeps each iteration fast (~1s scan).
        target_card: SupportCardComponent | None = None
        target_action: str = "enhance"
        for card in cards_to_limit_break:
            target_card = card
            target_action = "limit_break"
            break
        if target_card is None:
            for card in cards_to_enhance:
                target_card = card
                target_action = "enhance"
                break

        if target_card is None:
            current_frame = app.latest_frame
            if prev_frame is not None and check_frame_change(prev_frame, current_frame):
                logger.debug("Reached end of card list (no frame change after scroll)")
                break
            prev_frame = current_frame.copy() if current_frame is not None else None
            _scroll_card_list(app, width)
            failed_positions.clear()  # positions invalidated after scroll
            scroll_count += 1
            continue

        # ── Process this single card: select → view detail → enhance/limit_break → back
        card = target_card
        action_type = target_action
        actual_cap = _get_level_cap(card.rarity, card.stars) if card.rarity else 0
        card_label = f"{card.rarity} Lv{card.level}/{actual_cap} ★{card.stars or 0} ({action_type})"
        logger.info(f"▶ 开始处理卡片 {card_label}")

        # Step 1: Click card thumbnail to select it
        logger.debug(f"  [Step1] 点击缩略图 {card_label}")
        app.game_utils.click_element_and_wait_trigger(
            card.box, retries=3, timeout=2.0,
        )
        sleep(0.5)

        # Step 2: Navigate to detail page via 詳細を見る
        logger.debug(f"  [Step2] 进入详情页 {card_label}")
        if not _click_view_detail(app):
            logger.warning(f"  [Step2] 未找到「詳細を見る」，跳过此卡 {card_label}")
            _fail_pos = (round(card.box.cx / 15) * 15, round(card.box.cy / 15) * 15)
            failed_positions.add(_fail_pos)
            continue  # rescan on next iteration

        # Step 3: CLIP identify + whitelist check
        logger.debug(f"  [Step3] CLIP识别 {card_label}")
        clip_id: str | None = None
        frame = app.latest_frame
        if frame is not None and frame.size > 0:
            height = frame.shape[0]
            card_image = frame[:int(height * 0.20), :]

            clip_id = _try_clip_identify(app, card_image)
            if clip_id is None:
                _clip_learn_from_detail(app, card_image)
                clip_id = _try_clip_identify(app, card_image)

            logger.debug(f"  [Step3] CLIP id={clip_id or '未识别'} {card_label}")

            # 跳过本次运行中已确认无法强化的卡
            if clip_id and clip_id in failed_clip_ids:
                logger.info(f"  [Step3] 跳过（本次已确认无法强化）{clip_id}")
                _go_back_to_list(app)
                sleep(0.5)
                _ensure_on_card_list(app, max_attempts=3)
                continue  # rescan

            # Re-check whitelist with CLIP id
            if whitelist_mode and clip_id and clip_id not in whitelist_ids:
                logger.info(f"  [Step3] 不在白名单，跳过 {clip_id}")
                _go_back_to_list(app)
                sleep(0.5)
                _ensure_on_card_list(app, max_attempts=3)
                continue  # rescan

        # Step 4: Perform action based on type
        if action_type == "limit_break":
            logger.debug(f"  [Step4] 执行上限解放 {card_label}")
            lb_result = _perform_limit_break(app)
            if lb_result:
                total_limit_breaks += 1
                logger.success(f"  [Step4] 上限解放成功 {card_label} clip={clip_id or '?'}")
            else:
                logger.info(f"  [Step4] 上限解放失败 {card_label} clip={clip_id or '?'}")
                _fail_pos = (round(card.box.cx / 15) * 15, round(card.box.cy / 15) * 15)
                failed_positions.add(_fail_pos)
        else:
            logger.debug(f"  [Step4] 执行强化 {card_label}")
            result = _enhance_single_card(app)
            if result == "success":
                total_enhanced += 1
                logger.success(f"  [Step4] 强化成功 {card_label} clip={clip_id or '?'}")
            elif result == "no_materials":
                logger.warning(f"  [Step4] サポート強化Pt 不足，停止后续所有强化 {card_label}")
                materials_exhausted = True
                _fail_pos = (round(card.box.cx / 15) * 15, round(card.box.cy / 15) * 15)
                failed_positions.add(_fail_pos)
                if clip_id:
                    failed_clip_ids.add(clip_id)
            else:
                logger.info(f"  [Step4] 强化失败（{result}）{card_label} clip={clip_id or '?'}")
                _fail_pos = (round(card.box.cx / 15) * 15, round(card.box.cy / 15) * 15)
                failed_positions.add(_fail_pos)
                if clip_id:
                    failed_clip_ids.add(clip_id)
                    logger.debug(f"  [Step4] {clip_id} 加入本次跳过列表")
                else:
                    logger.debug(f"  [Step4] pos={_fail_pos} 加入本次跳过列表（CLIP未识别，用位置替代）")

        # Step 5: Return to card list and rescan on next iteration.
        logger.debug(f"  [Step5] 返回列表 {card_label}")
        _go_back_to_list(app)
        sleep(1)
        app.game_utils.wait_frame_stable(stable_count=2)
        if not _ensure_on_card_list(app, max_attempts=3):
            logger.warning(f"  [Step5] 返回列表超时，重新扫描 {card_label}")

        prev_frame = None
        scroll_count = 0

        # 如果材料耗尽且没有上限解放需要处理，提前退出
        if materials_exhausted and not auto_limit_break:
            logger.info("サポート強化Pt 已耗尽且未开启上限解放，结束本轮处理")
            break

    # Don't clear debug overlay here -- let boxes expire naturally (duration=300s)
    # so the user can review the final state of the card list.

    # ── サポート変換 phase: batch convert after all enhancement/limit break ──
    if auto_convert:
        logger.info("サポート変換 phase: entering batch convert mode")
        # Ensure we're on the card list page
        if _ensure_on_card_list(app, max_attempts=5):
            app.game_utils.wait_frame_stable(stable_count=2)
            converted = _perform_convert_batch(app)
            total_converted += converted
            if converted > 0:
                logger.success(f"サポート変換 batch completed: {converted} batch(es)")
            else:
                logger.info("サポート変換: no cards to convert or convert skipped")
        else:
            logger.warning("サポート変換 skipped: could not return to card list page")

    # 生成汇总消息
    summary_parts = [f"共强化 {total_enhanced} 张卡"]
    if total_limit_breaks > 0:
        summary_parts.append(f"上限解放 {total_limit_breaks} 张")
    if total_converted > 0:
        summary_parts.append(f"变换 {total_converted} 张")
    if materials_exhausted:
        summary_parts.append("（サポート強化Pt 已耗尽）")
    summary = "支援卡强化完成，" + "，".join(summary_parts)

    message_tools.info(summary, 10)
    logger.success(f"Support card enhancement complete: {total_enhanced} enhanced, "
                    f"{total_limit_breaks} limit breaks, {total_converted} converted"
                    f"{' (materials exhausted)' if materials_exhausted else ''}")
    return True


def _scroll_card_list(app: "AppProcessor", screen_width: int):
    """Scroll down the card list."""
    if isinstance(app.device, Android_App):
        _, screen_height = app.device.get_window_size()
        start_y = int(screen_height * 0.7)
        end_y = int(screen_height * 0.35)
        app.device.swipe(
            screen_width // 2, start_y,
            screen_width // 2, end_y,
            offset_y=0,
        )
    sleep(1)
    app.game_utils.wait_frame_stable()
