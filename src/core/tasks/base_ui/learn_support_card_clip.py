from time import sleep
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.constants.game.text.support_card_text import SupportCardText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.device.Android.app import Android_App
from src.core.inference.ocr_engine import OCRService
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.SupportCard import (
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


def _try_clip_identify(app: "AppProcessor", card_frame: np.ndarray) -> Optional[str]:
    try:
        db_card = app.clip_manager.support_card_clip.retrieve(card_frame)
        if db_card is not None:
            return db_card.id
    except Exception as e:
        logger.debug(f"CLIP identify failed: {e}")
    return None


def _clip_learn_from_detail(app: "AppProcessor", card_image: np.ndarray) -> Optional[str]:
    """Try to learn the card via OCR on the detail page header.
    Returns the learned card's database ID, or None if learning failed."""
    frame = app.latest_frame
    if frame is None or frame.size == 0:
        return None
    height = frame.shape[0]
    header = frame[:int(height * 0.20), :]
    ocr_result = ocr_service.ocr(header)
    if not ocr_result or not ocr_result.results:
        return None
    for item in ocr_result.results:
        if len(item.text) >= 3:
            status, db_result = support_card_db.search(item.text)
            if status and db_result:
                try:
                    app.clip_manager.support_card_clip.add_to_memory(card_image, db_result)
                    logger.debug(f"[CLIP] Learned: {db_result.name} ({db_result.id})")
                    return db_result.id
                except Exception as e:
                    logger.warning(f"[CLIP] Learn failed: {e}")
                return None
    return None


def action__learn_support_card_clip(app: "AppProcessor") -> bool:
    """
    手动任务：遍历所有支援卡列表，逐张进入详情页进行 CLIP 学习。

    流程:
      Card list → click thumbnail → click 詳細を見る → detail page
      → OCR card name → CLIP learn → Back Button → card list → next card → scroll
    """
    width, _ = app.device.get_window_size()
    prev_frame: Optional[np.ndarray] = None
    total_learned = 0
    total_already_known = 0
    total_failed = 0
    max_scroll_attempts = 50
    scroll_count = 0

    message_tools.info(i18n_text("backend.task.startSupportCardCLIP", fallback="开始支援卡 CLIP 学习，请确保已进入支援卡列表页面"), 5)
    sleep(2)

    while scroll_count < max_scroll_attempts:
        app.game_utils.wait_frame_stable(stable_count=2)

        # Parse visible cards on list page
        card_list = SupportCardListParser(app.latest_results).parse()
        if not card_list:
            logger.debug("No support cards detected, scrolling...")
            _scroll_card_list(app, width)
            scroll_count += 1
            continue

        # Filter non-occluded cards
        visible_cards = [c for c in card_list if not c.occluded]
        if not visible_cards:
            _scroll_card_list(app, width)
            scroll_count += 1
            continue

        total_cards = len(visible_cards)
        for card_idx, card in enumerate(visible_cards):
            card_label = f"[{card_idx + 1}/{total_cards}] {card.rarity or '?'} Lv{card.level}"
            logger.info(f"▶ CLIP学习: {card_label}")

            # Step 1: Click card thumbnail
            app.game_utils.click_element_and_wait_trigger(
                card.box, retries=3, timeout=2.0,
            )
            sleep(0.5)

            # Step 2: Enter detail page
            buttons = ButtonList(app.latest_results)
            detail_btn = buttons.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ_CONFIG)
            if detail_btn is None:
                # Retry once after waiting
                sleep(0.5)
                app.game_utils.wait_frame_stable(stable_count=2)
                buttons = ButtonList(app.latest_results)
                detail_btn = buttons.get_button_by_text(SupportCardText.VIEW_DETAIL, _FUZZ_CONFIG)
            if detail_btn is None:
                logger.warning(f"  詳細を見る not found, skipping {card_label}")
                continue

            app.game_utils.click_element_and_wait_trigger(detail_btn, retries=3, timeout=3.0)
            sleep(0.8)
            app.game_utils.wait_frame_stable(stable_count=2)

            # Step 3: Try CLIP identify first
            frame = app.latest_frame
            if frame is not None and frame.size > 0:
                height = frame.shape[0]
                card_image = frame[:int(height * 0.20), :]

                existing_id = _try_clip_identify(app, card_image)
                if existing_id:
                    logger.debug(f"  Already known: {existing_id}")
                    total_already_known += 1
                else:
                    # Learn via OCR
                    learned_id = _clip_learn_from_detail(app, card_image)
                    if learned_id:
                        total_learned += 1
                        logger.success(f"  Learned: {learned_id}")
                    else:
                        total_failed += 1
                        logger.warning(f"  Failed to learn {card_label}")

            # Step 4: Go back to card list
            try:
                app.game_utils.back_next_page()
            except TimeoutError:
                pass
            sleep(1)
            app.game_utils.wait_frame_stable(stable_count=2)
            app.game_utils.wait_for_label(BaseUILabels.SUPPORT_CARD, timeout=10)

        # Scroll down for more cards
        current_frame = app.latest_frame
        if prev_frame is not None and check_frame_change(prev_frame, current_frame):
            logger.debug("Reached end of card list")
            break
        prev_frame = current_frame.copy() if current_frame is not None else None
        _scroll_card_list(app, width)
        scroll_count += 1

    summary = (f"CLIP学习完成: 新学习 {total_learned} 张, "
               f"已知 {total_already_known} 张, 失败 {total_failed} 张")
    message_tools.info(summary, 10)
    logger.success(summary)
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
