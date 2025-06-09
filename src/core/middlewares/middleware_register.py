import cv2
import numpy as np

from src.core.CLIP_services.skill_card import SkillCardInfo
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.game_tools import extract_skill_card_and_info, get_current_location
from src.utils.logger import logger
from src.constants import *
from typing import TYPE_CHECKING

from src.utils.ocr_instance import get_ocr, OCRService, OCR_ResultList
from src.utils.yolo_tools import get_modal
from tests.ORB_test import handle_new_image

if TYPE_CHECKING:
    from app import AppProcessor

last_card_name = ""

def register_middlewares(processor: "AppProcessor"):
    @processor.register_middleware()
    @logger.catch
    def _init_location(app: "AppProcessor"):
        if app.game_status_manager.current_location is None:
            app.update_current_location()
            app.exec_task()
        return True


    # @processor.register_middleware()
    # @logger.catch
    # def _handle_unexpected_modal(app: "AppProcessor"):
    #     if app.latest_results.exists_label(base_labels.modal_header):
    #         modal_header = get_modal(app.latest_results, app.latest_frame, True)
    #         pass
    #     return True

    @processor.register_middleware()
    @logger.catch
    def _add_skill(app: "AppProcessor"):
        global last_card_name
        if app.game_status_manager.current_location == GamePageTypes.SUB_MENU.PRODUCER_ILLUSTRATED:
            current_location = get_current_location(app.latest_results)
            if current_location != GamePageTypes.SUB_MENU.PRODUCER_ILLUSTRATED:
                app.game_status_manager.current_location = current_location
                return
            roi , skill_card, card_info = extract_skill_card_and_info(app.latest_frame)
            if skill_card is None or card_info is None:
                return
            ocr_service = OCRService()
            ocr_result = ocr_service.ocr(card_info)
            if ocr_result is None:
                return
            ocr_result.auto_merge_lines(width_gap=100)
            card_title = ocr_result.get_y_min().text
            if card_title != last_card_name:
                last_card_name = card_title
                return
            card_info = ocr_result.exclude([ocr_result.get_y_min()])
            card_info = OCR_ResultList([item for item in card_info if len(item.text) > 2])
            skill_card_types = [base_labels.skill_card, base_labels.skill_card__mental, base_labels.skill_card__active, base_labels.skill_card__trap]

            if not app.clip_manager.skill_card_clip.add_to_memory(skill_card, SkillCardInfo(card_title, app.latest_results.filter_by_labels(skill_card_types).get_y_min_element().first().label.replace("Skill Card: ", ""), [item.text for item in card_info]), 0.97):
                logger.debug(app.clip_manager.skill_card_clip.retrieve(skill_card))