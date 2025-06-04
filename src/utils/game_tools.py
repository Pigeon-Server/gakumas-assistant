from typing import List

import cv2

from src.entity.Yolo import Yolo_Results, Yolo_Box
from src.constants import *
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.logger import logger
from src.utils.ocr_instance import get_ocr
from src.utils.opencv_tools import check_status_detection

@logger.catch
def get_current_location(boxes: Yolo_Results) -> str | None:
    if boxes.exists_label(base_labels.start_menu_logo): return GamePageTypes.START_GAME
    if boxes.exists_label(base_labels.general_loading1) or boxes.exists_label(
        base_labels.general_loading2): return GamePageTypes.LOADING
    # 映射标签 → 页面类型
    TAB_LABEL_TO_PAGE = {
        base_labels.tab_communicate: GamePageTypes.MAIN_MENU__COMMUNICATE,
        base_labels.tab_idol: GamePageTypes.MAIN_MENU__IDOL,
        base_labels.tab_home: GamePageTypes.MAIN_MENU__HOME,
        base_labels.tab_gacha: GamePageTypes.MAIN_MENU__GACHA,
        base_labels.tab_contest: GamePageTypes.MAIN_MENU__CONTEST,
        location.page__present: GamePageTypes.HOME_TAB.GIFT,
        location.page__daily_task: GamePageTypes.HOME_TAB.TASK,
        location.page__achievement: GamePageTypes.HOME_TAB.ACHIEVEMENT,
        location.page__achievement_idol: GamePageTypes.HOME_TAB.ACHIEVEMENT_SUB_PAGR.IDOL,
        location.page__achievement_producer: GamePageTypes.HOME_TAB.ACHIEVEMENT_SUB_PAGR.PRODUCER,
        location.page__achievement_other: GamePageTypes.HOME_TAB.ACHIEVEMENT_SUB_PAGR.OTHER,
        location.page__plan: GamePageTypes.HOME_TAB.MISSION_PASS,
        location.page__dispatch_work: GamePageTypes.HOME_TAB.WORK,
        location.page__shop: GamePageTypes.HOME_TAB.SHOP,
        location.page__shop_gem: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.GEM,
        location.page__shop_pack: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.PACK,
        location.page__shop_pass: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.PASS,
        location.page__shop_coin_gacha: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.COIN_GACHA,
        location.page__shop_daily_exchange: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.DAILY_EXCHANGE,
        location.page__shop_costume_exchange: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.COSTUME_EXCHANGE,
        location.page__shop_item_exchange: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.ITEM_EXCHANGE,
        location.page__shop_ticket_exchange: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.TICKET_EXCHANGE,
        location.page__contest: GamePageTypes.CONTEST.ARENA,
        location.page__the_road_to_idols: GamePageTypes.CONTEST.THE_ROAD_TO_IDOL,
        location.page__hatsusei_community: GamePageTypes.Communicate.MAIN_STORY,
        location.page__idol_community: GamePageTypes.Communicate.BOND_STORIES,
        location.page__produce_card_list: GamePageTypes.Communicate.SUPPORT_CARD_ARCHIVE,
        location.page__event_plot: GamePageTypes.Communicate.PAST_EVENTS,
    }
    MAIN_UI_TABS = list(TAB_LABEL_TO_PAGE.keys())[:5]
    if boxes.exists_all_labels(MAIN_UI_TABS):
        home_tab_bar = boxes.filter_by_labels(MAIN_UI_TABS)
        for item in home_tab_bar:
            if check_status_detection(item.frame):
                return TAB_LABEL_TO_PAGE.get(item.label)
    elif boxes.exists_label(base_labels.current_location):
        current_location = boxes.filter_by_label(base_labels.current_location).first()
        if current_location.frame is None or current_location.frame.size == 0:
            return GamePageTypes.UNKNOWN
        location_text = get_ocr(current_location.frame)
        if location_text is None:
            return GamePageTypes.UNKNOWN
        location_text = " ".join([ocr_item.text for ocr_item in location_text])
        for label in TAB_LABEL_TO_PAGE.keys():
            if label in location_text:
                return TAB_LABEL_TO_PAGE.get(label)
    return GamePageTypes.UNKNOWN
