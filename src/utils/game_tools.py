import cv2
import numpy as np

import config
from src.constants.location import Location
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.entity.Game.Components.Modal import Modal
from src.entity.Game.Components.SupportCard import SupportCard, SupportCardList
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.logger import logger
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.inference.ocr_engine import OCRService
from src.utils.contest_overlay_tools import detect_contest_grade_up_splash, detect_contest_season_overlay
from src.utils.opencv_tools import check_status_detection, get_mask_contours, extract_roi_from_mask, check_color, \
    filter_by_rectangle_shape, get_max_contour
from src.utils.performance_tools import timeit
from src.utils.string_tools import string_match, MatchConfig

ocr_service = OCRService()

@timeit
def get_current_location(boxes: Yolo_Results) -> str | None:
    # Priority 1: Check for start menu
    if boxes.exists_label(BaseUILabels.START_MENU_LOGO):
        return GamePageTypes.START_GAME
    
    # 映射标签 → 页面类型
    TAB_LABEL_TO_PAGE = {
        BaseUILabels.TAB_COMMUNICATE: GamePageTypes.MAIN_MENU__COMMUNICATE,
        BaseUILabels.TAB_IDOL: GamePageTypes.MAIN_MENU__IDOL,
        BaseUILabels.TAB_HOME: GamePageTypes.MAIN_MENU__HOME,
        BaseUILabels.TAB_GACHA: GamePageTypes.MAIN_MENU__GACHA,
        BaseUILabels.TAB_CONTEST: GamePageTypes.MAIN_MENU__CONTEST,
        Location.MAIN_UI.Page.PRESENT: GamePageTypes.HOME_TAB.GIFT,
        Location.MAIN_UI.Page.DAILY_TASK: GamePageTypes.HOME_TAB.TASK,
        Location.MAIN_UI.Page.ACHIEVEMENT: GamePageTypes.HOME_TAB.ACHIEVEMENT,
        Location.MAIN_UI.Page.ACHIEVEMENT_IDOL: GamePageTypes.HOME_TAB.ACHIEVEMENT_SUB_PAGE.IDOL,
        Location.MAIN_UI.Page.ACHIEVEMENT_PRODUCER: GamePageTypes.HOME_TAB.ACHIEVEMENT_SUB_PAGE.PRODUCER,
        Location.MAIN_UI.Page.ACHIEVEMENT_OTHER: GamePageTypes.HOME_TAB.ACHIEVEMENT_SUB_PAGE.OTHER,
        Location.MAIN_UI.Page.PLAN: GamePageTypes.HOME_TAB.PASS_REWARD,
        Location.MAIN_UI.Page.DISPATCH_WORK: GamePageTypes.HOME_TAB.WORK,
        Location.MAIN_UI.Page.SHOP: GamePageTypes.HOME_TAB.SHOP,
        Location.MAIN_UI.Page.SHOP_GEM: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.GEM,
        Location.MAIN_UI.Page.SHOP_PACK: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.PACK,
        Location.MAIN_UI.Page.SHOP_PASS: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.PASS,
        Location.MAIN_UI.Page.SHOP_COIN_GACHA: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.COIN_GACHA,
        Location.MAIN_UI.Page.SHOP_DAILY_EXCHANGE: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.DAILY_EXCHANGE,
        Location.MAIN_UI.Page.SHOP_COSTUME_EXCHANGE: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.COSTUME_EXCHANGE,
        Location.MAIN_UI.Page.SHOP_ITEM_EXCHANGE: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.ITEM_EXCHANGE,
        Location.MAIN_UI.Page.SHOP_TICKET_EXCHANGE: GamePageTypes.HOME_TAB.SHOP_SUB_PAGE.TICKET_EXCHANGE,
        Location.MAIN_UI.Page.CONTEST: GamePageTypes.CONTEST_TAB.ARENA,
        Location.MAIN_UI.Page.THE_ROAD_TO_IDOLS: GamePageTypes.CONTEST_TAB.THE_ROAD_TO_IDOL,
        Location.MAIN_UI.Page.HATSUSEI_COMMUNITY: GamePageTypes.Communicate_TAB.MAIN_STORY,
        Location.MAIN_UI.Page.IDOL_COMMUNITY: GamePageTypes.Communicate_TAB.BOND_STORIES,
        Location.MAIN_UI.Page.PRODUCE_CARD_LIST: GamePageTypes.Communicate_TAB.SUPPORT_CARD_ARCHIVE,
        Location.MAIN_UI.Page.EVENT_PLOT: GamePageTypes.Communicate_TAB.PAST_EVENTS,
        Location.MAIN_UI.Page.PRODUCER_ILLUSTRATED: GamePageTypes.SUB_MENU.PRODUCER_ILLUSTRATED,
    }
    MAIN_UI_TABS = list(TAB_LABEL_TO_PAGE.keys())[:5]
    
    # Priority 2: Check for main UI tabs (color detection)
    if boxes.exists_all_labels(MAIN_UI_TABS):
        home_tab_bar = boxes.filter_by_labels(MAIN_UI_TABS)
        for item in home_tab_bar:
            if check_status_detection(
                item.frame,
                upper_color=(13,215,255),
                lower_color=(5,149,205),
                background_upper_color=(30,115,255),
                background_lower_color=(0,35,225)
            ):
                return TAB_LABEL_TO_PAGE.get(item.label)
    
    # Priority 3: Check "Current Location" label with OCR fallback
    # This is more reliable than loading detection for identifying actual pages
    if current_location := boxes.filter_by_label(BaseUILabels.CURRENT_LOCATION):
        current_location = current_location.first()
        if current_location.frame is not None and current_location.frame.size > 0:
            ocr_result = ocr_service.ocr(current_location.frame)
            if ocr_result and len(ocr_result.results) > 0:
                location_text = "".join([ocr_item.text for ocr_item in ocr_result])
                logger.debug(f"OCR detected location text: '{location_text}'")
                match_result = string_match(location_text, list(TAB_LABEL_TO_PAGE.keys()), MatchConfig(fuzz_threshold=85))
                if match_result and match_result.status:
                    logger.debug(f"Matched location: '{match_result.result}' -> {TAB_LABEL_TO_PAGE.get(match_result.result)}")
                    return TAB_LABEL_TO_PAGE.get(match_result.result)
                else:
                    logger.debug(f"No match found for location text: '{location_text}'")
            else:
                logger.debug("OCR returned no text from Current Location frame")
        else:
            logger.debug("Current Location frame is None or empty")
    
    # Priority 4: Check for loading indicators (lowest priority)
    # Only return LOADING if no other location could be determined
    if detect_contest_season_overlay(boxes.frame):
        logger.debug("Contest season overlay detected, returning ARENA")
        return GamePageTypes.CONTEST_TAB.ARENA
    if detect_contest_grade_up_splash(boxes.frame):
        logger.debug("Contest grade-up splash detected, returning ARENA")
        return GamePageTypes.CONTEST_TAB.ARENA

    if boxes.exists_label(BaseUILabels.GENERAL_LOADING1) or boxes.exists_label(BaseUILabels.GENERAL_LOADING2):
        logger.debug("Loading indicators detected, returning LOADING")
        return GamePageTypes.LOADING
    
    return GamePageTypes.UNKNOWN

@timeit
def modal_body_extract_item_info(
        img,
        item_lower: tuple[int, int, int] = (0,0,0),
        item_upper: tuple[int, int, int] = (179,90,120),
        mark_lower: tuple[int, int, int] = (85,55,210),
        mark_upper: tuple[int, int, int] = (105,175,255)
):
    """
    在模态框中提取物品信息
    :param img: 图像
    :param item_lower: 物品外框下值
    :param item_upper: 物品外框上值
    :param mark_lower: 锚点下值（已放宽以抗 JPG 压缩噪点）
    :param mark_upper: 锚点上值
    :return: (item_image, item_info_image) or (None, None)
    """
    if img is None or img.size == 0:
        return None, None
    img_h, img_w = img.shape[:2]
    # 取出物品：最小面积按图像尺寸自适应，防止 JPG 噪点产生的小矩形
    min_item_area = max(50, int(img_h * img_w * 0.002))

    # 主路径：小核形态学，适用于高质量截图
    item_contours = get_mask_contours(img, item_lower, item_upper, iterations=2)
    item_contours = filter_by_rectangle_shape(item_contours, min_item_area)

    # 降级：更大形态学核弥合 JPG 压缩导致的边框断裂
    if not item_contours:
        item_contours = get_mask_contours(
            img, item_lower, item_upper,
            ksize=(5, 5), iterations=2,
        )
        item_contours = filter_by_rectangle_shape(item_contours, min_item_area)

    if not item_contours:
        return None, None
    item_x,item_y,item_w,item_h = cv2.boundingRect(get_max_contour(item_contours))
    item = img[item_y:item_y+item_h, item_x:item_x+item_w]
    # 取出物品信息锚点：统一要求 contour 面积大于噪点阈值
    mark_y = 0
    min_mark_size = max(5, int(min(img_h, img_w) * 0.01))
    contours = get_mask_contours(img[item_y+item_h:], mark_lower, mark_upper)
    for contour in contours:
        _x,_y,_w,_h = cv2.boundingRect(contour)
        if _h > min_mark_size and _w > min_mark_size:
            mark_y = min(_y, mark_y) if mark_y > 0 else _y
    mark_y = item_y+item_h+mark_y
    if mark_y <= item_y + item_h and not check_color(img[item_y+item_h:item_y+item_h+20, item_x+item_w:], (0,0,45), (0,0,108), threshold=5):
        item_info = img[max(0, item_y-10):item_y+item_h, item_x+item_w:]
    else:
        item_info = img[max(0, item_y-10):mark_y, item_x+item_w:]

    return item, item_info

@timeit
def get_modal(yolo_result: Yolo_Results, no_body: bool = False, quiet: bool = False) -> Modal | None:
    """
    获取模态框
    :param yolo_result: yolo结果集
    :param no_body: 不识别模态框主体（加速）
    :return: 解析后的 Modal 对象
    """
    return Modal.from_yolo_results(yolo_result, no_body=no_body, quiet=quiet)


@timeit
def get_support_card(yolo_box: Yolo_Box) -> SupportCard:
    """
    解析单张支援卡
    :param yolo_box: 单个 Support Card 的 Yolo_Box
    :return: 解析后的 SupportCard 对象
    """
    return SupportCard.from_yolo_box(yolo_box)


@timeit
def get_support_cards(yolo_result: Yolo_Results) -> SupportCardList:
    """
    获取当前画面支援卡识别结果
    :param yolo_result: yolo结果集
    :return: 解析后的 SupportCardList 对象
    """
    return SupportCardList.from_yolo_results(yolo_result)


@timeit
def extract_support_cards(
        image: np.ndarray,
        model: YoloModelFromONNX | None = None,
        conf_threshold: float = 0.7,
) -> SupportCardList:
    """
    从完整截图中提取支援卡信息
    :param image: 原始截图
    :param model: 可复用的 BASE_UI 模型
    :param conf_threshold: YOLO 置信度阈值
    :return: 解析后的 SupportCardList 对象
    """
    if image is None or image.size == 0:
        return SupportCardList()
    base_ui_model = model if model is not None else YoloModelFromONNX(config.model_config["BASE_UI"])
    yolo_result = Yolo_Results(base_ui_model(image, conf_threshold=conf_threshold), image)
    return get_support_cards(yolo_result)


def get_support_card_level_stars(yolo_result: Yolo_Results) -> SupportCardList:
    return get_support_cards(yolo_result)


def extract_support_card_level_stars(
        image: np.ndarray,
        model: YoloModelFromONNX | None = None,
        conf_threshold: float = 0.7,
) -> SupportCardList:
    return extract_support_cards(image, model=model, conf_threshold=conf_threshold)
