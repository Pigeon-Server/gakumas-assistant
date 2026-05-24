from time import sleep
from typing import Optional, Tuple, TYPE_CHECKING
import os
import numpy as np

from src.constants.game.text.general_text import GeneralText
from src.constants.path.debug_path import DebugPath
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.device.Android.app import Android_App
from src.core.inference.ocr_engine import OCRService
from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.CheckBox import CheckBox
from src.entity.Game.Components.TabBar import TabBar
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Yolo import Yolo_Box
from src.utils.debug_tools import DebugTools
from src.utils.game_database_tools import GakumasDatabase_ProduceCardDataUtils
from src.utils.i18n_tools import i18n_text
from src.utils.opencv_tools import check_frame_change
from src.utils.string_tools import string_match
from src.utils.ui_message_tools import UIMessage
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.main import AppProcessor

ocr_service = OCRService()
skill_card_database = GakumasDatabase_ProduceCardDataUtils()
message_tools = UIMessage()
debug_tools = DebugTools()

def __navigate_to_page(app: "AppProcessor") -> bool:
    """检查页面"""
    current_location = app.game_utils.update_current_location()
    while current_location != GamePageTypes.SUB_MENU.PRODUCER_ILLUSTRATED:
        logger.warning(f"Skill storage refresh requires P図鑑 page, current location: {current_location}")
        message_tools.info(i18n_text("backend.task.suspendSwitchToAlbum", fallback="任务已挂起，请手动切换到图鉴页面"), 30)
        app.task_queue.suspend_running_task()
        current_location = app.game_utils.update_current_location()

    if app.game_utils.wait_for_label(BaseUILabels.TAB_BAR):
        tabbar = TabBar(app.latest_results.filter_by_label(BaseUILabels.TAB_BAR).first())
        for tab_item in tabbar:
            if string_match(tab_item.text, GeneralText.SKILL_CARD):
                app.device.click_element(tab_item)
                return True

    message_tools.error(i18n_text("backend.task.tabBarNotFound", fallback="无法找到TabBar，刷新失败"))
    return False

def __validate_environment(app: "AppProcessor") -> bool:
    """验证环境"""
    if not app.game_utils.wait_for_label(BaseUILabels.BUTTON):
        message_tools.error(i18n_text("backend.task.requiredButtonNotFound", fallback="无法找到任务所需的按钮，刷新失败"))
        return False

    buttons = ButtonList(app.latest_results)
    enhance_btn = buttons.get_button_by_text(GeneralText.ENHANCE) or \
                  next((string_match(i.text, GeneralText.ENHANCE) for i in
                        [CheckBox(el) for el in app.latest_results.filter_by_label(BaseUILabels.CHECKBOX)]), None)

    idol_switch = buttons.get_button_by_text("切り替え")

    if not enhance_btn:
        message_tools.error(i18n_text("backend.task.cardAttributeSwitchNotFound", fallback="无法找到卡属性切换按钮，刷新失败"))
        return False
    if not idol_switch:
        message_tools.error(i18n_text("backend.task.idolSwitchNotFound", fallback="无法找到偶像切换按钮，刷新失败"))
        return False
    return True

def __process_single_card(app: "AppProcessor", skill_card, info_box: Yolo_Box):
    """处理单张卡片的核心逻辑"""
    debug_tools.add_box(skill_card.x, skill_card.y, skill_card.w, skill_card.h)
    app.device.click_element(skill_card)
    sleep(1)
    app.game_utils.wait_frame_stable(stable_count=2)

    # 裁剪图像
    y, h, x, w = info_box.y, info_box.h, info_box.x, info_box.w
    # 确保坐标不越界
    card_img = app.latest_frame[y:h, x:w]
    info_img = app.latest_frame[max(0, int(y-10)):h, w:]

    # 查重
    if app.clip_manager.skill_card_clip.retrieve(card_img) is not None:
        return

    # OCR 识别
    ocr_result = ocr_service.ocr(info_img)
    if not ocr_result:
        logger.warning("No valid skill card information.")
        return

    ocr_result.auto_merge_lines(width_gap=100)
    card_title = ocr_result.get_y_min().text
    logger.debug(f"Recognized title: {card_title}")

    # 数据库匹配与入库
    status, db_result = skill_card_database.search(card_title)
    if status:
        logger.debug(f"DB Match: {db_result}")
        app.clip_manager.skill_card_clip.add_to_memory(card_img, db_result)
        sleep(2)
    else:
        logger.warning("No search skill card info form game database.")

def __get_swipe_range(card_list, width) -> Tuple[int, int]:
    """计算滑动区域"""
    start = card_list.get_y_max_element().get_COL()[1]
    end = card_list.get_y_min_element().get_COL()[1]
    logger.debug(f"swipe_start={start}, swipe_end={end}")
    debug_tools.add_line(0, start, width, start, color=(0,255,0))
    debug_tools.add_line(0, end, width, end, color=(0,255,255))
    return start, end

def refresh_skill_storage(app: "AppProcessor"):
    """主任务入口函数"""
    if not __navigate_to_page(app):
        return False
    if not __validate_environment(app): return False

    os.makedirs(DebugPath.NoValidSkillCardInfo, exist_ok=True)

    # 状态变量
    width, _ = app.device.get_window_size()
    info_box: Optional[Yolo_Box] = None
    prev_page: Optional[np.ndarray] = None
    swipe_start, swipe_end = 0, 0

    while True:
        # 获取并筛选元素
        raw_cards = app.latest_results.filter_by_labels([
            BaseUILabels.SKILL_CARD, BaseUILabels.SKILL_CARD_ACTIVE,
            BaseUILabels.SKILL_CARD_MENTAL, BaseUILabels.SKILL_CARD_TRAP
        ])
        # 排除最上方的干扰项
        skill_card_list = raw_cards.remove_by_yolo_results(raw_cards.get_y_min_element())

        # 初始化布局参数（仅首次）
        if info_box is None:
            info_box = raw_cards.get_y_min_element()

        if not swipe_start:
            swipe_start, swipe_end = __get_swipe_range(skill_card_list, width)

        # 遍历处理
        for skill_card in skill_card_list:
            __process_single_card(app, skill_card, info_box)

        # 翻页逻辑
        if isinstance(app.device, Android_App):
            app.device.swipe(width // 2, swipe_start, width // 2, swipe_end, offset_y=0)

        sleep(1)
        app.game_utils.wait_frame_stable()
        debug_tools.clear_all()

        # 到底检测
        if prev_page is not None and check_frame_change(prev_page, app.latest_frame):
            break
        prev_page = app.latest_frame

    return True
