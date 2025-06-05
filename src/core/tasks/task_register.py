import cv2
import numpy as np

from src.core.tasks.base_ui.auto_contest import action__enter_contest_page, action__check_and_collect_rewards, \
    action__loop_challenge_contest
from src.core.tasks.base_ui.dispatch_work import action__enter_dispatch_page, action__dispatch_all_available_work
from src.core.tasks.base_ui.get_gift import action__enter_gift_page, action__has_gift_items, action__collect_all_gifts
from src.core.tasks.base_ui.start_game import (
    action__click_start_game,
    handle__network_error_modal_boxes,
    action__check_home_tab_exist
)
from src.entity.Game.Components.Button import Button, ButtonList
from src.entity.Game.Components.CheckBox import CheckBox
from src.entity.Game.Components.Contest import ContestList
from src.entity.Game.Components.TabBar import TabBar
from src.entity.Yolo import Yolo_Box
from src.utils.game_tools import get_current_location
from time import sleep
from src.entity.Game.Page.Types.index import GamePageTypes
from src.constants import *
from src.utils.logger import logger
from typing import TYPE_CHECKING

from src.utils.ocr_instance import get_ocr
from src.utils.opencv_tools import check_color_in_region
from src.utils.yolo_tools import get_modal

if TYPE_CHECKING:
    from app import AppProcessor

def register_tasks(processor: "AppProcessor"):
    @processor.register_task("start_game", "启动游戏", 60)
    @logger.catch
    def _task__start_game(app: "AppProcessor"):
        TIMEOUT = 30
        if not (app.game_status_manager.current_location == GamePageTypes.START_GAME or app.latest_results.exists_label(base_labels.start_menu_logo)):
            return
        if action__click_start_game(app, TIMEOUT) is not False:
            sleep(2)
            app.wait__loading()
            handle__network_error_modal_boxes(app)
        action__check_home_tab_exist(app)
        app.update_current_location()

    @processor.register_task("get_expenditure", "获取活动费", 30)
    @logger.catch
    def _task__get_expenditure(app: "AppProcessor"):
        app.go_home()
        app.wait__loading()
        if not app.wait_for_label(base_labels.home_get_expenditure):
            raise TimeoutError("Timeout waiting for [home:expenditure] to appear.")
        app.app.click_element(app.latest_results.filter_by_label(base_labels.home_get_expenditure).first())
        sleep(3)
        if modal := app.wait_for_modal("活動費", no_body=True, timeout=10):
            print(modal)
            app.app.click_element(modal.cancel_button)
            sleep(3)
            return True
        elif app.latest_results.exists_label(base_labels.tab_home):
            logger.warning("There are no claimable expenses")
            return True
        raise TimeoutError("Timeout waiting for modal to appear.")

    @processor.register_task("dispatch_work", "派遣任务", 30)
    @logger.catch
    def _task__dispatch_work(app: "AppProcessor"):
        app.go_home()
        app.wait__loading()
        action__enter_dispatch_page(app)
        action__dispatch_all_available_work(app)

    @processor.register_task("get_gift", "获取礼物/邮箱")
    @logger.catch
    def _task__get_gift(app: "AppProcessor"):
        app.go_home()
        app.wait__loading()
        action__enter_gift_page(app)
        if action__has_gift_items(app):
            action__collect_all_gifts(app)

    @processor.register_task("auto_contest", "自动每日竞技场")
    @logger.catch
    def _task__auto_contest(app: "AppProcessor"):
        app.go_home()
        app.wait__loading()
        action__enter_contest_page(app)
        action__check_and_collect_rewards(app)
        action__loop_challenge_contest(app)

    @processor.register_task("claim_task_rewards", "领取任务奖励")
    @logger.catch
    def _task__claim_task_rewards(app: "AppProcessor"):
        app.go_home()
        app.wait__loading()
        app.app.click_element(app.latest_results.filter_by_label(base_labels.home_daily_task).first())
        app.wait_for_label(base_labels.tab_bar)
        tab_bar = TabBar(app.latest_results.filter_by_label(base_labels.tab_bar).first())
        logger.debug(tab_bar)
        height, width = app.latest_frame.shape[:2]
        frame_cx = width // 2
        for tab in tab_bar:
            app.app.click_element(tab)
            sleep(3)
            buttons = app.latest_results.filter_by_label(base_labels.button)
            flag = False
            for button in buttons:
                if frame_cx - 10 < button.cx < frame_cx + 10 and not Button(button).is_disabled():
                    app.app.click_element(button)
                    flag = True
                    break
            if flag:
                modal = app.wait_for_modal("受取完了", no_body=True, timeout=10)
                app.app.click_element(modal.cancel_button)
                app.click_on_label(base_labels.close_button)
                logger.info(f"The task reward of {tab.text} has been claimed")
            else:
                logger.info(f"{tab.text} has no task rewards to be claimed")
            sleep(1)

