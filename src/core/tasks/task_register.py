import cv2
import numpy as np

from src.core.tasks.base_ui.dispatch_work import action__enter_dispatch_page, action__dispatch_all_available_work
from src.core.tasks.base_ui.start_game import (
    action__click_start_game,
    handle__network_error_modal_boxes,
    action__check_home_tab_exist
)
from src.entity.Game.Components.Button import Button, ButtonList
from src.entity.Game.Components.CheckBox import CheckBox
from src.entity.Game.Components.Contest import ContestList
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
        if not app.wait_for_label(base_labels.home_gift_btn):
            raise TimeoutError("Timeout waiting for [home:gift] to appear.")
        app.app.click_element(app.latest_results.filter_by_label(base_labels.home_gift_btn).first())
        sleep(3)
        if not app.latest_results.exists_label(base_labels.item):
            logger.info("Not to collect gifts")
            return
        ButtonList(app.latest_results).get_button_by_text("一括受取")
        app.app.click_element(app.latest_results.filter_by_label(base_labels.button).get_y_max_element().first())
        sleep(1)
        if app.wait_for_label(base_labels.modal_header, 10):
            modal = get_modal(app.latest_results)
            app.app.click_element(modal.cancel_button)
            sleep(1)
        else:
            raise TimeoutError("Timeout waiting for [modal:header] to appear.")

    @processor.register_task("auto_contest", "自动每日竞技场")
    @logger.catch
    def _task__auto_contest(app: "AppProcessor"):
        app.go_home()
        app.wait__loading()
        app.app.click_element(app.latest_results.filter_by_label(base_labels.tab_contest).first())
        app.update_current_location(GamePageTypes.MAIN_MENU__CONTEST)
        sleep(2)
        # 等待符合文本条件的按钮出现
        app.click_button("コンテスト")
        sleep(3)
        app.update_current_location(GamePageTypes.CONTEST.ARENA)
        # 判断是否有上赛季奖励，并领取
        height, width = app.latest_frame.shape[:2]
        items = app.latest_results.filter_by_label(base_labels.item)
        items_cx, items_cy = items.get_COL()
        # 当物品在下半屏幕出现时，判断为奖励
        if items and (height//2) < items_cy:
            app.app.click(items_cx, items_cy)
            sleep(2)
            app.app.click(items_cx, items_cy)
            logger.info("Last season's rewards have been claimed.")
            sleep(2)
        while True:
            # 获取所有挑战者
            contest = ContestList(app.latest_results, app.latest_frame)
            # 没有挑战时自动停止
            if not contest:
                logger.info("There is no contest.")
                break
            # 挑战战力最低的
            app.app.click_element(contest.get_combat_power_min())
            sleep(1)
            # 如果有空插槽，自动重新编队
            if app.latest_results.exists_label(base_labels.blank_slot):
                app.click_button("ユニッ卜編成")
                sleep(1)
                app.click_button("おまかせ")
                sleep(1)
                app.click_button("決定")
                sleep(0.5)
                app.click_button("閉じる")
                app.back_next_page()
            # 开始挑战
            app.click_button("挑戦開始")
            # 检查跳过所有选项是否启用
            app.wait_for_label(base_labels.checkbox)
            check_box = CheckBox(app.latest_results.filter_by_label(base_labels.checkbox).first())
            if check_box.checked is False:
                app.app.click_element(check_box)
            # 跳过战斗
            app.click_on_label(base_labels.skip_button)
            sleep(1)
            # 检查skip按钮是否消失
            while True:
                if not app.latest_results.exists_label(base_labels.skip_button):
                    break
                app.app.click(width//2, height//2)
                sleep(1)
            app.app.click(width//2, height//2)
            COUNT = 0
            WAIT = 15
            # 连续点击，等待按钮出现
            while COUNT < WAIT:
                if app.latest_results.exists_label(base_labels.button):
                    break
                else:
                    app.app.click(width//2, height//2)
                    sleep(1)
                    COUNT += 1
            if COUNT > WAIT:
                raise TimeoutError("Waiting for the challenge to end timeout")
            app.click_button("次へ")
            app.click_button("終了")
            sleep(1)
            # 自动领取等级奖励
            if app.latest_results.exists_label(base_labels.modal_header):
                modal = app.wait_for_modal("レート報酬", no_body=True)
                app.app.click_element(modal.cancel_button)
            app.wait__loading()
    #
    # @processor.register_task("claim_task_rewards", "领取任务奖励")
    # @logger.catch
    # def _task__claim_task_rewards(app: AppProcessor):
    #     if tab_home := app.latest_results.filter_by_label(base_labels.tab_home):
    #         tab_home = tab_home[0]
    #         app.app.click_element(tab_home)
    #     else:
    #         raise RuntimeError("当前不在主页")
