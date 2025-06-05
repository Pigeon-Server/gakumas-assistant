from time import sleep

from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Components.CheckBox import CheckBox
from src.entity.Game.Components.Contest import ContestList
from src.entity.Game.Page.Types.index import GamePageTypes
from src.constants import *
from src.utils.logger import logger
from src.utils.yolo_tools import get_modal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app import AppProcessor

def action__enter_contest_page(app: "AppProcessor"):
    """
    进入竞技场页面流程。
    包括点击主界面竞技场 Tab 和内部按钮。
    """
    app.app.click_element(app.latest_results.filter_by_label(base_labels.tab_contest).first())
    app.update_current_location(GamePageTypes.MAIN_MENU__CONTEST)
    sleep(2)
    app.click_button("コンテスト")  # 点击进入竞技场功能
    sleep(3)
    app.update_current_location(GamePageTypes.CONTEST.ARENA)

def action__check_and_collect_rewards(app: "AppProcessor"):
    """
    检查并领取上赛季奖励。
    奖励出现时通常位于屏幕下半部，通过点击领取。
    """
    height, width = app.latest_frame.shape[:2]
    items = app.latest_results.filter_by_label(base_labels.item)
    items_cx, items_cy = items.get_COL()
    if items and (height // 2) < items_cy:
        app.app.click(items_cx, items_cy)
        sleep(2)
        app.app.click(items_cx, items_cy)
        logger.info("Last season's rewards have been claimed.")
        sleep(2)

def action__loop_challenge_contest(app: "AppProcessor"):
    """
    持续挑战竞技场，直到没有可挑战对象为止。
    """
    height, width = app.latest_frame.shape[:2]
    while True:
        contest = ContestList(app.latest_results, app.latest_frame)
        if not contest:
            logger.info("There is no contest.")
            break
        app.app.click_element(contest.get_combat_power_min())
        sleep(1)
        if app.latest_results.exists_label(base_labels.blank_slot):
            _auto_form_team(app)
        _start_battle_and_skip(app, width, height)
        _finish_battle(app)

def _auto_form_team(app: "AppProcessor"):
    """
    如果有空的编队槽位，执行自动编队。
    依次点击：编成 -> おまかせ -> 決定 -> 閉じる。
    """
    app.click_button("ユニッ卜編成")
    sleep(1)
    app.click_button("おまかせ")
    sleep(1)
    app.click_button("決定")
    sleep(0.5)
    app.click_button("閉じる")
    app.back_next_page()

def _start_battle_and_skip(app: "AppProcessor", width: int, height: int):
    """
    发起挑战并跳过战斗过程。
    若勾选框未启用，自动勾选“跳过”。
    重复点击直到跳过按钮消失。
    """
    app.click_button("挑戦開始")
    app.wait_for_label(base_labels.checkbox)
    check_box = CheckBox(app.latest_results.filter_by_label(base_labels.checkbox).first())
    if not check_box.checked:
        app.app.click_element(check_box)
    app.click_on_label(base_labels.skip_button)
    sleep(1)
    while app.latest_results.exists_label(base_labels.skip_button):
        app.app.click(width // 2, height // 2)
        sleep(1)
    app.app.click(width // 2, height // 2)

def _finish_battle(app: "AppProcessor"):
    """
    处理战斗结束的按钮点击与奖励弹窗。
    超时等待过程中持续点击屏幕中部。
    """
    COUNT, WAIT = 0, 15
    while COUNT < WAIT:
        if app.latest_results.exists_label(base_labels.button):
            break
        app.app.click(app.latest_frame.shape[1] // 2, app.latest_frame.shape[0] // 2)
        sleep(1)
        COUNT += 1
    if COUNT >= WAIT:
        raise TimeoutError("Waiting for the challenge to end timeout")
    app.click_button("次へ")
    app.click_button("終了")
    sleep(1)
    if app.latest_results.exists_label(base_labels.modal_header):
        modal = app.wait_for_modal("レート報酬", no_body=True)
        app.app.click_element(modal.cancel_button)
    app.wait__loading()