import app

from time import sleep

from src.constants import *
from src.constants.base_ui import labels
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.yolo_tools import get_modal


def action__click_start_game(app: "app.AppProcessor", timeout=30):
    """动作：点击启动游戏"""
    if app.wait_for_label(labels.start_menu_click_continue_flag, timeout):
        if not app.click_on_label(labels.start_menu_click_continue_flag, timeout):
            raise TimeoutError("Failed to click on the continue flag within the timeout.")
    else:
        raise TimeoutError("Timeout waiting for continue flag in the start menu.")

def handle__network_error_modal_boxes(app: "app.AppProcessor"):
    """处理：通信错误模态框"""
    if app.latest_results.filter_by_label(labels.modal_header):
        modal = get_modal(app.latest_results, app.latest_frame)
        if modal.modal_title == modal_text.connection_error:
            if modal_text.ConnectionError_Body.Token_Fail in modal.modal_body:
                app.app.click_element(modal.cancel_button)
                action__click_start_game(app)
                sleep(2)
                app.wait__loading()
                handle__network_error_modal_boxes(app)
            if modal_text.ConnectionError_Body.Timeout in modal.modal_body:
                app.app.click_element(modal.confirm_button)
                app.wait__loading()
                handle__network_error_modal_boxes(app)

def action__check_home_tab_exist(app: "app.AppProcessor", timeout=30):
    """动作：检查主界面标识是否存在"""
    count = 0
    if app.latest_results.exists_label(labels.tab_home):
        return True
    while count < timeout:
        if close_btn := app.latest_results.filter_by_label(labels.close_button):
            app.app.click_element(close_btn.first())
            sleep(2)
        if skip_btn := app.latest_results.filter_by_label(labels.skip_button):
            app.app.click_element(skip_btn.first())
            sleep(2)
        elif app.latest_results.exists_label(labels.tab_home):
            return True
        else:
            height, width = app.latest_frame.shape[:2]
            app.app.click(width // 2, height // 2)
            count += 3
            sleep(3)
    raise TimeoutError("Timeout waiting for home tab exist in the timeout.")