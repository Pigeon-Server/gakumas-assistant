import app

from time import sleep
from src.constants import labels
from src.utils.yolo_tools import get_modal


def action__click_start_game(app: "app.AppProcessor", timeout=30):
    """动作：点击启动游戏"""
    if app.latest_results.get_yolo_boxs_by_label(labels.start_menu_logo):
        if app.wait_for_label(labels.start_menu_click_continue_flag, timeout):
            if not app.click_on_label(labels.start_menu_click_continue_flag, timeout):
                raise TimeoutError("Failed to click on the continue flag within the timeout.")
        else:
            raise TimeoutError("Timeout waiting for continue flag in the start menu.")
    return False

def handle__network_error_modal_boxes(app: "app.AppProcessor"):
    """处理：通信错误模态框"""
    if app.latest_results.get_yolo_boxs_by_label(labels.modal_header):
        modal = get_modal(app.latest_results.yolo_boxs, app.latest_frame)
        if modal.modal_title == "通信エラー":
            if "ERR_AM_IDTOKEN_FAIL" in modal.modal_body:
                app.app.click_element(modal.cancel_button)
                action__click_start_game(app)
                sleep(2)
                app.wait__loading()
                handle__network_error_modal_boxes(app)
            if "FBR_ERR_TIMEOUT":
                app.app.click_element(modal.confirm_button)
                app.wait__loading()
                handle__network_error_modal_boxes(app)

def action__check_home_tab_exist(app: "app.AppProcessor", timeout=30):
    """动作：检查主界面标识是否存在"""
    count = 0
    if app.latest_results.get_yolo_boxs_by_label(labels.tab_home):
        return True
    while count < timeout:
        if app.latest_results.get_yolo_boxs_by_label(labels.tab_home):
            return True
        else:
            height, width = app.latest_frame.shape[:2]
            app.app.click(int(height / 2), int(width / 2))
            count += 3
            sleep(3)
    raise TimeoutError("Timeout waiting for home tab exist in the timeout.")