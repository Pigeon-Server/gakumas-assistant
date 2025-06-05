from time import sleep

from src.entity.Game.Components.Button import ButtonList
from src.entity.Game.Page.Types.index import GamePageTypes
from src.constants import *
from src.utils.logger import logger
from src.utils.yolo_tools import get_modal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app import AppProcessor

def action__enter_gift_page(app: "AppProcessor"):
    """
    进入首页的礼物界面，若按钮不存在则抛出超时异常。
    """
    if not app.wait_for_label(base_labels.home_gift_btn):
        raise TimeoutError("Timeout waiting for [home:gift] to appear.")
    app.app.click_element(app.latest_results.filter_by_label(base_labels.home_gift_btn).first())
    app.update_current_location(GamePageTypes.HOME_TAB.GIFT)
    sleep(3)

def action__has_gift_items(app: "AppProcessor") -> bool:
    """
    判断当前界面是否存在可领取的礼物项目。
    :return: True 表示有礼物，False 表示没有
    """
    return app.latest_results.exists_label(base_labels.item)

def action__collect_all_gifts(app: "AppProcessor"):
    """
    尝试点击“一括受取”按钮并处理弹窗确认。
    如果弹窗未出现则抛出超时异常。
    """
    ButtonList(app.latest_results).get_button_by_text("一括受取")
    app.app.click_element(app.latest_results.filter_by_label(base_labels.button).get_y_max_element().first())
    sleep(1)
    if app.wait_for_label(base_labels.modal_header, 10):
        modal = get_modal(app.latest_results)
        app.app.click_element(modal.cancel_button)
        sleep(1)
    else:
        raise TimeoutError("Timeout waiting for [modal:header] to appear.")
