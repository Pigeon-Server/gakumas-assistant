from time import sleep
from typing import TYPE_CHECKING

import cv2

from src.constants import *
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.logger import logger
from src.utils.ocr_instance import get_ocr
from src.utils.opencv_tools import check_color_in_region
from src.utils.yolo_tools import get_modal

if TYPE_CHECKING:
    from app import AppProcessor

MAX_WORKS = 2

def action__enter_dispatch_page(app: "AppProcessor"):
    """进入页面并收取历史派遣结果逻辑"""
    if not app.wait_for_label(base_labels.home_dispatch_work):
        raise TimeoutError("Timeout waiting for [home:dispatch work] to appear.")
    app.app.click_element(app.latest_results.filter_by_label(base_labels.home_dispatch_work).first())
    sleep(1)
    app.wait__loading()

    count = 0

    while count < MAX_WORKS:
        app.update_current_location()
        if app.game_status_manager.current_location == GamePageTypes.HOME_TAB.WORK:
            return
        if app.wait_for_label(base_labels.modal_header, 3):
            modal = get_modal(app.latest_results, app.latest_frame, True)
            app.app.click_element(modal.cancel_button)
            count += 1
            sleep(3)
    else:
        raise RuntimeError("Too many attempts to claim daily dispatch task.")

def action__dispatch_all_available_work(app: "AppProcessor"):
    """派遣任务逻辑"""
    height, width = app.latest_frame.shape[:2]
    item_group = app.latest_results.filter_by_label(base_labels.item).group_yolo_boxes_by_position(10, width / 4)

    if len(item_group) != MAX_WORKS:
        raise RuntimeError("Error in calculating the range of the box body")

    for group in item_group:
        if _is_work_already_dispatched(app, group, width):
            continue
        _dispatch_single_work(app, group)
        sleep(3)
        app.wait_for_label(base_labels.avatar, 10)

def _is_work_already_dispatched(app: "AppProcessor", group, width):
    """判断该任务是否已派遣"""
    return group.get_vertical_range_elements(app.latest_results, width / 4).exists_label(base_labels.avatar)

def _is_avatar_busy(avatar, full_frame):
    """判断角色是否“工作中”"""
    h, w = full_frame.shape[:2]

    x1 = max(0, avatar.x - 10)
    y1 = max(0, avatar.y - 10)
    x2 = min(w, avatar.w)
    y2 = min(h, avatar.cy)

    target_frame = full_frame[y1:y2, x1:x2]

    ocr_result = get_ocr(target_frame)
    return "お仕事中" in [ocr_obj.text for ocr_obj in ocr_result]

def _is_avatar_guaranteed_success(avatar):
    """判断角色是否带有标志“好調：大成功確定”"""
    height, width = avatar.frame.shape[:2]
    region = (width / 2, 0, width, height / 2)
    rgb_lower = (96, 183, 238)
    rgb_upper = (89, 186, 260)
    return check_color_in_region(avatar.frame, region, rgb_lower, rgb_upper, 20)

def _assign_avatar_to_work(app: "AppProcessor", avatar=None):
    """选中角色并点击时长按钮"""
    if avatar: app.app.click_element(avatar)
    sleep(0.5)
    app.app.click_element(app.latest_results.filter_by_label(base_labels.button).get_y_min_element().first())
    sleep(1)
    app.wait_for_label(base_labels.button)

    duration_box = _select_work_duration(app)
    app.app.click_element(duration_box)
    sleep(1)
    app.app.click_element(app.latest_results.filter_by_label(base_labels.button).get_y_min_element().first())
    sleep(1)

    modal = app.wait_for_modal("お仕事開始確認", 10, no_body=True)
    app.app.click_element(modal.confirm_button)
    sleep(1)

def _select_work_duration(app: "AppProcessor"):
    """选择工作时长"""
    frame_h, frame_w = app.latest_frame.shape[:2]
    y_start = frame_h // 2
    y_end = int(app.latest_results.filter_by_label(base_labels.button).get_y_min_element().first().y)
    y_end = min(frame_h, max(y_start + 1, y_end))
    frame = app.latest_frame[y_start:y_end, 0:frame_w]

    ocr_results = get_ocr(frame)
    selects = ["4時間", "8時間", "12時間"]
    candidates = [
        Yolo_Box(
            x := o.x, y := y_start + o.y, w := x + o.w, h := y + o.h,
            f"button__{o.text}", app.latest_frame[y:h, x:w]
        )
        for o in ocr_results if o.text in selects
    ]
    return max(candidates, key=lambda box: selects.index(box.label.replace("button__", "")))

def _dispatch_single_work(app: "AppProcessor", group):
    """派遣单个任务"""
    app.app.click_element(group)
    sleep(1)
    def _exec():
        avatars = app.latest_results.filter_by_label(base_labels.avatar)
        avatars = Yolo_Results.from_boxes([avatar for avatar in avatars if avatar.x >= 10])
        for avatar in avatars:
            sleep(0.5)
            cv2.imshow("avatar", avatar.frame)
            cv2.waitKey(10)
            if _is_avatar_busy(avatar, app.latest_frame):
                logger.debug("Skip 'お仕事中' avatar")
                continue
            if _is_avatar_guaranteed_success(avatar):
                _assign_avatar_to_work(app, avatar)
                return True
        return False
    if not _exec():
        avatars = app.latest_results.filter_by_label(base_labels.avatar)
        x, y = avatars.get_COL()
        app.app.scrollY(x, y, -10)
        sleep(1)
        _exec()
        _assign_avatar_to_work(app)

