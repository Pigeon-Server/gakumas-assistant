from time import sleep
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.device.Android.app import Android_App
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.logger import logger
from src.core.inference.ocr_engine import OCRService
from src.utils.opencv_tools import check_color_in_region, check_color, check_frame_change
from src.utils.string_tools import string_match, MatchConfig

if TYPE_CHECKING:
    from src.main import AppProcessor

MAX_WORKS = 2
FLAG__Reconfigure_work_hour = False
ocr_service = OCRService()

def handle__work_dispatch_results(app: "AppProcessor"):
    """处理任务派遣结果"""
    count = 0

    while count < MAX_WORKS + 2:
        if app.game_utils.update_current_location() == GamePageTypes.HOME_TAB.WORK:
            return
        modal = app.game_utils.wait_for_modal(None, timeout=3, interval=0.5, no_body=True)
        if modal:
            close_button = modal.cancel_button or modal.confirm_button
            if close_button is None:
                logger.warning(f"Dispatch result modal '{modal.modal_title}' has no actionable button.")
                sleep(1)
                continue
            app.device.click_element(close_button)
            count += 1
            sleep(3)
    else:
        raise RuntimeError("Too many attempts to claim daily dispatch task.")

def action__dispatch_all_available_work(app: "AppProcessor"):
    """
    派遣所有可派遣的任务
    :param app: app实例
    :return:
    """
    global FLAG__Reconfigure_work_hour
    height, width = app.latest_frame.shape[:2]
    item_group: Optional[Yolo_Results] = None
    for i in range(MAX_WORKS + 1):
        if i >= MAX_WORKS:
            raise RuntimeError("Too many attempts have been made to obtain the number of dispatched tasks.")
        item_group = app.latest_results.filter_by_label(BaseUILabels.ITEM).group_yolo_boxes_by_position(10, width // 4)
        if len(item_group) != MAX_WORKS:
            logger.warning("The number of dispatched tasks is incorrect")
            sleep(1)
            continue
        break
    FLAG__Reconfigure_work_hour = False
    for group in item_group:
        # 跳过已派遣的组
        if _is_work_already_dispatched(app, group, width):
            continue
        app.device.click_element(group)
        dispatched = _dispatch_single_work(app)
        if not dispatched:
            logger.warning("dispatch_work: 当前派遣组未完成派遣，已跳过并继续后续流程。")
        sleep(3)
        app.game_utils.wait_for_label(BaseUILabels.ITEM, 5)


def _dismiss_connection_error_modal_if_present(app: "AppProcessor") -> bool:
    """若出现通信错误弹窗，点击重试并等待过场。"""
    modal = app.game_utils.try_get_modal(no_body=True)
    if modal is None:
        return False
    if not string_match(modal.modal_title, [ModalText.TITLE.CONNECTION_ERROR, ModalText.TITLE.INFO_FETCH_FAILED]):
        return False
    action_button = modal.confirm_button or modal.cancel_button
    if action_button is None:
        logger.warning("dispatch_work: connection modal has no actionable button.")
        return False
    logger.warning(f"dispatch_work: 检测到网络错误弹窗 '{modal.modal_title}'，点击重试。")
    app.device.click_element(action_button)
    sleep(1)
    app.game_utils.wait_loading()
    return True

def _is_work_already_dispatched(app: "AppProcessor", group, width):
    """判断该任务是否已派遣"""
    return group.get_vertical_range_elements(app.latest_results, width / 4).exists_label(BaseUILabels.AVATAR)

def _is_avatar_guaranteed_success(avatar):
    """判断角色是否带有标志“好調：大成功確定”"""
    height, width = avatar.frame.shape[:2]
    region = (width // 2, 0, width, height // 4)
    result = check_color_in_region(avatar.frame, (98,217,240), (100,255,255), region, 20)
    logger.debug(result)
    return result

def _assign_avatar_to_work(app: "AppProcessor", avatar: Yolo_Box = None):
    """
    选中角色并点击时长按钮
    :param app: app实例
    :param avatar: 要选择的头像（可选）
    :return:
    """
    if avatar:  # 当有头像元素时
        app.device.click_element(avatar)
        sleep(0.5)
    app.device.click_element(app.latest_results.filter_by_label(BaseUILabels.BUTTON).get_y_max_element().first())
    app.debug_tools.hide()
    sleep(1)
    while True:
        modal = app.game_utils.try_get_modal()
        exists_modal = modal is not None
        if not app.latest_results.exists_label(BaseUILabels.AVATAR) and not exists_modal:
            break
        if modal:
            if string_match(modal.modal_title, ModalText.TITLE.CONFIRM) and string_match(modal.modal_body_text, ModalText.BODY.DISPATCH_WORK_ERROR.OTHER_SELECTABLE_IDOLS):
                close_button = modal.cancel_button or modal.confirm_button
                if close_button is None:
                    logger.warning("Dispatch confirmation modal has no actionable button.")
                    return False
                app.device.click_element(close_button)
                sleep(0.5)
                return False
        sleep(0.1)
    app.game_utils.wait_for_label(BaseUILabels.BUTTON)
    _select_work_duration(app)
    sleep(1)
    app.device.click_element(app.latest_results.filter_by_label(BaseUILabels.BUTTON).get_y_max_element().first())
    sleep(1)
    modal = app.game_utils.wait_for_modal(ModalText.TITLE.WORK_START_CONFIRMATION, 10, no_body=True)
    app.device.click_element(modal.confirm_button)
    sleep(1)
    return True

def _select_work_duration(app: "AppProcessor"):
    """选择工作时长"""
    global FLAG__Reconfigure_work_hour
    if FLAG__Reconfigure_work_hour or not app.config_service().task__dispatch_work.reconfigure_work_hours.value:
        return
    frame_h, frame_w = app.latest_frame.shape[:2]
    y_start = frame_h // 2
    y_end = int(app.latest_results.filter_by_label(BaseUILabels.BUTTON).get_y_max_element().first().y)
    y_end = min(frame_h, max(y_start + 1, y_end))
    frame = app.latest_frame[y_start:y_end, 0:frame_w]

    ocr_results = ocr_service.ocr(frame)
    selects = {
        "4H": ButtonText.WORK.TIME.TIME_4H,
        "8H": ButtonText.WORK.TIME.TIME_8H,
        "12H": ButtonText.WORK.TIME.TIME_12H
    }

    candidates = [
        Yolo_Box(
            x := o.x, y := y_start + o.y, w := x + o.w, h := y + o.h,
            f"button__{o.text}", app.latest_frame[y:h, x:w]
        )
        for o in ocr_results if o.text in selects.values()
    ]

    # 根据配置选择目标按钮
    target_text = selects.get(app.config_service().task__dispatch_work.working_hours.value)
    app.device.click_element(
        next(
            (c for c in candidates if string_match(c.label, f"button__{target_text}", MatchConfig(fuzz_threshold=95))),
            candidates[-1]
        )
    )
    FLAG__Reconfigure_work_hour = True

def _dispatch_single_work(app: "AppProcessor") -> bool:
    """
    派遣单个任务
    :param app:
    :return:
    """
    app.game_utils.wait_loading()
    if not app.game_utils.wait_for_label(BaseUILabels.AVATAR):
        recovered = _dismiss_connection_error_modal_if_present(app)
        if recovered and app.game_utils.wait_for_label(BaseUILabels.AVATAR, timeout=8):
            logger.warning("dispatch_work: 网络弹窗恢复后继续派遣流程。")
        else:
            logger.warning("dispatch_work: 未找到可选头像，跳过当前派遣组。")
            return False
    rejected_avatars: set[Yolo_Box] = set()

    def _get_dispatch_avatars() -> Yolo_Results:
        """返回当前页可参与派遣选择的头像集合。"""
        avatars = app.latest_results.filter_by_label(BaseUILabels.AVATAR)
        return Yolo_Results.from_boxes([avatar for avatar in avatars if avatar.x >= 10])

    def _pick_fallback_avatar() -> Optional[Yolo_Box]:
        """
        在当前页里挑一个未被拒绝、且未处于派遣中的兜底头像。
        """
        for avatar in _get_dispatch_avatars():
            if avatar in rejected_avatars:
                continue
            if check_color(avatar.frame, (0,5,75), (179,120,190), threshold=50):
                continue
            return avatar
        return None

    def __try_dispatch_work__():
        """
        派遣任务
        :return:
        """
        app.debug_tools.clear_all()
        avatars = _get_dispatch_avatars()
        for avatar in avatars:
            if avatar in rejected_avatars:
                app.debug_tools.add_box(avatar.x, avatar.y, avatar.w, avatar.h, label="跳过，已尝试", color=(0,165,255))
                continue
            # 跳过正在工作中的角色
            working = check_color(avatar.frame, (0,5,75), (179,120,190), threshold=50)
            logger.debug(working)
            if working:
                app.debug_tools.add_box(avatar.x, avatar.y, avatar.w, avatar.h, label=f"跳过，已派遣", color=(255,255,0))
                continue
            # 大成功確定
            avatar_height, avatar_width = avatar.frame.shape[:2]
            app.debug_tools.add_box(
                avatar.x + avatar_width // 2,
                avatar.y,
                avatar.w,
                avatar.y + avatar_height // 4,
            )
            if _is_avatar_guaranteed_success(avatar):
                app.debug_tools.add_box(avatar.x, avatar.y, avatar.w, avatar.h, label="大成功确定", color=(0,255,0))
                if not _assign_avatar_to_work(app, avatar):
                    rejected_avatars.add(avatar)
                    continue
                app.debug_tools.clear_all()
                return True
            app.debug_tools.add_box(avatar.x, avatar.y, avatar.w, avatar.h, label="非优选")
        app.debug_tools.clear_all()
        return False
    avatar_group = _get_dispatch_avatars()
    if not avatar_group:
        logger.warning("dispatch_work: 头像列表为空，无法继续当前派遣组。")
        app.debug_tools.clear_all()
        return False
    avatar_group_x, avatar_group_y = avatar_group.get_COL()
    prev_frame: Optional[np.ndarray] = None
    while True:
        if __try_dispatch_work__():
            return True
        if isinstance(app.device, Android_App):
            app.device.swipe(avatar_group.w, avatar_group_y, avatar_group.x, avatar_group_y, 1)
        else:
            app.device.scrollY(avatar_group_x, avatar_group_y, -10)
        app.game_utils.wait_frame_stable()
        if prev_frame is not None and not check_frame_change(prev_frame, app.latest_frame):
            rejected_avatars.clear()
        avatar_groups = app.latest_results.filter_by_label(BaseUILabels.AVATAR).group_yolo_boxes_by_position(None, avatar_group_x//6)
        if not avatar_groups:
            logger.warning("dispatch_work: 滚动后未检测到头像分组。")
            break
        max_x_group = max(
            avatar_groups,
            key=lambda g: max(box.w for box in g.boxes),
        )
        if len(max_x_group) < 2:
            if __try_dispatch_work__():
                return True
            break

        if prev_frame is not None and check_frame_change(prev_frame, app.latest_frame):
            break

        prev_frame = app.latest_frame.copy()
    fallback_avatar = _pick_fallback_avatar()
    if fallback_avatar is None:
        logger.warning("No fallback avatar available after dispatch selection was rejected.")
        app.debug_tools.clear_all()
        return False
    assigned = _assign_avatar_to_work(app, fallback_avatar)
    app.debug_tools.clear_all()
    return bool(assigned)
