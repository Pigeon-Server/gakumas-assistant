from time import monotonic, sleep
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.modal_text import ModalText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.device.Android.app import Android_App
from src.core.inference.ocr_engine import OCRService, OCR_Result, OCR_ResultList
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Game.Components.Button import Button, ButtonList
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.game_tools import get_modal
from src.utils.logger import logger
from src.utils.opencv_tools import check_color_in_region, check_color, check_frame_change
from src.utils.string_tools import string_match, MatchConfig

if TYPE_CHECKING:
    from src.main import AppProcessor

MAX_WORKS = 2
FLAG__Reconfigure_work_hour = False
ocr_service = OCRService()
_WORK_SELECT_BUTTON_TEXT = "選択する"
_WORK_ACTION_BUTTON_MATCH = MatchConfig(fuzz_threshold=60, use_contains=True, normalize=True)

def handle__work_dispatch_results(app: "AppProcessor"):
    """处理派遣入口前置的结果弹窗，并确保最终停留在派遣页。"""
    count = 0

    while count < MAX_WORKS + 2:
        current_location = app.game_utils.update_current_location()
        if current_location == GamePageTypes.HOME_TAB.WORK:
            return
        if current_location == GamePageTypes.MAIN_MENU__HOME and not app.latest_results.exists_label(BaseUILabels.MODAL_HEADER):
            return

        modal = app.game_utils.wait_for_modal(None, timeout=3, interval=0.5, no_body=True)
        if modal is None:
            if app.latest_results.exists_label(BaseUILabels.MODAL_HEADER):
                modal = get_modal(app.latest_results, True, quiet=True)
            if modal is None:
                return

        close_button = modal.cancel_button or modal.confirm_button
        if close_button is None:
            logger.warning(f"Dispatch result modal '{modal.modal_title}' has no actionable button.")
            sleep(1)
            continue
        app.device.click_element(close_button)
        count += 1
        sleep(1.5)

    raise RuntimeError("Too many attempts to claim daily dispatch task.")


def ensure__work_dispatch_page_ready(app: "AppProcessor"):
    """若派遣结果弹窗关闭后回到主页，则重新进入派遣页。"""
    current_location = app.game_utils.update_current_location()
    if current_location == GamePageTypes.HOME_TAB.WORK:
        return
    if current_location == GamePageTypes.MAIN_MENU__HOME and app.latest_results.exists_label(BaseUILabels.HOME_DISPATCH_WORK):
        logger.info("dispatch_work: 派遣结果弹窗关闭后回到主页，重新进入派遣页。")
        app.game_utils.click_on_label(BaseUILabels.HOME_DISPATCH_WORK)
        app.game_utils.wait_loading()
        app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.WORK)
        return
    raise RuntimeError(f"dispatch_work: 处理派遣结果后未停留在派遣页，当前位置={current_location}")



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


def _matches_work_action_button_text(text: str | None, expected_texts: tuple[str, ...]) -> bool:
    """判断 OCR/按钮文本是否匹配派遣页底部操作按钮候选文案。"""
    if not text or not expected_texts:
        return False
    return any(string_match(text, expected, _WORK_ACTION_BUTTON_MATCH) for expected in expected_texts)


def _get_yolo_work_action_button(
    app: "AppProcessor",
    expected_texts: tuple[str, ...],
) -> Optional[Button]:
    """优先从 YOLO 的 BUTTON 检测结果中挑出底部主操作按钮。"""
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return None

    frame_h, frame_w = frame.shape[:2]
    min_width = int(frame_w * 0.18)
    bottom_threshold = int(frame_h * 0.72)
    center_tolerance = int(frame_w * 0.24)
    matched: list[Button] = []
    candidates: list[Button] = []

    for button in ButtonList(app.latest_results):
        if button is None or button.is_disabled():
            continue
        button_width = int(button.w - button.x)
        if button_width < min_width:
            continue
        if int(button.cy) < bottom_threshold:
            continue
        if abs(int(button.cx) - frame_w // 2) > center_tolerance:
            continue
        if _matches_work_action_button_text(button.text, expected_texts):
            matched.append(button)
        candidates.append(button)

    if matched:
        return max(matched, key=lambda item: (int(item.cy), int(item.w - item.x)))
    if candidates:
        return max(candidates, key=lambda item: (int(item.cy), int(item.w - item.x)))
    return None


def _collect_work_action_ocr_candidates(
    app: "AppProcessor",
) -> tuple[list[OCR_Result], int]:
    """收集底部区域 OCR 候选文本，并返回相对搜索区域的结果与顶部偏移量。"""
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return [], 0

    frame_h, frame_w = frame.shape[:2]
    search_top = int(frame_h * 0.68)
    search_frame = frame[search_top:frame_h, :]
    if search_frame.size == 0:
        return [], search_top

    gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    # BUTTON 漏检时，分别用原图/二值图/自适应阈值图做 OCR，尽量覆盖白字按钮和压缩噪声。
    ocr_frames: list[tuple[np.ndarray, float]] = [
        (search_frame, 1.0),
        (cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 1.0),
        (cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR), 1.0),
    ]

    deduped: list[OCR_Result] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for image, scale in ocr_frames:
        results = ocr_service.ocr(image)
        for item in results:
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            mapped = OCR_Result(
                x=int(round(item.x / scale)),
                y=int(round(item.y / scale)),
                w=int(round(item.w / scale)),
                h=int(round(item.h / scale)),
                text=text,
                confidence=getattr(item, "confidence", None),
            )
            signature = (
                mapped.text,
                mapped.x // 8,
                mapped.y // 8,
                max(1, mapped.w) // 8,
                max(1, mapped.h) // 8,
            )
            # 多路 OCR 结果位置通常只差几个像素；这里按粗粒度位置去重，避免重复候选放大权重。
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(mapped)

    merged = OCR_ResultList(deduped).auto_merge_lines(
        cy_range=max(4, int(frame_h * 0.004)),
        width_gap=max(12, int(frame_w * 0.02)),
    ) if deduped else OCR_ResultList([])
    combined = deduped + list(merged)
    return combined, search_top


def _find_work_action_button_from_ocr(
    app: "AppProcessor",
    expected_texts: tuple[str, ...],
) -> Optional[Yolo_Box]:
    """在 BUTTON 漏检时，通过底部 OCR 文本位置反推出主操作按钮点击区域。"""
    frame = getattr(app, "latest_frame", None)
    if frame is None or frame.size == 0:
        return None

    frame_h, frame_w = frame.shape[:2]
    bottom_threshold = int(frame_h * 0.72)
    center_tolerance = int(frame_w * 0.25)
    matched: list[tuple[Yolo_Box, float]] = []
    ocr_candidates, search_top = _collect_work_action_ocr_candidates(app)

    for item in ocr_candidates:
        text = str(getattr(item, "text", "") or "").strip()
        if not _matches_work_action_button_text(text, expected_texts):
            continue
        width = max(1, int(item.w))
        height = max(1, int(item.h))
        x1 = int(item.x)
        y1 = int(search_top + item.y)
        cx = x1 + width // 2
        cy = y1 + height // 2
        if cy < bottom_threshold:
            continue
        if abs(cx - frame_w // 2) > center_tolerance:
            continue

        # OCR 只能给出文本框，实际点击需要扩成整颗底部主按钮的可点击区域。
        pad_x = max(int(width * 0.9), int(frame_w * 0.05))
        pad_y = max(int(height * 1.6), int(frame_h * 0.02))
        box_x1 = max(0, x1 - pad_x)
        box_y1 = max(0, y1 - pad_y)
        box_x2 = min(frame_w, x1 + width + pad_x)
        box_y2 = min(frame_h, y1 + height + pad_y)
        box = Yolo_Box(
            box_x1,
            box_y1,
            box_x2,
            box_y2,
            BaseUILabels.BUTTON,
            frame[box_y1:box_y2, box_x1:box_x2].copy(),
        )
        match_bonus = 1.0 if any(text == expected for expected in expected_texts) else 0.0
        score = float(cy) + float(width) + match_bonus
        matched.append((box, score))

    if not matched:
        return None
    return max(matched, key=lambda item: item[1])[0]


def _get_work_action_button(
    app: "AppProcessor",
    expected_texts: tuple[str, ...],
    *,
    timeout: float = 2.5,
    interval: float = 0.3,
) -> Optional[Yolo_Box]:
    """统一获取派遣页底部主按钮，先查 YOLO，失败后退回 OCR。"""
    deadline = monotonic() + max(0.0, timeout)
    while True:
        button = _get_yolo_work_action_button(app, expected_texts)
        if button is None:
            button = _find_work_action_button_from_ocr(app, expected_texts)
        if button is not None:
            return button
        if monotonic() >= deadline:
            return None
        sleep(interval)

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
    action_button = _get_work_action_button(app, (_WORK_SELECT_BUTTON_TEXT, ButtonText.CONFIRM))
    if action_button is None:
        logger.warning("dispatch_work: 未定位到头像选择页底部操作按钮，跳过当前角色。")
        return False
    app.device.click_element(action_button)
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
    selected_duration = _select_work_duration(app)
    if not selected_duration:
        logger.warning("dispatch_work: 未识别到工作时长按钮，继续使用当前默认时长。")
    sleep(1)
    action_button = _get_work_action_button(app, (ButtonText.CONFIRM, _WORK_SELECT_BUTTON_TEXT))
    if action_button is None:
        logger.warning("dispatch_work: 未定位到派遣确认按钮，跳过当前角色。")
        return False
    app.device.click_element(action_button)
    sleep(1)
    modal = app.game_utils.wait_for_modal(ModalText.TITLE.WORK_START_CONFIRMATION, 10, no_body=True)
    if modal is None:
        recovered = _dismiss_connection_error_modal_if_present(app)
        if recovered:
            modal = app.game_utils.wait_for_modal(ModalText.TITLE.WORK_START_CONFIRMATION, 5, no_body=True)
    if modal is None:
        logger.warning("dispatch_work: 未等待到工作开始确认弹窗。")
        return False
    confirm_button = modal.confirm_button or modal.cancel_button
    if confirm_button is None:
        logger.warning("dispatch_work: 工作开始确认弹窗缺少可点击按钮。")
        return False
    app.device.click_element(confirm_button)
    sleep(1)
    return True

def _select_work_duration(app: "AppProcessor"):
    """选择工作时长"""
    global FLAG__Reconfigure_work_hour
    if FLAG__Reconfigure_work_hour or not app.config_service().task__dispatch_work.reconfigure_work_hours.value:
        return
    frame_h, frame_w = app.latest_frame.shape[:2]
    y_start = frame_h // 2
    action_button = _get_work_action_button(app, (ButtonText.CONFIRM, _WORK_SELECT_BUTTON_TEXT), timeout=1.5)
    y_end = int(action_button.y) if action_button is not None else int(frame_h * 0.86)
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
    if not candidates:
        logger.warning("dispatch_work: 当前页面未识别到任何工作时长按钮。")
        return False

    # 根据配置选择目标按钮
    target_text = selects.get(app.config_service().task__dispatch_work.working_hours.value)
    target_button = next(
        (c for c in candidates if string_match(c.label, f"button__{target_text}", MatchConfig(fuzz_threshold=95))),
        candidates[-1]
    )
    app.device.click_element(target_button)
    FLAG__Reconfigure_work_hour = True
    return True

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
