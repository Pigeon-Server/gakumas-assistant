from typing import Tuple

import cv2
import numpy as np

from src.constants import *
from src.entity.Game.Components.Button import Button
from src.entity.Game.Components.Modal import Modal
from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.ocr_instance import get_ocr
from src.utils.logger import logger

@logger.catch
def get_modal(yolo_result: Yolo_Results, frame: np.array, no_body: bool = False) -> Modal | None:
    """
    获取模态框
    :param no_body: 不识别模态框主体（加速）
    :param yolo_result: yolo 识别结果
    :param frame: 图像帧
    :return: 解析后的 Modal 对象
    """
    # try:
    # 获取模态框头部
    modal = yolo_result.filter_by_labels([base_labels.modal_header, base_labels.button])
    if not modal:
        raise ValueError("未找到模态框")
    modal_header = modal.filter_by_label(base_labels.modal_header).first()
    modal_header_text = get_ocr(modal_header.frame)[0].text
    # 获取确认和取消按钮
    buttons = modal.filter_by_label(base_labels.button).group_yolo_boxes_by_position(30, None)
    if buttons:
        buttons = buttons[0]
        confirm_button = buttons.get_x_max_element()
        cancel_button = buttons.get_x_min_element()
    else:
        # 不区分按钮类型时，取最下面的按钮作为取消按钮
        buttons = modal.filter_by_label(base_labels.button).get_y_min_element()
        confirm_button = None
        cancel_button = buttons
    if not confirm_button and not cancel_button:
        raise ValueError("未找到确认或取消按钮")
    confirm_button = Button(confirm_button.first()) if confirm_button else None
    cancel_button = Button(cancel_button.first()) if cancel_button else None
    # 计算模态框主体区域
    if confirm_button and cancel_button:
        modal_body_y = max(cancel_button.y, confirm_button.y)
    else:
        modal_body_y = confirm_button.y if confirm_button else cancel_button.y
    modal_body_frame = frame[modal_header.h:modal_body_y, modal_header.x:modal_header.w]
    modal_body_text = "" if no_body else " ".join([item.text for item in get_ocr(modal_body_frame)])
    modal_obj = Modal(modal_header_text, modal_body_text, confirm_button, cancel_button)
    # logger.debug(modal_obj)
    return modal_obj

@logger.catch
def check_status_detection(frame: np.array, threshold=0.15, upper_color: Tuple[int, int, int] = (22, 255, 255), lower_color: Tuple[int, int, int]=(8, 100, 100)):
    """
    选中状态检测：忽略白色背景，专注文本区域标记
    """
    if frame.size == 0:
        return False
    lower_color = np.array(lower_color)
    upper_color = np.array(upper_color)
    # 1. 预处理 - 转换为HSV和灰度
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. 创建白色背景掩码（忽略区域）
    white_mask = cv2.inRange(gray, 220, 255)  # 忽略亮色区域
    # 3. 创建橙色文本掩码（扩大检测范围）
    orange_mask = cv2.inRange(hsv, lower_color, upper_color)
    # 4. 组合掩码（只在非白色区域检测橙色）
    combined_mask = cv2.bitwise_and(orange_mask, cv2.bitwise_not(white_mask))
    # 5. 计算有效区域比例（排除白色后的区域）
    non_white_area = cv2.countNonZero(cv2.bitwise_not(white_mask))
    if non_white_area == 0:
        return False
    orange_ratio = cv2.countNonZero(combined_mask) / non_white_area
    return orange_ratio > threshold