import numpy as np

import src.constants as constants
from src.entity.Button import Button
from src.entity.Modal import Modal
from src.entity.Yolo_Box import Yolo_Box
from src.utils.ocr_instance import get_ocr
from src.utils.logger import logger


def find_element_in_same_row(list1: list[Yolo_Box], list2: list[Yolo_Box], tolerance_ratio: float = 0.5) -> list[
    tuple[Yolo_Box, Yolo_Box]]:
    """
    查找在同一行的元素
    :param list1:
    :param list2:
    :param tolerance_ratio:
    :return:
    """
    same_row_pairs = []

    for box1 in list1:
        for box2 in list2:
            # 计算允许的误差阈值（基于较小的按钮高度）
            tolerance = min(box1.h, box2.h) * tolerance_ratio

            # 判断两个按钮是否在同一行
            if abs(box1.y - box2.y) <= tolerance:
                same_row_pairs.append((box1, box2))

    return same_row_pairs

def get_element(yolo_result: list[Yolo_Box] | tuple[Yolo_Box, Yolo_Box], label: str):
    """获取带有相关标签的元素"""
    return [item for item in yolo_result if item.label == label]

def find_element(yolo_result: list[Yolo_Box] | tuple[Yolo_Box, Yolo_Box], label: str):
    """查找带有相关标签的元素"""
    return any(b.label == label for b in yolo_result)

def get_modal(yolo_result: list[Yolo_Box], frame: np.array) -> Modal | None:
    """
    获取模态框
    :param yolo_result: yolo 识别结果
    :param frame: 图像帧
    :return: 解析后的 Modal 对象
    """
    try:
        # 获取模态框头部
        modal_headers = get_element(yolo_result, constants.labels.modal_header)
        if not modal_headers:
            raise ValueError("未找到模态框头部")
        modal_header = modal_headers[0]
        modal_header_text = get_ocr(modal_header.frame)[0].text

        # 获取确认和取消按钮
        confirm_buttons = get_element(yolo_result, constants.labels.confirm_button)
        cancel_buttons = get_element(yolo_result, constants.labels.cancel_button)
        if not confirm_buttons or not cancel_buttons:
            raise ValueError("未找到确认或取消按钮")

        # 识别出在同一行的按钮
        result = find_element_in_same_row(confirm_buttons, cancel_buttons)
        if not result:
            raise ValueError("确认和取消按钮未匹配成功")

        confirm_button_box = get_element(result[0], constants.labels.confirm_button)[0]
        cancel_button_box = get_element(result[0], constants.labels.cancel_button)[0]

        # 解析按钮文本
        confirm_button_text = "".join([item.text for item in get_ocr(confirm_button_box.frame)])
        cancel_button_text = "".join([item.text for item in get_ocr(cancel_button_box.frame)])

        confirm_button = Button(confirm_button_box.x, confirm_button_box.y, confirm_button_box.w, confirm_button_box.h, confirm_button_text)
        cancel_button = Button(cancel_button_box.x, cancel_button_box.y, cancel_button_box.w, cancel_button_box.h, cancel_button_text)

        # 计算模态框主体区域
        modal_body_y = max(cancel_button.y, confirm_button.y)
        modal_body_frame = frame[modal_header.h:modal_body_y, modal_header.x:modal_header.w]
        modal_body_text = " ".join([item.text for item in get_ocr(modal_body_frame)])

        modal_obj = Modal(modal_header_text, modal_body_text, confirm_button, cancel_button)
        logger.debug(modal_obj)
        return modal_obj

    except Exception as e:
        logger.error(f"解析模态框失败: {e}")
        return None