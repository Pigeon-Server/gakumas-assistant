import abc

import numpy as np

from src.entity.Yolo import Yolo_Box, Yolo_Results


class BaseDevice(abc.ABC):

    def close(self):
        """
        释放设备持有的运行时资源
        """
        return None

    def __bool__(self) -> bool:
        pass

    def is_app_focused(self):
        pass

    def is_app_running(self):
        """
        判断游戏进程是否正在运行
        :return:
        """
        pass

    def start_game(self):
        """
        启动游戏
        :return:
        """
        pass

    def capture(self) -> np.ndarray:
        """
        截取游戏画面
        :return:
        """

    def click(self, x, y, el_label = ""):
        """
        点击窗口内容（坐标）
        :param x: x轴窗口内坐标
        :param y: y轴窗口内坐标
        :param el_label: 标签（debug用）
        :return:
        """


    def click_element(self, element: Yolo_Box | Yolo_Results):
        """
        点击窗口内容（元素）
        :param element: Yolo_Box | Yolo_Results
        :return:
        """
        self.click(*element.get_COL(), getattr(element, "label", ""))

    def scrollY(self, x, y, scroll_delta):
        """
        向上下滚动窗口
        :param x: x轴窗口内坐标
        :param y: y轴窗口内坐标
        :param scroll_delta: 滚动距离（各系统可能不大相同）
        :return:
        """

    def scrollX(self, x, y, scroll_delta):
        """
        向左右滚动窗口
        :param x: x轴窗口内坐标
        :param y: y轴窗口内坐标
        :param scroll_delta: 滚动距离（各系统可能不大相同）
        :return:
        """

    def bring_to_front(self):
        pass

    def get_diagnostics(self) -> dict:
        """
        返回设备层诊断信息（用于任务失败 dump）。
        子类可按设备类型扩展字段。
        """
        return {}
