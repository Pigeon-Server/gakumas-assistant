import ctypes
import sys
from time import sleep
from typing import Tuple

import cv2
import numpy as np
import pyautogui
import win32api
import win32con
import win32gui

from src.entity.Yolo import Yolo_Box, Yolo_Results
from src.utils.logger import logger

class Windows_App:
    __window_name: str
    def __init__(self, window_name):
        # if not self.is_admin():
        #     # 不是管理员，重新以管理员身份启动
        #     logger.warning("当前不是管理员权限，正在尝试以管理员身份重新启动...")
        #     ctypes.windll.shell32.ShellExecuteW(
        #         None, "runas", sys.executable, " ".join(sys.argv), None, 1
        #     )
        #     print(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        #     sys.exit()
        self.__window_name = window_name

    @staticmethod
    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    def __find_window(self):
        """
        获取窗口实例
        :return:
        """
        hwnd = win32gui.FindWindow(None, self.__window_name)
        if not hwnd:
            raise Exception(f'窗口 "{self.__window_name}" 未找到')
        return hwnd

    def __get_window_region(self):
        """
        获取窗口位置
        :return:
        """
        hwnd = self.__find_window()
        client_rect = win32gui.GetClientRect(hwnd)
        client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))
        client_width = client_rect[2]  # 客户区宽度
        client_height = client_rect[3]  # 客户区高度
        return client_left, client_top, client_width, client_height

    @logger.catch
    def capture(self):
        """
        截取窗口位置
        :return:
        """
        client_left, client_top, client_width, client_height = self.__get_window_region()
        # 截取客户区域
        screenshot = pyautogui.screenshot(
            region=(client_left, client_top, client_width, client_height)
        )

        # 转换为 OpenCV 格式
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _xy_abs_conversion(self, x, y) -> Tuple[int, int]:
        left, top, width, height = self.__get_window_region()
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"坐标超出有效范围: ({x}, {y}) 窗口尺寸: {width}x{height}")
        abs_x = left + x
        abs_y = top + y
        return abs_x, abs_y

    @logger.catch
    def click(self, x, y, el_label = ""):
        """
        点击窗口内容
        :param el_label:
        :param x:
        :param y:
        :return:
        """
        abs_x, abs_y = self._xy_abs_conversion(x, y)
        pyautogui.click(abs_x, abs_y, button='left')
        logger.debug(f"click {el_label}: {abs_x, abs_y}" if el_label else f"click: {abs_x, abs_y}")
        return True

    @logger.catch
    def click_element(self, element: Yolo_Box | Yolo_Results):
        self.click(*element.get_COL(), getattr(element, "label", ""))

    @logger.catch
    def scrollY(self, x, y, scroll_delta):
        if scroll_delta == 0:
            logger.warning("scroll delta is 0, skipping scroll")
            return
        abs_x, abs_y = self._xy_abs_conversion(x, y)
        pyautogui.moveTo(abs_x, abs_y)
        pyautogui.click(button='right', clicks=1, interval=0.3)
        for _ in range(abs(scroll_delta)):
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0,120 if scroll_delta > 0 else -120)
            sleep(0.1)
        logger.debug(f"scroll delta: {scroll_delta} (x:{abs_x} y:{abs_y})")