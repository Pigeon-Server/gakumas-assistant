import cv2
import numpy as np
import pyautogui
import win32gui
import pydirectinput

from src.utils.logger import logger

class Windows_App:
    __window_name: str
    def __init__(self, window_name):
        self.__window_name = window_name

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

    @logger.catch
    def click(self, x, y):
        """
        点击窗口内容
        :param x:
        :param y:
        :return:
        """
        left, top, width, height = self.__get_window_region()
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"坐标超出有效范围: ({x}, {y}) 窗口尺寸: {width}x{height}")
        abs_x = left + x
        abs_y = top + y
        # pydirectinput.moveTo(abs_x, abs_y)
        pydirectinput.click(abs_x, abs_y, button='left')
        print(f"click: {abs_x, abs_y}")
        return True