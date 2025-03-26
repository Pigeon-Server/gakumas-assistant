import threading

import cv2
import torch
import numpy as np
import config
import src.constants as constants
from typing import Union, Callable
from ultralytics import YOLO

from src.entity.Yolo_Box import Yolo_Box
from src.core.Android.app import Android_App
from src.core.Windows.app import Windows_App
from src.utils.task.core import TaskQueue
from src.utils.yolo_tools import get_modal, get_element, find_element
from src.utils.ocr_instance import init_ocr
from src.utils.logger import logger


class AppProcessor:
    """应用程序主处理器，封装核心业务流程"""
    device: str
    model: YOLO
    app: Android_App | Windows_App
    _debug_window_created: bool
    latest_frame: np.array
    latest_results: list[Yolo_Box] | None
    task_registry: dict[str, Callable] = {}

    def __init__(self):
        """初始化处理器"""
        self.device = self._detect_device()
        self.model = self._init_yolo(self.device)
        init_ocr()
        self.app = self._create_app_instance()
        self._debug_window_created = False  # 调试窗口创建状态
        self.task_queue = TaskQueue()

        # 存储最新帧和推理结果
        self.latest_frame = None
        self.latest_results = None

        # 启动后台线程进行帧捕获和推理
        self.running = True
        self.capture_thread = threading.Thread(target=self._thread__capture_and_infer, daemon=True)
        self.capture_thread.start()

    @classmethod
    def register_task(cls, task_name: str, description: str):
        """装饰器：注册任务到任务队列"""
        def decorator(func: Callable):
            cls.task_registry[task_name] = (func, description)
            return func
        return decorator


    @staticmethod
    def _detect_device() -> str:
        """检测并返回计算设备"""
        device = 'cuda' if torch.cuda.is_available() and not config.use_cpu else 'cpu'
        logger.info(f"use device: {device.upper()}")
        return device

    @staticmethod
    def _init_yolo(device) -> YOLO:
        """
        初始化YOLO模型
        """
        logger.debug("loading yolo model......")
        model = YOLO(config.model_path).to(device).eval()
        model.conf = config.conf_threshold  # 置信度阈值
        model.iou = config.iou_threshold    # IoU阈值
        model.half = torch.cuda.is_available()
        # 记录模型参数
        model_imgsz = model.args.get('imgsz', 640)
        logger.info(f"model size: {model_imgsz}")
        logger.info(f"置信度阈值: {model.conf:.2f}")
        logger.info(f"半精度模式: {model.half}")
        return model

    @staticmethod
    def _create_app_instance() -> Union[Android_App, Windows_App]:
        """
        创建平台特定的应用实例
        """
        mode = config.mode.lower()
        if mode == 'phone':
            logger.debug("初始化Android模式")
            return Android_App()

        if mode == 'pc':
            logger.debug("初始化Windows模式")
            # 处理Windows高DPI缩放问题
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
            return Windows_App(
                config.window_name
            )

        raise ValueError(f"无效的运行模式: {config.mode}")

    def _thread__capture_and_infer(self):
        while True:
            # 捕获屏幕帧
            frame = self.app.capture()

            # 有效性检查（修复ValueError）
            if frame is None or frame.size <= 0:
                continue

            self.latest_frame = frame

            results = self.model(
                frame,
                imgsz=self.model.args.get('imgsz', 640),
                verbose=False,
                stream=True
            )

            for result in results:
                if not result.boxes:
                    continue

                self.latest_results = self._parse_boxes(result, frame)

                if config.debug:
                    self._display_debug_info(frame, result)

    def _parse_boxes(self, result, frame: np.ndarray) -> list[Yolo_Box]:
        """
        解析YOLO检测框为业务对象

        参数:
            result: YOLO检测结果
            frame (np.ndarray): 原始图像帧

        返回:
            list[Yolo_Box]: 检测框对象列表
        """
        boxs = []
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxs.append(Yolo_Box(x1, y1, x2, y2, class_name, frame[y1:y2, x1:x2]))
        return boxs

    # def _handle_special_components(self, boxes: list[Yolo_Box], frame: np.array):
    #     """处理特殊界面组件（如模态对话框）"""
    #     # 处理启动游戏点击
    #     if find_element(boxes, constants.labels.start_menu_logo) and find_element(boxes,
    #                                                                               constants.labels.start_menu_click_continue_flag):
    #         return self._handle_game_start_menu(boxes)
    #     if find_element(boxes, constants.labels.modal_header) and ((not find_element(boxes,
    #                                                                                  constants.labels.confirm_button) and find_element(
    #             boxes, constants.labels.cancel_button)) or (find_element(boxes,
    #                                                                      constants.labels.confirm_button) and not find_element(
    #             boxes, src.constants.labels.cancel_button))):
    #         logger.debug("检测到仅关闭模态框")
    #         return self._handle_close_only_modal(boxes)
    #     # 处理模态框
    #     if find_element(boxes, constants.labels.modal_header) and (find_element(boxes,
    #                                                                             src.constants.labels.confirm_button) or find_element(
    #             boxes, src.constants.labels.cancel_button)):
    #         logger.debug("检测到模态对话框")
    #         get_modal(boxes, frame)

    # def _handle_game_start_menu(self, boxes: list[Yolo_Box]):
    #     """处理游戏启动界面"""
    #     if find_element(boxes, constants.labels.general_loading1):
    #         return False
    #     logger.info("click start_menu_click_continue_flag")
    #     element = get_element(boxes, constants.labels.start_menu_click_continue_flag)[0]
    #     self.app.click(*element.get_COL())
    #     return True

    # def _handle_close_only_modal(self, boxes: list[Yolo_Box]):
    #     logger.debug("close modal box")
    #     if el := get_element(boxes, constants.labels.confirm_button):
    #         return self.app.click(*el[0].get_COL())
    #     if el := get_element(boxes, constants.labels.cancel_button):
    #         return self.app.click(*el[0].get_COL())

    def _display_debug_info(self, frame: np.ndarray, result):
        """显示调试信息窗口"""
        h, w = frame.shape[:2]
        debug_frame = result.plot(
            conf=False,
            line_width=max(1, int(h / 600)),
            font_size=max(0.5, h / 1200),
            pil=False
        )

        if not self._debug_window_created:
            cv2.namedWindow(config.debug_window_name, cv2.WINDOW_NORMAL)
            self._debug_window_created = True

        cv2.resizeWindow(config.debug_window_name, w, h)
        cv2.imshow(config.debug_window_name, debug_frame)

    # def run(self):
    #     """主运行循环"""
    #     try:
    #         logger.info("启动主处理循环")
    #         while True:
    #             # 捕获屏幕帧
    #             frame = self.app.capture()
    #
    #             # 有效性检查（修复ValueError）
    #             if frame is not None and frame.size > 0:
    #                 self._process_frame(frame)
    #
    #             # 检查任务系统状态
    #             if config.debug and self._task_system.get_status()['current_task']:
    #                 logger.debug(f"当前任务: {self._task_system.get_status()['current_task']}")
    #
    #             # 处理退出指令
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break
    #     except Exception as e:
    #         logger.error(f"主循环异常: {str(e)}")
    #         raise
    #     finally:
    #         self.cleanup()

    def cleanup(self):
        """清理资源"""
        if config.debug and self._debug_window_created:
            cv2.destroyWindow(config.debug_window_name)

def main():
    """程序入口函数"""
    try:
        logger.info("启动应用程序")
        processor = AppProcessor()
        # processor.run()
    except Exception as e:
        logger.critical(f"致命错误: {str(e)}")
        raise
    finally:
        logger.info("应用程序退出")

if __name__ == '__main__':
    main()