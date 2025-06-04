import threading

import cv2
import torch
import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

import config
from typing import Union, Callable, List
from ultralytics import YOLO
from fastapi import FastAPI
from time import sleep

from src.core.Android.app import Android_App
from src.core.Web.routers import register_routes
from src.core.Web.websocket import WebSocketManager
from src.core.Windows.app import Windows_App
from src.core.middlewares.middleware_register import register_middlewares
from src.core.tasks.base_ui.start_game import action__click_start_game, handle__network_error_modal_boxes, \
    action__check_home_tab_exist
from src.core.tasks.task_register import register_tasks
from src.entity.Game.Game_Info import GameStatusManager
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.WebSocket_Data import WebSocket_Data
from src.core.task import TaskQueue
from src.entity.Yolo import YoloModelType, Yolo_Results
from src.utils.game_tools import get_current_location
from src.utils.ocr_instance import init_ocr
from src.utils.logger import logger

from src.constants import *
from src.utils.yolo_tools import get_modal


class AppProcessor:
    # 推理设备
    device: str
    # Yolo模型
    model: YOLO
    # 操作设备
    app: Android_App | Windows_App
    # 当前Yolo模型
    current_model_type: str
    # 最新帧
    latest_frame: np.array = None
    # 最新推理结果
    latest_results: Yolo_Results | None = None
    # 任务队列
    task_queue: TaskQueue
    # 捕获帧状态
    running: bool = False
    # 捕获帧线程
    capture_thread: threading.Thread = None
    # 暂停捕获帧标志
    _pause_capture_frame: bool = False
    # 中间件注册列表
    _middleware_registry: List[Callable]
    # 游戏状态管理器
    game_status_manager: GameStatusManager

    def __init__(self):
        self.app = self._create_app_instance()
        self.device = self._detect_device()
        self.load_model()
        init_ocr()
        self._middleware_registry = []
        self.task_queue = TaskQueue(self)
        self.game_status_manager = GameStatusManager()
        self.start()
        logger.success("Application Initialized")

    def load_model(self, model_type: str = YoloModelType.BASE_UI):
        """
        加载指定类型的Yolo模型
        :param model_type:
        :return:
        """
        def _init(model_type: str):
            logger.debug(f"Loading YOLO model {model_type}...")
            model_config = config.model_config.get(model_type)
            model = YOLO(model_config.get("model_path")).to(self.device).eval()
            model.conf = model_config.get("conf_threshold")
            model.iou = model_config.get("iou_threshold")
            model.half = torch.cuda.is_available()
            logger.info(f"Model size: {model.overrides.get('imgsz', 640)}")
            return model

        if model_type in [YoloModelType.BASE_UI, YoloModelType.PRODUCER]:
            self.pause_capture_frame()
            self.model = _init(model_type)
            self.current_model_type = model_type
            self.resume_capture_frame()
        else:
            raise ValueError(f'Unknown model type: {model_type}')

    def register_task(self, task_name: str, description: str, timeout: int | None = None):
        """实例方法：注册任务"""
        logger.debug(f"register task: {task_name}")
        def decorator(func: Callable):
            self.task_queue.reg_task(task_name, description, func, timeout)
        return decorator

    def register_middleware(self):
        """实例方法：注册中间件"""
        def decorator(func: Callable):
            logger.debug(f"register middleware: {func.__name__}")
            self._middleware_registry.append(func)
        return decorator

    def initialized(self):
        """生命周期方法：脚本初始化完毕"""
        self.update_current_location()

    def pause_capture_frame(self):
        if self.running and not self._pause_capture_frame:
            logger.debug("Pause capture frame......")
            self._pause_capture_frame = True
            self.capture_thread.join()
            logger.debug("Paused capture frame")

    def resume_capture_frame(self):
        if self.running and self._pause_capture_frame:
            self._pause_capture_frame = False
            self.capture_thread = threading.Thread(target=self._capture_and_infer, daemon=True)
            self.capture_thread.start()
            logger.debug("Resumed capture frame")

    @staticmethod
    def _detect_device() -> str:
        device = 'cuda' if torch.cuda.is_available() and not config.use_cpu else 'cpu'
        logger.info(f"Using device: {device.upper()}")
        return device

    @staticmethod
    def _create_app_instance() -> Union[Android_App, Windows_App]:
        mode = config.mode.lower()
        if mode == 'phone':
            logger.debug("Initializing Android mode")
            return Android_App()
        if mode == 'pc':
            logger.debug("Initializing Windows mode")
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
            return Windows_App(config.window_name)
        raise ValueError(f"Invalid mode: {config.mode}")

    def _capture_and_infer(self):
        while self.running and not self._pause_capture_frame:
            frame = self.app.capture()
            if frame is None or frame.size <= 0:
                sleep(0.5)
                continue
            self.latest_frame = frame
            results = self.model(frame, imgsz=self.model.args['imgsz'] if hasattr(self.model, 'args') else 640, verbose=False, stream=True)
            self.latest_results = Yolo_Results(results, self.model, self.latest_frame)
            self._send_frame_to_clients()

    @logger.catch
    def _send_frame_to_clients(self):
        """将最新的图像的二进制数据发送给 WebSocket 客户端。"""
        if self.latest_frame is None:
            return
        # 获取图像尺寸
        height, width = self.latest_frame.shape[:2]
        if not self.latest_results.results:
            _, encoded_frame = cv2.imencode('.jpg', self.latest_frame)
            frame_bytes = encoded_frame.tobytes()
            ws_manager.broadcast_sync(WebSocket_Data(None, f"{width},{height}".encode('utf-8') + b"," + frame_bytes))
        for result in self.latest_results.results:
            annotated_frame = result.plot(
                conf=False,
                line_width=max(1, int(height / 600)),
                font_size=max(0.5, height / 1200),
                pil=False
            )
            _, encoded_frame = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = encoded_frame.tobytes()
            ws_manager.broadcast_sync(WebSocket_Data(None, f"{width},{height}".encode('utf-8') + b"," + frame_bytes))
            self._exec_middleware()

    def _exec_middleware(self):
        """注册处理中间件"""
        for func in self._middleware_registry:
            func(self)

    def wait_for_label(self, label, timeout=30, interval=1, continuous=1):
        """等待指定标签的框出现"""
        wait_time = 0
        count = 0
        logger.debug(f"waiting label: {label}")
        while wait_time <= timeout:
            if count > continuous:
                return True
            if self.latest_results.filter_by_label(label):
                count += 1
                sleep(0.3)
                continue
            else:
                count = 0
            sleep(interval)
            wait_time += interval
        return False

    def wait_for_modal(self, modal_title, timeout=30, interval=1, no_body: bool = False):
        """等待指定标题的模态框出现"""
        wait_time = 0
        logger.debug(f"waiting modal: {modal_title}")
        while wait_time < timeout:
            if not (
                    self.latest_results.filter_by_label(base_labels.modal_header) and
                    self.latest_results.filter_by_label(base_labels.button)
            ):
                sleep(interval)
                wait_time += interval
                continue

            modal = get_modal(self.latest_results, self.latest_frame, no_body)
            if modal is None:
                wait_time += interval
                sleep(interval)
                continue
            print(modal)
            if modal_title is not None and modal_title in modal.modal_title:
                return modal
            elif modal_title is None:
                return modal
            else:
                wait_time += interval
                sleep(interval)
        return False

    def click_on_label(self, label, timeout=30, interval=1):
        """等待指定标签并点击"""
        wait_time = 0
        count = 0
        logger.debug(f"waiting click label: {label}")
        while wait_time < timeout:
            boxs = self.latest_results.filter_by_label(label)
            if boxs:
                self.app.click_element(boxs.first())
                return True
            else:
                count += 1
                if count >= 3:
                    break
                sleep(interval)
            wait_time += interval
        return False

    def wait__loading(self, timeout=60):
        """等待加载"""
        COUNT = 0
        sleep(3)
        while COUNT < timeout:
            logger.debug("Waiting for loading")
            if self.latest_results.filter_by_label(base_labels.general_loading1) or self.latest_results.filter_by_label(
                    base_labels.general_loading2):
                sleep(1)
                COUNT += 1
            else:
                logger.debug("Wait for the loading to finish")
                return True
        raise TimeoutError("Waiting for a load timeout")

    def go_home(self):
        self.update_current_location()
        if self.game_status_manager.current_location == GamePageTypes.MAIN_MENU__HOME:
            return
        for _ in range(5):
            logger.debug(f"[{_}]Try going home")
            main_menu_items = [
                value for name, value in vars(GamePageTypes).items()
                if name.startswith("MAIN_MENU__")
            ]
            if self.game_status_manager.current_location in main_menu_items:
                self.app.click_element(self.latest_results.filter_by_label(base_labels.tab_home).first())
                self.wait__loading()
                self.update_current_location()
                return
            elif go_home_btn := self.latest_results.filter_by_label(base_labels.go_home_btn):
                self.app.click_element(go_home_btn.first())
                self.wait__loading()
                self.update_current_location()
                return
            sleep(2)
        raise RuntimeError("Going home failed")


    def back_next_page(self):
        logger.debug("Going back next page")
        if self.wait_for_label(base_labels.back_btn, 3):
            self.app.click_element(self.latest_results.filter_by_label(base_labels.back_btn).first())
        else:
            raise TimeoutError("Waiting for a back button timeout")

    def update_current_location(self):
        logger.debug("Updating current location......")
        current_location = get_current_location(self.latest_results)
        if current_location and current_location != self.game_status_manager.current_location:
            self.game_status_manager.current_location = current_location
            logger.debug(f"Current location: {self.game_status_manager.current_location}")

    def start(self):
        if not self.running or self._pause_capture_frame:
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_and_infer, daemon=True)
            self.capture_thread.start()
            logger.success("Started inference thread.")

    def stop(self):
        if self.running:
            self.running = False
            self.capture_thread.join(timeout=3)
            logger.success("Stopped inference thread.")

    def exec_task(self):
        self.task_queue.exec_task()


app = FastAPI()
processor = AppProcessor()
ws_manager = WebSocketManager()

register_routes(app, processor, ws_manager)
register_tasks(processor)
register_middlewares(processor)

@app.on_event("shutdown")
def shutdown_event():
    processor.stop()

@app.on_event("startup")
def start_event():
    sleep(5)
    processor.start()