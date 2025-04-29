import threading

import cv2
import torch
import numpy as np
from skimage.filters.rank import modal
from starlette.websockets import WebSocket, WebSocketDisconnect

import config
from typing import Union, Callable
from ultralytics import YOLO
from fastapi import FastAPI
from time import sleep

from src.core.Android.app import Android_App
from src.core.Web.websocket import WebSocketManager
from src.core.Windows.app import Windows_App
from src.core.tasks.base_ui.start_game import action__click_start_game, \
    handle__network_error_modal_boxes, action__check_home_tab_exist
from src.entity.WebSocket_Data import WebSocket_Data
from src.entity.YOLO_Model_Type import YoloModelType
from src.entity.Yolo_Results import Yolo_Results
from src.core.task import TaskQueue
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
    latest_frame: np.array
    # 最新推理结果
    latest_results: Yolo_Results | None
    # 任务注册列表
    task_registry: dict[str, dict[str, Union[Callable, str]]] = {}
    # 任务队列
    task_queue: TaskQueue
    # 捕获帧状态
    running: bool = False
    # 捕获帧线程
    capture_thread: threading.Thread = None
    # 暂停捕获帧标志
    _pause_capture_frame: bool = False

    def __init__(self):
        self.app = self._create_app_instance()
        self.device = self._detect_device()
        self.load_model()
        init_ocr()
        self.task_queue = TaskQueue(self)
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


    def register_task(self, task_name: str, description: str):
        """实例方法：注册任务"""
        logger.debug(f"register task: {task_name}")
        def decorator(func: Callable):
            self.task_queue.reg_task(task_name, description, func)
        return decorator

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
            return False
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

    def wait_for_label(self, label, timeout=30, interval=1, continuous=1):
        """等待指定标签的框出现"""
        wait_time = 0
        count = 0
        logger.debug(f"waiting label: {label}")
        while wait_time <= timeout:
            if count > continuous:
                return True
            if self.latest_results.get_yolo_boxs_by_label(label):
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
        count = 0
        logger.debug(f"waiting modal: {modal_title}")
        while wait_time < timeout:
            if count > 3:
                modal = get_modal(self.latest_results.yolo_boxs, self.latest_frame, no_body)
                if modal.modal_title == modal_title:
                    return modal
                return False

            if not (self.latest_results.get_yolo_boxs_by_label(labels.modal_header) and (self.latest_results.get_yolo_boxs_by_label(labels.cancel_button) or self.latest_results.get_yolo_boxs_by_label(labels.confirm_button))):
                sleep(interval)
                count = 0
                wait_time += interval
                continue
            else:
                count += 1
                sleep(0.3)
        return False

    def click_on_label(self, label, timeout=30, interval=1):
        """等待指定标签并点击"""
        wait_time = 0
        count = 0
        logger.debug(f"waiting click label: {label}")
        while wait_time < timeout:
            boxs = self.latest_results.get_yolo_boxs_by_label(label)
            if boxs:
                self.app.click_element(boxs[0])
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
        while COUNT < timeout:
            logger.debug("Waiting for loading")
            if self.latest_results.get_yolo_boxs_by_label(labels.general_loading1) or self.latest_results.get_yolo_boxs_by_label(labels.general_loading2):
                sleep(1)
                COUNT += 1
            else:
                logger.debug("Wait for the loading to finish")
                return True
        raise TimeoutError("Waiting for a load timeout")

    def go_home(self):
        logger.debug("Going home")
        if not self.click_on_label(labels.go_home_btn, 3):
            raise TimeoutError("Waiting for a go home button timeout")

    def back_next_page(self):
        logger.debug("Going back next page")
        if self.wait_for_label(labels.back_btn, 3):
            self.app.click_element(self.latest_results.get_yolo_boxs_by_label(labels.back_btn)[0])
        else:
            raise TimeoutError("Waiting for a back button timeout")

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 处理端点"""
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.get("/start")
def start_inference():
    processor.exec_task()
    return {"message": "Inference started"}

@app.get("/stop")
def stop_inference():
    processor.task_queue.stop()
    return {"message": "Inference stopped"}

@app.get("/status")
def get_status():
    return {'status': processor.running}

@app.get("/get_registered_tasks")
def get_registered_tasks():
    return processor.task_queue.get_task_list()

@app.post("/disable_task/{task_name:str}")
def disable_task(task_name):
    return processor.task_queue.disable_task(task_name)

@app.get("/switch_yolo_model/base_ui")
def switch_yolo_model__base_ui():
    processor.load_model(YoloModelType.BASE_UI)
    return {"status": True}

@app.get("/switch_yolo_model/producer")
def switch_yolo_model__producer():
    processor.load_model(YoloModelType.PRODUCER)
    return {"status": True}

# @app.get("/get_enabled_tasks")
# def get_enabled_tasks():
#     return processor.task_queue.get_enabled_tasks()


@app.on_event("shutdown")
def shutdown_event():
    processor.stop()

@app.on_event("startup")
def start_event():
    sleep(5)
    processor.start()

def action__back_home(app: AppProcessor):
    """返回主页面"""


@processor.register_task("start_game", "启动游戏")
def _task__start_game(app: AppProcessor):
    TIMEOUT = 30
    if action__click_start_game(app, TIMEOUT) is not False:
        sleep(2)
        app.wait__loading()
        handle__network_error_modal_boxes(app)
    action__check_home_tab_exist(app)

@processor.register_task("get_expenditure", "获取活动费")
def _task__get_expenditure(app: AppProcessor):
    if tab_home := app.latest_results.get_yolo_boxs_by_label(labels.tab_home):
        tab_home = tab_home[0]
        app.app.click_element(tab_home)
        if not app.wait_for_label(labels.home_get_expenditure):
            raise TimeoutError("Timeout waiting for home expenditure to appear.")
        get_expenditure_btn = app.latest_results.get_yolo_boxs_by_label(labels.home_get_expenditure)[0]
        app.app.click_element(get_expenditure_btn)
        if modal := app.wait_for_modal("活動費", no_body=True):
            app.app.click_element(modal.cancel_button)
            sleep(3)
            return True
        else:
            raise TimeoutError("Timeout waiting for modal to appear.")
    else:
        raise RuntimeError("当前不在主页")

@processor.register_task("dispatch_work", "派遣任务")
def _task__dispatch_work(app: AppProcessor):
    if tab_home := app.latest_results.get_yolo_boxs_by_label(labels.tab_home):
        tab_home = tab_home[0]
        app.app.click_element(tab_home)
        if not app.wait_for_label(labels.home_dispatch_work):
            raise TimeoutError("Timeout waiting for home dispatch work to appear.")
        dispatch_work_btn = app.latest_results.get_yolo_boxs_by_label(labels.home_dispatch_work)[0]
        app.app.click_element(dispatch_work_btn)
        sleep(2)
        app.wait__loading()
        while True:
            if not app.latest_results.get_yolo_boxs_by_label(labels.modal_header):
                break
            if modal := app.wait_for_modal("お仕事完了", no_body=True):
                app.app.click_element(modal.confirm_button)
                sleep(2)
                if not app.wait_for_label(labels.modal_header, 10):
                    break
        # TODO 重新派遣
        sleep(1)
        app.go_home()
        app.wait__loading()
        return True
    else:
        raise RuntimeError("当前不在主页")