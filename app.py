import threading

import cv2
import torch
import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

import config
from typing import Union, Callable
from ultralytics import YOLO
from fastapi import FastAPI
from time import sleep

# from src.entity.Yolo_Box import Yolo_Box
from src.core.Android.app import Android_App
from src.core.Web.websocket import WebSocketManager
from src.core.Windows.app import Windows_App
from src.entity.WebSocket_Data import WebSocket_Data
from src.entity.YOLO_Model_Type import YoloModelType
from src.entity.Yolo_Results import Yolo_Results
from src.core.task import TaskQueue
from src.utils.ocr_instance import init_ocr
from src.utils.logger import logger

app = FastAPI()

class AppProcessor:
    # 推理设备
    device: str
    # Yolo模型
    model: YOLO
    # 操作设备
    app: Android_App | Windows_App
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
        self.device = self._detect_device()
        self.load_model()
        init_ocr()
        self.app = self._create_app_instance()
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

# 创建处理器实例
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

@processor.register_task("test1", "测试任务1")
def _task__test1(app):
    logger.debug("task1: run")
    sleep(3)
    logger.debug("task1: stop")
    return False

@processor.register_task("test2", "测试任务2")
def _task__test2(app):
    logger.debug("task2: run")
    sleep(5)
    logger.debug("task2: stop")
    return False

@processor.register_task("test3", "测试任务3")
def _task__test3(app):
    logger.debug("task3: run")
    sleep(3)
    logger.debug("task3: stop")
    return False