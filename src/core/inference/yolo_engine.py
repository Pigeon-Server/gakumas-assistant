import threading
import traceback
from threading import Thread
from time import sleep

from typing import Callable, List

import config
from src.constants.yolo.model_type import YoloModelType
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.BaseDevice import BaseDevice
from src.entity.Yolo import Yolo_Results
from src.utils.logger import logger



class YoloInferenceEngine:
    _engine: YoloModelFromONNX
    _device: BaseDevice
    _model_type: str
    _latest_results: Yolo_Results | None = None
    _pause_capture_frame: bool = False
    _capture_thread: Thread
    _infer_callback_list: List[Callable]
    _capture_failure_callback_list: List[Callable]
    _agnostic_nms_groups: list[set[int]] | None = None  # 跨类别NMS分组
    # Flags
    __flag_loop: bool = False  # 主循环
    __flag_pause: bool = False  # 暂停
    # Lock
    __action_lock: threading.Lock  # 动作锁
    __result_write_lock: threading.Lock  # 写入锁


    def __init__(self, device: BaseDevice):
        self._device = device
        self._infer_callback_list = []
        self._capture_failure_callback_list = []
        self.__action_lock = threading.Lock()
        self.__result_write_lock = threading.Lock()
        self.load_model()

    def set_device(self, device: BaseDevice):
        self._device = device

    def load_model(self, model_type: str = YoloModelType.BASE_UI):
        """
        加载指定类型的Yolo模型
        :param model_type:
        :return:
        """
        if model_type not in YoloModelType.__dict__.keys() and model_type in config.model_config.keys():
            raise ValueError(f'Unknown model type: {model_type}')
        if self.__flag_loop:
            self.pause()
        with self.__action_lock:
            logger.debug(f"Loading YOLO model {model_type}...")
            self._engine = YoloModelFromONNX(config.model_config[model_type])
            self._model_type = model_type
            # 从模型元数据中自动构建跨类别NMS分组（技能卡 Active/Mental/Trap 视为同类）
            self._agnostic_nms_groups = self._build_skill_card_nms_group()
        if self.__flag_loop:
            self.resume()

    def _build_skill_card_nms_group(self) -> list[set[int]] | None:
        """从模型标签中查找技能卡类别ID，构建跨类别NMS分组。"""
        skill_labels = {"Skill Card: Active", "Skill Card: Mental", "Skill Card: Trap"}
        group: set[int] = set()
        for cid, name in self._engine._model_meta.names.items():
            if name in skill_labels:
                group.add(cid)
        return [group] if len(group) >= 2 else None

    def start(self):
        """
        开始推理进程
        :return:
        """
        with self.__action_lock:
            if self.__flag_loop:
                return False
            self.__flag_pause = False
            self.__flag_loop = True
            self._capture_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._capture_thread.start()
            logger.success("Started inference thread.")
        return True

    def stop(self):
        """
        结束推理进程
        :return:
        """
        with self.__action_lock:
            if not self.__flag_loop:
                return False
            self.__flag_pause = False
            self.__flag_loop = False
            self._capture_thread.join(timeout=3)
            logger.success("Stopped inference thread.")
        return True

    def pause(self):
        """
        暂停推理
        :return:
        """
        with self.__action_lock:
            if self.__flag_pause:
                return False
            self.__flag_pause = True
            logger.debug("Paused inference frame")
            return True

    def resume(self):
        """
        恢复推理
        :return:
        """
        with self.__action_lock:
            if not self.__flag_pause:
                return False
            self.__flag_pause = False
            logger.debug("Resumed inference frame")
            return True

    @property
    def is_pause(self):
        with self.__action_lock:
            return self.__flag_pause

    @property
    def running(self):
        with self.__action_lock:
            return self.__flag_loop

    @property
    def latest_frame(self):
        try:
            return self._latest_results.frame
        except Exception as e:
            logger.warning(f"Get latest frame error: {e}")
            return None

    @property
    def latest_results(self):
        try:
            return self._latest_results
        except Exception as e:
            logger.warning(f"Get latest results error: {e}")
            return None

    @property
    def model_type(self):
        return self._model_type

    def register_infer_callback(self, func: Callable):
        with self.__action_lock:
            if func not in self._infer_callback_list:
                logger.debug(f"Register inference callback: {func.__name__}")
                self._infer_callback_list.append(func)
            else:
                logger.error(f"Inference callback already registered: {func}")

    def register_capture_failure_callback(self, func: Callable):
        with self.__action_lock:
            if func not in self._capture_failure_callback_list:
                logger.debug(f"Register capture failure callback: {func.__name__}")
                self._capture_failure_callback_list.append(func)
            else:
                logger.error(f"Capture failure callback already registered: {func}")


    def _exec_infer_callback(self):
        for callback in self._infer_callback_list:
            try:
                callback(self.latest_frame, self.latest_results)
            except Exception as e:
                logger.error(f"Inference callback failed: {e}")

    def _exec_capture_failure_callback(self, exc: Exception):
        for callback in self._capture_failure_callback_list:
            try:
                callback(exc)
            except Exception as callback_exc:
                logger.error(f"Capture failure callback failed: {callback_exc}")

    def _inference_loop(self):
        """
        截图并推理
        """
        while self.__flag_loop:
            if self.__flag_pause:
                sleep(0.1)
                continue
            try:
                frame = self._device.capture()
            except Exception as e:
                self._exec_capture_failure_callback(e)
                reason_getter = getattr(self._device, "get_unavailable_reason", None)
                if not self._device and callable(reason_getter):
                    logger.warning(f"Inference loop stopped: {reason_getter()}")
                else:
                    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()
                    logger.error(f"Inference loop failed: \n{tb_str}")
                self.__flag_pause = False
                self.__flag_loop = False
                return
            if frame is None or frame.size <= 0:
                sleep(0.1)
                continue
            results = self._engine(
                frame,
                conf_threshold=0.6,
                agnostic_nms_groups=self._agnostic_nms_groups
            )
            with self.__result_write_lock:
                self._latest_results = Yolo_Results(results, frame)
            self._exec_infer_callback()
        self.__flag_loop = False
