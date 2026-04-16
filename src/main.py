import os
import platform
import shutil
import threading
from typing import TYPE_CHECKING

import cv2
import config
from time import sleep

from src.constants.path.data_path import DataPath
from src.constants.path.debug_path import DebugPath
from src.constants.device.device_type import DeviceType
from src.constants.task_status import TaskStatus
from src.constants.websocket_actions import WebsocketActions
from src.core.device.Android.app import Android_App
from src.core.device.unavailable_device import UnavailableDevice
from src.core.inference.yolo_engine import YoloInferenceEngine
from src.core.services.task_service import TaskService
from src.core.services.resource_update_service import ResourceUpdateService
from src.core.web.websocket import WebSocketManager
from src.core.device.windows_compat import (
    create_windows_device,
    get_windows_unavailability_reason,
    windows_pc_mode_is_available,
)
from src.core.device.macos_compat import (
    create_mac_device,
    get_mac_unavailability_reason,
    mac_playtools_mode_is_available,
)
from src.core.services.config_service import ConfigService
from src.entity.BaseDevice import BaseDevice
from src.entity.Game.Game_Info import GameStatusManager
from src.entity.WebSocketData import WebSocketData
from src.utils.debug_tools import DebugTools
from src.utils.dmm_tools import extract_gakumas_launch_parameters
from src.utils.logger import logger
from src.utils.runtime_paths import resolve_data_str

if TYPE_CHECKING:
    from src.core.services.clip_services import CLIPServiceManager
    from src.core.services.game_utils import GameUtils

class AppProcessor:
    data_path: str
    # 配置服务
    config_service: ConfigService
    # 操作设备
    device: BaseDevice
    # 任务队列
    task_queue: TaskService
    # Yolo推理引擎
    yolo_engine: YoloInferenceEngine
    # 图像debug工具
    debug_tools: DebugTools
    # 游戏实用工具
    game_utils: "GameUtils | None"
    # 游戏状态管理器
    game_status_manager: GameStatusManager
    # 图像记忆管理器
    clip_manager: "CLIPServiceManager | None"
    # Websocket Session管理器
    ws_manager: WebSocketManager
    # 资源仓库更新检查服务
    resource_update_service: ResourceUpdateService

    def __init__(self):
        self._device_state_lock = threading.RLock()
        self._resource_state_lock = threading.RLock()
        self._resource_ready = False
        self._task_services_registered = False
        self._shutdown_requested = threading.Event()
        self._device_status = {
            "available": False,
            "code": "initializing",
            "message": "正在初始化设备...",
        }
        self._device_polling_thread = threading.Thread(target=self._poll_device_state_loop, daemon=True)
        self._device_polling_thread.start()
        self._init_environment()
        self._init_database()
        self.config_service = ConfigService()
        logger.debug(self.config_service())
        self.device = self.create_device_instance()
        self.yolo_engine = YoloInferenceEngine(self.device)
        self.debug_tools = DebugTools()
        self.yolo_engine.register_infer_callback(self._send_frame_to_clients)
        self.yolo_engine.register_capture_failure_callback(self._handle_device_capture_failure)
        self.task_queue = TaskService(self)
        self.game_status_manager = GameStatusManager()
        self.ws_manager = WebSocketManager()
        self.resource_update_service = ResourceUpdateService(self)
        self.clip_manager = None
        self.game_utils = None
        self._register_task_services()
        self._register_config_listening()
        if self.resource_update_service.has_required_resources():
            self.ensure_resource_dependencies_initialized()
        else:
            logger.warning("游戏数据库资源尚未就绪，等待用户确认后下载。")
            logger.success("Application Started In Bootstrap Mode")

    def _register_task_services(self):
        if self._task_services_registered:
            return
        from src.core.tasks.middlewares.middleware_register import register_middlewares
        from src.core.tasks.task_register import register_tasks

        register_tasks(self)
        register_middlewares(self)
        self._task_services_registered = True

    def _init_environment(self):
        """
        初始化环境
        """
        self.data_path = resolve_data_str()
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(resolve_data_str("CLIP"), exist_ok=True)
        if os.path.exists(DebugPath.BasePath()):
            shutil.rmtree(DebugPath.BasePath())
        os.makedirs(DebugPath.BasePath(), exist_ok=True)

    @staticmethod
    def _init_database():
        """
        初始化数据库
        """
        from src.models.base import db
        from src.models import all_models
        from playhouse.migrate import SqliteMigrator, migrate
        if db.is_closed():
            db.connect()
        db.create_tables(all_models)
        migrator = SqliteMigrator(db)
        for model in all_models:
            existing_columns = [c.name for c in db.get_columns(model._meta.table_name)]
            for field_name, field_obj in model._meta.fields.items():
                if field_name not in existing_columns:
                    logger.warning(f"Field '{field_name}' is missing in DB. Migrating...")
                    migrate(
                        migrator.add_column(model._meta.table_name, field_name, field_obj)
                    )
        db.close()
        from src.models.config import ConfigModel
        ConfigModel.update_database()
        logger.success("Database Initialized")

    @staticmethod
    def init_database():
        AppProcessor._init_database()

    @staticmethod
    def _load_game_database(force_reload: bool = False):
        from src.utils.game_database_tools import (
            GakumasDatabase_ItemDataUtils,
            GakumasDatabase_ProduceCardDataUtils,
            reload_loaded_game_databases,
        )
        if force_reload:
            reload_loaded_game_databases()
        GakumasDatabase_ItemDataUtils()
        GakumasDatabase_ProduceCardDataUtils()
        logger.success("Load game database successfully")

    def reload_game_database(self):
        self.ensure_resource_dependencies_initialized(force_reload=True)

    def is_resource_ready(self) -> bool:
        with self._resource_state_lock:
            return self._resource_ready

    def has_required_resources(self) -> bool:
        return self.resource_update_service.has_required_resources()

    def ensure_resource_dependencies_initialized(self, force_reload: bool = False) -> bool:
        with self._resource_state_lock:
            if not self.resource_update_service.has_required_resources():
                self._resource_ready = False
                return False

            self._load_game_database(force_reload=force_reload)

            if self.clip_manager is None:
                from src.core.services.clip_services import CLIPServiceManager

                self.clip_manager = CLIPServiceManager()

            if self.game_utils is None:
                from src.core.services.game_utils import GameUtils

                self.game_utils = GameUtils(self)

            self._resource_ready = True

        self.start_inference_if_possible()
        logger.success("Application Initialized")
        return True

    def start_background_services(self):
        self.resource_update_service.start()

    def request_app_shutdown(self):
        self._shutdown_requested.set()

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    def shutdown(self):
        self._shutdown_requested.set()
        try:
            self.task_queue.stop()
        except Exception as exc:
            logger.debug(f"Skip task queue stop during shutdown: {exc}")
        try:
            self.yolo_engine.stop()
        except Exception as exc:
            logger.debug(f"Skip inference stop during shutdown: {exc}")
        self._close_device(getattr(self, "device", None))

    def _close_device(self, device: BaseDevice | None):
        if device is None:
            return
        close = getattr(device, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            logger.debug(f"Skip device close: {exc}")

    def _describe_device_state(self, device: BaseDevice) -> dict:
        if bool(device):
            return {
                "available": True,
                "code": "ready",
                "message": "",
            }
        reason_getter = getattr(device, "get_unavailable_reason", None)
        code_getter = getattr(device, "get_unavailable_code", None)
        message = reason_getter() if callable(reason_getter) else "当前设备不可用。"
        code = code_getter() if callable(code_getter) else "device_unavailable"
        return {
            "available": False,
            "code": code,
            "message": message,
        }

    def _poll_device_state_loop(self):
        """定期检测设备状态并广播更新"""
        while not self.is_shutdown_requested():
            try:
                self._update_device_state(self.device)
                
                # If device recently became available and inference hasn't started, start it
                with self._device_state_lock:
                    is_available = self._device_status.get('available', False)
                if is_available and not getattr(self.yolo_engine, "running", False):
                    self.start_inference_if_possible()
            except Exception as exc:
                logger.debug(f"Device polling error: {exc}")
            sleep(2)

    def _update_device_state(self, device: BaseDevice):
        with self._device_state_lock:
            previous = dict(self._device_status)
            current = self._describe_device_state(device)
            self._device_status = current
        if previous != current:
            if current["available"]:
                logger.success("设备已就绪，已自动识别。")
            else:
                logger.warning(f"设备不可用：{current['message']}")
            self._broadcast_device_status(current)

    def _broadcast_device_status(self, status: dict):
        if not hasattr(self, "ws_manager"):
            return
        if not getattr(self.ws_manager, "active_connections", None):
            return
        try:
            self.ws_manager.broadcast_action_sync(
                WebsocketActions.Device.StatusChanged,
                WebSocketData(message=status),
            )
        except RuntimeError as exc:
            logger.debug(f"Skip device status websocket broadcast: {exc}")

    def build_app_status(self) -> dict:
        """构建前端状态栏需要的应用状态快照。"""
        current_task = self.task_queue.get_current_running_task()
        return {
            'platform': self.config_service().base.run_mode.value.lower(),
            'yolo': self.yolo_engine.running,
            'task': self.task_queue.queue_status(),
            'current_task': current_task.id if current_task else '',
            'device': self.get_device_status(),
            'game': {
                'current_location': self.game_status_manager.current_location,
                'player': {
                    'level': self.game_status_manager.player.level,
                    'gem': self.game_status_manager.player.gem,
                    'stamina': self.game_status_manager.player.stamina,
                }
            }
        }

    def broadcast_app_status(self):
        """通过 websocket 广播应用状态快照。"""
        if not hasattr(self, "ws_manager"):
            return
        if not getattr(self.ws_manager, "active_connections", None):
            return
        try:
            self.ws_manager.broadcast_action_sync(
                WebsocketActions.App.StatusChanged,
                WebSocketData(message=self.build_app_status()),
            )
        except RuntimeError as exc:
            logger.debug(f"Skip app status websocket broadcast: {exc}")

    def get_device_status(self) -> dict:
        with self._device_state_lock:
            return dict(self._device_status)

    def _handle_device_capture_failure(self, exc: Exception):
        self._update_device_state(self.device)

    def start_inference_if_possible(self) -> bool:
        if not self.is_resource_ready():
            return False
        if not self.device:
            return False
        if self.yolo_engine.running:
            return True
        return self.yolo_engine.start()

    def ensure_device_ready(self, force: bool = False, restart_inference: bool = False) -> bool:
        recreate_device = False
        old_device = None
        should_stop_before_recreate = False
        with self._device_state_lock:
            if not force and self.device:
                ready = True
                should_stop = False
                should_start = restart_inference and hasattr(self, "yolo_engine") and not self.yolo_engine.running
            else:
                recreate_device = True
                old_device = getattr(self, "device", None)
                should_stop_before_recreate = (
                    force and hasattr(self, "yolo_engine") and self.yolo_engine.running
                )
        if should_stop_before_recreate:
            self.yolo_engine.stop()
        if recreate_device and old_device is not None:
            self._close_device(old_device)
        if recreate_device:
            new_device = self.create_device_instance()
            with self._device_state_lock:
                self.device = new_device
                if hasattr(self, "yolo_engine"):
                    self.yolo_engine.set_device(new_device)
                ready = bool(new_device)
                should_stop = hasattr(self, "yolo_engine") and self.yolo_engine.running and not ready
                should_start = restart_inference and hasattr(self, "yolo_engine") and ready and not self.yolo_engine.running
        if should_stop:
            self.yolo_engine.stop()
        if should_start:
            self.yolo_engine.start()
        return ready

    def _register_config_listening(self):

        def update_device(key, old, new):
            logger.warning(f"Reinitialize device......")
            suspend_task = self.task_queue.get_current_suspend_task()
            if suspend_task:
                self.task_queue.suspend_running_task()
            status = self.yolo_engine.running
            self.yolo_engine.stop()
            self.ensure_device_ready(force=True, restart_inference=status)
            if suspend_task: self.task_queue.resume_suspended_task()
            return

        self.config_service.add_listener([
            "base.run_mode",
            "base.game_window_name",
            "base.adb_connect_mode",
            "base.adb_host",
            "base.adb_port",
            "base.adb_serial",
            "base.android_screen_capture_service",
            "base.android_touch_service",
            "base.game_package_name"
        ], update_device)

    @property
    def latest_frame(self):
        return self.yolo_engine.latest_frame

    @property
    def latest_results(self):
        return self.yolo_engine.latest_results

    def create_device_instance(self) -> BaseDevice:
        """
        创建设备操作实例
        """
        mode = self.config_service.base.run_mode.lower()
        if mode == DeviceType.PC and not windows_pc_mode_is_available():
            reason = get_windows_unavailability_reason()
            if platform.system() == "Windows":
                raise RuntimeError(reason)
            logger.warning(f"{reason} 已自动回退到 Phone 模式。")
            self.config_service.get_config().base.run_mode.set("Phone", touch=False)
            mode = DeviceType.PHONE
        if mode == DeviceType.MAC_PLAYTOOLS:
            if not mac_playtools_mode_is_available():
                reason = get_mac_unavailability_reason()
                if platform.system() == "Darwin":
                    logger.warning(reason)
                    device = UnavailableDevice(reason, "mac_playtools_unavailable")
                    self._update_device_state(device)
                    return device
                logger.warning(f"{reason} 已自动回退到 Phone 模式。")
                self.config_service.get_config().base.run_mode.set("Phone", touch=False)
                mode = DeviceType.PHONE
            else:
                logger.debug("Initializing MacPlayTools device")
                try:
                    device = create_mac_device()
                except Exception as exc:
                    device = UnavailableDevice(f"MacPlayTools 设备初始化失败：{exc}", "mac_playtools_init_failed")
                if not device:
                    device = UnavailableDevice(
                        getattr(device, "get_unavailable_reason", lambda: "MacPlayTools 设备不可用。")(),
                        getattr(device, "get_unavailable_code", lambda: "mac_playtools_unavailable")(),
                    )
                self._update_device_state(device)
                return device
        if mode == DeviceType.PHONE:
            logger.debug("Initializing Android device")
            try:
                device = Android_App()
            except Exception as exc:
                device = UnavailableDevice(f"Android 设备初始化失败：{exc}", "android_init_failed")
            if not device:
                device = UnavailableDevice(
                    getattr(device, "get_unavailable_reason", lambda: "Android 设备不可用。")(),
                    getattr(device, "get_unavailable_code", lambda: "android_unavailable")(),
                )
            self._update_device_state(device)
            return device
        if mode == DeviceType.PC:
            logger.debug("Initializing Windows device")
            try:
                device = create_windows_device()
            except Exception as exc:
                device = UnavailableDevice(str(exc), "windows_device_unavailable")
            self._update_device_state(device)
            return device
        raise ValueError(f"Invalid device type: {mode}")

    def _send_frame_to_clients(self, latest_frame, latest_results):
        """将最新的图像的二进制数据发送给 WebSocket 客户端。"""
        if latest_frame is None:
            return
        # 获取图像尺寸
        height, width = latest_frame.shape[:2]
        annotated_frame = latest_frame
        if latest_results:
            result_obj = latest_results.results
            if hasattr(result_obj, "plot"):
                try:
                    annotated_frame = result_obj.plot()
                except Exception as e:
                    logger.warning(f"Failed to annotate inference result, fallback to raw frame: {e}")
        annotated_frame = self.debug_tools.draw_boxes(annotated_frame)
        # 本地预览直接使用 BMP，无损且编码开销比 JPEG 更低。
        _, encoded_frame = cv2.imencode(
            '.bmp',
            annotated_frame,
        )
        frame_bytes = encoded_frame.tobytes()
        self.ws_manager.broadcast_sync(WebSocketData(None, f"{width},{height}".encode('utf-8') + b"," + frame_bytes))

    def exec_task(self, task_name: str = None):
        return self.task_queue.start_queue(task_name)
