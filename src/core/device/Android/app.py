import random
import re
import time
from typing import Optional

import adbutils
import cv2
import numpy as np
import requests
import uiautomator2 as u2

from src.constants.device.adb import ADBConnectMode, ADBOperation
from src.core.device.Android.adapters import ScrcpyAdapter, MinitouchAdapter, MaaTouchAdapter
from src.core.services.config_service import ConfigService
from src.entity.BaseDevice import BaseDevice
from src.utils.adb_runtime import describe_adb_error
from src.utils.adb_tools import start_DroidCast, ADBShell
from src.utils.i18n_tools import I18nText, i18n_text
from src.utils.debug_tools import DebugTools
from src.utils.logger import logger
from src.utils.performance_tools import timeit
from src.utils.task_debug_tools import (
    record_task_click,
    record_task_scroll,
    record_task_swipe,
)


debugger = DebugTools()

class Android_App(BaseDevice):
    _FOREGROUND_ACTIVITY_PATTERNS = (
        r"topResumedActivity[:=]\s*ActivityRecord\{[^\}]*\s+[^\s]+\s+(?P<package>[A-Za-z0-9._]+)\/(?P<activity>[^\s\}]+)",
        r"mResumedActivity[:=]\s*ActivityRecord\{[^\}]*\s+[^\s]+\s+(?P<package>[A-Za-z0-9._]+)\/(?P<activity>[^\s\}]+)",
        r"ResumedActivity[:=]\s*ActivityRecord\{[^\}]*\s+[^\s]+\s+(?P<package>[A-Za-z0-9._]+)\/(?P<activity>[^\s\}]+)",
        r"mResumeActivity[:=]\s*ActivityRecord\{[^\}]*\s+[^\s]+\s+(?P<package>[A-Za-z0-9._]+)\/(?P<activity>[^\s\}]+)",
        r"mFocusedApp=.*?(?P<package>[A-Za-z0-9._]+)\/(?P<activity>[^\s\}]+)",
        r"mCurrentFocus=.*?(?P<package>[A-Za-z0-9._]+)\/(?P<activity>[^\s\}]+)",
    )
    __config_service: ConfigService
    __adb_host: str
    __adb_port: int
    __adb_device: adbutils.AdbDevice | None = None
    __u2_device: Optional[u2.Device] = None
    __package_name: str
    __connect_mode: str
    __screen_capture_service: str
    __screen_touch_service: str
    __droidcast_service_status: bool = False
    __capture_service_shell: ADBShell | None = None
    __scrcpy_adapter: Optional[ScrcpyAdapter] = None
    __minitouch_adapter: Optional[MinitouchAdapter] = None
    __maatouch_adapter: Optional[MaaTouchAdapter] = None
    __unavailable_reason: str = ""
    __unavailable_code: str = ""

    def __init__(self) -> None:
        super().__init__()
        self.__config_service = ConfigService()
        self.__package_name = self.__config_service.base.game_package_name
        self.__connect_mode = self.__config_service.base.adb_connect_mode
        self.__adb_serial = self.__config_service.base.adb_serial
        self.__adb_host = self.__config_service.base.adb_host
        self.__adb_port = self.__config_service.base.adb_port
        self.__screen_capture_service = self.__config_service.base.android_screen_capture_service
        self.__screen_touch_service = self.__config_service.base.android_touch_service
        if not self.__connect_ADB():
            return
        self.__init_capture_service()
        self.__init_touch_service()
        if (
            self.__screen_capture_service == ADBOperation.ScreenCaptureService.uiautomator2
            or self.__screen_touch_service ==  ADBOperation.TouchService.uiautomator2
        ):
            self.__connect_uiautomator2()

    def __del__(self) -> None:
        self.close()

    def close(self):
        self._reset_runtime_services()
        self.__adb_device = None

    def __bool__(self) -> bool:
        return bool(self.__adb_device)

    def get_unavailable_reason(self) -> I18nText | str:
        return self.__unavailable_reason

    def get_unavailable_code(self) -> str:
        return self.__unavailable_code

    def _set_unavailable(self, code: str, reason: I18nText | str) -> bool:
        self.__unavailable_code = code
        self.__unavailable_reason = reason
        logger.warning(reason)
        return False

    def _get_adb_error_context(self) -> dict:
        if self.__connect_mode == ADBConnectMode.NETWORK:
            return {
                "connect_mode": "Network",
                "host": self.__adb_host,
                "port": self.__adb_port,
            }
        return {
            "connect_mode": "USB",
            "serial": self.__adb_serial,
        }

    def _reset_runtime_services(self):
        if self.__maatouch_adapter is not None:
            try:
                self.__maatouch_adapter.stop()
            except Exception:  # 清理阶段忽略停止失败
                pass
            self.__maatouch_adapter = None
        if self.__minitouch_adapter is not None:
            try:
                self.__minitouch_adapter.stop()
            except Exception:  # 清理阶段忽略停止失败
                pass
            self.__minitouch_adapter = None
        if self.__scrcpy_adapter is not None:
            try:
                self.__scrcpy_adapter.stop()
            except Exception:  # 清理阶段忽略停止失败
                pass
            self.__scrcpy_adapter = None
        if self.__capture_service_shell is not None:
            try:
                self.__capture_service_shell.close()
            except Exception:  # 清理阶段忽略关闭失败
                pass
            self.__capture_service_shell = None
        self.__droidcast_service_status = False
        self.__u2_device = None

    def _handle_runtime_adb_error(self, exc: Exception, action: str) -> None:
        """处理运行时 ADB 异常并保留可翻译的嵌套错误信息。"""
        code, reason = describe_adb_error(exc, **self._get_adb_error_context())
        self._reset_runtime_services()
        self.__adb_device = None
        self._set_unavailable(
            code,
            i18n_text(
                "backend.adb.initFailed",
                fallback=f"{action}失败：{reason}",
                message=reason,
            ),
        )
        raise RuntimeError(str(self.__unavailable_reason)) from exc

    def _verify_adb_transport(self) -> bool:
        if self.__adb_device is None:
            return self._set_unavailable(
                "adb_not_connected",
                i18n_text("backend.adb.notConnected", fallback="当前未连接到 ADB 设备。"),
            )
        try:
            self.__adb_device.shell("echo connected")
        except Exception as exc:
            code, reason = describe_adb_error(exc, **self._get_adb_error_context())
            self.__adb_device = None
            return self._set_unavailable(code, reason)
        self.__unavailable_code = ""
        self.__unavailable_reason = i18n_text("backend.adb.noError", fallback="")
        return True

    def __connect_ADB(self) -> bool:
        if self.__connect_mode == ADBConnectMode.USB:
            try:
                usb_devices = [dev.serial for dev in adbutils.adb.device_list()]
            except Exception as exc:
                code, reason = describe_adb_error(exc, connect_mode="USB", serial=self.__adb_serial)
                return self._set_unavailable(code, reason)
            if not self.__adb_serial and len(usb_devices) == 1:
                self.__adb_serial = usb_devices[0]
                logger.info(f"Auto selected USB ADB device: {self.__adb_serial}")
            if self.__adb_serial not in usb_devices:
                code, reason = describe_adb_error(
                    RuntimeError("USB ADB device not found"),
                    connect_mode="USB",
                    serial=self.__adb_serial,
                )
                return self._set_unavailable(code, reason)
            logger.debug(f"Try connect ADB(serial: {self.__adb_serial})")
            try:
                self.__adb_device = adbutils.adb.device(serial=self.__adb_serial)
            except Exception as e:
                code, reason = describe_adb_error(e, connect_mode="USB", serial=self.__adb_serial)
                return self._set_unavailable(code, reason)
            return self._verify_adb_transport()
        elif self.__connect_mode == ADBConnectMode.NETWORK:
            self._adb_host = self.__config_service.base.adb_host
            self._adb_port = self.__config_service.base.adb_port
            logger.debug(f"Try connect ADB(host: {self._adb_host}, port: {self._adb_port})......")
            try:
                adbutils.adb.connect(f"{self.__adb_host}:{self.__adb_port}")
                self.__adb_device = adbutils.adb.device(f"{self.__adb_host}:{self.__adb_port}")
            except Exception as e:
                code, reason = describe_adb_error(
                    e,
                    connect_mode="Network",
                    host=self.__adb_host,
                    port=self.__adb_port,
                )
                return self._set_unavailable(code, reason)
            return self._verify_adb_transport()
        else:
            return self._set_unavailable(
                "invalid_connect_mode",
                i18n_text(
                    "backend.adb.invalidConnectMode",
                    fallback=f"无效的 ADB 连接模式：{self.__connect_mode}",
                    mode=self.__connect_mode,
                ),
            )

    def __connect_uiautomator2(self):
        logger.debug("Try connect to UIAutomator2...")
        try:
            self.__u2_device = u2.connect(self.__adb_device.serial)
        except Exception as e:
            logger.warning(f"uiautomator2 Initial Error：\n{e}")
            self.__u2_device = None

    def __ensure_scrcpy_adapter(self) -> bool:
        if self.__scrcpy_adapter is None:
            self.__scrcpy_adapter = ScrcpyAdapter(self.__adb_device)
        return self.__scrcpy_adapter.start()

    def __ensure_minitouch_adapter(self) -> bool:
        if self.__minitouch_adapter is None:
            self.__minitouch_adapter = MinitouchAdapter(self.__adb_device)
        return self.__minitouch_adapter.start()

    def __ensure_maatouch_adapter(self) -> bool:
        if self.__maatouch_adapter is None:
            self.__maatouch_adapter = MaaTouchAdapter(self.__adb_device)
        return self.__maatouch_adapter.start()

    def __init_capture_service(self):
        match self.__screen_capture_service:
            case ADBOperation.ScreenCaptureService.ADB:
                pass
            case ADBOperation.ScreenCaptureService.uiautomator2:
                pass
            case ADBOperation.ScreenCaptureService.DroidCast:
                self.__droidcast_service_status, self.__capture_service_shell = start_DroidCast(self.__adb_device)
            case ADBOperation.ScreenCaptureService.scrcpy:
                if not self.__ensure_scrcpy_adapter():
                    logger.warning("scrcpy capture service unavailable, fallback to ADB")
                    self.__screen_capture_service = ADBOperation.ScreenCaptureService.ADB
            case _:
                logger.warning(f"Not support capture service: '{self.__screen_capture_service}', reverted to ADB")
                self.__screen_capture_service = ADBOperation.ScreenCaptureService.ADB

    def __init_touch_service(self):
        match self.__screen_touch_service:
            case ADBOperation.TouchService.ADB:
                pass
            case ADBOperation.TouchService.uiautomator2:
                pass
            case ADBOperation.TouchService.scrcpy:
                if not self.__ensure_scrcpy_adapter():
                    logger.warning("scrcpy touch service unavailable, fallback to ADB")
                    self.__screen_touch_service = ADBOperation.TouchService.ADB
            case ADBOperation.TouchService.maatouch:
                if not self.__ensure_maatouch_adapter():
                    logger.warning("MaaTouch service unavailable, fallback to ADB")
                    self.__screen_touch_service = ADBOperation.TouchService.ADB
            case ADBOperation.TouchService.minitouch:
                if not self.__ensure_minitouch_adapter():
                    logger.warning("minitouch service unavailable, fallback to ADB")
                    self.__screen_touch_service = ADBOperation.TouchService.ADB
            case _:
                logger.warning(f"Not support touch service: '{self.__screen_touch_service}', reverted to ADB")
                self.__screen_touch_service = ADBOperation.TouchService.ADB

    def _adb_shell(self, command: str) -> str:
        """
        执行一次 ADB shell 命令。

        该 helper 仅用于“尽力获取状态”的只读查询；
        失败时返回空字符串，而不是把临时查询异常向上传播。
        """
        try:
            return self.__adb_device.shell(command) or ""
        except Exception as exc:
            logger.debug(f"ADB shell failed for '{command}': {exc}")
            return ""

    def _iter_foreground_targets(self):
        """
        枚举当前前台候选包名/Activity。

        优先读取 uiautomator2 的当前应用信息，
        再 fallback 到 dumpsys activity/window 输出。
        """
        if self.__u2_device is not None:
            try:
                current_app = self.__u2_device.app_current()
                current_app = current_app or {}
                package = current_app.get("package")
                activity = current_app.get("activity") or ""
                if package:
                    yield package, activity
            except Exception as exc:
                logger.debug(f"uiautomator2 app_current failed: {exc}")

            try:
                package = self.__u2_device.info.get("currentPackageName")
                if package:
                    yield package, ""
            except Exception as exc:
                logger.debug(f"uiautomator2 currentPackageName failed: {exc}")

        for command in (
                "dumpsys activity activities",
                "dumpsys window windows",
        ):
            output = self._adb_shell(command)
            if not output:
                continue
            for pattern in self._FOREGROUND_ACTIVITY_PATTERNS:
                for match in re.finditer(pattern, output):
                    yield match.group("package"), match.group("activity")

    def start_game(self):
        """启动游戏"""
        self.__adb_device.app_start(self.__package_name)
        TIMEOUT = 30
        COUNT = 0
        while True:
            if COUNT > TIMEOUT:
                raise TimeoutError("Waiting start game timeout")
            if self.is_app_focused():
                break
            else:
                time.sleep(1)
                COUNT += 1

    @timeit
    def is_app_focused(self) -> bool:
        """判断游戏是否在前台"""
        for package, _activity in self._iter_foreground_targets():
            if package == self.__package_name:
                return True
        return False

    @timeit
    def is_app_running(self) -> bool:
        """判断游戏是否在运行"""
        if self.__u2_device is None:
            return self._adb_shell(f"pidof {self.__package_name}").strip() != ""
        return self.__package_name in self.__u2_device.app_list_running()

    @timeit
    def get_window_size(self):
        if not self.__u2_device:
            return self.__adb_device.window_size()
        return self.__u2_device.window_size()

    def get_diagnostics(self) -> dict:
        return {
            "device_type": "android",
            "connect_mode": self.__connect_mode,
            "serial": getattr(self.__adb_device, "serial", ""),
            "package_name": self.__package_name,
            "capture_service": self.__screen_capture_service,
            "touch_service": self.__screen_touch_service,
            "android": {
                "model": self._adb_shell("getprop ro.product.model").strip(),
                "brand": self._adb_shell("getprop ro.product.brand").strip(),
                "device": self._adb_shell("getprop ro.product.device").strip(),
                "android_release": self._adb_shell("getprop ro.build.version.release").strip(),
                "sdk_int": self._adb_shell("getprop ro.build.version.sdk").strip(),
            },
        }

    def __get_touch_service(self):
        match self.__screen_touch_service:
            case ADBOperation.TouchService.uiautomator2:
                if self.__u2_device is not None:
                    return self.__u2_device
                return self.__adb_device
            case ADBOperation.TouchService.scrcpy:
                if self.__ensure_scrcpy_adapter():
                    return self.__scrcpy_adapter
                logger.warning("scrcpy touch service unavailable, fallback to ADB for current action")
                return self.__adb_device
            case ADBOperation.TouchService.maatouch:
                if self.__ensure_maatouch_adapter():
                    return self.__maatouch_adapter
                logger.warning("MaaTouch service unavailable, fallback to ADB for current action")
                return self.__adb_device
            case ADBOperation.TouchService.minitouch:
                if self.__ensure_minitouch_adapter():
                    return self.__minitouch_adapter
                logger.warning("minitouch service unavailable, fallback to ADB for current action")
                return self.__adb_device
            case ADBOperation.TouchService.ADB:
                return self.__adb_device
            case _:
                logger.warning(f"Not support touch service: '{self.__screen_touch_service}', reverted to ADB")
                self.__screen_touch_service = ADBOperation.TouchService.ADB
                return self.__adb_device

    def _get_touch_service_with_retry(self):
        service = self.__get_touch_service()
        if service is not None:
            return service

        logger.warning("Touch service unavailable, reinitializing Android touch runtime once")
        self._reset_runtime_services()
        if not self.__connect_ADB():
            raise RuntimeError(self.__unavailable_reason or i18n_text("backend.adb.reconnectFailed", fallback="重新连接 ADB 设备失败"))
        self.__init_capture_service()
        self.__init_touch_service()
        if (
            self.__screen_capture_service == ADBOperation.ScreenCaptureService.uiautomator2
            or self.__screen_touch_service == ADBOperation.TouchService.uiautomator2
        ):
            self.__connect_uiautomator2()
        service = self.__get_touch_service()
        if service is None:
            self._set_unavailable("touch_service_unavailable", i18n_text(
                    "backend.adb.touchServiceUnavailable",
                    fallback="点击失败：触摸服务不可用",
                ))
            raise RuntimeError(self.__unavailable_reason)
        return service

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.8,
              offset_x=10, offset_y=10, safe_margin=50, hold_end=0.0, ease=None):
        """
        基础滑动方法：执行带安全检查和随机偏移的单次滑动
        :param start_x: 起始X
        :param start_y: 起始Y
        :param end_x: 结束X
        :param end_y: 结束Y
        :param duration: 滑动总时长，默认0.8秒
        :param offset_x: 随机偏移x值
        :param offset_y: 随机偏移y值
        :param safe_margin: 安全边距
        :param hold_end: 到达终点后保持不动的时间（秒），用于消除游戏惯性滑动
        :param ease: 缓动函数名（"out_quad" 等），None 为线性
        """
        width, height = self.get_window_size()
        def clamp(val, max_val):
            return max(safe_margin, min(max_val - safe_margin, val))
        safe_start_x = clamp(start_x, width)
        safe_start_y = clamp(start_y, height)
        safe_end_x = clamp(end_x, width)
        safe_end_y = clamp(end_y, height)
        debugger.add_line(
            safe_start_x,
            safe_start_y,
            safe_end_x,
            safe_end_y,
            color=(0, 255, 0),   # 绿色滑动轨迹
            thickness=3,
            duration=duration+1
        )
        debugger.add_point(
            safe_start_x,
            safe_start_y,
            color=(255, 255, 0), # 起点黄
            radius=6,
            duration=duration+1
        )
        debugger.add_point(
            safe_end_x,
            safe_end_y,
            color=(255, 0, 0),   # 终点红
            radius=6,
            duration=duration+1
        )
        # 模拟人类滑动的微小随机偏移 (轨迹随机化)
        offset_x = random.randint(0-offset_x, offset_x)
        offset_y = random.randint(0-offset_y, offset_y)
        # 将 duration 稍微随机化，避免死板的固定时长
        actual_duration = duration * random.uniform(0.9, 1.1)
        service = self._get_touch_service_with_retry()
        try:
            service.swipe(
                safe_start_x + offset_x,
                safe_start_y + offset_y,
                safe_end_x,
                safe_end_y,
                actual_duration,
                hold_end=hold_end,
                ease=ease,
            )
        except TypeError:
            # ADB / uiautomator2 等不支持 hold_end / ease 参数
            service.swipe(
                safe_start_x + offset_x,
                safe_start_y + offset_y,
                safe_end_x,
                safe_end_y,
                actual_duration,
            )
        record_task_swipe(
            safe_start_x + offset_x,
            safe_start_y + offset_y,
            safe_end_x,
            safe_end_y,
            duration=round(float(actual_duration), 3),
            source=f"android.{self.__screen_touch_service}",
        )
        # 增加随机短暂停顿 (模拟人类自然停顿)
        time.sleep(random.uniform(0.05, 0.1))

    def _scroll(self, x, y, direction, scroll_delta):
        """
        通用滚动方法，调用提取出的 swipe
        """
        record_task_scroll(
            x,
            y,
            direction=str(direction),
            delta=int(scroll_delta),
            source="android",
        )
        width, height = self.get_window_size()
        scroll_distance = int(height * 0.05)
        scroll_sign = 1 if scroll_delta > 0 else -1

        for _ in range(abs(scroll_delta)):
            # 计算滑动因子和当前距离
            scroll_factor = random.uniform(0.8, 1.2) if random.random() > 0.2 else random.uniform(1, 1.5)
            current_dist = int(scroll_distance * scroll_factor)

            if direction == ADBOperation.ScrollDirection.HORIZONTAL:
                end_x = x + (scroll_sign * current_dist)
                DebugTools().add_line(
                    x, y, x, end_x,
                    color=(100, 200, 255),
                    thickness=2,
                    duration=1.0
                )
                self.swipe(x, y, end_x, y, duration=random.uniform(0.1, 0.2))

            elif direction == ADBOperation.ScrollDirection.VERTICAL:
                end_y = y + (scroll_sign * current_dist)
                DebugTools().add_line(
                    x, y, x, end_y,
                    color=(100, 200, 255),
                    thickness=2,
                    duration=1.0
                )
                self.swipe(x, y, x, end_y, duration=random.uniform(0.1, 0.2))
            else:
                raise ValueError(f"Invalid direction: {direction}")

    def scrollY(self, x, y, scroll_delta):
        """纵向滚动（向上/向下滑动）"""
        self._scroll(x, y, ADBOperation.ScrollDirection.VERTICAL, scroll_delta)

    def scrollX(self, x, y, scroll_delta):
        self._scroll(x, y, ADBOperation.ScrollDirection.HORIZONTAL, scroll_delta)

    def click(self, x, y, el_label=""):
        """点击指定坐标"""
        debugger.add_crosshair(
            x, y,
            size=25,
            color=(255, 0, 0),
            thickness=1
        )
        service = self._get_touch_service_with_retry()
        try:
            service.click(x, y)
        except Exception as exc:
            self._handle_runtime_adb_error(exc, "点击")
        record_task_click(x, y, label=el_label, source="android")

    def capture(self) -> Optional[np.ndarray]:
        if not self.__bool__():
            return None
        try:
            match self.__screen_capture_service:
                case ADBOperation.ScreenCaptureService.ADB:
                    return cv2.cvtColor(np.asarray(self.__adb_device.screenshot()), cv2.COLOR_RGB2BGR)
                case ADBOperation.ScreenCaptureService.DroidCast:
                    if not self.__droidcast_service_status:
                        return cv2.cvtColor(np.asarray(self.__adb_device.screenshot()), cv2.COLOR_RGB2BGR)
                    response = requests.get("http://127.0.0.1:53516/screenshot")
                    if response.status_code != 200:
                        self.__init_capture_service()
                        return cv2.cvtColor(np.asarray(self.__adb_device.screenshot()), cv2.COLOR_RGB2BGR)
                    image_array = np.frombuffer(response.content, dtype=np.uint8)
                    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                case ADBOperation.ScreenCaptureService.uiautomator2:
                    return self.__u2_device.screenshot(format='opencv')
                case ADBOperation.ScreenCaptureService.scrcpy:
                    if self.__ensure_scrcpy_adapter():
                        frame = self.__scrcpy_adapter.capture(wait_timeout=1.0)
                        if frame is not None:
                            return frame
                    logger.warning("scrcpy frame unavailable, fallback to ADB screenshot for current capture")
                    return cv2.cvtColor(np.asarray(self.__adb_device.screenshot()), cv2.COLOR_RGB2BGR)
                case _:
                    logger.warning(f"Undefined screenshot service {self.__screen_capture_service}, reverted to ADB")
                    return cv2.cvtColor(np.asarray(self.__adb_device.screenshot()), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self._handle_runtime_adb_error(exc, "截图")
            return None
