import random
import subprocess
import time
from typing import Optional

import numpy as np

from src.core.device.MacOS.playtools_adapter import MacPlayToolsAdapter
from src.core.services.config_service import ConfigService
from src.entity.BaseDevice import BaseDevice
from src.utils.debug_tools import DebugTools
from src.utils.i18n_tools import I18nText, i18n_text
from src.utils.logger import logger
from src.utils.task_debug_tools import (
    record_task_click,
    record_task_scroll,
    record_task_swipe,
)

debugger = DebugTools()


class MacOS_App(BaseDevice):
    """
    macOS PlayCover/PlayTools 设备适配器。
    通过 MaaTools TCP 协议与 PlayCover 中运行的 iOS 应用通信，
    实现截屏、点击、滑动等操作。
    """

    _host: str = "localhost"
    _port: int = 0
    _adapter: Optional[MacPlayToolsAdapter] = None
    _unavailable_reason: I18nText | str = ""
    _unavailable_code: str = ""

    def __init__(self, host: str = "localhost", port: int = 0):
        self._host = host
        self._port = port
        self._adapter = None
        self._unavailable_reason: I18nText | str = ""
        self._unavailable_code = ""

        config_service = ConfigService()
        if port == 0:
            port = int(config_service.base.playtools_port or 0)
        if port == 0:
            raise ValueError(
                i18n_text(
                    "backend.device.mac.unavailable.port_not_configured",
                    fallback="MacPlayTools 端口未配置。请在 PlayCover 中启动游戏后，从窗口标题栏获取 [localhost:端口号] 并填入配置。",
                )
            )

        self._host = host
        self._port = port
        self._adapter = MacPlayToolsAdapter(host, port)
        self._unavailable_reason = ""
        self._unavailable_code = ""

        if not self._adapter.connect():
            self._unavailable_reason = i18n_text(
                "backend.device.mac.unavailable.connect_failed",
                fallback=f"无法连接到 MacPlayTools ({host}:{port})。请确认 PlayCover 中的游戏已启动且 MaaTools 已启用。",
                host=host,
                port=port,
            )
            self._unavailable_code = "playtools_connect_failed"
            logger.warning(self._unavailable_reason)

    def __del__(self):
        self.close()

    def close(self):
        adapter = self._adapter
        if adapter is not None:
            adapter.disconnect()
            self._adapter = None

    def __bool__(self) -> bool:
        return bool(self._adapter and self._adapter.connected)

    def get_unavailable_reason(self) -> I18nText | str:
        return self._unavailable_reason

    def get_unavailable_code(self) -> str:
        return self._unavailable_code

    # ── 应用状态 ──────────────────────────────────────────────

    def is_app_focused(self) -> bool:
        adapter = self._adapter
        if adapter is None or not adapter.connected:
            return False
        bundle_id = adapter.get_bundle_id()
        if not bundle_id:
            return False
        # 通过 lsappinfo 检查前台应用是否包含目标 bundle id
        try:
            result = subprocess.run(
                ["lsappinfo", "info", "-only", "bundleID", "-app", "front"],
                capture_output=True, text=True, timeout=5
            )
            return bundle_id in result.stdout
        except Exception:
            # 无法确认前台, 只要连接正常就假定在前台
            return adapter.connected

    def is_app_running(self) -> bool:
        return bool(self._adapter and self._adapter.connected)

    def start_game(self):
        logger.info("MacPlayTools 模式下请手动通过 PlayCover 启动游戏")

    def bring_to_front(self):
        adapter = self._adapter
        if adapter is None or not adapter.connected:
            return
        bundle_id = adapter.get_bundle_id()
        if bundle_id:
            try:
                subprocess.run(
                    ["open", "-b", bundle_id],
                    capture_output=True, timeout=5
                )
            except Exception as e:
                logger.debug(f"bring_to_front 失败: {e}")

    # ── 屏幕尺寸 ──────────────────────────────────────────────

    def get_window_size(self):
        adapter = self._adapter
        if adapter is None:
            return 0, 0
        w, h = adapter.screen_size
        if w == 0 or h == 0:
            logger.warning("MacPlayTools 屏幕尺寸未知")
        return w, h

    def get_diagnostics(self) -> dict:
        adapter = self._adapter
        bundle_id = adapter.get_bundle_id() if adapter is not None else ""
        return {
            "device_type": "mac_playtools",
            "host": self._host,
            "port": self._port,
            "bundle_id": bundle_id,
            "capture_method": "playtools.screencap_bgr/rgbx",
            "touch_method": "playtools.click/swipe",
        }

    # ── 截屏 ──────────────────────────────────────────────────

    def capture(self) -> Optional[np.ndarray]:
        adapter = self._adapter
        if adapter is None or not adapter.connected:
            return None
        # 优先使用 BGR 截屏 (MaaTools v3)，减少一次颜色空间转换
        frame = adapter.screencap_bgr()
        if frame is not None:
            return frame
        # 回退到 RGBX 截屏
        frame = adapter.screencap_rgbx()
        return frame

    # ── 触控操作 ──────────────────────────────────────────────

    def click(self, x, y, el_label=""):
        debugger.add_crosshair(
            x, y,
            size=25,
            color=(255, 0, 0),
            thickness=1
        )
        self._adapter.click(int(x), int(y))
        record_task_click(x, y, label=el_label, source="mac_playtools")

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.8,
              offset_x=10, offset_y=10, safe_margin=50, hold_end=0.0, ease=None):
        width, height = self.get_window_size()
        if width == 0 or height == 0:
            self._adapter.swipe(int(start_x), int(start_y), int(end_x), int(end_y),
                                duration, hold_end=hold_end, ease=ease)
            return

        def clamp(val, max_val):
            return max(safe_margin, min(max_val - safe_margin, val))

        safe_start_x = clamp(start_x, width)
        safe_start_y = clamp(start_y, height)
        safe_end_x = clamp(end_x, width)
        safe_end_y = clamp(end_y, height)

        debugger.add_line(
            safe_start_x, safe_start_y, safe_end_x, safe_end_y,
            color=(0, 255, 0), thickness=3, duration=duration + 1
        )
        debugger.add_point(safe_start_x, safe_start_y, color=(255, 255, 0), radius=6, duration=duration + 1)
        debugger.add_point(safe_end_x, safe_end_y, color=(255, 0, 0), radius=6, duration=duration + 1)

        offset_x = random.randint(-offset_x, offset_x)
        offset_y = random.randint(-offset_y, offset_y)
        actual_duration = duration * random.uniform(0.9, 1.1)

        self._adapter.swipe(
            safe_start_x + offset_x, safe_start_y + offset_y,
            safe_end_x, safe_end_y, actual_duration,
            hold_end=hold_end, ease=ease,
        )
        record_task_swipe(
            safe_start_x + offset_x,
            safe_start_y + offset_y,
            safe_end_x,
            safe_end_y,
            duration=round(float(actual_duration), 3),
            source="mac_playtools",
        )
        time.sleep(random.uniform(0.05, 0.1))

    def _scroll(self, x, y, direction, scroll_delta):
        record_task_scroll(
            x,
            y,
            direction=str(direction),
            delta=int(scroll_delta),
            source="mac_playtools",
        )
        width, height = self.get_window_size()
        scroll_distance = int(height * 0.05) if height > 0 else 50
        scroll_sign = 1 if scroll_delta > 0 else -1

        for _ in range(abs(scroll_delta)):
            scroll_factor = random.uniform(0.8, 1.2) if random.random() > 0.2 else random.uniform(1, 1.5)
            current_dist = int(scroll_distance * scroll_factor)

            if direction == "horizontal":
                end_x = x + (scroll_sign * current_dist)
                self.swipe(x, y, end_x, y, duration=random.uniform(0.1, 0.2))
            elif direction == "vertical":
                end_y = y + (scroll_sign * current_dist)
                self.swipe(x, y, x, end_y, duration=random.uniform(0.1, 0.2))

    def scrollY(self, x, y, scroll_delta):
        self._scroll(x, y, "vertical", scroll_delta)

    def scrollX(self, x, y, scroll_delta):
        self._scroll(x, y, "horizontal", scroll_delta)
