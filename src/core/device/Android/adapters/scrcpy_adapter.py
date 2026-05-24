import math
import socket
import threading
import time
from typing import Optional

import cv2
import numpy as np
from adbutils import AdbDevice, Network

from src.core.device.Android.adapters.scrcpy_control import (
    AMOTION_EVENT_ACTION_DOWN,
    AMOTION_EVENT_ACTION_MOVE,
    AMOTION_EVENT_ACTION_UP,
    serialize_touch_event,
)
from src.core.device.Android.adapters.scrcpy_resource import ScrcpyServerResource
from src.utils.logger import logger

CodecContext = None
_CODEC_IMPORT_ERROR = None


def _load_codec_context():
    global CodecContext, _CODEC_IMPORT_ERROR
    if CodecContext is not None:
        return CodecContext
    if _CODEC_IMPORT_ERROR is not None:
        return None
    try:
        from av.codec import CodecContext as av_codec_context
    except ImportError:
        return None
    CodecContext = av_codec_context
    return CodecContext


class ScrcpyAdapter:
    def __init__(
        self,
        adb_device: AdbDevice,
        max_width: int = 0,
        bitrate: int = 100_000_000,
        max_fps: int = 30,
        connection_timeout: int = 30000,
    ):
        self._adb_device = adb_device
        self._max_width = max_width
        self._bitrate = bitrate
        self._max_fps = max_fps
        self._connection_timeout = connection_timeout
        self._resource: Optional[ScrcpyServerResource] = None
        self._video_socket: Optional[socket.socket] = None
        self._control_socket: Optional[socket.socket] = None
        self._server_stream = None
        self._server_died = threading.Event()
        self._server_monitor_thread: Optional[threading.Thread] = None
        self._decoder_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._alive = False
        self._last_frame: Optional[np.ndarray] = None
        self._frame_size: Optional[tuple[int, int]] = None

    @property
    def available(self) -> bool:
        return _load_codec_context() is not None

    @property
    def alive(self) -> bool:
        return self._alive

    def start(self) -> bool:
        with self._state_lock:
            if self.alive:
                return True
            if not self.available:
                logger.warning("PyAV is not installed, fallback to ADB")
                return False

            self._resource = ScrcpyServerResource.discover()
            if self._resource is None:
                logger.warning("Official scrcpy-server not found in bin/, fallback to ADB")
                return False

            self.stop()
            try:
                self._deploy_server()
                self._video_socket = self._open_socket()
                self._control_socket = self._open_socket()
                self._alive = True
                self._decoder_thread = threading.Thread(
                    target=self._decode_video_stream,
                    name="scrcpy-video-decoder",
                    daemon=True,
                )
                self._decoder_thread.start()
                logger.debug(
                    f"scrcpy started from {self._resource.source}: "
                    f"{self._resource.server_path} (version {self._resource.version}, "
                    f"max_size={self._max_width}, bitrate={self._bitrate}, max_fps={self._max_fps})"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to start scrcpy adapter: {e}")
                self.stop()
                return False

    def stop(self):
        self._alive = False

        if self._server_monitor_thread is not None and self._server_monitor_thread.is_alive():
            self._server_monitor_thread.join(timeout=1.0)
        self._server_monitor_thread = None

        if self._video_socket is not None:
            try:
                self._video_socket.close()
            except Exception as e:
                logger.debug(f"Failed to close scrcpy video socket cleanly: {e}")
        self._video_socket = None

        if self._control_socket is not None:
            try:
                self._control_socket.close()
            except Exception as e:
                logger.debug(f"Failed to close scrcpy control socket cleanly: {e}")
        self._control_socket = None

        if self._server_stream is not None:
            try:
                self._server_stream.close()
            except Exception as e:
                logger.debug(f"Failed to stop scrcpy server cleanly: {e}")
        self._server_stream = None

        if self._decoder_thread is not None and self._decoder_thread.is_alive():
            self._decoder_thread.join(timeout=1.0)
        self._decoder_thread = None

    def capture(self, wait_timeout: float = 1.0) -> Optional[np.ndarray]:
        if not self.alive:
            return None

        deadline = time.time() + max(wait_timeout, 0)
        while self._last_frame is None and self.alive and time.time() < deadline:
            time.sleep(0.01)

        with self._frame_lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy()

    def click(self, x, y):
        if not self.alive:
            raise RuntimeError("scrcpy adapter is not running")

        x = int(x)
        y = int(y)
        self._send_touch_event(AMOTION_EVENT_ACTION_DOWN, x, y, pressure=1.0)
        time.sleep(0.01)
        self._send_touch_event(AMOTION_EVENT_ACTION_UP, x, y, pressure=0.0)

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.8):
        if not self.alive:
            raise RuntimeError("scrcpy adapter is not running")

        start_x, start_y = int(start_x), int(start_y)
        end_x, end_y = int(end_x), int(end_y)
        distance = max(abs(end_x - start_x), abs(end_y - start_y))
        steps = max(2, min(60, int(math.ceil(max(distance, 1) / 24))))
        if duration > 0:
            steps = max(steps, min(120, int(math.ceil(duration / 0.016))))

        step_delay = max(float(duration) / steps, 0.001)
        self._send_touch_event(AMOTION_EVENT_ACTION_DOWN, start_x, start_y, pressure=1.0)
        for index in range(1, steps):
            progress = index / steps
            move_x = round(start_x + (end_x - start_x) * progress)
            move_y = round(start_y + (end_y - start_y) * progress)
            self._send_touch_event(AMOTION_EVENT_ACTION_MOVE, move_x, move_y, pressure=1.0)
            time.sleep(step_delay)
        self._send_touch_event(AMOTION_EVENT_ACTION_UP, end_x, end_y, pressure=0.0)

    def _deploy_server(self):
        assert self._resource is not None

        remote_server_path = "/data/local/tmp/scrcpy-server.jar"
        self._adb_device.sync.push(str(self._resource.server_path), remote_server_path)

        commands = [
            f"CLASSPATH={remote_server_path}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            self._resource.version,
            "log_level=error",
            f"max_size={self._max_width}",
            f"max_fps={self._max_fps}",
            f"video_bit_rate={self._bitrate}",
            "video_codec=h264",
            "video=true",
            "audio=false",
            "control=true",
            "cleanup=false",
            "tunnel_forward=true",
            "stay_awake=false",
            "power_off_on_close=false",
            "clipboard_autosync=false",
            "raw_stream=true",
        ]
        self._server_died.clear()
        self._server_stream = self._adb_device.shell(commands, stream=True)
        self._server_monitor_thread = threading.Thread(
            target=self._monitor_server_stream,
            name="scrcpy-server-monitor",
            daemon=True,
        )
        self._server_monitor_thread.start()

    def _monitor_server_stream(self):
        stream = self._server_stream
        if stream is None:
            self._server_died.set()
            return
        try:
            while True:
                data = stream.recv(4096)
                if not data:
                    break
        except Exception as e:
            logger.debug(f"Scrcpy: 操作失败: {e}")

        self._server_died.set()

    def _open_socket(self) -> socket.socket:
        timeout_seconds = max(self._connection_timeout / 1000, 0.5)
        deadline = time.time() + timeout_seconds
        last_error = None
        while time.time() < deadline:
            if self._server_died.is_set():
                raise ConnectionError("scrcpy server process exited before socket was ready")
            try:
                sock = self._adb_device.create_connection(Network.LOCAL_ABSTRACT, "scrcpy")
                sock.settimeout(1.0)
                return sock
            except Exception as e:
                last_error = e
                time.sleep(0.05)

        raise ConnectionError(f"Failed to connect to scrcpy socket: {last_error}")

    def _decode_video_stream(self):
        codec_context = _load_codec_context()
        if codec_context is None:
            logger.warning("PyAV is not installed, scrcpy video decoder cannot start")
            self._alive = False
            return
        codec = codec_context.create("h264", "r")
        while self.alive and self._video_socket is not None:
            try:
                raw_h264 = self._video_socket.recv(0x10000)
                if not raw_h264:
                    raise ConnectionError("scrcpy video stream disconnected")

                packets = codec.parse(raw_h264)
                for packet in packets:
                    frames = codec.decode(packet)
                    for frame in frames:
                        image = self._frame_to_bgr(frame)
                        with self._frame_lock:
                            self._last_frame = image
                            self._frame_size = (image.shape[1], image.shape[0])
            except socket.timeout:
                continue
            except Exception as e:
                if self.alive:
                    logger.warning(f"scrcpy video decoder stopped: {e}")
                self._alive = False
                break

    @staticmethod
    def _frame_to_bgr(frame) -> np.ndarray:
        # Decode to RGB first, then convert explicitly for OpenCV to avoid channel ambiguity.
        rgb_image = frame.to_ndarray(format="rgb24")
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    def _send_touch_event(self, action: int, x: int, y: int, pressure: float):
        if self._control_socket is None:
            raise RuntimeError("scrcpy control socket is not connected")

        width, height = self._get_screen_size()
        payload = serialize_touch_event(
            x=x,
            y=y,
            width=width,
            height=height,
            action=action,
            pressure=pressure,
        )
        with self._control_lock:
            self._control_socket.sendall(payload)

    def _get_screen_size(self) -> tuple[int, int]:
        if self._frame_size is not None:
            return self._frame_size
        width, height = self._adb_device.window_size()
        return int(width), int(height)
