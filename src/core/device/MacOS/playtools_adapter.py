"""
MacPlayTools 适配器 —— 通过 TCP 与 PlayCover/PlayTools (MaaTools) 通信，
实现 macOS 上对 iOS 应用的截屏和触控操作。

协议版本: MaaTools v3
传输层: TCP，连接地址格式 localhost:<port>

握手:
    客户端 → 服务端: b'MAA\x00'  (4 字节)
    服务端 → 客户端: b'OKAY'     (4 字节)

消息帧:
    客户端 → 服务端: [u16 big-endian payload长度] + [payload]
    服务端 → 客户端: 根据命令不同，格式各异（见下方命令说明）

命令 (payload 前 4 字节):
    SCRN  (0x5343524e) - 截屏 (RGBX)
        响应: [u32 数据长度] + [RGBX 原始像素数据]
    BGR\x01 (0x42475201) - 截屏 (BGR)
        响应: [u32 宽度] + [u32 高度] + [u32 数据长度] + [BGR 原始像素数据]
    SIZE  (0x53495a45) - 获取屏幕尺寸
        响应: [u16 宽度] + [u16 高度]
    TUCH  (0x54554348) - 触控事件
        payload: [4字节命令] + [1字节phase] + [u16 x] + [u16 y]
        phase: 0=down, 1=move, 3=up
        无响应
    VERN  (0x5645524e) - 获取版本号
        响应: [u32 版本号]
    BNDL  (0x424e444c) - 获取 Bundle ID
        响应: [u32 长度] + [UTF-8 字符串]
    RECT  (0x52454354) - 获取窗口矩形
        响应: 8 个 u16 值 (frame.x, frame.y, frame.w, frame.h, content.x, content.y, content.w, content.h)
    TERM  (0x5445524d) - 终止应用
        无响应
"""

import math
import socket
import struct
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import logger


class MacPlayToolsAdapter:
    """通过 MaaTools v3 TCP 协议与 PlayCover 中运行的 iOS 应用通信。"""

    HANDSHAKE_MAGIC = b"MAA\x00"
    HANDSHAKE_REPLY = b"OKAY"

    CMD_SCREENCAP = b"SCRN"
    CMD_BGR_SCREENCAP = b"BGR\x01"
    CMD_SIZE = b"SIZE"
    CMD_TOUCH = b"TUCH"
    CMD_VERSION = b"VERN"
    CMD_BUNDLE = b"BNDL"
    CMD_RECT = b"RECT"
    CMD_TERMINATE = b"TERM"

    TOUCH_DOWN = 0
    TOUCH_MOVE = 1
    TOUCH_UP = 3

    def __init__(self, host: str = "localhost", port: int = 0):
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False
        self._screen_width = 0
        self._screen_height = 0
        self._version = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._screen_width, self._screen_height

    @property
    def version(self) -> int:
        return self._version

    # ── 连接管理 ──────────────────────────────────────────────

    def connect(self, timeout: float = 5.0) -> bool:
        with self._lock:
            self.disconnect()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((self._host, self._port))

                # 握手
                sock.sendall(self.HANDSHAKE_MAGIC)
                reply = self._recv_exact(sock, 4)
                if reply != self.HANDSHAKE_REPLY:
                    logger.warning(f"MacPlayTools 握手失败: 期望 {self.HANDSHAKE_REPLY!r}, 收到 {reply!r}")
                    sock.close()
                    return False

                self._sock = sock
                self._connected = True

                # 查询版本和屏幕尺寸
                self._version = self._query_version()
                self._screen_width, self._screen_height = self._query_size()
                logger.info(
                    f"MacPlayTools 已连接 {self._host}:{self._port} "
                    f"(version={self._version}, {self._screen_width}x{self._screen_height})"
                )
                return True
            except Exception as e:
                logger.warning(f"MacPlayTools 连接失败 ({self._host}:{self._port}): {e}")
                self._cleanup_socket()
                return False

    def disconnect(self):
        self._cleanup_socket()
        self._connected = False
        self._screen_width = 0
        self._screen_height = 0
        self._version = 0

    def _cleanup_socket(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception as e:
                logger.debug(f"PlayTools: 关闭socket失败: {e}")

            self._sock = None

    # ── 底层收发 ──────────────────────────────────────────────

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("连接已断开")
            buf.extend(chunk)
        return bytes(buf)

    def _send_command(self, payload: bytes):
        if self._sock is None:
            raise RuntimeError("MacPlayTools 尚未连接")
        length = len(payload)
        header = struct.pack(">H", length)
        self._sock.sendall(header + payload)

    def _recv_u16(self) -> int:
        data = self._recv_exact(self._sock, 2)
        return struct.unpack(">H", data)[0]

    def _recv_u32(self) -> int:
        data = self._recv_exact(self._sock, 4)
        return struct.unpack(">I", data)[0]

    # ── 协议命令 ──────────────────────────────────────────────

    def _query_version(self) -> int:
        self._send_command(self.CMD_VERSION)
        return self._recv_u32()

    def _query_size(self) -> Tuple[int, int]:
        self._send_command(self.CMD_SIZE)
        w = self._recv_u16()
        h = self._recv_u16()
        return w, h

    def screencap_bgr(self) -> Optional[np.ndarray]:
        """截屏并返回 BGR 格式的 numpy 数组。"""
        with self._lock:
            if not self._connected:
                return None
            try:
                self._send_command(self.CMD_BGR_SCREENCAP)
                width = self._recv_u32()
                height = self._recv_u32()
                data_len = self._recv_u32()
                if data_len == 0 or width == 0 or height == 0:
                    return None
                raw = self._recv_exact(self._sock, data_len)
                img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                return img.copy()
            except Exception as e:
                logger.warning(f"MacPlayTools BGR 截屏失败: {e}")
                self._mark_disconnected()
                return None

    def screencap_rgbx(self) -> Optional[np.ndarray]:
        """截屏并返回 BGR 格式的 numpy 数组 (通过 RGBX 转换)。"""
        with self._lock:
            if not self._connected:
                return None
            try:
                self._send_command(self.CMD_SCREENCAP)
                data_len = self._recv_u32()
                if data_len == 0:
                    return None
                raw = self._recv_exact(self._sock, data_len)
                if self._screen_width == 0 or self._screen_height == 0:
                    return None
                img = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self._screen_height, self._screen_width, 4)
                )
                bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                return bgr
            except Exception as e:
                logger.warning(f"MacPlayTools RGBX 截屏失败: {e}")
                self._mark_disconnected()
                return None

    def touch(self, phase: int, x: int, y: int):
        """发送触控事件。phase: 0=down, 1=move, 3=up"""
        with self._lock:
            if not self._connected:
                raise RuntimeError("MacPlayTools 尚未连接")
            try:
                payload = self.CMD_TOUCH + struct.pack(">BHH", phase, x, y)
                self._send_command(payload)
            except Exception as e:
                logger.warning(f"MacPlayTools 触控事件失败: {e}")
                self._mark_disconnected()
                raise

    def click(self, x: int, y: int):
        """在指定坐标执行点击。"""
        self.touch(self.TOUCH_DOWN, x, y)
        time.sleep(0.01)
        self.touch(self.TOUCH_UP, x, y)

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int,
              duration: float = 0.8, hold_end: float = 0.0, ease: str | None = None):
        """从起点滑动到终点。"""
        start_x, start_y = int(start_x), int(start_y)
        end_x, end_y = int(end_x), int(end_y)

        distance = max(abs(end_x - start_x), abs(end_y - start_y))
        steps = max(2, min(60, int(math.ceil(max(distance, 1) / 24))))
        if duration > 0:
            steps = max(steps, min(120, int(math.ceil(duration / 0.016))))

        step_delay = max(float(duration) / steps, 0.001)

        if ease == "out_quad":
            ease_fn = lambda t: 1 - (1 - t) ** 2
        else:
            ease_fn = lambda t: t

        self.touch(self.TOUCH_DOWN, start_x, start_y)
        for i in range(1, steps):
            progress = ease_fn(i / steps)
            move_x = round(start_x + (end_x - start_x) * progress)
            move_y = round(start_y + (end_y - start_y) * progress)
            self.touch(self.TOUCH_MOVE, move_x, move_y)
            time.sleep(step_delay)
        if hold_end > 0:
            self.touch(self.TOUCH_MOVE, end_x, end_y)
            time.sleep(hold_end)
        self.touch(self.TOUCH_UP, end_x, end_y)

    def get_bundle_id(self) -> str:
        with self._lock:
            if not self._connected:
                return ""
            try:
                self._send_command(self.CMD_BUNDLE)
                length = self._recv_u32()
                if length == 0:
                    return ""
                raw = self._recv_exact(self._sock, length)
                return raw.decode("utf-8")
            except Exception as e:
                logger.warning(f"MacPlayTools 获取 Bundle ID 失败: {e}")
                self._mark_disconnected()
                return ""

    def get_window_rect(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """返回 (frame, content) 两个矩形, 各为 (x, y, w, h)。"""
        with self._lock:
            if not self._connected:
                return (0, 0, 0, 0), (0, 0, 0, 0)
            try:
                self._send_command(self.CMD_RECT)
                values = []
                for _ in range(8):
                    values.append(self._recv_u16())
                frame = tuple(values[:4])
                content = tuple(values[4:])
                return frame, content
            except Exception as e:
                logger.warning(f"MacPlayTools 获取窗口矩形失败: {e}")
                self._mark_disconnected()
                return (0, 0, 0, 0), (0, 0, 0, 0)

    def terminate_app(self):
        with self._lock:
            if not self._connected:
                return
            try:
                self._send_command(self.CMD_TERMINATE)
            except Exception as e:
                logger.warning(f"MacPlayTools 终止应用失败: {e}")

    def _mark_disconnected(self):
        self._connected = False
        self._cleanup_socket()
