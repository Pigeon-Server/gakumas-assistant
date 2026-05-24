import socket
import time
from typing import Optional

from adbutils import AdbDevice, Network

from src.core.device.Android.adapters.minitouch_resource import MinitouchBinaryResource
from src.utils.logger import logger


# ── 缓动函数（ease functions）──────────────────────────
def _ease_linear(t: float) -> float:
    return t


def _ease_out_quad(t: float) -> float:
    return 1 - (1 - t) ** 2


_EASE_FUNCTIONS = {
    None: _ease_linear,
    "linear": _ease_linear,
    "out_quad": _ease_out_quad,
}


class MinitouchAdapter:
    SOCKET_NAME = "minitouch-gkmas"

    def __init__(self, adb_device: AdbDevice, connection_timeout: int = 3000):
        self._adb_device = adb_device
        self._connection_timeout = connection_timeout
        self._socket: Optional[socket.socket] = None
        self._process_stream = None
        self._pid: Optional[int] = None
        self._protocol_version: Optional[int] = None
        self._max_contacts: Optional[int] = None
        self._max_x: Optional[int] = None
        self._max_y: Optional[int] = None
        self._max_pressure: Optional[int] = None
        self._display_width: Optional[int] = None
        self._display_height: Optional[int] = None

    @property
    def alive(self) -> bool:
        return self._socket is not None

    def start(self) -> bool:
        if self.alive:
            return True

        sdk = self._get_sdk_version()
        if sdk >= 29:
            logger.warning(
                "minitouch on Android 10+ requires STFService forwarding per official README; fallback to ADB"
            )
            return False

        abi_list = self._get_supported_abis()
        resource = MinitouchBinaryResource.discover(abi_list, sdk)
        if resource is None:
            logger.warning(
                "Official minitouch binary not found in bin/minitouch/libs/<abi>/, fallback to ADB"
            )
            return False

        self.stop()
        try:
            remote_path = f"/data/local/tmp/{resource.binary_name}"
            self._adb_device.sync.push(str(resource.local_path), remote_path)
            self._adb_device.shell(["chmod", "755", remote_path])
            self._process_stream = self._adb_device.shell(
                [remote_path, "-n", self.SOCKET_NAME],
                stream=True,
            )
            self._socket = self._open_socket()
            self._read_header()
            self._refresh_display_size()
            logger.debug(
                f"minitouch started from {resource.source}: {resource.local_path} "
                f"(abi={resource.abi}, version={self._protocol_version})"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to start minitouch adapter: {e}")
            self.stop()
            return False

    def stop(self):
        if self._socket is not None:
            try:
                self._socket.sendall(b"r\n")
            except Exception as e:
            logger.debug(f"Minitouch: 操作失败: {e}")

            try:
                self._socket.close()
            except Exception as e:
                logger.debug(f"Failed to close minitouch socket cleanly: {e}")
        self._socket = None

        if self._pid is not None:
            try:
                self._adb_device.shell(["kill", str(self._pid)])
            except Exception as e:
            logger.debug(f"Minitouch: 操作失败: {e}")

        self._pid = None

        if self._process_stream is not None:
            try:
                self._process_stream.close()
            except Exception as e:
                logger.debug(f"Failed to stop minitouch process cleanly: {e}")
        self._process_stream = None

    def click(self, x, y):
        if not self.alive:
            raise RuntimeError("minitouch adapter is not running")

        touch_x, touch_y = self._map_to_touch_space(x, y)
        pressure = self._default_pressure()
        self._send_commands(
            [
                f"d 0 {touch_x} {touch_y} {pressure}",
                "c",
                "u 0",
                "c",
            ]
        )

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.8, hold_end=0.0, ease=None):
        if not self.alive:
            raise RuntimeError("minitouch adapter is not running")

        start_touch_x, start_touch_y = self._map_to_touch_space(start_x, start_y)
        end_touch_x, end_touch_y = self._map_to_touch_space(end_x, end_y)
        pressure = self._default_pressure()
        distance = max(abs(end_touch_x - start_touch_x), abs(end_touch_y - start_touch_y))
        steps = max(2, min(60, distance // 24 or 2))
        if duration > 0:
            steps = max(steps, min(120, int(duration / 0.016) or 2))
        step_delay = max(float(duration) / steps, 0.001)

        ease_fn = _EASE_FUNCTIONS.get(ease, _ease_linear)
        self._send_commands([f"d 0 {start_touch_x} {start_touch_y} {pressure}", "c"])
        for index in range(1, steps):
            progress = ease_fn(index / steps)
            move_x = round(start_touch_x + (end_touch_x - start_touch_x) * progress)
            move_y = round(start_touch_y + (end_touch_y - start_touch_y) * progress)
            self._send_commands([f"m 0 {move_x} {move_y} {pressure}", "c"])
            time.sleep(step_delay)
        if hold_end > 0:
            self._send_commands([f"m 0 {end_touch_x} {end_touch_y} {pressure}", "c"])
            time.sleep(hold_end)
        self._send_commands(["u 0", "c"])

    def _get_sdk_version(self) -> int:
        return int(self._adb_device.shell("getprop ro.build.version.sdk").strip())

    def _get_supported_abis(self) -> list[str]:
        abi_list_raw = self._adb_device.shell("getprop ro.product.cpu.abilist").strip()
        abi_raw = self._adb_device.shell("getprop ro.product.cpu.abi").strip()

        result = []
        for item in abi_list_raw.split(",") + [abi_raw]:
            abi = item.strip()
            if abi and abi not in result:
                result.append(abi)
        return result

    def _refresh_display_size(self):
        width, height = self._adb_device.window_size()
        self._display_width = max(int(width), 1)
        self._display_height = max(int(height), 1)

    def _open_socket(self) -> socket.socket:
        timeout_seconds = max(self._connection_timeout / 1000, 0.5)
        deadline = time.time() + timeout_seconds
        last_error = None
        while time.time() < deadline:
            try:
                sock = self._adb_device.create_connection(Network.LOCAL_ABSTRACT, self.SOCKET_NAME)
                sock.settimeout(1.0)
                return sock
            except Exception as e:
                last_error = e
                time.sleep(0.05)
        raise ConnectionError(f"Failed to connect to minitouch socket: {last_error}")

    def _read_header(self):
        version_line = self._recv_line()
        bounds_line = self._recv_line()
        pid_line = self._recv_line()

        if not version_line.startswith("v "):
            raise RuntimeError(f"Unexpected minitouch header: {version_line!r}")
        if not bounds_line.startswith("^ "):
            raise RuntimeError(f"Unexpected minitouch bounds: {bounds_line!r}")
        if not pid_line.startswith("$ "):
            raise RuntimeError(f"Unexpected minitouch pid line: {pid_line!r}")

        self._protocol_version = int(version_line.split()[1])
        _, max_contacts, max_x, max_y, max_pressure = bounds_line.split()
        self._max_contacts = int(max_contacts)
        self._max_x = int(max_x)
        self._max_y = int(max_y)
        self._max_pressure = int(max_pressure)
        self._pid = int(pid_line.split()[1])

    def _recv_line(self) -> str:
        if self._socket is None:
            raise RuntimeError("minitouch socket is not connected")

        data = bytearray()
        while True:
            chunk = self._socket.recv(1)
            if not chunk:
                raise ConnectionError("minitouch socket closed while reading header")
            if chunk == b"\n":
                break
            data.extend(chunk)
        return data.decode("utf-8", errors="replace").strip()

    def _send_commands(self, commands: list[str]):
        if self._socket is None:
            raise RuntimeError("minitouch socket is not connected")
        payload = ("\n".join(commands) + "\n").encode("utf-8")
        self._socket.sendall(payload)

    def _default_pressure(self) -> int:
        assert self._max_pressure is not None
        return max(1, self._max_pressure // 2)

    def _map_to_touch_space(self, x: int, y: int) -> tuple[int, int]:
        self._refresh_display_size()
        assert self._display_width is not None
        assert self._display_height is not None
        assert self._max_x is not None
        assert self._max_y is not None

        x = max(0, min(int(x), self._display_width - 1))
        y = max(0, min(int(y), self._display_height - 1))

        mapped_x = round((x / max(self._display_width - 1, 1)) * self._max_x)
        mapped_y = round((y / max(self._display_height - 1, 1)) * self._max_y)
        return mapped_x, mapped_y
