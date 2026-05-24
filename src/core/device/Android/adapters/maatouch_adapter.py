import re
import time
from typing import Optional

from adbutils import AdbConnection, AdbDevice

from src.core.device.Android.adapters.maatouch_resource import MaaTouchResource
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


class MaaTouchAdapter:
    PACKAGE_NAME = "com.shxyke.MaaTouch"
    APP_PROCESS_CLASS = "com.shxyke.MaaTouch.App"
    PROCESS_NAME = "maatouch-gkmas"
    REMOTE_APK_PATH = "/data/local/tmp/maatouch.apk"

    def __init__(self, adb_device: AdbDevice, connection_timeout: int = 3000):
        self._adb_device = adb_device
        self._connection_timeout = connection_timeout
        self._stream: Optional[AdbConnection] = None
        self._stdout_buffer = bytearray()
        self._max_contacts: Optional[int] = None
        self._max_pressure: Optional[int] = None
        self._display_width: Optional[int] = None
        self._display_height: Optional[int] = None

    @property
    def alive(self) -> bool:
        return self._stream is not None and not self._stream.closed

    def start(self) -> bool:
        if self.alive:
            return True

        resource = MaaTouchResource.discover()
        if resource is None:
            logger.warning(
                "Official MaaTouch artifact not found in bin/maatouch/, fallback to ADB"
            )
            return False

        self.stop()
        try:
            self._deploy_artifact(resource)
            candidate_classpaths = [self.REMOTE_APK_PATH]

            installed_apk_path = self._resolve_installed_apk_path()
            if installed_apk_path and installed_apk_path not in candidate_classpaths:
                candidate_classpaths.append(installed_apk_path)

            for classpath in candidate_classpaths:
                for app_process_root in ("/system/bin", "/"):
                    if self._try_start_stream(classpath, app_process_root):
                        logger.debug(
                            f"MaaTouch started from {resource.source}: {resource.local_path} "
                            f"(classpath={classpath}, root={app_process_root})"
                        )
                        return True

            logger.warning("Failed to start MaaTouch with any known app_process command")
            self.stop()
            return False
        except Exception as e:
            logger.warning(f"Failed to start MaaTouch adapter: {e}")
            self.stop()
            return False

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.send(b"r\n")
            except Exception:
                pass  # 忽略异常

            try:
                self._stream.close()
            except Exception as e:
                logger.debug(f"Failed to close MaaTouch stream cleanly: {e}")
        self._stream = None
        self._stdout_buffer = bytearray()

        for pid in self._find_process_ids():
            try:
                self._adb_device.shell(["kill", pid])
            except Exception:
                pass  # 忽略异常


    def click(self, x, y):
        if not self.alive:
            raise RuntimeError("MaaTouch adapter is not running")

        touch_x, touch_y = self._clamp_to_display(x, y)
        pressure = self._default_pressure()
        self._send_commands(
            [
                f"d 0 {touch_x} {touch_y} {pressure}",
                "c",
            ]
        )
        time.sleep(0.01)
        self._send_commands(["u 0", "c"])

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.8, hold_end=0.0, ease=None):
        if not self.alive:
            raise RuntimeError("MaaTouch adapter is not running")

        start_touch_x, start_touch_y = self._clamp_to_display(start_x, start_y)
        end_touch_x, end_touch_y = self._clamp_to_display(end_x, end_y)
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

    def press_key(self, keycode: int):
        if not self.alive:
            raise RuntimeError("MaaTouch adapter is not running")
        self._send_commands([f"k {int(keycode)} o", "c"])

    def input_text(self, text: str):
        if not self.alive:
            raise RuntimeError("MaaTouch adapter is not running")
        normalized = str(text).replace("\r", " ").replace("\n", " ").strip()
        if not normalized:
            return
        self._send_commands([f"t {normalized}", "c"])

    def _deploy_artifact(self, resource: MaaTouchResource):
        self._adb_device.sync.push(str(resource.local_path), self.REMOTE_APK_PATH)
        try:
            self._adb_device.install_remote(self.REMOTE_APK_PATH)
        except Exception as e:
            logger.debug(
                f"MaaTouch install attempt failed, continuing with raw APK classpath only: {e}"
            )

    def _resolve_installed_apk_path(self) -> Optional[str]:
        try:
            result = self._adb_device.shell(f"pm path {self.PACKAGE_NAME}")
        except Exception as e:
            logger.debug(f"Failed to query MaaTouch package path: {e}")
            return None

        match = re.search(r"package:(.*)", result)
        if match:
            apk_path = match.group(1).strip()
            return apk_path or None
        return None

    def _try_start_stream(self, classpath: str, app_process_root: str) -> bool:
        command = [
            f"CLASSPATH={classpath}",
            "app_process",
            app_process_root,
            f"--nice-name={self.PROCESS_NAME}",
            self.APP_PROCESS_CLASS,
        ]
        stream = self._adb_device.shell(command, stream=True)
        try:
            timeout_seconds = max(self._connection_timeout / 1000, 0.5)
            stream.conn.settimeout(timeout_seconds)
            self._stdout_buffer = bytearray()
            self._read_header(stream)
            self._stream = stream
            return True
        except Exception as e:
            logger.debug(
                f"MaaTouch launch attempt failed with classpath={classpath}, "
                f"root={app_process_root}: {e}"
            )
            try:
                stream.close()
            except Exception:
                pass  # 忽略异常

            self._stream = None
            return False

    def _read_header(self, stream: AdbConnection):
        deadline = time.time() + max(self._connection_timeout / 1000, 0.5)
        version_line = None
        bounds_line = None
        pid_line = None
        unexpected_lines: list[str] = []

        while time.time() < deadline:
            line = self._recv_line(stream, deadline).strip()
            if not line:
                continue
            if line.startswith("v "):
                version_line = line
                continue
            if line.startswith("^ "):
                bounds_line = line
                continue
            if line.startswith("$ "):
                pid_line = line
                if bounds_line is not None:
                    break
                continue
            unexpected_lines.append(line)
            if len(unexpected_lines) >= 3:
                break

        if version_line is not None:
            logger.debug(f"MaaTouch reported protocol version: {version_line}")
        if bounds_line is None or pid_line is None:
            summary = "; ".join(unexpected_lines) if unexpected_lines else "no header received"
            raise RuntimeError(f"Unexpected MaaTouch header: {summary}")

        _, max_contacts, width, height, max_pressure = bounds_line.split()
        self._max_contacts = int(max_contacts)
        self._display_width = max(int(width), 1)
        self._display_height = max(int(height), 1)
        self._max_pressure = max(int(max_pressure), 1)

    def _recv_line(self, stream: AdbConnection, deadline: float) -> str:
        while True:
            newline_index = self._stdout_buffer.find(b"\n")
            if newline_index != -1:
                data = self._stdout_buffer[:newline_index]
                del self._stdout_buffer[: newline_index + 1]
                return data.decode("utf-8", errors="replace")

            if time.time() >= deadline:
                raise TimeoutError("Timed out while waiting for MaaTouch output")

            chunk = stream.recv(4096)
            if not chunk:
                raise ConnectionError("MaaTouch stream closed unexpectedly")
            self._stdout_buffer.extend(chunk)

    def _send_commands(self, commands: list[str]):
        if self._stream is None:
            raise RuntimeError("MaaTouch stream is not connected")
        payload = ("\n".join(commands) + "\n").encode("utf-8")
        self._stream.send(payload)

    def _refresh_display_size(self):
        try:
            width, height = self._adb_device.window_size()
            self._display_width = max(int(width), 1)
            self._display_height = max(int(height), 1)
        except Exception:
            pass  # 忽略异常


    def _clamp_to_display(self, x: int, y: int) -> tuple[int, int]:
        self._refresh_display_size()
        width = max(int(self._display_width or 1), 1)
        height = max(int(self._display_height or 1), 1)
        return (
            max(0, min(int(x), width - 1)),
            max(0, min(int(y), height - 1)),
        )

    def _default_pressure(self) -> int:
        return max(1, int((self._max_pressure or 255) // 2))

    def _find_process_ids(self) -> list[str]:
        try:
            output = self._adb_device.shell(f"pidof {self.PROCESS_NAME}").strip()
        except Exception:
            return []
        return [pid for pid in output.split() if pid.isdigit()]
