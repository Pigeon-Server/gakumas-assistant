import os
import re
import threading
from time import sleep
from typing import Callable, Optional, Tuple

import adbutils
from adbutils import AdbDevice

from src.constants.device.adb import ADBOperation
from src.utils.logger import logger
from src.utils.runtime_paths import resolve_runtime_str

class ADBShell:
    __shell: adbutils.AdbConnection = None
    __callback: Optional[Callable] = None
    __message_thread: Optional[threading.Thread] = None
    __close_flag: bool = False
    def __init__(self, adb_device: AdbDevice):
        self.__shell = adb_device.open_shell('')

    def __bool__(self):
        return not bool(self.__shell.closed or self.__close_flag)

    def send_command(self, command: str):
        """发送命令到shell"""
        logger.debug(f"send command: {command}")
        self.__shell.send((command+"\n").encode('utf-8'))

    def open_message(self, message_callback: Callable):
        """打开消息获取线程"""
        if self.__message_thread is not None and self.__message_thread.is_alive():
            raise RuntimeError("open message thread is still alive")
        self.__callback = message_callback
        self.__message_thread = threading.Thread(target=self.__recv, daemon=True)
        self.__message_thread.start()

    def close_message(self):
        """关闭消息获取线程"""
        if self.__message_thread is not None and not self.__message_thread.is_alive():
            raise RuntimeError("message thread not is still alive")
        self.__close_flag = False
        self.__message_thread = None
        self.__callback = None

    def __recv(self):
        """读取终端"""
        while not self.__close_flag:
            try:
                data = self.__shell.recv(4096)
            except adbutils.AdbTimeout:
                sleep(0.05)
                continue
            if not data:
                sleep(0.05)
                continue
            text = data.decode("utf-8", errors="ignore")
            if self.__callback:
                self.__callback(text)
            sleep(0.05)

    def close(self):
        """关闭shell"""
        self.__close_flag = True
        if self.__message_thread is not None:
            del self.__message_thread
        self.__shell.close()

    def __del__(self):
        self.close()

def start_DroidCast(adb_device: AdbDevice) -> Tuple[bool, Optional[ADBShell]]:
    DC_STARTED = False
    DC_CLASSPATH_ERROR = False
    MAX_TIMEOUT = 30 * 10

    def shell_output_handler(msg: str):
        nonlocal DC_STARTED, DC_CLASSPATH_ERROR
        if "DroidCast main entry" in msg:
            DC_STARTED = True
        if "could not find class 'com.rayworks.droidcast.Main'" in msg:
            DC_CLASSPATH_ERROR = True

    droidcast_apk = resolve_runtime_str(ADBOperation.ScreenCaptureService.Bin.DroidCast)
    try:
        adb_device.sync.push(droidcast_apk, "/data/local/tmp")
        adb_device.install_remote(f"/data/local/tmp/{os.path.basename(ADBOperation.ScreenCaptureService.Bin.DroidCast)}")
        logger.debug("DroidCast pushed and installed via /data/local/tmp")
    except Exception as e:
        logger.error(f"Failed to push/install DroidCast: {e}")
        return False, None

    adb_shell = ADBShell(adb_device)
    adb_shell.open_message(shell_output_handler)
    classpath_value = f"/data/local/tmp/{os.path.basename(droidcast_apk)}"

    def start_droidcast(clspath: str):
        logger.debug(f"Starting DroidCast with CLASSPATH={clspath}")
        adb_shell.send_command(f"export CLASSPATH={clspath}")
        adb_shell.send_command("app_process /system/bin com.rayworks.droidcast.Main")

    start_droidcast(classpath_value)
    timeout_counter = 0
    while not DC_STARTED and not DC_CLASSPATH_ERROR and timeout_counter < 200:
        sleep(0.1)
        timeout_counter += 1
    if timeout_counter >= MAX_TIMEOUT:
        logger.error("DroidCast launch timeout: no response from device shell.")
        return False, None
    # 若 CLASSPATH 错误 → fallback 使用 pm path
    if DC_CLASSPATH_ERROR:
        logger.debug("DroidCast did not start with local CLASSPATH. Trying pm path fallback...")

        try:
            exec_result = adb_device.shell("pm path com.rayworks.droidcast")
        except Exception as e:
            logger.error(f"Failed to run 'pm path com.rayworks.droidcast': {e}")
            return False, None

        m = re.search(r'package:(.*)', exec_result)
        if m:
            package = m.group(1).strip()
            logger.debug(f"pm path returned: {package}")
            if not package:
                logger.error("pm path returned an empty package path. Cannot launch DroidCast.")
                return False, None
            start_droidcast(package)
            timeout_counter = 0
            while not DC_STARTED and timeout_counter < MAX_TIMEOUT:
                sleep(0.1)
                timeout_counter += 1
            if not DC_STARTED:
                logger.error("DroidCast failed to start even after pm path fallback.")
                return False, None
        else:
            logger.error(f"Unexpected pm path output: '{exec_result}'. Cannot extract apk path.")
            return False, None
    try:
        adb_device.forward("tcp:53516", "tcp:53516")
        logger.debug("DroidCast started. Access via http://localhost:53516/screenshot")
    except Exception as e:
        logger.error(f"Failed to setup port forwarding: {e}")
        return False, None
    return True, adb_shell
