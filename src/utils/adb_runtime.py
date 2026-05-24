from __future__ import annotations

from typing import Tuple

from src.utils.i18n_tools import i18n_text


ADB_MISSING_CODE = "adb_missing"
ADB_USB_NOT_FOUND_CODE = "adb_device_not_found"
ADB_DEVICE_DISCONNECTED_CODE = "adb_device_disconnected"
ADB_NETWORK_UNREACHABLE_CODE = "adb_network_unreachable"
ADB_UNKNOWN_CODE = "adb_error"


def is_adb_missing_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "no adb exe could be found" in message or "install adb on your system" in message


def describe_adb_error(
    exc: Exception,
    *,
    connect_mode: str | None = None,
    serial: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> Tuple[str, str]:
    if is_adb_missing_error(exc):
        return (
            ADB_MISSING_CODE,
            i18n_text(
                "backend.adb.missing",
                fallback="未安装 adb，请先安装 Android SDK Platform-Tools 并将 adb 加入 PATH。",
            ),
        )

    message = str(exc).strip() or exc.__class__.__name__
    normalized_message = message.lower()

    if "device '" in normalized_message and " not found" in normalized_message:
        if connect_mode == "Network":
            target = f"{host}:{port}" if host and port else "目标地址"
            return (
                ADB_DEVICE_DISCONNECTED_CODE,
                i18n_text(
                    "backend.adb.deviceDisconnectedTarget",
                    fallback=f"ADB 设备 {target} 已断开或未连接。请确认模拟器/设备正在运行。",
                    target=target,
                ),
            )
        if serial:
            return (
                ADB_DEVICE_DISCONNECTED_CODE,
                i18n_text(
                    "backend.adb.deviceDisconnectedSerial",
                    fallback=f"ADB 设备 {serial} 已断开或未连接。请确认 USB 已连接、已开启 USB 调试，并在 WebUI 中刷新设备列表。",
                    serial=serial,
                ),
            )
        return (
            ADB_DEVICE_DISCONNECTED_CODE,
            i18n_text(
                "backend.adb.deviceDisconnected",
                fallback="ADB 设备已断开或未连接。请确认设备在线后重试。",
            ),
        )

    if "device offline" in normalized_message:
        if connect_mode == "Network":
            target = f"{host}:{port}" if host and port else "目标地址"
            return (
                ADB_DEVICE_DISCONNECTED_CODE,
                i18n_text(
                    "backend.adb.deviceOfflineTarget",
                    fallback=f"ADB 设备 {target} 当前处于离线状态。请重试或重启设备。",
                    target=target,
                ),
            )
        if serial:
            return (
                ADB_DEVICE_DISCONNECTED_CODE,
                i18n_text(
                    "backend.adb.deviceOfflineSerial",
                    fallback=f"ADB 设备 {serial} 当前处于离线状态。请重新插拔设备并确认 USB 调试授权。",
                    serial=serial,
                ),
            )
        return (
            ADB_DEVICE_DISCONNECTED_CODE,
            i18n_text(
                "backend.adb.deviceOffline",
                fallback="ADB 设备当前处于离线状态。请重新连接设备后重试。",
            ),
        )

    if connect_mode == "USB":
        if serial:
            return (
                ADB_USB_NOT_FOUND_CODE,
                i18n_text(
                    "backend.adb.usbNotFoundSerial",
                    fallback=f"未找到所选 USB ADB 设备：{serial}。请确认设备已连接、已开启 USB 调试，并在 WebUI 中刷新设备列表后重新选择。",
                    serial=serial,
                ),
            )
        return (
            ADB_USB_NOT_FOUND_CODE,
            i18n_text(
                "backend.adb.usbNotFound",
                fallback="未检测到可用的 USB ADB 设备。请连接设备、开启 USB 调试，并在 WebUI 中刷新设备列表。",
            ),
        )

    if connect_mode == "Network":
        target = f"{host}:{port}" if host and port else "目标地址"
        return (
            ADB_NETWORK_UNREACHABLE_CODE,
            i18n_text(
                "backend.adb.networkUnavailable",
                fallback=f"未能连接到 ADB 设备 {target}。请确认 adb 已安装、设备已联网并已执行 adb tcpip / adb connect，原始错误：{message}",
                target=target,
                message=message,
            ),
        )

    return (
        ADB_UNKNOWN_CODE,
        i18n_text(
            "backend.adb.initFailed",
            fallback=f"ADB 初始化失败：{message}",
            message=message,
        ),
    )
