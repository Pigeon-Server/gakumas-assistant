import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from loguru import logger

from src.constants.task_status import TaskStatus
from src.utils.runtime_paths import get_runtime_root, resolve_log_path

_CURRENT_ARGS: argparse.Namespace | None = None


class CLIError(RuntimeError):
    """CLI 可预期错误，用于给出简洁提示。"""


def _json_default(value: Any):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _print_payload(payload: Any, *, as_json: bool = False):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        return
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _configure_logging(args: argparse.Namespace):
    level = "WARNING" if args.quiet else args.log_level
    if args.json and not args.log_level_explicit and not args.quiet:
        level = "WARNING"

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{thread.name}</cyan> | "
            "<cyan>{name}:{function}:{line}</cyan> <red>-</red> "
            "<level>{message}</level>"
        ),
        enqueue=True,
        backtrace=True,
    )

    log_dir = resolve_log_path()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_dir / f"{time.strftime('%Y-%m-%d')}.log"),
        rotation="00:00",
        retention="7 days",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{thread.name} | {name}:{function}:{line} - {message}"
        ),
        enqueue=True,
        backtrace=True,
    )


@contextmanager
def _suppress_native_output_when_needed():
    """在 JSON/quiet 模式下屏蔽 ADB、ONNX Runtime 等原生库直接写 fd 的噪声。"""
    if _CURRENT_ARGS is None or not (_CURRENT_ARGS.json or _CURRENT_ARGS.quiet):
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def _normalize_cwd():
    # 打包后从任意目录调用 CLI 时，旧代码中的相对路径仍按运行时根目录解析。
    os.chdir(str(get_runtime_root()))


def _init_database():
    from src.main import AppProcessor

    if _CURRENT_ARGS is not None:
        _configure_logging(_CURRENT_ARGS)
    AppProcessor.init_database()


def _create_processor(*, start_background_services: bool = False):
    from src.main import AppProcessor

    if _CURRENT_ARGS is not None:
        _configure_logging(_CURRENT_ARGS)
    with _suppress_native_output_when_needed():
        processor = AppProcessor()
    if start_background_services:
        processor.start_background_services()
    return processor


def _create_task_catalog_queue():
    """只注册任务元数据，不初始化设备、模型和资源依赖。"""
    _init_database()
    from src.core.services.config_service import ConfigService
    from src.core.services.task_service import TaskService
    from src.core.tasks.task_register import register_tasks

    fake_app = SimpleNamespace(
        is_resource_ready=lambda: False,
        ensure_device_ready=lambda **_kwargs: False,
        yolo_engine=SimpleNamespace(pause=lambda: None, running=False, is_model_switching=False),
        device=None,
        config_service=ConfigService(),
        broadcast_app_status=lambda: None,
    )
    task_queue = TaskService(fake_app)
    fake_app.task_queue = task_queue
    register_tasks(fake_app)
    return task_queue


def _create_resource_update_service():
    """资源命令使用的轻量 app，避免为了检查资源而启动设备和视觉模型。"""
    _init_database()
    from src.core.services.resource_update_service import ResourceUpdateService
    from src.main import AppProcessor

    class _ResourceCLIApp:
        clip_manager = None

        def __init__(self):
            self.task_queue = SimpleNamespace(queue_status=lambda: TaskStatus.PENDING)

        def reload_game_database(self):
            AppProcessor._load_game_database(force_reload=True)

    return ResourceUpdateService(_ResourceCLIApp())


def _format_bool(value: bool) -> str:
    return "是" if value else "否"


def _summarize_app_status(status: dict) -> str:
    device = status.get("device") or {}
    game = status.get("game") or {}
    player = game.get("player") or {}
    resources = status.get("resources") or {}
    lines = [
        f"运行模式：{status.get('platform', '')}",
        f"YOLO 推理：{_format_bool(bool(status.get('yolo')))}",
        f"任务状态：{status.get('task', '')}",
        f"当前任务：{status.get('current_task') or '-'}",
        f"挂起任务：{status.get('suspended_task') or '-'}",
        f"设备可用：{_format_bool(bool(device.get('available')))}",
    ]
    if device.get("message"):
        lines.append(f"设备消息：{device.get('message')}")
    if resources:
        lines.append(f"必要资源就绪：{_format_bool(bool(resources.get('required_resources_ready')))}")
        lines.append(f"资源发现更新：{_format_bool(bool(resources.get('has_update')))}")
        if resources.get("last_error"):
            lines.append(f"资源错误：{resources.get('last_error')}")
    lines.extend(
        [
            f"当前位置：{game.get('current_location') or '-'}",
            f"玩家等级：{player.get('level')}",
            f"宝石：{player.get('gem')}",
            f"体力：{player.get('stamina')}",
        ]
    )
    return "\n".join(lines)


def _command_status(args: argparse.Namespace) -> int:
    if args.full or args.refresh_device:
        processor = _create_processor(start_background_services=args.background_services)
        try:
            if args.refresh_device:
                processor.ensure_device_ready(force=True, restart_inference=True)
            status = processor.build_app_status()
            _print_payload(status if args.json else _summarize_app_status(status), as_json=args.json)
            return 0
        finally:
            processor.shutdown()

    _init_database()
    from src.core.services.config_service import ConfigService

    resource_status = _create_resource_update_service().get_status()
    status = {
        "platform": ConfigService().base.run_mode.lower(),
        "yolo": False,
        "task": TaskStatus.PENDING,
        "current_task": "",
        "suspended_task": "",
        "device": {
            "available": False,
            "code": "not_checked",
            "message": "轻量状态未检测设备；需要检测设备时请使用 status --full。",
        },
        "resources": {
            "required_resources_ready": resource_status.get("required_resources_ready", False),
            "has_update": resource_status.get("has_update", False),
            "last_error": resource_status.get("last_error", ""),
        },
        "game": {
            "current_location": "",
            "player": {"level": -1, "gem": -1, "stamina": -1},
        },
    }
    _print_payload(status if args.json else _summarize_app_status(status), as_json=args.json)
    return 0


def _visible_task_rows(task_queue, *, include_hidden: bool = False) -> list[dict]:
    if include_hidden:
        tasks = task_queue._task_list
        return [
            {
                "id": task.id,
                "description": task.task_name,
                "enable": task.enable,
                "status": task.status,
                "manual_only": task.manual_only,
                "hidden": task.hide,
                "allow_manual_suspend": task.allow_manual_suspend,
                "allow_manual_resume": task.allow_manual_resume,
                "last_run_time": task.last_run_time,
            }
            for task in tasks
        ]

    return [{"id": task_id, **task_info} for task_id, task_info in task_queue.get_task_list().items()]


def _format_task_table(rows: list[dict]) -> str:
    if not rows:
        return "没有已注册任务。"
    headers = ("任务ID", "名称", "启用", "状态", "手动", "隐藏")
    values = [
        (
            row["id"],
            row.get("description", ""),
            "是" if row.get("enable") else "否",
            row.get("status", ""),
            "是" if row.get("manual_only") else "否",
            "是" if row.get("hidden") else "否",
        )
        for row in rows
    ]
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in values))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in values:
        lines.append("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def _command_tasks_list(args: argparse.Namespace) -> int:
    task_queue = _create_task_catalog_queue()
    rows = _visible_task_rows(task_queue, include_hidden=args.include_hidden)
    _print_payload(rows if args.json else _format_task_table(rows), as_json=args.json)
    return 0


def _save_disabled_task_state(task_id: str, *, disabled: bool):
    from src.core.services.config_service import ConfigService

    service = ConfigService()
    config = deepcopy(service())
    disabled_tasks = list(config.base.disabled_tasks.value)
    if disabled:
        if task_id not in disabled_tasks:
            disabled_tasks.append(task_id)
    else:
        disabled_tasks = [item for item in disabled_tasks if item != task_id]
    config.base.disabled_tasks.set(disabled_tasks)
    service.save_config(config)
    return disabled_tasks


def _command_task_set_enabled(args: argparse.Namespace, *, enable: bool) -> int:
    task_queue = _create_task_catalog_queue()
    task = task_queue._find_task(args.task_id)
    if task is None:
        raise CLIError(f"任务不存在：{args.task_id}")
    _save_disabled_task_state(args.task_id, disabled=not enable)
    message = f"已{'启用' if enable else '禁用'}任务：{args.task_id}"
    payload = {"status": True, "task_id": args.task_id, "enable": enable, "message": message}
    _print_payload(payload if args.json else message, as_json=args.json)
    return 0


def _wait_for_task_queue(processor, *, timeout: float | None = None) -> dict:
    started_at = time.time()
    try:
        while True:
            if processor.task_queue.queue_status() == TaskStatus.PENDING:
                break
            if timeout is not None and time.time() - started_at > timeout:
                processor.task_queue.stop()
                raise CLIError(f"等待任务队列超时：{timeout} 秒")
            time.sleep(0.5)
    except KeyboardInterrupt:
        processor.task_queue.stop()
        raise CLIError("收到中断信号，已请求停止任务队列。")

    rows = _visible_task_rows(processor.task_queue, include_hidden=True)
    touched = [
        row for row in rows
        if row.get("last_run_time") and float(row["last_run_time"]) >= started_at - 1
    ]
    failed = [
        row for row in touched
        if row.get("status") in {TaskStatus.FAILED, TaskStatus.CANCELED}
    ]
    return {
        "status": not failed,
        "duration_seconds": round(time.time() - started_at, 2),
        "tasks": touched,
        "failed_tasks": failed,
    }


def _command_task_run(args: argparse.Namespace) -> int:
    processor = _create_processor(start_background_services=args.background_services)
    try:
        if args.apply_resources and not processor.is_resource_ready():
            ok, message, _status = processor.resource_update_service.apply_updates()
            if not ok:
                raise CLIError(message)
        if not processor.is_resource_ready():
            missing = processor.resource_update_service.get_missing_required_resources()
            raise CLIError(
                "运行任务前需要先准备游戏数据库和本地化资源。"
                "可执行 `python app.py --cli resources apply` 下载/更新资源。\n"
                f"缺失资源：{json.dumps(missing, ensure_ascii=False, default=_json_default)}"
            )
        if not processor.ensure_device_ready(restart_inference=True):
            device_status = processor.get_device_status()
            raise CLIError(device_status.get("message") or "当前设备不可用")

        if args.from_task:
            if not args.task_id:
                raise CLIError("使用 --from 时必须提供起始任务 ID。")
            started = processor.exec_task_from(args.task_id)
        else:
            started = processor.exec_task(args.task_id)
        if not started:
            raise CLIError("任务队列启动失败，请检查任务 ID、设备状态和资源状态。")
        result = _wait_for_task_queue(processor, timeout=args.timeout)
        if args.json:
            _print_payload(result, as_json=True)
        else:
            if result["status"]:
                _print_payload(f"任务队列执行完成，用时 {result['duration_seconds']} 秒。")
            else:
                failed_ids = ", ".join(row["id"] for row in result["failed_tasks"])
                _print_payload(f"任务队列执行结束，但存在失败/取消任务：{failed_ids}")
        return 0 if result["status"] else 2
    finally:
        processor.shutdown()


def _flatten_config(data: dict, prefix: str = "") -> list[dict]:
    rows = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "value" in value and "data_type" in value:
            ui = value.get("ui") or {}
            rows.append(
                {
                    "path": path,
                    "value": value.get("value"),
                    "default_value": value.get("default_value"),
                    "data_type": value.get("data_type"),
                    "label": ui.get("label"),
                    "hint": ui.get("hint"),
                    "options": ui.get("options") or [],
                }
            )
        elif isinstance(value, dict):
            rows.extend(_flatten_config(value, path))
    return rows


def _format_config_rows(rows: list[dict]) -> str:
    if not rows:
        return "没有匹配的配置项。"
    lines = []
    for row in rows:
        label = f"（{row['label']}）" if row.get("label") else ""
        lines.append(f"{row['path']}{label}")
        lines.append(f"  类型：{row['data_type']}")
        lines.append(f"  当前值：{json.dumps(row['value'], ensure_ascii=False, default=_json_default)}")
        if row.get("hint"):
            lines.append(f"  说明：{row['hint']}")
    return "\n".join(lines)


def _parse_config_value(raw_value: str, target_type: type):
    if target_type is bool:
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "是", "启用"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "否", "禁用"}:
            return False
        raise CLIError(f"无法把值解析为 bool：{raw_value}")
    if target_type is int:
        return int(raw_value)
    if target_type is float:
        return float(raw_value)
    if target_type in {list, dict, tuple}:
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise CLIError(f"{target_type.__name__} 类型配置需要传入 JSON：{exc}") from exc
        if target_type is tuple:
            return tuple(parsed)
        if not isinstance(parsed, target_type):
            raise CLIError(f"配置值类型错误，应为 {target_type.__name__}")
        return parsed
    return raw_value


def _command_config_list(args: argparse.Namespace) -> int:
    _init_database()
    from src.core.services.config_service import ConfigService

    data = ConfigService()().to_json_dict()
    rows = _flatten_config(data)
    if args.section:
        prefix = args.section.rstrip(".") + "."
        rows = [row for row in rows if row["path"] == args.section or row["path"].startswith(prefix)]
    _print_payload(rows if args.json else _format_config_rows(rows), as_json=args.json)
    return 0


def _command_config_get(args: argparse.Namespace) -> int:
    _init_database()
    from src.core.services.config_service import ConfigService

    service = ConfigService()
    try:
        item = service.item(args.path)
    except AttributeError as exc:
        raise CLIError(str(exc)) from exc
    payload = {
        "path": args.path,
        "value": item.value,
        "default_value": item.default_value,
        "data_type": item.data_type.__name__,
        "verify": item.verify,
        "use_verify": item.use_verify,
        "ui": item.ui.to_json_dict(),
    }
    if args.raw and not args.json:
        _print_payload(str(item.value))
    else:
        _print_payload(payload if args.json else json.dumps(payload, ensure_ascii=False, indent=2), as_json=args.json)
    return 0


def _command_config_set(args: argparse.Namespace) -> int:
    _init_database()
    from src.core.services.config_service import ConfigService

    service = ConfigService()
    config = deepcopy(service())
    try:
        item = config.get_item(args.path)
    except AttributeError as exc:
        raise CLIError(str(exc)) from exc

    value = _parse_config_value(args.value, item.data_type)
    payload = config.to_json_dict()
    current = payload
    keys = args.path.split(".")
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]]["value"] = value
    ok, errors = config.from_json_dict(payload)
    if not ok:
        raise CLIError("；".join(str(error) for error in errors))
    service.save_config(config)
    result = {"status": True, "path": args.path, "value": config.get_item(args.path).value}
    _print_payload(result if args.json else f"已保存配置：{args.path} = {result['value']!r}", as_json=args.json)
    return 0


def _command_config_reset(args: argparse.Namespace) -> int:
    _init_database()
    from src.core.services.config_service import ConfigService

    service = ConfigService()
    if args.path:
        config = deepcopy(service())
        try:
            item = config.get_item(args.path)
        except AttributeError as exc:
            raise CLIError(str(exc)) from exc
        item.reset()
        service.save_config(config)
        payload = {"status": True, "path": args.path, "value": item.value}
        _print_payload(payload if args.json else f"已重置配置：{args.path} = {item.value!r}", as_json=args.json)
        return 0
    service.reset_config()
    payload = {"status": True, "message": "已重置全部配置"}
    _print_payload(payload if args.json else payload["message"], as_json=args.json)
    return 0


def _command_config_export(args: argparse.Namespace) -> int:
    _init_database()
    from src.core.services.config_service import ConfigService

    data = ConfigService()().to_json_dict()
    text = json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(text + "\n", encoding="utf-8")
        payload = {"status": True, "output": str(output_path)}
        _print_payload(payload if args.json else f"已导出配置：{output_path}", as_json=args.json)
    else:
        print(text)
    return 0


def _command_config_import(args: argparse.Namespace) -> int:
    _init_database()
    from src.core.services.config_service import ConfigService

    input_path = Path(args.input)
    if not input_path.exists():
        raise CLIError(f"配置文件不存在：{input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    service = ConfigService()
    config = deepcopy(service())
    ok, errors = config.from_json_dict(data)
    if not ok:
        raise CLIError("；".join(str(error) for error in errors))
    service.save_config(config)
    payload = {"status": True, "input": str(input_path)}
    _print_payload(payload if args.json else f"已导入配置：{input_path}", as_json=args.json)
    return 0


def _command_resources_status(args: argparse.Namespace) -> int:
    service = _create_resource_update_service()
    status = service.get_status()
    if args.json:
        _print_payload(status, as_json=True)
        return 0
    lines = [
        f"必要资源就绪：{_format_bool(bool(status.get('required_resources_ready')))}",
        f"发现更新：{_format_bool(bool(status.get('has_update')))}",
        f"检查中：{_format_bool(bool(status.get('checking')))}",
        f"更新中：{_format_bool(bool(status.get('updating')))}",
        f"上次检查：{status.get('last_checked_at') or '-'}",
    ]
    if status.get("last_error"):
        lines.append(f"错误：{status['last_error']}")
    missing = status.get("missing_required_resources") or []
    if missing:
        lines.append("缺失资源：")
        lines.extend(f"  - {item['name']} ({item['path']}) 缺失 {item['missing_count']} 项" for item in missing)
    _print_payload("\n".join(lines))
    return 0


def _command_resources_check(args: argparse.Namespace) -> int:
    service = _create_resource_update_service()
    ok, message, status = service.manual_check_updates()
    payload = {"status": ok, "message": message, "data": status}
    _print_payload(payload if args.json else message, as_json=args.json)
    return 0 if ok else 2


def _command_resources_apply(args: argparse.Namespace) -> int:
    service = _create_resource_update_service()
    ok, message, status = service.apply_updates()
    payload = {"status": ok, "message": message, "data": status}
    _print_payload(payload if args.json else message, as_json=args.json)
    return 0 if ok else 2


def _command_adb_devices(args: argparse.Namespace) -> int:
    from src.utils.adb_runtime import describe_adb_error

    try:
        import adbutils

        with _suppress_native_output_when_needed():
            devices = [device.serial for device in adbutils.adb.device_list()]
        if args.usb:
            devices = [serial for serial in devices if ":" not in str(serial)]
        payload = {"available": True, "devices": devices, "message": ""}
        if args.json:
            _print_payload(payload, as_json=True)
        else:
            _print_payload("\n".join(devices) if devices else "未找到 ADB 设备。")
        return 0
    except Exception as exc:
        _, reason = describe_adb_error(exc, connect_mode="USB" if args.usb else None)
        payload = {"available": False, "devices": [], "message": reason}
        _print_payload(payload if args.json else reason, as_json=args.json)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python app.py --cli",
        description="Gakumas Assistant 命令行入口（不启动 WebUI / HttpAPI）。",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    parser.add_argument("--quiet", action="store_true", help="仅输出警告及以上日志")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="控制台日志等级，默认 INFO",
    )
    parser.set_defaults(log_level_explicit=False)

    # 标记用户是否显式传入 --log-level，便于 JSON 模式默认保持 stdout 干净。
    original_parse_known_args = parser.parse_known_args

    def _parse_known_args(args=None, namespace=None):
        arg_list = sys.argv[1:] if args is None else args
        parsed, extras = original_parse_known_args(args, namespace)
        parsed.log_level_explicit = "--log-level" in arg_list
        return parsed, extras

    parser.parse_known_args = _parse_known_args

    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="查看应用、设备、任务状态")
    status_parser.add_argument("--full", action="store_true", help="初始化完整核心后读取状态（会检测设备并加载模型）")
    status_parser.add_argument("--refresh-device", action="store_true", help="重新创建设备实例后再输出状态")
    status_parser.add_argument("--background-services", action="store_true", help="启动资源检查等后台服务")
    status_parser.set_defaults(func=_command_status)

    tasks_parser = subparsers.add_parser("tasks", help="任务队列相关命令")
    task_subparsers = tasks_parser.add_subparsers(dest="tasks_command", required=True)
    tasks_list_parser = task_subparsers.add_parser("list", help="列出已注册任务")
    tasks_list_parser.add_argument("--include-hidden", action="store_true", help="包含隐藏任务")
    tasks_list_parser.set_defaults(func=_command_tasks_list)

    tasks_run_parser = task_subparsers.add_parser("run", help="执行任务队列或单个任务")
    tasks_run_parser.add_argument("task_id", nargs="?", help="任务 ID；省略时执行所有启用的自动任务")
    tasks_run_parser.add_argument("--from", dest="from_task", action="store_true", help="从指定任务开始执行后续自动任务")
    tasks_run_parser.add_argument("--timeout", type=float, default=None, help="等待任务队列结束的最大秒数")
    tasks_run_parser.add_argument("--apply-resources", action="store_true", help="资源缺失时先尝试下载/更新资源")
    tasks_run_parser.add_argument("--background-services", action="store_true", help="启动资源检查等后台服务")
    tasks_run_parser.set_defaults(func=_command_task_run)

    tasks_enable_parser = task_subparsers.add_parser("enable", help="启用任务")
    tasks_enable_parser.add_argument("task_id", help="任务 ID")
    tasks_enable_parser.set_defaults(func=lambda args: _command_task_set_enabled(args, enable=True))

    tasks_disable_parser = task_subparsers.add_parser("disable", help="禁用任务")
    tasks_disable_parser.add_argument("task_id", help="任务 ID")
    tasks_disable_parser.set_defaults(func=lambda args: _command_task_set_enabled(args, enable=False))

    config_parser = subparsers.add_parser("config", help="配置读写命令")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_list_parser = config_subparsers.add_parser("list", help="列出配置项")
    config_list_parser.add_argument("--section", help="只显示指定分组，例如 base 或 task__auto_producer")
    config_list_parser.set_defaults(func=_command_config_list)

    config_get_parser = config_subparsers.add_parser("get", help="读取单个配置项")
    config_get_parser.add_argument("path", help="配置路径，例如 base.run_mode")
    config_get_parser.add_argument("--raw", action="store_true", help="只输出配置值")
    config_get_parser.set_defaults(func=_command_config_get)

    config_set_parser = config_subparsers.add_parser("set", help="写入单个配置项")
    config_set_parser.add_argument("path", help="配置路径，例如 base.run_mode")
    config_set_parser.add_argument("value", help="配置值；list/dict 需要传 JSON")
    config_set_parser.set_defaults(func=_command_config_set)

    config_reset_parser = config_subparsers.add_parser("reset", help="重置配置")
    config_reset_parser.add_argument("path", nargs="?", help="配置路径；省略时重置全部配置")
    config_reset_parser.set_defaults(func=_command_config_reset)

    config_export_parser = config_subparsers.add_parser("export", help="导出配置 JSON")
    config_export_parser.add_argument("--output", "-o", help="输出文件；省略时输出到 stdout")
    config_export_parser.set_defaults(func=_command_config_export)

    config_import_parser = config_subparsers.add_parser("import", help="导入配置 JSON")
    config_import_parser.add_argument("input", help="配置 JSON 文件")
    config_import_parser.set_defaults(func=_command_config_import)

    resources_parser = subparsers.add_parser("resources", help="资源仓库检查与更新")
    resource_subparsers = resources_parser.add_subparsers(dest="resources_command", required=True)
    resource_status_parser = resource_subparsers.add_parser("status", help="查看资源状态")
    resource_status_parser.set_defaults(func=_command_resources_status)
    resource_check_parser = resource_subparsers.add_parser("check", help="检查资源更新")
    resource_check_parser.set_defaults(func=_command_resources_check)
    resource_apply_parser = resource_subparsers.add_parser("apply", help="下载/更新资源并重载数据库")
    resource_apply_parser.set_defaults(func=_command_resources_apply)

    adb_parser = subparsers.add_parser("adb", help="ADB 辅助命令")
    adb_subparsers = adb_parser.add_subparsers(dest="adb_command", required=True)
    adb_devices_parser = adb_subparsers.add_parser("devices", help="列出 ADB 设备")
    adb_devices_parser.add_argument("--usb", action="store_true", help="只显示 USB 设备")
    adb_devices_parser.set_defaults(func=_command_adb_devices)

    return parser


def _normalize_global_option_order(argv: list[str]) -> list[str]:
    """允许 --json / --quiet / --log-level 放在子命令前后。"""
    global_options: list[str] = []
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in {"--json", "--quiet"}:
            global_options.append(item)
            index += 1
            continue
        if item == "--log-level" and index + 1 < len(argv):
            global_options.extend([item, argv[index + 1]])
            index += 2
            continue
        remaining.append(item)
        index += 1
    return global_options + remaining


def main(argv: list[str] | None = None) -> int:
    global _CURRENT_ARGS
    _normalize_cwd()
    parser = build_parser()
    normalized_argv = _normalize_global_option_order(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)
    _CURRENT_ARGS = args
    _configure_logging(args)

    try:
        return int(args.func(args) or 0)
    except CLIError as exc:
        payload = {"status": False, "message": str(exc)}
        _print_payload(payload if args.json else f"错误：{exc}", as_json=args.json)
        return 2
    except KeyboardInterrupt:
        payload = {"status": False, "message": "已中断。"}
        _print_payload(payload if args.json else "已中断。", as_json=args.json)
        return 130
    except Exception as exc:
        logger.exception(f"CLI 执行失败：{exc}")
        payload = {"status": False, "message": str(exc), "error_type": type(exc).__name__}
        _print_payload(payload if args.json else f"执行失败：{exc}", as_json=args.json)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
