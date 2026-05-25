import functools
import os
import sys
import time

from src.utils.logger import logger


def _format_frame_name(frame) -> str:
    func_name = frame.f_code.co_name
    self_obj = frame.f_locals.get("self")
    cls_obj = frame.f_locals.get("cls")
    if self_obj is not None:
        return f"{self_obj.__class__.__name__}.{func_name}"
    if isinstance(cls_obj, type):
        return f"{cls_obj.__name__}.{func_name}"
    return func_name


def _get_caller_info() -> str:
    try:
        caller_frame = sys._getframe(2)
    except ValueError:
        return "未知调用方"

    caller_name = _format_frame_name(caller_frame)
    caller_file = caller_frame.f_code.co_filename
    try:
        caller_file = os.path.relpath(caller_file, os.getcwd())
    except ValueError as exc:
        logger.debug("性能日志: 调用文件路径转换失败，使用原始路径: {}", exc)
    caller_line = caller_frame.f_lineno
    return f"{caller_name} ({caller_file}:{caller_line})"


def timeit(func):
    """测试函数执行时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        use_time = (end - start) * 1000
        if use_time > 300:
            caller_info = _get_caller_info()
            logger.warning(f"[{func.__qualname__}] 被 [{caller_info}] 调用，执行耗时: {use_time:.3f} ms")
        return result
    return wrapper
