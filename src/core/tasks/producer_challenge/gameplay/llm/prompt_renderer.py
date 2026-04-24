"""producer_challenge 的 LLM prompt 渲染层。"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import jinja2


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@functools.lru_cache(maxsize=1)
def _get_env() -> jinja2.Environment:
    """获取env并返回结果。

    Returns:
        jinja2.Environment: 返回值类型见注解，语义由函数用途决定。
    """
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_PROMPT_DIR)),
        keep_trailing_newline=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(template_name: str, **kwargs: Any) -> str:
    """渲染目标数据并返回结果。

    Args:
        template_name: 用于提供template、name相关输入。
        **kwargs: 用于提供kwargs相关输入。

    Returns:
        str: 处理后的文本结果。
    """
    return _get_env().get_template(template_name).render(**kwargs)
