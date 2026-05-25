"""LLM API 客户端 — OpenAI 兼容接口的封装。

特性:
  - 自动重试与指数退避
  - 连接池失效时重建客户端
  - 思考模型（reasoning）的输出解析
  - 结构化日志
"""

from time import sleep, time
from typing import Optional

from openai import OpenAI

from src.core.tasks.producer_challenge.gameplay.llm.config import (
    get_llm_config, MAX_RETRIES, RETRY_DELAY, TOP_P,
)
from src.utils.logger import logger


class LLMClient:
    """封装 OpenAI 兼容 API 的客户端，每次调用从 ConfigService 读取最新配置。"""

    def __init__(self):
        self._client: Optional[OpenAI] = None
        # 缓存上一次构建客户端时使用的连接参数，便于检测变更
        self._last_base_url: Optional[str] = None
        self._last_api_key: Optional[str] = None
        self._last_timeout: Optional[float] = None

    def _ensure_client(self, cfg: dict) -> OpenAI:
        """若连接参数未变则复用，否则重建客户端。"""
        if (
            self._client is not None
            and cfg["base_url"] == self._last_base_url
            and cfg["api_key"] == self._last_api_key
            and cfg["timeout"] == self._last_timeout
        ):
            return self._client
        self._client = OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            timeout=cfg["timeout"],
        )
        self._last_base_url = cfg["base_url"]
        self._last_api_key = cfg["api_key"]
        self._last_timeout = cfg["timeout"]
        return self._client

    def _recreate_client(self):
        """强制重建客户端连接（连接池失效时使用）。"""
        logger.debug("LLM: 重建 API 客户端连接")
        self._client = None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[str]:
        """发送聊天请求并返回模型回复内容。

        所有参数（model / max_tokens / temperature 等）从 ConfigService 实时读取。

        Returns:
            模型回复的文本内容，失败返回 None。
        """
        cfg = get_llm_config()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(MAX_RETRIES + 1):
            try:
                client = self._ensure_client(cfg)
                t0 = time()
                kwargs = {
                    "model": cfg["model"],
                    "messages": messages,
                    "max_tokens": cfg["max_tokens"],
                    "temperature": cfg["temperature"],
                    "top_p": TOP_P,
                    "reasoning_effort": str(cfg.get("reasoning_effort") or "medium"),
                }
                num_ctx = int(cfg.get("num_ctx") or 0)
                if num_ctx > 0:
                    kwargs["extra_body"] = {
                        "options": {"num_ctx": num_ctx},
                        "think": str(cfg.get("reasoning_effort") or "medium"),
                    }
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception as e:
                    message = str(e).lower()
                    if (
                        "reasoning_effort" in message
                        or "think" in message
                        or "unsupported" in message
                        or "extra_body" in message
                    ):
                        kwargs.pop("reasoning_effort", None)
                        kwargs.pop("extra_body", None)
                        resp = client.chat.completions.create(**kwargs)
                    else:
                        raise

                elapsed = time() - t0

                content = self._extract_content(resp)
                if content is not None:
                    logger.debug(f"LLM 响应 ({elapsed:.1f}s): {content[:100]}...")
                    return content

                logger.warning("LLM 响应为空")
                return None

            except Exception as e:
                logger.warning(f"LLM API 错误 (尝试 {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                if attempt < MAX_RETRIES:
                    self._recreate_client()
                    sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"LLM API 调用失败，已用尽所有重试: {e}")
                    return None
        return None

    @staticmethod
    def _extract_content(resp) -> Optional[str]:
        """从 API 响应中提取文本内容。

        兼容思考模型（reasoning model）的输出格式。
        """
        if not resp.choices:
            return None

        msg = resp.choices[0].message
        content = getattr(msg, "content", None) or ""

        # 清理思考标签（某些模型在 content 中混入 <think>...</think>）
        if "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return content if content else None
