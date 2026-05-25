"""VL (Vision-Language) 客户端 — 用于记忆卡面照片选择。

通过 OpenAI 兼容的 vision API 发送截图，让 VL 模型选出最优照片。
复用现有 LLM 配置（base_url / model / api_key 等）。
"""

import base64
import re
from time import sleep, time
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI

from src.core.tasks.producer_challenge.gameplay.llm.config import (
    get_llm_config, MAX_RETRIES, RETRY_DELAY,
)
from src.utils.logger import logger


class VLClient:
    """封装 OpenAI 兼容 Vision API 的客户端。"""

    def __init__(self):
        """处理init并返回结果。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
        self._client: Optional[OpenAI] = None
        self._last_base_url: Optional[str] = None
        self._last_api_key: Optional[str] = None
        self._last_timeout: Optional[float] = None

    def _ensure_client(self, cfg: dict) -> OpenAI:
        """处理ensure、client并返回结果。

        Args:
            cfg: 用于提供cfg相关输入。

        Returns:
            OpenAI: 返回值类型见注解。
        """
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
        """处理recreate、client并返回结果。

        Returns:
            返回处理结果，具体类型见返回注解。
        """
        logger.debug("VL: 重建 API 客户端连接")
        self._client = None

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """将 OpenCV 图像编码为 base64 字符串。"""
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def select_best_photo(
        self,
        photo_images: list[np.ndarray],
        prompt: str,
    ) -> Optional[int]:
        """使用 VL 模型从照片列表中选出最优照片。

        Args:
            photo_images: 各照片缩略图的 OpenCV 图像列表。
            prompt: 用户自定义提示词。

        Returns:
            选中照片的索引（0-based），失败返回 None。
        """
        if not photo_images:
            logger.warning("VL: 无照片可选")
            return None

        if len(photo_images) == 1:
            logger.debug("VL: 仅有一张照片，直接选择")
            return 0

        cfg = get_llm_config()

        # 构建包含多张图片的 content
        content_parts = []
        content_parts.append({
            "type": "text",
            "text": (
                f"{prompt}\n\n"
                f"共有 {len(photo_images)} 张照片，编号从 1 到 {len(photo_images)}。"
                f"请直接回复你选择的照片编号（数字），不要回复其他内容。"
            ),
        })

        for i, img in enumerate(photo_images):
            b64 = self._encode_image(img)
            content_parts.append({
                "type": "text",
                "text": f"照片 {i + 1}:",
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            })

        messages = [
            {
                "role": "user",
                "content": content_parts,
            }
        ]

        for attempt in range(MAX_RETRIES + 1):
            try:
                client = self._ensure_client(cfg)
                t0 = time()
                resp = client.chat.completions.create(
                    model=cfg["model"],
                    messages=messages,
                    max_tokens=64,
                    temperature=0.1,
                )
                elapsed = time() - t0

                content = self._extract_content(resp)
                if content is None:
                    logger.warning("VL: 响应为空")
                    return None

                logger.debug(f"VL 响应 ({elapsed:.1f}s): {content}")
                return self._parse_photo_index(content, len(photo_images))

            except Exception as e:
                logger.warning(
                    f"VL API 错误 (尝试 {attempt + 1}/{MAX_RETRIES + 1}): {e}"
                )
                if attempt < MAX_RETRIES:
                    self._recreate_client()
                    sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"VL API 调用失败，已用尽所有重试: {e}")
                    return None
        return None

    @staticmethod
    def _extract_content(resp) -> Optional[str]:
        """提取content并返回结果。

        Args:
            resp: 用于提供resp相关输入。

        Returns:
            Optional[str]: 返回值类型见注解。
        """
        if not resp.choices:
            return None
        msg = resp.choices[0].message
        content = getattr(msg, "content", None) or ""
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content if content else None

    @staticmethod
    def _parse_photo_index(text: str, total: int) -> Optional[int]:
        """从 VL 响应中解析照片编号（1-based → 0-based）。"""
        numbers = re.findall(r"\d+", text)
        if not numbers:
            logger.warning(f"VL: 无法从响应中解析数字: {text!r}")
            return None
        chosen = int(numbers[0])
        if 1 <= chosen <= total:
            logger.info(f"VL 选择照片 {chosen}/{total}")
            return chosen - 1
        logger.warning(f"VL: 返回的编号 {chosen} 超出范围 [1, {total}]")
        return None
