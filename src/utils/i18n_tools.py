from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


@dataclass(frozen=True)
class I18nText:
    """前后端共享的国际化文本结构。"""

    key: str
    params: dict[str, Any] | None = None
    fallback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。"""
        payload: dict[str, Any] = {"key": self.key}
        if self.params:
            payload["params"] = serialize_i18n_value(self.params)
        if self.fallback:
            payload["fallback"] = self.fallback
        return payload

    def __str__(self) -> str:
        """返回便于日志输出的可读文本。"""
        return self.fallback or self.key


def i18n_text(key: str, fallback: str | None = None, **params: Any) -> I18nText:
    """构造国际化文本对象。"""
    normalized_params = params or None
    return I18nText(key=key, params=normalized_params, fallback=fallback)


def i18n_text_from_params(
    key: str,
    params: dict[str, Any] | None = None,
    fallback: str | None = None,
) -> I18nText:
    """使用现成参数字典构造国际化文本对象。"""
    return I18nText(key=key, params=params or None, fallback=fallback)


def serialize_i18n_value(value: Any) -> Any:
    """递归转换国际化对象与嵌套结构为可 JSON 序列化数据。"""
    if isinstance(value, I18nText):
        return value.to_dict()
    if is_dataclass(value):
        return serialize_i18n_value(asdict(value))
    if isinstance(value, dict):
        return {
            key: serialize_i18n_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [serialize_i18n_value(item) for item in value]
    return value


def ensure_i18n_text(value: I18nText | str | None, fallback_key: str) -> I18nText:
    """确保返回值为国际化文本对象。"""
    if isinstance(value, I18nText):
        return value
    if value:
        return i18n_text(fallback_key, fallback=str(value))
    return i18n_text(fallback_key)


def translate_like_payload(value: I18nText | str | dict | None, fallback_key: str) -> dict[str, Any]:
    """将消息值规范化为前端可识别的国际化对象字典。"""
    if isinstance(value, I18nText):
        return value.to_dict()
    if isinstance(value, dict):
        serialized = serialize_i18n_value(value)
        if isinstance(serialized, dict) and serialized.get("key"):
            return serialized
        return i18n_text(fallback_key, fallback=str(serialized)).to_dict()
    if value:
        return i18n_text(fallback_key, fallback=str(value)).to_dict()
    return i18n_text(fallback_key).to_dict()


def normalize_i18n_key_segment(value: Any) -> str:
    """将任意值规范化为国际化键片段。"""
    text = str(value or "").strip()
    if not text:
        return "empty"
    normalized_chars: list[str] = []
    last_was_separator = False
    for char in text:
        if char.isascii() and char.isalnum():
            if char.isupper() and normalized_chars and not last_was_separator:
                normalized_chars.append("_")
            normalized_chars.append(char.lower())
            last_was_separator = False
            continue
        if not last_was_separator:
            normalized_chars.append("_")
            last_was_separator = True
    normalized = "".join(normalized_chars).strip("_")
    return normalized or "value"
