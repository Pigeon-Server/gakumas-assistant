"""LLM 配置模块 — 从 ConfigService 读取全局 LLM 设置。"""

from src.core.services.config_service import ConfigService


def get_llm_config() -> dict:
    base = ConfigService().items.base
    return {
        "base_url":          str(base.llm_base_url),
        "model":             str(base.llm_model),
        "api_key":           str(base.llm_api_key),
        "timeout":           float(base.llm_timeout),
        "max_tokens":        int(base.llm_max_tokens) or None,
        "num_ctx":           int(base.llm_num_ctx),
        "reasoning_effort":  str(base.llm_reasoning_effort or "medium"),
        "temperature":       float(base.llm_temperature),
    }


def get_insight_config() -> dict:
    base = ConfigService().items.base
    main = get_llm_config()
    max_tokens_raw = int(base.llm_insight_max_tokens or 0)
    num_ctx_raw = int(base.llm_insight_num_ctx or 0)
    return {
        "enabled":          bool(base.llm_insight_enabled),
        "base_url":         str(base.llm_insight_base_url or "").strip() or main["base_url"],
        "model":            str(base.llm_insight_model or "").strip() or main["model"],
        "api_key":          str(base.llm_insight_api_key or "").strip() or main["api_key"],
        "timeout":          float(base.llm_insight_timeout or 120),
        "max_tokens":       max_tokens_raw if max_tokens_raw > 0 else None,
        "num_ctx":          num_ctx_raw if num_ctx_raw > 0 else None,
        "reasoning_effort": str(base.llm_insight_reasoning_effort or "medium"),
        "temperature":      float(base.llm_insight_temperature or 0.2),
    }


# 非 ConfigService 管理的运行时常量
MAX_RETRIES = 2
RETRY_DELAY = 2.0
TOP_P = 0.9
MAX_ACTIVE_EFFECTS = 15
MAX_ACTIVE_ENCHANTS = 10
