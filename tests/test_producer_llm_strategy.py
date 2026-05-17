from types import SimpleNamespace
from typing import Any

from src.core.tasks.producer_challenge.gameplay.llm.insight_data import InsightData
from src.core.tasks.producer_challenge.gameplay.llm.message_builder import build_user_prompt
from src.core.tasks.producer_challenge.gameplay.strategy import llm_strategy as llm_strategy_module
from src.core.tasks.producer_challenge.gameplay.strategy.llm_strategy import LLMStrategy


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("no fake response left")
        return self._responses.pop(0)


def _response(content: str, reasoning: str = "", finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=reasoning),
                finish_reason=finish_reason,
            )
        ]
    )


def _runtime_llm_config(**overrides: object) -> dict[str, object]:
    """构造测试用主 LLM 运行时配置。"""
    config = {
        "base_url": "http://llm.test/v1/",
        "model": "config-model",
        "api_key": "config-key",
        "timeout": 30.0,
        "temperature": 0.4,
        "max_tokens": None,
        "num_ctx": 4096,
        "reasoning_effort": "medium",
    }
    config.update(overrides)
    return config


def _runtime_insight_config(**overrides: object) -> dict[str, object]:
    """构造测试用洞察 LLM 运行时配置。"""
    config = {
        "enabled": True,
        "base_url": "http://insight.test/v1/",
        "model": "insight-model",
        "api_key": "insight-key",
        "timeout": 60.0,
        "temperature": 0.2,
        "max_tokens": None,
        "num_ctx": 2048,
        "reasoning_effort": "low",
    }
    config.update(overrides)
    return config


def _patch_runtime_config(
    monkeypatch: Any,
    llm_config: dict[str, object] | None = None,
    insight_config: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """替换策略模块的配置读取函数。"""
    active_llm_config = llm_config or _runtime_llm_config()
    active_insight_config = insight_config or _runtime_insight_config()
    monkeypatch.setattr(llm_strategy_module, "get_llm_config", lambda: dict(active_llm_config))
    monkeypatch.setattr(llm_strategy_module, "get_insight_config", lambda: dict(active_insight_config))
    return active_llm_config, active_insight_config


def test_llm_retry_forces_direct_answer_without_reasoning_echo(monkeypatch):
    _patch_runtime_config(monkeypatch)
    strategy = LLMStrategy(
        model="fake-model",
        max_tokens=4096,
        think="low",
        num_ctx=8192,
        temperature=0.3,
    )
    monkeypatch.setattr(llm_strategy_module, "build_system_prompt", lambda _phase, _snapshot: "sys")

    fake = _FakeCompletions(
        [
            _response("", reasoning="long reasoning only", finish_reason="length"),
            _response('{"why_this":"直接得分","why_not_others":"回合不足","action_index":2}'),
        ]
    )
    strategy._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

    result = strategy._call_and_parse(
        "prompt body",
        {
            "phase": "lesson",
            "legal_actions": [0, 1, 2, 3],
            "llm_snapshot": {},
        },
    )

    index, reasoning = result
    assert index == 2
    assert reasoning["why_this"] == "直接得分"
    assert len(fake.calls) == 2
    second = fake.calls[1]
    assert second["temperature"] == 0.0
    assert second["max_tokens"] == 64
    if "extra_body" in second:
        assert "think" not in second["extra_body"]
    assert all(message["role"] != "assistant" for message in second["messages"])
    assert any(
        "只输出 JSON" in str(message.get("content", ""))
        for message in second["messages"]
        if message.get("role") == "user"
    )


def test_llm_retry_when_content_empty_without_reasoning(monkeypatch):
    _patch_runtime_config(monkeypatch)
    strategy = LLMStrategy(
        model="fake-model",
        max_tokens=256,
        think="false",
        temperature=0.2,
    )
    monkeypatch.setattr(llm_strategy_module, "build_system_prompt", lambda _phase, _snapshot: "sys")

    fake = _FakeCompletions(
        [
            _response("", reasoning="", finish_reason="stop"),
            _response('{"why_this":"保留技能卡","why_not_others":"当前不需要","action_index":1}'),
        ]
    )
    strategy._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

    result = strategy._call_and_parse(
        "prompt body",
        {
            "phase": "skill_reward",
            "legal_actions": [0, 1],
            "llm_snapshot": {},
        },
    )

    index, reasoning = result
    assert index == 1
    assert reasoning["why_this"] == "保留技能卡"
    assert len(fake.calls) == 2


def test_llm_strategy_reads_runtime_config_dynamically(monkeypatch):
    llm_config, _insight_config = _patch_runtime_config(
        monkeypatch,
        llm_config=_runtime_llm_config(model="first-model", max_tokens=None, num_ctx=4096),
    )
    strategy = LLMStrategy()

    assert strategy.model == "first-model"
    assert strategy.max_tokens is None
    assert strategy.num_ctx == 4096
    assert strategy._session.num_ctx == 4096

    llm_config.update(
        {
            "model": "second-model",
            "temperature": 0.15,
            "max_tokens": 512,
            "num_ctx": 8192,
            "reasoning_effort": "high",
        }
    )
    strategy._refresh_runtime_config()

    assert strategy.model == "second-model"
    assert strategy.temperature == 0.15
    assert strategy.max_tokens == 512
    assert strategy.num_ctx == 8192
    assert strategy._session.num_ctx == 8192
    assert strategy.reasoning_effort == "high"


def test_llm_strategy_treats_zero_token_and_context_as_auto(monkeypatch):
    _patch_runtime_config(
        monkeypatch,
        llm_config=_runtime_llm_config(max_tokens=0, num_ctx=0),
    )
    strategy = LLMStrategy()

    assert strategy.max_tokens is None
    assert strategy.num_ctx == 0
    assert strategy._session.num_ctx == 8192


def test_llm_strategy_keeps_explicit_overrides_over_runtime_config(monkeypatch):
    llm_config, _insight_config = _patch_runtime_config(
        monkeypatch,
        llm_config=_runtime_llm_config(model="config-model", max_tokens=1024),
    )
    strategy = LLMStrategy(model="explicit-model", max_tokens=0, think="low")

    assert strategy.model == "explicit-model"
    assert strategy.max_tokens is None
    assert strategy.reasoning_effort == "low"

    llm_config.update({"model": "changed-model", "max_tokens": 2048, "reasoning_effort": "high"})
    strategy._refresh_runtime_config()

    assert strategy.model == "explicit-model"
    assert strategy.max_tokens is None
    assert strategy.reasoning_effort == "low"


def test_build_prompt_injects_strategy_insights():
    prompt = build_user_prompt(
        {
            "phase": "lesson",
            "position": "lesson_idle",
            "llm_snapshot": {
                "phase": "lesson",
                "position": "lesson_idle",
                "scenario": "hajime",
                "difficulty": "regular",
                "week": 3,
                "stage_context": {"label": "课程出牌", "description": "", "available_action_summary": ""},
                "planning": {},
                "produce_goals": {"summary": "", "exam_criteria": [], "training_tasks": []},
                "exam_criteria": [],
                "training_tasks": [],
                "turn": 1,
                "remaining": 1,
                "battle_kind_label": "レッスン",
                "score": 90,
                "target": 100,
                "ratio": "90%",
                "stamina": 10,
                "max_stamina": 30,
                "genki": 0,
                "parameter_stats": {},
                "p_items": [],
                "formation_abilities": [],
                "formation_events": [],
                "hand": [],
                "deck_count": 0,
                "deck_summary": "",
                "deck_cards": [],
                "zone_counts": {"deck": 0, "grave": 0, "hold": 0, "lost": 0},
                "offensive_counts": {"hand": 0, "deck": 0, "grave": 0, "hold": 0},
                "resources": {},
                "observability": {},
            },
            "llm_actions": [
                {
                    "index": 0,
                    "label": "直撃",
                    "kind": "produce_card",
                    "description": "直接得分",
                    "available": True,
                    "gain_summary": "immediate_score_gain",
                }
            ],
            "legal_actions": [0],
        },
        strategy_insights=[
            InsightData(
                strategy_description="残り1回合且需立即过线时，优先直接得分",
                when_to_apply="剩余回合很少且有得分缺口",
                when_not_to_apply="还有足够铺垫窗口",
            )
        ],
    )

    assert "## 历史策略洞察" in prompt
    assert "残り1回合且需立即过线时" in prompt
    assert "## 当前可选动作" in prompt
