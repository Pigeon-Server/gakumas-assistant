"""策略洞察系统端到端测试。"""

import pytest

from src.core.tasks.producer_challenge.gameplay.llm.insight_data import InsightData
from src.core.tasks.producer_challenge.gameplay.llm.insight_store import InsightStore
from src.core.tasks.producer_challenge.gameplay.llm.message_builder import (
    build_insight_reference_text,
    build_user_prompt,
)


# ── 测试用 DB fixture ─────────────────────────────────


@pytest.fixture(autouse=True)
def _setup_test_db(tmp_path, monkeypatch):
    """使用临时数据库运行测试。"""
    from peewee import SqliteDatabase

    test_db = SqliteDatabase(str(tmp_path / "test_insights.db"))
    from src.models.strategy_insight import ProduceStrategyInsight

    test_db.bind([ProduceStrategyInsight], bind_refs=False, bind_backrefs=False)
    test_db.create_tables([ProduceStrategyInsight], safe=True)

    import src.core.tasks.producer_challenge.gameplay.llm.insight_store as _store_mod
    original = _store_mod._insight_store
    _store_mod._insight_store = None

    yield test_db

    _store_mod._insight_store = original
    test_db.close()


# ── InsightStore CRUD 测试 ────────────────────────────


def test_save_and_retrieve_insight(_setup_test_db):
    store = InsightStore()
    data = InsightData(
        insight_type="step",
        scenario="hajime",
        phase="schedule",
        strategy_description="前期堆叠好印象效果，后期用高倍率打分卡兑现",
        when_to_apply="路线偏向好印象且剩余回合>5时",
        when_not_to_apply="体力不足或已进入打分阶段时",
        decision_family="schedule",
        idol_plan_type="ProducePlanType_Plan2",
    )
    iid = store.save_insight(data)
    assert iid > 0

    state = {
        "phase": "schedule",
        "llm_snapshot": {
            "scenario": "hajime",
            "idol_plan_type": "ProducePlanType_Plan2",
            "planning": {},
        },
    }
    results = store.retrieve_insights(state, limit=5)
    assert len(results) >= 1
    assert any(r.id == iid for r in results)
    found = [r for r in results if r.id == iid][0]
    assert "好印象" in found.strategy_description
    assert found.decision_family == "schedule"


def test_record_usage_promotes_status(_setup_test_db):
    store = InsightStore()
    data = InsightData(
        insight_type="step",
        scenario="hajime",
        phase="lesson",
        strategy_description="出牌策略",
        decision_family="combat",
    )
    iid = store.save_insight(data)

    row = store.get_by_ids([iid])[0]
    assert row.validation_status == "draft"

    store.record_usage(iid, "success")
    store.record_usage(iid, "success")

    row = store.get_by_ids([iid])[0]
    assert row.validation_status == "trusted"
    assert row.evidence_count == 2


def test_retrieve_filters_by_scenario(_setup_test_db):
    store = InsightStore()
    store.save_insight(InsightData(
        insight_type="step", scenario="hajime", phase="schedule",
        strategy_description="hajime策略", decision_family="schedule",
    ))
    store.save_insight(InsightData(
        insight_type="step", scenario="nia", phase="schedule",
        strategy_description="nia策略", decision_family="schedule",
    ))

    state = {"phase": "schedule", "llm_snapshot": {"scenario": "hajime", "planning": {}}}
    results = store.retrieve_insights(state, limit=10)
    assert all(r.scenario == "hajime" for r in results)


# ── Prompt 注入测试 ───────────────────────────────────


def test_build_insight_reference_text():
    insights = [
        InsightData(
            strategy_description="前期堆叠好印象效果",
            when_to_apply="路线偏向好印象",
            when_not_to_apply="体力不足时",
        ),
        InsightData(
            strategy_description="考试前1-2周集中打分",
            when_to_apply="距离考试2周内",
        ),
    ]
    text = build_insight_reference_text(insights)
    assert "前期堆叠好印象效果" in text
    assert "路线偏向好印象" in text
    assert "体力不足时" in text
    assert "考试前1-2周集中打分" in text


def test_build_user_prompt_injects_insights():
    insights = [
        InsightData(
            strategy_description="残り1回合且需立即过线时，优先直接得分",
            when_to_apply="剩余回合很少且有得分缺口",
            when_not_to_apply="还有足够铺垫窗口",
        ),
    ]
    prompt = build_user_prompt(
        {
            "phase": "lesson",
            "position": "lesson_idle",
            "llm_snapshot": {
                "phase": "lesson", "position": "lesson_idle",
                "scenario": "hajime", "difficulty": "regular",
                "week": 3, "stage_context": {"label": "课程出牌", "description": "", "available_action_summary": ""},
                "planning": {}, "produce_goals": {"summary": "", "exam_criteria": [], "training_tasks": []},
                "turn": 1, "remaining": 1, "battle_kind_label": "レッスン",
                "score": 90, "target": 100, "ratio": "90%",
                "stamina": 10, "max_stamina": 30, "genki": 0,
                "parameter_stats": {}, "p_items": [], "hand": [],
                "deck_count": 0, "deck_summary": "", "deck_cards": [],
                "zone_counts": {"deck": 0, "grave": 0, "hold": 0, "lost": 0},
                "offensive_counts": {"hand": 0, "deck": 0, "grave": 0, "hold": 0},
                "resources": {}, "observability": {},
            },
            "llm_actions": [
                {"index": 0, "label": "直撃", "kind": "produce_card", "description": "直接得分",
                 "available": True, "gain_summary": "immediate_score_gain"},
            ],
            "legal_actions": [0],
        },
        strategy_insights=insights,
    )
    assert "## 历史策略洞察" in prompt
    assert "残り1回合且需立即过线时" in prompt
    assert "## 当前可选动作" in prompt


# ── 端到端: DB → 检索 → Prompt ────────────────────────


def test_end_to_end_db_to_prompt(_setup_test_db):
    store = InsightStore()
    store.save_insight(InsightData(
        insight_type="step", scenario="hajime", phase="schedule",
        strategy_description="在体力充足时优先选择提升参数的动作",
        when_to_apply="体力>20且距离考试>3周",
        when_not_to_apply="体力不足或临近考试",
        decision_family="schedule",
        idol_plan_type="ProducePlanType_Plan2",
    ))
    store.save_insight(InsightData(
        insight_type="phase", scenario="hajime", phase="schedule",
        strategy_description="好印象流派节奏：前4周堆好印象，5-8周过渡，9-12周打分",
        when_to_apply="路线偏向好印象",
        decision_family="schedule",
        idol_plan_type="ProducePlanType_Plan2",
    ))

    state = {
        "phase": "schedule",
        "llm_snapshot": {
            "scenario": "hajime",
            "idol_plan_type": "ProducePlanType_Plan2",
            "planning": {},
        },
    }
    retrieved = store.retrieve_insights(state, limit=5)
    assert len(retrieved) == 2

    text = build_insight_reference_text(retrieved)
    assert "体力充足时" in text
    assert "好印象流派节奏" in text

    prompt = build_user_prompt(
        {
            "phase": "schedule", "position": "schedule_idle",
            "llm_snapshot": {
                "phase": "schedule", "position": "schedule_idle",
                "scenario": "hajime", "difficulty": "regular",
                "week": 5, "stage_context": {"label": "日程选择", "description": "", "available_action_summary": ""},
                "planning": {}, "produce_goals": {"summary": "", "exam_criteria": [], "training_tasks": []},
                "turn": 1, "remaining": 1, "battle_kind_label": "スケジュール",
                "score": 0, "target": 0, "ratio": "0%",
                "stamina": 25, "max_stamina": 30, "genki": 0,
                "parameter_stats": {}, "p_items": [], "hand": [],
                "deck_count": 0, "deck_summary": "", "deck_cards": [],
                "zone_counts": {"deck": 0, "grave": 0, "hold": 0, "lost": 0},
                "offensive_counts": {"hand": 0, "deck": 0, "grave": 0, "hold": 0},
                "resources": {}, "observability": {},
            },
            "llm_actions": [
                {"index": 0, "label": "レッスン", "kind": "lesson", "description": "提升参数",
                 "available": True, "gain_summary": "vocal_param_gain"},
                {"index": 1, "label": "営業", "kind": "business", "description": "推进fan vote",
                 "available": True, "gain_summary": "fan_vote_progress"},
            ],
            "legal_actions": [0, 1],
        },
        strategy_insights=retrieved,
    )
    assert "## 历史策略洞察" in prompt
    assert "体力充足时" in prompt
    assert "好印象流派节奏" in prompt
    assert "## 当前可选动作" in prompt
