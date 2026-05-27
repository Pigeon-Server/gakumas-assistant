"""DecisionDumper 单元测试。

验证:
  - start_session 创建目录
  - record 写入 JSON 文件并包含完整字段
  - update_last_resolved 正确覆写文件
  - get_summary 统计准确
  - 单例模式正常工作
  - 禁用时不写入
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

# 使用 monkeypatch 替换 _DUMP_ROOT 来避免写入真实日志目录
from src.core.tasks.producer_challenge.gameplay.llm import decision_dumper
from src.core.tasks.producer_challenge.gameplay.llm.decision_dumper import (
    DecisionDumper,
    DecisionRecord,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """每个测试前重置单例。"""
    DecisionDumper._instance = None
    yield
    DecisionDumper._instance = None


@pytest.fixture
def tmp_dump_root(tmp_path):
    """将 dump 根目录重定向到临时目录。"""
    original = decision_dumper._DUMP_ROOT
    decision_dumper._DUMP_ROOT = tmp_path
    yield tmp_path
    decision_dumper._DUMP_ROOT = original


def _make_decision_state(phase: str = "lesson", position: str = "card_play") -> dict:
    return {
        "phase": phase,
        "position": position,
        "week": 3,
        "revision": 42,
        "llm_snapshot": {"hp": 100, "turn": 2},
        "stage_context": {},
        "candidates": [
            {"index": 0, "id": "skill_1", "title": "元気キャラ"},
            {"index": 1, "id": "skill_2", "title": "集中"},
            {"index": 2, "id": "end_turn", "title": "ターン終了"},
        ],
        "llm_actions": [{"index": 0}, {"index": 1}, {"index": 2}],
        "legal_actions": [0, 1, 2],
    }


class TestDecisionDumperSingleton:
    """单例模式测试。"""

    def test_get_instance_returns_same(self):
        a = DecisionDumper.get_instance()
        b = DecisionDumper.get_instance()
        assert a is b

    def test_separate_instances_after_reset(self):
        a = DecisionDumper.get_instance()
        DecisionDumper._instance = None
        b = DecisionDumper.get_instance()
        assert a is not b


class TestStartSession:
    """start_session 测试。"""

    def test_creates_directory(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        path = dumper.start_session("test_session")
        assert path.exists()
        assert path.is_dir()
        assert "test_session" in str(path)

    def test_auto_session_id(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        path = dumper.start_session()
        assert path.exists()
        # 自动 ID 为时间戳格式
        assert path.name[0:4].isdigit()

    def test_resets_sequence(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        dumper.start_session("s1")
        dumper.record(decision_state=_make_decision_state(), chosen_index=0)
        assert dumper.record_count == 1
        dumper.start_session("s2")
        assert dumper.record_count == 0


class TestRecord:
    """record 写入测试。"""

    def test_writes_json_file(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        dumper.start_session("rec_test")
        rec = dumper.record(
            decision_state=_make_decision_state(),
            system_prompt="你是助手",
            user_prompt="请选择卡片",
            llm_raw_content="选择索引1",
            llm_raw_reasoning="因为集中更好",
            llm_cleaned_output="1",
            llm_model="test-model",
            llm_elapsed_sec=1.5,
            chosen_index=1,
            resolved_index=1,
            resolved_name="集中",
            total_elapsed_sec=2.0,
        )
        assert rec is not None
        # 检查文件存在
        filepath = dumper.session_dir / "0000_lesson.json"
        assert filepath.exists()

        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["seq"] == 0
        assert data["phase"] == "lesson"
        assert data["position"] == "card_play"
        assert data["week"] == 3
        assert data["llm_call"]["system_prompt"] == "你是助手"
        assert data["llm_call"]["raw_reasoning"] == "因为集中更好"
        assert data["decision"]["chosen_index"] == 1
        assert data["decision"]["resolved_index"] == 1
        assert data["decision"]["fallback_used"] is False

    def test_increments_sequence(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        dumper.start_session("seq_test")
        dumper.record(decision_state=_make_decision_state(phase="lesson"), chosen_index=0)
        dumper.record(decision_state=_make_decision_state(phase="consult"), chosen_index=1)
        assert dumper.record_count == 2
        assert (dumper.session_dir / "0000_lesson.json").exists()
        assert (dumper.session_dir / "0001_consult.json").exists()


class TestUpdateLastResolved:
    """update_last_resolved 测试。"""

    def test_overwrites_json(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        dumper.start_session("update_test")
        dumper.record(
            decision_state=_make_decision_state(),
            chosen_index=1,
            resolved_index=1,
            resolved_name="集中",
        )
        # 兜底覆盖
        dumper.update_last_resolved(
            resolved_index=2,
            resolved_name="ターン終了",
            fallback_used=True,
            fallback_reason="无可打出卡片",
        )
        filepath = dumper.session_dir / "0000_lesson.json"
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["decision"]["resolved_index"] == 2
        assert data["decision"]["resolved_name"] == "ターン終了"
        assert data["decision"]["fallback_used"] is True
        assert data["decision"]["fallback_reason"] == "无可打出卡片"

    def test_noop_when_no_records(self, tmp_dump_root):
        """没有记录时不应报错。"""
        dumper = DecisionDumper.get_instance()
        dumper.start_session("empty")
        dumper.update_last_resolved(resolved_index=0)  # 不应抛异常


class TestGetSummary:
    """get_summary 统计测试。"""

    def test_empty_summary(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        dumper.start_session("summary_empty")
        s = dumper.get_summary()
        assert s["total"] == 0

    def test_correct_counts(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        dumper.start_session("summary_full")
        dumper.record(
            decision_state=_make_decision_state(phase="lesson"),
            chosen_index=0,
            llm_elapsed_sec=1.0,
        )
        dumper.record(
            decision_state=_make_decision_state(phase="lesson"),
            chosen_index=1,
            llm_elapsed_sec=2.0,
            fallback_used=True,
            fallback_reason="test",
        )
        dumper.record(
            decision_state=_make_decision_state(phase="consult"),
            chosen_index=0,
            llm_elapsed_sec=0.5,
        )
        s = dumper.get_summary()
        assert s["total"] == 3
        assert s["by_phase"]["lesson"] == 2
        assert s["by_phase"]["consult"] == 1
        assert s["fallback_count"] == 1
        assert s["total_llm_time_sec"] == 3.5
        assert s["avg_llm_time_sec"] == pytest.approx(3.5 / 3, abs=0.01)


class TestDisabled:
    """禁用时不写入测试。"""

    def test_record_returns_none_when_disabled(self, tmp_dump_root):
        dumper = DecisionDumper.get_instance()
        dumper.enabled = False
        dumper.start_session("disabled")
        result = dumper.record(decision_state=_make_decision_state(), chosen_index=0)
        assert result is None
        assert dumper.record_count == 0


class TestDecisionRecord:
    """DecisionRecord 序列化测试。"""

    def test_to_dict_structure(self):
        rec = DecisionRecord(
            seq=0,
            timestamp="2024-01-01T00:00:00",
            phase="lesson",
            position="card_play",
            chosen_index=1,
            resolved_index=1,
            llm_raw_reasoning="这是思维链",
        )
        d = rec.to_dict()
        assert "llm_call" in d
        assert "decision" in d
        assert d["llm_call"]["raw_reasoning"] == "这是思维链"
        assert d["decision"]["chosen_index"] == 1
