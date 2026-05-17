from __future__ import annotations

from src.core.tasks.producer_challenge.gameplay.llm.prompt_renderer import render


def _term_block(text: str) -> str:
    start = text.index("## 术语速查")
    end = text.index("流派联动", start) if "流派联动" in text[start:] else text.index("规则", start)
    return text[start:end].strip()


def test_system_prompts_share_common_term_block():
    lesson = render("system_lesson.j2")
    p_drink = render("system_p_drink.j2")
    exam = render("system_exam.j2")
    skill_reward = render("system_skill_reward.j2")

    common = _term_block(lesson)
    assert common == _term_block(p_drink)
    assert common in exam
    assert common in skill_reward
    assert "这是一份公共术语块" not in common
    assert "好調 / 絶好調 / 集中 / 好印象" in common
