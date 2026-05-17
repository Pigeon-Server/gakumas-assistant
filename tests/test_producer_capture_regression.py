from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.tasks.producer_challenge.context import ProduceContext
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.gameplay.consult import ConsultHandler
from src.core.tasks.producer_challenge.gameplay.dialogue import DialogueHandler
from src.core.tasks.producer_challenge.gameplay.lesson import LessonHandler
from src.core.tasks.producer_challenge.gameplay.p_drink import PDrinkHandler
from src.core.tasks.producer_challenge.gameplay.schedule import ScheduleHandler
from src.core.tasks.producer_challenge.gameplay.skill_reward import SkillRewardHandler
from src.core.tasks.producer_challenge.ui import classify_gameplay_phase, classify_pipeline_position
from src.entity.Yolo import Yolo_Box, Yolo_Results

from src.core.tasks.producer_challenge.gameplay import consult as consult_module
from src.core.tasks.producer_challenge.gameplay import decision as decision_module
from src.core.tasks.producer_challenge.gameplay import dialogue as dialogue_module
from src.core.tasks.producer_challenge.gameplay import lesson as lesson_module
from src.core.tasks.producer_challenge.gameplay import p_drink as p_drink_module
from src.core.tasks.producer_challenge.gameplay import schedule as schedule_module
from src.core.tasks.producer_challenge.gameplay import skill_reward as skill_reward_module
from src.core.tasks.producer_challenge import ui as ui_module
from src.core.tasks.producer_challenge.ui import gameplay_state as gameplay_state_module

CAPTURE_ROOT = Path(__file__).parent / "produce_gameplay_captures"
PRODUCER_SECTIONS = ("PRODUCER_png", "PRODUCER_jpg")


class _DebugToolsStub:
    def __init__(self):
        self.boxes: list[dict] = []

    def add_box(self, x, y, w, h, **kwargs):
        self.boxes.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                **kwargs,
            }
        )


class _DeviceStub:
    def __init__(self):
        self.clicks: list[tuple[int, int, str]] = []

    def click(self, x, y, el_label=""):
        self.clicks.append((int(x), int(y), str(el_label or "")))

    def click_element(self, element):
        self.click(element.cx, element.cy, getattr(element, "label", ""))


def _meta_files(folder: str) -> list[Path]:
    return sorted((CAPTURE_ROOT / folder).glob("*_meta.json"))


def _results_from_meta(meta_path: Path, section: str) -> Yolo_Results:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    boxes: list[Yolo_Box] = []
    for item in data.get(section, []):
        x, y, w, h = item["box"]
        frame = np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
        boxes.append(Yolo_Box(x, y, x + w, y + h, item["label"], frame))
    return Yolo_Results.from_boxes(boxes)


def _iter_capture_results(folder: str):
    for meta_path in _meta_files(folder):
        for section in PRODUCER_SECTIONS:
            yield meta_path.name, section, _results_from_meta(meta_path, section)


def _make_app(results: Yolo_Results):
    device = _DeviceStub()
    debug_tools = _DebugToolsStub()
    return SimpleNamespace(
        latest_results=results,
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        device=device,
        debug_tools=debug_tools,
        clip_manager=None,
    )


@pytest.fixture(autouse=True)
def _stub_ocr(monkeypatch):
    for module in (
        consult_module,
        decision_module,
        dialogue_module,
        lesson_module,
        p_drink_module,
        schedule_module,
        skill_reward_module,
    ):
        monkeypatch.setattr(module, "ocr_text", lambda _image: "", raising=False)


PHASE_POSITION_CASES = [
    ("schedule_select", "schedule", "schedule_recommend", ""),
    ("action_clicked", "schedule", "schedule_selected", ""),
    ("after_action_confirm", "unknown", "transition_empty", ""),
    ("after_schedule_confirm", "schedule", "schedule_idle", ""),
    ("after_action_transition_wait", "schedule", "schedule_idle", ""),
    ("after_schedule_transition_wait", "dialogue", "dialogue_options", ""),
    ("commu_choice", "dialogue", "dialogue_options", ""),
    ("dialogue_after_second_tap", "p_drink", "p_drink_idle", ""),
    ("p_drink_select", "p_drink", "p_drink_idle", ""),
    ("p_drink_after_first_tap", "p_drink", "p_drink_selected", ""),
    ("after_p_drink_confirm", "unknown", "transition_hud", ""),
    ("lesson", "lesson", "lesson_idle", ""),
    ("transition", "lesson", "lesson_idle", ""),
    ("lesson_after_first_card_tap", "lesson", "lesson_selected", ""),
    ("after_card_confirm", "lesson", "lesson_selected", ""),
    ("after_reward_center_tap", "unknown", "transition_hud", ""),
    ("after_skill_reward_confirm", "unknown", "transition_hud", ""),
    ("after_skill_reward_showcase_tap", "unknown", "transition_hud", ""),
    ("stage_after_skillcard_reward_tap", "skill_reward", "skill_reward_idle", ""),
    ("skill_reward_after_first_tap", "skill_reward", "skill_reward_selected", ""),
    ("week2_present_support_checkpoint_1", "unknown", "transition_hud", ""),
    ("week2_present_support_checkpoint_2", "unknown", "transition_hud", ""),
    ("week2_present_support_checkpoint_3", "unknown", "transition_hud", ""),
    ("week2_present_support_after_last_showcase", "dialogue", "dialogue_options", ""),
    ("week2_present_support_checkpoint_4", "p_drink", "p_drink_idle", ""),
    ("week2_present_support_checkpoint_5", "unknown", "transition_hud", ""),
    ("week2_present_support_checkpoint_6", "unknown", "transition_hud", ""),
    ("week2_present_support_stage_done", "unknown", "transition_hud", ""),
    ("week2_present_support_stage_done_2", "skill_reward", "skill_reward_idle", ""),
    ("week2_present_support_final_reward", "skill_reward", "skill_reward_selected", ""),
    ("week4_soudan_after_transition_wait", "consult", "consult_exchange", ""),
    ("week4_soudan_after_confirm", "unknown", "transition_empty", ""),
    ("soudan_enhance_after_tap", "consult", "consult_enhancement_preview", "consult_exchange"),
    ("soudan_enhance_card_selected", "consult", "consult_enhancement_preview", "consult_exchange"),
    ("soudan_enhance_card_selected_2", "consult", "consult_enhancement_ready", "consult_enhancement_preview"),
    ("soudan_enhance_after_modal_confirm", "consult", "consult_exchange", ""),
    ("soudan_enhance_stage_done", "consult", "consult_exchange", ""),
    ("consult_exit_after_tap", "schedule", "schedule_recommend", ""),
    ("exam_checkpoint_1", "exam", "exam_idle", ""),
    ("exam_after_end_turn_confirm", "exam", "exam_idle", ""),
    ("exam_skip_batch_checkpoint", "exam", "exam_idle", ""),
    ("exam_result_chain_1", "exam", "exam_idle", ""),
    ("exam_after_skip_tap", "modal", "gameplay_modal", ""),
    ("week5_katsudo_checkpoint_2", "unknown", "transition_hud", ""),
    ("week5_katsudo_checkpoint_3", "p_drink", "p_drink_idle", ""),
    ("week5_katsudo_shikyu_after_confirm", "unknown", "transition_hud", ""),
    ("stage_done_ready_to_log", "unknown", "transition_hud", ""),
    ("stage_finish_before_plan", "unknown", "transition_hud", ""),
    ("stage_finish_for_plan_checkpoint", "unknown", "transition_empty", ""),
    ("stage_finish_for_plan_stable", "schedule", "schedule_idle", ""),
    ("startup_modal_1", "modal", "gameplay_modal", ""),
    ("soudan_enhance_confirmed", "modal", "gameplay_modal", ""),
    ("after_simple_doubletap_retry", "result", "result", ""),
    # ── 新增：扩展覆盖的场景截图 ──────────────────────────────────────────────
    # 动作/排程相关
    ("after_action_summary_tap",             "unknown",      "transition_empty",       ""),
    ("after_transition_wait",                "schedule",     "schedule_idle",          ""),
    ("current_state",                        "schedule",     "schedule_selected",      ""),
    ("schedule",                             "schedule",     "schedule_selected",      ""),
    ("schedule_after_first_tap",             "schedule",     "schedule_selected",      ""),
    ("schedule_after_second_tap",            "unknown",      "transition_empty",       ""),
    ("lesson_post_achievement_1",            "schedule",     "schedule_idle",          ""),
    ("fast_chain_checkpoint_2",              "schedule",     "schedule_idle",          ""),
    # 课程相关
    ("current_after_image_plan",             "lesson",       "lesson_idle",            ""),
    ("current_progress_after_plan",          "lesson",       "lesson_idle",            ""),
    ("lesson_after_pdrink_tap",              "lesson",       "lesson_idle",            ""),
    ("lesson_after_second_card_tap",         "lesson",       "lesson_idle",            ""),
    ("lesson_after_use_confirm_modal",       "lesson",       "lesson_idle",            ""),
    ("lesson_final_turn_checkpoint",         "lesson",       "lesson_idle",            ""),
    ("lesson_late_checkpoint_1",             "lesson",       "lesson_idle",            ""),
    ("lesson_mid_checkpoint_1",              "lesson",       "lesson_idle",            ""),
    ("lesson_mid_checkpoint_2",              "lesson",       "lesson_idle",            ""),
    ("lesson_second_card_selected",          "lesson",       "lesson_selected",        ""),
    ("lesson_second_card_used",              "lesson",       "lesson_idle",            ""),
    ("lesson_turn2",                         "lesson",       "lesson_selected",        ""),
    ("lesson_zero_cost_card_selected",       "lesson",       "lesson_selected",        ""),
    ("lesson_zero_cost_card_used",           "lesson",       "lesson_idle",            ""),
    ("week3_vo_lesson_stable",               "lesson",       "lesson_idle",            ""),
    ("chase_lesson_batch_progress",          "lesson",       "lesson_idle",            ""),  # 追い込みレッスン
    # 对话相关
    ("after_second_start_tap",               "dialogue",     "dialogue_options",       ""),
    ("commu_choice2",                        "dialogue",     "dialogue_options",       ""),
    ("current_after_interrupt",              "dialogue",     "dialogue_options",       ""),
    ("current_live_1",                       "dialogue",     "dialogue_options",       ""),
    ("dialogue3_after_first_tap",            "dialogue",     "dialogue_options",       ""),
    ("dialogue_after_first_tap",             "dialogue",     "dialogue_options",       ""),
    ("post_start",                           "dialogue",     "dialogue_options",       ""),
    ("week1_focus_dialogue_after_confirm",   "dialogue",     "dialogue_options",       ""),
    ("week1_focus_dialogue_delayed_confirm", "dialogue",     "dialogue_options",       ""),
    ("week1_focus_dialogue_fastforward",     "dialogue",     "dialogue_options",       ""),
    ("week1_focus_dialogue_hotspot_try",     "dialogue",     "dialogue_options",       ""),
    ("week1_focus_dialogue_reconfirm",       "dialogue",     "dialogue_options",       ""),
    ("week6_after_transition_wait",          "dialogue",     "dialogue_options",       ""),
    ("fail_run_tail_1",                      "dialogue",     "dialogue_continue",      ""),
    ("fail_run_tail_2",                      "dialogue",     "dialogue_continue",      ""),
    ("post_exam_fastforward_confirmed",      "dialogue",     "dialogue_continue",      ""),
    ("post_memory_effect_1",                 "dialogue",     "dialogue_continue",      ""),
    # 弹窗相关
    ("lesson_end_checkpoint",               "modal",        "gameplay_modal",         ""),
    ("startup_modal_2",                     "modal",        "gameplay_modal",         ""),
    ("startup_modal_3",                     "modal",        "gameplay_modal",         ""),
    ("startup_modal_fast_forward",          "modal",        "gameplay_modal",         ""),
    ("startup_modal_skip",                  "modal",        "gameplay_modal",         ""),
    ("voice_modal",                         "modal",        "gameplay_modal",         ""),
    ("fail_run_closeout_1",                 "modal",        "gameplay_modal",         ""),
    ("exam_result_after_next_1",            "modal",        "gameplay_modal",         ""),
    ("memory_regen_action_1",               "modal",        "gameplay_modal",         ""),
    # 技能奖励相关
    ("dialogue3_after_second_tap",          "skill_reward", "skill_reward_idle",      ""),
    ("week6_dialogue_branch2_after_confirm","skill_reward", "skill_reward_idle",      ""),
    ("fail_run_closeout_2",                 "skill_reward", "skill_reward_selected",  ""),
    ("fail_run_final_stable",               "skill_reward", "skill_reward_selected",  ""),
    ("memory_regen_choice_right",           "skill_reward", "skill_reward_selected",  ""),
    ("memory_regen_result_page",            "skill_reward", "skill_reward_selected",  ""),
    # 过渡帧（有HUD标签）
    ("current_stage_checkpoint_before_plan","unknown",      "transition_hud",         ""),
    ("lesson_after_summary_tap",            "unknown",      "transition_hud",         ""),
    ("week3_vo_lesson_after_confirm",       "unknown",      "transition_hud",         ""),
    ("chase_lesson_intro_1",                "unknown",      "transition_hud",         ""),
    # 过渡帧（无HUD标签 / 游戏循环外界面）
    ("after_startup_0",                     "unknown",      "transition_empty",       ""),
    ("confirm_page",                        "unknown",      "transition_empty",       ""),
    ("current_exam_takeover",               "unknown",      "transition_empty",       ""),
    ("lesson_stage_done_or_fail",           "lesson",       "lesson_summary_showcase", ""),
    ("pre_start_confirm",                   "unknown",      "transition_empty",       ""),
    ("week6_after_schedule_confirm",        "unknown",      "transition_empty",       ""),
    ("fail_run_home_or_next",               "unknown",      "transition_empty",       ""),
    ("fail_run_back_home",                  "unknown",      "transition_empty",       ""),
    ("fail_run_memory_generate_1",          "unknown",      "transition_empty",       ""),
    ("fail_run_memory_generate_2",          "unknown",      "transition_empty",       ""),
    ("fail_run_result_tail_1",              "unknown",      "transition_empty",       ""),
    ("fail_run_closeout_home_1",            "unknown",      "transition_empty",       ""),
    ("fail_run_closeout_home_2",            "unknown",      "transition_empty",       ""),
    ("exam_fail_end_produce_1",             "unknown",      "transition_empty",       ""),
    ("closeout_event_rewards_1",            "unknown",      "transition_empty",       ""),
    ("post_exam_loading_1",                 "unknown",      "transition_empty",       ""),
    ("fast_chain_checkpoint_1",             "unknown",      "transition_empty",       ""),
    ("memory_regen_action_2",               "unknown",      "transition_empty",       ""),
    ("memory_regen_action_3",               "unknown",      "transition_empty",       ""),
    ("memory_regen_correct_tap",            "unknown",      "transition_empty",       ""),
    ("memory_regen_wait",                   "unknown",      "transition_empty",       ""),
]


@pytest.mark.parametrize(("folder", "expected_phase", "expected_position", "last_position"), PHASE_POSITION_CASES)
def test_capture_phase_and_position_regression(folder, expected_phase, expected_position, last_position):
    seen = 0
    for meta_name, section, results in _iter_capture_results(folder):
        ctx = ProduceContext()
        if last_position:
            ctx.last_stable_position = last_position
        actual_phase = classify_gameplay_phase(results, ctx=ctx)
        actual_position = classify_pipeline_position(results, ctx=ctx)
        assert actual_phase == expected_phase, (folder, meta_name, section, actual_phase)
        assert actual_position == expected_position, (folder, meta_name, section, actual_position)
        seen += 1
    assert seen >= 2


def _phase_position(results: Yolo_Results, ctx: ProduceContext) -> tuple[str, str]:
    phase = classify_gameplay_phase(results, ctx=ctx)
    position = classify_pipeline_position(results, ctx=ctx)
    ctx.set_phase(phase)
    ctx.set_position(position)
    return phase, position


def test_schedule_retry_sequence_uses_capture_frames():
    handler = ScheduleHandler()
    ctx = ProduceContext()

    first_results = _results_from_meta(_meta_files("schedule_select")[0], "PRODUCER_png")
    app = _make_app(first_results)
    phase, position = _phase_position(first_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.pending_schedule_index == 0
    assert app.debug_tools.boxes

    second_results = _results_from_meta(_meta_files("schedule_after_first_tap")[0], "PRODUCER_png")
    app.latest_results = second_results
    phase, position = _phase_position(second_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.pending_schedule_index is None
    assert ctx.current_week == 1
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "confirm_schedule_action",
        "retry_limit": 12,
        "retry_sleep": 0.7,
    }


def test_dialogue_retry_sequence_uses_capture_frames():
    handler = DialogueHandler()
    ctx = ProduceContext()

    first_results = _results_from_meta(_meta_files("commu_choice")[0], "PRODUCER_png")
    app = _make_app(first_results)
    phase, position = _phase_position(first_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.pending_dialogue_option_index == 0
    assert app.debug_tools.boxes

    second_results = _results_from_meta(_meta_files("dialogue_after_first_tap")[0], "PRODUCER_png")
    app.latest_results = second_results
    phase, position = _phase_position(second_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.pending_dialogue_option_index is None
    assert ctx.dialogue_choices_made == 1


def test_lesson_summary_showcase_uses_capture_frames():
    handler = LessonHandler()
    ctx = ProduceContext()

    results = _results_from_meta(_meta_files("lesson_stage_done_or_fail")[0], "PRODUCER_png")
    app = _make_app(results)
    phase, position = _phase_position(results, ctx)

    result = handler.handle(app, ctx, phase, position)

    assert result.status == "ok"
    assert app.device.clicks == [(540, 819, "lesson-summary-showcase")]
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "lesson_summary_showcase",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_short_parameter_gain_text_without_action_info_is_lesson_summary_showcase(monkeypatch):
    results = _results_from_meta(_meta_files("lesson_stage_done_or_fail")[0], "PRODUCER_png")
    filtered_boxes = [
        box for box in results.boxes
        if box.label != ProducerLabels.PC_ACTION_INFO
    ]
    results.boxes = filtered_boxes
    results.frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
    debug_tools = _DebugToolsStub()
    app = SimpleNamespace(
        latest_results=results,
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        device=_DeviceStub(),
        debug_tools=debug_tools,
        clip_manager=None,
    )
    ctx = ProduceContext()

    monkeypatch.setattr(ui_module, "ocr_text", lambda _image: "Vo.が43上昇した!", raising=False)
    monkeypatch.setattr(gameplay_state_module, "ocr_text", lambda _image: "Vo.が43上昇した!", raising=False)

    phase = classify_gameplay_phase(results, ctx=ctx)
    position = classify_pipeline_position(results, ctx=ctx)
    result = LessonHandler().handle(app, ctx, phase, position)

    assert phase == "lesson"
    assert position == "lesson_summary_showcase"
    assert result.status == "ok"


def test_p_drink_retry_sequence_uses_capture_frames():
    handler = PDrinkHandler()
    ctx = ProduceContext()

    first_results = _results_from_meta(_meta_files("p_drink_select")[0], "PRODUCER_png")
    app = _make_app(first_results)
    phase, position = _phase_position(first_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.pending_p_drink_index == 0
    assert app.debug_tools.boxes

    second_results = _results_from_meta(_meta_files("p_drink_after_first_tap")[0], "PRODUCER_png")
    app.latest_results = second_results
    phase, position = _phase_position(second_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.pending_p_drink_index is None


def test_skill_reward_retry_sequence_uses_capture_frames():
    handler = SkillRewardHandler()
    ctx = ProduceContext()

    first_results = _results_from_meta(_meta_files("stage_after_skillcard_reward_tap")[0], "PRODUCER_png")
    app = _make_app(first_results)
    phase, position = _phase_position(first_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.pending_skill_reward_index == 0
    assert app.debug_tools.boxes

    second_results = _results_from_meta(_meta_files("skill_reward_after_first_tap")[0], "PRODUCER_png")
    app.latest_results = second_results
    phase, position = _phase_position(second_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.pending_skill_reward_index is None


def test_present_support_skill_card_showcase_capture_regression(monkeypatch):
    results = _results_from_meta(
        _meta_files("present_support_skill_card_showcase")[0],
        "PRODUCER_png",
    )
    ctx = ProduceContext()

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.ui.collect_frame_text",
        lambda _results: "心のアルバム パラメータ+3 元気+3 次のターン、スキルカードを引く 2ターン後、スキルカードを引く 重複不可 レッスン中1回",
    )

    phase = classify_gameplay_phase(results, ctx=ctx)
    position = classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == "skill_reward"
    assert position == "skill_reward_showcase"


def test_present_support_p_drink_receive_confirm_capture_regression(monkeypatch):
    results = _results_from_meta(
        _meta_files("present_support_p_drink_receive_confirm")[0],
        "PRODUCER_png",
    )
    ctx = ProduceContext()
    ctx.last_stable_position = "p_drink_selected"

    monkeypatch.setattr(
        "src.core.tasks.producer_challenge.ui.collect_frame_text",
        lambda _results: "活動支給 ホットコーヒー 今やる 受け取る",
    )

    phase = classify_gameplay_phase(results, ctx=ctx)
    position = classify_pipeline_position(results, ctx=ctx, phase=phase)

    assert phase == "p_drink"
    assert position == "p_drink_selected"


def test_consult_enhancement_sequence_uses_capture_frames():
    handler = ConsultHandler()
    ctx = ProduceContext()

    exchange_results = _results_from_meta(_meta_files("week4_soudan_after_transition_wait")[0], "PRODUCER_png")
    app = _make_app(exchange_results)
    phase, position = _phase_position(exchange_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.handler_state["consult_last_subaction"] == "open_enhancement"
    assert app.debug_tools.boxes

    preview_results = _results_from_meta(_meta_files("soudan_enhance_after_tap")[0], "PRODUCER_png")
    app.latest_results = preview_results
    phase, position = _phase_position(preview_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.handler_state["consult_last_subaction"] == "select_enhancement_target"
    assert ctx.handler_state["consult_enhancement_target"]

    ready_results = _results_from_meta(_meta_files("soudan_enhance_card_selected_2")[0], "PRODUCER_png")
    app.latest_results = ready_results
    phase, position = _phase_position(ready_results, ctx)
    result = handler.handle(app, ctx, phase, position)
    assert result.status == "ok"
    assert ctx.handler_state["consult_last_subaction"] == "confirm_enhancement"
    assert ctx.handler_state["consult_auto_used_enhancement"] is True
