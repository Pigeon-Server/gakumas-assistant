from types import SimpleNamespace

import numpy as np

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge import ui as ui_module
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import skill_reward as skill_reward_module
from src.core.tasks.producer_challenge.gameplay.skill_reward import (
    SkillRewardHandler,
    SkillRewardStepResult,
)


class _ResultsStub:
    def __init__(self, labels):
        self._labels = set(labels)
        self.frame = np.zeros((2340, 1080, 3), dtype=np.uint8)

    def exists_label(self, label):
        return label in self._labels

    def filter_by_label(self, label):
        if label in self._labels:
            return [SimpleNamespace(label=label, x=0, y=0, w=100, h=100, cx=50, cy=50, frame=np.zeros((10, 10, 3), dtype=np.uint8))]
        return []

    def __bool__(self):
        return True


class _DeviceStub:
    def __init__(self):
        self.clicks = []

    def click(self, x, y, el_label=""):
        self.clicks.append((int(x), int(y), str(el_label or "")))

    def click_element(self, element):
        self.click(element.cx, element.cy, getattr(element, "label", ""))


def test_classify_gameplay_state_detects_skill_reward_showcase(monkeypatch):
    results = _ResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_TARGET,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.PC_ACTION_INFO,
        ProducerLabels.SKILL_CARD_MENTAL,
        ProducerLabels.P_DRINK,
    })
    monkeypatch.setattr(
        ui_module,
        "ocr_text",
        lambda _frame: "中間試験 戻す ワクワクが止まらない+ リーリヤ ワクワクが止まらないを強化しました",
    )

    phase = ui_module.classify_gameplay_phase(results)
    position = ui_module.classify_pipeline_position(results, phase=phase)

    assert phase == "skill_reward"
    assert position == GameplayPosition.SKILL_REWARD_SHOWCASE


def test_classify_gameplay_state_detects_skill_reward_selection_with_hud(monkeypatch):
    results = _ResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_TARGET,
        ProducerLabels.PC_P_POINT,
    })
    panel = results.frame[int(results.frame.shape[0] * 0.56):int(results.frame.shape[0] * 0.93), int(results.frame.shape[1] * 0.05):int(results.frame.shape[1] * 0.95)]
    panel[:] = 240
    monkeypatch.setattr(
        ui_module,
        "ocr_text",
        lambda _frame: "受け取るスキルカードを選んでください。 受け取る あと2回 再抽選",
    )

    phase = ui_module.classify_gameplay_phase(results)
    position = ui_module.classify_pipeline_position(results, phase=phase)

    assert phase == "skill_reward"
    assert position == GameplayPosition.SKILL_REWARD_IDLE


def test_classify_gameplay_state_prefers_skill_reward_over_battle_hand_hud(monkeypatch):
    results = _ResultsStub({
        ProducerLabels.PC_PROGRESS,
        ProducerLabels.PC_STAMINA,
        ProducerLabels.PC_TARGET,
        ProducerLabels.PC_P_POINT,
        ProducerLabels.SKILL_CARD_MENTAL,
        ProducerLabels.P_DRINK,
    })
    # 模拟真机奖励页现场：三张卡、底栏饮料/空槽、下半屏大白面板，
    # 但“受け取る”按钮本身没有被 YOLO 检出来。
    results.frame[int(results.frame.shape[0] * 0.56):int(results.frame.shape[0] * 0.93), int(results.frame.shape[1] * 0.05):int(results.frame.shape[1] * 0.95)] = 240
    original_filter_by_label = results.filter_by_label

    def _filter_by_label(label):
        if label == ProducerLabels.SKILL_CARD_MENTAL:
            return [
                SimpleNamespace(label=label, x=230, y=1460, w=430, h=1660, cx=330, cy=1560, frame=np.zeros((10, 10, 3), dtype=np.uint8)),
                SimpleNamespace(label=label, x=440, y=1540, w=635, h=1740, cx=538, cy=1640, frame=np.zeros((10, 10, 3), dtype=np.uint8)),
                SimpleNamespace(label=label, x=650, y=1540, w=845, h=1740, cx=748, cy=1640, frame=np.zeros((10, 10, 3), dtype=np.uint8)),
            ]
        if label == ProducerLabels.P_DRINK:
            return [
                SimpleNamespace(label=label, x=70, y=2220, w=185, h=2338, cx=128, cy=2280, frame=np.zeros((10, 10, 3), dtype=np.uint8)),
            ]
        return original_filter_by_label(label)

    monkeypatch.setattr(results, "filter_by_label", _filter_by_label)
    monkeypatch.setattr(
        ui_module,
        "ocr_text",
        lambda _frame: "受け取るスキルカードを選んでください 前途洋々 パラメータ+8 元気+7 受け取る 獲得ガイド",
    )

    phase = ui_module.classify_gameplay_phase(results)
    position = ui_module.classify_pipeline_position(results, phase=phase)

    assert phase == "skill_reward"
    assert position == GameplayPosition.SKILL_REWARD_SELECTED


def test_skill_reward_handler_advances_showcase():
    device = _DeviceStub()
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(set()),
        device=device,
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        clip_manager=None,
    )
    ctx = ProduceContext()
    ctx.pending_skill_reward_index = 0
    ctx.pending_skill_reward_label = "ワクワクが止まらない+"
    handler = SkillRewardHandler()

    result = handler.handle(app, ctx, "skill_reward", GameplayPosition.SKILL_REWARD_SHOWCASE)

    assert result.status == "ok"
    assert device.clicks == [(540, 2059, "skill_reward_showcase_advance")]
    assert ctx.pending_skill_reward_index == 0
    assert ctx.pending_skill_reward_label == "ワクワクが止まらない+"
    assert ctx.handler_state["unknown_retry_override"] == {
        "reason": "skill_reward_showcase_transition",
        "retry_limit": 15,
        "retry_sleep": 1.0,
    }


def test_skill_reward_handler_sets_transition_retry_after_selection(monkeypatch):
    device = _DeviceStub()
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(set()),
        device=device,
        debug_tools=SimpleNamespace(add_box=lambda *args, **kwargs: None),
        clip_manager=None,
    )
    ctx = ProduceContext()
    handler = SkillRewardHandler()

    monkeypatch.setattr(
        skill_reward_module,
        "execute_skill_reward_step",
        lambda *_args, **_kwargs: SkillRewardStepResult(status="selected"),
    )

    result = handler.handle(app, ctx, "skill_reward", GameplayPosition.SKILL_REWARD_IDLE)

    assert result.status == "ok"
    assert ctx.handler_state.get("unknown_retry_override") is None
