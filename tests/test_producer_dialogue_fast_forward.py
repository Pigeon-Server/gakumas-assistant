from types import SimpleNamespace

import numpy as np

from src.constants.game.producer_gameplay import GameplayPosition
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.core.tasks.producer_challenge.context import ProduceContext
from src.core.tasks.producer_challenge.gameplay import dialogue as dialogue_module


class _BoxList(list):
    def first(self):
        return self[0] if self else None


class _ResultsStub:
    def __init__(self, labels, *, label_boxes=None):
        self._labels = set(labels)
        self._label_boxes = dict(label_boxes or {})

    def filter_by_label(self, label):
        if label in self._label_boxes:
            return _BoxList(self._label_boxes[label])
        if label in self._labels:
            return _BoxList([SimpleNamespace(cx=300, cy=300, x=260, y=260, w=340, h=340)])
        return _BoxList()


class _DeviceStub:
    def __init__(self):
        self.clicks = []

    def click(self, x, y, el_label=""):
        self.clicks.append((int(x), int(y), str(el_label or "")))

    def click_element(self, element):
        self.clicks.append(("element", element))


def test_dialogue_fast_forward_clicks_once_then_advances(monkeypatch):
    monkeypatch.setattr(
        dialogue_module,
        "collect_dialogue_option_candidates",
        lambda *_args, **_kwargs: [],
    )

    ff_box = SimpleNamespace(cx=200, cy=210, x=160, y=180, w=240, h=240)
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(
            {ProducerLabels.PLOT_FAST_FORWARD_BUTTON},
            label_boxes={ProducerLabels.PLOT_FAST_FORWARD_BUTTON: [ff_box]},
        ),
        device=_DeviceStub(),
    )
    ctx = ProduceContext()

    first = dialogue_module.execute_dialogue_step(
        app,
        ctx,
        position=GameplayPosition.DIALOGUE_CONTINUE,
    )
    second = dialogue_module.execute_dialogue_step(
        app,
        ctx,
        position=GameplayPosition.DIALOGUE_CONTINUE,
    )

    assert first is not None and first.status == "fast_forward"
    assert second is not None and second.status == "advanced"
    assert app.device.clicks[0] == ("element", ff_box)
    assert app.device.clicks[1] == (540, 1919, "dialogue-advance")


def test_dialogue_fast_forward_flag_resets_when_button_disappears(monkeypatch):
    monkeypatch.setattr(
        dialogue_module,
        "collect_dialogue_option_candidates",
        lambda *_args, **_kwargs: [],
    )

    ff_box = SimpleNamespace(cx=200, cy=210, x=160, y=180, w=240, h=240)
    results_with_ff = _ResultsStub(
        {BaseUILabels.PLOT_FAST_FORWARD_BUTTON},
        label_boxes={BaseUILabels.PLOT_FAST_FORWARD_BUTTON: [ff_box]},
    )
    results_without_ff = _ResultsStub(set())
    device = _DeviceStub()
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=results_with_ff,
        device=device,
    )
    ctx = ProduceContext()

    first = dialogue_module.execute_dialogue_step(
        app,
        ctx,
        position=GameplayPosition.DIALOGUE_CONTINUE,
    )
    app.latest_results = results_without_ff
    dialogue_module.execute_dialogue_step(
        app,
        ctx,
        position=GameplayPosition.DIALOGUE_CONTINUE,
    )
    app.latest_results = results_with_ff
    third = dialogue_module.execute_dialogue_step(
        app,
        ctx,
        position=GameplayPosition.DIALOGUE_CONTINUE,
    )

    assert first is not None and first.status == "fast_forward"
    assert third is not None and third.status == "fast_forward"
    assert device.clicks[0] == ("element", ff_box)
    assert device.clicks[2] == ("element", ff_box)


def test_dialogue_fast_forward_orange_enabled_state_skips_toggle(monkeypatch):
    monkeypatch.setattr(
        dialogue_module,
        "collect_dialogue_option_candidates",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        dialogue_module,
        "probe_fast_forward_enabled_state",
        lambda *_args, **_kwargs: (True, 0.42),
    )

    ff_box = SimpleNamespace(cx=200, cy=210, x=160, y=180, w=240, h=240)
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(
            {ProducerLabels.PLOT_FAST_FORWARD_BUTTON},
            label_boxes={ProducerLabels.PLOT_FAST_FORWARD_BUTTON: [ff_box]},
        ),
        device=_DeviceStub(),
    )
    ctx = ProduceContext()

    result = dialogue_module.execute_dialogue_step(
        app,
        ctx,
        position=GameplayPosition.DIALOGUE_CONTINUE,
    )

    assert result is not None and result.status == "advanced"
    assert app.device.clicks == [(540, 1919, "dialogue-advance")]


def test_dialogue_continue_prefers_skip_button(monkeypatch):
    monkeypatch.setattr(
        dialogue_module,
        "collect_dialogue_option_candidates",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        dialogue_module,
        "_try_enable_story_fast_forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("有 skip 时不应先点快进")),
    )

    skip_box = SimpleNamespace(cx=220, cy=210, x=180, y=180, w=260, h=240)
    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(
            {BaseUILabels.SKIP_BUTTON},
            label_boxes={BaseUILabels.SKIP_BUTTON: [skip_box]},
        ),
        device=_DeviceStub(),
    )
    ctx = ProduceContext()

    result = dialogue_module.execute_dialogue_step(
        app,
        ctx,
        position=GameplayPosition.DIALOGUE_CONTINUE,
    )

    assert result is not None and result.status == "skipped"
    assert app.device.clicks == [("element", skip_box)]


def test_dialogue_options_probes_action_info_and_enriches_description(monkeypatch):
    option0 = SimpleNamespace(name="o0")
    option1 = SimpleNamespace(name="o1")
    candidates = [
        dialogue_module.DialogueOptionCandidate(index=0, title="选项A", selected=False, box=option0, metadata={}),
        dialogue_module.DialogueOptionCandidate(index=1, title="选项B", selected=False, box=option1, metadata={}),
    ]

    monkeypatch.setattr(
        dialogue_module,
        "collect_dialogue_option_candidates",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(dialogue_module, "_is_outing_context", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        dialogue_module,
        "_probe_dialogue_option_effects",
        lambda _app, _cands: (
            _cands[0].metadata.update({"option_effect": "体力+5"}),
            _cands[1].metadata.update({"option_effect": "Pポイント+20"}),
        ),
    )
    monkeypatch.setattr(dialogue_module, "decide_dialogue_option", lambda *_args, **_kwargs: 0)

    app = SimpleNamespace(
        latest_frame=np.zeros((2340, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub({ProducerLabels.PC_ACTION_INFO}),
        device=_DeviceStub(),
    )
    ctx = ProduceContext()

    result = dialogue_module.execute_dialogue_step(
        app,
        ctx,
        position=GameplayPosition.DIALOGUE_OPTIONS,
    )

    assert result is not None and result.status == "selected"
    assert candidates[0].metadata["description"].startswith("効果: ")
    assert candidates[1].metadata["description"].startswith("効果: ")
    assert app.device.clicks == [("element", option0)]


def test_enrich_dialogue_option_descriptions_prefers_db_description():
    candidate = dialogue_module.DialogueOptionCandidate(
        index=0,
        title="ビジュアルを重点的に",
        selected=False,
        box=None,
        metadata={
            "option_effect": "OCR噪声描述",
            "dialogue_db_description": "ビジュアル上昇+20",
        },
    )

    dialogue_module._enrich_dialogue_option_descriptions([candidate])

    assert candidate.metadata["description"] == "効果: ビジュアル上昇+20"


def test_dialogue_option_info_context_allows_progress_fallback_without_effect_text():
    option_boxes = [
        SimpleNamespace(cx=320, cy=1510, x=180, y=1440, w=900, h=1560),
        SimpleNamespace(cx=320, cy=1670, x=180, y=1600, w=900, h=1720),
    ]
    app = SimpleNamespace(
        latest_results=_ResultsStub(
            {ProducerLabels.PC_PROGRESS},
            label_boxes={ProducerLabels.UNIVERSAL_OPTIONS: option_boxes},
        ),
    )

    assert dialogue_module._is_dialogue_option_info_context(
        app,
        GameplayPosition.DIALOGUE_OPTIONS,
    )


def test_dialogue_option_info_context_allows_progress_fallback_with_effect_text(monkeypatch):
    option_boxes = [
        SimpleNamespace(cx=320, cy=1510, x=180, y=1440, w=900, h=1560),
        SimpleNamespace(cx=320, cy=1670, x=180, y=1600, w=900, h=1720),
    ]
    app = SimpleNamespace(
        latest_results=_ResultsStub(
            {ProducerLabels.PC_PROGRESS},
            label_boxes={ProducerLabels.UNIVERSAL_OPTIONS: option_boxes},
        ),
    )
    monkeypatch.setattr(dialogue_module, "_extract_action_info_description", lambda _app: "効果: 体力回復+10")

    assert dialogue_module._is_dialogue_option_info_context(
        app,
        GameplayPosition.DIALOGUE_OPTIONS,
    )


def test_extract_action_info_description_fallback_reads_option_upper_region(monkeypatch):
    option_boxes = [
        SimpleNamespace(cx=320, cy=1510, x=180, y=1440, w=900, h=1560),
        SimpleNamespace(cx=320, cy=1670, x=180, y=1600, w=900, h=1720),
    ]
    app = SimpleNamespace(
        latest_frame=np.zeros((1920, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(
            {ProducerLabels.PC_PROGRESS},
            label_boxes={ProducerLabels.UNIVERSAL_OPTIONS: option_boxes},
        ),
    )
    monkeypatch.setattr(dialogue_module, "ocr_text", lambda _img: "効果: 体力回復+10")

    text = dialogue_module._extract_action_info_description(app)

    assert "体力回復+10" in text


def test_extract_action_info_description_white_panel_near_anchor(monkeypatch):
    option_boxes = [
        SimpleNamespace(cx=540, cy=1500, x=180, y=1440, w=900, h=1560),
        SimpleNamespace(cx=540, cy=1660, x=180, y=1600, w=900, h=1720),
    ]
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    # 在估计锚点附近放置白色信息框（上方、居中、宽于单个选项）。
    frame[1180:1330, 140:940] = 255
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub(
            {ProducerLabels.PC_PROGRESS},
            label_boxes={ProducerLabels.UNIVERSAL_OPTIONS: option_boxes},
        ),
    )
    monkeypatch.setattr(
        dialogue_module._ACTION_INFO_OCR,
        "ocr",
        lambda _img: (_ for _ in ()).throw(RuntimeError("mock ocr fail")),
    )
    # 仅对白色信息框尺寸返回有效文本，验证确实命中了白框兜底。
    monkeypatch.setattr(
        dialogue_module,
        "ocr_text",
        lambda img: "効果: 体力回復+10"
        if abs(int(img.shape[0]) - 150) <= 12 and abs(int(img.shape[1]) - 800) <= 20
        else "",
    )

    text = dialogue_module._extract_action_info_description(app)

    assert "体力回復+10" in text


def test_extract_action_info_description_white_panel_far_from_anchor_is_rejected(monkeypatch):
    option_boxes = [
        SimpleNamespace(cx=540, cy=1500, x=180, y=1440, w=900, h=1560),
        SimpleNamespace(cx=540, cy=1660, x=180, y=1600, w=900, h=1720),
    ]
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    # 放一个离估计中心过远的白块，应被几何约束过滤掉。
    frame[760:900, 140:940] = 255
    app = SimpleNamespace(
        latest_frame=frame,
        latest_results=_ResultsStub(
            {ProducerLabels.PC_PROGRESS},
            label_boxes={ProducerLabels.UNIVERSAL_OPTIONS: option_boxes},
        ),
    )
    monkeypatch.setattr(
        dialogue_module._ACTION_INFO_OCR,
        "ocr",
        lambda _img: (_ for _ in ()).throw(RuntimeError("mock ocr fail")),
    )
    monkeypatch.setattr(
        dialogue_module,
        "ocr_text",
        lambda img: "効果: 体力回復+10"
        if abs(int(img.shape[0]) - 140) <= 12 and abs(int(img.shape[1]) - 800) <= 20
        else "",
    )

    text = dialogue_module._extract_action_info_description(app)

    assert text == ""


def test_probe_dialogue_option_effects_waits_for_refresh_after_click(monkeypatch):
    option0 = SimpleNamespace(name="o0")
    option1 = SimpleNamespace(name="o1")
    candidates = [
        dialogue_module.DialogueOptionCandidate(index=0, title="选项A", selected=False, box=option0, metadata={}),
        dialogue_module.DialogueOptionCandidate(index=1, title="选项B", selected=False, box=option1, metadata={}),
    ]
    effect_iter = iter(["", "効果: 体力+5", "", "効果: Pポイント+20"])
    wait_calls = []
    app = SimpleNamespace(
        latest_frame=np.zeros((1920, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(set()),
        device=_DeviceStub(),
        game_utils=SimpleNamespace(wait_frame_stable=lambda **_kwargs: wait_calls.append(1)),
    )

    monkeypatch.setattr(
        dialogue_module,
        "_extract_action_info_description",
        lambda _app: next(effect_iter, ""),
    )
    monkeypatch.setattr(dialogue_module.time, "sleep", lambda *_args, **_kwargs: None)

    dialogue_module._probe_dialogue_option_effects(app, candidates)

    assert candidates[0].metadata["option_effect"] == "効果: 体力+5"
    assert candidates[1].metadata["option_effect"] == "効果: Pポイント+20"
    assert app.device.clicks == [("element", option0), ("element", option1)]
    assert len(wait_calls) >= 2


def test_probe_dialogue_option_effects_does_not_early_break_without_action_info(monkeypatch):
    option0 = SimpleNamespace(name="o0")
    option1 = SimpleNamespace(name="o1")
    candidates = [
        dialogue_module.DialogueOptionCandidate(index=0, title="选项A", selected=False, box=option0, metadata={}),
        dialogue_module.DialogueOptionCandidate(index=1, title="选项B", selected=False, box=option1, metadata={}),
    ]
    app = SimpleNamespace(
        latest_frame=np.zeros((1920, 1080, 3), dtype=np.uint8),
        latest_results=_ResultsStub(set()),
        device=_DeviceStub(),
    )
    calls = {"count": 0}

    def _mock_read(*_args, **_kwargs):
        calls["count"] += 1
        return "" if calls["count"] == 1 else "効果: ボーカル上昇+55"

    monkeypatch.setattr(dialogue_module, "_read_action_info_after_option_click", _mock_read)
    monkeypatch.setattr(dialogue_module.time, "sleep", lambda *_args, **_kwargs: None)

    dialogue_module._probe_dialogue_option_effects(app, candidates)

    assert app.device.clicks == [("element", option0), ("element", option1)]
    assert candidates[0].metadata.get("option_effect") in {None, ""}
    assert candidates[1].metadata.get("option_effect") == "効果: ボーカル上昇+55"
