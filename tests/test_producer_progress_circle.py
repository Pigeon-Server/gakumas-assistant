"""进度圆圈 OCR 解析 + HSV 卡片状态检测 单元测试。

测试覆盖:
  - _parse_progress_circle: 干净输入 / OCR 噪点输入 / 边界情况
  - _is_card_grayed_out: 灰色卡片（低亮度）和正常卡片
  - _is_card_exchanged: 已交换卡片（低亮度+低饱和度）和正常卡片
"""
import numpy as np
import pytest

from src.core.tasks.producer_challenge.gameplay import decision as decision_module

_parse = decision_module._parse_progress_circle


# ────────────────────────────────────────────
#  _parse_progress_circle: 基本格式
# ────────────────────────────────────────────

class TestParseProgressCircleClean:
    """干净 OCR 文本解析。"""

    def test_clear_not_achieved(self):
        """'CLEARまで10' → 未 CLEAR, 距 CLEAR 还需 10"""
        result = _parse("CLEARまで10")
        assert result is not None
        assert result["clear_achieved"] is False
        assert result["remaining_to_clear"] == 10
        assert result["remaining_to_perfect"] == 0

    def test_perfect_stage(self):
        """'PERFECTまで175CLEAR' → 已 CLEAR, 距 PERFECT 还需 175"""
        result = _parse("PERFECTまで175CLEAR")
        assert result is not None
        assert result["clear_achieved"] is True
        assert result["remaining_to_clear"] == 0
        assert result["remaining_to_perfect"] == 175

    def test_perfect_small_number(self):
        """'PERFECTまで5CLEAR' → 已 CLEAR, 距 PERFECT 还需 5"""
        result = _parse("PERFECTまで5CLEAR")
        assert result is not None
        assert result["clear_achieved"] is True
        assert result["remaining_to_perfect"] == 5

    def test_clear_large_number(self):
        """'CLEARまで999' → 未 CLEAR, 距 CLEAR 还需 999"""
        result = _parse("CLEARまで999")
        assert result is not None
        assert result["clear_achieved"] is False
        assert result["remaining_to_clear"] == 999


# ────────────────────────────────────────────
#  _parse_progress_circle: OCR 噪点
# ────────────────────────────────────────────

class TestParseProgressCircleNoisy:
    """OCR 噪点/误读场景。"""

    def test_perfec7_variant(self):
        """'PERFEC7まで50CLEAR' → 识别为 PERFECT 阶段"""
        result = _parse("PERFEC7まで50CLEAR")
        assert result is not None
        assert result["clear_achieved"] is True
        assert result["remaining_to_perfect"] == 50

    def test_c1ear_made_variant(self):
        """'C1EARまて15' → 识别为 CLEAR 阶段（C1EAR + まて）"""
        result = _parse("C1EARまて15")
        assert result is not None
        assert result["clear_achieved"] is False
        assert result["remaining_to_clear"] == 15

    def test_perfecf_variant(self):
        """'PERFECFまで80CLEAR' → 识别为 PERFECT 阶段"""
        result = _parse("PERFECFまで80CLEAR")
        assert result is not None
        assert result["clear_achieved"] is True
        assert result["remaining_to_perfect"] == 80

    def test_ciear_variant(self):
        """'CIEARまで42' → 识别为 CLEAR 阶段"""
        result = _parse("CIEARまで42")
        assert result is not None
        assert result["clear_achieved"] is False
        assert result["remaining_to_clear"] == 42

    def test_spaces_and_fullwidth(self):
        """含空格/全角空格的文本也能正确解析"""
        result = _parse("PERFECT　まで 100 CLEAR")
        assert result is not None
        assert result["clear_achieved"] is True
        assert result["remaining_to_perfect"] == 100


# ────────────────────────────────────────────
#  _parse_progress_circle: 非进度圆圈文本
# ────────────────────────────────────────────

class TestParseProgressCircleReject:
    """不属于进度圆圈的文本应返回 None。"""

    def test_plain_number(self):
        assert _parse("175") is None

    def test_empty_string(self):
        assert _parse("") is None

    def test_none_input(self):
        assert _parse(None) is None

    def test_random_text(self):
        assert _parse("Hello World 42") is None

    def test_only_number_no_keyword(self):
        assert _parse("12345") is None


# ────────────────────────────────────────────
#  HSV 卡片状态检测辅助
# ────────────────────────────────────────────

def _make_box(x, y, x2, y2):
    """创建模拟的 Yolo_Box 用于 HSV 检测测试。"""
    from types import SimpleNamespace
    return SimpleNamespace(x=x, y=y, w=x2, h=y2, cx=(x + x2) // 2, cy=(y + y2) // 2)


def _make_solid_frame(w, h, rgb):
    """创建纯色 RGB frame (uint8)。"""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = rgb
    return frame


def _make_frame_with_region(fw, fh, box, rgb):
    """创建背景黑色但指定区域为 rgb 颜色的 frame。"""
    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    frame[box.y:box.h, box.x:box.w] = rgb
    return frame


def _add_jpg_noise(frame, strength=5, seed=42):
    """在 frame 上叠加随机噪点，模拟 JPG 压缩伪影。"""
    rng = np.random.RandomState(seed)
    noise = rng.randint(-strength, strength + 1, frame.shape, dtype=np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


# ────────────────────────────────────────────
#  _is_card_grayed_out 测试
# ────────────────────────────────────────────

from src.core.tasks.producer_challenge.gameplay.consult import _is_card_grayed_out


class TestIsCardGrayedOut:

    def test_normal_card_bright(self):
        """亮色卡片 V≈230 应判定为正常"""
        box = _make_box(10, 10, 110, 210)
        # RGB (200, 200, 230): 浅蓝色, HSV V≈230
        frame = _make_frame_with_region(200, 300, box, (200, 200, 230))
        assert _is_card_grayed_out(box, frame) is False

    def test_grayed_card_dark(self):
        """灰暗卡片 V≈120 应判定为灰色"""
        box = _make_box(10, 10, 110, 210)
        # RGB (100, 100, 120): 暗灰, HSV V≈120
        frame = _make_frame_with_region(200, 300, box, (100, 100, 120))
        assert _is_card_grayed_out(box, frame) is True

    def test_borderline_below_threshold(self):
        """亮度恰好低于阈值 170 → 灰色"""
        box = _make_box(10, 10, 110, 210)
        frame = _make_frame_with_region(200, 300, box, (160, 160, 160))
        assert _is_card_grayed_out(box, frame) is True

    def test_borderline_above_threshold(self):
        """亮度恰好高于阈值 170 → 正常"""
        box = _make_box(10, 10, 110, 210)
        frame = _make_frame_with_region(200, 300, box, (180, 180, 180))
        assert _is_card_grayed_out(box, frame) is False

    def test_robust_to_jpg_noise(self):
        """加 JPG 噪点后检测结果不变"""
        box = _make_box(10, 10, 110, 210)
        # 正常卡片加噪
        bright_frame = _make_frame_with_region(200, 300, box, (200, 200, 230))
        noisy_bright = _add_jpg_noise(bright_frame, strength=8)
        assert _is_card_grayed_out(box, noisy_bright) is False

        # 灰色卡片加噪
        dark_frame = _make_frame_with_region(200, 300, box, (100, 100, 120))
        noisy_dark = _add_jpg_noise(dark_frame, strength=8)
        assert _is_card_grayed_out(box, noisy_dark) is True

    def test_empty_region_returns_false(self):
        """空区域（大小为0）应返回 False"""
        box = _make_box(10, 10, 10, 10)  # w==x → 宽度为 0
        frame = _make_solid_frame(200, 300, (200, 200, 200))
        assert _is_card_grayed_out(box, frame) is False


# ────────────────────────────────────────────
#  _is_card_exchanged 测试
# ────────────────────────────────────────────

from src.core.tasks.producer_challenge.gameplay.consult import _is_card_exchanged


class TestIsCardExchanged:

    def test_normal_card(self):
        """正常未交换卡片：高亮度 + 高饱和度"""
        box = _make_box(10, 10, 110, 210)
        # RGB (200, 50, 50): 红色, V≈200, S较高
        frame = _make_frame_with_region(200, 300, box, (200, 50, 50))
        assert _is_card_exchanged(box, frame) is False

    def test_exchanged_card(self):
        """已交换卡片：低亮度 + 低饱和度（灰蒙蒙遮罩）"""
        box = _make_box(10, 10, 110, 210)
        # RGB (100, 100, 100): 纯灰, V=100, S=0
        frame = _make_frame_with_region(200, 300, box, (100, 100, 100))
        assert _is_card_exchanged(box, frame) is True

    def test_dark_but_saturated_not_exchanged(self):
        """暗色但高饱和 → 不是已交换（可能是暗色卡片）"""
        box = _make_box(10, 10, 110, 210)
        # RGB (150, 20, 20): 暗红, V≈150, S很高
        frame = _make_frame_with_region(200, 300, box, (150, 20, 20))
        assert _is_card_exchanged(box, frame) is False

    def test_bright_desaturated_not_exchanged(self):
        """亮色但低饱和 → 不是已交换（V > threshold）"""
        box = _make_box(10, 10, 110, 210)
        # RGB (230, 225, 220): 略偏暖白, V≈230, S低
        frame = _make_frame_with_region(200, 300, box, (230, 225, 220))
        assert _is_card_exchanged(box, frame) is False

    def test_robust_to_jpg_noise(self):
        """JPG 噪点不影响检测结果"""
        box = _make_box(10, 10, 110, 210)
        # 正常卡片加噪
        normal_frame = _make_frame_with_region(200, 300, box, (200, 50, 50))
        noisy_normal = _add_jpg_noise(normal_frame, strength=8)
        assert _is_card_exchanged(box, noisy_normal) is False

        # 已交换卡片加噪
        exchanged_frame = _make_frame_with_region(200, 300, box, (100, 100, 100))
        noisy_exchanged = _add_jpg_noise(exchanged_frame, strength=8)
        assert _is_card_exchanged(box, noisy_exchanged) is True

    def test_empty_region_returns_false(self):
        """空区域应返回 False"""
        box = _make_box(10, 10, 10, 10)
        frame = _make_solid_frame(200, 300, (100, 100, 100))
        assert _is_card_exchanged(box, frame) is False
