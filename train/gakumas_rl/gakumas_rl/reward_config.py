"""参数化奖励配置，把所有奖励权重集中管理，支持 CLI / 文件 / API 覆盖。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class RewardConfig:
    """统一奖励参数集。

    所有权重均可通过 CLI ``--reward-<snake_case>=<value>`` 或
    JSON 文件 ``--reward-config=<path>`` 覆盖。
    """

    # ── 奖励模式 ──────────────────────────────────────────────────
    #  'score' : 追求考试高评价
    #  'clear' : 追求课程过线 / 效率优先
    reward_mode: str = 'score'

    # ── Dense shaping（潜势差分）权重 ──────────────────────────────
    #  shape_scale 控制潜势差分在总奖励中的比例
    shape_scale: float = 1.85
    #  5 个潜势函数各自的权重
    goal_weight: float = 0.95        # Φ_goal：离通关目标的距离
    eval_weight: float = 1.55        # Φ_eval：终局评价边际价值
    archetype_weight: float = 1.10   # Φ_archetype：流派资源估值
    risk_weight: float = 0.70        # Φ_risk：负面状态 / 体力风险
    efficiency_weight: float = 0.28  # Φ_efficiency：收官效率

    # ── Dense 子项参数 ─────────────────────────────────────────────
    turn_window_weight: float = 0.80       # 回合颜色窗口价值权重
    judging_alignment_weight: float = 0.70  # 审查基准与当前三维匹配度权重
    efficiency_gate: float = 0.92          # Φ_efficiency 启动阈值
    efficiency_overshoot_penalty: float = 0.32  # 超线惩罚系数

    # ── 资源估值缩放（Φ_archetype）──────────────────────────────
    #  sense / logic / anomaly 流派各自资源的 soft_cap 与权重
    #  这些只是默认值；实际的 per-resource 估值在 exam_runtime 的
    #  _phi_archetype() 中已用 _resource_curve() 做了边际递减压缩，
    #  这里提供流派级别的总系数缩放。
    sense_resource_scale: float = 1.0
    logic_resource_scale: float = 1.0
    anomaly_resource_scale: float = 1.0

    # ── Sparse 终局奖励 ───────────────────────────────────────────
    terminal_pass_reward: float = 5.5      # 考试达标一次性奖励
    terminal_eval_weight: float = 5.0      # 终局分数曲线加权
    terminal_stamina_weight: float = 0.45  # 剩余体力加权
    terminal_speed_weight: float = 0.40    # 剩余回合加权
    terminal_failure_weight: float = 5.0   # 失败惩罚（越大惩罚越重）
    terminal_force_end_bonus: float = 1.8  # 达到最高分自动结束奖励
    terminal_nia_bonus: float = 1.25       # NIA fan vote 终局奖励
    lesson_clear_reward: float = 4.0       # 课程 Clear 一次性奖励
    lesson_perfect_reward: float = 6.5     # 课程 Perfect 一次性奖励
    overshoot_penalty: float = 0.95        # 超线惩罚（终局）

    # ── 截断 / 惩罚机制 ──────────────────────────────────────────
    invalid_action_penalty: float = -0.45  # 选择了不可用动作的即时惩罚
    skip_turn_penalty: float = -0.02       # 空过回合（未出牌直接结束回合）的惩罚
    stamina_death_penalty: float = -1.0    # 体力归零导致强制结束时的额外惩罚
    consecutive_end_turn_penalty: float = -0.04  # 连续结束回合的递增惩罚

    # ── 里程碑奖励（Milestone）──────────────────────────────────
    #  在分数首次达到某些关键比例时发放一次性奖励
    milestone_25_reward: float = 0.0       # 首次达到 25% 目标
    milestone_50_reward: float = 0.5       # 首次达到 50% 目标
    milestone_75_reward: float = 1.0       # 首次达到 75% 目标
    milestone_100_reward: float = 2.0      # 首次达到 100% 目标（通关）

    # ── 额外密集信号 ──────────────────────────────────────────────
    score_delta_scale: float = 0.0         # 直接用分数差分做密集奖励的缩放
    resource_gain_scale: float = 0.0       # 正向资源增量密集奖励缩放
    card_play_reward: float = 0.0          # 每次成功出牌的微小正向奖励
    drink_use_reward: float = 0.0          # 成功使用饮料的微小正向奖励

    # ── 全局缩放 ──────────────────────────────────────────────────
    reward_scale: float = 1.0              # 最终奖励值全局缩放
    reward_clip: float = 25.0             # >0 时对最终奖励做 clip(-c, c)

    def to_dict(self) -> dict[str, Any]:
        """转成普通 dict，方便序列化与日志记录。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RewardConfig':
        """从 dict 构建，忽略无关的 key。"""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> 'RewardConfig':
        """从 JSON 文件加载。"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        """保存到 JSON 文件。"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def merge(self, overrides: dict[str, Any]) -> 'RewardConfig':
        """用 dict 中的值覆盖当前配置，返回新实例。"""
        current = self.to_dict()
        valid_keys = {f.name for f in fields(self.__class__)}
        for k, v in overrides.items():
            if k in valid_keys:
                current[k] = v
        return self.__class__.from_dict(current)


# ── 预置 profile ──────────────────────────────────────────────────

SCORE_PROFILE = RewardConfig(
    reward_mode='score',
    shape_scale=1.85,
    goal_weight=0.95,
    eval_weight=1.55,
    archetype_weight=1.10,
    risk_weight=0.70,
    efficiency_weight=0.28,
    turn_window_weight=0.85,
    judging_alignment_weight=0.85,
    efficiency_gate=0.92,
    efficiency_overshoot_penalty=0.32,
    terminal_pass_reward=5.5,
    terminal_eval_weight=5.0,
    terminal_stamina_weight=0.45,
    terminal_speed_weight=0.40,
    terminal_failure_weight=5.0,
    terminal_force_end_bonus=1.8,
    terminal_nia_bonus=1.25,
    lesson_clear_reward=4.0,
    lesson_perfect_reward=6.5,
    overshoot_penalty=0.95,
    invalid_action_penalty=-0.60,
    skip_turn_penalty=-0.02,
    consecutive_end_turn_penalty=-0.04,
    reward_clip=25.0,
    milestone_50_reward=0.5,
    milestone_75_reward=1.0,
    milestone_100_reward=2.0,
)

CLEAR_PROFILE = RewardConfig(
    reward_mode='clear',
    shape_scale=1.55,
    goal_weight=1.45,
    eval_weight=1.00,
    archetype_weight=0.68,
    risk_weight=1.18,
    efficiency_weight=0.60,
    turn_window_weight=0.45,
    judging_alignment_weight=0.45,
    efficiency_gate=0.75,
    efficiency_overshoot_penalty=1.45,
    terminal_pass_reward=7.2,
    terminal_eval_weight=3.8,
    terminal_stamina_weight=0.90,
    terminal_speed_weight=0.95,
    terminal_failure_weight=7.0,
    terminal_force_end_bonus=2.1,
    terminal_nia_bonus=1.10,
    lesson_clear_reward=5.4,
    lesson_perfect_reward=8.2,
    overshoot_penalty=2.60,
    invalid_action_penalty=-0.95,
    skip_turn_penalty=-0.08,
    stamina_death_penalty=-2.3,
    consecutive_end_turn_penalty=-0.15,
    reward_clip=25.0,
    milestone_50_reward=0.8,
    milestone_75_reward=1.5,
    milestone_100_reward=3.0,
)

REWARD_PROFILES: dict[str, RewardConfig] = {
    'score': SCORE_PROFILE,
    'clear': CLEAR_PROFILE,
}


def build_reward_config(
    reward_mode: str = 'score',
    reward_config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> RewardConfig:
    """根据 mode / 文件 / CLI 覆盖构建奖励配置。

    优先级：overrides > config_file > profile_defaults
    """
    config = REWARD_PROFILES.get(reward_mode, SCORE_PROFILE)
    if reward_config_path:
        file_config = RewardConfig.from_json(reward_config_path)
        config = file_config
    if overrides:
        config = config.merge(overrides)
    # 确保 reward_mode 一致
    if config.reward_mode != reward_mode and not reward_config_path:
        config = config.merge({'reward_mode': reward_mode})
    return config
