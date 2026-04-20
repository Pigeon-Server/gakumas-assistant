"""自动训练配置和早停机制。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np


@dataclass
class AutoTrainingConfig:
    """自动训练配置。"""

    # 自动步数配置
    full_auto: bool = False
    auto_total_timesteps: bool = False
    min_timesteps: int = 100_000
    max_timesteps: int = 5_000_000
    timesteps_per_eval: int = 50_000
    dynamic_eval_schedule: bool = False
    dynamic_checkpoint_schedule: bool = False
    target_num_evaluations: int = 12
    min_eval_interval: int = 10_000
    max_eval_interval: int = 100_000

    # 早停配置
    enable_early_stopping: bool = False
    patience: int = 10  # 多少次评估无改善后停止
    min_delta: float = 0.01  # 最小改善阈值
    baseline_episodes: int = 5  # 基线评估回合数

    # 动态学习率
    enable_lr_schedule: bool = False
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 5


@dataclass(frozen=True)
class ExamRandomizationCurriculumConfig:
    """按训练进度逐步放开考试随机化轴。"""

    enabled: bool = False
    stage_type_start_ratio: float = 0.10
    use_after_item_start_ratio: float = 0.25


class EarlyStoppingCallback:
    """早停回调，类似 YOLO 的训练策略。"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.01,
        baseline_episodes: int = 5,
        min_training_steps: int = 0,
        verbose: bool = True,
        window_size: int = 5,
    ):
        """初始化早停回调。

        Args:
            patience: 多少次评估无改善后停止
            min_delta: 最小改善阈值（相对改善）
            baseline_episodes: 基线评估回合数
            min_training_steps: 在允许早停前至少训练的步数
            verbose: 是否打印详细信息
            window_size: 滑动窗口大小
        """
        self.patience = patience
        self.min_delta = min_delta
        self.baseline_episodes = baseline_episodes
        self.min_training_steps = max(int(min_training_steps), 0)
        self.verbose = verbose
        self.window_size = max(int(window_size), 1)

        self.best_mean_reward = -np.inf
        self.best_window_mean = -np.inf
        self.wait = 0
        self.stopped_step = 0
        self.should_stop = False
        self.history: list[dict[str, Any]] = []
        self._recent_rewards: list[float] = []

    def on_evaluation(self, step: int, mean_reward: float, std_reward: float) -> bool:
        """评估后调用，使用滑动窗口平均进行早停判断。

        Args:
            step: 当前训练步数
            mean_reward: 平均奖励
            std_reward: 奖励标准差

        Returns:
            是否应该停止训练
        """
        self.history.append({
            'step': step,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
        })

        # 维护单次最佳（用于 best checkpoint 等）
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

        # 滑动窗口
        self._recent_rewards.append(mean_reward)
        if len(self._recent_rewards) > self.window_size:
            self._recent_rewards.pop(0)

        # 窗口未满时不触发早停
        if len(self._recent_rewards) < self.window_size:
            if self.verbose:
                print(
                    f"[EarlyStopping] Step {step}: Filling window "
                    f"({len(self._recent_rewards)}/{self.window_size}), "
                    f"current: {mean_reward:.4f}"
                )
            return False

        window_mean = float(np.mean(self._recent_rewards))

        if self.best_window_mean == -np.inf:
            improvement = np.inf
        else:
            improvement = (window_mean - self.best_window_mean) / max(abs(self.best_window_mean), 1e-8)

        if window_mean > self.best_window_mean + self.min_delta:
            self.best_window_mean = window_mean
            self.wait = 0
            if self.verbose:
                print(
                    f"[EarlyStopping] Step {step}: New best window mean {window_mean:.4f} "
                    f"(improvement: {improvement:.2%}, current: {mean_reward:.4f})"
                )
        else:
            if step < self.min_training_steps:
                if self.verbose:
                    print(
                        f"[EarlyStopping] Step {step}: Warmup phase, early stopping disabled until "
                        f"{self.min_training_steps}"
                    )
                return False

            self.wait += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] Step {step}: No improvement ({self.wait}/{self.patience}), "
                    f"window_mean: {window_mean:.4f}, best_window_mean: {self.best_window_mean:.4f}, "
                    f"current: {mean_reward:.4f}"
                )

            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_step = step
                if self.verbose:
                    print(f"[EarlyStopping] Early stopping triggered at step {step}")
                return True

        return False

    def get_summary(self) -> dict[str, Any]:
        """获取早停摘要。"""
        return {
            'stopped': self.should_stop,
            'stopped_step': self.stopped_step,
            'best_mean_reward': float(self.best_mean_reward),
            'best_window_mean': float(self.best_window_mean),
            'window_size': self.window_size,
            'min_training_steps': self.min_training_steps,
            'patience': self.patience,
            'wait': self.wait,
            'history': self.history,
        }


class DynamicEvaluationScheduler:
    """根据训练阶段和评估趋势动态调整评估/落盘间隔。"""

    def __init__(
        self,
        config: AutoTrainingConfig,
        estimated_total_timesteps: int,
        rollout_steps: int,
    ):
        """初始化动态评估调度器。"""
        self.config = config
        self.estimated_total_timesteps = max(int(estimated_total_timesteps), 1)
        self.rollout_steps = max(int(rollout_steps), 1)
        self.min_interval = self._align_interval(max(int(config.min_eval_interval), self.rollout_steps))
        self.max_interval = self._align_interval(max(int(config.max_eval_interval), self.min_interval))
        self.base_interval = self._resolve_base_interval()

    def _align_interval(self, interval: int) -> int:
        """把间隔对齐到 rollout 步长。"""
        aligned = ((max(int(interval), 1) + self.rollout_steps - 1) // self.rollout_steps) * self.rollout_steps
        return max(aligned, self.rollout_steps)

    def _resolve_base_interval(self) -> int:
        """基于估算步数和 hint 计算基准评估间隔。"""
        hinted_interval = max(int(self.config.timesteps_per_eval), 0)
        target_interval = 0
        if int(self.config.target_num_evaluations) > 0:
            target_interval = self.estimated_total_timesteps // int(self.config.target_num_evaluations)

        candidates = [value for value in (hinted_interval, target_interval) if value > 0]
        base_interval = min(candidates) if candidates else self.max_interval
        return max(self.min_interval, min(self._align_interval(base_interval), self.max_interval))

    def _progress_factor(self, current_step: int) -> float:
        """按训练进度决定评估频率。"""
        progress = min(max(float(current_step) / float(self.estimated_total_timesteps), 0.0), 1.0)
        if progress < 0.15:
            return 1.5
        if progress < 0.45:
            return 1.0
        if progress < 0.8:
            return 0.75
        return 0.5

    def _trend_factor(self, history: list[dict[str, Any]]) -> float:
        """按最近评估走势决定评估频率。"""
        analysis = analyze_training_progress(history)
        factor = 1.0
        if analysis['trend'] == 'improving':
            factor *= 1.1
        elif analysis['trend'] == 'stable':
            factor *= 0.7
        elif analysis['trend'] == 'degrading':
            factor *= 0.55
        if analysis.get('converged'):
            factor *= 0.5
        return factor

    def interval_for(self, current_step: int, history: list[dict[str, Any]] | None = None) -> int:
        """为当前训练状态计算下一次评估间隔。"""
        eval_history = history or []
        factor = self._progress_factor(current_step)
        if eval_history:
            factor *= self._trend_factor(eval_history)
        if current_step < int(self.config.min_timesteps):
            factor = max(factor, 1.0)

        interval = int(round(self.base_interval * factor))
        interval = max(self.min_interval, min(self._align_interval(interval), self.max_interval))
        return interval

    def next_step(self, current_step: int, history: list[dict[str, Any]] | None = None) -> int:
        """返回下一次评估/落盘应发生的步数。"""
        return max(int(current_step), 0) + self.interval_for(current_step, history)


def make_json_serializable(value: Any) -> Any:
    """把训练元数据里的 numpy / Path / 复杂容器递归转成标准 JSON 类型。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(item) for item in value]
    return str(value)


def estimate_total_timesteps(
    env_config: dict[str, Any],
    target_performance: float | None = None,
    min_timesteps: int = 100_000,
    max_timesteps: int = 5_000_000,
) -> int:
    """根据环境配置自动估算所需训练步数。

    Args:
        env_config: 环境配置
        target_performance: 目标性能（可选）
        min_timesteps: 最小步数
        max_timesteps: 最大步数

    Returns:
        估算的训练步数
    """
    mode = env_config.get('mode', 'exam')
    scenario = env_config.get('scenario', 'nia_master')

    base_timesteps = {
        'exam': {
            'score': 600_000,
            'clear': 360_000,
        },
        'lesson': {
            'default': 300_000,
        },
        'planning': {
            'default': 1_000_000,
        },
    }

    if mode == 'exam':
        reward_mode = env_config.get('exam_reward_mode', 'score')
        estimated = base_timesteps['exam'].get(reward_mode, 500_000)
    elif mode == 'lesson':
        estimated = base_timesteps['lesson']['default']
    else:
        estimated = base_timesteps['planning']['default']

    difficulty_multipliers = {
        'first_star_regular': 0.8,
        'first_star_pro': 1.0,
        'first_star_master': 1.2,
        'first_star_legend': 1.5,
        'nia_pro': 1.0,
        'nia_master': 1.3,
    }
    multiplier = difficulty_multipliers.get(scenario, 1.0)
    if mode == 'exam':
        if bool(env_config.get('exam_randomize_context') or False):
            multiplier *= 1.20
        if bool(env_config.get('exam_randomize_stage_type') or False):
            multiplier *= 1.10
        if bool(env_config.get('exam_randomize_use_after_item') or False):
            multiplier *= 1.05
        if str(env_config.get('exam_starting_stamina_mode') or 'full') == 'random':
            multiplier *= 1.05
    estimated = int(estimated * multiplier)

    estimated = max(min_timesteps, min(estimated, max_timesteps))
    return estimated


def save_training_metadata(
    run_dir: Path,
    config: dict[str, Any],
    early_stopping_summary: dict[str, Any] | None = None,
) -> None:
    """保存训练元数据。

    Args:
        run_dir: 运行目录
        config: 训练配置
        early_stopping_summary: 早停摘要
    """
    metadata = {
        'config': config,
        'early_stopping': early_stopping_summary,
    }

    metadata_path = run_dir / 'training_metadata.json'
    with metadata_path.open('w', encoding='utf-8') as f:
        json.dump(make_json_serializable(metadata), f, indent=2, ensure_ascii=False)


def extract_evaluation_stats(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    """从不同后端的评估结果中提取均值和标准差。"""
    if 'mean_reward' in payload:
        mean_reward = payload.get('mean_reward')
        std_reward = payload.get('std_reward')
        return (
            float(mean_reward) if mean_reward is not None else None,
            float(std_reward) if std_reward is not None else None,
        )

    result = payload.get('result') or payload.get('evaluation') or payload
    candidates = [
        result.get('episode_reward_mean'),
        result.get('episode_return_mean'),
        (result.get('env_runners') or {}).get('episode_reward_mean'),
        (result.get('env_runners') or {}).get('episode_return_mean'),
        (result.get('evaluation') or {}).get('episode_reward_mean'),
        (result.get('evaluation') or {}).get('episode_return_mean'),
        ((result.get('evaluation') or {}).get('env_runners') or {}).get('episode_reward_mean'),
        ((result.get('evaluation') or {}).get('env_runners') or {}).get('episode_return_mean'),
    ]
    std_candidates = [
        result.get('episode_reward_std'),
        result.get('episode_return_std'),
        (result.get('env_runners') or {}).get('episode_reward_std'),
        (result.get('env_runners') or {}).get('episode_return_std'),
        (result.get('evaluation') or {}).get('episode_reward_std'),
        (result.get('evaluation') or {}).get('episode_return_std'),
        ((result.get('evaluation') or {}).get('env_runners') or {}).get('episode_reward_std'),
        ((result.get('evaluation') or {}).get('env_runners') or {}).get('episode_return_std'),
    ]

    mean_reward = next((float(value) for value in candidates if value is not None), None)
    std_reward = next((float(value) for value in std_candidates if value is not None), None)
    return mean_reward, std_reward


def load_training_history(run_dir: Path) -> list[dict[str, Any]]:
    """加载训练历史。

    Args:
        run_dir: 运行目录

    Returns:
        评估历史列表
    """
    eval_log = run_dir / 'evaluations.jsonl'
    if not eval_log.exists():
        return []

    history = []
    with eval_log.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                payload = json.loads(line)
                mean_reward, std_reward = extract_evaluation_stats(payload)
                if mean_reward is not None and 'mean_reward' not in payload:
                    payload['mean_reward'] = mean_reward
                if std_reward is not None and 'std_reward' not in payload:
                    payload['std_reward'] = std_reward
                history.append(payload)

    return history


def analyze_training_progress(history: list[dict[str, Any]]) -> dict[str, Any]:
    """分析训练进度。

    Args:
        history: 评估历史

    Returns:
        分析结果
    """
    if not history:
        return {
            'converged': False,
            'trend': 'unknown',
            'improvement_rate': 0.0,
        }

    rewards = [float(h['mean_reward']) for h in history if h.get('mean_reward') is not None]
    if not rewards:
        return {
            'converged': False,
            'trend': 'unknown',
            'improvement_rate': 0.0,
        }

    if len(rewards) >= 3:
        recent_rewards = rewards[-3:]
        trend_slope = (recent_rewards[-1] - recent_rewards[0]) / max(len(recent_rewards) - 1, 1)

        if trend_slope > 0.01:
            trend = 'improving'
        elif trend_slope < -0.01:
            trend = 'degrading'
        else:
            trend = 'stable'
    else:
        trend = 'insufficient_data'

    if len(rewards) >= 2:
        improvement_rate = (rewards[-1] - rewards[0]) / max(abs(rewards[0]), 1e-8)
    else:
        improvement_rate = 0.0

    if len(rewards) >= 5:
        recent_std = np.std(rewards[-5:])
        converged = recent_std < 0.05 * abs(np.mean(rewards[-5:]))
    else:
        converged = False

    return {
        'converged': converged,
        'trend': trend,
        'improvement_rate': float(improvement_rate),
        'best_reward': float(max(rewards)),
        'latest_reward': float(rewards[-1]),
        'num_evaluations': len(rewards),
    }
