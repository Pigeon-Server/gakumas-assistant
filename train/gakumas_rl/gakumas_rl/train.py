"""命令行入口，支持 dry-run、观测调试和正式训练。"""

from __future__ import annotations

import argparse
import json
import sys

from .auto_training import AutoTrainingConfig, ExamRandomizationCurriculumConfig
from .backends import TrainingSpec, run_training
from .service import LoadoutConfig, build_env_from_config, loadout_summary


def parse_args() -> argparse.Namespace:
    """解析训练与调试命令行参数。"""

    parser = argparse.ArgumentParser(description='Train Gakumas RL environments.')
    parser.add_argument('--mode', choices=['planning', 'exam', 'lesson'], default='exam')
    parser.add_argument('--backend', choices=['sb3', 'rllib', 'torch'], default='sb3')
    parser.add_argument('--scenario', default='nia_master')
    parser.add_argument('--stage-type', default=None)
    parser.add_argument('--exam-reward-mode', choices=['score', 'clear'], default='score')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--total-timesteps', type=int, default=100_000)
    parser.add_argument('--checkpoint-freq', '--checkpoint-steps', dest='checkpoint_freq', type=int, default=10_000)
    parser.add_argument('--eval-freq', type=int, default=10_000)
    parser.add_argument('--test-report-freq', '--report-freq', dest='test_report_freq', type=int, default=0, help='每隔多少训练步生成一次测试报告/回放；0 表示禁用')
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--rollout-steps', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--learning-rate-final', type=float, default=None, help='线性学习率衰减末值；留空表示固定学习率')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--run-dir', default='')
    parser.add_argument('--auto-resume', action='store_true')
    parser.add_argument('--rllib-num-workers', type=int, default=0)
    parser.add_argument('--rllib-num-envs-per-worker', type=int, default=1)
    parser.add_argument('--rllib-num-gpus', type=float, default=0.0)
    parser.add_argument('--rllib-train-batch-size', type=int, default=0, help='RLlib 每次 learner 更新使用的总 batch；0 表示自动推导')
    parser.add_argument('--rllib-minibatch-size', type=int, default=0, help='RLlib PPO 的 minibatch 大小；0 表示自动推导')
    parser.add_argument('--rllib-num-epochs', type=int, default=4, help='RLlib PPO 每批数据的训练轮数；0 表示使用框架默认值')
    parser.add_argument('--rllib-rollout-fragment-length', type=int, default=0, help='RLlib 每个 worker 单次采样 fragment 长度；0 表示沿用 rollout-steps')

    parser.add_argument('--idol-card-id', default='', help='可选，显式指定单个偶像卡；留空或传 all 时默认在全偶像池中采样')
    parser.add_argument('--producer-level', type=int, default=35)
    parser.add_argument('--idol-rank', type=int, default=0)
    parser.add_argument('--dearness-level', type=int, default=0)
    parser.add_argument('--use-after-item', action='store_true')
    parser.add_argument('--exam-score-bonus-multiplier', type=float, default=None)
    parser.add_argument('--fan-votes', type=float, default=None, help='exam-only / NIA 模式下显式指定粉丝投票数')
    parser.add_argument('--exam-randomize-context', action='store_true', help='在每次考试 reset 时随机采样一套局外上下文')
    parser.add_argument('--exam-stat-jitter-ratio', type=float, default=0.1, help='考试上下文随机化时的三属性抖动比例')
    parser.add_argument('--exam-score-bonus-jitter-ratio', type=float, default=0.05, help='考试上下文随机化时的倍率抖动比例')
    parser.add_argument('--exam-randomize-use-after-item', action='store_true', help='考试上下文随机化时允许 before/after P 道具切换')
    parser.add_argument('--exam-randomize-stage-type', action='store_true', help='考试上下文随机化时允许在场景的多个 stage 间采样')
    parser.add_argument('--exam-randomization-curriculum', action='store_true', help='按训练进度逐步打开 stage_type / use_after_item 随机化，避免一开始就把泛化轴全压给策略')
    parser.add_argument('--exam-curriculum-stage-type-start-ratio', type=float, default=0.10, help='总训练进度达到该比例后启用 stage_type 随机化')
    parser.add_argument('--exam-curriculum-use-after-item-start-ratio', type=float, default=0.25, help='总训练进度达到该比例后启用 use_after_item 随机化')
    parser.add_argument('--exam-starting-stamina-mode', choices=['full', 'random'], default='full', help='exam-only 模式下的开场体力策略')
    parser.add_argument('--exam-starting-stamina-min-ratio', type=float, default=0.6, help='随机开场体力的最小比例')
    parser.add_argument('--exam-starting-stamina-max-ratio', type=float, default=1.0, help='随机开场体力的最大比例')
    parser.add_argument('--lesson-action-type', default='', help='lesson 模式下固定课程类型；留空时在当前场景的普通/SP课程池中采样')
    parser.add_argument('--lesson-level-index', type=int, default=0, help='lesson 模式下固定课程等级序号；0 表示按主数据候选随机采样')
    parser.add_argument(
        '--manual-exam-setup',
        action='append',
        default=[],
        help='手工 exam 配置 jsonl；提供后 exam/lesson 环境将按记录采样真实牌组/饮料/P物品',
    )
    parser.add_argument(
        '--guarantee-card-effect',
        action='append',
        default=[],
        help='保底某类效果标签牌数量，例如 review=3 / 打分=4 / 好印象=2 / 元气=2',
    )
    parser.add_argument(
        '--force-card',
        action='append',
        default=[],
        help='强制加入按主轴分组的卡 JSON，例如 {"好印象":["p_card..."]} / {"干劲":["p_card..."]} 或 @force_cards.json；重复传入会合并，且计入保底数量',
    )

    parser.add_argument('--include-deck-features', action='store_true', help='在观测空间中包含牌库组成特征')

    parser.add_argument('--pretrained-checkpoint', default=None, help='BC 预训练 checkpoint 路径，用于热启动 PPO')

    parser.add_argument('--llm-reward', action='store_true', help='启用 LLM reward shaping')
    parser.add_argument('--llm-reward-model', default='qwen3:4b', help='ollama 模型名 (default: qwen3:4b)')
    parser.add_argument('--llm-reward-base-url', default='http://localhost:11434', help='ollama API 地址')
    parser.add_argument('--llm-reward-weight', type=float, default=0.3, help='LLM reward 权重 (default: 0.3)')
    parser.add_argument('--llm-reward-eval-interval', type=int, default=1, help='每隔几步调用 LLM (default: 1)')

    # ── 奖励工程参数 ──
    parser.add_argument('--reward-config', default=None, help='奖励配置 JSON 文件路径')
    parser.add_argument('--reward-shape-scale', type=float, default=None, help='潜势差分缩放')
    parser.add_argument('--reward-goal-weight', type=float, default=None, help='Φ_goal 权重')
    parser.add_argument('--reward-eval-weight', type=float, default=None, help='Φ_eval 权重')
    parser.add_argument('--reward-archetype-weight', type=float, default=None, help='Φ_archetype 权重')
    parser.add_argument('--reward-risk-weight', type=float, default=None, help='Φ_risk 权重')
    parser.add_argument('--reward-efficiency-weight', type=float, default=None, help='Φ_efficiency 权重')
    parser.add_argument('--reward-turn-window-weight', type=float, default=None, help='回合颜色窗口权重')
    parser.add_argument('--reward-judging-alignment-weight', type=float, default=None, help='审查基准匹配权重')
    parser.add_argument('--reward-efficiency-overshoot-penalty', type=float, default=None, help='提前达标后的超线惩罚')
    parser.add_argument('--reward-terminal-pass', type=float, default=None, help='考试通过奖励')
    parser.add_argument('--reward-terminal-failure', type=float, default=None, help='考试失败惩罚')
    parser.add_argument('--reward-terminal-force-end-bonus', type=float, default=None, help='达到 force_end_score 的收官奖励')
    parser.add_argument('--reward-lesson-clear', type=float, default=None, help='课程 Clear 奖励')
    parser.add_argument('--reward-lesson-perfect', type=float, default=None, help='课程 Perfect 奖励')
    parser.add_argument('--reward-overshoot-penalty', type=float, default=None, help='超线惩罚')
    parser.add_argument('--reward-invalid-action-penalty', type=float, default=None, help='无效动作惩罚')
    parser.add_argument('--reward-skip-turn-penalty', type=float, default=None, help='空过回合惩罚')
    parser.add_argument('--reward-consecutive-end-turn-penalty', type=float, default=None, help='连续结束回合惩罚')
    parser.add_argument('--reward-stamina-death-penalty', type=float, default=None, help='体力归零额外惩罚')
    parser.add_argument('--reward-milestone-50', type=float, default=None, help='50%% 目标里程碑奖励')
    parser.add_argument('--reward-milestone-75', type=float, default=None, help='75%% 目标里程碑奖励')
    parser.add_argument('--reward-milestone-100', type=float, default=None, help='100%% 目标里程碑奖励')
    parser.add_argument('--reward-score-delta-scale', type=float, default=None, help='分数差分密集奖励缩放')
    parser.add_argument('--reward-card-play-reward', type=float, default=None, help='出牌微奖励')
    parser.add_argument('--reward-scale', type=float, default=None, help='全局奖励缩放')
    parser.add_argument('--reward-clip', type=float, default=None, help='奖励裁剪范围 (0=不裁剪)')

    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--print-observation', action='store_true')
    parser.add_argument('--print-loadout', action='store_true')

    parser.add_argument('--auto-train', action='store_true', help='自动训练模式：自动估算步数、动态调整评估/落盘间隔，并默认启用早停；高随机性环境不建议长期训练时使用')
    parser.add_argument('--auto-timesteps', action='store_true', help='仅自动估算训练步数，不会像 --auto-train 那样默认开启早停')
    parser.add_argument('--auto-min-timesteps', type=int, default=100_000, help='自动训练允许早停前的最少训练步数')
    parser.add_argument('--auto-max-timesteps', type=int, default=5_000_000, help='自动训练的最大训练步数上限')
    parser.add_argument('--auto-eval-interval', type=int, default=50_000, help='自动训练的评估基准间隔，实际会按训练进度动态调整')
    parser.add_argument('--early-stopping', action='store_true', help='启用早停机制；高随机性环境或服务器长期训练通常建议关闭')
    parser.add_argument('--early-stopping-patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.01, help='早停最小改善阈值')
    return parser.parse_args()


def _build_loadout_config(args: argparse.Namespace) -> LoadoutConfig:
    """从命令行参数构造偶像配装配置。"""

    return LoadoutConfig(
        idol_card_id=args.idol_card_id,
        producer_level=args.producer_level,
        idol_rank=args.idol_rank,
        dearness_level=args.dearness_level,
        use_after_item=(True if args.use_after_item else None),
        exam_score_bonus_multiplier=args.exam_score_bonus_multiplier,
        fan_votes=args.fan_votes,
        exam_randomize_context=args.exam_randomize_context,
        exam_stat_jitter_ratio=args.exam_stat_jitter_ratio,
        exam_score_bonus_jitter_ratio=args.exam_score_bonus_jitter_ratio,
        exam_randomize_use_after_item=args.exam_randomize_use_after_item,
        exam_randomize_stage_type=args.exam_randomize_stage_type,
    )


_REWARD_CLI_MAP = {
    'reward_shape_scale': 'shape_scale',
    'reward_goal_weight': 'goal_weight',
    'reward_eval_weight': 'eval_weight',
    'reward_archetype_weight': 'archetype_weight',
    'reward_risk_weight': 'risk_weight',
    'reward_efficiency_weight': 'efficiency_weight',
    'reward_turn_window_weight': 'turn_window_weight',
    'reward_judging_alignment_weight': 'judging_alignment_weight',
    'reward_efficiency_overshoot_penalty': 'efficiency_overshoot_penalty',
    'reward_terminal_pass': 'terminal_pass_reward',
    'reward_terminal_failure': 'terminal_failure_weight',
    'reward_terminal_force_end_bonus': 'terminal_force_end_bonus',
    'reward_lesson_clear': 'lesson_clear_reward',
    'reward_lesson_perfect': 'lesson_perfect_reward',
    'reward_overshoot_penalty': 'overshoot_penalty',
    'reward_invalid_action_penalty': 'invalid_action_penalty',
    'reward_skip_turn_penalty': 'skip_turn_penalty',
    'reward_consecutive_end_turn_penalty': 'consecutive_end_turn_penalty',
    'reward_stamina_death_penalty': 'stamina_death_penalty',
    'reward_milestone_50': 'milestone_50_reward',
    'reward_milestone_75': 'milestone_75_reward',
    'reward_milestone_100': 'milestone_100_reward',
    'reward_score_delta_scale': 'score_delta_scale',
    'reward_card_play_reward': 'card_play_reward',
    'reward_scale': 'reward_scale',
    'reward_clip': 'reward_clip',
}


def _collect_reward_overrides(args: argparse.Namespace) -> dict | None:
    """从 CLI 参数中收集非 None 的奖励配置覆盖。"""

    overrides = {}
    for cli_attr, config_key in _REWARD_CLI_MAP.items():
        val = getattr(args, cli_attr, None)
        if val is not None:
            overrides[config_key] = val
    return overrides if overrides else None


def _build_env_config(args: argparse.Namespace) -> dict:
    """把命令行参数整理成环境构建配置。"""

    return {
        'mode': args.mode,
        'scenario': args.scenario,
        'stage_type': args.stage_type,
        'exam_reward_mode': args.exam_reward_mode,
        'seed': args.seed,
        'idol_card_id': args.idol_card_id,
        'producer_level': args.producer_level,
        'idol_rank': args.idol_rank,
        'dearness_level': args.dearness_level,
        'use_after_item': True if args.use_after_item else None,
        'exam_score_bonus_multiplier': args.exam_score_bonus_multiplier,
        'fan_votes': args.fan_votes,
        'exam_randomize_context': args.exam_randomize_context,
        'exam_stat_jitter_ratio': args.exam_stat_jitter_ratio,
        'exam_score_bonus_jitter_ratio': args.exam_score_bonus_jitter_ratio,
        'exam_randomize_use_after_item': args.exam_randomize_use_after_item,
        'exam_randomize_stage_type': args.exam_randomize_stage_type,
        'exam_starting_stamina_mode': args.exam_starting_stamina_mode,
        'exam_starting_stamina_min_ratio': args.exam_starting_stamina_min_ratio,
        'exam_starting_stamina_max_ratio': args.exam_starting_stamina_max_ratio,
        'lesson_action_type': args.lesson_action_type,
        'lesson_level_index': args.lesson_level_index,
        'manual_exam_setup_paths': list(args.manual_exam_setup),
        'guarantee_card_effects': list(args.guarantee_card_effect),
        'force_card_groups': list(args.force_card),
        'include_deck_features': args.include_deck_features,
        'llm_reward_enabled': args.llm_reward,
        'llm_reward_model': args.llm_reward_model,
        'llm_reward_base_url': args.llm_reward_base_url,
        'llm_reward_weight': args.llm_reward_weight,
        'llm_reward_eval_interval': args.llm_reward_eval_interval,
        'reward_config_path': args.reward_config,
        'reward_overrides': _collect_reward_overrides(args),
    }


def _argument_was_provided(flags: str | tuple[str, ...], argv: list[str]) -> bool:
    """判断某个命令行参数是否由用户显式传入。"""

    candidates = (flags,) if isinstance(flags, str) else flags
    return any(token == flag or token.startswith(f'{flag}=') for token in argv for flag in candidates)


def main() -> int:
    """命令行入口，支持 dry-run、观测调试与正式训练。"""

    raw_argv = sys.argv[1:]
    args = parse_args()
    loadout_config = _build_loadout_config(args)
    env_config = _build_env_config(args)

    if args.print_loadout:
        print(json.dumps(loadout_summary(args.scenario, loadout_config), ensure_ascii=False, indent=2))

    env = build_env_from_config(env_config)
    obs, info = env.reset(seed=args.seed)
    if args.print_observation:
        payload = {
            'global_shape': list(obs['global'].shape),
            'action_features_shape': list(obs['action_features'].shape),
            'action_mask_shape': list(obs['action_mask'].shape),
            'action_labels': info.get('action_labels', []),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.dry_run:
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(16):
            valid_actions = [idx for idx, flag in enumerate(obs['action_mask']) if flag > 0.5]
            if not valid_actions:
                break
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(json.dumps({'dry_run_reward': total_reward, 'terminated': terminated, 'truncated': truncated, 'info': info}, ensure_ascii=False, indent=2))
        return 0

    user_set_eval_freq = _argument_was_provided('--eval-freq', raw_argv)
    user_set_checkpoint_freq = _argument_was_provided(('--checkpoint-freq', '--checkpoint-steps'), raw_argv)
    auto_base_eval_interval = max(int(args.auto_eval_interval), max(int(args.rollout_steps), 1))
    auto_min_eval_interval = max(max(int(args.rollout_steps), 1), auto_base_eval_interval // 4)
    auto_max_eval_interval = max(auto_base_eval_interval * 2, auto_min_eval_interval)

    auto_config = None
    total_timesteps = args.total_timesteps
    checkpoint_freq = args.checkpoint_freq
    eval_freq = args.eval_freq
    enable_early_stopping = args.early_stopping

    if args.auto_train or args.auto_timesteps:
        auto_config = AutoTrainingConfig(
            full_auto=args.auto_train,
            auto_total_timesteps=True,
            min_timesteps=args.auto_min_timesteps,
            max_timesteps=args.auto_max_timesteps,
            timesteps_per_eval=auto_base_eval_interval,
            dynamic_eval_schedule=(args.auto_train and not user_set_eval_freq),
            dynamic_checkpoint_schedule=(args.auto_train and not user_set_checkpoint_freq),
            target_num_evaluations=12,
            min_eval_interval=auto_min_eval_interval,
            max_eval_interval=auto_max_eval_interval,
            enable_early_stopping=(args.auto_train or args.early_stopping),
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            baseline_episodes=args.eval_episodes,
        )

    if args.auto_train:
        total_timesteps = args.auto_max_timesteps
        enable_early_stopping = True
        if not user_set_eval_freq:
            eval_freq = auto_base_eval_interval
        if not user_set_checkpoint_freq:
            checkpoint_freq = auto_base_eval_interval
        eval_schedule = 'dynamic' if auto_config and auto_config.dynamic_eval_schedule else f'fixed:{eval_freq:,}'
        checkpoint_schedule = 'dynamic' if auto_config and auto_config.dynamic_checkpoint_schedule else f'fixed:{checkpoint_freq:,}'
        print(
            f"[AutoTrain] min_steps={args.auto_min_timesteps:,}, max_steps={args.auto_max_timesteps:,}, "
            f"eval_schedule={eval_schedule}, checkpoint_schedule={checkpoint_schedule}, "
            f"patience={args.early_stopping_patience}"
        )

    result = run_training(
        TrainingSpec(
            backend=args.backend,
            env_config=env_config,
            total_timesteps=total_timesteps,
            checkpoint_freq=checkpoint_freq,
            test_report_freq=args.test_report_freq,
            eval_freq=eval_freq,
            eval_episodes=args.eval_episodes,
            rollout_steps=args.rollout_steps,
            learning_rate=args.learning_rate,
            learning_rate_final=args.learning_rate_final,
            device=args.device,
            run_dir=args.run_dir,
            auto_resume=args.auto_resume,
            seed=args.seed,
            rllib_num_workers=args.rllib_num_workers,
            rllib_num_envs_per_worker=args.rllib_num_envs_per_worker,
            rllib_num_gpus=args.rllib_num_gpus,
            rllib_train_batch_size=args.rllib_train_batch_size,
            rllib_minibatch_size=args.rllib_minibatch_size,
            rllib_num_epochs=args.rllib_num_epochs,
            rllib_rollout_fragment_length=args.rllib_rollout_fragment_length,
            auto_config=auto_config,
            exam_randomization_curriculum=ExamRandomizationCurriculumConfig(
                enabled=bool(args.exam_randomization_curriculum),
                stage_type_start_ratio=args.exam_curriculum_stage_type_start_ratio,
                use_after_item_start_ratio=args.exam_curriculum_use_after_item_start_ratio,
            ),
            enable_early_stopping=enable_early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            pretrained_checkpoint=args.pretrained_checkpoint,
        )
    )
    print(
        json.dumps(
            {
                'backend': result.backend,
                'run_dir': str(result.run_dir),
                'latest_checkpoint': str(result.latest_checkpoint) if result.latest_checkpoint else None,
                'total_timesteps': result.total_timesteps,
                'evaluation_log': str(result.evaluation_log) if result.evaluation_log else None,
                'metadata_log': str(result.metadata_log) if result.metadata_log else None,
                'replay_html': str(result.replay_html) if result.replay_html else None,
                'replay_json': str(result.replay_json) if result.replay_json else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
