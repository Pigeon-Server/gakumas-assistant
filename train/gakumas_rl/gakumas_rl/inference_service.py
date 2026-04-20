"""统一的推理服务接口，支持多种RL后端。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExamState:
    """考试状态的完整描述。"""

    # 基础属性
    vocal: int
    dance: int
    visual: int
    stamina: int
    max_stamina: int

    # 考试状态
    score: int
    target_score: int  # 目标分数（可变更）
    turn: int
    max_turns: int

    # 资源状态
    block: int  # 元气消耗
    review: int  # 集中
    aggressive: int  # 元气
    concentration: int  # 集中指针
    full_power_point: int  # 全力点数
    parameter_buff: int  # 好调
    lesson_buff: int  # 课程增益

    # 指针状态
    stance: str  # neutral/concentration/full_power/preservation
    stance_level: int

    # 手牌信息
    hand_cards: list[dict[str, Any]]  # 当前手牌列表
    deck_count: int  # 牌库数量
    grave_count: int  # 弃牌堆数量

    # 饮料信息
    drinks: list[dict[str, Any]]  # P饮料列表

    # P道具效果
    status_enchants: list[str]  # 状态附魔ID列表

    # N.I.A专属
    fan_votes: int | None = None  # 粉丝投票数

    # 回忆卡（支援卡）
    support_cards: list[str] | None = None  # 回忆卡ID列表

    # 其他状态
    gimmicks: list[dict[str, Any]] | None = None  # 场地机制


@dataclass
class InferenceRequest:
    """推理请求。"""

    state: ExamState
    legal_actions: list[dict[str, Any]]  # 合法动作列表
    deterministic: bool = True  # 是否使用确定性策略


@dataclass
class InferenceResponse:
    """推理响应。"""

    action_index: int  # 选择的动作索引
    confidence: float  # 置信度
    action_id: str = ""  # 动作稳定标识（优先由客户端传入）
    db_id: str = ""  # 实体类动作对应的主库 ID
    action_kind: str = ""  # 动作类别
    value_estimate: float | None = None  # 价值估计（可选）
    policy_probs: list[float] | None = None  # 策略概率分布（可选）


class RLBackend(ABC):
    """RL后端抽象基类。"""

    @abstractmethod
    def load_model(self, checkpoint_path: Path, *, device: str = 'cpu') -> None:
        """加载模型checkpoint。"""
        pass

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理。"""
        pass

    @abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """获取后端信息。"""
        pass


class PPOBackend(RLBackend):
    """PPO后端实现（Stable-Baselines3）。"""

    def __init__(self):
        self.model = None
        self.env = None

    def load_model(self, checkpoint_path: Path, *, device: str = 'cpu') -> None:
        """加载PPO模型。"""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise RuntimeError("PPO backend requires stable-baselines3")

        self.model = PPO.load(str(checkpoint_path), device=device)

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """执行PPO推理。"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # 构造观测
        obs = self._state_to_observation(request.state, request.legal_actions)

        # 执行推理
        action, _states = self.model.predict(obs, deterministic=request.deterministic)
        action_index = int(action)

        # 通过稳定字段回传动作身份，避免协议依赖 OCR 名称
        action_id = ""
        db_id = ""
        action_kind = ""
        if action_index < len(request.legal_actions):
            action_payload = request.legal_actions[action_index]
            action_id = str(action_payload.get('action_id') or action_payload.get('id') or "")
            db_id = str(action_payload.get('db_id') or "")
            action_kind = str(action_payload.get('kind') or "")

        # 获取价值估计（如果可用）
        value_estimate = None
        try:
            value_estimate = float(self.model.policy.predict_values(obs)[0])
        except Exception:
            pass

        return InferenceResponse(
            action_index=action_index,
            action_id=action_id,
            db_id=db_id,
            action_kind=action_kind,
            confidence=1.0 if request.deterministic else 0.0,
            value_estimate=value_estimate,
        )

    def _state_to_observation(self, state: ExamState, legal_actions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """将状态转换为模型观测格式。"""
        # 构造全局特征
        global_features = np.array([
            state.turn / max(state.max_turns, 1),
            state.score / max(state.target_score, 1),
            state.stamina / max(state.max_stamina, 1),
            state.vocal / 1200.0,
            state.dance / 1200.0,
            state.visual / 1200.0,
            state.block / 30.0,
            state.review / 30.0,
            state.aggressive / 30.0,
            state.concentration / 30.0,
            state.full_power_point / 30.0,
            state.parameter_buff / 30.0,
            state.lesson_buff / 30.0,
            1.0 if state.stance == 'concentration' else 0.0,
            1.0 if state.stance == 'full_power' else 0.0,
            1.0 if state.stance == 'preservation' else 0.0,
            len(state.hand_cards) / 10.0,
            state.deck_count / 40.0,
            state.grave_count / 40.0,
            len(state.drinks) / 5.0,
        ], dtype=np.float32)

        # 补齐到43维（与训练时一致）
        if len(global_features) < 43:
            global_features = np.pad(global_features, (0, 43 - len(global_features)))

        # 构造动作特征（简化版）
        max_actions = 52  # 48手牌 + 3饮料 + 1结束回合
        action_features = np.zeros((max_actions, 100), dtype=np.float32)  # 假设特征维度100

        # 构造动作掩码
        action_mask = np.zeros(max_actions, dtype=np.float32)
        for i, action in enumerate(legal_actions[:max_actions]):
            if action.get('available', True):
                action_mask[i] = 1.0

        return {
            'global': global_features,
            'action_features': action_features,
            'action_mask': action_mask,
        }

    def get_backend_info(self) -> dict[str, Any]:
        """获取PPO后端信息。"""
        return {
            'backend': 'ppo',
            'framework': 'stable-baselines3',
            'model_loaded': self.model is not None,
        }


class DQNBackend(RLBackend):
    """DQN后端实现（预留接口）。"""

    def __init__(self):
        self.model = None

    def load_model(self, checkpoint_path: Path, *, device: str = 'cpu') -> None:
        """加载DQN模型。"""
        del checkpoint_path, device
        raise NotImplementedError("DQN backend not yet implemented")

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """执行DQN推理。"""
        raise NotImplementedError("DQN backend not yet implemented")

    def get_backend_info(self) -> dict[str, Any]:
        """获取DQN后端信息。"""
        return {
            'backend': 'dqn',
            'framework': 'custom',
            'model_loaded': False,
        }


class AlphaZeroBackend(RLBackend):
    """AlphaZero后端实现（预留接口）。"""

    def __init__(self):
        self.model = None
        self.mcts_simulations = 100

    def load_model(self, checkpoint_path: Path, *, device: str = 'cpu') -> None:
        """加载AlphaZero模型。"""
        del checkpoint_path, device
        raise NotImplementedError("AlphaZero backend not yet implemented")

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """执行AlphaZero推理（带MCTS）。"""
        raise NotImplementedError("AlphaZero backend not yet implemented")

    def get_backend_info(self) -> dict[str, Any]:
        """获取AlphaZero后端信息。"""
        return {
            'backend': 'alphazero',
            'framework': 'custom',
            'mcts_simulations': self.mcts_simulations,
            'model_loaded': False,
        }


class InferenceService:
    """统一的推理服务。"""

    def __init__(
        self,
        backend_type: str = 'ppo',
        *,
        checkpoint_path: str | Path | None = None,
        device: str = 'cpu',
    ):
        """初始化推理服务。

        Args:
            backend_type: 后端类型 (ppo/dqn/alphazero)
        """
        self.backend = self._create_backend(backend_type)
        self.backend_type = backend_type
        self.device = str(device or 'cpu')
        self.checkpoint_path = str(checkpoint_path or "")
        if checkpoint_path:
            self.load_model(checkpoint_path, device=self.device)

    def _create_backend(self, backend_type: str) -> RLBackend:
        """创建后端实例。"""
        backends = {
            'ppo': PPOBackend,
            'dqn': DQNBackend,
            'alphazero': AlphaZeroBackend,
        }

        if backend_type not in backends:
            raise ValueError(f"Unknown backend type: {backend_type}")

        return backends[backend_type]()

    def load_model(self, checkpoint_path: str | Path, *, device: str | None = None) -> None:
        """加载模型。"""
        resolved_device = str(device or self.device or 'cpu')
        resolved_path = Path(checkpoint_path)
        self.backend.load_model(resolved_path, device=resolved_device)
        self.device = resolved_device
        self.checkpoint_path = str(resolved_path)

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理。"""
        return self.backend.predict(request)

    def get_info(self) -> dict[str, Any]:
        """获取服务信息。"""
        return {
            'service': 'gakumas_rl_inference',
            'backend_type': self.backend_type,
            'device': self.device,
            'checkpoint_path': self.checkpoint_path,
            'backend_info': self.backend.get_backend_info(),
        }
