"""RLlib 动作掩码模型适配，兼容 old/new API stack。"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from .model import MaskedPolicyValueNet

DEFAULT_RLLIB_MODEL_CONFIG: dict[str, Any] = {
    'hidden_dim': 256,
}


def _resolve_hidden_dim(model_config: dict[str, Any] | None) -> int:
    """解析 RLlib 模型隐藏层大小。"""

    config = dict(DEFAULT_RLLIB_MODEL_CONFIG)
    if model_config:
        config.update(model_config)
    return int(config.get('hidden_dim', 256))


class RLlibMaskedActionsModel(TorchModelV2, nn.Module):
    """让 RLlib 旧 API stack 复用仓库内的动作掩码策略网络。"""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: dict[str, Any],
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        original_space = getattr(obs_space, 'original_space', obs_space)
        global_dim = int(np.prod(original_space['global'].shape))
        action_feature_dim = int(original_space['action_features'].shape[-1])
        if 'hidden_dim' in kwargs:
            hidden_dim = int(kwargs['hidden_dim'])
        else:
            hidden_dim = _resolve_hidden_dim(model_config.get('custom_model_config', {}))
        self.net = MaskedPolicyValueNet(global_dim, action_feature_dim, hidden_dim=hidden_dim)
        self._value_out: torch.Tensor | None = None

    def forward(self, input_dict, state, seq_lens):
        """从 Dict 观测恢复原始结构，并输出带 mask 的 logits。"""

        obs = restore_original_dimensions(input_dict['obs'], self.obs_space, tensorlib='torch')
        global_obs = obs['global'].float()
        action_features = obs['action_features'].float()
        action_mask = obs['action_mask'].float()
        logits, value = self.net(global_obs, action_features, action_mask)
        self._value_out = value
        return logits, state

    def value_function(self) -> torch.Tensor:
        """返回上一轮 forward 缓存的 state value。"""

        if self._value_out is None:
            raise ValueError('value_function() called before forward().')
        return self._value_out.reshape(-1)


class GakumasActionMaskingTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """RLlib 新 API stack 使用的动作掩码 RLModule。"""

    REQUIRED_OBS_KEYS = ('global', 'action_features', 'action_mask')

    @override(TorchRLModule)
    def setup(self) -> None:
        """按现有 exam 观测结构搭建共享策略/价值网络。"""

        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError('GakumasActionMaskingTorchRLModule 需要 Dict observation space。')
        observation_spaces = getattr(self.observation_space, 'spaces', {})
        missing_keys = [key for key in self.REQUIRED_OBS_KEYS if key not in observation_spaces]
        if missing_keys:
            raise ValueError(f'GakumasActionMaskingTorchRLModule 缺少 observation keys: {missing_keys}')

        hidden_dim = _resolve_hidden_dim(self.model_config)
        global_dim = int(np.prod(observation_spaces['global'].shape))
        action_dim = int(observation_spaces['action_features'].shape[-1])
        self.net = MaskedPolicyValueNet(global_dim, action_dim, hidden_dim=hidden_dim)
        self._checked_observations = False

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, torch.Tensor]:
        _, logits = self._compute_embeddings_and_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, torch.Tensor]:
        embeddings, logits = self._compute_embeddings_and_logits(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings=None):
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_logits(batch)
        return self.net.value_head(embeddings).squeeze(-1)

    def _compute_embeddings_and_logits(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从观测 batch 中构造 PPO 需要的 logits 与 value embeddings。"""

        obs = self._extract_obs(batch)
        global_obs = obs['global'].float()
        action_features = obs['action_features'].float()
        action_mask = obs['action_mask'].float()

        global_hidden = self.net.global_encoder(global_obs)
        action_hidden = self.net.action_encoder(action_features)
        expanded_global = global_hidden.unsqueeze(1).expand(-1, action_hidden.size(1), -1)
        logits = self.net.logit_head(torch.cat([expanded_global, action_hidden], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(action_mask <= 0, -1.0e9)

        mask_expanded = action_mask.unsqueeze(-1)
        masked_sum = (action_hidden * mask_expanded).sum(dim=1)
        mask_count = mask_expanded.sum(dim=1).clamp(min=1.0)
        action_context = masked_sum / mask_count
        embeddings = torch.cat([global_hidden, action_context], dim=-1)
        return embeddings, logits

    def _extract_obs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """从 RLlib batch 中提取并校验观测字典。"""

        obs = batch[Columns.OBS]
        if not isinstance(obs, dict):
            raise ValueError('GakumasActionMaskingTorchRLModule 期望 batch[Columns.OBS] 为 dict。')
        if not self._checked_observations:
            missing_keys = [key for key in self.REQUIRED_OBS_KEYS if key not in obs]
            if missing_keys:
                raise ValueError(f'观测缺少动作掩码所需字段: {missing_keys}')
            self._checked_observations = True
        return obs


def register_rllib_model() -> str:
    """注册旧 API stack 的自定义 RLlib 模型，并返回固定名称。"""

    model_name = 'gakumas_masked_actions_model'
    registry = getattr(register_rllib_model, '_registered', set())
    if model_name not in registry:
        ModelCatalog.register_custom_model(model_name, RLlibMaskedActionsModel)
        registry.add(model_name)
        register_rllib_model._registered = registry
    return model_name


def build_rllib_module_spec(model_config: dict[str, Any] | None = None) -> RLModuleSpec:
    """构造 RLlib 新 API stack 使用的 RLModuleSpec。"""

    resolved_config = dict(DEFAULT_RLLIB_MODEL_CONFIG)
    if model_config:
        resolved_config.update(model_config)
    return RLModuleSpec(
        module_class=GakumasActionMaskingTorchRLModule,
        model_config=resolved_config,
    )
