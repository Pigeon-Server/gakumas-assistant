"""推理服务的 HTTP API 接口。"""

from __future__ import annotations

import argparse
import os
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference_service import ExamState, InferenceRequest, InferenceService

router = APIRouter(prefix='/api/inference', tags=['inference'])

_ENV_BACKEND = 'GAKUMAS_RL_INFERENCE_BACKEND'
_ENV_CHECKPOINT = 'GAKUMAS_RL_INFERENCE_CHECKPOINT'
_ENV_DEVICE = 'GAKUMAS_RL_INFERENCE_DEVICE'

# 全局推理服务实例
_inference_service: InferenceService | None = None


class PredictRequest(BaseModel):
    """Battle 推理请求。"""

    # 基础属性
    vocal: int
    dance: int
    visual: int
    stamina: int
    max_stamina: int

    # 考试/课程状态
    score: int
    target_score: int
    turn: int
    max_turns: int

    # 资源状态
    block: int = 0
    review: int = 0
    aggressive: int = 0
    concentration: int = 0
    full_power_point: int = 0
    parameter_buff: int = 0
    lesson_buff: int = 0

    # 指针状态
    stance: str = "neutral"
    stance_level: int = 0

    # 手牌信息
    hand_cards: list[dict[str, Any]] = Field(default_factory=list)
    deck_count: int = 0
    grave_count: int = 0

    # 饮料信息
    drinks: list[dict[str, Any]] = Field(default_factory=list)

    # P道具/附魔
    status_enchants: list[str] = Field(default_factory=list)

    # N.I.A 专属
    fan_votes: int | None = None

    # 其他状态
    support_cards: list[str] | None = None
    gimmicks: list[dict[str, Any]] | None = None

    # 合法动作列表
    legal_actions: list[dict[str, Any]] = Field(default_factory=list)

    # 推理参数
    deterministic: bool = True


class PredictResponse(BaseModel):
    """推理响应。"""

    action_index: int
    action_id: str = ""
    db_id: str = ""
    action_kind: str = ""
    confidence: float
    value_estimate: float | None = None
    policy_probs: list[float] | None = None


def _resolve_server_config(
    *,
    backend_type: str | None = None,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> tuple[str, str, str]:
    """解析 server 启动配置。"""

    resolved_backend = str(
        backend_type
        or os.getenv(_ENV_BACKEND)
        or 'ppo'
    ).strip()
    resolved_checkpoint = str(
        checkpoint_path
        or os.getenv(_ENV_CHECKPOINT)
        or ''
    ).strip()
    resolved_device = str(
        device
        or os.getenv(_ENV_DEVICE)
        or 'cpu'
    ).strip()
    if not resolved_checkpoint:
        raise RuntimeError(
            'RL inference server 启动失败：未提供 checkpoint。'
            f'请通过参数 --checkpoint 或环境变量 {_ENV_CHECKPOINT} 指定模型。'
        )
    return resolved_backend, resolved_checkpoint, resolved_device


def _build_service(
    *,
    backend_type: str | None = None,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> InferenceService:
    """根据启动配置创建并预加载模型。"""

    resolved_backend, resolved_checkpoint, resolved_device = _resolve_server_config(
        backend_type=backend_type,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return InferenceService(
        backend_type=resolved_backend,
        checkpoint_path=resolved_checkpoint,
        device=resolved_device,
    )


@router.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """执行推理。"""

    global _inference_service

    if _inference_service is None:
        raise HTTPException(status_code=500, detail='Inference service is not initialized.')

    try:
        state = ExamState(
            vocal=request.vocal,
            dance=request.dance,
            visual=request.visual,
            stamina=request.stamina,
            max_stamina=request.max_stamina,
            score=request.score,
            target_score=request.target_score,
            turn=request.turn,
            max_turns=request.max_turns,
            block=request.block,
            review=request.review,
            aggressive=request.aggressive,
            concentration=request.concentration,
            full_power_point=request.full_power_point,
            parameter_buff=request.parameter_buff,
            lesson_buff=request.lesson_buff,
            stance=request.stance,
            stance_level=request.stance_level,
            hand_cards=request.hand_cards,
            deck_count=request.deck_count,
            grave_count=request.grave_count,
            drinks=request.drinks,
            status_enchants=request.status_enchants,
            fan_votes=request.fan_votes,
            support_cards=request.support_cards,
            gimmicks=request.gimmicks,
        )
        inference_request = InferenceRequest(
            state=state,
            legal_actions=request.legal_actions,
            deterministic=request.deterministic,
        )
        response = _inference_service.predict(inference_request)
        return PredictResponse(
            action_index=response.action_index,
            action_id=response.action_id,
            db_id=response.db_id,
            action_kind=response.action_kind,
            confidence=response.confidence,
            value_estimate=response.value_estimate,
            policy_probs=response.policy_probs,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get('/info')
def get_info() -> dict[str, Any]:
    """获取推理服务信息。"""

    global _inference_service

    if _inference_service is None:
        return {
            'status': 'error',
            'message': 'Inference service is not initialized.',
        }
    return {
        'status': 'ready',
        'info': _inference_service.get_info(),
    }


def register_inference_routes(app: FastAPI) -> None:
    """注册推理路由到 FastAPI 应用。"""

    app.include_router(router)


def create_inference_app(
    *,
    backend_type: str | None = None,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> FastAPI:
    """创建独立的推理服务应用，并在启动时预加载模型。"""

    global _inference_service
    _inference_service = _build_service(
        backend_type=backend_type,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    app = FastAPI(
        title="Gakumas RL Inference Service",
        description="统一的 RL 推理服务（模型在服务启动时加载）",
        version="2.0.0",
    )
    register_inference_routes(app)
    return app


def parse_args() -> argparse.Namespace:
    """解析推理服务命令行参数。"""

    parser = argparse.ArgumentParser(description='Start Gakumas RL inference service.')
    parser.add_argument('--backend', default=None, help='后端类型（ppo/dqn/alphazero），默认读环境变量')
    parser.add_argument('--checkpoint', default=None, help='模型 checkpoint 路径，默认读环境变量')
    parser.add_argument('--device', default=None, help='模型加载设备（cpu/cuda），默认读环境变量')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8001, help='监听端口')
    return parser.parse_args()


def main() -> int:
    """命令行启动入口。"""

    import uvicorn

    args = parse_args()
    app = create_inference_app(
        backend_type=args.backend,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
