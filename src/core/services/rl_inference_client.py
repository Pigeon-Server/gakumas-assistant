"""RL 推理服务 HTTP 客户端。

与 ``train/gakumas_rl`` 的 FastAPI 推理服务通信。
模型在服务端启动时固定加载，客户端只负责状态查询和预测调用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import logger

_DEFAULT_TIMEOUT = 5.0
_PREDICT_TIMEOUT = 10.0


class RLInferenceClient:
    """RL 推理服务的 HTTP 客户端封装。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8100",
        *,
        info_timeout: float = _DEFAULT_TIMEOUT,
        predict_timeout: float = _PREDICT_TIMEOUT,
    ):
        self._base_url = base_url.rstrip("/")
        self._info_timeout = float(info_timeout)
        self._predict_timeout = float(predict_timeout)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    @property
    def base_url(self) -> str:
        return self._base_url

    # ── 公共 API ────────────────────────────────────────

    def is_available(self) -> bool:
        """检查推理服务是否可用。"""
        try:
            resp = self._session.get(
                f"{self._base_url}/api/inference/info",
                timeout=self._info_timeout,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def get_info(self) -> Dict[str, Any]:
        """获取推理服务状态信息。"""
        try:
            resp = self._session.get(
                f"{self._base_url}/api/inference/info",
                timeout=self._info_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning(f"[RL] 获取推理服务状态失败: {exc}")
            return {"status": "error", "message": str(exc)}

    def predict(
        self,
        exam_state: Dict[str, Any],
        legal_actions: List[Dict[str, Any]],
        deterministic: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """执行推理预测。

        Parameters
        ----------
        exam_state:
            当前考试状态，字段对应 ``PredictRequest``。
        legal_actions:
            合法动作列表。
        deterministic:
            是否使用确定性策略。

        Returns
        -------
        预测结果字典（含 ``action_index``, ``action_id``, ``db_id`` 等字段），
        失败时返回 ``None``。
        """
        payload = {
            **exam_state,
            "legal_actions": legal_actions,
            "deterministic": deterministic,
        }
        try:
            resp = self._session.post(
                f"{self._base_url}/api/inference/predict",
                json=payload,
                timeout=self._predict_timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.debug(
                f"[RL] 推理结果: action={result.get('action_index')}, "
                f"confidence={result.get('confidence', 0):.3f}"
            )
            return result
        except requests.RequestException as exc:
            logger.warning(f"[RL] 推理请求失败: {exc}")
            return None

    def close(self) -> None:
        """关闭 HTTP 会话。"""
        self._session.close()

    def __del__(self):
        try:
            self._session.close()
        except Exception:
            pass
