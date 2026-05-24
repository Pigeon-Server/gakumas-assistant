"""周行動アイコン CLIP 服务。

将周行动按钮的截图特征存入 CLIP 记忆库，用于跨场景快速识别
schedule action 类型（レッスン / おでかけ / 相談 等），降低对 OCR 的依赖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.models.clip import CLIPayload_ScheduleAction, CLIPMemory
from src.utils.clip_tools import CLIPTools


@dataclass
class ScheduleActionIdentity:
    """CLIP 检索返回的周行动身份信息。"""
    action_id: str
    param_kind: str = ""
    rl_action_type: str = ""


class ScheduleActionCLIP(CLIPTools):
    """周行動アイコン専用 CLIP 服务。

    使用 ``schedule_action`` 命名空间，按 action_id 唯一索引。
    """

    def __init__(self, session):
        super().__init__(session, "schedule_action")

    def _save_payload(
        self,
        image: np.ndarray,
        features: np.ndarray,
        payload: ScheduleActionIdentity,
    ):
        """将周行动图标写入 CLIP 图片目录，并持久化 payload。"""
        safe_name = payload.action_id.replace(":", "_").replace("/", "_")
        image_save_path = os.path.join(self._image_file_path, f"{safe_name}.png")
        if not os.path.exists(image_save_path):
            cv2.imwrite(image_save_path, image)
        obj, _created = CLIPayload_ScheduleAction.get_or_create(
            action_id=payload.action_id,
            defaults={
                "param_kind": payload.param_kind,
                "rl_action_type": payload.rl_action_type,
            },
        )
        return obj

    def _load_payload(
        self, payload_ref: CLIPayload_ScheduleAction
    ) -> Optional[ScheduleActionIdentity]:
        """从 DB payload 记录还原 ScheduleActionIdentity。"""
        if payload_ref is None:
            return None
        return ScheduleActionIdentity(
            action_id=str(payload_ref.action_id),
            param_kind=str(payload_ref.param_kind or ""),
            rl_action_type=str(payload_ref.rl_action_type or ""),
        )

    def add_to_memory(
        self,
        image: np.ndarray,
        payload: ScheduleActionIdentity,
        similarity_threshold: float = 0.96,
        save_image: bool = False,
    ) -> bool:
        """将周行动图标加入 CLIP 记忆库。

        相同 action_id 已有记录时跳过，避免重复 ONNX 推理。
        """
        # 快速路径：已有相同 action_id 的记忆条目则跳过
        payload_ref = CLIPayload_ScheduleAction.get_or_none(
            CLIPayload_ScheduleAction.action_id == payload.action_id,
        )
        if payload_ref is not None and CLIPMemory.select().where(
            CLIPMemory._payload_id == str(payload_ref.get_id())
        ).exists():
            return False
        return super().add_to_memory(image, payload, similarity_threshold, save_image)

    def retrieve(
        self,
        image: np.ndarray,
        similarity_threshold: float = 0.92,
    ) -> Optional[ScheduleActionIdentity]:
        """从 CLIP 记忆库检索最匹配的周行动。

        周行动图标差异较大（颜色/形状都不同），使用 0.92 阈值。
        """
        result = super().retrieve(image, similarity_threshold)
        if result is None:
            return None
        return result.payload
