import os
from typing import Optional

import cv2
import numpy as np

from src.entity.Game.Database.IdolCard import IdolCard
from src.models.clip import CLIPayload_IdolCard, CLIPMemory
from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils
from src.utils.logger import logger
from src.utils.clip_tools import CLIPTools

idol_card_db = GakumasDatabase_IdolCardDataUtils()


class IdolCardCLIP(CLIPTools):
    def __init__(self, session):
        """初始化偶像卡 CLIP 服务，使用 ``idol_cards`` 命名空间。"""
        super().__init__(session, "idol_cards")

    def _load_payload(self, payload_ref: "CLIPayload_IdolCard") -> Optional[IdolCard]:
        """根据 payload 引用从数据库加载对应的 IdolCard 对象。"""
        return idol_card_db.get_by_id(payload_ref.id)

    def _save_payload(self, image: np.ndarray, features: np.ndarray, payload: IdolCard):
        """将偶像卡图片写入 CLIP 图片目录，并持久化 payload 引用到数据库。

        Args:
            image: 原始图像数组（BGR）。
            features: CLIP 模型提取的特征向量。
            payload: 对应的 :class:`~src.entity.Game.Database.IdolCard.IdolCard` 对象。

        Returns:
            持久化后的 :class:`~src.models.clip.CLIPayload_IdolCard` 实例。
        """
        image_save_path = os.path.join(self._image_file_path, f"{payload.id}.png")
        if not os.path.exists(image_save_path):
            cv2.imwrite(image_save_path, image)
        obj, created = CLIPayload_IdolCard.get_or_create(
            id=payload.id,
        )
        return obj

    def add_to_memory(self, image: np.ndarray, payload: IdolCard, similarity_threshold=0.96,
                      save_image=False, augment=True):
        """将偶像卡图片加入 CLIP 记忆库，相似度超过阈值的条目自动跳过。

        Args:
            image: 偶像卡截图（BGR 数组）。
            payload: 对应的 :class:`~src.entity.Game.Database.IdolCard.IdolCard` 对象。
            similarity_threshold: 余弦相似度阈值，超过此值视为重复（默认 0.96）。
            save_image: 是否同时保存图片文件。
            augment: 是否存储增强版本特征向量以提升鲁棒性。

        Returns:
            是否成功写入（重复条目返回 False）。
        """
        # 快速路径：若 ID 已在记忆库中则跳过耗时的 ONNX 推理
        payload_ref = CLIPayload_IdolCard.get_or_none(CLIPayload_IdolCard.id == payload.id)
        if payload_ref is not None and CLIPMemory.select().where(
            CLIPMemory._payload_id == str(payload_ref.get_id())
        ).exists():
            return False
        logger.debug(f"[IdolCardCLIP] Add: {payload.id} ({getattr(payload, 'name', '')})")
        return super().add_to_memory(image, payload, similarity_threshold, save_image,
                                     augment=augment)

    def add_variant_to_memory(
            self,
            image: np.ndarray,
            payload: IdolCard,
            similarity_threshold: float = 0.96,
            save_image: bool = False,
            augment: bool = False,
    ) -> bool:
        """向已存在 payload 追加变体图像，不走“已有 ID 直接跳过”的快速路径。"""
        logger.debug(f"[IdolCardCLIP] Add variant: {payload.id} ({getattr(payload, 'name', '')})")
        return CLIPTools.add_to_memory(
            self,
            image,
            payload,
            similarity_threshold=similarity_threshold,
            save_image=save_image,
            augment=augment,
        )

    def retrieve(self, image: np.ndarray, similarity_threshold: float = 0.95) -> Optional[IdolCard]:
        """从 CLIP 记忆库检索最匹配的偶像卡。

        Args:
            image: 待识别的截图（BGR 数组）。
            similarity_threshold: 最低余弦相似度阈值（默认 0.96）。

        Returns:
            相似度达标的 :class:`~src.entity.Game.Database.IdolCard.IdolCard`，
            未命中时返回 ``None``。
        """
        result = super().retrieve(image, similarity_threshold)
        if result is None:
            return None
        return result.payload
