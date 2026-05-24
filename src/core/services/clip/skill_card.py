import os
from typing import Optional

import cv2
import numpy as np

from src.entity.Game.Database.ProduceCard import ProduceCard
from src.models.clip import CLIPayload_SkillCard, CLIPMemory
from src.utils.game_database_tools import GakumasDatabase_ProduceCardDataUtils
from src.utils.clip_tools import CLIPTools

produce_card_db = GakumasDatabase_ProduceCardDataUtils()

class SkillCardCLIP(CLIPTools):
    def __init__(self, session):
        """初始化技能卡 CLIP 服务，使用 ``skill_card`` 命名空间。"""
        super().__init__(session, "skill_card")

    def _load_payload(self, payload_ref: "CLIPayload_SkillCard") -> Optional[ProduceCard]:
        """根据 payload 引用从数据库加载对应的 ProduceCard 对象。"""
        return produce_card_db.get_by_id(f"{payload_ref.id}.{payload_ref.level}")

    def _save_payload(self, image: np.ndarray, features: np.ndarray, payload: ProduceCard):
        """将技能卡图片写入 CLIP 图片目录，并持久化 payload 引用到数据库。

        文件以 ``{id}_{upgradeCount}.png`` 命名，不同升级等级分开存储。

        Args:
            image: 原始图像数组（BGR）。
            features: CLIP 模型提取的特征向量。
            payload: 对应的 :class:`~src.entity.Game.Database.ProduceCard.ProduceCard` 对象。

        Returns:
            持久化后的 :class:`~src.models.clip.CLIPayload_SkillCard` 实例。
        """
        image_save_path = os.path.join(self._image_file_path, f"{payload.id}_{payload.upgradeCount}.png")
        if not os.path.exists(image_save_path):
            cv2.imwrite(image_save_path, image)
        obj, created = CLIPayload_SkillCard.get_or_create(
            id=payload.id,
            level=payload.upgradeCount,
        )
        return obj

    def add_to_memory(self, image: np.ndarray, payload: ProduceCard, similarity_threshold=0.98, save_image=False):
        """将技能卡图片加入 CLIP 记忆库，相似度超过阈值的条目自动跳过。

        Args:
            image: 技能卡截图（BGR 数组）。
            payload: 对应的 :class:`~src.entity.Game.Database.ProduceCard.ProduceCard` 对象。
            similarity_threshold: 余弦相似度阈值，超过此值视为重复（默认 0.98）。
            save_image: 是否同时保存图片文件。

        Returns:
            是否成功写入（重复条目返回 False）。
        """
        # 快速路径：若 ID 已在记忆库中则跳过耗时的 ONNX 推理
        payload_ref = CLIPayload_SkillCard.get_or_none(
            CLIPayload_SkillCard.id == payload.id,
            CLIPayload_SkillCard.level == payload.upgradeCount,
        )
        if payload_ref is not None and CLIPMemory.select().where(
            CLIPMemory._payload_id == str(payload_ref.get_id())
        ).exists():
            return False
        return super().add_to_memory(image, payload, similarity_threshold, save_image)

    def retrieve(self, image: np.ndarray, similarity_threshold: float = 0.98) -> Optional[ProduceCard]:
        """从 CLIP 记忆库检索最匹配的技能卡。

        Args:
            image: 待识别的截图（BGR 数组）。
            similarity_threshold: 最低余弦相似度阈值（默认 0.98）。

        Returns:
            相似度达标的 :class:`~src.entity.Game.Database.ProduceCard.ProduceCard`，
            未命中时返回 ``None``。
        """
        result = super().retrieve(image, similarity_threshold)
        if result is None:
            return None
        return result.payload
