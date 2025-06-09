from dataclasses import dataclass
from typing import Optional, List

import clip
import torch
import os
import pickle
import numpy as np
from PIL import Image
import cv2

from src.utils.logger import logger

# 载入 CLIP 模型和预处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

@dataclass
class CLIPMemoryItem:
    payload: any
    features: any

@dataclass
class CLIPRetrieveData:
    payload: any
    similarity: float

class CLIPTools:
    _memory_file_path: str
    _memory: List[CLIPMemoryItem]

    def _load(self):
        """加载记忆"""
        if os.path.exists(self._memory_file_path):
            with open(self._memory_file_path, 'rb') as f:
                self._memory = pickle.load(f)
        else:
            self._memory = []

    def _save(self):
        """保存记忆到本地"""
        with open(self._memory_file_path, 'wb') as f:
            pickle.dump(self._memory, f)

    @classmethod
    def _cosine_similarity(cls, a, b):
        """计算余弦相似度"""
        return (a @ b.T).squeeze(0) / (a.norm() * b.norm())

    def __init__(self, save_file_name: str):
        self._memory_file_path = os.path.join(os.getcwd(), "model/CLIP", save_file_name+".pkl")
        os.makedirs(os.path.dirname(self._memory_file_path), exist_ok=True)
        logger.info(f"Loading CLIP model from {self._memory_file_path}")
        self._load()

    def add_to_memory(self, image_frame: np.array, payload, similarity_threshold: float = 0.9) -> bool:
        """
        添加图像到记忆中
        :param image_frame: 图像
        :param payload: 载荷
        :param similarity_threshold:
        :return:
        """
        pil_image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))

        # 预处理图像并获取图像特征向量
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        # 获取图像特征向量
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # 检查图像是否已经在记忆库中
        for data in self._memory:
            # 计算当前图像与已存图像特征的余弦相似度
            similarity = self._cosine_similarity(image_features, data.features)

            # 如果相似度超过阈值，认为是重复图像
            if similarity > similarity_threshold:
                logger.debug(f"Image already exists with similarity: {similarity.item():.4f}")
                return False

        # 如果图像未找到重复，添加到记忆库
        self._memory.append(CLIPMemoryItem(payload, image_features))
        logger.debug(f"Added image to memory")
        # 保存到本地文件
        self._save()
        return True

    def retrieve(self, image_frame: np.array, similarity_threshold: float = 0.9) -> Optional[CLIPRetrieveData]:
        """
        使用图像检索记忆
        :param image_frame: 图像
        :param similarity_threshold: 阈值
        :return: CLIPRetrieveData | None
        """
        # 获取图像特征向量
        pil_image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # 计算图像与记忆库中所有图像特征的相似度
        similarities = []
        for data in self._memory:
            similarity = self._cosine_similarity(image_features, data.features)
            similarities.append(similarity.item())

        # 按相似度排序并返回最匹配的载荷
        if similarities:
            best_match_idx = np.argmax(similarities)
            best_match_similarity = similarities[best_match_idx]

            if best_match_similarity > similarity_threshold:
                matched_payload = self._memory[best_match_idx].payload
                return CLIPRetrieveData(matched_payload, best_match_similarity)

        return None