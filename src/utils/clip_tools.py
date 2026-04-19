import abc
import os
import pickle
from abc import abstractmethod

import cv2
import numpy as np

from dataclasses import dataclass
from typing import Optional, Any, List

from src.core.inference.ONNX import CLIPModelFromONNX
from src.models.clip import CLIPMemory
from src.utils.logger import logger
from src.utils.runtime_paths import resolve_data_str

@dataclass
class CLIPRetrieveData:
    payload: Any
    similarity: float


def _augment_image(image: np.ndarray) -> List[np.ndarray]:
    """生成多种数据增强版本，提升 CLIP 检索对噪点/色偏/压缩的鲁棒性。

    包含位置偏移增强以应对 YOLO 边界框在不同截图间的微小漂移。

    Returns:
        增强后的图像列表（不含原图）。
    """
    augmented = []
    h, w = image.shape[:2]

    # 1. JPEG 压缩伪影（Q30 ~ 中低质量）
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, buf = cv2.imencode(".jpg", image, encode_param)
    augmented.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))

    # 2. 高斯噪点 (σ=15)
    noise = np.random.normal(0, 15, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    # 3. 亮度/对比度偏移（亮度+20, 对比度×1.15）
    bright = cv2.convertScaleAbs(image, alpha=1.15, beta=20)
    augmented.append(bright)

    # 4. 色调偏移（HSV H通道±8）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + 8) % 180
    augmented.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))

    # 5. 轻微缩放（0.85x 后 resize 回原尺寸，模拟分辨率差异）
    small = cv2.resize(image, (int(w * 0.85), int(h * 0.85)), interpolation=cv2.INTER_AREA)
    augmented.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))

    # 6-9. 随机裁剪/平移偏移 — 模拟 YOLO 边界框在不同帧间的漂移（±8%）
    shift_ratio = 0.08
    for dx_sign, dy_sign in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        dx = int(w * shift_ratio * dx_sign)
        dy = int(h * shift_ratio * dy_sign)
        # 仿射平移 + 边缘像素填充（避免黑边）
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(shifted)

    return augmented


class CLIPTools(abc.ABC):
    _image_file_path: str
    _clip_name: str
    _engine: CLIPModelFromONNX

    def __init__(self, model_session: CLIPModelFromONNX, clip_name: str):
        self._engine = model_session
        self._clip_name = clip_name
        self._image_file_path = resolve_data_str("CLIP", clip_name)
        os.makedirs(self._image_file_path, exist_ok=True)

    @ staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray):
        """计算向量相似度"""
        a = a.flatten()
        b = b.flatten()
        return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))).item()

    @abstractmethod
    def _save_payload(self, image: np.ndarray, features: np.ndarray, payload: Any):
        pass

    @abstractmethod
    def _load_payload(self, payload_ref):
        pass

    def _cleanup_stale_memories(
        self,
        stale_memory_ids: list[str],
        *,
        sample_messages: list[str],
    ) -> None:
        """清理失效的 CLIP 记忆条目，避免后续检索重复刷屏。"""
        if not stale_memory_ids:
            return

        deleted = 0
        try:
            delete_query = CLIPMemory.delete()
            memory_uuid_field = getattr(CLIPMemory, "uuid", None)
            if memory_uuid_field is not None and hasattr(memory_uuid_field, "in_"):
                delete_query = delete_query.where(memory_uuid_field.in_(stale_memory_ids))
            if hasattr(delete_query, "execute"):
                deleted = int(delete_query.execute() or 0)
        except Exception as exc:  # noqa: BLE001 - 清理失败不应中断主流程
            logger.warning(f"CLIP '{self._clip_name}' 清理失效记忆条目失败: {exc}")
            deleted = 0

        detail = "; ".join(sample_messages[:3])
        if len(sample_messages) > 3:
            detail = f"{detail}; 其余 {len(sample_messages) - 3} 条已省略"
        if deleted > 0:
            logger.warning(
                f"CLIP '{self._clip_name}' 已清理 {deleted} 条失效记忆条目: {detail}"
            )
        else:
            logger.warning(
                f"CLIP '{self._clip_name}' 检测到 {len(stale_memory_ids)} 条失效记忆条目: {detail}"
            )

    def add_to_memory(self, image: np.ndarray, payload, similarity_threshold: float = 0.9,
                      save_image: bool = False, augment: bool = True) -> bool:
        image_features = self._engine.forward(image)
        if image_features is None:
            raise RuntimeError("Image features is None")

        # 获取目标 payload 的标识（用于类内偏差验证）
        target_payload_id = str(getattr(payload, "id", "") or "")

        # 与同类（相同 payload id）已有样本的最高相似度，用于类内偏差检测
        best_inclass_similarity = -1.0
        # 缓存 _payload_id(UUID) → domain id 的映射，避免重复查询
        _uuid_to_domain_id: dict[str, str] = {}
        for memory in CLIPMemory.select().where(CLIPMemory.clip_name == self._clip_name):
            existing_features = pickle.loads(memory.features)
            similarity = self._cosine_similarity(image_features, existing_features)
            # 记录与同类样本的最高相似度
            raw_payload_id = str(getattr(memory, "_payload_id", "") or "")
            if target_payload_id and raw_payload_id:
                if raw_payload_id not in _uuid_to_domain_id:
                    try:
                        ref = memory.load_payload()
                        _uuid_to_domain_id[raw_payload_id] = str(getattr(ref, "id", "") or getattr(ref, "action_id", "") or raw_payload_id)
                    except Exception:
                        _uuid_to_domain_id[raw_payload_id] = raw_payload_id
                mem_domain_id = _uuid_to_domain_id[raw_payload_id]
                if mem_domain_id == target_payload_id:
                    if similarity > best_inclass_similarity:
                        best_inclass_similarity = similarity

        # 类内偏差验证：若已有同类样本，但新图像与同类样本相似度过低，说明图像与目标类不相关，拒绝学习
        # 下限阈值基于 similarity_threshold 动态计算，留出足够余量避免误拒合法变体
        _INCLASS_MIN_SIMILARITY = max(0.72, similarity_threshold - 0.18)
        if best_inclass_similarity >= 0 and best_inclass_similarity < _INCLASS_MIN_SIMILARITY:
            logger.warning(
                f"CLIP '{self._clip_name}' 类内偏差验证失败：图像与目标类 "
                f"{target_payload_id} 最高相似度仅 {best_inclass_similarity:.3f}，"
                f"低于下限 {_INCLASS_MIN_SIMILARITY:.3f}，拒绝学习"
            )
            return False

        payload_ref = self._save_payload(image, image_features, payload)

        # 存储原始特征
        CLIPMemory.save_vector(
            clip_name=self._clip_name,
            payload_obj=payload_ref,
            features=pickle.dumps(image_features)
        )

        # 存储增强版本的特征向量，提升对噪点/色偏/压缩的鲁棒性
        # 使用更宽松的阈值（0.95）判断增强特征是否足够不同，
        # 避免主去重阈值（通常 0.98）过于严格导致所有增强版本被过滤掉
        _AUG_DEDUP_THRESHOLD = 0.95
        if augment:
            aug_count = 0
            for aug_image in _augment_image(image):
                aug_features = self._engine.forward(aug_image)
                if aug_features is None:
                    continue
                # 只存储与原始特征差异足够大的增强版本
                aug_sim = self._cosine_similarity(image_features, aug_features)
                if aug_sim < _AUG_DEDUP_THRESHOLD:
                    CLIPMemory.save_vector(
                        clip_name=self._clip_name,
                        payload_obj=payload_ref,
                        features=pickle.dumps(aug_features)
                    )
                    aug_count += 1
            if aug_count > 0:
                logger.debug(f"Stored {aug_count} augmented feature vectors")

        return True

    def retrieve(self, image: np.ndarray, similarity_threshold: float = 0.9) -> Optional[CLIPRetrieveData]:
        image_features = self._engine.forward(image)
        if image_features is None:
            raise RuntimeError("Image features is None")

        best_similarity = -1
        best_payload = None
        stale_memory_ids: list[str] = []
        stale_messages: list[str] = []

        for memory in CLIPMemory.select().where(CLIPMemory.clip_name == self._clip_name):
            existing_features = pickle.loads(memory.features)
            similarity = self._cosine_similarity(image_features, existing_features)

            if similarity > best_similarity:
                # CLIP 记忆库可能残留已失效的 payload 引用。
                # 这类脏数据不应直接中断主流程，而应跳过后继续寻找下一条候选。
                try:
                    payload_ref = memory.load_payload()
                except Exception as exc:  # noqa: BLE001 - 这里需要兜底脏库数据
                    memory_id = str(
                        getattr(memory, "uuid", None)
                        or getattr(memory, "id", None)
                        or "unknown"
                    )
                    stale_memory_ids.append(memory_id)
                    stale_messages.append(f"{memory_id}: {exc}")
                    continue
                if payload_ref is None:
                    logger.debug(
                        f"CLIP '{self._clip_name}' 记忆条目 "
                        f"{getattr(memory, 'uuid', None) or getattr(memory, 'id', '?')} 未加载到 payload，跳过"
                    )
                    continue
                best_similarity = similarity
                best_payload = payload_ref

        self._cleanup_stale_memories(
            stale_memory_ids,
            sample_messages=stale_messages,
        )

        if best_payload is not None and best_similarity > similarity_threshold:
            return CLIPRetrieveData(self._load_payload(best_payload), best_similarity)

        # 未达到阈值时记录最佳候选的相似度，便于诊断 CLIP 识别率问题
        if best_payload is not None and best_similarity > 0.5:
            logger.debug(
                f"CLIP '{self._clip_name}' 最佳候选相似度 {best_similarity:.3f} "
                f"低于阈值 {similarity_threshold:.3f}，未命中"
            )

        return None
