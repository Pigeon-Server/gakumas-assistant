from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.utils.logger import logger
from src.utils.clip_tools import CLIPTools

@dataclass
class ItemInfo:
    name: str
    info: List[str]


class ItemCLIP(CLIPTools):
    def __init__(self):
        super().__init__("items")

    def add_to_memory(self, image_frame: np.array, payload: ItemInfo, similarity_threshold = 0.97):
        return super().add_to_memory(image_frame, payload, similarity_threshold)

    def retrieve(self, image_frame: np.array, similarity_threshold: float = 0.9) -> Optional[ItemInfo]:
        result = super().retrieve(image_frame, similarity_threshold)
        logger.debug(f"result: {result}")
        if result is None:
            return None
        return result.payload