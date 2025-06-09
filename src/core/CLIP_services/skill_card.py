from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.utils.logger import logger
from src.utils.clip_tools import CLIPTools, CLIPRetrieveData


@dataclass
class SkillCardInfo:
    name: str
    type: str
    info: List[str]

class SkillCardCLIP(CLIPTools):
    def __init__(self):
        logger.info("Loading Skill Card CLIP Data......")
        super().__init__("skill_card")


    def add_to_memory(self, image_frame: np.array, payload: SkillCardInfo, similarity_threshold = 0.97):
        return super().add_to_memory(image_frame, payload, similarity_threshold)

    def retrieve(self, image_frame: np.array, similarity_threshold: float = 0.9) -> Optional[SkillCardInfo]:
        result = super().retrieve(image_frame, similarity_threshold)
        if result is None:
            return None
        return result.payload
