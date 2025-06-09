from src.core.CLIP_services.item import ItemCLIP
from src.core.CLIP_services.skill_card import SkillCardCLIP

class CLIPServiceManager:
    skill_card_clip: SkillCardCLIP
    item_clip: ItemCLIP
    def __init__(self):
        self.skill_card_clip = SkillCardCLIP()
        self.item_clip = ItemCLIP()