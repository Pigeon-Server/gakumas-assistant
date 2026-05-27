import json
import importlib
import os
import re
import threading
import hashlib
import pickle
from typing import Any, Dict, List, TextIO

import yaml

from src.entity.Game.Database.Character import Character, CharacterLocalization
from src.entity.Game.Database.EffectGroup import EffectGroup, EffectGroupLocalization
from src.entity.Game.Database.General import GeneralProduceDescriptionsLocalization
from src.entity.Game.Database.IdolCard import IdolCard, IdolCardLocalization
from src.entity.Game.Database.Item import Item, ItemLocalization
from src.entity.Game.Database.ProduceCard import ProduceCard, ProduceCardLocalization
from src.entity.Game.Database.ProduceCardCustomize import (
    ProduceCardCustomize,
    ProduceCardCustomizeLocalization,
)
from src.entity.Game.Database.ProduceCardGrowEffect import ProduceCardGrowEffect
from src.entity.Game.Database.ProduceCardSearch import (
    ProduceCardSearch,
    ProduceCardSearchLocalization,
)
from src.entity.Game.Database.ProduceCardStatusEnchant import (
    ProduceCardStatusEnchant,
    ProduceCardStatusEnchantLocalization,
)
from src.entity.Game.Database.ProduceDrink import ProduceDrink, ProduceDrinkLocalization
from src.entity.Game.Database.ProduceExamEffect import ProduceExamEffect
from src.entity.Game.Database.ProduceExamStatusEnchant import (
    ProduceExamStatusEnchant,
    ProduceExamStatusEnchantLocalization,
)
from src.entity.Game.Database.ProduceExamTrigger import ProduceExamTrigger
from src.entity.Game.Database.ProduceItem import ProduceItem, ProduceItemLocalization
from src.entity.Game.Database.ProduceSkill import ProduceSkill, ProduceSkillLocalization
from src.entity.Game.Database.SupportCard import SupportCard, SupportCardLocalization
from src.entity.Game.Database.SupportCardProduceSkillLevelAssist import SupportCardProduceSkillLevelAssist
from src.entity.Game.Database.SupportCardProduceSkillLevelDance import SupportCardProduceSkillLevelDance
from src.entity.Game.Database.SupportCardProduceSkillLevelVisual import SupportCardProduceSkillLevelVisual
from src.entity.Game.Database.SupportCardProduceSkillLevelVocal import SupportCardProduceSkillLevelVocal
from src.utils.data_converter import DataConverter
from src.utils.logger import logger
from src.utils.runtime_paths import resolve_cache_str, resolve_existing_resource_path
from src.utils.string_tools import string_match, MatchConfig

_CACHE_DIR = resolve_cache_str("yaml_db")


class _SingletonByFileMeta(type):
    _instances = {}
    _lock = threading.RLock()

    def __call__(cls, data_file=None, *args, **kwargs):
        if data_file is None:
            data_file = cls._get_default_data_file()
        key = (cls, os.path.abspath(data_file))
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    cls._instances[key] = super().__call__(data_file, *args, **kwargs)
        return cls._instances[key]


class _BaseYamlDatabase(metaclass=_SingletonByFileMeta):
    data_cls = None
    loc_cls = None
    default_data_file_parts: tuple[str, ...] = ()
    _diff_file: str = None
    _data: List[Any] = None
    _map: Dict[str, Any] = None
    _raw_id_map: Dict[str, List[Any]] = None

    def __init__(self, data_file):
        self._diff_file = data_file
        if not os.path.exists(data_file):
            raise FileNotFoundError(data_file)
        self._load_database()

    @classmethod
    def _get_default_data_file(cls):
        if not cls.default_data_file_parts:
            raise ValueError(f"{cls.__name__} does not define a default data file")
        return str(resolve_existing_resource_path(*cls.default_data_file_parts))

    @classmethod
    def _preprocess_yaml_data(cls, f: TextIO) -> str:
        content = f.read()
        content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", content)
        content = content.replace("\t", "    ")
        return content

    @classmethod
    def _load_localization(cls, data_file_path, data_entity):
        loc_file = str(
            resolve_existing_resource_path(
                "assets",
                "GakumasTranslationData",
                "local-files",
                "masterTrans",
                f"{os.path.splitext(os.path.basename(data_file_path))[0]}.json",
            )
        )
        if not os.path.exists(loc_file):
            return []

        with open(loc_file, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries.get("data", []):
            if pd := entry.get("produceDescriptions"):
                entry["produceDescriptions"] = [d for d in pd if isinstance(d, dict)]
            if upd := entry.get("upgradeProduceCardProduceDescriptions"):
                entry["upgradeProduceCardProduceDescriptions"] = [
                    d for d in upd if isinstance(d, dict)
                ]

        return [
            DataConverter.from_dict(data_entity, entry)
            for entry in entries.get("data", [])
        ]

    def _get_file_hash(self) -> str:
        with open(self._diff_file, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_cache_path(self, file_hash: str) -> str:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        return os.path.join(_CACHE_DIR, f"{file_hash}.pkl")

    def _load_yaml(self) -> list[dict]:
        file_hash = self._get_file_hash()
        cache_path = self._get_cache_path(file_hash)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except (EOFError, OSError, pickle.PickleError, TypeError, ValueError) as exc:
                logger.debug("GameDatabase: 读取缓存失败，将重新解析: {}", exc)

        with open(self._diff_file, "r", encoding="utf-8") as f:
            content = self._preprocess_yaml_data(f)
        data = yaml.load(content, Loader=yaml.CSafeLoader)
        if data is None:
            return []
        if not isinstance(data, list):
            raise TypeError(
                f"{self._diff_file} parsed to {type(data).__name__}, expected list"
            )

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.debug(f"GameDatabase: 保存缓存失败: {e}")

        return data

    def _load_objects(self, entries) -> list:
        return [DataConverter.from_dict(self.data_cls, entry) for entry in entries]

    def _load_localization_data(self):
        if not self.loc_cls:
            return {}
        locs = self._load_localization(self._diff_file, self.loc_cls)
        return self._build_loc_map(locs)

    def _build_loc_map(self, loc_objects):
        return {o.id: o for o in loc_objects}

    def _build_map_key(self, obj):
        return getattr(obj, "id")

    @staticmethod
    def _id_list_to_objects(ids: List[str], db: "_BaseYamlDatabase") -> List[Any]:
        return [db.get_by_id(v) for v in ids]

    def _load_database(self):
        entries = self._load_yaml()
        objects = self._load_objects(entries)
        loc_map = self._load_localization_data()

        for obj in objects:
            if self.loc_cls:
                obj.localization = loc_map.get(self._build_map_key(obj))

        self._data = objects
        self._map = {self._build_map_key(o): o for o in objects}
        self._raw_id_map = {}
        for o in objects:
            if hasattr(o, "id"):
                self._raw_id_map.setdefault(str(getattr(o, "id")), []).append(o)

        logger.success(
            f"[{self.__class__.__name__}] {len(self._data)} records loaded from {self._diff_file}"
        )

    def reload(self):
        self._load_database()
        return self

    def get_all_item(self):
        return self._data

    def get_map(self):
        return self._map

    def get_by_id(self, id):
        return self._map.get(id)

    def get_all_by_raw_id(self, raw_id) -> List[Any]:
        if raw_id is None:
            return []
        return self._raw_id_map.get(str(raw_id), [])

    def get_by_raw_id(self, raw_id):
        items = self.get_all_by_raw_id(raw_id)
        return items[0] if items else None

    def has_raw_id(self, raw_id) -> bool:
        return len(self.get_all_by_raw_id(raw_id)) > 0

    def search_by_name(self, keyword, match_config=None):
        name_map = {c.name: c for c in self._data if hasattr(c, "name") and c.name}
        if match_config is not None and not match_config.normalize:
            cfg = MatchConfig(
                use_regex=match_config.use_regex,
                use_fuzz=match_config.use_fuzz,
                fuzz_threshold=match_config.fuzz_threshold,
                use_contains=match_config.use_contains,
                normalize=True,
            )
        else:
            cfg = MatchConfig(normalize=True) if match_config is None else match_config
        result = string_match(keyword, list(name_map.keys()), cfg)
        if not result:
            return False, None
        return True, name_map[result.result]


class GakumasDatabase_ItemDataUtils(_BaseYamlDatabase):
    data_cls = Item
    loc_cls = ItemLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "Item.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def search(self, ocr_result, match_config=None):
        return self.search_by_name(ocr_result, match_config)


class GakumasDatabase_CharacterDataUtils(_BaseYamlDatabase):
    data_cls = Character
    loc_cls = CharacterLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "Character.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)


class GakumasDatabase_EffectGroupDataUtils(_BaseYamlDatabase):
    data_cls = EffectGroup
    loc_cls = EffectGroupLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "EffectGroup.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)


class GakumasDatabase_ExamTriggerDataUtils(_BaseYamlDatabase):
    data_cls = ProduceExamTrigger
    loc_cls = GeneralProduceDescriptionsLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceExamTrigger.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)


class GakumasDatabase_ExamEffectDataUtils(_BaseYamlDatabase):
    data_cls = ProduceExamEffect
    loc_cls = GeneralProduceDescriptionsLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceExamEffect.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)


class GakumasDatabase_GrowEffectDataUtils(_BaseYamlDatabase):
    data_cls = ProduceCardGrowEffect
    loc_cls = None
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceCardGrowEffect.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _load_database(self):
        super()._load_database()

        exam_triggers = GakumasDatabase_ExamTriggerDataUtils()
        exam_effects = GakumasDatabase_ExamEffectDataUtils()

        for ge in self._data:
            ge.playProduceExamTriggerCls = exam_triggers.get_by_id(ge.playProduceExamTriggerId)
            ge.playEffectProduceExamTriggerCls = exam_triggers.get_by_id(
                ge.playEffectProduceExamTriggerId
            )
            ge.targetPlayEffectProduceExamTriggerClss = [
                exam_triggers.get_by_id(tid)
                for tid in ge.targetPlayEffectProduceExamTriggerIds
            ]
            ge.playProduceExamEffectCls = exam_effects.get_by_id(ge.playProduceExamEffectId)
            ge.targetPlayProduceExamEffectClss = [
                exam_effects.get_by_id(eid) for eid in ge.targetPlayProduceExamEffectIds
            ]


class GakumasDatabase_ProduceCardSearchDataUtils(_BaseYamlDatabase):
    data_cls = ProduceCardSearch
    loc_cls = ProduceCardSearchLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceCardSearch.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _load_database(self):
        super()._load_database()

        effect_groups = GakumasDatabase_EffectGroupDataUtils()
        for search in self._data:
            search.effectGroupClss = self._id_list_to_objects(search.effectGroupIds, effect_groups)


class GakumasDatabase_CardStatusEnchantDataUtils(_BaseYamlDatabase):
    data_cls = ProduceCardStatusEnchant
    loc_cls = ProduceCardStatusEnchantLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceCardStatusEnchant.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _load_database(self):
        super()._load_database()

        exam_trigger = GakumasDatabase_ExamTriggerDataUtils()
        grow_effects = GakumasDatabase_GrowEffectDataUtils()

        for enchant in self._data:
            enchant.produceExamTriggerCls = exam_trigger.get_by_id(enchant.produceExamTriggerId)
            enchant.produceCardGrowEffectClss = self._id_list_to_objects(
                enchant.produceCardGrowEffectIds, grow_effects
            )


class GakumasDatabase_ExamStatusEnchantDataUtils(_BaseYamlDatabase):
    data_cls = ProduceExamStatusEnchant
    loc_cls = ProduceExamStatusEnchantLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceExamStatusEnchant.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _load_database(self):
        super()._load_database()

        exam_trigger = GakumasDatabase_ExamTriggerDataUtils()
        exam_effect = GakumasDatabase_ExamEffectDataUtils()

        for enchant in self._data:
            enchant.produceExamTriggerCls = exam_trigger.get_by_id(enchant.produceExamTriggerId)
            enchant.produceExamEffectClss = self._id_list_to_objects(
                enchant.produceExamEffectIds, exam_effect
            )


class GakumasDatabase_ProduceCardCustomizeDataUtils(_BaseYamlDatabase):
    data_cls = ProduceCardCustomize
    loc_cls = ProduceCardCustomizeLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceCardCustomize.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _build_map_key(self, obj):
        return f"{obj.id}.{obj.customizeCount}"

    def _build_loc_map(self, loc_objects):
        return {f"{o.id}.{o.customizeCount}": o for o in loc_objects}

    def _load_database(self):
        super()._load_database()

        grow_effects = GakumasDatabase_GrowEffectDataUtils()
        for customize in self._data:
            customize.produceCardGrowEffectClss = self._id_list_to_objects(
                customize.produceCardGrowEffectIds, grow_effects
            )


class GakumasDatabase_ProduceCardDataUtils(_BaseYamlDatabase):
    data_cls = ProduceCard
    loc_cls = ProduceCardLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceCard.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _build_map_key(self, card):
        return f"{card.id}.{card.upgradeCount}"

    def _build_loc_map(self, loc_objects):
        return {f"{o.id}.{o.upgradeCount}": o for o in loc_objects}

    def _load_database(self):
        super()._load_database()

        exam_effect = GakumasDatabase_ExamEffectDataUtils()
        exam_trigger = GakumasDatabase_ExamTriggerDataUtils()
        customize_db = GakumasDatabase_ProduceCardCustomizeDataUtils()
        customize_by_id = {}
        for customize in customize_db.get_all_item():
            customize_by_id.setdefault(customize.id, []).append(customize)
        for customize_list in customize_by_id.values():
            customize_list.sort(key=lambda c: c.customizeCount)

        for card in self._data:
            for e in card.playEffects:
                if e.produceExamTriggerId:
                    e.produceExamTriggerCls = exam_trigger.get_by_id(e.produceExamTriggerId)
                if e.produceExamEffectId:
                    e.produceExamEffectCls = exam_effect.get_by_id(e.produceExamEffectId)

            card.playProduceExamTriggerCls = exam_trigger.get_by_id(card.playProduceExamTriggerId)
            card.moveProduceExamTriggerClss = self._id_list_to_objects(
                card.moveProduceExamTriggerIds, exam_trigger
            )
            card.moveProduceExamEffectClss = self._id_list_to_objects(
                card.moveProduceExamEffectIds, exam_effect
            )
            card.produceCardCustomizeMap = {
                customize_id: list(customize_by_id.get(customize_id, []))
                for customize_id in card.produceCardCustomizeIds
            }
            card.produceCardCustomizeClss = [
                customize
                for customize_id in card.produceCardCustomizeIds
                for customize in customize_by_id.get(customize_id, [])
            ]

    def search(self, ocr_result, match_config=None):
        return self.search_by_name(ocr_result, match_config)


class GakumasDatabase_ProduceItemDataUtils(_BaseYamlDatabase):
    data_cls = ProduceItem
    loc_cls = ProduceItemLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceItem.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _load_database(self):
        super()._load_database()

        effect_groups = GakumasDatabase_EffectGroupDataUtils()
        for item in self._data:
            item.effectGroupClss = self._id_list_to_objects(item.effectGroupIds, effect_groups)

    def search(self, ocr_result, match_config=None):
        return self.search_by_name(ocr_result, match_config)


class GakumasDatabase_ProduceDrinkDataUtils(_BaseYamlDatabase):
    data_cls = ProduceDrink
    loc_cls = ProduceDrinkLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceDrink.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _load_database(self):
        super()._load_database()

        effect_groups = GakumasDatabase_EffectGroupDataUtils()
        for drink in self._data:
            drink.effectGroupClss = self._id_list_to_objects(drink.effectGroupIds, effect_groups)

    def search(self, ocr_result, match_config=None):
        return self.search_by_name(ocr_result, match_config)


class GakumasDatabase_ProduceSkillDataUtils(_BaseYamlDatabase):
    data_cls = ProduceSkill
    loc_cls = ProduceSkillLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "ProduceSkill.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _build_map_key(self, obj):
        return f"{obj.id}.{obj.level}"

    def _build_loc_map(self, loc_objects):
        return {f"{o.id}.{o.level}": o for o in loc_objects}


class GakumasDatabase_IdolCardDataUtils(_BaseYamlDatabase):
    data_cls = IdolCard
    loc_cls = IdolCardLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "IdolCard.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _load_database(self):
        super()._load_database()

        character_db = GakumasDatabase_CharacterDataUtils()
        produce_card_db = GakumasDatabase_ProduceCardDataUtils()
        produce_item_db = GakumasDatabase_ProduceItemDataUtils()

        produce_card_base_map = {}
        for card in produce_card_db.get_all_item():
            if card.id not in produce_card_base_map or card.upgradeCount == 0:
                produce_card_base_map[card.id] = card

        for idol_card in self._data:
            idol_card.characterCls = character_db.get_by_id(idol_card.characterId)
            idol_card.produceCardCls = produce_card_base_map.get(idol_card.produceCardId)
            idol_card.beforeProduceItemCls = produce_item_db.get_by_id(
                idol_card.beforeProduceItemId
            )
            idol_card.afterProduceItemCls = produce_item_db.get_by_id(
                idol_card.afterProduceItemId
            )

    def search(self, ocr_result, match_config=None):
        return self.search_by_name(ocr_result, match_config)


class _SupportCardSkillLevelMixin:
    """Mixin for SupportCardProduceSkillLevel tables that lack an ``id`` field."""

    data_cls = None
    loc_cls = None
    default_data_file_parts: tuple[str, ...] = ()

    def _build_map_key(self, obj):
        return f"{obj.supportCardId}|{obj.produceSkillId}|{obj.produceSkillLevel}"


class GakumasDatabase_SupportCardSkillLevelVocalUtils(_SupportCardSkillLevelMixin, _BaseYamlDatabase):
    data_cls = SupportCardProduceSkillLevelVocal
    loc_cls = None
    default_data_file_parts = ("assets", "gakumasu-diff", "SupportCardProduceSkillLevelVocal.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)


class GakumasDatabase_SupportCardSkillLevelDanceUtils(_SupportCardSkillLevelMixin, _BaseYamlDatabase):
    data_cls = SupportCardProduceSkillLevelDance
    loc_cls = None
    default_data_file_parts = ("assets", "gakumasu-diff", "SupportCardProduceSkillLevelDance.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)


class GakumasDatabase_SupportCardSkillLevelVisualUtils(_SupportCardSkillLevelMixin, _BaseYamlDatabase):
    data_cls = SupportCardProduceSkillLevelVisual
    loc_cls = None
    default_data_file_parts = ("assets", "gakumasu-diff", "SupportCardProduceSkillLevelVisual.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)


class GakumasDatabase_SupportCardSkillLevelAssistUtils(_SupportCardSkillLevelMixin, _BaseYamlDatabase):
    data_cls = SupportCardProduceSkillLevelAssist
    loc_cls = None
    default_data_file_parts = ("assets", "gakumasu-diff", "SupportCardProduceSkillLevelAssist.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)


def build_support_card_skill_descriptions() -> dict[str, list[dict]]:
    """构建支援卡 → 技能槽位列表的映射（含各等级描述）。

    通过关联 SupportCardProduceSkillLevel → ProduceSkill 数据，
    为每张支援卡汇总其所有技能槽位，每个槽位包含各等级的描述文本和解锁条件。

    Returns:
        {支援卡 ID: [
            {
                "order": int,          # 技能槽位序号
                "levels": [
                    {
                        "cardLevel": int,   # 解锁所需卡等级
                        "skillLevel": int,  # 技能等级
                        "description": str, # 技能描述文本
                    },
                    ...
                ]
            },
            ...
        ]} 的映射字典。
    """
    skill_db = GakumasDatabase_ProduceSkillDataUtils()

    # 加载所有 4 种类型的 SkillLevel 数据
    all_skill_levels = []
    for utils_cls in (
        GakumasDatabase_SupportCardSkillLevelVocalUtils,
        GakumasDatabase_SupportCardSkillLevelDanceUtils,
        GakumasDatabase_SupportCardSkillLevelVisualUtils,
        GakumasDatabase_SupportCardSkillLevelAssistUtils,
    ):
        try:
            db = utils_cls()
            all_skill_levels.extend(db.get_all_item())
        except Exception as e:
            logger.debug(f"GameDatabase: 操作失败: {e}")


    # 按 (supportCardId, order, produceSkillId) 收集各等级
    from collections import defaultdict
    card_slots: dict[str, dict[int, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for sl in all_skill_levels:
        card_id = sl.supportCardId
        order = sl.order or 0
        skill_id = sl.produceSkillId
        card_slots[card_id][order][skill_id].append(sl)

    # 组装结果
    result: dict[str, list[dict]] = {}
    for card_id, orders in card_slots.items():
        slots = []
        for order in sorted(orders.keys()):
            skills_for_order = orders[order]
            # 通常一个 order 只有一个 produceSkillId
            for skill_id, level_entries in skills_for_order.items():
                levels = []
                for sl in sorted(level_entries, key=lambda x: x.produceSkillLevel or 1):
                    key = f"{sl.produceSkillId}.{sl.produceSkillLevel}"
                    skill = skill_db.get_by_id(key)
                    if not skill:
                        continue
                    source = skill.localization if skill.localization else skill
                    desc_list = getattr(source, "produceDescriptions", [])
                    combined = _concat_produce_descriptions(desc_list)
                    # Also collect Japanese description for OCR matching
                    combined_ja = ""
                    if getattr(skill, "localization", None):
                        combined_ja = _concat_produce_descriptions(
                            getattr(skill, "produceDescriptions", [])
                        )
                    if combined or combined_ja:
                        entry_dict: dict[str, Any] = {
                            "cardLevel": sl.supportCardLevel or 1,
                            "skillLevel": sl.produceSkillLevel or 1,
                            "description": combined or combined_ja,
                        }
                        if combined_ja:
                            entry_dict["description_ja"] = combined_ja
                        levels.append(entry_dict)
                if levels:
                    slots.append({"order": order, "levels": levels})
        if slots:
            result[card_id] = slots
    return result


def get_skill_descriptions_at_level(
    skill_slots: list[dict], card_level: int
) -> list[str]:
    """从技能槽位列表中提取指定卡等级下的生效描述。

    对每个技能槽位，从后向前遍历等级列表，找到 cardLevel <= card_level 的最高一条。

    Args:
        skill_slots: build_support_card_skill_descriptions 返回的槽位列表。
        card_level: 当前卡等级。

    Returns:
        按槽位序号排列的描述文本列表。
    """
    descs = []
    for slot in skill_slots:
        levels = slot.get("levels", [])
        best = None
        for entry in reversed(levels):
            if entry["cardLevel"] <= card_level:
                best = entry
                break
        if best:
            descs.append(best["description"])
    return descs


def _strip_html_tags(text: str) -> str:
    """移除字符串中的 HTML 标签（如 <nobr>, </nobr> 等）。"""
    return re.sub(r"<[^>]+>", "", text)


def _concat_produce_descriptions(descs_objs, item_db=None) -> str:
    """将 produceDescriptions token 列表按顺序拼接为单一展示文本。

    游戏数据中 produceDescriptions 是一组语义 token，每个 token 的 text 字段
    构成最终文本的一部分，需全部拼接才能得到完整描述。

    Args:
        descs_objs: produceDescriptions 列表（dict 或 dataclass）。
        item_db: 可选的 P 物品数据库；传入后，ProduceItem 类型 token
                 会用本地化物品名替换原始（日文）text。
    Returns:
        拼接并去除 HTML 标签后的单一描述字符串。
    """
    parts = []
    for d in descs_objs:
        if isinstance(d, dict):
            dtype = d.get("produceDescriptionType", "")
            text = d.get("text", "")
            target_id = d.get("targetId", "")
        else:
            dtype = getattr(d, "produceDescriptionType", "")
            text = getattr(d, "text", "")
            target_id = getattr(d, "targetId", "")

        # ProduceItem 类型：优先用本地化物品名替换原始文本
        if dtype == "ProduceDescriptionType_ProduceItem" and target_id and item_db:
            item = item_db.get_by_id(target_id)
            if item:
                loc = getattr(item, "localization", None)
                name = (getattr(loc, "name", None) if loc else None) or getattr(item, "name", text)
                if name:
                    parts.append(name)
                    continue

        if text:
            parts.append(_strip_html_tags(text))
    return "".join(parts).strip()


def build_support_card_event_items() -> dict[str, list[dict]]:
    """构建支援卡 → 附带 P 物品 / スキルカード 的映射。

    通过关联 ProduceEventSupportCard → ProduceStepEventDetail，
    提取 ProduceDescriptionType_ProduceItem 和 ProduceDescriptionType_ProduceCard
    两类事件奖励。

    Returns:
        {支援卡 ID: [
            {
                "id": str,        # 物品/卡牌ID
                "kind": str,      # "item" | "card"
                "name": str,      # 名称（优先本地化）
                "rarity": str,    # 稀有度枚举值
                "planType": str,  # 路线枚举值
                "category": str,  # 卡牌类别（仅 card 有值）
                "assetId": str,   # 资源ID（可用于图片）
                "descriptions": [str],  # 效果描述文本列表
            },
            ...
        ]}
    """

    # 加载 ProduceEventSupportCard（无 id 字段，用 AutoDataUtils 行号索引）
    event_sc_yaml = str(resolve_existing_resource_path("assets", "gakumasu-diff", "ProduceEventSupportCard.yaml"))
    event_sc_db = GakumasDatabase_AutoDataUtils(data_file=event_sc_yaml, table_name="ProduceEventSupportCard")

    # 加载 ProduceStepEventDetail（有 id 字段，可按 id get）
    step_event_yaml = str(resolve_existing_resource_path("assets", "gakumasu-diff", "ProduceStepEventDetail.yaml"))
    step_event_db = GakumasDatabase_AutoDataUtils(data_file=step_event_yaml, table_name="ProduceStepEventDetail")

    # 加载 ProduceItem（用专用 Utils，含本地化名称）
    produce_item_db = GakumasDatabase_ProduceItemDataUtils()

    # 加载 ProduceCard（用专用 Utils，含本地化名称）
    produce_card_db = GakumasDatabase_ProduceCardDataUtils()

    # 构建 supportCardId → [produceStepEventDetailId] 映射
    card_to_events: dict[str, list[str]] = {}
    for event in event_sc_db.get_all_item():
        cid = getattr(event, "supportCardId", None)
        did = getattr(event, "produceStepEventDetailId", None)
        if cid and did:
            card_to_events.setdefault(cid, []).append(did)

    result: dict[str, list[dict]] = {}
    for card_id, event_detail_ids in card_to_events.items():
        items_found = []
        seen_ids: set[str] = set()
        for detail_id in event_detail_ids:
            detail = step_event_db.get_by_id(detail_id)
            if not detail:
                continue
            for desc in getattr(detail, "produceDescriptions", []):
                if isinstance(desc, dict):
                    dtype = desc.get("produceDescriptionType", "")
                    target_id = desc.get("targetId", None)
                    text = desc.get("text", "")
                else:
                    dtype = getattr(desc, "produceDescriptionType", "")
                    target_id = getattr(desc, "targetId", None)
                    text = getattr(desc, "text", "")

                if dtype == "ProduceDescriptionType_ProduceItem":
                    if not target_id or target_id in seen_ids:
                        continue
                    seen_ids.add(target_id)
                    item = produce_item_db.get_by_id(target_id)
                    if not item:
                        continue
                    name = None
                    if item.localization and getattr(item.localization, "name", None):
                        name = item.localization.name
                    if not name:
                        name = item.name
                    # 收集效果描述（将所有 token 拼接为单一描述文本）
                    source = item.localization if item.localization else item
                    desc_concat = _concat_produce_descriptions(getattr(source, "produceDescriptions", []))
                    descs = [desc_concat] if desc_concat else []
                    items_found.append({
                        "id": target_id,
                        "kind": "item",
                        "name": name or target_id,
                        "rarity": item.rarity or "",
                        "planType": item.planType or "",
                        "category": "",
                        "assetId": item.assetId or "",
                        "descriptions": descs,
                    })

                elif dtype == "ProduceDescriptionType_ProduceCard":
                    if not target_id or target_id in seen_ids:
                        continue
                    seen_ids.add(target_id)
                    # ProduceCard 的 key 格式是 "{id}.{upgradeCount}"，事件给的是 upgradeCount=0
                    card_obj = produce_card_db.get_by_id(f"{target_id}.0")
                    if not card_obj:
                        # fallback: 直接用 text 作为名称
                        items_found.append({
                            "id": target_id,
                            "kind": "card",
                            "name": text or target_id,
                            "rarity": "",
                            "planType": "",
                            "category": "",
                            "assetId": "",
                            "descriptions": [],
                        })
                        continue
                    name = None
                    if card_obj.localization and getattr(card_obj.localization, "name", None):
                        name = card_obj.localization.name
                    if not name:
                        name = card_obj.name
                    # 收集卡牌效果描述（将所有 token 拼接为单一描述文本）
                    source = card_obj.localization if card_obj.localization else card_obj
                    desc_concat = _concat_produce_descriptions(getattr(source, "produceDescriptions", []))
                    descs = [desc_concat] if desc_concat else []
                    items_found.append({
                        "id": target_id,
                        "kind": "card",
                        "name": name or target_id,
                        "rarity": card_obj.rarity or "",
                        "planType": card_obj.planType or "",
                        "category": card_obj.category or "",
                        "assetId": card_obj.assetId or "",
                        "descriptions": descs,
                    })

        if items_found:
            result[card_id] = items_found
    return result


def build_support_card_level_limits() -> dict[str, list[dict]]:
    """构建 supportCardLevelLimitId → 各突破阶段等级上限 的映射。

    数据来自 SupportCardLevelLimit.yaml，其中 rank=Unknown 为未突破，
    rank=__1~__4 为 1~4 阶突破。

    Returns:
        {supportCardLevelLimitId: [
            {"rank": 0, "levelLimit": 20},  # Unknown=0星
            {"rank": 1, "levelLimit": 25},  # __1=1星
            ...
        ]}
    """
    limit_yaml = str(resolve_existing_resource_path(
        "assets", "gakumasu-diff", "SupportCardLevelLimit.yaml"
    ))
    limit_db = GakumasDatabase_AutoDataUtils(
        data_file=limit_yaml, table_name="SupportCardLevelLimit"
    )
    from collections import defaultdict
    id_map: dict[str, list[dict]] = defaultdict(list)
    rank_order = {
        "SupportCardLevelLimitRank_Unknown": 0,
        "SupportCardLevelLimitRank__1": 1,
        "SupportCardLevelLimitRank__2": 2,
        "SupportCardLevelLimitRank__3": 3,
        "SupportCardLevelLimitRank__4": 4,
    }
    for entry in limit_db.get_all_item():
        eid = getattr(entry, "id", None)
        rank_str = getattr(entry, "rank", "")
        level_limit = getattr(entry, "levelLimit", 0)
        if eid and level_limit:
            id_map[eid].append({
                "rank": rank_order.get(rank_str, 0),
                "levelLimit": level_limit,
            })
    # 按 rank 排序
    for k in id_map:
        id_map[k].sort(key=lambda x: x["rank"])
    return dict(id_map)


def build_support_card_events() -> dict[str, list[dict]]:
    """构建支援卡 → サポートイベント一览 的映射。

    通过 ProduceEventSupportCard → ProduceStepEventDetail → ProduceStory
    关联事件标题、解锁等级、效果描述。

    Returns:
        {支援卡 ID: [
            {
                "number": int,           # 事件编号
                "supportCardLevel": int,  # 解锁所需卡等级
                "title": str,            # 事件标题
                "descriptions": [str],   # 事件效果描述
            },
            ...
        ]}
    """

    event_sc_yaml = str(resolve_existing_resource_path(
        "assets", "gakumasu-diff", "ProduceEventSupportCard.yaml"
    ))
    event_sc_db = GakumasDatabase_AutoDataUtils(
        data_file=event_sc_yaml, table_name="ProduceEventSupportCard"
    )

    step_event_yaml = str(resolve_existing_resource_path(
        "assets", "gakumasu-diff", "ProduceStepEventDetail.yaml"
    ))
    step_event_db = GakumasDatabase_AutoDataUtils(
        data_file=step_event_yaml, table_name="ProduceStepEventDetail"
    )

    story_yaml = str(resolve_existing_resource_path(
        "assets", "gakumasu-diff", "ProduceStory.yaml"
    ))
    story_db = GakumasDatabase_AutoDataUtils(
        data_file=story_yaml, table_name="ProduceStory"
    )

    # 加载 P 物品数据库（用于替换 ProduceItem token 中的本地化名称）
    produce_item_db = GakumasDatabase_ProduceItemDataUtils()

    from collections import defaultdict
    result: dict[str, list[dict]] = defaultdict(list)

    for event in event_sc_db.get_all_item():
        card_id = getattr(event, "supportCardId", None)
        number = getattr(event, "number", 0)
        unlock_level = getattr(event, "supportCardLevel", 1)
        detail_id = getattr(event, "produceStepEventDetailId", None)
        if not card_id or not detail_id:
            continue

        detail = step_event_db.get_by_id(detail_id)
        story_id = getattr(detail, "produceStoryId", None) if detail else None

        # 事件标题
        title = ""
        title_ja = ""
        if story_id:
            story = story_db.get_by_id(story_id)
            if story:
                loc = getattr(story, "localization", None)
                if loc and getattr(loc, "title", None):
                    title = loc.title
                # Japanese title from base (non-localized) object
                if getattr(story, "title", None):
                    title_ja = story.title
                if not title:
                    title = title_ja

        # 事件效果描述：将所有 token 拼接为单一字符串（ProduceItem 类型用本地化名称替换）
        descriptions = []
        if detail:
            descs_raw = getattr(detail, "produceDescriptions", [])
            concat = _concat_produce_descriptions(descs_raw, item_db=produce_item_db)
            if concat:
                descriptions = [concat]

        event_entry: dict[str, Any] = {
            "number": number,
            "supportCardLevel": unlock_level,
            "title": title or f"イベント{number}",
            "descriptions": descriptions,
        }
        if title_ja:
            event_entry["title_ja"] = title_ja
        result[card_id].append(event_entry)

    # 按 number 排序
    for k in result:
        result[k].sort(key=lambda x: x["number"])
    return dict(result)


class GakumasDatabase_SupportCardDataUtils(_BaseYamlDatabase):
    data_cls = SupportCard
    loc_cls = SupportCardLocalization
    default_data_file_parts = ("assets", "gakumasu-diff", "SupportCard.yaml")

    def __init__(self, data_file=None):
        super().__init__(data_file)

    def _load_database(self):
        super()._load_database()

        search_db = GakumasDatabase_ProduceCardSearchDataUtils()
        for support_card in self._data:
            support_card.upgradeProduceCardSearchCls = search_db.get_by_id(
                support_card.upgradeProduceCardSearchId
            )

    def search(self, ocr_result, match_config=None):
        return self.search_by_name(ocr_result, match_config)


def _resolve_table_schema_cls(table_name: str):
    module = importlib.import_module(f"src.entity.Game.Database.{table_name}")
    data_cls = getattr(module, table_name, None)
    if data_cls is None:
        raise AttributeError(
            f"Dataclass `{table_name}` not found in module src.entity.Game.Database.{table_name}"
        )
    loc_cls = getattr(module, f"{table_name}Localization", None)
    return data_cls, loc_cls


class GakumasDatabase_AutoDataUtils(_BaseYamlDatabase):
    """
    通用数据库加载器：用于加载尚未写专用 DataUtils 的表。
    - 自动按表名导入 dataclass
    - 自动判断 key 策略：id / id+field / 行号索引
    """

    data_cls = None
    loc_cls = None

    def __init__(self, data_file: str, table_name: str = None):
        self.table_name = table_name or os.path.splitext(os.path.basename(data_file))[0]
        self.data_cls, self.loc_cls = _resolve_table_schema_cls(self.table_name)
        self._key_fields: List[str] = []
        self._use_row_index_key = False
        super().__init__(data_file)

    def _detect_key_mode(self, objects: List[Any]):
        if not objects:
            self._use_row_index_key = True
            self._key_fields = []
            return

        annotations = getattr(self.data_cls, "__annotations__", {})
        has_id_field = "id" in annotations and all(hasattr(o, "id") for o in objects)
        if not has_id_field:
            self._use_row_index_key = True
            self._key_fields = []
            return

        ids = [getattr(o, "id", None) for o in objects]
        if len(ids) == len(set(ids)):
            self._use_row_index_key = False
            self._key_fields = ["id"]
            return

        candidate_fields = [
            "level",
            "upgradeCount",
            "customizeCount",
            "rank",
            "step",
            "phase",
            "order",
        ]
        candidate_fields.extend(
            [
                f
                for f in annotations.keys()
                if f not in ("id",) and f not in candidate_fields
            ]
        )

        for field_name in candidate_fields:
            values = [getattr(o, field_name, None) for o in objects]
            if any(isinstance(v, (list, dict, tuple, set)) for v in values):
                continue
            keys = [f"{id_val}.{val}" for id_val, val in zip(ids, values)]
            if len(keys) == len(set(keys)):
                self._use_row_index_key = False
                self._key_fields = ["id", field_name]
                return

        self._use_row_index_key = True
        self._key_fields = []

    def _compose_key(self, obj: Any, idx: int) -> str:
        if self._use_row_index_key:
            return str(idx)
        if not self._key_fields:
            return str(idx)
        parts = []
        for field_name in self._key_fields:
            parts.append(str(getattr(obj, field_name, "")))
        return ".".join(parts)

    def _load_database(self):
        entries = self._load_yaml()
        objects = self._load_objects(entries)
        self._detect_key_mode(objects)

        loc_map = {}
        if self.loc_cls:
            loc_objects = self._load_localization(self._diff_file, self.loc_cls)
            if self._use_row_index_key:
                if loc_objects and hasattr(loc_objects[0], "id"):
                    for loc in loc_objects:
                        loc_id = getattr(loc, "id", None)
                        if loc_id is not None:
                            loc_map[str(loc_id)] = loc
                else:
                    for i, loc in enumerate(loc_objects):
                        loc_map[str(i)] = loc
            else:
                for i, loc in enumerate(loc_objects):
                    loc_map[self._compose_key(loc, i)] = loc
                    loc_id = getattr(loc, "id", None)
                    if loc_id is not None:
                        loc_map.setdefault(str(loc_id), loc)

        self._data = objects
        self._map = {}
        self._raw_id_map = {}

        for i, obj in enumerate(objects):
            key = self._compose_key(obj, i)
            self._map[key] = obj
            if hasattr(obj, "id"):
                self._raw_id_map.setdefault(str(getattr(obj, "id")), []).append(obj)

            if self.loc_cls:
                obj.localization = loc_map.get(key)
                if obj.localization is None and hasattr(obj, "id"):
                    obj.localization = loc_map.get(str(getattr(obj, "id")))

        key_mode = "row_index" if self._use_row_index_key else ".".join(self._key_fields)
        logger.success(
            f"[{self.__class__.__name__}:{self.table_name}] "
            f"{len(self._data)} records loaded, key={key_mode}"
        )


_SPECIALIZED_TABLE_UTILS = {
    "Item": GakumasDatabase_ItemDataUtils,
    "Character": GakumasDatabase_CharacterDataUtils,
    "EffectGroup": GakumasDatabase_EffectGroupDataUtils,
    "ProduceExamTrigger": GakumasDatabase_ExamTriggerDataUtils,
    "ProduceExamEffect": GakumasDatabase_ExamEffectDataUtils,
    "ProduceCardGrowEffect": GakumasDatabase_GrowEffectDataUtils,
    "ProduceCardSearch": GakumasDatabase_ProduceCardSearchDataUtils,
    "ProduceCardStatusEnchant": GakumasDatabase_CardStatusEnchantDataUtils,
    "ProduceExamStatusEnchant": GakumasDatabase_ExamStatusEnchantDataUtils,
    "ProduceCardCustomize": GakumasDatabase_ProduceCardCustomizeDataUtils,
    "ProduceCard": GakumasDatabase_ProduceCardDataUtils,
    "ProduceItem": GakumasDatabase_ProduceItemDataUtils,
    "ProduceDrink": GakumasDatabase_ProduceDrinkDataUtils,
    "ProduceSkill": GakumasDatabase_ProduceSkillDataUtils,
    "IdolCard": GakumasDatabase_IdolCardDataUtils,
    "SupportCard": GakumasDatabase_SupportCardDataUtils,
}


def list_available_game_database_tables() -> List[str]:
    base_path = resolve_existing_resource_path("assets", "gakumasu-diff")
    if not base_path.exists():
        return []
    return sorted(
        [
            os.path.splitext(name)[0]
            for name in os.listdir(base_path)
            if name.endswith(".yaml")
        ]
    )


_DYNAMIC_TABLE_UTILS: Dict[str, type] = {}
_RELATION_BIND_LOCK = threading.RLock()


def _build_dynamic_table_loader_cls(table_name: str):
    class_name = f"GakumasDatabase_{table_name}DataUtils"
    if class_name in globals():
        return globals()[class_name]

    def __init__(self, data_file=None, _table_name=table_name):
        if data_file is None:
            data_file = str(resolve_existing_resource_path("assets", "gakumasu-diff", f"{table_name}.yaml"))
        GakumasDatabase_AutoDataUtils.__init__(
            self, data_file=data_file, table_name=_table_name
        )

    dynamic_cls = type(
        class_name,
        (GakumasDatabase_AutoDataUtils,),
        {
            "__doc__": f"Auto generated loader for `{table_name}` table.",
            "__init__": __init__,
            "default_data_file_parts": ("assets", "gakumasu-diff", f"{table_name}.yaml"),
        },
    )
    globals()[class_name] = dynamic_cls
    return dynamic_cls


def _register_dynamic_table_utils():
    for table_name in list_available_game_database_tables():
        if table_name in _SPECIALIZED_TABLE_UTILS:
            continue
        _DYNAMIC_TABLE_UTILS[table_name] = _build_dynamic_table_loader_cls(table_name)


def get_game_database_loader_cls(table_name: str):
    if table_name in _SPECIALIZED_TABLE_UTILS:
        return _SPECIALIZED_TABLE_UTILS[table_name]
    if table_name not in _DYNAMIC_TABLE_UTILS:
        if table_name not in list_available_game_database_tables():
            raise KeyError(f"Unknown game database table: {table_name}")
        _DYNAMIC_TABLE_UTILS[table_name] = _build_dynamic_table_loader_cls(table_name)
    return _DYNAMIC_TABLE_UTILS[table_name]


def get_game_database(table_name: str):
    loader_cls = get_game_database_loader_cls(table_name)
    return loader_cls()


def _extract_relation_field(field_name: str):
    if field_name == "id":
        return None

    match = re.match(r"^(.+?)Ids(\d+)$", field_name)
    if match:
        stem = match.group(1)
        return stem, True, f"{field_name}Clss"

    if field_name.endswith("Ids"):
        stem = field_name[:-3]
        if stem:
            return stem, True, f"{stem}Clss"

    match = re.match(r"^(.+?)Id(\d+)$", field_name)
    if match:
        stem = match.group(1)
        return stem, False, f"{field_name}Cls"

    if field_name.endswith("Id"):
        stem = field_name[:-2]
        if stem:
            return stem, False, f"{stem}Cls"

    return None


def _split_camel_tokens(name: str) -> List[str]:
    if not name:
        return []
    return re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z0-9]+", name)


def _candidate_tables_from_stem(stem: str, table_set: set[str]) -> List[str]:
    tokens = _split_camel_tokens(stem)
    if not tokens:
        return []

    candidates = []
    for i in range(len(tokens)):
        candidate = "".join(t[:1].upper() + t[1:] for t in tokens[i:])
        if candidate in table_set and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _iter_relation_values(obj: Any, field_name: str, is_list: bool) -> List[str]:
    value = getattr(obj, field_name, None)
    if value is None:
        return []
    if is_list:
        if not isinstance(value, list):
            return []
        return [v for v in value if isinstance(v, str) and v]
    if isinstance(value, str) and value:
        return [value]
    return []


def _relation_alias_populated(source_db: _BaseYamlDatabase, alias_name: str) -> bool:
    for obj in source_db.get_all_item()[:100]:
        value = getattr(obj, alias_name, None)
        if value not in (None, [], {}, ""):
            return True
    return False


def _pick_relation_target(
    source_db: _BaseYamlDatabase,
    field_name: str,
    is_list: bool,
    candidates: List[str],
    db_map: Dict[str, _BaseYamlDatabase],
):
    sample_values = []
    for obj in source_db.get_all_item():
        sample_values.extend(_iter_relation_values(obj, field_name, is_list))
        if len(sample_values) >= 400:
            break
    if not sample_values:
        return None

    best_table = None
    best_hits = 0
    best_ratio = 0.0

    for table_name in candidates:
        target_db = db_map.get(table_name)
        if target_db is None:
            continue
        hits = 0
        for value in sample_values:
            if target_db.has_raw_id(value):
                hits += 1
        if hits == 0:
            continue
        ratio = hits / len(sample_values)
        if ratio < 0.15 and hits < 20:
            continue
        if hits > best_hits or (hits == best_hits and ratio > best_ratio):
            best_hits = hits
            best_ratio = ratio
            best_table = table_name

    return best_table


def _bind_relation_field(
    source_db: _BaseYamlDatabase,
    target_db: _BaseYamlDatabase,
    field_name: str,
    alias_name: str,
    is_list: bool,
) -> int:
    bound_count = 0

    for obj in source_db.get_all_item():
        raw_values = _iter_relation_values(obj, field_name, is_list)
        if is_list:
            resolved = []
            for value in raw_values:
                matches = target_db.get_all_by_raw_id(value)
                if len(matches) == 1:
                    resolved.append(matches[0])
                elif len(matches) > 1:
                    resolved.extend(matches)
            setattr(obj, alias_name, resolved)
            if resolved:
                bound_count += 1
        else:
            resolved = target_db.get_by_raw_id(raw_values[0]) if raw_values else None
            setattr(obj, alias_name, resolved)
            if resolved is not None:
                bound_count += 1

    return bound_count


def bind_game_database_relations(
    db_map: Dict[str, _BaseYamlDatabase],
    overwrite_existing: bool = False,
):
    relation_specs = []
    table_set = set(db_map.keys())

    for source_table_name, source_db in db_map.items():
        annotations = getattr(source_db.data_cls, "__annotations__", {})
        for field_name in annotations.keys():
            relation_field = _extract_relation_field(field_name)
            if relation_field is None:
                continue

            stem, is_list, alias_name = relation_field
            if not overwrite_existing and _relation_alias_populated(source_db, alias_name):
                continue

            candidates = _candidate_tables_from_stem(stem, table_set)
            if not candidates:
                continue

            target_table_name = _pick_relation_target(
                source_db=source_db,
                field_name=field_name,
                is_list=is_list,
                candidates=candidates,
                db_map=db_map,
            )
            if not target_table_name:
                continue

            target_db = db_map[target_table_name]
            bound_count = _bind_relation_field(
                source_db=source_db,
                target_db=target_db,
                field_name=field_name,
                alias_name=alias_name,
                is_list=is_list,
            )
            if bound_count <= 0:
                continue

            relation_specs.append(
                (
                    source_table_name,
                    field_name,
                    alias_name,
                    target_table_name,
                    bound_count,
                )
            )

    logger.success(f"[Relations] bound {len(relation_specs)} field relations")
    return relation_specs


def preload_all_game_databases(bind_relations: bool = True) -> Dict[str, _BaseYamlDatabase]:
    _register_dynamic_table_utils()

    db_map = {}
    for table_name in list_available_game_database_tables():
        db_map[table_name] = get_game_database(table_name)

    if bind_relations:
        with _RELATION_BIND_LOCK:
            bind_game_database_relations(db_map)

    return db_map


def reload_loaded_game_databases():
    with _RELATION_BIND_LOCK, _SingletonByFileMeta._lock:
        old_instances = dict(_SingletonByFileMeta._instances)
        if not old_instances:
            return {}
        _SingletonByFileMeta._instances = {}
        rebuilt_old_instances = {}
        for key, old_instance in old_instances.items():
            loader_cls, data_file = key
            fresh_instance = loader_cls(data_file)
            old_instance.__dict__.clear()
            old_instance.__dict__.update(fresh_instance.__dict__)
            rebuilt_old_instances[key] = old_instance
        current_instances = dict(_SingletonByFileMeta._instances)
        current_instances.update(rebuilt_old_instances)
        _SingletonByFileMeta._instances = current_instances
        return {
            os.path.splitext(os.path.basename(db._diff_file))[0]: db
            for db in current_instances.values()
            if getattr(db, "_diff_file", None)
        }


_register_dynamic_table_utils()
