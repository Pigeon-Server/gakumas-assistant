from dataclasses import dataclass
from pathlib import Path


GAKUMAS_DIFF_FILES = (
    "Item.yaml",
    "Character.yaml",
    "IdolCard.yaml",
    "SupportCard.yaml",
    "SupportCardLevel.yaml",
    "SupportCardLevelLimit.yaml",
    "SupportCardProduceSkillFilter.yaml",
    "SupportCardProduceSkillLevelAssist.yaml",
    "SupportCardProduceSkillLevelDance.yaml",
    "SupportCardProduceSkillLevelVisual.yaml",
    "SupportCardProduceSkillLevelVocal.yaml",
    "ProduceCard.yaml",
    "ProduceCardCustomize.yaml",
    "ProduceCardSearch.yaml",
    "ProduceCardStatusEnchant.yaml",
    "ProduceEventSupportCard.yaml",
    "ProduceStepEventDetail.yaml",
    "EffectGroup.yaml",
    "ProduceExamEffect.yaml",
    "ProduceExamTrigger.yaml",
    "ProduceCardGrowEffect.yaml",
    "ProduceExamStatusEnchant.yaml",
    "ProduceItem.yaml",
    "ProduceDrink.yaml",
    "ProduceSkill.yaml",
    "ProduceSetting.yaml",
    "Setting.yaml",
)

TRANSLATION_SOURCE_FILES = (
    "Item.yaml",
    "Character.yaml",
    "IdolCard.yaml",
    "SupportCard.yaml",
    "ProduceCard.yaml",
    "ProduceCardCustomize.yaml",
    "ProduceCardSearch.yaml",
    "ProduceCardStatusEnchant.yaml",
    "ProduceStepEventDetail.yaml",
    "EffectGroup.yaml",
    "ProduceExamEffect.yaml",
    "ProduceExamTrigger.yaml",
    "ProduceExamStatusEnchant.yaml",
    "ProduceItem.yaml",
    "ProduceDrink.yaml",
    "ProduceSkill.yaml",
)

TRANSLATION_FILES = tuple(Path(name).with_suffix(".json").name for name in TRANSLATION_SOURCE_FILES)


@dataclass(frozen=True)
class ResourceRepositoryDefinition:
    name: str
    path: str
    owner: str
    repo: str
    default_url: str = ""
    config_key: str = ""
    required_subdir: str = ""
    required_files: tuple[str, ...] = ()

    def iter_required_relative_paths(self) -> tuple[str, ...]:
        if not self.required_files:
            return ()
        prefix = Path(self.required_subdir) if self.required_subdir else Path()
        return tuple(str(prefix / file_name).replace("\\", "/") for file_name in self.required_files)


RESOURCE_REPOSITORIES = (
    ResourceRepositoryDefinition(
        name="gakumasu-diff",
        path="assets/gakumasu-diff",
        owner="vertesan",
        repo="gakumasu-diff",
        default_url="https://github.com/vertesan/gakumasu-diff.git",
        config_key="base.gakumasu_diff_repository_url",
        required_files=GAKUMAS_DIFF_FILES,
    ),
    ResourceRepositoryDefinition(
        name="GakumasTranslationData",
        path="assets/GakumasTranslationData",
        owner="chinosk6",
        repo="GakumasTranslationData",
        default_url="https://github.com/chinosk6/GakumasTranslationData.git",
        config_key="base.gakumas_translation_data_repository_url",
        required_subdir="local-files/masterTrans",
        required_files=TRANSLATION_FILES,
    ),
)
