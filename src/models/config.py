import ast
from datetime import datetime

from peewee import AutoField, CharField, BooleanField, DateTimeField

from src.utils.logger import logger
from src.models.base import BaseModel
from src.entity.Config import (
    Config as ConfigEntity,
    ConfigItem,
    apply_legacy_producer_decision_backend,
)


class ConfigModel(BaseModel):
    id = AutoField(primary_key=True)
    key = CharField(unique=True)
    value = CharField(default="")
    verify = CharField(unique=False, default=None)
    use_verify = BooleanField(default=False)
    last_modified_time = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "config"

    @staticmethod
    def _verify_value(config_item: ConfigItem) -> str:
        return config_item.verify or ""

    @classmethod
    def load_config(cls) -> ConfigEntity:
        config = ConfigEntity()
        for row in cls.select():
            keys = row.key.split(".")
            if len(keys) != 2: # 跳过非法 key
                logger.warning(f"Skip load config: {row}")
                continue

            section_name, item_name = keys

            if section_name == "base" and item_name == "producer_decision_backend":
                apply_legacy_producer_decision_backend(config, row.value)
                continue

            section = getattr(config, section_name, None)
            if section is None:
                logger.warning(f"Config entity not category: {section_name}")
                continue

            config_item: ConfigItem = getattr(section, item_name, None)
            if not isinstance(config_item, ConfigItem):
                continue
            try:
                cast_value = cls.cast_to_type(row.value, config_item.data_type)
            except Exception as e:
                logger.warning(f"Failed to cast value: {row.value}\n{e}")
                cast_value = config_item.default_value

            config_item.value = cast_value
            config_item.last_modified_time = row.last_modified_time

        return config

    @classmethod
    def save_config(cls, config_entity: ConfigEntity = ConfigEntity):
        for section_name in dir(config_entity):
            if section_name.startswith("_"):
                continue

            section = getattr(config_entity, section_name)
            for attr_name in dir(section):
                if attr_name.startswith("__"):
                    continue

                config_item = getattr(section, attr_name)
                if isinstance(config_item, ConfigItem):
                    key = f"{section_name}.{attr_name}"
                    if config_item.data_type == list:
                        config_item.value = list(set(config_item.value))
                    value_str = str(config_item.value)

                    config_row, created = cls.get_or_create(key=key, defaults={
                        'value': value_str,
                        'verify': cls._verify_value(config_item),
                        'use_verify': config_item.use_verify or False,
                        'last_modified_time': datetime.now()
                    })
                    verify_value = cls._verify_value(config_item)
                    metadata_changed = (
                        config_row.verify != verify_value
                        or config_row.use_verify != bool(config_item.use_verify)
                    )
                    if not created and (config_row.value != value_str or metadata_changed):
                        config_row.value = value_str
                        config_row.verify = verify_value
                        config_row.use_verify = config_item.use_verify
                        config_row.last_modified_time = datetime.now()
                        config_row.save()

    @classmethod
    def update_database(cls):
        config = ConfigEntity()
        for section_name in dir(config):
            if section_name.startswith("__"):
                continue

            section = getattr(config, section_name)
            for attr_name in dir(section):
                if attr_name.startswith("__"):
                    continue
                config_item = getattr(section, attr_name)
                if not isinstance(config_item, ConfigItem):
                    continue
                key = f"{section_name}.{attr_name}"
                verify_value = cls._verify_value(config_item)
                row = cls.get_or_none(cls.key == key)
                if row is None:
                    if not cls.create(
                        key=key,
                        value=config_item.default_value,
                        verify=verify_value,
                        use_verify=config_item.use_verify,
                        last_modified_time=datetime.now()
                    ):
                        logger.warning(f"Failed to create config key: {key}")
                        return
                    logger.info(f"Created config key: {key}")
                    continue

                if row.verify != verify_value or row.use_verify != bool(config_item.use_verify):
                    row.verify = verify_value
                    row.use_verify = config_item.use_verify
                    row.save()
                    logger.info(f"Updated config schema metadata: {key}")

    @staticmethod
    def cast_to_type(value: str, target_type: type):
        value = str(value)
        if target_type == bool:
            return value.lower() in ("true", "1", "yes", "on", "True")
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type in [dict, list, tuple]:
            try:
                return ast.literal_eval(value)
            except ValueError as e:
                logger.warning(f"Failed to cast value: {value}\n{e}")
                return []
        elif target_type == str:
            return value
        else:
            raise ValueError(f"Unsupported cast type: {target_type}")


    @classmethod
    def init_config(cls):
        if len(cls().select()) >= 0:
            cls.save_config(ConfigEntity())
            return
