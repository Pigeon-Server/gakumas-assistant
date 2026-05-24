from peewee import *
from uuid import uuid4

from src.models.base import BaseModel

MODEL_REGISTRY = {}

class _BaseCLIPPayload(BaseModel):
    uuid = UUIDField(default=uuid4, unique=True, primary_key=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        MODEL_REGISTRY[cls.__name__] = cls   # 自动注册模型类

class CLIPayload_Item(_BaseCLIPPayload):
    id = CharField(unique=True)

class CLIPayload_SkillCard(_BaseCLIPPayload):
    id = CharField()
    level = IntegerField(default=0)

    class Meta:
        indexes = (
            (('id', 'level'), True),
        )

class CLIPayload_SupportCard(_BaseCLIPPayload):
    id = CharField(unique=True)

class CLIPayload_IdolCard(_BaseCLIPPayload):
    id = CharField(unique=True)

class CLIPayload_ProduceDrink(_BaseCLIPPayload):
    id = CharField(unique=True)

class CLIPayload_ProduceItem(_BaseCLIPPayload):
    id = CharField(unique=True)

class CLIPayload_ScheduleAction(_BaseCLIPPayload):
    """周行動アイコン CLIP ペイロード。action_id をキーとして保持。"""
    action_id = CharField(unique=True)
    param_kind = CharField(default="")
    rl_action_type = CharField(default="")

class CLIPMemory(BaseModel):
    uuid = UUIDField(default=uuid4, unique=True, primary_key=True)
    clip_name = CharField()
    _payload_model = CharField()   # 存模型类名
    _payload_id = CharField()      # 存对象的主键
    features = BlobField()

    @classmethod
    def save_vector(cls, clip_name, features: bytes, payload_obj) -> "CLIPMemory":
        payload_model_name = payload_obj.__class__.__name__
        return cls.create(
            clip_name=clip_name,
            _payload_model=payload_model_name,
            _payload_id=payload_obj.uuid,
            features=features
        )

    def load_payload(self):
        ModelClass = MODEL_REGISTRY[self._payload_model]
        return ModelClass.get_by_id(self._payload_id)

    @classmethod
    def find_by_payload(cls, payload_obj):
        model_name = payload_obj.__class__.__name__
        return cls.select().where(
            (cls._payload_model == model_name) &
            (cls._payload_id == str(payload_obj.get_id()))
        )
