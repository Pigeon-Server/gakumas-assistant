import json
import uuid

from datetime import date, datetime

from src.utils.i18n_tools import I18nText, serialize_i18n_value


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')  # 格式化 date 对象
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, I18nText):
            return obj.to_dict()
        else:
            serialized = serialize_i18n_value(obj)
            if serialized is not obj:
                return serialized
            return json.JSONEncoder.default(self, obj)
