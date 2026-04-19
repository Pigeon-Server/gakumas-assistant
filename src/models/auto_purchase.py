from datetime import datetime

from peewee import AutoField, CharField, DateTimeField, IntegerField

from src.models.base import BaseModel


class AutoPurchaseExchangeRecord(BaseModel):
    id = AutoField(primary_key=True)
    item_id = CharField(index=True)
    item_name = CharField()
    page_money_before = IntegerField(null=True)
    modal_money_before = IntegerField(null=True)
    modal_money_after = IntegerField(null=True)
    money_delta = IntegerField(null=True)
    owned_before = IntegerField(null=True)
    owned_after = IntegerField(null=True)
    purchase_quantity = IntegerField(null=True)
    exchange_limit_before = IntegerField(null=True)
    exchange_limit_after = IntegerField(null=True)
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "auto_purchase_exchange_record"
