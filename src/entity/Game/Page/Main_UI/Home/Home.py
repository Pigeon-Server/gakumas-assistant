from src.entity.Game.Components.Button import Button
from src.entity.Game.Page.Main_UI.BasePage import BasePage
from src.entity.Game.Page.Types.index import GamePageTypes
from src.entity.Yolo_Results import Yolo_Results
from src.constants import *


class HomePage(BasePage):
    id: GamePageTypes.MAIN_MENU__HOME
    produce_btn: Button
    shop_btn: Button
    gift_btn: Button
    task_btn: Button
    work_btn: Button
    achievement_btn: Button
    get_expenditure_btn: Button

    def __init__(self, yolo_results: Yolo_Results):
        self.produce_btn = Button(yolo_results.get_yolo_boxs_by_label(labels.home_produce_btn, True), True)
        self.shop_btn = Button(yolo_results.get_yolo_boxs_by_label(labels.home_shop_btn, True), True)
        self.gift_btn = Button(yolo_results.get_yolo_boxs_by_label(labels.home_gift_btn, True), True)
        self.task_btn = Button(yolo_results.get_yolo_boxs_by_label(labels.home_daily_task, True), True)
        self.work_btn = Button(yolo_results.get_yolo_boxs_by_label(labels.home_dispatch_work, True), True)
        self.achievement_btn = Button(yolo_results.get_yolo_boxs_by_label(labels.home_achievement, True), True)
        self.get_expenditure_btn = Button(yolo_results.get_yolo_boxs_by_label(labels.home_get_expenditure, True), True)
        super().__init__()