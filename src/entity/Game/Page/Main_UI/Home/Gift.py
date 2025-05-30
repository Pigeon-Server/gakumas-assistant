from dataclasses import dataclass

from src.entity.Game.Components.Button import Button
from src.entity.Game.Components.TabBar import TabBar
from src.entity.Game.Page.Main_UI.BasePage import BasePage
from src.entity.Game.Page.Types.index import GamePageTypes
from src.constants import *
from src.entity.Yolo_Results import Yolo_Results


@dataclass
class GiftPage(BasePage):
    id: GamePageTypes.HOME_TAB.GIFT
    tabs: TabBar
    claim_all: Button


    def __init__(self, yolo_results: Yolo_Results):
        BasePage.__init__(self)
        self.tabs = TabBar(yolo_results.get_yolo_boxs_by_label(labels.tab_bar, True))
        self.claim_all = Button()