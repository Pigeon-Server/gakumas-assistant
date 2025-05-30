from src.entity.Game.Page.Types.Tabs.Contest import Contest
from src.entity.Game.Page.Types.Tabs.Gacha import Gacha
from src.entity.Game.Page.Types.Tabs.Home import HomeTab
from src.entity.Game.Page.Types.Tabs.Idol import IdolTab
from src.entity.Game.Page.Types.Tabs.Communicate import Communicate


class GamePageTypes:
    START_GAME = "START_GAME"
    LOADING = "LOADING"
    DOWNLOADING = "DOWNLOADING"
    UNKNOWN = "UNKNOWN"
    #
    MAIN_MENU__GACHA = "MAIN_MENU__GACHA"
    MAIN_MENU__CONTEST = "MAIN_MENU__CONTEST"
    MAIN_MENU__HOME = "MAIN_MENU__HOME"
    MAIN_MENU__IDOL = "MAIN_MENU__IDOL"
    MAIN_MENU__COMMUNICATE = "MAIN_MENU__COMMUNICATE"
    #
    GACHA = Gacha
    CONTEST = Contest
    HOME_TAB = HomeTab
    IDOL_TAB = IdolTab
    Communicate = Communicate


