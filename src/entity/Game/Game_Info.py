from src.entity.Game.Page.Types.index import GamePageTypes


class PlayerInfo:
    # 账户等级
    level: int = -1
    # 体力
    stamina: int = -1
    # 宝石/抽卡资源
    gem: int = -1

class GameStatusManager:
    player: PlayerInfo = PlayerInfo()
    current_location: str = GamePageTypes.UNKNOWN
    page_entity: object

    def __init__(self):
        pass