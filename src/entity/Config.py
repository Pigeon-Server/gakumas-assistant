import copy
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Any, Tuple, Dict

from src.constants.device.adb import ADBOperation, ADBConnectMode
from src.constants.ocr.backend import OCR_BACKEND_VERIFY, OCRBackendType
from src.utils.logger import logger


def _config_enum_values(enum_cls) -> List[str]:
    return [
        value for key, value in enum_cls.__dict__.items()
        if not key.startswith("__") and isinstance(value, str)
    ]


def _default_run_mode() -> str:
    if platform.system() == "Windows":
        return "PC"
    if platform.system() == "Darwin":
        return "MacPlayTools"
    return "Phone"


def _run_mode_options() -> List[Dict[str, Any]]:
    pc_option: Dict[str, Any] = {
        "title": "电脑端（DMM）",
        "value": "PC",
    }
    try:
        from src.core.device.windows_compat import (
            get_windows_unavailability_reason,
            windows_pc_mode_is_available,
        )

        if not windows_pc_mode_is_available():
            pc_option["disabled"] = True
            pc_option["disabled_reason"] = get_windows_unavailability_reason()
    except Exception:
        if platform.system() != "Windows":
            pc_option["disabled"] = True
            pc_option["disabled_reason"] = "PC / DMM 模式仅支持 Windows。"

    mac_option: Dict[str, Any] = {
        "title": "macOS PlayCover",
        "value": "MacPlayTools",
    }
    if platform.system() != "Darwin":
        mac_option["disabled"] = True
        mac_option["disabled_reason"] = "MacPlayTools 模式仅支持 macOS (Apple Silicon)。"

    return [
        pc_option,
        {"title": "手机端", "value": "Phone"},
        mac_option,
    ]


def _ocr_backend_options() -> List[Dict[str, Any]]:
    vision_option: Dict[str, Any] = {
        "title": "Vision（macOS 原生 OCR）",
        "value": OCRBackendType.VISION,
    }
    if platform.system() != "Darwin":
        vision_option["disabled"] = True
        vision_option["disabled_reason"] = "Vision OCR 仅在 macOS 可用。"
    return [
        {"title": "自动", "value": OCRBackendType.AUTO},
        {"title": "RapidOCR", "value": OCRBackendType.RAPIDOCR},
        vision_option,
    ]


@dataclass
class ConfigItemUI:
    label: Optional[str] = None
    hint: Optional[str] = None
    component: Optional[str] = None
    options: List[Dict[str, Any]] = field(default_factory=list)
    visible_if: Optional[Dict[str, Any]] = None
    readonly: bool = False
    resettable: bool = False
    auto_generate: bool = True
    order: int = 0

    def to_json_dict(self) -> dict:
        return {
            "label": self.label,
            "hint": self.hint,
            "component": self.component,
            "options": self.options,
            "visible_if": self.visible_if,
            "readonly": self.readonly,
            "resettable": self.resettable,
            "auto_generate": self.auto_generate,
            "order": self.order,
        }


@dataclass
class ConfigItem:
    """
    配置项
    """
    # 默认值
    default_value: Any
    # 数据类型
    data_type: type = str
    # 验证规则（正则表达式）
    verify: Optional[str] = None
    # 是否启用验证
    use_verify: bool = False
    # 配置项实际值
    value: Any = None
    # 最后修改时间
    last_modified_time: datetime = None
    # 前端展示元数据
    ui: ConfigItemUI = field(default_factory=ConfigItemUI)

    def __post_init__(self):
        if self.value is None:
            self.value = copy.deepcopy(self.default_value)

    def set(self, value: Any, touch: bool = True):
        self.value = value
        if touch:
            self.last_modified_time = datetime.now()
        return self.value

    def reset(self, touch: bool = True):
        return self.set(copy.deepcopy(self.default_value), touch=touch)

    def unwrap(self):
        return self.value

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    def __bool__(self):
        return bool(self.value)

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __contains__(self, item):
        return item in self.value

    def __getitem__(self, item):
        return self.value[item]

    def __getattr__(self, item):
        return getattr(self.value, item)

    def __eq__(self, other):
        if isinstance(other, ConfigItem):
            other = other.value
        return self.value == other

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)


@dataclass
class ConfigVerifyError:
    section: str
    field: str
    message: str

    def __str__(self):
        return f"[{self.section}.{self.field}]: {self.message}"


class _BaseConfigGroup:
    def __init__(self):
        # 遍历类定义的所有 ConfigItem
        for name, attr in vars(type(self)).items():
            if isinstance(attr, ConfigItem):
                # 深拷贝，生成实例独立副本
                setattr(self, name, copy.deepcopy(attr))
            elif isinstance(attr, _BaseConfigGroup):
                # 嵌套结构的情况（递归初始化）
                setattr(self, name, attr.__class__())

    def __str__(self):
        return self._to_str()

    def _to_str(self, indent=0):
        lines = []
        prefix = " " * indent
        for name, attr in vars(self).items():  # 只遍历实例属性
            if isinstance(attr, ConfigItem):
                lines.append(f"{prefix}{name}: {attr.value!r}")
            elif isinstance(attr, _BaseConfigGroup):
                lines.append(f"{prefix}{name}:")
                lines.append(attr._to_str(indent + 4))
        return "\n".join(lines)

class _Base(_BaseConfigGroup):
    """脚本基本配置"""

    # 脚本运行模式
    run_mode = ConfigItem(
        default_value=_default_run_mode(),
        data_type=str,
        verify=r"Phone|PC|MacPlayTools",
        use_verify=True,
        ui=ConfigItemUI(
            label="运行模式",
            hint="脚本的执行模式（需重启生效）",
            component="select",
            options=_run_mode_options(),
            order=10,
        )
    )
    ocr_backend = ConfigItem(
        default_value=OCRBackendType.AUTO,
        data_type=str,
        verify=OCR_BACKEND_VERIFY,
        use_verify=True,
        ui=ConfigItemUI(
            label="OCR 后端",
            hint="auto：macOS 优先 Vision，其他平台使用 RapidOCR；失败时会回退到 RapidOCR（修改后需重启生效）",
            component="select",
            options=_ocr_backend_options(),
            order=15,
        ),
    )
    # 游戏窗口名
    game_window_name = ConfigItem(
        default_value="gakumas",
        data_type=str,
        ui=ConfigItemUI(
            label="游戏窗口名",
            hint="默认：gakumas（修改后需重启生效）",
            resettable=True,
            visible_if={"base.run_mode": "PC"},
            order=100,
        )
    )
    # 自动启动游戏
    auto_start_game = ConfigItem(
        default_value=False,
        data_type=bool,
        ui=ConfigItemUI(
            label="自动启动游戏",
            hint="当游戏未启动时是否自动启动游戏",
            component="switch",
            order=20,
        )
    )
    # adb连接模式
    adb_connect_mode = ConfigItem(
        default_value=ADBConnectMode.NETWORK,
        data_type=str,
        verify="|".join(v for k, v in ADBConnectMode.__dict__.items() if not k.startswith("__") and not callable(v)),
        use_verify=True,
        ui=ConfigItemUI(
            label="ADB连接模式",
            hint="安卓调试桥的连接模式，手机建议使用USB，模拟器可使用网络连接（修改后需重启生效）",
            component="select",
            options=[
                {"title": "网络连接", "value": "Network"},
                {"title": "USB连接", "value": "USB"},
            ],
            visible_if={"base.run_mode": "Phone"},
            order=30,
        )
    )
    # adb地址
    adb_host = ConfigItem(
        default_value="127.0.0.1",
        data_type=str,
        ui=ConfigItemUI(
            label="ADB主机名",
            hint="安卓调试桥的ip地址，模拟器一般是127.0.0.1",
            resettable=True,
            visible_if={"base.run_mode": "Phone", "base.adb_connect_mode": "Network"},
            order=40,
        )
    )
    # adb端口(Network)
    adb_port = ConfigItem(
        default_value="5555",
        data_type=int,
        ui=ConfigItemUI(
            label="ADB端口",
            hint="安卓调试桥的端口，默认5555，Android11以上为系统随机",
            component="number",
            resettable=True,
            visible_if={"base.run_mode": "Phone", "base.adb_connect_mode": "Network"},
            order=50,
        )
    )
    # adb端口(USB)
    adb_serial = ConfigItem(
        default_value="",
        data_type=str,
        ui=ConfigItemUI(
            label="通过USB连接的ADB设备",
            hint="请选择通过USB连接的设备，如未找到设备请尝试刷新列表",
            component="adb_devices",
            visible_if={"base.run_mode": "Phone", "base.adb_connect_mode": "USB"},
            order=60,
        )
    )
    # Android截图服务
    android_screen_capture_service = ConfigItem(
        default_value=ADBOperation.ScreenCaptureService.ADB,
        data_type=str,
        verify="|".join(_config_enum_values(ADBOperation.ScreenCaptureService)),
        use_verify=True,
        ui=ConfigItemUI(
            label="ADB截图方式",
            hint="scrcpy / DroidCast 的延迟通常优于 ADB 截图；scrcpy 需要将官方 Releases 的 scrcpy-server 放到 bin",
            component="select",
            options=[
                {"title": "scrcpy", "value": "scrcpy"},
                {"title": "DroidCast", "value": "DroidCast"},
                {"title": "ADB", "value": "ADB"},
            ],
            visible_if={"base.run_mode": "Phone"},
            order=70,
        )
    )
    # Android点击服务
    android_touch_service = ConfigItem(
        default_value=ADBOperation.TouchService.ADB,
        data_type=str,
        verify="|".join(_config_enum_values(ADBOperation.TouchService)),
        use_verify=True,
        ui=ConfigItemUI(
            label="ADB点击屏幕方式",
            hint="可选 MaaTouch / minitouch / scrcpy；MaaTouch 需放入官方构建产物到 bin/maatouch 或用 workflow 生成，minitouch 需放入官方构建产物到 bin/minitouch 且当前只支持 Android 9 及以下",
            component="select",
            options=[
                {"title": "MaaTouch", "value": "maatouch"},
                {"title": "minitouch", "value": "minitouch"},
                {"title": "scrcpy", "value": "scrcpy"},
                {"title": "ADB", "value": "ADB"},
            ],
            visible_if={"base.run_mode": "Phone"},
            order=80,
        )
    )
    # 游戏APP名
    game_package_name = ConfigItem(
        default_value="com.bandainamcoent.idolmaster_gakuen",
        data_type=str,
        ui=ConfigItemUI(
            label="游戏包名",
            hint="默认：com.bandainamcoent.idolmaster_gakuen（修改后需重启生效）",
            resettable=True,
            visible_if={"base.run_mode": "Phone"},
            order=90,
        )
    )
    # MacPlayTools 端口
    playtools_port = ConfigItem(
        default_value=0,
        data_type=int,
        ui=ConfigItemUI(
            label="PlayTools 端口",
            hint="PlayCover 游戏窗口标题栏 [localhost:端口号] 中的端口号（修改后需重启生效）",
            component="number",
            resettable=True,
            visible_if={"base.run_mode": "MacPlayTools"},
            order=95,
        )
    )
    # 禁用任务列表
    disabled_tasks = ConfigItem(
        default_value=[],
        data_type=list,
        ui=ConfigItemUI(
            label="禁用任务列表",
            hint="配置禁用任务列表",
            component="disabled_tasks",
            order=110,
        )
    )
    # 是否启用自动运行
    enabled_auto_startup = ConfigItem(
        default_value=False,
        data_type=bool,
        ui=ConfigItemUI(
            label="每日自动执行脚本",
            hint="开启后会在设定时间自动开始执行任务队列",
            component="switch",
            order=120,
        )
    )
    # 自动运行触发时间
    auto_startup_time = ConfigItem(
        default_value="12:00",
        data_type=str,
        verify=r"(?:[01]\d|2[0-3]):[0-5]\d",
        use_verify=True,
        ui=ConfigItemUI(
            label="自动运行触发时间",
            hint="24小时制，格式为 HH:MM",
            component="time",
            visible_if={"base.enabled_auto_startup": True},
            order=130,
        )
    )
    # 是否启用资源仓库更新检查
    enabled_check_resource_updates = ConfigItem(
        default_value=True,
        data_type=bool,
        ui=ConfigItemUI(
            label="定时检查资源仓库更新",
            hint="按设定周期检查 assets/GakumasTranslationData 与 assets/gakumasu-diff 是否有上游更新",
            component="switch",
            order=135,
        )
    )
    # 启动时检查资源仓库更新
    check_resource_updates_on_startup = ConfigItem(
        default_value=True,
        data_type=bool,
        ui=ConfigItemUI(
            label="启动时检查资源仓库更新",
            hint="每次启动后立即检查一次资源仓库更新",
            component="switch",
            order=136,
        )
    )
    # 资源仓库定时检查周期
    resource_update_check_period = ConfigItem(
        default_value="daily",
        data_type=str,
        verify=r"daily|every_3_days|weekly",
        use_verify=True,
        ui=ConfigItemUI(
            label="资源仓库检查周期",
            hint="仅用于定时检查",
            component="select",
            options=[
                {"title": "每天", "value": "daily"},
                {"title": "每 3 天", "value": "every_3_days"},
                {"title": "每周", "value": "weekly"},
            ],
            visible_if={"base.enabled_check_resource_updates": True},
            order=137,
        )
    )
    gakumasu_diff_repository_url = ConfigItem(
        default_value="https://github.com/vertesan/gakumasu-diff.git",
        data_type=str,
        verify=r"(?:(?:https?|ssh|git)://\S+|git@[^:]+:\S+)",
        use_verify=True,
        ui=ConfigItemUI(
            label="gakumasu-diff 仓库 URL",
            hint="资源下载与更新使用的仓库地址。修改后会立即重新检查更新状态。（如果你不知道这是什么，请不要更改）",
            resettable=True,
            order=138,
        )
    )
    gakumas_translation_data_repository_url = ConfigItem(
        default_value="https://github.com/chinosk6/GakumasTranslationData.git",
        data_type=str,
        verify=r"(?:(?:https?|ssh|git)://\S+|git@[^:]+:\S+)",
        use_verify=True,
        ui=ConfigItemUI(
            label="GakumasTranslationData 仓库 URL",
            hint="资源下载与更新使用的仓库地址。修改后会立即重新检查更新状态。（如果你不知道这是什么，请不要更改）",
            resettable=True,
            order=139,
        )
    )
    # 是否启用游戏资源下载（通过官方服务器获取游戏资源文件）
    enable_game_asset_download = ConfigItem(
        default_value=False,
        data_type=bool,
        ui=ConfigItemUI(
            label="是否启用游戏资源下载",
            hint="使用GkmasObjectManager从游戏服务器下载游戏资源文件。需要有互联网连接。",
            component="switch",
            order=140,
        )
    )
    prefer_game_asset_image = ConfigItem(
        default_value=False,
        data_type=bool,
        ui=ConfigItemUI(
            label="在 UI 中始终使用游戏资源显示",
            hint="启用后，UI 中的物品/支援卡/技能卡图片将始终优先使用从游戏服务器下载的资源图片，而非游戏过程中截图识别的图像。",
            component="switch",
            order=141,
        )
    )

    # --- 自动培育决策后端配置 ---
    producer_decision_backend = ConfigItem(
        default_value="llm",
        data_type=str,
        verify=r"llm|rl_battle",
        use_verify=True,
        ui=ConfigItemUI(
            label="自动培育决策后端",
            hint="llm：所有阶段走大模型；rl_battle：仅 lesson / exam 走 RL 推理服务",
            component="select",
            options=[
                {"title": "LLM（默认）", "value": "llm"},
                {"title": "RL Battle（lesson / exam）", "value": "rl_battle"},
            ],
            order=290,
        ),
    )
    rl_inference_base_url = ConfigItem(
        default_value="http://127.0.0.1:8001",
        data_type=str,
        ui=ConfigItemUI(
            label="RL 推理服务地址",
            hint="无状态 RL 推理服务地址；模型在服务启动时固定加载",
            visible_if={"base.producer_decision_backend": "rl_battle"},
            order=295,
        ),
    )
    rl_inference_timeout = ConfigItem(
        default_value=10.0,
        data_type=float,
        ui=ConfigItemUI(
            label="RL 推理超时(秒)",
            hint="lesson / exam 请求 RL 服务的超时时间",
            visible_if={"base.producer_decision_backend": "rl_battle"},
            order=296,
        ),
    )

    # --- LLM 决策配置 ---
    llm_base_url = ConfigItem(
        default_value="http://127.0.0.1:11434/v1/",
        data_type=str,
        ui=ConfigItemUI(
            label="LLM API 地址",
            hint="OpenAI 兼容 API 端点（Ollama / vLLM / OpenAI 等）",
            order=300,
        ),
    )
    llm_model = ConfigItem(
        default_value="gpt-oss:20b",
        data_type=str,
        ui=ConfigItemUI(
            label="LLM 模型",
            hint="模型名称（如 gpt-oss:20b, qwen3:4b 等）",
            order=310,
        ),
    )
    llm_api_key = ConfigItem(
        default_value="ollama",
        data_type=str,
        ui=ConfigItemUI(
            label="LLM API Key",
            hint="API 密钥（Ollama 默认 ollama）",
            order=320,
        ),
    )
    llm_timeout = ConfigItem(
        default_value=60.0,
        data_type=float,
        ui=ConfigItemUI(
            label="LLM 超时(秒)",
            hint="API 请求超时时间",
            order=330,
        ),
    )
    llm_max_tokens = ConfigItem(
        default_value=4096,
        data_type=int,
        ui=ConfigItemUI(
            label="LLM 最大输出 Token",
            hint="输出 token 上限（包含思考+回答）",
            order=340,
        ),
    )
    llm_num_ctx = ConfigItem(
        default_value=8192,
        data_type=int,
        ui=ConfigItemUI(
            label="LLM 上下文窗口",
            hint="上下文窗口大小（Ollama 专用，影响显存占用）",
            order=350,
        ),
    )
    llm_temperature = ConfigItem(
        default_value=0.3,
        data_type=float,
        ui=ConfigItemUI(
            label="LLM 温度",
            hint="生成温度，越低越确定（0.0 ~ 1.0）",
            order=360,
        ),
    )



class _Task:
    """任务配置"""

    class DispatchWork(_BaseConfigGroup):
        # 每次重新配置工作时间
        reconfigure_work_hours = ConfigItem(default_value=True, data_type=bool)
        # 工作时间
        working_hours = ConfigItem(default_value="12H", data_type=str, verify=r"4H|8H|12H", use_verify=True)

    class AutoPurchase(_BaseConfigGroup):
        # 是否购买每周礼包
        weekly_gift = ConfigItem(default_value=True, data_type=bool)
        # 每日购买的物品
        daily_buy_list = ConfigItem(default_value=[], data_type=list)
        # 自动刷新可购买列表（免费）
        refresh_shop = ConfigItem(default_value=True, data_type=bool)
        # 使用石头刷新列表
        use_gem_refresh = ConfigItem(default_value=False, data_type=bool)

    class AutoContest(_BaseConfigGroup):
        # 挑战前自动重新配置队伍
        auto_reconfigure_team_before_challenge = ConfigItem(default_value=False, data_type=bool)
        # 挑战顺序
        challenge_order = ConfigItem(default_value="random", data_type=str, verify=r"random|highest_power|lowest_power|balanced_power", use_verify=True)

    class AutoEnhancementSupportCard(_BaseConfigGroup):
        # 是否强化 R 卡
        enhance_r = ConfigItem(default_value=False, data_type=bool)
        # R 卡最大强化等级（4★上限=40）
        enhance_r_max_level = ConfigItem(default_value=40, data_type=int)
        # 是否强化 SR 卡
        enhance_sr = ConfigItem(default_value=True, data_type=bool)
        # SR 卡最大强化等级（4★上限=50）
        enhance_sr_max_level = ConfigItem(default_value=50, data_type=int)
        # 是否强化 SSR 卡
        enhance_ssr = ConfigItem(default_value=True, data_type=bool)
        # SSR 卡最大强化等级（4★上限=60）
        enhance_ssr_max_level = ConfigItem(default_value=60, data_type=int)
        # 是否自动执行上限解放（需要同名卡作为素材）
        auto_limit_break = ConfigItem(default_value=False, data_type=bool)
        # 是否自动执行サポート変換（将多余卡片变换为サポートの証）
        auto_convert = ConfigItem(default_value=False, data_type=bool)
        # 白名单模式（仅强化白名单内的卡）
        whitelist_mode = ConfigItem(default_value=False, data_type=bool)
        # 白名单卡 ID 列表
        whitelist_card_ids = ConfigItem(default_value=[], data_type=list)

    class AutoProducer(_BaseConfigGroup):
        # 剧本选择: "hajime" (初) / "nia" (NIA)
        scenario = ConfigItem(
            default_value="hajime",
            data_type=str,
            verify=r"hajime|nia",
            use_verify=True,
            ui=ConfigItemUI(
                label="剧本",
                hint="选择培育剧本",
                component="select",
                options=[
                    {"title": "初（HAJIME）", "value": "hajime"},
                    {"title": "NIA", "value": "nia"},
                ],
                order=10,
            ),
        )
        # HAJIME 难度: "regular" / "pro" / "master" / "legend"
        difficulty = ConfigItem(
            default_value="regular",
            data_type=str,
            verify=r"regular|pro|master|legend",
            use_verify=True,
            ui=ConfigItemUI(
                label="难度",
                hint="选择培育难度",
                component="select",
                options=[
                    {"title": "Regular（produce-001）", "value": "regular"},
                    {"title": "Pro（produce-002）", "value": "pro"},
                    {"title": "Master（produce-003）", "value": "master"},
                    {"title": "Legend（produce-006）", "value": "legend"},
                ],
                visible_if={"task__auto_producer.scenario": "hajime"},
                order=20,
            ),
        )
        # NIA 难度: "pro" / "master"
        nia_difficulty = ConfigItem(
            default_value="pro",
            data_type=str,
            verify=r"pro|master",
            use_verify=True,
            ui=ConfigItemUI(
                label="NIA 难度",
                hint="选择 NIA 剧本难度",
                component="select",
                options=[
                    {"title": "Pro", "value": "pro"},
                    {"title": "Master", "value": "master"},
                ],
                visible_if={"task__auto_producer.scenario": "nia"},
                order=21,
            ),
        )
        # 目标偶像卡 ID（通过 CLIP 匹配，留空使用默认选中的卡）
        target_idol_card_id = ConfigItem(
            default_value="",
            data_type=str,
            ui=ConfigItemUI(
                label="目标偶像卡",
                hint="目标 P アイドル ID（留空使用默认选中的卡；需先执行「刷新偶像卡存储」学习卡片特征）",
                order=30,
            ),
        )
        # 支援卡编成模式: "auto" (おまかせ) / "preset" (预设编号)
        support_card_mode = ConfigItem(
            default_value="auto",
            data_type=str,
            verify=r"auto|preset",
            use_verify=True,
            ui=ConfigItemUI(
                label="支援卡编成",
                hint="自动编成或使用预设编号",
                component="select",
                options=[
                    {"title": "自动编成", "value": "auto"},
                    {"title": "预设编号", "value": "preset"},
                ],
                order=40,
            ),
        )
        # 预设支援卡编号
        support_card_preset_index = ConfigItem(
            default_value=1,
            data_type=int,
            ui=ConfigItemUI(
                label="支援卡预设编号",
                hint="使用第几组预设编成",
                visible_if={"task__auto_producer.support_card_mode": "preset"},
                order=50,
            ),
        )
        # 记忆编成模式
        memory_mode = ConfigItem(
            default_value="auto",
            data_type=str,
            verify=r"auto|preset",
            use_verify=True,
            ui=ConfigItemUI(
                label="记忆编成",
                hint="自动编成或使用预设编号",
                component="select",
                options=[
                    {"title": "自动编成", "value": "auto"},
                    {"title": "预设编号", "value": "preset"},
                ],
                order=60,
            ),
        )
        # 预设记忆编号
        memory_preset_index = ConfigItem(
            default_value=1,
            data_type=int,
            ui=ConfigItemUI(
                label="记忆预设编号",
                hint="使用第几组预设编成",
                visible_if={"task__auto_producer.memory_mode": "preset"},
                order=70,
            ),
        )
        # 自动编排记忆时「レンタルを使用」复选框
        use_rental = ConfigItem(
            default_value=True,
            data_type=bool,
            ui=ConfigItemUI(
                label="使用租赁记忆",
                hint="自动编排记忆时勾选「レンタルを使用」复选框",
                component="switch",
                order=75,
            ),
        )
        # 開始確認页是否使用加成道具
        use_boost_items = ConfigItem(
            default_value=False,
            data_type=bool,
            ui=ConfigItemUI(
                label="使用加成道具",
                hint="開始確認页面是否使用加成道具（編成詳细按钮上方）",
                component="switch",
                order=80,
            ),
        )
        # 是否恢复上次中断的培育（检测到「プロデュース再開」弹窗时点击「再開する」）
        resume_interrupted = ConfigItem(
            default_value=True,
            data_type=bool,
            ui=ConfigItemUI(
                label="恢复中断培育",
                hint="检测到上次中断的培育时自动恢复（点击「再開する」），而非放弃重新开始",
                component="switch",
                order=82,
            ),
        )
        # 行程决策阶段 P手帳 读取策略
        schedule_notebook_mode = ConfigItem(
            default_value="before_decision",
            data_type=str,
            verify=r"disabled|before_decision",
            use_verify=True,
            ui=ConfigItemUI(
                label="P手帐读取策略",
                hint="disabled：不读取；before_decision：仅在周行动自动决策前读取（决策后不再触发）",
                component="select",
                options=[
                    {"title": "关闭读取", "value": "disabled"},
                    {"title": "仅决策前读取", "value": "before_decision"},
                ],
                order=83,
            ),
        )
        # 记忆卡面（フォト）选择模式
        memory_photo_mode = ConfigItem(
            default_value="first",
            data_type=str,
            verify=r"first|vl",
            use_verify=True,
            ui=ConfigItemUI(
                label="记忆卡面选择",
                hint="培育结束后选择记忆卡面的方式",
                component="select",
                options=[
                    {"title": "默认选择第一个", "value": "first"},
                    {"title": "VL 视觉模型自动选择最优卡面", "value": "vl"},
                ],
                order=85,
            ),
        )
        # VL 模型自定义提示词
        memory_photo_vl_prompt = ConfigItem(
            default_value=(
                "以下图片是游戏中可选的记忆卡照片缩略图列表。"
                "请选出构图最好、角色表情最生动、最具观赏性的一张照片。"
                "只需要返回你选择的照片编号（从1开始），不要返回其他内容。"
            ),
            data_type=str,
            ui=ConfigItemUI(
                label="VL 选卡面提示词",
                hint="自定义 VL 模型选择卡面时使用的提示词（留空使用默认）",
                visible_if={"task__auto_producer.memory_photo_mode": "vl"},
                order=86,
            ),
        )

class _DMMPlayerConfig(_BaseConfigGroup):
    """DMMPlayer启动器配置"""
    game_exe_path = ConfigItem(
        default_value="",
        data_type=str,
        ui=ConfigItemUI(
            label="游戏安装目录",
            hint="游戏安装路径，指向gakumas.exe（默认自动获取，非必要无需修改）",
            visible_if={"base.run_mode": "PC"},
            order=140,
        )
    )
    viewer_id = ConfigItem(
        default_value="",
        data_type=str,
        ui=ConfigItemUI(
            label="Viewer ID",
            hint="自动获取，非必要无需修改",
            readonly=True,
            visible_if={"base.run_mode": "PC"},
            order=150,
        )
    )
    open_id = ConfigItem(
        default_value="",
        data_type=str,
        ui=ConfigItemUI(
            label="Open ID",
            hint="自动获取，非必要无需修改",
            readonly=True,
            visible_if={"base.run_mode": "PC"},
            order=160,
        )
    )
    pf_token = ConfigItem(
        default_value="",
        data_type=str,
        ui=ConfigItemUI(
            label="PF Token",
            hint="自动获取，非必要无需修改",
            readonly=True,
            visible_if={"base.run_mode": "PC"},
            order=170,
        )
    )


@dataclass
class Config(_BaseConfigGroup):
    base: _Base = field(default_factory=_Base)
    dmm_player: _DMMPlayerConfig = field(default_factory=_DMMPlayerConfig)
    task__auto_purchase: _Task.AutoPurchase = field(default_factory=_Task.AutoPurchase)
    task__auto_contest: _Task.AutoContest = field(default_factory=_Task.AutoContest)
    task__dispatch_work: _Task.DispatchWork = field(default_factory=_Task.DispatchWork)
    task__auto_enhancement_support_card: _Task.AutoEnhancementSupportCard = field(default_factory=_Task.AutoEnhancementSupportCard)
    task__auto_producer: _Task.AutoProducer = field(default_factory=_Task.AutoProducer)

    def to_json_dict(self) -> dict:
        def serialize_group(group):
            result = {}
            for name, attr in vars(group).items():  # 遍历实例属性
                if isinstance(attr, ConfigItem):
                    value = None
                    if attr.value is not None:
                        value = attr.value
                    elif attr.default_value is not None:
                        value = attr.default_value
                    else:
                        target_type = attr.data_type
                        if target_type == bool:
                            value = False
                        elif target_type == int:
                            value = 0
                        elif target_type == float:
                            value = 0.0
                        elif target_type in [dict, list, tuple]:
                            value = target_type([])
                        elif target_type == str:
                            value = ""
                        else:
                            logger.warning(f"Unsupported cast type: {target_type}")

                    result[name] = {
                        "value": value,
                        "default_value": attr.default_value,
                        "data_type": attr.data_type.__name__,
                        "verify": attr.verify,
                        "use_verify": attr.use_verify,
                        "last_modified_time": attr.last_modified_time.isoformat() if attr.last_modified_time else None,
                        "ui": attr.ui.to_json_dict(),
                    }
                elif isinstance(attr, _BaseConfigGroup):
                    result[name] = serialize_group(attr)
            return result

        return serialize_group(self)

    def get_item(self, path: str) -> ConfigItem:
        current = self
        for key in path.split("."):
            if not hasattr(current, key):
                raise AttributeError(f"Config path not found: {path}")
            current = getattr(current, key)
        if not isinstance(current, ConfigItem):
            raise AttributeError(f"Config path is not a ConfigItem: {path}")
        return current

    def from_json_dict(self, data: dict) -> Tuple[bool,List[ConfigVerifyError]]:
        errors = []

        def apply_group(group, group_data, group_name=""):
            for attr_name, attr_value in group_data.items():
                if not hasattr(group, attr_name):
                    continue
                item = getattr(group, attr_name)
                full_name = f"{group_name}.{attr_name}" if group_name else attr_name

                if isinstance(item, ConfigItem):
                    value = attr_value.get("value", item.default_value)
                    if value is None or item.value == value:
                        continue
                    if not isinstance(value, item.data_type):
                        try:
                            value = item.data_type(value)
                        except Exception:
                            errors.append(ConfigVerifyError(
                                group_name,
                                attr_name,
                                f"类型错误，应为 {item.data_type.__name__}"
                            ))
                            continue  # 跳过赋值
                    # 正则校验
                    if item.use_verify and item.verify:
                        if not re.fullmatch(item.verify, str(value)):
                            errors.append(ConfigVerifyError(
                                group_name,
                                attr_name,
                                f"值 '{value}' 不符合正则规则: {item.verify}"
                            ))
                            continue  # 跳过赋值
                    # 赋值
                    item.set(value)

                elif isinstance(item, _BaseConfigGroup):
                    # 递归处理嵌套
                    apply_group(item, attr_value, full_name)

        apply_group(self, data)
        return not bool(errors), errors
