"""
游戏资源下载服务

通过 GkmasObjectManager从游戏服务器获取游戏资源素材
下载采用线程池并行加速，已存在的文件自动跳过（可通过 force=True 强制重新下载）。
下载完成后可触发 CLIP 训练，将新图片导入记忆库（已学习条目自动跳过）。
"""

import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, Thread
from typing import TYPE_CHECKING, Callable, Optional

from src.utils.i18n_tools import i18n_text, serialize_i18n_value
from src.utils.logger import logger
from src.utils.runtime_paths import resolve_data_path

if TYPE_CHECKING:
    from src.core.services.clip_services import CLIPServiceManager

# ---------------------------------------------------------------------------
# GkmasObjectManager vendor 路径注入
# ---------------------------------------------------------------------------

_VENDOR_GOM_PATH = Path(__file__).resolve().parents[3] / "vendor" / "GkmasObjectManager"

_download_lock = Lock()

GAME_ASSETS_DIR = "game_assets"
SUPPORT_CARD_SUBDIR = "support_cards"
IDOL_CARD_SUBDIR = "idol_cards"
ITEM_SUBDIR = "items"
SKILL_CARD_SUBDIR = "skill_cards"

# GkmasObjectManager manifest.search() 使用 re.match（自动锚定行首），资源名含 .unity3d 后缀
SUPPORT_CARD_THUMB_PATTERN = r"img_general_csprt.*thumb-square"
IDOL_CARD_THUMB_PATTERN = r"img_general_cidol.*thumb-square"
SUPPORT_CARD_FULL_PATTERN = r"img_general_csprt-\d+-\d{4}_full\."
ITEM_THUMB_PATTERN = r"img_general_item_"
PITEM_THUMB_PATTERN = r"img_general_pitem_"
SKILL_CARD_THUMB_PATTERN = r"img_general_skillcard_"

SUPPORT_CARD_FULL_SUBDIR = "support_cards_full"
IDOL_CARD_FULL_SUBDIR = "idol_cards_full"

# 并行下载线程数（避免对服务端造成过大压力）
_DOWNLOAD_WORKERS = min(2, os.cpu_count())


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------


def _ensure_gom_in_path() -> None:
    """向 sys.path 注入 vendor/GkmasObjectManager，支持不经 pip 直接使用子模块。"""
    if _VENDOR_GOM_PATH.exists() and str(_VENDOR_GOM_PATH) not in sys.path:
        sys.path.insert(0, str(_VENDOR_GOM_PATH))


def _is_gom_available() -> bool:
    """检查 GkmasObjectManager 是否可用（优先使用 vendor 子模块）。"""
    _ensure_gom_in_path()
    try:
        import GkmasObjectManager  # noqa: F401
        return True
    except ImportError:
        return False


def _get_asset_subdir(subdir: str) -> Path:
    """获取指定子目录的完整路径，不存在则自动创建。"""
    path = resolve_data_path(GAME_ASSETS_DIR, subdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_game_assets_dir() -> Path:
    """获取支援卡图片存储目录路径（兼容旧接口）。"""
    return _get_asset_subdir(SUPPORT_CARD_SUBDIR)


# ---------------------------------------------------------------------------
# 便捷查询
# ---------------------------------------------------------------------------


def get_support_card_image_path(card_id: str) -> Optional[Path]:
    """若支援卡游戏资源图片已缓存，返回本地路径；否则返回 None。"""
    path = _get_asset_subdir(SUPPORT_CARD_SUBDIR) / f"{card_id}.png"
    return path if path.exists() else None


def has_support_card_image(card_id: str) -> bool:
    """判断指定支援卡的游戏资源图片是否已缓存。"""
    return get_support_card_image_path(card_id) is not None


def get_idol_card_image_path(card_id: str) -> Optional[Path]:
    """若偶像卡游戏资源缩略图已缓存，返回本地路径；否则返回 None。"""
    path = _get_asset_subdir(IDOL_CARD_SUBDIR) / f"{card_id}.png"
    return path if path.exists() else None


def has_idol_card_image(card_id: str) -> bool:
    """判断指定偶像卡的游戏资源缩略图是否已缓存。"""
    return get_idol_card_image_path(card_id) is not None


def get_support_card_full_image_path(card_id: str) -> Optional[Path]:
    """若支援卡全尺寸游戏资源图片已缓存，返回本地路径；否则返回 None。"""
    path = _get_asset_subdir(SUPPORT_CARD_FULL_SUBDIR) / f"{card_id}.png"
    return path if path.exists() else None


def has_support_card_full_image(card_id: str) -> bool:
    """判断指定支援卡的全尺寸图片是否已缓存。"""
    return get_support_card_full_image_path(card_id) is not None


def get_idol_card_full_image_path(card_id: str, skin: int = 0) -> Optional[Path]:
    """若偶像卡全尺寸图片已缓存，返回本地路径；否则返回 None。"""
    path = _get_asset_subdir(IDOL_CARD_FULL_SUBDIR) / f"{card_id}_{skin}.png"
    return path if path.exists() else None


def has_idol_card_full_image(card_id: str, skin: int = 0) -> bool:
    """判断指定偶像卡的全尺寸图片是否已缓存。"""
    return get_idol_card_full_image_path(card_id, skin) is not None


def get_item_image_path(item_id: str) -> Optional[Path]:
    """若道具游戏资源图片已缓存，返回本地路径；否则返回 None。"""
    path = _get_asset_subdir(ITEM_SUBDIR) / f"{item_id}.png"
    return path if path.exists() else None


def has_item_image(item_id: str) -> bool:
    """判断指定道具的游戏资源图片是否已缓存。"""
    return get_item_image_path(item_id) is not None


def get_skill_card_image_path(asset_id: str) -> Optional[Path]:
    """若技能卡游戏资源图片已缓存，返回本地路径；否则返回 None。

    Args:
        asset_id: ProduceCard.assetId 去掉 ``img_general_`` 前缀后的部分，
                  例如 ``skillcard_act-0_001``。
    """
    path = _get_asset_subdir(SKILL_CARD_SUBDIR) / f"{asset_id}.png"
    return path if path.exists() else None


def has_skill_card_image(asset_id: str) -> bool:
    """判断指定技能卡的游戏资源图片是否已缓存。"""
    return get_skill_card_image_path(asset_id) is not None


def get_downloaded_card_count() -> int:
    """获取已缓存的支援卡图片数量。"""
    return len(list(_get_asset_subdir(SUPPORT_CARD_SUBDIR).glob("*.png")))


def get_downloaded_item_count() -> int:
    """获取已缓存的道具图片数量。"""
    return len(list(_get_asset_subdir(ITEM_SUBDIR).glob("*.png")))


def get_downloaded_skill_card_count() -> int:
    """获取已缓存的技能卡图片数量。"""
    return len(list(_get_asset_subdir(SKILL_CARD_SUBDIR).glob("*.png")))


# ---------------------------------------------------------------------------
# 下载状态管理
# ---------------------------------------------------------------------------


class GameAssetDownloadStatus:
    """各类资源下载进度的线程安全状态容器。"""

    def __init__(self):
        self._lock = Lock()
        self.downloading = False
        self.progress = 0
        self.total = 0
        self.message = ""
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        """序列化为可 JSON 化的字典。"""
        with self._lock:
            return {
                "downloading": self.downloading,
                "progress": self.progress,
                "total": self.total,
                "message": serialize_i18n_value(self.message),
                "error": serialize_i18n_value(self.error),
            }

    def set(self, **kwargs) -> None:
        """线程安全地批量更新状态字段。"""
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)


_status = GameAssetDownloadStatus()


def get_download_status() -> dict:
    """获取当前下载状态的快照字典。"""
    return _status.to_dict()



# ---------------------------------------------------------------------------
# 内部并行下载核心
# ---------------------------------------------------------------------------


def _build_asset_id_map(db_list: list | None) -> dict[str, str]:
    """将数据库列表转换为 assetId -> entry_id 的映射字典。

    Args:
        db_list: 含 id/assetId 字段的数据库对象列表，可以是 dict 或 dataclass。

    Returns:
        以资源 ID（及其 stem）为键、条目 ID 为值的映射字典。
    """
    mapping: dict[str, str] = {}
    if not db_list:
        return mapping
    for entry in db_list:
        if isinstance(entry, dict):
            asset_id = entry.get("assetId")
            entry_id = entry.get("id")
        else:
            asset_id = getattr(entry, "assetId", None)
            entry_id = getattr(entry, "id", None)
        if asset_id and entry_id:
            mapping[asset_id] = entry_id
            mapping[Path(asset_id).stem] = entry_id
    return mapping


def _download_assets_parallel(
    objects: list,
    asset_dir: Path,
    asset_id_map: dict[str, str],
    name_transform: Callable[[str], str],
    force: bool,
    workers: int,
    status_label: str,
) -> tuple[int, int]:
    """使用线程池并行下载资源列表。

    每个条目先写入独立临时目录，成功后原子移至目标路径，
    已存在且不强制覆盖的条目直接跳过。

    Args:
        objects: GkmasResource 实例列表。
        asset_dir: 目标输出目录。
        asset_id_map: 资源名（或其 stem）到输出文件名（不含 .png）的映射。
        name_transform: 当映射中无法命中时，将资源名 stem 转为文件名的回退函数。
        force: 为 True 时强制覆盖已存在的文件。
        workers: 并发线程数。
        status_label: 状态消息中展示的资源类型名称。

    Returns:
        (新下载数量, 已跳过数量) 的二元组。
    """
    total = len(objects)
    _status.set(
        total=total,
        message=i18n_text(
            "backend.gameAsset.downloading",
            fallback=f"正在下载 {status_label}...",
            label=status_label,
        ),
    )

    downloaded = 0
    skipped = 0
    counter_lock = Lock()

    def _fetch_one(obj) -> str:
        """下载单个资源；返回 'new'、'skip' 或 'fail'。"""
        bare = Path(obj.name).stem  # 去除扩展名
        transformed = name_transform(bare)
        file_id = (
            asset_id_map.get(obj.name)
            or asset_id_map.get(bare)
            or asset_id_map.get(transformed)
            or transformed
        )
        target = asset_dir / f"{file_id}.png"

        if target.exists() and not force:
            return "skip"

        with tempfile.TemporaryDirectory() as tmp:
            try:
                obj.download(path=tmp, categorize=False, image_format="PNG")
                for f in Path(tmp).rglob("*.png"):
                    shutil.move(str(f), str(target))
                    return "new"
                return "fail"
            except Exception as exc:
                logger.debug(f"[GameAsset] Failed to download {obj.name}: {exc}")
                return "fail"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, obj): obj for obj in objects}
        for future in as_completed(futures):
            result = future.result()
            with counter_lock:
                if result == "skip":
                    skipped += 1
                elif result == "new":
                    downloaded += 1
                _status.set(progress=downloaded + skipped)

    return downloaded, skipped


def _download_typed_assets(
    pattern: str,
    subdir: str,
    name_transform: Callable[[str], str],
    db_list: list | None,
    force: bool,
    label: str,
) -> bool:
    """统一的资源下载流程，供各类型资产下载函数复用。

    流程：检查 GOM 可用 → 获取清单 → 搜索匹配对象 → 并行下载 → 更新状态。

    Args:
        pattern: 传给 ``manifest.search()`` 的正则模式。
        subdir: 目标子目录名（位于 ``game_assets/`` 下）。
        name_transform: 将资源名 stem 映射为输出文件名的回退函数。
        db_list: 数据库条目列表，含 id/assetId 字段；为 None 时下载所有匹配项。
        force: 是否强制覆盖已存在文件。
        label: 状态消息中展示的资源类型名称。

    Returns:
        操作是否成功完成。
    """
    if not _is_gom_available():
        msg = "GkmasObjectManager 未就绪（请确认 vendor/GkmasObjectManager 子模块已初始化）"
        _status.set(error=i18n_text("backend.api.objectManagerUnavailable", fallback=msg))
        logger.warning(msg)
        return False

    if not _download_lock.acquire(blocking=False):
        logger.warning("Game asset download already in progress")
        return False

    try:
        _status.set(
            downloading=True, progress=0, total=0,
            message=i18n_text(
                "backend.gameAsset.fetchingManifestWithLabel",
                fallback=f"正在获取资源清单（{label}）...",
                label=label,
            ),
            error=None,
        )

        import GkmasObjectManager as gom

        logger.info(f"Fetching game manifest for {label}...")
        manifest = gom.fetch()

        _status.set(
            message=i18n_text(
                "backend.gameAsset.searchingWithLabel",
                fallback=f"正在检索 {label} 资源...",
                label=label,
            ),
        )
        objects = manifest.search(pattern)

        if not objects:
            _status.set(
                downloading=False,
                message=i18n_text(
                    "backend.gameAsset.noMatchingResources",
                    fallback=f"未找到 {label} 资源",
                    label=label,
                ),
                error=i18n_text("backend.gameAsset.noMatchingObjects", fallback="no matching objects"),
            )
            logger.warning(f"No {label} objects found in manifest")
            return False

        logger.info(f"Found {len(objects)} {label} objects in manifest")

        asset_dir = _get_asset_subdir(subdir)
        asset_id_map = _build_asset_id_map(db_list)

        downloaded, skipped = _download_assets_parallel(
            objects=objects,
            asset_dir=asset_dir,
            asset_id_map=asset_id_map,
            name_transform=name_transform,
            force=force,
            workers=_DOWNLOAD_WORKERS,
            status_label=label,
        )

        total_cached = len(list(asset_dir.glob("*.png")))
        _status.set(
            downloading=False,
            message=i18n_text(
                "backend.gameAsset.downloadCompleted",
                fallback=(
                    f"{label}下载完成：{downloaded} 张新图片，"
                    f"{skipped} 张已跳过（共缓存 {total_cached} 张）"
                ),
                label=label,
                downloaded=downloaded,
                skipped=skipped,
                total=total_cached,
            ),
            error=None,
        )
        logger.success(
            f"{label} download finished: {downloaded} new, {skipped} skipped, "
            f"{total_cached} total cached"
        )
        return True

    except Exception as exc:
        _status.set(
            downloading=False,
            error=i18n_text("backend.gameAsset.downloadFailedError", fallback=str(exc), error=str(exc)),
            message=i18n_text(
                "backend.gameAsset.downloadFailed",
                fallback=f"{label}下载失败",
                label=label,
            ),
        )
        logger.error(f"Failed to download {label}: {exc}")
        return False
    finally:
        _download_lock.release()


# ---------------------------------------------------------------------------
# 公共下载接口
# ---------------------------------------------------------------------------


def download_support_card_images(
    card_db_list: list | None = None,
    force: bool = False,
) -> bool:
    """下载支援卡方形缩略图到 game_assets/support_cards。

    使用 GkmasObjectManager 获取清单并下载匹配 ``img_general_csprt.*thumb-square``
    的资源，以 ``{card_id}.png`` 命名保存，已存在文件默认跳过。
    下载采用线程池并行加速。

    Args:
        card_db_list: 支援卡数据库列表（含 id/assetId 字段），为 None 时下载全部匹配资源。
        force: 为 True 时强制重新下载已存在的图片。

    Returns:
        操作是否成功完成。
    """
    return _download_typed_assets(
        pattern=SUPPORT_CARD_THUMB_PATTERN,
        subdir=SUPPORT_CARD_SUBDIR,
        name_transform=lambda bare: bare.replace("img_general_", "").replace("_thumb-square", ""),
        db_list=card_db_list,
        force=force,
        label="支援卡缩略图",
    )


def download_idol_card_images(
    card_db_list: list | None = None,
    force: bool = False,
) -> bool:
    """下载偶像卡缩略图到 game_assets/idol_cards。"""
    return _download_typed_assets(
        pattern=IDOL_CARD_THUMB_PATTERN,
        subdir=IDOL_CARD_SUBDIR,
        name_transform=lambda bare: re.sub(
            r"_\d+-thumb-square$",
            "",
            bare.replace("img_general_", ""),
        ),
        db_list=card_db_list,
        force=force,
        label="偶像卡缩略图",
    )


def download_item_images(
    item_db_list: list | None = None,
    force: bool = False,
) -> bool:
    """下载道具图片到 game_assets/items。

    使用 GkmasObjectManager 获取清单并下载匹配 ``img_general_item_`` 前缀
    的资源，以 ``{item_id}.png`` 命名保存，已存在文件默认跳过。
    若传入 item_db_list，则优先按 assetId→id 映射命名；否则使用资源名去前缀。

    Args:
        item_db_list: 道具数据库列表（含 id/assetId 字段），为 None 时下载全部匹配资源。
        force: 为 True 时强制重新下载已存在的图片。

    Returns:
        操作是否成功完成。
    """
    return _download_typed_assets(
        pattern=ITEM_THUMB_PATTERN,
        subdir=ITEM_SUBDIR,
        name_transform=lambda bare: bare.replace("img_general_", ""),
        db_list=item_db_list,
        force=force,
        label="道具图片",
    )


def download_pitem_images(
    force: bool = False,
) -> bool:
    """下载 P 物品图片到 game_assets/items（img_general_pitem_ 资源）。"""
    return _download_typed_assets(
        pattern=PITEM_THUMB_PATTERN,
        subdir=ITEM_SUBDIR,
        name_transform=lambda bare: bare.replace("img_general_", ""),
        db_list=None,
        force=force,
        label="P物品图片",
    )


def download_skill_card_images(
    card_db_list: list | None = None,
    force: bool = False,
) -> bool:
    """下载技能卡图片到 game_assets/skill_cards。

    使用 GkmasObjectManager 获取清单并下载匹配 ``img_general_skillcard_`` 前缀
    的资源，以 ``{asset_id_去前缀}.png`` 命名保存，已存在文件默认跳过。
    同一 assetId 的不同升级等级共享同一图片文件。

    Args:
        card_db_list: 技能卡数据库列表（含 id/assetId 字段），为 None 时下载全部匹配资源。
        force: 为 True 时强制重新下载已存在的图片。

    Returns:
        操作是否成功完成。
    """
    return _download_typed_assets(
        pattern=SKILL_CARD_THUMB_PATTERN,
        subdir=SKILL_CARD_SUBDIR,
        name_transform=lambda bare: bare.replace("img_general_", ""),
        db_list=card_db_list,
        force=force,
        label="技能卡图片",
    )


def download_support_card_full_images(
    card_db_list: list | None = None,
    force: bool = False,
) -> bool:
    """下载支援卡全尺寸图片到 game_assets/support_cards_full。"""
    return _download_typed_assets(
        pattern=SUPPORT_CARD_FULL_PATTERN,
        subdir=SUPPORT_CARD_FULL_SUBDIR,
        name_transform=lambda bare: bare.replace("img_general_", "").replace("_full", ""),
        db_list=card_db_list,
        force=force,
        label="支援卡全尺寸图",
    )


def download_single_support_card_full_image(card_id: str, asset_id: str) -> bool:
    """按需下载单张支援卡的全尺寸图片。

    直接搜索匹配 ``img_general_{asset_id}_full`` 的对象并下载，
    不占用全局 ``_download_lock``，适合在用户查看卡牌详情时按需触发。

    Args:
        card_id: 支援卡 ID（如 ``s_card-3-0031``），用作输出文件名。
        asset_id: 支援卡的 assetId（如 ``csprt-3-0031``），用于清单搜索。

    Returns:
        已成功下载（或文件已存在）时返回 True，否则返回 False。
    """
    import re as _re

    target = _get_asset_subdir(SUPPORT_CARD_FULL_SUBDIR) / f"{card_id}.png"
    if target.exists():
        return True

    if not _is_gom_available():
        return False

    try:
        import GkmasObjectManager as gom
        manifest = gom.fetch()
        pattern = rf"img_general_{_re.escape(asset_id)}_full\."
        objects = manifest.search(pattern)
        if not objects:
            logger.warning(f"[GameAsset] No full image found for {card_id} (assetId={asset_id})")
            return False

        obj = objects[0]
        with tempfile.TemporaryDirectory() as tmp:
            obj.download(path=tmp, categorize=False, image_format="PNG")
            for f in Path(tmp).rglob("*.png"):
                shutil.move(str(f), str(target))
                logger.success(f"[GameAsset] Downloaded full image for {card_id}")
                return True
        return False
    except Exception as exc:
        logger.debug(f"[GameAsset] Failed single full image for {card_id}: {exc}")
        return False


def download_single_idol_card_full_image(card_id: str, asset_id: str, skin: int = 0) -> bool:
    """按需下载单张偶像卡的全尺寸图片。"""
    target = _get_asset_subdir(IDOL_CARD_FULL_SUBDIR) / f"{card_id}_{skin}.png"
    if target.exists():
        return True

    if not _is_gom_available():
        return False

    try:
        import GkmasObjectManager as gom

        manifest = gom.fetch()
        pattern = rf"img_general_{re.escape(asset_id)}_{skin}-full\."
        objects = manifest.search(pattern)
        if not objects:
            logger.warning(
                f"[GameAsset] No idol full image found for {card_id} "
                f"(assetId={asset_id}, skin={skin})"
            )
            return False

        obj = objects[0]
        with tempfile.TemporaryDirectory() as tmp:
            obj.download(path=tmp, categorize=False, image_format="PNG")
            for f in Path(tmp).rglob("*.png"):
                shutil.move(str(f), str(target))
                logger.success(f"[GameAsset] Downloaded idol full image for {card_id} skin={skin}")
                return True
        return False
    except Exception as exc:
        logger.debug(f"[GameAsset] Failed idol full image for {card_id}: {exc}")
        return False


def _run_download_phase(
    manifest,
    subdir: str,
    pattern: str,
    name_transform: Callable[[str], str],
    db_list: list | None,
    force: bool,
    label: str,
) -> bool:
    """在已持有 _download_lock 的情况下，对单个资源类型执行下载流程。

    与 :func:`_download_typed_assets` 的区别：不管理锁，也不修改 downloading 标志，
    仅更新 message/progress/total，适合在 :func:`download_all_for_dialog` 等
    多阶段组合下载中使用。

    Returns:
        操作是否成功完成（未找到资源时返回 False 但不中断调用方）。
    """
    _status.set(
        message=i18n_text(
            "backend.gameAsset.searchingWithLabel",
            fallback=f"正在检索 {label} 资源...",
            label=label,
        ),
        progress=0,
        total=0,
    )
    objects = manifest.search(pattern)
    if not objects:
        logger.warning(f"No {label} objects found in manifest")
        return False

    logger.info(f"Found {len(objects)} {label} objects")
    asset_dir = _get_asset_subdir(subdir)
    asset_id_map = _build_asset_id_map(db_list)

    downloaded, skipped = _download_assets_parallel(
        objects=objects,
        asset_dir=asset_dir,
        asset_id_map=asset_id_map,
        name_transform=name_transform,
        force=force,
        workers=_DOWNLOAD_WORKERS,
        status_label=label,
    )
    total_cached = len(list(asset_dir.glob("*.png")))
    _status.set(
        message=i18n_text(
            "backend.gameAsset.phaseCompleted",
            fallback=(
                f"{label}：{downloaded} 张新图片，{skipped} 张已跳过"
                f"（共缓存 {total_cached} 张）"
            ),
            label=label,
            downloaded=downloaded,
            skipped=skipped,
            total=total_cached,
        ),
    )
    logger.success(
        f"{label} done: {downloaded} new, {skipped} skipped, {total_cached} cached"
    )
    return True


def download_all_for_dialog(
    card_db_list: list | None = None,
    force: bool = False,
    clip_manager: "CLIPServiceManager | None" = None,
) -> bool:
    """在单次锁持有周期内依次下载所有支援卡对话框所需资源。

    下载顺序：缩略图 → 全尺寸图 → 技能卡图 → P物品图。
    仅获取一次 GOM manifest，status.downloading 保持 True 直到全部完成，
    防止前端轮询在阶段间间隙误判为"下载完毕"而停止。

    Args:
        card_db_list: 支援卡数据库列表（含 id / assetId 字段）。
        force: 是否强制重新下载已存在的文件。
        clip_manager: 可选 CLIP 管理器，下载完缩略图后触发训练。

    Returns:
        操作是否完整执行（GOM 不可用时返回 False）。
    """
    if not _is_gom_available():
        msg = "GkmasObjectManager 未就绪（请确认 vendor/GkmasObjectManager 子模块已初始化）"
        _status.set(error=i18n_text("backend.api.objectManagerUnavailable", fallback=msg))
        logger.warning(msg)
        return False

    if not _download_lock.acquire(blocking=False):
        logger.warning("Game asset download already in progress")
        return False

    try:
        _status.set(
            downloading=True, progress=0, total=0,
            message=i18n_text("backend.gameAsset.fetchingManifest", fallback="正在获取资源清单..."),
            error=None,
        )
        import GkmasObjectManager as gom

        logger.info("Fetching game manifest for full dialog asset download...")
        manifest = gom.fetch()

        # ── Phase 1: 支援卡缩略图 ──────────────────────────────────────────
        _run_download_phase(
            manifest,
            subdir=SUPPORT_CARD_SUBDIR,
            pattern=SUPPORT_CARD_THUMB_PATTERN,
            name_transform=lambda bare: bare.replace("img_general_", "").replace("_thumb-square", ""),
            db_list=card_db_list,
            force=force,
            label="支援卡缩略图",
        )

        # 缩略图完成后触发 CLIP 训练（不等待）
        if clip_manager is not None:
            from threading import Thread as _T
            _T(
                target=train_clip_from_game_assets,
                args=(clip_manager, SUPPORT_CARD_SUBDIR),
                daemon=True,
            ).start()

        # ── Phase 2: 支援卡全尺寸图 ──────────────────────────────────────
        _run_download_phase(
            manifest,
            subdir=SUPPORT_CARD_FULL_SUBDIR,
            pattern=SUPPORT_CARD_FULL_PATTERN,
            name_transform=lambda bare: bare.replace("img_general_", "").replace("_full", ""),
            db_list=card_db_list,
            force=force,
            label="支援卡全尺寸图",
        )

        # ── Phase 3: 技能卡图 ────────────────────────────────────────────
        _run_download_phase(
            manifest,
            subdir=SKILL_CARD_SUBDIR,
            pattern=SKILL_CARD_THUMB_PATTERN,
            name_transform=lambda bare: bare.replace("img_general_", ""),
            db_list=None,
            force=force,
            label="技能卡图片",
        )

        # ── Phase 4: P物品图 ─────────────────────────────────────────────
        _run_download_phase(
            manifest,
            subdir=ITEM_SUBDIR,
            pattern=PITEM_THUMB_PATTERN,
            name_transform=lambda bare: bare.replace("img_general_", ""),
            db_list=None,
            force=force,
            label="P物品图片",
        )

        _status.set(
            downloading=False,
            message=i18n_text("backend.gameAsset.dialogAssetsCompleted", fallback="支援卡相关资源全部下载完成"),
            error=None,
        )
        logger.success("download_all_for_dialog: all phases complete")
        return True

    except Exception as exc:
        _status.set(
            downloading=False,
            error=i18n_text("backend.gameAsset.downloadFailedError", fallback=str(exc), error=str(exc)),
            message=i18n_text("backend.gameAsset.bulkDownloadFailed", fallback="资源下载失败"),
        )
        logger.error(f"download_all_for_dialog failed: {exc}")
        return False
    finally:
        _download_lock.release()


def download_support_card_images_async(
    card_db_list: list | None = None,
    force: bool = False,
    clip_manager: "CLIPServiceManager | None" = None,
) -> Thread:
    """在后台线程异步下载支援卡缩略图，立即返回线程对象。

    下载完成后若传入 ``clip_manager`` 则自动触发 CLIP 训练。

    Args:
        card_db_list: 支援卡数据库列表，参见 :func:`download_support_card_images`。
        force: 是否强制覆盖已存在文件。
        clip_manager: 可选的 CLIP 管理器，传入后下载完成会自动训练。

    Returns:
        已启动的 :class:`~threading.Thread` 实例。
    """
    def _worker():
        download_support_card_images(card_db_list, force)
        if clip_manager is not None:
            train_clip_from_game_assets(clip_manager, SUPPORT_CARD_SUBDIR)

    thread = Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def download_idol_card_images_async(
    card_db_list: list | None = None,
    force: bool = False,
    clip_manager: "CLIPServiceManager | None" = None,
) -> Thread:
    """在后台线程异步下载偶像卡缩略图，并在完成后触发 CLIP 训练。"""

    def _worker():
        download_idol_card_images(card_db_list, force)
        if clip_manager is not None:
            train_clip_from_game_assets(clip_manager, IDOL_CARD_SUBDIR)

    thread = Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def download_item_images_async(
    item_db_list: list | None = None,
    force: bool = False,
    clip_manager: "CLIPServiceManager | None" = None,
) -> Thread:
    """在后台线程异步下载道具图片，立即返回线程对象。

    下载完成后若传入 ``clip_manager`` 则自动触发 CLIP 训练。

    Args:
        item_db_list: 道具数据库列表，参见 :func:`download_item_images`。
        force: 是否强制覆盖已存在文件。
        clip_manager: 可选的 CLIP 管理器，传入后下载完成会自动训练。

    Returns:
        已启动的 :class:`~threading.Thread` 实例。
    """
    def _worker():
        download_item_images(item_db_list, force)
        if clip_manager is not None:
            train_clip_from_game_assets(clip_manager, ITEM_SUBDIR)

    thread = Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def download_skill_card_images_async(
    card_db_list: list | None = None,
    force: bool = False,
    clip_manager: "CLIPServiceManager | None" = None,
) -> Thread:
    """在后台线程异步下载技能卡图片，立即返回线程对象。

    下载完成后若传入 ``clip_manager`` 则自动触发 CLIP 训练。

    Args:
        card_db_list: 技能卡数据库列表，参见 :func:`download_skill_card_images`。
        force: 是否强制覆盖已存在文件。
        clip_manager: 可选的 CLIP 管理器，传入后下载完成会自动训练。

    Returns:
        已启动的 :class:`~threading.Thread` 实例。
    """
    def _worker():
        download_skill_card_images(card_db_list, force)
        if clip_manager is not None:
            train_clip_from_game_assets(clip_manager, SKILL_CARD_SUBDIR)

    thread = Thread(target=_worker, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# CLIP 训练接口
# ---------------------------------------------------------------------------


def train_clip_from_game_assets(
    clip_manager: "CLIPServiceManager",
    subdir: str = SUPPORT_CARD_SUBDIR,
) -> int:
    """使用已缓存的游戏资源图片训练 CLIP 记忆库，已学习的条目自动跳过。

    遍历指定子目录下的所有 PNG 文件，尝试将其加入对应 CLIP 服务的记忆库。
    CLIPTools.add_to_memory() 内部会通过余弦相似度去重，相似度超阈值的图片
    不会重复写入，因此多次调用是幂等安全的。

    Args:
        clip_manager: 已初始化的 :class:`~src.core.services.clip_services.CLIPServiceManager`。
        subdir: 要训练的资源子目录名，可选值为
            ``SUPPORT_CARD_SUBDIR``、``ITEM_SUBDIR``、``SKILL_CARD_SUBDIR``。

    Returns:
        本次新增到记忆库的图片数量。
    """
    import cv2

    from src.utils.game_database_tools import (
        GakumasDatabase_IdolCardDataUtils,
        GakumasDatabase_ItemDataUtils,
        GakumasDatabase_ProduceCardDataUtils,
        GakumasDatabase_SupportCardDataUtils,
    )

    asset_dir = _get_asset_subdir(subdir)
    png_files = list(asset_dir.glob("*.png"))
    if not png_files:
        logger.info(f"[GameAsset CLIP] No images found in {subdir}, skipping training")
        return 0

    if subdir == SUPPORT_CARD_SUBDIR:
        clip_svc = clip_manager.support_card_clip
        db = GakumasDatabase_SupportCardDataUtils()
        # Build assetId -> entry fallback map (handles files named csprt-X-XXXX from old downloads)
        _asset_id_fallback = {
            getattr(e, 'assetId', None): e
            for e in (db.get_all_item() or [])
            if getattr(e, 'assetId', None)
        }
        def _get_payload(file_stem: str):
            return db.get_by_id(file_stem) or _asset_id_fallback.get(file_stem)

    elif subdir == IDOL_CARD_SUBDIR:
        clip_svc = clip_manager.idol_card_clip
        db = GakumasDatabase_IdolCardDataUtils()
        _asset_id_fallback = {
            getattr(e, 'assetId', None): e
            for e in (db.get_all_item() or [])
            if getattr(e, 'assetId', None)
        }

        def _get_payload(file_stem: str):
            return db.get_by_id(file_stem) or _asset_id_fallback.get(file_stem)

    elif subdir == ITEM_SUBDIR:
        clip_svc = clip_manager.item_clip
        db = GakumasDatabase_ItemDataUtils()
        def _get_payload(file_stem: str):
            return db.get_by_id(file_stem)

    elif subdir == SKILL_CARD_SUBDIR:
        clip_svc = clip_manager.skill_card_clip
        db = GakumasDatabase_ProduceCardDataUtils()
        # 文件名形如 skillcard_men-3_043-ssmk，assetId 形如 img_general_skillcard_men-3_043-ssmk
        # 但 ProduceCard map key 是 {id}.{upgradeCount}，不能直接用 assetId 查
        # 因此反向建立 bare_assetId -> card 映射，同 assetId 保留 upgradeCount 最小的条目
        _asset_id_to_card: dict = {}
        for _card in (db.get_all_item() or []):
            _asset_id = getattr(_card, 'assetId', None)
            if not _asset_id:
                continue
            _bare = _asset_id.replace("img_general_", "")
            _existing = _asset_id_to_card.get(_bare)
            if _existing is None or _card.upgradeCount < _existing.upgradeCount:
                _asset_id_to_card[_bare] = _card
        def _get_payload(file_stem: str):
            return _asset_id_to_card.get(file_stem)

    else:
        logger.warning(f"[GameAsset CLIP] Unknown subdir: {subdir}")
        return 0

    added = 0
    missing_entries: list[str] = []
    for png_path in png_files:
        file_stem = png_path.stem  # 文件名去 .png
        payload = _get_payload(file_stem)
        if payload is None:
            missing_entries.append(file_stem)
            continue

        image = cv2.imread(str(png_path))
        if image is None:
            logger.debug(f"[GameAsset CLIP] Failed to read image: {png_path}")
            continue

        try:
            was_added = clip_svc.add_to_memory(image, payload, save_image=False)
            if was_added:
                added += 1
                logger.debug(f"[GameAsset CLIP] Trained: {file_stem}")
        except Exception as exc:
            logger.debug(f"[GameAsset CLIP] Error training {file_stem}: {exc}")

    if missing_entries:
        sample = ", ".join(missing_entries[:8])
        logger.debug(
            f"[GameAsset CLIP] {subdir}: skipped {len(missing_entries)} cache files "
            f"without current DB entry"
            + (f" (samples: {sample})" if sample else "")
        )

    logger.info(
        f"[GameAsset CLIP] {subdir}: {added} new entries added "
        f"({len(png_files) - added} already learned or skipped)"
    )
    return added


def train_clip_from_game_assets_async(
    clip_manager: "CLIPServiceManager",
    subdir: str = SUPPORT_CARD_SUBDIR,
) -> Thread:
    """在后台线程异步触发 CLIP 训练，立即返回线程对象。

    Args:
        clip_manager: 已初始化的 :class:`~src.core.services.clip_services.CLIPServiceManager`。
        subdir: 要训练的资源子目录名。

    Returns:
        已启动的 :class:`~threading.Thread` 实例。
    """
    thread = Thread(
        target=train_clip_from_game_assets,
        args=(clip_manager, subdir),
        daemon=True,
    )
    thread.start()
    return thread
