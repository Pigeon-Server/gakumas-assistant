import os.path
from copy import copy

import adbutils
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.core.services import game_asset_service

from src.constants.task_status import TaskStatus
from src.constants.websocket_actions import WebsocketActions
from src.core.web.websocket import WebSocketManager
from typing import TYPE_CHECKING

from src.entity.WebSocketData import WebSocketData
from src.utils.adb_runtime import describe_adb_error
from src.utils.dmm_tools import extract_gakumas_launch_parameters
from src.utils.game_database_tools import (
    GakumasDatabase_IdolCardDataUtils,
    GakumasDatabase_ItemDataUtils,
    GakumasDatabase_ProduceItemDataUtils,
    GakumasDatabase_SupportCardDataUtils,
    _concat_produce_descriptions,
)
from src.utils.logger import logger
from src.utils.i18n_tools import I18nText, i18n_text, serialize_i18n_value, translate_like_payload
from src.utils.runtime_paths import resolve_data_str, resolve_runtime_str
from src.utils.task_failure_package import resolve_task_failure_package_download

if TYPE_CHECKING:
    from src.main import AppProcessor

def _api_return(status: bool, message: I18nText | str | dict = '', data: dict | list = None):
    return {
        'status': status,
        'message': serialize_i18n_value(message),
        'data': serialize_i18n_value(data),
    }

def register_routes(app: FastAPI, processor: "AppProcessor", ws_manager: WebSocketManager):
    def _resource_not_ready_response():
        return _api_return(
            False,
            translate_like_payload(None, "backend.api.resourceNotReady"),
        )

    def _get_item_db():
        if not processor.is_resource_ready():
            raise RuntimeError(str(i18n_text("backend.api.gameDatabaseNotReady", fallback="游戏数据库资源尚未就绪")))
        return GakumasDatabase_ItemDataUtils()

    def _get_support_card_db():
        if not processor.is_resource_ready():
            raise RuntimeError(str(i18n_text("backend.api.gameDatabaseNotReady", fallback="游戏数据库资源尚未就绪")))
        return GakumasDatabase_SupportCardDataUtils()

    def _get_idol_card_db():
        if not processor.is_resource_ready():
            raise RuntimeError(str(i18n_text("backend.api.gameDatabaseNotReady", fallback="游戏数据库资源尚未就绪")))
        return GakumasDatabase_IdolCardDataUtils()

    def _get_produce_item_db():
        if not processor.is_resource_ready():
            raise RuntimeError(str(i18n_text("backend.api.gameDatabaseNotReady", fallback="游戏数据库资源尚未就绪")))
        return GakumasDatabase_ProduceItemDataUtils()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            initial_sent = await ws_manager.send_action(
                websocket,
                WebsocketActions.App.StatusChanged,
                WebSocketData(message=processor.build_app_status()),
            )
            if not initial_sent:
                return
            while True:
                data = await websocket.receive_json()
                if not data.get("action"):
                    continue
                action = data.get("action")
                data = data.get("data")
                if action == WebsocketActions.BaseActionFlag + ":" + WebsocketActions.WebsocketHeartBeat.Ping:
                    pong_sent = await ws_manager.send_action(websocket, WebsocketActions.WebsocketHeartBeat.Pong)
                    if not pong_sent:
                        break
        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected")
        except RuntimeError as e:
            if ws_manager.is_normal_disconnect_error(e, websocket):
                logger.debug("WebSocket session closed normally [{}]: {}", type(e).__name__, e)
            else:
                logger.exception(f"Websocket Error: {e}")
        except Exception as e:
            logger.exception(f"Websocket Error: {e}")
        finally:
            ws_manager.disconnect(websocket)

    @app.get("/api/task/start")
    def start_task_queue():
        """启动任务队列"""
        if not processor.is_resource_ready():
            return _resource_not_ready_response()
        if not processor.ensure_device_ready(restart_inference=True):
            return _api_return(
                False,
                processor.get_device_status().get("message")
                or translate_like_payload(None, "backend.app.deviceUnavailable"),
            )
        if not processor.exec_task():
            return _api_return(False, translate_like_payload(None, "backend.api.taskQueueStartFailed"))
        return _api_return(True, translate_like_payload(None, "backend.api.ok"))

    @app.get("/api/task/start/{task_name:str}")
    def run_task(task_name: str):
        """
        运行任务（单个）
        :param task_name: 任务名
        :return:
        """
        if not processor.is_resource_ready():
            return _resource_not_ready_response()
        if not processor.ensure_device_ready(restart_inference=True):
            return _api_return(
                False,
                processor.get_device_status().get("message")
                or translate_like_payload(None, "backend.app.deviceUnavailable"),
            )
        if not processor.exec_task(task_name):
            return _api_return(False, translate_like_payload(None, "backend.api.taskStartFailed"))
        return _api_return(True, translate_like_payload(None, "backend.api.ok"))

    @app.get("/api/task/start_from/{task_name:str}")
    def run_task_from(task_name: str):
        """从指定任务开始执行后续自动任务"""
        if not processor.is_resource_ready():
            return _resource_not_ready_response()
        if not processor.ensure_device_ready(restart_inference=True):
            return _api_return(
                False,
                processor.get_device_status().get("message")
                or translate_like_payload(None, "backend.app.deviceUnavailable"),
            )
        task = processor.task_queue._find_task(task_name)
        if task is None:
            return _api_return(False, translate_like_payload(None, "backend.api.invalidTaskName"))
        if task.manual_only:
            return _api_return(False, translate_like_payload(None, "backend.api.manualOnlyRunFromUnsupported"))
        if not processor.exec_task_from(task_name):
            return _api_return(False, translate_like_payload(None, "backend.api.runFromFailed"))
        return _api_return(True, translate_like_payload(None, "backend.api.ok"))

    @app.get("/api/task/suspend")
    def suspend_task():
        """
        挂起任务
        :return:
        """
        current_running_task = processor.task_queue.get_current_running_task()
        if not current_running_task:
            return _api_return(False, translate_like_payload(None, "backend.api.noRunningTask"))
        if not current_running_task.allow_manual_suspend:
            return _api_return(False, translate_like_payload(None, "backend.api.suspendUnsupported"))
        processor.task_queue.suspend_running_task()
        return _api_return(True, translate_like_payload(None, "backend.api.ok"))

    @app.get("/api/task/resume")
    def resume_task():
        """
        恢复任务
        :return:
        """
        if processor.task_queue.queue_status() != TaskStatus.SUSPENDED:
            return _api_return(False, translate_like_payload(None, "backend.api.noSuspendedTask"))
        if not processor.task_queue.get_current_suspend_task().allow_manual_resume:
            return _api_return(False, translate_like_payload(None, "backend.api.resumeUnsupported"))
        if processor.task_queue.get_current_running_task() is not None:
            logger.debug(f"Current running task: {processor.task_queue.get_current_running_task()}")
            return _api_return(False, translate_like_payload(None, "backend.api.resumeBlockedByInsertedTask"))
        processor.task_queue.resume_suspended_task()
        return _api_return(True, translate_like_payload(None, "backend.api.ok"))


    @app.get("/api/task/stop")
    def stop_task_queue():
        """停止任务队列"""
        processor.task_queue.stop()
        return _api_return(True, translate_like_payload(None, "backend.api.ok"))

    @app.get("/api/task/failure_package/download/{package_id}")
    def download_task_failure_package(package_id: str):
        """下载任务失败日志压缩包（重启前有效）。"""
        package_path = resolve_task_failure_package_download(package_id)
        if package_path is None:
            return _api_return(False, translate_like_payload(None, "backend.api.taskFailurePackageMissing"))
        return FileResponse(
            str(package_path),
            media_type="application/zip",
            filename=package_path.name,
        )

    @app.get("/api/status")
    def get_status():
        """获取服务状态"""
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), processor.build_app_status())

    @app.post("/api/app/shutdown")
    def shutdown_app():
        processor.request_app_shutdown()
        return _api_return(True, translate_like_payload(None, "backend.api.shutdownStarted"))

    @app.get("/api/task/get_registered_tasks")
    def get_registered_tasks():
        """获取所有已注册的任务"""
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), processor.task_queue.get_task_list())

    @app.post("/api/task/disable/{task_name:str}")
    def disable_task(task_name):
        """
        禁用任务
        :param task_name: 任务id
        :return:
        """
        new_config = processor.config_service().base.disabled_tasks.value.append(task_name)
        processor.config_service.save_config(new_config)
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), processor.task_queue.disable_task(task_name))

    @app.post("/api/task/enable/{task_name:str}")
    def enable_task(task_name):
        """
        启用任务
        :param task_name: 任务id
        :return:
        """
        new_config = processor.config_service().base.disabled_tasks.value.remove(task_name)
        processor.config_service.save_config(new_config)
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), processor.task_queue.enable_task(task_name))

    # @app.get("/api/debug/switch_yolo_model/{model_name:str}")
    # def switch_yolo_model(model: str):
    #     model_list = ["base_ui", "producer"]
    #     if model.lower() not in model_list:
    #         return _api_return(False, "Invalid model name")
    #     processor.yolo_engine.load_model(model.upper())
    #     return _api_return(True, f"model switched to {model}")

    @app.get("/api/config")
    def get_all_config():
        """
        获取所有配置
        :return:
        """
        config = processor.config_service()
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), config.to_json_dict())

    @app.get("/api/config/tools/reset_config")
    def reset_config():
        """
        重置所有配置
        :return:
        """
        processor.config_service.reset_config()
        return get_all_config()

    @app.get("/api/config/tools/refresh_ddm_token")
    def refresh_ddm_token():
        """刷新DDMPlayer Token"""
        ddm_cfg = processor.config_service().dmm_player
        try:
            result = extract_gakumas_launch_parameters()
            ddm_cfg.game_exe_path.value = result.exe_path
            ddm_cfg.viewer_id.value = result.viewer_id
            ddm_cfg.open_id.value = result.open_id
            ddm_cfg.pf_token.value = result.pf_token
            processor.config_service.save_config()
        except Exception as e:
            logger.exception("刷新DDM Token失败")
            return _api_return(False, translate_like_payload(i18n_text("backend.api.refreshDmmTokenFailed", fallback="提取游戏启动参数失败，请检查游戏是否正在运行"), "backend.api.refreshDmmTokenFailed"))
        return _api_return(True, translate_like_payload(None, "backend.api.ok"))

    @app.get("/api/config/{task_name:str}")
    def get_task_config(task_name: str):
        """
        获取单个任务配置
        :param task_name: 任务id
        :return:
        """
        if task_name not in processor.task_queue.get_task_list().keys():
            return _api_return(False, translate_like_payload(None, "backend.api.invalidTaskName"))
        all_config = processor.config_service().to_json_dict()
        task_name = f"task__{task_name}"
        if task_name not in all_config.keys():
            return _api_return(False, translate_like_payload(None, "backend.api.taskConfigMissing"))
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), all_config[task_name])

    @app.put("/api/config")
    async def set_all_config(request: Request):
        """
        保存所有任务配置
        :param request:
        :return:
        """
        data = await request.json()
        config = copy(processor.config_service())
        status, errors = config.from_json_dict(data)
        if status:
            processor.config_service.save_config(config)
            return _api_return(True, translate_like_payload(None, "backend.api.ok"), config.to_json_dict())
        else:
            return _api_return(False, translate_like_payload(None, "backend.api.genericError"), {f"{e.section}.{e.field}": e.message for e in errors})

    @app.put("/api/config/{task_name:str}")
    async def set_task_config(request: Request, task_name: str):
        """
        保存单个任务配置
        :param request:
        :param task_name: 任务id
        :return:
        """
        if task_name not in processor.task_queue.get_task_list().keys():
            return _api_return(False, translate_like_payload(None, "backend.api.invalidTaskName"))
        config = copy(processor.config_service())
        all_config = config.to_json_dict()
        task_name = f"task__{task_name}"
        if task_name not in all_config.keys():
            return _api_return(False, translate_like_payload(None, "backend.api.taskConfigMissing"))
        data = await request.json()
        # 合并新的 task 配置
        all_config[task_name] = data
        status, errors = config.from_json_dict(all_config)
        if status:
            processor.config_service.save_config(config)
            return _api_return(True, translate_like_payload(None, "backend.api.ok"), config.to_json_dict()[task_name])
        else:
            return _api_return(False, translate_like_payload(None, "backend.api.genericError"), {f"{e.section}.{e.field}": e.message for e in errors})

    @app.get("/api/resource_update/status")
    def get_resource_update_status():
        """获取资源仓库更新状态"""
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), processor.resource_update_service.get_status())

    @app.post("/api/resource_update/check")
    def check_resource_updates():
        """手动检查资源仓库更新"""
        status, message, data = processor.resource_update_service.manual_check_updates()
        return _api_return(status, message, data)

    @app.post("/api/resource_update/apply")
    def apply_resource_updates():
        """更新资源仓库并重新加载游戏数据库"""
        status, message, data = processor.resource_update_service.apply_updates()
        return _api_return(status, message, data)

    @app.get("/api/adb/devices")
    def get_adb_devices():
        """获取所有ADB设备"""
        try:
            devices = [s.serial for s in adbutils.adb.device_list()]
            return _api_return(True, translate_like_payload(None, "backend.api.ok"), {
                "devices": devices,
                "available": True,
                "message": i18n_text("backend.adb.noError", fallback=""),
            })
        except Exception as exc:
            _, reason = describe_adb_error(exc)
            return _api_return(True, translate_like_payload(None, "backend.api.ok"), {
                "devices": [],
                "available": False,
                "message": reason,
            })

    @app.get("/api/adb/devices/usb")
    def get_adb_usb_serial_list():
        """获取使用USB连接的ADB设备"""
        try:
            serial_list = adbutils.adb.device_list()
            serial_list = [s.serial for s in serial_list if ":" not in str(s.serial)]
            return _api_return(True, translate_like_payload(None, "backend.api.ok"), {
                "devices": serial_list,
                "available": True,
                "message": i18n_text("backend.adb.noError", fallback=""),
            })
        except Exception as exc:
            _, reason = describe_adb_error(exc, connect_mode="USB")
            return _api_return(True, translate_like_payload(None, "backend.api.ok"), {
                "devices": [],
                "available": False,
                "message": reason,
            })

    @app.get("/api/item/list")
    def get_all_items():
        """获取所有物品列表"""
        try:
            items = _get_item_db().get_all_item()
        except RuntimeError as exc:
            logger.warning("获取物品列表失败: {}", exc)
            return _api_return(False, translate_like_payload(None, "backend.api.gameDatabaseNotReady"))
        all_items = []
        for item in items:
            all_items.append({
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "acquisitionRouteDescription": item.acquisitionRouteDescription,
                "translation": {
                    "name": item.localization.name,
                    "description": item.localization.description,
                    "acquisitionRouteDescription": item.localization.acquisitionRouteDescription,
                } if item.localization else {},
                "image": os.path.exists(os.path.join(processor.data_path, f"CLIP/items/{item.id}.png")),
                "gameAssetImage": game_asset_service.has_item_image(item.id),
            })
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), all_items)

    @app.get("/api/idol_card/list")
    def get_all_idol_cards():
        """获取所有偶像卡列表（用于自动培育目标卡浏览器）。"""
        try:
            produce_item_db = _get_produce_item_db()
            cards = _get_idol_card_db().get_all_item()
        except RuntimeError as exc:
            logger.warning("获取偶像卡列表失败: {}", exc)
            return _api_return(False, translate_like_payload(None, "backend.api.gameDatabaseNotReady"))

        def _primary_attribute(card):
            stats = {
                "vocal": getattr(card, "produceVocal", 0) or 0,
                "dance": getattr(card, "produceDance", 0) or 0,
                "visual": getattr(card, "produceVisual", 0) or 0,
            }
            return max(stats, key=stats.get)

        def _character_name(card):
            character = getattr(card, "characterCls", None)
            if character is None:
                return ""
            localization = getattr(character, "localization", None)
            localized_name = f"{getattr(localization, 'lastName', '')}{getattr(localization, 'firstName', '')}".strip()
            if localized_name:
                return localized_name
            return f"{getattr(character, 'lastName', '')}{getattr(character, 'firstName', '')}".strip()

        def _serialize_produce_card(source):
            if source is None:
                return None
            localization = getattr(source, "localization", None)
            asset_key = (getattr(source, "assetId", "") or "").replace("img_general_", "")
            return {
                "id": source.id,
                "name": source.name,
                "assetId": source.assetId,
                "description": _concat_produce_descriptions(getattr(source, "produceDescriptions", []), produce_item_db),
                "translation": {
                    "name": localization.name,
                    "description": _concat_produce_descriptions(getattr(localization, "produceDescriptions", []), produce_item_db),
                } if localization else {},
                "hasImage": bool(asset_key) and game_asset_service.has_skill_card_image(asset_key),
            }

        def _serialize_produce_item(source):
            if source is None:
                return None
            localization = getattr(source, "localization", None)
            asset_key = (getattr(source, "assetId", "") or "").replace("img_general_", "")
            return {
                "id": source.id,
                "name": source.name,
                "assetId": source.assetId,
                "description": _concat_produce_descriptions(getattr(source, "produceDescriptions", []), produce_item_db),
                "translation": {
                    "name": localization.name,
                    "description": _concat_produce_descriptions(getattr(localization, "produceDescriptions", []), produce_item_db),
                } if localization else {},
                "hasImage": bool(asset_key) and os.path.exists(
                    resolve_data_str("game_assets", "items", f"{asset_key}.png")
                ),
            }

        all_cards = []
        for card in cards:
            clip_exists = os.path.exists(os.path.join(processor.data_path, f"CLIP/idol_cards/{card.id}.png"))
            full_image_exists = os.path.exists(
                resolve_data_str("game_assets", "idol_cards_full", f"{card.id}_0.png")
            )
            all_cards.append({
                "id": card.id,
                "name": card.name,
                "assetId": card.assetId,
                "rarity": card.rarity,
                "planType": card.planType,
                "characterId": card.characterId,
                "characterName": _character_name(card),
                "examEffectType": card.examEffectType,
                "isLimited": card.isLimited,
                "primaryAttribute": _primary_attribute(card),
                "produceVocal": card.produceVocal,
                "produceDance": card.produceDance,
                "produceVisual": card.produceVisual,
                "produceVocalGrowthRatePermil": card.produceVocalGrowthRatePermil,
                "produceDanceGrowthRatePermil": card.produceDanceGrowthRatePermil,
                "produceVisualGrowthRatePermil": card.produceVisualGrowthRatePermil,
                "produceStamina": card.produceStamina,
                "translation": {
                    "name": card.localization.name,
                } if card.localization else {},
                "produceCard": _serialize_produce_card(getattr(card, "produceCardCls", None)),
                "beforeProduceItem": _serialize_produce_item(getattr(card, "beforeProduceItemCls", None)),
                "afterProduceItem": _serialize_produce_item(getattr(card, "afterProduceItemCls", None)),
                "hasImage": clip_exists,
                "hasFullImage": full_image_exists,
            })
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), all_cards)

    @app.get("/api/support_card/list")
    def get_all_support_cards():
        """获取所有支援卡列表（含技能描述文本）"""
        try:
            cards = _get_support_card_db().get_all_item()
        except RuntimeError as exc:
            logger.warning("获取支援卡列表失败: {}", exc)
            return _api_return(False, translate_like_payload(None, "backend.api.gameDatabaseNotReady"))

        # 构建支援卡技能描述映射（从 ProduceSkill 关联获取）
        from src.utils.game_database_tools import (
            build_support_card_skill_descriptions,
            build_support_card_event_items,
            build_support_card_level_limits,
            build_support_card_events,
            get_skill_descriptions_at_level,
        )
        try:
            skill_slots_map = build_support_card_skill_descriptions()
        except Exception:
            skill_slots_map = {}
        try:
            event_items_map = build_support_card_event_items()
        except Exception:
            event_items_map = {}
        try:
            level_limits_map = build_support_card_level_limits()
        except Exception:
            level_limits_map = {}
        try:
            events_map = build_support_card_events()
        except Exception:
            events_map = {}

        all_cards = []
        for card in cards:
            clip_exists = os.path.exists(os.path.join(processor.data_path, f"CLIP/support_cards/{card.id}.png"))
            game_asset_exists = game_asset_service.has_support_card_image(card.id)
            game_asset_full_exists = game_asset_service.has_support_card_full_image(card.id)

            # 技能槽位数据（含各等级描述）
            skill_slots = skill_slots_map.get(card.id, [])
            # 等级上限数据（从 SupportCardLevelLimit 获取）
            level_limits = level_limits_map.get(card.supportCardLevelLimitId, [])
            # 默认展示等级：取满突破最高等级
            if level_limits:
                default_level = max(ll["levelLimit"] for ll in level_limits)
            else:
                default_level = 40 if card.rarity != "SupportCardRarity_R" else 30
            skill_descs = get_skill_descriptions_at_level(skill_slots, default_level)

            all_cards.append({
                "id": card.id,
                "name": card.name,
                "type": card.type,
                "planType": card.planType,
                "rarity": card.rarity,
                "assetId": card.assetId,
                "characterIds": card.characterIds,
                "isLimited": card.isLimited,
                "produceCardUpgradePermil": card.produceCardUpgradePermil,
                "produceCardUpgradeLessonParameterType": card.produceCardUpgradeLessonParameterType,
                "levelLimits": level_limits,
                "translation": {
                    "name": card.localization.name,
                } if card.localization else {},
                "skillDescriptions": skill_descs,
                "skillSlots": skill_slots,
                "eventItems": event_items_map.get(card.id, []),
                "events": events_map.get(card.id, []),
                "image": clip_exists,
                "gameAssetImage": game_asset_exists,
                "gameAssetFullImage": game_asset_full_exists,
            })
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), all_cards)

    @app.get("/api/game_asset/status")
    def get_game_asset_status():
        """获取游戏资源下载状态"""
        return _api_return(True, translate_like_payload(None, "backend.api.ok"), {
            "available": game_asset_service._is_gom_available(),
            "downloadedCount": game_asset_service.get_downloaded_card_count(),
            **game_asset_service.get_download_status(),
        })

    @app.post("/api/game_asset/download_support_cards")
    def trigger_download_support_cards():
        """触发下载支援卡缩略图"""
        config = processor.config_service()
        if not config.base.enable_game_asset_download.value:
            return _api_return(False, translate_like_payload(None, "backend.api.imageDownloadDisabled"))

        if not game_asset_service._is_gom_available():
            return _api_return(False, translate_like_payload(None, "backend.api.objectManagerUnavailable"))

        status = game_asset_service.get_download_status()
        if status["downloading"]:
            return _api_return(False, translate_like_payload(None, "backend.api.downloadInProgress"))

        try:
            cards = _get_support_card_db().get_all_item()
        except RuntimeError as exc:
            logger.warning("获取支援卡列表失败: {}", exc)
            return _api_return(False, translate_like_payload(None, "backend.api.gameDatabaseNotReady"))

        game_asset_service.download_support_card_images_async(
            card_db_list=cards,
            clip_manager=processor.clip_manager,
        )
        return _api_return(True, translate_like_payload(None, "backend.api.supportCardThumbDownloadStarted"))

    @app.post("/api/game_asset/download_support_cards_full")
    def trigger_download_support_cards_full():
        """触发下载支援卡全尺寸图片"""
        config = processor.config_service()
        if not config.base.enable_game_asset_download.value:
            return _api_return(False, translate_like_payload(None, "backend.api.imageDownloadDisabled"))

        if not game_asset_service._is_gom_available():
            return _api_return(False, translate_like_payload(None, "backend.api.objectManagerUnavailable"))

        status = game_asset_service.get_download_status()
        if status["downloading"]:
            return _api_return(False, translate_like_payload(None, "backend.api.downloadInProgress"))

        try:
            cards = _get_support_card_db().get_all_item()
        except RuntimeError as exc:
            logger.warning("获取支援卡列表失败: {}", exc)
            return _api_return(False, translate_like_payload(None, "backend.api.gameDatabaseNotReady"))

        from threading import Thread
        Thread(
            target=game_asset_service.download_support_card_full_images,
            args=(cards, False),
            daemon=True,
        ).start()
        return _api_return(True, translate_like_payload(None, "backend.api.supportCardFullDownloadStarted"))

    @app.post("/api/game_asset/download_card_full/{card_id}")
    def download_single_card_full(card_id: str):
        """按需下载单张支援卡全尺寸图片（查看详情时触发）"""
        config = processor.config_service()
        if not config.base.enable_game_asset_download.value:
            return _api_return(False, translate_like_payload(None, "backend.api.imageDownloadFeatureDisabledShort"))
        if not game_asset_service._is_gom_available():
            return _api_return(False, translate_like_payload(None, "backend.api.objectManagerUnavailableShort"))
        if game_asset_service.has_support_card_full_image(card_id):
            return _api_return(True, translate_like_payload(None, "backend.api.downloadAlreadyExists"), {"exists": True})
        try:
            cards = _get_support_card_db().get_all_item()
        except RuntimeError as exc:
            logger.warning("获取支援卡列表失败: {}", exc)
            return _api_return(False, translate_like_payload(None, "backend.api.gameDatabaseNotReady"))
        card = next((c for c in cards if c.id == card_id), None)
        if not card:
            return _api_return(False, translate_like_payload(i18n_text("backend.api.cardNotFound", fallback=f"未找到卡牌：{card_id}", cardId=card_id), "backend.api.cardNotFound"))
        from threading import Thread
        Thread(
            target=game_asset_service.download_single_support_card_full_image,
            args=(card_id, card.assetId),
            daemon=True,
        ).start()
        return _api_return(True, translate_like_payload(None, "backend.api.downloadStarted"))

    @app.post("/api/game_asset/auto_download")
    def trigger_auto_download():
        """批量下载所有支援卡相关图片（在设置页面触发）"""
        config = processor.config_service()
        if not config.base.enable_game_asset_download.value:
            return _api_return(False, translate_like_payload(None, "backend.api.imageDownloadFeatureDisabledShort"))

        if not game_asset_service._is_gom_available():
            return _api_return(False, translate_like_payload(None, "backend.api.objectManagerUnavailableShort"))

        status = game_asset_service.get_download_status()
        if status["downloading"]:
            return _api_return(True, translate_like_payload(None, "backend.api.downloadAlreadyRunning"))

        try:
            cards = _get_support_card_db().get_all_item()
        except RuntimeError as exc:
            logger.warning("获取支援卡列表失败: {}", exc)
            return _api_return(False, translate_like_payload(None, "backend.api.gameDatabaseNotReady"))

        from threading import Thread

        Thread(
            target=game_asset_service.download_all_for_dialog,
            kwargs={"card_db_list": cards, "clip_manager": processor.clip_manager},
            daemon=True,
        ).start()
        return _api_return(True, translate_like_payload(None, "backend.api.supportCardAutoDownloadStarted"))

    app.mount(
        "/assets",
        StaticFiles(directory=resolve_runtime_str("dist", "assets"), html=True),
        name="static",
    )
    clip_image_dir = resolve_data_str("CLIP")
    os.makedirs(clip_image_dir, exist_ok=True)
    app.mount(
        "/api/clip_image",
        StaticFiles(directory=clip_image_dir, html=False),
        name="clip_images",
    )

    game_assets_dir = resolve_data_str("game_assets")
    os.makedirs(game_assets_dir, exist_ok=True)
    app.mount(
        "/api/game_assets",
        StaticFiles(directory=game_assets_dir, html=False),
        name="game_assets",
    )

    @app.get("/")
    def read_index():
        return FileResponse(resolve_runtime_str("dist", "index.html"))
