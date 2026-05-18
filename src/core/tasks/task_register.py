import time

from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.device.Android.app import Android_App
from src.core.device.windows_compat import is_windows_device
from time import sleep

from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.debug_tools import DebugTools
from src.utils.i18n_tools import i18n_text
from src.utils.logger import logger
from src.utils.ui_message_tools import UIMessage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.main import AppProcessor

GAME_RUNNING = False
debug_tools = DebugTools()


def _wait_until(condition, timeout: float, interval: float = 1.0) -> bool:
    """
    在给定超时时间内轮询条件函数。

    返回 True 表示条件在 timeout 内满足；
    返回 False 表示轮询超时。
    """
    deadline = time.time() + timeout
    while time.time() <= deadline:
        if condition():
            return True
        sleep(interval)
    return False


def _is_startup_screen_ready(processor: "AppProcessor") -> bool:
    """
    判断启动阶段是否已经进入“可继续执行业务任务”的界面。

    只要当前位置不再是 UNKNOWN，或当前出现了可处理模态框，
    就认为启动页已经准备完成。
    """
    latest_results = getattr(processor, "latest_results", None)
    if latest_results is None:
        return False

    current_location = processor.game_utils.update_current_location()
    if current_location and current_location != GamePageTypes.UNKNOWN:
        return True

    return latest_results.exists_label(BaseUILabels.MODAL_HEADER)


def _ensure_windows_game_foreground(processor: "AppProcessor") -> bool:
    """
    尝试将 Windows 端游戏窗口恢复到前台。

    仅在游戏已运行且设备确认为 Windows PC 模式时执行。
    """
    if not is_windows_device(processor.device):
        return False
    if not processor.device.is_app_running():
        return False
    if processor.device.is_app_focused():
        return True

    logger.debug("Game switch to front......")
    processor.device.bring_to_front()
    return _wait_until(processor.device.is_app_focused, timeout=5, interval=0.25)


def register_tasks(processor: "AppProcessor"):
    @processor.task_queue.register_pre_queue_start()
    def _pre__check_adb_connect():
        """
        检查ADB连接并尝试重连
        :return:
        """
        def _check():
            try:
                logger.debug(f"device bool: {bool(processor.device)}, try capture size={processor.device.capture().size}")
                return bool(processor.device) and processor.device.capture().size != 0
            except Exception as e:
                logger.warning(f"screen capture test failed: {e}")
                return False
        if isinstance(processor.device, Android_App):
            MAX_TRY = 3
            TRY_COUNT = 0
            while TRY_COUNT < MAX_TRY:
                if not _check():
                    logger.warning(f"[{TRY_COUNT}]Adb connect disconnect, Try reconnect")
                    processor.create_device_instance()
                else:
                    logger.success(f"Adb reconnection was successful")
                    return True
                sleep(1)
                TRY_COUNT += 1
            logger.error(f"The maximum number of adb reconnections has been reached")
            return False
        return True

    @processor.task_queue.register_pre_queue_start()
    def _pre__resume_yolo_inference():
        if processor.yolo_engine.running:
            return True
        processor.yolo_engine.start()
        processor.yolo_engine.resume()
        return True

    @processor.task_queue.register_pre_queue_start()
    def _pre__wait_frame():
        TIMEOUT = 30
        START_TIME = time.time()
        while True:
            if time.time() - START_TIME > TIMEOUT:
                raise TimeoutError()
            if processor.yolo_engine.latest_frame is None:
                sleep(0.25)
                continue
            if processor.yolo_engine.latest_frame.size != 0:
                break

    @processor.task_queue.register_pre_queue_start()
    def _pre__start_game():
        global GAME_RUNNING
        GAME_RUNNING = False
        _ui_msg = UIMessage()

        if isinstance(processor.device, Android_App):
            if not processor.config_service().base.auto_start_game.value:
                # 未开启自动启动，检查游戏是否已在前台
                if processor.device.is_app_focused():
                    GAME_RUNNING = True
                    return True
                # 游戏不在前台，检查进程是否存在
                if processor.device.is_app_running():
                    _ui_msg.warning(i18n_text("backend.task.gameNotInForeground", fallback="游戏未在前台运行，请手动切回游戏后重试"), timeout=5)
                else:
                    _ui_msg.error(i18n_text("backend.task.gameNotStarted", fallback="游戏未启动，请先手动启动游戏后重试"), timeout=5)
                logger.warning("游戏未在前台，auto_start_game 已关闭，取消任务队列")
                return False

            if processor.device.is_app_focused():
                GAME_RUNNING = True
                return True
            logger.debug("Ensure Android game is foreground......")
            processor.device.start_game()
            return True

        game_running = processor.device.is_app_running()
        logger.debug(f"Game running: {game_running}")
        if game_running:
            GAME_RUNNING = True
            if processor.device.is_app_focused():
                return True

            if _ensure_windows_game_foreground(processor):
                return True

            if not processor.config_service().base.auto_start_game.value:
                _ui_msg.warning(i18n_text("backend.task.gameNotInForeground", fallback="游戏未在前台运行，请手动切回游戏后重试"), timeout=5)
                logger.warning("游戏未在前台，任务队列取消执行")
                return False

            processor.device.start_game()
            return True

        if not processor.config_service().base.auto_start_game.value:
            _ui_msg.warning(i18n_text("backend.task.gameNotStarted", fallback="游戏未启动，请先手动启动游戏后重试"), timeout=5)
            logger.warning("游戏未启动，auto_start_game 已关闭，取消任务队列")
            return False

        processor.device.start_game()
        return _wait_until(processor.device.is_app_running, timeout=120, interval=1)

    @processor.task_queue.register_pre_queue_start()
    def _pre__resume_yolo_engine():
        """恢复Yolo引擎"""
        if processor.yolo_engine.running is False:
            processor.yolo_engine.start()
        processor.yolo_engine.resume()

    @processor.task_queue.register_pre_queue_start()
    def _pre__wait_game_start():
        """等待游戏启动"""
        if not processor.config_service().base.auto_start_game.value:
            # 未开启自动启动时，前面 _pre__start_game 已校验过游戏状态
            return True
        if GAME_RUNNING:
            return True
        logger.debug("wait game start......")
        return _wait_until(lambda: _is_startup_screen_ready(processor), timeout=120, interval=1)

    @processor.task_queue.register_task("start_game", "启动游戏", 3600)
    def _task__start_game(app: "AppProcessor"):
        from src.core.tasks.base_ui.start_game import (
            action__click_start_game,
            action__wait_enter_home,
        )

        sleep(2)
        current_location = app.game_utils.update_current_location()
        if current_location == GamePageTypes.START_GAME:
            action__click_start_game(app)
            app.game_utils.wait_loading()
            action__wait_enter_home(app)
        elif current_location in [GamePageTypes.UNKNOWN, GamePageTypes.LOADING]:
            if current_location == GamePageTypes.LOADING:
                app.game_utils.wait_loading()
            action__wait_enter_home(app)
        app.game_utils.update_current_location()

    @processor.task_queue.register_task("get_expenditure", "获取活动费", 60)
    def _task__get_expenditure(app: "AppProcessor"):
        from src.core.tasks.base_ui.get_expenditure import action__claim_expenditure

        return action__claim_expenditure(app)

    @processor.task_queue.register_task("dispatch_work", "派遣任务", 120)
    def _task__work_dispatch(app: "AppProcessor"):
        from src.core.tasks.base_ui.dispatch_work import (
            action__dispatch_all_available_work,
            ensure__work_dispatch_page_ready,
            handle__work_dispatch_results,
        )
        from src.core.tasks.base_ui.goto_pages import goto__work_dispatch_page

        goto__work_dispatch_page(app)
        handle__work_dispatch_results(app)
        ensure__work_dispatch_page_ready(app)
        action__dispatch_all_available_work(app)

    @processor.task_queue.register_task("get_gift", "获取礼物/邮箱")
    def _task__get_gift(app: "AppProcessor"):
        from src.core.tasks.base_ui.get_gift import (
            action__collect_all_gifts,
            action__has_gift_items,
        )
        from src.core.tasks.base_ui.goto_pages import goto__gift_page

        goto__gift_page(app)
        if action__has_gift_items(app):
            action__collect_all_gifts(app)

    @processor.task_queue.register_task("auto_purchase", "自动每日交换")
    def _task__automated_purchase(app: "AppProcessor"):
        from src.core.tasks.base_ui.auto_purchase import (
            action__daily_exchange,
            action__receive_weekly_gift,
        )
        from src.core.tasks.base_ui.goto_pages import goto__shop_page

        goto__shop_page(app)
        if app.config_service().task__auto_purchase.weekly_gift.value:
            action__receive_weekly_gift(app)
        action__daily_exchange(app)

    @processor.task_queue.register_task("auto_enhancement_support_card", "自动强化支援卡")
    def _task__auto_enhancement_support_card(app: "AppProcessor"):
        from src.core.tasks.base_ui.auto_enhancement_support_card import (
            action__auto_enhance_support_cards,
        )
        from src.core.tasks.base_ui.goto_pages import goto_support_card_list_page

        goto_support_card_list_page(app)
        action__auto_enhance_support_cards(app)

    @processor.task_queue.register_task("auto_contest", "自动每日竞技场")
    def _task__automated_contest(app: "AppProcessor"):
        from src.core.tasks.base_ui.auto_contest import (
            action__check_and_collect_rewards,
            action__loop_challenge_contest,
        )
        from src.core.tasks.base_ui.goto_pages import goto__contest_page

        goto__contest_page(app)
        sleep(3)
        action__check_and_collect_rewards(app)
        action__loop_challenge_contest(app)

    @processor.task_queue.register_task("claim_task_rewards", "领取任务奖励")
    def _task__claim_task_rewards(app: "AppProcessor"):
        from src.core.tasks.base_ui.claim_task_rewards import claim_task_rewards
        from src.core.tasks.base_ui.goto_pages import goto__claim_task_rewards_page

        goto__claim_task_rewards_page(app)
        claim_task_rewards(app)

    @processor.task_queue.register_task("claim_pass_rewards", "领取通行证奖励")
    def _task__claim_pass_rewards(app: "AppProcessor"):
        from src.core.tasks.base_ui.claim_pass_rewards import claim_pass_rewards
        from src.core.tasks.base_ui.goto_pages import goto__claim_pass_rewards

        goto__claim_pass_rewards(app)
        claim_pass_rewards(app)

    @processor.task_queue.register_task("auto_producer", "自动培育（Bata）", -1)
    def _task__auto_producer(app: "AppProcessor"):
        from src.core.tasks.producer_challenge import build_produce_pipeline
        from src.core.tasks.producer_challenge.context import ProduceContext

        cfg = app.config_service().task__auto_producer
        # NIA 使用独立的 nia_difficulty 配置
        if cfg.scenario.value == "nia":
            difficulty = cfg.nia_difficulty.value
        else:
            difficulty = cfg.difficulty.value
        ctx = ProduceContext(
            scenario=cfg.scenario.value,
            difficulty=difficulty,
            target_idol_card_id=cfg.target_idol_card_id.value,
            support_card_mode=cfg.support_card_mode.value,
            support_card_preset_index=cfg.support_card_preset_index.value,
            memory_mode=cfg.memory_mode.value,
            memory_preset_index=cfg.memory_preset_index.value,
            use_rental=cfg.use_rental.value,
            use_boost_items=cfg.use_boost_items.value,
            schedule_notebook_mode=cfg.schedule_notebook_mode.value,
            resume_interrupted=cfg.resume_interrupted.value,
            allow_ap_recovery=cfg.allow_ap_recovery.value,
            allow_destroy_production_data=cfg.allow_destroy_production_data.value,
        )
        base = app.config_service().base

        # 读取3个独立决策源配置
        schedule_backend = str(getattr(base, "schedule_decision_backend", "llm") or "llm")
        battle_backend = str(getattr(base, "battle_decision_backend", "llm") or "llm")
        other_backend = str(getattr(base, "other_decision_backend", "llm") or "llm")

        # 根据配置为每个决策源注入对应策略
        # 统一策略对象：内部根据 phase 路由，互不干扰
        # 1. 周行程决策
        if schedule_backend == "rl_battle":
            from src.core.tasks.producer_challenge.gameplay.strategy.rl_strategy import RLStrategy
            ctx.schedule_strategy = RLStrategy(
                base_url=str(base.rl_inference_base_url),
                predict_timeout=float(base.rl_inference_timeout),
            )
        elif schedule_backend == "algo":
            from src.core.tasks.producer_challenge.gameplay.strategy.algo_strategy import AlgoStrategy
            ctx.schedule_strategy = AlgoStrategy()
        elif schedule_backend == "rl":
            from src.core.tasks.producer_challenge.gameplay.strategy.rl_strategy import RLStrategy
            ctx.schedule_strategy = RLStrategy(
                base_url=str(base.rl_inference_base_url),
                predict_timeout=float(base.rl_inference_timeout),
            )
        else:
            if not ctx.schedule_strategy:
                from src.core.tasks.producer_challenge.gameplay.strategy.llm_strategy import LLMStrategy
                ctx.schedule_strategy = LLMStrategy()

        # 2. 战斗决策（lesson/exam）
        if battle_backend == "rl_battle":
            from src.core.tasks.producer_challenge.gameplay.strategy.rl_strategy import RLStrategy
            battle_strategy = RLStrategy(
                base_url=str(base.rl_inference_base_url),
                predict_timeout=float(base.rl_inference_timeout),
            )
            ctx.lesson_strategy = battle_strategy
            ctx.exam_strategy = battle_strategy
        elif battle_backend == "algo":
            from src.core.tasks.producer_challenge.gameplay.strategy.algo_strategy import AlgoStrategy
            battle_strategy = AlgoStrategy()
            ctx.lesson_strategy = battle_strategy
            ctx.exam_strategy = battle_strategy
        elif battle_backend == "rl":
            from src.core.tasks.producer_challenge.gameplay.strategy.rl_strategy import RLStrategy
            battle_strategy = RLStrategy(
                base_url=str(base.rl_inference_base_url),
                predict_timeout=float(base.rl_inference_timeout),
            )
            ctx.lesson_strategy = battle_strategy
            ctx.exam_strategy = battle_strategy
        else:
            if not ctx.lesson_strategy:
                from src.core.tasks.producer_challenge.gameplay.strategy.llm_strategy import LLMStrategy
                battle_strategy = LLMStrategy()
                ctx.lesson_strategy = battle_strategy
                ctx.exam_strategy = battle_strategy

        # 3. 其他决策（dialogue/p_drink/skill_reward/consult/item_select/modal）
        if other_backend == "algo":
            from src.core.tasks.producer_challenge.gameplay.strategy.algo_strategy import AlgoStrategy
            other_strategy = AlgoStrategy()
            ctx.dialogue_strategy = other_strategy
            ctx.p_drink_strategy = other_strategy
            ctx.skill_reward_strategy = other_strategy
            ctx.consult_strategy = other_strategy
            ctx.item_select_strategy = other_strategy
            ctx.modal_strategy = other_strategy
        elif other_backend == "rl" or other_backend == "rl_battle":
            from src.core.tasks.producer_challenge.gameplay.strategy.rl_strategy import RLStrategy
            other_strategy = RLStrategy(
                base_url=str(base.rl_inference_base_url),
                predict_timeout=float(base.rl_inference_timeout),
            )
            ctx.dialogue_strategy = other_strategy
            ctx.p_drink_strategy = other_strategy
            ctx.skill_reward_strategy = other_strategy
            ctx.consult_strategy = other_strategy
            ctx.item_select_strategy = other_strategy
            ctx.modal_strategy = other_strategy
        else:
            if not ctx.dialogue_strategy:
                from src.core.tasks.producer_challenge.gameplay.strategy.llm_strategy import LLMStrategy
                other_strategy = LLMStrategy()
                ctx.dialogue_strategy = other_strategy
                ctx.p_drink_strategy = other_strategy
                ctx.skill_reward_strategy = other_strategy
                ctx.consult_strategy = other_strategy
                ctx.item_select_strategy = other_strategy
                ctx.modal_strategy = other_strategy

        logger.info(
            "决策后端配置: schedule={}, battle={}, other={}",
            schedule_backend, battle_backend, other_backend,
        )

        pipeline = build_produce_pipeline()
        pipeline.run(app, ctx)

    @processor.task_queue.register_task("void_task", "测试任务", hide=True)
    def _task__void_task(app: "AppProcessor"):
        logger.success("void_task!")
        return True

    @processor.task_queue.register_task("refresh_skill_storage", "刷新技能卡存储", disabled_middleware=True, manual_only=True, allow_manual_resume=True)
    def _task__refresh_skill_storage(app: "AppProcessor"):
        from src.core.tasks.base_ui.refresh_skill_storage import refresh_skill_storage

        refresh_skill_storage(app)

    @processor.task_queue.register_task("learn_support_card_clip", "刷新支援卡存储", manual_only=True)
    def _task__learn_support_card_clip(app: "AppProcessor"):
        from src.core.tasks.base_ui.learn_support_card_clip import action__learn_support_card_clip
        from src.core.tasks.base_ui.goto_pages import goto_support_card_list_page

        goto_support_card_list_page(app)
        action__learn_support_card_clip(app)

    @processor.task_queue.register_task("learn_idol_card_clip", "刷新偶像卡存储", manual_only=True)
    def _task__learn_idol_card_clip(app: "AppProcessor"):
        from src.core.tasks.base_ui.goto_pages import goto_idol_card_list_page
        from src.core.tasks.base_ui.learn_idol_card_clip import action__learn_idol_card_clip

        goto_idol_card_list_page(app)
        action__learn_idol_card_clip(app)
