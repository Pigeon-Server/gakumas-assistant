"""Step 3: 选择难度（Regular / Pro / Master / Legend / NIA Pro / NIA Master）。

在剧本选择页面上直接点击对应难度标签即可进入偶像卡选择。

HAJIME 剧本: Regular(produce-001), Pro(produce-002), Master(produce-003), Legend(produce-006)
NIA 剧本: Pro(produce-004), Master(produce-005)

Legend 难度在 HAJIME 页面第二页（需要向左滑动切换到 Legend 选择页）。
NIA 剧本使用与 HAJIME 不同的难度页面（PRODUCER_NIA 标签），
其中 NIA Pro 和 NIA Master 在同一个 NIA 剧本页面上，
NIA Pro 对应默认选项，NIA Master 需要向下滑动或在列表中选择。
"""

from time import sleep
from typing import TYPE_CHECKING

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.core.exceptions.TaskException import TaskUserMessage
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    click_modal_action_with_retry,
    find_button,
    inertial_swipe,
    wait_frame_stable,
)
from src.core.tasks.producer_challenge.shared.common import ocr_text
from src.entity.Game.Page.Types.index import GamePageTypes
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor

# 剧本 × 难度 → YOLO 标签映射
# HAJIME 难度直接映射到独立标签
# NIA 的 Pro/Master 都在 NIA 标签页内（统一通过 PRODUCER_NIA 进入）
_HAJIME_DIFFICULTY_LABEL_MAP = {
    "regular": BaseUILabels.PRODUCER_REGULAR,
    "pro": BaseUILabels.PRODUCER_PRO,
    "master": BaseUILabels.PRODUCER_MASTER,
    # Legend 没有独立标签——位于 HAJIME 第二页，需要先滑动再点击
}

# 偶像选择页的特征标签
_IDOL_SELECTION_LABELS = (
    BaseUILabels.PRODUCE_CARD_VOCAL,
    BaseUILabels.PRODUCE_CARD_DANCE,
    BaseUILabels.PRODUCE_CARD_VISUAL,
)

MAX_SWIPE_ATTEMPTS = 5


class SelectDifficultyStep(ProduceStep):
    """根据剧本和难度进入对应入口，并等待偶像卡选择页出现。"""

    step_name = "select_difficulty"
    skip_on_resume = True

    def validate(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """确认上下文中已经解析出有效的培育路线。

        Args:
            app: 当前应用处理器；此处不直接使用，但保留统一步骤签名。
            ctx: 培育上下文，要求 `ctx.produce_id` 已在前面步骤或初始化阶段写入。

        Returns:
            bool: 只要 `ctx.produce_id` 不为空就返回 True，表示可以继续执行难度选择。
        """
        return ctx.produce_id is not None

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """根据剧本与目标难度进入对应的偶像卡选择页。

        Args:
            app: 当前应用处理器，用于点击难度标签、滑动页面和等待加载完成。
            ctx: 培育上下文；函数会读取 `scenario` 与 `effective_difficulty`，并依此分派到
                HAJIME 普通难度、HAJIME Legend 或 NIA 的不同处理流程。

        Returns:
            bool: 成功进入偶像卡选择页时返回 True。

        Raises:
            ValueError: 上下文中的剧本标识不受支持时抛出。
        """
        scenario = ctx.scenario.lower()
        difficulty = ctx.effective_difficulty.lower()

        if scenario == "hajime" and difficulty == "legend":
            return self._select_hajime_legend(app, ctx)
        elif scenario == "hajime":
            return self._select_hajime_difficulty(app, ctx, difficulty)
        elif scenario == "nia":
            return self._select_nia_difficulty(app, ctx, difficulty)
        else:
            raise ValueError(f"未知剧本: {scenario!r}")

    def _select_hajime_difficulty(self, app: "AppProcessor", ctx: "ProduceContext", difficulty: str) -> bool:
        """点击 HAJIME 普通难度标签，并处理 AP 恢复后的重试。

        Args:
            app: 当前应用处理器。
            ctx: 当前培育上下文；该参数主要用于保持与其它分支一致的签名。
            difficulty: 目标难度，只允许 `regular`、`pro`、`master`。

        Returns:
            bool: 成功进入偶像卡选择页时返回 True。

        Raises:
            TimeoutError: 难度标签未出现，或多次重试后仍未进入偶像卡选择页时抛出。
            RuntimeError: 标签出现后瞬间消失，无法安全点击时抛出。
        """
        label = _HAJIME_DIFFICULTY_LABEL_MAP[difficulty]
        MAX_RETRIES = 3

        for retry in range(MAX_RETRIES):
            if not app.game_utils.wait_for_label(label, timeout=10):
                raise TimeoutError(f"未检测到难度标签: {label}")

            boxes = app.latest_results.filter_by_label(label)
            if not boxes:
                raise RuntimeError(f"难度标签 {label} 瞬间消失")
            app.game_utils.click_element_and_wait_trigger(boxes.first(), retries=3)

            app.game_utils.wait_loading()
            try:
                return self._wait_idol_selection_page(app, ctx)
            except TimeoutError:
                # AP 恢复后可能回到难度页，检测难度标签存在则重试
                if app.latest_results.exists_label(label):
                    logger.info(f"AP 恢复后回到难度选择页，重新点击 {difficulty} (retry {retry + 1})")
                    continue
                raise

        raise TimeoutError(f"难度选择重试 {MAX_RETRIES} 次仍未进入偶像选择页")

    def _select_hajime_legend(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """通过横向滑动进入 HAJIME 的 Legend 难度页并点击进入。

        Args:
            app: 当前应用处理器，用于执行滑动和点击 Legend 按钮。
            ctx: 当前培育上下文；此分支暂不直接读取额外字段，但保留统一签名。

        Returns:
            bool: 成功从 Legend 页面进入偶像卡选择页时返回 True。

        Raises:
            TimeoutError: 多次滑动后仍未找到 Legend 页面时抛出。
        """
        app.game_utils.wait_location_update(GamePageTypes.HOME_TAB.PRODUCER)
        h, w = app.latest_frame.shape[:2]
        cy = h // 2

        # 先检查当前是否已经在 Legend 页面
        if self._is_legend_page_visible(app):
            logger.debug("已经在 Legend 难度页面")
            return self._click_legend_and_enter(app, ctx)

        # 从 HAJIME 页面向左滑动到 Legend
        for attempt in range(MAX_SWIPE_ATTEMPTS):
            logger.debug(f"尝试向左滑动到 Legend ({attempt + 1}/{MAX_SWIPE_ATTEMPTS})")
            inertial_swipe(app, w * 3 // 4, cy, w // 4, cy)
            if self._is_legend_page_visible(app):
                return self._click_legend_and_enter(app, ctx)

        raise TimeoutError(f"滑动 {MAX_SWIPE_ATTEMPTS} 次后仍未找到 Legend 页面")

    def _is_legend_page_visible(self, app: "AppProcessor") -> bool:
        """检测 Legend 页面——不存在 HAJIME 普通难度标签，且存在包含「レジェンド」文本的按钮。"""
        # 如果存在普通难度标签则不在 Legend 页
        hajime_labels = (
            BaseUILabels.PRODUCER_REGULAR,
            BaseUILabels.PRODUCER_PRO,
            BaseUILabels.PRODUCER_MASTER,
        )
        if any(app.latest_results.exists_label(lbl) for lbl in hajime_labels):
            return False

        return find_button(app, ProduceText.LEGEND, fuzz_threshold=70, use_contains=False) is not None

    def _click_legend_and_enter(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """在 Legend 页面点击「レジェンド」按钮进入偶像卡选择。"""
        legend_btn = find_button(app, ProduceText.LEGEND, fuzz_threshold=70, use_contains=False)
        if legend_btn is None:
            raise RuntimeError("Legend 按钮消失")

        app.game_utils.click_element_and_wait_trigger(legend_btn, retries=3)
        app.game_utils.wait_loading()
        return self._wait_idol_selection_page(app, ctx)

    def _select_nia_difficulty(self, app: "AppProcessor", ctx: "ProduceContext", difficulty: str) -> bool:
        """NIA 剧本 Pro/Master 选择。

        NIA 页面统一通过 PRODUCER_NIA 标签进入。
        Pro 是默认（上方）选项，Master 在下方。
        使用 OCR 在按钮列表中匹配对应文本。
        """
        # NIA 页面上 Pro/Master 以按钮形式展示
        nia_diff_text = {
            "pro": "プロ",
            "master": "マスター",
        }

        target_text = nia_diff_text[difficulty]

        if not app.game_utils.wait_for_label(BaseUILabels.PRODUCER_NIA, timeout=10):
            raise TimeoutError("未检测到 NIA 标签")

        # 尝试匹配目标难度按钮
        for _ in range(10):
            diff_btn = find_button(app, target_text, fuzz_threshold=70, use_contains=False)
            if diff_btn is not None:
                app.game_utils.click_element_and_wait_trigger(diff_btn, retries=3)
                app.game_utils.wait_loading()
                return self._wait_idol_selection_page(app, ctx)
            sleep(1)

        # 回退：NIA 只有一个标签时，直接点击
        logger.warning(f"NIA 难度按钮 '{target_text}' 未找到，尝试直接点击 NIA 标签")
        boxes = app.latest_results.filter_by_label(BaseUILabels.PRODUCER_NIA)
        if boxes:
            app.game_utils.click_element_and_wait_trigger(boxes.first(), retries=3)
            app.game_utils.wait_loading()
            return self._wait_idol_selection_page(app, ctx)

        raise TimeoutError(f"NIA 难度 '{difficulty}' 选择失败")

    @staticmethod
    def _wait_idol_selection_page(app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """等待偶像卡选择页出现，自动处理 AP 不足弹窗。
        AP 恢复后若仍在难度选择页，抛出 TimeoutError 由外层重试。"""
        ap_recovered = False
        for attempt in range(20):
            # 检查是否已到偶像选择页
            if any(app.latest_results.exists_label(lbl) for lbl in _IDOL_SELECTION_LABELS):
                wait_frame_stable(app, timeout=2.5)
                logger.debug("成功进入偶像卡选择页")
                return True

            # AP 恢复后回到了难度选择页 → 让外层重新点击难度
            if ap_recovered:
                diff_labels = list(_HAJIME_DIFFICULTY_LABEL_MAP.values())
                if any(app.latest_results.exists_label(lbl) for lbl in diff_labels):
                    logger.info("AP 恢复后仍在难度选择页，需重新点击难度")
                    raise TimeoutError("AP 恢复后回到难度选择页")

            # 检查 AP 不足弹窗
            text = ocr_text(app.latest_frame)
            if ProduceText.AP_SHORTAGE in text:
                if not getattr(ctx, "allow_ap_recovery", True):
                    logger.warning("检测到 AP 不足弹窗，但用户配置禁止自动恢复")
                    SelectDifficultyStep._cancel_ap_recovery_and_return_home(app)
                logger.info("检测到 AP 不足弹窗，尝试自动恢复")
                SelectDifficultyStep._handle_ap_shortage(app)
                ap_recovered = True
                continue

            sleep(1)

        raise TimeoutError("等待偶像卡选择页超时")

    @staticmethod
    def _cancel_ap_recovery_and_return_home(app: "AppProcessor") -> None:
        """拒绝 AP 恢复弹窗，并返回主页终止本次培育。"""
        modal = app.game_utils.try_get_modal(no_body=True, require_header=False)
        cancelled = False
        if modal is not None:
            cancelled = click_modal_action_with_retry(
                app,
                modal,
                prefer_confirm=False,
                retries=2,
                timeout=4.0,
                action_name="select_difficulty ap recovery cancel",
            )
        else:
            for text in (ProduceText.AP_CANCEL, ButtonText.CLOSE):
                button = find_button(app, text, fuzz_threshold=60)
                if button is None:
                    continue
                cancelled = app.game_utils.click_element_and_wait_trigger(
                    button,
                    retries=2,
                    timeout=3.0,
                )
                if not cancelled:
                    app.device.click_element(button)
                    cancelled = True
                break

        if not cancelled:
            raise TimeoutError("未能取消 AP 恢复弹窗")

        wait_frame_stable(app, timeout=2.0)
        try:
            app.game_utils.go_home()
        except Exception as exc:
            raise TaskUserMessage("用户配置禁止 AP 恢复，且取消后返回主页失败") from exc
        raise TaskUserMessage("用户配置禁止 AP 恢复，已取消并返回主页")

    @staticmethod
    def _handle_ap_shortage(app: "AppProcessor") -> None:
        """处理 AP 不足弹窗的完整流程：
        1. AP 不足 → 点击 回復する
        2. AP 回復 → 点击 使う（APドリンク旁的按钮）
        3. 确认对话框 → 点击确认
        4. 回復完了 → 点击确认
        
        注意: 两层弹窗可能叠加显示，OCR 同时包含 AP不足 和 AP回復 文本，
        需要通过按钮位置来区分当前应该操作的弹窗层。
        """
        from src.core.inference.ocr_backends.factory import create_ocr_backend
        ocr = create_ocr_backend()
        MAX_STEPS = 15

        for step in range(MAX_STEPS):
            sleep(1.5)
            frame = app.latest_frame
            if frame is None:
                continue

            # 用 OCR 获取文本和位置
            result = ocr.infer(frame)
            texts = result.txts
            boxes = result.boxes
            full_text = "".join(texts)

            # 查找 OCR 文本中特定关键词的位置
            def find_ocr_center(keyword):
                """在 OCR 结果中查找包含关键词的文本框中心。"""
                for t, b in zip(texts, boxes):
                    if keyword in t:
                        cx = (b[0][0] + b[2][0]) / 2
                        cy = (b[0][1] + b[2][1]) / 2
                        return int(cx), int(cy)
                return None

            yolo_buttons = app.latest_results.filter_by_label(BaseUILabels.BUTTON)

            # 阶段优先级：确认消费 > AP恢复（使う）> AP不足（回復する）。
            # 三层弹窗可能同时显示，确认消费弹窗在最底层(cy最大)，需要优先处理

            # 阶段 3：确认消费对话框（“消費して / 回復しますか”）。
            if "消費して" in full_text or "回復しますか" in full_text:
                # 用 OCR 找最下方“回復する”按钮（确认消费弹窗底部）。
                recover_positions = []
                for t, b in zip(texts, boxes):
                    if "回復する" in t:
                        cy = (b[0][1] + b[2][1]) / 2
                        cx = (b[0][0] + b[2][0]) / 2
                        recover_positions.append((int(cx), int(cy)))
                if recover_positions:
                    # 选 cy 最大的按钮（最底部弹窗的“回復する”）。
                    pos = max(recover_positions, key=lambda p: p[1])
                    logger.debug(f"AP消費確認: 点击回復する OCR位置 {pos}")
                    app.device.click(pos[0], pos[1], el_label="回復する")
                    sleep(2)
                    continue

            # 阶段 2：AP恢复弹窗出现 → 点击“使う”。
            if ProduceText.AP_RECOVERY in full_text and ProduceText.AP_DRINK in full_text:
                pos = find_ocr_center("使う")
                if pos:
                    logger.debug(f"AP回復: 点击使う OCR位置 {pos}")
                    app.device.click(pos[0], pos[1], el_label="使う")
                    sleep(2)
                    continue
                # 如果没找到“使う”文本，则找“閉じる”退出。
                close_pos = find_ocr_center("閉じる")
                if close_pos:
                    logger.warning("AP回復: 未找到使う按钮，点击閉じる退出")
                    app.device.click(close_pos[0], close_pos[1], el_label="閉じる")
                    sleep(1.5)
                    return

            # 阶段 1：仅出现 AP 不足（尚未打开 AP 恢复弹窗）→ 点击“回復する”。
            if ProduceText.AP_SHORTAGE in full_text and ProduceText.AP_RECOVERY not in full_text:
                pos = find_ocr_center(ProduceText.AP_RECOVER_BUTTON)
                if pos:
                    logger.debug(f"AP不足: 点击回復する OCR位置 {pos}")
                    app.device.click(pos[0], pos[1], el_label=ProduceText.AP_RECOVER_BUTTON)
                    sleep(1.5)
                    continue

            # 阶段: 回復完了 → 有按钮就点
            if ProduceText.AP_RECOVERY_COMPLETE in full_text or ProduceText.AP_RECOVERY_DONE in full_text:
                if yolo_buttons:
                    btn = yolo_buttons.first()
                    logger.debug(f"AP回復完了: 点击确认 ({btn.cx:.0f}, {btn.cy:.0f})")
                    app.device.click_element(btn)
                    sleep(1.5)
                    continue

            # 如果没有 AP 相关文本，已恢复完毕
            if ProduceText.AP_SHORTAGE not in full_text and ProduceText.AP_RECOVERY not in full_text:
                logger.info("AP 恢复完毕，回到正常流程")
                return

        logger.warning("AP 恢复步骤超过上限，继续等待")
