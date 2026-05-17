"""Step 12: 处理培育结果画面并返回主页。

培育结束后游戏会依次展示（实际顺序基于连调实测）：
  1. フォト選択画面 — 选择照片（支持 VL 模型自动选卡面），点击「次へ」
  2. メモリー生成画面 — 圆形「生成」按钮（YOLO 无法检测）
  3. 生成动画 / Loading
  4. プロデュース評価 — 评价得分，TAP 推进
  5. メモリーカード展示 — TAP 推进（可能多页）
  6. スキルカード展示 — TAP 推进（可能多页）
  7. メモリー生成完了 — TAP 推进
  8. 結果汇总页 — 有 Confirm / Universal 按钮
  9. 成就 / 奖励进度页 — 有 Confirm 按钮（可能多页）
  10. 完了する / プロデュース履歴 — 必须点击「完了する」（左侧按钮）
  11. 受取完了弹窗 — 点击按钮关闭
  12. イベントPt 页 — 点击按钮推进
  13. Loading → 回到主页

注意：
  - 很多中间画面没有 YOLO 标签，只能靠 OCR 识别
  - 「生成」按钮是圆形设计，YOLO 不会检测到
  - 最终页有两个 Universal button 并排，必须用 OCR 区分
"""

from time import sleep, time
from typing import TYPE_CHECKING

import cv2

from src.constants.game.text.button_text import ButtonText
from src.constants.game.text.produce_text import ProduceText
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.yolo.labels.producer_Labels import ProducerLabels
from src.constants.yolo.model_type import YoloModelType
from src.core.inference.ocr_engine import OCRService
from src.constants.game.producer_gameplay import GameplayPosition
from src.core.tasks.producer_challenge.context import GameplayPhase
from src.core.tasks.producer_challenge.gameplay.strategy.llm_strategy import LLMStrategy
from src.core.tasks.producer_challenge.shared.common import (
    click_relative_point,
    ocr_text,
)
from src.core.tasks.producer_challenge.steps.base import ProduceStep
from src.core.tasks.producer_challenge.ui import (
    collect_frame_text,
    click_modal_action_with_retry,
    detect_gameplay_state,
    find_button,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.main import AppProcessor


# ── OCR 关键词 ──────────────────────────────────────────
# 照片选择画面（记忆卡面照片选择）关键词
_PHOTO_SELECT_KEYWORDS = (
    ProduceText.MEMORY_PHOTO_SELECT,  # 请选择要作为记忆的照片。
    ProduceText.MEMORY_PHOTO_SELECT_SHORT,
)

# 需要 TAP 推进的画面关键词
_TAP_SCREEN_KEYWORDS = (
    ProduceText.MEMORY_GENERATION_COMPLETE,  # メモリー生成完了（记忆卡）
    ProduceText.TAP,
    ProduceText.PRODUCE_EVALUATION,          # 培育评价得分
)

# 结果链相关关键词（用于判断是否仍在结果流程中）
_RESULT_CHAIN_KEYWORDS = (
    ProduceText.MEMORY_GENERATION_COMPLETE,
    ProduceText.MEMORY_SELECT,
    ProduceText.MEMORY_PHOTO_SELECT,
    ProduceText.PRODUCE_RESULT,
    ProduceText.ACHIEVEMENT_PROGRESS,
    ProduceText.EVENT_REWARD_PROGRESS,
    ProduceText.EVENT_POINT,
    ProduceText.FAILED,
    ProduceText.FINAL_PRODUCE_EVALUATION,
    ProduceText.REWARD_ITEMS,
    ProduceText.MEMORY_OCR_VARIANTS[1],
    ProduceText.PRODUCE,
    ProduceText.RECEIVE_COMPLETE,
    ProduceText.REWARD,
    ButtonText.COMPLETE,
    ProduceText.EVENT,
    ProduceText.FINAL_EXAM,
    ProduceText.MEMORY_PHOTO_SELECT_SHORT,
)

# “完了する”页面的 OCR 标识。
_COMPLETE_PAGE_KEYWORDS = (ButtonText.COMPLETE,)
# “プロデュース履歴”页面的 OCR 标识。
_HISTORY_PAGE_KEYWORD = ProduceText.PRODUCE_HISTORY

_RESULT_DETAIL_KEYWORDS = (ButtonText.BACK, ProduceText.RESULT_RESTORED)
_RESULT_DETAIL_LABELS = {
    ProducerLabels.PC_PROGRESS,
    ProducerLabels.PC_P_POINT,
    ProducerLabels.PC_ACTION_INFO,
    ProducerLabels.SKILL_CARD_ACTIVE,
    ProducerLabels.SKILL_CARD_MENTAL,
    ProducerLabels.SKILL_CARD_TRAP,
    ProducerLabels.SKILL_CARD_INFO,
    ProducerLabels.P_DRINK,
}
_RESULT_OCR = OCRService()


class HandleResultsStep(ProduceStep):
    """处理培育结算链路，并在必要时恢复 gameplay 或最终回到主页。"""

    step_name = "handle_results"

    @staticmethod
    def _flush_llm_session(ctx: "ProduceContext") -> None:
        """结果链完成后补做一次最终会话 flush。"""
        strategy = getattr(ctx, "schedule_strategy", None)
        if isinstance(strategy, LLMStrategy):
            strategy.flush_session(ctx)

    def execute(self, app: "AppProcessor", ctx: "ProduceContext") -> bool:
        """推进培育结果链，必要时恢复 gameplay，最终确保回到主页。

        结果链既可能一路结算回主页，也可能在考试失败后通过「再挑戦」重新回到
        gameplay。为此本步骤会在 PRODUCER 模型下持续推进结果页面，识别当前是
        继续结算、恢复到 gameplay，还是已经返回主页；若恢复到了 gameplay，
        会重新调用 `ProduceGameplayLoopStep` 继续跑到下一次退出点。

        Args:
            app: 当前应用处理器，用于点击结果链按钮、切换模型与回主页。
            ctx: 培育上下文；本步骤会更新 gameplay phase / position，必要时依赖
                `handler_state` 判断主循环是否已经完成收尾。

        Returns:
            bool: 成功回到主页时返回 True；若恢复 gameplay 或手动回主页失败，会返回 False
                或抛出异常。
        """
        try:
            # 如果主循环已通过 produce_finishing 推进结果链并确认到主页，直接跳过
            if ctx.handler_state.get("produce_finishing"):
                logger.info("HandleResults: 主循环已完成培育收尾，跳过 step 12")
                logger.success("培育完成，已返回主页")
                return True

            ctx.gameplay_phase = "results"
            logger.info("开始处理培育结果画面")
            post_state: tuple[str, str, str] | None = None

            # 支持多次 retry 循环：考试失败→再挑戦→考试→再次失败→再挑戦…
            _MAX_RESULT_RETRIES = 5
            for attempt in range(_MAX_RESULT_RETRIES):
                # 阶段一：在 PRODUCER 模型下跳过结果页面
                post_state = self._skip_result_screens(app, ctx, timeout=120)

                # 阶段二：切回 BASE_UI 模型
                logger.info("切换回 BASE_UI 模型")
                app.switch_yolo_model(YoloModelType.BASE_UI, settle_seconds=2.0)

                if post_state is None:
                    post_state = self._detect_post_result_state(app, ctx)
                if post_state[0] == "resume":
                    phase, position = post_state[1], post_state[2]
                    logger.info(f"结果链已回到 gameplay，恢复主循环: phase={phase}, position={position}")
                    if hasattr(ctx, "set_phase"):
                        ctx.set_phase(phase)
                    else:
                        ctx.gameplay_phase = phase
                    if hasattr(ctx, "set_position"):
                        ctx.set_position(position)
                    else:
                        ctx.gameplay_position = position
                    from src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop import (
                        ProduceGameplayLoopStep,
                    )

                    logger.info("切换回 PRODUCER 模型用于 gameplay loop")
                    app.switch_yolo_model(YoloModelType.PRODUCER, settle_seconds=1.5)
                    if not ProduceGameplayLoopStep().execute(app, ctx):
                        logger.error("恢复 gameplay 主循环失败")
                        return False
                    logger.info(f"gameplay loop 再次退出 (attempt {attempt + 1})，继续处理结果")
                    continue

                if post_state[0] == "home":
                    break
                break

            if post_state is None:
                post_state = ("result", "", "")

            # 阶段三：等待回到主页（处理残余弹窗）
            if post_state[0] != "home" and not self._wait_for_home(app, timeout=40):
                logger.warning("未自动回到主页，尝试手动导航")
                try:
                    app.game_utils.go_home(max_try=5)
                except RuntimeError:
                    logger.error("返回主页失败")
                    return False

            logger.success("培育完成，已返回主页")
            return True
        finally:
            self._flush_llm_session(ctx)


    # ── 阶段一：结果链推进 ─────────────────────────────────

    @staticmethod
    def _skip_result_screens(
        app: "AppProcessor",
        ctx: "ProduceContext | None" = None,
        timeout: int = 120,
    ) -> tuple[str, str, str] | None:
        """处理结果链页面推进，并在必要时返回恢复状态。

        Args:
            app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
            ctx: 培育上下文对象，保存跨步骤状态与策略配置。
            timeout: 结果链推进阶段的最长等待时间。

        Returns:
            tuple[str, str, str] | None: 若检测到可恢复 gameplay 或已回主页，返回状态元组；
                否则返回 None 表示交给后续状态探测兜底。
        """
        start = time()
        consecutive_no_progress = 0
        same_text_count = 0           # 连续相同 OCR 文本计数（检测画面卡住）
        last_text_hash = ""           # 上一次 OCR 文本摘要
        retry_click_count = 0         # 连续点击「再挑戦」次数（检测 ticket 用尽）
        MAX_NO_PROGRESS = 15          # 连续无法推进则退出
        MAX_SAME_TEXT = 8             # 同一画面持续 N 次则尝试强制推进
        MAX_RETRY_CLICKS = 3          # 连续点击“再挑戦”无效，可能 ticket 用尽，改点“次へ”。

        while time() - start < timeout:
            sleep(1.0)
            results = app.latest_results
            frame = app.latest_frame
            if results is None or frame is None:
                consecutive_no_progress += 1
                if consecutive_no_progress >= MAX_NO_PROGRESS:
                    break
                continue

            # ── 快速主页检测（PRODUCER 模型偶尔也能看到 Tab 标签）──
            if results.exists_label(BaseUILabels.TAB_HOME):
                logger.debug("结果链: 检测到主页标签，退出")
                break

            # ── OCR 识别当前画面 ──
            frame_text = ocr_text(frame)
            labels = [r.label for r in results]
            # 缩短文本用于日志（最多80字符）
            short_text = frame_text.replace("\n", " ")[:80]
            logger.debug(f"结果链: OCR=[{short_text}] YOLO={labels}")

            post_state = HandleResultsStep._detect_post_result_state(
                app,
                ctx,
                frame_text=frame_text,
                labels=labels,
            )
            if post_state[0] == "home":
                logger.debug("结果链: 检测到主页，提前退出")
                return post_state
            if post_state[0] == "resume":
                logger.info(
                    "结果链: 检测到已回到 gameplay，停止结果链推进: "
                    f"phase={post_state[1]}, position={post_state[2]}"
                )
                return post_state

            # ── 同一画面检测（防止死循环）──
            text_hash = frame_text.replace(" ", "").replace("\n", "")[:100]
            if text_hash == last_text_hash and text_hash:
                same_text_count += 1
            else:
                same_text_count = 0
                last_text_hash = text_hash

            # 画面长时间无变化 → 强制尝试关闭/退出
            if same_text_count >= MAX_SAME_TEXT:
                logger.warning(f"结果链: 画面卡住 {same_text_count} 次，尝试强制推进")
                # 先尝试 Close Button
                close_boxes = results.filter_by_label(ProducerLabels.CLOSE_BUTTON)
                if close_boxes:
                    app.device.click_element(close_boxes.first())
                else:
                    # 点击底部关闭按钮常见位置
                    click_relative_point(app, x_ratio=0.5, y_ratio=0.94, label="stuck-close")
                sleep(2.0)
                same_text_count = 0
                consecutive_no_progress += 1
                if consecutive_no_progress >= MAX_NO_PROGRESS:
                    break
                continue

            # ── 0a. 考试不合格 — 点击「再挑戦」按钮触发再挑战确认弹窗 ──
            if ProduceText.FAILED in frame_text.replace(" ", ""):
                # 多次点击“再挑戦”仍无变化，可能 ticket 用尽，改为点击“次へ”放弃重试。
                if retry_click_count >= MAX_RETRY_CLICKS:
                    logger.warning(
                        f"结果链: 再挑戦已点击 {retry_click_count} 次无效，"
                        "判断 ticket 用尽，改为点击「次へ」放弃重试"
                    )
                    # “次へ”按钮通常在右下角（Confirm button）。
                    confirm_boxes = list(results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
                    if confirm_boxes:
                        app.device.click_element(confirm_boxes[0])
                    else:
                        click_relative_point(app, x_ratio=0.73, y_ratio=0.95, label="next-fallback")
                    retry_click_count = 0
                    sleep(2.0)
                    consecutive_no_progress = 0
                    continue

                retry_btn = find_button(app, ButtonText.RETRY, fuzz_threshold=50)
                if retry_btn:
                    logger.info(
                        f"结果链: 考试不合格，find_button 命中 再挑戦 "
                        f"cx={getattr(retry_btn, 'cx', '?')} cy={getattr(retry_btn, 'cy', '?')} "
                        f"text={getattr(retry_btn, 'text', '?')}"
                    )
                    app.device.click_element(retry_btn)
                    logger.info("结果链: 考试不合格，点击「再挑戦」按钮")
                else:
                    # 兜底：直接用坐标点击左下角「再挑戦」按钮
                    logger.info("结果链: 考试不合格，find_button 未命中，使用坐标兜底")
                    click_relative_point(app, x_ratio=0.27, y_ratio=0.95, label="retry-fallback")
                    logger.info("结果链: 考试不合格，兜底点击左下角区域（再挑戦）")
                retry_click_count += 1
                sleep(2.0)
                consecutive_no_progress = 0
                continue

            # ── 0b. 照片选择画面 — VL 模型自动选择最优卡面 ──
            if HandleResultsStep._is_photo_select_screen(frame_text):
                logger.info("结果链: 检测到フォト選択画面（记忆卡面照片选择）")
                HandleResultsStep._handle_photo_selection(app, frame)
                consecutive_no_progress = 0
                continue

            # ── 1. 「生成」按钮页 — 圆形按钮 YOLO 无法检测 ──
            if HandleResultsStep._is_generate_screen(frame_text, labels):
                logger.debug("结果链: 检测到「生成」按钮页，点击生成")
                gen_btn = find_button(app, ButtonText.GENERATE, fuzz_threshold=50)
                if gen_btn:
                    app.device.click_element(gen_btn)
                else:
                    # 圆形按钮在画面偏下方 y≈0.77
                    click_relative_point(app, x_ratio=0.5, y_ratio=0.77, label="生成-button")
                sleep(3.0)
                consecutive_no_progress = 0
                continue

            # ── 2. “プロデュース履歴”详情页：误触后先关闭（优先于完了检测）──
            if HandleResultsStep._is_history_page(frame_text, labels):
                logger.debug("结果链: 检测到プロデュース履歴详情页，关闭")
                close_boxes = results.filter_by_label(ProducerLabels.CLOSE_BUTTON)
                if close_boxes:
                    app.device.click_element(close_boxes.first())
                else:
                    # Close Button 在底部 y≈0.94（PRODUCER 模型可能检测不到）
                    click_relative_point(app, x_ratio=0.5, y_ratio=0.94, label="history-close")
                sleep(1.5)
                consecutive_no_progress = 0
                continue

            # ── 3. “完了する”页：必须点击左侧按钮。──
            if HandleResultsStep._is_complete_page(frame_text):
                logger.debug("结果链: 检测到「完了する」页面")
                complete_btn = find_button(app, ButtonText.COMPLETE, fuzz_threshold=50)
                if complete_btn:
                    app.device.click_element(complete_btn)
                    logger.debug("结果链: 点击「完了する」按钮")
                else:
                    # 兜底：两个并排按钮时取左侧（cx 较小的）
                    btns = results.filter_by_label(ProducerLabels.BUTTON)
                    confirms = results.filter_by_label(ProducerLabels.CONFIRM_BUTTON)
                    all_btns = list(btns) + list(confirms)
                    if len(all_btns) >= 2:
                        leftmost = min(all_btns, key=lambda b: b.cx)
                        app.device.click_element(leftmost)
                        logger.debug("结果链: 点击左侧按钮（兜底）")
                    elif all_btns:
                        app.device.click_element(all_btns[0])
                    else:
                        # “完了する”按钮通常在左下角。
                        click_relative_point(app, x_ratio=0.27, y_ratio=0.92, label="完了-fallback")
                sleep(2.0)
                consecutive_no_progress = 0
                continue

            # ── 4. TAP 推进画面（评价/展示/生成完了）──
            if HandleResultsStep._is_tap_screen(frame_text):
                logger.debug("结果链: TAP 推进画面")
                click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="result-tap")
                sleep(1.5)
                consecutive_no_progress = 0
                continue

            # ── 5. Skip 按钮 ──
            skip_boxes = results.filter_by_label(ProducerLabels.SKIP_BUTTON)
            if skip_boxes:
                app.device.click_element(skip_boxes.first())
                logger.debug("结果链: 点击 Skip")
                sleep(1.5)
                consecutive_no_progress = 0
                continue

            # ── 6. 快进按钮（剧情过场）──
            ff = results.filter_by_label(ProducerLabels.PLOT_FAST_FORWARD_BUTTON)
            if ff:
                app.device.click_element(ff.first())
                sleep(0.5)
                consecutive_no_progress = 0
                continue

            # ── 7. YOLO 检测到的按钮 ──
            clicked_btn = HandleResultsStep._click_best_button(app, results, frame_text)
            if clicked_btn:
                logger.debug(f"结果链: 点击按钮 [{clicked_btn}]")
                sleep(1.5)
                consecutive_no_progress = 0
                continue

            # ── 8. 结果链内容但无按钮 — TAP 推进 ──
            if HandleResultsStep._text_matches_result_chain(frame_text):
                click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="result-chain-tap")
                logger.debug("结果链: 文本匹配结果链，TAP推进")
                sleep(1.5)
                # 不重置 consecutive_no_progress — 纯文本匹配可能是卡住
                consecutive_no_progress += 1
                if consecutive_no_progress >= MAX_NO_PROGRESS:
                    logger.debug("结果链: 结果链文本匹配但无法推进，退出")
                    break
                continue

            # ── 9. Loading 画面 ──
            normalized_loading_text = frame_text.upper().replace(" ", "")
            if any(token in normalized_loading_text for token in ProduceText.LOADING_TOKENS):
                logger.debug("结果链: Loading 中，等待")
                sleep(2.0)
                consecutive_no_progress = 0
                continue

            # ── 10. 完全无内容 — 可能是过渡动画 ──
            consecutive_no_progress += 1
            if consecutive_no_progress >= MAX_NO_PROGRESS:
                logger.debug("结果链: 连续无法推进，退出")
                break
            click_relative_point(app, x_ratio=0.5, y_ratio=0.5, label="result-empty")
            sleep(1.0)

        elapsed = round(time() - start, 1)
        logger.debug(f"结果链处理完毕，耗时 {elapsed}s")
        return None

    # ── 画面类型判断辅助 ─────────────────────────────────

    @staticmethod
    def _is_photo_select_screen(frame_text: str) -> bool:
        """判断是否是フォト選択画面（记忆卡面照片选择页）。"""
        text = frame_text.replace(" ", "").replace("\n", "")
        return any(kw in text for kw in _PHOTO_SELECT_KEYWORDS)

    @staticmethod
    def _handle_photo_selection(app: "AppProcessor", frame) -> None:
        """处理记忆卡面照片选择——根据配置使用 VL 模型或默认选择第一个。"""
        from src.core.services.config_service import ConfigService

        cfg = ConfigService().items
        mode = str(cfg.task__auto_producer.memory_photo_mode).lower()

        if mode != "vl":
            logger.info("记忆卡面选择: 使用默认模式（第一个），点击「次へ」")
            _click_next_button(app)
            return

        logger.info("记忆卡面选择: 使用 VL 模型自动选择最优卡面")
        prompt = str(cfg.task__auto_producer.memory_photo_vl_prompt).strip()

        # 提取照片缩略图区域
        photo_images = _extract_photo_thumbnails(frame)
        if not photo_images:
            logger.warning("VL: 未能提取照片缩略图，使用默认（第一个）")
            _click_next_button(app)
            return

        logger.info(f"VL: 提取到 {len(photo_images)} 张照片缩略图")

        try:
            from src.core.tasks.producer_challenge.gameplay.llm.vl_client import VLClient
            vl = VLClient()
            chosen_index = vl.select_best_photo(photo_images, prompt)
        except Exception as e:
            logger.error(f"VL 模型调用失败: {e}，回退到默认选择")
            chosen_index = None

        if chosen_index is not None and chosen_index > 0:
            # 点击选中的照片（index=0 已经是默认选中的，不需要点击）
            _click_photo_by_index(app, frame, chosen_index, len(photo_images))
            sleep(1.0)
        elif chosen_index is None:
            logger.info("VL 未返回有效选择，使用默认（第一个）")

        _click_next_button(app)

    @staticmethod
    def _is_generate_screen(frame_text: str, labels: list[str]) -> bool:
        """判断是否是メモリー生成页（圆形「生成」按钮）。
        特征：OCR 只有「生成」或「MEMORY」+「生成」，且无 YOLO 按钮标签。
        """
        text = frame_text.replace(" ", "").replace("\n", "")
        has_generate = ButtonText.GENERATE in text
        has_memory = any(token in text for token in ProduceText.MEMORY_OCR_VARIANTS)
        # 排除「记忆生成完了」（那是 TAP 推进页）
        if ProduceText.MEMORY_GENERATION_COMPLETE in text:
            return False
        if ButtonText.REGENERATE in text:
            return False
        # 排除「再生成」按钮页（有其他按钮）
        has_any_button = any(
            "button" in lbl.lower() or "Button" in lbl
            for lbl in labels
        )
        return has_generate and (has_memory or len(text) < 10) and not has_any_button

    @staticmethod
    def _is_complete_page(frame_text: str) -> bool:
        """判断是否是最终「完了する」页面。
        特征：同时有「完了する」和「プロデュース履歴」或「メモリー変換」文本。
        """
        text = frame_text.replace(" ", "")
        has_complete = ButtonText.COMPLETE in text
        has_history = _HISTORY_PAGE_KEYWORD in text
        has_memory_convert = ProduceText.MEMORY_CONVERT in text
        return has_complete and (has_history or has_memory_convert)

    @staticmethod
    def _is_history_page(frame_text: str, labels: list[str]) -> bool:
        """判断是否是「プロデュース履歴」详情页（误触后进入的页面）。
        特征：有「プロデュース履歴」+ 详情内容（定期公演/編成/PLV等），
        但没有「完了する」按钮。
        注意：PRODUCER 模型可能检测不到 Close Button，所以不依赖 YOLO 标签。
        """
        text = frame_text.replace(" ", "")
        has_history = _HISTORY_PAGE_KEYWORD in text
        has_complete = ButtonText.COMPLETE in text
        # 详情页特有关键词（用于区分“完了する”页面）。
        has_detail = any(kw in text for kw in ProduceText.RESULT_HISTORY_DETAIL_TOKENS)
        return has_history and has_detail and not has_complete

    @staticmethod
    def _is_tap_screen(frame_text: str) -> bool:
        """判断是否是需要 TAP 推进的展示画面。"""
        text = frame_text.replace(" ", "").replace("\n", "")
        for keyword in _TAP_SCREEN_KEYWORDS:
            if keyword in text:
                return True
        return False

    @staticmethod
    def _text_matches_result_chain(frame_text: str) -> bool:
        """判断 OCR 文本是否属于结果链的一部分。"""
        text = frame_text.replace(" ", "")
        if not text:
            return False
        for keyword in _RESULT_CHAIN_KEYWORDS:
            if keyword in text:
                return True
        return False

    @staticmethod
    def _is_result_detail_page(frame_text: str, labels: list[str]) -> bool:
        """识别结果链里的卡片详情/回退页，避免误判成已恢复 gameplay。"""
        text = frame_text.replace(" ", "").replace("\n", "")
        if not text or not any(keyword in text for keyword in _RESULT_DETAIL_KEYWORDS):
            return False
        label_set = set(labels)
        return len(label_set & _RESULT_DETAIL_LABELS) >= 3

    @staticmethod
    def _click_ocr_text(
        app: "AppProcessor",
        frame,
        queries: tuple[str, ...] | list[str],
        *,
        prefer_rightmost: bool = False,
    ) -> str | None:
        """直接点击 OCR 识别到的指定文本，适合 YOLO 缺标签的结果页按钮。"""
        if frame is None:
            return None
        matches = _RESULT_OCR.ocr(frame)
        if hasattr(matches, "search"):
            matches = matches.search(list(queries), None)
        else:
            expected = set(queries)
            matches = [item for item in matches if getattr(item, "text", "") in expected]
        matches = list(matches)
        if not matches:
            return None
        target = max(matches, key=lambda item: item.cx) if prefer_rightmost else min(
            matches,
            key=lambda item: item.cx,
        )
        app.device.click(target.cx, target.cy, label=f"ocr:{target.text}")
        return target.text

    @staticmethod
    def _detect_post_result_state(
        app: "AppProcessor",
        ctx: "ProduceContext | None" = None,
        *,
        frame_text: str | None = None,
        labels: list[str] | None = None,
    ) -> tuple[str, str, str]:
        """判断结果链当前是已回主页、已恢复 gameplay，还是仍在结果流中。

        之所以单独做这层判定，是因为结果链中会混入一些看起来像 gameplay modal 的页面，
        甚至会误触进入结果详情页。如果直接把所有非 UNKNOWN 状态都交回主循环，容易把结果页
        误判成可继续接管的 gameplay，因此这里会先排除结果详情与通用结果弹窗，再决定是否恢复。
        """
        results = app.latest_results
        if results is not None and results.exists_label(BaseUILabels.TAB_HOME):
            return ("home", "", "")

        phase, position = detect_gameplay_state(app, ctx)
        phase_value = getattr(phase, "value", str(phase or ""))
        position_value = getattr(position, "value", str(position or ""))
        if labels is None:
            if results is not None and hasattr(results, "__iter__"):
                labels = [r.label for r in results]
            else:
                labels = []
        if frame_text is None:
            frame_text = collect_frame_text(results) if results is not None else ""
            if not frame_text and getattr(app, "latest_frame", None) is not None:
                frame_text = ocr_text(app.latest_frame)

        if HandleResultsStep._is_result_detail_page(frame_text, labels):
            return ("result", "", "")
        if phase == GameplayPhase.MODAL or phase_value == GameplayPhase.MODAL.value:
            # 结果链中常会插入“早送り確認”“確認”等通用弹窗；
            # 只有明确识别成具体 gameplay modal 时，才交回主循环。
            if (
                position == GameplayPosition.GAMEPLAY_MODAL
                or position_value == GameplayPosition.GAMEPLAY_MODAL
            ):
                return ("result", "", "")
            return ("resume", phase_value, position_value)
        if phase in {
            GameplayPhase.UNKNOWN,
            GameplayPhase.RESULT,
            GameplayPhase.LOADING,
            GameplayPhase.NONE,
        } or phase_value in {
            GameplayPhase.UNKNOWN.value,
            GameplayPhase.RESULT.value,
            GameplayPhase.LOADING.value,
            GameplayPhase.NONE.value,
        }:
            return ("result", "", "")
        return ("resume", phase_value, position_value)

    # ── 按钮点击辅助 ─────────────────────────────────────

    @staticmethod
    def _click_best_button(app: "AppProcessor", results, frame_text: str = "") -> str | None:
        """智能选择并点击最佳按钮。

        优先级：
        1. 如果画面有「完了する」，优先点左侧按钮
        2. OCR 匹配「閉じる」/「次へ」/「確認」 → 精确点击
        3. Universal Confirm button → 直接点击
        4. Universal button → 底部优先
        返回点击的按钮描述，无按钮返回 None。
        """
        # 特殊处理：若 OCR 包含“完了する”，优先点击左侧按钮。
        clean_text = frame_text.replace(" ", "")
        if ButtonText.COMPLETE in clean_text:
            complete_btn = find_button(app, ButtonText.COMPLETE, fuzz_threshold=50)
            if complete_btn:
                app.device.click_element(complete_btn)
                return f"{ButtonText.COMPLETE}(OCR)"
            # 兜底：取最左侧按钮
            btns = list(results.filter_by_label(ProducerLabels.BUTTON)) + \
                   list(results.filter_by_label(ProducerLabels.CONFIRM_BUTTON))
            if len(btns) >= 2:
                leftmost = min(btns, key=lambda b: b.cx)
                app.device.click_element(leftmost)
                return f"{ButtonText.COMPLETE}(左侧兜底)"

        # 尝试 OCR 匹配关键按钮文本
        for text_key, label in (
            (ButtonText.CLOSE, ButtonText.CLOSE),
            (ButtonText.NEXT, ButtonText.NEXT),
            (ButtonText.CONFIRM, ButtonText.CONFIRM),
        ):
            btn = find_button(app, text_key, fuzz_threshold=50)
            if btn:
                app.device.click_element(btn)
                return label

        # Confirm 按钮（底部优先）
        confirms = results.filter_by_label(ProducerLabels.CONFIRM_BUTTON)
        if confirms:
            target = max(list(confirms), key=lambda b: b.cy)
            app.device.click_element(target)
            return f"Confirm@{round(target.cy)}"

        # Universal 按钮（底部优先）
        buttons = results.filter_by_label(ProducerLabels.BUTTON)
        if buttons:
            target = max(list(buttons), key=lambda b: b.cy)
            app.device.click_element(target)
            return f"Button@{round(target.cy)}"

        return None

    # ── 阶段三：等待主页 ─────────────────────────────────

    @staticmethod
    def _wait_for_home(app: "AppProcessor", timeout: int = 40) -> bool:
        """等待回到主页（BASE_UI 模型下检测 Tab: Home 标签）。

        同时处理残余弹窗（受取完了、イベントPt 等）。
        """
        start = time()
        while time() - start < timeout:
            results = app.latest_results
            if results is None:
                sleep(1)
                continue

            # 检测主页底部 tab bar
            if results.exists_label(BaseUILabels.TAB_HOME):
                return True

            # 处理残余弹窗（BASE_UI 模型下 try_get_modal 可靠）
            modal = app.game_utils.try_get_modal(no_body=True)
            if modal:
                click_modal_action_with_retry(
                    app, modal,
                    prefer_confirm=True,
                    retries=2,
                    timeout=3.0,
                    action_name="post-result modal",
                )
                sleep(1)
                continue

            # 通用按钮兜底（可能还有残余结果页未关闭）
            buttons = results.filter_by_label(BaseUILabels.BUTTON)
            if buttons:
                target = max(list(buttons), key=lambda b: b.cy)
                app.device.click_element(target)
                sleep(1.5)
                continue

            sleep(1)
        return False


# ── 照片选择画面辅助函数 ─────────────────────────────────

def _find_photo_grid(frame) -> tuple[list[tuple[int, int]], int, int]:
    """检测フォト選択画面中照片网格的行区间与列范围。

    返回 (rows, x_start, x_end):
        rows   — [(y1, y2), ...] 每行照片的纵坐标范围
        x_start, x_end — 网格的横坐标范围
    """
    import numpy as np

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # 1) 用 OCR 定位标题和底部按钮，确定网格纵向范围
    ocr_results = _RESULT_OCR.ocr(frame)
    title_bottom = int(h * 0.33)  # 默认值
    btn_top = int(h * 0.85)
    for r in ocr_results:
        text = getattr(r, "text", "")
        if (
            ProduceText.MEMORY_PHOTO_SELECT_SHORT in text
            or ProduceText.MEMORY_PHOTO_SELECT_PREFIX in text
        ):
            title_bottom = r.y + r.h + 5
        elif (
            ButtonText.NEXT in text
            or ProduceText.PHOTO_SAVE_TO_DEVICE in text
            or ProduceText.PHOTO_SAVE_TO_ALBUM in text
        ):
            btn_top = min(btn_top, r.y - 10)

    # 2) 在标题下方到按钮上方区域做逐行方差扫描，找照片行
    scan_region = gray[title_bottom:btn_top, int(w * 0.03):int(w * 0.97)]
    threshold = 20
    in_row = False
    rows: list[tuple[int, int]] = []
    start_y = 0
    for y in range(scan_region.shape[0]):
        row_std = float(np.std(scan_region[y, :]))
        if not in_row and row_std > threshold:
            start_y = y
            in_row = True
        elif in_row and row_std < threshold:
            if y - start_y > 30:
                rows.append((title_bottom + start_y, title_bottom + y))
            in_row = False
    if in_row and scan_region.shape[0] - start_y > 30:
        rows.append((title_bottom + start_y, title_bottom + scan_region.shape[0]))

    x_start = int(w * 0.03)
    x_end = int(w * 0.97)
    return rows, x_start, x_end


def _extract_photo_thumbnails(frame) -> list:
    """从フォト選択画面提取照片缩略图列表。

    使用 OCR 定位标题/按钮 → 方差扫描检测行 → 3 等分列 → 过滤空图。
    """
    import numpy as np

    if frame is None:
        return []

    rows, x_start, x_end = _find_photo_grid(frame)
    if not rows:
        logger.warning("VL: 未检测到照片行")
        return []

    total_w = x_end - x_start
    col_w = total_w // 3
    margin = 8

    photos = []
    for y1, y2 in rows:
        for ci in range(3):
            cx1 = x_start + ci * col_w + margin
            cx2 = x_start + (ci + 1) * col_w - margin
            crop = frame[y1 + margin : y2 - margin, cx1:cx2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            if float(np.std(gray)) > 10:
                photos.append(crop)

    return photos


def _click_photo_by_index(app, frame, index: int, total: int) -> None:
    """点击photo、by、index并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。
        frame: 待识别图像帧。
        index: 用于提供index相关输入。
        total: 用于提供total相关输入。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    if frame is None:
        return

    rows, x_start, x_end = _find_photo_grid(frame)
    if not rows:
        logger.warning("VL: 未检测到照片行，无法点击")
        return

    col_w = (x_end - x_start) // 3
    row_idx = index // 3
    col_idx = index % 3

    if row_idx >= len(rows):
        logger.warning(f"VL: 目标行 {row_idx} 超出检测到的行数 {len(rows)}")
        return

    y1, y2 = rows[row_idx]
    cx = x_start + col_idx * col_w + col_w // 2
    cy = (y1 + y2) // 2

    logger.info(f"VL: 点击照片 {index + 1}/{total}，坐标 ({cx}, {cy})")
    app.device.click(cx, cy, label=f"vl-photo-{index + 1}")


def _click_next_button(app) -> None:
    """点击next、按钮并返回结果。

    Args:
        app: 应用处理器实例，负责截图、检测结果访问与点击/滑动交互。

    Returns:
        None: 仅产生副作用，不返回业务值。
    """
    next_btn = find_button(app, ButtonText.NEXT, fuzz_threshold=50)
    if next_btn:
        app.device.click_element(next_btn)
        logger.debug("记忆卡面选择: 点击「次へ」按钮")
    else:
        # 兜底：“次へ”按钮位于底部居中 y≈0.93。
        click_relative_point(app, x_ratio=0.5, y_ratio=0.93, label="photo-next-fallback")
        logger.debug("记忆卡面选择: 兜底点击底部「次へ」位置")
    sleep(2.0)
