#!/usr/bin/env python3
"""培育流程连调脚本 — 直接启动 AppProcessor 并运行 produce pipeline。

用法:
  python scripts/run_produce_debug.py                  # 完整 pipeline
  python scripts/run_produce_debug.py --step 11        # 从指定步骤开始
  python scripts/run_produce_debug.py --only-loop      # 仅运行 gameplay loop
  python scripts/run_produce_debug.py --capture         # 截图+YOLO识别
  python scripts/run_produce_debug.py --phase           # 检测当前画面阶段
"""

import argparse
import os
import sys
import time
import traceback

# 将项目根目录加入路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.utils.logger import logger


def init_app():
    """初始化 AppProcessor 并等待推理引擎就绪。"""
    from src.main import AppProcessor

    logger.info("正在初始化 AppProcessor ...")
    app = AppProcessor()

    # 确保资源已初始化
    if not app.is_resource_ready():
        app.ensure_resource_dependencies_initialized()
    assert app.is_resource_ready(), "游戏资源未就绪"

    # 启动推理引擎
    if not app.yolo_engine.running:
        app.yolo_engine.start()
    app.yolo_engine.resume()

    # 等待第一帧
    logger.info("等待首帧图像 ...")
    deadline = time.time() + 15
    while time.time() < deadline:
        if app.latest_frame is not None and app.latest_frame.size > 0:
            break
        time.sleep(0.3)
    else:
        raise TimeoutError("等待首帧超时")

    logger.success(f"AppProcessor 就绪 | 设备: {type(app.device).__name__} | "
                   f"帧大小: {app.latest_frame.shape}")
    return app


def _wait_for_fresh_results(
    app,
    previous_results=None,
    *,
    timeout: float = 3.0,
    poll_interval: float = 0.15,
):
    """等待新一帧推理结果，避免读到模型切换前或上一次命令残留的快照。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        results = app.latest_results
        if results is not None and results is not previous_results:
            return results
        time.sleep(poll_interval)

    fallback_results = app.latest_results
    if fallback_results is not None and fallback_results is not previous_results:
        return fallback_results
    return None


def _ensure_debug_model(
    app,
    *,
    model_type: str,
):
    """切到目标 YOLO 模型，并等待第一份新鲜结果。"""
    previous_results = app.latest_results
    current_model = getattr(app.yolo_engine, "model_type", None)
    if current_model != model_type:
        logger.info(f"切换 YOLO 模型: {current_model} -> {model_type}")
        app.yolo_engine.load_model(model_type)

    results = _wait_for_fresh_results(app, previous_results)
    if results is None:
        raise TimeoutError(f"切换到 {model_type} 后未等到新的推理结果")
    return results


def _collect_results_snapshots(
    app,
    *,
    model_type: str,
    sample_count: int = 3,
    sample_interval: float = 0.35,
):
    """采集多份同模型快照，过滤单帧空结果或切页动画造成的瞬时抖动。"""
    if sample_count <= 0:
        raise ValueError("sample_count 必须大于 0")

    samples = [_ensure_debug_model(app, model_type=model_type)]
    previous_results = samples[-1]

    for _ in range(sample_count - 1):
        time.sleep(sample_interval)
        results = _wait_for_fresh_results(app, previous_results, timeout=2.0)
        if results is None:
            break
        samples.append(results)
        previous_results = results

    return samples


def _select_capture_snapshot(samples):
    """为截图挑选最稳定的一帧：优先非空、优先检测框更多、同分取最新。"""
    if not samples:
        raise ValueError("samples 不能为空")

    best_index = len(samples) - 1
    best_results = samples[best_index]
    best_score = len(best_results) if best_results is not None else -1

    for index, results in enumerate(samples):
        if results is None:
            continue
        score = len(results)
        if score > best_score or (score == best_score and index > best_index):
            best_index = index
            best_results = results
            best_score = score

    return best_results


def _select_phase_probe(probes):
    """phase 探测优先采用最近一次非 unknown 结果，避免被尾部空帧误导。"""
    if not probes:
        raise ValueError("probes 不能为空")

    from src.core.tasks.producer_challenge.context import GameplayPhase

    stable_probes = [probe for probe in probes if probe[0] != GameplayPhase.UNKNOWN]
    return stable_probes[-1] if stable_probes else probes[-1]


def _probe_gameplay_state(
    app,
    ctx,
    *,
    model_type: str,
):
    """基于多份同模型快照探测当前 gameplay state。"""
    from src.core.tasks.producer_challenge.ui import classify_gameplay_state

    sample_count = max(1, int(ctx.handler_state.get("unknown_retry_limit", 2) or 0) + 1)
    sample_interval = max(0.15, float(ctx.handler_state.get("unknown_retry_sleep", 0.4) or 0.0))
    samples = _collect_results_snapshots(
        app,
        model_type=model_type,
        sample_count=sample_count,
        sample_interval=sample_interval,
    )

    probes = []
    for index, results in enumerate(samples, start=1):
        phase, position = classify_gameplay_state(results, ctx=ctx)
        logger.info(
            f"phase probe[{index}/{len(samples)}]: boxes={len(results)}, phase={phase}, position={position}"
        )
        probes.append((phase, position, results))

    return _select_phase_probe(probes)


def _log_yolo_results(results):
    """打印单份 YOLO 结果，方便真机现场快速比对。"""
    if results is None:
        logger.warning("无 YOLO 结果")
        return

    logger.info("=== YOLO 检测结果 ===")
    all_items = list(results)
    for item in all_items:
        label = getattr(item, "label", "?")
        conf = getattr(item, "confidence", 0)
        x, y, w, h = int(item.x), int(item.y), int(item.w), int(item.h)
        logger.info(f"  [{label}] conf={conf:.2f} box=({x},{y},{w},{h})")

    logger.info(f"共 {len(all_items)} 个检测框")


def build_context(app, difficulty_override=None, use_llm=False):
    """根据当前配置构建 ProduceContext。"""
    from src.core.tasks.producer_challenge.context import ProduceContext
    from src.utils.game_database_tools import GakumasDatabase_IdolCardDataUtils

    cfg = app.config_service().task__auto_producer
    # 优先使用参数传入的覆盖值，其次用 app 上保存的命令行覆盖值
    override = difficulty_override or getattr(app, '_debug_difficulty_override', None)
    difficulty = override or (cfg.nia_difficulty.value if cfg.scenario.value == "nia" else cfg.difficulty.value)
    ctx = ProduceContext(
        scenario=cfg.scenario.value,
        difficulty=difficulty,
        target_idol_card_id=cfg.target_idol_card_id.value,
        support_card_mode=cfg.support_card_mode.value,
        support_card_preset_index=int(cfg.support_card_preset_index.value),
        memory_mode=cfg.memory_mode.value,
        memory_preset_index=int(cfg.memory_preset_index.value),
        use_rental=cfg.use_rental.value,
        use_boost_items=cfg.use_boost_items.value,
        resume_interrupted=getattr(app, '_debug_resume', False),
    )
    # 调试脚本支持从中途/only-loop 续跑，先挂上目标偶像卡主库数据作为参数 OCR 种子。
    ctx.selected_idol_card = GakumasDatabase_IdolCardDataUtils().get_by_id(ctx.target_idol_card_id)
    logger.info(f"ProduceContext: {ctx}")
    # 联调要求：遇到未知页面时立刻停下并保留现场，避免脚本继续盲点。
    ctx.handler_state["pause_on_unknown"] = True
    # 真机偶发单帧漏检时，先被动复检几帧；只有连续 unknown 才真正暂停。
    ctx.handler_state["unknown_retry_limit"] = 2
    ctx.handler_state["unknown_retry_sleep"] = 0.4
    # 周行动确认后的剧情切换通常比普通空帧更长，单独给更大的重试预算。
    # 新增 lesson/exam 入场演出后，这段过场实机可能持续 7s 以上，
    # 需要更长窗口才能等到手牌页稳定下来。
    ctx.handler_state["schedule_confirm_unknown_retry_limit"] = 12
    ctx.handler_state["schedule_confirm_unknown_retry_sleep"] = 0.7
    # 剧情快进/跳过后常会接长过场，再进入结果页或下一段事件。
    ctx.handler_state["dialogue_transition_unknown_retry_limit"] = 8
    ctx.handler_state["dialogue_transition_unknown_retry_sleep"] = 0.7
    # 技能卡奖励选卡/确认后也可能接一小段演出，再回到周行动页。
    # 单独拆开设置，便于后续只调这一段，不影响普通 loading。
    ctx.handler_state["skill_reward_transition_unknown_retry_limit"] = 15
    ctx.handler_state["skill_reward_transition_unknown_retry_sleep"] = 1.0
    # loading 后的纯过场可能持续数秒，给下一轮 unknown 更长的等待窗口。
    ctx.handler_state["loading_unknown_retry_limit"] = 15
    ctx.handler_state["loading_unknown_retry_sleep"] = 1.0

    # 注入 LLM 决策策略
    if use_llm:
        llm_url = getattr(app, '_debug_llm_url', None) or "http://192.168.100.10:11434/v1/"
        llm_model = getattr(app, '_debug_llm_model', None) or "gpt-oss:20b"
        from src.core.tasks.producer_challenge.gameplay.llm_strategy import inject_llm_strategy
        inject_llm_strategy(ctx, base_url=llm_url, model=llm_model)

    return ctx


def cmd_capture(app, *, model_type=None):
    """截取当前画面并打印 YOLO 识别结果。"""
    import cv2
    from src.constants.yolo.model_type import YoloModelType

    target_model = model_type or getattr(app.yolo_engine, "model_type", None) or YoloModelType.PRODUCER
    samples = _collect_results_snapshots(
        app,
        model_type=target_model,
        sample_count=3,
        sample_interval=0.3,
    )
    for index, sample in enumerate(samples, start=1):
        logger.info(f"capture probe[{index}/{len(samples)}]: model={target_model}, boxes={len(sample)}")

    results = _select_capture_snapshot(samples)
    frame = getattr(results, "frame", None)
    if frame is None or frame.size == 0:
        frame = app.latest_frame
    if frame is None or frame.size == 0:
        raise RuntimeError("无法获取稳定截图")

    # 保存原始截图
    out_dir = os.path.join(ROOT, "out", "debug_captures")
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    raw_path = os.path.join(out_dir, f"capture_{ts}.png")
    cv2.imwrite(raw_path, frame)
    logger.info(f"截图保存: {raw_path}")
    logger.info(f"capture selected: model={target_model}, boxes={len(results)}")
    _log_yolo_results(results)


def cmd_phase(app, *, model_type=None):
    """检测当前画面的 gameplay phase 和 position。"""
    from src.constants.yolo.model_type import YoloModelType
    from src.entity.Game.Components.Button import ButtonList

    ctx = build_context(app)
    target_model = model_type or YoloModelType.PRODUCER
    phase, position, results = _probe_gameplay_state(app, ctx, model_type=target_model)
    logger.info(
        f"当前画面阶段: phase={phase}, position={position}, model={target_model}, boxes={len(results)}"
    )

    # 打印按钮文本
    buttons = ButtonList(results)
    if buttons:
        logger.info("=== 按钮列表 ===")
        for btn in buttons:
            logger.info(f"  [{btn.text}] pos=({btn.cx},{btn.cy})")


def cmd_run_full(app, start_step=1, use_llm=False):
    """运行完整 pipeline（可从指定步骤开始）。"""
    from src.core.tasks.producer_challenge import build_produce_pipeline
    
    ctx = build_context(app, use_llm=use_llm)
    pipeline = build_produce_pipeline()

    if start_step > 1:
        logger.info(f"跳过前 {start_step - 1} 步")
        pipeline.steps = pipeline.steps[start_step - 1:]

        # 跳过的步骤可能包含模型切换（handle_startup_modals → PRODUCER）
        # gameplay loop 需要 PRODUCER 模型，确保已切换
        from src.constants.yolo.model_type import YoloModelType
        app.yolo_engine.load_model(YoloModelType.PRODUCER)
        import time as _time
        _time.sleep(1.5)
        logger.info("已切换到 PRODUCER 模型（跳步模式）")

    logger.info(f"开始执行 pipeline ({len(pipeline.steps)} 步)")
    for idx, step in enumerate(pipeline.steps, start_step):
        logger.info(f"  [{idx}] {step.step_name}")

    try:
        pipeline.run(app, ctx)
        logger.success("Pipeline 执行完毕!")
    except Exception as e:
        logger.error(f"Pipeline 执行失败: {e}")
        traceback.print_exc()
        # 保存当前画面快照用于调试
        cmd_capture(app)
        raise


def cmd_run_loop(app, use_llm=False):
    """仅运行 gameplay loop 步骤（假设已在 gameplay 中）。"""
    from src.constants.yolo.model_type import YoloModelType
    from src.core.tasks.producer_challenge.steps.runtime.produce_gameplay_loop import (
        ProduceGameplayLoopStep,
    )

    # gameplay loop 使用 PRODUCER 模型
    app.yolo_engine.load_model(YoloModelType.PRODUCER)
    import time as _time
    _time.sleep(1.5)

    ctx = build_context(app, use_llm=use_llm)
    step = ProduceGameplayLoopStep()

    logger.info("直接运行 ProduceGameplayLoopStep ...")
    try:
        result = step.execute(app, ctx)
        logger.success(f"Gameplay loop 完成: {result}")

        # 自动衔接结果处理
        if result:
            from src.core.tasks.producer_challenge.steps.finalize.handle_results import HandleResultsStep
            logger.info("开始处理结果画面 ...")
            result_step = HandleResultsStep()
            result_ok = result_step.execute(app, ctx)
            logger.success(f"结果处理完成: {result_ok}")
    except Exception as e:
        logger.error(f"Gameplay loop 失败: {e}")
        traceback.print_exc()
        cmd_capture(app)
        raise


def main():
    parser = argparse.ArgumentParser(description="培育流程连调脚本")
    parser.add_argument("--capture", action="store_true", help="截图+YOLO识别")
    parser.add_argument("--phase", action="store_true", help="检测当前画面阶段")
    parser.add_argument("--only-loop", action="store_true", help="仅运行 gameplay loop")
    parser.add_argument("--step", type=int, default=1, help="从指定步骤开始 (1-12)")
    parser.add_argument("--resume", action="store_true",
                        help="恢复中断的培育（检测到再开弹窗时点击再開する）")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="覆盖难度设置 (regular/pro/master/legend)")
    parser.add_argument("--llm", action="store_true",
                        help="显式启用 LLM 决策策略（默认已启用）")
    parser.add_argument("--no-llm", action="store_true",
                        help="禁用 LLM 决策策略，回退到纯规则逻辑")
    parser.add_argument("--llm-url", type=str, default="http://192.168.100.10:11434/v1/",
                        help="LLM API 地址 (默认: http://192.168.100.10:11434/v1/)")
    parser.add_argument("--llm-model", type=str, default="gpt-oss:20b",
                        help="LLM 模型名 (默认: gpt-oss:20b)")
    args = parser.parse_args()

    app = init_app()

    # 保存难度覆盖到全局以便 build_context 使用
    app._debug_difficulty_override = getattr(args, 'difficulty', None)
    # 保存 LLM 配置
    app._debug_llm_url = getattr(args, 'llm_url', None)
    app._debug_llm_model = getattr(args, 'llm_model', None)
    # 保存恢复模式标记
    app._debug_resume = getattr(args, 'resume', False)

    use_llm = True
    if args.no_llm:
        use_llm = False
    elif args.llm:
        use_llm = True

    try:
        from src.constants.yolo.model_type import YoloModelType

        if args.capture:
            cmd_capture(app, model_type=YoloModelType.PRODUCER)
        elif args.phase:
            cmd_phase(app, model_type=YoloModelType.PRODUCER)
        elif args.only_loop:
            cmd_run_loop(app, use_llm=use_llm)
        else:
            cmd_run_full(app, start_step=args.step, use_llm=use_llm)
    except KeyboardInterrupt:
        logger.warning("用户中断")
    finally:
        # 打印 LLM 统计
        if use_llm:
            try:
                from src.core.tasks.producer_challenge.gameplay.llm_strategy import LLMStrategy
                # 尝试获取策略实例的统计
                logger.info("[LLM] 决策统计已记录在日志中")
            except Exception:
                pass
        try:
            app.yolo_engine.stop()
        except Exception:
            pass
        logger.info("清理完毕")


if __name__ == "__main__":
    main()
