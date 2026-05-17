#!/usr/bin/env python3
"""离线批量测试 detect_gameplay_phase 阶段检测准确率。

使用已采集的截图 + YOLO PRODUCER 模型，验证 classify_phase 逻辑。
"""
import cv2
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.constants.yolo.model_type import YoloModelType
from src.core.inference.ONNX import YoloModelFromONNX

CAP_DIR = os.path.join(os.path.dirname(__file__), "produce_gameplay_captures")
model = YoloModelFromONNX(config.model_config[YoloModelType.PRODUCER])


def detect(img):
    r = model(img, conf_threshold=0.4)
    L = {}
    for b, s, c in zip(r.boxes, r.scores, r.class_ids):
        n = model._model_meta.names.get(int(c), str(c))
        if n not in L:
            L[n] = []
        cx = int(b[0] + b[2] / 2)
        cy = int(b[1] + b[3] / 2)
        L[n].append({"cx": cx, "cy": cy, "b": [int(x) for x in b]})
    return L


def classify_phase(L):
    """与 ui.py detect_gameplay_phase 保持一致的离线版本。"""
    s = set(L.keys())
    a = "Producer Challenge: Action" in s
    r = "Producer Challenge: Recommend Action" in s
    p = "Producer Challenge: Progress" in s
    sk = bool(s & {"Skill Card: Active", "Skill Card: Mental", "Skill Card: Trap"})
    ts = "Producer Challenge: Training: Score" in s
    tr = "Producer Challenge: Training: Remaining Rounds" in s
    ff = "Fast Forward Button" in s
    opts = "Universal Options" in s
    modal = "Universal Modal Header" in s
    skip = "Skip Button" in s
    pd = "P Drink" in s
    cb = "Universal Confirm button" in s
    ub = "Universal button" in s
    db = "Universal Disable Button" in s
    si = "Skill Card: Info" in s

    if modal:
        return "modal"
    if sk and (ts or tr):
        return "lesson"
    if (sk or si) and not a and not ts and (ub or cb or db):
        return "skill_reward"
    if (a or r) and p:
        return "schedule"
    if opts and not a:
        return "dialogue"
    if ff and not a and not sk:
        return "dialogue"
    if pd and not a and not sk and not ff and not opts:
        pd_items = L.get("P Drink", [])
        if any(item["cy"] < 1900 for item in pd_items):
            return "p_drink"
    if skip and not a and not sk:
        return "result"
    if not s:
        return "empty"
    return "unknown"


# ── 期望映射: 文件夹名 → 期望阶段 ──
FOLDER_EXPECTED = {
    "schedule": "schedule",
    "schedule_select": "schedule",
    "schedule_selected": "schedule",
    "action_clicked": "schedule",
    "after_schedule_confirm": "schedule",
    "lesson": "lesson",
    "lesson_turn2": "lesson",
    "after_card_confirm": "lesson",
    "after_card_use": "lesson",
    "transition": "lesson",       # 过渡帧仍含 lesson 元素
    "p_drink_select": "p_drink",
    "lesson_end": "p_drink",      # lesson结束后进入 P Drink
    "voice_modal": "modal",
    "commu_choice": "dialogue",
    "commu_choice2": "dialogue",
}

# flow_step_* 目录单独映射
FLOW_STEP_EXPECTED = {
    "flow_step_00": "p_drink",
}
# flow_step_01 ~ flow_step_29 均为 skill_reward
for i in range(1, 30):
    FLOW_STEP_EXPECTED[f"flow_step_{i:02d}"] = "skill_reward"

# full_flow 已知标注 (来自实际 YOLO 检测验证)
FULL_FLOW_EXPECTED = {
    "001.png": "modal",
    "003.png": "modal",
    "005.png": "modal",
    # 012-019 是过渡画面(HUD可见但P Drink仅在底栏), 不是p_drink选择
    "020.png": "schedule",
}


def collect_test_cases():
    """收集所有可测试的 (图片路径, 期望阶段) 对。"""
    cases = []

    # 1. 命名文件夹中的截图
    for folder, expected in FOLDER_EXPECTED.items():
        d = os.path.join(CAP_DIR, folder)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".png"):
                cases.append((os.path.join(d, f), expected, f"{folder}/{f}"))

    # 2. live/ 目录中的截图（从文件名取阶段）
    live_dir = os.path.join(CAP_DIR, "live")
    if os.path.isdir(live_dir):
        for f in sorted(os.listdir(live_dir)):
            if not f.endswith(".png") or f.startswith("current"):
                continue
            # 文件名格式: s01_dialogue.png
            parts = f.replace(".png", "").split("_", 1)
            if len(parts) == 2:
                expected = parts[1]
                cases.append((os.path.join(live_dir, f), expected, f"live/{f}"))

    # 3. full_flow/ 已标注子集
    ff_dir = os.path.join(CAP_DIR, "full_flow")
    if os.path.isdir(ff_dir):
        for fname, expected in FULL_FLOW_EXPECTED.items():
            fp = os.path.join(ff_dir, fname)
            if os.path.exists(fp):
                cases.append((fp, expected, f"full_flow/{fname}"))

    return cases


def main():
    cases = collect_test_cases()
    if not cases:
        print("未找到测试用例！")
        sys.exit(1)

    total = len(cases)
    passed = 0
    failed = []

    print(f"共 {total} 个测试用例\n")
    print(f"{'名称':40s} {'期望':15s} {'实际':15s} {'结果'}")
    print("-" * 80)

    for img_path, expected, name in cases:
        img = cv2.imread(img_path)
        if img is None:
            print(f"{name:40s} {'?':15s} {'READ_FAIL':15s} SKIP")
            continue
        L = detect(img)
        actual = classify_phase(L)

        ok = actual == expected
        if ok:
            passed += 1
            status = "OK"
        else:
            failed.append((name, expected, actual, sorted(L.keys())))
            status = "FAIL"

        print(f"{name:40s} {expected:15s} {actual:15s} {status}")

    print(f"\n{'=' * 80}")
    print(f"通过: {passed}/{total}  ({100 * passed / total:.1f}%)")

    if failed:
        print(f"\n失败用例 ({len(failed)}):")
        for name, exp, act, labels in failed:
            print(f"  {name}: 期望={exp}, 实际={act}")
            print(f"    检测标签: {labels}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
