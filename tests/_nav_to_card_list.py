"""Navigate device back to support card list page."""
import os, sys, time, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.device.Android.app import Android_App
from src.core.inference.ONNX import YoloModelFromONNX
from src.entity.Yolo import Yolo_Results
from src.entity.Game.Components.Button import ButtonList
from src.constants.yolo.labels.baseUI_Labels import BaseUILabels
from src.constants.game.text.support_card_text import SupportCardText
from src.utils.string_tools import MatchConfig

_FUZZ = MatchConfig(use_fuzz=True, fuzz_threshold=70)

device = Android_App()
model = YoloModelFromONNX("model/base_ui.onnx")

for i in range(8):
    frame = device.capture()
    yr = Yolo_Results(model(frame), frame)
    labels = list(set(b.label for b in yr.boxes))
    sc_count = sum(1 for b in yr.boxes if b.label == "Support Card")
    print(f"Step {i}: support_cards={sc_count} labels={labels}")
    if sc_count >= 3:
        print("On card list page!")
        cv2.imwrite("/tmp/card_list_current.png", frame)
        break
    # Check for cancel button (enhance page)
    btns = ButtonList(yr)
    cancel = btns.get_button_by_text(SupportCardText.ENHANCE_CANCEL, _FUZZ)
    if cancel:
        print("  Found cancel, clicking...")
        device.click_element(cancel)
        time.sleep(1.5)
        continue
    # Check for back button
    if yr.exists_label(BaseUILabels.BACK_BTN):
        print("  Found back button, clicking...")
        device.click_element(yr.filter_by_label(BaseUILabels.BACK_BTN).first())
        time.sleep(1.5)
        continue
    print("  No known buttons, pressing system back...")
    device.back()
    time.sleep(1.5)

print("Done")
