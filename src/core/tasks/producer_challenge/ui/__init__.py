"""producer_challenge 的 UI 辅助入口。"""

from src.core.tasks.producer_challenge.shared.common import ocr_text

from .common import (
    click_modal_action_with_retry,
    click_top_right_action,
    find_button,
    get_buttons,
    has_button,
    inertial_swipe,
    is_final_confirm_page,
    is_memory_selection_page,
    wait_for_final_confirm_page,
    wait_for_memory_selection_page,
    wait_frame_stable,
)
from .gameplay_actions import (
    click_recommend_action,
    handle_p_drink_select,
    handle_skill_card_selection,
    handle_skill_reward_selection,
)
from .gameplay_state import (
    classify_gameplay_phase,
    classify_gameplay_state,
    classify_pipeline_position,
    collect_button_like_texts,
    collect_frame_text,
    detect_gameplay_phase,
    detect_gameplay_state,
    get_pipeline_position,
)
from .navigation import go_back_in_gameplay, go_home_from_gameplay
from .preset import (
    build_preset_swipe_paths,
    get_current_preset_index,
    get_preset_swipe_paths,
    parse_preset_index,
    select_preset_by_horizontal_swipe,
)

__all__ = [
    "build_preset_swipe_paths",
    "classify_gameplay_phase",
    "classify_gameplay_state",
    "classify_pipeline_position",
    "click_modal_action_with_retry",
    "click_recommend_action",
    "click_top_right_action",
    "collect_button_like_texts",
    "collect_frame_text",
    "detect_gameplay_phase",
    "detect_gameplay_state",
    "find_button",
    "get_buttons",
    "get_current_preset_index",
    "get_pipeline_position",
    "get_preset_swipe_paths",
    "go_back_in_gameplay",
    "go_home_from_gameplay",
    "handle_p_drink_select",
    "handle_skill_card_selection",
    "handle_skill_reward_selection",
    "has_button",
    "inertial_swipe",
    "is_final_confirm_page",
    "is_memory_selection_page",
    "ocr_text",
    "parse_preset_index",
    "select_preset_by_horizontal_swipe",
    "wait_for_final_confirm_page",
    "wait_for_memory_selection_page",
    "wait_frame_stable",
]
