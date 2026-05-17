from types import SimpleNamespace

import cv2

import src.entity.Game.Components.TabBar as tabbar_module
from src.entity.Game.Components.TabBar import (
    TabBar,
    _expand_tab_region,
    _expand_tab_text_region,
    _extract_tab_word_boxes,
    _filter_centered_word_boxes,
    _is_selected_tab_frame,
    _merge_word_boxes_into_tab_regions,
    _normalize_tab_text,
)
from src.entity.Yolo import Yolo_Box


def test_merge_word_boxes_into_tab_regions_merges_stacked_lines_only():
    word_boxes = [
        (21, 43, 235, 51),
        (91, 13, 166, 38),
        (345, 0, 404, 16),
        (351, 13, 398, 38),
    ]

    regions = _merge_word_boxes_into_tab_regions(word_boxes, width=500, height=51)

    assert regions == [
        (21, 13, 235, 51),
        (345, 0, 404, 38),
    ]


def test_extract_tab_word_boxes_filters_background_bars_from_selected_tab():
    image = cv2.imread("tests/tabbar.png")

    boxes = _extract_tab_word_boxes(image)

    assert boxes == [
        (141, 28, 212, 61),
        (451, 28, 569, 61),
    ]


def test_normalize_tab_text_strips_new_prefix_noise():
    assert _normalize_tab_text("NEWAP") == "AP"
    assert _normalize_tab_text("NEUWコンテスト") == "コンテスト"
    assert _normalize_tab_text("デイリー") == "デイリー"


def test_filter_centered_word_boxes_keeps_only_middle_band_text():
    boxes = [
        (91, 13, 166, 38),
        (345, 0, 404, 16),
        (351, 13, 398, 38),
    ]

    centered = _filter_centered_word_boxes(boxes, height=51)

    assert centered == [
        (91, 13, 166, 38),
        (351, 13, 398, 38),
    ]


def test_is_selected_tab_frame_matches_sample_tab_images():
    samples = [
        ("tests/tabbar.png", (21, 0, 317, 85), True),
        ("tests/tabbar.png", (450, 0, 677, 85), False),
        ("tests/tabbar4.png", (21, 13, 235, 51), True),
        ("tests/tabbar4.png", (345, 0, 404, 38), False),
        ("tests/tabbar2.png", (9, 12, 102, 54), True),
        ("tests/tabbar2.png", (105, 14, 206, 37), False),
        ("tests/tabbar3.png", (391, 0, 515, 47), True),
        ("tests/tabbar3.png", (29, 1, 138, 47), False),
    ]

    for image_path, (x1, y1, x2, y2), expected in samples:
        image = cv2.imread(image_path)
        frame = image[y1:y2, x1:x2]
        assert _is_selected_tab_frame(frame) is expected


def test_expand_tab_region_keeps_selected_state_without_growing_to_full_button():
    image = cv2.imread("tests/tabbar4.png")
    x1, y1, x2, y2 = _expand_tab_region(91, 13, 166, 38, image.shape[1], image.shape[0])

    assert (x1, y1, x2, y2) == (87, 11, 170, 50)
    assert _is_selected_tab_frame(image[y1:y2, x1:x2]) is True


def test_expand_tab_text_region_stays_in_middle_band():
    image = cv2.imread("tests/tabbar4.png")
    x1, y1, x2, y2 = _expand_tab_text_region(351, 13, 398, 38, image.shape[1], image.shape[0])

    assert (x1, y1, x2, y2) == (347, 11, 402, 40)


def test_tabbar_items_use_middle_text_box_for_bounds(monkeypatch):
    image = cv2.imread("tests/tabbar4.png")

    def fake_ocr(cropped):
        text = "マニー" if cropped.shape[1] > 70 else "AP"
        return [SimpleNamespace(text=text)]

    monkeypatch.setattr(tabbar_module.ocr_service, "ocr", fake_ocr)

    tabbar = TabBar(Yolo_Box(0, 0, image.shape[1], image.shape[0], "TAB_BAR", image))

    assert [(item.text, (item.x, item.y, item.w, item.h)) for item in tabbar] == [
        ("マニー", (87, 11, 170, 40)),
        ("AP", (347, 11, 402, 40)),
    ]
    assert tabbar.selected == tabbar.tab_items[0]


def test_tabbar_skips_groups_without_centered_text_boxes(monkeypatch):
    image = cv2.imread("tests/tabbar4.png")

    monkeypatch.setattr(
        tabbar_module,
        "_extract_tab_word_boxes",
        lambda _frame: [
            (10, 0, 50, 12),
            (91, 13, 166, 38),
            (351, 13, 398, 38),
        ],
    )

    def fake_ocr(cropped):
        text = "マニー" if cropped.shape[1] > 70 else "AP"
        return [SimpleNamespace(text=text)]

    monkeypatch.setattr(tabbar_module.ocr_service, "ocr", fake_ocr)

    tabbar = TabBar(Yolo_Box(0, 0, image.shape[1], image.shape[0], "TAB_BAR", image))

    assert [(item.text, (item.x, item.y, item.w, item.h)) for item in tabbar] == [
        ("マニー", (87, 11, 170, 40)),
        ("AP", (347, 11, 402, 40)),
    ]


def test_tabbar_skips_groups_without_centered_text_boxes(monkeypatch):
    image = cv2.imread("tests/tabbar4.png")

    monkeypatch.setattr(
        tabbar_module,
        "_extract_tab_word_boxes",
        lambda _frame: [
            (10, 0, 50, 12),
            (91, 13, 166, 38),
            (351, 13, 398, 38),
        ],
    )

    def fake_ocr(cropped):
        text = "マニー" if cropped.shape[1] > 70 else "AP"
        return [SimpleNamespace(text=text)]

    monkeypatch.setattr(tabbar_module.ocr_service, "ocr", fake_ocr)

    tabbar = TabBar(Yolo_Box(0, 0, image.shape[1], image.shape[0], "TAB_BAR", image))

    assert [(item.text, (item.x, item.y, item.w, item.h)) for item in tabbar] == [
        ("マニー", (87, 11, 170, 40)),
        ("AP", (347, 11, 402, 40)),
    ]
