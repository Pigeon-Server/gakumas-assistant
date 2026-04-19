from copy import copy
from dataclasses import dataclass,field
from typing import List, Tuple, Union, Optional, Any

import numpy as np
from src.core.inference.ONNX import ONNXYoloResult
from src.utils.number import median
from src.utils.opencv_tools import intersection_area


@dataclass
class Yolo_Box:
    """
    YOLO 单个检测框封装类。

    Attributes:
        x, y, w, h: 框的位置和尺寸。
        label: 类别标签。
        frame: 框住的图像区域帧。
        cx, cy: 框中心点坐标。
    """
    x: float
    y: float
    w: float
    h: float
    label: str
    cx: int
    cy: int
    frame: Optional[np.ndarray] = field(repr=False)

    def __init__(self, x: float, y: float, w: float, h: float, label: str | None, frame: Optional[np.ndarray]):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label if label else self.__class__.__name__
        self.frame = frame
        self.cx = int(median(self.x, self.w))
        self.cy = int(median(self.y, self.h))

    def __eq__(self, other):
        """
        自定义相等比较：根据框的坐标、尺寸、标签判断两个 Yolo_Box 是否相等。
        """
        if isinstance(other, Yolo_Box):
            return (self.x == other.x and self.y == other.y and
                    self.w == other.w and self.h == other.h and
                    self.label == other.label)
        return False

    def __hash__(self):
        """
        自定义哈希：通过框的坐标、尺寸和标签计算哈希值。
        """
        return hash((self.x, self.y, self.w, self.h, self.label))

    def get_COL(self) -> Tuple[int, int]:
        return self.cx, self.cy

@dataclass
class Yolo_Results:
    """
    YOLO检测结果封装类，用于提取、筛选和分组Yolo_Box。
    
    Attributes:
        results: 原始YOLO模型结果。
        boxes: 提取后的目标框列表。
    """
    results: Any
    boxes: List[Yolo_Box]
    frame: Optional[np.ndarray]
    def __init__(self, yolo_results, frame: np.ndarray):
        self.boxes = []
        self.results = yolo_results
        # Copy once up-front; all box crops are views of this owned copy.
        # This protects against the caller mutating 'frame' later while adding
        # zero extra memory versus individual per-box copies.
        self.frame = frame.copy()
        frame_height, frame_width = self.frame.shape[:2]
        for index, box in enumerate(yolo_results):
            x, y, box_width, box_height = map(int, box)
            # 规范化坐标，防止出现负值或越界。
            x = max(0, min(frame_width, x))
            y = max(0, min(frame_height, y))
            box_width = max(0, box_width)
            box_height = max(0, box_height)
            x2 = min(frame_width, x + box_width)
            y2 = min(frame_height, y + box_height)
            label_id = int(yolo_results.class_ids[index])
            label = yolo_results.model_mata.names[label_id]
            self.boxes.append(Yolo_Box(x, y, x2, y2, label, self.frame[y:y2, x:x2]))
        self.sort_boxes()

    def __bool__(self):
        return bool(self.boxes)

    def __len__(self):
        return len(self.boxes)

    def __iter__(self):
        return iter(self.boxes)

    def __getitem__(self, index):
        return self.from_boxes(self.boxes[index])

    def sort_boxes(self, vertical_thresh: int = 20):
        """
        从左上到右下对 boxes 排序。
        vertical_thresh: 两个框的中心点 y 坐标差异小于该值，则认为在同一行。
        以行首元素的 cy 为锚点判断是否同行，避免滑动窗口导致跨行元素被错误合并。
        """
        # 先按 y 升序粗排（先上后下）
        sorted_boxes = sorted(self.boxes, key=lambda b: b.cy)
        # 分行：把 boxes 分为多行，每行内再按 x 排序
        lines = []
        current_line: list = []
        line_anchor_cy: float | None = None
        for box in sorted_boxes:
            if line_anchor_cy is None or abs(box.cy - line_anchor_cy) <= vertical_thresh:
                if not current_line:
                    line_anchor_cy = box.cy  # 锚定行首
                current_line.append(box)
            else:
                lines.append(sorted(current_line, key=lambda b: b.cx))
                current_line = [box]
                line_anchor_cy = box.cy
        if current_line:
            lines.append(sorted(current_line, key=lambda b: b.cx))
        # 扁平化行列表
        self.boxes = [box for line in lines for box in line]

    @classmethod
    def from_boxes(cls, boxes: List[Yolo_Box]) -> "Yolo_Results":
        """
        通过已有的 Yolo_Box 列表构建 Yolo_Results 实例
        """
        inst = cls.__new__(cls)
        inst.results = []
        inst.boxes = boxes
        inst.frame = None
        inst.sort_boxes()
        return inst

    def first(self):
        return self.boxes[0] if self.boxes else None

    def index(self, index):
        return self.boxes[index]

    def filter_by_label(self, label: str) -> "Yolo_Results":
        """
        按类别获取目标框。

        Args:
            label: str 类别名称。

        Returns:
            返回符合条件的Yolo_Results实例
        """
        return self.from_boxes([box for box in self.boxes if box.label == label])

    def filter_by_labels(self, labels: List[str]) -> "Yolo_Results":
        """
        按多个类别名称筛选目标框。

        Args:
            labels: 标签名列表，如 ["button", "checkbox"]

        Returns:
            所有匹配标签的 Yolo_Results实例，可能为空
        """
        return self.from_boxes(
            [box for box in self.boxes if box.label in labels]
        )

    def remove_by_label(self, label: str) -> None:
        """
        移除指定标签的元素。

        Args:
            label: 要移除的标签名。
        """
        self.boxes = [box for box in self.boxes if box.label != label]

    def remove_by_yolo_results(self, other_yolo_results: "Yolo_Results") -> "Yolo_Results":
        """
        移除与指定 Yolo_Results 对象中的元素相同的目标框。

        Args:
            other_yolo_results: 要删除的 Yolo_Results 对象。
        """
        other_boxes = set(other_yolo_results.boxes)
        return self.from_boxes([box for box in self.boxes if box not in other_boxes])

    def remove_by_yolo_box(self, yolo_box: Yolo_Box) -> "Yolo_Results":
        """
        移除指定的 Yolo_Box 元素。

        Args:
            yolo_box: 要移除的 Yolo_Box 对象。
        """
        return self.from_boxes([box for box in self.boxes if box != yolo_box])

    def remove_by_yolo_boxes(self, target_yolo_boxes: List[Yolo_Box]) -> "Yolo_Results":
        """
        移除与指定 Yolo_Box 列表中的元素相同的目标框。

        Args:
            target_yolo_boxes: 要删除的 Yolo_Box 对象列表。
        """
        other_boxes = set(target_yolo_boxes)
        return self.from_boxes([box for box in self.boxes if box not in other_boxes])

    def exists_label(self, label: str) -> bool:
        """
        查找是否存在目标标签
        :param label: 标签名
        :return:
        """
        return any(b.label == label for b in self.boxes)

    def exists_all_labels(self, labels: List[str]) -> bool:
        """
        判断所有指定的标签是否都存在于当前框集合中。

        Args:
            labels: 标签名列表

        Returns:
            True 表示全部标签都存在，False 表示有任意一个不存在
        """
        existing_labels = {box.label for box in self.boxes}
        return all(label in existing_labels for label in labels)

    def get_y_min_element(self) -> Optional["Yolo_Results"]:
        """返回Y轴最小的元素（最靠上）"""
        if not self.boxes:
            return None
        return self.from_boxes([min(self.boxes, key=lambda box: box.y)])

    def get_y_max_element(self) -> Optional["Yolo_Results"]:
        """返回Y轴最小的元素（最靠下）"""
        if not self.boxes:
            return None
        return self.from_boxes([max(self.boxes, key=lambda box: box.h)])

    def get_x_min_element(self) -> Optional["Yolo_Results"]:
        """返回X轴最小的元素（最靠左）"""
        if not self.boxes:
            return None
        return self.from_boxes([min(self.boxes, key=lambda box: box.x)])

    def get_x_max_element(self) -> Optional["Yolo_Results"]:
        """"返回X轴最大的元素（最靠右）"""
        if not self.boxes:
            return None
        return self.from_boxes([max(self.boxes, key=lambda box: box.w)])

    def get_center_x_range_element(self, x_value, range_: int) -> "Yolo_Results":
        """
        获取中心点X坐标范围内的元素
        :param x_value: 目标X坐标值
        :param range_: 范围
        :return:
        """
        return self.from_boxes(
            [el for el in self.boxes if el.cx - range_ <= x_value <= el.cx + range_]
        )

    def get_y_range_element(self, y_value, range_: int) -> "Yolo_Results":
        """
        获取中心点Y坐标范围内的元素
        :param y_value: 目标Y坐标值
        :param range_: 范围
        :return:
        """
        return self.from_boxes(
            [el for el in self.boxes if el.cy - range_ <= y_value <= el.cy + range_]
        )

    def match_rows_with(self, other_boxes: list[Yolo_Box], tolerance_ratio: float = 0.5) -> list[
        tuple[Yolo_Box, Yolo_Box]]:
        """
        查找在同一行的匹配元素对（自身与另一个列表中元素）。

        Args:
            other_boxes: 另一个Yolo_Box列表。
            tolerance_ratio: 高度容差比例（默认为 0.5）。

        Returns:
            在同一行的元素对列表。
        """
        same_row_pairs = []

        for box1 in self.boxes:
            for box2 in other_boxes:
                # 计算允许的误差阈值（基于较小的按钮高度）
                tolerance = min(box1.h, box2.h) * tolerance_ratio

                # 判断两个按钮是否在同一行
                if abs(box1.y - box2.y) <= tolerance:
                    same_row_pairs.append((box1, box2))

        return same_row_pairs

    def group_yolo_boxes_by_position(
            self,
            row_thresh: Optional[int] = 30,
            col_thresh: Optional[int] = 120,
            mode: str = 'center',
            margin: int = 10
    ) -> List["Yolo_Results"]:
        """
        自动根据x/y距离分组Yolo元素
        :param row_thresh: 最大行容差（像素），为None表示不分行
        :param col_thresh: 最大列容差（像素），为None表示不分列
        :param mode: 中心点模式：'center'，边到边模式：'edge'
        :param margin: 边框大小容差（像素）
        :return: 分组后的Yolo_Results对象列表
        """

        for box in self.boxes:
            box.cx, box.cy = box.get_COL()

        boxes_to_process = self.boxes

        # ===== 分行 =====
        if row_thresh is not None:
            if mode == 'center':
                boxes_sorted = sorted(boxes_to_process, key=lambda b: b.cy)
            else:
                boxes_sorted = sorted(boxes_to_process, key=lambda b: b.y)

            rows = []
            for box in boxes_sorted:
                matched = False
                for row in rows:
                    ref = row[0]
                    if mode == 'center':
                        y_dist = abs(box.cy - ref.cy)
                    else:
                        y_dist = box.y - (ref.y + ref.h)

                    if -margin <= y_dist <= row_thresh:
                        row.append(box)
                        matched = True
                        break
                if not matched:
                    rows.append([box])
        else:
            # 不分行，视为一整行
            rows = [boxes_to_process]

        grouped: List["Yolo_Results"] = []

        for row in rows:
            # ===== 分列 =====
            if col_thresh is not None:
                if mode == 'center':
                    row_sorted = sorted(row, key=lambda b: b.cx)
                else:
                    row_sorted = sorted(row, key=lambda b: b.x)

                group = [row_sorted[0]]
                for prev, curr in zip(row_sorted, row_sorted[1:]):
                    if mode == 'center':
                        x_dist = curr.get_COL()[0] - prev.get_COL()[0]
                    else:
                        x_dist = curr.x - (prev.x + prev.w)

                    if -margin <= x_dist <= col_thresh:
                        group.append(curr)
                    else:
                        grouped.append(self.from_boxes(group))
                        group = [curr]
                grouped.append(self.from_boxes(group))
            else:
                # 不分列，整行为一个组
                grouped.append(self.from_boxes(row))

        return grouped

    def find_containing_groups(
            self,
            container_label: str,
            include_labels: Union[str, List[str]],
            relation: str = "all",
            contain_threshold: float = 0.9
    ) -> List["Yolo_Results"]:
        """
        查找包含其他框的主框组合（如容器+子组件）。

        Args:
            container_label: 容器框的标签名（如 "panel", "card"）
            include_labels: 被包含框的标签名（可为单个或多个）
            relation: 匹配关系，"all" 表示必须全部包含，"or" 表示包含任一即可
            contain_threshold: 被包含阈值（被包含框有多少比例在容器内，默认0.9）

        Returns:
            所有满足条件的 Yolo_Results 列表，每组包含一个容器框和若干个子框
        """
        if isinstance(include_labels, str):
            include_labels = [include_labels]

        def contain_ratio(container: Yolo_Box, child: Yolo_Box) -> float:
            def _area(b: Yolo_Box) -> float:
                return max(0.0, b.w * b.h)

            inter = intersection_area(container.x, container.y, container.w, container.h, child.x, child.y, child.w, child.h)
            child_area = _area(child)
            return inter / child_area if child_area > 0 else 0.0

        result_groups = []

        for container in self.boxes:
            if container.label != container_label:
                continue

            included = []
            for other in self.boxes:
                if other == container or other.label not in include_labels:
                    continue
                if contain_ratio(container, other) >= contain_threshold:
                    included.append(other)

            if relation == "all":
                matched_labels = {box.label for box in included}
                if set(include_labels).issubset(matched_labels):
                    result_groups.append(self.from_boxes([container] + included))
            elif relation == "or":
                if included:
                    result_groups.append(self.from_boxes([container] + included))
            else:
                raise ValueError(f"不支持的 relation 类型: {relation}（应为 'all' 或 'or'）")

        return result_groups

    @property
    def x(self):
        return min(box.x for box in self.boxes)

    @property
    def y(self):
        return min(box.y for box in self.boxes)

    @property
    def w(self):
        return max(box.w for box in self.boxes)

    @property
    def h(self):
        return max(box.h for box in self.boxes)

    def get_COL(self) -> Tuple[float, float]:
        """获取集合的中心点"""
        if not self.boxes:
            raise ValueError("The number of boxes is 0, and the center point cannot be obtained")
        # min_x = min(box.x for box in self.boxes)
        # max_x = max(box.w for box in self.boxes)
        # min_y = min(box.y for box in self.boxes)
        # max_y = max(box.h for box in self.boxes)
        center_x = (self.x + self.w) / 2
        center_y = (self.y + self.h) / 2
        return int(center_x), int(center_y)

    def get_vertical_range_elements(self, all_boxes: "Yolo_Results", x_tolerance: float) -> "Yolo_Results":
        """获取与本组垂直对齐的其他框"""
        center_x, _ = self.get_COL()
        return self.from_boxes([
            box for box in all_boxes.boxes if abs(box.get_COL()[0] - center_x) <= x_tolerance
        ])
