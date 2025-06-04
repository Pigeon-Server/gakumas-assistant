from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import numpy as np
from ultralytics import YOLO

from src.utils.number import median

class YoloModelType:
    BASE_UI: str = 'BASE_UI'
    PRODUCER: str = 'PRODUCER'

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
    frame: np.ndarray
    cx: int
    cy: int

    def __init__(self, x: float, y: float, w: float, h: float, label: str, frame: np.ndarray):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.frame = frame
        self.cx = int(median(self.x, self.w))
        self.cy = int(median(self.y, self.h))

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
    results: any
    boxes: list[Yolo_Box]
    def __init__(self, yolo_results, model: YOLO, frame: np.array):
        self.results = list(yolo_results)
        self.boxes = []
        for result in self.results:
            if not hasattr(result, 'boxes'):
                continue
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                self.boxes.append(Yolo_Box(x1, y1, x2, y2, class_name, frame[y1:y2, x1:x2]))

    def __bool__(self):
        return bool(self.boxes)

    def __len__(self):
        return len(self.boxes)

    def __iter__(self):
        return iter(self.boxes)

    @classmethod
    def from_boxes(cls, boxes: List[Yolo_Box]) -> "Yolo_Results":
        """
        通过已有的 Yolo_Box 列表构建 Yolo_Results 实例
        """
        inst = cls.__new__(cls)
        inst.results = []
        inst.boxes = boxes
        return inst

    def first(self):
        return self.boxes[0]

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
        """返回Y轴最小的元素（最靠下）"""
        if not self.boxes:
            return None
        return self.from_boxes([min(self.boxes, key=lambda box: box.y)])

    def get_y_max_element(self) -> Optional["Yolo_Results"]:
        """返回Y轴最小的元素（最靠上）"""
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
            relation: str = "all"
    ) -> List["Yolo_Results"]:
        """
        查找包含其他框的主框组合（如容器+子组件）。

        Args:
            container_label: 容器框的标签名（如 "panel", "card"）
            include_labels: 被包含框的标签名（可为单个或多个）
            relation: 匹配关系，"all" 表示必须全部包含，"or" 表示包含任一即可

        Returns:
            所有满足条件的 Yolo_Results 列表，每组包含一个容器框和若干个子框
        """
        if isinstance(include_labels, str):
            include_labels = [include_labels]

        result_groups = []

        for container in self.boxes:
            if container.label != container_label:
                continue

            included = []
            for other in self.boxes:
                if other == container or other.label not in include_labels:
                    continue

                # 判断是否被包含（边界在容器内部）
                if (container.x <= other.x and
                        container.y <= other.y and
                        container.x + container.w >= other.x + other.w and
                        container.y + container.h >= other.y + other.h):
                    included.append(other)

            if relation == "all":
                include_label_set = set(include_labels)
                matched_labels = {box.label for box in included}
                if include_label_set.issubset(matched_labels):
                    result_groups.append(self.from_boxes([container] + included))
            elif relation == "or":
                if included:
                    result_groups.append(self.from_boxes([container] + included))
            else:
                raise ValueError(f"不支持的 relation 类型: {relation}（应为 'all' 或 'or'）")

        return result_groups


    def get_COL(self) -> Tuple[float, float]:
        """获取集合的中心点"""
        if not self.boxes:
            raise ValueError("The number of boxes is 0, and the center point cannot be obtained")
        min_x = min(box.x for box in self.boxes)
        max_x = max(box.w for box in self.boxes)
        min_y = min(box.y for box in self.boxes)
        max_y = max(box.h for box in self.boxes)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        return int(center_x), int(center_y)

    def get_vertical_range_elements(self, all_boxes: "Yolo_Results", x_tolerance: float) -> "Yolo_Results":
        """获取与本组垂直对齐的其他框"""
        center_x, _ = self.get_COL()
        return self.from_boxes([
            box for box in all_boxes.boxes if abs(box.get_COL()[0] - center_x) <= x_tolerance
        ])