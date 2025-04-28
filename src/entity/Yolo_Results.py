from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from src.entity.Yolo_Box import Yolo_Box

@dataclass
class Yolo_Results:
    results: any
    yolo_boxs: list[Yolo_Box]
    def __init__(self, yolo_results, model: YOLO, frame: np.array):
        self.results = list(yolo_results)
        self.yolo_boxs = []
        for result in yolo_results:
            if not hasattr(result, 'boxes'):
                continue
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                self.yolo_boxs.append(Yolo_Box(x1, y1, x2, y2, class_name, frame[y1:y2, x1:x2]))
