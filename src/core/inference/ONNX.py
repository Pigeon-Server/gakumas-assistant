import ast
import colorsys
import os
import json
import threading
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import cv2
import onnxruntime as ort

from src.utils.dml_manager import DMLManager
from src.utils.logger import logger
from src.utils.opencv_tools import letterbox, center_crop
from src.utils.runtime_paths import resolve_runtime_str


@dataclass
class ONNXYoloModelMeta:
    imgsz: Tuple[int, int]
    names: Dict[int, str]
    colors: Dict[int, Tuple[int, int, int]]


@dataclass
class ONNXExportMeta:
    version: str | None
    args: Dict[str, object]
    end2end: bool

@dataclass
class ONNXYoloResult:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    model_mata: ONNXYoloModelMeta
    image: np.ndarray

    def __bool__(self):
        return bool(self.boxes.size > 0)

    def __len__(self):
        return len(self.boxes)

    def __iter__(self):
        return iter(self.boxes)

    def plot(
            self,
            line_width: int = 2,
            font_size: float = 0.5,
    ) -> np.ndarray:
        img = self.image.copy()
        for box, score, cls in zip(self.boxes, self.scores, self.class_ids):
            x, y, w, h = box.astype(int)
            color = self.model_mata.colors.get(int(cls), (0, 255, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, line_width)
            label = f"{self.model_mata.names.get(int(cls), cls)}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                1,
            )
            label_x = max(x, 0)
            label_y = y - 10 if y - 10 > label_height else y + label_height + 4
            cv2.rectangle(
                img,
                (label_x, label_y - label_height - 4),
                (label_x + label_width, label_y + 2),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return img

@dataclass
class ONNXYoloClassifyResult:
    class_id: int
    score: float
    probs: np.ndarray
    model_meta: ONNXYoloModelMeta
    image: np.ndarray

    def __bool__(self):
        return self.score > 0

    @property
    def class_name(self):
        return self.model_meta.names.get(self.class_id, str(self.class_id))

    def plot(
            self,
            line_width: int = 2,
            font_size: float = 0.5,
    ) -> np.ndarray:
        img = self.image.copy()
        color = (0, 255, 0)
        label = f"{self.class_name}: {self.score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            1,
        )
        padding = max(line_width, 4)
        cv2.rectangle(
            img,
            (padding, padding),
            (padding + label_width + 8, padding + label_height + 12),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            img,
            label,
            (padding + 4, padding + label_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return img

class YoloModelFromONNX:
    _model_meta: ONNXYoloModelMeta
    _export_meta: ONNXExportMeta
    _engine: ort.InferenceSession
    _model_dir: str
    _model_file: str
    _model_name: str
    _model_input_name: str
    _output_layout: str
    _output_shape: Tuple[object, ...]
    def __init__(self, model_path: str) -> None:
        """
        初始化ONNX模型
        :param model_path: 模型地址
        """
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        self._model_dir, self._model_file = os.path.split(model_path)
        self._model_name = os.path.splitext(self._model_file)[0]
        self._load_model_meta()
        self._engine = DMLManager.create_dml_session(model_path)
        self._model_input_name = self._engine.get_inputs()[0].name
        self._output_shape = tuple(self._engine.get_outputs()[0].shape)
        self._output_layout = self._detect_output_layout(self._output_shape)
        logger.debug(
            "Loaded ONNX model {} with Ultralytics {} output {} -> {}",
            self._model_name,
            self._export_meta.version or "unknown",
            self._output_shape,
            self._output_layout,
        )

    @staticmethod
    def _pastel_palette(n: int):
        colors = []
        for i in range(n):
            h = i / n
            s = 0.45      # 更柔
            v = 0.95
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append((int(r*255), int(g*255), int(b*255)))
        return colors

    def _load_model_meta(self):
        meta_path = os.path.join(self._model_dir, f"{self._model_name}_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        imgsz = json.loads(meta["imgsz"])
        names_mapping = ast.literal_eval(meta["names"])
        raw_args = meta.get("args", "{}")
        try:
            export_args = ast.literal_eval(raw_args) if raw_args else {}
        except (ValueError, SyntaxError):
            export_args = {}
        palette_255 = self._pastel_palette(len(names_mapping))
        color_mapping = {
            name_id: color
            for name_id, color in zip(names_mapping.keys(), palette_255)
        }
        self._model_meta = ONNXYoloModelMeta(imgsz, names_mapping, color_mapping)
        self._export_meta = ONNXExportMeta(
            version=meta.get("version"),
            args=export_args if isinstance(export_args, dict) else {},
            end2end=self._parse_metadata_bool(meta.get("end2end")),
        )

    @staticmethod
    def _parse_metadata_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_version_tuple(version: str | None) -> Tuple[int, ...]:
        if not version:
            return ()
        parts: list[int] = []
        for part in str(version).split("."):
            digits = "".join(ch for ch in part if ch.isdigit())
            if not digits:
                break
            parts.append(int(digits))
        return tuple(parts)

    def _detect_output_layout(self, output_shape: Tuple[object, ...]) -> str:
        if not output_shape:
            return "legacy"
        if len(output_shape) >= 2 and output_shape[-1] == 6:
            return "standalone"
        if len(output_shape) >= 2 and output_shape[1] == 6:
            return "standalone"

        version_tuple = self._parse_version_tuple(self._export_meta.version)
        if version_tuple >= (8, 4, 38):
            if self._export_meta.args.get("nms") is True:
                return "standalone"
        return "legacy"

    @staticmethod
    def _empty_result(input_image: np.ndarray, model_meta: ONNXYoloModelMeta) -> ONNXYoloResult:
        return ONNXYoloResult(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            model_meta,
            input_image,
        )

    @staticmethod
    def _flatten_nms_keep(keep) -> list[int]:
        if keep is None:
            return []
        keep_array = np.asarray(keep)
        if keep_array.size == 0:
            return []
        return [int(index) for index in keep_array.reshape(-1).tolist()]

    @staticmethod
    def _to_box_xywh_from_xyxy(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            ratio: float,
            dw: float,
            dh: float,
    ) -> list[float] | None:
        left = (x1 - dw) / ratio
        top = (y1 - dh) / ratio
        right = (x2 - dw) / ratio
        bottom = (y2 - dh) / ratio
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        if width <= 0 or height <= 0:
            return None
        return [left, top, width, height]

    @staticmethod
    def _normalize_standalone_rows(results: np.ndarray) -> np.ndarray:
        outputs = np.squeeze(results[0])
        if outputs.ndim == 1:
            outputs = outputs.reshape(1, -1)
        if outputs.ndim != 2:
            raise ValueError(f"Unsupported standalone output rank: {outputs.ndim}")
        if outputs.shape[-1] == 6:
            return outputs
        if outputs.shape[0] == 6:
            return outputs.T
        raise ValueError(f"Unsupported standalone output shape: {outputs.shape}")

    def _postprocess_standalone(
            self,
            input_image: np.ndarray,
            results: np.ndarray,
            conf_threshold: float,
            ratio: float,
            dw: float,
            dh: float,
    ) -> ONNXYoloResult:
        rows = self._normalize_standalone_rows(results)
        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        for row in rows:
            score = float(row[4])
            if score < conf_threshold:
                continue
            box = self._to_box_xywh_from_xyxy(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                ratio,
                dw,
                dh,
            )
            if box is None:
                continue
            boxes.append(box)
            scores.append(score)
            class_ids.append(int(round(float(row[5]))))

        if not boxes:
            return self._empty_result(input_image, self._model_meta)
        return ONNXYoloResult(
            np.asarray(boxes, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
            np.asarray(class_ids, dtype=np.int64),
            self._model_meta,
            input_image,
        )

    def _postprocess_legacy(
            self,
            input_image: np.ndarray,
            results: np.ndarray,
            conf_threshold: float,
            iou_threshold: float,
            ratio: float,
            dw: float,
            dh: float,
            agnostic_nms_groups: list[set[int]] | None = None,
    ) -> ONNXYoloResult:
        outputs = np.transpose(np.squeeze(results[0]))
        if outputs.ndim == 1:
            outputs = outputs.reshape(1, -1)
        if outputs.size == 0:
            return self._empty_result(input_image, self._model_meta)

        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= conf_threshold:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = x - w / 2
                top = y - h / 2

                left = (left - dw) / ratio
                top = (top - dh) / ratio
                width = w / ratio
                height = h / ratio

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        if not boxes:
            return self._empty_result(input_image, self._model_meta)

        nms_boxes = []
        nms_scores = []
        nms_class_ids = []

        agnostic_map: dict[int, int] = {}
        if agnostic_nms_groups:
            for gi, group in enumerate(agnostic_nms_groups):
                for cid in group:
                    agnostic_map[cid] = gi

        grouped: dict[tuple, list[int]] = {}
        for i, cid in enumerate(class_ids):
            if cid in agnostic_map:
                key = ("agnostic", agnostic_map[cid])
            else:
                key = ("class", int(cid))
            grouped.setdefault(key, []).append(i)

        for indices_in_group in grouped.values():
            group_boxes = [boxes[i] for i in indices_in_group]
            group_scores = [scores[i] for i in indices_in_group]
            group_class_ids = [class_ids[i] for i in indices_in_group]

            keep = self._flatten_nms_keep(
                cv2.dnn.NMSBoxes(group_boxes, group_scores, conf_threshold, iou_threshold)
            )
            nms_boxes.extend([group_boxes[i] for i in keep])
            nms_scores.extend([group_scores[i] for i in keep])
            nms_class_ids.extend([group_class_ids[i] for i in keep])

        if not nms_boxes:
            return self._empty_result(input_image, self._model_meta)
        return ONNXYoloResult(
            np.asarray(nms_boxes, dtype=np.float32),
            np.asarray(nms_scores, dtype=np.float32),
            np.asarray(nms_class_ids, dtype=np.int64),
            self._model_meta,
            input_image
        )

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        图像预处理
        :param img: 图像
        :return:
        """
        img_letterbox, ratio, (dw, dh) = letterbox(img, self._model_meta.imgsz)
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgb = img_rgb.transpose(2, 0, 1)
        return np.expand_dims(img_rgb, axis=0), ratio, dw, dh

    def _postprocess(
            self,
            input_image: np.ndarray,
            results: np.ndarray,
            conf_threshold: float,
            iou_threshold: float,
            ratio: float,
            dw: float,
            dh: float,
            agnostic_nms_groups: list[set[int]] | None = None,
    ) -> ONNXYoloResult:
        """
        后处理模型输出
        :param input_image: 输入图像
        :param results: 模型推理结果
        :param conf_threshold: 得分阈值
        :param iou_threshold: NMS阈值
        :param agnostic_nms_groups: 跨类别NMS分组，每个 set 内的 class_id 视为同类进行NMS。
                                    例如 [{0, 1, 2}] 表示 class 0/1/2 之间的重叠框会互相抑制。
        :return:
        """
        if self._output_layout == "standalone":
            return self._postprocess_standalone(
                input_image,
                results,
                conf_threshold,
                ratio,
                dw,
                dh,
            )
        return self._postprocess_legacy(
            input_image,
            results,
            conf_threshold,
            iou_threshold,
            ratio,
            dw,
            dh,
            agnostic_nms_groups=agnostic_nms_groups,
        )

    def __call__(self, img: np.ndarray, conf_threshold: float = 0.5, iou_threshold: float = 0.5,
                 agnostic_nms_groups: list[set[int]] | None = None) -> ONNXYoloResult:
        input_tensor, ratio, dw, dh = self._preprocess(img)
        outputs = DMLManager.run(
            self._engine,
            {self._model_input_name: input_tensor}
        )
        return self._postprocess(img, outputs, conf_threshold, iou_threshold, ratio, dw, dh,
                                 agnostic_nms_groups=agnostic_nms_groups)

class YoloClassifyModelFromONNX:
    _model_meta: ONNXYoloModelMeta
    _engine: ort.InferenceSession
    _model_input_name: str

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        model_dir, model_file = os.path.split(model_path)
        model_name = os.path.splitext(model_file)[0]

        meta_path = os.path.join(model_dir, f"{model_name}_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        imgsz = json.loads(meta["imgsz"])
        names_mapping = ast.literal_eval(meta["names"])

        self._model_meta = ONNXYoloModelMeta(imgsz, names_mapping, {})
        self._engine = DMLManager.create_dml_session(model_path)
        self._model_input_name = self._engine.get_inputs()[0].name

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(img, self._model_meta.imgsz)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgb = img_rgb.transpose(2, 0, 1)
        return np.expand_dims(img_rgb, axis=0)

    def __call__(self, img: np.ndarray) -> ONNXYoloClassifyResult:
        input_tensor = self._preprocess(img)
        outputs = DMLManager.run(self._engine, {self._model_input_name: input_tensor})
        probs = outputs[0][0]
        class_id = int(np.argmax(probs))
        score = float(probs[class_id])
        return ONNXYoloClassifyResult(class_id, score, probs, self._model_meta, img)

class CLIPModelFromONNX:
    session: ort.InferenceSession
    _input_name: str
    _lock: threading.Lock

    def __init__(self, model_path: str=None):
        if not model_path or not os.path.exists(model_path):
            model_path = resolve_runtime_str("model", "clip_visual.onnx")
        self.session = DMLManager.create_dml_session(model_path)
        self._input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _, (_, _) = letterbox(image, (224, 224))
        image = center_crop(image)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))  # [HWC] -> [CHW]
        return image[np.newaxis, :].astype(np.float32)

    def forward(self, image: np.ndarray) -> Optional[np.ndarray]:
        input_tensor = self._preprocess(image)
        try:
            output = DMLManager.run(
                self.session,
                {self._input_name: input_tensor}
            )
            return output[0]
        except Exception as e:
            logger.error(e)
            return None
