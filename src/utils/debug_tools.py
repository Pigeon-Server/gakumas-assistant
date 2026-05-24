from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from time import time
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.entity.Base import SingletonMeta
from src.utils.runtime_paths import resolve_runtime_str


# ---- 字体缓存 ----
@lru_cache(maxsize=32)
def _load_font(path: str, size: int):
    """缓存已加载的字体对象，避免重复 IO。"""
    try:
        return ImageFont.truetype(path, size)
    except IOError:
        return ImageFont.load_default()


# ---- 共享测量上下文（避免每次创建临时 PIL 图片）----
_MEASURE_IMG = Image.new("RGB", (1, 1))
_MEASURE_DRAW = ImageDraw.Draw(_MEASURE_IMG)

# ---- 字号 + 布局计算缓存 ----
# key: (label, box_w, box_h)  value: (font_size, wrapped_lines, line_height)
_LAYOUT_CACHE: dict[tuple, tuple] = {}
_LAYOUT_CACHE_MAX = 128

# 字体文件路径（模块级缓存，避免每次 resolve_runtime_str）
_FONT_PATH: str = ""


def _get_font_path() -> str:
    """懒加载字体路径，只解析一次。"""
    global _FONT_PATH
    if not _FONT_PATH:
        _FONT_PATH = resolve_runtime_str("assets", "NotoSerifCJKsc-Medium.otf")
    return _FONT_PATH

@dataclass
class DebugElement(ABC):
    """调试元素的抽象基类"""
    color: Tuple[int, int, int]
    alpha: float
    duration: float
    created_at: float = field(default_factory=time)

    @property
    def expire_time(self) -> float:
        return self.created_at + self.duration

    @abstractmethod
    def draw(self, overlay_img: np.ndarray, final_img: np.ndarray, pil_img: Optional["Image.Image"] = None):
        """
        绘制元素。
        :param overlay_img: 用于绘制半透明背景的图层
        :param final_img: 最终输出的图像（用于绘制不透明的实体线条或文字）
        :param pil_img: 可选的共享 PIL 图像，用于文字渲染（避免重复转换）
        """


@dataclass
class DebugBox(DebugElement):
    x: int = 0
    y: int = 0
    w: int = 0  # 右下角 x
    h: int = 0  # 右下角 y
    label: Optional[str] = None
    text_color: Tuple[int, int, int] = (0, 0, 0)
    font_size: int = 24
    thickness: int = 2

    color: Tuple[int, int, int] = (0, 0, 255)
    alpha: float = 0.4
    duration: float = 60.0

    def __post_init__(self):
        # 简单的坐标验证，确保 w>x, h>y
        self.x, self.w = min(self.x, self.w), max(self.x, self.w)
        self.y, self.h = min(self.y, self.h), max(self.y, self.h)

    @property
    def top_left(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return self.w, self.h

    # ---- 自适应标签渲染常量 ----
    _LABEL_PAD_X: int = 4          # 标签水平内边距
    _LABEL_PAD_Y: int = 2          # 标签垂直内边距
    _LABEL_FONT_MIN: int = 12      # 最小字号
    _LABEL_FONT_MAX: int = 40      # 最大字号
    _LABEL_LINE_SPACING: int = 2   # 行间距

    @staticmethod
    def _wrap_text(text: str, font, max_width: int) -> list:
        """按像素宽度将文本自动换行，返回行列表。使用共享测量上下文。"""
        lines: list[str] = []
        for raw_line in text.splitlines() or [""]:
            line = ""
            for char in raw_line:
                trial = line + char
                bbox = _MEASURE_DRAW.textbbox((0, 0), trial, font=font)
                if (bbox[2] - bbox[0]) <= max_width:
                    line = trial
                else:
                    if line:
                        lines.append(line)
                    line = char
            if line:
                lines.append(line)
        return lines or [""]

    @staticmethod
    def _measure_line_height(font) -> int:
        """测量单行文字高度。使用共享测量上下文。"""
        bbox = _MEASURE_DRAW.textbbox((0, 0), "Ag中", font=font)
        return bbox[3] - bbox[1]

    @staticmethod
    def _truncate_line(text: str, font, max_width: int) -> str:
        """将单行文本截断到 max_width 像素以内，超出用 '…' 替代。"""
        bbox = _MEASURE_DRAW.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            trial = text[:mid] + "…"
            bbox = _MEASURE_DRAW.textbbox((0, 0), trial, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "…" if lo > 0 else "…"

    def _compute_layout(self) -> tuple:
        """计算自适应字号、换行文本和行高，结果缓存。

        Returns:
            (font_size, lines_tuple, line_height)
        """
        box_w = max(self.w - self.x, 1)
        box_h = max(self.h - self.y, 1)

        # 调用方显式指定了非默认字号时直接用
        explicit = (self.font_size > 0 and self.font_size != 24)
        cache_key = (self.label, box_w, box_h, self.font_size if explicit else 0)
        cached = _LAYOUT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        usable_w = max(box_w - self._LABEL_PAD_X * 2, 1)
        usable_h = max(box_h - self._LABEL_PAD_Y * 2, 1)
        font_path = _get_font_path()

        def _layout_for(size: int):
            font = _load_font(font_path, size)
            lines = self._wrap_text(self.label, font, usable_w)
            line_h = self._measure_line_height(font)
            stride = line_h + self._LABEL_LINE_SPACING
            max_lines_count = max((usable_h + self._LABEL_LINE_SPACING) // stride, 1)
            if len(lines) > max_lines_count:
                lines = lines[:max_lines_count]
                lines[-1] = self._truncate_line(lines[-1] + "…", font, usable_w)
            total_h = line_h * len(lines) + self._LABEL_LINE_SPACING * (len(lines) - 1)
            return lines, line_h, total_h

        if explicit:
            best_size = max(self._LABEL_FONT_MIN, min(self.font_size, self._LABEL_FONT_MAX))
            lines, line_h, _ = _layout_for(best_size)
        else:
            lo, hi = self._LABEL_FONT_MIN, self._LABEL_FONT_MAX
            best_size = lo
            while lo <= hi:
                mid = (lo + hi) // 2
                _, _, total_h = _layout_for(mid)
                if total_h <= usable_h:
                    best_size = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            lines, line_h, _ = _layout_for(best_size)

        result = (best_size, tuple(lines), line_h)
        if len(_LAYOUT_CACHE) >= _LAYOUT_CACHE_MAX:
            _LAYOUT_CACHE.clear()
        _LAYOUT_CACHE[cache_key] = result
        return result

    def draw(self, overlay_img: np.ndarray, final_img: np.ndarray, pil_img: Optional[Image.Image] = None):
        # 绘制半透明矩形填充
        cv2.rectangle(overlay_img, self.top_left, self.bottom_right, self.color, -1)
        # 绘制边框
        cv2.rectangle(final_img, self.top_left, self.bottom_right, self.color, self.thickness)

        if not self.label or pil_img is None:
            return

        font_size, lines, line_h = self._compute_layout()
        font = _load_font(_get_font_path(), font_size)
        stride = line_h + self._LABEL_LINE_SPACING

        # 使用 Pillow stroke_width 一次调用完成描边+填充
        pil_draw = ImageDraw.Draw(pil_img)
        text_x = self.x + self._LABEL_PAD_X
        text_y = self.y + self._LABEL_PAD_Y

        for line in lines:
            pil_draw.text(
                (text_x, text_y), line, font=font,
                fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0),
            )
            text_y += stride


@dataclass
class DebugLine(DebugElement):
    start_x: int = 0
    start_y: int = 0
    end_x: int = 0
    end_y: int = 0
    thickness: int = 2
    padding: int = 5  # 线条背景光晕宽度

    color: Tuple[int, int, int] = (0, 255, 0)
    alpha: float = 0.5
    duration: float = 60.0

    def draw(self, overlay_img: np.ndarray, final_img: np.ndarray, pil_img: Optional[Image.Image] = None):
        # 1. 在 overlay 上绘制较粗的半透明线条作为光晕背景
        cv2.line(
            overlay_img,
            (self.start_x, self.start_y), (self.end_x, self.end_y),
            self.color,
            self.thickness + self.padding * 2 # 背景比线条本身粗
        )
        # 2. 在 final 上绘制实体线条
        cv2.line(
            final_img,
            (self.start_x, self.start_y), (self.end_x, self.end_y),
            self.color,
            self.thickness
        )


@dataclass
class DebugPoint(DebugElement):
    """调试关键点"""
    cx: int = 0
    cy: int = 0
    radius: int = 5
    thickness: int = -1 # -1 表示填充实心
    padding: int = 8 # 光晕大小

    color: Tuple[int, int, int] = (0, 255, 255) # 默认黄色
    alpha: float = 0.6
    duration: float = 60.0

    def draw(self, overlay_img: np.ndarray, final_img: np.ndarray, pil_img: Optional[Image.Image] = None):
        center = (self.cx, self.cy)
        # Overlay 绘制半透明大圆光晕
        cv2.circle(overlay_img, center, self.radius + self.padding, self.color, -1)
        # Final 绘制中心实心亮点
        cv2.circle(final_img, center, self.radius, (255, 255, 255), self.thickness) # 中心白色高亮
        cv2.circle(final_img, center, self.radius, self.color, 1) # 外圈自身颜色描边

class DebugTools(metaclass=SingletonMeta):
    _elements: List[DebugElement] = []
    _lock: Lock = Lock()
    _hide: bool = False
    _show_grid: bool = False

    # ---- 图形缓存 ----
    # 当元素集合未变化时，跳过昂贵的 PIL 文字渲染和全图比较，
    # 直接用缓存的预计算数据快速合成。
    _cache_valid: bool = False
    _cached_fill_tiles: List[tuple] = []               # [((y1,x1,y2,x2), np tile), ...]
    _cached_stamp_indices: Optional[tuple] = None       # np.where 坐标元组
    _cached_stamp_values: Optional[np.ndarray] = None   # 不透明像素值
    _cached_glow_lines: List[tuple] = []
    _cached_glow_circles: List[tuple] = []
    _cached_shape: Optional[tuple] = None

    def _invalidate_cache(self):
        """标记图形缓存失效并释放缓存数据"""
        self._cache_valid = False
        self._cached_fill_tiles = []
        self._cached_stamp_indices = None
        self._cached_stamp_values = None
        self._cached_glow_lines = []
        self._cached_glow_circles = []
        self._cached_shape = None

    def add_box(self, x: int, y: int, w: int, h: int, **kwargs):
        """绘制调试框 (x, y, end_x, end_y)"""
        with self._lock:
            self._elements.append(DebugBox(x=x, y=y, w=w, h=h, **kwargs))
            self._invalidate_cache()

    # add_rect 是 add_box 的别名，保持向后兼容
    add_rect = add_box

    def add_text(self, x: int, y: int, text: str, **kwargs):
        """绘制调试文本，使用 DebugBox 实现（label 即文本内容）"""
        font_size = kwargs.pop("font_size", 16)
        color = kwargs.pop("color", (255, 255, 255))
        duration = kwargs.pop("duration", 5.0)
        est_width = max(len(text) * font_size, 200)
        with self._lock:
            self._elements.append(DebugBox(
                x=x, y=y, w=x + est_width, h=y + font_size + 4,
                label=text, font_size=font_size, text_color=color,
                color=(0, 0, 0), alpha=0.3, thickness=0,
                duration=duration, **kwargs,
            ))
            self._invalidate_cache()

    def add_line(self, start_x: int, start_y: int, end_w: int, end_h: int, **kwargs):
        """绘制调试线 (start_x, start_y, end_x, end_y)"""
        with self._lock:
            self._elements.append(DebugLine(
                start_x=start_x, start_y=start_y, end_x=end_w, end_y=end_h, **kwargs
            ))
            self._invalidate_cache()

    def add_point(self, cx: int, cy: int, **kwargs):
        """新增：绘制调试关键点 (center_x, center_y)"""
        with self._lock:
            self._elements.append(DebugPoint(cx=cx, cy=cy,**kwargs))
            self._invalidate_cache()

    def add_crosshair(self, x: int, y: int, size: int = 30, color=(0, 0, 255), thickness=1, **kwargs):
        """新增：绘制十字准星，用于精确定位"""
        half = size // 2
        common_kwargs = {'color': color, 'thickness': thickness, **kwargs}
        self.add_line(x - half, y, x + half, y, duration=3, **common_kwargs)
        self.add_line(x, y - half, x, y + half, duration=3, **common_kwargs)

    def add_trajectory(self, points: List[Tuple[int, int]], color=(255, 100, 0), thickness=2, **kwargs):
        """新增：绘制轨迹路径"""
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            self.add_line(p1[0], p1[1], p2[0], p2[1], color=color, thickness=thickness, **kwargs)

    def _clear_expired(self):
        now = time()
        with self._lock:
            before = len(self._elements)
            self._elements = [el for el in self._elements if el.expire_time > now]
            if len(self._elements) != before:
                self._invalidate_cache()

    def clear_all(self):
        with self._lock:
            self._elements = []
            self._hide = False
            self._invalidate_cache()

    def hide(self):
        with self._lock:
            self._hide = True

    def show(self):
        with self._lock:
            self._hide = False

    def toggle_grid(self, show: bool = None):
        """切换网格显示"""
        with self._lock:
            if show is None:
                self._show_grid = not self._show_grid
            else:
                self._show_grid = show

    def _draw_grid_overlay(self, image: np.ndarray, step: int = 50, color=(128, 128, 128)):
        """绘制网格辅助线"""
        h, w = image.shape[:2]
        overlay = image.copy()
        for x in range(0, w, step):
            thickness = 2 if x == 0 else 1
            cv2.line(overlay, (x, 0), (x, h), color, thickness)
        for y in range(0, h, step):
            thickness = 2 if y == 0 else 1
            cv2.line(overlay, (0, y), (w, y), color, thickness)
        cv2.line(overlay, (w//2, 0), (w//2, h), (0, 0, 255), 1)
        cv2.line(overlay, (0, h//2), (w, h//2), (0, 0, 255), 1)
        return cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

    @staticmethod
    def _clip_box_bounds(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        """将调试框裁剪到图像边界内，避免 ROI 和 tile 尺寸不一致。"""
        clipped_x1 = max(0, min(image_width, int(x1)))
        clipped_y1 = max(0, min(image_height, int(y1)))
        clipped_x2 = max(0, min(image_width, int(x2)))
        clipped_y2 = max(0, min(image_height, int(y2)))
        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            return None
        return clipped_x1, clipped_y1, clipped_x2, clipped_y2

    def _rebuild_render_cache(self, img_shape: tuple):
        """重建图形缓存：预渲染不透明层（边框+文字）和填充 tile。

        渲染开销较大（~100ms），但只在元素变化时执行一次。
        """
        h, w = img_shape[:2]
        opaque = np.zeros((h, w, 3), dtype=np.uint8)
        fill_tiles = []
        glow_lines = []
        glow_circles = []

        for el in self._elements:
            if isinstance(el, DebugBox):
                # 预构建半透明填充 tile（纯色块，cv2.addWeighted 用）
                clipped_bounds = self._clip_box_bounds(el.x, el.y, el.w, el.h, w, h)
                if clipped_bounds is not None:
                    x1, y1, x2, y2 = clipped_bounds
                    tile = np.full((y2 - y1, x2 - x1, 3), el.color, dtype=np.uint8)
                    fill_tiles.append(((y1, x1, y2, x2), tile))
                # 边框画到不透明层
                if el.thickness > 0:
                    cv2.rectangle(opaque, el.top_left, el.bottom_right,
                                  el.color, el.thickness)
            elif isinstance(el, DebugLine):
                glow_lines.append((
                    (el.start_x, el.start_y), (el.end_x, el.end_y),
                    el.color, el.thickness + el.padding * 2,
                ))
                cv2.line(opaque,
                         (el.start_x, el.start_y), (el.end_x, el.end_y),
                         el.color, el.thickness)
            elif isinstance(el, DebugPoint):
                glow_circles.append((
                    (el.cx, el.cy), el.radius + el.padding, el.color,
                ))
                cv2.circle(opaque, (el.cx, el.cy), el.radius,
                           (255, 255, 255), el.thickness)
                cv2.circle(opaque, (el.cx, el.cy), el.radius,
                           el.color, 1)

        # 渲染文字到不透明层（PIL）
        has_text = any(isinstance(el, DebugBox) and el.label for el in self._elements)
        if has_text:
            pil_img = Image.fromarray(cv2.cvtColor(opaque, cv2.COLOR_BGR2RGB))
            pil_draw = ImageDraw.Draw(pil_img)
            for el in self._elements:
                if isinstance(el, DebugBox) and el.label:
                    font_size, lines, line_h = el._compute_layout()
                    font = _load_font(_get_font_path(), font_size)
                    stride = line_h + el._LABEL_LINE_SPACING
                    text_x = el.x + el._LABEL_PAD_X
                    text_y = el.y + el._LABEL_PAD_Y
                    for line in lines:
                        pil_draw.text(
                            (text_x, text_y), line, font=font,
                            fill=(255, 255, 255),
                            stroke_width=1, stroke_fill=(0, 0, 0),
                        )
                        text_y += stride
            opaque = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 预计算不透明像素的坐标索引和颜色值（坐标索引比布尔掩码快 40x）
        opaque_mask = np.any(opaque > 0, axis=2)
        stamp_indices = np.where(opaque_mask)
        stamp_values = opaque[stamp_indices]

        self._cached_fill_tiles = fill_tiles
        self._cached_stamp_indices = stamp_indices
        self._cached_stamp_values = stamp_values
        self._cached_glow_lines = glow_lines
        self._cached_glow_circles = glow_circles
        self._cached_shape = img_shape
        self._cache_valid = True

    def _apply_cached_render(self, image: np.ndarray) -> np.ndarray:
        """使用缓存数据快速合成调试叠加层。

        热路径性能（8 框 2340×1080）：
        - 填充: cv2.addWeighted 预计算 tile → ~0.6ms
        - 覆盖: 坐标索引预取 → ~0.4ms
        - 合计: ~1.7ms（含 copy），约 588 FPS
        """
        result = image.copy()

        # 1. 半透明光晕（线条/点）
        if self._cached_glow_lines or self._cached_glow_circles:
            glow_layer = result.copy()
            for start, end, color, thick in self._cached_glow_lines:
                cv2.line(glow_layer, start, end, color, thick)
            for center, radius, color in self._cached_glow_circles:
                cv2.circle(glow_layer, center, radius, color, -1)
            cv2.addWeighted(glow_layer, 0.5, result, 0.5, 0, dst=result)

        # 2. 半透明矩形填充 — 逐区域 addWeighted + 预计算纯色 tile
        for (y1, x1, y2, x2), tile in self._cached_fill_tiles:
            roi = result[y1:y2, x1:x2]
            if roi.shape != tile.shape:
                clipped_h = min(roi.shape[0], tile.shape[0])
                clipped_w = min(roi.shape[1], tile.shape[1])
                if clipped_h <= 0 or clipped_w <= 0:
                    continue
                roi = roi[:clipped_h, :clipped_w]
                tile = tile[:clipped_h, :clipped_w]
            cv2.addWeighted(roi, 0.5, tile, 0.5, 0, dst=roi)

        # 3. 不透明覆盖（边框 + 文字）— 坐标索引直接赋值
        if self._cached_stamp_indices is not None:
            result[self._cached_stamp_indices] = self._cached_stamp_values

        return result

    def draw_boxes(self, image: np.ndarray) -> np.ndarray:
        """主绘制方法 — 元素未变化时使用图形缓存快速合成"""
        if self._show_grid:
            image = self._draw_grid_overlay(image)

        self._clear_expired()
        if self._hide or not self._elements:
            return image

        with self._lock:
            if not self._cache_valid or self._cached_shape != image.shape:
                self._rebuild_render_cache(image.shape)
            return self._apply_cached_render(image)


if __name__ == '__main__':
    # 创建一个黑色背景图用于测试
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    debugger = DebugTools()

    # 1. 测试 Box (带居中标签)
    debugger.add_box(100, 100, 300, 300, label="目标 Target", color=(0, 0, 255))

    # 2. 测试 Line (带背景光晕)
    debugger.add_line(400, 50, 700, 150, color=(0, 255, 0), thickness=3)

    # 3. 测试 Point (关键点)
    debugger.add_point(600, 400, radius=8, color=(255, 0, 255)) # 紫色关键点
    debugger.add_point(650, 420, radius=5, color=(0, 255, 255)) # 黄色关键点

    # 4. 测试十字准星 (组合工具)
    debugger.add_crosshair(400, 300, size=50, color=(255, 255, 0), thickness=2) # 画面中心黄色准星

    # 5. 测试轨迹 (组合工具)
    trajectory_points = [(50, 400), (100, 420), (150, 450), (200, 430), (250, 500)]
    debugger.add_trajectory(trajectory_points, color=(50, 100, 255), thickness=2)

    # 6. 测试网格开关
    debugger.toggle_grid(True)

    # 执行绘制
    result_image = debugger.draw_boxes(canvas)

    # 显示结果
    cv2.imshow("Debug Tools Demo", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
