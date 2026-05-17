from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from peewee import SqliteDatabase

import config
from src.core.inference.ONNX import YoloModelFromONNX
from src.core.tasks.base_ui import auto_purchase
from src.entity.Yolo import Yolo_Results
from src.models.auto_purchase import AutoPurchaseExchangeRecord
from src.utils.game_database_tools import GakumasDatabase_ItemDataUtils
from src.utils.game_tools import get_modal

SAMPLES_DIR = Path(__file__).resolve().parent / "auto_purchase_samples"
PAGE_IMAGE_PATH = SAMPLES_DIR / "daily_exchange_page_money.png"
MODAL_IMAGE_PATH = SAMPLES_DIR / "exchange_confirm_modal.png"
EXPECTED_ITEM_ID = "item-limitovermaterial-2-b-3"
EXPECTED_PAGE_MONEY = 441_069


def _jpeg_compress(frame: np.ndarray, quality: int = 35) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok is True
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded is not None
    return decoded


def _add_gaussian_noise(frame: np.ndarray, sigma: float = 10) -> np.ndarray:
    rng = np.random.RandomState(42)
    noise = rng.normal(0, sigma, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _load_sample(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None, f"Cannot load sample image: {path}"
    return image


def _build_results(image: np.ndarray, yolo_model: YoloModelFromONNX) -> Yolo_Results:
    return Yolo_Results(yolo_model(image, conf_threshold=0.7), image)


class _DebugToolsStub:
    def add_box(self, *args, **kwargs):
        return None

    def show(self):
        return None

    def clear_all(self):
        return None


class _ResultsSequenceApp:
    def __init__(self, results):
        self._results = list(results)
        self._index = 0
        self.debug_tools = _DebugToolsStub()

    @property
    def latest_results(self):
        result = self._results[min(self._index, len(self._results) - 1)]
        if self._index < len(self._results) - 1:
            self._index += 1
        return result


@pytest.fixture(scope="session")
def yolo_model():
    return YoloModelFromONNX(config.model_config["BASE_UI"])


@pytest.fixture(scope="session")
def item_db():
    return GakumasDatabase_ItemDataUtils()


@pytest.mark.parametrize(
    ("variant_name", "transform"),
    [
        ("origin", lambda image: image),
        ("jpeg35", lambda image: _jpeg_compress(image, quality=35)),
        ("gaussian10", lambda image: _add_gaussian_noise(image, sigma=10)),
    ],
)
def test_extract_exchange_confirmation_stats_on_multiple_modal_variants(variant_name, transform, yolo_model):
    image = transform(_load_sample(MODAL_IMAGE_PATH))
    results = _build_results(image, yolo_model)
    modal = get_modal(results, no_body=False)

    assert modal is not None, f"modal parse failed for variant={variant_name}"
    assert "確認" in modal.modal_title

    app = SimpleNamespace(debug_tools=_DebugToolsStub())
    stats = auto_purchase._extract_exchange_confirmation_stats(app, modal)

    assert stats.owned_before == 1238, variant_name
    assert stats.owned_after == 1248, variant_name
    assert stats.purchase_quantity == 10, variant_name
    assert stats.exchange_limit_before == 1, variant_name
    assert stats.exchange_limit_after == 0, variant_name
    assert stats.modal_money_before == EXPECTED_PAGE_MONEY, variant_name
    assert stats.modal_money_after == 437_069, variant_name


def test_read_daily_exchange_money_multiframe_uses_majority_vote(yolo_model):
    origin = _build_results(_load_sample(PAGE_IMAGE_PATH), yolo_model)
    jpeg = _build_results(_jpeg_compress(_load_sample(PAGE_IMAGE_PATH), quality=35), yolo_model)
    noisy = _build_results(_add_gaussian_noise(_load_sample(PAGE_IMAGE_PATH), sigma=10), yolo_model)
    app = _ResultsSequenceApp([origin, jpeg, noisy])

    money = auto_purchase._read_daily_exchange_money_multiframe(
        app,
        sample_count=3,
        max_attempts=3,
        interval=0,
    )

    assert money == EXPECTED_PAGE_MONEY


def test_save_exchange_record_uses_main_database_item_id(item_db):
    item_data = item_db.get_by_id(EXPECTED_ITEM_ID)
    assert item_data is not None

    stats = auto_purchase.ExchangeConfirmationStats(
        owned_before=1238,
        owned_after=1248,
        purchase_quantity=10,
        exchange_limit_before=1,
        exchange_limit_after=0,
        modal_money_before=EXPECTED_PAGE_MONEY,
        modal_money_after=437_069,
    )

    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx([AutoPurchaseExchangeRecord]):
        test_db.connect()
        test_db.create_tables([AutoPurchaseExchangeRecord])

        auto_purchase._save_exchange_record(item_data, EXPECTED_PAGE_MONEY, stats)
        row = AutoPurchaseExchangeRecord.get()

    assert row.item_id == EXPECTED_ITEM_ID
    assert row.item_name == "ロジックノート（ビジュアル）"
    assert row.purchase_quantity == 10
    assert row.owned_before == 1238
    assert row.owned_after == 1248
    assert row.page_money_before == EXPECTED_PAGE_MONEY
    assert row.modal_money_after == 437_069
    assert row.money_delta == 4000
