import hashlib
import json
import os
import platform
import shutil
import tempfile
import threading
from pathlib import Path

import onnxruntime as ort

from src.utils.logger import logger
from src.utils.runtime_paths import resolve_cache_path

class DMLManager:
    _lock = threading.Lock()
    _session_model_paths: dict[int, str] = {}
    _session_provider_names: dict[int, tuple[str, ...]] = {}
    _session_replacements: dict[int, ort.InferenceSession] = {}
    _preferred_execution_providers = (
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )

    @classmethod
    def run(cls, session: ort.InferenceSession, feeds: dict):
        with cls._lock:
            active_session = cls._resolve_session(session)
            try:
                return active_session.run(None, feeds)
            except Exception as exc:
                fallback_session = cls._fallback_to_cpu_session(active_session, exc)
                if fallback_session is None:
                    raise
                return fallback_session.run(None, feeds)

    @classmethod
    def get_lock(cls):
        with cls._lock:
            return cls._lock

    @classmethod
    def get_session_providers(cls) -> list[str]:
        available_providers = set(ort.get_available_providers())
        providers = [
            provider
            for provider in cls._preferred_execution_providers
            if provider in available_providers
        ]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    @staticmethod
    def _normalize_provider_names(providers) -> list[str]:
        normalized: list[str] = []
        for provider in providers:
            if isinstance(provider, tuple):
                normalized.append(str(provider[0]))
            else:
                normalized.append(str(provider))
        return normalized

    @classmethod
    def _remember_session(cls, session: ort.InferenceSession, model_path: str, providers) -> ort.InferenceSession:
        session_id = id(session)
        cls._session_model_paths[session_id] = model_path
        cls._session_provider_names[session_id] = tuple(cls._normalize_provider_names(providers))
        return session

    @classmethod
    def _resolve_session(cls, session: ort.InferenceSession) -> ort.InferenceSession:
        current = session
        visited: set[int] = set()
        while True:
            current_id = id(current)
            replacement = cls._session_replacements.get(current_id)
            if replacement is None or current_id in visited:
                return current
            visited.add(current_id)
            current = replacement

    # 自由维度覆盖列表：(名称, 值)
    # 修改此列表后，CoreML 缓存 key 会自动变更，旧缓存被清理
    _free_dimension_overrides: list[tuple[str, int]] = [
        ("batch", 1),
        ("batch_size", 1),
    ]

    @classmethod
    def _build_session_options(cls, extra_overrides: dict[str, int] | None = None) -> ort.SessionOptions:
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        # 将模型中的动态维度固定为具体值，
        # 避免 CoreML 因 unbounded dimension 拒绝编译子图而全部回退到 CPU
        for name, value in cls._free_dimension_overrides:
            so.add_free_dimension_override_by_name(name, value)
        if extra_overrides:
            for name, value in extra_overrides.items():
                so.add_free_dimension_override_by_name(name, value)
        return so

    @staticmethod
    def _read_model_imgsz(model_path: str) -> tuple[int, int] | None:
        """从模型的伴随 meta JSON 文件中读取 imgsz"""
        model_dir, model_file = os.path.split(model_path)
        model_name = os.path.splitext(model_file)[0]
        meta_path = os.path.join(model_dir, f"{model_name}_meta.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            imgsz = json.loads(meta.get("imgsz", "null"))
            if isinstance(imgsz, list) and len(imgsz) == 2:
                return int(imgsz[0]), int(imgsz[1])
        except Exception as e:
            logger.debug(f"DMLManager: 操作失败: {e}")

        return None

    @classmethod
    def _build_dim_overrides_for_model(cls, model_path: str) -> dict[str, int]:
        """根据模型 meta 文件自动推导空间维度的覆盖值"""
        overrides: dict[str, int] = {}
        imgsz = cls._read_model_imgsz(model_path)
        if imgsz is not None:
            overrides["height"] = imgsz[0]
            overrides["width"] = imgsz[1]
        return overrides

    @classmethod
    def _create_cpu_session(cls, model_path: str) -> ort.InferenceSession:
        extra = cls._build_dim_overrides_for_model(model_path)
        return ort.InferenceSession(
            model_path,
            sess_options=cls._build_session_options(extra),
            providers=["CPUExecutionProvider"],
        )

    @classmethod
    def _fallback_to_cpu_session(
            cls,
            session: ort.InferenceSession,
            exc: Exception,
    ) -> ort.InferenceSession | None:
        session_id = id(session)
        model_path = cls._session_model_paths.get(session_id)
        provider_names = cls._session_provider_names.get(session_id)
        if not model_path or not provider_names:
            return None
        if tuple(provider_names) == ("CPUExecutionProvider",):
            return None

        logger.warning(
            "ONNX runtime failed under providers {} for {}, fallback to CPUExecutionProvider: {}",
            list(provider_names),
            model_path,
            exc,
        )
        fallback_session = cls._remember_session(
            cls._create_cpu_session(model_path),
            model_path,
            ["CPUExecutionProvider"],
        )
        cls._session_replacements[session_id] = fallback_session
        return fallback_session

    @staticmethod
    def _get_cache_root() -> Path:
        candidates = []
        if custom_cache_dir := os.environ.get("GAKUMAS_CACHE_DIR"):
            candidates.append(Path(custom_cache_dir))

        candidates.append(resolve_cache_path("onnxruntime"))
        candidates.append(Path(tempfile.gettempdir()) / "gakumas-assistant")

        for base_dir in candidates:
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
                return base_dir
            except OSError:
                continue

        raise RuntimeError("No writable cache directory available for ONNX Runtime.")

    @classmethod
    def _build_model_cache_key(cls, model_path: str, provider_options: dict[str, str],
                               extra_dim_overrides: dict[str, int] | None = None) -> str:
        model_scope = hashlib.sha256(str(Path(model_path).resolve()).encode("utf-8")).hexdigest()[:12]
        digest = hashlib.sha256()
        for file_path in cls._iter_model_fingerprint_files(model_path):
            digest.update(str(file_path).encode("utf-8"))
            with open(file_path, "rb") as model_file:
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(repr(sorted(provider_options.items())).encode("utf-8"))
        # 维度覆盖会影响 CoreML 编译产物，纳入缓存 key 以避免使用旧缓存
        if cls._free_dimension_overrides:
            digest.update(repr(cls._free_dimension_overrides).encode("utf-8"))
        if extra_dim_overrides:
            digest.update(repr(sorted(extra_dim_overrides.items())).encode("utf-8"))
        return f"{model_scope}-{digest.hexdigest()[:12]}"

    @staticmethod
    def _iter_model_fingerprint_files(model_path: str) -> list[Path]:
        """
        返回参与 CoreML 缓存 key 的模型文件集合。

        说明：ONNX 可能采用 external data（例如 xxx.onnx.data），
        仅哈希 .onnx 主文件会导致模型更新后缓存 key 不变。
        """
        model_file = Path(model_path).resolve()
        candidates = [model_file]
        # 常见 external data 命名：model.onnx.data
        candidates.append(Path(f"{model_file}.data"))
        # 兼容切片 external data：model.onnx.data.0 / .1 ...
        candidates.extend(sorted(model_file.parent.glob(f"{model_file.name}.data*")))
        files: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                files.append(candidate)
        return files

    @staticmethod
    def _extract_coreml_cache_dir(providers) -> Path | None:
        for provider in providers:
            if not isinstance(provider, tuple):
                continue
            provider_name, provider_options = provider
            if provider_name != "CoreMLExecutionProvider":
                continue
            if not isinstance(provider_options, dict):
                continue
            cache_dir = provider_options.get("ModelCacheDirectory")
            if not cache_dir:
                continue
            return Path(str(cache_dir))
        return None

    @staticmethod
    def _clear_coreml_cache_dir(cache_dir: Path) -> None:
        if not cache_dir.exists():
            return
        try:
            shutil.rmtree(cache_dir)
            logger.warning("Removed invalid CoreML cache directory: {}", cache_dir)
        except OSError as exc:
            logger.warning("Failed to remove invalid CoreML cache {}: {}", cache_dir, exc)

    @classmethod
    def _prune_stale_coreml_cache(cls, cache_root: Path, model_path: str, model_cache_key: str) -> None:
        cache_dir = cache_root / "coreml-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_scope = model_cache_key.split("-", 1)[0]

        for child in cache_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name == model_cache_key:
                continue
            if child.name.startswith(f"{model_scope}-") or child.name.isdigit():
                try:
                    shutil.rmtree(child)
                    logger.info("Removed stale CoreML cache directory: {}", child)
                except OSError as exc:
                    logger.warning("Failed to remove stale CoreML cache {}: {}", child, exc)

    @classmethod
    def _build_provider_config(cls, model_path: str, extra_dim_overrides: dict[str, int] | None = None):
        providers = []
        available_providers = cls.get_session_providers()

        if "DmlExecutionProvider" in available_providers:
            providers.append("DmlExecutionProvider")

        if "CoreMLExecutionProvider" in available_providers:
            cache_root = cls._get_cache_root()
            tmp_dir = cache_root / "tmp"
            cache_dir = cache_root / "coreml-cache"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            # CoreML compilation can fail under the default macOS temp directory in sandboxed
            # or packaged environments, so pin it to an app-owned writable cache directory.
            os.environ["TMPDIR"] = str(tmp_dir)

            mac_version = platform.mac_ver()[0]
            try:
                major_version = int(mac_version.split(".", 1)[0]) if mac_version else 0
            except ValueError:
                major_version = 0
            model_format = "MLProgram" if major_version >= 12 else "NeuralNetwork"
            provider_options = {
                "ModelFormat": model_format,
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "0",
                "EnableOnSubgraphs": "0",
            }
            model_cache_key = cls._build_model_cache_key(model_path, provider_options, extra_dim_overrides)
            cls._prune_stale_coreml_cache(cache_root, model_path, model_cache_key)
            cache_dir = cache_root / "coreml-cache" / model_cache_key
            cache_dir.mkdir(parents=True, exist_ok=True)

            providers.append(
                (
                    "CoreMLExecutionProvider",
                    provider_options | {"ModelCacheDirectory": str(cache_dir)},
                )
            )

        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

        return providers

    @staticmethod
    def create_dml_session(model_path: str) -> ort.InferenceSession:
        extra = DMLManager._build_dim_overrides_for_model(model_path)
        so = DMLManager._build_session_options(extra)
        providers = DMLManager._build_provider_config(model_path, extra)
        logger.debug(f"Create ONNX session with providers: {providers}")
        try:
            return DMLManager._remember_session(
                ort.InferenceSession(
                    model_path,
                    sess_options=so,
                    providers=providers,
                ),
                model_path,
                providers,
            )
        except Exception as exc:
            fallback_providers = ["CPUExecutionProvider"]
            if providers == fallback_providers:
                raise
            coreml_cache_dir = DMLManager._extract_coreml_cache_dir(providers)
            if coreml_cache_dir is not None:
                # CoreML 缓存损坏时先清理并重试一次加速会话，再回退 CPU。
                DMLManager._clear_coreml_cache_dir(coreml_cache_dir)
                retry_providers = DMLManager._build_provider_config(model_path, extra)
                try:
                    logger.warning(
                        "Retry accelerated ONNX session after clearing CoreML cache for {}",
                        model_path,
                    )
                    return DMLManager._remember_session(
                        ort.InferenceSession(
                            model_path,
                            sess_options=so,
                            providers=retry_providers,
                        ),
                        model_path,
                        retry_providers,
                    )
                except Exception as retry_exc:
                    logger.warning(
                        "Retry accelerated ONNX session failed for {}: {}",
                        model_path,
                        retry_exc,
                    )
            logger.warning(
                "Create accelerated ONNX session failed for {}, fallback to CPUExecutionProvider: {}",
                model_path,
                exc,
            )
            return DMLManager._remember_session(
                ort.InferenceSession(
                    model_path,
                    sess_options=so,
                    providers=fallback_providers,
                ),
                model_path,
                fallback_providers,
            )
