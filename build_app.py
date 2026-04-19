import io
import importlib.util
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from datetime import datetime
from pathlib import Path

from src.constants.resource_repositories import (
    GAKUMAS_DIFF_FILES,
    RESOURCE_REPOSITORIES,
    TRANSLATION_FILES,
)
from src.utils.runtime_paths import (
    RUNTIME_METADATA_FILE_NAME,
    STORAGE_MODE_MERGED,
    STORAGE_MODE_PORTABLE,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
WEBUI_DIR = PROJECT_ROOT / "web-ui"
PROJECT_NAME = "Gakumas Assistant"
TARGET_PLATFORM = platform.system()
NUITKA_OUTPUT_DIR = PROJECT_ROOT / "out"
APP_DIST_DIR = NUITKA_OUTPUT_DIR / f"{PROJECT_NAME}.dist"
APP_BUNDLE_DIR = NUITKA_OUTPUT_DIR / f"{PROJECT_NAME}.app"
MACOS_PORTABLE_DIR = NUITKA_OUTPUT_DIR / PROJECT_NAME
LOGO = PROJECT_ROOT / "assets" / "images" / "gakumas_logo.png"
NUITKA_REPORT_PATH = PROJECT_ROOT / ".cache" / "nuitka-compilation-report.xml"
WEBUI_DIST_DIR = PROJECT_ROOT / "dist"
WEBUI_BUILD_INPUTS = (
    WEBUI_DIR / "src",
    WEBUI_DIR / "public",
    WEBUI_DIR / "index.html",
    WEBUI_DIR / "package.json",
    WEBUI_DIR / "package-lock.json",
    WEBUI_DIR / "vite.config.mjs",
    WEBUI_DIR / "eslint.config.js",
)

MODEL_FILES = [
    "base_ui.onnx",
    "base_ui_meta.json",
    "contest_blocker.onnx",
    "contest_blocker_meta.json",
    "producer.onnx",
    "producer_meta.json",
    "clip_visual.onnx",
]
RAPIDOCR_DATA_FILES = [
    "rapidocr/default_models.yaml",
    "rapidocr/config.yaml",
    "rapidocr/models/ch_PP-OCRv5_mobile_det.onnx",
    "rapidocr/models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "rapidocr/models/japan_PP-OCRv4_rec_infer.onnx",
]
PYWEBVIEW_DATA_DIRECTORIES = [
    "js",
    "lib",
]
DEFAULT_PACKAGE_STORAGE_MODE = STORAGE_MODE_PORTABLE
VALID_PACKAGE_STORAGE_MODES = {STORAGE_MODE_PORTABLE, STORAGE_MODE_MERGED}


def _detect_qt_plugin_name() -> str | None:
    for package_name, plugin_name in (
        ("PySide6", "pyside6"),
        ("PyQt6", "pyqt6"),
        ("PySide2", "pyside2"),
        ("PyQt5", "pyqt5"),
    ):
        if importlib.util.find_spec(package_name):
            return plugin_name
    return None


def _normalize_package_storage_mode(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"portable", "bundled", "local"}:
        return STORAGE_MODE_PORTABLE
    if normalized in {"merged", "managed", "user", "userdata"}:
        return STORAGE_MODE_MERGED
    return None


def _get_package_storage_mode(argv: list[str] | None = None) -> str:
    argv = list(sys.argv[1:] if argv is None else argv)
    raw_mode = os.getenv("GAKUMAS_PACKAGE_STORAGE_MODE", DEFAULT_PACKAGE_STORAGE_MODE)

    for arg in argv:
        if arg in {"--portable", "--storage-mode=portable"}:
            raw_mode = STORAGE_MODE_PORTABLE
        elif arg in {"--merged", "--storage-mode=merged"}:
            raw_mode = STORAGE_MODE_MERGED
        elif arg.startswith("--storage-mode="):
            raw_mode = arg.split("=", 1)[1]

    normalized_mode = _normalize_package_storage_mode(raw_mode)
    if normalized_mode is None:
        valid_modes = ", ".join(sorted(VALID_PACKAGE_STORAGE_MODES))
        raise SystemExit(f"Invalid storage mode: {raw_mode!r}. Expected one of: {valid_modes}")
    return normalized_mode


def ignore_unnecessary(_dir, files):
    ignore_list = {".git", ".gitignore", "__pycache__", ".DS_Store"}
    return [name for name in files if name in ignore_list]


def _get_output_filename() -> str:
    return f"{PROJECT_NAME}.exe" if TARGET_PLATFORM == "Windows" else PROJECT_NAME


def update_game_database():
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def _iter_files(path: Path):
    if not path.exists():
        return
    if path.is_file():
        yield path
        return
    for file_path in path.rglob("*"):
        if file_path.is_file():
            yield file_path


def _get_latest_mtime(paths) -> float:
    latest_mtime = 0.0
    for path in paths:
        for file_path in _iter_files(path):
            latest_mtime = max(latest_mtime, file_path.stat().st_mtime)
    return latest_mtime


def _has_built_webui_dist() -> bool:
    index_html = WEBUI_DIST_DIR / "index.html"
    assets_dir = WEBUI_DIST_DIR / "assets"
    return index_html.exists() and assets_dir.exists()


def _webui_dist_is_fresh() -> bool:
    if not _has_built_webui_dist():
        return False
    latest_source_mtime = _get_latest_mtime(WEBUI_BUILD_INPUTS)
    latest_dist_mtime = _get_latest_mtime((WEBUI_DIST_DIR,))
    return latest_dist_mtime >= latest_source_mtime


def build_webui():
    if (
        not os.getenv("GITHUB_ACTIONS")
        and not os.getenv("FORCE_WEBUI_BUILD")
        and _webui_dist_is_fresh()
    ):
        print("Skipping web-ui build, existing dist is up to date")
        return

    npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
    npm_cache_dir = PROJECT_ROOT / ".cache" / "npm"
    node_modules_dir = WEBUI_DIR / "node_modules"
    vite_binary = node_modules_dir / ".bin" / ("vite.cmd" if platform.system() == "Windows" else "vite")
    npm_cache_dir.mkdir(parents=True, exist_ok=True)
    npm_env = os.environ.copy()
    npm_env.setdefault("NPM_CONFIG_CACHE", str(npm_cache_dir))
    install_args = ["ci"] if (WEBUI_DIR / "package-lock.json").exists() else ["install"]
    try:
        subprocess.run([npm_cmd, *install_args], cwd=str(WEBUI_DIR), check=True, env=npm_env)
    except subprocess.CalledProcessError:
        if install_args != ["ci"] or os.getenv("GITHUB_ACTIONS"):
            raise
        print("npm ci failed locally, retrying with npm install")
        try:
            subprocess.run([npm_cmd, "install"], cwd=str(WEBUI_DIR), check=True, env=npm_env)
        except subprocess.CalledProcessError:
            if not node_modules_dir.exists():
                raise
            print("npm install failed locally, reusing existing node_modules")
    if not vite_binary.exists():
        if not os.getenv("GITHUB_ACTIONS") and _has_built_webui_dist():
            print("Vite is unavailable locally, reusing existing dist")
            return
        raise RuntimeError("Vite is unavailable. Reinstall web-ui dependencies before building.")
    subprocess.run([npm_cmd, "run", "build"], cwd=str(WEBUI_DIR), check=True, env=npm_env)


def _add_rapidocr_data_files(nuitka_cmd: list[str]):
    purelib_dir = Path(sysconfig.get_paths()["purelib"])
    missing_files = []
    for relative_path in RAPIDOCR_DATA_FILES:
        source_path = purelib_dir / Path(relative_path)
        if not source_path.exists():
            missing_files.append(source_path)
    if missing_files:
        print("RapidOCR runtime files are missing, bootstrapping models before packaging")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from build_app import bootstrap_rapidocr_runtime_files; bootstrap_rapidocr_runtime_files()",
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
    for relative_path in RAPIDOCR_DATA_FILES:
        source_path = purelib_dir / Path(relative_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        nuitka_cmd.append(f"--include-data-files={source_path}={relative_path}")


def bootstrap_rapidocr_runtime_files():
    """直接创建 RapidOCR 实例以触发模型文件下载。

    不经过 OCRLoader（它在 macOS 会自动选择 vision 后端而跳过 RapidOCR 模型下载）。
    """
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    RapidOCR(params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.JAPAN,
        "Rec.model_type": ModelType.MOBILE,
    })


def _add_nuitka_dependency_pruning(nuitka_cmd: list[str], storage_mode: str):
    # Uvicorn standard extras pull in optional reload/runtime helpers that this app does not use.
    nofollow_modules = [
        "tkinter",
        "pytouch",
        "touch",
        "matplotlib",
        "pandas",
        "onnxruntime.training",
        "onnxruntime.quantization",
        "onnxruntime.tools",
        "watchfiles",
        "watchgod",
        "uvloop",
        "dotenv",
    ]
    if TARGET_PLATFORM == "Windows":
        # Windows 上 pywebview 使用 EdgeChromium，需要排除 Qt 平台模块以避免与 no-qt 插件冲突
        nofollow_modules.extend(
            [
                "PySide6",
                "shiboken6",
                "qtpy",
                "webview.platforms.qt",
            ]
        )
    if TARGET_PLATFORM == "Darwin":
        nofollow_modules.extend(
            [
                "PySide6",
                "shiboken6",
                "qtpy",
                "webview.platforms.qt",
            ]
        )
        if not _use_embedded_webview(storage_mode):
            nofollow_modules.extend(
                [
                    "AppKit",
                    "Cocoa",
                    "WebKit",
                    "webview.platforms.cocoa",
                ]
            )
        # --standalone 模式下 Nuitka 不允许包含 Foundation/objc/Vision（需要 --mode=app），
        # 未使用 app bundle 时必须将其排除，否则编译报 FATAL
        if not _use_macos_app_bundle(storage_mode):
            nofollow_modules.extend(["Foundation", "objc", "Vision"])
    if TARGET_PLATFORM == "Linux":
        nofollow_modules.extend(
            [
                "Foundation",
                "AppKit",
                "Cocoa",
                "objc",
                "WebKit",
            ]
        )
    for module_name in nofollow_modules:
        nuitka_cmd.append(f"--nofollow-import-to={module_name}")

    # Use Nuitka's anti-bloat switches to stop scanning deployment-irrelevant module families.
    noinclude_modes = {
        "--noinclude-setuptools-mode": "nofollow",
        "--noinclude-pytest-mode": "nofollow",
        "--noinclude-unittest-mode": "nofollow",
        "--noinclude-pydoc-mode": "nofollow",
        "--noinclude-IPython-mode": "nofollow",
        "--noinclude-dask-mode": "nofollow",
        "--noinclude-numba-mode": "nofollow",
    }
    for option_name, mode in noinclude_modes.items():
        nuitka_cmd.append(f"{option_name}={mode}")

    noinclude_data_patterns = [
        "torch/include/**",
        "torch/bin/protoc*",
    ]
    for pattern in noinclude_data_patterns:
        nuitka_cmd.append(f"--noinclude-data-files={pattern}")
    if _use_embedded_webview(storage_mode):
        # app.py imports pywebview dynamically, so keep the package entrypoint explicit.
        # Platform backends are included selectively to avoid fighting Nuitka's pywebview plugin.
        nuitka_cmd.append("--include-module=webview")
        if TARGET_PLATFORM == "Linux":
            nuitka_cmd.append("--include-module=webview.platforms.qt")
        elif TARGET_PLATFORM == "Darwin":
            nuitka_cmd.append("--include-module=webview.platforms.cocoa")
        elif TARGET_PLATFORM == "Windows":
            nuitka_cmd.append("--include-module=webview.platforms.edgechromium")
    if TARGET_PLATFORM == "Darwin" and _use_macos_app_bundle(storage_mode):
        # Foundation/objc/Vision 仅在 app bundle 模式下包含，Nuitka 要求 Foundation 必须用 --mode=app
        for module_name in ("Foundation", "objc", "Vision"):
            if importlib.util.find_spec(module_name):
                nuitka_cmd.append(f"--include-module={module_name}")
    if TARGET_PLATFORM == "Linux":
        qt_plugin_name = _detect_qt_plugin_name()
        if qt_plugin_name is None:
            raise RuntimeError(
                "No supported Qt binding is installed. Install pywebview[qt] or another supported Qt extra before packaging."
        )
        nuitka_cmd.append(f"--enable-plugin={qt_plugin_name}")
    elif TARGET_PLATFORM == "Darwin":
        if not _use_macos_app_bundle(storage_mode):
            nuitka_cmd.append("--disable-plugin=pywebview")
    elif TARGET_PLATFORM == "Windows":
        # 手动指定平台模块后禁用 pywebview 插件，避免与 no-qt 插件产生冲突
        nuitka_cmd.append("--disable-plugin=pywebview")



def _add_optional_nuitka_report(nuitka_cmd: list[str]):
    if not os.getenv("NUITKA_REPORT"):
        return
    NUITKA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nuitka_cmd.append(f"--report={NUITKA_REPORT_PATH}")
    nuitka_cmd.append("--report-diffable")


def _copy_file(source: Path, target: Path):
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_directory(source: Path, target: Path):
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore_unnecessary)


def _copy_selected_files(source_dir: Path, target_dir: Path, file_names: list[str]):
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        source_path = source_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, target_dir / file_name)


def _remove_existing_path(path: Path):
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Failed to clean previous build output: {path}. Close processes that may be using packaged files and retry."
        ) from exc


def _prepare_nuitka_output_paths():
    if os.getenv("NUITKA_CLEAN_BUILD"):
        _remove_existing_path(NUITKA_OUTPUT_DIR / "app.build")
    _remove_existing_path(NUITKA_OUTPUT_DIR / "app.build")
    _remove_existing_path(NUITKA_OUTPUT_DIR / "app.dist")
    _remove_existing_path(APP_DIST_DIR)
    _remove_existing_path(APP_BUNDLE_DIR)
    _remove_existing_path(MACOS_PORTABLE_DIR)
    _remove_existing_path(NUITKA_OUTPUT_DIR / _get_output_filename())


def _use_macos_app_bundle(storage_mode: str) -> bool:
    return TARGET_PLATFORM == "Darwin" and storage_mode == STORAGE_MODE_MERGED


def _use_embedded_webview(storage_mode: str) -> bool:
    return not (TARGET_PLATFORM == "Darwin" and storage_mode == STORAGE_MODE_PORTABLE)


def _get_nuitka_platform_options(storage_mode: str) -> list[str]:
    compilation_mode = "--standalone"
    nuitka_cmd = []
    if _use_macos_app_bundle(storage_mode):
        compilation_mode = "--mode=app"
        nuitka_cmd.extend(
            [
                f"--output-folder-name={PROJECT_NAME}",
                f"--macos-app-name={PROJECT_NAME}",
            ]
        )
    elif TARGET_PLATFORM == "Darwin":
        nuitka_cmd.append(f"--output-folder-name={PROJECT_NAME}")

    nuitka_cmd = [
        compilation_mode,
        "--module-parameter=torch-disable-jit=no",
        f"--output-filename={_get_output_filename()}",
        f"--output-dir={NUITKA_OUTPUT_DIR}",
        "--no-deployment-flag=self-execution",
        *nuitka_cmd,
    ]
    if TARGET_PLATFORM == "Windows":
        nuitka_cmd.append("--enable-plugin=no-qt")

    if shutil.which("upx"):
        nuitka_cmd.append("--enable-plugin=upx")

    if TARGET_PLATFORM == "Windows":
        nuitka_cmd.append(f"--windows-icon-from-ico={LOGO}")
        console_mode = "attach" if storage_mode == STORAGE_MODE_PORTABLE else "disable"
        nuitka_cmd.append(f"--windows-console-mode={console_mode}")
    elif TARGET_PLATFORM == "Linux":
        nuitka_cmd.append(f"--linux-icon={LOGO}")
    elif _use_macos_app_bundle(storage_mode):
        if LOGO.exists():
            nuitka_cmd.append(f"--macos-app-icon={LOGO}")
        else:
            nuitka_cmd.append("--macos-app-icon=none")
        nuitka_cmd.append("--macos-app-console-mode=disable")

    return nuitka_cmd


def _resolve_git_dir(repo_path: Path) -> Path | None:
    git_entry = repo_path / ".git"
    if git_entry.is_dir():
        return git_entry
    if not git_entry.is_file():
        return None
    first_line = git_entry.read_text(encoding="utf-8").splitlines()[0].strip()
    if not first_line.startswith("gitdir:"):
        return None
    git_dir = Path(first_line.split(":", 1)[1].strip())
    if not git_dir.is_absolute():
        git_dir = (repo_path / git_dir).resolve()
    return git_dir


def _read_git_ref(git_dir: Path, ref_name: str) -> str:
    ref_path = git_dir.joinpath(*ref_name.split("/"))
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    packed_refs_path = git_dir / "packed-refs"
    if not packed_refs_path.exists():
        return ""
    with packed_refs_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            commit_sha, current_ref, *_ = line.split()
            if current_ref == ref_name:
                return commit_sha
    return ""


def _read_git_version(repo_path: Path) -> dict[str, str] | None:
    git_dir = _resolve_git_dir(repo_path)
    if git_dir is None:
        return None
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head_value = head_path.read_text(encoding="utf-8").strip()
    if not head_value:
        return None
    if head_value.startswith("ref: "):
        ref_name = head_value[5:].strip()
        commit_sha = _read_git_ref(git_dir, ref_name)
        if not commit_sha:
            return None
        return {
            "commit": commit_sha,
            "branch": ref_name.split("/")[-1],
            "source": "git",
        }
    return {
        "commit": head_value,
        "branch": "",
        "source": "git",
    }


def _load_repository_metadata(repository) -> dict[str, str]:
    source_metadata_path = (
        PROJECT_ROOT
        / "data"
        / "resource_repository_versions"
        / f"{repository.name}.json"
    )
    if source_metadata_path.exists():
        with source_metadata_path.open("r", encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
        metadata.setdefault("source", "metadata")
        return metadata

    git_version = _read_git_version(PROJECT_ROOT / repository.path)
    if git_version is not None:
        return {
            "name": repository.name,
            "path": repository.path.replace("\\", "/"),
            "owner": repository.owner,
            "repo": repository.repo,
            "commit": git_version.get("commit", ""),
            "branch": git_version.get("branch", ""),
            "source": git_version.get("source", "git"),
        }

    return {
        "name": repository.name,
        "path": repository.path.replace("\\", "/"),
        "owner": repository.owner,
        "repo": repository.repo,
        "commit": "",
        "branch": "",
        "source": "build",
    }


def _write_resource_repository_versions(app_dist_path: Path):
    target_dir = app_dist_path / "data" / "resource_repository_versions"
    target_dir.mkdir(parents=True, exist_ok=True)
    updated_at = datetime.now().isoformat(timespec="seconds")
    for repository in RESOURCE_REPOSITORIES:
        metadata = _load_repository_metadata(repository)
        metadata.update(
            {
                "name": repository.name,
                "path": repository.path.replace("\\", "/"),
                "owner": repository.owner,
                "repo": repository.repo,
                "updated_at": updated_at,
            }
        )
        target_path = target_dir / f"{repository.name}.json"
        with target_path.open("w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)


def _copy_runtime_files(app_dist_path: Path, storage_mode: str):
    purelib_dir = Path(sysconfig.get_paths()["purelib"])
    _copy_file(
        PROJECT_ROOT / "assets" / "images" / "gakumas_logo.png",
        app_dist_path / "assets" / "images" / "gakumas_logo.png",
    )
    _copy_file(
        PROJECT_ROOT / "assets" / "NotoSerifCJKsc-Medium.otf",
        app_dist_path / "assets" / "NotoSerifCJKsc-Medium.otf",
    )
    include_repository_assets = os.getenv("INCLUDE_GAME_RESOURCE_REPOSITORIES", "").lower() in {"1", "true", "yes"}
    if include_repository_assets:
        _copy_selected_files(
            PROJECT_ROOT / "assets" / "gakumasu-diff",
            app_dist_path / "assets" / "gakumasu-diff",
            list(GAKUMAS_DIFF_FILES),
        )
        _copy_selected_files(
            PROJECT_ROOT / "assets" / "GakumasTranslationData" / "local-files" / "masterTrans",
            app_dist_path / "assets" / "GakumasTranslationData" / "local-files" / "masterTrans",
            list(TRANSLATION_FILES),
        )
    else:
        print("Skipping bundled game database and localization resources; they will be downloaded on first launch")
    _copy_directory(PROJECT_ROOT / "bin", app_dist_path / "bin")
    _copy_directory(PROJECT_ROOT / "dist", app_dist_path / "dist")
    _copy_selected_files(PROJECT_ROOT / "model", app_dist_path / "model", MODEL_FILES)
    if _use_embedded_webview(storage_mode):
        for directory_name in PYWEBVIEW_DATA_DIRECTORIES:
            source_dir = purelib_dir / "webview" / directory_name
            if source_dir.exists():
                _copy_directory(source_dir, app_dist_path / "webview" / directory_name)
    _write_resource_repository_versions(app_dist_path)


def _write_runtime_metadata(runtime_root: Path, storage_mode: str):
    metadata_path = runtime_root / RUNTIME_METADATA_FILE_NAME
    metadata = {
        "storage_mode": storage_mode,
        "embedded_webview": _use_embedded_webview(storage_mode),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_directory_contents(source: Path, target: Path):
    if not source.exists():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target_child = target / child.name
        if child.is_dir():
            shutil.copytree(child, target_child, dirs_exist_ok=True, ignore=ignore_unnecessary)
        else:
            shutil.copy2(child, target_child)


def _get_macos_bundle_versions() -> tuple[str, str]:
    ref_name = os.getenv("GITHUB_REF_NAME", "").strip()
    short_version = "0.0.0"
    if ref_name.startswith("v") and any(char.isdigit() for char in ref_name):
        short_version = ref_name[1:]

    build_version = os.getenv("GITHUB_SHA", "").strip()[:12]
    if not build_version:
        git_version = _read_git_version(PROJECT_ROOT) or {}
        build_version = git_version.get("commit", "")[:12]
    if not build_version:
        build_version = datetime.now().strftime("%Y%m%d%H%M%S")

    return short_version, build_version


def _generate_macos_bundle_icon(resources_dir: Path) -> Path | None:
    if not LOGO.exists():
        return None
    icns_path = resources_dir / "AppIcon.icns"

    try:
        from PIL import Image

        with Image.open(LOGO) as icon_image:
            icon_image.convert("RGBA").save(
                icns_path,
                format="ICNS",
                sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
            )
        return icns_path
    except (ImportError, OSError, ValueError):
        if icns_path.exists():
            icns_path.unlink()

    if not shutil.which("sips") or not shutil.which("iconutil"):
        return None

    icon_specs = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            iconset_dir = Path(temp_dir) / "AppIcon.iconset"
            iconset_dir.mkdir(parents=True, exist_ok=True)
            for file_name, size in icon_specs.items():
                subprocess.run(
                    [
                        "sips",
                        "-z",
                        str(size),
                        str(size),
                        str(LOGO),
                        "--out",
                        str(iconset_dir / file_name),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            subprocess.run(
                ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (FileNotFoundError, subprocess.CalledProcessError):
        if icns_path.exists():
            icns_path.unlink()
        return None

    return icns_path


def _write_macos_bundle_metadata(bundle_dir: Path, icon_file: Path | None):
    short_version, build_version = _get_macos_bundle_versions()
    info_plist_path = bundle_dir / "Contents" / "Info.plist"
    plist_data = {}
    if info_plist_path.exists():
        with info_plist_path.open("rb") as file_obj:
            plist_data = plistlib.load(file_obj)

    plist_data.update(
        {
            "CFBundleDisplayName": PROJECT_NAME,
            "CFBundleExecutable": PROJECT_NAME,
            "CFBundleIdentifier": "com.pigeonserver.gakumasassistant",
            "CFBundleInfoDictionaryVersion": "6.0",
            "CFBundleName": PROJECT_NAME,
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": short_version,
            "CFBundleVersion": build_version,
            "NSHighResolutionCapable": True,
        }
    )
    if icon_file is not None:
        plist_data["CFBundleIconFile"] = icon_file.name

    info_plist_path.parent.mkdir(parents=True, exist_ok=True)
    with info_plist_path.open("wb") as file_obj:
        plistlib.dump(plist_data, file_obj)


def _codesign_macos_bundle(bundle_dir: Path) -> None:
    """对 macOS app bundle 进行 ad-hoc 深度签名。

    --deep 会递归签名 .app 内所有 Mach-O 文件，确保 Gatekeeper 将整包视为
    一个整体来验证，而不是逐一拦截内部的 .so/.dylib。
    """
    if not shutil.which("codesign"):
        print("警告：未找到 codesign 工具，跳过 app bundle 签名步骤")
        return
    print(f"正在对 app bundle 进行 ad-hoc 深度签名：{bundle_dir.name}")
    subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(bundle_dir)],
        check=True,
        capture_output=True,
    )
    print("app bundle 签名完成")


def _create_macos_bundle_launch_script(bundle_dir: Path) -> None:
    """在 app bundle 旁边创建首次运行辅助脚本。

    与便携版同理：用户下载解压后 macOS 会给 .app 包附加 com.apple.quarantine，
    导致"无法打开，因为无法验证开发者"。此脚本清除隔离属性后用 open 打开 .app，
    用户只需在第一次运行时右键→打开此脚本即可。
    """
    script_name = "启动助手（首次运行）.command"
    bundle_name = bundle_dir.name  # e.g. "Gakumas Assistant.app"
    script_content = f"""#!/bin/bash
# 移除 macOS Gatekeeper 隔离标记，解除"无法验证开发者"的拦截提示。
# 仅需在首次运行时执行一次：右键 → 打开，之后可直接双击 .app 启动。
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
APP_PATH="$SCRIPT_DIR/{bundle_name}"
echo "正在移除 macOS 隔离标记，请稍候..."
xattr -cr "$APP_PATH"
xattr -d com.apple.quarantine "$0" 2>/dev/null || true
echo "完成，正在启动应用..."
open "$APP_PATH"
"""
    # 脚本放在 .app 同级目录（即 out/）
    script_path = bundle_dir.parent / script_name
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)


def _finalize_macos_bundle(bundle_dir: Path):
    if not bundle_dir.exists():
        raise FileNotFoundError(bundle_dir)
    resources_dir = bundle_dir / "Contents" / "Resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    icon_file = _generate_macos_bundle_icon(resources_dir)
    _write_macos_bundle_metadata(bundle_dir, icon_file)
    _codesign_macos_bundle(bundle_dir)
    _create_macos_bundle_launch_script(bundle_dir)


def _codesign_macos_portable(output_dir: Path) -> None:
    """对 macOS 便携版输出目录中的所有 Mach-O 文件进行 ad-hoc 重签名。

    Nuitka 编译时会附加 ad-hoc 签名，但部分文件签名可能不一致。
    重签名确保每个 .so/.dylib 都有有效的签名，避免 Gatekeeper 在加载
    动态库时以"签名无效"为由拦截（与隔离属性问题独立）。
    """
    if not shutil.which("codesign"):
        print("警告：未找到 codesign 工具，跳过 ad-hoc 重签名步骤")
        return

    targets: list[Path] = []
    main_binary = output_dir / _get_output_filename()
    if main_binary.exists():
        targets.append(main_binary)
    targets.extend(output_dir.rglob("*.so"))
    targets.extend(output_dir.rglob("*.dylib"))

    print(f"正在对 {len(targets)} 个 Mach-O 文件进行 ad-hoc 重签名...")
    for target in targets:
        subprocess.run(
            ["codesign", "--force", "--sign", "-", str(target)],
            check=True,
            capture_output=True,
        )
    print("ad-hoc 重签名完成")


def _create_macos_launch_script(output_dir: Path) -> None:
    """在便携版目录中创建首次运行辅助脚本。

    用户下载解压后，macOS 会给所有文件附加 com.apple.quarantine 隔离属性，
    导致 Gatekeeper 逐一拦截 .so/.dylib 并报"Apple 无法验证..."。
    此脚本通过 xattr -cr 递归清除隔离属性后再启动主程序，
    用户只需在第一次运行时右键→打开此脚本即可，之后可直接运行主程序。
    """
    script_name = "启动助手（首次运行）.command"
    main_binary_name = _get_output_filename()
    script_content = f"""#!/bin/bash
# 移除 macOS Gatekeeper 隔离标记，解除"Apple 无法验证..."的拦截提示。
# 仅需在首次运行时执行一次：右键 → 打开，之后可直接运行主程序。
APP_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
echo "正在移除 macOS 隔离标记，请稍候..."
xattr -cr "$APP_DIR"
echo "完成，正在启动应用..."
exec "$APP_DIR/{main_binary_name}"
"""
    script_path = output_dir / script_name
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)


def _finalize_macos_portable_output() -> Path:
    if not APP_DIST_DIR.exists():
        raise FileNotFoundError(APP_DIST_DIR)
    _remove_existing_path(MACOS_PORTABLE_DIR)
    APP_DIST_DIR.rename(MACOS_PORTABLE_DIR)
    _codesign_macos_portable(MACOS_PORTABLE_DIR)
    _create_macos_launch_script(MACOS_PORTABLE_DIR)
    return MACOS_PORTABLE_DIR


def _create_macos_bundle_launcher(macos_dir: Path):
    binary_path = macos_dir / PROJECT_NAME
    if not binary_path.exists():
        raise FileNotFoundError(binary_path)

    bundled_binary_path = macos_dir / f"{PROJECT_NAME}-bin"
    binary_path.rename(bundled_binary_path)

    launcher_script = f"""#!/bin/sh
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
exec "$SCRIPT_DIR/{bundled_binary_path.name}" "$@"
"""
    binary_path.write_text(launcher_script, encoding="utf-8")
    binary_path.chmod(0o755)


def _create_macos_app_bundle(app_dist_path: Path):
    bundle_dir = APP_BUNDLE_DIR
    macos_dir = bundle_dir / "Contents" / "MacOS"
    resources_dir = bundle_dir / "Contents" / "Resources"

    _remove_existing_path(bundle_dir)
    _copy_directory_contents(app_dist_path, macos_dir)
    resources_dir.mkdir(parents=True, exist_ok=True)

    icon_file = _generate_macos_bundle_icon(resources_dir)
    _write_macos_bundle_metadata(bundle_dir, icon_file)
    _create_macos_bundle_launcher(macos_dir)


def build_project(storage_mode: str = DEFAULT_PACKAGE_STORAGE_MODE):
    if os.getenv("GITHUB_ACTIONS"):
        update_game_database()

    nuitka_cache_dir = Path(os.environ.setdefault("NUITKA_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "nuitka")))
    nuitka_cache_dir.mkdir(parents=True, exist_ok=True)
    build_webui()
    _prepare_nuitka_output_paths()

    nuitka_cmd = _get_nuitka_platform_options(storage_mode)

    if os.getenv("GITHUB_ACTIONS") and TARGET_PLATFORM == "Windows":
        nuitka_cmd.append("--mingw64")
    if os.getenv("GITHUB_ACTIONS"):
        nuitka_cmd.append("--assume-yes-for-downloads")
    if os.getenv("NUITKA_SHOW_PROGRESS"):
        nuitka_cmd.append("--show-progress")

    _add_nuitka_dependency_pruning(nuitka_cmd, storage_mode)
    _add_optional_nuitka_report(nuitka_cmd)
    _add_rapidocr_data_files(nuitka_cmd)

    subprocess.run(
        [sys.executable, "-m", "nuitka", *nuitka_cmd, "app.py"],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    runtime_target_dir = APP_DIST_DIR
    if _use_macos_app_bundle(storage_mode):
        runtime_target_dir = APP_BUNDLE_DIR / "Contents" / "MacOS"
    _copy_runtime_files(runtime_target_dir, storage_mode)
    _write_runtime_metadata(runtime_target_dir, storage_mode)
    final_output_path = APP_DIST_DIR
    if _use_macos_app_bundle(storage_mode):
        _finalize_macos_bundle(APP_BUNDLE_DIR)
        final_output_path = APP_BUNDLE_DIR
    elif TARGET_PLATFORM == "Darwin":
        final_output_path = _finalize_macos_portable_output()
    print(f"Packaged runtime storage mode: {storage_mode}")
    print(f"Final packaged output: {final_output_path}")


if __name__ == "__main__":
    build_project(storage_mode=_get_package_storage_mode())
