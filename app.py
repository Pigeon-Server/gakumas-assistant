import asyncio
import importlib
import os
import platform
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager
import io
from time import sleep

import uvicorn
from fastapi import FastAPI

from src.utils.runtime_paths import embedded_webview_enabled, get_runtime_root, resolve_runtime_str

# Normalize the process working directory early so legacy relative paths
# resolve against the packaged runtime root instead of the caller's shell cwd.
os.chdir(str(get_runtime_root()))

if platform.system() == "Linux":
    os.environ.setdefault("QT_API", "pyside6")
    os.environ.setdefault("FORCE_QT_API", "1")
    os.environ.setdefault("PYWEBVIEW_GUI", "qt")

from src.core.web.routers import register_routes
from src.utils.logger import logger
from src.utils.args import args
from src.main import AppProcessor

WEBVIEW_MODULE = None
WEBVIEW_IMPORT_ERROR = None

PYWEBVIEW_WINDOW_BRIDGE = None

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class PyWebviewWindowBridge:
    def __init__(self):
        self.window = None
        self._maximized = False
        self._normal_bounds = None

    def bind_window(self, window):
        self.window = window
        self._maximized = False
        self._normal_bounds = None

    def _capture_bounds(self):
        if self.window is None:
            return None
        return {
            "x": int(self.window.x),
            "y": int(self.window.y),
            "width": int(self.window.width),
            "height": int(self.window.height),
        }

    def _apply_bounds(self, bounds):
        if self.window is None or not bounds:
            return
        self.window.resize(int(bounds["width"]), int(bounds["height"]))
        self.window.move(int(bounds["x"]), int(bounds["y"]))

    def get_window_state(self):
        return {
            "frameless": bool(getattr(self.window, "frameless", False)),
            "maximized": self._maximized,
        }

    def minimize_window(self):
        if self.window is not None:
            self.window.minimize()
        return self.get_window_state()

    def toggle_maximize_window(self):
        if self.window is None:
            return self.get_window_state()
        if platform.system() == "Darwin":
            if self._maximized:
                self._apply_bounds(self._normal_bounds)
            else:
                screen = getattr(self.window, "screen", None)
                self._normal_bounds = self._capture_bounds()
                if screen is None:
                    self.window.maximize()
                else:
                    self.window.resize(int(screen.width), int(screen.height))
                    self.window.move(int(screen.x), int(screen.y))
            self._maximized = not self._maximized
            return self.get_window_state()
        if self._maximized:
            self.window.restore()
        else:
            self.window.maximize()
        self._maximized = not self._maximized
        return self.get_window_state()

    def close_window(self):
        if self.window is not None:
            threading.Timer(0.1, self.window.destroy).start()
        return {"closing": True}


def start_webapp(core_processor: AppProcessor):
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        core_processor.ws_manager.set_fastapi_loop(asyncio.get_event_loop())
        core_processor.start_background_services()
        yield
        pass
    webapp = FastAPI(lifespan=lifespan)
    register_routes(webapp, core_processor, core_processor.ws_manager)
    uvicorn.run(
        webapp,
        host=args.host,
        port=args.port,
        log_level="info" if args.http_server_info else "warning",
        reload=False
    )


def _webview_enabled_for_build() -> bool:
    return embedded_webview_enabled()


def _load_pywebview_module():
    global WEBVIEW_MODULE, WEBVIEW_IMPORT_ERROR
    if WEBVIEW_MODULE is not None:
        return WEBVIEW_MODULE
    if WEBVIEW_IMPORT_ERROR is not None:
        raise RuntimeError("pywebview is unavailable") from WEBVIEW_IMPORT_ERROR
    if not _webview_enabled_for_build():
        WEBVIEW_IMPORT_ERROR = RuntimeError("Embedded WebView is disabled for this build.")
        raise RuntimeError("Embedded WebView is disabled for this build.") from WEBVIEW_IMPORT_ERROR

    try:
        if platform.system() == "Linux":
            importlib.import_module("webview.platforms.qt")
        elif platform.system() == "Darwin":
            os.environ.setdefault("PYWEBVIEW_GUI", "cocoa")
            importlib.import_module("webview.platforms.cocoa")
        WEBVIEW_MODULE = importlib.import_module("webview")
        return WEBVIEW_MODULE
    except Exception as import_error:
        WEBVIEW_IMPORT_ERROR = import_error
        raise RuntimeError("pywebview is unavailable") from import_error


def _start_pywebview(url: str):
    global PYWEBVIEW_WINDOW_BRIDGE
    webview_module = _load_pywebview_module()

    window_options: dict[str, object] = {
        "width": 1200,
        "height": 800,
        "text_select": True,
    }
    if platform.system() in {"Windows", "Darwin"}:
        window_options.update(
            {
                "frameless": True,
                "shadow": True,
                "easy_drag": False,
            }
        )
    if platform.system() == "Windows":
        window_options["easy_drag"] = True

    PYWEBVIEW_WINDOW_BRIDGE = PyWebviewWindowBridge()
    window_options["js_api"] = PYWEBVIEW_WINDOW_BRIDGE

    window = webview_module.create_window("Gakumas Assistant", url, **window_options)
    PYWEBVIEW_WINDOW_BRIDGE.bind_window(window)
    icon_path = resolve_runtime_str("assets", "images", "gakumas_logo.png")
    start_kwargs = {}
    # On Windows, pywebview's WinForms backend passes the icon to
    # System.Drawing.Icon() which only accepts .ico files and crashes
    # with PNG images. Skip the custom icon on Windows and let pywebview
    # fall back to the default executable icon.
    if platform.system() != "Windows" and os.path.exists(icon_path):
        start_kwargs["icon"] = icon_path
    if platform.system() == "Linux":
        start_kwargs["gui"] = "qt"
    elif platform.system() == "Darwin":
        start_kwargs["gui"] = "cocoa"
    webview_module.start(**start_kwargs)


def _shutdown_processor(app_processor: AppProcessor):
    try:
        app_processor.shutdown()
    except Exception as shutdown_error:
        logger.warning(f"Shutdown processor failed: {shutdown_error}")


def _run_server_forever(
    url: str,
    app_processor: AppProcessor,
    open_browser: bool = False,
):
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as browser_error:
            logger.warning(f"Open browser failed: {browser_error}")
    logger.success(f"Server started at {url}")
    try:
        while True:
            sleep(0.1)
            if app_processor.is_shutdown_requested():
                logger.info("Shutdown requested from browser mode. Shutting down.")
                break
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down.")
    finally:
        _shutdown_processor(app_processor)
    raise SystemExit(0)

if __name__ == "__main__":
    processor = AppProcessor()
    webapp_thread = threading.Thread(target=start_webapp, args=(processor,), daemon=True)
    webapp_thread.start()
    processor.start_inference_if_possible()
    # noinspection HttpUrlsUsage
    server_url = f"http://{args.host}:{args.port}"
    use_embedded_webview = _webview_enabled_for_build() and not args.not_use_webview
    if use_embedded_webview:
        try:
            _start_pywebview(server_url)
            _shutdown_processor(processor)
            raise SystemExit(0)
        except Exception as pywebview_error:
            logger.warning(f"PyWebView unavailable, fallback to browser mode: {pywebview_error}")
            _run_server_forever(server_url, processor, open_browser=True)
    else:
        if not _webview_enabled_for_build():
            logger.info("Embedded WebView is disabled for this build. Launching in browser mode.")
        _run_server_forever(server_url, processor, open_browser=not args.not_use_webview)