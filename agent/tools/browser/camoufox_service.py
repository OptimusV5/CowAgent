"""Camoufox Python backend for the browser tool.

This adapter uses daijro/camoufox through its Playwright-compatible sync API.
The page operation layer is inherited from BrowserService; only browser launch
and shutdown differ from the default Chromium Playwright backend.
"""

import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from agent.tools.browser.browser_service import BrowserService, _should_use_headless
from common.log import logger
from common.proxy import normalize_proxy_url
from common.utils import expand_path


_DEFAULT_USER_DATA_DIR = "~/.cow/camoufox_profile"
_DEFAULT_IDLE_TIMEOUT = 10
_DEFAULT_INSTALL_HINT = (
    "Install Camoufox with: python3 -m pip install -U camoufox && "
    "python3 -m camoufox fetch"
)
_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument ['\"]([^'\"]+)['\"]")
_OPTIONAL_CAMOUFOX_OPTIONS = {
    "fingerprint_preset",
    "humanize",
    "geoip",
    "block_images",
    "block_webrtc",
    "disable_coop",
    "main_world_eval",
    "enable_cache",
}


def _compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop empty values so Camoufox can apply its own defaults."""
    return {k: v for k, v in data.items() if v is not None and v != "" and v != [] and v != {}}


def _enabled(value: Any) -> bool:
    """Return whether an optional Camoufox-only feature was explicitly enabled."""
    return value not in (None, False, "", [], {})


def _unexpected_kwarg(err: Exception) -> str:
    match = _UNEXPECTED_KWARG_RE.search(str(err))
    return match.group(1) if match else ""


def _parse_proxy(value: str) -> Optional[Dict[str, str]]:
    proxy = normalize_proxy_url(value)
    if not proxy:
        return None
    parsed = urlsplit(proxy)
    scheme = "socks5" if parsed.scheme == "socks5h" else parsed.scheme
    server = f"{scheme}://{parsed.hostname}:{parsed.port}"
    result = {"server": server}
    if parsed.username:
        result["username"] = parsed.username
    if parsed.password:
        result["password"] = parsed.password
    return result


class CamoufoxBrowserService(BrowserService):
    """BrowserService-compatible backend powered by daijro/camoufox."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        browser_cfg = config or {}
        camoufox_cfg = browser_cfg.get("camoufox") if isinstance(browser_cfg.get("camoufox"), dict) else {}
        merged = dict(browser_cfg)
        merged.update(camoufox_cfg or {})
        merged.setdefault("idle_timeout", _DEFAULT_IDLE_TIMEOUT)
        super().__init__(merged)
        self._camoufox_ctx = None
        self._camoufox_browser_or_context = None

    def _launch_browser(self):
        """Launch Camoufox on the background thread."""
        try:
            from camoufox.sync_api import Camoufox
        except Exception as e:
            raise RuntimeError(f"Camoufox Python package is not installed. {_DEFAULT_INSTALL_HINT}") from e

        if self._cdp_endpoint:
            raise RuntimeError("Camoufox backend does not support cdp_endpoint. Use engine=playwright for CDP mode.")

        if self._headless is None:
            headless_cfg = self._config.get("headless")
            self._headless = headless_cfg if headless_cfg is not None else _should_use_headless()

        launch_args = list(self._config.get("launch_args") or [])
        if self._headless:
            launch_args.append("--no-sandbox")

        viewport_w = int(self._config.get("viewport_width", 1280) or 1280)
        viewport_h = int(self._config.get("viewport_height", 720) or 720)
        viewport = {"width": viewport_w, "height": viewport_h}

        options = self._build_camoufox_options(launch_args, viewport)
        persistent = self._launch_mode == "persistent"
        if persistent:
            os.makedirs(self._user_data_dir, exist_ok=True)
            logger.info(
                f"[Browser] Launching Camoufox (persistent, headless={self._headless}, "
                f"profile={self._user_data_dir})"
            )
        else:
            logger.info(f"[Browser] Launching Camoufox (fresh, headless={self._headless})")

        try:
            launched = self._launch_camoufox_context(Camoufox, options)
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "no such file" in msg or "browser version" in msg or "fetch" in msg:
                raise RuntimeError(f"Camoufox browser runtime is not ready. {_DEFAULT_INSTALL_HINT}. Original error: {e}") from e
            if "singletonlock" in msg or "profile" in msg or "lock" in msg:
                raise RuntimeError(
                    f"Camoufox profile '{self._user_data_dir}' is in use by another process. "
                    "Close the other browser / cow instance, or set a different "
                    "tools.browser.camoufox.user_data_dir."
                ) from e
            raise

        self._camoufox_browser_or_context = launched
        if persistent:
            self._browser = None
            self._context = launched
            pages = self._context.pages
            self._page = pages[0] if pages else self._context.new_page()
        else:
            self._browser = launched
            self._context = self._browser.new_context(viewport=viewport)
            self._page = self._context.new_page()

        self._wire_close_listeners()
        logger.info("[Browser] Camoufox ready")

    def _launch_camoufox_context(self, Camoufox, options: Dict[str, Any]):
        """Enter Camoufox, dropping unsupported optional kwargs for older versions."""
        launch_options = dict(options)
        dropped = []
        while True:
            ctx = Camoufox(**launch_options)
            try:
                launched = ctx.__enter__()
                self._camoufox_ctx = ctx
                if dropped:
                    logger.warning(
                        "[Browser] Camoufox launched without unsupported options: "
                        + ", ".join(dropped)
                    )
                return launched
            except Exception as e:
                try:
                    ctx.__exit__(type(e), e, e.__traceback__)
                except Exception:
                    pass
                self._camoufox_ctx = None
                key = _unexpected_kwarg(e)
                if key and key in launch_options and key in _OPTIONAL_CAMOUFOX_OPTIONS:
                    launch_options.pop(key, None)
                    dropped.append(key)
                    logger.warning(
                        f"[Browser] Installed Camoufox does not support option '{key}'; "
                        "retrying without it. Upgrade Camoufox to use this option."
                    )
                    continue
                raise

    def _build_camoufox_options(self, launch_args: List[str], viewport: Dict[str, int]) -> Dict[str, Any]:
        """Build Camoufox launch kwargs.

        ``proxy`` here is the effective browser-page proxy (page traffic), already
        resolved from ``tools.browser.camoufox.proxy`` plus ``proxy_default`` /
        per-call ``use_proxy``. It is unrelated to ``backend_proxy``, which only
        proxies CowAgent's HTTP calls to the Camofox REST API.
        """
        persistent = self._launch_mode == "persistent"
        raw_proxy = str(self._config.get("proxy") or "").strip()
        proxy = _parse_proxy(raw_proxy)
        os_choice = str(self._config.get("os") or "").strip().lower()
        if os_choice not in ("windows", "macos", "linux"):
            os_choice = ""

        window = self._config.get("window")
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            window = (int(viewport["width"]), int(viewport["height"]))

        firefox_user_prefs = self._config.get("firefox_user_prefs")
        if not isinstance(firefox_user_prefs, dict):
            firefox_user_prefs = {}
        else:
            firefox_user_prefs = dict(firefox_user_prefs)
        if raw_proxy.startswith("socks5h://"):
            firefox_user_prefs.setdefault("network.proxy.socks_remote_dns", True)

        raw_config = self._config.get("config")
        if not isinstance(raw_config, dict):
            raw_config = {}

        options = {
            "persistent_context": persistent,
            "headless": self._headless,
            "args": launch_args,
            "window": tuple(window),
            "os": os_choice,
            "config": raw_config,
            "firefox_user_prefs": firefox_user_prefs,
            "proxy": proxy,
        }
        for key in ("block_images", "block_webrtc", "disable_coop", "main_world_eval", "enable_cache"):
            if _enabled(self._config.get(key)):
                options[key] = self._config.get(key)
        if _enabled(self._config.get("humanize", True)):
            options["humanize"] = self._config.get("humanize", True)
        if _enabled(self._config.get("geoip", False)):
            options["geoip"] = self._config.get("geoip")
        if _enabled(self._config.get("fingerprint_preset", False)):
            options["fingerprint_preset"] = self._config.get("fingerprint_preset")
        executable_path = str(self._config.get("executable_path") or "").strip()
        if executable_path:
            options["executable_path"] = expand_path(executable_path)
        browser_version = str(self._config.get("browser") or "").strip()
        if browser_version:
            options["browser"] = browser_version
        if persistent:
            options["user_data_dir"] = self._user_data_dir
        return _compact_dict(options)

    def _shutdown_browser(self):
        """Close Camoufox context manager resources on the background thread."""
        self._cancel_idle_timer()

        if self._camoufox_ctx:
            try:
                self._camoufox_ctx.__exit__(None, None, None)
            except Exception as e:
                logger.debug(f"[Browser] camoufox close error: {e}")
        else:
            for obj, label in [
                (self._context, "context"),
                (self._browser, "browser"),
            ]:
                try:
                    if obj:
                        obj.close()
                except Exception as e:
                    logger.debug(f"[Browser] camoufox {label} close error: {e}")

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._camoufox_browser_or_context = None
        self._camoufox_ctx = None
        logger.info("[Browser] Camoufox closed")


__all__ = ["CamoufoxBrowserService"]
