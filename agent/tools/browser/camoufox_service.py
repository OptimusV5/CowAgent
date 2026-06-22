"""Camoufox Python backend for the browser tool.

This adapter uses daijro/camoufox through its Playwright-compatible sync API.
The page operation layer is inherited from BrowserService; only browser launch
and shutdown differ from the default Chromium Playwright backend.
"""

import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from agent.tools.browser.browser_service import BrowserService, _should_use_headless
from common.log import logger
from common.proxy import normalize_proxy_url
from common.utils import expand_path


_DEFAULT_USER_DATA_DIR = "~/.cow/camoufox_profile"
_DEFAULT_INSTALL_HINT = (
    "Install Camoufox with: python3 -m pip install -U camoufox && "
    "python3 -m camoufox fetch"
)


def _compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop empty values so Camoufox can apply its own defaults."""
    return {k: v for k, v in data.items() if v is not None and v != "" and v != [] and v != {}}


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
            self._camoufox_ctx = Camoufox(**options)
            launched = self._camoufox_ctx.__enter__()
        except Exception as e:
            self._camoufox_ctx = None
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

    def _build_camoufox_options(self, launch_args: List[str], viewport: Dict[str, int]) -> Dict[str, Any]:
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
            "humanize": self._config.get("humanize", True),
            "geoip": self._config.get("geoip", False),
            "fingerprint_preset": self._config.get("fingerprint_preset", False),
            "block_images": self._config.get("block_images", None),
            "block_webrtc": self._config.get("block_webrtc", None),
            "disable_coop": self._config.get("disable_coop", None),
            "main_world_eval": self._config.get("main_world_eval", None),
            "enable_cache": self._config.get("enable_cache", None),
            "config": raw_config,
            "firefox_user_prefs": firefox_user_prefs,
            "proxy": proxy,
        }
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
