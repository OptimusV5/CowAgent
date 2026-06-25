"""Browser service factory."""

import copy
import json
from typing import Any, Dict, Optional

from agent.tools.browser.browser_service import BrowserService
from common.log import logger


def _normalize_browser_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both direct browser config and nested tools.browser config."""
    if not isinstance(config, dict):
        return {}
    tools_cfg = config.get("tools")
    if isinstance(tools_cfg, dict) and isinstance(tools_cfg.get("browser"), dict):
        return tools_cfg.get("browser") or {}
    return config


def saved_camoufox_proxy(browser_cfg: Dict[str, Any]) -> str:
    """Return the saved Camoufox browser-page proxy URL from config (may be unused at launch)."""
    if not isinstance(browser_cfg, dict):
        return ""
    camoufox_cfg = browser_cfg.get("camoufox")
    if not isinstance(camoufox_cfg, dict):
        camoufox_cfg = {}
    return str(camoufox_cfg.get("proxy") or browser_cfg.get("proxy") or "").strip()


def resolve_effective_proxy(
    browser_cfg: Dict[str, Any],
    use_proxy: Optional[bool] = None,
) -> str:
    """Resolve the Camoufox browser-page proxy to apply at launch time.

    ``backend_proxy`` (Camofox REST API proxy) is unrelated; only ``camoufox.proxy``
    is considered here. Missing ``proxy_default`` is treated as false.
    """
    saved = saved_camoufox_proxy(browser_cfg)
    if not saved:
        return ""
    proxy_default = browser_cfg.get("proxy_default", False) is True
    if use_proxy is True:
        return saved
    if use_proxy is False:
        return ""
    return saved if proxy_default else ""


def _effective_browser_config(
    config: Dict[str, Any],
    use_proxy: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build browser config with the effective Camoufox page proxy applied."""
    browser_cfg = _normalize_browser_config(config or {})
    effective = copy.deepcopy(browser_cfg)
    proxy = resolve_effective_proxy(effective, use_proxy)
    camoufox = effective.get("camoufox")
    if not isinstance(camoufox, dict):
        camoufox = {}
        effective["camoufox"] = camoufox
    camoufox["proxy"] = proxy
    return effective


def _service_signature(config: Dict[str, Any], use_proxy: Optional[bool] = None) -> str:
    """Return a stable signature for config values that affect backend choice."""
    browser_cfg = _effective_browser_config(config or {}, use_proxy)
    try:
        return json.dumps(browser_cfg, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        return repr(sorted(browser_cfg.items()))


def _playwright_config(browser_cfg: Dict[str, Any]) -> Dict[str, Any]:
    nested = browser_cfg.get("playwright")
    if isinstance(nested, dict):
        merged = dict(browser_cfg)
        merged.update(nested)
        return merged
    return browser_cfg


def create_browser_service(config: Dict[str, Any] = None, use_proxy: Optional[bool] = None):
    """Create the configured browser backend.

    Supported engines:
      - playwright (default): local Playwright Chromium backend.
      - camofox: @askjo/camofox-browser REST API backend.
      - camoufox: daijro/camoufox Python Playwright-compatible backend.
      - auto: try camofox if healthy, otherwise fall back to Playwright.

  ``use_proxy`` controls whether the saved Camoufox page proxy is applied when
  ``proxy_default`` is false. It does not accept arbitrary proxy URLs.
    """
    config = config or {}
    browser_cfg = _effective_browser_config(config, use_proxy)
    engine = str(browser_cfg.get("engine") or "playwright").strip().lower()

    if engine == "camofox":
        from agent.tools.browser.camofox_service import CamofoxBrowserService
        return CamofoxBrowserService(browser_cfg)

    if engine == "camoufox":
        from agent.tools.browser.camoufox_service import CamoufoxBrowserService
        return CamoufoxBrowserService(browser_cfg)

    if engine == "auto":
        try:
            from agent.tools.browser.camofox_service import CamofoxBrowserService
            service = CamofoxBrowserService(browser_cfg)
            if service.health().get("ok"):
                return service
            logger.info("[Browser] Camofox not healthy; falling back to Playwright")
        except Exception as e:
            logger.info(f"[Browser] Camofox unavailable; falling back to Playwright: {e}")

    return BrowserService(_playwright_config(browser_cfg))


__all__ = [
    "create_browser_service",
    "_service_signature",
    "_normalize_browser_config",
    "saved_camoufox_proxy",
    "resolve_effective_proxy",
    "_effective_browser_config",
]
