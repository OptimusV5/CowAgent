"""Browser service factory."""

import json
from typing import Any, Dict

from agent.tools.browser.browser_service import BrowserService
from common.log import logger


def _service_signature(config: Dict[str, Any]) -> str:
    """Return a stable signature for config values that affect backend choice."""
    browser_cfg = _normalize_browser_config(config or {})
    try:
        return json.dumps(browser_cfg, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        return repr(sorted(browser_cfg.items()))


def _normalize_browser_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both direct browser config and nested tools.browser config."""
    if not isinstance(config, dict):
        return {}
    tools_cfg = config.get("tools")
    if isinstance(tools_cfg, dict) and isinstance(tools_cfg.get("browser"), dict):
        return tools_cfg.get("browser") or {}
    return config


def _playwright_config(browser_cfg: Dict[str, Any]) -> Dict[str, Any]:
    nested = browser_cfg.get("playwright")
    if isinstance(nested, dict):
        merged = dict(browser_cfg)
        merged.update(nested)
        return merged
    return browser_cfg


def create_browser_service(config: Dict[str, Any] = None):
    """Create the configured browser backend.

    Supported engines:
      - playwright (default): local Playwright Chromium backend.
      - camofox: camofox-browser REST API backend.
      - auto: try camofox if healthy, otherwise fall back to Playwright.
    """
    config = config or {}
    browser_cfg = _normalize_browser_config(config)
    engine = str(browser_cfg.get("engine") or "playwright").strip().lower()

    if engine == "camofox":
        from agent.tools.browser.camofox_service import CamofoxBrowserService
        return CamofoxBrowserService(browser_cfg)

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


__all__ = ["create_browser_service", "_service_signature"]
