"""
Browser tool - Control a browser for web navigation and interaction.

Uses the configured browser backend under the hood. Browser instance is lazily
started on first use, reused across tool calls within the same session, and
cleaned up via close().

Launch modes (configured under `tools.browser` in config.json):
  - persistent (default): Chromium runs with a persistent user_data_dir
    (default `~/.cow/browser_profile`), so cookies and login state survive
    across runs. The user only needs to log in once.
  - cdp: When `cdp_endpoint` is set, attach to an externally launched Chrome
    via the Chrome DevTools Protocol. Lets the agent reuse the user's real
    browser (with all logins / extensions / true fingerprints).
  - fresh: Set `persistent` to false to fall back to a clean context every run.
"""

import json
import os
from typing import Dict, Any, Optional

from agent.tools.base_tool import BaseTool, ToolResult
from agent.tools.browser.factory import (
    create_browser_service,
    _service_signature,
    _normalize_browser_config,
    saved_camoufox_proxy,
)
from common.log import logger


class BrowserTool(BaseTool):
    """Single tool exposing all browser actions via an 'action' parameter."""

    name: str = "browser"
    description: str = (
        "Control a browser to navigate web pages, interact with elements, and extract content. "
        "The active backend is configured by CowAgent and may be Playwright, Camofox REST, or Camoufox. "
        "Actions: navigate, snapshot, click, fill, select, scroll, screenshot, wait, back, forward, "
        "get_text, press, evaluate.\n\n"
        "Workflow: navigate (auto-includes snapshot with element refs) → click/fill/select by ref → snapshot to verify.\n\n"
        "Use snapshot as the primary way to read pages. Use screenshot + send to show key results to the user. "
        "For login/CAPTCHA/authorization etc., screenshot and ask the user for help. "
        "Login state is persisted across sessions (cookies / localStorage are kept in a "
        "user profile directory), so once the user logs in to a site, the agent can keep "
        "using it without logging in again.\n\n"
        "Proxy (Camoufox only): set tools.browser.camoufox.proxy in config to save a browser-page "
        "proxy. Pass use_proxy=true when visiting sites that are blocked or not directly reachable "
        "from mainland China (or set proxy_default to true to always use the saved proxy). By default "
        "the saved proxy is off. Only the pre-configured proxy can be enabled — arbitrary proxy URLs "
        "are not accepted at runtime."
    )

    params: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "The browser action to perform. One of: "
                    "navigate, snapshot, click, fill, select, scroll, "
                    "screenshot, wait, back, forward, get_text, press, evaluate"
                ),
                "enum": [
                    "navigate", "snapshot", "click", "fill", "select", "scroll",
                    "screenshot", "wait", "back", "forward", "get_text", "press",
                    "evaluate"
                ]
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for 'navigate' action)"
            },
            "ref": {
                "anyOf": [{"type": "integer"}, {"type": "string"}],
                "description": "Element ref from snapshot (number for Playwright/Camoufox, e.g. 1; string for Camofox REST, e.g. e1)"
            },
            "selector": {
                "type": "string",
                "description": "CSS selector as fallback when ref is unavailable (for click/fill/select/wait/get_text)"
            },
            "text": {
                "type": "string",
                "description": "Text to type (for 'fill' action)"
            },
            "value": {
                "type": "string",
                "description": "Option value (for 'select' action)"
            },
            "key": {
                "type": "string",
                "description": "Key to press, e.g. Enter, Tab, Escape (for 'press' action)"
            },
            "direction": {
                "type": "string",
                "description": "Scroll direction: up, down, left, right (for 'scroll' action, default: down)"
            },
            "script": {
                "type": "string",
                "description": "JavaScript code to execute (for 'evaluate' action)"
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full page screenshot (for 'screenshot' action, default: false)"
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (optional, default varies by action)"
            },
            "use_proxy": {
                "type": "boolean",
                "description": (
                    "Camoufox only: when true, route browser page traffic through the proxy saved "
                    "in tools.browser.camoufox.proxy. Set this when the target site is blocked or "
                    "not directly reachable from mainland China. When false, force a direct "
                    "connection even if proxy_default is enabled. Omit to follow proxy_default "
                    "(off by default). Only the pre-configured proxy is used — arbitrary proxy URLs "
                    "are not accepted."
                )
            }
        },
        "required": ["action"]
    }

    _shared_service = None
    _shared_signature: str = ""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.cwd = self.config.get("cwd", os.getcwd())
        self._service = None
        self._service_signature = ""
        # Active proxy mode for the live browser session. None means "not yet
        # decided" (cold start); it is seeded from proxy_default on the first
        # call that omits use_proxy, and only changes on an explicit True/False.
        self._active_use_proxy = None

    def _effective_config(self) -> dict:
        """Read the latest runtime config so Web console switches hot-apply."""
        try:
            from agent.tools.tool_manager import ToolManager
            tm = ToolManager()
            runtime_config = getattr(tm, "tool_configs", {}).get(self.name)
            if isinstance(runtime_config, dict):
                self.config = runtime_config
        except Exception as e:
            logger.debug(f"[Browser] Failed to read runtime browser config: {e}")
        return self.config

    def _validate_use_proxy(self, use_proxy: Optional[bool]) -> Optional[ToolResult]:
        if use_proxy is not True:
            return None
        browser_cfg = _normalize_browser_config(self._effective_config())
        engine = str(browser_cfg.get("engine") or "playwright").strip().lower()
        if engine != "camoufox":
            return ToolResult.fail(
                "Error: use_proxy is only supported with tools.browser.engine=camoufox"
            )
        if not saved_camoufox_proxy(browser_cfg):
            return ToolResult.fail(
                "Error: use_proxy=true but no Camoufox browser proxy is configured "
                "(set tools.browser.camoufox.proxy)"
            )
        return None

    def _resolve_proxy_mode(self, use_proxy: Optional[bool]) -> bool:
        """Resolve the proxy mode for this call and update the active session mode.

        Tri-state semantics:
          - explicit True/False: switch the active mode (may rebuild service).
          - omitted (None): keep the active mode untouched. If no mode has been
            decided yet (cold start), seed it once from proxy_default.

        proxy_default only seeds the initial value; it is never re-read on
        subsequent omitted calls, so an active proxied session stays sticky.
        """
        if use_proxy is True or use_proxy is False:
            self._active_use_proxy = use_proxy
        elif self._active_use_proxy is None:
            browser_cfg = _normalize_browser_config(self._effective_config())
            self._active_use_proxy = browser_cfg.get("proxy_default", False) is True
        return self._active_use_proxy

    def _get_service(self):
        """Get or create the browser service, sharing across copies."""
        config = self._effective_config()
        if self._active_use_proxy is None:
            # Defensive: a handler called _get_service without going through
            # execute(); seed the active mode from proxy_default once.
            self._resolve_proxy_mode(None)
        signature = _service_signature(config, self._active_use_proxy)
        if self._service is not None and self._service_signature == signature:
            return self._service

        # Reuse shared service across tool copies within the same session
        if BrowserTool._shared_service is not None and BrowserTool._shared_signature == signature:
            self._service = BrowserTool._shared_service
            self._service_signature = signature
            return self._service

        if BrowserTool._shared_service is not None and BrowserTool._shared_signature != signature:
            try:
                BrowserTool._shared_service.close()
            except Exception as e:
                logger.debug(f"[Browser] Failed to close stale service: {e}")
            BrowserTool._shared_service = None
            BrowserTool._shared_signature = ""

        self._service = create_browser_service(config, use_proxy=self._active_use_proxy)
        self._service_signature = signature
        BrowserTool._shared_service = self._service
        BrowserTool._shared_signature = signature
        return self._service

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        action = args.get("action", "").strip().lower()
        if not action:
            return ToolResult.fail("Error: 'action' parameter is required")

        use_proxy = args.get("use_proxy") if "use_proxy" in args else None
        if use_proxy is not None and not isinstance(use_proxy, bool):
            return ToolResult.fail("Error: 'use_proxy' must be a boolean")
        proxy_err = self._validate_use_proxy(use_proxy)
        if proxy_err:
            return proxy_err
        self._resolve_proxy_mode(use_proxy)

        handler = self._ACTION_MAP.get(action)
        if not handler:
            valid = ", ".join(sorted(self._ACTION_MAP.keys()))
            return ToolResult.fail(f"Unknown action '{action}'. Valid actions: {valid}")

        try:
            return handler(self, args)
        except Exception as e:
            logger.error(f"[Browser] Action '{action}' error: {e}")
            return ToolResult.fail(f"Browser error ({action}): {e}")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_navigate(self, args: Dict[str, Any]) -> ToolResult:
        url = args.get("url", "").strip()
        if not url:
            return ToolResult.fail("Error: 'url' is required for navigate action")
        # Only auto-prepend https:// for bare hosts; preserve file://, about:, data:, etc.
        if "://" not in url and not url.startswith(("about:", "data:")):
            url = "https://" + url
        timeout = args.get("timeout", 30000)
        service = self._get_service()
        result = service.navigate(url, timeout=timeout)
        if "error" in result:
            return ToolResult.fail(result["error"])
        # Auto-snapshot after navigation so the agent gets page content in one call
        snapshot_text = service.snapshot()
        return ToolResult.success(
            f"Navigated to: {result['url']}\nTitle: {result['title']}\nStatus: {result['status']}\n\n"
            f"--- Page Snapshot ---\n{snapshot_text}"
        )

    def _do_snapshot(self, args: Dict[str, Any]) -> ToolResult:
        selector = args.get("selector")
        text = self._get_service().snapshot(selector=selector)
        return ToolResult.success(text)

    def _do_click(self, args: Dict[str, Any]) -> ToolResult:
        ref = args.get("ref")
        selector = args.get("selector")
        timeout = args.get("timeout", 5000)
        result = self._get_service().click(ref=ref, selector=selector, timeout=timeout)
        if "error" in result:
            return ToolResult.fail(result["error"])
        return ToolResult.success(f"Clicked successfully. Use 'snapshot' to see updated page.")

    def _do_fill(self, args: Dict[str, Any]) -> ToolResult:
        text = args.get("text", "")
        ref = args.get("ref")
        selector = args.get("selector")
        timeout = args.get("timeout", 5000)
        if not text and text != "":
            return ToolResult.fail("Error: 'text' is required for fill action")
        result = self._get_service().fill(text, ref=ref, selector=selector, timeout=timeout)
        if "error" in result:
            return ToolResult.fail(result["error"])
        return ToolResult.success(f"Filled text into element. Use 'snapshot' to verify.")

    def _do_select(self, args: Dict[str, Any]) -> ToolResult:
        value = args.get("value", "")
        ref = args.get("ref")
        selector = args.get("selector")
        timeout = args.get("timeout", 5000)
        if not value:
            return ToolResult.fail("Error: 'value' is required for select action")
        result = self._get_service().select(value, ref=ref, selector=selector, timeout=timeout)
        if "error" in result:
            return ToolResult.fail(result["error"])
        return ToolResult.success(f"Selected option '{value}'.")

    def _do_scroll(self, args: Dict[str, Any]) -> ToolResult:
        direction = args.get("direction", "down")
        amount = args.get("timeout", 500)  # reuse timeout field or default
        if "amount" in args:
            amount = args["amount"]
        result = self._get_service().scroll(direction=direction, amount=amount)
        if "error" in result:
            return ToolResult.fail(result["error"])
        pos = f"scrollY={result.get('scrollY', '?')}/{result.get('scrollHeight', '?')}"
        return ToolResult.success(f"Scrolled {direction}. Position: {pos}")

    def _do_screenshot(self, args: Dict[str, Any]) -> ToolResult:
        full_page = args.get("full_page", False)
        filepath = self._get_service().screenshot(full_page=full_page, cwd=self.cwd)
        return ToolResult.success(f"Screenshot saved to: {filepath}")

    def _do_wait(self, args: Dict[str, Any]) -> ToolResult:
        selector = args.get("selector")
        timeout = args.get("timeout", 5000)
        result = self._get_service().wait(selector=selector, timeout=timeout)
        if "error" in result:
            return ToolResult.fail(result["error"])
        return ToolResult.success(f"Wait completed.")

    def _do_back(self, args: Dict[str, Any]) -> ToolResult:
        result = self._get_service().go_back()
        if "error" in result:
            return ToolResult.fail(result["error"])
        return ToolResult.success(f"Navigated back to: {result['url']}")

    def _do_forward(self, args: Dict[str, Any]) -> ToolResult:
        result = self._get_service().go_forward()
        if "error" in result:
            return ToolResult.fail(result["error"])
        return ToolResult.success(f"Navigated forward to: {result['url']}")

    def _do_get_text(self, args: Dict[str, Any]) -> ToolResult:
        selector = args.get("selector", "").strip()
        if not selector:
            return ToolResult.fail("Error: 'selector' is required for get_text action")
        result = self._get_service().get_text(selector)
        if "error" in result:
            return ToolResult.fail(result["error"])
        return ToolResult.success(result["text"])

    def _do_press(self, args: Dict[str, Any]) -> ToolResult:
        key = args.get("key", "").strip()
        if not key:
            return ToolResult.fail("Error: 'key' is required for press action")
        result = self._get_service().press(key)
        if "error" in result:
            return ToolResult.fail(result["error"])
        return ToolResult.success(f"Pressed key: {key}")

    def _do_evaluate(self, args: Dict[str, Any]) -> ToolResult:
        script = args.get("script", "").strip()
        if not script:
            return ToolResult.fail("Error: 'script' is required for evaluate action")
        result = self._get_service().evaluate(script)
        if "error" in result:
            return ToolResult.fail(result["error"])
        val = result.get("result")
        if isinstance(val, (dict, list)):
            return ToolResult.success(json.dumps(val, ensure_ascii=False, indent=2))
        return ToolResult.success(str(val) if val is not None else "(no return value)")

    # Action dispatch table
    _ACTION_MAP = {
        "navigate": _do_navigate,
        "snapshot": _do_snapshot,
        "click": _do_click,
        "fill": _do_fill,
        "select": _do_select,
        "scroll": _do_scroll,
        "screenshot": _do_screenshot,
        "wait": _do_wait,
        "back": _do_back,
        "forward": _do_forward,
        "get_text": _do_get_text,
        "press": _do_press,
        "evaluate": _do_evaluate,
    }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def copy(self):
        """Share browser instance across tool copies (avoids re-launching)."""
        new_tool = BrowserTool(self.config)
        new_tool.model = self.model
        new_tool.context = getattr(self, "context", None)
        new_tool.cwd = self.cwd
        new_tool._service = self._service
        new_tool._service_signature = self._service_signature
        new_tool._active_use_proxy = self._active_use_proxy
        return new_tool

    def close(self):
        """Release browser resources."""
        if self._service:
            self._service.close()
            self._service = None
            self._service_signature = ""
        BrowserTool._shared_service = None
        BrowserTool._shared_signature = ""
        logger.info("[Browser] BrowserTool closed")

    @classmethod
    def reset_shared_service(cls):
        """Close the shared browser backend so config changes take effect."""
        if cls._shared_service:
            try:
                cls._shared_service.close()
            except Exception as e:
                logger.debug(f"[Browser] Failed to reset shared service: {e}")
        cls._shared_service = None
        cls._shared_signature = ""
