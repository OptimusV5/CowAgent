# encoding:utf-8
"""Unit tests for browser proxy on-demand enablement."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.tools.browser.factory import (
    _effective_browser_config,
    _service_signature,
    resolve_effective_proxy,
    saved_camoufox_proxy,
)
from agent.tools.browser import browser_tool as browser_tool_mod
from agent.tools.browser.browser_tool import BrowserTool


class TestBrowserProxyResolution(unittest.TestCase):
  PROXY = "socks5h://127.0.0.1:1080"

  def _cfg(self, **overrides):
    base = {
      "engine": "camoufox",
      "proxy_default": False,
      "camoufox": {"proxy": self.PROXY},
    }
    base.update(overrides)
    return base

  def test_saved_proxy_read_from_camoufox(self):
    self.assertEqual(saved_camoufox_proxy(self._cfg()), self.PROXY)

  def test_proxy_default_false_requires_use_proxy(self):
    cfg = self._cfg()
    self.assertEqual(resolve_effective_proxy(cfg), "")
    self.assertEqual(resolve_effective_proxy(cfg, use_proxy=True), self.PROXY)
    self.assertEqual(resolve_effective_proxy(cfg, use_proxy=False), "")

  def test_missing_proxy_default_treated_as_false(self):
    cfg = {"camoufox": {"proxy": self.PROXY}}
    self.assertEqual(resolve_effective_proxy(cfg), "")
    self.assertEqual(resolve_effective_proxy(cfg, use_proxy=True), self.PROXY)

  def test_proxy_default_true_applies_without_use_proxy(self):
    cfg = self._cfg(proxy_default=True)
    self.assertEqual(resolve_effective_proxy(cfg), self.PROXY)
    self.assertEqual(resolve_effective_proxy(cfg, use_proxy=False), "")

  def test_effective_config_strips_proxy_when_disabled(self):
    effective = _effective_browser_config(self._cfg())
    self.assertEqual(effective["camoufox"]["proxy"], "")

  def test_effective_config_applies_proxy_when_requested(self):
    effective = _effective_browser_config(self._cfg(), use_proxy=True)
    self.assertEqual(effective["camoufox"]["proxy"], self.PROXY)

  def test_service_signature_changes_with_use_proxy(self):
    cfg = self._cfg()
    sig_direct = _service_signature(cfg)
    sig_proxy = _service_signature(cfg, use_proxy=True)
    self.assertNotEqual(sig_direct, sig_proxy)


class TestBrowserToolUseProxyValidation(unittest.TestCase):
  def test_use_proxy_true_without_saved_proxy_fails(self):
    tool = BrowserTool({"engine": "camoufox", "proxy_default": False, "camoufox": {}})
    result = tool.execute({"action": "snapshot", "use_proxy": True})
    self.assertEqual(result.status, "error")
    self.assertIn("no Camoufox browser proxy", result.result)

  def test_use_proxy_true_on_playwright_fails(self):
    tool = BrowserTool({"engine": "playwright"})
    result = tool.execute({"action": "snapshot", "use_proxy": True})
    self.assertEqual(result.status, "error")
    self.assertIn("only supported with", result.result)

  def test_use_proxy_must_be_boolean(self):
    tool = BrowserTool({"engine": "camoufox", "camoufox": {"proxy": "socks5://127.0.0.1:1080"}})
    result = tool.execute({"action": "snapshot", "use_proxy": "yes"})
    self.assertEqual(result.status, "error")
    self.assertIn("must be a boolean", result.result)


class TestBrowserToolProxyStickiness(unittest.TestCase):
  """The active proxy mode must stick across calls that omit use_proxy."""

  PROXY = "socks5h://127.0.0.1:1080"

  def _cfg(self, proxy_default=False):
    return {
      "engine": "camoufox",
      "proxy_default": proxy_default,
      "camoufox": {"proxy": self.PROXY},
    }

  def setUp(self):
    BrowserTool.reset_shared_service()
    self.built_modes = []

    def fake_create(config, use_proxy=None):
      self.built_modes.append(use_proxy)
      return MagicMock(name=f"service(use_proxy={use_proxy})")

    self._create_patch = patch.object(
      browser_tool_mod, "create_browser_service", side_effect=fake_create
    )
    self._create_patch.start()
    # Keep config resolution deterministic; avoid reading the global ToolManager.
    self._cfg_patch = patch.object(
      BrowserTool, "_effective_config", autospec=True, side_effect=lambda self: self.config
    )
    self._cfg_patch.start()

  def tearDown(self):
    self._cfg_patch.stop()
    self._create_patch.stop()
    BrowserTool.reset_shared_service()

  def test_cold_start_omitted_follows_proxy_default_false(self):
    tool = BrowserTool(self._cfg(proxy_default=False))
    tool.execute({"action": "snapshot"})
    self.assertEqual(self.built_modes, [False])
    self.assertIs(tool._active_use_proxy, False)

  def test_cold_start_omitted_follows_proxy_default_true(self):
    tool = BrowserTool(self._cfg(proxy_default=True))
    tool.execute({"action": "snapshot"})
    self.assertEqual(self.built_modes, [True])
    self.assertIs(tool._active_use_proxy, True)

  def test_explicit_true_then_omitted_stays_proxied(self):
    tool = BrowserTool(self._cfg(proxy_default=False))
    tool.execute({"action": "snapshot", "use_proxy": True})
    tool.execute({"action": "snapshot"})
    tool.execute({"action": "snapshot"})
    # Built exactly once; omitted calls must not rebuild or drop the proxy.
    self.assertEqual(self.built_modes, [True])
    self.assertIs(tool._active_use_proxy, True)

  def test_explicit_false_switches_back_to_direct(self):
    tool = BrowserTool(self._cfg(proxy_default=False))
    tool.execute({"action": "snapshot", "use_proxy": True})
    tool.execute({"action": "snapshot", "use_proxy": False})
    self.assertEqual(self.built_modes, [True, False])
    self.assertIs(tool._active_use_proxy, False)

  def test_multi_step_flow_keeps_single_service(self):
    tool = BrowserTool(self._cfg(proxy_default=False))
    tool.execute({"action": "navigate", "url": "https://example.com", "use_proxy": True})
    tool.execute({"action": "snapshot"})
    tool.execute({"action": "click", "ref": 1})
    # navigate -> snapshot -> click must reuse the same proxied service.
    self.assertEqual(self.built_modes, [True])
    self.assertIs(tool._active_use_proxy, True)

  def test_omitted_does_not_reread_proxy_default(self):
    tool = BrowserTool(self._cfg(proxy_default=False))
    tool.execute({"action": "snapshot", "use_proxy": True})
    # Flip proxy_default in config; an omitted call must ignore it and stay proxied.
    tool.config["proxy_default"] = False
    tool.execute({"action": "snapshot"})
    self.assertEqual(self.built_modes, [True])
    self.assertIs(tool._active_use_proxy, True)


if __name__ == "__main__":
  unittest.main()
