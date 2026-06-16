"""Camofox REST backend for the browser tool."""

import base64
import json
import os
import subprocess
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests

from common.log import logger
from common.utils import expand_path


_DEFAULT_BASE_URL = "http://127.0.0.1:9377"
_DEFAULT_INSTALL_DIR = "~/.cow/browser/camofox"
_DEFAULT_USER_ID = "cow-agent"
_DEFAULT_SESSION_KEY = "default"


class CamofoxBrowserService:
    """BrowserService-compatible adapter for camofox-browser REST API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        camofox_cfg = self._config.get("camofox") if isinstance(self._config.get("camofox"), dict) else {}
        self._cfg = {**self._config, **(camofox_cfg or {})}

        self._base_url = str(self._cfg.get("base_url") or os.environ.get("CAMOFOX_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self._access_key = str(self._cfg.get("access_key") or os.environ.get("CAMOFOX_ACCESS_KEY") or "").strip()
        self._admin_key = str(self._cfg.get("admin_key") or os.environ.get("CAMOFOX_ADMIN_KEY") or "").strip()
        self._user_id = str(self._cfg.get("user_id") or _DEFAULT_USER_ID)
        self._session_key = str(self._cfg.get("session_key") or _DEFAULT_SESSION_KEY)
        self._timeout = float(self._cfg.get("request_timeout") or 30)
        self._auto_start = bool(self._cfg.get("auto_start", False))
        self._managed = bool(self._cfg.get("managed", False))
        self._install_dir = expand_path(str(self._cfg.get("install_dir") or _DEFAULT_INSTALL_DIR))
        self._port = int(self._cfg.get("port") or 9377)
        self._process: Optional[subprocess.Popen] = None
        self._tab_id: Optional[str] = None
        self._screenshot_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._access_key:
            headers["Authorization"] = f"Bearer {self._access_key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request(self, method: str, path: str, **kwargs):
        self._ensure_running()
        headers = kwargs.pop("headers", None) or {}
        merged_headers = self._headers()
        merged_headers.update(headers)
        try:
            resp = requests.request(
                method,
                self._url(path),
                headers=merged_headers,
                timeout=self._timeout,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Camofox request failed: {e}") from e
        if resp.status_code >= 400:
            text = resp.text[:500]
            raise RuntimeError(f"Camofox {method} {path} failed ({resp.status_code}): {text}")
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return resp.json()
        if resp.content:
            return resp.content
        return {}

    def health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(self._url("/health"), timeout=5, headers=self._headers())
            if resp.status_code >= 400:
                return {"ok": False, "status": resp.status_code, "error": resp.text[:300]}
            data = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
            return {"ok": True, **data}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Managed process lifecycle
    # ------------------------------------------------------------------

    def _ensure_running(self):
        health = self.health()
        if health.get("ok"):
            return
        if not self._auto_start:
            raise RuntimeError(f"Camofox is not reachable at {self._base_url}: {health.get('error')}")
        self.start()
        deadline = time.time() + 20
        last = health
        while time.time() < deadline:
            time.sleep(0.5)
            last = self.health()
            if last.get("ok"):
                return
        raise RuntimeError(f"Camofox did not become healthy: {last.get('error')}")

    def _resolve_executable(self) -> str:
        configured = str(self._cfg.get("executable") or os.environ.get("CAMOFOX_BROWSER_BIN") or "").strip()
        candidates = []
        if configured:
            candidates.append(expand_path(configured))
        candidates.extend([
            os.path.join(self._install_dir, "node_modules", ".bin", "camofox-browser"),
            os.path.join(self._install_dir, "node_modules", "@askjo", "camofox-browser", "bin", "camofox-browser.js"),
        ])
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    def start(self) -> Dict[str, Any]:
        if self.health().get("ok"):
            return {"ok": True, "already_running": True}
        if not self._managed:
            return {"ok": False, "error": "Camofox managed mode is disabled"}

        executable = self._resolve_executable()
        if not executable:
            return {"ok": False, "error": f"camofox-browser is not installed under {self._install_dir}"}

        env = os.environ.copy()
        env["CAMOFOX_PORT"] = str(self._port)
        if self._access_key:
            env["CAMOFOX_ACCESS_KEY"] = self._access_key
        if self._admin_key:
            env["CAMOFOX_ADMIN_KEY"] = self._admin_key
        env.setdefault("CAMOFOX_PROFILE_DIR", os.path.join(expand_path("~/.cow"), "camofox", "profiles"))
        env.setdefault("CAMOFOX_COOKIES_DIR", os.path.join(expand_path("~/.cow"), "camofox", "cookies"))
        env.setdefault("CAMOFOX_TRACES_DIR", os.path.join(expand_path("~/.cow"), "camofox", "traces"))
        for key in ("CAMOFOX_PROFILE_DIR", "CAMOFOX_COOKIES_DIR", "CAMOFOX_TRACES_DIR"):
            os.makedirs(env[key], exist_ok=True)

        cmd = [executable]
        if executable.endswith(".js"):
            cmd = ["node", executable]
        self._process = subprocess.Popen(
            cmd,
            cwd=self._install_dir if os.path.isdir(self._install_dir) else None,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info(f"[Browser] Started managed Camofox pid={self._process.pid}")
        return {"ok": True, "pid": self._process.pid}

    def stop(self) -> Dict[str, Any]:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            return {"ok": True, "stopped": True}
        if self._admin_key:
            try:
                resp = requests.post(
                    self._url("/stop"),
                    headers={"Content-Type": "application/json", "x-admin-key": self._admin_key},
                    data="{}",
                    timeout=5,
                )
                if resp.status_code >= 400:
                    return {"ok": False, "status": resp.status_code, "error": resp.text[:300]}
                data = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
                return {"ok": True, **data}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return {"ok": False, "error": "Camofox admin_key is required to stop an external or previously-started process"}

    def close(self):
        self._tab_id = None
        if bool(self._cfg.get("stop_on_close", False)):
            self.stop()

    # ------------------------------------------------------------------
    # BrowserService-compatible actions
    # ------------------------------------------------------------------

    def _ensure_tab(self, url: str = "") -> str:
        if self._tab_id:
            return self._tab_id
        payload = {"userId": self._user_id, "sessionKey": self._session_key}
        if url:
            payload["url"] = url
        result = self._request("POST", "/tabs", data=json.dumps(payload))
        tab_id = result.get("tabId")
        if not tab_id:
            raise RuntimeError(f"Camofox did not return tabId: {result}")
        self._tab_id = tab_id
        return tab_id

    def navigate(self, url: str, timeout: int = 30000) -> Dict[str, Any]:
        try:
            if not self._tab_id:
                result = self._request("POST", "/tabs", data=json.dumps({
                    "userId": self._user_id,
                    "sessionKey": self._session_key,
                    "url": url,
                }))
                self._tab_id = result.get("tabId")
                return {"url": result.get("url", url), "title": result.get("title", ""), "status": result.get("status") or "ok"}
            result = self._request("POST", f"/tabs/{quote(self._tab_id)}/navigate", data=json.dumps({
                "userId": self._user_id,
                "url": url,
                "sessionKey": self._session_key,
            }))
            return {"url": result.get("url", url), "title": result.get("title", ""), "status": result.get("status") or "ok"}
        except Exception as e:
            return {"error": f"Navigation failed: {e}"}

    def snapshot(self, selector: Optional[str] = None) -> str:
        try:
            tab_id = self._ensure_tab()
            result = self._request(
                "GET",
                f"/tabs/{quote(tab_id)}/snapshot?userId={quote(self._user_id)}",
                headers={"Content-Type": "application/json"},
            )
            url = result.get("url", "")
            refs_count = result.get("refsCount", 0)
            body = result.get("snapshot", "")
            header = f"Page: {url}\nInteractive elements: {refs_count}\n---"
            if result.get("hasMore") and result.get("nextOffset") is not None:
                body += f"\n... [snapshot truncated, nextOffset={result.get('nextOffset')}]"
            return f"{header}\n{body}"
        except Exception as e:
            return f"[Snapshot error: {e}]"

    def click(self, ref=None, selector: Optional[str] = None, timeout: int = 5000) -> Dict[str, Any]:
        try:
            tab_id = self._ensure_tab()
            payload = {"userId": self._user_id}
            if ref is not None:
                payload["ref"] = str(ref)
            if selector:
                payload["selector"] = selector
            result = self._request("POST", f"/tabs/{quote(tab_id)}/click", data=json.dumps(payload))
            return {"clicked": True, **(result if isinstance(result, dict) else {})}
        except Exception as e:
            return {"error": f"Click failed: {e}"}

    def fill(self, text: str, ref=None, selector: Optional[str] = None, timeout: int = 5000) -> Dict[str, Any]:
        try:
            tab_id = self._ensure_tab()
            payload = {"userId": self._user_id, "text": text, "clear": True}
            if ref is not None:
                payload["ref"] = str(ref)
            if selector:
                payload["selector"] = selector
            result = self._request("POST", f"/tabs/{quote(tab_id)}/type", data=json.dumps(payload))
            return {"filled": True, "text": text, **(result if isinstance(result, dict) else {})}
        except Exception as e:
            return {"error": f"Fill failed: {e}"}

    def select(self, value: str, ref=None, selector: Optional[str] = None, timeout: int = 5000) -> Dict[str, Any]:
        if ref is not None and not selector:
            return {"error": "Camofox select by ref is not supported; provide a CSS selector"}
        if not selector:
            return {"error": "Provide selector for select action"}
        script = (
            "((sel, value) => {"
            "const el = document.querySelector(sel);"
            "if (!el) return {error: 'selector not found'};"
            "el.value = value;"
            "el.dispatchEvent(new Event('input', {bubbles: true}));"
            "el.dispatchEvent(new Event('change', {bubbles: true}));"
            "return {selected: true, value: el.value};"
            "})"
        )
        return self.evaluate(f"{script}({json.dumps(selector)}, {json.dumps(value)})")

    def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        try:
            tab_id = self._ensure_tab()
            result = self._request("POST", f"/tabs/{quote(tab_id)}/scroll", data=json.dumps({
                "userId": self._user_id,
                "direction": direction,
                "amount": int(amount),
            }))
            return {"scrolled": direction, "amount": amount, **(result if isinstance(result, dict) else {})}
        except Exception as e:
            return {"error": f"Scroll failed: {e}"}

    def wait(self, selector: Optional[str] = None, timeout: int = 5000, state: str = "visible") -> Dict[str, Any]:
        try:
            tab_id = self._ensure_tab()
            payload = {"userId": self._user_id, "timeout": int(timeout)}
            if selector:
                payload["selector"] = selector
            result = self._request("POST", f"/tabs/{quote(tab_id)}/wait", data=json.dumps(payload))
            return {"waited": True, **(result if isinstance(result, dict) else {})}
        except Exception as e:
            return {"error": f"Wait failed: {e}"}

    def go_back(self) -> Dict[str, Any]:
        return self._history("back")

    def go_forward(self) -> Dict[str, Any]:
        return self._history("forward")

    def _history(self, action: str) -> Dict[str, Any]:
        try:
            tab_id = self._ensure_tab()
            result = self._request("POST", f"/tabs/{quote(tab_id)}/{action}", data=json.dumps({"userId": self._user_id}))
            return {"url": result.get("url", ""), "title": result.get("title", "")}
        except Exception as e:
            return {"error": f"Go {action} failed: {e}"}

    def screenshot(self, full_page: bool = False, cwd: str = "") -> str:
        tab_id = self._ensure_tab()
        result = self._request("GET", f"/tabs/{quote(tab_id)}/screenshot?userId={quote(self._user_id)}")
        data = ""
        mime_type = "image/png"
        if isinstance(result, dict):
            shot = result.get("screenshot") or {}
            data = shot.get("data", "")
            mime_type = shot.get("mimeType", "image/png")
        if not data:
            raise RuntimeError(f"Camofox screenshot returned no image data: {result}")
        ext = ".png" if "png" in mime_type else ".jpg"
        save_dir = self._get_screenshot_dir(cwd)
        filepath = os.path.join(save_dir, f"screenshot_{uuid.uuid4().hex[:8]}{ext}")
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(data))
        return filepath

    def get_text(self, selector: str) -> Dict[str, Any]:
        script = (
            "() => {"
            f"const el = document.querySelector({json.dumps(selector)});"
            "return el ? (el.innerText || el.textContent || '') : '';"
            "}"
        )
        result = self.evaluate(script)
        if "error" in result:
            return result
        return {"text": result.get("result") or ""}

    def evaluate(self, script: str) -> Dict[str, Any]:
        try:
            tab_id = self._ensure_tab()
            result = self._request("POST", f"/tabs/{quote(tab_id)}/evaluate", data=json.dumps({
                "userId": self._user_id,
                "expression": script,
            }))
            return {"result": result.get("result") if isinstance(result, dict) else result}
        except Exception as e:
            return {"error": f"Evaluate failed: {e}"}

    def press(self, key: str) -> Dict[str, Any]:
        try:
            tab_id = self._ensure_tab()
            result = self._request("POST", f"/tabs/{quote(tab_id)}/press", data=json.dumps({
                "userId": self._user_id,
                "key": key,
            }))
            return {"pressed": key, **(result if isinstance(result, dict) else {})}
        except Exception as e:
            return {"error": f"Press failed: {e}"}

    def _get_screenshot_dir(self, cwd: str = "") -> str:
        if self._screenshot_dir and os.path.isdir(self._screenshot_dir):
            return self._screenshot_dir
        base = cwd or os.getcwd()
        d = os.path.join(base, "tmp")
        os.makedirs(d, exist_ok=True)
        self._screenshot_dir = d
        return d
