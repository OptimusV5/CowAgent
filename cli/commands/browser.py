"""cow browser - Manage browser backend configuration."""

import json
import os
import subprocess
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import click

from cli.utils import ensure_sys_path, load_config_json, save_config_json


def _mask_proxy_url(value: str) -> str:
    ensure_sys_path()
    from common.proxy import mask_proxy_url
    return mask_proxy_url(value)


def _normalize_proxy_url(value: str) -> str:
    ensure_sys_path()
    from common.proxy import normalize_proxy_url
    return normalize_proxy_url(value)


def _browser_config(cfg: dict) -> dict:
    tools = cfg.get("tools")
    if not isinstance(tools, dict):
        tools = {}
    browser = tools.get("browser")
    if not isinstance(browser, dict):
        browser = {}
    tools["browser"] = browser
    cfg["tools"] = tools
    return browser


def _camofox_config(browser: dict) -> dict:
    node = browser.get("camofox")
    if not isinstance(node, dict):
        node = {}
    browser["camofox"] = node
    return node


def _camoufox_config(browser: dict) -> dict:
    node = browser.get("camoufox")
    if not isinstance(node, dict):
        node = {}
    browser["camoufox"] = node
    return node


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _health(base_url: str, access_key: str = "") -> dict:
    req = Request(base_url.rstrip("/") + "/health")
    if access_key:
        req.add_header("Authorization", f"Bearer {access_key}")
    try:
        with urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            return {"ok": True, "status": resp.status, **data}
    except HTTPError as e:
        return {"ok": False, "status": e.code, "error": e.read().decode("utf-8", errors="replace")[:300]}
    except (URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": str(e)}


@click.group()
def browser():
    """Manage browser backend."""


@browser.command()
def status():
    """Show configured browser backend."""
    cfg = load_config_json()
    browser_cfg = cfg.get("tools", {}).get("browser", {}) if isinstance(cfg.get("tools"), dict) else {}
    engine = browser_cfg.get("engine", "playwright") if isinstance(browser_cfg, dict) else "playwright"
    camofox = browser_cfg.get("camofox", {}) if isinstance(browser_cfg, dict) else {}
    click.echo(f"Engine: {engine}")
    if isinstance(camofox, dict):
        click.echo(f"Camofox URL: {camofox.get('base_url', 'http://127.0.0.1:9377')}")
        if camofox.get("access_key"):
            click.echo(f"Camofox access key: {_mask(camofox.get('access_key', ''))}")
        if camofox.get("admin_key"):
            click.echo(f"Camofox admin key: {_mask(camofox.get('admin_key', ''))}")
        click.echo(f"Camofox managed: {bool(camofox.get('managed', False))}")
        click.echo(f"Camofox auto-start: {bool(camofox.get('auto_start', False))}")
    camoufox = browser_cfg.get("camoufox", {}) if isinstance(browser_cfg, dict) else {}
    if isinstance(camoufox, dict):
        click.echo(f"Camoufox user data dir: {camoufox.get('user_data_dir', '~/.cow/camoufox_profile')}")
        click.echo(f"Camoufox persistent: {camoufox.get('persistent', True) is not False}")
        if camoufox.get("proxy"):
            click.echo(f"Camoufox browser proxy: {_mask_proxy_url(camoufox.get('proxy', ''))}")
        click.echo(f"Camoufox proxy default: {browser_cfg.get('proxy_default', False) is True}")
        if camoufox.get("os"):
            click.echo(f"Camoufox target OS: {camoufox.get('os')}")


@browser.command()
@click.argument("engine", type=click.Choice(["playwright", "camofox", "camoufox", "auto"], case_sensitive=False))
@click.option("--base-url", default="", help="Camofox server URL.")
@click.option("--access-key", default="", help="Camofox access key.")
@click.option("--admin-key", default="", help="Camofox admin key for POST /stop.")
@click.option("--managed/--external", default=None, help="Whether CowAgent manages the Camofox process.")
@click.option("--auto-start/--no-auto-start", default=None, help="Whether CowAgent auto-starts managed Camofox.")
@click.option("--port", type=int, default=None, help="Managed Camofox port.")
@click.option("--user-data-dir", default="", help="Camoufox persistent profile directory.")
@click.option("--browser-proxy", default="", help="Camoufox browser traffic proxy.")
@click.option("--proxy-default/--no-proxy-default", default=None, help="Whether Camoufox uses the saved browser proxy by default.")
@click.option("--target-os", type=click.Choice(["", "windows", "macos", "linux"], case_sensitive=False), default="", help="Camoufox fingerprint target OS.")
@click.option("--persistent/--fresh", default=None, help="Whether Camoufox uses a persistent profile.")
def switch(engine, base_url, access_key, admin_key, managed, auto_start, port, user_data_dir, browser_proxy, proxy_default, target_os, persistent):
    """Switch browser backend."""
    engine = engine.lower()
    cfg = load_config_json()
    browser_cfg = _browser_config(cfg)
    browser_cfg["engine"] = engine

    if engine in ("camofox", "auto"):
        camofox = _camofox_config(browser_cfg)
        if base_url:
            camofox["base_url"] = base_url
        elif "base_url" not in camofox:
            camofox["base_url"] = "http://127.0.0.1:9377"
        if access_key:
            camofox["access_key"] = access_key
        if admin_key:
            camofox["admin_key"] = admin_key
        if managed is not None:
            camofox["managed"] = bool(managed)
        if auto_start is not None:
            camofox["auto_start"] = bool(auto_start)
        if port is not None:
            camofox["port"] = port

    if engine == "camoufox":
        camoufox = _camoufox_config(browser_cfg)
        if user_data_dir:
            camoufox["user_data_dir"] = user_data_dir
        elif "user_data_dir" not in camoufox:
            camoufox["user_data_dir"] = "~/.cow/camoufox_profile"
        if browser_proxy:
            camoufox["proxy"] = _normalize_proxy_url(browser_proxy)
        if proxy_default is not None:
            browser_cfg["proxy_default"] = bool(proxy_default)
        if target_os:
            camoufox["os"] = target_os.lower()
        if persistent is not None:
            camoufox["persistent"] = bool(persistent)

    save_config_json(cfg)
    click.echo(f"Browser engine set to: {engine}")
    click.echo("Restart CowAgent for CLI changes to affect a running process, or switch from the Web console for hot apply.")


@browser.command()
def doctor():
    """Check configured browser backend."""
    cfg = load_config_json()
    browser_cfg = cfg.get("tools", {}).get("browser", {}) if isinstance(cfg.get("tools"), dict) else {}
    engine = browser_cfg.get("engine", "playwright") if isinstance(browser_cfg, dict) else "playwright"
    click.echo(f"Configured engine: {engine}")
    if engine in ("camofox", "auto"):
        camofox = browser_cfg.get("camofox", {}) if isinstance(browser_cfg, dict) else {}
        base_url = camofox.get("base_url", "http://127.0.0.1:9377")
        result = _health(base_url, camofox.get("access_key", ""))
        if result.get("ok"):
            click.echo(f"Camofox health: ok ({base_url})")
        else:
            click.echo(f"Camofox health: failed ({base_url}) - {result.get('error') or result.get('status')}")
    elif engine == "camoufox":
        try:
            import camoufox  # noqa: F401
            click.echo("Camoufox package: installed")
        except Exception as e:
            click.echo(f"Camoufox package: missing ({e})")
            click.echo("Install with: cow install-browser --engine camoufox")
            return
        try:
            from camoufox.pkgman import launch_path
            browser_path = launch_path()
            if browser_path and os.path.exists(browser_path):
                click.echo(f"Camoufox browser runtime: installed ({browser_path})")
            else:
                click.echo("Camoufox browser runtime: missing")
                click.echo("Fetch with: python3 -m camoufox fetch")
        except Exception as e:
            click.echo(f"Camoufox browser runtime: missing ({e})")
            click.echo("Fetch with: python3 -m camoufox fetch")
    else:
        try:
            import playwright  # noqa: F401
            click.echo("Playwright package: installed")
        except Exception as e:
            click.echo(f"Playwright package: missing ({e})")


@browser.command()
def start():
    """Start managed Camofox if configured."""
    cfg = load_config_json()
    browser_cfg = cfg.get("tools", {}).get("browser", {}) if isinstance(cfg.get("tools"), dict) else {}
    camofox = browser_cfg.get("camofox", {}) if isinstance(browser_cfg, dict) else {}
    base_url = camofox.get("base_url", "http://127.0.0.1:9377")
    if _health(base_url, camofox.get("access_key", "")).get("ok"):
        click.echo(f"Camofox already running at {base_url}")
        return
    install_dir = os.path.expanduser(camofox.get("install_dir", "~/.cow/browser/camofox"))
    bin_path = os.path.join(install_dir, "node_modules", ".bin", "camofox-browser")
    if not os.path.exists(bin_path):
        click.echo("Camofox is not installed. Run: cow install-browser --engine camofox", err=True)
        raise SystemExit(1)
    env = os.environ.copy()
    env["CAMOFOX_PORT"] = str(camofox.get("port", 9377))
    if camofox.get("access_key"):
        env["CAMOFOX_ACCESS_KEY"] = camofox["access_key"]
    if camofox.get("admin_key"):
        env["CAMOFOX_ADMIN_KEY"] = camofox["admin_key"]
    home = os.path.expanduser("~/.cow")
    env.setdefault("CAMOFOX_PROFILE_DIR", os.path.join(home, "camofox", "profiles"))
    env.setdefault("CAMOFOX_COOKIES_DIR", os.path.join(home, "camofox", "cookies"))
    env.setdefault("CAMOFOX_TRACES_DIR", os.path.join(home, "camofox", "traces"))
    for key in ("CAMOFOX_PROFILE_DIR", "CAMOFOX_COOKIES_DIR", "CAMOFOX_TRACES_DIR"):
        os.makedirs(env[key], exist_ok=True)
    proc = subprocess.Popen([bin_path], cwd=install_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    click.echo(f"Started Camofox pid={proc.pid}")
    deadline = time.time() + 15
    while time.time() < deadline:
        if _health(base_url, camofox.get("access_key", "")).get("ok"):
            click.echo(f"Camofox ready at {base_url}")
            return
        time.sleep(0.5)
    click.echo("Camofox started but health check did not pass yet.", err=True)


@browser.command()
def stop():
    """Stop Camofox through the admin endpoint when configured."""
    cfg = load_config_json()
    browser_cfg = cfg.get("tools", {}).get("browser", {}) if isinstance(cfg.get("tools"), dict) else {}
    camofox = browser_cfg.get("camofox", {}) if isinstance(browser_cfg, dict) else {}
    base_url = camofox.get("base_url", "http://127.0.0.1:9377")
    admin_key = camofox.get("admin_key", "")
    req = Request(base_url.rstrip("/") + "/stop", method="POST", data=b"{}")
    req.add_header("Content-Type", "application/json")
    if admin_key:
        req.add_header("x-admin-key", admin_key)
    try:
        with urlopen(req, timeout=5) as resp:
            click.echo(resp.read().decode("utf-8", errors="replace") or f"Stopped ({resp.status})")
    except Exception as e:
        click.echo(f"Failed to stop Camofox: {e}", err=True)
        raise SystemExit(1)
