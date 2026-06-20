# encoding:utf-8

from typing import Dict, Optional
from urllib.parse import urlsplit, urlunsplit


SUPPORTED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}


def normalize_proxy_url(value: str) -> str:
    """Return a normalized proxy URL or raise ValueError for unsupported input."""
    proxy = str(value or "").strip()
    if not proxy:
        return ""

    parsed = urlsplit(proxy)
    scheme = parsed.scheme.lower()
    if scheme not in SUPPORTED_PROXY_SCHEMES:
        raise ValueError("proxy scheme must be one of: http, https, socks5, socks5h")
    if not parsed.hostname:
        raise ValueError("proxy host is required")
    try:
        port = parsed.port
    except ValueError as e:
        raise ValueError("proxy port must be a valid integer between 1 and 65535") from e
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("proxy port must be between 1 and 65535")
    if port is None:
        raise ValueError("proxy port is required")

    return urlunsplit((
        scheme,
        parsed.netloc,
        parsed.path or "",
        parsed.query or "",
        parsed.fragment or "",
    ))


def proxy_dict(value: str) -> Optional[Dict[str, str]]:
    proxy = normalize_proxy_url(value)
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def config_proxy_dict(key: str = "proxy") -> Optional[Dict[str, str]]:
    from config import conf

    return proxy_dict(conf().get(key) or "")


def mask_proxy_url(value: str) -> str:
    proxy = str(value or "").strip()
    if not proxy:
        return ""
    try:
        parsed = urlsplit(proxy)
        if parsed.password is None:
            return proxy
        username = parsed.username or ""
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        auth = f"{username}:****@" if username else "****@"
        return urlunsplit((
            parsed.scheme,
            f"{auth}{host}{port}",
            parsed.path or "",
            parsed.query or "",
            parsed.fragment or "",
        ))
    except Exception:
        return proxy
