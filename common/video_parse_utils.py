# encoding:utf-8

import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


YT_DLP_PROXY_SITE_DEFS = {
    "bilibili": {
        "label": "Bilibili",
        "domains": ("bilibili.com", "b23.tv"),
    },
    "youtube": {
        "label": "YouTube",
        "domains": ("youtube.com", "youtu.be", "googlevideo.com"),
    },
    "twitter": {
        "label": "X / Twitter",
        "domains": ("x.com", "twitter.com"),
    },
    "tiktok": {
        "label": "TikTok",
        "domains": ("tiktok.com",),
    },
    "douyin": {
        "label": "Douyin",
        "domains": ("douyin.com",),
    },
    "instagram": {
        "label": "Instagram",
        "domains": ("instagram.com",),
    },
    "facebook": {
        "label": "Facebook",
        "domains": ("facebook.com", "fb.watch"),
    },
    "vimeo": {
        "label": "Vimeo",
        "domains": ("vimeo.com",),
    },
}


def normalize_yt_dlp_proxy_sites(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item).strip() for item in value]
    else:
        raw_items = []

    sites = []
    seen = set()
    for item in raw_items:
        if item in YT_DLP_PROXY_SITE_DEFS and item not in seen:
            sites.append(item)
            seen.add(item)
    return sites


def yt_dlp_proxy_site_options() -> List[Dict[str, Any]]:
    return [
        {
            "id": site_id,
            "label": meta["label"],
            "domains": list(meta["domains"]),
        }
        for site_id, meta in YT_DLP_PROXY_SITE_DEFS.items()
    ]


def should_proxy_yt_dlp_url(url: str, site_ids: Any) -> bool:
    selected = normalize_yt_dlp_proxy_sites(site_ids)
    if not selected:
        return False
    host = (urlparse(url).hostname or "").strip(".").lower()
    if not host:
        return False
    for site_id in selected:
        for domain in YT_DLP_PROXY_SITE_DEFS[site_id]["domains"]:
            domain = domain.lower()
            if host == domain or host.endswith("." + domain):
                return True
    return False


def resolve_yt_dlp_command() -> Tuple[Optional[List[str]], str]:
    """Resolve yt-dlp even when CowAgent runs from a project virtualenv not on PATH."""
    binary = shutil.which("yt-dlp")
    if binary:
        return [binary], binary

    candidates = []
    executable = sys.executable
    if executable:
        bin_dir = os.path.dirname(executable)
        candidates.append(os.path.join(bin_dir, "yt-dlp"))
        candidates.append(os.path.join(bin_dir, "yt-dlp.exe"))

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_venv_bin = os.path.join(project_root, ".venv", "bin")
    candidates.append(os.path.join(project_venv_bin, "yt-dlp"))
    candidates.append(os.path.join(project_venv_bin, "yt-dlp.exe"))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return [candidate], candidate

    try:
        import yt_dlp  # noqa: F401
        if executable:
            return [executable, "-m", "yt_dlp"], f"{executable} -m yt_dlp"
    except Exception:
        pass
    return None, ""
