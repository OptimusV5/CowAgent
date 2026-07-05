import html
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from agent.tools.base_tool import BaseTool, ToolResult
from common.log import logger
from config import conf


DEFAULT_TIMEOUT = 20
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)
DEFAULT_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
    "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
)


class WeiboParseError(Exception):
    pass


class WeiboParseTool(BaseTool):
    """Fetch a Weibo status body and the first page of comments from m.weibo.cn."""

    name: str = "weibo_parse"
    description: str = (
        "Fetch and parse a Weibo post by ID. Use this when the user provides a Weibo ID "
        "or m.weibo.cn status URL and asks for the Weibo正文, comments, 评论, or summary. "
        "The tool returns the post body and first-page comments using the configured Weibo cookie."
    )

    params: dict = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Weibo status ID, e.g. 5316674700182948.",
            },
            "weibo_id": {
                "type": "string",
                "description": "Alias of id. Accepts a numeric Weibo ID.",
            },
            "url": {
                "type": "string",
                "description": "Optional m.weibo.cn status URL. The status ID is extracted from the URL.",
            },
        },
        "required": [],
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.cwd = self.config.get("cwd", os.getcwd())

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        args = args or {}
        try:
            runtime = self._runtime_config()
            weibo_id = self._resolve_weibo_id(args)
            if not runtime["cookie"]:
                raise WeiboParseError("Missing Weibo cookie. Configure tools.weibo_parse.cookie in Web config.")

            status_url = f"https://m.weibo.cn/status/{weibo_id}"
            self.report_progress("正在获取微博正文...")
            status_html = self._request_text(status_url, runtime, is_document=True)

            render_data = self._extract_render_data(status_html)
            status = self._find_status(render_data, weibo_id)
            if not status:
                status = self._fallback_status_from_html(status_html, weibo_id)
            if not status:
                raise WeiboParseError("Could not parse Weibo status body from response")

            comments = self._extract_comments(render_data)
            comments_source = "status_page"
            if not comments:
                self.report_progress("正在获取第一页评论...")
                comments = self._fetch_first_page_comments(weibo_id, runtime)
                comments_source = "hotflow"

            result = {
                "id": weibo_id,
                "url": status_url,
                "status": self._normalize_status(status),
                "comments": [self._normalize_comment(item) for item in comments],
                "_meta": {
                    "source": "m.weibo.cn",
                    "status_url": status_url,
                    "comments_source": comments_source,
                    "comment_count": len(comments),
                    "user_agent": runtime["user_agent"],
                },
            }
            return ToolResult.success(result)
        except WeiboParseError as e:
            logger.warning(f"[WeiboParse] {e}")
            return ToolResult.fail(f"Error: {e}")
        except requests.Timeout:
            return ToolResult.fail("Error: Weibo request timed out")
        except requests.ConnectionError as e:
            logger.warning(f"[WeiboParse] Connection failed: {e}")
            return ToolResult.fail("Error: Failed to connect to m.weibo.cn")
        except Exception as e:
            logger.error(f"[WeiboParse] Unexpected error: {e}", exc_info=True)
            return ToolResult.fail(f"Error: Weibo parsing failed: {e}")

    def _runtime_config(self) -> Dict[str, Any]:
        cfg = self._latest_tool_config()
        timeout = cfg.get("timeout", DEFAULT_TIMEOUT)
        try:
            timeout = max(1, int(timeout))
        except Exception:
            timeout = DEFAULT_TIMEOUT
        return {
            "cookie": self._normalize_cookie(
                cfg.get("cookie") or os.environ.get("WEIBO_COOKIE") or ""
            ),
            "user_agent": str(cfg.get("user_agent") or os.environ.get("WEIBO_USER_AGENT") or DEFAULT_USER_AGENT).strip(),
            "timeout": timeout,
        }

    def _latest_tool_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if isinstance(self.config, dict):
            cfg.update(self.config)
        try:
            tools_cfg = conf().get("tools", {})
            if isinstance(tools_cfg, dict) and isinstance(tools_cfg.get("weibo_parse"), dict):
                cfg.update(tools_cfg["weibo_parse"])
        except Exception as e:
            logger.debug(f"[WeiboParse] Failed to refresh runtime config from conf(): {e}")
        self.config = cfg
        return cfg

    def _resolve_weibo_id(self, args: Dict[str, Any]) -> str:
        raw = str(args.get("id") or args.get("weibo_id") or args.get("url") or "").strip()
        if not raw:
            raise WeiboParseError("Missing required parameter: id")
        if raw.startswith("http://") or raw.startswith("https://"):
            parsed = urlparse(raw)
            parts = [p for p in (parsed.path or "").split("/") if p]
            if "status" in parts:
                idx = parts.index("status")
                if idx + 1 < len(parts):
                    raw = parts[idx + 1]
            else:
                raw = parts[-1] if parts else raw
        raw = raw.split("?")[0].split("#")[0].strip()
        if not re.fullmatch(r"\d{6,}", raw):
            raise WeiboParseError(f"Invalid Weibo ID: {raw}")
        return raw

    def _headers(self, runtime: Dict[str, Any], is_document: bool = False) -> Dict[str, str]:
        headers = {
            "accept": DEFAULT_ACCEPT if is_document else "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cookie": runtime["cookie"],
            "priority": "u=0, i",
            "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "user-agent": runtime["user_agent"],
        }
        if is_document:
            headers.update({
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "upgrade-insecure-requests": "1",
            })
        else:
            headers.update({
                "referer": "https://m.weibo.cn/",
                "x-requested-with": "XMLHttpRequest",
            })
            xsrf = self._cookie_value(runtime["cookie"], "XSRF-TOKEN")
            if xsrf:
                headers["x-xsrf-token"] = xsrf
        return headers

    def _request_text(self, url: str, runtime: Dict[str, Any], is_document: bool = False) -> str:
        response = requests.get(
            url,
            headers=self._headers(runtime, is_document=is_document),
            timeout=runtime["timeout"],
            allow_redirects=True,
        )
        if response.status_code >= 400:
            raise WeiboParseError(f"Weibo request failed: HTTP {response.status_code}")
        response.encoding = response.encoding or "utf-8"
        text = response.text
        if "登录" in text and ("passport.weibo" in text or "m.weibo.cn/login" in text):
            raise WeiboParseError("Weibo cookie appears invalid or expired")
        return text

    def _fetch_first_page_comments(self, weibo_id: str, runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = (
            "https://m.weibo.cn/comments/hotflow"
            f"?id={weibo_id}&mid={weibo_id}&max_id_type=0"
        )
        response = requests.get(
            url,
            headers=self._headers(runtime, is_document=False),
            timeout=runtime["timeout"],
            allow_redirects=True,
        )
        if response.status_code >= 400:
            raise WeiboParseError(f"Weibo comments request failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except Exception as e:
            raise WeiboParseError(f"Weibo comments response is not JSON: {e}") from e
        if data.get("ok") not in (1, True, "1") and not data.get("data"):
            message = data.get("msg") or data.get("message") or "unknown error"
            raise WeiboParseError(f"Weibo comments request failed: {message}")
        comments = self._get_nested(data, "data", "data")
        return comments if isinstance(comments, list) else []

    def _extract_render_data(self, text: str) -> Any:
        for marker in ("$render_data", "render_data", "__INITIAL_STATE__"):
            idx = text.find(marker)
            while idx >= 0:
                eq = text.find("=", idx)
                if eq < 0:
                    break
                start = self._first_json_start(text, eq + 1)
                if start >= 0:
                    raw = self._balanced_json(text, start)
                    if raw:
                        try:
                            return json.loads(raw)
                        except Exception:
                            pass
                idx = text.find(marker, idx + len(marker))
        return {}

    def _first_json_start(self, text: str, offset: int) -> int:
        candidates = [pos for pos in (text.find("{", offset), text.find("[", offset)) if pos >= 0]
        return min(candidates) if candidates else -1

    def _balanced_json(self, text: str, start: int) -> str:
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        quote = ""
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                continue
            if ch in ("'", '"'):
                # m.weibo render data is JSON, but tolerate quoted JS strings enough
                # to avoid counting braces inside strings.
                in_string = True
                quote = ch
                continue
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[start: idx + 1]
        return ""

    def _find_status(self, data: Any, weibo_id: str) -> Optional[Dict[str, Any]]:
        candidates = []
        for item in self._walk_dicts(data):
            if "mblog" in item and isinstance(item.get("mblog"), dict):
                candidates.append(item["mblog"])
            if "status" in item and isinstance(item.get("status"), dict):
                candidates.append(item["status"])
            if self._looks_like_status(item):
                candidates.append(item)

        for item in candidates:
            item_id = str(item.get("id") or item.get("mid") or item.get("idstr") or "")
            if item_id == weibo_id:
                return item
        return candidates[0] if candidates else None

    def _fallback_status_from_html(self, text: str, weibo_id: str) -> Optional[Dict[str, Any]]:
        title = ""
        match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
        if match:
            title = self._strip_html(match.group(1))
        description = ""
        match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            text,
            flags=re.I | re.S,
        )
        if match:
            description = self._strip_html(match.group(1))
        body = description or title
        if not body:
            return None
        return {"id": weibo_id, "text": body}

    def _extract_comments(self, data: Any) -> List[Dict[str, Any]]:
        for item in self._walk_dicts(data):
            for key in ("comments", "comment_data", "commentList", "hotflow"):
                value = item.get(key)
                if isinstance(value, list):
                    comments = [c for c in value if isinstance(c, dict) and self._looks_like_comment(c)]
                    if comments:
                        return comments
            value = self._get_nested(item, "data", "data")
            if isinstance(value, list):
                comments = [c for c in value if isinstance(c, dict) and self._looks_like_comment(c)]
                if comments:
                    return comments
        return []

    def _normalize_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        user = status.get("user") if isinstance(status.get("user"), dict) else {}
        return {
            "id": str(status.get("id") or status.get("mid") or status.get("idstr") or ""),
            "created_at": status.get("created_at") or "",
            "user": {
                "id": str(user.get("id") or user.get("idstr") or ""),
                "screen_name": user.get("screen_name") or "",
            },
            "text": self._status_text(status),
            "reposts_count": self._safe_int(status.get("reposts_count")),
            "comments_count": self._safe_int(status.get("comments_count")),
            "attitudes_count": self._safe_int(status.get("attitudes_count")),
        }

    def _normalize_comment(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        return {
            "id": str(comment.get("id") or comment.get("idstr") or ""),
            "created_at": comment.get("created_at") or "",
            "user": {
                "id": str(user.get("id") or user.get("idstr") or ""),
                "screen_name": user.get("screen_name") or "",
            },
            "text": self._strip_html(comment.get("text_raw") or comment.get("text") or ""),
            "like_count": self._safe_int(comment.get("like_count") or comment.get("like_counts")),
        }

    def _status_text(self, status: Dict[str, Any]) -> str:
        candidates = [
            status.get("text_raw"),
            self._get_nested(status, "longText", "longTextContent"),
            status.get("text"),
        ]
        for value in candidates:
            text = self._strip_html(value or "")
            if text:
                return text
        return ""

    def _looks_like_status(self, item: Dict[str, Any]) -> bool:
        if not isinstance(item, dict):
            return False
        if not any(key in item for key in ("text", "text_raw", "longText")):
            return False
        return any(key in item for key in ("id", "mid", "idstr", "user"))

    def _looks_like_comment(self, item: Dict[str, Any]) -> bool:
        if not isinstance(item, dict):
            return False
        if not any(key in item for key in ("text", "text_raw")):
            return False
        return "user" in item or "created_at" in item or "like_count" in item or "like_counts" in item

    def _walk_dicts(self, data: Any):
        stack = [data]
        seen = set()
        while stack:
            cur = stack.pop()
            obj_id = id(cur)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            if isinstance(cur, dict):
                yield cur
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)

    def _strip_html(self, value: str) -> str:
        text = html.unescape(str(value or ""))
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_cookie(self, cookie: str) -> str:
        return re.sub(r"\s+", " ", str(cookie or "").strip())

    def _cookie_value(self, cookie: str, name: str) -> str:
        for part in str(cookie or "").split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key.strip() == name:
                return value.strip()
        return ""

    def _get_nested(self, data: Dict[str, Any], *keys: str) -> Any:
        cur = data
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0
