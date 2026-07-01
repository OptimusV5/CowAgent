import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests

from agent.tools.base_tool import BaseTool, ToolResult
from agent.tools.utils.truncate import format_size
from common.log import logger
from common.proxy import normalize_proxy_url, proxy_dict
from common.utils import expand_path
from common.video_parse_utils import (
    normalize_yt_dlp_proxy_sites,
    resolve_yt_dlp_command,
    should_proxy_yt_dlp_url,
)
from config import conf


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com"
DEFAULT_PROMPT = (
    "请用中文总结这个视频，返回 JSON。字段包括：summary、main_points、timeline、"
    "spoken_content、visual_content、on_screen_text、uncertainties。不要输出 Markdown。"
)
DEFAULT_DOWNLOAD_TIMEOUT = 600
DEFAULT_FFMPEG_TIMEOUT = 300
DEFAULT_GEMINI_TIMEOUT = 600
DEFAULT_PROCESSING_TIMEOUT = 300
DEFAULT_SPLIT_DURATION_THRESHOLD = 0
DEFAULT_MAX_SEGMENTS = 20
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024

SUPPORTED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
}

VIDEO_FILE_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".avi",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".3gp",
}

AUDIO_FILE_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".aac",
    ".opus",
    ".ogg",
    ".wav",
    ".flac",
}

MIME_ALIASES = {
    "video/x-msvideo": "video/avi",
    "video/vnd.avi": "video/avi",
    "video/x-ms-wmv": "video/wmv",
}

_YOUTUBE_HOSTS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "www.music.youtube.com",
})
_YOUTUBE_SHORT_HOSTS = frozenset({"youtu.be", "www.youtu.be"})
_YOUTUBE_VIDEO_ID_RE = re.compile(r"^[\w-]{11}$")


class VideoParseError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class VideoParseModelFallbackError(VideoParseError):
    def __init__(
        self,
        message: str,
        model_errors: List[Dict[str, Any]],
        fallbackable: bool = False,
        status_code: Optional[int] = None,
    ):
        super().__init__(message, status_code=status_code)
        self.model_errors = model_errors
        self.fallbackable = fallbackable


class VideoParseTool(BaseTool):
    """Analyze video files or video links using Gemini Files API."""

    name: str = "video_parse"
    description: str = (
        "Analyze, summarize, or extract information from a video link or uploaded/local video file. "
        "Use this whenever the user sends a video URL or video file and asks to understand, summarize, "
        "parse, describe, transcribe, or extract timeline/content from the video. "
        "For YouTube URLs with Gemini, the tool first tries direct YouTube URL analysis via Gemini "
        "generateContent (no local download). If that fails and fallback is enabled, it downloads "
        "with yt-dlp, merges split audio/video with ffmpeg stream copy, uploads the final video to "
        "Gemini Files API, and returns JSON/text analysis. Non-YouTube URLs always use the download path."
    )

    params: dict = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "HTTP/HTTPS video page URL to download and analyze.",
            },
            "file_path": {
                "type": "string",
                "description": "Local path to an uploaded or existing video file to analyze.",
            },
            "source": {
                "type": "string",
                "description": "Optional fallback source; can be either a video URL or local file path.",
            },
            "prompt": {
                "type": "string",
                "description": "Optional custom prompt for Gemini. Defaults to Chinese JSON video summary fields.",
            },
            "model": {
                "type": "string",
                "description": "Optional primary Gemini model name. Web video_parse config takes precedence when set.",
            },
            "models": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional ordered Gemini model fallback list. The tool tries each model until one succeeds. "
                    "Web video_parse config takes precedence when set."
                ),
            },
            "keep_temp": {
                "type": "boolean",
                "description": "Keep temporary local files for debugging. Defaults to tools.video_parse.keep_temp.",
            },
        },
        "required": [],
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.cwd = self.config.get("cwd", os.getcwd())

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        args = args or {}
        runtime = self._runtime_config(args)
        work_dir = None
        local_source_path = None
        remote_file_names: List[str] = []
        success = False
        youtube_direct_error: Optional[str] = None
        youtube_direct_model_errors: List[Dict[str, Any]] = []
        youtube_direct_attempted = False

        try:
            url, file_path = self._resolve_input(args)
            self._validate_api_key(runtime)

            if url and self._should_use_youtube_direct(url, runtime):
                youtube_direct_attempted = True
                try:
                    result = self._analyze_youtube_url_direct_with_model_fallback(url, runtime, runtime["prompt"])
                    meta = result.setdefault("_meta", {})
                    meta.update({
                        "source_url": url,
                        "youtube_direct_attempted": True,
                        "fallback_used": False,
                    })
                    meta.setdefault("model", runtime["model"])
                    success = True
                    return ToolResult.success(result)
                except Exception as e:
                    youtube_direct_error = str(e)
                    if isinstance(e, VideoParseModelFallbackError):
                        youtube_direct_model_errors = e.model_errors
                    if not self._should_youtube_direct_fallback(e, runtime):
                        logger.warning(
                            f"[VideoParse] YouTube direct analysis failed (non-recoverable): {e}"
                        )
                        diagnostics = {
                            "_meta": {
                                "mode": "youtube_url_direct",
                                "model": runtime["model"],
                                "models": runtime.get("models") or [runtime["model"]],
                                "source_url": url,
                                "youtube_direct_attempted": True,
                                "fallback_used": False,
                                "youtube_direct_error": youtube_direct_error,
                            }
                        }
                        if isinstance(e, VideoParseModelFallbackError):
                            diagnostics["_meta"]["model_errors"] = e.model_errors
                        return ToolResult.fail(f"Error: {youtube_direct_error}", ext_data=diagnostics)
                    logger.warning(
                        f"[VideoParse] YouTube direct analysis failed, falling back to download: {e}"
                    )

            if url:
                self._validate_commands(url_required=True)
            else:
                self._validate_commands(url_required=False)
            work_dir = self._make_work_dir(runtime)

            if url:
                self.report_progress("正在下载视频...")
                yt_dlp_proxy = runtime["proxy"] if should_proxy_yt_dlp_url(url, runtime.get("yt_dlp_proxy_sites")) else ""
                media_paths = self._download_video(
                    url,
                    work_dir,
                    runtime["download_timeout"],
                    runtime["cookie_file"],
                    yt_dlp_proxy,
                )
            else:
                local_source_path = self._resolve_local_path(file_path)
                media_paths = [local_source_path]

            self.report_progress("正在检查音视频流...")
            final_path, merge_performed = self._prepare_final_video(
                media_paths,
                work_dir,
                runtime["ffmpeg_timeout"],
            )

            if os.path.getsize(final_path) <= 0:
                raise VideoParseError("Video file is empty")
            mime_type = self._detect_mime_type(final_path)
            if mime_type not in SUPPORTED_VIDEO_MIME_TYPES:
                supported = ", ".join(sorted(SUPPORTED_VIDEO_MIME_TYPES))
                raise VideoParseError(
                    f"Unsupported video MIME type: {mime_type}. Supported: {supported}"
                )
            final_path = self._ensure_ascii_upload_path(final_path, work_dir)

            result = self._analyze_video_or_segments_with_model_fallback(
                final_path=final_path,
                mime_type=mime_type,
                work_dir=work_dir,
                runtime=runtime,
                remote_file_names=remote_file_names,
            )
            meta = result.setdefault("_meta", {})
            meta.setdefault("model", runtime["model"])
            meta.update({
                "mime_type": mime_type,
                "merge_performed": merge_performed,
                "temp_files_deleted": not runtime["keep_temp"],
                "source_file_deleted": bool(local_source_path and runtime["delete_source_on_success"]),
                "remote_file_deleted": bool(remote_file_names and runtime["delete_remote_file"]),
            })
            if youtube_direct_attempted:
                meta["youtube_direct_attempted"] = True
                meta["fallback_used"] = bool(youtube_direct_error)
                if url:
                    meta["source_url"] = url
                if youtube_direct_error:
                    meta["youtube_direct_error"] = youtube_direct_error
                if youtube_direct_model_errors:
                    meta["youtube_direct_model_errors"] = youtube_direct_model_errors
            if runtime["keep_temp"]:
                meta["temp_dir"] = work_dir
            success = True
            return ToolResult.success(result)

        except VideoParseModelFallbackError as e:
            logger.warning(f"[VideoParse] {e}")
            diagnostics = {
                "_meta": {
                    "model": runtime["model"],
                    "models": runtime.get("models") or [runtime["model"]],
                    "model_errors": e.model_errors,
                }
            }
            if youtube_direct_attempted:
                diagnostics["_meta"]["youtube_direct_attempted"] = True
                diagnostics["_meta"]["fallback_used"] = bool(youtube_direct_error)
                if youtube_direct_error:
                    diagnostics["_meta"]["youtube_direct_error"] = youtube_direct_error
                if youtube_direct_model_errors:
                    diagnostics["_meta"]["youtube_direct_model_errors"] = youtube_direct_model_errors
            return ToolResult.fail(f"Error: {e}", ext_data=diagnostics)
        except VideoParseError as e:
            logger.warning(f"[VideoParse] {e}")
            return ToolResult.fail(f"Error: {e}")
        except requests.Timeout:
            logger.warning("[VideoParse] Gemini request timed out")
            return ToolResult.fail(f"Error: Gemini request timed out after {runtime['gemini_timeout']}s")
        except requests.ConnectionError as e:
            logger.warning(f"[VideoParse] Gemini connection failed: {e}")
            return ToolResult.fail("Error: Failed to connect to Gemini API")
        except Exception as e:
            logger.error(f"[VideoParse] Unexpected error: {e}", exc_info=True)
            return ToolResult.fail(f"Error: Video parsing failed: {e}")
        finally:
            if success:
                self._cleanup_after_success(
                    work_dir=work_dir,
                    local_source_path=local_source_path,
                    file_names=remote_file_names,
                    runtime=runtime,
                )
            else:
                if runtime.get("delete_remote_file"):
                    self._delete_remote_files(remote_file_names, runtime)
                if work_dir and not runtime.get("keep_temp"):
                    self._remove_dir(work_dir)

    def _runtime_config(self, args: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self._latest_tool_config()
        api_base = (
            cfg.get("api_base")
            or os.environ.get("GEMINI_API_BASE")
            or conf().get("gemini_api_base")
            or DEFAULT_API_BASE
        )
        upload_api_base = cfg.get("upload_api_base") or api_base
        models = self._configured_models(cfg, args)
        keep_temp = args.get("keep_temp")
        if keep_temp is None:
            keep_temp = self._bool_value(cfg.get("keep_temp", False))

        return {
            "api_key": cfg.get("api_key") or os.environ.get("GEMINI_API_KEY") or conf().get("gemini_api_key", ""),
            "api_base": str(api_base).rstrip("/"),
            "upload_api_base": str(upload_api_base).rstrip("/"),
            "model": models[0],
            "models": models,
            "prompt": args.get("prompt") or cfg.get("prompt") or DEFAULT_PROMPT,
            "download_timeout": self._int_value(cfg.get("download_timeout"), DEFAULT_DOWNLOAD_TIMEOUT),
            "ffmpeg_timeout": self._int_value(cfg.get("ffmpeg_timeout"), DEFAULT_FFMPEG_TIMEOUT),
            "gemini_timeout": self._int_value(cfg.get("gemini_timeout"), DEFAULT_GEMINI_TIMEOUT),
            "processing_timeout": self._int_value(cfg.get("processing_timeout"), DEFAULT_PROCESSING_TIMEOUT),
            "split_duration_threshold_sec": max(
                0,
                self._int_value(cfg.get("split_duration_threshold_sec"), DEFAULT_SPLIT_DURATION_THRESHOLD),
            ),
            "max_segments": max(0, self._int_value(cfg.get("max_segments"), DEFAULT_MAX_SEGMENTS)),
            "max_video_bytes": self._int_value(cfg.get("max_video_bytes"), MAX_VIDEO_BYTES),
            "temp_dir": cfg.get("temp_dir") or "",
            "keep_temp": bool(keep_temp),
            "delete_source_on_success": self._bool_value(cfg.get("delete_source_on_success", True)),
            "delete_remote_file": self._bool_value(cfg.get("delete_remote_file", True)),
            "prefer_json": self._bool_value(cfg.get("prefer_json", True)),
            "cookie_file": str(cfg.get("cookie_file") or "").strip(),
            "proxy": str(cfg.get("proxy") or "").strip(),
            "yt_dlp_proxy_sites": normalize_yt_dlp_proxy_sites(cfg.get("yt_dlp_proxy_sites")),
            "youtube_direct_enabled": self._bool_value(cfg.get("youtube_direct_enabled", True)),
            "youtube_direct_fallback": self._bool_value(cfg.get("youtube_direct_fallback", True)),
        }

    def _latest_tool_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if isinstance(self.config, dict):
            cfg.update(self.config)

        try:
            tools_cfg = conf().get("tools", {})
            if isinstance(tools_cfg, dict) and isinstance(tools_cfg.get("video_parse"), dict):
                cfg.update(tools_cfg["video_parse"])
        except Exception as e:
            logger.debug(f"[VideoParse] Failed to refresh runtime config from conf(): {e}")

        self.config = cfg
        return cfg

    def _configured_models(self, cfg: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
        candidates = (
            cfg.get("models"),
            cfg.get("model"),
            args.get("models"),
            args.get("model"),
            os.environ.get("GEMINI_VIDEO_MODELS"),
            os.environ.get("GEMINI_VIDEO_MODEL"),
            DEFAULT_MODEL,
        )
        for value in candidates:
            models = self._normalize_model_list(value)
            if models:
                return models
        return [DEFAULT_MODEL]

    def _normalize_model_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = re.split(r"[\n\r,，]+", value)
        elif isinstance(value, (list, tuple)):
            raw_items = value
        else:
            raw_items = [value]

        models = []
        seen = set()
        for item in raw_items:
            model = str(item or "").strip()
            if not model or model in seen:
                continue
            seen.add(model)
            models.append(model)
        return models

    def _runtime_for_model(self, runtime: Dict[str, Any], model: str) -> Dict[str, Any]:
        model_runtime = dict(runtime)
        model_runtime["model"] = model
        return model_runtime

    def _model_error_info(self, model: str, error: Exception) -> Dict[str, Any]:
        info = {
            "model": model,
            "error": str(error),
        }
        status_code = getattr(error, "status_code", None)
        if status_code:
            info["status_code"] = status_code
        return info

    def _model_fallback_error(
        self,
        action: str,
        model_errors: List[Dict[str, Any]],
        fallbackable: bool = False,
    ) -> VideoParseModelFallbackError:
        details = "; ".join(
            "{}: {}".format(item.get("model") or "unknown", self._tail(item.get("error") or "", 500))
            for item in model_errors
        )
        status_code = None
        for item in reversed(model_errors):
            if item.get("status_code"):
                status_code = item["status_code"]
                break
        return VideoParseModelFallbackError(
            f"All configured Gemini models failed during {action}: {details}",
            model_errors=model_errors,
            fallbackable=fallbackable,
            status_code=status_code,
        )

    def _attach_model_fallback_meta(
        self,
        meta: Dict[str, Any],
        models: List[str],
        success_index: int,
        errors: List[Dict[str, Any]],
    ) -> None:
        meta["model"] = models[success_index]
        meta["models"] = models
        meta["models_attempted"] = models[: success_index + 1]
        meta["model_fallback_used"] = success_index > 0
        if errors:
            meta["model_errors"] = errors

    def _resolve_input(self, args: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        url = (args.get("url") or "").strip()
        file_path = (args.get("file_path") or "").strip()
        source = (args.get("source") or "").strip()

        if source and not url and not file_path:
            parsed = urlparse(source)
            if parsed.scheme in ("http", "https"):
                url = source
            else:
                file_path = source

        if url and file_path:
            raise VideoParseError("Provide either url or file_path, not both")
        if not url and not file_path:
            raise VideoParseError("Either url, file_path, or source is required")

        if url:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise VideoParseError("Video URL must start with http:// or https://")
        return url or None, file_path or None

    def _validate_api_key(self, runtime: Dict[str, Any]) -> None:
        if not runtime["api_key"]:
            raise VideoParseError(
                "Missing GEMINI_API_KEY. Configure it with env_config(action='set', key='GEMINI_API_KEY', value='...') "
                "or config.json gemini_api_key."
            )
        if runtime.get("proxy"):
            normalize_proxy_url(runtime["proxy"])

    def _validate_commands(self, url_required: bool) -> None:
        yt_dlp_cmd, _ = resolve_yt_dlp_command()
        if url_required and not yt_dlp_cmd:
            raise VideoParseError(
                "Missing dependency: yt-dlp. Install it with: python3 -m pip install -U yt-dlp"
            )
        missing = [cmd for cmd in ("ffmpeg", "ffprobe") if not shutil.which(cmd)]
        if missing:
            raise VideoParseError(
                "Missing system dependency: {}. Please install ffmpeg and ensure ffmpeg/ffprobe are in PATH.".format(
                    ", ".join(missing)
                )
            )

    def _is_youtube_url(self, url: str) -> bool:
        parsed = urlparse((url or "").strip())
        host = (parsed.hostname or "").lower()
        if not host:
            return False
        if host == "googlevideo.com" or host.endswith(".googlevideo.com"):
            return False

        path = parsed.path or ""

        if host in _YOUTUBE_SHORT_HOSTS:
            video_id = path.strip("/").split("/")[0]
            return bool(video_id and _YOUTUBE_VIDEO_ID_RE.match(video_id))

        if host not in _YOUTUBE_HOSTS:
            return False

        if path.startswith("/playlist"):
            return False

        if path.startswith("/watch"):
            return self._youtube_watch_has_video_id(parsed)

        for prefix in ("/shorts/", "/embed/", "/live/", "/v/"):
            if path.startswith(prefix):
                return bool(self._youtube_path_video_id(path, prefix))

        return False

    def _youtube_watch_has_video_id(self, parsed) -> bool:
        qs = parse_qs(parsed.query or "")
        video_ids = qs.get("v") or []
        video_id = str(video_ids[0]).strip() if video_ids else ""
        return bool(video_id and _YOUTUBE_VIDEO_ID_RE.match(video_id))

    def _youtube_path_video_id(self, path: str, prefix: str) -> Optional[str]:
        remainder = path[len(prefix):]
        video_id = remainder.split("/")[0].split("?")[0]
        if video_id and _YOUTUBE_VIDEO_ID_RE.match(video_id):
            return video_id
        return None

    def _should_use_youtube_direct(self, url: str, runtime: Dict[str, Any]) -> bool:
        if not runtime.get("youtube_direct_enabled", True):
            return False
        if not self._is_youtube_url(url):
            return False
        if not runtime.get("api_key"):
            return False
        return True

    def _should_youtube_direct_fallback(self, error: Exception, runtime: Dict[str, Any]) -> bool:
        if not runtime.get("youtube_direct_fallback", True):
            return False
        if isinstance(error, VideoParseModelFallbackError):
            return error.fallbackable
        # Default to fallback; only refuse for clearly non-recoverable errors so that
        # transient/network/parse failures still go through the download path.
        return not self._is_youtube_direct_non_fallbackable(error)

    def _is_youtube_direct_non_fallbackable(self, error: Exception) -> bool:
        # Genuine auth/quota/model errors will not be fixed by downloading first.
        # Note: HTTP 403/PERMISSION_DENIED is treated as recoverable because
        # age/region-gated videos may still succeed via local yt-dlp cookies/proxy.
        status_code = getattr(error, "status_code", None)
        if status_code in (401, 429):
            return True
        message = str(error).lower()
        non_fallback_patterns = (
            "api key not valid",
            "api_key_invalid",
            "invalid api key",
            "quota",
            "resource exhausted",
            "unauthenticated",
            "authentication credentials",
            "request had invalid authentication",
        )
        if any(pattern in message for pattern in non_fallback_patterns):
            return True
        if "model" in message and ("not found" in message or "does not exist" in message):
            return True
        return False

    def _analyze_youtube_url_direct_with_model_fallback(
        self,
        url: str,
        runtime: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        models = runtime.get("models") or [runtime["model"]]
        errors: List[Dict[str, Any]] = []
        fallbackable = False
        for index, model in enumerate(models):
            model_runtime = self._runtime_for_model(runtime, model)
            self.report_progress(f"正在通过 YouTube 直链调用 Gemini 解析（模型 {model}）...")
            try:
                result = self._analyze_youtube_url_direct(url, model_runtime, prompt=prompt)
                self._attach_model_fallback_meta(result.setdefault("_meta", {}), models, index, errors)
                return result
            except Exception as e:
                errors.append(self._model_error_info(model, e))
                if self._should_youtube_direct_fallback(e, model_runtime):
                    fallbackable = True
                logger.warning(f"[VideoParse] YouTube direct analysis failed with model {model}: {e}")
        raise self._model_fallback_error(
            "YouTube direct analysis",
            errors,
            fallbackable=fallbackable,
        )

    def _analyze_youtube_url_direct(
        self,
        url: str,
        runtime: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        response_data = self._generate_content_youtube_url(url, runtime, prompt=prompt)
        raw_text = self._extract_candidate_text(response_data)
        parsed = self._parse_json_text(raw_text)
        result = parsed if isinstance(parsed, dict) else {}
        result.setdefault("raw_text", raw_text)
        result["_meta"] = {
            "mode": "youtube_url_direct",
            "usage": response_data.get("usageMetadata") or {},
        }
        return result

    def _generate_content_youtube_url(
        self,
        youtube_url: str,
        runtime: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = "{}/v1beta/models/{}:generateContent".format(runtime["api_base"], runtime["model"])
        payload = {
            "contents": [
                {
                    "parts": [
                        {"file_data": {"file_uri": youtube_url}},
                        {"text": prompt if prompt is not None else runtime["prompt"]},
                    ]
                }
            ]
        }
        if runtime.get("prefer_json"):
            payload["generationConfig"] = {"response_mime_type": "application/json"}

        response = requests.post(
            url,
            headers={
                "x-goog-api-key": runtime["api_key"],
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=runtime["gemini_timeout"],
            proxies=proxy_dict(runtime.get("proxy") or ""),
        )
        self._raise_for_gemini_error(response, "generate content from YouTube URL")
        return response.json()

    def _make_work_dir(self, runtime: Dict[str, Any]) -> str:
        base = runtime.get("temp_dir") or os.path.join(self.cwd, "tmp", "video_parse")
        base = expand_path(base)
        work_dir = os.path.join(base, uuid.uuid4().hex)
        os.makedirs(work_dir, exist_ok=True)
        return work_dir

    def _download_video(
        self,
        url: str,
        work_dir: str,
        timeout: int,
        cookie_file: str = "",
        proxy: str = "",
    ) -> List[str]:
        yt_dlp_cmd, yt_dlp_display = resolve_yt_dlp_command()
        if not yt_dlp_cmd:
            raise VideoParseError("Missing dependency: yt-dlp. Install it with: python3 -m pip install -U yt-dlp")
        cmd = yt_dlp_cmd + [
            "--no-playlist",
            "--no-progress",
            "-f",
            "bv*[vcodec^=avc1]+ba[ext=m4a]/bv*[vcodec^=avc1]+ba/b[ext=mp4]/b",
            "-o",
            os.path.join(work_dir, "%(id)s.%(ext)s"),
        ]
        cookie_file = str(cookie_file or "").strip()
        if cookie_file:
            cookie_file = expand_path(cookie_file)
            if not os.path.isfile(cookie_file):
                raise VideoParseError(f"Configured yt-dlp cookie file not found: {cookie_file}")
            cmd.extend(["--cookies", cookie_file])
        proxy = str(proxy or "").strip()
        if proxy:
            cmd.extend(["--proxy", normalize_proxy_url(proxy)])
        cmd.append(url)
        proxy_note = " with proxy" if proxy else ""
        logger.info(f"[VideoParse] Downloading video{proxy_note}: {url} -> {work_dir} using {yt_dlp_display}")
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise VideoParseError(f"yt-dlp timed out after {timeout}s")

        output = completed.stdout or ""
        if completed.returncode != 0:
            raise VideoParseError(
                "yt-dlp failed with exit code {}:\n{}".format(
                    completed.returncode,
                    self._tail(output),
                )
            )

        media_paths = self._scan_media_files(work_dir)
        if not media_paths:
            raise VideoParseError(
                "yt-dlp completed but no usable media file was found.\n{}".format(
                    self._tail(output)
                )
            )
        return media_paths

    def _resolve_local_path(self, path: str) -> str:
        expanded = expand_path(path)
        if not os.path.isabs(expanded):
            expanded = os.path.abspath(os.path.join(self.cwd, expanded))
        if not os.path.isfile(expanded):
            raise VideoParseError(f"Video file not found: {expanded}")
        return expanded

    def _scan_media_files(self, root: str) -> List[str]:
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if not os.path.isfile(path):
                    continue
                if filename.endswith((".download", ".part", ".tmp")):
                    continue
                if os.path.getsize(path) <= 0:
                    continue
                if self._probe_media(path) or self._guess_media_streams(path):
                    paths.append(path)
        paths.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return paths

    def _prepare_final_video(
        self,
        media_paths: List[str],
        work_dir: str,
        timeout: int,
    ) -> Tuple[str, bool]:
        candidates = []
        for path in media_paths:
            probe = self._probe_media(path) or self._guess_media_streams(path)
            if probe:
                candidates.append({
                    "path": path,
                    "size": os.path.getsize(path),
                    "has_video": probe["has_video"],
                    "has_audio": probe["has_audio"],
                })

        if not candidates:
            raise VideoParseError("No valid video/audio stream found in input")

        combined = [c for c in candidates if c["has_video"] and c["has_audio"]]
        if combined:
            combined.sort(key=lambda c: c["size"], reverse=True)
            return combined[0]["path"], False

        video_candidates = [c for c in candidates if c["has_video"]]
        audio_candidates = [c for c in candidates if c["has_audio"] and not c["has_video"]]
        if not video_candidates:
            raise VideoParseError("No video stream found in input")

        video_candidates.sort(key=lambda c: c["size"], reverse=True)
        if not audio_candidates:
            return video_candidates[0]["path"], False

        audio_candidates.sort(key=lambda c: c["size"], reverse=True)
        final_path = os.path.join(work_dir, "final.mp4")
        self.report_progress("检测到音视频分离，正在无损合并...")
        self._merge_streams(
            video_candidates[0]["path"],
            audio_candidates[0]["path"],
            final_path,
            timeout,
        )
        return final_path, True

    def _probe_media(self, path: str) -> Optional[Dict[str, bool]]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            path,
        ]
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            stdout = completed.stdout.decode("utf-8", errors="replace").strip()
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            if completed.returncode != 0:
                logger.warning(
                    f"[VideoParse] ffprobe exited with {completed.returncode} for {path}: "
                    f"{self._tail(stderr or stdout, 500)}"
                )
                return None
            data = json.loads(stdout or "{}")
            if stderr:
                logger.debug(f"[VideoParse] ffprobe diagnostics for {path}: {self._tail(stderr, 500)}")
        except Exception as e:
            logger.warning(f"[VideoParse] ffprobe parse failed for {path}: {e}")
            return None

        streams = data.get("streams") or []
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        if not has_video and not has_audio:
            return None
        return {"has_video": has_video, "has_audio": has_audio}

    def _guess_media_streams(self, path: str) -> Optional[Dict[str, bool]]:
        ext = os.path.splitext(path)[1].lower()
        if ext in AUDIO_FILE_EXTENSIONS:
            logger.warning(f"[VideoParse] ffprobe failed for audio-like file, using extension fallback: {path}")
            return {"has_video": False, "has_audio": True}
        if ext in VIDEO_FILE_EXTENSIONS:
            logger.warning(f"[VideoParse] ffprobe failed for video-like file, using extension fallback: {path}")
            return {"has_video": True, "has_audio": True}
        return None

    def _merge_streams(self, video_path: str, audio_path: str, final_path: str, timeout: int) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            "-shortest",
            final_path,
        ]
        logger.info(f"[VideoParse] Merging split streams into {final_path}")
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise VideoParseError(f"ffmpeg merge timed out after {timeout}s")

        if completed.returncode != 0 or not os.path.isfile(final_path):
            raise VideoParseError(
                "ffmpeg stream-copy merge failed:\n{}".format(self._tail(completed.stdout or ""))
            )

    def _validate_video_size(self, path: str, max_bytes: int) -> None:
        size = os.path.getsize(path)
        if size <= 0:
            raise VideoParseError("Video file is empty")
        if size > max_bytes:
            raise VideoParseError(
                "Video file too large: {} > {}".format(format_size(size), format_size(max_bytes))
            )

    def _detect_mime_type(self, path: str) -> str:
        guessed, _ = mimetypes.guess_type(path)
        mime_type = guessed or ""
        if not mime_type and shutil.which("file"):
            try:
                out = subprocess.check_output(
                    ["file", "-b", "--mime-type", path],
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                mime_type = out.decode("utf-8", errors="replace").strip()
            except Exception:
                mime_type = ""

        ext = os.path.splitext(path)[1].lower()
        if ext == ".mp4" and not mime_type:
            mime_type = "video/mp4"
        elif ext == ".mov":
            mime_type = "video/quicktime"
        elif ext == ".avi":
            mime_type = "video/avi"
        elif ext == ".wmv":
            mime_type = "video/wmv"
        elif ext in (".mpg", ".mpeg"):
            mime_type = "video/mpeg"

        return MIME_ALIASES.get(mime_type, mime_type or "application/octet-stream")

    def _ensure_ascii_upload_path(self, path: str, work_dir: str) -> str:
        basename = os.path.basename(path)
        try:
            basename.encode("ascii")
            return path
        except UnicodeEncodeError:
            pass

        ext = os.path.splitext(path)[1].lower() or ".mp4"
        ascii_path = os.path.join(work_dir, f"video_{uuid.uuid4().hex}{ext}")
        if os.path.abspath(path) == os.path.abspath(ascii_path):
            return path
        shutil.copy2(path, ascii_path)
        logger.info(f"[VideoParse] Copied upload file to ASCII path: {ascii_path}")
        return ascii_path

    def _analyze_video_or_segments(
        self,
        final_path: str,
        mime_type: str,
        work_dir: str,
        runtime: Dict[str, Any],
        remote_file_names: List[str],
    ) -> Dict[str, Any]:
        duration_sec = self._probe_duration(final_path)
        split_threshold = runtime.get("split_duration_threshold_sec", 0)
        if not split_threshold or not duration_sec or duration_sec <= split_threshold:
            if split_threshold and not duration_sec:
                logger.warning("[VideoParse] Unable to detect video duration; falling back to single-file analysis")
            analysis = self._analyze_single_video(
                final_path,
                mime_type,
                runtime,
                remote_file_names,
                progress_prefix="视频",
            )
            result = analysis["parsed"] if isinstance(analysis["parsed"], dict) else {}
            result.setdefault("raw_text", analysis["raw_text"])
            result["_meta"] = {
                "mode": "single",
                "duration_sec": duration_sec,
                "split_duration_threshold_sec": split_threshold,
                "usage": analysis["usage"],
            }
            if not runtime["delete_remote_file"] and analysis.get("file_uri"):
                result["_meta"]["file_uri"] = analysis["file_uri"]
            return result

        max_segments = runtime.get("max_segments", DEFAULT_MAX_SEGMENTS)
        estimated_segments = int(math.ceil(duration_sec / split_threshold))
        if max_segments and estimated_segments > max_segments:
            raise VideoParseError(
                "Video duration {}s exceeds split limit: estimated {} segments > max_segments {}. "
                "Increase split_duration_threshold_sec or max_segments.".format(
                    round(duration_sec, 2),
                    estimated_segments,
                    max_segments,
                )
            )

        self.report_progress("视频超过阈值，正在无损切分...")
        segments = self._split_video(
            final_path,
            work_dir,
            split_threshold,
            runtime["ffmpeg_timeout"],
            runtime["max_video_bytes"],
        )
        if max_segments and len(segments) > max_segments:
            raise VideoParseError(
                f"Video split produced {len(segments)} segments, which exceeds max_segments {max_segments}"
            )

        segment_results = []
        usage_total = {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
        }
        for segment in segments:
            index = segment["index"]
            prompt = self._segment_prompt(
                runtime["prompt"],
                index,
                len(segments),
                segment.get("start_sec"),
                segment.get("end_sec"),
            )
            analysis = self._analyze_single_video(
                segment["path"],
                segment["mime_type"],
                runtime,
                remote_file_names,
                prompt=prompt,
                progress_prefix=f"第 {index}/{len(segments)} 段视频",
            )
            usage = analysis["usage"]
            for key in usage_total:
                usage_total[key] += self._int_value(usage.get(key), 0)
            segment_result = {
                "index": index,
                "start_sec": segment.get("start_sec"),
                "end_sec": segment.get("end_sec"),
                "duration_sec": segment.get("duration_sec"),
                "mime_type": segment["mime_type"],
                "raw_text": analysis["raw_text"],
                "parsed": analysis["parsed"] if analysis["parsed"] is not None else {},
                "usage": usage,
            }
            if runtime["keep_temp"]:
                segment_result["path"] = segment["path"]
            if not runtime["delete_remote_file"] and analysis.get("file_uri"):
                segment_result["file_uri"] = analysis["file_uri"]
            segment_results.append(segment_result)

        return {
            "mode": "segmented",
            "segments": segment_results,
            "_meta": {
                "mode": "segmented",
                "duration_sec": duration_sec,
                "split_duration_threshold_sec": split_threshold,
                "segment_count": len(segment_results),
                "max_segments": max_segments,
                "usage_total": usage_total,
            },
        }

    def _analyze_video_or_segments_with_model_fallback(
        self,
        final_path: str,
        mime_type: str,
        work_dir: str,
        runtime: Dict[str, Any],
        remote_file_names: List[str],
    ) -> Dict[str, Any]:
        models = runtime.get("models") or [runtime["model"]]
        errors: List[Dict[str, Any]] = []
        for index, model in enumerate(models):
            model_runtime = self._runtime_for_model(runtime, model)
            if index:
                self.report_progress(f"前一个模型解析失败，正在切换到 {model} 重试...")
            try:
                result = self._analyze_video_or_segments(
                    final_path=final_path,
                    mime_type=mime_type,
                    work_dir=work_dir,
                    runtime=model_runtime,
                    remote_file_names=remote_file_names,
                )
                self._attach_model_fallback_meta(result.setdefault("_meta", {}), models, index, errors)
                return result
            except Exception as e:
                errors.append(self._model_error_info(model, e))
                logger.warning(f"[VideoParse] Video analysis failed with model {model}: {e}")
        raise self._model_fallback_error("video file analysis", errors)

    def _analyze_single_video(
        self,
        path: str,
        mime_type: str,
        runtime: Dict[str, Any],
        remote_file_names: List[str],
        prompt: Optional[str] = None,
        progress_prefix: str = "视频",
    ) -> Dict[str, Any]:
        self._validate_video_size(path, runtime["max_video_bytes"])
        self.report_progress(f"正在上传{progress_prefix}到 Gemini Files API...")
        file_info = self._upload_file(path, mime_type, runtime)
        file_uri = self._file_field(file_info, "uri")
        file_name = self._file_field(file_info, "name")
        if file_name:
            remote_file_names.append(file_name)
        if not file_uri:
            raise VideoParseError("Gemini Files API did not return file.uri")

        if file_name:
            self.report_progress(f"正在等待 Gemini 完成{progress_prefix}处理...")
            file_info = self._wait_for_file_active(file_name, runtime)
            file_uri = self._file_field(file_info, "uri") or file_uri

        self.report_progress(f"正在调用 Gemini 解析{progress_prefix}...")
        response_data = self._generate_content_when_ready(
            file_name,
            file_uri,
            mime_type,
            runtime,
            prompt=prompt,
        )
        raw_text = self._extract_candidate_text(response_data)
        parsed = self._parse_json_text(raw_text)
        return {
            "raw_text": raw_text,
            "parsed": parsed,
            "usage": response_data.get("usageMetadata") or {},
            "file_name": file_name,
            "file_uri": file_uri,
        }

    def _segment_prompt(
        self,
        base_prompt: str,
        index: int,
        total: int,
        start_sec: Optional[float],
        end_sec: Optional[float],
    ) -> str:
        start = "未知" if start_sec is None else f"{start_sec:.2f}s"
        end = "未知" if end_sec is None else f"{end_sec:.2f}s"
        return (
            f"这是原始视频切分后的第 {index}/{total} 段，时间范围约 {start} - {end}。\n"
            "请只解析这一段内容，并在时间线和总结中保留该分段上下文，便于后续合并成完整视频理解。\n"
            f"用户原始要求：{base_prompt}"
        )

    def _probe_duration(self, path: str) -> Optional[float]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ]
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            stdout = completed.stdout.decode("utf-8", errors="replace").strip()
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            if completed.returncode != 0:
                logger.warning(
                    f"[VideoParse] ffprobe duration exited with {completed.returncode} for {path}: "
                    f"{self._tail(stderr or stdout, 500)}"
                )
                return None
            data = json.loads(stdout or "{}")
            candidates = []
            format_duration = self._float_value(self._get_nested(data, "format", "duration"))
            if format_duration:
                candidates.append(format_duration)
            for stream in data.get("streams") or []:
                stream_duration = self._float_value(stream.get("duration"))
                if stream_duration:
                    candidates.append(stream_duration)
            if not candidates:
                return None
            return max(candidates)
        except Exception as e:
            logger.warning(f"[VideoParse] ffprobe duration parse failed for {path}: {e}")
            return None

    def _split_video(
        self,
        path: str,
        work_dir: str,
        segment_seconds: int,
        timeout: int,
        max_bytes: int,
    ) -> List[Dict[str, Any]]:
        segment_dir = os.path.join(work_dir, "segments")
        os.makedirs(segment_dir, exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_FILE_EXTENSIONS:
            ext = ".mp4"
        output_pattern = os.path.join(segment_dir, f"segment_%04d{ext}")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            output_pattern,
        ]
        logger.info(f"[VideoParse] Splitting video into {segment_seconds}s segments: {path} -> {segment_dir}")
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise VideoParseError(f"ffmpeg split timed out after {timeout}s")

        if completed.returncode != 0:
            raise VideoParseError(
                "ffmpeg stream-copy split failed:\n{}".format(self._tail(completed.stdout or ""))
            )

        segment_paths = []
        for filename in sorted(os.listdir(segment_dir)):
            segment_path = os.path.join(segment_dir, filename)
            if os.path.isfile(segment_path) and os.path.getsize(segment_path) > 0:
                segment_paths.append(segment_path)
        if not segment_paths:
            raise VideoParseError("ffmpeg split completed but no segment file was produced")

        segments = []
        current_start = 0.0
        for index, segment_path in enumerate(segment_paths, start=1):
            probe = self._probe_media(segment_path) or self._guess_media_streams(segment_path)
            if not probe or not probe["has_video"]:
                raise VideoParseError(f"Split segment has no video stream: {segment_path}")
            self._validate_video_size(segment_path, max_bytes)
            segment_mime = self._detect_mime_type(segment_path)
            if segment_mime not in SUPPORTED_VIDEO_MIME_TYPES:
                raise VideoParseError(f"Unsupported segment MIME type: {segment_mime}")
            segment_duration = self._probe_duration(segment_path)
            fallback_start = float((index - 1) * segment_seconds)
            start_sec = current_start if current_start > 0 or index == 1 else fallback_start
            duration = segment_duration if segment_duration else float(segment_seconds)
            end_sec = start_sec + duration
            current_start = end_sec
            segments.append({
                "index": index,
                "path": segment_path,
                "mime_type": segment_mime,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "duration_sec": round(duration, 2),
            })
        return segments

    def _upload_file(self, path: str, mime_type: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
        num_bytes = os.path.getsize(path)
        display_name = os.path.basename(path)
        start_url = "{}/upload/v1beta/files".format(runtime["upload_api_base"])
        headers = {
            "x-goog-api-key": runtime["api_key"],
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        response = requests.post(
            start_url,
            headers=headers,
            json={"file": {"display_name": display_name}},
            timeout=runtime["gemini_timeout"],
            proxies=proxy_dict(runtime.get("proxy") or ""),
        )
        self._raise_for_gemini_error(response, "create upload session")

        upload_url = response.headers.get("X-Goog-Upload-URL") or response.headers.get("x-goog-upload-url")
        if not upload_url:
            raise VideoParseError("Gemini upload session did not return X-Goog-Upload-URL")

        upload_headers = {
            "Content-Length": str(num_bytes),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        with open(path, "rb") as f:
            upload_response = requests.post(
                upload_url,
                headers=upload_headers,
                data=f,
                timeout=runtime["gemini_timeout"],
                proxies=proxy_dict(runtime.get("proxy") or ""),
            )
        self._raise_for_gemini_error(upload_response, "upload video")
        return upload_response.json()

    def _wait_for_file_active(
        self,
        file_name: str,
        runtime: Dict[str, Any],
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        deadline = deadline or time.time() + runtime["processing_timeout"]
        last_info = {}
        last_logged_state = None
        while True:
            info = self._get_file(file_name, runtime)
            last_info = info
            state = str(self._file_field(info, "state") or "").upper()
            display_state = state or "UNKNOWN"
            if display_state != last_logged_state:
                logger.info(f"[VideoParse] Gemini file {file_name} state={display_state}")
                last_logged_state = display_state

            if state == "ACTIVE":
                return info
            if state in ("FAILED", "ERROR"):
                raise VideoParseError(f"Gemini file processing failed for {file_name}")
            if time.time() >= deadline:
                raise VideoParseError(
                    f"Timed out waiting for Gemini file to become ACTIVE; last state={display_state}"
                )
            time.sleep(2)

        return last_info

    def _get_file(self, file_name: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
        url = "{}/v1beta/{}".format(runtime["upload_api_base"], file_name)
        response = requests.get(
            url,
            headers={"x-goog-api-key": runtime["api_key"]},
            timeout=runtime["gemini_timeout"],
            proxies=proxy_dict(runtime.get("proxy") or ""),
        )
        self._raise_for_gemini_error(response, "get uploaded file")
        return response.json()

    def _generate_content_when_ready(
        self,
        file_name: Optional[str],
        file_uri: str,
        mime_type: str,
        runtime: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        deadline = time.time() + runtime["processing_timeout"]
        attempt = 0
        while True:
            try:
                return self._generate_content(file_uri, mime_type, runtime, prompt=prompt)
            except VideoParseError as e:
                if not file_name or not self._is_file_not_active_error(e):
                    raise

                attempt += 1
                if time.time() >= deadline:
                    raise VideoParseError(
                        f"Gemini file {file_name} did not become usable before timeout: {e}"
                    )

                logger.info(
                    f"[VideoParse] Gemini file {file_name} is not usable yet; "
                    f"waiting before retry #{attempt}"
                )
                self.report_progress("Gemini 文件仍在处理中，继续等待...")
                file_info = self._wait_for_file_active(file_name, runtime, deadline=deadline)
                file_uri = self._file_field(file_info, "uri") or file_uri
                time.sleep(min(2 * attempt, 10))

    def _generate_content(
        self,
        file_uri: str,
        mime_type: str,
        runtime: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = "{}/v1beta/models/{}:generateContent".format(runtime["api_base"], runtime["model"])
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "file_data": {
                                "mime_type": mime_type,
                                "file_uri": file_uri,
                            }
                        },
                        {"text": prompt if prompt is not None else runtime["prompt"]},
                    ]
                }
            ]
        }
        if runtime.get("prefer_json"):
            payload["generationConfig"] = {"response_mime_type": "application/json"}

        response = requests.post(
            url,
            headers={
                "x-goog-api-key": runtime["api_key"],
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=runtime["gemini_timeout"],
            proxies=proxy_dict(runtime.get("proxy") or ""),
        )
        self._raise_for_gemini_error(response, "generate content")
        return response.json()

    def _extract_candidate_text(self, data: Dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        parts_text = []
        for candidate in candidates:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                text = part.get("text")
                if text:
                    parts_text.append(text)
        text = "\n".join(parts_text).strip()
        if not text:
            raise VideoParseError("Gemini returned no text content")
        return text

    def _parse_json_text(self, text: str) -> Any:
        cleaned = self._strip_code_fence(text)
        for candidate in self._json_candidates(cleaned):
            try:
                return json.loads(candidate)
            except Exception:
                pass

            try:
                from json_repair import repair_json
                repaired = repair_json(candidate)
                return json.loads(repaired)
            except Exception:
                pass
        return None

    def _json_candidates(self, text: str) -> List[str]:
        candidates = [text.strip()]
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())
        match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())
        return [c for c in candidates if c]

    def _strip_code_fence(self, text: str) -> str:
        stripped = text.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            return fence.group(1).strip()
        return stripped

    def _cleanup_after_success(
        self,
        work_dir: Optional[str],
        local_source_path: Optional[str],
        file_names: List[str],
        runtime: Dict[str, Any],
    ) -> None:
        if runtime.get("delete_remote_file"):
            self._delete_remote_files(file_names, runtime)

        if local_source_path and runtime.get("delete_source_on_success"):
            self._remove_file(local_source_path)

        if work_dir and not runtime.get("keep_temp"):
            self._remove_dir(work_dir)

    def _delete_remote_files(self, file_names: List[str], runtime: Dict[str, Any]) -> None:
        for file_name in list(dict.fromkeys(file_names or [])):
            self._delete_remote_file(file_name, runtime)

    def _delete_remote_file(self, file_name: str, runtime: Dict[str, Any]) -> None:
        try:
            url = "{}/v1beta/{}".format(runtime["upload_api_base"], file_name)
            response = requests.delete(
                url,
                headers={"x-goog-api-key": runtime["api_key"]},
                timeout=min(runtime["gemini_timeout"], 60),
                proxies=proxy_dict(runtime.get("proxy") or ""),
            )
            if response.status_code >= 400:
                logger.warning(
                    f"[VideoParse] Failed to delete Gemini file {file_name}: "
                    f"HTTP {response.status_code} {response.text[:300]}"
                )
        except Exception as e:
            logger.warning(f"[VideoParse] Failed to delete Gemini file {file_name}: {e}")

    def _remove_file(self, path: str) -> None:
        try:
            if os.path.isfile(path):
                os.remove(path)
                logger.info(f"[VideoParse] Removed file: {path}")
        except Exception as e:
            logger.warning(f"[VideoParse] Failed to remove file {path}: {e}")

    def _remove_dir(self, path: str) -> None:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                logger.info(f"[VideoParse] Removed temp dir: {path}")
        except Exception as e:
            logger.warning(f"[VideoParse] Failed to remove temp dir {path}: {e}")

    def _raise_for_gemini_error(self, response: requests.Response, action: str) -> None:
        if response.status_code < 400:
            return
        message = response.text
        try:
            data = response.json()
            message = self._get_nested(data, "error", "message") or message
        except Exception:
            pass
        raise VideoParseError(
            f"Gemini API failed to {action}: HTTP {response.status_code}: {message}",
            status_code=response.status_code,
        )

    def _file_field(self, data: Dict[str, Any], key: str) -> Any:
        value = self._get_nested(data, "file", key)
        if value is not None:
            return value
        if isinstance(data, dict):
            return data.get(key)
        return None

    def _is_file_not_active_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return "not in an active state" in message or "not in a active state" in message

    def _get_nested(self, data: Dict[str, Any], *keys: str) -> Any:
        cur = data
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    def _tail(self, text: str, limit: int = 2000) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[-limit:]

    def _int_value(self, value: Any, default: int) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except Exception:
            return default

    def _float_value(self, value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            parsed = float(value)
            if parsed <= 0:
                return None
            return parsed
        except Exception:
            return None

    def _bool_value(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
