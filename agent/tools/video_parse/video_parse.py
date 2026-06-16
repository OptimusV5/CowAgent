import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from agent.tools.base_tool import BaseTool, ToolResult
from agent.tools.utils.truncate import format_size
from common.log import logger
from common.utils import expand_path
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

MIME_ALIASES = {
    "video/x-msvideo": "video/avi",
    "video/vnd.avi": "video/avi",
    "video/x-ms-wmv": "video/wmv",
}


class VideoParseError(Exception):
    pass


class VideoParseTool(BaseTool):
    """Analyze video files or video links using Gemini Files API."""

    name: str = "video_parse"
    description: str = (
        "Analyze, summarize, or extract information from a video link or uploaded/local video file. "
        "Use this whenever the user sends a video URL or video file and asks to understand, summarize, "
        "parse, describe, transcribe, or extract timeline/content from the video. "
        "For URLs, the tool downloads with you-get, merges split audio/video with ffmpeg stream copy, "
        "uploads the final video to Gemini Files API, and returns JSON/text analysis."
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
                "description": "Optional Gemini model name. Defaults to gemini-2.5-flash.",
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
        file_name = None
        success = False

        try:
            url, file_path = self._resolve_input(args)
            self._validate_api_key(runtime)
            self._validate_commands(url_required=bool(url))
            work_dir = self._make_work_dir(runtime)

            if url:
                self.report_progress("正在下载视频...")
                media_paths = self._download_video(url, work_dir, runtime["download_timeout"])
            else:
                local_source_path = self._resolve_local_path(file_path)
                media_paths = [local_source_path]

            self.report_progress("正在检查音视频流...")
            final_path, merge_performed = self._prepare_final_video(
                media_paths,
                work_dir,
                runtime["ffmpeg_timeout"],
            )

            self._validate_video_size(final_path, runtime["max_video_bytes"])
            mime_type = self._detect_mime_type(final_path)
            if mime_type not in SUPPORTED_VIDEO_MIME_TYPES:
                supported = ", ".join(sorted(SUPPORTED_VIDEO_MIME_TYPES))
                raise VideoParseError(
                    f"Unsupported video MIME type: {mime_type}. Supported: {supported}"
                )

            self.report_progress("正在上传视频到 Gemini Files API...")
            file_info = self._upload_file(final_path, mime_type, runtime)
            file_uri = self._file_field(file_info, "uri")
            file_name = self._file_field(file_info, "name")
            if not file_uri:
                raise VideoParseError("Gemini Files API did not return file.uri")

            if file_name:
                self.report_progress("正在等待 Gemini 完成视频处理...")
                file_info = self._wait_for_file_active(file_name, runtime)
                file_uri = self._file_field(file_info, "uri") or file_uri

            self.report_progress("正在调用 Gemini 解析视频...")
            response_data = self._generate_content_when_ready(file_name, file_uri, mime_type, runtime)
            raw_text = self._extract_candidate_text(response_data)
            parsed = self._parse_json_text(raw_text)
            usage = response_data.get("usageMetadata") or {}

            result = parsed if isinstance(parsed, dict) else {}
            result.setdefault("raw_text", raw_text)
            result["_meta"] = {
                "model": runtime["model"],
                "mime_type": mime_type,
                "merge_performed": merge_performed,
                "temp_files_deleted": not runtime["keep_temp"],
                "source_file_deleted": bool(local_source_path and runtime["delete_source_on_success"]),
                "remote_file_deleted": bool(file_name and runtime["delete_remote_file"]),
                "usage": usage,
            }
            if runtime["keep_temp"]:
                result["_meta"]["temp_dir"] = work_dir
            if not runtime["delete_remote_file"]:
                result["_meta"]["file_uri"] = file_uri
            success = True
            return ToolResult.success(result)

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
                    file_name=file_name,
                    runtime=runtime,
                )
            elif work_dir and not runtime.get("keep_temp"):
                self._remove_dir(work_dir)

    def _runtime_config(self, args: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config or {}
        api_base = (
            cfg.get("api_base")
            or os.environ.get("GEMINI_API_BASE")
            or conf().get("gemini_api_base")
            or DEFAULT_API_BASE
        )
        upload_api_base = cfg.get("upload_api_base") or api_base
        model = (
            args.get("model")
            or cfg.get("model")
            or os.environ.get("GEMINI_VIDEO_MODEL")
            or DEFAULT_MODEL
        )
        keep_temp = args.get("keep_temp")
        if keep_temp is None:
            keep_temp = self._bool_value(cfg.get("keep_temp", False))

        return {
            "api_key": cfg.get("api_key") or os.environ.get("GEMINI_API_KEY") or conf().get("gemini_api_key", ""),
            "api_base": str(api_base).rstrip("/"),
            "upload_api_base": str(upload_api_base).rstrip("/"),
            "model": str(model).strip() or DEFAULT_MODEL,
            "prompt": args.get("prompt") or cfg.get("prompt") or DEFAULT_PROMPT,
            "download_timeout": self._int_value(cfg.get("download_timeout"), DEFAULT_DOWNLOAD_TIMEOUT),
            "ffmpeg_timeout": self._int_value(cfg.get("ffmpeg_timeout"), DEFAULT_FFMPEG_TIMEOUT),
            "gemini_timeout": self._int_value(cfg.get("gemini_timeout"), DEFAULT_GEMINI_TIMEOUT),
            "processing_timeout": self._int_value(cfg.get("processing_timeout"), DEFAULT_PROCESSING_TIMEOUT),
            "max_video_bytes": self._int_value(cfg.get("max_video_bytes"), MAX_VIDEO_BYTES),
            "temp_dir": cfg.get("temp_dir") or "",
            "keep_temp": bool(keep_temp),
            "delete_source_on_success": self._bool_value(cfg.get("delete_source_on_success", True)),
            "delete_remote_file": self._bool_value(cfg.get("delete_remote_file", True)),
            "prefer_json": self._bool_value(cfg.get("prefer_json", True)),
        }

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

    def _validate_commands(self, url_required: bool) -> None:
        if url_required and not shutil.which("you-get"):
            raise VideoParseError(
                "Missing dependency: you-get. Install project requirements or run: pip install you-get"
            )
        missing = [cmd for cmd in ("ffmpeg", "ffprobe") if not shutil.which(cmd)]
        if missing:
            raise VideoParseError(
                "Missing system dependency: {}. Please install ffmpeg and ensure ffmpeg/ffprobe are in PATH.".format(
                    ", ".join(missing)
                )
            )

    def _make_work_dir(self, runtime: Dict[str, Any]) -> str:
        base = runtime.get("temp_dir") or os.path.join(self.cwd, "tmp", "video_parse")
        base = expand_path(base)
        work_dir = os.path.join(base, uuid.uuid4().hex)
        os.makedirs(work_dir, exist_ok=True)
        return work_dir

    def _download_video(self, url: str, work_dir: str, timeout: int) -> List[str]:
        cmd = ["you-get", "-o", work_dir, url]
        logger.info(f"[VideoParse] Downloading video: {url} -> {work_dir}")
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise VideoParseError(f"you-get timed out after {timeout}s")

        output = completed.stdout or ""
        if completed.returncode != 0:
            raise VideoParseError(
                "you-get failed with exit code {}:\n{}".format(
                    completed.returncode,
                    self._tail(output),
                )
            )

        media_paths = self._scan_media_files(work_dir)
        if not media_paths:
            raise VideoParseError(
                "you-get completed but no usable media file was found.\n{}".format(
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
                if self._probe_media(path):
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
            probe = self._probe_media(path)
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
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60)
            data = json.loads(out.decode("utf-8", errors="replace") or "{}")
        except Exception:
            return None

        streams = data.get("streams") or []
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        if not has_video and not has_audio:
            return None
        return {"has_video": has_video, "has_audio": has_audio}

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
        url = "{}/v1beta/{}".format(runtime["api_base"], file_name)
        response = requests.get(
            url,
            headers={"x-goog-api-key": runtime["api_key"]},
            timeout=runtime["gemini_timeout"],
        )
        self._raise_for_gemini_error(response, "get uploaded file")
        return response.json()

    def _generate_content_when_ready(
        self,
        file_name: Optional[str],
        file_uri: str,
        mime_type: str,
        runtime: Dict[str, Any],
    ) -> Dict[str, Any]:
        deadline = time.time() + runtime["processing_timeout"]
        attempt = 0
        while True:
            try:
                return self._generate_content(file_uri, mime_type, runtime)
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

    def _generate_content(self, file_uri: str, mime_type: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
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
                        {"text": runtime["prompt"]},
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
        work_dir: str,
        local_source_path: Optional[str],
        file_name: Optional[str],
        runtime: Dict[str, Any],
    ) -> None:
        if runtime.get("delete_remote_file") and file_name:
            self._delete_remote_file(file_name, runtime)

        if local_source_path and runtime.get("delete_source_on_success"):
            self._remove_file(local_source_path)

        if not runtime.get("keep_temp"):
            self._remove_dir(work_dir)

    def _delete_remote_file(self, file_name: str, runtime: Dict[str, Any]) -> None:
        try:
            url = "{}/v1beta/{}".format(runtime["api_base"], file_name)
            response = requests.delete(
                url,
                headers={"x-goog-api-key": runtime["api_key"]},
                timeout=min(runtime["gemini_timeout"], 60),
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
        raise VideoParseError(f"Gemini API failed to {action}: HTTP {response.status_code}: {message}")

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

    def _bool_value(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
