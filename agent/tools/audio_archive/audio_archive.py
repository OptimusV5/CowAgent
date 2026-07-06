import os
import re
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

from agent.tools.base_tool import BaseTool, ToolResult
from common.log import logger
from common.utils import expand_path
from config import conf


AUDIO_EXTENSIONS = {
    ".aac", ".amr", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma",
}
DEFAULT_RCLONE_BIN = "rclone"
DEFAULT_RCLONE_REMOTE = "alistdav"
DEFAULT_BASE_DIR = "/quark/COW"
DEFAULT_TIMEOUT = 300
DEFAULT_TITLE = "会议纪要"


class AudioArchiveError(Exception):
    pass


class AudioArchiveTool(BaseTool):
    """Archive local audio files to configured cloud storage via rclone."""

    name: str = "audio_archive"
    description: str = (
        "Archive a local audio file to configured cloud storage after you finish transcribing it "
        "and generating a meeting summary or meeting minutes. Use this tool when the user asks to "
        "summarize, transcribe, 整理, 生成会议纪要, or create a Word/PDF meeting note from an audio "
        "file. First call audio_transcribe, then generate the meeting note, then call this tool with "
        "the original audio file_path and a concise title based on the meeting note title. The title "
        "must not include file extensions like .docx, .pdf, .wav, or .mp3."
    )

    params: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Local audio file path to archive. Use the file_path returned by audio_transcribe.",
            },
            "title": {
                "type": "string",
                "description": (
                    "Concise archive title, usually the meeting-minutes document title. "
                    "Do not include file extensions like .docx, .pdf, .wav, .mp3."
                ),
            },
            "date": {
                "type": "string",
                "description": "Optional archive date in YYYY-MM-DD. Used to choose the YYYY-MM remote directory.",
            },
            "delete_local": {
                "type": "boolean",
                "description": "Optional override for deleting the local temp audio after successful upload.",
            },
        },
        "required": ["file_path", "title"],
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.cwd = self.config.get("cwd", os.getcwd())

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        args = args or {}
        try:
            runtime = self._runtime_config()
            if not runtime["enabled"]:
                raise AudioArchiveError("audio_archive is disabled in tools.audio_archive.enabled")

            file_path = self._resolve_file_path(args.get("file_path"))
            title = self._sanitize_title(args.get("title"))
            if not title:
                title = DEFAULT_TITLE
            archive_date = self._parse_date(args.get("date"))
            delete_local = args.get("delete_local")
            if delete_local is None:
                delete_local = runtime["delete_local_on_success"]
            else:
                delete_local = bool(delete_local)

            remote_dir = self._remote_dir(runtime, archive_date)
            filename = self._archive_filename(file_path, title, archive_date)
            remote_path = f"{remote_dir.rstrip('/')}/{filename}"

            self.report_progress(f"正在创建音频归档目录: {remote_dir}")
            self._run_rclone([runtime["rclone_bin"], "mkdir", remote_dir], runtime["timeout"], "mkdir")

            self.report_progress(f"正在归档音频到网盘: {filename}")
            self._run_rclone(
                [runtime["rclone_bin"], "copyto", file_path, remote_path],
                runtime["timeout"],
                "copyto",
            )

            deleted_local = False
            delete_local_skipped_reason = ""
            if delete_local:
                deleted_local, delete_local_skipped_reason = self._delete_local_temp_file(file_path)

            result = {
                "archived": True,
                "file_path": file_path,
                "remote_dir": remote_dir,
                "remote_path": remote_path,
                "title": title,
                "deleted_local": deleted_local,
            }
            if delete_local_skipped_reason:
                result["delete_local_skipped_reason"] = delete_local_skipped_reason
            logger.info(f"[AudioArchive] Archived audio: {file_path} -> {remote_path}")
            return ToolResult.success(result)
        except AudioArchiveError as e:
            logger.warning(f"[AudioArchive] {e}")
            return ToolResult.fail(f"Error: {e}")
        except Exception as e:
            logger.error(f"[AudioArchive] Unexpected error: {e}", exc_info=True)
            return ToolResult.fail(f"Error: Audio archive failed: {e}")

    def _runtime_config(self) -> Dict[str, Any]:
        cfg = self._latest_tool_config()
        return {
            "enabled": self._bool_value(cfg.get("enabled", True)),
            "rclone_bin": str(cfg.get("rclone_bin") or DEFAULT_RCLONE_BIN).strip(),
            "rclone_remote": str(cfg.get("rclone_remote") or DEFAULT_RCLONE_REMOTE).strip(),
            "base_dir": str(cfg.get("base_dir") or DEFAULT_BASE_DIR).strip(),
            "delete_local_on_success": self._bool_value(cfg.get("delete_local_on_success", True)),
            "timeout": self._int_value(cfg.get("timeout"), DEFAULT_TIMEOUT),
        }

    def _latest_tool_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if isinstance(self.config, dict):
            cfg.update(self.config)
        tools_cfg = conf().get("tools", {})
        if isinstance(tools_cfg, dict) and isinstance(tools_cfg.get(self.name), dict):
            cfg.update(tools_cfg.get(self.name) or {})
        return cfg

    def _resolve_file_path(self, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            raise AudioArchiveError("file_path is required")
        if raw.startswith("file://"):
            raw = raw[7:]
        path = os.path.abspath(expand_path(raw))
        if not os.path.isfile(path):
            raise AudioArchiveError(f"audio file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext not in AUDIO_EXTENSIONS:
            raise AudioArchiveError(f"unsupported audio file extension: {ext or '(none)'}")
        return path

    def _sanitize_title(self, value: Any) -> str:
        title = str(value or "").strip()
        title = re.sub(r"^[#*\-\s>]+", "", title).strip()
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
        title = re.sub(
            r"\.(?:docx?|pdf|md|txt|wav|mp3|m4a|opus|aac|flac|ogg|amr|wma)\s*$",
            "",
            title,
            flags=re.IGNORECASE,
        )
        title = re.sub(r"[*_`~\[\]（）()【】]+", "", title).strip()
        title = re.sub(r"[\\/:*?\"<>|]", "_", title)
        title = re.sub(r"\s+", "_", title).strip("._- ")
        return title[:80]

    def _parse_date(self, value: Any) -> datetime:
        raw = str(value or "").strip()
        if not raw:
            return datetime.now()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        raise AudioArchiveError("date must be YYYY-MM-DD")

    def _remote_dir(self, runtime: Dict[str, Any], archive_date: datetime) -> str:
        remote = runtime["rclone_remote"].rstrip(":")
        if not remote:
            raise AudioArchiveError("tools.audio_archive.rclone_remote is required")
        base_dir = "/" + runtime["base_dir"].strip("/")
        month_dir = archive_date.strftime("%Y-%m")
        return f"{remote}:{base_dir}/{month_dir}"

    def _archive_filename(self, file_path: str, title: str, archive_date: datetime) -> str:
        ext = os.path.splitext(file_path)[1].lower() or ".audio"
        suffix = archive_date.strftime("%Y%m%d_%H%M%S")
        return f"{title}_{suffix}{ext}"

    def _run_rclone(self, cmd: list, timeout: int, action: str) -> None:
        rclone_bin = cmd[0]
        if os.path.sep not in rclone_bin and shutil.which(rclone_bin) is None:
            raise AudioArchiveError(f"rclone not found: {rclone_bin}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise AudioArchiveError(f"rclone {action} timed out after {timeout}s") from e
        except Exception as e:
            raise AudioArchiveError(f"rclone {action} failed: {e}") from e
        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip()
            raise AudioArchiveError(
                f"rclone {action} failed: returncode={result.returncode}, output={output[:500]}"
            )

    def _delete_local_temp_file(self, file_path: str) -> tuple:
        workspace_root = os.path.abspath(expand_path(conf().get("agent_workspace", "~/cow")))
        tmp_roots = [
            os.path.abspath(os.path.join(workspace_root, "tmp")),
            os.path.abspath(os.path.join(self.cwd, "tmp")),
        ]
        abs_path = os.path.abspath(file_path)
        for tmp_root in tmp_roots:
            try:
                if os.path.commonpath([tmp_root, abs_path]) == tmp_root:
                    os.remove(abs_path)
                    logger.info(f"[AudioArchive] Removed local temp audio: {abs_path}")
                    return True, ""
            except FileNotFoundError:
                return False, "file already removed"
            except Exception as e:
                return False, f"failed to remove local file: {e}"
        return False, "local file is not under a recognized tmp directory"

    @staticmethod
    def _bool_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _int_value(value: Any, default: int) -> int:
        try:
            parsed = int(value)
            return parsed if parsed > 0 else default
        except Exception:
            return default
