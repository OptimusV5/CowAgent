import os
from typing import Any, Dict

from agent.tools.base_tool import BaseTool, ToolResult
from bridge.reply import ReplyType


class AudioTranscribe(BaseTool):
    """Transcribe audio files through CowAgent's configured ASR pipeline."""

    name: str = "audio_transcribe"
    description: str = (
        "Transcribe a local audio or voice file to text using CowAgent's configured speech recognition provider. "
        "Use this whenever the user asks to convert, transcribe, recognize, or understand an audio/voice file. "
        "Do not inspect env_config or call OpenAI/curl audio APIs yourself; this tool uses the Models page ASR configuration. "
        "When the conversation provides temporary domain terms, names, or correction rules, pass hotwords and replace_json for this call only. "
        "When you transcribe an uploaded audio file for a meeting summary or meeting minutes, use the returned file_path "
        "to call audio_archive after you finish generating the summary/title."
    )
    params: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Local path to the uploaded audio/voice file to transcribe.",
            },
            "instance_id": {
                "type": "string",
                "description": "Optional ASR OpenAI-compatible instance id for this transcription only. Leave empty to use the selected ASR instance.",
            },
            "model": {
                "type": "string",
                "description": "Optional ASR model override for this transcription only.",
            },
            "response_format": {
                "type": "string",
                "description": "Optional response_format override, e.g. verbose_json. Omit when not needed.",
            },
            "hotwords": {
                "type": "string",
                "description": "Optional comma-separated hotwords for this transcription only, e.g. OpenAI|10,飞书|10,CowAgent|11.",
            },
            "replace_json": {
                "type": "object",
                "description": "Optional post-processing replacement map for this transcription only, e.g. {\"扣agent\":\"CowAgent\"}.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["file_path"],
    }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        file_path = (args.get("file_path") or "").strip()
        if not file_path:
            return ToolResult.fail("Error: file_path is required")
        file_path = os.path.expanduser(file_path)
        if not os.path.isfile(file_path):
            return ToolResult.fail(f"Error: audio file not found: {file_path}")

        from bridge.bridge import Bridge

        options = {}
        for key in ("instance_id", "model", "response_format", "hotwords"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                options[key] = value.strip()
        replace_json = args.get("replace_json")
        if isinstance(replace_json, dict) and replace_json:
            options["replace_json"] = {str(k): str(v) for k, v in replace_json.items()}
        elif isinstance(replace_json, str) and replace_json.strip():
            options["replace_json"] = replace_json.strip()

        reply = Bridge().fetch_voice_to_text(file_path, options=options or None)
        if reply is None:
            return ToolResult.fail("Error: ASR returned no result")
        if reply.type == ReplyType.TEXT:
            return ToolResult.success({
                "text": reply.content or "",
                "file_path": file_path,
            })
        return ToolResult.fail(reply.content or "Error: ASR failed")
