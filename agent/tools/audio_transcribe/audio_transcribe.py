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
        "Do not inspect env_config or call OpenAI/curl audio APIs yourself; this tool uses the Models page ASR configuration."
    )
    params: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Local path to the uploaded audio/voice file to transcribe.",
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

        reply = Bridge().fetch_voice_to_text(file_path)
        if reply is None:
            return ToolResult.fail("Error: ASR returned no result")
        if reply.type == ReplyType.TEXT:
            return ToolResult.success({"text": reply.content or ""})
        return ToolResult.fail(reply.content or "Error: ASR failed")
