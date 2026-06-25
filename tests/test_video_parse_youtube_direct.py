# encoding:utf-8
"""Unit tests for YouTube direct URL analysis in video_parse."""
import json
import os
import sys
import unittest
from unittest.mock import ANY, MagicMock, patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.tools.video_parse.video_parse import VideoParseError, VideoParseTool


class TestYouTubeUrlRecognition(unittest.TestCase):
    def setUp(self):
        self.tool = VideoParseTool({"api_key": "test-key"})

    def test_accepts_watch_url(self):
        self.assertTrue(self.tool._is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))

    def test_accepts_youtu_be(self):
        self.assertTrue(self.tool._is_youtube_url("https://youtu.be/dQw4w9WgXcQ"))

    def test_accepts_shorts(self):
        self.assertTrue(self.tool._is_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ"))

    def test_accepts_embed(self):
        self.assertTrue(self.tool._is_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ"))

    def test_accepts_live(self):
        self.assertTrue(self.tool._is_youtube_url("https://www.youtube.com/live/dQw4w9WgXcQ"))

    def test_accepts_music_youtube(self):
        self.assertTrue(self.tool._is_youtube_url("https://music.youtube.com/watch?v=dQw4w9WgXcQ"))

    def test_accepts_watch_with_playlist_param(self):
        self.assertTrue(
            self.tool._is_youtube_url(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLtest"
            )
        )

    def test_rejects_playlist_only(self):
        self.assertFalse(self.tool._is_youtube_url("https://www.youtube.com/playlist?list=PLtest"))

    def test_rejects_watch_list_only(self):
        self.assertFalse(self.tool._is_youtube_url("https://www.youtube.com/watch?list=PLtest"))

    def test_rejects_googlevideo(self):
        self.assertFalse(
            self.tool._is_youtube_url(
                "https://rr1---sn-abc.googlevideo.com/videoplayback?expire=1"
            )
        )

    def test_rejects_non_youtube(self):
        self.assertFalse(self.tool._is_youtube_url("https://example.com/video.mp4"))

    def test_rejects_watch_invalid_id(self):
        self.assertFalse(self.tool._is_youtube_url("https://www.youtube.com/watch?v=tooshort"))

    def test_rejects_watch_overlong_id(self):
        self.assertFalse(self.tool._is_youtube_url("https://www.youtube.com/watch?v=waytoolongvideoid"))

    def test_rejects_youtu_be_invalid_id(self):
        self.assertFalse(self.tool._is_youtube_url("https://youtu.be/short"))


class TestYouTubeDirectExecution(unittest.TestCase):
    YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    OTHER_URL = "https://example.com/video.mp4"

    def _tool(self, **config):
        cfg = {
            "api_key": "test-key",
            "youtube_direct_enabled": True,
            "youtube_direct_fallback": True,
        }
        cfg.update(config)
        return VideoParseTool(cfg)

    def _gemini_response(self, summary="测试摘要"):
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps({"summary": summary})}]}}],
            "usageMetadata": {"totalTokenCount": 42},
        }

    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_direct_path_skips_download(self, mock_generate, mock_download):
        tool = self._tool()
        mock_generate.return_value = self._gemini_response()

        result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "success")
        mock_generate.assert_called_once()
        mock_download.assert_not_called()
        self.assertEqual(result.result["_meta"]["mode"], "youtube_url_direct")
        self.assertFalse(result.result["_meta"]["fallback_used"])
        self.assertTrue(result.result["_meta"]["youtube_direct_attempted"])

    @patch.object(VideoParseTool, "_remove_dir")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_direct_success_skips_temp_dir_cleanup(self, mock_generate, mock_remove_dir):
        tool = self._tool()
        mock_generate.return_value = self._gemini_response()

        result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "success")
        # No temp dir was created on the direct path, so cleanup must be a no-op.
        mock_remove_dir.assert_not_called()

    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_direct_success_emits_no_cleanup_warning(self, mock_generate):
        tool = self._tool()
        mock_generate.return_value = self._gemini_response()

        with self.assertNoLogs("log", level="WARNING"):
            result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "success")

    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_direct_payload_uses_youtube_url(self, mock_generate):
        tool = self._tool()
        mock_generate.return_value = self._gemini_response()

        tool.execute({"url": self.YOUTUBE_URL})

        mock_generate.assert_called_once_with(self.YOUTUBE_URL, ANY, prompt=ANY)

    def test_cleanup_after_success_noop_when_no_work_dir(self):
        tool = self._tool()
        runtime = tool._runtime_config({})
        with patch.object(VideoParseTool, "_remove_dir") as mock_remove_dir, patch.object(
            VideoParseTool, "_remove_file"
        ) as mock_remove_file, self.assertNoLogs("log", level="WARNING"):
            tool._cleanup_after_success(
                work_dir=None,
                local_source_path=None,
                file_names=[],
                runtime=runtime,
            )
        mock_remove_dir.assert_not_called()
        mock_remove_file.assert_not_called()

    @patch.object(VideoParseTool, "_remove_dir")
    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_prepare_final_video")
    @patch.object(VideoParseTool, "_analyze_video_or_segments")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_download_path_still_cleans_temp_dir(
        self,
        mock_generate,
        mock_analyze,
        mock_prepare,
        mock_download,
        mock_remove_dir,
    ):
        tool = self._tool(youtube_direct_enabled=False)
        mock_download.return_value = ["/tmp/video.mp4"]
        mock_prepare.return_value = ("/tmp/final.mp4", False)

        with patch.object(VideoParseTool, "_validate_commands"), patch.object(
            VideoParseTool, "_make_work_dir", return_value="/tmp/work"
        ), patch("os.path.getsize", return_value=1024), patch.object(
            VideoParseTool, "_detect_mime_type", return_value="video/mp4"
        ), patch.object(
            VideoParseTool, "_ensure_ascii_upload_path", side_effect=lambda p, _: p
        ):
            mock_analyze.return_value = {"summary": "downloaded", "_meta": {"mode": "single"}}
            result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "success")
        mock_generate.assert_not_called()
        mock_remove_dir.assert_called_once_with("/tmp/work")

    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_prepare_final_video")
    @patch.object(VideoParseTool, "_analyze_video_or_segments")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_fallback_on_recoverable_error(
        self,
        mock_generate,
        mock_analyze,
        mock_prepare,
        mock_download,
    ):
        tool = self._tool()
        mock_generate.side_effect = VideoParseError(
            "Gemini API failed to generate content from YouTube URL: HTTP 404: not found"
        )
        mock_download.return_value = ["/tmp/video.mp4"]
        mock_prepare.return_value = ("/tmp/final.mp4", False)

        with patch.object(VideoParseTool, "_validate_commands"), patch.object(
            VideoParseTool, "_make_work_dir", return_value="/tmp/work"
        ), patch("os.path.getsize", return_value=1024), patch.object(
            VideoParseTool, "_detect_mime_type", return_value="video/mp4"
        ), patch.object(
            VideoParseTool, "_ensure_ascii_upload_path", side_effect=lambda p, _: p
        ):
            mock_analyze.return_value = {"summary": "fallback", "_meta": {"mode": "single"}}
            result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "success")
        mock_download.assert_called_once()
        meta = result.result["_meta"]
        self.assertTrue(meta["youtube_direct_attempted"])
        self.assertTrue(meta["fallback_used"])
        self.assertIn("youtube_direct_error", meta)

    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_no_fallback_on_invalid_api_key(self, mock_generate, mock_download):
        tool = self._tool()
        mock_generate.side_effect = VideoParseError(
            "Gemini API failed to generate content from YouTube URL: HTTP 400: API key not valid",
            status_code=400,
        )

        result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "error")
        mock_download.assert_not_called()
        self.assertIn("API key not valid", result.result)

    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_failure_path_carries_meta(self, mock_generate, mock_download):
        tool = self._tool()
        mock_generate.side_effect = VideoParseError(
            "Gemini API failed to generate content from YouTube URL: HTTP 429: quota exceeded",
            status_code=429,
        )

        result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "error")
        mock_download.assert_not_called()
        self.assertIsInstance(result.ext_data, dict)
        meta = result.ext_data["_meta"]
        self.assertTrue(meta["youtube_direct_attempted"])
        self.assertFalse(meta["fallback_used"])
        self.assertEqual(meta["source_url"], self.YOUTUBE_URL)
        self.assertIn("model", meta)
        self.assertIn("youtube_direct_error", meta)

    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_prepare_final_video")
    @patch.object(VideoParseTool, "_analyze_video_or_segments")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_fallback_on_permission_denied(
        self,
        mock_generate,
        mock_analyze,
        mock_prepare,
        mock_download,
    ):
        tool = self._tool()
        mock_generate.side_effect = VideoParseError(
            "Gemini API failed to generate content from YouTube URL: HTTP 403: PERMISSION_DENIED",
            status_code=403,
        )
        mock_download.return_value = ["/tmp/video.mp4"]
        mock_prepare.return_value = ("/tmp/final.mp4", False)

        with patch.object(VideoParseTool, "_validate_commands"), patch.object(
            VideoParseTool, "_make_work_dir", return_value="/tmp/work"
        ), patch("os.path.getsize", return_value=1024), patch.object(
            VideoParseTool, "_detect_mime_type", return_value="video/mp4"
        ), patch.object(
            VideoParseTool, "_ensure_ascii_upload_path", side_effect=lambda p, _: p
        ):
            mock_analyze.return_value = {"summary": "fallback", "_meta": {"mode": "single"}}
            result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "success")
        mock_download.assert_called_once()
        self.assertTrue(result.result["_meta"]["fallback_used"])

    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_prepare_final_video")
    @patch.object(VideoParseTool, "_analyze_video_or_segments")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_fallback_on_json_decode_error(
        self,
        mock_generate,
        mock_analyze,
        mock_prepare,
        mock_download,
    ):
        tool = self._tool()
        mock_generate.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_download.return_value = ["/tmp/video.mp4"]
        mock_prepare.return_value = ("/tmp/final.mp4", False)

        with patch.object(VideoParseTool, "_validate_commands"), patch.object(
            VideoParseTool, "_make_work_dir", return_value="/tmp/work"
        ), patch("os.path.getsize", return_value=1024), patch.object(
            VideoParseTool, "_detect_mime_type", return_value="video/mp4"
        ), patch.object(
            VideoParseTool, "_ensure_ascii_upload_path", side_effect=lambda p, _: p
        ):
            mock_analyze.return_value = {"summary": "fallback", "_meta": {"mode": "single"}}
            result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "success")
        mock_download.assert_called_once()
        self.assertTrue(result.result["_meta"]["fallback_used"])

    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_prepare_final_video")
    @patch.object(VideoParseTool, "_analyze_video_or_segments")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_disabled_direct_uses_download_path(
        self,
        mock_generate,
        mock_analyze,
        mock_prepare,
        mock_download,
    ):
        tool = self._tool(youtube_direct_enabled=False)
        mock_download.return_value = ["/tmp/video.mp4"]
        mock_prepare.return_value = ("/tmp/final.mp4", False)

        with patch.object(VideoParseTool, "_validate_commands"), patch.object(
            VideoParseTool, "_make_work_dir", return_value="/tmp/work"
        ), patch("os.path.getsize", return_value=1024), patch.object(
            VideoParseTool, "_detect_mime_type", return_value="video/mp4"
        ), patch.object(
            VideoParseTool, "_ensure_ascii_upload_path", side_effect=lambda p, _: p
        ):
            mock_analyze.return_value = {"summary": "downloaded", "_meta": {"mode": "single"}}
            result = tool.execute({"url": self.YOUTUBE_URL})

        self.assertEqual(result.status, "success")
        mock_generate.assert_not_called()
        mock_download.assert_called_once()
        self.assertNotIn("youtube_direct_attempted", result.result.get("_meta", {}))

    @patch.object(VideoParseTool, "_download_video")
    @patch.object(VideoParseTool, "_prepare_final_video")
    @patch.object(VideoParseTool, "_analyze_video_or_segments")
    @patch.object(VideoParseTool, "_generate_content_youtube_url")
    def test_non_youtube_url_uses_download_path(
        self,
        mock_generate,
        mock_analyze,
        mock_prepare,
        mock_download,
    ):
        tool = self._tool()
        mock_download.return_value = ["/tmp/video.mp4"]
        mock_prepare.return_value = ("/tmp/final.mp4", False)

        with patch.object(VideoParseTool, "_validate_commands"), patch.object(
            VideoParseTool, "_make_work_dir", return_value="/tmp/work"
        ), patch("os.path.getsize", return_value=1024), patch.object(
            VideoParseTool, "_detect_mime_type", return_value="video/mp4"
        ), patch.object(
            VideoParseTool, "_ensure_ascii_upload_path", side_effect=lambda p, _: p
        ):
            mock_analyze.return_value = {"summary": "downloaded", "_meta": {"mode": "single"}}
            result = tool.execute({"url": self.OTHER_URL})

        self.assertEqual(result.status, "success")
        mock_generate.assert_not_called()
        mock_download.assert_called_once()

    def test_fallback_on_timeout(self):
        tool = self._tool()
        self.assertTrue(
            tool._should_youtube_direct_fallback(requests.Timeout("timed out"), tool._runtime_config({}))
        )

    def test_no_fallback_when_disabled(self):
        tool = self._tool(youtube_direct_fallback=False)
        error = VideoParseError(
            "Gemini API failed to generate content from YouTube URL: HTTP 404: not found",
            status_code=404,
        )
        self.assertFalse(tool._should_youtube_direct_fallback(error, tool._runtime_config({})))


class TestYouTubeFallbackDecision(unittest.TestCase):
    def setUp(self):
        self.tool = VideoParseTool({"api_key": "test-key"})
        self.runtime = self.tool._runtime_config({})

    def _fb(self, error):
        return self.tool._should_youtube_direct_fallback(error, self.runtime)

    def test_fallback_on_http_408(self):
        self.assertTrue(self._fb(VideoParseError("HTTP 408: request timeout", status_code=408)))

    def test_fallback_on_connection_error(self):
        self.assertTrue(self._fb(requests.ConnectionError("connection reset by peer")))

    def test_fallback_on_json_decode_error(self):
        self.assertTrue(self._fb(json.JSONDecodeError("Expecting value", "", 0)))

    def test_fallback_on_bare_internal_error(self):
        self.assertTrue(self._fb(VideoParseError("internal error")))

    def test_fallback_on_empty_response(self):
        self.assertTrue(self._fb(VideoParseError("Gemini returned no text content")))

    def test_fallback_on_permission_denied(self):
        self.assertTrue(self._fb(VideoParseError("HTTP 403: PERMISSION_DENIED", status_code=403)))

    def test_no_fallback_on_quota_429(self):
        self.assertFalse(self._fb(VideoParseError("HTTP 429: quota exceeded", status_code=429)))

    def test_no_fallback_on_auth_401(self):
        self.assertFalse(self._fb(VideoParseError("HTTP 401: unauthenticated", status_code=401)))

    def test_no_fallback_on_invalid_api_key(self):
        self.assertFalse(self._fb(VideoParseError("HTTP 400: API key not valid", status_code=400)))

    def test_no_fallback_on_model_not_found(self):
        self.assertFalse(self._fb(VideoParseError("model gemini-x does not exist", status_code=404)))


def _load_video_parse_config_handler():
    """Import VideoParseConfigHandler, stubbing the optional third-party `web` dep."""
    if "web" not in sys.modules:
        sys.modules["web"] = MagicMock()
    from channel.web.web_channel import VideoParseConfigHandler

    return VideoParseConfigHandler


class TestWebVideoParseDefaultConfig(unittest.TestCase):
    def test_default_config_includes_youtube_direct_keys(self):
        handler = _load_video_parse_config_handler()

        defaults = handler._default_config()
        self.assertIn("youtube_direct_enabled", defaults)
        self.assertIn("youtube_direct_fallback", defaults)
        self.assertTrue(defaults["youtube_direct_enabled"])
        self.assertTrue(defaults["youtube_direct_fallback"])

    def test_merge_defaults_preserves_youtube_direct_keys(self):
        handler = _load_video_parse_config_handler()

        merged = handler._merge_defaults(
            {"youtube_direct_enabled": False, "youtube_direct_fallback": False}
        )
        self.assertFalse(merged["youtube_direct_enabled"])
        self.assertFalse(merged["youtube_direct_fallback"])


if __name__ == "__main__":
    unittest.main()
