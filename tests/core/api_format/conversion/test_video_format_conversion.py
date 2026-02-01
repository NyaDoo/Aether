"""
视频格式转换单元测试

覆盖重点：
- OpenAI <-> Gemini 视频请求格式转换
- 视频任务响应格式转换
"""

from __future__ import annotations

from src.core.api_format.conversion.normalizers.gemini import GeminiNormalizer
from src.core.api_format.conversion.normalizers.openai import OpenAINormalizer
from src.core.api_format.conversion.registry import FormatConversionRegistry


def _make_registry() -> FormatConversionRegistry:
    reg = FormatConversionRegistry()
    reg.register(OpenAINormalizer())
    reg.register(GeminiNormalizer())
    return reg


class TestVideoRequestConversion:
    """视频请求格式转换测试"""

    def test_openai_to_gemini_video_request(self) -> None:
        """OpenAI Sora -> Gemini Veo 请求格式转换"""
        reg = _make_registry()

        openai_request = {
            "model": "sora-2",
            "prompt": "A cat playing piano",
            "size": "1280x720",
            "seconds": 8,
        }

        gemini_request = reg.convert_video_request(openai_request, "openai:video", "gemini:video")

        # 验证 Gemini 格式
        assert "instances" in gemini_request
        assert "parameters" in gemini_request
        assert isinstance(gemini_request["instances"], list)
        assert len(gemini_request["instances"]) > 0

        instance = gemini_request["instances"][0]
        assert instance["prompt"] == "A cat playing piano"

        params = gemini_request["parameters"]
        assert params["aspectRatio"] == "16:9"
        assert params["resolution"] == "720p"
        assert params["durationSeconds"] == 8

    def test_gemini_to_openai_video_request(self) -> None:
        """Gemini Veo -> OpenAI Sora 请求格式转换"""
        reg = _make_registry()

        gemini_request = {
            "model": "veo-3.1-generate-preview",
            "instances": [{"prompt": "A beautiful sunset over mountains"}],
            "parameters": {
                "aspectRatio": "16:9",
                "resolution": "1080p",
                "durationSeconds": 5,
            },
        }

        openai_request = reg.convert_video_request(gemini_request, "gemini:video", "openai:video")

        # 验证 OpenAI 格式
        assert openai_request["prompt"] == "A beautiful sunset over mountains"
        assert openai_request["model"] == "veo-3.1-generate-preview"
        assert openai_request["seconds"] == 5
        assert openai_request["size"] == "1920x1080"

    def test_same_format_no_conversion(self) -> None:
        """相同格式不进行转换"""
        reg = _make_registry()

        openai_request = {
            "model": "sora-2",
            "prompt": "Test",
            "size": "720x1280",
            "seconds": 4,
        }

        result = reg.convert_video_request(openai_request, "openai:video", "openai:video")
        assert result == openai_request

    def test_conversion_with_reference_image(self) -> None:
        """带参考图片的请求转换"""
        reg = _make_registry()

        openai_request = {
            "model": "sora-2",
            "prompt": "Animate this image",
            "size": "1280x720",
            "seconds": 4,
            "input_reference": "base64_encoded_image_data",
        }

        gemini_request = reg.convert_video_request(openai_request, "openai:video", "gemini:video")

        instance = gemini_request["instances"][0]
        assert "image" in instance
        assert instance["image"]["bytesBase64Encoded"] == "base64_encoded_image_data"


class TestVideoTaskConversion:
    """视频任务响应格式转换测试"""

    def test_gemini_to_openai_processing_task(self) -> None:
        """Gemini 处理中任务 -> OpenAI 格式"""
        reg = _make_registry()

        gemini_response = {
            "name": "operations/12345",
            "done": False,
            "metadata": {"progress": 50},
        }

        openai_response = reg.convert_video_task(gemini_response, "gemini:video", "openai:video")

        # OpenAI 格式使用 status 字段
        assert openai_response["status"] in ("queued", "processing")
        assert "id" in openai_response

    def test_gemini_to_openai_completed_task(self) -> None:
        """Gemini 完成任务 -> OpenAI 格式"""
        reg = _make_registry()

        gemini_response = {
            "name": "operations/12345",
            "done": True,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [{"video": {"uri": "https://example.com/video.mp4"}}]
                }
            },
        }

        openai_response = reg.convert_video_task(gemini_response, "gemini:video", "openai:video")

        assert openai_response["status"] == "completed"
        assert openai_response["progress"] == 100

    def test_openai_to_gemini_processing_task(self) -> None:
        """OpenAI 处理中任务 -> Gemini 格式"""
        reg = _make_registry()

        openai_response = {
            "id": "task_12345",
            "object": "video",
            "status": "processing",
            "progress": 30,
            "created_at": 1700000000,
        }

        gemini_response = reg.convert_video_task(openai_response, "openai:video", "gemini:video")

        # Gemini 格式使用 done 字段
        assert gemini_response["done"] is False
        assert "name" in gemini_response

    def test_openai_to_gemini_completed_task(self) -> None:
        """OpenAI 完成任务 -> Gemini 格式"""
        reg = _make_registry()

        openai_response = {
            "id": "task_12345",
            "object": "video",
            "status": "completed",
            "progress": 100,
            "created_at": 1700000000,
            "completed_at": 1700000100,
        }

        gemini_response = reg.convert_video_task(openai_response, "openai:video", "gemini:video")

        assert gemini_response["done"] is True


class TestVideoConversionCapabilities:
    """视频格式转换能力检查测试"""

    def test_can_convert_video(self) -> None:
        """检查视频格式转换能力"""
        reg = _make_registry()

        assert reg.can_convert_video("openai:video", "gemini:video") is True
        assert reg.can_convert_video("gemini:video", "openai:video") is True
        assert reg.can_convert_video("openai:video", "openai:video") is True

    def test_cannot_convert_unsupported_format(self) -> None:
        """不支持的格式无法转换"""
        reg = _make_registry()

        # 没有注册 Claude normalizer 用于视频
        assert reg.can_convert_video("openai:video", "claude:video") is False
