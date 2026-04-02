"""Tests for OpenAIInferenceService.

Static methods are pure functions. Constructor and generate() mock the openai client.
"""

from unittest.mock import MagicMock, patch

import pytest

from gigaevo.memory.openai_inference import OpenAIInferenceService

# ===========================================================================
# _resolve_base_url (static, pure)
# ===========================================================================


class TestResolveBaseUrl:
    def test_explicit_url(self):
        result = OpenAIInferenceService._resolve_base_url(
            "http://localhost:8000", "sk-abc"
        )
        assert result == "http://localhost:8000"

    def test_explicit_url_with_whitespace(self):
        result = OpenAIInferenceService._resolve_base_url(
            "  http://localhost:8000  ", "sk-abc"
        )
        assert result == "http://localhost:8000"

    def test_openrouter_key_detection(self):
        result = OpenAIInferenceService._resolve_base_url(None, "sk-or-abc123")
        assert result == "https://openrouter.ai/api/v1"

    def test_regular_key_no_base_url(self):
        result = OpenAIInferenceService._resolve_base_url(None, "sk-abc123")
        assert result is None

    def test_empty_base_url_with_openrouter_key(self):
        result = OpenAIInferenceService._resolve_base_url("", "sk-or-abc")
        assert result == "https://openrouter.ai/api/v1"


# ===========================================================================
# _extract_content_text (static, pure)
# ===========================================================================


class TestExtractContentText:
    def test_string(self):
        assert OpenAIInferenceService._extract_content_text("hello") == "hello"

    def test_list_of_dicts(self):
        parts = [{"text": "a"}, {"text": "b"}]
        assert OpenAIInferenceService._extract_content_text(parts) == "ab"

    def test_list_with_objects(self):
        part = MagicMock()
        part.text = "obj"
        assert OpenAIInferenceService._extract_content_text([part]) == "obj"

    def test_list_without_text(self):
        parts = [{"type": "image"}]
        assert OpenAIInferenceService._extract_content_text(parts) == ""

    def test_none(self):
        assert OpenAIInferenceService._extract_content_text(None) == ""

    def test_int(self):
        assert OpenAIInferenceService._extract_content_text(42) == ""

    def test_empty_list(self):
        assert OpenAIInferenceService._extract_content_text([]) == ""


# ===========================================================================
# _extract_total_tokens (static, pure)
# ===========================================================================


class TestExtractTotalTokens:
    def test_total_present(self):
        assert (
            OpenAIInferenceService._extract_total_tokens({"total_tokens": 100}) == 100
        )

    def test_prompt_and_completion(self):
        usage = {"prompt_tokens": 30, "completion_tokens": 70}
        assert OpenAIInferenceService._extract_total_tokens(usage) == 100

    def test_input_output_fallback(self):
        usage = {"input_tokens": 20, "output_tokens": 80}
        assert OpenAIInferenceService._extract_total_tokens(usage) == 100

    def test_no_info(self):
        assert OpenAIInferenceService._extract_total_tokens({}) is None

    def test_partial_prompt_only(self):
        assert OpenAIInferenceService._extract_total_tokens({"prompt_tokens": 50}) == 50

    def test_total_takes_precedence(self):
        usage = {"total_tokens": 200, "prompt_tokens": 30, "completion_tokens": 70}
        assert OpenAIInferenceService._extract_total_tokens(usage) == 200


# ===========================================================================
# Constructor
# ===========================================================================


class TestInit:
    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key is required"):
            OpenAIInferenceService(model_name="gpt-4", api_key="")

    @patch("gigaevo.memory.openai_inference.OpenAI", create=True)
    def test_valid_init(self, mock_openai_cls):
        # Mock the import inside the constructor
        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            svc = OpenAIInferenceService(
                model_name="gpt-4",
                api_key="sk-test",
            )
            assert svc.model_name == "gpt-4"

    def test_none_api_key_raises(self):
        with pytest.raises((ValueError, TypeError)):
            OpenAIInferenceService(model_name="gpt-4", api_key=None)  # type: ignore[arg-type]


# ===========================================================================
# generate()
# ===========================================================================


class TestGenerate:
    def _make_service(self):
        """Create service with mocked OpenAI client."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            svc = OpenAIInferenceService(
                model_name="gpt-4",
                api_key="sk-test",
                base_url="http://localhost:8000",
            )
        return svc

    def test_returns_4_tuple(self):
        svc = self._make_service()

        # Mock the response
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model_dump.return_value = {
            "usage": {"total_tokens": 42},
        }
        svc.client.chat.completions.create = MagicMock(return_value=mock_response)

        content, payload, tokens, cost = svc.generate("test prompt")
        assert content == "Hello world"
        assert tokens == 42
        assert cost is None  # no cost in response

    def test_empty_choices(self):
        svc = self._make_service()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.model_dump.return_value = {"usage": {}}
        svc.client.chat.completions.create = MagicMock(return_value=mock_response)

        content, _, _, _ = svc.generate("test")
        assert content == ""

    def test_openrouter_headers(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            svc = OpenAIInferenceService(
                model_name="gpt-4",
                api_key="sk-or-test",
                openrouter_referer="http://example.com",
                openrouter_title="Test App",
            )
        assert svc._is_openrouter is True

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.model_dump.return_value = {"usage": {}}
        svc.client.chat.completions.create = MagicMock(return_value=mock_response)

        svc.generate("test")
        call_kwargs = svc.client.chat.completions.create.call_args[1]
        assert "extra_headers" in call_kwargs
        assert call_kwargs["extra_headers"]["HTTP-Referer"] == "http://example.com"
        assert call_kwargs["extra_headers"]["X-Title"] == "Test App"

    def test_cost_extracted(self):
        svc = self._make_service()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.model_dump.return_value = {
            "usage": {"total_tokens": 10, "cost": 0.005},
        }
        svc.client.chat.completions.create = MagicMock(return_value=mock_response)

        _, _, _, cost = svc.generate("test")
        assert cost == 0.005
