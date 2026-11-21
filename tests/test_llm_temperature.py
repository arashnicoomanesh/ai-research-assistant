"""
Smoke test for LLM temperature and max_tokens preservation.

Verifies that temperature=0.0 and max_tokens=0 are correctly passed
to the Ollama API instead of being replaced by defaults.
"""
import json
from unittest.mock import patch, MagicMock

from src.core.llm import OllamaLLM


class TestLLMTemperaturePreservation:
    """Tests for temperature and max_tokens parameter preservation."""

    def setup_method(self):
        """Clear session before each test."""
        OllamaLLM.close_session()

    def teardown_method(self):
        """Clear session after each test."""
        OllamaLLM.close_session()

    def test_generate_temperature_zero(self):
        """temperature=0.0 should be sent as 0.0, not replaced by default."""
        llm = OllamaLLM(host="http://localhost:11434", model="test-model")

        with patch.object(llm, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response
            mock_get_session.return_value = mock_session

            llm.generate("test prompt", temperature=0.0)

            call_args = mock_session.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["options"]["temperature"] == 0.0, \
                f"Expected 0.0, got {payload['options']['temperature']}"

    def test_generate_max_tokens_zero(self):
        """max_tokens=0 should be sent as 0, not replaced by default."""
        llm = OllamaLLM(host="http://localhost:11434", model="test-model")

        with patch.object(llm, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response
            mock_get_session.return_value = mock_session

            llm.generate("test prompt", max_tokens=0)

            call_args = mock_session.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["options"]["num_predict"] == 0, \
                f"Expected 0, got {payload['options']['num_predict']}"

    def test_chat_temperature_zero(self):
        """temperature=0.0 in chat should be preserved."""
        llm = OllamaLLM(host="http://localhost:11434", model="test-model")

        with patch.object(llm, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "test"}}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response
            mock_get_session.return_value = mock_session

            llm.chat([{"role": "user", "content": "test"}], temperature=0.0)

            call_args = mock_session.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["options"]["temperature"] == 0.0, \
                f"Expected 0.0, got {payload['options']['temperature']}"

    def test_generate_temperature_none_uses_default(self):
        """temperature=None should use config default."""
        llm = OllamaLLM(host="http://localhost:11434", model="test-model")

        with patch.object(llm, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response
            mock_get_session.return_value = mock_session

            llm.generate("test prompt", temperature=None)

            call_args = mock_session.post.call_args
            payload = call_args.kwargs["json"]
            # Default from config is 0.1
            assert payload["options"]["temperature"] == 0.1, \
                f"Expected default 0.1, got {payload['options']['temperature']}"

    def test_generate_max_tokens_none_uses_default(self):
        """max_tokens=None should use config default."""
        llm = OllamaLLM(host="http://localhost:11434", model="test-model")

        with patch.object(llm, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response
            mock_get_session.return_value = mock_session

            llm.generate("test prompt", max_tokens=None)

            call_args = mock_session.post.call_args
            payload = call_args.kwargs["json"]
            # Default from config is 2048
            assert payload["options"]["num_predict"] == 2048, \
                f"Expected default 2048, got {payload['options']['num_predict']}"