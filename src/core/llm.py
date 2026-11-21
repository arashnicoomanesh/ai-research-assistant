"""
LLM client for interacting with Ollama.

Provides a clean interface for generating completions using local models via Ollama.
Supports streaming and non-streaming responses with proper error handling.
"""
import json
import logging
import threading
from typing import Optional, Iterator, List, Dict, Any
import requests
from src.core.config import config

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Client for Ollama LLM API."""

    _session: Optional[requests.Session] = None
    _session_lock = threading.Lock()

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize Ollama LLM client.

        Args:
            host: Ollama server URL (default from config)
            model: Model name to use (default from config)
            timeout: Request timeout in seconds (default from config)
        """
        self.host = host or config.ollama.host
        self.model = model or config.ollama.llm_model
        self.timeout = timeout if timeout is not None else config.ollama.llm_timeout_seconds
        self._base_url = f"{self.host}/api"
        logger.info(f"OllamaLLM initialized: model={self.model}, timeout={self.timeout}s")

    @classmethod
    def _get_session(cls) -> requests.Session:
        """Get or create the shared requests.Session (thread-safe)."""
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:
                    cls._session = requests.Session()
                    logger.debug("Created shared requests.Session for OllamaLLM")
        return cls._session

    @classmethod
    def close_session(cls) -> None:
        """Close the shared session (for cleanup/testing)."""
        with cls._session_lock:
            if cls._session is not None:
                cls._session.close()
                cls._session = None
                logger.debug("Closed shared requests.Session for OllamaLLM")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a completion from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default from config)
            max_tokens: Maximum tokens to generate (default from config)
            stream: Whether to stream the response
            **kwargs: Additional options passed to Ollama

        Returns:
            Generated text response
        """
        if stream:
            return "".join(self.generate_stream(prompt, system_prompt, temperature, max_tokens, **kwargs))

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else config.generation.temperature,
                "num_predict": max_tokens if max_tokens is not None else config.generation.max_tokens,
                **kwargs
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        response = self._get_session().post(
            f"{self._base_url}/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate a streaming completion from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options passed to Ollama

        Yields:
            Text chunks as they are generated
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else config.generation.temperature,
                "num_predict": max_tokens if max_tokens is not None else config.generation.max_tokens,
                **kwargs
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        response = self._get_session().post(
            f"{self._base_url}/generate",
            json=payload,
            timeout=self.timeout,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done", False):
                    break

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Chat completion using Ollama's chat API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional options

        Returns:
            Generated response text
        """
        if stream:
            return "".join(self.chat_stream(messages, temperature, max_tokens, **kwargs))

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else config.generation.temperature,
                "num_predict": max_tokens if max_tokens is not None else config.generation.max_tokens,
                **kwargs
            }
        }

        response = self._get_session().post(
            f"{self._base_url}/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Streaming chat completion.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional options

        Yields:
            Text chunks
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else config.generation.temperature,
                "num_predict": max_tokens if max_tokens is not None else config.generation.max_tokens,
                **kwargs
            }
        }

        response = self._get_session().post(
            f"{self._base_url}/chat",
            json=payload,
            timeout=self.timeout,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done", False):
                    break

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            response = self._get_session().get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """List available models on Ollama server."""
        try:
            response = self._get_session().get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except requests.RequestException:
            return []


# Convenience function for quick usage
def get_llm() -> OllamaLLM:
    """Get a default LLM client instance."""
    return OllamaLLM()