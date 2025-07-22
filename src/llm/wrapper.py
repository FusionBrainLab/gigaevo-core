"""LLM wrapper with robust error handling and retry logic.

Provides a reliable interface for OpenAI API calls with comprehensive
error handling, validation, and logging.
"""

from abc import ABC, abstractmethod
import asyncio
import random
import time
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field, computed_field

from src.exceptions import LLMAPIError, LLMValidationError, ensure_not_none


class LLMConfig(BaseModel):
    """Configuration for LLM wrapper with Pydantic validation."""

    max_retries: int = Field(
        default=3, ge=0, description="Maximum number of retries"
    )
    retry_delay: float = Field(
        default=1.0, ge=0, description="Base delay between retries"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p sampling"
    )
    max_prompt_length: int = Field(
        default=100000, gt=0, description="Maximum prompt length"
    )
    api_endpoint: Optional[str] = Field(
        default=None, description="Custom API endpoint URL"
    )
    proxy_user: Optional[str] = Field(
        default=None, description="Proxy username"
    )
    proxy_password: Optional[str] = Field(
        default=None, description="Proxy password"
    )
    proxy_host: Optional[str] = Field(default=None, description="Proxy host")
    proxy_port: Optional[int] = Field(default=None, description="Proxy port")

    @computed_field
    @property
    def proxy_url(self) -> Optional[str]:
        if self.proxy_host is None or self.proxy_port is None:
            return None
        return f"socks5://{self.proxy_user}:{self.proxy_password}@{self.proxy_host}:{self.proxy_port}"


class LLMInterface(ABC):
    """Interface for LLM wrapper."""

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Generate a response from the LLM with robust error handling."""

    @abstractmethod
    async def generate_async(self, *args, **kwargs) -> str:
        """Asynchronously generate a response from the LLM with robust error handling."""

    @property
    @abstractmethod
    def model(self) -> str: ...


class LLMWrapper(LLMInterface):
    """LLM wrapper with robust error handling and retry logic.

    Always creates both sync and async clients for maximum flexibility.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        """Initialize the LLM wrapper with both sync and async clients.

        Args:
            model: Model name to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key (if not provided, will use environment variable)
            system_prompt: Optional default system prompt
            config: Optional configuration settings
        """
        ensure_not_none(model, "model")
        if not isinstance(model, str) or not model.strip():
            raise LLMValidationError("model must be a non-empty string")

        self._model = model
        self.system_prompt = system_prompt
        self.config = config or LLMConfig()

        # Initialize both clients
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.config.api_endpoint:
            client_kwargs["base_url"] = self.config.api_endpoint

        self.sync_client = OpenAI(
            **client_kwargs,
            http_client=(
                httpx.Client(proxy=self.config.proxy_url)
                if self.config.proxy_url
                else None
            ),
        )
        self.async_client = AsyncOpenAI(
            **client_kwargs,
            http_client=(
                httpx.AsyncClient(proxy=self.config.proxy_url)
                if self.config.proxy_url
                else None
            ),
        )

        logger.info(f"[LLMWrapper] Initialized with model: {model}")

    def _validate_user_prompt(self, user_prompt: str) -> None:
        """Validate user prompt input."""
        ensure_not_none(user_prompt, "user_prompt")
        if not isinstance(user_prompt, str):
            raise LLMValidationError("user_prompt must be a string")

        if not user_prompt.strip():
            raise LLMValidationError("user_prompt cannot be empty")

        if len(user_prompt) > self.config.max_prompt_length:
            raise LLMValidationError(
                f"user_prompt too long ({len(user_prompt)} > {self.config.max_prompt_length})"
            )

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        transient_errors = (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
        )

        if isinstance(error, APIStatusError):
            return error.status_code in [429, 500, 502, 503, 504]

        return isinstance(error, transient_errors)

    def _handle_api_error(self, error: Exception, attempt: int) -> None:
        """Handle and log API errors appropriately."""
        if isinstance(error, AuthenticationError):
            logger.error(
                f"[LLMWrapper] Authentication failed - check API key: {error}"
            )
            raise LLMAPIError(f"Authentication error: {error}")

        elif isinstance(error, BadRequestError):
            logger.error(f"[LLMWrapper] Bad request: {error}")
            raise LLMAPIError(f"Invalid request: {error}")

        elif isinstance(error, RateLimitError):
            logger.warning(
                f"[LLMWrapper] Rate limit exceeded (attempt {attempt}): {error}"
            )

        elif isinstance(error, APITimeoutError):
            logger.warning(
                f"[LLMWrapper] API timeout (attempt {attempt}): {error}"
            )

        elif isinstance(error, APIConnectionError):
            logger.warning(
                f"[LLMWrapper] Connection error (attempt {attempt}): {error}"
            )

        elif isinstance(error, InternalServerError):
            logger.warning(
                f"[LLMWrapper] Server error (attempt {attempt}): {error}"
            )

        else:
            logger.error(f"[LLMWrapper] Unexpected API error: {error}")
            raise LLMAPIError(f"API error: {error}")

    def _prepare_messages(
        self,
        user_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Prepare messages for API call."""
        if messages is not None:
            return messages

        effective_system_prompt = system_prompt or self.system_prompt

        if effective_system_prompt:
            return [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            return [{"role": "user", "content": user_prompt}]

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    def generate(
        self,
        user_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a response from the LLM with robust error handling.

        Args:
            user_prompt: The user's prompt
            messages: Optional custom message history (overrides system_prompt)
            system_prompt: Optional system prompt for this call (overrides instance default)
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response string
        """
        self._validate_user_prompt(user_prompt)
        prepared_messages = self._prepare_messages(
            user_prompt, messages, system_prompt
        )

        api_params = {
            "model": self.model,
            "messages": prepared_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            **kwargs,
        }

        if self.config.max_tokens:
            api_params["max_tokens"] = self.config.max_tokens

        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(
                    f"[LLMWrapper] Attempting generation (attempt {attempt + 1})"
                )

                response = self.sync_client.chat.completions.create(
                    **api_params
                )

                if not response.choices:
                    raise LLMAPIError("API returned no choices")

                content = response.choices[0].message.content

                if content is None:
                    raise LLMAPIError("API returned None content")

                logger.debug("[LLMWrapper] Generation successful")
                return content

            except Exception as error:
                last_error = error
                self._handle_api_error(error, attempt + 1)

                if (
                    not self._should_retry(error)
                    or attempt >= self.config.max_retries
                ):
                    break

                delay = self.config.retry_delay * (2**attempt)
                logger.info(f"[LLMWrapper] Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

        error_msg = f"Failed after {self.config.max_retries + 1} attempts. Last error: {last_error}"
        logger.error(f"[LLMWrapper] {error_msg}")
        raise LLMAPIError(error_msg)

    async def generate_async(
        self,
        user_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Asynchronously generate a response from the LLM with robust error handling.

        Args:
            user_prompt: The user's prompt
            messages: Optional custom message history (overrides system_prompt)
            system_prompt: Optional system prompt for this call (overrides instance default)
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response string
        """
        self._validate_user_prompt(user_prompt)
        prepared_messages = self._prepare_messages(
            user_prompt, messages, system_prompt
        )

        api_params = {
            "model": self.model,
            "messages": prepared_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            **kwargs,
        }

        if self.config.max_tokens:
            api_params["max_tokens"] = self.config.max_tokens

        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(
                    f"[LLMWrapper] Attempting async generation (attempt {attempt + 1})"
                )

                response = await self.async_client.chat.completions.create(
                    **api_params
                )

                if not response.choices:
                    raise LLMAPIError("API returned no choices")

                content = response.choices[0].message.content

                if content is None:
                    raise LLMAPIError("API returned None content")

                logger.debug("[LLMWrapper] Async generation successful")
                return content

            except Exception as error:
                last_error = error
                self._handle_api_error(error, attempt + 1)

                if (
                    not self._should_retry(error)
                    or attempt >= self.config.max_retries
                ):
                    break

                delay = self.config.retry_delay * (2**attempt)
                logger.info(f"[LLMWrapper] Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)

        error_msg = f"Failed after {self.config.max_retries + 1} attempts. Last error: {last_error}"
        logger.error(f"[LLMWrapper] {error_msg}")
        raise LLMAPIError(error_msg)

    def set_system_prompt(self, system_prompt: Optional[str]) -> None:
        """Set the default system prompt."""
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise LLMValidationError("system_prompt must be a string or None")

        self.system_prompt = system_prompt
        logger.info(f"[LLMWrapper] Updated system prompt")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "model": self.model,
            "config": self.config.model_dump(),
            "has_system_prompt": self.system_prompt is not None,
        }


class MultiModelLLMWrapper(LLMInterface):
    """Wrapper for multiple LLM models with probabilistic selection."""

    def __init__(
        self,
        *,
        models: List[str],
        probabilities: List[float],
        api_key: str,
        system_prompt: Optional[str] = None,
        configs: List[LLMConfig],
    ):
        """Initialize the multi-model wrapper.

        Args:
            models: List of model names
            probabilities: List of selection probabilities (will be normalized)
            api_key: API key for all models
            system_prompt: Optional system prompt for all models
            configs: Optional list of configs, one per model (if not provided, uses defaults)
        """
        # Input validation
        ensure_not_none(models, "models")
        ensure_not_none(probabilities, "probabilities")
        ensure_not_none(api_key, "api_key")

        if not models:
            raise LLMValidationError("models list cannot be empty")

        if len(models) != len(probabilities):
            raise LLMValidationError(
                f"models and probabilities must have same length: {len(models)} != {len(probabilities)}"
            )

        if any(p <= 0 for p in probabilities):
            raise LLMValidationError("all probabilities must be positive")

        elif len(configs) != len(models):
            raise LLMValidationError(
                f"configs and models must have same length: {len(configs)} != {len(models)}"
            )

        self.models = list(models)  # Make a copy
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.configs = list(configs)  # Make a copy

        # Normalize probabilities
        total_prob = sum(probabilities)
        self.probabilities = [p / total_prob for p in probabilities]

        # Initialize model wrappers
        try:
            self.llm_wrappers = [
                LLMWrapper(model, api_key, system_prompt, config)
                for model, config in zip(models, configs)
            ]
        except Exception as e:
            raise LLMValidationError(f"Failed to initialize LLM wrappers: {e}")

        # Default model for property access (first model)
        self._default_model = self.models[0]
        self.selected_model: Optional[LLMWrapper] = None

        logger.info(
            f"[MultiModelLLMWrapper] Initialized with {len(models)} models: {models}"
        )

    def select_model(self) -> LLMWrapper:
        """Select a model based on the probabilities."""
        try:
            return random.choices(
                self.llm_wrappers, weights=self.probabilities
            )[0]
        except Exception as e:
            logger.error(f"[MultiModelLLMWrapper] Model selection failed: {e}")
            # Fallback to first model
            return self.llm_wrappers[0]

    def generate(self, *args, **kwargs) -> str:
        """Generate a response from the LLM with robust error handling."""
        self.selected_model = self.select_model()
        try:
            return self.selected_model.generate(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"[MultiModelLLMWrapper] Generation failed with model {self.selected_model.model}: {e}"
            )
            raise

    async def generate_async(self, *args, **kwargs) -> str:
        """Asynchronously generate a response from the LLM with robust error handling."""
        self.selected_model = self.select_model()
        try:
            return await self.selected_model.generate_async(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"[MultiModelLLMWrapper] Async generation failed with model {self.selected_model.model}: {e}"
            )
            raise

    @property
    def model(self) -> str:
        """Get the currently selected model name, or default if none selected."""
        if self.selected_model is not None:
            return self.selected_model.model
        return self._default_model

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "models": self.models,
            "probabilities": self.probabilities,
            "selected_model": (
                self.selected_model.model if self.selected_model else None
            ),
            "default_model": self._default_model,
            "has_system_prompt": self.system_prompt is not None,
        }
