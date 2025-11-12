from collections.abc import AsyncIterator, Iterator
import random
from typing import Any, AsyncIterator, Iterator, Optional

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

class MultiModelRouter(Runnable):
    """Probabilistic model selector - drop-in replacement for ChatOpenAI.

    This router implements the same interface as ChatOpenAI, making it a true
    drop-in replacement. It probabilistically selects one of the provided models
    for each invocation, enabling A/B testing and load distribution.

    Supports all ChatOpenAI features:
    - Structured output via .with_structured_output()
    - Streaming
    - Async operations
    - LCEL chaining
    - Built-in Langfuse tracing (automatic, immediate mode)

    Attributes:
        models: List of ChatOpenAI models to choose from
        probabilities: Normalized probability distribution over models
        langfuse_handler: Langfuse callback handler for tracing (auto-created)
        langfuse_client: Langfuse client for manual flushing

    Example:
        >>> models = [
        ...     ChatOpenAI(model="gpt-4"),
        ...     ChatOpenAI(model="gpt-3.5-turbo")
        ... ]
        >>> router = MultiModelRouter(models, [0.8, 0.2])
        >>>
        >>> # Works exactly like ChatOpenAI - all calls automatically traced
        >>> response = await router.ainvoke("Hello!")
        >>>
        >>> # Supports structured output
        >>> router_with_schema = router.with_structured_output(MySchema)
    """

    def __init__(self, models: list[ChatOpenAI], probabilities: list[float]):
        """Initialize router with models and selection probabilities.

        Automatically enables Langfuse tracing if LANGFUSE_PUBLIC_KEY and
        LANGFUSE_SECRET_KEY environment variables are set.

        Args:
            models: List of LangChain ChatOpenAI instances
            probabilities: Selection probabilities (will be normalized)

        Raises:
            ValueError: If models/probabilities length mismatch or invalid probs
        """
        if len(models) != len(probabilities):
            raise ValueError(
                f"models and probabilities must have same length: "
                f"{len(models)} != {len(probabilities)}"
            )

        if any(p <= 0 for p in probabilities):
            raise ValueError("All probabilities must be positive")

        self.models = models
        total = sum(probabilities)
        self.probabilities = [p / total for p in probabilities]
        self._default_model = models[0]
        self.selected_model: Optional[ChatOpenAI] = None
        self.langfuse_handler = None
        self.langfuse_client = None

        # Auto-initialize Langfuse tracing
        self.langfuse_handler = CallbackHandler()
          
        # Create client for immediate flushing and manual control
        self.langfuse_client = Langfuse(
            flush_at=1,          # Send every event immediately
            flush_interval=1,    # Check every 1 second
        )
        
        # Warn if Langfuse is not initialized
        if self.langfuse_handler is None or self.langfuse_client is None:
            logger.warning(
                "[MultiModelRouter] Langfuse tracing is disabled. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables to enable tracing."
            )
        else:
            logger.info("[MultiModelRouter] Langfuse tracing enabled (immediate mode)")
        
        self.selected_model: ChatOpenAI | None = None
        logger.info(f"[MultiModelRouter] Initialized with {len(models)} models")

    def _select_model(self) -> ChatOpenAI:
        """Select a model based on probabilities."""
        self.selected_model = random.choices(self.models, weights=self.probabilities)[0]
        return self.selected_model

    def _add_langfuse_to_config(self, config: Optional[RunnableConfig]) -> RunnableConfig:
        """Add Langfuse handler and metadata to config if tracing is enabled."""
        if not self.langfuse_handler:
            return config

        cfg = dict(config or {})
    
        # Add Langfuse handler to callbacks
        callbacks = cfg.setdefault("callbacks", [])
        if self.langfuse_handler not in callbacks:
            callbacks.append(self.langfuse_handler)

        # Add model metadata and tags
        if self.selected_model:
            metadata = dict(cfg.get("metadata", {}))
            model_data = str(self.selected_model)
            metadata["selected_model"] = model_data
            cfg["metadata"] = metadata

        return cfg


    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Synchronously select and invoke a model.

        Automatically traces the call to Langfuse if enabled.

        Args:
            input: Input to the model (messages or string)
            config: Optional runnable configuration
            **kwargs: Additional arguments passed to model

        Returns:
            Model response (BaseMessage)
        """
        model = self._select_model()
        config = self._add_langfuse_to_config(config)
        return model.invoke(input, config, **kwargs)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Asynchronously select and invoke a model.

        Automatically traces the call to Langfuse if enabled.

        Args:
            input: Input to the model (messages or string)
            config: Optional runnable configuration
            **kwargs: Additional arguments passed to model

        Returns:
            Model response (BaseMessage)
        """
        model = self._select_model()
        config = self._add_langfuse_to_config(config)
        return await model.ainvoke(input, config, **kwargs)

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        """Stream response from selected model.

        Automatically traces the call to Langfuse if enabled.

        Args:
            input: Input to the model
            config: Optional runnable configuration
            **kwargs: Additional arguments passed to model

        Yields:
            Response chunks (BaseMessage)
        """
        model = self._select_model()
        config = self._add_langfuse_to_config(config)
        return model.stream(input, config, **kwargs)

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        """Async stream response from selected model.

        Automatically traces the call to Langfuse if enabled.

        Args:
            input: Input to the model
            config: Optional runnable configuration
            **kwargs: Additional arguments passed to model

        Yields:
            Response chunks (BaseMessage)
        """
        model = self._select_model()
        config = self._add_langfuse_to_config(config)
        async for chunk in model.astream(input, config, **kwargs):
            yield chunk

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "MultiModelRouter":
        """Return a router with structured output - wraps all models.

        This creates a new router where each model is wrapped with structured output.
        Langfuse tracing is automatically preserved.

        Args:
            schema: Pydantic model or JSON schema
            **kwargs: Additional arguments for structured output

        Returns:
            New MultiModelRouter with structured output
        """
        wrapped_models = [
            model.with_structured_output(schema, **kwargs) for model in self.models
        ]
        new_router = MultiModelRouter(wrapped_models, self.probabilities)
        
        # Preserve Langfuse handler from parent router
        new_router.langfuse_handler = self.langfuse_handler
        new_router.langfuse_client = self.langfuse_client
        
        return new_router

    def flush_traces(self):
        """Flush pending Langfuse traces. Call at end of script/request."""
        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
                logger.info("[MultiModelRouter] Flushed Langfuse traces")
            except Exception as e:
                logger.warning(f"[MultiModelRouter] Failed to flush traces: {e}")


def create_chat_model(
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    top_p: float = 1.0,
    top_k: int | None = None,
    base_url: str | None = None,
    max_retries: int = 3,
    request_timeout: float = 60.0,
    **kwargs,
) -> ChatOpenAI:
    """Factory for creating LangChain ChatOpenAI models with sensible defaults.

    Args:
        model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
        api_key: OpenAI API key or compatible API key
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (provider-specific)
        base_url: Custom API endpoint (e.g., for OpenRouter, local models)
        max_retries: Number of retry attempts on failure
        request_timeout: Request timeout in seconds
        **kwargs: Additional model_kwargs passed to the API

    Returns:
        Configured ChatOpenAI instance

    Example:
        >>> model = create_chat_model(
        ...     model="Qwen3-235B-A22B-Thinking-2507",
        ...     api_key=os.getenv("OPENAI_API_KEY"),
        ...     temperature=0.6,
        ...     max_tokens=81920,
        ...     base_url="http://localhost:8777/v1"
        ... )
    """
    # Build model_kwargs with defaults
    # Note: top_k is intentionally not included as it's not supported by OpenAI's
    # structured output API (beta parse). For providers that support it, pass via kwargs.
    model_kwargs = {}
    model_kwargs.update(kwargs)  # Merge additional kwargs

    chat_model = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        model_kwargs=model_kwargs,
        base_url=base_url,
        max_retries=max_retries,
        request_timeout=request_timeout,
    )

    logger.info(
        f"[create_chat_model] Created ChatOpenAI model: {model} "
        f"(base_url={base_url}, max_retries={max_retries})"
    )

    return chat_model


def create_multi_model_router(
    model_configs: list[dict], probabilities: list[float], api_key: str
) -> MultiModelRouter:
    """Create multi-model router for probabilistic model selection.

    Convenience function that creates multiple ChatOpenAI models from configs
    and wraps them in a MultiModelRouter. Langfuse tracing is automatically
    enabled if environment variables are set.

    Args:
        model_configs: List of config dicts for create_chat_model()
        probabilities: Selection probabilities for each model
        api_key: API key to use for all models

    Returns:
        Configured MultiModelRouter with automatic Langfuse tracing

    Raises:
        ValueError: If configs/probabilities length mismatch

    Example:
        >>> # Set environment variables for Langfuse (optional):
        >>> # export LANGFUSE_PUBLIC_KEY="pk-lf-..."
        >>> # export LANGFUSE_SECRET_KEY="sk-lf-..."
        >>>
        >>> configs = [
        ...     {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000},
        ...     {"model": "gpt-3.5-turbo", "temperature": 0.8, "max_tokens": 500}
        ... ]
        >>> router = create_multi_model_router(configs, [0.8, 0.2], api_key)
        >>> # All calls are automatically traced if Langfuse is configured
        >>> response = await router.ainvoke("Hello!")
    """
    if len(model_configs) != len(probabilities):
        raise ValueError(
            f"model_configs and probabilities must have same length: "
            f"{len(model_configs)} != {len(probabilities)}"
        )

    models = [create_chat_model(api_key=api_key, **cfg) for cfg in model_configs]

    return MultiModelRouter(models, probabilities)
