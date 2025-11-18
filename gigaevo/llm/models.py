from collections.abc import AsyncIterator, Iterator
import os
import random
from typing import Any, Optional

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger

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
        langfuse_handler: Langfuse callback handler for tracing (auto-created if env vars set)

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

        # Auto-initialize Langfuse tracing if environment variables are set
        self.langfuse_handler: Optional[CallbackHandler] = None
        if self._is_langfuse_available():
            try:
                # Configure for immediate flushing: flush after 1 event, every 1 second
                self.langfuse_handler = CallbackHandler(
                    flush_at=1,
                    flush_interval=1
                )
                logger.info("[MultiModelRouter] Langfuse tracing enabled (immediate mode)")
            except Exception as e:
                logger.warning(
                    f"[MultiModelRouter] Failed to initialize Langfuse tracing: {e}. "
                    "Tracing will be disabled."
                )
                self.langfuse_handler = None
        else:
            logger.debug(
                "[MultiModelRouter] Langfuse tracing is disabled. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables to enable tracing."
            )
        
        logger.info(f"[MultiModelRouter] Initialized with {len(models)} models")

    @staticmethod
    def _is_langfuse_available() -> bool:
        """Check if Langfuse environment variables are set."""
        return bool(
            os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
        )

    def _select_model(self) -> ChatOpenAI:
        """Select a model based on probabilities."""
        self.selected_model = random.choices(self.models, weights=self.probabilities)[0]
        return self.selected_model

    def _add_langfuse_to_config(
        self, config: Optional[RunnableConfig]
    ) -> RunnableConfig:
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
        
        return new_router
