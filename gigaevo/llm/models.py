from collections.abc import AsyncIterator, Iterator
import random
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger


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

    Attributes:
        models: List of ChatOpenAI models to choose from
        probabilities: Normalized probability distribution over models

    Example:
        >>> models = [
        ...     ChatOpenAI(model="gpt-4"),
        ...     ChatOpenAI(model="gpt-3.5-turbo")
        ... ]
        >>> router = MultiModelRouter(models, [0.8, 0.2])
        >>>
        >>> # Works exactly like ChatOpenAI
        >>> response = await router.ainvoke("Hello!")
        >>>
        >>> # Supports structured output
        >>> router_with_schema = router.with_structured_output(MySchema)
    """

    def __init__(self, models: list[ChatOpenAI], probabilities: list[float]):
        """Initialize router with models and selection probabilities.

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
        self.selected_model: ChatOpenAI | None = None
        logger.info(f"[MultiModelRouter] Initialized with {len(models)} models")

    def _select_model(self) -> ChatOpenAI:
        """Select a model based on probabilities."""
        self.selected_model = random.choices(self.models, weights=self.probabilities)[0]
        return self.selected_model

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Synchronously select and invoke a model.

        Args:
            input: Input to the model (messages or string)
            config: Optional runnable configuration
            **kwargs: Additional arguments passed to model

        Returns:
            Model response (BaseMessage)
        """
        model = self._select_model()
        return model.invoke(input, config, **kwargs)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Asynchronously select and invoke a model.

        Args:
            input: Input to the model (messages or string)
            config: Optional runnable configuration
            **kwargs: Additional arguments passed to model

        Returns:
            Model response (BaseMessage)
        """
        model = self._select_model()
        return await model.ainvoke(input, config, **kwargs)

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        """Stream response from selected model.

        Args:
            input: Input to the model
            config: Optional runnable configuration
            **kwargs: Additional arguments passed to model

        Yields:
            Response chunks (BaseMessage)
        """
        model = self._select_model()
        return model.stream(input, config, **kwargs)

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        """Async stream response from selected model.

        Args:
            input: Input to the model
            config: Optional runnable configuration
            **kwargs: Additional arguments passed to model

        Yields:
            Response chunks (BaseMessage)
        """
        model = self._select_model()
        async for chunk in model.astream(input, config, **kwargs):
            yield chunk

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "MultiModelRouter":
        """Return a router with structured output - wraps all models.

        This creates a new router where each model is wrapped with structured output.

        Args:
            schema: Pydantic model or JSON schema
            **kwargs: Additional arguments for structured output

        Returns:
            New MultiModelRouter with structured output
        """
        wrapped_models = [
            model.with_structured_output(schema, **kwargs) for model in self.models
        ]
        return MultiModelRouter(wrapped_models, self.probabilities)
