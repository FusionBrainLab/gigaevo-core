from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbsGenerator(ABC):
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    @abstractmethod
    def generate_single(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate one response.
        Return shape: {"text": str, "json": dict | None, "response": dict}.
        Generation parameters such as temperature and max_tokens are configured
        by the concrete generator and do not need to be repeated here.
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str] | None = None,
        messages_list: list[list[dict[str, str]]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate a batch of responses.
        Return shape: [{"text": str, "json": dict | None, "response": dict}, ...].
        Generation parameters such as temperature and max_tokens are configured
        by the concrete generator and do not need to be repeated here.
        """
        pass
