from __future__ import annotations

import json
import types
from typing import Any, Generic, TypeVar

from loguru import logger

from src.programs.core_types import StageIO
from src.programs.program import Program
from src.programs.stages.base import Stage
from src.programs.stages.common import AnyContainer, Box, StringContainer
from src.programs.stages.stage_registry import StageRegistry

K = TypeVar("K")
V = TypeVar("V")


class MergeDictInputs(StageIO, Generic[K, V]):
    first: Box[dict[K, V]]
    second: Box[dict[K, V]]


@StageRegistry.register(description="Merge two dictionaries")
class MergeDictStage(Stage, Generic[K, V]):
    """
    Merge two dictionaries ({**first, **second}); second overwrites conflicts.
    """

    InputsModel = MergeDictInputs[Any, Any]
    OutputModel = Box[dict[Any, Any]]
    cacheable: bool = True

    async def compute(self, program: Program) -> StageIO:
        first = self.params.first.data
        second = self.params.second.data

        merged = {**first, **second}
        logger.debug(
            "[{}] merged {} + {} -> {} keys",
            type(self).__name__,
            len(first),
            len(second),
            len(merged),
        )
        return self.__class__.OutputModel(data=merged)

    @classmethod
    def __class_getitem__(cls, params):  # supports MergeDictStage[float] & [K, V]
        """
        Returns a dynamic subclass with InputsModel/OutputModel specialized
        to the provided K,V types.
        """
        K_t, V_t = params
        return cls._make_specialized_class(K_t, V_t)

    @classmethod
    def _make_specialized_class(
        cls, K_t: Any, V_t: Any
    ) -> type["MergeDictStage[K, V]"]:
        """
        Build a dynamic subclass whose I/O models are typed as:
        InputsModel = MergeDictInputs[K_t, V_t]
        OutputModel = DictContainer[K_t, V_t]
        """

        def _exec_body(ns):
            ns["__doc__"] = {cls.__doc__}
            ns["InputsModel"] = MergeDictInputs[K_t, V_t]
            ns["OutputModel"] = Box[dict[K_t, V_t]]
            ns["cacheable"] = cls.cacheable
            ns["compute"] = cls.compute  # reuse implementation

        return types.new_class(cls.__name__, (cls,), exec_body=_exec_body)


@StageRegistry.register(description="Parse JSON string into Python value")
class ParseJSONStage(Stage):
    InputsModel = StringContainer
    OutputModel = AnyContainer
    cacheable: bool = True

    async def compute(self, program: Program) -> StageIO:
        s = self.params.data
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg} at pos {e.pos}") from e
        logger.debug(
            "[{}] parsed JSON -> {}", type(self).__name__, type(parsed).__name__
        )
        return AnyContainer(data=parsed)


@StageRegistry.register(description="Stringify Python value to JSON")
class StringifyJSONStage(Stage):
    InputsModel = AnyContainer
    OutputModel = StringContainer
    cacheable: bool = True

    def __init__(self, *, indent: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.indent = indent

    async def compute(self, program: Program) -> StageIO:
        obj = self.params.data
        try:
            s = json.dumps(obj, indent=self.indent)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert to JSON: {e}") from e
        logger.debug(
            "[{}] stringified {} -> {} chars",
            type(self).__name__,
            type(obj).__name__,
            len(s),
        )
        return StringContainer(data=s)
