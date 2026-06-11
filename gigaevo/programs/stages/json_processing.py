from __future__ import annotations

import types
from typing import Any, Generic, TypeVar, cast

from loguru import logger

from gigaevo.programs.core_types import StageIO
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.stage_registry import StageRegistry

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

    async def compute(self, program: Program) -> StageIO:
        params = cast("MergeDictInputs[Any, Any]", self.params)
        first = params.first.data
        second = params.second.data

        merged = {**first, **second}
        overlapping_keys = len(first) + len(second) - len(merged)
        logger.debug(
            "[{}] merged {} + {} -> {} keys ({} overlapping)",
            type(self).__name__,
            len(first),
            len(second),
            len(merged),
            overlapping_keys,
        )
        return self.__class__.OutputModel(data=merged)

    @classmethod
    def __class_getitem__(cls, params):  # type: ignore[override]  # dynamic generic specialization
        """
        Returns a dynamic subclass with InputsModel/OutputModel specialized
        to the provided K,V types.
        """
        K_t, V_t = params
        return cls._make_specialized_class(K_t, V_t)

    @classmethod
    def _make_specialized_class(cls, K_t: Any, V_t: Any) -> type[MergeDictStage[K, V]]:
        def _exec_body(ns):
            ns["__doc__"] = {cls.__doc__}
            ns["InputsModel"] = MergeDictInputs[K_t, V_t]
            ns["OutputModel"] = Box[dict[K_t, V_t]]
            ns["cache_handler"] = cls.cache_handler
            ns["compute"] = cls.compute  # reuse implementation

        return types.new_class(cls.__name__, (cls,), exec_body=_exec_body)

    @classmethod
    def create_typed(cls, key_type: type, value_type: type):
        """Factory for Hydra configs: returns MergeDictStage[K, V] class.

        Args:
            key_type: Type for dictionary keys
            value_type: Type for dictionary values

        Returns:
            Specialized MergeDictStage class

        Usage in Hydra:
            _target_: gigaevo.programs.stages.json_processing.MergeDictStage.create_typed_factory
            _partial_: true
            key_type: ${get_object:builtins.str}
            value_type: ${get_object:builtins.float}
        """
        return cls[key_type, value_type]  # type: ignore[index]  # dynamic generic specialization via __class_getitem__


class StrFloatDictInputs(StageIO):
    """Inputs for merging two str->float dictionaries."""

    first: Box[dict[str, float]]
    second: Box[dict[str, float]]


@StageRegistry.register(
    description="Merge two str→float dicts (e.g., metrics); second overwrites first"
)
class MergeStrFloatDict(Stage):
    """
    Specialized stage for merging two dictionaries with string keys and float values.
    Common use case: merging metrics from different stages.

    The second dictionary overwrites any overlapping keys from the first.

    Example:
        first = {"accuracy": 0.9, "loss": 0.1}
        second = {"f1_score": 0.85, "loss": 0.08}
        result = {"accuracy": 0.9, "loss": 0.08, "f1_score": 0.85}
    """

    InputsModel = StrFloatDictInputs
    OutputModel = Box[dict[str, float]]

    async def compute(self, program: Program) -> StageIO:
        params = cast(StrFloatDictInputs, self.params)
        first = params.first.data
        second = params.second.data

        merged = {**first, **second}
        overlapping_keys = len(first) + len(second) - len(merged)

        logger.debug(
            "[MergeStrFloatDict] merged {} + {} -> {} keys ({} overlapping)",
            len(first),
            len(second),
            len(merged),
            overlapping_keys,
        )

        if overlapping_keys > 0:
            overlap_names = set(first.keys()) & set(second.keys())
            logger.info(
                "[MergeStrFloatDict] overlapping keys (using second): {}",
                sorted(overlap_names),
            )

        return Box[dict[str, float]](data=merged)
