"""Lazy exports for stage modules and common stage symbols.

Avoid importing every stage module at package import time. Some stage modules
pull in LLM agent factories, which can create circular imports during mutation
context initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBMODULES = {
    "base",
    "collector",
    "complexity",
    "formatter",
    "insights",
    "insights_lineage",
    "json_processing",
    "llm_score",
    "metrics",
    "optimization",
    "python_executors",
    "runtime_metrics",
    "validation",
}

_EXPORTS: dict[str, tuple[str, str]] = {
    "Stage": ("gigaevo.programs.stages.base", "Stage"),
    "RelatedCollectorBase": (
        "gigaevo.programs.stages.collector",
        "RelatedCollectorBase",
    ),
    "ComputeComplexityStage": (
        "gigaevo.programs.stages.complexity",
        "ComputeComplexityStage",
    ),
    "GetCodeLengthStage": (
        "gigaevo.programs.stages.complexity",
        "GetCodeLengthStage",
    ),
    "FormatterStage": ("gigaevo.programs.stages.formatter", "FormatterStage"),
    "InsightsStage": ("gigaevo.programs.stages.insights", "InsightsStage"),
    "LineagesFromAncestors": (
        "gigaevo.programs.stages.insights_lineage",
        "LineagesFromAncestors",
    ),
    "LineageStage": ("gigaevo.programs.stages.insights_lineage", "LineageStage"),
    "LineagesToDescendants": (
        "gigaevo.programs.stages.insights_lineage",
        "LineagesToDescendants",
    ),
    "MergeDictStage": (
        "gigaevo.programs.stages.json_processing",
        "MergeDictStage",
    ),
    "ParseJSONStage": (
        "gigaevo.programs.stages.json_processing",
        "ParseJSONStage",
    ),
    "StringifyJSONStage": (
        "gigaevo.programs.stages.json_processing",
        "StringifyJSONStage",
    ),
    "LLMScoreStage": ("gigaevo.programs.stages.llm_score", "LLMScoreStage"),
    "EnsureMetricsStage": (
        "gigaevo.programs.stages.metrics",
        "EnsureMetricsStage",
    ),
    "NormalizeMetricsStage": (
        "gigaevo.programs.stages.metrics",
        "NormalizeMetricsStage",
    ),
    "CMANumericalOptimizationStage": (
        "gigaevo.programs.stages.optimization",
        "CMANumericalOptimizationStage",
    ),
    "CMAOptimizationOutput": (
        "gigaevo.programs.stages.optimization",
        "CMAOptimizationOutput",
    ),
    "OptunaOptimizationOutput": (
        "gigaevo.programs.stages.optimization",
        "OptunaOptimizationOutput",
    ),
    "OptunaOptimizationStage": (
        "gigaevo.programs.stages.optimization",
        "OptunaOptimizationStage",
    ),
    "CallFileFunction": (
        "gigaevo.programs.stages.python_executors",
        "CallFileFunction",
    ),
    "CallProgramFunction": (
        "gigaevo.programs.stages.python_executors",
        "CallProgramFunction",
    ),
    "CallProgramFunctionWithFixedArgs": (
        "gigaevo.programs.stages.python_executors",
        "CallProgramFunctionWithFixedArgs",
    ),
    "CallValidatorFunction": (
        "gigaevo.programs.stages.python_executors",
        "CallValidatorFunction",
    ),
    "execution": ("gigaevo.programs.stages.python_executors", "execution"),
    "RuntimeFitnessStage": (
        "gigaevo.programs.stages.runtime_metrics",
        "RuntimeFitnessStage",
    ),
    "ValidateCodeStage": (
        "gigaevo.programs.stages.validation",
        "ValidateCodeStage",
    ),
}

__all__ = sorted(_SUBMODULES | set(_EXPORTS))


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
