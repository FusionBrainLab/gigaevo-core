"""Feature extraction for eval-time prediction.

``FeatureExtractor`` is a Protocol (structural subtyping) — users implement
it for domain-specific features without inheriting anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


@runtime_checkable
class FeatureExtractor(Protocol):
    """Extract numeric features from a Program for eval-time prediction.

    Implementations MUST be pure (no side effects) and fast — they are
    called synchronously in the DAG launch hot path.
    """

    def extract(self, program: Program) -> dict[str, float]: ...


class CodeFeatureExtractor:
    """Default feature extractor: code-level structural features.

    Works for any problem domain.  No external dependencies.
    """

    def extract(self, program: Program) -> dict[str, float]:
        code = program.code
        return {
            "code_length": float(len(code)),
            "num_lines": float(code.count("\n") + 1),
            "num_function_defs": float(code.count("def ")),
            "num_loop_constructs": float(code.count("for ") + code.count("while ")),
        }


class CompositeFeatureExtractor:
    """Compose multiple extractors.  Key conflicts: last writer wins."""

    def __init__(self, extractors: list[FeatureExtractor]) -> None:
        if not extractors:
            raise ValueError("At least one extractor required")
        self._extractors = extractors

    def extract(self, program: Program) -> dict[str, float]:
        features: dict[str, float] = {}
        for ext in self._extractors:
            features.update(ext.extract(program))
        return features
