"""Feature extraction for eval-time prediction.

``FeatureExtractor`` is a Protocol (structural subtyping) — users implement
it for domain-specific features without inheriting anything.
"""

from __future__ import annotations

import re
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


class ChainFeatureExtractor:
    """Feature extractor for chain-definition programs (HoVer, HotpotQA, etc.).

    Programs are Python functions returning a dict with ``system_prompt``
    and ``steps`` (a list of tool/LLM step configs).  Eval time is dominated
    by the number and verbosity of LLM steps — more tokens in prompts means
    longer LLM inference per step.

    Works for any chain-based problem (HoVer, HotpotQA, or custom chains).
    """

    _TOOL_STEP_RE = re.compile(r'"step_type"\s*:\s*"tool"')
    _LLM_STEP_RE = re.compile(r'"step_type"\s*:\s*"llm"')
    _DEP_RE = re.compile(r'"dependencies"\s*:\s*\[([^\]]*)\]')

    def extract(self, program: Program) -> dict[str, float]:
        code = program.code

        n_tool_steps = float(len(self._TOOL_STEP_RE.findall(code)))
        n_llm_steps = float(len(self._LLM_STEP_RE.findall(code)))
        n_total_steps = n_tool_steps + n_llm_steps

        # Total length of string literals >= 10 chars.
        # Captures system_prompt, stage_action, example_reasoning, aim, etc.
        # Longer strings = more LLM tokens = longer eval.
        total_string_content = sum(
            len(m.group(0)) for m in re.finditer(r'"[^"]{10,}"', code)
        )

        # Deep retrieval steps use k=10 vs k=7, take longer
        n_deep_retrieval = float(code.count('"retrieve_deep"'))

        # Total retrieval steps (both retrieve and retrieve_deep)
        n_retrievals = n_deep_retrieval + float(
            len(re.findall(r'"retrieve"(?!_)', code))
        )

        # Few-shot example blocks add significant token count
        n_examples = float(len(re.findall(r"Example \d+", code)))

        # Non-empty system_prompt adds per-step overhead
        has_system_prompt = float(
            '"system_prompt": ""' not in code and '"system_prompt"' in code
        )

        # Max dependency fan-in: steps with more dependencies receive
        # more context from prior steps -> more tokens -> longer inference
        max_deps = 0.0
        for m in self._DEP_RE.finditer(code):
            deps_str = m.group(1).strip()
            if deps_str:
                n_deps = len([d for d in deps_str.split(",") if d.strip()])
                max_deps = max(max_deps, float(n_deps))

        # DAG depth: longest path from any root to any leaf.
        # Steps are numbered 1..N in the code; deps_list is in order of appearance.
        dep_lists: list[list[int]] = []
        for m in self._DEP_RE.finditer(code):
            deps_str = m.group(1).strip()
            if deps_str:
                dep_lists.append(
                    [int(d.strip()) for d in deps_str.split(",") if d.strip()]
                )
            else:
                dep_lists.append([])

        dag_depth = 0.0
        if dep_lists:
            # depth[i] = longest path ending at step i (0-indexed)
            n = len(dep_lists)
            depth = [0] * n
            for i in range(n):
                for d in dep_lists[i]:
                    idx = d - 1  # 1-indexed → 0-indexed
                    if 0 <= idx < n:
                        depth[i] = max(depth[i], depth[idx] + 1)
            dag_depth = float(max(depth) + 1)  # +1: count nodes, not edges

        return {
            "code_length": float(len(code)),
            "n_tool_steps": n_tool_steps,
            "n_llm_steps": n_llm_steps,
            "n_total_steps": n_total_steps,
            "total_string_content": float(total_string_content),
            "n_deep_retrieval": n_deep_retrieval,
            "n_retrievals": n_retrievals,
            "n_examples": n_examples,
            "has_system_prompt": has_system_prompt,
            "max_dependency_fan_in": max_deps,
            "dag_depth": dag_depth,
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
