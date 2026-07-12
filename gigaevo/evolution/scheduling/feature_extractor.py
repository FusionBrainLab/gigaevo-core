"""Feature extraction for eval-time prediction.

``FeatureExtractor`` is a Protocol (structural subtyping) — users implement
it for domain-specific features without inheriting anything.
"""

from __future__ import annotations

import json
import re
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import ValidationError

if TYPE_CHECKING:
    from gigaevo.programs.program import Program

_CHAIN_TYPES: ModuleType | None | bool = False


def _chain_spec_types() -> ModuleType | None:
    """CARL parse-layer models (problems.chains.types), or None when absent.

    Lazy: mmar-carl is an optional 3.12-gated extra, so the import must not
    run at module load. None → semantic features degrade to zeros; a chains
    run without the extra fails loudly at evaluation anyway.
    """
    global _CHAIN_TYPES
    if _CHAIN_TYPES is False:
        try:
            from problems.chains import types as chain_types
        except ImportError:
            _CHAIN_TYPES = None
        else:
            _CHAIN_TYPES = chain_types
    return _CHAIN_TYPES  # type: ignore[return-value]


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

    # Passages returned per retrieval tool call. No class-spec home exists:
    # the k's are call-site literals in each problem's validate.py
    # (make_retrieve_fn(..., k=7) / retrieve_deep k=10).
    _PASSAGES_PER_TOOL = {"retrieve": 7.0, "retrieve_deep": 10.0}
    # Seed-program convention for "field intentionally blank" (initial_programs
    # baselines) — prompt content, not schema, so not in problems.chains.types.
    _NONE_PLACEHOLDER = "<none>"

    def _semantic_features(self, code: str) -> dict[str, float]:
        """Strategy features from a json_document chain spec.

        The spec is parsed with the CARL parse-layer models
        (``problems.chains.types.RawChainSpec``) and guidance fields come from
        ``STRUCTURED_FIELDS`` there — the extractor sees exactly the schema
        ``validate_chain_spec`` enforces.

        - ``hop_depth``: max over tool steps of 1 + #tool steps in the
          transitive upstream dependency closure (retrieve→reason→retrieve
          chain length; 0 without tool steps).
        - ``passages_fetched``: total evidence budget, k-weighted per
          retrieval tool call.
        - ``instr_chars``: characters of STRUCTURED_FIELDS (``<none>``
          placeholders excluded) + system_prompt.

        Non-JSON or schema-invalid code yields zeros — pair the chains_bd3d
        behavior space with program_format=json_document only.
        """
        zeros = {"hop_depth": 0.0, "passages_fetched": 0.0, "instr_chars": 0.0}
        try:
            raw = json.loads(code)
        except ValueError:
            return zeros
        if not isinstance(raw, dict):
            return zeros
        chain_types = _chain_spec_types()
        if chain_types is None:
            return zeros
        try:
            spec = chain_types.RawChainSpec.model_validate(raw)
        except ValidationError:
            return zeros

        deps_by_num = {s.number: s.dependencies for s in spec.steps}
        tool_steps = [s for s in spec.steps if s.step_type == "tool"]
        tool_numbers = {s.number for s in tool_steps}

        hop_depth = 0.0
        passages = 0.0
        for step in tool_steps:
            passages += self._PASSAGES_PER_TOOL.get(step.step_config.tool_name, 0.0)
            upstream: set[int] = set()
            stack = list(step.dependencies)
            while stack:
                p = stack.pop()
                if p in upstream:
                    continue
                upstream.add(p)
                stack.extend(deps_by_num.get(p, []))
            hop_depth = max(hop_depth, float(1 + len(upstream & tool_numbers)))

        instr_chars = len(spec.system_prompt)
        for step in spec.steps:
            for field in chain_types.STRUCTURED_FIELDS:
                value = getattr(step, field, None)
                if isinstance(value, str) and value and value != self._NONE_PLACEHOLDER:
                    instr_chars += len(value)

        return {
            "hop_depth": hop_depth,
            "passages_fetched": passages,
            "instr_chars": float(instr_chars),
        }

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
            **self._semantic_features(code),
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
