"""Frozen-topology prompt diff for problems/chains RawChainSpec genomes.

The chain topology — step count, step types, dependency wiring, and tool
configs — is frozen to the parents' shared structure; only the prompt text of
LLM steps is mutable. The schema exposes exactly one required slot per LLM
step position, each a keep-by-id with optional field edits, so topology
mutation is unrepresentable by construction. Companion to the free-topology
AllowedToolChainChanges (problems/chains/tool_chain_diff.py).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import ConfigDict, Field, TypeAdapter, create_model

from gigaevo.evolution.mutation.allowed_changes import (
    AllowedChanges,
    DiffSchema,
    DiffStructuredOutputBase,
)
from gigaevo.exceptions import MutationError
from gigaevo.llm.schema_compat import portable_json_schema
from problems.chains.tool_chain_diff import StepEdits, _query_source_text
from problems.chains.types import LLMStep, RawChainSpec, ToolStep


class FrozenChainDiffBase(DiffStructuredOutputBase):
    """Static shape of the per-call diff model; _diff_model narrows the enums."""


def _topology_signature(spec: RawChainSpec) -> tuple:
    sig: list[tuple] = []
    for step in spec.steps:
        if isinstance(step, ToolStep):
            sig.append(
                (
                    "tool",
                    step.number,
                    tuple(step.dependencies),
                    step.step_config.tool_name,
                    tuple(sorted(step.step_config.input_mapping.items())),
                )
            )
        else:
            sig.append(("llm", step.number, tuple(step.dependencies)))
    return tuple(sig)


class FrozenTopologyChainChanges(AllowedChanges):
    def build_schema(self, parents: dict[str, str]) -> DiffSchema:
        specs = self._parse(parents)
        adapter = TypeAdapter(self._diff_model(specs))
        schema = portable_json_schema(
            {**adapter.json_schema(), "title": "frozen_chain_diff"}
        )
        return DiffSchema(json_schema=schema, validate=adapter.validate_python)

    def render_parents(self, parents: dict[str, str]) -> str:
        specs = self._parse(parents)
        blocks = []
        for ns, spec in specs.items():
            lines = [f"=== Parent {ns} ==="]
            if spec.system_prompt:
                lines.append(f"system_prompt: {spec.system_prompt}")
            ids = self._id_by_number(ns, spec)
            for step in spec.steps:
                if isinstance(step, ToolStep):
                    lines.append(
                        f"step {step.number} [frozen tool] "
                        f"tool={step.step_config.tool_name} "
                        f"query={_query_source_text(step)}"
                    )
                else:
                    lines.append(
                        f"step {step.number} = slot_{step.number} | "
                        f"id {ids[step.number]} | deps={step.dependencies} | "
                        f"title: {step.title}"
                    )
                    for field in ("aim", "stage_action", "reasoning_questions"):
                        value = getattr(step, field, "")
                        if value:
                            lines.append(f"    {field}: {value}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def apply(self, diff: Any, parents: dict[str, str]) -> str:
        specs = self._parse(parents)
        try:
            wire = self._transcribe(diff, specs)
            RawChainSpec.model_validate(wire)
        except MutationError:
            raise
        except Exception as e:
            raise MutationError(f"diff_apply_assertion: {e}") from e
        return json.dumps(wire, ensure_ascii=False, indent=2)

    def describe(self) -> str:
        return (
            "FROZEN-TOPOLOGY CHAIN DIFF (prompt edits only)\n"
            "- The chain's structure is FROZEN: step count, step order, tool "
            "steps, and all dependency wiring are fixed and cannot be changed. "
            "Tool steps never appear in the diff; they are copied verbatim.\n"
            "- The diff has one required slot per LLM step, named slot_<step "
            "number> to match the rendered parent (e.g. step 4 = slot_4).\n"
            "- Each slot picks the step to keep by its rendered id and may "
            "attach 'edits' overriding any of: title, aim, stage_action, "
            "reasoning_questions. Omitted edit fields keep the parent's text.\n"
            "- base_parent = the parent whose lineage this child continues.\n"
            "- Improve the child by rewriting the prompt text of at least one "
            "step; a diff with no edits reproduces the parent and wastes the "
            "attempt."
        )

    def _parse(self, parents: dict[str, str]) -> dict[str, RawChainSpec]:
        if not parents:
            raise MutationError("carl_validation_error: no parents provided")
        specs: dict[str, RawChainSpec] = {}
        for ns, code in parents.items():
            try:
                specs[ns] = RawChainSpec.model_validate_json(code)
            except Exception as e:
                raise MutationError(f"carl_validation_error: parent {ns}: {e}") from e
        signatures = {ns: _topology_signature(spec) for ns, spec in specs.items()}
        first_ns = next(iter(signatures))
        for ns, signature in signatures.items():
            if signature != signatures[first_ns]:
                raise MutationError(
                    f"frozen_topology_mismatch: parent {ns} topology differs "
                    f"from parent {first_ns}"
                )
        return specs

    @staticmethod
    def _id_by_number(ns: str, spec: RawChainSpec) -> dict[int, str]:
        out: dict[int, str] = {}
        for step in spec.steps:
            if isinstance(step, LLMStep):
                out[step.number] = f"{ns.lower()}{len(out) + 1}"
        return out

    def _diff_model(self, specs: dict[str, RawChainSpec]) -> type[FrozenChainDiffBase]:
        reference = next(iter(specs.values()))
        slot_fields: dict[str, Any] = {}
        for step in reference.steps:
            if not isinstance(step, LLMStep):
                continue
            ids = tuple(
                self._id_by_number(ns, spec)[step.number] for ns, spec in specs.items()
            )
            form = create_model(
                f"FrozenSlot{step.number}",
                __config__=ConfigDict(extra="forbid"),
                id=(Literal[ids], ...),
                edits=(StepEdits, Field(default_factory=StepEdits)),
            )
            slot_fields[f"slot_{step.number}"] = (form, ...)
        return create_model(
            "FrozenChainDiff",
            __base__=FrozenChainDiffBase,
            base_parent=(Literal[tuple(specs)], ...),
            **slot_fields,
        )

    def _transcribe(
        self, diff: FrozenChainDiffBase, specs: dict[str, RawChainSpec]
    ) -> dict:
        by_id = {
            self._id_by_number(ns, spec)[step.number]: step
            for ns, spec in specs.items()
            for step in spec.steps
            if isinstance(step, LLMStep)
        }
        base = specs[diff.base_parent]
        steps: list[dict] = []
        for step in base.steps:
            if isinstance(step, ToolStep):
                steps.append(step.model_dump(exclude={"frozen"}))
            else:
                slot = getattr(diff, f"slot_{step.number}")
                data = by_id[slot.id].model_dump(
                    exclude={"number", "dependencies", "step_type", "frozen"}
                )
                data |= slot.edits.model_dump(exclude_none=True)
                steps.append(
                    {
                        "number": step.number,
                        "step_type": "llm",
                        "dependencies": list(step.dependencies),
                        **data,
                    }
                )
        return {"system_prompt": base.system_prompt, "steps": steps}
