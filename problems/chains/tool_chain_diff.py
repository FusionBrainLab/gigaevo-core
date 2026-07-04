"""Tool-aware positional-slot DAG-diff for problems/chains RawChainSpec genomes.

Extends the LLM-only positional-slot pattern (gigaevo/chains/dag_changes.py) to
chains whose genome is RawChainSpec — {system_prompt, steps:[LLMStep | ToolStep]}
— so retrieval steps are first-class in the diff. Each slot is keep (reuse an
LLM step by id + edits), new_llm, or new_tool (a tool_name from available_tools
+ one query source). Tool query wiring is emitted as absolute $history[N] from
the renumbered source position, so a reordering diff can never point a retrieve
at the wrong upstream output. Design: experiments/carl_tool_chain_diff_design.md.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
    model_validator,
)

from gigaevo.evolution.mutation.allowed_changes import (
    AllowedChanges,
    DiffSchema,
    DiffStructuredOutputBase,
)
from gigaevo.exceptions import MutationError
from gigaevo.llm.schema_compat import portable_json_schema
from problems.chains.types import LLMStep, RawChainSpec, ToolStep

_LLM_EDIT_FIELDS = ("title", "aim", "stage_action", "reasoning_questions")

# aim/stage_action/title must stay non-empty or RawChainSpec's LLMStep rejects
# the chain; min_length pushes that invariant into the schema so it is
# unrepresentable-to-break
_EDIT_FIELD_DEFS: dict[str, Any] = {
    name: (
        str | None,
        Field(default=None, min_length=1)
        if name in ("title", "aim", "stage_action")
        else Field(default=None),
    )
    for name in _LLM_EDIT_FIELDS
}
StepEdits = create_model(
    "ToolChainStepEdits", __config__=ConfigDict(extra="forbid"), **_EDIT_FIELD_DEFS
)


class ToolChainDiffBase(DiffStructuredOutputBase):
    """Static shape of the per-call diff model; _diff_model narrows the enums."""


def _slots_contiguous(diff: BaseModel) -> BaseModel:
    empty = None
    for name in type(diff).model_fields:
        if not name.startswith("slot_"):
            continue
        if getattr(diff, name) is None:
            empty = name
        elif empty is not None:
            raise ValueError(
                f"{name} is filled after empty {empty}; fill slots consecutively from slot_1"
            )
    return diff


def _query_source_text(step: ToolStep) -> str:
    ref = step.step_config.input_mapping.get("query", "")
    if ref == "$outer_context":
        return "outer_context"
    if ref == "$history[-1]":
        return "output of the preceding step"
    if ref.startswith("$history[") and ref.endswith("]"):
        idx = ref[len("$history[") : -1]
        if idx.isdigit():
            return f"output of step {int(idx) + 1}"
    return ref


class AllowedToolChainChanges(AllowedChanges):
    def __init__(
        self,
        *,
        min_steps: int = 1,
        max_steps: int,
        available_tools: list[str],
        outer_context_token: str = "$outer_context",
    ):
        if not 1 <= min_steps <= max_steps:
            raise ValueError(f"invalid step bounds: min={min_steps} max={max_steps}")
        if not available_tools:
            raise ValueError("available_tools must be non-empty")
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.available_tools = list(available_tools)
        self.outer_context_token = outer_context_token

    def build_schema(self, parents: dict[str, str]) -> DiffSchema:
        specs = self._parse(parents)
        adapter = TypeAdapter(self._diff_model(specs))
        schema = portable_json_schema(
            {**adapter.json_schema(), "title": "tool_chain_diff"}
        )

        def validate(payload: Any) -> Any:
            try:
                return adapter.validate_python(payload)
            except ValidationError:
                repaired = self._compact_payload(payload)
                if repaired is None:
                    raise
                validated = adapter.validate_python(repaired)
                logger.warning(
                    "tool_chain_diff: repaired non-contiguous slot payload "
                    "(filled slots {} compacted)",
                    sorted(
                        int(k.removeprefix("slot_"))
                        for k, v in payload.items()
                        if k.startswith("slot_") and v is not None
                    ),
                )
                return validated

        return DiffSchema(json_schema=schema, validate=validate)

    @staticmethod
    def _compact_payload(payload: Any) -> dict | None:
        """Slide filled slots left to close a gap the model left, remapping every
        slot reference, so a well-formed-but-non-contiguous diff becomes valid.
        Order is preserved and edges are backward, so the result transcribes
        identically to a contiguous emission. Returns None (keep the original
        error) when there is no gap or a surviving slot references the hole."""
        if not isinstance(payload, dict):
            return None
        names = sorted(
            (k for k in payload if k.startswith("slot_")),
            key=lambda s: int(s.removeprefix("slot_")),
        )
        filled = [
            (int(k.removeprefix("slot_")), payload[k])
            for k in names
            if payload[k] is not None
        ]
        old = [i for i, _ in filled]
        if old == list(range(1, len(old) + 1)):
            return None
        remap = {o: n for n, (o, _) in enumerate(filled, start=1)}
        rebuilt: dict[str, Any] = {}
        try:
            for n, (_, body) in enumerate(filled, start=1):
                if not isinstance(body, dict):
                    return None
                body = dict(body)
                deps = body.get("dependencies")
                if isinstance(deps, list):
                    new_deps = []
                    for r in deps:
                        j = int(str(r).removeprefix("slot_"))
                        if j not in remap:
                            return None
                        new_deps.append(f"slot_{remap[j]}")
                    body["dependencies"] = new_deps
                qsrc = body.get("query_source")
                if isinstance(qsrc, str) and qsrc.startswith("slot_"):
                    j = int(qsrc.removeprefix("slot_"))
                    if j not in remap:
                        return None
                    body["query_source"] = f"slot_{remap[j]}"
                rebuilt[f"slot_{n}"] = body
        except (ValueError, TypeError):
            return None
        out = {k: v for k, v in payload.items() if not k.startswith("slot_")}
        out.update(rebuilt)
        return out

    def render_parents(self, parents: dict[str, str]) -> str:
        specs = self._parse(parents)
        blocks = []
        for ns, spec in specs.items():
            lines = [f"=== Parent {ns} ==="]
            if spec.system_prompt:
                lines.append(f"system_prompt: {spec.system_prompt}")
            sid_by_number = {
                step.number: sid for sid, step in self._llm_id_map(ns, spec).items()
            }
            for step in spec.steps:
                deps = [sid_by_number.get(d, str(d)) for d in step.dependencies]
                if isinstance(step, LLMStep):
                    sid = sid_by_number[step.number]
                    lines.append(f"{sid} | deps={deps or '[]'} | title: {step.title}")
                    for field in ("aim", "stage_action", "reasoning_questions"):
                        value = getattr(step, field, "")
                        if value:
                            lines.append(f"    {field}: {value}")
                else:
                    lines.append(
                        f"[tool step, reproduce via new_tool] "
                        f"tool={step.step_config.tool_name} "
                        f"query={_query_source_text(step)}"
                    )
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
        tools = ", ".join(self.available_tools)
        return (
            "POSITIONAL-SLOT CHAIN DIFF (LLM + tool steps)\n"
            f"- The child chain is slot_1..slot_{self.max_steps}, filled consecutively: "
            f"use {self.min_steps}..{self.max_steps} steps and set every unused trailing "
            "slot to null (no gaps). slot_1 is required.\n"
            "- base_parent = the parent whose lineage this child continues.\n"
            "- Slot forms: keep = reuse a rendered LLM step by id (parent-prefixed) with "
            "optional 'edits'; new_llm = a fresh LLM step with title/aim/stage_action/"
            "reasoning_questions; new_tool = a retrieval step choosing tool_name from "
            f"[{tools}] and one query_source.\n"
            "- Tool steps are re-expressed via new_tool (not kept by id): a tool step is "
            "fully defined by its tool_name and query_source, so nothing is lost.\n"
            "- query_source is 'outer_context' (the raw task input) or an EARLIER slot "
            "whose output becomes the retrieval query; the schema offers only earlier "
            "slots, so wiring is always backward and reorder-safe.\n"
            "- For LLM steps, 'dependencies' lists the earlier slots whose outputs feed "
            "the step; slot_1 takes none. Any base step you omit is deleted."
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
        return specs

    def _llm_id_map(self, ns: str, spec: RawChainSpec) -> dict[str, LLMStep]:
        out: dict[str, LLMStep] = {}
        for step in spec.steps:
            if isinstance(step, LLMStep):
                out[f"{ns.lower()}{len(out) + 1}"] = step
        return out

    def _slot_forms(self, k: int, all_ids: tuple[str, ...]) -> list[type]:
        dep_field: dict[str, Any] = {}
        qsrc = ["outer_context"]
        if k > 1:
            dep_ref: Any = Literal[tuple(f"slot_{j}" for j in range(1, k))]
            dep_field = {
                "dependencies": (
                    list[dep_ref],  # type: ignore[valid-type]
                    Field(default_factory=list, max_length=k - 1),
                )
            }
            qsrc += [f"slot_{j}" for j in range(1, k)]
        forms: list[type] = []
        if all_ids:
            forms.append(
                create_model(
                    f"ToolKeep{k}",
                    __config__=ConfigDict(extra="forbid"),
                    kind=(Literal["keep"], ...),
                    id=(Literal[all_ids], ...),
                    edits=(StepEdits, Field(default_factory=StepEdits)),
                    **dep_field,
                )
            )
        forms.append(
            create_model(
                f"ToolNewLlm{k}",
                __config__=ConfigDict(extra="forbid"),
                kind=(Literal["new_llm"], ...),
                title=(str, Field(..., min_length=1)),
                aim=(str, Field(..., min_length=1)),
                stage_action=(str, Field(..., min_length=1)),
                reasoning_questions=(str, ""),
                **dep_field,
            )
        )
        forms.append(
            create_model(
                f"ToolNewTool{k}",
                __config__=ConfigDict(extra="forbid"),
                kind=(Literal["new_tool"], ...),
                tool_name=(Literal[tuple(self.available_tools)], ...),
                query_source=(Literal[tuple(qsrc)], ...),
            )
        )
        return forms

    def _diff_model(self, specs: dict[str, RawChainSpec]) -> type[ToolChainDiffBase]:
        all_ids = tuple(
            sid for ns, spec in specs.items() for sid in self._llm_id_map(ns, spec)
        )
        slot_fields: dict[str, Any] = {}
        for k in range(1, self.max_steps + 1):
            forms = self._slot_forms(k, all_ids)
            union: Any = forms[0]
            for form in forms[1:]:
                union = union | form
            if k <= self.min_steps:
                slot_fields[f"slot_{k}"] = (union, ...)
            else:
                slot_fields[f"slot_{k}"] = (union | None, None)
        return create_model(
            "ToolChainDiff",
            __base__=ToolChainDiffBase,
            base_parent=(Literal[tuple(specs)], ...),
            **slot_fields,
            __validators__={
                # stub wants Callable but pydantic docs pass decorator proxies here
                "check_contiguous": model_validator(mode="after")(_slots_contiguous)  # type: ignore[dict-item]
            },
        )

    @staticmethod
    def _slot_deps(slot: Any, k: int) -> list[int]:
        refs = getattr(slot, "dependencies", []) if k > 1 else []
        return sorted({int(r.removeprefix("slot_")) for r in refs})

    def _transcribe(
        self, diff: ToolChainDiffBase, specs: dict[str, RawChainSpec]
    ) -> dict:
        by_id = {
            sid: step
            for ns, spec in specs.items()
            for sid, step in self._llm_id_map(ns, spec).items()
        }
        base = specs[diff.base_parent]
        steps: list[dict] = []
        for k in range(1, self.max_steps + 1):
            slot = getattr(diff, f"slot_{k}")
            if slot is None:
                break
            if slot.kind == "keep":
                data = by_id[slot.id].model_dump(
                    exclude={"number", "dependencies", "step_type", "frozen"}
                )
                data |= slot.edits.model_dump(exclude_none=True)
                steps.append(
                    {
                        "number": k,
                        "step_type": "llm",
                        "dependencies": self._slot_deps(slot, k),
                        **data,
                    }
                )
            elif slot.kind == "new_llm":
                steps.append(
                    {
                        "number": k,
                        "step_type": "llm",
                        "dependencies": self._slot_deps(slot, k),
                        "title": slot.title,
                        "aim": slot.aim,
                        "stage_action": slot.stage_action,
                        "reasoning_questions": slot.reasoning_questions,
                    }
                )
            else:
                if slot.query_source == "outer_context":
                    ref, deps = self.outer_context_token, []
                else:
                    j = int(slot.query_source.removeprefix("slot_"))
                    ref, deps = f"$history[{j - 1}]", [j]
                steps.append(
                    {
                        "number": k,
                        "step_type": "tool",
                        "dependencies": deps,
                        "title": f"{slot.tool_name} (step {k})",
                        "step_config": {
                            "tool_name": slot.tool_name,
                            "input_mapping": {"query": ref},
                        },
                    }
                )
        return {"system_prompt": base.system_prompt, "steps": steps}
