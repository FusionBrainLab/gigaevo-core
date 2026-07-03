"""Positional-slot DAG-diff vocabulary for CARL reasoning chains.

Genomes are CARL wire JSON (ReasoningChain.to_dict plus the platform extras
task_description / is_output_step). A diff IS the full child chain as fixed
object properties slot_1..slot_max, each keeping a rendered parent step by id
or introducing a new one; unused trailing slots are null::

    {"reasoning": "add a verification step",
     "base_parent": "A",
     "slot_1": {"kind": "keep", "id": "a1"},
     "slot_2": {"kind": "new", "title": "Verify facts",
                "aim": "Cross-check the draft", "dependencies": ["slot_1"]},
     "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_2"]},
     "slot_4": null, ...}

Each slot's dependency enum offers only earlier slots (slot_1 has no
dependencies field at all), so self- and forward-references are
unrepresentable in the grammar; the one model-validator tripwire is
contiguous fill, which also makes references to null slots impossible.
Keep-ids form one global parent-prefixed enum, so keeps may reference ANY
rendered parent (crossover); base_parent is lineage attribution. A single
model keeps the grammar well below strict compilers' complexity cliff
(per-parent branching exceeds it at 8 slots). The schema is emitted in the
portable subset via gigaevo.llm.schema_compat.
Design: experiments/carl_dag_diff_dependency_rule_fix.md.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mmar_carl.chain import ReasoningChain
from mmar_carl.models.steps import LLMStepDescription
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
    model_validator,
)

from gigaevo.evolution.mutation.allowed_changes import AllowedChanges, DiffSchema
from gigaevo.exceptions import MutationError
from gigaevo.llm.schema_compat import portable_json_schema

CONTENT_FIELDS = tuple(
    name
    for name in (
        "title",
        "aim",
        "stage_action",
        "reasoning_questions",
        "example_reasoning",
    )
    if name in LLMStepDescription.model_fields
)

# aim/title must stay non-empty or LLMStepDescription's validator rejects the chain;
# min_length pushes that invariant into the schema so it is unrepresentable-to-break
_EDIT_FIELD_DEFS: dict[str, Any] = {
    name: (
        str | None,
        Field(default=None, min_length=1)
        if name in ("title", "aim")
        else Field(default=None),
    )
    for name in CONTENT_FIELDS
}
StepEdits = create_model(
    "StepEdits", __config__=ConfigDict(extra="forbid"), **_EDIT_FIELD_DEFS
)


class ChainDagDiffBase(BaseModel):
    """Static shape of the per-call diff model; _diff_model narrows the enums."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    base_parent: str


def _base_ids(ns: str, chain: ReasoningChain) -> list[str]:
    return [f"{ns.lower()}{i + 1}" for i in range(len(chain.steps))]


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


def _slot_models(k: int, all_ids: tuple[str, ...]) -> tuple[type, type]:
    """Keep/new models for position k; the dependency enum offers only earlier slots."""
    dep_field: dict[str, Any] = {}
    if k > 1:
        dep_ref: Any = Literal[tuple(f"slot_{j}" for j in range(1, k))]
        # maxItems=k-1 caps the array in the grammar itself: degenerate repetition
        # loops hit a forced ']' instead of running until token limits
        dep_field = {
            "dependencies": (
                list[dep_ref],  # type: ignore[valid-type]
                Field(default_factory=list, max_length=k - 1),
            )
        }
    keep = create_model(
        f"Keep{k}",
        __config__=ConfigDict(extra="forbid"),
        kind=(Literal["keep"], ...),
        id=(Literal[all_ids], ...),
        edits=(StepEdits, Field(default_factory=StepEdits)),
        **dep_field,
    )
    new = create_model(
        f"New{k}",
        __config__=ConfigDict(extra="forbid"),
        kind=(Literal["new"], ...),
        step_type=(Literal["llm"], "llm"),
        title=(str, Field(..., min_length=1)),
        aim=(str, Field(..., min_length=1)),
        stage_action=(str, ""),
        reasoning_questions=(str, ""),
        **dep_field,
    )
    return keep, new


class AllowedDagChanges(AllowedChanges):
    def __init__(self, *, min_steps: int = 1, max_steps: int = 8):
        if not 1 <= min_steps <= max_steps:
            raise ValueError(f"invalid step bounds: min={min_steps} max={max_steps}")
        self.min_steps = min_steps
        self.max_steps = max_steps

    def build_schema(self, parents: dict[str, str]) -> DiffSchema:
        chains, _ = self._parse(parents)
        adapter = TypeAdapter(self._diff_model(chains))
        schema = portable_json_schema(
            {**adapter.json_schema(), "title": "chain_dag_diff"}
        )
        return DiffSchema(json_schema=schema, validate=adapter.validate_python)

    def render_parents(self, parents: dict[str, str]) -> str:
        chains, extras = self._parse(parents)
        blocks = []
        for ns, chain in chains.items():
            lines = [f"=== Parent {ns} ==="]
            if extras[ns]:
                lines.append(f"chain task_description: {extras[ns]}")
            ids = _base_ids(ns, chain)
            # dependencies carry CARL step numbers, which need not match positions
            num_to_sid = {step.number: sid for sid, step in zip(ids, chain.steps)}
            for sid, step in zip(ids, chain.steps):
                deps = [num_to_sid.get(d, str(d)) for d in step.dependencies]
                lines.append(f"{sid} | deps={deps or '[]'} | title: {step.title}")
                for field in (
                    "aim",
                    "stage_action",
                    "reasoning_questions",
                    "example_reasoning",
                ):
                    value = getattr(step, field, "")
                    if value:
                        lines.append(f"    {field}: {value}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def apply(self, diff: Any, parents: dict[str, str]) -> str:
        chains, extras = self._parse(parents)
        try:
            wire = self._transcribe(diff, chains, extras)
            reparsed = ReasoningChain.from_dict(
                json.loads(json.dumps(wire)), use_typed_steps=True
            )
            if len(reparsed.steps) != len(wire["steps"]):
                raise ValueError(
                    f"round-trip step count {len(reparsed.steps)} != {len(wire['steps'])}"
                )
        except MutationError:
            raise
        except Exception as e:
            raise MutationError(f"diff_apply_assertion: {e}") from e
        return json.dumps(wire, ensure_ascii=False, indent=2)

    def describe(self) -> str:
        return (
            "POSITIONAL-SLOT CHAIN DIFF\n"
            f"- The child chain is slot_1..slot_{self.max_steps}, filled consecutively: "
            f"use {self.min_steps}..{self.max_steps} steps and set every unused trailing "
            "slot to null (no gaps). slot_1 is required.\n"
            "- base_parent = the parent whose lineage this child continues (pick the "
            "parent you take the most from).\n"
            "- Slot forms: keep = reuse a rendered step by id from ANY parent (ids are "
            "parent-prefixed), optionally overriding fields via 'edits'; new = a fresh "
            "LLM step with title/aim/stage_action/reasoning_questions. Any base step you "
            "omit is deleted; keeping the same id twice duplicates it.\n"
            "- Wiring is explicit: each slot's 'dependencies' lists the earlier slots "
            "whose outputs it consumes; the schema only offers earlier slots, and slot_1 "
            "takes none (it reads the raw task input). Dependencies always refer to slot "
            "positions in the NEW chain, never to base ids.\n"
            "- The last filled slot is the output step; its answer is what gets scored."
        )

    def _parse(
        self, parents: dict[str, str]
    ) -> tuple[dict[str, ReasoningChain], dict[str, str]]:
        chains: dict[str, ReasoningChain] = {}
        extras: dict[str, str] = {}
        if not parents:
            raise MutationError("carl_validation_error: no parents provided")
        for ns, code in parents.items():
            try:
                doc = json.loads(code)
                chains[ns] = ReasoningChain.from_dict(doc, use_typed_steps=True)
            except Exception as e:
                raise MutationError(f"carl_validation_error: parent {ns}: {e}") from e
            if not chains[ns].steps:
                raise MutationError(f"carl_validation_error: parent {ns} has no steps")
            for step in chains[ns].steps:
                if not isinstance(step, LLMStepDescription):
                    raise MutationError(
                        f"carl_validation_error: parent {ns} step {step.number} has "
                        f"unsupported type {type(step).__name__}; the diff grammar "
                        "covers LLM steps only"
                    )
            extras[ns] = doc.get("task_description", "")
        return chains, extras

    def _diff_model(self, chains: dict[str, ReasoningChain]) -> type[ChainDagDiffBase]:
        all_ids = tuple(
            sid for ns, chain in chains.items() for sid in _base_ids(ns, chain)
        )
        slot_fields: dict[str, Any] = {}
        for k in range(1, self.max_steps + 1):
            keep, new = _slot_models(k, all_ids)
            if k <= self.min_steps:
                slot_fields[f"slot_{k}"] = (keep | new, ...)
            else:
                slot_fields[f"slot_{k}"] = (keep | new | None, None)
        return create_model(
            "ChainDagDiff",
            __base__=ChainDagDiffBase,
            base_parent=(Literal[tuple(chains)], ...),
            **slot_fields,
            __validators__={
                # stub wants Callable but pydantic docs pass decorator proxies here
                "check_contiguous": model_validator(mode="after")(_slots_contiguous)  # type: ignore[dict-item]
            },
        )

    def _transcribe(
        self,
        diff: ChainDagDiffBase,
        chains: dict[str, ReasoningChain],
        extras: dict[str, str],
    ) -> dict:
        base = chains[diff.base_parent]
        by_id = {
            sid: step
            for ns, chain in chains.items()
            for sid, step in zip(_base_ids(ns, chain), chain.steps)
        }
        steps = []
        for k in range(1, self.max_steps + 1):
            slot = getattr(diff, f"slot_{k}")
            if slot is None:
                break
            refs = slot.dependencies if k > 1 else []
            deps = sorted({int(ref.removeprefix("slot_")) for ref in refs})
            if slot.kind == "keep":
                data = by_id[slot.id].model_dump(exclude={"number", "dependencies"})
                data |= slot.edits.model_dump(exclude_none=True)
            else:
                data = dict.fromkeys(CONTENT_FIELDS, "")
                data |= slot.model_dump(include=set(CONTENT_FIELDS))
            steps.append(LLMStepDescription(number=k, dependencies=deps, **data))
        # base.to_dict() scaffold preserves chain-level fields the diff never
        # touches (search_config, replan_policy, timeout, ...); only steps change
        wire = base.to_dict()
        wire["steps"] = ReasoningChain(steps=steps).to_dict()["steps"]
        # platform extras that carl's round-trip drops: derived, never LLM-emitted
        wire["task_description"] = extras[diff.base_parent]
        wire["steps"][-1]["is_output_step"] = True
        return wire
