"""Run programs → eligible mutation records for the librarian write path.

Validates the mutator's typed ``mutation_output`` changes (each already a
structured ``{description, explanation}`` item) into ``Improvement``s, converts
eligible programs (parented, strictly-valid fitness, unseen) into
``ProgramRecord``s, and owns the cross-sweep dedup bookkeeping so the live hook
and the post-run hook never re-ingest the same program. Pure with respect to the
store and the LLM — it only reads program metadata.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gigaevo.evolution.mutation.constants import MUTATION_OUTPUT_METADATA_KEY
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program


class Improvement(BaseModel):
    """A single normalised mutation change: what changed, and the stated why."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(description="What changed, in one sentence.")
    explanation: str = Field(
        default="", description="The stated motivation for the change."
    )


class ProgramRecord(BaseModel):
    """Metadata extracted from a Program for the librarian write path.

    Created from a raw Program object; carries only the fields the librarian
    needs (no stage results, no raw execution data).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Program id.")
    fitness: float = Field(
        description="Fitness value under the configured fitness key."
    )
    generation: int = Field(description="Generation the program was created in.")
    parents: list[str] = Field(default_factory=list, description="Parent program ids.")
    improvements: list[Improvement] = Field(
        default_factory=list,
        description="Typed mutation changes that produced this program.",
    )
    strategy: str = Field(
        default="", description="Mutation archetype reported in the mutation output."
    )
    task_description: str = Field(
        default="", description="Task description active for this run."
    )
    task_description_summary: str = Field(
        default="", description="Condensed form of the task description."
    )
    code: str = Field(default="", description="Program source code.")
    base_parent_id: str = Field(
        default="",
        description="Id of the base parent the mutator anchored the child to.",
    )
    parent_code: str = Field(
        default="",
        description="Source code of the base parent, when available.",
    )


class MutationOutput(BaseModel):
    """Validated `mutation_output` metadata blob attached to a mutated Program."""

    model_config = ConfigDict(extra="ignore")

    changes: list[Improvement] = Field(
        default_factory=list,
        description="Typed changes emitted by the mutator; each already a {description, explanation} item.",
    )
    archetype: str = Field(
        default="", description="Mutation archetype label; empty when absent."
    )
    base_parent: int = Field(
        default=1,
        description="1-based index of the parent the mutator anchored the child to.",
    )

    @field_validator("changes", mode="before")
    @classmethod
    def coerce_none_changes(cls, value: Any) -> Any:
        return value or []

    @field_validator("archetype", mode="before")
    @classmethod
    def coerce_none_archetype(cls, value: Any) -> Any:
        return value or ""

    @field_validator("base_parent", mode="before")
    @classmethod
    def coerce_none_base_parent(cls, value: Any) -> Any:
        return value or 1


def program_to_record(
    program: Program,
    task_description: str,
    task_description_summary: str,
    fitness_key: str = "fitness",
    parent_codes: dict[str, str] | None = None,
) -> ProgramRecord:
    """Convert a Program to a ProgramRecord for the librarian write path.

    The program must carry a metric under ``fitness_key`` (callers filter
    eligibility first); a missing key raises rather than minting a default.
    """
    raw_output = program.metadata.get(MUTATION_OUTPUT_METADATA_KEY)
    mutation_output = (
        MutationOutput.model_validate(raw_output)
        if isinstance(raw_output, dict)
        else MutationOutput()
    )
    parents = list(program.lineage.parents)
    # The librarian diffs the child against the parent the mutator anchored it to
    # (1-based ``base_parent``), not whichever parent the selector happened to list
    # first; in ≥2-parent rewrite mode the base may be a later donor's sibling. Out
    # of range falls back to the first parent, matching freeze_base_parent_snapshot.
    base_index = mutation_output.base_parent - 1
    if base_index < 0 or base_index >= len(parents):
        base_index = 0
    base_parent_id = parents[base_index] if parents else ""
    parent_code = parent_codes.get(base_parent_id, "") if parent_codes else ""
    return ProgramRecord(
        id=program.id,
        fitness=program.metrics[fitness_key],
        generation=program.lineage.generation,
        parents=parents,
        improvements=list(mutation_output.changes),
        strategy=mutation_output.archetype,
        task_description=task_description,
        task_description_summary=task_description_summary,
        code=program.code,
        base_parent_id=base_parent_id,
        parent_code=parent_code,
    )


def record_note(record: ProgramRecord) -> str:
    """One-line mutation note from a record's typed improvements."""
    note = "; ".join(
        imp.description.strip()
        for imp in record.improvements
        if imp.description.strip()
    )
    return note or "Unspecified change"


class ProgramRecordExtractor:
    """Filters a run's programs to eligible mutation records, once each.

    Skips: root programs (no parents), programs without a strictly-valid
    fitness (missing/non-positive ``is_valid``; missing, non-finite, or
    sentinel fitness), and already-seen ids. Tracks seen ids so a timed-out or
    cancelled ingest can be rolled back via :meth:`forget`.
    """

    def __init__(
        self,
        *,
        task_description: str,
        fitness_key: str,
        metrics_context: MetricsContext,
    ) -> None:
        self._task_description = task_description
        self._fitness_key = fitness_key
        self._metrics_context = metrics_context
        self._seen_ids: set[str] = set()

    @property
    def seen_ids(self) -> set[str]:
        return self._seen_ids

    def extract(
        self,
        programs: list[Program],
        *,
        task_description_summary: str,
        posterior_programs: list[Program] | None = None,
    ) -> list[ProgramRecord]:
        """Eligible programs converted to records, marking each seen.

        Parent code resolves from ``posterior_programs`` (the full pool) when
        provided: live sweeps cap ``programs`` to the newest window, and mutation
        parents are usually older archive elites outside it — without the full
        pool the librarian loses the parent code its diff reconciliation needs
        and silently degrades to ungrounded authoring mid-run.
        """
        eligible: list[Program] = []
        for prog in programs:
            if not prog.lineage.parents:
                continue
            if (
                self._metrics_context.strict_fitness(prog.metrics, self._fitness_key)
                is None
            ):
                continue
            if prog.id in self._seen_ids:
                continue
            eligible.append(prog)

        code_pool = programs if posterior_programs is None else posterior_programs
        parent_codes: dict[str, str] = {p.id: p.code for p in code_pool if p.code}
        records = [
            program_to_record(
                p,
                self._task_description,
                task_description_summary,
                self._fitness_key,
                parent_codes=parent_codes,
            )
            for p in eligible
        ]
        self._seen_ids.update(p.id for p in eligible)
        return records

    def forget(self, ids: set[str]) -> None:
        """Roll back records whose ingest failed so a later sweep retries them."""
        if not ids:
            return
        self._seen_ids -= ids
