"""ProgramRecordExtractor: a run's programs → eligible mutation records.

Owns the cross-sweep dedup bookkeeping (``_seen_ids``) and the running record
log so the live hook and the post-run hook never re-ingest the same program.
Pure with respect to the store and the LLM — it only reads program metadata.
"""

from __future__ import annotations

from gigaevo.memory.ideas_tracker.fitness import valid_fitness
from gigaevo.memory.ideas_tracker.models import ProgramRecord, program_to_record
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program


def record_note(record: ProgramRecord) -> str:
    """One-line mutation note from a record's normalised improvements."""
    note = "; ".join(
        imp.description.strip()
        for imp in record.improvements
        if imp.description.strip()
    )
    return note or "Unspecified change"


class ProgramRecordExtractor:
    """Filters a run's programs to eligible mutation records, once each.

    Skips: root programs (no parents), programs without a validated fitness
    (missing/non-positive ``is_valid``; missing, non-finite, or sentinel
    fitness), and already-seen ids. Tracks seen ids and the running record log so
    a timed-out or cancelled ingest can be rolled back via :meth:`forget`.
    """

    def __init__(
        self,
        *,
        task_description: str,
        fitness_key: str,
        metrics_context: MetricsContext | None = None,
    ) -> None:
        self._task_description = task_description
        self._fitness_key = fitness_key
        self._metrics_context = metrics_context
        self._all_records: list[ProgramRecord] = []
        self._seen_ids: set[str] = set()

    @property
    def seen_ids(self) -> set[str]:
        return self._seen_ids

    @property
    def all_records(self) -> list[ProgramRecord]:
        return self._all_records

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
            if valid_fitness(prog, self._fitness_key, self._metrics_context) is None:
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
        self._all_records.extend(records)
        self._seen_ids.update(p.id for p in eligible)
        return records

    def forget(self, ids: set[str]) -> None:
        """Roll back records whose ingest failed so a later sweep retries them."""
        if not ids:
            return
        self._seen_ids -= ids
        self._all_records = [r for r in self._all_records if r.id not in ids]
