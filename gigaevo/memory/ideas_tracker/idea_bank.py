"""
IdeaBank: stores and manages Idea objects for an IdeaTracker session.

Also contains usage-payload helpers (build, merge) previously in utils/helpers.py.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    ClassificationChunk,
    Idea,
    IdeaExplanation,
    IdeaUpdate,
    UsageEntry,
    UsagePayload,
)
from gigaevo.memory.utils import median, to_float

# ---------------------------------------------------------------------------
# Usage-payload helpers  (was utils/helpers.py)
# ---------------------------------------------------------------------------


def build_usage_payload(task_to_deltas: dict[str, list[float]]) -> UsagePayload:
    usage_entries: list[UsageEntry] = []
    total_deltas: list[float] = []
    for task_summary in sorted(task_to_deltas):
        deltas = [
            d
            for raw in task_to_deltas[task_summary]
            if (d := to_float(raw)) is not None
        ]
        if not deltas:
            continue
        usage_entries.append(
            UsageEntry(
                task_description_summary=task_summary,
                used_count=len(deltas),
                fitness_delta_per_use=deltas,
                median_delta_fitness=median(deltas),
            )
        )
        total_deltas.extend(deltas)
    return UsagePayload(
        entries=usage_entries,
        total_used=len(total_deltas),
        median_delta_fitness=median(total_deltas) if total_deltas else None,
    )


def _extract_task_deltas(usage: UsagePayload | Any) -> dict[str, list[float]]:
    if isinstance(usage, UsagePayload):
        result: dict[str, list[float]] = {}
        for entry in usage.entries:
            deltas = [d for d in entry.fitness_delta_per_use if d is not None]
            if deltas:
                result.setdefault(entry.task_description_summary, []).extend(deltas)
        return result
    # Legacy dict path: {"used": {"entries": [...]}}
    if not isinstance(usage, dict):
        return {}
    used = usage.get("used")
    if not isinstance(used, dict):
        return {}
    entries = used.get("entries")
    if not isinstance(entries, list):
        return {}
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        task = str(entry.get("task_description_summary") or "").strip()
        if not task:
            continue
        raw_deltas = entry.get("fitness_delta_per_use") or entry.get("fitness_deltas")
        if not isinstance(raw_deltas, list):
            continue
        deltas = [d for raw in raw_deltas if (d := to_float(raw)) is not None]
        if deltas:
            result.setdefault(task, []).extend(deltas)
    return result


def merge_usage_payloads(existing: Any, incoming: Any) -> UsagePayload:
    """Merge two usage payloads, combining per-task fitness-delta lists."""
    existing_deltas = _extract_task_deltas(existing)
    incoming_deltas = _extract_task_deltas(incoming)
    if not existing_deltas and not incoming_deltas:
        if isinstance(existing, UsagePayload):
            return existing
        if isinstance(incoming, UsagePayload):
            return incoming
        return UsagePayload()
    merged: dict[str, list[float]] = {k: list(v) for k, v in existing_deltas.items()}
    for task, deltas in incoming_deltas.items():
        merged.setdefault(task, []).extend(deltas)
    return build_usage_payload(merged)


# ---------------------------------------------------------------------------
# IdeaBank
# ---------------------------------------------------------------------------


class IdeaBank:
    """
    Stores and manages Idea objects for an IdeaTracker session.

    Provides add/get/update/enrich operations and produces chunked
    representations for LLM classification calls (ClassifyingAnalyzer).
    All mutations return a new Idea via model_copy — the internal list
    is updated in place but ideas themselves are treated as immutable.
    """

    def __init__(self, chunk_size: int = 5) -> None:
        self._ideas: list[Idea] = []
        self._id_index: dict[str, int] = {}  # O(1) id → list-index lookup
        self._chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add(self, idea: Idea) -> None:
        """Append an Idea. Reassigns id if it already exists in the bank."""
        if idea.id in self._id_index:
            idea = idea.model_copy(update={"id": str(uuid4())})
        self._id_index[idea.id] = len(self._ideas)
        self._ideas.append(idea)

    def apply(self, result: AnalysisResult) -> None:
        """Add all new_ideas and apply all updates from an AnalysisResult."""
        for idea in result.new_ideas:
            self.add(idea)
        for upd in result.updates:
            self.update(upd)

    def update(self, upd: IdeaUpdate) -> bool:
        """Apply an IdeaUpdate to an existing Idea. Returns False if not found."""
        idx = self._index(upd.idea_id)
        if idx is None:
            return False
        idea = self._ideas[idx]
        patches: dict[str, Any] = {}

        if upd.programs:
            merged_programs = list(dict.fromkeys(idea.programs + upd.programs))
            patches["programs"] = merged_programs

        if upd.generation and upd.generation > idea.last_generation:
            patches["last_generation"] = upd.generation

        if upd.new_description is not None:
            archive_entry = {
                f"{upd.idea_id}-update": {
                    "description": idea.description,
                    "programs": list(idea.programs),
                    "explanations": list(idea.explanation.entries),
                }
            }
            patches["aliases"] = idea.aliases + [archive_entry]
            patches["description"] = upd.new_description

        new_entries = idea.explanation.entries + (
            [upd.motivation] if upd.motivation else []
        )
        patches["explanation"] = IdeaExplanation(
            entries=new_entries,
            summary=idea.explanation.summary,
        )

        self._ideas[idx] = idea.model_copy(update=patches)
        return True

    def enrich(
        self,
        idea_id: str,
        *,
        keywords: list[str],
        summary: str,
        task_summary: str,
    ) -> bool:
        """Set keywords, explanation summary, and task_description_summary. Returns False if not found."""
        idx = self._index(idea_id)
        if idx is None:
            return False
        idea = self._ideas[idx]
        self._ideas[idx] = idea.model_copy(
            update={
                "keywords": keywords,
                "explanation": IdeaExplanation(
                    entries=idea.explanation.entries,
                    summary=summary,
                ),
                "task_description_summary": task_summary,
            }
        )
        return True

    def apply_usage_updates(
        self, usage_updates: dict[str, UsagePayload | dict[str, Any]]
    ) -> None:
        """Merge per-card usage payloads into matching ideas."""
        for i, idea in enumerate(self._ideas):
            update = usage_updates.get(str(idea.id or ""))
            if update:
                self._ideas[i] = idea.model_copy(
                    update={"usage": merge_usage_payloads(idea.usage, update)}
                )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, idea_id: str) -> Idea | None:
        """Return the Idea with the given id, or None."""
        idx = self._index(idea_id)
        return self._ideas[idx] if idx is not None else None

    def all_ideas(self) -> list[Idea]:
        """Return all ideas in insertion order."""
        return list(self._ideas)

    def classification_chunks(self) -> list[ClassificationChunk]:
        """
        Return ideas grouped into fixed-size chunks for LLM classification.

        Each chunk contains a formatted text block and short-id mappings,
        matching the format expected by ClassifyingAnalyzer._classify_against_bank.
        """
        if not self._ideas:
            return []
        chunks: list[ClassificationChunk] = []
        for i in range(0, len(self._ideas), self._chunk_size):
            batch = self._ideas[i : i + self._chunk_size]
            short_ids = [
                {
                    "id": idea.id,
                    "short_id": idea.id.split("-")[0],
                    "description": idea.description,
                }
                for idea in batch
            ]
            text = "".join(
                f"[{s['short_id']}]: {s['description']} \n " for s in short_ids
            )
            chunks.append(ClassificationChunk(text=text, short_ids=short_ids))
        return chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index(self, idea_id: str) -> int | None:
        return self._id_index.get(idea_id)
