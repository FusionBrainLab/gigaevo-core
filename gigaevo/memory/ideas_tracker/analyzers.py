"""
Idea analysers for IdeasTracker.

Analyzer      — Protocol that both analysers implement.
ClassifyingAnalyzer — Sequential per-program LLM classification against the idea bank.
ClusteringAnalyzer  — Batch embedding + DBSCAN + async LLM refinement pipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Protocol, runtime_checkable
import uuid

from dotenv import load_dotenv
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from gigaevo.memory.ideas_tracker.idea_bank import IdeaBank
from gigaevo.memory.ideas_tracker.llm import LLMClient
from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    EmbeddedIdea,
    Idea,
    IdeaExplanation,
    IdeaUpdate,
    ProgramRecord,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Analyzer(Protocol):
    """
    Common interface for idea analysers.

    Both ClassifyingAnalyzer and ClusteringAnalyzer implement this protocol,
    allowing IdeaTracker to use either without branching on type.
    """

    model: str

    def analyze(self, records: list[ProgramRecord], bank: IdeaBank) -> AnalysisResult:
        """Extract and classify improvement ideas from a batch of program records."""
        ...

    async def analyze_async(
        self, records: list[ProgramRecord], bank: IdeaBank
    ) -> AnalysisResult:
        """Async version of analyze — used by IdeaTracker._run to avoid asyncio nesting."""
        ...

    def call(self, step: str, content: str | dict[str, str] = "") -> str:
        """Synchronous LLM call — used by the enrichment step in IdeaTracker."""
        ...

    async def call_async(self, step: str, content: str | dict[str, str] = "") -> str:
        """Asynchronous LLM call — used by the enrichment step in IdeaTracker."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _split_id(idea_ref: str) -> tuple[str, int]:
    """
    Parse ``shortId:sequence`` from classify_ext LLM output.

    If the model omits ``:sequence``, returns sequence 1 (best-effort).
    Strips surrounding square brackets if present.
    """
    raw = idea_ref.strip()
    if ":" not in raw:
        return raw.strip("[]"), 1
    left, right = raw.split(":", 1)
    try:
        seq = int(right.strip("[]"))
    except ValueError:
        seq = 1
    return left.strip("[]"), seq


# ---------------------------------------------------------------------------
# ClassifyingAnalyzer  (was IdeaAnalyzer)
# ---------------------------------------------------------------------------


@dataclass
class _PendingIdeas:
    """
    Scratch object tracking classification state for a single program's improvements.

    Private to ClassifyingAnalyzer — not exported.
    """

    items: list[dict[str, Any]] = field(default_factory=list)
    mapping: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_improvements(cls, improvements: list[dict[str, str]]) -> _PendingIdeas:
        items = [
            {
                "description": i["description"],
                "motivation": i.get("explanation", ""),
                "classified": False,
                "target_id": "",
                "rewrite": False,
            }
            for i in improvements
        ]
        pending = cls(items=items)
        pending.refresh_mapping()
        return pending

    def refresh_mapping(self) -> None:
        mapping: dict[int, str] = {}
        c = 1
        for item in self.items:
            if not item["classified"]:
                mapping[c] = item["description"]
                c += 1
        self.mapping = mapping

    def mark_classified(self, seq_num: int, target_id: str, rewrite: bool) -> None:
        desc = self.mapping.get(seq_num)
        if desc is None:
            logger.warning(
                "ClassifyingAnalyzer: no pending idea at position {}", seq_num
            )
            return
        for item in self.items:
            if item["description"] == desc:
                item["target_id"] = target_id
                item["classified"] = True
                item["rewrite"] = rewrite
                break

    @property
    def unclassified_count(self) -> int:
        return sum(1 for i in self.items if not i["classified"])

    def as_numbered_text(self) -> str:
        lines: list[str] = []
        c = 1
        for item in self.items:
            if not item["classified"]:
                lines.append(f"{c}) {item['description']} \n")
                c += 1
        return "".join(lines)


class ClassifyingAnalyzer:
    """
    Classifies incoming improvement ideas against an existing idea bank using an LLM.

    Processes programs sequentially. For each program, asks the LLM whether each
    incoming idea is new, an update to an existing idea, or a rewrite of one.
    The bank is read at call time via analyze(records, bank) — not stored at construction,
    so the same analyser instance can serve multiple IdeaTracker sessions.

    Args:
        model: LLM model identifier.
        base_url: Optional OpenAI-compatible API base URL.
        reasoning: Optional OpenRouter reasoning settings (e.g. {"effort": "low"}).
        retry_attempts: LLM call retries on JSON parse failure.
        description_rewriting: If True, allow the LLM to rewrite idea descriptions.
    """

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        base_url: str | None = None,
        reasoning: dict[str, Any] | None = None,
        retry_attempts: int = 10,
        description_rewriting: bool = True,
    ) -> None:
        self.model = model
        self._reasoning = reasoning or {}
        self._retry_attempts = retry_attempts
        self._description_rewriting = description_rewriting
        self._llm = LLMClient(model=model, base_url=base_url)

    def call(self, step: str, content: str | dict[str, str] = "") -> str:
        """Synchronous LLM call — used by IdeaTracker enrichment step."""
        return self._llm.call(step, content, self._reasoning)

    async def call_async(self, step: str, content: str | dict[str, str] = "") -> str:
        """Asynchronous LLM call — used by IdeaTracker enrichment step."""
        return await self._llm.call_async(step, content, self._reasoning)

    def analyze(self, records: list[ProgramRecord], bank: IdeaBank) -> AnalysisResult:
        """
        Classify all program improvements against the bank.

        Returns an AnalysisResult with new ideas to add and updates to apply.
        """
        result = AnalysisResult()
        for record in tqdm(records, leave=False, desc="Classifying programs"):
            pending = _PendingIdeas.from_improvements(record.improvements)
            if not pending.items:
                continue
            self._classify_against_bank(pending, bank.classification_chunks())
            self._apply_pending_to_result(pending, record, result)
        return result

    async def analyze_async(
        self, records: list[ProgramRecord], bank: IdeaBank
    ) -> AnalysisResult:
        """Async wrapper — runs synchronous analyze() in a thread pool to avoid blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.analyze, records, bank)

    def _classify_against_bank(self, pending: _PendingIdeas, chunks: list) -> None:
        """Classify pending ideas against each bank chunk, updating pending in place."""
        for chunk in chunks:
            if pending.unclassified_count == 0:
                break
            unclassified_text = pending.as_numbered_text()
            prompt = f" Existing Ideas: \n {chunk.text} \n Incoming Ideas: \n {unclassified_text}"
            parsed: dict[str, list[Any]] = {
                "present_ideas": [],
                "new_ideas": [],
                "updated_ideas": [],
            }
            for _ in range(self._retry_attempts):
                try:
                    raw = self._llm.call("classify_ext", prompt, self._reasoning)
                    parsed = json.loads(raw)
                    break
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "ClassifyingAnalyzer: LLM returned invalid JSON (retrying): {}",
                        exc,
                    )
                except Exception as exc:
                    logger.error("ClassifyingAnalyzer: LLM call failed: {}", exc)

            for ref in parsed.get("present_ideas", []):
                if not isinstance(ref, str):
                    continue
                short_id, seq = _split_id(ref)
                full_id = self._resolve_id(short_id, chunk.short_ids)
                if full_id:
                    pending.mark_classified(seq, full_id, False)

            for item in parsed.get("updated_ideas", []):
                if not isinstance(item, dict):
                    continue
                item_id = item.get("id")
                if not isinstance(item_id, str):
                    continue
                short_id, seq = _split_id(item_id)
                full_id = self._resolve_id(short_id, chunk.short_ids)
                if full_id:
                    pending.mark_classified(seq, full_id, True)

            pending.refresh_mapping()

    def _resolve_id(self, short_id: str, short_ids: list[dict[str, str]]) -> str:
        """Map a short UUID prefix to a full UUID, or return '' if not found."""
        for entry in short_ids:
            if entry["short_id"] == short_id:
                return entry["id"]
        return ""

    def _apply_pending_to_result(
        self, pending: _PendingIdeas, record: ProgramRecord, result: AnalysisResult
    ) -> None:
        """Convert classified/unclassified pending items into AnalysisResult entries."""
        for item in pending.items:
            if not item["classified"]:
                result.new_ideas.append(
                    Idea(
                        description=item["description"],
                        strategy=record.strategy,
                        task_description=record.task_description,
                        last_generation=record.generation,
                        programs=[record.id],
                        explanation=IdeaExplanation(
                            entries=[item["motivation"]] if item["motivation"] else []
                        ),
                    )
                )
            elif item["rewrite"]:
                result.updates.append(
                    IdeaUpdate(
                        idea_id=item["target_id"],
                        programs=[record.id],
                        generation=record.generation,
                        new_description=item["description"]
                        if self._description_rewriting
                        else None,
                        motivation=item["motivation"] or None,
                    )
                )
            else:
                result.updates.append(
                    IdeaUpdate(
                        idea_id=item["target_id"],
                        programs=[record.id],
                        generation=record.generation,
                        motivation=item["motivation"] or None,
                    )
                )


# ---------------------------------------------------------------------------
# ClusteringAnalyzer  (was IdeaAnalyzerFast)
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, allowing surrounding prose or fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group())
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _validate_partition(
    included: list[Any], rejected: list[Any], start: int, end: int
) -> tuple[list[int], list[int]]:
    """Validate that included + rejected form an exact partition of start..end (inclusive)."""
    if not isinstance(included, list) or not isinstance(rejected, list):
        raise ValueError("included and rejected must be lists")
    if start > end:
        raise ValueError("empty index range")
    inc = [int(x) for x in included]
    rej = [int(x) for x in rejected]
    expected = set(range(start, end + 1))
    if len(inc + rej) != len(expected) or set(inc + rej) != expected:
        raise ValueError("partition does not cover the range exactly once")
    return inc, rej


class IdeaCluster:
    """
    A mutable cluster of EmbeddedIdea instances.

    Internal working object for ClusteringAnalyzer — not exported as a data model.
    """

    def __init__(self, cluster_id: str) -> None:
        self.cluster_id = cluster_id
        self.center: list[float] = []
        self.members: list[EmbeddedIdea] = []
        self.index_to_card: dict[int, EmbeddedIdea] = {}
        self.has_changed: bool = True

    @property
    def size(self) -> int:
        return len(self.members)

    def add_member(self, card: EmbeddedIdea) -> None:
        card.cluster_id = self.cluster_id
        self.members.append(card)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.index_to_card = {i + 1: c for i, c in enumerate(self.members)}

    def prune_stale(self) -> None:
        """Remove members whose cluster_id no longer matches this cluster."""
        self.members = [c for c in self.members if c.cluster_id == self.cluster_id]
        self._rebuild_index()

    def numbered_text(self) -> str:
        return "".join(
            f"{i + 1}) {c.description} \n" for i, c in enumerate(self.members)
        )

    def numbered_groups(self, subgroup_size: int) -> list[str]:
        if subgroup_size < 1:
            raise ValueError("subgroup_size must be >= 1")
        n = len(self.members)
        if n == 0:
            return []
        if subgroup_size >= n:
            return [self.numbered_text()]
        groups: list[str] = []
        for i in range(0, n, subgroup_size):
            chunk = self.members[i : i + subgroup_size]
            groups.append(
                "".join(f"{i + j + 1}) {c.description} \n" for j, c in enumerate(chunk))
            )
        return groups


class ClusteringAnalyzer:
    """
    Groups improvement ideas by semantic similarity using embeddings, DBSCAN,
    and async LLM refinement.

    Processes all program records in a single batch. Does not consult the existing
    bank — always returns new_ideas with an empty updates list. The bank parameter
    in analyze() is accepted for protocol compatibility but ignored.

    Args:
        model: LLM model for refine and representative steps.
        embeddings_model: sentence-transformers model id.
        base_url: Optional API base URL override.
        reasoning: Optional OpenRouter reasoning settings.
        batch_size: Encoding batch size.
        min_samples_for_dbscan: Below this count, skip DBSCAN and use one cluster.
        dbscan_eps: Cosine DBSCAN epsilon.
        dbscan_min_samples: DBSCAN min_samples.
        max_attempts: LLM call retries per step.
        max_rounds: Refinement loop upper bound.
        refine_subgroup_size: Max ideas per refine LLM call.
        llm_max_concurrent: Semaphore limit for async LLM calls.
    """

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        embeddings_model: str = "sentence-transformers/all-mpnet-base-v2",
        base_url: str | None = None,
        reasoning: dict[str, Any] | None = None,
        batch_size: int = 32,
        min_samples_for_dbscan: int = 4,
        dbscan_eps: float = 0.25,
        dbscan_min_samples: int = 2,
        max_attempts: int = 10,
        max_rounds: int = 20,
        refine_subgroup_size: int = 20,
        llm_max_concurrent: int = 100,
    ) -> None:
        self.model = model
        self._reasoning = reasoning or {}
        self._batch_size = batch_size
        self._min_samples = min_samples_for_dbscan
        self._dbscan_eps = dbscan_eps
        self._dbscan_min_samples = dbscan_min_samples
        self._max_attempts = max_attempts
        self._max_rounds = max_rounds
        self._subgroup_size = refine_subgroup_size
        self._llm = LLMClient(
            model=model, base_url=base_url, max_concurrent=llm_max_concurrent
        )
        self._embed_model = SentenceTransformer(embeddings_model)
        self._benchmark_times: list[float] = []
        self._benchmark_clusters: list[int] = []

    def call(self, step: str, content: str | dict[str, str] = "") -> str:
        """Synchronous LLM call — used by IdeaTracker enrichment step."""
        return self._llm.call(step, content, self._reasoning)

    async def call_async(self, step: str, content: str | dict[str, str] = "") -> str:
        """Asynchronous LLM call — used by IdeaTracker enrichment step."""
        return await self._llm.call_async(step, content, self._reasoning)

    def analyze(self, records: list[ProgramRecord], _bank: IdeaBank) -> AnalysisResult:
        """
        Embed, cluster, refine, and return one Idea per surviving cluster.

        _bank is accepted for protocol compatibility but not used by this analyzer.
        NOTE: Do not call this from within an async context — use analyze_async instead.
        """
        return AnalysisResult(new_ideas=asyncio.run(self._run_async(records)))

    async def analyze_async(
        self, records: list[ProgramRecord], _bank: IdeaBank
    ) -> AnalysisResult:
        """Async implementation — runs the full embed/cluster/refine pipeline."""
        return AnalysisResult(new_ideas=await self._run_async(records))

    # ------------------------------------------------------------------
    # Async pipeline
    # ------------------------------------------------------------------

    async def _run_async(self, records: list[ProgramRecord]) -> list[Idea]:
        cards = self._flatten_to_cards(records)
        if not cards:
            return []
        t0 = time.perf_counter()
        self._embed(cards)
        clusters = self._build_clusters(cards)
        await self._refine_loop(clusters, t0)
        for c in clusters:
            c.prune_stale()
        clusters = [c for c in clusters if c.size > 0]
        tasks = [self._cluster_to_idea(c, {p.id: p for p in records}) for c in clusters]
        return list(await asyncio.gather(*tasks))

    def _flatten_to_cards(self, records: list[ProgramRecord]) -> list[EmbeddedIdea]:
        cards: list[EmbeddedIdea] = []
        for record in records:
            for imp in record.improvements:
                cards.append(
                    EmbeddedIdea(
                        description=str(imp.get("description", "")),
                        source_program_id=record.id,
                        change_motivation=str(imp.get("explanation", "")),
                    )
                )
        return cards

    def _embed(self, cards: list[EmbeddedIdea]) -> None:
        texts = [c.description for c in cards]
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vecs = self._embed_model.encode(
                batch, convert_to_numpy=True, show_progress_bar=False
            )
            for i, vec in enumerate(vecs):
                cards[start + i].embedding = vec.astype(np.float64).tolist()

    def _mean_center(self, members: list[EmbeddedIdea]) -> list[float]:
        if not members:
            return []
        return (
            np.array([m.embedding for m in members], dtype=np.float64)
            .mean(axis=0)
            .tolist()
        )

    def _build_clusters(self, cards: list[EmbeddedIdea]) -> list[IdeaCluster]:
        n = len(cards)
        if n < self._min_samples:
            c = IdeaCluster(str(uuid.uuid4()))
            for card in cards:
                c.add_member(card)
            c.center = self._mean_center(c.members)
            return [c]

        mat = np.array([c.embedding for c in cards], dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.where(norms == 0, 1.0, norms)
        labels = (
            DBSCAN(
                eps=self._dbscan_eps,
                min_samples=self._dbscan_min_samples,
                metric="cosine",
            )
            .fit(mat)
            .labels_
        )

        clusters: list[IdeaCluster] = []
        for label in sorted(set(labels.tolist())):
            idxs = np.where(labels == label)[0]
            if label == -1:
                for i in idxs:
                    cl = IdeaCluster(str(uuid.uuid4()))
                    cl.add_member(cards[int(i)])
                    cl.center = self._mean_center(cl.members)
                    clusters.append(cl)
            else:
                cl = IdeaCluster(str(uuid.uuid4()))
                for i in idxs:
                    cl.add_member(cards[int(i)])
                cl.center = self._mean_center(cl.members)
                clusters.append(cl)
        return clusters

    async def _refine_loop(self, clusters: list[IdeaCluster], t0: float) -> None:
        self._benchmark_times.clear()
        self._benchmark_clusters.clear()
        pbar = tqdm(total=self._max_rounds, desc="Refinement rounds")
        for _ in range(self._max_rounds):
            for c in clusters:
                c.prune_stale()
            clusters[:] = [c for c in clusters if c.size > 0]
            eligible = [c for c in clusters if c.size >= 2 and c.has_changed]
            if not eligible:
                break
            pairs = list(
                zip(
                    eligible,
                    await asyncio.gather(*[self._refine_cluster(c) for c in eligible]),
                    strict=True,
                )
            )
            changed = self._apply_refinements(clusters, pairs)
            pbar.update(1)
            pbar.set_postfix(clusters=len(clusters))
            self._benchmark_times.append(time.perf_counter() - t0)
            self._benchmark_clusters.append(len(clusters))
            if not changed:
                break
        pbar.close()

    async def _refine_cluster(
        self, cluster: IdeaCluster
    ) -> tuple[list[int], list[int]] | None:
        sg = self._subgroup_size
        groups = cluster.numbered_groups(sg)
        n = cluster.size
        if not groups:
            return None

        async def run_subgroup(
            gi: int, text: str
        ) -> tuple[list[int], list[int]] | None:
            i0 = gi * sg
            start, end = i0 + 1, i0 + min(sg, n - i0)
            for _ in range(self._max_attempts):
                try:
                    raw = await self._llm.call_async(
                        "cluster_fast_refine", text, self._reasoning
                    )
                    data = _extract_json_object(raw)
                    return _validate_partition(
                        data.get("included", []), data.get("rejected", []), start, end
                    )
                except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                    continue
            return None

        parts = await asyncio.gather(
            *[run_subgroup(gi, t) for gi, t in enumerate(groups)]
        )
        if any(p is None for p in parts):
            return None
        merged_inc: list[int] = []
        merged_rej: list[int] = []
        for p in parts:
            assert p is not None
            merged_inc.extend(p[0])
            merged_rej.extend(p[1])
        return merged_inc, merged_rej

    def _apply_refinements(
        self,
        clusters: list[IdeaCluster],
        pairs: list[tuple[IdeaCluster, tuple[list[int], list[int]] | None]],
    ) -> bool:
        changed = False
        for cluster, parsed in pairs:
            if parsed is None:
                cluster.has_changed = True
                changed = True
                continue
            inc_idx, rej_idx = parsed
            if not rej_idx:
                cluster.has_changed = False
                continue
            changed = True
            cluster.has_changed = True
            inc_set = set(inc_idx)
            rej_set = set(rej_idx)
            included_cards = [cluster.index_to_card[i] for i in sorted(inc_set)]
            rejected_cards = [cluster.index_to_card[i] for i in sorted(rej_set)]
            cluster.members = included_cards
            cluster._rebuild_index()
            for c in included_cards:
                c.cluster_id = cluster.cluster_id
            if rejected_cards:
                new_cluster = IdeaCluster(str(uuid.uuid4()))
                seen: set[str] = set()
                for c in rejected_cards:
                    if c.id not in seen:
                        seen.add(c.id)
                        new_cluster.add_member(c)
                if new_cluster.size > 0:
                    new_cluster.center = self._mean_center(new_cluster.members)
                    clusters.append(new_cluster)
        clusters[:] = [c for c in clusters if c.size > 0]
        return changed

    async def _cluster_to_idea(
        self, cluster: IdeaCluster, records_by_id: dict[str, ProgramRecord]
    ) -> Idea:
        members = cluster.members
        if not members:
            raise ValueError("empty cluster")

        if len(members) == 1:
            rep = members[0]
        else:
            rep = await self._pick_representative(cluster) or members[0]

        prog = records_by_id.get(rep.source_program_id)
        strategy = prog.strategy if prog else ""
        task_description = prog.task_description if prog else ""
        gen = prog.generation if prog else 0

        all_gens = [
            records_by_id[m.source_program_id].generation
            for m in members
            if m.source_program_id in records_by_id
        ]
        last_gen = max(all_gens) if all_gens else gen

        programs = list(
            dict.fromkeys(m.source_program_id for m in members if m.source_program_id)
        )
        motivations = [m.change_motivation for m in members if m.change_motivation]
        other_descriptions = [
            m.description for m in members if m is not rep and m.description
        ]

        if len(members) > 1:
            desc = await self._synthesise_description(
                rep.description, other_descriptions, motivations
            )
            description = desc or rep.description
        else:
            description = rep.description

        return Idea(
            description=description,
            strategy=strategy,
            task_description=task_description,
            last_generation=last_gen,
            programs=programs,
            explanation=IdeaExplanation(entries=motivations),
        )

    async def _pick_representative(self, cluster: IdeaCluster) -> EmbeddedIdea | None:
        text = cluster.numbered_text()
        for _ in range(self._max_attempts):
            try:
                raw = await self._llm.call_async(
                    "cluster_fast_representative", text, self._reasoning
                )
                data = _extract_json_object(raw)
                idx = int(data["representative_index"])
                if 1 <= idx <= cluster.size:
                    return cluster.index_to_card.get(idx)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                continue
        return None

    async def _synthesise_description(
        self,
        rep_description: str,
        other_descriptions: list[str],
        motivations: list[str],
    ) -> str:
        all_desc = "".join(f"{k}) {d} \n" for k, d in enumerate(other_descriptions))
        all_motiv = "".join(f"{k}) {m} \n" for k, m in enumerate(motivations))
        prompt = {
            "<INSERT_REP>": f"- {rep_description}",
            "<INSERT_DES>": all_desc,
            "<INSERT_EXPL>": all_motiv,
        }
        for _ in range(self._max_attempts):
            try:
                return await self._llm.call_async(
                    "cluster_desc_synth", prompt, self._reasoning
                )
            except Exception as exc:
                logger.error("ClusteringAnalyzer desc_synth failed: {}", exc)
        return ""
