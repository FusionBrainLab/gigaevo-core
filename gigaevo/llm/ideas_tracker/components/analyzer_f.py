"""
Fast idea clustering: sentence embeddings, DBSCAN (or single-cluster fallback),
and async LLM refinement into :class:`~gigaevo.llm.ideas_tracker.components.data_components.RecordCardExtended` records.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any
import uuid

from dotenv import load_dotenv
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from gigaevo.llm.ideas_tracker.components.data_components import (
    ClusterCard,
    ProgramRecord,
    RecordCardEmbedding,
    RecordCardExtended,
    RefinementRoundResult,
)
from gigaevo.llm.ideas_tracker.components.prompt_manager import PromptManager
from gigaevo.llm.ideas_tracker.utils.it_logger import IdeasTrackerLogger

load_dotenv()


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


def _validate_partition_global(
    included: list[Any], rejected: list[Any], start: int, end: int
) -> tuple[list[int], list[int]]:
    """Partition of inclusive index range ``start..end`` (e.g. a refinement subgroup)."""
    if not isinstance(included, list) or not isinstance(rejected, list):
        raise ValueError("included and rejected must be lists")
    if start > end:
        raise ValueError("empty index range")
    inc = [int(x) for x in included]
    rej = [int(x) for x in rejected]
    all_idx = inc + rej
    expected = set(range(start, end + 1))
    if len(all_idx) != len(expected):
        raise ValueError("partition size mismatch")
    if set(all_idx) != expected:
        raise ValueError("indices must cover the range exactly once")
    return inc, rej


class IdeaAnalyzerFast:
    """
    Embedding + DBSCAN (or single-cluster fallback) + async LLM refinement pipeline.

    Expands each program's improvements into embedding cards, clusters them, splits
    mixed clusters via the refine prompt (optionally in fixed-size index subgroups),
    then builds one :class:`~gigaevo.llm.ideas_tracker.components.data_components.RecordCardExtended`
    per surviving cluster (representative description, other members in ``aliases``).
    """

    def __init__(
        self,
        model: str = "deepseek/deepseek-v3.2",
        embeddings_model: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 32,
        reasoning: dict[str, Any] | None = None,
        base_url: str | None = None,
        min_samples_for_dbscan: int = 4,
        dbscan_eps: float = 0.25,
        dbscan_min_samples: int = 2,
        max_attempts: int = 10,
        max_rounds: int = 20,
        recompute_center: bool = False,
        refine_subgroup_size: int = 20,
    ) -> None:
        """
        Args:
            model: Chat model name for OpenAI-compatible API (refine + representative steps).
            embeddings_model: ``sentence-transformers`` model id for idea embeddings.
            batch_size: Encode batch size when computing embeddings.
            reasoning: Optional OpenRouter ``extra_body.reasoning`` (e.g. ``{"effort": "low"}``).
            base_url: Optional API base URL; else from ``OPENAI_BASE_URL`` / ``BASE_URL`` / ``LLM_BASE_URL``.
            min_samples_for_dbscan: If fewer idea cards than this, skip DBSCAN and use one cluster.
            dbscan_eps: Cosine DBSCAN ``eps`` (after L2-normalizing embedding rows).
            dbscan_min_samples: DBSCAN ``min_samples``.
            max_attempts: Retries per LLM call when JSON validation fails.
            max_rounds: Upper bound on refinement rounds (splitting rejects into a pool cluster).
            recompute_center: If True, refresh cluster embedding centroid after membership changes.
            refine_subgroup_size: Max ideas per refine LLM call; indices stay global across subgroups.
        """
        self.model = model
        self.embeddings_model_name = embeddings_model
        self.batch_size = batch_size
        self.reasoning = reasoning or {}
        self.base_url = str(base_url).strip() if base_url is not None else None
        self._is_openrouter = False
        self.logger: IdeasTrackerLogger | None = None
        self.description_rewriting = False

        self.min_samples_for_dbscan = min_samples_for_dbscan
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.max_attempts = max_attempts
        self.max_rounds = max_rounds
        self.recompute_center = recompute_center
        self.refine_subgroup_size = refine_subgroup_size

        self.prompt_manager = PromptManager()
        self.embeddings_model = SentenceTransformer(self.embeddings_model_name)
        self.embedd_size = int(self.embeddings_model.encode("Test sentence").shape[0])

        self._cards: list[RecordCardEmbedding] = []
        self.program_by_id: dict[str, ProgramRecord] = {}
        self.embedding_id_pairs: list[tuple[list[float], str]] = []

        self._init_llm_clients()

    def _init_llm_clients(self) -> None:
        """Build ``AsyncOpenAI`` from env (API key, optional base URL, OpenRouter detection)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment or .env file."
            )
        env_base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("BASE_URL")
            or os.getenv("LLM_BASE_URL")
        )
        base_url = env_base_url or self.base_url
        if not base_url and api_key.startswith("sk-or-"):
            base_url = "https://openrouter.ai/api/v1"
        self._is_openrouter = bool(base_url and "openrouter.ai" in base_url)

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.llm_async = AsyncOpenAI(**client_kwargs)

        if self.logger is not None:
            self.logger.log_init(
                component="IdeaAnalyzerFast",
                model_name=self.model,
            )

    def call_llm(self, step_name: str, prompt_content: str) -> str:
        """
        Synchronous chat completion for code paths that use the sync API (e.g. enrichment).

        :class:`~gigaevo.llm.ideas_tracker.components.analyzer.IdeaAnalyzer` exposes the same
        method; without it, :meth:`IdeaTracker.enrich_ideas` fails silently and leaves
        keywords/summaries empty.
        """
        return asyncio.run(self._call_llm_async(step_name, prompt_content))

    def ingest_programs(
        self, programs: list[ProgramRecord]
    ) -> list[RecordCardEmbedding]:
        """
        Flatten each program's ``improvements`` into :class:`RecordCardEmbedding` cards.

        Sets :attr:`program_by_id` and :attr:`_cards`.
        """
        self.program_by_id = {p.id: p for p in programs}
        cards: list[RecordCardEmbedding] = []
        for program in programs:
            for imp in program.improvements:
                desc = str(imp.get("description", ""))
                explanation = str(imp.get("explanation", ""))
                cards.append(
                    RecordCardEmbedding(
                        id=str(uuid.uuid4()),
                        description=desc,
                        source_program_id=program.id,
                        change_motivation=explanation,
                    )
                )
        self._cards = cards
        return cards

    def create_embeddings(self, cards: list[RecordCardEmbedding]) -> None:
        """Encode each card's ``description`` and fill ``embedding`` and :attr:`embedding_id_pairs`."""
        if not cards:
            return
        texts = [c.description for c in cards]
        self.embedding_id_pairs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs = self.embeddings_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for i, vec in enumerate(vecs):
                card = cards[start + i]
                emb = vec.astype(np.float64).tolist()
                card.embedding = emb
                self.embedding_id_pairs.append((emb, card.id))

    def _mean_center(self, members: list[RecordCardEmbedding]) -> list[float]:
        """Mean embedding vector across members (empty list → empty list)."""
        if not members:
            return []
        arr = np.array([m.embedding for m in members], dtype=np.float64)
        return arr.mean(axis=0).tolist()

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row for cosine distance used by DBSCAN."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def build_initial_clusters(
        self, cards: list[RecordCardEmbedding]
    ) -> list[ClusterCard]:
        """
        Cluster embedding cards with DBSCAN on cosine distance, or one cluster if too few points.

        Noise points (label ``-1``) become singleton clusters. Each cluster gets a mean ``center``.
        """
        n = len(cards)
        if n == 0:
            return []
        if n < self.min_samples_for_dbscan:
            cid = str(uuid.uuid4())
            cluster = ClusterCard(cluster_id=cid, center=[], members=[])
            for c in cards:
                cluster.add_member(c)
            cluster.center = self._mean_center(cluster.members)
            cluster.rebuild_index_to_card()
            return [cluster]

        mat = np.array([c.embedding for c in cards], dtype=np.float64)
        mat = self._normalize_rows(mat)
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="cosine",
        ).fit(mat)
        labels = clustering.labels_
        out: list[ClusterCard] = []
        unique_labels = sorted(set(labels.tolist()))
        for label in unique_labels:
            idxs = np.where(labels == label)[0]
            if label == -1:
                for i in idxs:
                    sub = [cards[int(i)]]
                    cid = str(uuid.uuid4())
                    cluster = ClusterCard(cluster_id=cid, center=[], members=[])
                    for c in sub:
                        cluster.add_member(c)
                    cluster.center = self._mean_center(cluster.members)
                    cluster.rebuild_index_to_card()
                    out.append(cluster)
            else:
                sub = [cards[int(i)] for i in idxs]
                cid = str(uuid.uuid4())
                cluster = ClusterCard(cluster_id=cid, center=[], members=[])
                for c in sub:
                    cluster.add_member(c)
                cluster.center = self._mean_center(cluster.members)
                cluster.rebuild_index_to_card()
                out.append(cluster)
        return out

    def _maybe_recompute_center(self, cluster: ClusterCard) -> None:
        """Update ``cluster.center`` to the mean embedding when :attr:`recompute_center` is True."""
        if self.recompute_center and cluster.members:
            cluster.center = self._mean_center(cluster.members)

    async def _call_llm_async(self, step_name: str, prompt_content: str) -> str:
        """
        Run one chat completion: prompts ``{step_name}__system`` and ``{step_name}__user`` with ``<INSERT>``.

        Returns assistant message text (may be empty).
        """
        prompt_system = self.prompt_manager.load_prompt(
            prompt_name=f"{step_name}__system"
        )
        prompt_user = self.prompt_manager.load_prompt(
            prompt_name=f"{step_name}__user",
            insert_data=prompt_content,
        )
        request_kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            "model": self.model,
            "temperature": 0,
        }
        if self._is_openrouter and self.reasoning:
            request_kwargs["extra_body"] = {"reasoning": self.reasoning}
        if self._is_openrouter and self.model.startswith("google/"):
            request_kwargs["extra_body"] = {"provider": {"order": ["google-ai-studio"]}}
        try:
            response = await self.llm_async.chat.completions.create(**request_kwargs)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""
        content = response.choices[0].message.content
        return content or ""

    async def _refine_cluster_llm(
        self, cluster: ClusterCard
    ) -> tuple[list[int], list[int]] | None:
        """
        Ask the refine model to partition cluster ideas into keep vs reject (global 1-based indices).

        Large clusters are split into :attr:`refine_subgroup_size` chunks; results are merged.
        Returns ``None`` if any subgroup fails validation after :attr:`max_attempts` retries.
        """
        sg = self.refine_subgroup_size
        groups = cluster.numbered_idea_groups(sg)
        n = cluster.size
        if not groups:
            return None

        async def run_subgroup(
            gi: int, text: str
        ) -> tuple[list[int], list[int]] | None:
            i0 = gi * sg
            chunk_len = min(sg, n - i0)
            start = i0 + 1
            end = i0 + chunk_len
            for _ in range(self.max_attempts):
                try:
                    raw = await self._call_llm_async("cluster_fast_refine", text)
                    data = _extract_json_object(raw)
                    return _validate_partition_global(
                        data.get("included", []),
                        data.get("rejected", []),
                        start,
                        end,
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

    async def _representative_llm(self, cluster: ClusterCard) -> int | None:
        """
        Return a 1-based ``representative_index`` for the cluster, or ``None`` if parsing fails.

        Uses the full :meth:`~gigaevo.llm.ideas_tracker.components.data_components.ClusterCard.numbered_ideas_text`.
        """
        text = cluster.numbered_ideas_text()
        n = cluster.size
        for _ in range(self.max_attempts):
            try:
                raw = await self._call_llm_async("cluster_fast_representative", text)
                data = _extract_json_object(raw)
                idx = int(data["representative_index"])
                if 1 <= idx <= n:
                    return idx
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                continue
        return None

    async def _gather_refine_results(
        self, eligible: list[ClusterCard]
    ) -> list[tuple[ClusterCard, tuple[list[int], list[int]] | None]]:
        """Run :meth:`_refine_cluster_llm` for each cluster concurrently."""
        tasks = [self._refine_cluster_llm(c) for c in eligible]
        results = await asyncio.gather(*tasks)
        return list(zip(eligible, results, strict=True))

    def _new_cluster_from_rejected_cards(
        self, rejected_cards: list[RecordCardEmbedding]
    ) -> ClusterCard | None:
        """
        Create one new :class:`ClusterCard` whose members are exactly the rejected
        embedding cards (deduped by ``card.id``). Returns ``None`` if nothing to add.
        """
        if not rejected_cards:
            return None
        pool_id = str(uuid.uuid4())
        pool = ClusterCard(cluster_id=pool_id, center=[], members=[])
        seen_ids: set[str] = set()
        for c in rejected_cards:
            if c.id in seen_ids:
                continue
            seen_ids.add(c.id)
            pool.add_member(c)
        if pool.size == 0:
            return None
        pool.center = self._mean_center(pool.members)
        pool.rebuild_index_to_card()
        self._maybe_recompute_center(pool)
        return pool

    def _apply_refine_round(
        self,
        clusters: list[ClusterCard],
        pairs: list[tuple[ClusterCard, tuple[list[int], list[int]] | None]],
    ) -> RefinementRoundResult:
        """
        Apply included/rejected index lists: shrink clusters; for each cluster that
        yields rejects, append **one** new cluster containing **only** that cluster's
        rejected cards (not merged with rejects from other clusters).

        Sets ``cluster.has_changed``: ``True`` if the partition had rejects or parse
        failed (``None``); ``False`` if there were no rejects (refinement stable for
        that cluster).

        Mutates ``clusters`` in place. Failed parses (``None``) count as change.
        """
        aggregate_changed = False

        for cluster, parsed in pairs:
            if parsed is None:
                aggregate_changed = True
                cluster.has_changed = True
                continue
            included_idx, rejected_idx = parsed
            if len(rejected_idx) == 0:
                cluster.has_changed = False
                continue
            aggregate_changed = True
            cluster.has_changed = True
            inc_set = set(included_idx)
            rej_set = set(rejected_idx)
            included_cards = [cluster.index_to_card[i] for i in sorted(inc_set)]
            rejected_cards = [cluster.index_to_card[i] for i in sorted(rej_set)]
            cluster.members = included_cards
            cluster.rebuild_index_to_card()
            for c in included_cards:
                c.cluster_id = cluster.cluster_id
            self._maybe_recompute_center(cluster)

            reject_cluster = self._new_cluster_from_rejected_cards(rejected_cards)
            if reject_cluster is not None:
                clusters.append(reject_cluster)

        clusters[:] = [c for c in clusters if c.size > 0]
        return RefinementRoundResult(has_changed=aggregate_changed)

    async def refine_clusters_loop(self, clusters: list[ClusterCard]) -> None:
        """
        Repeatedly refine clusters with at least two members and ``has_changed`` True
        until stable or :attr:`max_rounds`. Clusters with ``has_changed`` False (no rejects
        in the last refine) are skipped.
        """
        pbar = tqdm(total=self.max_rounds, desc="Refinement rounds")
        for step in range(self.max_rounds):
            for c in clusters:
                c.prune_stale_members()
            clusters[:] = [c for c in clusters if c.size > 0]

            eligible = [c for c in clusters if c.size >= 2 and c.has_changed]
            if not eligible:
                break

            pairs = await self._gather_refine_results(eligible)
            round_result = self._apply_refine_round(clusters, pairs)
            if not round_result.has_changed:
                break
            pbar.update(1)
            pbar.set_postfix(clusters_count=len(clusters))
        pbar.close()

    async def build_record_extended_async(
        self, cluster: ClusterCard, program_by_id: dict[str, ProgramRecord]
    ) -> RecordCardExtended:
        """
        Build a :class:`~gigaevo.llm.ideas_tracker.components.data_components.RecordCardExtended` for one cluster.

        Chooses a representative idea (LLM for multi-member clusters), sets metadata from its program,
        unions program ids, aggregates explanations, and stores non-representative descriptions under ``aliases``.
        """
        members = list(cluster.members)
        if not members:
            raise ValueError("empty cluster")
        if len(members) == 1:
            rep = members[0]
        else:
            idx = await self._representative_llm(cluster)
            if idx is None:
                rep = members[0]
            else:
                rep = cluster.index_to_card.get(idx, members[0])
        prog = program_by_id.get(rep.source_program_id)
        if prog is None:
            category = ""
            strategy = ""
            task_description = ""
            gen_from_rep = 0
        else:
            category = prog.category
            strategy = prog.strategy
            task_description = prog.task_description
            gen_from_rep = prog.generation

        programs_union: list[str] = []
        seen: set[str] = set()
        for m in members:
            if m.source_program_id and m.source_program_id not in seen:
                seen.add(m.source_program_id)
                programs_union.append(m.source_program_id)

        last_gen = gen_from_rep
        for m in members:
            p = program_by_id.get(m.source_program_id)
            if p is not None:
                last_gen = max(last_gen, p.generation)

        motivations = [m.change_motivation for m in members if m.change_motivation]
        first_mot = motivations[0] if motivations else ""

        new_id = str(uuid.uuid4())
        extended = RecordCardExtended(
            id=new_id,
            category=category,
            description=rep.description,
            task_description=task_description,
            strategy=strategy,
            programs=programs_union,
            change_motivation=first_mot,
            last_generation=last_gen,
        )
        for mot in motivations[1:]:
            extended.add_explanation(mot)

        # Same shape as RecordCardExtended.update_idea: one dict per alias entry.
        for m in members:
            if m is rep:
                continue

            extended.update_idea(
                experiment_id="0",
                program_id=m.source_program_id,
                generation=0,
                new_description=m.description,
                change_motivation=m.change_motivation,
            )

        return extended

    async def run(self, programs: list[ProgramRecord]) -> list[RecordCardExtended]:
        """
        End-to-end pipeline: ingest → embed → cluster → refine → one extended record per cluster.

        Returns:
            One :class:`~gigaevo.llm.ideas_tracker.components.data_components.RecordCardExtended` per non-empty cluster.
        """
        cards = self.ingest_programs(programs)
        if not cards:
            return []
        self.create_embeddings(cards)
        clusters = self.build_initial_clusters(cards)
        await self.refine_clusters_loop(clusters)
        for c in clusters:
            c.prune_stale_members()
        clusters[:] = [c for c in clusters if c.size > 0]

        out: list[RecordCardExtended] = []
        for c in clusters:
            if c.size == 0:
                continue
            ext = await self.build_record_extended_async(c, self.program_by_id)
            out.append(ext)
        return out

    def process_ideas(self) -> None:
        """Deprecated stub; use async run()."""
        raise NotImplementedError("Use await IdeaAnalyzerFast.run(programs)")
