"""One-shot Redis seeder for dashboard demo data (db=14).

Populates db=14 with 4 realistic GigaEvo runs, each with programs,
metrics history, archive (1D, 150 bins), run_state, and lineage.

Usage:
    PYTHONPATH=. ~/envs/evo_fast/bin/python problems/dashboard/seed_redis.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import math
import os
import random
import uuid

import redis as redis_lib

from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.json import dumps as gjson_dumps

REDIS_DB = int(os.environ.get("DASHBOARD_REDIS_DB", "14"))
REDIS_HOST = "localhost"
REDIS_PORT = 6379

RUN_SPECS: list[dict] = [
    {
        "prefix": "hotpotqa/cold_start",
        "label": "T1",
        "total_gens": 50,
        "current_gen": 14,
        "start_fitness": 0.41,
        "end_fitness": 0.623,
        "n_programs": 80,
        "valid_frac": 0.875,
        "noise": 0.008,
    },
    {
        "prefix": "hotpotqa/nlp_prompts",
        "label": "T2",
        "total_gens": 50,
        "current_gen": 22,
        "start_fitness": 0.44,
        "end_fitness": 0.591,
        "n_programs": 120,
        "valid_frac": 0.500,
        "noise": 0.005,
    },
    {
        "prefix": "hotpotqa/thinking",
        "label": "T3",
        "total_gens": 30,
        "current_gen": 7,
        "start_fitness": 0.55,
        "end_fitness": 0.658,
        "n_programs": 60,
        "valid_frac": 0.833,
        "noise": 0.012,
    },
    {
        "prefix": "hotpotqa/p3_crossover",
        "label": "T4",
        "total_gens": 50,
        "current_gen": 50,
        "start_fitness": 0.38,
        "end_fitness": 0.612,
        "n_programs": 200,
        "valid_frac": 0.780,
        "noise": 0.006,
    },
]

MUTATIONS = [
    "rewrite_retrieval_step",
    "optimize_prompt_template",
    "add_chain_of_thought",
    "compress_context_window",
    "ensemble_answering",
    "rerank_retrieved_docs",
    "multi_hop_decompose",
    "answer_verification_step",
    "bridge_entity_extraction",
    "confidence_calibration",
]

CODE_TEMPLATES = [
    """\
def entrypoint(question: str) -> str:
    docs = retrieve(question, top_k=5)
    context = "\\n".join(docs)
    answer = llm_answer(question, context)
    return answer
""",
    """\
def entrypoint(question: str) -> str:
    entities = extract_entities(question)
    docs = []
    for e in entities[:3]:
        docs.extend(retrieve(e, top_k=3))
    context = "\\n".join(docs[:8])
    cot = llm_chain_of_thought(question, context)
    return extract_answer(cot)
""",
    """\
def entrypoint(question: str) -> str:
    initial_docs = retrieve(question, top_k=10)
    reranked = rerank(initial_docs, question)
    context = "\\n".join(reranked[:5])
    candidate = llm_answer(question, context)
    verified = verify_answer(candidate, context)
    return verified
""",
    """\
def entrypoint(question: str) -> str:
    sub_questions = decompose(question)
    sub_answers = []
    for sq in sub_questions:
        docs = retrieve(sq, top_k=3)
        ctx = "\\n".join(docs)
        sub_answers.append(llm_answer(sq, ctx))
    final_ctx = "\\n".join(sub_answers)
    return llm_answer(question, final_ctx)
""",
    """\
def entrypoint(question: str) -> str:
    docs = retrieve(question, top_k=8)
    answers = []
    for i in range(0, len(docs), 2):
        ctx = "\\n".join(docs[i:i+2])
        a, conf = llm_answer_with_confidence(question, ctx)
        answers.append((a, conf))
    answers.sort(key=lambda x: x[1], reverse=True)
    return answers[0][0]
""",
]


def _fitness_curve(
    n: int, start: float, end: float, noise: float, rng: random.Random
) -> list[float]:
    vals = []
    for i in range(n):
        t = i / max(n - 1, 1)
        v = start + (end - start) * (1 / (1 + math.exp(-8 * (t - 0.4))))
        v += rng.gauss(0, noise)
        vals.append(round(max(0.0, min(1.0, v)), 4))
    return vals


def _write_program(pipe, prefix: str, p: Program) -> None:
    key = f"{prefix}:program:{p.id}"
    data = p.to_dict()
    pipe.set(key, gjson_dumps(data))
    pipe.sadd(f"{prefix}:status:{p.state.value}", p.id)


def seed_run(r: redis_lib.Redis, spec: dict, rng: random.Random) -> tuple[int, int]:
    """Seed one run into Redis. Returns (program_count, total_db_keys)."""
    prefix = spec["prefix"]
    n_gens = spec["current_gen"]
    total_gens = spec["total_gens"]
    start_f = spec["start_fitness"]
    end_f = spec["end_fitness"]
    noise = spec["noise"]
    n_programs = spec["n_programs"]
    valid_frac = spec["valid_frac"]

    now = datetime.now(UTC)

    # ---- Build programs with lineage ----
    programs: list[Program] = []

    # Generation 1: root programs
    roots: list[Program] = []
    for i in range(3):
        p = Program(
            id=str(uuid.uuid4()),
            code=CODE_TEMPLATES[i % len(CODE_TEMPLATES)],
            lineage=Lineage(generation=1),
            state=ProgramState.DONE,
            created_at=now - timedelta(hours=n_gens * 2 + rng.uniform(0, 1)),
        )
        fitness = rng.uniform(start_f, start_f + 0.05)
        p.add_metrics({"fitness": fitness, "is_valid": 1.0})
        roots.append(p)
        programs.append(p)

    # Subsequent generations
    living = list(roots)
    progs_per_gen = max(1, (n_programs - 3) // max(n_gens - 1, 1))

    for gen in range(2, n_gens + 1):
        n_new = progs_per_gen if gen < n_gens else max(1, n_programs - len(programs))
        for _ in range(n_new):
            pool = living[-8:] if len(living) >= 8 else living
            parent = rng.choice(pool)
            mutation = rng.choice(MUTATIONS)
            code = rng.choice(CODE_TEMPLATES)

            child = Program.create_child(parents=[parent], code=code, mutation=mutation)
            child.state = ProgramState.DONE
            child.created_at = now - timedelta(
                hours=(n_gens - gen) * 2 + rng.uniform(0, 1)
            )

            t = (gen - 1) / max(n_gens - 1, 1)
            fitness = start_f + (end_f - start_f) * (1 / (1 + math.exp(-8 * (t - 0.4))))
            fitness += rng.gauss(0, noise)
            fitness = round(max(0.0, min(1.0, fitness)), 4)
            is_valid = 1.0 if rng.random() < valid_frac else 0.0
            child.add_metrics({"fitness": fitness * is_valid, "is_valid": is_valid})
            parent.lineage.add_child(child.id)
            programs.append(child)
            living.append(child)

    # ---- Ensure 4-hop ancestor chain for best program ----
    valid_progs = [p for p in programs if p.metrics.get("is_valid", 0.0) > 0.5]
    if valid_progs:
        valid_progs.sort(key=lambda p: p.metrics.get("fitness", 0.0), reverse=True)
        best = valid_progs[0]
        cur = best
        for _ in range(4):
            if cur.lineage.parents:
                pid = cur.lineage.parents[0]
                found = next((p for p in programs if p.id == pid), None)
                if found:
                    cur = found
                    continue
            # Create synthetic ancestor
            ancestor = Program(
                id=str(uuid.uuid4()),
                code=CODE_TEMPLATES[rng.randint(0, len(CODE_TEMPLATES) - 1)],
                lineage=Lineage(generation=max(1, cur.lineage.generation - 1)),
                state=ProgramState.DONE,
                created_at=cur.created_at - timedelta(hours=2),
            )
            ancestor.add_metrics(
                {
                    "fitness": round(rng.uniform(start_f, end_f * 0.85), 4),
                    "is_valid": 1.0,
                }
            )
            cur.lineage.parents = [ancestor.id]
            ancestor.lineage.add_child(cur.id)
            programs.append(ancestor)
            cur = ancestor

    # ---- Write programs to Redis ----
    pipe = r.pipeline(transaction=False)
    for p in programs:
        _write_program(pipe, prefix, p)
    pipe.execute()

    # ---- Archive (1D, 150 bins, fitness dimension) ----
    valid_sorted = sorted(
        [p for p in programs if p.metrics.get("is_valid", 0.0) > 0.5],
        key=lambda p: p.metrics.get("fitness", 0.0),
        reverse=True,
    )
    archive_mapping: dict[str, str] = {}
    for p in valid_sorted:
        fitness = p.metrics.get("fitness", 0.0)
        bin_idx = min(149, int(fitness * 150))
        if str(bin_idx) not in archive_mapping:
            archive_mapping[str(bin_idx)] = p.id
    if archive_mapping:
        r.hset(f"{prefix}:archive", mapping=archive_mapping)

    # ---- run_state ----
    r.hset(
        f"{prefix}:run_state",
        mapping={
            "engine:total_generations": str(n_gens),
            "engine:total_gens_planned": str(total_gens),
        },
    )

    # ---- Metrics history ----
    gen_means = _fitness_curve(n_gens, start_f, end_f, noise * 2, rng)
    gen_stds = [round(rng.uniform(0.01, 0.04), 4) for _ in range(n_gens)]
    gen_frontier: list[float] = []
    best_seen = start_f
    for v in gen_means:
        best_seen = max(best_seen, v)
        gen_frontier.append(round(best_seen, 4))

    progs_per_gen_metric = max(3, n_programs // n_gens)
    gen_valid = [
        int(progs_per_gen_metric * valid_frac * rng.uniform(0.85, 1.1))
        for _ in range(n_gens)
    ]
    gen_invalid = [max(0, progs_per_gen_metric - v) for v in gen_valid]

    def _write_hist(key: str, values: list, step_offset: int = 0) -> None:
        p2 = r.pipeline(transaction=False)
        for i, v in enumerate(values):
            p2.rpush(key, json.dumps({"s": step_offset + i, "v": v}))
        p2.execute()

    _write_hist(
        f"{prefix}:metrics:history:program_metrics:valid_gen_fitness_mean", gen_means
    )
    _write_hist(
        f"{prefix}:metrics:history:program_metrics:valid_gen_fitness_std", gen_stds
    )
    _write_hist(
        f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness", gen_frontier
    )
    _write_hist(
        f"{prefix}:metrics:history:program_metrics:programs_valid_count", gen_valid
    )
    _write_hist(
        f"{prefix}:metrics:history:program_metrics:programs_invalid_count", gen_invalid
    )
    _write_hist(
        f"{prefix}:metrics:history:program_metrics:programs_total_count",
        [gen_valid[i] + gen_invalid[i] for i in range(n_gens)],
    )

    # Validator duration history
    val_durs = [round(rng.uniform(1.5, 5.0), 3) for _ in range(min(60, n_programs))]
    _write_hist(
        f"{prefix}:metrics:history:dag_runner:dag:internals"
        ":CallValidatorFunction:stage_duration",
        val_durs,
    )

    return len(programs), r.dbsize()


def main() -> None:
    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    r.flushdb()
    print(f"Seeding Redis db={REDIS_DB}...")

    rng = random.Random(42)
    total_programs = 0
    for spec in RUN_SPECS:
        n_progs, n_keys = seed_run(r, spec, rng)
        print(f"  {spec['prefix']}: {n_progs} programs, db has {n_keys} total keys")
        total_programs += n_progs

    print(
        f"\nSeeded 4 runs into Redis db={REDIS_DB} — "
        f"{total_programs} programs, {r.dbsize()} keys"
    )
    r.close()


if __name__ == "__main__":
    main()
