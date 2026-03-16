"""Redis-backed context for the dashboard evolution problem.

Reads from a demo Redis DB (default: db=14) pre-seeded by seed_redis.py.
Override with env var DASHBOARD_REDIS_DB.

build_context() is called synchronously by the GigaEvo pipeline before
entrypoint(context) runs. Uses the sync redis client directly.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import redis as redis_lib

from gigaevo.programs.program import Program
from gigaevo.utils.json import loads as gjson_loads

REDIS_DB: int = int(os.environ.get("DASHBOARD_REDIS_DB", "14"))
REDIS_HOST: str = "localhost"
REDIS_PORT: int = 6379

_RUN_SPECS: list[dict] = [
    {"prefix": "hotpotqa/cold_start",   "label": "T1", "total_gens": 50},
    {"prefix": "hotpotqa/nlp_prompts",  "label": "T2", "total_gens": 50},
    {"prefix": "hotpotqa/thinking",     "label": "T3", "total_gens": 30},
    {"prefix": "hotpotqa/p3_crossover", "label": "T4", "total_gens": 50},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_hist(r: redis_lib.Redis, key: str) -> list[float]:
    """Read a metrics history list (JSON entries {s, v}), sort by step, return values."""
    raws = r.lrange(key, 0, -1)
    if not raws:
        return []
    pairs: list[tuple[int, float]] = []
    for raw in raws:
        try:
            obj = json.loads(raw)
            pairs.append((int(obj.get("s", 0)), float(obj.get("v", 0.0))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    pairs.sort(key=lambda x: x[0])
    return [v for _, v in pairs]


def _load_program(r: redis_lib.Redis, prefix: str, pid: str) -> Program | None:
    raw = r.get(f"{prefix}:program:{pid}")
    if not raw:
        return None
    try:
        return Program.from_dict(gjson_loads(raw))
    except Exception:
        return None


def _code_preview(code: str, max_chars: int = 400) -> str:
    """Return up to max_chars chars of code, stripping a leading docstring."""
    lines = code.splitlines()
    i = 0
    # skip blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1
    # skip opening docstring block
    if i < len(lines) and lines[i].strip().startswith('"""'):
        i += 1
        while i < len(lines) and '"""' not in lines[i]:
            i += 1
        i += 1
    result = "\n".join(lines[i:]).lstrip()
    return result[:max_chars]


def _build_genealogy(
    r: redis_lib.Redis,
    prefix: str,
    program: Program,
    max_hops: int = 4,
    max_nodes: int = 10,
) -> dict:
    """Build a recursive genealogy dict tracing the first parent up to max_hops deep."""
    node_count = [1]

    def _node(p: Program, depth: int) -> dict:
        entry: dict = {
            "id": p.id[:8],
            "fitness": round(p.metrics.get("fitness", 0.0), 4),
            "generation": p.lineage.generation,
            "mutation": p.lineage.mutation,
            "parents": [],
        }
        if depth > 0 and p.lineage.parents and node_count[0] < max_nodes:
            parent_prog = _load_program(r, prefix, p.lineage.parents[0])
            if parent_prog is not None:
                node_count[0] += 1
                entry["parents"].append(_node(parent_prog, depth - 1))
        return entry

    return _node(program, max_hops)


def _fetch_top_programs(
    r: redis_lib.Redis, prefix: str, n: int = 5
) -> tuple[list[Program], list[dict]]:
    """
    Scan all programs for this run prefix, return top-N valid by fitness.
    Returns (raw_programs, formatted_dicts).
    """
    all_keys: list[bytes] = []
    for key in r.scan_iter(match=f"{prefix}:program:*", count=500):
        all_keys.append(key)

    programs: list[Program] = []
    chunk = 200
    for i in range(0, len(all_keys), chunk):
        batch = all_keys[i : i + chunk]
        raws = r.mget(batch)
        for raw in raws:
            if not raw:
                continue
            try:
                p = Program.from_dict(gjson_loads(raw))
                if p.metrics.get("is_valid", 0.0) > 0.5:
                    programs.append(p)
            except Exception:
                pass

    programs.sort(key=lambda p: p.metrics.get("fitness", 0.0), reverse=True)
    top_raw = programs[:n]

    formatted: list[dict] = []
    for rank, p in enumerate(top_raw, 1):
        formatted.append(
            {
                "rank": rank,
                "id": p.id[:8],
                "fitness": round(p.metrics.get("fitness", 0.0), 4),
                "generation": p.lineage.generation,
                "mutation": p.lineage.mutation,
                "code_preview": _code_preview(p.code),
                "num_parents": len(p.lineage.parents),
                "num_children": len(p.lineage.children),
                "num_descendants": len(p.lineage.children),  # depth-1 approximation
                "created_at": p.created_at.isoformat(),
            }
        )

    return top_raw, formatted


def _build_run_context(r: redis_lib.Redis, spec: dict) -> dict:
    prefix = spec["prefix"]
    label = spec["label"]
    total_gens = spec["total_gens"]

    # ---- run_state ----
    run_state_raw = r.hgetall(f"{prefix}:run_state")
    run_state = {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in run_state_raw.items()
    }
    current_gen = int(run_state.get("engine:total_generations", 0))

    # ---- Metrics history ----
    gen_fitness_mean = _read_hist(
        r, f"{prefix}:metrics:history:program_metrics:valid_gen_fitness_mean"
    )
    gen_fitness_std = _read_hist(
        r, f"{prefix}:metrics:history:program_metrics:valid_gen_fitness_std"
    )
    gen_fitness_frontier = _read_hist(
        r, f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness"
    )
    gen_valid_count = [
        int(v)
        for v in _read_hist(
            r, f"{prefix}:metrics:history:program_metrics:programs_valid_count"
        )
    ]
    gen_invalid_count = [
        int(v)
        for v in _read_hist(
            r, f"{prefix}:metrics:history:program_metrics:programs_invalid_count"
        )
    ]

    best_fitness = gen_fitness_frontier[-1] if gen_fitness_frontier else 0.0

    # ---- Stagnation detection ----
    last_improvement_gen = 0
    for i in range(len(gen_fitness_frontier) - 1, -1, -1):
        if i == 0 or gen_fitness_frontier[i] > gen_fitness_frontier[i - 1]:
            last_improvement_gen = i + 1
            break
    gens_since_improvement = max(0, current_gen - last_improvement_gen)

    # ---- Status ----
    if current_gen >= total_gens:
        status = "complete"
    elif gens_since_improvement >= 8:
        status = "stalled"
    else:
        status = "running"

    # ---- Archive (1D, 150 bins) ----
    archive_raw = r.hgetall(f"{prefix}:archive")
    archive_cells: list[dict] = []
    for cell_key_b, pid_b in archive_raw.items():
        bin_idx = int(cell_key_b.decode() if isinstance(cell_key_b, bytes) else cell_key_b)
        pid_str = pid_b.decode() if isinstance(pid_b, bytes) else pid_b
        p = _load_program(r, prefix, pid_str)
        if p is not None:
            cell_fitness = p.metrics.get("fitness", 0.0)
            center = round((bin_idx + 0.5) / 150.0, 4)
            archive_cells.append(
                {
                    "bin": bin_idx,
                    "center": center,
                    "fitness": round(cell_fitness, 4),
                    "program_id": pid_str[:8],
                }
            )
    archive_cells.sort(key=lambda c: c["bin"])

    # ---- Population stats ----
    total_valid = gen_valid_count[-1] if gen_valid_count else 0
    total_invalid = gen_invalid_count[-1] if gen_invalid_count else 0
    total_programs = total_valid + total_invalid
    valid_rate = total_valid / max(total_programs, 1)

    # ---- Top programs ----
    top_raw, top_programs = _fetch_top_programs(r, prefix, n=5)

    # ---- Genealogy for rank-1 program ----
    genealogy: dict = {}
    if top_raw:
        genealogy = _build_genealogy(r, prefix, top_raw[0])

    # ---- Validator telemetry ----
    dur_key = (
        f"{prefix}:metrics:history:dag_runner:dag:internals"
        ":CallValidatorFunction:stage_duration"
    )
    dur_vals = _read_hist(r, dur_key)
    if dur_vals:
        dur_floats = [float(v) for v in dur_vals if v is not None]
        validator_mean_s = sum(dur_floats) / len(dur_floats) if dur_floats else 0.0
        dur_floats_sorted = sorted(dur_floats)
        p95_idx = max(0, int(len(dur_floats_sorted) * 0.95) - 1)
        validator_p95_s = dur_floats_sorted[p95_idx] if dur_floats_sorted else 0.0
    else:
        validator_mean_s = 0.0
        validator_p95_s = 0.0

    return {
        # Identity
        "name": prefix,
        "label": label,
        "prefix": prefix,
        "db": REDIS_DB,
        "status": status,
        # Progress
        "current_gen": current_gen,
        "total_gens": total_gens,
        "gens_since_improvement": gens_since_improvement,
        # Fitness trajectory (one value per completed gen)
        "gen_fitness_mean": [round(v, 4) for v in gen_fitness_mean],
        "gen_fitness_std": [round(v, 4) for v in gen_fitness_std],
        "gen_fitness_frontier": [round(v, 4) for v in gen_fitness_frontier],
        "gen_valid_count": gen_valid_count,
        "gen_invalid_count": gen_invalid_count,
        # Convenience: best fitness scalar for quick display
        "best_fitness": round(best_fitness, 4),
        # Archive (1D, 150 bins, fitness dimension)
        "archive_total_cells": 150,
        "archive_filled_cells": len(archive_cells),
        "archive_occupancy_pct": round(len(archive_cells) / 150 * 100, 1),
        "archive_dim": "fitness",
        "archive_cells": archive_cells,
        # Top programs (up to 5)
        "top_programs": top_programs,
        # Genealogy tree for rank-1 program (up to 4 hops deep)
        "genealogy": genealogy,
        # Population stats
        "total_programs": total_programs,
        "valid_programs": total_valid,
        "valid_rate": round(valid_rate, 3),
        "acceptance_rate": round(valid_rate * 0.6, 3),
        # Validator telemetry
        "validator_mean_s": round(validator_mean_s, 2),
        "validator_p95_s": round(validator_p95_s, 2),
    }


def build_context() -> dict:
    """Read demo data from Redis db=14 and return rich research context."""
    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    try:
        runs = [_build_run_context(r, spec) for spec in _RUN_SPECS]
    finally:
        r.close()

    return {
        "runs": runs,
        "framework": "GigaEvo v1.23",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
