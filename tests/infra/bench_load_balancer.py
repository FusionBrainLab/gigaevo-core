#!/usr/bin/env python3
"""Integration benchmark: BalancedChatOpenAI vs static ChatOpenAI.

Simulates realistic production load: 4 concurrent runs × 8 mutation calls
using actual mutation-length prompts. Servers are auto-discovered from
experiments/infrastructure.yaml or specified via environment variables.

Usage:
    # Auto-discover from infrastructure.yaml:
    PYTHONPATH=. ~/envs/evo_fast/bin/python tests/infra/bench_load_balancer.py

    # Explicit endpoints (comma-separated):
    LLM_ENDPOINTS=http://host1:8777/v1,http://host2:8777/v1 \\
    CHAIN_ENDPOINTS=http://host3:8001/v1,http://host4:8001/v1 \\
    PYTHONPATH=. ~/envs/evo_fast/bin/python tests/infra/bench_load_balancer.py

    # Control concurrency:
    N_RUNS=2 MUTANTS_PER_RUN=4 ... bench_load_balancer.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import time

from langchain_openai import ChatOpenAI
from loguru import logger
import yaml

from gigaevo.infra.balanced_chat import BalancedChatOpenAI
from gigaevo.infra.endpoint_pool import EndpointPool

# ---------------------------------------------------------------------------
# Configuration from env / infrastructure.yaml
# ---------------------------------------------------------------------------

_INFRA_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "infrastructure.yaml"
)


def _discover_endpoints(server_type: str) -> list[dict]:
    """Read endpoints from infrastructure.yaml."""
    if not _INFRA_PATH.exists():
        return []
    with open(_INFRA_PATH) as f:
        infra = yaml.safe_load(f)
    section = infra.get(server_type, {})
    endpoints = section.get("endpoints", [])
    port = section.get("port")
    result = []
    for ep in endpoints:
        host = ep["host"]
        p = ep.get("port", port)
        result.append({"url": f"http://{host}:{p}/v1", "label": ep.get("label", host)})
    return result


def _get_endpoints(env_key: str, server_type: str) -> list[str]:
    """Get endpoints from env var or infrastructure.yaml."""
    env_val = os.environ.get(env_key)
    if env_val:
        return [u.strip() for u in env_val.split(",") if u.strip()]
    discovered = _discover_endpoints(server_type)
    return [ep["url"] for ep in discovered]


def _check_endpoint(url: str) -> tuple[str, bool, str]:
    """Probe an endpoint. Returns (url, reachable, model_or_error)."""
    import json
    import urllib.request

    try:
        models_url = f"{url}/models"
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
        model = data["data"][0]["id"] if data.get("data") else "unknown"
        return url, True, model
    except Exception as e:
        return url, False, str(e)


# ---------------------------------------------------------------------------
# Realistic prompts (from gigaevo/prompts/)
# ---------------------------------------------------------------------------

# Mutation-style system prompt (shortened but representative length)
_MUTATION_SYSTEM = """You are an expert in evolutionary optimization, focusing on performance-driven mutation of python programs.

Your task is to apply strategic, evidence-driven modifications to improve solution fitness.
You operate within an evolutionary framework where programs are iteratively mutated and evaluated.

AVAILABLE METRICS:
- fitness (higher is better): Overall solution quality score
- accuracy (higher is better): Correctness on validation set
"""

# Realistic user prompt with parent program code (~3KB, typical mutation context)
_MUTATION_USER = """EVOLUTIONARY MUTATION: Adaptive Code Evolution

Transform the program using program insights and historical lineage intelligence.

## INTELLIGENCE INPUTS

**PROGRAM INSIGHTS**:
- threshold_tuning [rigid] (high): The retrieval threshold k=5 is hardcoded. Passages vary in relevance density. Adaptive k based on initial result quality could improve recall by 15-20%.
- query_formulation [beneficial] (medium): Multi-hop query construction using entity extraction shows positive delta +0.05 in recent generations. Preserve and extend this pattern.
- evidence_integration [fragile] (medium): The gap detection in step 5 misses implicit relationships when passages use synonyms for key entities.

**LINEAGE INSIGHTS**:
- strategy: refinement | description: Increased retrieval depth from k=5 to k=10 in third hop improved recall by 8% | delta: +0.08
- strategy: exploration | description: Added synonym expansion for query terms but introduced noise | delta: -0.02
- strategy: imitation | description: Copied entity extraction pattern from top performer, improved first-hop accuracy | delta: +0.05

**EVOLUTIONARY STATISTICS**:
| Generation | Best | Avg | Valid% | Children |
|-----------|------|-----|--------|----------|
| 0 | 0.500 | 0.420 | 85% | 1.2 |
| 1 | 0.550 | 0.460 | 88% | 1.4 |
| 2 | 0.580 | 0.490 | 90% | 1.3 |
| 3 ← | 0.600 | 0.510 | 91% | 1.5 |

## PARENT PROGRAM (fitness=0.600)

```python
def entrypoint():
    return {
        "system_prompt": "You are an expert in multi-hop claim verification...",
        "steps": [
            {"number": 1, "title": "Retrieve first-hop passages", "step_type": "tool",
             "step_config": {"tool_name": "retrieve", "input_mapping": {"query": "$outer_context"}},
             "dependencies": [], "frozen": True},
            {"number": 2, "title": "Summarize first-hop evidence", "step_type": "llm",
             "aim": "Extract claim-specific evidence with citations and identify all gaps.",
             "stage_action": "Break down the claim into key components...",
             "dependencies": [1], "frozen": False},
            {"number": 3, "title": "Generate second-hop query", "step_type": "llm",
             "aim": "Formulate a precise second-hop query targeting missing evidence.",
             "stage_action": "Based on identified gaps, select related gaps that share a common entity...",
             "dependencies": [2], "frozen": False},
            {"number": 4, "title": "Retrieve second-hop passages", "step_type": "tool",
             "step_config": {"tool_name": "retrieve", "input_mapping": {"query": "$history[-1]"}},
             "dependencies": [3], "frozen": True},
            {"number": 5, "title": "Summarize second-hop evidence", "step_type": "llm",
             "aim": "Integrate evidence with citations, note differences, identify remaining gaps.",
             "stage_action": "Compare first-hop and second-hop passages...",
             "dependencies": [2, 4], "frozen": False},
            {"number": 6, "title": "Generate third-hop query", "step_type": "llm",
             "aim": "Formulate precise third-hop query targeting critical missing evidence.",
             "stage_action": "Identify up to two critical missing pieces...",
             "dependencies": [5], "frozen": False},
            {"number": 7, "title": "Retrieve third-hop passages", "step_type": "tool",
             "step_config": {"tool_name": "retrieve_deep", "input_mapping": {"query": "$history[-1]"}},
             "dependencies": [6], "frozen": True},
        ],
    }
```

Select an archetype and produce the mutated program.
Respond with: archetype, justification (2-3 sentences), insights_used (1-3), and the complete code."""

# Chain-style prompt (shorter, simulates validation call)
_CHAIN_SYSTEM = """You are an expert in multi-hop claim verification. Given a claim and retrieved passages, determine if the claim is SUPPORTED or NOT SUPPORTED by the evidence."""

_CHAIN_USER = """Claim: "The Eiffel Tower was designed by Gustave Eiffel's company and completed in 1889 for the World's Fair."

Retrieved passages:
[1] Eiffel Tower | The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
[2] 1889 World's Fair | The 1889 Exposition Universelle was a world's fair held in Paris from 6 May to 31 October 1889. The Eiffel Tower served as the entrance arch to the fair.
[3] Construction History | Construction of the Eiffel Tower began in January 1887 and was completed on 31 March 1889. It was built as the centerpiece of the 1889 World's Fair.

Based on the evidence, is the claim SUPPORTED or NOT SUPPORTED? Explain your reasoning step by step, citing specific passages."""


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


async def _call_llm(client, system: str, user: str, label: str) -> float:
    """Make one LLM call, return latency in seconds."""
    from langchain_core.messages import HumanMessage, SystemMessage

    t0 = time.perf_counter()
    try:
        resp = await client.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        latency = time.perf_counter() - t0
        tokens = getattr(resp, "response_metadata", {}).get("token_usage", {})
        total_tok = tokens.get("total_tokens", "?")
        logger.info(f"[{label}] {latency:.1f}s | {total_tok} tokens")
        return latency
    except Exception as e:
        latency = time.perf_counter() - t0
        logger.error(f"[{label}] {latency:.1f}s ERROR: {e}")
        return latency


async def _clean_redis(pool_name: str, endpoints: list[str], redis_url: str) -> None:
    """Clean Redis state for a pool."""
    pool = EndpointPool(pool_name, endpoints, redis_url=redis_url)
    r = pool._get_async()
    await r.delete(pool._inflight_key)
    for ep in endpoints:
        await r.delete(pool._stats_key(ep))
        await r.delete(pool._cooldown_key(ep))
    await pool.close()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def scenario_static(
    endpoints: list[str],
    model: str,
    system: str,
    user: str,
    n_runs: int,
    calls_per_run: int,
    label: str,
) -> float:
    """Each run pinned to one server (current production behavior)."""

    async def run_batch(run_id: int) -> list[float]:
        ep = endpoints[run_id % len(endpoints)]
        client = ChatOpenAI(
            model=model,
            base_url=ep,
            api_key="none",
            temperature=0,
            max_tokens=1024,
            request_timeout=120,
        )
        return await asyncio.gather(
            *[
                _call_llm(client, system, user, f"{label}-static-r{run_id}-m{i}")
                for i in range(calls_per_run)
            ]
        )

    t0 = time.perf_counter()
    await asyncio.gather(*[run_batch(i) for i in range(n_runs)])
    total = time.perf_counter() - t0
    return total


async def scenario_balanced(
    endpoints: list[str],
    model: str,
    system: str,
    user: str,
    n_runs: int,
    calls_per_run: int,
    label: str,
    redis_url: str,
) -> float:
    """All runs share balanced pool (proposed behavior)."""
    pool_name = f"bench_{label}"
    await _clean_redis(pool_name, endpoints, redis_url)

    clients = [
        BalancedChatOpenAI(
            model=model,
            endpoints=endpoints,
            pool_name=pool_name,
            redis_url=redis_url,
            api_key="none",
            temperature=0,
            max_tokens=1024,
            request_timeout=120,
        )
        for _ in range(n_runs)
    ]

    async def run_batch(run_id: int) -> list[float]:
        return await asyncio.gather(
            *[
                _call_llm(
                    clients[run_id], system, user, f"{label}-balanced-r{run_id}-m{i}"
                )
                for i in range(calls_per_run)
            ]
        )

    t0 = time.perf_counter()
    await asyncio.gather(*[run_batch(i) for i in range(n_runs)])
    total = time.perf_counter() - t0

    # Show distribution
    stats = await clients[0]._pool.get_stats()
    dist = {ep.split("//")[1].split("/")[0]: s["requests"] for ep, s in stats.items()}
    print(f"  Distribution: {dist}")

    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/15")
    n_runs = int(os.environ.get("N_RUNS", "4"))
    mutants_per_run = int(os.environ.get("MUTANTS_PER_RUN", "8"))

    # Discover endpoints
    mutation_eps = _get_endpoints("LLM_ENDPOINTS", "mutation_servers")
    chain_eps = _get_endpoints("CHAIN_ENDPOINTS", "chain_servers")

    if not mutation_eps and not chain_eps:
        print(
            "ERROR: No endpoints found. Set LLM_ENDPOINTS/CHAIN_ENDPOINTS or check infrastructure.yaml"
        )
        sys.exit(1)

    # Probe endpoints
    print("=" * 70)
    print("LLM Load Balancer — Integration Benchmark")
    print("=" * 70)

    if mutation_eps:
        print(f"\nMutation servers ({len(mutation_eps)}):")
        mutation_model = None
        live_mutation = []
        for url, ok, info in [_check_endpoint(u) for u in mutation_eps]:
            status = f"✓ {info}" if ok else f"✗ {info}"
            print(f"  {url} {status}")
            if ok:
                live_mutation.append(url)
                mutation_model = info
        mutation_eps = live_mutation
    else:
        print("\nNo mutation endpoints configured — skipping mutation benchmarks")

    if chain_eps:
        print(f"\nChain servers ({len(chain_eps)}):")
        chain_model = None
        live_chain = []
        for url, ok, info in [_check_endpoint(u) for u in chain_eps]:
            status = f"✓ {info}" if ok else f"✗ {info}"
            print(f"  {url} {status}")
            if ok:
                live_chain.append(url)
                chain_model = info
        chain_eps = live_chain
    else:
        print("\nNo chain endpoints configured — skipping chain benchmarks")

    print(f"\nConfig: {n_runs} runs × {mutants_per_run} calls/run")
    total_calls = n_runs * mutants_per_run

    # --- Mutation benchmark ---
    if len(mutation_eps) >= 2:
        print(f"\n{'=' * 70}")
        print(
            f"MUTATION BENCHMARK: {total_calls} calls ({n_runs} runs × {mutants_per_run})"
        )
        print(
            f"Prompt: ~{len(_MUTATION_SYSTEM) + len(_MUTATION_USER)} chars (realistic mutation context)"
        )
        print(f"{'=' * 70}")

        print("\n[A] Static (each run → 1 server):")
        t_static = await scenario_static(
            mutation_eps,
            mutation_model,
            _MUTATION_SYSTEM,
            _MUTATION_USER,
            n_runs,
            mutants_per_run,
            "mutation",
        )
        print(
            f"  Total: {t_static:.1f}s | Per-call avg: {t_static * 1000 / total_calls:.0f}ms"
        )

        print("\n[B] Balanced (all runs → all servers):")
        t_balanced = await scenario_balanced(
            mutation_eps,
            mutation_model,
            _MUTATION_SYSTEM,
            _MUTATION_USER,
            n_runs,
            mutants_per_run,
            "mutation",
            redis_url,
        )
        print(
            f"  Total: {t_balanced:.1f}s | Per-call avg: {t_balanced * 1000 / total_calls:.0f}ms"
        )

        speedup = t_static / t_balanced if t_balanced > 0 else 0
        print(f"\n  → Mutation speedup: {speedup:.2f}x {'✓' if speedup > 1 else '✗'}")

    # --- Chain benchmark ---
    if len(chain_eps) >= 2:
        print(f"\n{'=' * 70}")
        print(
            f"CHAIN BENCHMARK: {total_calls} calls ({n_runs} runs × {mutants_per_run})"
        )
        print(
            f"Prompt: ~{len(_CHAIN_SYSTEM) + len(_CHAIN_USER)} chars (chain validation)"
        )
        print(f"{'=' * 70}")

        print("\n[A] Static (each run → 1 server):")
        t_static = await scenario_static(
            chain_eps,
            chain_model,
            _CHAIN_SYSTEM,
            _CHAIN_USER,
            n_runs,
            mutants_per_run,
            "chain",
        )
        print(
            f"  Total: {t_static:.1f}s | Per-call avg: {t_static * 1000 / total_calls:.0f}ms"
        )

        print("\n[B] Balanced (all runs → all servers):")
        t_balanced = await scenario_balanced(
            chain_eps,
            chain_model,
            _CHAIN_SYSTEM,
            _CHAIN_USER,
            n_runs,
            mutants_per_run,
            "chain",
            redis_url,
        )
        print(
            f"  Total: {t_balanced:.1f}s | Per-call avg: {t_balanced * 1000 / total_calls:.0f}ms"
        )

        speedup = t_static / t_balanced if t_balanced > 0 else 0
        print(f"\n  → Chain speedup: {speedup:.2f}x {'✓' if speedup > 1 else '✗'}")

    print(f"\n{'=' * 70}")
    print("Done")


if __name__ == "__main__":
    asyncio.run(main())
