# LLM Load Balancer

Redis-coordinated load balancer that distributes LLM requests across multiple server endpoints. Shared by all concurrent experiment runs.

## Problem

Each experiment run is statically pinned to one LLM server. With 4 runs and 4 servers, each run uses only its assigned server. Since runs spend 93-95% of time in chain validation, mutation servers sit idle most of the time.

## Solution

`BalancedChatOpenAI` — a drop-in `ChatOpenAI` replacement that routes each request to the least-loaded, fastest server across all runs.

## Quick Start

Wire the load-balanced router into your experiment file by selecting
``BalancedChatOpenAIConfig`` from the LLM schemas (or build it via
``llm_presets``); endpoints are then listed once on that config and
every run shares the pool. No per-run ``base_url`` override needed.

## How It Works

```
Run D1 calls BalancedChatOpenAI.ainvoke("mutate this program...")
  │
  ├─ 1. EndpointPool.acquire()  [~0.05ms, atomic Lua script on Redis DB 15]
  │     Read inflight counts + EMA latency for all endpoints
  │     Score = inflight × 1000 + ema_latency_ms
  │     Pick lowest score (random tiebreak among ties)
  │     HINCRBY +1 atomically
  │
  ├─ 2. Delegate to ChatOpenAI(base_url=selected_endpoint)
  │     [15-60s for mutation, 1-5s for chain]
  │
  └─ 3. On completion:
        HINCRBY -1 (release)
        Update EMA latency: ema = 0.3 × new + 0.7 × old
        Record metrics (request_count, latency_ms, errors)
```

All runs share the same Redis keys → full cross-run visibility.

## Routing Logic

Selection uses a **weighted score** per endpoint:

```
score = inflight_count × LATENCY_WEIGHT + ema_latency_ms
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `LATENCY_WEIGHT` | 1000 | 1 extra inflight ≈ 1s extra latency penalty |
| `EMA alpha` | 0.3 | Recent 3-5 requests dominate the average |
| `cooldown_secs` | 60 | How long to skip a failed endpoint |

**Example**: Server A has 0 inflight but EMA of 5000ms (slow). Server B has 2 inflight but EMA of 100ms (fast). Scores: A=5000, B=2100. Server B wins despite higher load.

**Adaptation**: When a server slows down (e.g., test eval running on it), its EMA rises within 3-5 requests. Traffic automatically diverts to faster servers. When it recovers, EMA drops and traffic returns.

## Redis Keys (DB 15)

```
llm_pool:{pool}:inflight              Hash: {endpoint_url → count}
llm_pool:{pool}:cooldown:{url_hash}   String with TTL (marks unhealthy)
llm_pool:{pool}:stats:{url_hash}      Hash: {requests, errors, total_latency_ms, ema_latency_ms}
```

Inspect with:
```bash
redis-cli -n 15 HGETALL llm_pool:mutation:inflight
redis-cli -n 15 HGETALL llm_pool:mutation:stats:$(python -c "import hashlib; print(hashlib.sha256(b'http://10.226.72.211:8777/v1').hexdigest()[:12])")
```

## Configuration

### config/llm/balanced.yaml

```yaml
llm:
  _target_: gigaevo.llm.models.MultiModelRouter
  models:
    - _target_: gigaevo.infra.balanced_chat.BalancedChatOpenAI
      model: ${model_name}
      api_key: ${oc.env:OPENAI_API_KEY}
      temperature: ${temperature}
      max_tokens: ${max_tokens}
      pool_name: "mutation"
      redis_url: "redis://localhost:6379/15"
      cooldown_secs: 60
      endpoints:
        - "http://10.226.72.211:8777/v1"
        - "http://10.226.15.38:8777/v1"
        - "http://10.226.185.47:8777/v1"
        - "http://10.225.51.251:8777/v1"
  probabilities: [1.0]
```

To add/remove servers: edit the `endpoints` list. No code changes needed.

## Metrics

Per-endpoint metrics written via LogWriter (same stream as token tracking):

```
pool/{pool_name}/{endpoint_label}/request_count
pool/{pool_name}/{endpoint_label}/error_count
pool/{pool_name}/{endpoint_label}/latency_ms
pool/{pool_name}/{endpoint_label}/avg_latency_ms
```

## Benchmarks

Tested with real production servers (Qwen3-235B mutation, Qwen3-8B chain):

| Scenario | Static | Balanced | Speedup |
|----------|--------|----------|---------|
| 8 concurrent long-output mutations (1 run) | 59.8s | 32.4s | **1.85×** |
| Chain with 1 overloaded server | 69.4s | 24.8s | **2.80×** |
| Short output, all healthy (worst case) | 12.3s | 13.3s | 0.92× |

## Architecture

```
gigaevo/infra/
  endpoint_pool.py     # EndpointPool — Lua-based atomic selection + EMA
  balanced_chat.py     # BalancedChatOpenAI — drop-in ChatOpenAI wrapper
  pool_metrics.py      # PoolMetricsTracker — per-endpoint observability

config/llm/
  balanced.yaml        # Hydra config group

tests/infra/
  test_endpoint_pool.py   # 26 tests (selection, cooldown, EMA, cross-run)
  test_balanced_chat.py   # 12 tests (invoke, failover, structured output, metrics)
  bench_load_balancer.py  # Integration benchmark (auto-discovers servers)
```
