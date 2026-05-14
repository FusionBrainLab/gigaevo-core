# gigaevo.monitoring — live in-run instrumentation

Two parallel daemon-thread loops, both started from `run.py` before
`asyncio.run(run_experiment(...))`. Both are gated by a Hydra config
sibling under `cfg.<name>.*` and follow the same shape: a tick-based
loop, fully self-contained per iteration, daemonic (no graceful
shutdown).

| Module | Hydra group | What it emits |
|---|---|---|
| `live_profiler` | `cfg.live_profiler.*` | Re-renders `profile_live.html` from the run log so you can refresh a browser tab while the experiment is mutating. |
| `live_frontier_compare` | `cfg.live_frontier_compare.*` | Periodic comparison snapshot: current-best-vs-frontier-best and current-mean-vs-frontier-mean per tracked metric. Emits via loguru and optionally Telegram. |

Default frontier source for `live_frontier_compare` is `"hof"` — the
current run's hall-of-fame as written to Redis by
`gigaevo.utils.metrics_tracker.MetricsTracker`. Other sources
(`"archive"`, `"reference_set"`) are reserved for future
implementations.
