# GigaEvo Redis Key Schema

Every run uses one Redis DB (0–15), set via `redis.db=N`.
All keys are prefixed with `problem.name` (e.g. `chains/hotpotqa/static`).

---

## Key Categories

### Program storage

| Key pattern | Type | Contains |
|---|---|---|
| `{prefix}:program:{id}` | hash | Program source, metadata, fitness scores |
| `{prefix}:archive` | hash | MAP-Elites archive: bin → best program id |
| `{prefix}:queue` | list | Work queue for exec_runner workers |

### Metrics history

All metrics history keys have the form:
```
{prefix}:metrics:history:{source}:{metric_name}
```
Each is a Redis **list** where each entry is a JSON object:
```json
{"s": <step>, "t": <unix_timestamp>, "v": <value>, "k": "scalar"}
```

The `"s"` field meaning depends on the metric source:

| Source | `"s"` meaning |
|--------|--------------|
| `program_metrics:valid_program_*` | Global program index (0-based) |
| `program_metrics:valid_gen_*` | Generation number |
| `program_metrics:valid_iter_*` | MAP-Elites iteration (= generation) |
| `program_metrics:valid_frontier_*` | Global program index **at time of frontier improvement** |
| `dag_runner:*` | DAG execution step counter |

### Key metrics and what to read them for

| What you want | Key | How to read |
|---|---|---|
| **Current generation** | `valid_iter_fitness_mean` | last entry `"s"` field |
| **Best val fitness** | `valid_frontier_fitness` | last entry `"v"` field |
| **Total programs evaluated** | `program_metrics:programs_total_count` | last entry `"v"` field |
| **Valid programs** | `program_metrics:programs_valid_count` | last entry `"v"` field |
| **Mean fitness this gen** | `valid_gen_fitness_mean` | last entry `"v"` field |

**Common mistake**: using `llen(valid_frontier_fitness)` as the generation count.
This is WRONG — it counts frontier improvements (a small number), not generations.

---

## Canonical Tools — Use These, Don't Query Redis Directly

| Task | Tool | Command |
|---|---|---|
| Live status (gen, fitness, PIDs) | `tools/status.py` | `PYTHONPATH=. python tools/status.py --run prefix@db:label ...` |
| Top N programs | `tools/top_programs.py` | `PYTHONPATH=. python tools/top_programs.py --run prefix@db:label -n 10` |
| Export all data to CSV | `tools/redis2pd.py` | `PYTHONPATH=. python tools/redis2pd.py --run prefix@db:label` |
| Fitness curves plot | `tools/comparison.py` | `PYTHONPATH=. python tools/comparison.py --run prefix@db:label ...` |
| Kill workers + flush | `tools/flush.py` | `PYTHONPATH=. python tools/flush.py --db N [--confirm]` |

**Never write ad-hoc Redis queries** to answer questions these tools already answer.
If a tool gives wrong results, fix the tool — don't work around it with inline Python.

---

## Limitation: Generation Count from Redis

`valid_iter_fitness_mean` last `"s"` is the best Redis-based gen estimate, but it
only updates when a valid program completes. For runs with slow or many-failing
evaluations (e.g. 600-sample), it lags behind the true generation count.

The watchdog uses a more reliable method: counting `"Phase 1: Idle confirmed"` lines
in the run log. `tools/status.py` could be improved to accept a `--log` path and use
the same method. Until then, for accurate gen count on slow runs, check the log:

```bash
grep -c "Phase 1: Idle confirmed" experiments/<name>/run_q.log
```

---

## Per-Experiment Status Script

Each experiment should have `experiments/<name>/run_status.sh` with the correct
`--run`, `--pid`, and `--watchdog` args pre-filled. Always run this — never
construct the status.py invocation from scratch each time.
