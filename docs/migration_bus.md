# Cross-Run Migration Bus

Share rejected-but-valid programs between parallel evolution runs solving the same problem.

## Concept

When a valid program is rejected by one run's archive (niche occupied by a better program), it is published to a shared Redis Stream. Other runs consume the stream, exclusively claim programs via `SETNX`, and import them as orphans into their own archives.

```
Run A:  evaluate -> ingest (priority) -> publish rejects to bus
Run B:  drain bus -> claim exclusively -> import orphans -> select & mutate -> ...
```

Key properties:
- **Async**: no barrier synchronization between runs
- **Exclusive**: SETNX ensures each program is claimed by exactly one run
- **Orphan semantics**: imported programs have empty lineage (no cross-DB parent references)
- **Problem-agnostic**: works with any GigaEvo problem

## Quick Start

```bash
# Terminal 1
python run.py problem.name=vartodd pipeline=mcts_evo migration_bus=bus redis.db=0

# Terminal 2
python run.py problem.name=vartodd pipeline=mcts_evo migration_bus=bus redis.db=2

# Terminal 3
python run.py problem.name=vartodd pipeline=mcts_evo migration_bus=bus redis.db=3
```

All runs share a migration stream on Redis DB 15 (configurable via `migration_bus_db`).

## Configuration

### Presets

| Preset | File | Description |
|--------|------|-------------|
| `disabled` (default) | `config/migration_bus/disabled.yaml` | No bus, normal EvolutionEngine |
| `bus` | `config/migration_bus/bus.yaml` | Fully-connected — accept from any other run |
| `ring` | `config/migration_bus/ring.yaml` | Ring — accept only from predecessor |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `migration_bus_db` | 15 | Redis DB for the migration stream |
| `migration_node.max_buffer_size` | 30 | Max buffered arrivals before dropping |
| `migration_node.consume_interval` | 3.0s | Polling interval |
| `migration_node.max_consume_per_poll` | 20 | Max messages per poll |
| `migration_node.transport.max_stream_len` | 1000 | Redis Stream MAXLEN (approximate) |
| `migration_node.transport.claim_ttl` | 120s | SETNX claim expiry |

### Ring Topology

```bash
python run.py migration_bus=ring \
  migration_bus_ring_ids='["vartodd@db0","vartodd@db2","vartodd@db3"]' \
  redis.db=0
```

## Architecture

```
BusedEvolutionEngine(EvolutionEngine)
  |-- _notify_hook() override -> publish rejects
  |-- step() override -> drain bus before elite selection
  |
MigrationNode (buffer + background poll + orphan conversion)
  |
RedisStreamTransport (XADD + XREAD + SET NX)
  |
Topology (BusTopology | RingTopology)
```

## Monitoring

```bash
# Stream length
redis-cli -n 15 XLEN "gigaevo:<problem>:migration_bus"

# Active claims
redis-cli -n 15 KEYS "gigaevo:<problem>:migration_bus:claim:*" | wc -l

# Look for import/publish log lines
grep "MigrationBus" /tmp/run_*.log
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No imports happening | Runs on different `migration_bus_db` | Ensure all use same DB |
| `[MigrationBus] Buffer full` | Consumer too slow | Increase `max_buffer_size` or decrease `consume_interval` |
| Duplicate imports | Different `stream_key` per run | Ensure `problem.name` matches |
| Claims expiring before processing | `claim_ttl` too short | Increase `claim_ttl` |
