# GigaEvo CLI

`gigaevo` is the supported interface for inspecting runs, exporting program
data, plotting results, and operating experiment manifests. Run
`gigaevo --help` for the command list and `gigaevo COMMAND --help` for the
authoritative options of a command.

## Install

```bash
pip install -e .
gigaevo --help
```

Plotting and profiler commands require the plotting extra:

```bash
pip install -e ".[plotting]"
```

## Argument Order

Global options must appear before the command. Command-specific options appear
after it.

```bash
# Correct: -r and global -f are before the command.
gigaevo -r chains/hover/full7@4:run-a -f json trajectory --tail 20

# Correct: logs -f is the command-local --follow flag.
gigaevo -e hover/my-experiment logs run-a -f
```

The global target options are mutually exclusive:

- `-e/--experiment TASK/NAME` loads Redis runs from
  `experiments/TASK/NAME/experiment.yaml`. The manifest launcher selects
  `storage=redis` because its run identity is a Redis prefix and DB.
- `-r/--run SPEC` targets a run directly. It is repeatable and can refer to
  Redis or disk storage.

Commands such as `flush` and `inspect` take their own Redis DB options and do
not require `-e` or `-r`.

## Run Specifications

### Redis

| Form | Example | Meaning |
|---|---|---|
| `prefix@db:label` | `chains/hover/full7@4:run-a` | Explicit prefix, DB, and display label |
| `prefix@db` | `chains/hover/full7@4` | Label defaults to `prefix@db` |
| `db:label` | `4:run-a` | Discover the prefix in DB 4 |
| `db` or `@db` | `4` or `@4` | Discover the prefix; use the default label |

Prefix discovery scans instance-lock, run-state, and program keys, so it also
works after a cleanly stopped run has released its lock. It fails when the DB
has no discovered prefix or more than one. Use `gigaevo inspect --db 4` and
then pass the full `prefix@db` form when a DB is ambiguous.

### Disk

Disk specs point to a Hydra output directory from a `storage=disk` run, its
`storage` directory, or directly to its one prefix directory:

```text
outputs/run/storage/
  chains_hover_full7/
    programs/
    archives/
```

Accepted forms:

| Form | Example |
|---|---|
| Absolute path | `/data/runs/run-a/storage` |
| Relative path containing `/` | `outputs/run-a/storage` |
| Explicit relative path | `./storage` or `../run-a/storage` |
| Path with display label | `outputs/run-a/storage:run-a` |
| Hydra output directory | `outputs/run-a` |
| Direct prefix directory | `outputs/run-a/storage/chains_hover_full7` |

When the storage root is given, it must contain exactly one directory with a
`programs/` child. If it contains several prefixes, point `-r` directly at the
desired prefix directory. Disk access is read-only and does not take the
writer lock.

### Labels

Labels identify runs in tables, plot legends, positional export filters, and
multi-run output filenames. Labels must be unique within one command. Add an
explicit `:label` when two disk paths resolve to the same prefix name.

For multi-run exports, labels are made filesystem-safe. For example, the
default Redis label `chains/hover/full7@4` becomes
`chains_hover_full7@4` in the filename while remaining unchanged in output
metadata.

## Backend Support

| Command | Redis | Disk | CSV/log files | Notes |
|---|:---:|:---:|:---:|---|
| `top` | yes | yes | no | Reads stored programs |
| `export csv` | yes | yes | no | Reads stored programs |
| `export frontier` | yes | yes | no | Reads stored programs |
| `plot comparison` | yes | yes | yes | CSV input uses `--from-csv` |
| `plot trajectory` | yes | yes | no | Reads stored programs |
| `plot arms-race` | yes | yes | no | Requires unique paired labels |
| `status` | yes | no | no | Uses Redis engine state and metrics |
| `trajectory` | yes | yes | no | Reads persisted metric histories |
| `metrics` | yes | yes | no | Reads persisted metric histories |
| `checkpoint` | yes | no | no | Read-only one-shot status snapshot |
| `logs` | no | no | yes | Reads experiment log files |
| `events` | no | no | yes | Audits and plots canonical log events |
| `profiler` | no | no | yes | Profiles run logs |
| `manifest`, `launch` | yes | no | manifest/logs | Experiment control plane |
| `watchdog` | yes | no | logs | Legacy scheduled monitoring |
| `inspect`, `flush` | yes | no | no | Direct Redis operations |

`status` and `checkpoint` reject disk paths because disk program storage does
not contain their live PID and engine-state contract. `trajectory` and
`metrics` read the standard `<run-dir>/metrics/*.jsonl` files next to
`<run-dir>/storage`. If a run overrides either storage root independently,
target Redis or inspect the custom metrics directory directly.

## Common Workflows

### Inspect Programs

```bash
# Redis
gigaevo -r chains/hover/full7@4:run-a top -n 5 --code

# Disk
gigaevo -r outputs/run-a/storage:run-a top -n 5 --code

# Lower metric values are better
gigaevo -r outputs/run-a/storage top --metric loss --minimize
```

### Export Data

```bash
# One run writes the exact path.
gigaevo -r chains/hover/full7@4:run-a export csv -o data/run-a.csv

# Several runs fan out to data/runs_run-a.csv and data/runs_run-b.csv.
gigaevo \
  -r outputs/run-a/storage:run-a \
  -r outputs/run-b/storage:run-b \
  export csv -o data/runs.csv

# Cumulative best-by-generation frontier for a minimization metric.
gigaevo -r outputs/run-a/storage \
  export frontier --metric loss --minimize -o data/frontier.csv
```

`export frontier` computes the best value within each generation and then the
cumulative best through that generation. Maximization is the default; pass
`--minimize` when lower values are better.

### Plot Runs

```bash
gigaevo \
  -r outputs/run-a/storage:run-a \
  -r chains/hover/full7@4:run-b \
  plot comparison -o plots/ --metric fitness

gigaevo -r outputs/run-a/storage \
  plot trajectory -o plots/loss --metric loss --minimize

gigaevo plot comparison \
  --from-csv data/run-a.csv:run-a \
  --from-csv data/run-b.csv:run-b \
  -o plots/
```

`--from-csv` is mutually exclusive with `-e` and `-r`. Its input must use the
schema produced by `gigaevo export csv`, including `iteration` and the selected
`metric_<name>` column.

### Inspect Run Metrics

```bash
gigaevo -e hover/my-experiment status
gigaevo -r chains/hover/full7@4:run-a trajectory --tail 20
gigaevo -r chains/hover/full7@4:run-a metrics --tag "valid/frontier/*"
gigaevo -r outputs/run-a trajectory --tail 20
gigaevo -r outputs/run-a metrics --tag "valid/frontier/*"
gigaevo -r chains/hover/full7@4:run-a checkpoint
```

`trajectory --tail N` keeps the last N rows for every selected run and metric.
The frontier history is already canonical, so the command does not assume that
the metric is maximized.

`checkpoint` is a read-only Redis status snapshot. Scheduled watchdog
monitoring is legacy functionality and is not required for normal CLI use.

### Logs and Profiling

```bash
gigaevo -e hover/my-experiment logs
gigaevo -e hover/my-experiment logs run-a -f
gigaevo logs --file outputs/run-a/run.log -n 100
gigaevo profiler --file outputs/run-a/run.log --out-dir reports/run-a
```

### Manifest and Lifecycle

```bash
gigaevo -e hover/my-experiment manifest get contract.runs -f json
gigaevo -e hover/my-experiment manifest update lifecycle.status running
gigaevo -e hover/my-experiment manifest gate running
gigaevo -e hover/my-experiment manifest record-pids \
  --pids-file /tmp/pids --labels run-a,run-b
gigaevo -e hover/my-experiment launch
```

Available manifest operations are listed by `gigaevo manifest --help`; there
is no separate `manifest set` command.

## Output Formats

The global `-f/--format` option supports `table`, `json`, `csv`, and
`markdown`. It must be placed before the command. Commands that also expose a
local `-f/--format` accept it after the command as an override.

```bash
gigaevo -r chains/hover/full7@4 -f json top -n 1
gigaevo -r chains/hover/full7@4 top -n 1 -f json
```

## Troubleshooting

**A disk path was parsed as Redis**

Use a path containing `/`, such as `outputs/run/storage`, or prefix a local
name with `./`, such as `./storage`.

**A disk root contains several prefixes**

Point directly at one prefix directory:

```bash
gigaevo -r outputs/run/storage/chains_hover_full7 top
```

**Duplicate run label**

Add explicit unique labels:

```bash
gigaevo -r outputs/a/storage:a -r outputs/b/storage:b top
```

**Global option rejected after a command**

Move `-e`, `-r`, global `-f`, `--redis-host`, or `--redis-port` before the
command name.

**Redis DB has multiple prefixes**

```bash
gigaevo inspect --db 4
gigaevo -r exact/prefix@4:run-a status
```

## Lazy Loading

The root command keeps a static command-help registry, so `gigaevo --help`
does not import every command module. `LazyGroup.get_command()` imports only
the selected command. Each `_LAZY_SUBCOMMANDS` entry contains its module,
attribute, and root-help text in one typed record; tests enforce that root help
remains lazy.
