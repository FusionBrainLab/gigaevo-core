# Design — Looping analysis for output-only autoresearch on the Sequential-Maze

Date: 2026-06-25
Status: design (approved in brainstorm; pending spec review)
Scope owner: maze benchmark / autoresearch analytics

## 1. Motivation & research question

The Sequential-Maze is provably hard for off-the-shelf optimizers: only a sequential
path-follower reaches the global, and every classical black-box method stalls near the
mouth (certified by `gate.py`; figures in `docs/audits/maze_benchmark/`). That makes it an
ideal probe for a different question about *automated research itself*:

> When an output-only autoresearch engine attacks a deceptive task, does it **loop** —
> cycle through approaches without ever advancing toward the qualitatively-new idea
> (here, sequential path-following) — and can we detect and characterize that looping
> automatically, post hoc, from a run trace?

"Output-only" means the thing being evolved (an optimizer) can only probe the objective's
scalar output — it never sees the landscape internals. This is already enforced for evolved
optimizers via the `counted` closure in `validate.py`.

We are **not** building or installing an autoresearch engine here. The run can be produced
by anything (GigaEvo, a Karpathy-style loop, a hand-rolled script). We are building the
**engine-agnostic analytics layer** that ingests a trace and reports looping.

### Success criteria (verifiable)
1. On synthetic traces with *injected, known* loop structure, the analyzer labels the loop
   type correctly.
2. On a *healthy* (monotone-progress) trace, it reports **no loop** — no false positive.
3. On a real multi-attempt trace (classical-battery replay, §6) it produces an
   interpretable verdict plus plots.

## 2. Operational definition of "looping"

**Looping = high behavioral recurrence WITH flat best-progress.**

The progress axis is load-bearing: it separates *looping/trapped* (recurrent at a
non-solution) from *converged* (also recurrent, but at maximum progress). A run that keeps
returning to the same behavior because it solved the task is not looping.

"Progress" in v1 is the privileged maze `progress()` (available because we use the
`MazeDescriptor`). When the task-agnostic `GeometricDescriptor` is later swapped in,
best-so-far *score* substitutes for progress in every "flat-progress" rule below.

Two signals only (per the brainstorm), both derivable from probing output:
- **Behavioral footprint** — what each attempt *did*: the points it queried on the
  objective and the values it got.
- **Score trajectory** — the scalar output / score series over research-time.

We deliberately do **not** use artifact (code/idea) semantics or lineage-graph structure in
v1.

## 3. Architecture (four units behind seams)

The analyzer consumes a **trace file**, never the engine. Each engine needs only a thin
adapter that emits the schema. Units are isolated and independently testable.

```
trace (JSONL)  ->  Descriptor  ->  Detectors  ->  Taxonomer  ->  Report
   capture.py      descriptors.py   recurrence.py   taxonomy.py    report.py
                                    triage.py
```

### 3.1 Trace schema (`trace.py`) — the contract
JSONL. One run header line, then one record per attempt:

```jsonc
{"type":"run","engine":"<name>","task":"maze_insane","seed":0,"dim":12,
 "budget":7000,"global_min_value":<float>}
{"type":"attempt","i":0,"queries":[[x0,...,x_{d-1},f], ...],
 "returned_x":[...],"score":<float in [0,1]>}
```

- `queries` is the per-attempt footprint (length ≤ budget). For large runs the writer may
  downsample queries to a cap (e.g. keep every k-th plus the running argmin); the cap is
  recorded in the run header so descriptors stay comparable.
- `score` is whatever the engine optimized (for the maze: path-based `progress`-derived
  fitness once the maze branch is wired into `validate.py`; for raw replay we record the
  privileged `progress` directly).
- Loaded into frozen dataclasses `RunHeader`, `Attempt`, `Footprint`.

### 3.2 Descriptor (`descriptors.py`) — pluggable
```python
class Descriptor(Protocol):
    def describe(self, fp: Footprint) -> DescriptorResult: ...
# DescriptorResult: vector: np.ndarray  (comparable, fixed-length)
#                   labels: dict        (interpretable, human-readable fields)
```
- **v1: `MazeDescriptor`** (privileged — fine, we are the post-hoc analyst, not the
  optimizer). Using the maze tree / `progress()`, it maps each queried point to a
  (path-depth, basin id) and summarizes the footprint as:
  `[max_path_depth, basin_visit_histogram, terminal_basin_id, best_f, query_spread]`.
- **Later swap-in (seam only): `GeometricDescriptor`** — task-agnostic, from the raw query
  cloud (centroid, spread, hull volume, best f, improvement rate, query entropy, terminal
  point). Documented, not implemented in v1.

### 3.3 Detectors
- **`recurrence.py` — RecurrenceAnalyzer (RQA).** Input: descriptor vectors `T×d` ordered
  by research-time. Normalize, compute the pairwise distance matrix, threshold at a
  recurrence radius ε → recurrence matrix R. Metrics: recurrence rate (RR), determinism
  (DET), mean diagonal-line length (L), trapping time (TT), laminarity (LAM). These are the
  standard RQA quantities for *cycling* (diagonal structure) and *being stuck* (vertical
  structure).
- **`triage.py` — ScoreTriage.** From the scalar series: best-so-far curve, plateau length,
  steps-since-last-improvement, and the max-progress-over-time curve. Cheap, universal, and
  the no-progress gate for the taxonomy. Always on.

### 3.4 Taxonomer (`taxonomy.py`) — verdict
Combines RQA + triage + descriptor labels into one verdict label + evidence:

| Label | Operational rule (calibrated thresholds, §5) |
|---|---|
| `healthy` | max-progress rising over the run, or final max-progress ≈ 1; recurrence only in the high-progress tail |
| `frozen` | RR very high, behavioral novelty ≈ 0 (descriptors near-identical), progress flat |
| `archetype-orbit` | DET high with ≥2 distinct recurring behavior clusters visited cyclically, progress flat |
| `decoy-capture` | terminal-basin distribution dominated by one non-global (decoy) basin (share ≥ τ), max-progress < 1 and flat |
| `diffuse-thrash` | RR low, high novelty, progress flat (the wandering contrast class) |

`decoy-capture` requires the privileged descriptor; the others work on any descriptor.

### 3.5 Report (`report.py`)
Markdown verdict + evidence table, plus PNGs: the **behavior recurrence plot**, the
**basin-visit timeline** (attempt index × basin, colored by path-depth), and the
**best-progress curve**. Styled like the existing maze figures. A YouTrack page is an
optional later wrapper, not part of v1.

### 3.6 Capture adapter (`capture.py`)
Engine-agnostic helper that wraps an objective so every call is logged, and writes the
schema. v1 provides: (a) `counted`-style wrapping usable from `validate.py`, and (b) a
**classical-battery replay** that runs the existing `benchmark.py` methods in sequence on an
instance and emits a real multi-attempt trace (no LLM, fully reproducible — used by §6).

### 3.7 CLI
One thin entry: `analyze <trace.jsonl> [--descriptor maze] -> report dir`. Because this is a
user-facing command, `tools/README.md` is updated in the same change (canonical-doc rule).

## 4. Module layout (co-located with the problem)
```
problems/sequential_landscape/looping/
  __init__.py
  DESIGN.md          (this file)
  trace.py           schema dataclasses + JSONL read/write
  descriptors.py     Descriptor Protocol + MazeDescriptor
  recurrence.py      RQA
  triage.py          score/progress triage
  taxonomy.py        LoopTaxonomer + verdict
  report.py          markdown + PNG
  capture.py         objective wrapper + classical-battery replay
  cli.py             thin CLI
tests/problems/sequential_landscape/test_looping_analysis.py
```

## 5. Calibration (no magic constants)
ε (recurrence radius), plateau/flatness thresholds, novelty cutoff, and τ (decoy-share) are
**fit on the validation set**, not hard-coded blind:
- ε defaults to a percentile of the pairwise descriptor-distance distribution.
- progress-flatness / plateau thresholds are chosen so the *healthy* trace is never flagged
  and the *injected* loops always are.
The chosen defaults ship with a small calibration note recording how they were derived, so
they are auditable and can be re-derived if the descriptor changes.

## 6. Validation & testing (TDD)
- **Unit (RED→GREEN).** Pure factory functions build ground-truth traces:
  `make_oscillation_trace` (alternates two decoys), `make_frozen_trace`,
  `make_progress_trace` (monotone, healthy), `make_thrash_trace` (high novelty, no
  progress). Assert each detector's quantities and the taxonomer's label, **including the
  no-false-positive case** on the healthy trace.
- **Integration on real footprints.** Classical-battery replay on `maze_insane` →
  `capture.py` → analyzer; assert verdict is `archetype-orbit` with flat progress. Needs no
  LLM and no installed autoresearch engine, so it is reproducible now.
- Lint clean (ruff); tests run via `/run-tests` targeting the new test file.

## 7. Scope & non-goals (v1)
**In:** trace schema, capture adapter + classical replay, `MazeDescriptor`, RQA + score
triage, taxonomer, markdown+PNG report, thin CLI, the test suite above.

**Out:** the task-agnostic `GeometricDescriptor` (seam only), artifact/code semantics,
lineage-graph analysis, any *intervention* / loop-breaking feedback into a live engine,
installing an autoresearch engine, and a YouTrack page.

## 8. Open dependencies
- The live `validate.py` currently scores the equal-volume **chain** ladder, not the maze +
  path-based `progress()`. The analytics layer does not require that wiring (it reads a
  trace and can record privileged `progress` directly in replay), but a *real GigaEvo run*
  against the maze does. That wiring is tracked separately (`plans/2026-06-25-insane-maze-landscape.md`).
