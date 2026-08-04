# hex_range_basic — direct/GigaEvo-baseline arm (size-generalizing hexagon packing)

One of two matched arms (the other is `problems/hex_range_improve`). The LLM
evolves a monolithic `Solver.solve(hex_num, seed)` program; grading runs the
frozen direct evaluator (repeated seeded calls, no incumbent) independently at
every train size.

## Grading

`validate.py` → `problems._harness.benchmarks.hex_range_grading.grade_range_direct`
(shared with the improve arm — same sandbox, verifier, acceptance rule,
grading seed and per-size wall-clock slice `PER_N_SECONDS`).

- Train sizes: N ∈ {11, 12, 15, 17, 23}, equal slice each.
- fitness = −(mean over valid N of L_N/√N + 1 × #invalid N); all-invalid → −1000.
- Holdout {13, 14, 16} and post-hoc {18–22} are graded ONLY after evolution —
  never add them here.

`metrics.yaml` is byte-identical to `problems/hex_range_improve/metrics.yaml`,
and the task descriptions differ only in the interface block. Keep both
invariants when editing either arm.

## Seed provenance (matched pairs — do not edit)

`initial_programs/gemini1-5_direct.py` derive from
`problems/hexagon_pack_general/initial_programs/gemini1-5_direct.py` — the
pre-existing direct-interface ports of the SAME five paper-era Gemini-3-Pro
improver programs the improve arm seeds with verbatim
(`problems/hexagon_improver/initial_programs/gemini1-5.py`). Identical
optimization kernels; pair i differs across arms only in interface — the
experiment's treatment variable.

Exactly two mechanical edits were applied for this harness (transformer with
asserted-unique replacements; kernels untouched):

1. `_DIRECT_TIME_BUDGET_S` 240 → 20 s — the 240 was tuned to the old 300 s
   single-call budget; 20 fits the ~30 s per-size slice.
2. A thin `Solver` class + `entrypoint()` returning it, matching the frozen
   direct contract (`solve(hex_num, seed)` called repeatedly by
   `DirectEvaluator`).

Editing a seed in one arm without its pair breaks the matching.
