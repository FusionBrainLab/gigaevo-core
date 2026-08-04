# hex_range_improve — ImprovEvolve arm (size-generalizing hexagon packing)

One of two matched arms (the other is `problems/hex_range_basic`). The LLM
evolves the modular `generate_config / improve / perturb` class; grading runs
the frozen Stage A/B controller independently at every train size.

## Grading

`validate.py` → `problems._harness.benchmarks.hex_range_grading.grade_range_controller`
(shared with the basic arm — same sandbox, verifier, acceptance rule,
grading seed and per-size wall-clock slice `PER_N_SECONDS`).

- Train sizes: N ∈ {11, 12, 15, 17, 23}, equal slice each.
- fitness = −(mean over valid N of L_N/√N + 1 × #invalid N); all-invalid → −1000.
- Holdout {13, 14, 16} and post-hoc {18–22} are graded ONLY after evolution —
  never add them here.

`metrics.yaml` is byte-identical to `problems/hex_range_basic/metrics.yaml`,
and the task descriptions differ only in the interface block. Keep both
invariants when editing either arm.

## Seed provenance (matched pairs — do not edit)

`initial_programs/gemini1-5.py` are byte-for-byte copies of
`problems/hexagon_improver/initial_programs/gemini1-5.py` — the paper-era
Gemini-3-Pro improver seed pool the published ImprovEvolve numbers started
from. The basic arm's `gemini{i}_direct.py` is the direct-interface port of
the SAME program i (identical optimization kernels), so pair i differs
across arms only in interface — the experiment's treatment variable. Editing
a seed in one arm without its pair breaks the matching.
