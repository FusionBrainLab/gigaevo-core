# CARL Chain Problems

CARL chain problems evolve JSON chain specs rather than Python source files.
All recipes use `program_format=json_document`. Start from the
[recommended defaults](#recommended-defaults-2026-07-all-in-closeout) below;
the later sections are comparability recipes for the older 1D-archive
baselines.

## Recommended Defaults (2026-07 ALL-IN closeout)

These defaults come from the two-pair ALL-IN A/B on
`chains/hover/full7_vectorized` with **Qwen3-235B-A22B-Thinking-2507 as the
mutation LLM** (via the internal LiteLLM proxy; all evidence numbers below are
under that mutator). Closeout: `experiments/hover/diff_memory/JOURNAL.md`,
2026-07-12. The executable form of this recipe, including endpoint smoke
checks and treatment self-checks, is
`experiments/hover/diff_memory/launch_bd3d_noise_allin.sh`.

```bash
# Routes — see "Local LiteLLM Setup" below for the env vars:
#   LITELLM_BASE_URL, CARL_MUTATION_MODEL (Qwen3-235B-A22B-Thinking-2507),
#   HOVER_CHAIN_URL / HOVER_CHAIN_MODEL (Qwen/Qwen3-8B executor).
CHAIN_DEFAULTS=(
    storage=disk
    problem.name=chains/hover/full7_vectorized   # vectorized problem: per-sample score vectors
    program_format=json_document
    mutation=carl_with_retrieval_tools
    llm=single
    llm_base_url="$LITELLM_BASE_URL"
    model_name="$CARL_MUTATION_MODEL"
    algorithm=chains_bd3d                        # 3D strategy-space archive
    enable_chain_structural_metrics=true         # required by chains_bd3d
    archive_selector=paired_bootstrap            # paired per-sample acceptance gate
    num_parents=1
    max_mutants=250
    max_tokens=60000
    stage_timeout=7200
    dag_timeout=14400
)

# Memory ON (recommended exploratory default). The memory bank goes to the
# default checkpoint dir under the run's output dir; override checkpoint_dir
# only when a bank must live at a specific path (e.g. shared across tooling).
# `memory/write=live` is the shipped memory=v2 default (same-run read+write);
# it is passed explicitly here to mirror the launcher. Crediting is internal to
# memory=v2 (the v1 memory/crediting group is gone).
python run.py "${CHAIN_DEFAULTS[@]}" pipeline=memory_guided \
    memory=v2 memory/write=live memory/llm=qwen_instruct \
    memory.llm.models.0.base_url="$LITELLM_BASE_URL"

# Memory OFF (control arm — keep one in any A/B):
python run.py "${CHAIN_DEFAULTS[@]}" pipeline=guided memory=none
```

Headline result (Qwen-235B mutator, 250 mutants/run): the recipe's best
program reached **62.7 ± 1.9 TEST hard** (K=5), beating all prior
Qwen-mutator finals (56.7–59.0) and the best no-memory arm by +7.3 pts; on
val it re-evaluated to a statistical tie with the prior champions
(0.8229 ± 0.0097 vs 0.8298/0.8291, n.s.). The mutation LLM is a swappable
lever — `llm=gemini35_flash` replaces the three `llm=single`/URL/model
overrides (see `config/llm/gemini35_flash.yaml`).

Why each piece (evidence from the ALL-IN campaign, N=2 MEM/NOMEM pairs):

- **Vectorized problem** — stores a per-sample score vector per program; this
  enables the paired gate, paired crediting, and MMR winner selection. For a
  new task, create a vectorized problem variant whose `validate()` returns
  per-sample scores (see `problems/chains/hover/full7_vectorized/`).
- **`chains_bd3d`** — 27–41 occupied cells vs 13 with the 1D archive; the
  campaign-best program came from a hop-depth niche the 1D archive never held.
- **`paired_bootstrap` selector** — single-eval fitness noise is σ≈0.008; the
  paired gate re-adjudicates each challenger vs the cell incumbent on shared
  samples (949 decisions, 0 fallbacks across the campaign).
- **Memory ON** — split 1/1 on within-pair val MMR, but the campaign-best
  program on held-out TEST (+7.3 pts hard over best no-memory) was a
  memory-arm product and paired crediting was verified live. Treat it as the
  exploratory default and keep a no-memory control arm; known caveat: card
  injection ran above the 30–55% design band (58–62%).

Winner selection and reporting protocol (do NOT rank by single-eval fitness):

1. Pool candidates across arms and MMR-rank them: pairwise P(A beats B) via
   paired bootstrap on the stored score vectors, Bradley–Terry fit
   (`experiments/hover/diff_memory/mmr_top10.py`).
2. Re-evaluate finalists K=5 on val with per-claim vectors for headline
   numbers and paired significance tests
   (`experiments/hover/diff_memory/reeval_winners_vec.py`).
3. Report held-out TEST (hard metric for HoVer) with a seed-program anchor
   evaluated under the same protocol.

`memory=none` is the no-memory baseline: no cards are read and no memory writer
runs. Disk storage uses the normal run-local default
`${hydra:runtime.output_dir}/storage`.

## Local LiteLLM Setup

There are two LLM routes in the HoVer CARL runs:

| Route | Used for | Configure with |
|---|---|---|
| Mutation LLM | Writes new chain diffs | `llm=single`, `llm_base_url`, `model_name` |
| Chain executor | Runs each chain during validation | `HOVER_CHAIN_URL`, `HOVER_CHAIN_MODEL` |

```bash
export OPENAI_API_KEY=${OPENAI_API_KEY:-sk-local}
export LITELLM_BASE_URL=${LITELLM_BASE_URL:-http://127.0.0.1:4000/v1}
export CARL_MUTATION_MODEL=${CARL_MUTATION_MODEL:-Qwen3-235B-A22B-Thinking-2507}
export HOVER_CHAIN_URL=${HOVER_CHAIN_URL:-$LITELLM_BASE_URL}
export HOVER_CHAIN_MODEL=${HOVER_CHAIN_MODEL:-Qwen/Qwen3-8B}
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
```

## HoVer Tool-Diff Run

This is the no-memory form of the tool-aware CARL diff experiment used by
the Qwen diff HoVer runs. It is configured for comparability with the previous
Qwen diff baseline by relying on the standard defaults: the archive is
one-dimensional on `fitness`, uses the bounded-gap fitness selector from
`single_island_no_distant_parents`, and does not add chain-topology behavior
metrics.

```bash
CARL_ARGS=(
    problem.name=chains/hover/full7
    pipeline=guided
    program_format=json_document
    memory=none
    mutation=carl_with_retrieval_tools
    llm=single
    llm_base_url="$LITELLM_BASE_URL"
    model_name="$CARL_MUTATION_MODEL"
    num_parents=1
    max_mutants=250
    max_tokens=60000
    stage_timeout=7200
    dag_timeout=14400
)

# Config-only smoke check; does not spend LLM tokens.
python run.py "${CARL_ARGS[@]}" --cfg job

# Launch after the config resolves as expected.
python run.py "${CARL_ARGS[@]}"
```

`max_tokens` is the per-call completion budget for the mutation LLM route. It
does not cap total experiment spend, validation/executor calls, or the number
of mutants; `max_mutants` is the run budget. For thinking models this budget may
include hidden reasoning tokens plus the final structured diff.

The resolved config should show:

```text
pipeline.id: guided
program_format.feature: JsonDocumentEvaluationFeature
program_loader.pattern: '*.json'
memory.provider: NullMemoryProvider
mutation: StructuredDiffMutationOperator + AllowedToolChainChanges
behavior_space.keys: ['${primary_key}']
elite_selector: FitnessProportionalTournamentBoundedGapEliteSelector
enable_chain_structural_metrics: false
```

Stored program metrics should be exactly:

```text
fitness
is_valid
n_steps
n_tool_steps
```

`loader.pattern` is only the Hydra override alias provided by
`config/loader/directory.yaml`; the instantiated loader field to check in the
resolved config is `program_loader.pattern`.

## HoVer Tool-Diff With Memory

Use `pipeline=memory_guided` when cards should be read before mutation and
written after evaluation. Keep the same fitness-only archive as the no-memory
baseline when you want the run to be comparable to the previous Qwen diff runs;
the treatment difference is then memory, not a different MAP-Elites behavior
space.

```bash
python run.py \
    problem.name=chains/hover/full7 \
    pipeline=memory_guided \
    program_format=json_document \
    memory=v2 \
    memory/write=live \
    memory/llm=qwen_instruct \
    checkpoint_dir="$PWD/SHARE_HOVER_DIFF_MEMORY" \
    mutation=carl_with_retrieval_tools \
    llm=single \
    llm_base_url="$LITELLM_BASE_URL" \
    model_name="$CARL_MUTATION_MODEL" \
    memory.llm.models.0.base_url="$LITELLM_BASE_URL" \
    num_parents=1 \
    max_mutants=250 \
    max_tokens=60000 \
    stage_timeout=7200 \
    dag_timeout=14400 \
    --cfg job
```

For two-copy memory launches with endpoint smoke checks and independent memory
banks, use the experiment launcher only after confirming its `COMMON_ARGS`
match the comparable settings above:

```bash
OPENAI_API_KEY=sk-gigaevo ./experiments/hover/diff_memory/launch.sh
```

`launch.sh` pins the 3d topology archive (`algorithm=topology_3d_ret`). For the
fitness-only archive that mirrors the no-memory baseline, use
`./experiments/hover/diff_memory/launch_baseline_memory.sh` instead.

## Summarizer DAG-Diff Run

The older LLM-step summarizer experiment uses the same no-memory JSON setup but
a different problem and diff schema:

```bash
python run.py \
    problem.name=chains/summarizer \
    pipeline=guided \
    program_format=json_document \
    memory=none \
    mutation=structured_diff_chains \
    llm=single \
    llm_base_url="$LITELLM_BASE_URL" \
    model_name="$CARL_MUTATION_MODEL" \
    max_mutants=10 \
    max_tokens=60000 \
    --cfg job
```

That config should resolve to `StructuredDiffMutationOperator` with
`AllowedDagChanges`.
