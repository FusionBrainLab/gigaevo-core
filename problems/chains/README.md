# CARL Chain Problems

CARL chain problems evolve JSON chain specs rather than Python source files.
Use `program_format=json_document` with the standard guided pipeline:

```bash
python run.py problem.name=chains/hover/full7 \
    pipeline=guided \
    program_format=json_document \
    mutation=carl_with_retrieval_tools \
    memory=none \
    --cfg job
```

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
    memory=full \
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
