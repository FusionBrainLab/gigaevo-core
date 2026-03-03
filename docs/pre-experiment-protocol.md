# Pre-Experiment Protocol

**Version**: 1.0 (2026-03-04)
**Status**: Mandatory. Every GigaEvo experiment must complete all phases in order.
**Rationale**: Three successive launch bugs (repr-contamination, missing prompts_dir × 2) were
caused by insufficient pre-launch checks. This protocol exists to make those bugs impossible to
repeat.

---

## Phase 0: Pre-Registration

Must be complete before any code is written or any results are observed.

- [ ] Write a plan document in `docs/plans/YYYY-MM-DD-<experiment-name>.md`
  - Research question and hypotheses (null + alternative)
  - Design table: one row per run — `redis.db`, `pipeline`, `prompts`, `problem.name`,
    `llm_base_url`, `HOTPOTQA_CHAIN_URL` (or equivalent), seed, val set
  - Controlled variables: every config field held constant, with its value
  - Success criteria: exact numeric thresholds, pre-registered
  - Monitoring plan: gen-5 smoke check, gen-10/25/50 checkpoints, early-termination rules
  - Statistical power analysis
- [ ] Commit the plan document **before any implementation begins**. Record the commit hash.
- [ ] Pre-registration commit hash: `____________`

---

## Phase 1: Implementation

- [ ] Implement all code and config changes
- [ ] For every pipeline YAML that will be used with `prompts=<custom>`:
  - [ ] Confirm `prompts_dir: ${prompts.dir}` appears in the `evolution_context` block
  - [ ] Confirm `prompts_dir: ${prompts.dir}` appears in `mutation_operator` (via `_base.yaml` or explicitly)
  - [ ] If either is missing: **stop and fix before proceeding**
- [ ] For every `validate.py` that returns `tuple[dict, list[...]]`:
  - [ ] Confirm `pipeline=hotpotqa_asi`, `pipeline=hotpotqa_reflective`, or another pipeline
    with a custom `FormatterStage` that handles tuples
  - [ ] `pipeline=standard` is **forbidden** for tuple-returning validate.py — it calls `repr()`
- [ ] Run the unit tests: `~/envs/evo_fast/bin/python -m pytest`
- [ ] Run `ruff check . && ruff format --check .`

---

## Phase 2: Dry-Run Verification

**This phase is mandatory for every run, without exception.**

Run `dry_run=true` for each planned run and verify every field below. Do not proceed until
all items are checked.

```bash
HOTPOTQA_CHAIN_URL=<chain_url> python run.py \
    <all run params> dry_run=true 2>&1 | tee dry_run_<run_label>.txt
```

Then verify:

### 2a. Config fields (from `=== FULL RESOLVED CONFIG ===`)

- [ ] `redis.db` — matches design table for this run
- [ ] `redis.host` / `redis.port` — correct Redis instance
- [ ] `problem.name` — matches design table (`chains/hotpotqa/static` vs `static_r`, etc.)
- [ ] `problem.dir` — path exists and resolves to the correct directory
- [ ] `prompts.dir` — correct value:
  - For control runs: `null` or package default
  - For treatment runs with custom prompts: correct absolute path (e.g., `.../gigaevo/prompts/hotpotqa`)
- [ ] `pipeline_builder._target_` — correct class (e.g., `ASIPipelineBuilder`)
- [ ] `max_generations`, `max_mutations_per_generation`, `max_elites_per_generation`,
  `num_parents`, `primary_resolution` — all match design table
- [ ] `llm_base_url` — correct mutation LLM server IP and port

### 2b. Runtime metadata (from `=== RUNTIME METADATA ===`)

**[PROBLEM DIR]**
- [ ] Path exists
- [ ] `validate.py` present
- [ ] `pipeline.py` present (if applicable)
- [ ] `initial_programs/` present (if seeding from this dir)

**[VALIDATE.PY]**
- [ ] Module path is correct
- [ ] `Returns:` annotation:
  - If `dict` → `pipeline=standard` is acceptable
  - If `tuple[dict, list[dict]]` → **must not use `pipeline=standard`**

**[PIPELINE BUILDER]**
- [ ] Class matches expected pipeline
- [ ] `FormatterStage override:`:
  - If validate.py returns `tuple`: must NOT show `none (base — calls repr() on non-str)`
  - If validate.py returns `dict`: base is fine

**[PROMPT FILES]**
- [ ] For each of the 8 files (`mutation/system`, `mutation/user`, `insights/system`,
  `insights/user`, `lineage/system`, `lineage/user`, `scoring/system`, `scoring/user`):
  - For **control** runs: all 8 should show `[default]`
  - For **treatment** runs with `prompts=hotpotqa` (or custom):
    - Files that were intentionally customized: show `[CUSTOM]`
    - Files that are intentionally kept default: show `[default]`
  - Any `[MISSING]` that was not expected: **stop and fix before proceeding**
- [ ] For each `[CUSTOM]` file: read the snippet and confirm it is the NLP-specific version,
  not the default optimizer text

**[SEED PROGRAMS]**
- [ ] Dir path contains the correct seed commit hash
- [ ] At least 1 `.py` file present with `(entrypoint: ✓)`

**[MUTATION LLM]**
- [ ] URL matches design table for this run
- [ ] `Model ID:` is the expected model (e.g., `Qwen/Qwen3-235B-A22B`)
- [ ] `Thinking:` — confirm `✓` if thinking mode is required

**[REDIS]**
- [ ] `0 keys (empty)` — if any keys exist: flush with `redis-cli -n <db> FLUSHDB`, then re-run dry-run

**[ENVIRONMENT VARIABLES]**
- [ ] `HOTPOTQA_CHAIN_URL` (or problem-specific server env var) is set to the correct chain
  server URL for this run
- [ ] `NO_PROXY` / `no_proxy` includes all internal server IPs used in this run
- [ ] No unexpected env vars that could override config

---

## Phase 3: Preflight Checks (automated via launch.sh)

The launch script must perform these checks and abort on failure:

- [ ] All vLLM servers (mutation LLMs + chain servers) return HTTP 200 on `/v1/models`
- [ ] All chain servers respond with `<think>` in a test completion (thinking mode confirmed)
- [ ] All Redis DBs are empty (`dbsize() == 0`)
- [ ] Seed directory and `initial_programs/` exist

If any check fails: **do not launch**. Fix the issue and re-run from Phase 2.

---

## Phase 4: Launch

The `[verify]` block in the launch script runs all dry-runs and pauses for `read`. Do not
press Enter until Phase 2 is fully complete for all runs.

- [ ] Phase 2 complete for Run 1: `______`
- [ ] Phase 2 complete for Run 2: `______`
- [ ] Phase 2 complete for Run 3: `______`
- [ ] Phase 2 complete for Run 4: `______`
- [ ] Pressed Enter in launch script
- [ ] Recorded PIDs: `______`
- [ ] Recorded launch time (UTC): `______`
- [ ] Appended PIDs and launch time to pre-registration doc as actual launch record

---

## Phase 5: Post-Launch Monitoring

Per the monitoring plan in the pre-registration doc:

- [ ] Gen 5 (~2h): smoke check — all PIDs alive, Redis keys growing, no crashes
- [ ] Gen 10: checkpoint evaluation — extract best-by-val, run test eval, record metrics
- [ ] Gen 25: midpoint checkpoint — same metrics, check val-test gap trajectory
- [ ] Gen 50: final evaluation — full test eval, statistical analysis, write report

Any deviation from expected behavior (crash, stall, unexpected results) must be recorded
as an **amendment** in the pre-registration doc with commit hash and date.

---

## Compatibility Reference

### validate.py return type → required pipeline

| `validate.py` returns | Allowed pipelines | **Forbidden** |
|---|---|---|
| `dict` | `standard`, `hotpotqa_asi`, `hotpotqa_reflective` | — |
| `tuple[dict, list[dict]]` | `hotpotqa_asi`, `hotpotqa_reflective` | **`standard`** |

### Custom prompts → required pipeline YAML fields

When using `prompts=<custom>` (any non-default prompts config):

| Config location | Required field |
|---|---|
| `evolution_context` block in pipeline YAML | `prompts_dir: ${prompts.dir}` |
| `mutation_operator` block (or via `_base.yaml`) | `prompts_dir: ${prompts.dir}` |

If either is missing, the custom prompts are **silently ignored** — the system uses defaults
with no error. Verify via `[PROMPT FILES]` in dry-run output.

---

## Amendment Protocol

When a pre-registered plan must change after registration (bug fix, config correction, etc.):

1. Record the amendment in the pre-registration doc under a dedicated **Amendments** section
2. Include: what changed, why, the relevant commit hash, and the impact on design validity
3. Classify the amendment's impact:
   - **No confound**: change applied uniformly to all runs (e.g., pipeline bug fix)
   - **Confound introduced**: change applies differently across runs — document and assess
   - **Run invalidated**: the affected run(s) must be excluded from analysis

---

## Quick-Reference: Common Failure Modes

| Symptom | Root cause | Detection | Fix |
|---|---|---|---|
| Mutation prompts contain raw Python `repr()` output | `pipeline=standard` + tuple-returning `validate.py` | `[PIPELINE BUILDER]` shows `none (base)` | Switch to `pipeline=hotpotqa_asi` |
| Custom prompts silently ignored; `[PROMPT FILES]` shows all `[default]` | `prompts_dir` missing from `evolution_context` in pipeline YAML | `[PROMPT FILES]` in dry-run | Add `prompts_dir: ${prompts.dir}` to pipeline YAML |
| Chain server produces non-thinking output | Server restarted in non-thinking mode | `[preflight]` thinking check fails | Restart server with correct config |
| Redis DB not empty at launch | Previous run not flushed | `[REDIS]` in dry-run shows `N keys` | `redis-cli -n <db> FLUSHDB` |
| Wrong seed used | `program_loader.problem_dir` misconfigured | `[SEED PROGRAMS]` dir path wrong | Fix `program_loader.problem_dir` in launch params |
| HOTPOTQA_CHAIN_URL missing from env | Not exported before `nohup python run.py` | `[ENVIRONMENT VARIABLES]` — var absent | Export before launch or prefix with `VAR=val` |
