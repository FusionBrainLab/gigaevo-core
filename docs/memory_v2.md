# Memory v2: Causal Bayesian Core

Memory v2 keeps the useful operational machinery from memory v1: the Pydantic
card schema, local store, live writer, online authored-action equivalence,
program exemplars, lineage exclusion, and selection leases. It replaces the
entangled reputation and auction path with one replayable hierarchical causal
bandit.

The full posterior, selection, lifecycle, diagrams, and plain-English Bayesian
glossary are in [the Bayesian system report](memory_v2_bayesian_system_report.md).

This first iteration is intentionally narrow. It supports one proposed card per
mutation and `num_parents=1` on bounded MAP-Elites tasks. It does not claim
multi-card attribution, crossover identification, or full RAG-policy OPE.

## Run Surface

The base experiment now selects memory v2, the metadata-routing memory pipeline,
and one parent. A normal bounded task therefore needs no memory overrides:

```bash
python run.py problem.name=heilbron
```

The equivalent fully explicit HoVer recipe is:

```bash
python run.py \
  storage=disk \
  problem.name=chains/hover/full7_vectorized \
  archive_selector=paired_bootstrap \
  program_format=json_document \
  mutation=carl_with_retrieval_tools \
  algorithm=chains_bd3d \
  enable_chain_structural_metrics=true \
  num_parents=1 \
  pipeline=memory_guided_noise \
  memory=v2 memory/write=live
```

Normal runs use 70% delivery and 30% matched control after validation. Balanced
validation launchers override both
`memory.posterior_config.reference_offer_probability` and
`memory.policy_config.offer_probability` to `0.50`. Use
`pipeline=guided memory=none` for a true no-memory arm.

The startup validator rejects multi-parent use, metadata-dropping pipelines,
and refresh coalescing. A causal decision belongs to one mutation attempt;
reusing a cached parent assignment for concurrent mutations would give one
decision several outcomes.

The saved production smoke command and audit are in
`experiments/hover/memory_v2_smoke`.

## Immutable Intervention

A `CardSnapshot` freezes the exact text block delivered to the mutator:

```text
[card 1] id=<bank_card_id>
<payload>
```

`treatment_id` is the stable `bank_card_id`; the payload digest is only an audit
key for the exact block. The context stage recognizes the block as preformatted
and does not wrap it a second time. Rewriting a near-duplicate card therefore
keeps the same Bayesian arm and continues accumulating its randomized evidence.

Evidence is isolated by a compact `EnvironmentFingerprint`: task, mutation
model settings, algorithm, the concrete `MutationOperator` class, program
format, and pipeline. The operator belongs to each decision's frozen
environment context; it is not a card field or a card-ranking feature. Exact
card content is already identified by `treatment_id`; broader source and run
provenance belongs in the experiment manifest rather than the statistical
schema.

The environment fields have the following contract:

| Field | Meaning | Statistical role |
| --- | --- | --- |
| `task_key` | Stable memory task identity, such as a concrete benchmark instance. | Separates task-specific cards and causal evidence. |
| `problem_name` | Resolved problem implementation/configuration. | Preserved for audit and exact replay. |
| `llm.model_name` | Configured mutation model, including its provider prefix. | Part of the safety-calibration group because models have different failure rates. |
| `llm.base_url` | Inference endpoint used by the mutation model. | Preserved for audit; it is not a posterior feature. |
| `llm.temperature` | Mutation sampling temperature. | Preserved because it changes the proposal distribution. |
| `mutation_operator` | Concrete `MutationOperator` class that creates the child. | Part of the safety-calibration group because operators have different validity behavior. |
| `program_format` | Representation read and produced by mutation. | Prevents evidence from incompatible program representations being mixed. |
| `pipeline` | Evaluation and memory-injection pipeline. | Identifies the intervention semantics used by the decision. |
| `algorithm` | Evolution/archive algorithm governing parent selection and context. | Identifies the search process that generated the decision context. |

These are typed decision provenance, not learned card features. Online
posteriors consume the frozen numeric evolutionary context; the calibration CLI
groups safety priors by the complete typed fingerprint so none of these fields
is silently pooled across runs.

## Decision And Outcome Lifecycle

At each mutation attempt, v2 freezes and durably records:

- the typed parent identity, metrics, generation, and reward bounds;
- the complete MAP-Elites snapshot described below;
- the complete eligible-bank action universe and its immutable card snapshots;
- the optional agentic RAG applicability assessment, including its frozen policy
  fingerprint, selected card ids, summary, and neutral/failure state;
- pending counts for stable treatments and bank lineages;
- the evidence, context, candidate, posterior, and policy hashes;
- posterior fit diagnostics and every candidate posterior summary;
- abstention, proposal, conditional offer, and joint action probabilities;
- the sampled action and frozen reward/risk nuisance predictions.

The decision is committed before prompt exposure. If a card is delivered, its
selection lease is reserved with compare-and-swap semantics. The child link is
persisted before the child enters program storage. Every decision then accepts
one immutable terminal whose parent, child link, metric direction, and metric
bounds are checked against the frozen decision.

Evaluation-invalid results and post-decision mutation/prompt failures are
invalid outcomes. Pre-treatment or administrative failures are censored.
Discarded or missing children are explicitly closed, and startup reconciliation
closes orphan decisions. Censored and OPE-ineligible rows remain auditable but
do not enter posterior fitting. SQLite records decision, child-link, and
terminal UTC timestamps independently of immutable payload hashes, so retries
cannot manufacture timestamp conflicts.

## MAP-Elites Context

The model uses real archive state without treating unstable dynamic cell IDs as
exchangeable categories. Each decision freezes:

- behavior schema and archive fingerprints;
- raw, semantic-normalized, dynamic-normalized, and binned coordinates;
- archive coverage and size;
- absolute oriented parent fitness and archive quality quantile;
- local neighbor occupancy, parent cell, iteration, and generation.

Each binning object owns its stable model transform. Memory v2 reads the common
ordered axis schema from the live behavior spaces; HoVer, for example, uses
linear normalization for `hop_depth` and log normalization for
`passages_fetched` and `instr_chars`. Mutable archive bounds never change these
model coordinates. The minimal shared context is intercept, parent fitness,
progress, and the task's behavior values. Each card has one shrunk contextual
deviation over intercept, fitness, and those behavior values. No dynamic cell
coordinate, quadratic, lexical, keyword, category, or hashed-text feature enters
the posterior.

## Bounded Hierarchical Utility

`memory.credit.lineage_depth=1` is the exact proximal endpoint: the oriented
parent-child gain divided by the configured metric range. At larger depth, one
delayed reward row is derived for each original randomized decision. Its clock
is a fixed number of later mutation opportunities in the root's pre-treatment
MAP-Elites island. At maturity, a bounded breadth-first scan follows only the
root child lineage, stops at the configured depth and opportunity cutoff, and
uses the best descendant fitness relative to the original parent. Siblings are
excluded and descendants never become independent reward rows. Invalid roots
train safety immediately, but their full-utility row waits for the same follow-up
window as valid roots. Unfinished run-tail endpoints remain pending rather than
being filled with zero.

This local opportunity clock does not condition on how often the lineage itself
was selected. Reselection speed and descendant fertility remain part of the
card's policy-conditional evolutionary effect, while islands with different
update rates do not share one clock. The immediate terminal still trains the
safety head; only matured lineage rows train valid reward. This endpoint is not
direct MAP-Elites archive contribution.

Invalidity is modeled as a separate treatment-dependent hurdle. The final
action value combines valid gain with the frozen parent's worst feasible gain
under invalidity:

```text
D | x,j,a ~ Bernoulli(p_a)                         # every terminal
Y | D=0,x,j,a ~ Normal(g_Y + a tau_Y, sigma)       # valid gain only
v_a = clip(E[Y | D=0,x,j,a], parent_gain_bounds)
logit(p_a) = g_D(x) + a tau_D(x,j)
q_a = (1-p_a) v_a + p_a worst_feasible_gain(parent)
effect(x,j) = q_1 - q_0
```

The design is `baseline(x) + A * card_effect(x)`: proposed-but-withheld controls
all use the same contextual baseline, while delivered cards add their lineage and
stable card effect. When enabled, a frozen RAG-applicability indicator is one
additional shared treatment-effect contrast. It is learned from the same randomized
outcomes; it is not a hand-written reward or a candidate gate. A fixed conditional
offer probability `e` supplies randomized overlap (`0.7` by default for normal
runs); mixed offer propensities inside one fitted ledger are rejected by this first
implementation.
Invalidity learns from every randomized terminal; valid gain learns only where a
gain exists. The composite value still assigns invalidity its parent-specific
pessimistic consequence instead of treating it as missing or zero.

The reward posterior uses proper configured shrinkage priors over shared,
card-lineage, and contextual effects. Conditional coefficient
posteriors are Gaussian; residual scale is integrated as a one-dimensional
Bayesian mixture with explicit quadrature convergence diagnostics. Random-effect
prior scales are fixed configuration, not weakly identified learned
hyperparameters. Upper-support truncation or numerical failure makes the policy
abstain; concentration near zero unexplained residual noise is logged but is not
itself a failed fit.

The safety model uses a proper-prior logistic MAP fit and Laplace covariance.
L-BFGS is polished or replaced by a Newton solve, and gradients, objective,
Hessian conditioning, iterations, residual diagnostics, and the per-treatment
offer-probability hash are persisted on every decision. Numerical failure is
fail-closed.

If child and base evaluations have the same non-empty ordered cohort digest,
the reward likelihood uses the analytic paired-difference standard error. A
scalar result or cohort mismatch records unknown measurement uncertainty; it is
never mislabeled as exact.

## Risk-Gated Probability Matching

For each card, the Laplace safety posterior induces a bivariate Gaussian over
control and treatment invalidity logits. Adaptive conditional-normal quadrature
with an explicit absolute-error tolerance computes the conservative posterior
probability of the configured acceptable event. The default event is

```text
q = P(p_1 - p_0 <= incremental_cap | history).
```

The default admits a card when `q > alpha`, equivalently excluding it only when
the posterior is at least `1 - alpha` confident that incremental invalidity
exceeds the cap. There is no task-independent absolute invalidity ceiling:
task, model, and mutation-operator baselines can differ substantially. The
optional `credible_joint_safe` mode instead requires at least `1 - alpha`
probability that both an explicit treated-invalidity ceiling and the
incremental cap hold. Treated-risk and incremental-risk upper bounds and the
integration error are persisted. Numerical or tolerance failure excludes the
card, so admission has no Monte Carlo error and is independent of summary RNG.

For admitted cards, an event-keyed finite set of shared posterior worlds votes
for the highest usable effect or abstention. Those configured winner counts are
the behavior policy, so the sampled categorical probability is exact for that
finite-world policy. Its binomial Monte Carlo standard error is logged as a
diagnostic. A configured uniform-exploration mixture gives every feasible card
in the full eligible bank nonzero proposal support. Before this policy, an
agentic plan/retrieve/reflect pass may label a small semantic subset as
applicable to the current parent. It never removes a card from the action
universe: the posterior compares every eligible card in one shared-world pool.
The label enters only through the learned treatment-effect contrast above, so a
retrieval failure yields an empty, neutral label rather than a different
candidate policy. Retrieval rank is never treated as causal reward evidence.

After proposing card `j`, v2 randomizes actual delivery:

```text
rho(j)                 proposal probability
e(j)                   conditional offer probability
rho(j) * e(j)          delivered joint probability
rho(j) * (1 - e(j))    withheld-control joint probability
```

`e(j)` is a fixed configured overlap probability. Both arms consume the same
pending budget over the complete stable card lineage, preventing delayed
controls or historical absorbed ids from bypassing the concurrency cap.

## Content Authoring and Bayesian Retirement

The writer inspects each parent→child diff and observed outcome, authors at most
one conditional hypothesis, retrieves neighbors from that authored action, and
uses strict same-action/same-condition equivalence before admission. Program
exemplars use the same protocol and retain the best concrete representative of
each semantic strategy family. There is no union prose, merge decision, or
periodic exhaustive consolidation. The writer does not restamp heuristic
efficacy events into the v2 model.

Causal retirement refits the current causal posterior, requires randomized
treated and pooled-control support across multiple discrete MAP-Elites
island/cell contexts, and protects every pending lineage alias. Selection
defines helpfulness relative to zero; retirement uses a normalized practical
utility boundary so a well-supported neutral card can eventually leave the
bank. A card is removed only when the Wilson Monte Carlo upper bound for
`P(safe and practically useful)` is below the configured threshold in every
supported context under both unassessed and applicable RAG states. Optimizer,
residual-scale boundary, posterior-boundary, or safety-integration uncertainty
in either reward head vetoes irreversible retirement. The admission gate
consumes a one-use verdict, rechecks the exact treatment revision and evidence
version, protects live and historical-alias leases, and applies foreign-task
positive-evidence retention before deletion.

Each new evidence version may be judged again; "one-use" describes verdict
consumption, not a fixed-sample sequential-testing guarantee. Deterministic
evidence-version RNG and the Wilson upper bound make repeated checks
conservative, but posterior misspecification remains a risk. Censored outcomes
are excluded under a conditionally non-informative-censoring assumption, and a
card that stops receiving proposals before minimum support fails-keep. The
causal ledger and card bank are separate stores, leaving a narrow
ledger-version/read-to-bank-delete interval outside a shared transaction.

## Durable Evidence

The causal source of truth is:

```text
<checkpoint_dir>/memory_v2_selection_evidence.sqlite3
```

Decisions, child links, terminals, and event ordinals share one transactionally
consistent ledger. This separate filename deliberately starts the new
candidate-universe/applicability schema without migrating, restamping, or
rewriting earlier ledgers. Payload hashes are verified on every read, and writes use
full synchronization. SQLite locking on the project NFS mount is not trusted:
v2 detects network filesystems, runs the live database on node-local scratch,
then fsyncs and atomically replaces a checkpoint mirror after every causal
write. Treatment is returned only after that mirror succeeds. JSONL memory
events remain telemetry, never causal evidence.

## OPE Scope

`ConditionalOfferDREvaluator` evaluates only a changed delivery gate under the
logged behavior proposal distribution. Its target policy receives a validated
`PreDecisionUnit` and cannot inspect outcomes. The estimator uses frozen
decision-time reward/risk regressions and reports overlap, effective sample
size, and maximum importance weight. Standard error is clustered by independent
run when at least two runs exist; a single run reports `se=None`.

This is not full proposal-policy OPE or a causal estimate of the RAG assessor.
RAG applicability is an adaptive, non-randomized pre-treatment covariate, while
the complete eligible bank remains the logged action universe. Comparing
agentic and null applicability requires prospective independent runs.

## Smoke Analytics

```bash
python experiments/hover/memory_v2_smoke/analyze.py \
  --ledger <checkpoint_dir>/memory_v2_selection_evidence.sqlite3 \
  --output-dir <run_dir>/memory_v2_analytics
```

The audit independently validates SQLite and payload hashes, probability mass,
conditional and joint propensities, overlap, finite-world uncertainty, safety
membership, bounded outcomes, optimizer health, immutable terminal contracts,
timestamps, and evidence accounting. It exports complete decision, candidate
posterior, and terminal CSV traces plus a probability/posterior/MAP dashboard
and conditional-offer calibration plot, plus a conditional-offer DR sensitivity
trace. Hard gates require both randomized arms, terminal closure, posterior
updating, minimum decision/candidate/proposal counts, and acceptable assignment
balance.

## Safety-Prior Calibration

Safety failure rates depend on the task, model, concrete mutation operator, and
the rest of the typed execution environment. The CLI therefore groups by the
complete fingerprint rather than applying a circle-packing prior to a
structured-diff chain run or pooling different pipelines:

```bash
gigaevo -f json memory calibrate-safety \
  <run-or-checkpoint-dir> [<another-ledger-or-run> ...] \
  --output safety_calibration.json
```

The command hash-checks every ledger, reconstructs eligible invalid/outcome
rows, and replays each candidate prior using the exact
`fitted_observation_ids` frozen before that decision. It ranks the grid by
prequential Bernoulli log loss, then emits the best candidate that also passes
the configured minimum retention for both cold-start and later-new-card strata.
Gate replay evaluates every frozen candidate set, including decisions where the
deployed policy abstained; predictive scoring remains restricted to closed,
eligible proposed outcomes.
By default it tests an outcome-independent shared-effect grid corresponding to
half, unchanged, and double treatment odds. The Jeffreys-smoothed
treated/control log-odds contrast is reported as a descriptive diagnostic but
is not inserted into the scored grid, because doing so would leak current and
future outcomes into earlier replay predictions. This keeps the control
invalidity baseline and the shared effect of card delivery distinct;
card-specific deviations stay zero-mean.
The unconstrained calibration winner remains visible, so a conflict between
honest calibration and bootstrap admission cannot be hidden. The report also
includes Brier score, calibration bias, treated/control calibration,
equal-count calibration bins, gate-retention diagnostics, and exact Hydra
overrides for all five scored prior parameters. Multiple incompatible
environment keys produce separate reports. `--min-gate-retention` controls the
minimum candidate retention, with a default of 25%.

A fixed candidate's predictions are prequential, but selecting the best grid
candidate on those same trajectories is retrospective model selection. Its
score is therefore a development estimate. A recommendation from one
trajectory is labeled provisional and still needs a fresh run. The CLI also
reports the homoscedastic overlap proxy for alternative delivery rates. Under
that standard design approximation, 70% delivery retains 84% of the
binary-contrast information of a 50/50 experiment per proposal; 75% retains 75%,
and 80% retains 64%. These rows do not estimate the effect of changing the full
proposal policy, and gate replay does not identify the trajectory induced by
changing admission.

## Deliberately Deferred

- mutation-level crossover and multi-card slate randomization;
- full-policy evaluation or optimization of the agentic RAG assessor;
- sparse or low-rank posterior updates for long histories with many retired
  treatments;
- full proposal-policy OPE and adaptive confidence sequences;
- archive-contribution credit or LLM self-reported use attribution;
- change-point or nonstationary state-space models;
- a separate frozen-snapshot MAP-Elites archive-contribution endpoint.

These are explicit extension points, not hidden claims of the current endpoint.

## Foundations

- Randomized contextual treatment/control estimation follows the contextual-bandit
  treatment in [Krishnamurthy et al., ICML
  2018](https://proceedings.mlr.press/v80/krishnamurthy18a.html).
- Bank/card partial pooling follows the mixed-effect Thompson-sampling
  perspective in [Aouali et al., AISTATS
  2023](https://proceedings.mlr.press/v206/aouali23a.html).
- Conditional-offer evaluation follows doubly robust policy evaluation from
  [Dudik et al., ICML
  2011](https://www.microsoft.com/en-us/research/publication/doubly-robust-policy-evaluation-and-learning/).
- The restriction on single-run uncertainty reflects adaptive-data inference
  requirements described by [Karampatziakis et al., ICML
  2021](https://proceedings.mlr.press/v139/karampatziakis21a.html).
