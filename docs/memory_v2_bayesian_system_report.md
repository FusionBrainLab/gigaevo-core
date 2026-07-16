# Memory v2 Bayesian System Report

## Executive verdict

Memory v2 achieves the intended first-iteration replacement:

- v1 remains responsible for card content: insight authoring, program exemplars,
  deduplication, consolidation, storage, lineage exclusion, and leases;
- v1 reputation, gain restamping, bootstrap bidding, Thompson auction, efficacy
  rendering, and no-card heuristics do not control v2 selection;
- one replayable hierarchical Bayesian model now owns reward, invalidity risk,
  contextual card comparison, conservative safety admission, probability matching,
  and harm retirement;
- every delivered card has a randomized withheld-card control and an exact logged
  probability;
- the first iteration deliberately evaluates the complete eligible same-task bank.
  It does not use LLM, lexical, keyword, category, embedding, or code-similarity
  retrieval for selection.

This is a solid causal core for an empty-bank smoke without an arbitrary task-wide
admission cap. The smoke first validates authoring and admission, then selection and
causal credit after cards exist. It is not yet a large-bank retrieval system, a
multi-card attribution model, or a full archive-contribution model.

At cold start, cards with no v2 randomized evidence are deliberately close to
exchangeable. The policy explores them instead of pretending that v1 observational
scores or an unlogged semantic ranker are causal evidence. Differentiation becomes
trustworthy only after randomized treated/control outcomes accumulate.

## System architecture

![Memory v2 content, decision, and evidence planes](diagrams/memory_v2_architecture.png)

### What is retained from v1

The content plane keeps the useful and already tested memory machinery:

1. The LLM librarian reads successful parent-child differences and proposes insight
   cards.
2. The writer can also preserve strong program exemplars.
3. Admission reconciles duplicates, updates known cards, and consolidates related
   cards.
4. The local store persists the active card bank.
5. Lineage exclusions prevent a child from immediately receiving its ancestors'
   cards again.
6. Selection leases protect in-flight assignments and consolidation aliases.

The LLM is therefore still important for **writing good cards**. It is not used to
retrieve or rank cards for the mutator in this first v2 policy.

### What is replaced from v1

The following v1 values may remain on historical card objects for audit, but v2 does
not use them for inference or selection:

- `posterior_a` and `posterior_b` reputation counts;
- founding or later-use gain restamping;
- empirical-Bayes cohort priors;
- bootstrap expected-value bids;
- the Thompson/bootstrap auction;
- no-card evidence and legacy budget gates;
- efficacy text derived from the old posterior;
- LLM research retrieval as a selection shortlist.

The v2 causal ledger is the only training source for the new posterior. Historical
v1 events are descriptive shadow-analysis data and are explicitly marked
`training_eligible=false`.

## What a card means statistically

A stored card has two identities:

- **bank lineage**: the conceptual card across rewrites and consolidations;
- **card snapshot**: the exact text block shown to the mutator; its stable card id
  remains the Bayesian treatment identity.

The delivered intervention is frozen as:

```text
[card 1] id=<bank_card_id>
<payload>
```

The treatment id hashes that entire rendered block. Editing one word creates a new
snapshot. Near-duplicate merges retain the stable treatment id, so randomized
evidence continues pooling across the card's audited content snapshots.

**Plain English:** a card is treated like a medicine label. The family name says
which medicine it belongs to; the exact formulation says which version was actually
administered.

## Frozen evolutionary context

Each decision records a typed, pre-treatment snapshot of the parent and its current
MAP-Elites archive:

- oriented parent fitness and its bounded metric range;
- fitness rank within the current archive;
- iteration and generation progress;
- archive size and coverage;
- parent-cell occupancy and local-neighbor occupancy;
- raw behavior values;
- fixed semantic-normalized behavior values;
- live dynamic-normalized behavior values;
- live axis bounds, bins, cell coordinates, schema hash, and archive fingerprint.

For HoVer the configured behavior axes are `hop_depth`, `passages_fetched`, and
`instr_chars`.

### Dynamic MAP-Elites axes

The system does **not** assume that cell `(2, 4, 1)` means the same thing forever.
Dynamic axes can change their live minimum, maximum, and therefore cell assignment as
the archive evolves.

Each binning object owns a stable model transform and exposes two separate methods:

1. `normalize_for_model()`: fixed domain bounds and transform, used by the posterior;
2. `normalize_for_archive()`: current live bounds, used only to describe rebinning.

Transient cell indexes and live coordinates are recorded for audit and local archive
state, but they are not regression features. The posterior therefore sees the same
model coordinate for the same raw value after the archive is rebinned.

**Plain English:** if a classroom changes its grading curve, the model records both
the student's raw exam score and their percentile under today's curve. It never
pretends that “seat 12” is a permanent kind of student.

This removes dynamic-axis drift from the statistical basis. General evolutionary
nonstationarity can still be added later if experiment traces justify it.

## Modeled endpoint

At `lineage_depth=1`, memory v2 models the immediate, bounded parent-to-child
change in the primary fitness metric. Positive always means better. For metric
bounds `[l, u]`, parent fitness `f_p`, and child fitness `f_c`, a
higher-is-better task uses:

```text
Y = (f_c - f_p) / (u - l)
```

The bounds come from the problem's primary `MetricSpec`, not from memory or the
observed sample. HoVer declares `fitness` as higher-is-better with `[0, 1]` in
`problems/chains/hover/full7_vectorized/metrics.yaml`. A parent at `0.80` therefore
has feasible gain bounds `[-0.80, +0.20]`. A missing finite primary range makes v2
fail before selection rather than estimate a convenient range from the run.

The signs reverse for lower-is-better metrics. The feasible gain interval is frozen
from the parent's location inside `[l, u]`. Observations outside it are rejected.

At larger depth, the reward endpoint follows the root child and its descendants
through a fixed number of later mutation opportunities in the root's frozen
MAP-Elites island. A depth-limited graph scan then freezes one row containing the
best lineage fitness relative to the original parent. Other roots' siblings are
excluded, and descendants are never treated as independent reward samples. This
keeps update velocity and reselection fertility inside policy-conditional utility
without conditioning maturation on the lineage surviving for a chosen number of
expansions. It remains distinct from direct MAP-Elites archive contribution.

Terminals have three states:

- **valid**: an evaluable child with a bounded numeric gain;
- **invalid**: treatment occurred, but mutation or evaluation failed;
- **censored**: treatment effect cannot be observed, for example a pre-treatment or
  administrative failure.

Invalid outcomes train the risk model. They never receive an invented numeric reward.
Censored outcomes remain auditable but do not train either outcome likelihood.

## Bayesian posterior

![Hierarchical reward and invalidity posterior](diagrams/memory_v2_posterior_hierarchy.png)

For card revision `j`, context `x`, and delivered-card indicator `A`, the design
vector is:

```text
z_j(A, x) = [ baseline(x), A * effect_j(x) ]
```

The effect weight is `0` for withheld control and `1` for delivered treatment.
All proposed-but-withheld cards therefore share the same contextual control arm;
only delivered cards add a stable card effect. Normal post-validation runs use
fixed probability `e = 0.7`; balanced validation runs use `e = 0.5` to collect
information faster.

### Valid-gain model

For valid terminals:

```text
Y_i | beta, sigma ~ Normal(z_i^T beta, sigma^2 + s_i^2)
```

Here:

- `beta` contains shared, bank-lineage, and contextual coefficients;
- `sigma` is unexplained mutation-to-mutation variability;
- `s_i` is known paired-evaluation uncertainty when available, or an explicitly
  configured unknown-measurement scale for scalar evaluations.

The residual prior is:

```text
log(sigma) ~ Normal(log(0.20), 0.75^2),  sigma in [0.01, 5]
```

Conditional on a chosen `sigma`, the coefficient posterior is Gaussian and computed
analytically. Adaptive one-dimensional quadrature integrates over uncertainty in
`sigma`, including multiple possible modes and convergence diagnostics.

**Plain English:** among valid children, the model estimates how much the card tends
to change fitness while admitting that mutations are noisy and some evaluations are
measured more precisely than others.

### Invalidity model

Every non-censored terminal, valid or invalid, trains:

```text
D_i ~ Bernoulli(sigmoid(z_i^T gamma))
```

`D_i = 1` means invalid. The generic cold-start control prior is 5%; an
environment-calibrated profile can replace it. The shared treatment intercept has
its own log-odds prior mean (zero generically), so a high mutation-operator baseline
and a lower card-treated risk do not have to be collapsed into one misleading
probability. Context slopes and card deviations remain zero-mean. A proper-prior
logistic MAP fit finds the most plausible coefficients, and a Laplace covariance
approximates posterior uncertainty around that point. Optimizer, gradient, and
Hessian-conditioning diagnostics are persisted. Numerical failure is fail-closed.

**Plain English:** this is a separate model for “how likely is this card to break the
mutation?” A card cannot look good merely because its failures disappeared from the
reward sample.

### Hurdle utility

For action `a` in `{0, 1}`:

```text
v_a = clip(E[Y_a], parent's feasible gain bounds)
q_a = (1 - p_a) * v_a + p_a * worst_feasible_gain(parent)
Delta_j(x) = q_1 - q_0
```

`p_a` is invalidity risk. Invalidity receives the parent's worst feasible gain, not
zero. `Delta_j(x)` is the card's context-specific usable effect relative to its
withheld control.

**Plain English:** expected value combines “how much does it help when it works?” and
“how often does it fail?” A risky card is penalized by a real downside.

## Hierarchy and partial pooling

With `K=3` behavior axes, the complete shared context has only six values:

- intercept;
- oriented normalized parent fitness;
- normalized log progress;
- three stable behavior coordinates.

The baseline and shared treatment effect each use that six-value vector. Every card
lineage has one shrunk five-value deviation: intercept, parent fitness, and the three
MAP coordinates. Each stable card adds one contextual deviation. Card rankings
can therefore vary with parent fitness and MAP position without unrelated feature
families.

The coefficient count is:

```text
2*C + B*H + R
C = 3 + K
H = 2 + K
```

The smoke starts with `B=R=0`; there are no card effects to estimate before the
writer authors the first card. As the bank grows, `B` and `R` grow with the card
lineages and immutable revisions represented in the causal evidence. Every added
block is strongly shrunk toward zero until evidence justifies movement.

Configured reward prior standard deviations are:

| Component | Prior SD |
|---|---:|
| Shared baseline | 0.75 |
| Shared card effect | 0.35 |
| Card contextual deviation | 0.25 |

The invalidity model uses baseline/shared/card scales of
`0.15/0.20/0.60/0.30` and a 5% prior invalidity intercept. These scales are fixed
configuration, not weakly identified learned hyperparameters.

**Plain English:** every card starts near the overall average. A card can earn its own
reputation with evidence, and related content snapshots share the same statistical strength, but
small samples are pulled back toward the common center. This prevents one lucky child
from creating an extreme reputation.

### Task boundary

The first implementation is task-local. Candidate filtering requires the same task,
and a ledger fits one task environment, so HoVer evidence cannot alter a different
task's card posterior. There is no fake cross-task pooling from task names or hashes.

The hierarchy can later add `global -> task -> card` deviations when real
randomized evidence exists across multiple tasks. That extension does not require a
new decision, ledger, or policy contract; it only extends the feature map and priors.

## How one card is selected

![Complete single-card selection path](diagrams/memory_v2_card_selection.png)

For each mutation attempt:

1. **Freeze context.** Read the typed parent, fitness, progress, and current
   MAP-Elites snapshot before treatment.
2. **Snapshot the bank.** Keep nonempty insight/program cards from the same task and
   apply lineage exclusions. V2 does not impose a task-wide admission cap.
3. **Do not semantically prefilter.** No lexical or LLM feature influences inclusion
   or ordering. Every eligible card has a policy path.
4. **Resolve card snapshots.** Render and hash the exact singleton card text while
   retaining the stable card id as the treatment identity.
5. **Apply pending budget.** Exclude a lineage when it already has two in-flight
   proposed assignments, including withheld controls and absorbed aliases.
6. **Fit only prior evidence.** Load eligible closed terminals committed before this
   decision and fit the reward and invalidity posteriors.
7. **Predict every candidate.** Draw 1,024 shared posterior samples and record effect,
   gain, risk, and uncertainty summaries.
8. **Certify safety.** A card must have at least 90% conservative posterior probability
   that treated invalidity is at most 25% and incremental invalidity is at most 10%.
   Deterministic quadrature tolerance is `1e-8`; numerical uncertainty excludes the
   card.
9. **Build the proposal distribution.** In 512 shared posterior worlds, each safe card
   wins when it has the largest positive usable effect. Otherwise abstention wins.
10. **Preserve exploration.** Mix 5% uniform mass over all safe cards:

    ```text
    rho_j = 0.95 * winner_count_j / 512 + 0.05 / number_of_safe_cards
    rho_0 = 0.95 * abstention_count / 512
    ```

11. **Draw one proposal.** Sample either one card or abstention from that exact finite
    distribution.
12. **Randomize delivery.** If card `j` was proposed, deliver it with probability
    `e=0.7` in a normal run; otherwise withhold it as the matched control. Use
    `e=0.5` for balanced validation.
13. **Commit before exposure.** Persist the candidate set, posterior diagnostics,
    every probability, sampled action, and frozen predictions before the prompt can
    see a card.
14. **Inject at most one card.** Only the delivered branch enters the mutator. The
    withheld branch still records the proposed revision and consumes a lease until its
    terminal closes.

The logged leaf probabilities are:

```text
P(abstain)                 = rho_0
P(propose j and deliver)   = rho_j * 0.7
P(propose j and withhold)  = rho_j * 0.3
```

### How the system decides which card is better

It does not assign a permanent global score. In each posterior world, it estimates
the delivered-minus-withheld hurdle utility for every card under the **current
numeric evolutionary context**. A card is better for this parent when more posterior
worlds say it has the largest positive safe effect.

At cold start, identical prior structure means safe cards receive approximately
uniform exploration. After evidence accumulates, card rankings can differ by parent
fitness and stable behavior coordinates; progress changes the shared value of using
memory. No claim about semantic code/card compatibility is made in this iteration.

## Decision and evidence lifecycle

One causal unit belongs to one mutation attempt and one parent:

```text
active attempt
  -> context and candidates frozen
  -> decision committed
  -> one card delivered or withheld
  -> child linked before storage exposure
  -> exactly one terminal recorded
  -> lease released only at terminal closure
  -> later decisions may refit using this evidence
```

The SQLite causal ledger stores immutable decision, child-link, and terminal records.
Payload hashes are verified on read. JSONL memory events remain telemetry and cannot
train the posterior.

The outcome taxonomy prevents common attribution errors:

- evaluation-invalid and post-treatment mutation failures are invalid outcomes;
- pre-treatment failures are censored;
- orphan decisions are reconciled and explicitly closed;
- mixed offer propensities are rejected by this first model;
- v1 gains never silently warm the v2 posterior.

## Bank growth and retirement

The v2 configuration does not reject a genuinely new card because the same-task
bank reached an arbitrary size. Deduplication, consolidation, bounded program
exemplars, and evidence-backed harm retirement still control redundant or harmful
content.

A card is not evicted simply because it has a low posterior mean. Retirement requires:

- a fresh current posterior fit;
- at least two direct treated and two direct withheld outcomes for the stable card;
- at least two distinct observed modeled contexts;
- no pending assignment anywhere in its consolidation lineage;
- successful reward and safety numerical diagnostics;
- a 99% Wilson upper confidence bound for `P(safe and helpful)` at or below 5% in
  every observed context.

The verdict is invalidated if the ledger changes before deletion. This makes eviction
conservative and evidence-driven while allowing bad cards eventually to free capacity.

## What the smoke run must monitor

The first real experiment is a machinery and calibration gate, not a performance
claim. The analytics must inspect:

- number of decisions, proposals, abstentions, deliveries, controls, and terminals;
- complete probability mass and exact joint treatment/control propensities;
- exposure counts per bank lineage and stable card;
- posterior effect mean, spread, and probability positive for every candidate;
- control and treated invalidity probabilities and safety-set membership;
- reward quadrature errors, scale-boundary mass, and coefficient convergence;
- invalidity optimizer success, gradient residual, and Hessian conditioning;
- parent fitness, semantic coordinates, dynamic coordinates, coverage, and occupancy;
- posterior changes over decision prefixes;
- assignment balance, overlap, effective sample size, and conditional-offer DR
  sensitivity;
- terminal closure, bounded gains, immutable hashes, and ledger accounting.

Expected cold-start behavior is broad uncertainty and near-symmetric safe-card
proposal mass. Balanced validation expects 50/50 delivery/control; normal runs
expect 70/30. Early posterior movement is diagnostic, not enough to declare a
card superior.

## ELI5 Bayesian glossary

| Term | Plain-English meaning in memory v2 |
|---|---|
| Prior | What the model believes before seeing v2 outcomes. Cards start near neutral and uncertain, with a small prior failure rate. |
| Likelihood | The rule describing how probable the observed gains/failures would be under a possible set of card effects. |
| Posterior | The updated range of plausible card effects after combining prior and observed randomized outcomes. It is a distribution, not one score. |
| Bayesian | Keep uncertainty about many possible explanations, then update their plausibility when evidence arrives. |
| Contextual | A card's value may depend on parent fitness, evolution progress, archive state, and behavior coordinates. |
| Hierarchical model | Effects exist at shared, card-family, and card-by-context levels instead of fitting isolated cards independently. |
| Partial pooling | Cards share evidence where appropriate but retain individual effects. It lies between one global score and unrelated per-card scores. |
| Shrinkage | With little data, extreme estimates are pulled toward the common center. Strong evidence can overcome that pull. |
| Treatment | The proposed card was actually shown to the mutator. |
| Control | The same card was proposed but deliberately withheld, giving a comparable no-card outcome. |
| Propensity | The logged probability that the policy assigned a particular action. Here it includes proposal probability and the configured offer probability (0.7 normally, 0.5 in balanced validation). |
| Shared control baseline | Every withheld proposal uses the same contextual no-card baseline; a delivered card adds its contextual effect. |
| Nuisance effect | A necessary background term, such as some parents/cards naturally having different baseline outcomes, that is not itself the causal card effect of interest. |
| Gaussian / Normal | A bell-shaped uncertainty model used for valid numeric gains and coefficient uncertainty. |
| Bernoulli | A two-outcome model used for valid versus invalid terminals. |
| Logistic model | Converts a linear score into a probability between zero and one for invalidity. |
| Residual noise | Outcome variation the known context and card effects do not explain. |
| Measurement uncertainty | Noise from evaluation itself, kept separate from mutation-to-mutation residual variation when paired data allow it. |
| Hurdle model | First model whether the outcome is invalid; then model numeric gain only when valid; finally combine both into usable value. |
| MAP estimate | The single most plausible logistic coefficient setting after considering data and prior. This is used as the center of the safety approximation. |
| Laplace approximation | Approximate the posterior near its most plausible point with a bell shape whose width comes from local curvature. Used for invalidity, not claimed as exact full Bayes. |
| Quadrature | Careful numerical integration over one uncertain value by evaluating weighted points and checking convergence. Used for reward residual noise and safety probability. |
| Posterior world | One coherent draw of all uncertain effects. Cards are compared inside the same world so shared uncertainty is preserved. |
| Posterior probability positive | Fraction of plausible worlds in which delivering the card is better than withholding it for this context. |
| Credible safety constraint | Admit a card only when at least 90% of posterior belief satisfies the configured invalidity limits. |
| Probability matching | Propose cards in proportion to how often they are the best safe positive action across posterior worlds. |
| Exploration | The 5% uniform component that continues testing every safe eligible card instead of permanently locking onto an early winner. |
| Abstention | The no-proposal action wins a posterior world when no safe card has positive usable effect. |
| Covariance / correlation | Uncertainties that move together. Shared posterior worlds preserve the fact that cards depend on common coefficients and evidence. |
| Monte Carlo | Approximate a probability by counting outcomes across many reproducible posterior draws. The 512 proposal worlds define the finite behavior policy. |
| Identifiability | Having enough randomized treated/control evidence to distinguish a card effect from background differences. |
| Cold start | No eligible v2 outcomes yet. The policy honestly relies on priors and exploration rather than fabricated confidence. |
| OPE | Off-policy evaluation: estimate how another delivery rule might have performed using logged randomized data. Current scope changes only the conditional offer gate. |
| Doubly robust (DR) | An OPE estimator combining logged propensities and frozen outcome predictions; it can remain consistent if one of those two components is correct under its assumptions. |
| Effective sample size | A diagnostic for how much usable weighted evidence remains after an OPE policy differs from logged behavior. |

## Claims boundary and deferred work

The current implementation is intentionally precise about what it does not claim:

- no LLM or semantic retrieval during card selection;
- no semantic program representation in the posterior;
- no multi-card slate or crossover attribution;
- no direct archive-contribution credit;
- no full retrieval/proposal-policy OPE;
- no learned hierarchy-scale hyperpriors;
- no exact full-Bayes claim: reward coefficients are conditionally conjugate with
  checked scale integration; invalidity uses MAP/Laplace;
- no sparse posterior representation for hundreds of historical revisions;
- no explicit change-point/state-space model for nonstationarity.

These omissions keep the first core auditable and extensible. The first justified
extension for a substantially larger bank is a randomized, propensity-logged
candidate proposal stage, followed by the same Bayesian selector within its support.

## Implementation map

- Candidate bank: [`gigaevo/memory_v2/candidates.py`](../gigaevo/memory_v2/candidates.py)
- Typed context and revisions: [`gigaevo/memory_v2/models.py`](../gigaevo/memory_v2/models.py)
- MAP-Elites snapshot: [`gigaevo/memory_v2/context.py`](../gigaevo/memory_v2/context.py)
- Hierarchical features: [`gigaevo/memory_v2/features.py`](../gigaevo/memory_v2/features.py)
- Reward/safety posterior: [`gigaevo/memory_v2/posterior.py`](../gigaevo/memory_v2/posterior.py)
- Safe probability matching: [`gigaevo/memory_v2/policy.py`](../gigaevo/memory_v2/policy.py)
- Atomic decision provider: [`gigaevo/memory_v2/provider.py`](../gigaevo/memory_v2/provider.py)
- Causal ledger: [`gigaevo/memory_v2/ledger.py`](../gigaevo/memory_v2/ledger.py)
- Content-only v2 writer bridge: [`gigaevo/memory_v2/writer.py`](../gigaevo/memory_v2/writer.py)
- Conservative retirement: [`gigaevo/memory_v2/eviction.py`](../gigaevo/memory_v2/eviction.py)
- Production configuration: [`config/memory/v2.yaml`](../config/memory/v2.yaml)
- Smoke launcher and analytics: [`experiments/hover/memory_v2_smoke`](../experiments/hover/memory_v2_smoke)
