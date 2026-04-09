# GAN Training Techniques Survey for Adversarial Co-Evolution

**Purpose**: Comprehensive mapping of GAN training innovations to GigaEvo's adversarial
co-evolution system, targeting the Improver stagnation problem observed in
`adversarial/heilbron-prover`.

**System context**: G = Constructor population (MAP-Elites archive of Python programs placing
11 points in a triangle to maximize minimum triangle area); D = Improver population (MAP-Elites
archive of Python programs that try to improve G's placements). Both are evolved via LLM-guided
mutation (Qwen3-235B). No gradients exist -- mutation is stochastic LLM code generation.

**Core problem**: D stagnates because as G improves (100% resistance), D cannot find
improvements. P2_B in heilbron-prover had 0% acceptance rate for 45 consecutive generations.

---

## Part I: Training Dynamics and Stability

### 1.1 Wasserstein Distance / Earth Mover's Distance

**Paper**: Arjovsky, Chintala & Bottou, "Wasserstein GAN" (ICML 2017).

**Problem solved in GANs**: The Jensen-Shannon divergence used in vanilla GANs saturates when
the generator and discriminator distributions have disjoint support (which happens early in
training and whenever the discriminator is too good). When JS divergence saturates, the
generator gradient vanishes -- it receives zero useful learning signal. This manifests as
training instability and mode collapse.

**Core mechanism**: Replace the JS divergence with the Wasserstein-1 (Earth Mover's) distance:
W(P_r, P_g) = inf_{gamma in Pi(P_r, P_g)} E_{(x,y)~gamma}[||x - y||]. Unlike JS, the
Wasserstein distance is continuous and differentiable almost everywhere, providing meaningful
gradients even when the distributions don't overlap. The discriminator becomes a "critic" --
it outputs a real-valued score (not a probability) that approximates the Wasserstein distance.
The critic must be a 1-Lipschitz function, enforced via weight clipping (later improved by
gradient penalty).

**Key insight for co-evolution**: The fundamental problem is the same. When Constructors reach
100% resistance, the Improver fitness signal saturates at 0 (binary: "could not improve").
This is the JS-divergence saturation analog -- the Improver receives no gradient about *how
close* it came to finding an improvement.

**Mapping to GigaEvo**:

1. **Replace binary resistance with continuous distance**: Instead of scoring Improvers as 0/1
   (improved or didn't), compute a continuous "closeness to improvement" metric. For the
   Heilbronn problem: if the Improver's modified configuration has min_area = 0.0350 and the
   Constructor's original had min_area = 0.0355, the Improver gets credit proportional to how
   close it came (delta = -0.0005, better than delta = -0.010). Concretely:

   ```python
   # Current: binary
   improver_fitness = 1.0 if improved_min_area > original_min_area else 0.0

   # Wasserstein analog: continuous
   delta = improved_min_area - original_min_area
   improver_fitness = sigmoid(delta / temperature)  # smooth transition around delta=0
   # or: improver_fitness = max(0, delta) + epsilon * max(0, -delta + margin)
   ```

2. **Partial credit for near-improvements**: An Improver that moves one point and gets 99.9% of
   the way to an improvement should receive much higher fitness than one that random-shuffles
   all 11 points. The current binary signal treats both as equal failures.

3. **Implementation**: Modify `evaluate.py` in `problems/heilbron_adversarial/pop_b/` to return
   a continuous fitness value. Change the MAP-Elites selection criterion for Pop B from binary
   success to continuous delta. This requires changing `metrics.yaml` and the fitness
   computation in `validate.py`.

**Experiment idea**: "Continuous Improver fitness" -- replace binary improvement signal with
continuous delta-based fitness. Measure whether Improver stagnation is reduced.

---

### 1.2 Gradient Penalty (WGAN-GP)

**Paper**: Gulrajani, Ahmed, Arjovsky, Dumoulin & Courville, "Improved Training of Wasserstein
GANs" (NeurIPS 2017).

**Problem solved in GANs**: WGAN's weight clipping to enforce the Lipschitz constraint is crude
-- it biases the critic toward simple functions (capacity underuse) and can still lead to
exploding/vanishing gradients. Weight clipping pushes weights to the extremes of the clipping
range.

**Core mechanism**: Replace weight clipping with a gradient penalty term added to the critic
loss: lambda * E_{x_hat}[(||nabla_{x_hat} D(x_hat)||_2 - 1)^2], where x_hat are points
sampled uniformly along lines between real and generated data. This softly enforces the
1-Lipschitz constraint by penalizing gradients that deviate from unit norm.

**Key insight for co-evolution**: The gradient penalty ensures the critic (discriminator)
remains *informative everywhere in the input space*, not just at the boundary between real and
fake. It prevents the critic from becoming overconfident (outputting extreme values that
provide no useful gradient to G).

**Mapping to GigaEvo**:

1. **Penalize Improver over-specialization**: An Improver that achieves huge improvements on
   easy Constructors but zero improvement on hard Constructors is analogous to a critic with
   non-unit gradients -- informative in one region, useless in another. Penalize Improvers
   whose performance variance across opponents is extreme:

   ```python
   # Improver evaluated against K Constructors, gets deltas [d_1, ..., d_K]
   mean_delta = mean(deltas)
   variance_penalty = lambda_gp * variance(deltas)
   improver_fitness = mean_delta - variance_penalty
   ```

   This encourages Improvers to develop *general* improvement strategies rather than
   tricks that work on one Constructor but fail on all others.

2. **Interpolated difficulty evaluation**: Inspired by the interpolation in WGAN-GP, evaluate
   Improvers not just against the best and worst Constructors but against "interpolated"
   difficulty levels. In practice: maintain a difficulty-sorted archive of Constructors and
   sample opponents from the full spectrum, not just the extremes. This is closely related to
   GenEnv's alpha-curriculum (Section 5.4).

3. **Implementation**: In `FetchOpponentResultsStage`, modify opponent selection to sample from
   the full Constructor archive stratified by difficulty (min_area), rather than always picking
   the top-K. Add a fitness penalty term for variance across opponents.

---

### 1.3 Spectral Normalization

**Paper**: Miyato, Kataoka, Koyama & Yoshida, "Spectral Normalization for Generative
Adversarial Networks" (ICLR 2018).

**Problem solved in GANs**: Stabilizes discriminator training by constraining its Lipschitz
constant. Unlike gradient penalty (applied to loss), spectral normalization is applied
directly to network weights: W_SN = W / sigma(W) where sigma(W) is the largest singular value.
This is cheaper than gradient penalty (no extra forward-backward pass) and provides a tighter,
more uniform Lipschitz bound.

**Core mechanism**: For each weight matrix W in the discriminator, divide by its spectral norm
(largest singular value) at each training step. This ensures ||D(x) - D(y)|| <= ||x - y||
for all inputs, preventing the discriminator from creating sharp decision boundaries that
provide vanishing gradients to the generator.

**Key insight for co-evolution**: Spectral normalization prevents the discriminator from being
"too powerful" -- from creating decision boundaries so sharp that the generator gets no useful
signal. The analog in co-evolution is that the Improver becomes so capable that it trivially
breaks easy Constructors but provides no signal for hard ones.

**Mapping to GigaEvo**:

1. **Virulence cap on Improvers** (from literature brief R6): Limit how much an Improver can
   change a configuration. If an Improver moves all 11 points, it's essentially generating
   a new configuration from scratch -- this is the "unbounded discriminator" problem. Cap
   the Hamming distance (number of points moved) or the L2 distance in coordinate space:

   ```python
   # In Improver's evaluate.py
   points_moved = np.sum(np.any(improved_points != original_points, axis=1))
   if points_moved > K:  # K=3-4
       return {"is_valid": False, "reason": "moved too many points"}

   # Or: soft penalty
   move_distance = np.linalg.norm(improved_points - original_points)
   fitness = delta_min_area - lambda_sn * max(0, move_distance - threshold)
   ```

2. **Cap Improver improvement magnitude**: An Improver that claims +0.030 improvement (nearly
   doubling min_area) is almost certainly rebuilding from scratch. Cap credited improvement
   at some fraction of the Constructor's current fitness:

   ```python
   capped_delta = min(delta, cap_fraction * original_min_area)
   ```

3. **Implementation**: Add constraints to `problems/heilbron_adversarial/pop_b/validate.py`.
   The virulence cap is problem-specific but the concept generalizes: limit the "power" of
   the adversarial population to prevent it from being trivially effective on easy targets
   while useless on hard ones.

**Experiment idea**: "Virulence-capped Improvers" -- limit Improvers to modifying at most K=3
points. Test whether this forces more targeted, informative improvements.

---

### 1.4 Two-Timescale Update Rule (TTUR)

**Paper**: Heusel, Ramsauer, Unterthiner, Nessler & Hochreiter, "GANs Trained by a Two
Time-Scale Update Rule Converge to a Local Nash Equilibrium" (NeurIPS 2017).

**Problem solved in GANs**: Simultaneous updates to G and D often fail to converge -- the
optimization oscillates or diverges. When both players update at the same rate, neither can
stabilize because the other is constantly shifting.

**Core mechanism**: Use different learning rates for G and D: eta_D >> eta_G (typically 4:1).
The faster D converges to an approximate best response given the current G, while the slower G
makes small steps informed by a near-optimal D. The theoretical result: under mild regularity
conditions, this converges to a local Nash equilibrium. The FID (Frechet Inception Distance)
metric was also introduced in this paper as a byproduct.

**Key insight for co-evolution**: This is the single most directly actionable technique for the
Improver stagnation problem. If Improvers get the same number of evaluations per generation as
Constructors, but face a harder task (finding improvements is harder than generating
configurations), they are effectively under-trained relative to the Constructor. TTUR says:
give the harder role more compute.

**Mapping to GigaEvo**:

1. **Asymmetric generation ratios**: Give Improvers R times more evaluations per epoch than
   Constructors (R = 3-5). In steady-state engine terms, set
   `max_mutations_per_generation` differently per population:

   ```yaml
   # experiment.yaml
   runs:
     P1_A:  # Constructor
       max_mutations_per_generation: 8
     P1_B:  # Improver
       max_mutations_per_generation: 24  # 3x more
   ```

   This gives Improvers 3x more attempts to find improvements per Constructor epoch.
   Constructors advance slowly (more stable fitness target for Improvers), while Improvers
   get more exploration budget.

2. **Asymmetric archive sizes**: Larger Improver archive = more diverse attack strategies
   maintained. Constructors can have a smaller archive focused on the highest-quality
   configurations:

   ```yaml
   P1_A:  # Constructor
     max_elites_per_generation: 8
   P1_B:  # Improver
     max_elites_per_generation: 24  # 3x larger archive
   ```

3. **Alternating phase lengths**: Instead of strict 1:1 alternation (GAME-style), run the
   Improver for 3 epochs per 1 Constructor epoch. The MainRunSyncHook already supports
   generation drift -- relax the sync constraint from 1 to R.

4. **Implementation**: Modify `experiment.yaml` per-run configs. The steady-state engine
   already supports different `max_mutations_per_generation` per run. The MainRunSyncHook
   needs modification to tolerate asymmetric generation counts.

**Experiment idea**: "TTUR-inspired asymmetric compute" -- 3:1 Improver:Constructor evaluation
ratio. The single highest-priority technique to test based on GAN theory and the specific
stagnation pattern observed.

---

### 1.5 R1/R2 Gradient Penalties

**Paper**: Mescheder, Geiger & Nowozin, "Which Training Methods for GANs do actually
Converge?" (ICML 2018).

**Problem solved in GANs**: Provided theoretical analysis showing that many GAN training methods
do NOT converge, even on simple problems. Identified that gradient penalties on the
*discriminator* applied only at real data points (R1) or generated data points (R2) are
sufficient and simpler than WGAN-GP's interpolation-based penalty.

**Core mechanism**: R1 penalty: lambda/2 * E_{x~P_r}[||nabla_x D(x)||^2]. Applied only at
real data points. The key insight is that you only need to regularize the discriminator near
the data manifold, not everywhere. R1 alone is sufficient for local convergence. R2 is the
symmetric version applied at generated points.

**Key insight for co-evolution**: R1 regularization says: keep the discriminator smooth *near
the real data*. The co-evolution analog: keep the Improver's evaluation smooth *near good
Constructors*. If the Improver can differentiate between a min_area=0.0355 and
min_area=0.0356 Constructor but treats both as "impossible to improve", the fitness landscape
is discontinuous where it matters most.

**Mapping to GigaEvo**:

1. **Smooth evaluation near the frontier**: When evaluating Improvers against top-tier
   Constructors (the "real data" in GAN terms), use a more sensitive fitness function.
   Instead of binary "improved / didn't improve", measure the *gradient of improvement
   difficulty*:

   ```python
   # For Constructors near the frontier (top 20% by actual_fitness):
   # Use finer-grained fitness with smaller epsilon
   if constructor_quality > 0.8:
       improver_fitness = continuous_delta_with_high_precision(delta)
   # For easy Constructors (bottom 50%):
   # Binary is fine -- improvement is trivial
   else:
       improver_fitness = float(delta > 0)
   ```

2. **Adaptive fitness resolution**: As the Constructor archive quality increases, automatically
   increase the precision of the Improver fitness signal. This mirrors R1's focus on
   regularization where it matters (near good solutions).

---

### 1.6 Exponential Moving Average of G Weights (EMA)

**Paper**: Karras, Aila, Laine & Lehtinen (ProGAN, ICLR 2018); later refined in StyleGAN 1/2/3.
Also Yazici et al., "The Unusual Effectiveness of Averaging in GAN Training" (ICLR 2019).

**Problem solved in GANs**: Generator weights oscillate during training due to the adversarial
dynamic. The instantaneous G at any training step may produce artifacts. EMA smooths these
oscillations: theta_G_ema = beta * theta_G_ema + (1 - beta) * theta_G, with beta ~0.999.
The EMA model is used for evaluation/inference, not training.

**Core mechanism**: Maintain a running exponential moving average of generator parameters.
This averages out the oscillations caused by the adversarial training dynamic and produces
more stable outputs. The EMA generator typically produces higher quality and more diverse
outputs than the instantaneous generator.

**Key insight for co-evolution**: In MAP-Elites co-evolution, the archive IS the "model". The
archive changes every generation as new programs are accepted. But the opponent sees a snapshot
of this archive. If the archive is volatile (programs entering and exiting rapidly), opponents
train against a moving target.

**Mapping to GigaEvo**:

1. **Hall-of-Fame archive with decay weights**: Instead of evaluating Improvers only against the
   current Constructor archive, maintain a weighted hall-of-fame where recent champions have
   high weight and older ones decay exponentially:

   ```python
   # Hall of fame: (program, weight) pairs
   hof_weight[program] = beta^(current_gen - program_gen) * original_fitness
   # Sample opponents from HoF with probability proportional to weight
   ```

   This smooths the opponent distribution the same way EMA smooths generator weights.

2. **Sticky elite slots**: Once a program enters the top-K of the archive, keep it for at least
   M generations (a "cooldown" period) even if a marginally better program appears. This
   prevents rapid archive churn that destabilizes opponent evaluation.

3. **EMA fitness signal**: When computing a program's fitness across opponent evaluations from
   multiple generations, use EMA of the fitness values rather than the latest snapshot:

   ```python
   program.ema_fitness = beta * program.ema_fitness + (1-beta) * latest_fitness
   ```

4. **Implementation**: Modify the archive acceptance criterion or add a separate HoF data
   structure alongside the MAP-Elites archive. The HoF is read-only for opponent selection;
   the MAP-Elites archive continues to drive selection as before.

**Experiment idea**: "Smoothed opponent archive" -- EMA-weighted hall-of-fame for opponent
selection. Reduces target oscillation for both populations.

---

## Part II: Architecture Innovations

### 2.1 Progressive Growing (ProGAN)

**Paper**: Karras, Aila, Laine & Lehtinen, "Progressive Growing of GANs for Improved Quality,
Stability, and Variation" (ICLR 2018).

**Problem solved in GANs**: Training GANs on high-resolution images from scratch is unstable.
Low-resolution structure must be learned first before adding high-frequency details.

**Core mechanism**: Start training on 4x4 images. Once stable, add layers to both G and D to
handle 8x8. Continue doubling resolution until the target resolution (1024x1024). Each
resolution transition uses a fade-in: the new higher-resolution layers are blended in smoothly
(alpha goes from 0 to 1) to avoid sudden shocks. The key: both G and D see the same resolution
at each stage, so the "difficulty" is always matched.

**Key insight for co-evolution**: Start both populations on an easier version of the problem,
then progressively increase difficulty. This is a curriculum learning strategy that ensures
both populations stay in the "zone of proximal development" where learning is productive.

**Mapping to GigaEvo**:

1. **Progressive problem scaling**: Start with fewer points (e.g., n=5 points in a triangle,
   C(5,3)=10 triplets) where the Heilbronn problem is easier. Once both populations stabilize,
   increase to n=7, then n=9, then n=11. Each transition uses a warm-start from the previous
   level's best programs (modified to handle more points).

   ```
   Phase 1 (gen 0-20):  n=5 points, 10 triplets
   Phase 2 (gen 20-40): n=7 points, 35 triplets
   Phase 3 (gen 40-60): n=9 points, 84 triplets
   Phase 4 (gen 60+):   n=11 points, 165 triplets
   ```

2. **Progressive opponent count**: Start evaluating against 1 opponent, then 3, then 5. The
   current heilbron-prover already ramps `n_opponents` -- formalize this as a progressive
   schedule.

3. **Progressive constraint tightening**: Start with a loose definition of "improvement"
   (delta > 0.001), then tighten to delta > 0.0001, then delta > 0. This keeps the Improver
   task achievable at every stage.

4. **Implementation**: Requires modifying `evaluate.py` to accept a `difficulty_level`
   parameter. The watchdog or a new `ProgressiveScheduler` stage advances the difficulty
   based on generation count or performance milestones.

**Experiment idea**: "Progressive Heilbronn" -- start at n=7, transition to n=11. Test whether
curriculum learning prevents Improver stagnation by keeping difficulty matched.

---

### 2.2 StyleGAN 1/2/3: Mapping Network and Style Injection

**Papers**: Karras, Laine & Aila, "A Style-Based Generator Architecture for GANs" (CVPR 2019);
Karras, Laine, Aittala et al., "Analyzing and Improving the Image Quality of StyleGAN"
(CVPR 2020); Karras, Aittala, Laine et al., "Alias-Free Generative Adversarial Networks"
(NeurIPS 2021).

**Problem solved in GANs**: Disentangling the latent space to give the generator fine-grained
control over different aspects of generation. The mapping network f: Z -> W transforms a
simple latent z into an intermediate latent w that is better suited for style injection.
Style mixing enables multi-scale control.

**Core mechanism**: The mapping network is an 8-layer MLP that maps z to w (an intermediate
latent space W). w is then transformed into per-layer "styles" (scale and bias) that modulate
feature maps at each resolution via adaptive instance normalization (AdaIN). Noise is injected
at each layer for stochastic variation. Key innovations across versions: weight
demodulation (SG2), path length regularization (SG2), equivariance and alias-free operations
(SG3).

**Key insight for co-evolution**: StyleGAN decouples *what* to generate (high-level style) from
*how* to generate it (per-layer details). The analog: separate the *strategy* of a program from
its *implementation*.

**Mapping to GigaEvo**:

1. **Structured mutation prompts with hierarchical control**: When mutating a Constructor,
   provide separate prompt sections for:
   - **Global strategy** (W-space): "Place points along equidistant arcs" vs "Use repulsive
     force simulation" vs "Optimize worst triangle directly"
   - **Local adjustments** (noise injection): "Perturb point 7 by 0.01 in the x-direction"

   The mutation LLM first decides which strategic level to modify, then makes changes at
   that level. This prevents mutations that simultaneously change strategy AND implementation
   (which are the analog of entangled latent spaces).

2. **Style mixing for crossover**: When performing crossover (num_parents=2), take the
   high-level strategy from one parent and the low-level implementation from another:

   ```
   Parent A: "simulated annealing strategy" + "specific cooling schedule"
   Parent B: "repulsive force strategy" + "specific force parameters"
   Child: "simulated annealing strategy" + "specific force parameters" (mixed)
   ```

   This is the evolutionary analog of style mixing regularization.

3. **Implementation**: Modify mutation prompts to explicitly separate strategy from
   implementation. This is a prompt engineering change, not a code change. The LLM can be
   instructed to first identify the parent's strategy, then decide whether to modify at the
   strategy or implementation level.

---

### 2.3 Self-Attention GAN (SAGAN)

**Paper**: Zhang, Goodfellow, Metaxas & Odena, "Self-Attention Generative Adversarial
Networks" (ICML 2019).

**Problem solved in GANs**: Convolutional GANs struggle with long-range dependencies -- they can
generate realistic local textures but fail at global structure (e.g., dogs with too many legs).
Convolutions have limited receptive fields.

**Core mechanism**: Add self-attention layers to both G and D. Self-attention computes:
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V, where Q, K, V are learned projections of
feature maps. This allows each spatial position to attend to all other positions, capturing
global structure. SAGAN also introduced spectral normalization for G (not just D) and
the TTUR learning rate schedule.

**Key insight for co-evolution**: Programs in MAP-Elites are evaluated independently. There is no
mechanism for one program to "attend to" or be informed by the structural patterns across the
entire archive. A Constructor doesn't know that 80% of the archive uses the same strategy.

**Mapping to GigaEvo**:

1. **Archive-wide pattern analysis in mutation prompts**: Before mutating a program, analyze the
   entire archive to identify dominant strategies, missing strategy niches, and
   underrepresented regions:

   ```
   ARCHIVE ANALYSIS (self-attention analog):
   - 6/8 elites use repulsive force simulation (dominant strategy)
   - 1/8 uses simulated annealing (minority strategy)
   - 1/8 uses direct optimization (minority strategy)
   - No elite uses geometric construction (missing niche)
   Suggestion: explore geometric construction approaches.
   ```

2. **Cross-program attention for mutation**: The mutation prompt includes not just the parent
   program but a summary of how other archive members differ:

   ```
   Your program uses strategy X.
   The best-performing program uses strategy Y (fitness 0.035).
   The most different program from yours uses strategy Z (fitness 0.030).
   ```

   This gives the LLM "attention" across the archive.

3. **Implementation**: Add an `ArchiveAnalysisStage` to the mutation pipeline that reads the
   full archive, clusters programs by strategy (using the LLM or simple code similarity), and
   produces a summary injected into the mutation context. This is expensive (one LLM call per
   mutation) but could be cached per generation.

---

### 2.4 MSG-GAN (Multi-Scale Gradient)

**Paper**: Karnewar & Wang, "MSG-GAN: Multi-Scale Gradients for Generative Adversarial
Networks" (CVPR 2020).

**Problem solved in GANs**: The discriminator only sees the final generated image. Gradient
signal must propagate through all generator layers. Intermediate layers get weak, distorted
gradients. Mode collapse and training instability result.

**Core mechanism**: Connect the generator's intermediate feature maps directly to the
discriminator at matching scales. The discriminator receives multi-scale inputs: the final
image AND intermediate representations at 4x4, 8x8, 16x16, etc. This provides direct gradient
paths from D to each G layer, eliminating the gradient bottleneck.

**Key insight for co-evolution**: Evaluating only the final output (the point configuration)
discards information about the *process* that generated it. The Improver sees only the final
placement, not the intermediate steps (force calculations, annealing schedule, etc.).

**Mapping to GigaEvo**:

1. **Multi-scale evaluation**: Evaluate Constructor programs not just on their final output but
   on intermediate qualities:
   - **Scale 1 (global)**: min_area of the full 11-point configuration
   - **Scale 2 (local)**: distribution of pairwise distances between points
   - **Scale 3 (process)**: convergence behavior of the algorithm (if it's iterative)

   Improver fitness is a weighted combination of improvements at each scale:
   ```python
   improver_fitness = w1*delta_min_area + w2*delta_spread + w3*delta_convergence
   ```

2. **Expose intermediate state to Improvers**: If the Constructor uses an iterative algorithm
   (simulated annealing, gradient descent), expose the intermediate configurations at steps
   t=10, t=50, t=100. The Improver can then target the weakest intermediate state, providing
   more granular feedback.

3. **Implementation**: Modify Constructor's `entrypoint()` to optionally return intermediate
   states. Add an `IntermediateStateEvaluator` that assesses improvement potential at each
   intermediate point.

---

## Part III: Mode Collapse Prevention

### 3.1 Minibatch Discrimination

**Paper**: Salimans, Goodfellow, Zaremba, Cheung, Radford & Chen, "Improved Techniques for
Training GANs" (NeurIPS 2016).

**Problem solved in GANs**: Mode collapse -- the generator produces limited diversity (e.g.,
generating only one type of face). The discriminator can't detect this because it evaluates
samples independently.

**Core mechanism**: Compute pairwise distances between samples within a minibatch. For each
sample, compute the L1 distance of its features to all other samples in the batch, apply an
exponential kernel, and sum. This "closeness" feature is concatenated to the discriminator's
input. The discriminator can now detect when all generated samples are too similar (low
pairwise distances), penalizing mode collapse.

**Key insight for co-evolution**: MAP-Elites already provides diversity pressure via behavioral
descriptors. But behavioral descriptors might not capture *strategic diversity*. The archive
could contain 8 programs that all use the same strategy but differ in minor parameters,
occupying different behavioral descriptor cells.

**Mapping to GigaEvo**:

1. **Strategy-level diversity enforcement**: When evaluating a Constructor against the
   Constructor archive (for fitness), penalize programs whose strategies are too similar to
   existing archive members. Measure strategy similarity by:
   - Code edit distance (AST diff)
   - Output similarity (L2 distance between point configurations)
   - Shared subroutines (import overlap)

   ```python
   # In fitness computation:
   avg_distance = mean([code_distance(program, elite) for elite in archive])
   diversity_bonus = alpha * sigmoid(avg_distance - threshold)
   adjusted_fitness = raw_fitness + diversity_bonus
   ```

2. **Opponent diversity requirement**: When selecting K opponents for evaluation, ensure they
   are *diverse* (from different behavioral descriptor cells, or using K-means clustering on
   strategies as GAME does). This prevents both populations from co-adapting to a single narrow
   strategy.

3. **Batch evaluation signal for Improvers**: Evaluate each Improver against a batch of N
   Constructors simultaneously. Report the Improver's success rate across the batch:
   ```
   Improver X: improved 3/5 Constructors (successes on C1, C3, C5; failures on C2, C4)
   ```
   This gives the Improver signal about *which types* of Constructors it can and cannot crack.

4. **Implementation**: The `FetchOpponentResultsStage` already selects K opponents. Modify the
   selection criterion to maximize diversity (K-means or archive cell spread) rather than
   selecting top-K by fitness.

---

### 3.2 Unrolled GANs

**Paper**: Metz, Poole, Pfau & Sohl-Dickstein, "Unrolled Generative Adversarial Networks"
(ICLR 2017).

**Problem solved in GANs**: The generator optimizes against the *current* discriminator, but by
the time G updates, D has also changed. This myopic optimization leads to oscillation and mode
collapse. G needs to anticipate how D will respond to its updates.

**Core mechanism**: Instead of optimizing G against D_current, unroll D for K steps:
D_unrolled = D after K gradient updates on G's current output. Then optimize G against
D_unrolled. This gives G a "lookahead" -- it produces outputs that are robust to the next K
discriminator updates. Computationally expensive (K extra forward/backward passes per G step)
but dramatically reduces mode collapse.

**Key insight for co-evolution**: Constructors currently optimize against the *current* Improver
archive. But by the time the Constructor archive is updated, the Improver archive has shifted.
The Constructor needs to anticipate future Improver evolution.

**Mapping to GigaEvo**:

1. **Lookahead opponent evaluation**: Instead of evaluating Constructors against the current
   Improver archive, evaluate against a *projected* future archive. Concretely: run the
   Improver population for K extra generations in a sandbox (using cheap evaluation), then
   use the resulting archive as opponents for Constructor evaluation:

   ```
   Current Improver archive (gen N) -> Run K more Improver generations (sandbox)
   -> Projected Improver archive (gen N+K) -> Evaluate Constructors against this
   ```

   This is expensive but gives Constructors a preview of upcoming attacks.

2. **Simplified version -- Historical Improver trajectory**: Instead of full unrolling, show
   the Constructor the *trajectory* of Improver improvements: "At gen 5, best Improver did X.
   At gen 10, best Improver did Y. At gen 15, best Improver did Z. Extrapolate: by gen 20,
   Improvers will likely try W. Defend against W."

   This is cheaper (no sandbox needed) and gives the mutation LLM a "gradient direction" for
   Improver evolution.

3. **Implementation**: Option 1 (full unrolling) requires significant infrastructure --
   sandboxed Redis DBs, temporary engine instances. Not practical for the next experiment.
   Option 2 (trajectory injection) is a prompt engineering change: add "Improver evolution
   history" to the Constructor's mutation context. Achievable within the current pipeline.

**Experiment idea**: "Improver trajectory injection" -- show Constructors the last 5 generations
of Improver evolution in their mutation prompt. Test whether this anticipatory signal improves
Constructor robustness.

---

### 3.3 Historical Averaging

**Paper**: Salimans et al., "Improved Techniques for Training GANs" (NeurIPS 2016).

**Problem solved in GANs**: Training oscillation. G and D chase each other in parameter space
without converging. Historical averaging adds a penalty that keeps parameters close to their
historical average: ||theta - (1/t) sum_{i=1}^{t} theta_i||^2. This dampens oscillation by
penalizing large deviations from the running mean.

**Core mechanism**: For both G and D, add a regularization term to the loss:
L_hist = lambda * ||theta_t - theta_bar_t||^2, where theta_bar_t = (1/t) sum theta_i.
This is a form of implicit averaging that encourages convergence by making the optimization
trajectory "sticky" -- it resists sudden parameter jumps.

**Key insight for co-evolution**: Archive churn in MAP-Elites can cause the opponent distribution
to oscillate. If the Constructor archive suddenly shifts from strategy A to strategy B, the
Improver's training signal changes abruptly, potentially destabilizing Improver evolution.

**Mapping to GigaEvo**:

1. **Archive momentum / conservative acceptance**: Make archive acceptance more conservative
   when the archive has been stable for many generations. A new program must not just beat the
   current occupant but beat it by a margin proportional to the occupant's tenure:

   ```python
   # Acceptance criterion with historical averaging
   required_margin = base_margin + tenure_penalty * occupant_generations_held
   accept = candidate_fitness > occupant_fitness + required_margin
   ```

   This makes the archive "sticky" -- established programs are harder to dislodge, reducing
   oscillation.

2. **Ensemble opponent evaluation**: Evaluate against a mix of current archive members AND
   historical archive members (hall-of-fame). Weight recent members higher but keep historical
   members in the evaluation pool. This is the EMA approach from Section 1.6 reframed as
   historical averaging.

3. **Implementation**: Modify the archive acceptance criterion in `map_elites_archive.py` to
   include a tenure-based margin. Or maintain a separate HoF (see Section 1.6).

---

### 3.4 Diversity-Sensitive Loss / Mode-Seeking Loss

**Paper**: Mao, Li, Xie, Lau, Wang & Smolley, "On the Effectiveness of Least Squares
Generative Adversarial Networks" (related); Yang et al., "Diversity-Sensitive Conditional
Generative Adversarial Networks" (ICLR 2019).

**Problem solved in GANs**: Standard GAN losses incentivize the generator to produce the single
"safest" output per input condition. The mode-seeking loss explicitly rewards diversity:
L_ms = max(0, 1 - ||G(z1) - G(z2)|| / ||z1 - z2||) -- the generator is penalized if two
different latent codes produce similar outputs (mode collapse).

**Core mechanism**: For a pair of latent codes (z1, z2), compute the ratio of output diversity
to input diversity. If the generator maps different z's to similar outputs, the ratio is low
and the loss is high. This forces the generator to use the full latent space, preventing mode
collapse.

**Key insight for co-evolution**: MAP-Elites provides structural diversity via behavioral
descriptors. But within each cell, there's no diversity pressure -- the cell contains exactly
one program. If all cells converge to similar strategies (just with different parameter
values), strategic diversity is lost.

**Mapping to GigaEvo**:

1. **Cross-cell strategy diversity bonus**: When a new program is accepted into the archive,
   give it a bonus if its *strategy* (not just its behavioral descriptor) differs from
   neighboring cells:

   ```python
   strategy_novelty = mean([strategy_distance(program, neighbor) for neighbor in adjacent_cells])
   adjusted_fitness = raw_fitness + beta * strategy_novelty
   ```

2. **Diverse mutation instruction**: When generating mutations, explicitly instruct the LLM to
   produce outputs that are *strategically different* from the parent:

   ```
   DIVERSITY INSTRUCTION:
   Your parent uses simulated annealing. Previous mutations have also used
   simulated annealing variants. This time, try a COMPLETELY DIFFERENT approach
   (geometric construction, force-directed, random search, etc.)
   ```

3. **Niching pressure on Improvers**: Prevent all Improvers from converging to the same attack
   strategy. If 7/8 Improvers use "move the worst-triangle vertex," explicitly encourage
   the 8th to try a different approach.

---

### 3.5 PacGAN

**Paper**: Lin, Khetan, Fakhri & Makhzani, "PacGAN: The Power of Two Samples in Generative
Adversarial Networks" (NeurIPS 2018).

**Problem solved in GANs**: Mode collapse when the discriminator evaluates single samples. Even
with minibatch discrimination, subtle mode collapse can persist.

**Core mechanism**: Pack multiple generated samples together and feed the *pack* to the
discriminator. D receives (x1, x2, ..., xm) as a single input. If the generator has mode
collapse (all samples identical), the packed input reveals this immediately -- all elements
in the pack are the same. The discriminator can trivially distinguish this from packs of real
(diverse) data. Pack size m controls the trade-off: larger m = stronger anti-collapse but
more expensive.

**Key insight for co-evolution**: Evaluate Constructors not individually but as *packs*. If the
Constructor archive produces 8 programs that all generate the same point configuration (or
nearly), this is mode collapse and should be penalized.

**Mapping to GigaEvo**:

1. **Pack-based Improver evaluation**: Give the Improver a *pack* of K=3-5 Constructor outputs
   at once. The Improver must find improvements for the *worst* member of the pack. This
   rewards Improvers that can handle diverse Constructors:

   ```python
   # Pack evaluation
   pack = sample_diverse_constructors(archive, k=3)
   min_improvement = min([improver(c) - c.fitness for c in pack])
   improver_fitness = min_improvement  # Must improve ALL members
   ```

2. **Pack-based Constructor evaluation**: Evaluate a Constructor's output against *packs* of
   Improvers. The Constructor's resistance is its worst-case resistance across the pack:

   ```python
   pack = sample_diverse_improvers(archive, k=3)
   worst_case_resistance = min([resistance(constructor, imp) for imp in pack])
   constructor_fitness = alpha * quality + (1-alpha) * worst_case_resistance
   ```

3. **Implementation**: Modify `FetchOpponentResultsStage` to use pack-based evaluation.
   The n_opponents parameter already controls how many opponents are evaluated -- the change
   is in how the results are aggregated (min instead of mean).

---

## Part IV: Training Tricks That Actually Matter

### 4.1 Label Smoothing

**Paper**: Salimans et al. (NeurIPS 2016); also Szegedy et al., "Rethinking the Inception
Architecture" (2016) for the general technique.

**Problem solved in GANs**: The discriminator becomes overconfident -- it outputs 0.0 for fake
and 1.0 for real with absolute certainty. This creates sharp gradients that destabilize
training and provide a misleading signal to the generator.

**Core mechanism**: Replace hard labels (0/1) with soft labels (0.0/0.9 or 0.1/0.9). One-sided
label smoothing (only smooth real labels: 1.0 -> 0.9, keep fake at 0.0) is preferred, as
smoothing fake labels encourages the generator to produce samples near the decision boundary
rather than near real data.

**Key insight for co-evolution**: Binary fitness signals ("improved" / "didn't improve") create
the same overconfidence problem. A Constructor with 100% resistance is scored as "perfect" even
if Improvers came within epsilon of breaking it.

**Mapping to GigaEvo**:

1. **Soft resistance scores**: Replace binary resistance with a smoothed version:

   ```python
   # Current: binary
   resistance = sum(1 for imp in improvers if imp.delta <= 0) / len(improvers)

   # Smoothed:
   resistance = sum(sigmoid(-imp.delta / temperature) for imp in improvers) / len(improvers)
   # temperature controls smoothing: lower T = closer to binary, higher T = softer
   ```

   With smoothing, a Constructor that has resistance=0.999 (Improvers almost-but-not-quite
   succeeded) is distinguished from resistance=1.0 (Improvers weren't even close).

2. **One-sided smoothing for Constructors**: Smooth the "perfect resistance" signal (1.0 -> 0.9)
   but keep "broken" as a hard signal (0.0 stays 0.0). This keeps Constructors "hungry" --
   even at 100% resistance they see room for improvement.

3. **Implementation**: Modify the resistance computation in `evaluate.py`. This is a one-line
   change that could have significant impact on Constructor stagnation (the flip side of
   Improver stagnation -- Constructors stop improving because their fitness saturates).

---

### 4.2 Instance Noise

**Paper**: Sonderby, Caballero, Theis, Shi & Huszar, "Amortised MAP Inference for Image
Super-Resolution" (related); Arjovsky & Bottou (2017); Roth, Lucchi, Nowozin & Hofmann,
"Stabilizing Training of Generative Adversarial Networks through Regularization" (NeurIPS
2017).

**Problem solved in GANs**: When the real and generated data distributions have disjoint support
(they live on different manifolds), the discriminator can separate them perfectly, killing the
gradient. Adding noise to both real and generated data makes their supports overlap, ensuring
the discriminator always has a non-trivial task.

**Core mechanism**: Add Gaussian noise to both real and generated samples before feeding them to
the discriminator: D(x + epsilon), D(G(z) + epsilon'), where epsilon ~ N(0, sigma^2 I).
Sigma is annealed from a high value to zero during training. The noise "blurs" the
distributions so they overlap, preventing perfect discrimination and ensuring continuous
gradients.

**Key insight for co-evolution**: When Constructors reach near-optimal configurations, the
difference between "can be improved" and "cannot be improved" is infinitesimal. Evaluation
noise from the Improver's stochastic code execution may swamp this signal. But we can also
*add* useful noise to smooth the evaluation landscape.

**Mapping to GigaEvo**:

1. **Noisy evaluation for Improvers**: When evaluating an Improver against a Constructor, add
   small random perturbations to the Constructor's output before passing it to the Improver:

   ```python
   # In Improver evaluation:
   noisy_points = constructor_output + np.random.normal(0, sigma, size=(11, 2))
   noisy_points = clip_to_triangle(noisy_points)
   improved_points = improver(noisy_points)
   ```

   With sigma > 0, even a near-optimal Constructor configuration has "nearby" versions that
   are easier to improve. The Improver always has a non-trivial task.

2. **Anneal sigma over generations**: Start with high sigma (easy -- even good configurations
   are perturbed enough to be improvable), decrease toward zero as both populations improve.
   This mirrors the instance noise annealing in GANs.

3. **Alternative: Evaluation with multiple noise seeds**: Evaluate each Improver-Constructor
   pair M times with different noise seeds. The Improver's fitness is the mean improvement
   across seeds. This smooths out evaluation noise and provides a more reliable gradient.

4. **Implementation**: Modify `evaluate.py` to accept a `noise_sigma` parameter. The watchdog
   or a scheduler decreases sigma over generations.

**Experiment idea**: "Annealed evaluation noise" -- start with sigma=0.05, anneal to 0 over
50 generations. Test whether this keeps Improvers in the productive learning zone.

---

### 4.3 Feature Matching

**Paper**: Salimans et al., "Improved Techniques for Training GANs" (NeurIPS 2016).

**Problem solved in GANs**: The generator optimizes a single scalar (D's output). Feature
matching replaces this with a richer signal: minimize ||E[f(x_real)] - E[f(G(z))]||^2 where
f(x) are intermediate feature activations from D. The generator sees *which features* of real
data it's failing to match, not just a binary real/fake verdict.

**Core mechanism**: Extract features from an intermediate layer of the discriminator. The
generator loss becomes the L2 distance between the expected features of real data and the
expected features of generated data. This provides a richer gradient signal because the
generator sees which aspects of the data distribution it needs to match (textures, edges,
color distributions) rather than just whether D classified it correctly.

**Key insight for co-evolution**: This is the foundational insight for your heilbron-v2
experiment. The "intermediate features" in the co-evolution setting are the opponent's
*strategies* (source code), not just the scalar outcome.

**Mapping to GigaEvo**:

This is already the core design of heilbron-v2 (Section 2.2 of the literature brief and the
full 01_design.md). The mapping is:

1. **Intermediate features = opponent source code**: The most informative "feature" of an
   Improver is its strategy (how it moves points, which heuristics it uses). This is directly
   readable from the source code.

2. **Feature matching loss = structured feedback in mutation prompt**: Instead of "Improver
   improved you by 0.003", the Constructor sees "Improver moved point 7 toward the centroid
   to reduce the worst triangle (3,7,10) from area 0.012 to area 0.015". The mutation LLM
   can then evolve a defense against this specific attack.

3. **Bidirectional feature matching**: Direction 1 (Improver code -> Constructor) is the
   standard feature matching analog. Direction 2 (Constructor code -> Improver) is the
   reverse: the Improver sees Constructor defense strategies and evolves targeted attacks.
   This is unique to the co-evolution setting (GANs only do D->G feature matching).

4. **What heilbron-v2 is testing**: K=1 vs K=3 opponent code blocks -- effectively the
   "minibatch size" for feature matching. K=3 provides a richer feature signal but with more
   noise and context cost.

**Status**: Already designed and approved for heilbron/adversarial-v2. No additional experiment
needed for this technique specifically.

---

### 4.4 Differentiable Augmentation (DiffAugment)

**Paper**: Zhao, Zhang, Liu, So & Ermon, "Differentiable Augmentation for Data-Efficient
GAN Training" (NeurIPS 2020).

**Problem solved in GANs**: GANs overfit with limited training data. The discriminator memorizes
real samples and provides a degenerate signal to the generator. Standard augmentation can't be
applied to GANs naively because augmentation applied only to real data biases the discriminator.

**Core mechanism**: Apply the SAME differentiable augmentations (color jitter, translation,
cutout) to BOTH real and generated images before passing to the discriminator. Because the
augmentations are differentiable, gradients flow through them to the generator. Crucially,
both real and fake are augmented identically, so the discriminator can't use the presence/
absence of augmentation as a signal. This dramatically improves data efficiency (competitive
results with 20% of the data).

**Key insight for co-evolution**: The Constructor archive is small (8 elites). The Improver
trains against the same 8 opponents repeatedly. This is the "limited data" problem --
Improvers overfit to the specific archive members rather than learning general improvement
strategies.

**Mapping to GigaEvo**:

1. **Augmented opponent evaluation**: When presenting a Constructor to an Improver for
   evaluation, randomly augment the Constructor's output while preserving its essential
   structure:

   ```python
   # Augmentation: small random perturbation of Constructor output
   augmented_points = constructor_output + np.random.uniform(-0.01, 0.01, size=(11, 2))
   augmented_points = clip_to_triangle(augmented_points)
   # Improver must improve the augmented version
   improved = improver(augmented_points)
   ```

   This is identical to instance noise (Section 4.2) but motivated differently: the goal is
   to prevent Improver overfitting to specific configurations, not to smooth the distribution.

2. **Symmetric augmentation**: Apply the same augmentation to both the Constructor's output
   AND the Improver's output before comparing:

   ```python
   # Both outputs get the SAME perturbation
   noise = np.random.normal(0, sigma, (11, 2))
   original_noisy = constructor_output + noise
   improved_noisy = improver_output + noise
   delta = min_area(improved_noisy) - min_area(original_noisy)
   ```

   This prevents the Improver from exploiting augmentation artifacts.

3. **Implementation**: Modify `evaluate.py` to apply random augmentations. The key is applying
   them symmetrically to both populations' outputs.

---

### 4.5 Adaptive Discriminator Augmentation (ADA)

**Paper**: Karras, Aittala, Hellsten, Laine, Lehtinen & Aila, "Training Generative Adversarial
Networks with Limited Data" (NeurIPS 2020).

**Problem solved in GANs**: Even with DiffAugment, choosing the right augmentation strength is
difficult. Too little = discriminator overfits. Too much = training destabilizes.

**Core mechanism**: Monitor the discriminator's overfitting heuristic r_t = E[sign(D(x_real))]
(the fraction of real images the discriminator classifies as "real" with high confidence). If
r_t is too high (discriminator overfits), increase augmentation probability p. If r_t is too
low (too much augmentation), decrease p. The augmentation probability p is adjusted every 4
minibatches using a simple PID controller.

**Key insight for co-evolution**: ADA dynamically adjusts difficulty based on the
discriminator's current performance. This is the same principle as GenEnv's alpha-curriculum
but with an automated feedback loop rather than a fixed schedule.

**Mapping to GigaEvo**:

1. **Adaptive opponent difficulty**: Monitor the Improver's success rate (the analog of the
   discriminator's "overfitting" metric). If success rate is too low (< 10%), the task is
   too hard -- increase evaluation noise sigma, select easier Constructors as opponents, or
   increase the Improver's generation budget. If success rate is too high (> 50%), the task
   is too easy -- decrease noise, select harder Constructors:

   ```python
   # ADA-style adaptive controller
   target_success_rate = 0.3  # 30% target
   current_success_rate = improver_successes / improver_evaluations  # rolling window

   if current_success_rate < target_success_rate - 0.05:
       sigma *= 1.1  # easier (more noise)
       # or: shift opponent selection toward easier Constructors
   elif current_success_rate > target_success_rate + 0.05:
       sigma *= 0.9  # harder (less noise)
       # or: shift opponent selection toward harder Constructors
   ```

2. **Adaptive K (number of opponents)**: If the Improver is struggling (low success rate),
   reduce the number of opponents it must improve (K=3 -> K=1). If thriving, increase K.

3. **Implementation**: Add an `AdaptiveDifficultyController` that monitors Improver success
   rate via Redis metrics and adjusts evaluation parameters each generation. The watchdog
   could implement this as a per-generation hook.

**Experiment idea**: "ADA-style adaptive difficulty" -- automatically adjust Improver task
difficulty to maintain 30% success rate. This is the most sophisticated approach to the
stagnation problem and subsumes many simpler techniques.

---

### 4.6 Lazy Regularization

**Paper**: Karras et al., "Analyzing and Improving the Image Quality of StyleGAN" (CVPR 2020,
StyleGAN2).

**Problem solved in GANs**: R1 regularization is expensive (requires computing gradients of
gradients). Applying it every step wastes compute without proportional benefit.

**Core mechanism**: Apply regularization (R1 gradient penalty, path length regularization) every
K steps instead of every step. StyleGAN2 uses K=16 for R1 and K=8 for path length reg. This
saves ~40% of training time with negligible quality loss, because the regularization effect
persists across multiple steps.

**Key insight for co-evolution**: Expensive evaluation procedures (evaluating against all
opponents, computing diverse fitness metrics) don't need to happen every generation.

**Mapping to GigaEvo**:

1. **Lazy opponent evaluation**: Full opponent evaluation (all K opponents, feedback extraction)
   is expensive. Run it every R=3 generations; in between, use cached opponent results from
   the last full evaluation:

   ```python
   if generation % R == 0:
       full_opponent_evaluation(program)
       cache_results(program)
   else:
       use_cached_results(program)
   ```

2. **Lazy feedback extraction**: Generating the structured feedback (opponent code blocks) is
   the most token-expensive part of the mutation prompt. Refresh it every R generations rather
   than every generation.

3. **Lazy archive analysis**: The "self-attention" archive analysis (Section 2.3) is expensive.
   Run it every R=5 generations and cache the result.

4. **Implementation**: Add a `lazy_evaluation_period` config parameter. The
   `OpponentFeedbackStage` caches its output for R generations before refreshing.

---

### 4.7 Top-K Training for G

**Paper**: Sinha, Zhao, Goyal, Raffel & Courville, "Top-k Training of GANs: Improving GAN
Training by Throwing Away Bad Samples" (NeurIPS 2020).

**Problem solved in GANs**: Not all generated samples are useful for training. Many are so bad
that the discriminator trivially rejects them, providing zero useful gradient. Worse, gradient
from these bad samples can destabilize training by pointing G in unhelpful directions.

**Core mechanism**: For each minibatch of M generated samples, compute D's score for each,
sort, and only backpropagate through the top-K (highest-scored) samples. The bottom (M-K)
samples are discarded. K starts at M (all samples used) and linearly decreases to M/2 during
training. This focuses the generator's gradient on its "best attempts", which are most
informative for learning.

**Key insight for co-evolution**: Many mutated programs are terrible -- they crash, produce
invalid outputs, or are vastly worse than their parents. These programs provide no useful
evolutionary signal. Worse, if they occupy mutation budget, they slow down evolution.

**Mapping to GigaEvo**:

1. **Top-K archive admission with dynamic K**: Instead of accepting any program that beats the
   current archive occupant, only accept programs in the top-K of all candidates generated this
   generation:

   ```python
   candidates = [program for program in generation if program.is_valid]
   candidates.sort(key=lambda p: p.fitness, reverse=True)
   top_k = candidates[:K]
   for program in top_k:
       try_add_to_archive(program)
   # Bottom candidates are discarded -- their fitness signal is too weak to be useful
   ```

2. **Selective mutation context**: When building mutation prompts, only include information from
   *successful* opponent interactions (where the Improver actually found an improvement). Failed
   interactions (Improver couldn't improve) are less informative and add noise to the prompt.

3. **Progressive K schedule for Improvers**: Start by accepting all valid Improver programs
   (K=100% -- build initial diversity). Gradually increase selectivity (K=50% by gen 50) as the
   archive matures. This mirrors the linear K schedule in the original paper.

4. **Implementation**: Modify the archive acceptance logic to batch-evaluate candidates per
   generation and select only top-K. For the steady-state engine, buffer candidates for N
   evaluations, then batch-select.

---

## Part V: Game Theory and Equilibrium

### 5.1 Nash Equilibrium and Local Convergence Theory

**Paper**: Multiple foundational works. Key papers: Goodfellow et al. (2014, original GAN);
Nagarajan & Kolter, "Gradient descent GAN optimization is locally stable" (NeurIPS 2017);
Mescheder et al. (ICML 2018, "Which Training Methods...").

**Problem solved in GANs**: Understanding when and why GAN training converges. A GAN at
equilibrium is a Nash equilibrium of the two-player game: G cannot improve given D, and D
cannot improve given G. The theoretical question: does gradient descent converge to such an
equilibrium?

**Core mechanism**: At a local Nash equilibrium (theta_G*, theta_D*), the Jacobian of the
gradient vector field has all eigenvalues with positive real parts (the equilibrium is stable).
Nagarajan & Kolter (2017) showed that GAN training with simultaneous gradient updates IS
locally stable under mild conditions. Mescheder et al. (2018) showed that many popular
variants (vanilla GAN, WGAN without penalty, etc.) are NOT locally convergent, and identified
R1 regularization as a sufficient condition for local convergence.

**Key insight for co-evolution**: The heilbron-prover result shows convergence in one dimension
(Constructor actual_fitness approaches 0.0365) but NOT as a Nash equilibrium (Improver
stagnates, not at best response). This is a *Stackelberg equilibrium* (leader-follower), not
a Nash equilibrium. The Constructor leads; the Improver follows and gives up.

**Mapping to GigaEvo**:

1. **Diagnose the equilibrium type**: The current outcome is a "mediocre stable state"
   (Ficici & Pollack, 2000) -- the system converges to a point where neither population has
   sufficient signal to improve, but it's not a true Nash equilibrium. To reach Nash, both
   populations must be at best response.

2. **Sufficient conditions for co-evolutionary convergence**:
   - Both populations must have non-trivial fitness gradients at the equilibrium
   - The fitness landscape must be locally smooth (no discontinuities in the improvement signal)
   - Update rates must be compatible (TTUR guarantees this for two-timescale updates)

3. **Practical diagnosis**: Measure the "gradient" analogs at the current fixed point:
   - **Constructor gradient**: Does increasing actual_fitness by epsilon increase resistance?
     If resistance is already 100%, the answer is "doesn't matter" -- the Constructor's
     selection signal is saturated.
   - **Improver gradient**: Does modifying the Improver's strategy by epsilon increase its
     success rate? If success rate is 0%, the answer is "no useful direction found" -- the
     Improver's search signal is dead.

4. **Action**: The convergence theory says: ensure both populations face smooth, non-saturating
   fitness functions. This points toward: continuous Improver fitness (Section 1.1), soft
   resistance (Section 4.1), and difficulty calibration (Section 5.4).

---

### 5.2 Consensus Optimization

**Paper**: Mescheder, Nowozin & Geiger, "The Numerics of GANs" (NeurIPS 2017).

**Problem solved in GANs**: Simultaneous gradient descent on the min-max GAN objective creates
rotational dynamics -- the vector field has a large rotational component that causes cycling
rather than convergence. Pure gradient descent spirals outward in these rotational fields.

**Core mechanism**: Add a "consensus" regularization term that penalizes the magnitude of the
gradient: L_consensus = gamma * ||nabla_theta L||^2. This shrinks the rotational component
of the vector field, converting rotational dynamics into convergent ones. Intuitively: if both
players are experiencing large gradients (rapidly changing strategies), penalize this instability.

**Key insight for co-evolution**: Co-evolutionary cycling is the exact analog of GAN rotational
dynamics. Strategy A beats strategy B, B evolves to beat A, A evolves to beat the new B, etc.
-- the system rotates through strategies without converging.

**Mapping to GigaEvo**:

1. **Archive churn penalty**: Monitor the rate of archive changes per generation. If the churn
   rate is high (many cells changing occupants frequently), this indicates rotational dynamics.
   Add a penalty that makes acceptance harder when churn is high:

   ```python
   churn_rate = archive_changes_last_5_gens / archive_size
   acceptance_threshold = base_threshold + gamma * churn_rate
   # Higher churn -> harder to enter archive -> dampens rotation
   ```

2. **Momentum-based opponent selection**: Instead of always evaluating against the latest
   archive, select opponents from a mix of current and recent archives. This dampens the
   rotational component by averaging over the recent trajectory.

3. **Detect and break cycles**: Monitor fitness trajectories for oscillation patterns (A rises
   while B falls, then B rises while A falls). If detected, intervene by freezing one
   population for N generations while the other catches up. This is a discrete version of
   consensus optimization.

4. **Implementation**: Add a cycle detection heuristic to the watchdog. The watchdog already
   monitors fitness trajectories -- add a correlation analysis between Constructor and
   Improver fitness changes. Negative correlation = rotational dynamics = trigger intervention.

---

### 5.3 Competitive Gradient Descent (CGD) and Symplectic Gradient Adjustment (SGA)

**Papers**: Schafer & Anandkumar, "Competitive Gradient Descent" (NeurIPS 2019); Balduzzi,
Racaniere, Martens, Foerster, Tuyls & Graepel, "The Mechanics of n-Player Differentiable
Games" (ICML 2018).

**Problem solved in GANs**: Standard simultaneous gradient descent (SGD) ignores the game-
theoretic structure of the GAN objective. Each player's gradient is computed assuming the
other player is fixed, but both update simultaneously. This leads to the rotational dynamics
described in Section 5.2.

**Core mechanism (CGD)**: Compute a "competitive gradient" that accounts for the opponent's
anticipated response. Instead of the naive gradient, each player takes a step that is optimal
given the other player's expected counter-step. Mathematically: solve a linear system involving
the cross-player Hessian (the second-order interaction between G and D parameters).

**Core mechanism (SGA)**: Decompose the game's Jacobian into symmetric (potential) and
antisymmetric (Hamiltonian/rotational) components. Standard gradient descent follows both.
SGA adds a correction term that cancels the Hamiltonian (rotational) component, leaving only
the potential (convergent) component. This eliminates cycling.

**Key insight for co-evolution**: We don't have gradients, so we can't directly compute
cross-player Hessians or Jacobian decompositions. But the principles translate to selection
pressure design.

**Mapping to GigaEvo**:

1. **Opponent-aware fitness**: Instead of evaluating Constructor fitness independently, include
   a term that anticipates the Improver's likely next move. Show the Constructor the
   Improver's *trajectory* (Section 3.2, unrolled GANs) and reward Constructors that are
   robust to the *predicted* next Improver strategy, not just the current one.

2. **Anti-cycling selection pressure**: When selecting parents for mutation, prefer programs
   that DON'T simply reverse the last opponent's improvement. If the Improver moved point 7
   right, and the Constructor responds by moving point 7 left, this is a rotational cycle.
   Detect and penalize such "tit-for-tat" patterns:

   ```python
   # In mutation selection:
   if program_undoes_last_opponent_move(candidate, opponent_history):
       candidate.selection_weight *= 0.5  # penalize cycling
   ```

3. **Cooperative component injection**: In the fitness function, add a small cooperative term
   that rewards *both* populations when overall solution quality (actual_fitness) improves:

   ```python
   # Constructor fitness:
   constructor_fitness = 0.5*quality + 0.4*resistance + 0.1*global_progress_bonus
   # Improver fitness:
   improver_fitness = 0.5*improvement_rate + 0.4*adversarial + 0.1*global_progress_bonus

   # global_progress_bonus = actual_fitness improvement over last N gens
   ```

   This injects a small "potential" (convergent) component into what would otherwise be a
   pure "Hamiltonian" (rotational) game.

---

### 5.4 GenEnv / Alpha-Curriculum (Difficulty Calibration)

**Paper**: GenEnv (2025); conceptually related to the zone of proximal development (Vygotsky),
curriculum learning (Bengio et al. 2009), and self-paced learning.

**Problem solved**: Adversarial co-evolution where the "discriminator" (Improver) faces tasks
that are either too easy (trivial improvements on weak Constructors) or too hard (impossible
improvements on strong Constructors). Neither extreme provides useful learning signal.

**Core mechanism**: The environment generator receives reward:
R_env = exp(-beta * (success_rate - alpha)^2)
where alpha is the target success rate (typically 0.5). This peaks when the agent succeeds
exactly alpha fraction of the time. Mathematical justification: intermediate difficulty
(50% success) maximizes the expected squared gradient norm -- the strongest learning signal.

**Key insight for co-evolution**: The Improver stagnation in heilbron-prover occurs because
success rate drops to 0% -- the Constructors became too hard. GenEnv's alpha-curriculum would
automatically calibrate difficulty to keep Improvers at ~30-50% success rate.

**Mapping to GigaEvo**:

1. **Difficulty-calibrated opponent selection**: Maintain a difficulty-ranked archive of
   Constructors (ranked by actual_fitness). For Improver evaluation, select opponents from the
   zone where the Improver's historical success rate is ~30-50%:

   ```python
   # Compute Improver success rate vs each Constructor difficulty band
   bands = [constructors_with_fitness_in_range(low, high) for low, high in difficulty_bands]
   success_rates = [improver_success_rate_vs_band(band) for band in bands]

   # Select opponents from bands where success rate is near target (0.3-0.5)
   target_bands = [b for b, sr in zip(bands, success_rates) if 0.2 < sr < 0.6]
   opponents = sample_from(target_bands, k=n_opponents)
   ```

2. **Difficulty as behavioral descriptor**: Use the Constructor's difficulty level (estimated
   from Improver success rates) as a behavioral descriptor dimension in the MAP-Elites archive.
   This ensures the archive maintains Constructors at diverse difficulty levels, not just the
   hardest ones.

3. **Adaptive alpha**: Start with alpha=0.5 (easy -- Improvers should succeed half the time).
   As Improvers strengthen, gradually decrease alpha to 0.3, then 0.2. This implements
   curriculum learning within the alpha-curriculum framework.

4. **Implementation**: Add a `DifficultyCalibrator` that:
   a. Tracks Improver success rate per Constructor difficulty band
   b. Selects opponents from the band closest to alpha
   c. Reports calibration statistics to Redis metrics for monitoring

**Experiment idea**: "Alpha-curriculum opponent selection" -- replace top-K opponent selection
with difficulty-calibrated selection targeting 30-50% Improver success rate. Highest-priority
technique for directly addressing stagnation.

---

## Part VI: Self-Play and Debate

### 6.1 AI Safety Debate

**Paper**: Irving, Christiano & Amodei, "AI Safety via Debate" (2018).

**Problem solved**: Aligning AI systems without direct access to ground truth. Two AI agents
debate over the correct answer, with a human judge determining the winner. The key insight:
even if the human can't solve the problem directly, they can judge which debater made better
arguments.

**Core mechanism**: Two agents take opposing positions and argue over multiple rounds. Each
agent can see the other's arguments and must respond. The human judge evaluates the final
transcript. Theoretical result: in the limit, honest argumentation is the dominant strategy
(lying debaters can always be caught by honest opponents).

**Key insight for co-evolution**: The Constructor-Improver dynamic IS a debate. The Constructor
argues "this configuration is optimal" (by constructing it). The Improver argues "this
configuration is not optimal" (by finding improvements). The "judge" is the evaluation function.

**Mapping to GigaEvo**:

1. **Multi-round improvement attempts**: Instead of one-shot evaluation (Improver gets one
   chance to improve), allow K rounds of "debate":
   - Round 1: Improver suggests improvement. Constructor responds with a counter-configuration.
   - Round 2: Improver attacks the counter. Constructor responds again.
   - Final: Judge (evaluation function) determines if any round's improvement held.

   This gives the Improver multiple attempts with feedback, increasing the probability of
   finding an improvement and providing richer signal.

2. **Argument legibility**: Require Improvers to produce not just an improved configuration but
   a *justification* (which triangle was weakest, why the move helps). This justification flows
   into the Constructor's mutation prompt as "structured feedback" (already planned in v2).

3. **Implementation**: Modify the evaluation pipeline to support multi-round evaluation. Each
   round adds latency, so limit to K=2-3 rounds. The Improver's code receives the previous
   round's result as additional context.

---

### 6.2 Self-Play in AlphaGo/AlphaZero

**Papers**: Silver et al., "Mastering the Game of Go with Deep Neural Networks and Tree Search"
(Nature, 2016); Silver et al., "Mastering the Game of Go without Human Knowledge" (Nature,
2017); Silver et al., "A General Reinforcement Learning Algorithm that Masters Chess, Shogi,
and Go through Self-Play" (Science, 2018).

**Problem solved**: Learning to play games at superhuman level through self-play, without human
expert data. The key challenge: preventing catastrophic forgetting (the agent forgets how to
handle older strategies as it learns new ones).

**Core mechanism**: (1) MCTS (Monte Carlo Tree Search) guided by a neural network for
evaluation and move probabilities. (2) Self-play against the *current best* version of the
agent. (3) Evaluation against a fixed pool of previous best versions -- a new version must win
55% of games against the current champion to replace it. (4) The training data is generated by
self-play and stored in a replay buffer spanning many games.

**Anti-forgetting mechanisms**:
- **Champion checkpoint**: New policies must beat the current champion. This prevents regression.
- **Replay buffer**: Training data includes games from many previous versions, not just the
  latest. The agent continually trains on a mix of old and new experiences.
- **ELO rating**: Track progress via an ELO rating system that evaluates against a fixed pool.

**Key insight for co-evolution**: AlphaZero prevents forgetting through (1) the champion
checkpoint mechanism and (2) training on diverse historical data. Both are applicable to
evolutionary co-evolution.

**Mapping to GigaEvo**:

1. **Champion checkpoint mechanism**: The MAP-Elites archive should only accept programs that
   beat the current best by a margin (already partially implemented via `significant_change`).
   Extend this: new Constructors must also demonstrate resistance against the top-5 historical
   Improvers, not just the current archive.

2. **Replay buffer for opponent evaluation**: Maintain a buffer of the last N=50 Improver
   programs (not just the current archive's 8 elites). Evaluate Constructors against a random
   sample from this buffer. This ensures Constructors remain robust to diverse attack
   strategies, even old ones that left the archive.

3. **ELO-like rating system**: Assign each Constructor and Improver an ELO rating based on
   pairwise evaluation results. Use ELO instead of raw fitness for selection. ELO naturally
   accounts for opponent strength -- beating a strong opponent is worth more than beating a
   weak one:

   ```python
   # After Improver evaluates Constructor:
   if improved:
       improver_elo += K * (1 - expected_score(improver_elo, constructor_elo))
       constructor_elo += K * (0 - expected_score(constructor_elo, improver_elo))
   else:
       improver_elo += K * (0 - expected_score(improver_elo, constructor_elo))
       constructor_elo += K * (1 - expected_score(constructor_elo, improver_elo))
   ```

4. **Implementation**: Add ELO tracking to `metrics.yaml` and store ratings in Redis. Modify
   archive selection to use ELO instead of raw fitness. This is a significant change but
   provides a more principled selection signal.

---

### 6.3 Population-Based Training (PBT)

**Paper**: Jaderberg, Dalibard, Osindero et al., "Population Based Training of Neural Networks"
(2017); Jaderberg et al., "Human-Level Performance in First-Person Multiplayer Games with
Population-Based Deep Reinforcement Learning" (Science, 2019).

**Problem solved**: Hyperparameter optimization during training. Standard approaches either
use fixed hyperparameters (suboptimal) or expensive grid/random search (parallel). PBT
evolves hyperparameters alongside model training.

**Core mechanism**: Maintain a population of agents, each with its own hyperparameters. Every
N steps: (1) Evaluate all agents. (2) Bottom 20% of agents copy weights AND hyperparameters
from a randomly selected top-20% agent (exploit). (3) Copied hyperparameters are randomly
perturbed (explore). This simultaneously trains models and tunes hyperparameters.

**Key insight for co-evolution**: GigaEvo's MAP-Elites archive evolves programs but NOT the
evolutionary hyperparameters (mutation rate, prompt strategy, evaluation parameters). These
could be co-evolved using PBT.

**Mapping to GigaEvo**:

1. **Evolve mutation hyperparameters alongside programs**: Each program in the archive carries
   metadata about the mutation settings that produced it (temperature, prompt variant,
   parent selection strategy). When a program succeeds, its mutation settings are "inherited"
   (copied) to future mutations. Failed programs' settings are perturbed:

   ```python
   program.mutation_metadata = {
       "temperature": 0.7,  # LLM temperature
       "prompt_variant": "strategic",  # which prompt template
       "parent_selection": "tournament",  # selection method
   }
   # When mutating a child from this parent:
   child.mutation_metadata = perturb(program.mutation_metadata)
   ```

2. **PBT for evaluation parameters**: The evaluation parameters (n_opponents, noise_sigma,
   difficulty_target_alpha) could be evolved using PBT. If a setting produces more improvements,
   it's propagated; if it stagnates, it's perturbed.

3. **Implementation**: Store mutation metadata in `program.metadata`. The
   `MutationContextBuilderStage` reads the parent's metadata and uses it to configure the
   LLM call. Modify `MutationAgent` to accept variable temperature and prompt variants.

---

### 6.4 Fictitious Play and Double Oracle

**Papers**: Brown (1951, "Iterative Solution of Games by Fictitious Play"); McMahan, Gordon &
Blum, "Planning in the Presence of Cost Functions Controlled by an Adversary" (ICML 2003,
double oracle).

**Problem solved**: Finding equilibria in two-player games when the strategy space is too large
to enumerate. Fictitious play: each player best-responds to the opponent's historical average
strategy. Double oracle: iteratively expand the strategy set -- start with a small set, find
the equilibrium, add the best response to the current equilibrium, repeat.

**Core mechanism (Fictitious Play)**: At each round, player i observes the empirical frequency
of player j's past strategies. Player i then plays the best response to this empirical
distribution. Under certain conditions (zero-sum games, identical interests games), this
converges to a Nash equilibrium.

**Core mechanism (Double Oracle)**: Start with a small "restricted" game. Solve for the Nash
equilibrium of the restricted game. Each player then computes a best response to the opponent's
equilibrium strategy over the FULL strategy space. If the best response isn't already in the
restricted game, add it. Repeat until no new strategies are found (= reached the true Nash
equilibrium of the full game).

**Key insight for co-evolution**: MAP-Elites archives are a form of "restricted strategy set."
The archive holds a finite set of strategies per population. The double oracle framework
suggests: find the equilibrium of the archive game, then specifically search for strategies
that break this equilibrium.

**Mapping to GigaEvo**:

1. **Payoff matrix equilibrium analysis (from ASRO)**: Compute the full N_A x N_B payoff matrix
   between all Constructor and Improver archive members. Solve for the mixed Nash equilibrium.
   The equilibrium probabilities reveal which Constructors are vulnerable (high probability in
   the Improver's optimal mix) and which Improvers are useless (zero probability):

   ```python
   # Compute payoff matrix
   payoff = np.zeros((len(constructors), len(improvers)))
   for i, c in enumerate(constructors):
       for j, imp in enumerate(improvers):
           payoff[i, j] = evaluate(imp, c)  # delta_fitness

   # Solve for mixed Nash
   constructor_mix, improver_mix = solve_nash(payoff)

   # Constructors with high mix weight are most vulnerable
   # Improvers with zero mix weight are useless
   ```

2. **Directed mutation from equilibrium analysis**: Use the Nash analysis to direct mutation:
   - "This Constructor has 40% weight in the Improver's optimal mix -- it's the most
     vulnerable. Mutate it to reduce vulnerability."
   - "This Improver has 0% weight -- it's useless against the current Constructor archive.
     Mutate it to specifically target the highest-weight Constructors."

3. **Double oracle for archive expansion**: After computing the Nash equilibrium of the current
   archives, specifically search for a strategy that is a best response to the opponent's
   equilibrium mix. This targeted search is more efficient than random mutation because it
   focuses on strategies that would be immediately useful.

4. **Implementation**: Compute the payoff matrix every N generations using cached evaluation
   results. Use scipy's linear programming to solve the zero-sum game. Inject the equilibrium
   analysis into mutation prompts as context.

**Experiment idea**: "PSRO-guided mutation" -- compute payoff matrix between archives, solve
for Nash equilibrium, and use the analysis to guide mutation prompts. Test whether targeted
mutation is more efficient than random.

---

### 6.5 PSRO (Policy-Space Response Oracles)

**Paper**: Lanctot, Zambaldi, Gruslys et al., "A Unified Game-Theoretic Approach to Multiagent
Reinforcement Learning" (NeurIPS 2017).

**Problem solved**: Scaling game-theoretic solution concepts to large strategy spaces. Pure
fictitious play and double oracle require enumerating strategies, which is intractable in
continuous or large discrete spaces. PSRO uses RL to compute approximate best responses.

**Core mechanism**: Maintain a population of policies (strategies) for each player. Compute the
meta-game Nash equilibrium over this population. Use RL to train a new policy that is the best
response to the opponent's Nash mixture. Add the new policy to the population. Repeat. This
iteratively expands the strategy set in the most useful direction.

**Key PSRO variants**:
- **Rectified PSRO (2019)**: Only train against opponents that have positive probability in
  the Nash mixture (ignore irrelevant opponents).
- **Pipeline PSRO (2020)**: Parallelize the best-response computation.
- **JPSRO (2021)**: Joint best response -- both players compute best responses simultaneously.

**Key insight for co-evolution**: This is the most principled framework for adversarial
co-evolution and directly addresses the stagnation problem. Instead of evolving against the
entire opponent archive (which may contain many useless opponents), evolve against the Nash
mixture (which weights each opponent by its strategic importance).

**Mapping to GigaEvo**:

1. **PSRO as the meta-algorithm**: Replace the current "evaluate against top-K opponents" with
   PSRO:
   - Step 1: Compute payoff matrix between Constructor and Improver archives
   - Step 2: Solve for Nash equilibrium mixture
   - Step 3: Evolve new Constructor as best response to Nash Improver mixture
   - Step 4: Evolve new Improver as best response to Nash Constructor mixture
   - Step 5: Add new strategies to archives, goto Step 1

2. **Nash-weighted opponent selection**: When selecting opponents for evaluation, weight by
   Nash mixture probabilities instead of fitness:

   ```python
   nash_weights = solve_nash(payoff_matrix)
   opponents = np.random.choice(opponent_archive, size=K, p=nash_weights)
   ```

3. **Rectified PSRO for efficiency**: Only evolve against opponents with positive Nash weight.
   If 3 out of 8 Improvers have zero Nash weight (they're redundant), don't waste evaluation
   budget on them.

4. **Implementation**: This requires: (a) periodic payoff matrix computation (every N gens),
   (b) Nash equilibrium solver (scipy linear programming for zero-sum games), (c) modified
   opponent selection in `FetchOpponentResultsStage` to use Nash weights. The payoff matrix
   can be computed from cached evaluation results in Redis.

**Experiment idea**: "PSRO-guided co-evolution" -- implement Nash-weighted opponent selection
within the existing MAP-Elites framework. This is the most theoretically principled approach
and subsumes many ad hoc techniques.

---

## Part VII: Recent SOTA (2023-2025)

### 7.1 GigaGAN

**Paper**: Yu, Sohn, Kim & Shin, "Scaling up GANs for Text-to-Image Synthesis" (CVPR 2023).

**Problem solved**: GANs had fallen behind diffusion models for text-to-image generation.
GigaGAN scaled GANs to 1B parameters and competitive quality with diffusion models.

**Key innovations**:
- **Sample-adaptive kernel selection**: Convolution kernels are dynamically computed from the
  input text, allowing different text prompts to activate different filter banks.
- **Interleaved attention**: Self-attention and cross-attention layers interleaved with
  convolutions at multiple scales.
- **Multi-scale training**: Generator and discriminator operate at multiple scales
  simultaneously (MSG-GAN style).
- **Matching-aware loss**: Additional loss term that penalizes misalignment between text and
  generated image features.

**Mapping to GigaEvo**:

1. **Input-adaptive mutation**: Different Constructor programs should receive different mutation
   strategies based on their current approach. A program using simulated annealing should
   receive mutation prompts tuned for annealing (adjust temperature, modify cooling schedule),
   while a geometric construction program should receive geometry-specific prompts (change arc
   radii, adjust vertex positions). This is the "sample-adaptive kernel" analog:

   ```python
   # In MutationContextBuilderStage:
   strategy = classify_strategy(parent_program)  # LLM call or heuristic
   prompt_variant = strategy_to_prompt_variant(strategy)
   mutation_prompt = load_prompt(prompt_variant)
   ```

2. **Matching-aware evaluation**: Penalize programs whose code complexity doesn't match their
   output quality. A 500-line program that produces min_area=0.020 is less efficient than a
   50-line program with the same output. Efficiency pressure encourages cleaner, more robust
   solutions.

---

### 7.2 StyleGAN-T and Projected GANs

**Papers**: Sauer, Schwarz & Geiger, "StyleGAN-T: Unlocking the Power of GANs for Fast
Large-Scale Text-to-Image Synthesis" (ICML 2023); Sauer, Schwarz, Geiger & Geiger, "Projected
GANs Converge Faster" (NeurIPS 2021).

**Problem solved**: GANs train slowly because the discriminator must learn useful features from
scratch. Projected GANs use a pre-trained feature extractor (CLIP, EfficientNet) to project
both real and generated images into a feature space, then discriminate in that space. The
discriminator starts with useful features rather than learning from scratch.

**Core mechanism**: Replace the learned discriminator features with projections from a
pre-trained, frozen feature network: D(F(x_real), F(G(z))), where F is a pre-trained feature
extractor. The discriminator only learns the final classification head on top of fixed features.
This provides a strong, stable feature representation from the start.

**Key insight for co-evolution**: The Improver's "feature extraction" -- understanding the
Constructor's strategy from its source code -- is performed by the LLM during mutation.
A pre-trained, powerful LLM provides a strong "feature extractor" from the start. But the
features it extracts could be improved with domain-specific context.

**Mapping to GigaEvo**:

1. **Pre-trained analysis as "projection"**: Before showing opponent code to the mutation LLM,
   pass it through a separate analysis step that extracts high-level features:

   ```
   Step 1: Analyze opponent code -> "Uses simulated annealing, targets worst triangle,
            applies local perturbation of ~0.01 magnitude"
   Step 2: Inject this analysis (not raw code) into mutation prompt
   ```

   This is the "parsed critique" option mentioned in heilbron-v2's design (Section 2, design
   choice). It's the Projected GAN analog: use a pre-trained LLM to extract features, then
   evolve based on those features rather than raw code.

2. **Frozen vs adaptive analysis**: In Projected GANs, the feature extractor is frozen. The
   analog: keep the analysis prompt fixed (frozen features) rather than evolving it alongside
   the programs. This provides stability.

3. **Implementation**: Add an `OpponentAnalysisStage` before `OpponentFeedbackStage`. This
   stage uses the LLM to analyze opponent code and produce a structured summary. The summary
   (not raw code) is injected into the mutation prompt. Trade-off: one extra LLM call per
   mutation, but potentially higher signal-to-noise ratio.

---

### 7.3 Lessons from Diffusion Models That Apply to Adversarial Training

**Key papers**: Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models" (NeurIPS 2020);
Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021); Karras,
Aittala, Aila & Laine, "Elucidating the Design Space of Diffusion-Based Generative Models"
(NeurIPS 2022).

**Why diffusion models are relevant**: Diffusion models displaced GANs as SOTA for image
generation by solving the exact problems GANs struggled with: mode coverage, training
stability, and diversity. Understanding *how* they succeeded reveals what GANs were missing.

**Key lessons for co-evolution**:

1. **Noise schedule = difficulty schedule**: Diffusion models train on a spectrum of noise
   levels, from nearly clean (easy denoising) to pure noise (hard denoising). The model sees
   the FULL difficulty spectrum every training batch. Analog: evaluate Constructors/Improvers
   across the full difficulty spectrum every generation, not just at the hardest level.
   **This is the GenEnv alpha-curriculum applied comprehensively.**

2. **Progressive denoising = incremental improvement**: Diffusion models generate images by
   iterating through many small denoising steps, not one giant leap. Analog: require Improvers
   to make *small, incremental improvements* rather than one-shot rewrites. Each improvement
   step provides feedback for the next.

3. **Score matching = gradient estimation**: Diffusion models learn the *score function*
   (gradient of log probability) at each noise level. The analog for co-evolution: learn not
   just "is this configuration good?" but "in which direction should this configuration be
   changed to become better?" This is exactly what structured feedback (opponent code) provides
   -- the direction of improvement.

4. **Classifier-free guidance = self-play signal amplification**: CFG (Ho & Salimans, 2021)
   amplifies the conditional signal by subtracting the unconditional signal:
   score_guided = score_unconditional + w * (score_conditional - score_unconditional).
   Analog: compare the Constructor's performance against random opponents (unconditional)
   vs. targeted opponents (conditional). The *difference* is the adversarial signal:

   ```python
   adversarial_signal = fitness_vs_targeted_improvers - fitness_vs_random_improvers
   # This measures how much the targeted Improver exploits Constructor-specific weaknesses
   ```

---

## Part VIII: Integrated Experiment Recommendations

Based on the full survey, here are the 5 highest-priority experiments ranked by expected
information gain and feasibility within the GigaEvo framework.

### Experiment 1: heilbron/adversarial-v2 (ALREADY DESIGNED)

**Techniques used**: Feature matching (4.3), bidirectional feedback, minibatch discrimination
(K=1 vs K=3).

**Status**: Fully designed, reviewed, and ready for implementation. This is the immediate
next experiment.

**What it tests**: Whether raw opponent code as structured feedback improves Constructor
actual_fitness and reduces Improver stagnation.

**Expected outcome**: Based on GAN theory, this should help if the LLM can parse opponent
strategies from code. If NULL, the follow-up is parsed critique (Projected GAN analog, 7.2).

---

### Experiment 2: TTUR + Alpha-Curriculum (Asymmetric Compute + Difficulty Calibration)

**Techniques used**: TTUR (1.4), GenEnv alpha-curriculum (5.4), ADA-style adaptive difficulty
(4.5).

**Design**: 2x2 factorial:
- IV1: Improver compute ratio (1:1 vs 3:1 Improver:Constructor evaluations)
- IV2: Opponent selection (top-K vs difficulty-calibrated targeting 30% success rate)

**Why this combination**: TTUR and alpha-curriculum address the same root cause (Improver
stagnation) through complementary mechanisms. TTUR gives Improvers more attempts; alpha-
curriculum gives them more achievable targets. The interaction is important: more attempts on
impossible targets (TTUR alone) may not help; achievable targets without enough attempts
(alpha-curriculum alone) may not help either.

**Implementation requirements**:
- Asymmetric `max_mutations_per_generation` per run (existing config)
- Difficulty-sorted Constructor archive with calibrated sampling (new `DifficultyCalibrator`
  stage)
- Improver success rate tracking (new Redis metric)

**Primary hypothesis**: The combination of 3:1 compute ratio AND difficulty-calibrated
opponent selection yields > 10% Improver acceptance rate after gen 20 (vs 0% in baseline).

---

### Experiment 3: Continuous Improver Fitness (Wasserstein Analog)

**Techniques used**: Wasserstein distance (1.1), label smoothing (4.1), instance noise (4.2).

**Design**: 2x1 (treatment vs control within existing adversarial framework):
- Treatment: Improver fitness = continuous delta-based score with soft resistance
- Control: current binary improvement fitness

**Why**: The binary fitness signal is the most fundamental problem. All other techniques add
complexity, but if the base fitness signal is discontinuous, no amount of prompt engineering
or compute allocation will fix the gradient death problem.

**Implementation requirements**:
- Modify `evaluate.py` and `validate.py` in Pop B to return continuous fitness
- Modify `metrics.yaml` to track continuous Improver fitness
- Soft resistance computation in Pop A evaluation

**Primary hypothesis**: Continuous fitness provides a non-zero gradient to Improvers even at
high Constructor quality, preventing acceptance rate collapse.

---

### Experiment 4: PSRO-Guided Opponent Selection (Game-Theoretic)

**Techniques used**: PSRO (6.5), payoff matrix (6.4), Nash equilibrium (5.1), double oracle
(6.4).

**Design**: 2x1:
- Treatment: Nash-weighted opponent selection from periodically computed payoff matrix
- Control: uniform random opponent selection from archive

**Why**: This is the most theoretically principled approach. It replaces ad hoc opponent
selection heuristics with a game-theoretically optimal strategy. If it works, it validates
PSRO as the meta-algorithm for adversarial co-evolution with MAP-Elites.

**Implementation requirements**:
- Payoff matrix computation (periodic, every N gens) from cached evaluations
- Nash equilibrium solver (scipy.optimize.linprog for zero-sum games)
- Modified `FetchOpponentResultsStage` to accept sampling weights
- New Redis key for payoff matrix storage

**Primary hypothesis**: Nash-weighted selection focuses evolutionary pressure on strategically
important opponents, improving convergence to Nash equilibrium.

---

### Experiment 5: Progressive Difficulty with EMA Archive (Combined Stabilization)

**Techniques used**: Progressive growing (2.1), EMA (1.6), historical averaging (3.3),
consensus optimization (5.2).

**Design**: Exploratory (no control, 4 runs with different schedules):
- All runs: EMA-weighted hall-of-fame for opponent selection + progressive difficulty schedule
- Pair 1: Fast progression (n=7 -> n=11 over 50 gens)
- Pair 2: Slow progression (n=7 -> n=11 over 100 gens)

**Why**: Combines training stabilization (EMA archive) with curriculum learning (progressive
difficulty). Tests whether starting at an easier problem and gradually increasing difficulty
prevents the stagnation that occurs when both populations face the full difficulty from gen 1.

**Implementation requirements**:
- Parameterized problem difficulty (n_points as a config parameter)
- EMA-weighted archive for opponent selection
- Difficulty scheduler (generation-based or milestone-based)
- Programs must generalize across problem sizes (or be re-evolved at each transition)

**Primary hypothesis**: Progressive difficulty prevents early stagnation by keeping both
populations in the zone of productive learning throughout training.

---

## Summary Table: Technique to Experiment Priority

| Technique | Primary Mechanism | Maps to Experiment | Priority |
|---|---|---|---|
| Feature matching (4.3) | Opponent code as gradient | Exp 1 (v2, designed) | IMMEDIATE |
| TTUR (1.4) | Asymmetric compute | Exp 2 | HIGH |
| Alpha-curriculum (5.4) | Difficulty calibration | Exp 2 | HIGH |
| Wasserstein / continuous fitness (1.1) | Smooth gradient | Exp 3 | HIGH |
| Label smoothing (4.1) | Soft resistance | Exp 3 | HIGH |
| PSRO (6.5) | Nash-weighted selection | Exp 4 | MEDIUM-HIGH |
| Payoff matrix (6.4) | Equilibrium analysis | Exp 4 | MEDIUM-HIGH |
| Progressive growing (2.1) | Curriculum learning | Exp 5 | MEDIUM |
| EMA archive (1.6) | Training stabilization | Exp 5 | MEDIUM |
| ADA (4.5) | Adaptive difficulty | Exp 2 (built-in) | MEDIUM |
| Instance noise (4.2) | Smoothed evaluation | Exp 3 or standalone | MEDIUM |
| Minibatch discrimination (3.1) | Diverse opponents | Any experiment | LOW (MAP-Elites already provides) |
| Spectral norm / virulence cap (1.3) | Bounded Improver power | Standalone or Exp 2 add-on | LOW |
| Unrolled GANs (3.2) | Lookahead | Too expensive for now | LOW |
| Top-K training (4.7) | Selective mutation | Quality-of-life improvement | LOW |
| Lazy regularization (4.6) | Efficient evaluation | Efficiency optimization | LOW |
| PBT (6.3) | Hyperparameter evolution | Future work | LOW |
| ELO rating (6.2) | Principled selection | Exp 4 alternative | LOW |
| Style mixing (2.2) | Strategy/implementation separation | Prompt engineering | LOW |
| Self-attention / archive analysis (2.3) | Global awareness | Prompt engineering | LOW |

---

## Key Insight: The Stagnation Problem Has Multiple Causes

The Improver stagnation observed in heilbron-prover likely results from the intersection of
several problems, not a single root cause:

1. **Signal death** (Wasserstein, label smoothing): Binary fitness -> zero gradient at high
   Constructor quality
2. **Task mismatch** (TTUR, alpha-curriculum): Improver task is harder but gets equal compute
   and faces maximum-difficulty opponents
3. **Information poverty** (feature matching): Improver sees only scalar outcomes, not
   opponent strategies
4. **Selection inefficiency** (PSRO): Opponents chosen by fitness, not strategic importance

Each experiment targets one or two of these causes. The optimal long-term solution likely
combines all four: continuous fitness + asymmetric compute + structured feedback + Nash-weighted
selection. But scientific methodology requires isolating variables, so the experiments are
designed to test mechanisms individually or in theoretically motivated pairs.

---

*Document version: 1.0. Generated 2026-04-08 for NeurIPS 2026 paper planning.*
