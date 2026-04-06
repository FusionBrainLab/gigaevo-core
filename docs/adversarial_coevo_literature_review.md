# Adversarial Co-Evolution with LLMs: Literature Review

**Date**: April 2026
**Purpose**: Identify the most promising symmetric adversarial tasks for a NeurIPS 2026 paper on adversarial co-evolution with MAP-Elites and LLM-guided mutation.
**NeurIPS 2026 deadline**: Abstract May 4, full paper May 6 (AoE).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Competitive Landscape: Who Is Doing What](#2-competitive-landscape)
3. [Classic Co-Evolution Theory: What Makes Arms Races Work](#3-classic-co-evolution-theory)
4. [LLM-Based Program Evolution: State of the Art](#4-llm-based-program-evolution)
5. [Candidate Adversarial Tasks: Detailed Analysis](#5-candidate-adversarial-tasks)
6. [Comparative Assessment Matrix](#6-comparative-assessment-matrix)
7. [Recommended Direction](#7-recommended-direction)
8. [References](#8-references)

---

## 1. Executive Summary

The intersection of adversarial co-evolution, quality-diversity (QD) algorithms, and LLM-guided code generation is an emerging but sparsely populated research space. As of April 2026, the key players are:

- **GAME** (ALife 2025) — Two-population MAP-Elites without LLMs, tested on battle games, robot wrestling, and deck building
- **DRQ** (Sakana AI, Jan 2026) — LLM-guided evolution in Core War, single toy domain, no QD/MAP-Elites
- **FMSP/QDSP** (RLC 2025) — Foundation model self-play with QD, tested on Car Tag and Gandalf (jailbreaking)
- **MAGIC** (Feb 2026) — Co-evolving attacker-defender for LLM safety via RL, no code evolution
- **CoCoEvo** (Feb 2025) — Co-evolving programs and test cases for code generation, cooperative not adversarial
- **CURE** (NeurIPS 2025 Spotlight) — Co-evolving coder and tester via RL, cooperative, fine-tunes weights

**Our unique position**: adversarial (not cooperative) + MAP-Elites QD (not vanilla EA or RL) + LLM-guided code mutation (not neuroevolution or weight updates) + general-purpose framework. Nobody has combined all four. The gap is real, but it requires a compelling task domain to be publishable.

**The core problem with our pilot** (optimizer vs deceptive landscape): The task is structurally asymmetric. Landscapes can trivially achieve 100% fitness by gen 6 because creating a deceptive landscape is categorically easier than solving arbitrary optimization problems. No arms race can emerge when one side has already won.

**What we need**: A task where both sides write Python code, neither has a structural advantage, partial credit enables gradual improvement, and interesting emergent strategies can arise.

---

## 2. Competitive Landscape

### 2.1 GAME: Generational Adversarial MAP-Elites

**Paper**: "Adversarial Coevolutionary Illumination with Generational Adversarial MAP-Elites" (ALife 2025, extended arxiv version Oct 2025)

**What it does**: Alternates which population is evolved each generation. Uses vision embedding model for behavior characterization (no hand-crafted descriptors). Tested on:
- Multi-agent battle game
- Soft-robot wrestling
- Deck building game

**Key findings**:
- Arms-race-like dynamics observed
- Generational extinction (starting each gen from scratch) increases open-endedness
- Neutral mutations preserved as stepping stones to high performance
- Acknowledged limitation: "capacity for truly open-ended discovery remains constrained by the finite nature of the underlying search spaces"

**Gap vs us**: No LLM mutation. Uses neuroevolution (neural network parameters). Smaller search spaces. No code-level programs.

### 2.2 DRQ: Digital Red Queen

**Paper**: "Digital Red Queen: Adversarial Program Evolution in Core War with LLMs" (Sakana AI + MIT, Jan 2026)

**What it does**: Self-play in Core War. Each round, LLM evolves a new warrior to defeat all predecessors. Sequential (not population-based).

**Key findings**:
- Warriors become increasingly general vs held-out human warriors
- Less behaviorally diverse across independent runs (convergent evolution)
- Classic Red Queen dynamics observed

**Gap vs us**: No QD/MAP-Elites — single lineage, no diversity maintenance. Single domain (Core War assembly). No two-population coevolution — sequential self-play against history.

**Suggestion from paper**: "similarly minimal self-play approaches could prove useful in other more practical multi-agent adversarial domains, like real-world cybersecurity or combating drug resistance."

### 2.3 FMSP/QDSP: Foundation Model Self-Play

**Paper**: "Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models" (RLC 2025)

**What it does**: Three variants — vanilla FMSP, Novelty-Search Self-Play, Quality-Diversity Self-Play. Uses foundation models to generate code policies. Tested on:
- **Car Tag**: pursuer-evader, continuous control. Policies use RL, tree search, heuristics.
- **Gandalf**: LLM jailbreaking simulation (6 levels of difficulty).

**Key findings**:
- QDSP (diversity + quality) performs best overall
- FM enables "leaping across local optima in policy space" via external knowledge
- Implicit arms races: attack strategies prompt automatic defense patching

**Gap vs us**: Does not use MAP-Elites archive structure. Gandalf is asymmetric (attacker vs static defender). Car Tag is toy-scale.

### 2.4 MAGIC: Co-Evolving Attacker-Defender for LLM Safety

**Paper**: "MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety" (Feb 2026)

**What it does**: Multi-turn multi-agent RL. Attacker rewrites queries into deceptive prompts; defender optimizes refusal policy. Uses RL training, not code evolution.

**Key findings**:
- Novel combinatorial attack strategies emerged through iterative RL
- Superior defense rates without hurting helpfulness
- Attack Pool Benchmark promotes progressively harder attacks

**Gap vs us**: RL-based (weight updates), not code evolution. Asymmetric by design (attacker and defender have different architectures). Domain-specific to LLM safety.

### 2.5 CoCoEvo and CURE: Cooperative Co-Evolution

**CoCoEvo** (Feb 2025): Co-evolves programs and test cases for code generation. LLM-based crossover and mutation. Programs and tests drive each other's improvement. SOTA on code generation benchmarks.

**CURE** (NeurIPS 2025 Spotlight): Co-evolves coder LLM and unit tester LLM via RL. 5.3% improvement on code accuracy. Cooperative, not adversarial.

**Gap vs us**: Both are cooperative (coder and tester help each other). Not adversarial co-evolution. CURE fine-tunes model weights, not code. CoCoEvo is closer but misses the adversarial framing and QD diversity.

### 2.6 Other Relevant Work

- **SATLUTION** (Sep 2025): LLM-based autonomous SAT solver evolution. Beats SAT Competition 2025 winners. 10K+ LOC evolved. Not adversarial — static benchmark.
- **AlphaEvolve** (DeepMind, May 2025): Gemini-powered coding agent for algorithm discovery. Static evaluators, no adversarial component.
- **ReEvo** (NeurIPS 2024): LLM as hyper-heuristic with reflective evolution. Static optimization, no adversarial.
- **Learning Game-Playing Agents** (Aug 2025): Python code policies for Atari games via LLM. Single-player only, no adversarial tested.
- **Cultural Evolution of Cooperation** (Dec 2024): LLM agents in iterated Donor Game. Natural language strategies, not code.

---

## 3. Classic Co-Evolution Theory: What Makes Arms Races Work

### 3.1 Key Pathologies

From decades of competitive co-evolution research (Popovici, De Jong, Ficici, Cartlidge, Bullock):

1. **Cycling / Rock-Paper-Scissors**: Strategies cycle without progress. Common in intransitive games where no clear dominance hierarchy exists. Population forgets old solutions when the opponent shifts.

2. **Disengagement**: One population becomes so dominant that fitness variation in the losing population drops to zero. No selection gradient remains. Populations drift randomly.

3. **Over-specialization**: Strategies become narrowly adapted to current opponents rather than generally strong. Similar to overfitting in ML.

4. **Mediocre stable states**: System converges to a suboptimal equilibrium that is locally stable for both populations.

### 3.2 Conditions for Sustained Arms Races

From the literature, arms races require:

| Condition | Why It Matters |
|-----------|----------------|
| **Gradual fitness signal** | Binary win/lose causes disengagement. Partial credit keeps selection pressure alive. |
| **Strategy richness** | Search space must support diverse, novel strategies. Simple spaces exhaust quickly. |
| **Intransitivity management** | Some intransitivity is good (prevents convergence), but too much causes cycling. |
| **Diversity preservation** | Populations must maintain strategic diversity. MAP-Elites helps here. |
| **Moderate virulence** | "Parasites" (adversaries) should occasionally lose to maintain selection gradient for "hosts." (Cartlidge & Bullock, 2004) |
| **Symmetry** | If one side is structurally easier, it wins instantly and the arms race collapses. |
| **Stepping stones** | Neutral mutations must be preserved as intermediate steps. (GAME finding) |

### 3.3 Why MAP-Elites Helps

MAP-Elites naturally addresses several co-evolutionary pathologies:
- **Anti-cycling**: Archive preserves solutions from all generations. Old strategies are never forgotten.
- **Anti-disengagement**: Diverse archive ensures there are always opponents of varying difficulty.
- **Anti-over-specialization**: Behavior characterization forces strategies to be diverse, not just fit.
- **Stepping stones**: Neutral mutations survive in their niche even if not globally optimal.

This is a theoretically motivated advantage we should emphasize in the paper.

### 3.4 Hillis's Sorting Networks: The Classic Success Story

Hillis (1990) co-evolved sorting networks against adversarial test inputs. Key lessons:
- Parasites (test inputs) evolved to expose weaknesses in sorting networks
- Coevolution found better networks than evolution alone
- Required initial population with some features of optimal solutions
- Result: a network only 1 comparator longer than the best known human solution

This is the canonical example of productive adversarial co-evolution. It works because:
- Both sides have comparable complexity
- Partial credit (how many inputs are correctly sorted)
- Rich strategy space for both sides
- Clear metric of progress

---

## 4. LLM-Based Program Evolution: State of the Art

### 4.1 What Works for LLM Code Evolution

From FunSearch, AlphaEvolve, SATLUTION, OpenELM, and ReEvo, the following patterns emerge:

**Successful tasks share**:
- Self-contained Python functions (or small programs)
- Automated evaluation (no human judgment needed)
- Rich strategy space where many valid approaches exist
- Clear numerical fitness signal
- Modest computational requirements per evaluation (seconds, not hours)

**LLM mutation strengths**:
- Can make semantic leaps (restructure algorithm, try completely different approach)
- Knowledge of algorithm design, data structures, heuristics
- Can incorporate natural language feedback about what went wrong
- Effective crossover (combine ideas from two parent programs)

**LLM mutation weaknesses**:
- Struggles with precise numerical tuning (RL/gradient methods are better)
- Code length limitations (~200-500 lines practical maximum per function)
- Can hallucinate non-existent APIs or libraries
- Requires clean evaluation feedback to improve

### 4.2 Scale of Prior Art

| System | Code Scale | Eval Time | Domain |
|--------|-----------|-----------|--------|
| FunSearch | ~50-100 lines | seconds | cap set, bin packing |
| AlphaEvolve | full codebase | minutes | matrix multiplication, scheduling |
| SATLUTION | 10K+ LOC (C++) | minutes | SAT solver |
| OpenELM | ~100 lines | seconds | Sodarace, image generation |
| DRQ | ~50 lines assembly | seconds | Core War |
| FMSP/QDSP | ~100 lines Python | seconds | Car Tag, Gandalf |

Our system should target **50-200 line Python functions** with **< 30 second evaluation time** for practical iteration speed.

---

## 5. Candidate Adversarial Tasks: Detailed Analysis

### 5.1 Solver vs Instance Generator (Combinatorial Optimization)

**Concept**: Population A evolves solvers (heuristic algorithms) for NP-hard problems. Population B evolves instance generators that create hard instances for the solvers.

**Specific variants**:
- **SAT solver vs SAT instance generator**: Solver writes a DPLL/CDCL-style heuristic. Generator creates CNF formulas designed to be hard for the solver.
- **TSP solver vs TSP instance generator**: Solver writes a tour construction/improvement heuristic. Generator places cities in configurations that confuse the solver.
- **Graph coloring solver vs graph generator**: Similar structure.

**Prior art**: Coevolutionary algorithm portfolio optimization (Springer 2025), adversarial hard instance generation for neural combinatorial solvers (ICLR 2022), ToughSAT instance generator, hardness-adaptive curriculum learning for TSP (2022).

**Symmetry analysis**: Moderate. Generating hard instances is generally easier than solving them (information asymmetry). However, the generator's job becomes harder as solvers improve. The fitness signal is gradual (runtime ratio, optimality gap). SAT has the risk that generators can create trivially hard instances (random 3-SAT near the phase transition).

**Arms race likelihood**: Good if constrained properly. The generator must produce structured (not random) instances. The solver has rich strategy space (variable ordering, clause learning, restart policies). Prior work shows coevolution improves solver generalization.

**Novelty**: Medium. Solver-instance coevolution is well-studied in EC literature. The LLM-guided + MAP-Elites angle is new. SATLUTION showed LLMs can evolve real solvers. But nobody has done the adversarial version with LLMs.

**Implementation**: Straightforward. SAT/TSP evaluation is fast. Both sides write Python functions. Clear metrics (runtime, optimality gap, solution quality).

**Assessment**:
- Symmetry: 5/10 (generator side is easier)
- Arms race: 7/10
- Novelty: 6/10
- Feasibility: 9/10

### 5.2 Iterated Prisoner's Dilemma / Matrix Game Strategies

**Concept**: Population A and Population B each evolve strategies for a repeated matrix game. Each strategy is a Python function that decides actions based on history.

**Specific variants**:
- **IPD**: Classic cooperate/defect. Rich strategy space (tit-for-tat, grim trigger, Pavlov, extortion strategies, memory-N strategies).
- **Iterated Colonel Blotto**: Allocate limited resources across N battlefields each round.
- **Hawk-Dove with reputation**: Extended game with reputation tracking.

**Prior art**: Extensive classical EC work on IPD coevolution. LLM agents in IPD (2024 — tested Llama, Claude, GPT-4o). Cultural Evolution of Cooperation (Dec 2024 — natural language strategies in Donor Game). ALYMPICS (2025 — LLM agents in game theory scenarios).

**Symmetry analysis**: High. Both sides play the same game with the same action space. IPD is perfectly symmetric. Blotto is symmetric if budgets are equal.

**Arms race likelihood**: Mixed. IPD has a known problem: tit-for-tat variants tend to dominate quickly, and the strategy space (despite being infinite in theory) empirically converges. Extortion strategies (Press & Dyson, 2012) add complexity but are fragile to evolution. Colonel Blotto has richer strategic structure and natural intransitivity.

**Novelty**: Medium-low for IPD (heavily studied). Higher for Colonel Blotto with LLM code evolution (nobody has done this). The "LLMs write Python strategies that play games" angle exists in the cultural evolution literature but with natural language, not code.

**Implementation**: Very easy. Evaluation is milliseconds. Both sides write `def choose_action(history) -> action`. Clear metric (average payoff across many rounds).

**Assessment (IPD)**:
- Symmetry: 10/10
- Arms race: 4/10 (converges quickly)
- Novelty: 3/10
- Feasibility: 10/10

**Assessment (Colonel Blotto)**:
- Symmetry: 9/10
- Arms race: 7/10 (rich intransitive strategies)
- Novelty: 7/10
- Feasibility: 9/10

### 5.3 Adversarial Test Generation vs Program Synthesis

**Concept**: Population A evolves programs that solve a class of problems. Population B evolves test cases that distinguish correct programs from incorrect ones. Both sides write Python.

**Specific variants**:
- **Algorithm correctness**: A writes a sorting/searching/graph algorithm. B writes edge-case inputs that expose bugs.
- **Specification discovery**: A writes programs matching unknown specifications. B writes tests that reveal specification violations.

**Prior art**: CoCoEvo (Feb 2025) does cooperative program-test co-evolution. CURE (NeurIPS 2025) co-evolves coder and tester cooperatively. CodeContests+ generates adversarial tests for competitive programming. EvalPlus (NeurIPS 2023) showed 80x test augmentation drops GPT-4 from 88.4% to 76.2%.

**Symmetry analysis**: Low. Writing tests is fundamentally easier than writing correct programs. The test side will dominate because any edge case (empty input, very large input, negative numbers) is easy to generate, while writing correct algorithms requires genuine algorithmic understanding.

**Arms race likelihood**: Low for straightforward algorithm correctness — test generators win easily. Could work better for ambiguous specifications where "correct" behavior is debatable, but then evaluation becomes subjective.

**Novelty**: Low. CoCoEvo already exists (cooperative variant). The adversarial framing adds something but is not dramatically different.

**Assessment**:
- Symmetry: 3/10 (tests are much easier)
- Arms race: 3/10
- Novelty: 4/10
- Feasibility: 8/10

### 5.4 Attack vs Defense Code (Cybersecurity-Flavored)

**Concept**: Population A evolves attack programs (e.g., payloads, exploit generators). Population B evolves defense programs (e.g., filters, validators, detection heuristics). Pure Python, no real vulnerabilities needed.

**Specific variants**:
- **Payload obfuscation vs detection**: A evolves Python functions that encode a target payload to evade pattern detection. B evolves Python filters that detect encoded payloads.
- **Steganography vs steganalysis**: A embeds hidden messages in data. B detects hidden messages.
- **Evasive programs vs sandboxes**: A writes programs that behave differently when observed. B writes analysis programs that detect evasion.

**Prior art**: Adversarial GP for cybersecurity (Toutouh). Coevolution of attacks and defenses in CyberBattleSim. MAGIC (Feb 2026) for LLM safety. Malware source code generation via LLMs (2025).

**Symmetry analysis**: Varies by variant. Obfuscation vs detection can be balanced if the obfuscation space is constrained. Steganography vs steganalysis has natural balance (information-theoretic limits apply to both sides).

**Arms race likelihood**: Good for steganography/steganalysis. The encoding and detection methods have comparable complexity. Natural escalation: simple encoding -> frequency analysis detects it -> more sophisticated encoding -> statistical tests detect it -> etc.

**Novelty**: High. Nobody has done adversarial code co-evolution for steganography or obfuscation with LLMs + MAP-Elites.

**Implementation**: Moderate complexity. Need to define a clean interface (what constitutes a payload, what the detection API looks like). Evaluation is fast (microseconds for encoding/detection).

**Assessment (Steganography)**:
- Symmetry: 7/10
- Arms race: 7/10
- Novelty: 8/10
- Feasibility: 7/10

### 5.5 Competitive Scheduling/Resource Games

**Concept**: Two agents compete for shared resources. Each writes a Python strategy function. Examples: competitive job scheduling, bandwidth allocation, auction bidding.

**Specific variant — Competitive Scheduling Game**:
- N jobs arrive over T time steps
- Each agent controls M machines
- Jobs have different values and processing times
- Both agents bid on incoming jobs; highest bidder gets the job (pays the bid)
- Fitness = total value of completed jobs minus bids paid

**Prior art**: NeurIPS 2024 Auto-Bidding Competition (48 agents, 500M records). Coevolutionary agents in combinatorial auctions (Springer 2010). Game-theoretic resource allocation (extensive literature).

**Symmetry analysis**: Perfect if both agents start with same resources.

**Arms race likelihood**: Good. Rich strategy space (when to bid high, when to pass, portfolio management, opponent modeling). Natural intransitivity (aggressive bidding beats passive, but loses to selective bidding, which loses to aggressive in other situations).

**Novelty**: Medium. Auction/bidding coevolution has precedent in EC. The LLM code evolution angle is new. The NeurIPS auto-bidding competition shows the community cares about this.

**Implementation**: Moderate. Need to design the game mechanics carefully to ensure balance.

**Assessment**:
- Symmetry: 9/10
- Arms race: 6/10
- Novelty: 6/10
- Feasibility: 7/10

### 5.6 Heuristic vs Adversarial Input for Optimization (Improved Version)

**Concept**: Redesign of our current optimizer-vs-landscape task to fix the asymmetry problem.

**Key fix**: Instead of "optimizer vs landscape function," make it "heuristic A vs heuristic B" where both solve the SAME problem but compete on a shared instance. The instance is drawn from a distribution that co-evolves (or is fixed).

**Specific variant — Solver Duel**:
- Both populations evolve solvers for the same problem class (e.g., graph coloring, scheduling)
- Evaluation: random instance is generated; both solvers run; solver with better solution wins
- Fitness is relative (win rate against current opponent population)
- MAP-Elites BC: algorithm features (e.g., runtime, memory usage, greediness)

**Symmetry analysis**: Perfect. Both sides do the exact same task.

**Arms race likelihood**: Moderate. Risk of convergence to similar strategies (since both are optimizing the same objective). But MAP-Elites diversity should help — e.g., one niche has fast-but-approximate solvers, another has slow-but-optimal ones.

**Novelty**: High as a framework concept. But the competitive solver setup has been studied in algorithm selection literature.

**Assessment**:
- Symmetry: 10/10
- Arms race: 5/10 (risk of convergence)
- Novelty: 6/10
- Feasibility: 8/10

### 5.7 Sorting Networks vs Adversarial Inputs (Hillis Revisited with LLMs)

**Concept**: Classic Hillis task but with LLM-generated Python code. Population A evolves sorting network implementations (comparator-based). Population B evolves input sequences that the networks fail to sort correctly.

**Prior art**: Hillis (1990) — the foundational co-evolution success story. Extensive follow-up work. Co-evolving faults for fault-tolerant sorting networks.

**Symmetry analysis**: Low-moderate. Generating adversarial inputs is easier than constructing valid sorting networks. The input side's search space is simpler (permutations of N elements) vs the network side (sequences of comparators that must form a valid sorting network).

**Arms race likelihood**: Demonstrated empirically by Hillis. But the LLM angle may struggle — sorting networks are highly structured objects that LLMs may not handle well. The "network" side requires precise combinatorial construction.

**Novelty**: Low-medium. Directly revisiting a 35-year-old experiment with LLMs. Interesting if results are dramatically different, but risky.

**Assessment**:
- Symmetry: 4/10
- Arms race: 6/10 (empirically demonstrated)
- Novelty: 4/10
- Feasibility: 6/10 (LLMs may struggle with sorting network structure)

### 5.8 Prompt Attack vs Prompt Defense (LLM Jailbreaking)

**Concept**: Population A evolves Python functions that construct adversarial prompts to extract a secret from an LLM. Population B evolves Python functions that construct system prompts/filters to protect the secret.

**Prior art**: FMSP/QDSP tested on Gandalf (jailbreaking). MAGIC (Feb 2026) co-evolves attacker/defender via RL. Red teaming literature is enormous. DeepTeam framework (Nov 2025). "The Attacker Moves Second" (2025).

**Symmetry analysis**: Debatable. Some argue attacking is easier (attacker needs one success, defender must block all attacks). Others argue defending is easier (just refuse everything). In practice, the constraint of maintaining helpfulness makes defense harder, creating reasonable balance.

**Arms race likelihood**: Good. MAGIC demonstrated novel combinatorial attack strategies emerging through co-evolution. The strategy space is very rich (encoding, role-play, multi-turn manipulation vs input filtering, output checking, instruction reinforcement).

**Novelty**: Medium. MAGIC and FMSP already explored this territory. Our angle (MAP-Elites QD + code-level evolution rather than RL weight updates) adds something, but we'd be the third paper on this exact topic.

**Implementation complexity**: Moderate. Requires LLM inference for each evaluation (cost and latency). Need to define what constitutes "success" precisely.

**Assessment**:
- Symmetry: 6/10
- Arms race: 7/10
- Novelty: 4/10 (MAGIC, FMSP already exist)
- Feasibility: 6/10 (LLM inference cost per eval)

### 5.9 Steganography: Encoder vs Decoder (NEW PROPOSAL)

**Concept**: A pure information-theoretic competitive coding task.

- **Population A (Encoders)**: Write Python functions that embed a secret message into a cover medium (text, numbers, or structured data) such that the output looks "natural" according to a statistical test.
- **Population B (Decoders/Detectors)**: Write Python functions that analyze data and determine whether a hidden message is present, and optionally extract it.

**Game mechanics**:
- Encoder receives (message, cover_data) and returns modified_data
- Detector receives (data) and returns (is_stego: bool, confidence: float)
- Encoder fitness: message successfully recoverable + detector can't distinguish from natural data
- Detector fitness: correctly classify stego vs natural data + extract messages

**Why this works**:
- **Perfect symmetry of difficulty**: Information theory guarantees that better encoding requires deeper statistical analysis, and better detection requires more sophisticated encoding. Neither side has a structural advantage.
- **Gradual arms race signal**: Detection confidence is continuous (0-1). Encoding success is measurable (bit error rate). No binary collapse.
- **Rich strategy space**: Substitution ciphers -> frequency-preserving encoding -> context-dependent encoding -> learned statistical models -> adversarial perturbation-based encoding. Detection: frequency analysis -> n-gram statistics -> entropy measures -> distribution tests.
- **Self-contained Python**: Both sides write pure functions, no external dependencies needed.
- **Fast evaluation**: Microseconds per encode/detect operation.
- **Interesting emergent strategies**: Could discover novel steganographic techniques or statistical tests.
- **Publishable**: Novel intersection of steganography, co-evolution, QD, and LLM code evolution.

**Potential issue**: The "naturalness" constraint requires a reference distribution. We can use a simple statistical test (character frequency, word frequency for text) or provide reference data samples.

**Assessment**:
- Symmetry: 8/10
- Arms race: 8/10
- Novelty: 9/10
- Feasibility: 7/10

### 5.10 Function Approximation Duel

**Concept**: A novel symmetric task designed specifically for productive co-evolution.

- **Population A (Generators)**: Write Python functions that generate mathematical functions f(x) that are hard to approximate.
- **Population B (Approximators)**: Write Python functions that approximate arbitrary f(x) using limited compute/samples.

**Game mechanics**:
- Generator produces a function f: R^d -> R with constraints (bounded, Lipschitz, etc.)
- Approximator gets N sample points and must predict f at test points
- Generator fitness: approximation error of best approximators
- Approximator fitness: negative approximation error on generator functions

**Why this could work**:
- Both sides write Python functions
- Generators can create fractal functions, highly oscillatory functions, functions with narrow spikes, etc.
- Approximators can use interpolation, regression, basis functions, local fitting, etc.
- Natural arms race: simple polynomials -> polynomial approximators -> pathological functions -> adaptive methods -> etc.

**Potential issue**: Generator side might be easier (just add more oscillations). Need to constrain function smoothness.

**Assessment**:
- Symmetry: 6/10 (generators may have slight advantage)
- Arms race: 6/10
- Novelty: 7/10
- Feasibility: 8/10

### 5.11 Competitive Code Golf / Program Compression

**Concept**: Two populations compete on program equivalence and length.

- **Population A (Writers)**: Write programs that compute some function as compactly as possible
- **Population B (Distinguishers)**: Write test inputs that distinguish between programs that look equivalent but aren't

This is essentially program testing co-evolution but focused on the compactness-correctness tradeoff.

**Assessment**: Similar issues to 5.3 (asymmetric). Skipping detailed analysis.

### 5.12 Colonel Blotto with Evolving Strategies (DETAILED PROPOSAL)

**Concept**: A game-theoretic resource allocation competition.

**Game mechanics**:
- N battlefields (e.g., N=5 or N=10)
- Each player has B total units to allocate across battlefields
- Player wins a battlefield if they allocated more units to it
- Player's score = number of battlefields won (or weighted sum if battlefields have different values)
- Iterated: play M rounds, can adapt based on opponent's history
- Each strategy is a Python function: `def allocate(history, round_num, n_fields, budget) -> list[int]`

**Why this is excellent**:
- **Perfect symmetry**: Same action space, same rules, same budget
- **Rich strategy space**: Uniform spread, concentrate-and-sacrifice, pattern recognition, randomization, opponent modeling, Bayesian updating
- **Natural intransitivity**: A beats B, B beats C, C beats A (well-known property of Colonel Blotto)
- **Gradual fitness**: Win ratio over many rounds gives smooth signal
- **History-dependent**: Strategies can model opponents, creating escalating sophistication
- **LLM-friendly**: The strategy function is conceptually simple but algorithmically deep
- **Known theoretical richness**: Colonel Blotto has deep game-theoretic foundations (Borel 1921, Roberson 2006)
- **Practical relevance**: Models resource allocation in politics, military, advertising, sports

**MAP-Elites behavior characterization ideas**:
- Aggressiveness (how concentrated are allocations)
- Adaptiveness (how much do allocations change based on history)
- Entropy of allocation distribution
- Number of "sacrificed" battlefields (always allocate 0)

**Prior art**: Python Blotto implementations exist on GitHub. EC work on Blotto strategies exists but is sparse. Nobody has done LLM-guided Blotto strategy evolution with QD.

**Assessment**:
- Symmetry: 10/10
- Arms race: 8/10 (natural intransitivity drives escalation)
- Novelty: 8/10 (for NeurIPS: game theory + QD + LLM evolution is fresh)
- Feasibility: 9/10 (simple to implement, fast to evaluate)

---

## 6. Comparative Assessment Matrix

| Task | Symmetry | Arms Race | Novelty | Feasibility | Total | Key Risk |
|------|----------|-----------|---------|-------------|-------|----------|
| **Colonel Blotto** | 10 | 8 | 8 | 9 | **35** | May converge to mixed strategies |
| **Steganography** | 8 | 8 | 9 | 7 | **32** | Defining "naturalness" baseline |
| **Solver vs Instance Generator** | 5 | 7 | 6 | 9 | **27** | Asymmetric difficulty |
| **Cybersecurity (obfuscation/detection)** | 7 | 7 | 8 | 7 | **29** | Defining clean API |
| **Competitive Scheduling** | 9 | 6 | 6 | 7 | **28** | Complex game design |
| **Prompt Attack/Defense** | 6 | 7 | 4 | 6 | **23** | MAGIC/FMSP already exist; LLM inference cost |
| **Solver Duel** | 10 | 5 | 6 | 8 | **29** | Both converge to same strategy |
| **Colonel Blotto (IPD variant)** | 10 | 4 | 3 | 10 | **27** | IPD converges quickly |
| **Function Approximation** | 6 | 6 | 7 | 8 | **27** | Generator may dominate |
| **Sorting Networks** | 4 | 6 | 4 | 6 | **20** | LLMs bad at combinatorics |
| **Adversarial Tests vs Programs** | 3 | 3 | 4 | 8 | **18** | Heavily asymmetric |

---

## 7. Recommended Direction

### Primary Task: Colonel Blotto Resource Game

**Rationale**:
1. **Highest total score** (35/40) driven by perfect symmetry and strong arms race potential
2. **Natural intransitivity** is a known theoretical property, not an accident — this is the ideal condition for productive co-evolution without cycling
3. **Game-theoretic depth** provides rich analytical framework for the paper (Nash equilibria, best responses, regret analysis)
4. **MAP-Elites is perfectly suited**: behavior characterization over allocation patterns creates meaningful niches (aggressive, defensive, adaptive, randomized)
5. **LLM strengths match the task**: writing allocation strategies requires algorithmic creativity, opponent modeling, and pattern recognition — exactly what LLMs are good at
6. **Implementation simplicity**: Both sides write `allocate(history, round, n_fields, budget) -> allocations`. Evaluation is milliseconds. No external dependencies.
7. **Paper narrative**: "We demonstrate that adversarial co-evolution with MAP-Elites and LLM-guided mutation discovers sophisticated game-theoretic strategies in Colonel Blotto, a classical resource allocation game with provably no pure-strategy Nash equilibrium."
8. **NeurIPS relevance**: Intersects game theory (well-established at NeurIPS), open-ended evolution (growing interest), and LLM capabilities (hot topic).

### Secondary Task (for generality): Steganography Encoder vs Detector

**Rationale**:
1. Highest novelty score (9/10) — nobody has done this
2. Complementary to Colonel Blotto (information theory vs game theory)
3. Demonstrates framework generality across very different domains
4. Could yield genuinely novel steganographic techniques

### Paper Structure

A multi-domain paper showing:
1. Framework description (adversarial MAP-Elites + LLM mutation)
2. Colonel Blotto experiments (primary domain, deep analysis)
3. Steganography experiments (secondary domain, generality demonstration)
4. Analysis: What strategies emerged? Did arms races sustain? How does MAP-Elites diversity prevent co-evolutionary pathologies?
5. Comparison to baselines: non-adversarial evolution, adversarial without QD, QD without LLM

### Risk Mitigation

- **Colonel Blotto convergence risk**: If strategies converge to mixed-strategy Nash equilibrium too quickly, increase battlefield count or add battlefield-specific values
- **Steganography "naturalness" definition**: Use simple statistical tests (character bigram frequency for text stego) as reference; evolve more sophisticated tests as part of the detector population
- **Tight deadline (1 month to submission)**: Colonel Blotto can be implemented in days given existing infrastructure. Steganography requires more design work.

---

## 8. References

### Adversarial Co-Evolution

- GAME: "Adversarial Coevolutionary Illumination with Generational Adversarial MAP-Elites" — [arxiv.org/abs/2505.06617](https://arxiv.org/abs/2505.06617), ALife 2025
- DRQ: "Digital Red Queen: Adversarial Program Evolution in Core War with LLMs" — [arxiv.org/abs/2601.03335](https://arxiv.org/abs/2601.03335), Sakana AI + MIT, Jan 2026
- FMSP/QDSP: "Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models" — [arxiv.org/abs/2507.06466](https://arxiv.org/abs/2507.06466), RLC 2025
- MAGIC: "A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety" — [arxiv.org/abs/2602.01539](https://arxiv.org/abs/2602.01539), Feb 2026
- CoCoEvo: "Co-Evolution of Programs and Test Cases to Enhance Code Generation" — [arxiv.org/abs/2502.10802](https://arxiv.org/abs/2502.10802), Feb 2025
- CURE: "Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning" — [arxiv.org/abs/2506.03136](https://arxiv.org/abs/2506.03136), NeurIPS 2025 Spotlight
- Cartlidge & Bullock: "Combating Coevolutionary Disengagement by Reducing Parasite Virulence" — [eprints.soton.ac.uk/261440](https://eprints.soton.ac.uk/261440/2/Combating.pdf), 2004
- "Overcoming Binary Adversarial Optimisation with Competitive Coevolution" — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-70071-2_8), PPSN 2024
- "Ranking Diversity Benefits Coevolutionary Algorithms on an Intransitive Game" — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-70071-2_14), PPSN 2024
- "A review of landmark articles in the field of co-evolutionary computing" — [arxiv.org/abs/1506.05082](https://arxiv.org/pdf/1506.05082)
- "Global progress in competitive co-evolution: a systematic comparison of alternative methods" — [Frontiers](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1470886/full), 2024
- "Substitution of the Fittest: A Novel Approach for Mitigating Disengagement" — [arxiv.org/abs/2108.03156](https://arxiv.org/pdf/2108.03156)
- GECCO 2025 Tutorial: "Coevolutionary Computation for Adversarial Deep Learning" — [gecco-2025.sigevo.org](https://gecco-2025.sigevo.org/Tutorial?itemId=5116)

### LLM-Based Program Evolution

- FunSearch: "Mathematical discoveries from program search with large language models" — [Nature](https://www.nature.com/articles/s41586-023-06924-6), 2024
- AlphaEvolve: "A coding agent for scientific and algorithmic discovery" — [DeepMind](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf), May 2025
- SATLUTION: "Autonomous Code Evolution Meets NP-Completeness" — [arxiv.org/abs/2509.07367](https://arxiv.org/abs/2509.07367), Sep 2025
- OpenELM: "Evolution Through Large Models" — [github.com/CarperAI/OpenELM](https://github.com/CarperAI/OpenELM)
- EvoPrompt: "Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers" — [arxiv.org/abs/2309.08532](https://arxiv.org/abs/2309.08532), ICLR 2024
- ReEvo: "Large Language Models as Hyper-Heuristics with Reflective Evolution" — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf)
- EvoTune: "Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning" — [arxiv.org/abs/2504.05108](https://arxiv.org/html/2504.05108v4), COLM 2025
- "Learning Game-Playing Agents with Generative Code Optimization" — [arxiv.org/abs/2508.19506](https://arxiv.org/abs/2508.19506), Aug 2025
- "Evolving code with a large language model" — [Springer](https://link.springer.com/article/10.1007/s10710-024-09494-2), MIT, 2024
- EvoLattice: "Persistent Internal-Population Evolution through Multi-Alternative QD" — [arxiv.org/abs/2512.13857](https://arxiv.org/html/2512.13857)

### Game Theory and Multi-Agent LLMs

- "Cultural Evolution of Cooperation among LLM Agents" — [arxiv.org/abs/2412.10270](https://arxiv.org/abs/2412.10270), Dec 2024
- Multi-Agent Evolve: "LLM Self-Improve through Multi-Agent Co-evolution" — [arxiv.org/abs/2510.23595](https://arxiv.org/html/2510.23595v1)
- ALYMPICS: "LLM Agents Meet Game Theory" — [ACL 2025](https://aclanthology.org/2025.coling-main.193.pdf)
- NeurIPS 2024 Auto-Bidding Competition — [OpenReview](https://openreview.net/forum?id=ZejUjZUF6i)
- Colonel Blotto resource allocation — [arxiv.org/abs/2603.25979](https://arxiv.org/html/2603.25979), Mar 2026

### Software Testing and Adversarial Generation

- CodeContests+: "High-Quality Test Case Generation for Competitive Programming" — [arxiv.org/abs/2506.05817](https://arxiv.org/abs/2506.05817)
- EvalPlus: NeurIPS 2023 — 80x test augmentation reveals LLM weaknesses
- "Adversarial co-evolution of attack and defense in cybersecurity" — [ACM](https://dl.acm.org/doi/pdf/10.1145/3205651.3208287)
- "Coevolutionary Construction of Parallel Algorithm Portfolio Optimization" — [Springer](https://link.springer.com/chapter/10.1007/978-981-96-2841-4_6), 2025

### Classic Co-Evolution Theory

- Hillis (1990): Co-evolving parasites improve simulated evolution as an optimization procedure
- Press & Dyson (2012): Iterated Prisoner's Dilemma contains strategies that dominate any evolutionary opponent — [PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC3387070/)
- Borel (1921): La theorie du jeu et les equations integrales (Colonel Blotto origins)
- Dawkins & Krebs (1979): Arms races between and within species — [Royal Society](https://royalsocietypublishing.org/doi/10.1098/rspb.1979.0081)

### NeurIPS 2026

- Call for Papers — [neurips.cc/Conferences/2026/CallForPapers](https://neurips.cc/Conferences/2026/CallForPapers)
- Deadlines: Abstract May 4, Full paper May 6 (AoE) — [neurips.cc/Conferences/2026/Dates](https://neurips.cc/Conferences/2026/Dates)
