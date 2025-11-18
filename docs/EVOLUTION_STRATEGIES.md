# GigaEvo Evolution Strategies: MAP-Elites & Multi-Island

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [MAP-Elites Algorithm](#map-elites-algorithm)
- [Multi-Island System](#multi-island-system)
- [Configuration](#configuration)
- [Selection Strategies](#selection-strategies)
- [Migration](#migration)
- [Creating Custom Components](#creating-custom-components)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

---

## Overview

GigaEvo's evolution system is powered by **MAP-Elites** (Multi-dimensional Archive of Phenotypic Elites), a quality-diversity algorithm that maintains a diverse population of high-performing solutions.

### Why MAP-Elites?

Unlike traditional evolutionary algorithms that converge to a single "best" solution, MAP-Elites:

- ✅ **Maintains Diversity**: Archives solutions across the entire behavior space
- ✅ **Explores Tradeoffs**: Finds optimal solutions for different behavior characteristics
- ✅ **Avoids Local Optima**: Multiple regions explored simultaneously
- ✅ **Provides Insights**: Reveals relationships between behaviors and fitness

### Multi-Island Enhancement

GigaEvo extends MAP-Elites with **multi-island** evolution:

```
Island 1 (Fitness)           Island 2 (Simplicity)         Island 3 (Speed)
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ Behavior Space:  │        │ Behavior Space:  │        │ Behavior Space:  │
│  - Fitness       │        │  - Fitness       │        │  - Fitness       │
│  - Validity      │        │  - Complexity    │        │  - Runtime       │
│                  │   ←→   │                  │   ←→   │                  │
│ Optimizer:       │        │ Optimizer:       │        │ Optimizer:       │
│  Maximize        │        │  Maximize        │        │  Maximize        │
│  Fitness         │        │  Simplicity      │        │  Speed           │
└──────────────────┘        └──────────────────┘        └──────────────────┘
         ↓                           ↓                           ↓
              Migration (best solutions exchange)
```

**Benefits:**
- **Parallel Objectives**: Each island optimizes for different criteria
- **Cross-Pollination**: Migration shares successful solutions
- **Robustness**: Failure in one island doesn't affect others
- **Scalability**: Easy to add new optimization objectives

---

## Core Concepts

### 1. Behavior Space

A **behavior space** defines the dimensions along which programs are characterized:

```python
BehaviorSpace(
    feature_bounds={
        "fitness": (0.0, 100.0),      # Primary objective
        "validity": (0.0, 1.0),       # Constraint satisfaction
    },
    resolution={
        "fitness": 20,    # 20 bins for fitness dimension
        "validity": 5,    # 5 bins for validity dimension
    },
    binning_types={
        "fitness": BinningType.LINEAR,
        "validity": BinningType.LINEAR,
    }
)
```

**Key Properties:**
- **Dimensions**: Metrics used to characterize programs (e.g., fitness, complexity, validity)
- **Bounds**: Min/max values for each dimension
- **Resolution**: Number of bins (cells) per dimension
- **Binning Type**: How values map to bins (linear, logarithmic, square_root, quantile)

**Total Cells**: `resolution[dim1] × resolution[dim2] × ... × resolution[dimN]`

Example: 20 × 5 = **100 cells** in the above space

### 2. Archive

The **archive** stores the best program for each cell in the behavior space:

```
Fitness (20 bins)
    ↓
    ├─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
    │P│ │P│ │P│P│ │P│ │ │P│P│ │P│ │P│P│ │P│
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
Validity│P│P│P│ │P│ │P│ │P│ │ │P│P│P│P│ │P│ │
(5 bins)├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    │ │P│ │P│ │P│P│ │ │P│P│ │P│P│ │ │P│P│
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    │P│ │P│P│ │ │P│P│P│ │ │P│ │P│P│ │ │P│
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    │ │P│ │ │P│P│ │P│ │P│P│ │P│ │P│P│P│ │
    └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

P = Program elite in this cell
```

**Operations:**
- **Add**: Insert program if it improves the cell's current elite
- **Select**: Choose elites for mutation (various strategies)
- **Remove**: Enforce size limits using removal policies

### 3. Island

An **island** is an independent MAP-Elites population with its own:

```python
Island(
    island_id="fitness_island",
    max_size=100,                     # Archive capacity
    behavior_space=BehaviorSpace(...),
    archive_selector=SumArchiveSelector(...),  # How to compare programs
    elite_selector=FitnessProportional(...),   # How to pick parents
    archive_remover=FitnessRemover(...),       # How to enforce limits
    migrant_selector=TopFitness(...),          # How to pick migrants
)
```

**Lifecycle:**
1. **Add**: Receive new programs (mutants or migrants)
2. **Select Elites**: Choose parents for mutation
3. **Migrate**: Send best solutions to other islands
4. **Enforce Limits**: Remove excess programs if over capacity

### 4. Selectors

Selectors determine which programs to use for different purposes:

| Selector Type | Purpose | Examples |
|---------------|---------|----------|
| **ArchiveSelector** | Decide if new program replaces cell elite | SumArchiveSelector, ParetoArchiveSelector |
| **EliteSelector** | Choose parents for mutation | Random, FitnessProportional, Tournament |
| **MigrantSelector** | Choose programs to send to other islands | TopFitness, Random, Diverse |
| **ArchiveRemover** | Remove excess programs | FitnessRemover, AgeRemover, DiversityRemover |

---

## Architecture

### Component Hierarchy

```
┌───────────────────────────────────────────────────────────┐
│                    EvolutionEngine                         │
│  • Coordinates evolution loop                             │
│  • Manages program lifecycle (FRESH → EVOLVING)           │
│  • Enforces generation limits                             │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ↓
┌───────────────────────────────────────────────────────────┐
│               MapElitesMultiIsland (Strategy)              │
│  • Manages multiple islands                               │
│  • Routes mutants to appropriate islands                  │
│  • Orchestrates migration between islands                 │
│  • Aggregates metrics across islands                      │
└─────────────────────────┬─────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Island 1 │  │ Island 2 │  │ Island 3 │
    │          │  │          │  │          │
    │ Archive  │  │ Archive  │  │ Archive  │
    │ Space    │  │ Space    │  │ Space    │
    │ Selector │  │ Selector │  │ Selector │
    └──────────┘  └──────────┘  └──────────┘
```

### Evolution Loop

The **EvolutionEngine** orchestrates the entire evolution process:

```
┌─────────────────────────────────────────────────────────────┐
│                    EVOLUTION GENERATION                      │
└─────────────────────────────────────────────────────────────┘
    ↓
Phase 1: Wait for Idle
    └→ Block until no DAGs are running
    ↓
Phase 2: Select & Mutate
    ├→ Strategy.select_elites(N) → Pick parents
    ├→ MutationOperator.mutate() → Generate mutants
    └→ Store mutants in Redis (state: FRESH)
    ↓
Phase 3: Wait for Mutant DAGs
    └→ Block until mutant evaluation completes
    ↓
Phase 4: Ingest Completed Programs
    ├→ For each completed program:
    │   ├→ If already in strategy: restore to EVOLVING
    │   ├→ If accepted: Strategy.add() → EVOLVING
    │   └→ If rejected: → DISCARDED
    ↓
Phase 5: Refresh Evolving Programs
    ├→ Flip all EVOLVING → FRESH
    └→ Re-run lineage/insight stages with updated data
    ↓
Phase 6: Wait for Refresh DAGs
    └→ Block until refresh completes
    ↓
[Repeat]
```

---

## MAP-Elites Algorithm

### Classic MAP-Elites

The fundamental algorithm (single-island):

```
1. Initialize empty archive (behavior space)
2. Add seed programs to archive
3. Loop:
    a. Select elite program(s) from archive
    b. Mutate elite(s) to create offspring
    c. Evaluate offspring (run through DAG)
    d. For each offspring:
        - Compute behavior characteristics
        - Map to cell in behavior space
        - If cell empty OR offspring better than current elite:
            * Replace cell with offspring
    e. Repeat until termination
```

### Behavior Space Mapping

Programs map to cells based on their metrics:

```python
def get_cell(self, metrics: dict[str, float]) -> tuple[int, ...]:
    """Map program metrics to behavior space cell."""
    cell_indices = []
    for feature in self.behavior_keys:
        value = metrics[feature]
        bounds = self.feature_bounds[feature]
        resolution = self.resolution[feature]

        # Bin the value
        bin_idx = discretize(value, bounds, resolution, binning_type)
        cell_indices.append(bin_idx)

    return tuple(cell_indices)

# Example:
metrics = {"fitness": 75.3, "validity": 0.82}
cell = get_cell(metrics)  # → (15, 4)
```

### Archive Update

When a new program arrives:

```python
async def add(self, program: Program) -> bool:
    # 1. Determine which cell this program belongs to
    cell = self.behavior_space.get_cell(program.metrics)

    # 2. Get current elite in that cell (if any)
    current_elite = await self.archive.get_elite(cell)

    # 3. Compare using ArchiveSelector
    if current_elite is None or self.selector.is_better(program, current_elite):
        # 4. Replace elite
        await self.archive.set_elite(cell, program)
        return True

    return False
```

---

## Multi-Island System

### Island Independence

Each island operates independently:

```python
# Island 1: Maximize fitness + validity
Island(
    island_id="fitness_island",
    behavior_space=BehaviorSpace(
        keys=["fitness", "validity"],
        resolution=[20, 5]
    ),
    # Selector prioritizes fitness
    archive_selector=SumArchiveSelector(["fitness"])
)

# Island 2: Maximize fitness + simplicity
Island(
    island_id="simplicity_island",
    behavior_space=BehaviorSpace(
        keys=["fitness", "complexity_score"],
        resolution=[20, 10]
    ),
    # Selector prioritizes fitness AND low complexity
    archive_selector=SumArchiveSelector(
        ["fitness", "complexity_score"],
        higher_is_better=[True, False]  # Minimize complexity
    )
)
```

### Mutant Routing

New mutants are routed to appropriate islands:

```python
class MutantRouter:
    async def route_mutant(
        self,
        program: Program,
        candidate_islands: list[Island]
    ) -> Island | None:
        """Decide which island should receive this mutant."""
        # Examples:
        # - Random: random.choice(islands)
        # - Behavior-based: island with matching characteristics
        # - Fitness-based: island where program is most competitive
```

**Routing Strategies:**
- **Random**: Equal distribution across islands
- **Fitness-Aware**: Route to island where program has best relative fitness
- **Behavior-Aware**: Route based on program's behavior characteristics
- **Adaptive**: Learn which islands work best for different program types

### Migration

Periodic exchange of successful solutions between islands:

```python
async def _perform_migration(self):
    """Migrate elites between islands."""
    # 1. Collect migrants from each island
    migrants = []
    for island in self.islands:
        migrants.extend(await island.select_migrants(N))

    # 2. Shuffle for randomness
    random.shuffle(migrants)

    # 3. Send each migrant to a different island
    for migrant in migrants:
        source_island_id = migrant.metadata["current_island"]

        # Choose destination (not source island)
        destinations = [i for i in self.islands
                       if i.id != source_island_id]
        destination = await self.router.route_mutant(migrant, destinations)

        # 4. Add to destination, remove from source
        if await destination.add(migrant):
            await source_island.remove(migrant.id)
```

**Migration Parameters:**
- `migration_interval`: Generations between migrations (e.g., 50)
- `max_migrants_per_island`: Programs each island exports (e.g., 5)
- `enable_migration`: Toggle migration on/off

**Benefits:**
- **Diversity**: Successful solutions spread across behavior spaces
- **Cross-Pollination**: Ideas from one optimization objective help others
- **Robustness**: Prevents any single island from stagnating

---

## Configuration

### Single-Island Setup

Simplest configuration (one behavior space):

```yaml
# config/algorithm/single_island.yaml
behavior_space:
  _target_: gigaevo.config.helpers.build_behavior_space
  keys:
    - fitness
    - validity
  bounds:
    - [0.0, 100.0]
    - [0.0, 1.0]
  resolutions:
    - 20
    - 5
  binning_types:
    - LINEAR
    - LINEAR

islands:
  - _target_: gigaevo.evolution.strategies.map_elites.IslandConfig
    island_id: main_island
    max_size: 100
    behavior_space: ${behavior_space}

    # Archive comparison: sum of fitness
    archive_selector:
      _target_: gigaevo.evolution.strategies.map_elites.SumArchiveSelector
      fitness_keys: [fitness]
      fitness_key_higher_is_better: [true]

    # Parent selection: fitness-proportional
    elite_selector:
      _target_: gigaevo.evolution.strategies.map_elites.FitnessProportionalEliteSelector
      fitness_key: fitness
      fitness_key_higher_is_better: true

    # Remove lowest-fitness programs when full
    archive_remover:
      _target_: gigaevo.evolution.strategies.map_elites.FitnessArchiveRemover
      fitness_key: fitness
      fitness_key_higher_is_better: true

    # Select top-fitness migrants
    migrant_selector:
      _target_: gigaevo.evolution.strategies.map_elites.TopFitnessMigrantSelector
      fitness_key: fitness
      fitness_key_higher_is_better: true

evolution_strategy:
  _target_: gigaevo.evolution.strategies.map_elites.MapElitesMultiIsland
  island_configs: ${islands}
  program_storage: ${ref:redis_storage}
  migration_interval: 50
  enable_migration: false  # No migration with single island
  max_migrants_per_island: 5
```

### Multi-Island Setup

Multiple islands with different objectives:

```yaml
# config/algorithm/multi_island.yaml

# Island 1: Fitness + Validity
fitness_behavior_space:
  _target_: gigaevo.config.helpers.build_behavior_space
  keys: [fitness, validity]
  bounds: [[0.0, 100.0], [0.0, 1.0]]
  resolutions: [20, 5]

# Island 2: Fitness + Simplicity
simplicity_behavior_space:
  _target_: gigaevo.config.helpers.build_behavior_space
  keys: [fitness, complexity_score]
  bounds: [[0.0, 100.0], [0, 1000]]
  resolutions: [20, 10]

islands:
  # Island 1: Optimize fitness
  - _target_: gigaevo.evolution.strategies.map_elites.IslandConfig
    island_id: fitness_island
    max_size: 75
    behavior_space: ${fitness_behavior_space}
    archive_selector:
      _target_: gigaevo.evolution.strategies.map_elites.SumArchiveSelector
      fitness_keys: [fitness]
      fitness_key_higher_is_better: [true]
    # ... (other selectors)

  # Island 2: Optimize simplicity
  - _target_: gigaevo.evolution.strategies.map_elites.IslandConfig
    island_id: simplicity_island
    max_size: 75
    behavior_space: ${simplicity_behavior_space}
    archive_selector:
      _target_: gigaevo.evolution.strategies.map_elites.SumArchiveSelector
      fitness_keys: [fitness, complexity_score]
      fitness_key_higher_is_better: [true, false]  # Minimize complexity
    # ... (other selectors)

evolution_strategy:
  _target_: gigaevo.evolution.strategies.map_elites.MapElitesMultiIsland
  island_configs: ${islands}
  program_storage: ${ref:redis_storage}
  migration_interval: 50      # Migrate every 50 generations
  enable_migration: true
  max_migrants_per_island: 5  # Each island exports 5 migrants
```

### Binning Types

Control how metrics map to bins:

```python
class BinningType(Enum):
    LINEAR = "linear"           # Equal-width bins
    LOGARITHMIC = "logarithmic" # Log-scaled bins (for exponential distributions)
    SQUARE_ROOT = "square_root" # Square root scaling (moderate non-linearity)
```

**Use Cases:**
- **LINEAR**: Default, works for most metrics
- **LOGARITHMIC**: When values span multiple orders of magnitude (e.g., 0.001 to 1000)
---

## Selection Strategies

### Archive Selector

Determines which program should occupy a cell:

#### SumArchiveSelector

Compares programs by summing weighted fitness values:

```python
SumArchiveSelector(
    fitness_keys=["fitness", "validity"],
    fitness_key_higher_is_better=[True, True],
    weights=[1.0, 0.5]  # Fitness has 2x weight of validity
)

# Comparison:
program_a_score = 80.0 * 1.0 + 0.9 * 0.5 = 80.45
program_b_score = 75.0 * 1.0 + 1.0 * 0.5 = 75.50
# Program A wins
```

#### ParetoArchiveSelector

Uses Pareto dominance (multi-objective optimization):

```python
ParetoArchiveSelector(
    fitness_keys=["fitness", "validity"],
    fitness_key_higher_is_better=[True, True]
)

# Program A dominates B if:
# - A is better or equal in ALL objectives
# - A is strictly better in AT LEAST ONE objective
```

### Elite Selector

Chooses parents for mutation:

#### RandomEliteSelector

Uniform random selection:

```python
RandomEliteSelector()
# All elites have equal probability
```

**Pros**: Maximum diversity
**Cons**: Wastes compute on poor solutions

#### FitnessProportionalEliteSelector

Selection probability proportional to fitness:

```python
FitnessProportionalEliteSelector(
    fitness_key="fitness",
    fitness_key_higher_is_better=True
)

# P(program) ∝ fitness(program)
```

**Pros**: Focuses on promising solutions
**Cons**: Can prematurely converge

#### ScalarTournamentEliteSelector

Tournament selection (K random contestants, pick best):

```python
ScalarTournamentEliteSelector(
    fitness_key="fitness",
    fitness_key_higher_is_better=True,
    tournament_size=3
)

# For each selection:
# 1. Randomly pick 3 programs
# 2. Return the one with highest fitness
```

**Pros**: Good balance of exploitation and exploration
**Cons**: Requires tuning tournament_size

### Migrant Selector

Chooses programs to send to other islands:

#### TopFitnessMigrantSelector

Send the best programs:

```python
TopFitnessMigrantSelector(
    fitness_key="fitness",
    fitness_key_higher_is_better=True
)
# Always migrates top-N by fitness
```

#### RandomMigrantSelector

Random selection (diversity-focused):

```python
RandomMigrantSelector()
# Uniform random from archive
```

### Archive Remover

Enforces size limits when archive is full:

#### FitnessArchiveRemover

Remove lowest-fitness programs:

```python
FitnessArchiveRemover(
    fitness_key="fitness",
    fitness_key_higher_is_better=True
)
# Sorts by fitness, removes worst programs
```

#### AgeArchiveRemover

Remove oldest programs (FIFO):

```python
AgeArchiveRemover()
# Programs must have 'created_at' timestamp
```

#### DiversityArchiveRemover

Remove programs to maximize behavioral diversity:

```python
DiversityArchiveRemover(
    behavior_keys=["fitness", "validity"]
)
# Keeps programs that are maximally spread in behavior space
```

---

## Migration

### Migration Lifecycle

```
Generation N:
    ┌──────────────┐
    │ Island 1     │  Elites: [A, B, C, D, E]
    └──────────────┘
    ┌──────────────┐
    │ Island 2     │  Elites: [F, G, H, I, J]
    └──────────────┘

Generation N + migration_interval (e.g., N+50):
    ↓
Step 1: Select Migrants
    ┌──────────────┐
    │ Island 1     │  Sends: [A, B]  (top 2 by fitness)
    └──────────────┘
    ┌──────────────┐
    │ Island 2     │  Sends: [F, G]  (top 2 by fitness)
    └──────────────┘
    ↓
Step 2: Route Migrants
    Island 1 receives: [F, G] from Island 2
    Island 2 receives: [A, B] from Island 1
    ↓
Step 3: Add to Destination Archives
    ┌──────────────┐
    │ Island 1     │  Tries to add F, G to its archive
    └──────────────┘  (Success if they improve cells)
    ┌──────────────┐
    │ Island 2     │  Tries to add A, B to its archive
    └──────────────┘
    ↓
Step 4: Remove from Source Archives
    If F successfully added to Island 1:
        Remove F from Island 2
    (Ensures no duplicates across islands)
```

### Migration Policies

#### Conservative Migration

Infrequent, small batches:

```yaml
migration_interval: 100     # Every 100 generations
max_migrants_per_island: 3  # Only 3 per island
```

**Pros**: Islands develop distinct strategies
**Cons**: Slow cross-pollination

#### Aggressive Migration

Frequent, large batches:

```yaml
migration_interval: 20      # Every 20 generations
max_migrants_per_island: 10 # 10 per island
```

**Pros**: Rapid sharing of good solutions
**Cons**: Islands may homogenize

#### Adaptive Migration

Adjust based on island progress:

```python
# Pseudo-code
if island_has_stagnated():
    migration_rate = HIGH  # Force new ideas
else:
    migration_rate = LOW   # Let island explore
```

---

## Creating Custom Components

### Custom Archive Selector

```python
from gigaevo.evolution.strategies.selectors import ArchiveSelector

class WeightedParetoArchiveSelector(ArchiveSelector):
    """Pareto dominance with weighted objectives."""

    def __init__(
        self,
        fitness_keys: list[str],
        fitness_key_higher_is_better: list[bool],
        weights: list[float]
    ):
        self.keys = fitness_keys
        self.higher = fitness_key_higher_is_better
        self.weights = weights

    def is_better(
        self,
        candidate: Program,
        current: Program
    ) -> bool:
        """True if candidate should replace current."""
        # Apply weights to each objective
        candidate_scores = [
            candidate.metrics[k] * w
            for k, w in zip(self.keys, self.weights)
        ]
        current_scores = [
            current.metrics[k] * w
            for k, w in zip(self.keys, self.weights)
        ]

        # Check Pareto dominance
        better_in_any = False
        worse_in_any = False

        for c_score, curr_score, higher in zip(
            candidate_scores, current_scores, self.higher
        ):
            if higher:
                if c_score > curr_score:
                    better_in_any = True
                elif c_score < curr_score:
                    worse_in_any = True
            else:
                if c_score < curr_score:
                    better_in_any = True
                elif c_score > curr_score:
                    worse_in_any = True

        # Dominates if better in at least one and not worse in any
        return better_in_any and not worse_in_any
```

### Custom Elite Selector

```python
from gigaevo.evolution.strategies.elite_selectors import EliteSelector

class NoveltyEliteSelector(EliteSelector):
    """Select programs that are behaviorally novel."""

    def __init__(self, behavior_keys: list[str], k_nearest: int = 5):
        self.behavior_keys = behavior_keys
        self.k_nearest = k_nearest

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        # 1. Compute novelty score for each program
        novelty_scores = []
        for program in programs:
            behavior = [program.metrics[k] for k in self.behavior_keys]

            # Find K nearest neighbors
            distances = []
            for other in programs:
                if other.id == program.id:
                    continue
                other_behavior = [other.metrics[k] for k in self.behavior_keys]
                dist = self._euclidean_distance(behavior, other_behavior)
                distances.append(dist)

            distances.sort()
            novelty = sum(distances[:self.k_nearest]) / self.k_nearest
            novelty_scores.append((program, novelty))

        # 2. Sort by novelty (descending)
        novelty_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Return top N most novel programs
        return [prog for prog, _score in novelty_scores[:total]]

    @staticmethod
    def _euclidean_distance(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
```

### Custom Mutant Router

```python
from gigaevo.evolution.strategies.mutant_router import MutantRouter

class BehaviorAwareMutantRouter(MutantRouter):
    """Route mutants based on their behavior characteristics."""

    async def route_mutant(
        self,
        program: Program,
        candidate_islands: list[Island]
    ) -> Island | None:
        # 1. Compute behavior distance to each island's archive
        best_island = None
        min_distance = float('inf')

        for island in candidate_islands:
            # Get representative programs from island
            elites = await island.get_elites()
            if not elites:
                continue

            # Compute average distance
            distances = []
            for elite in elites[:10]:  # Sample for efficiency
                dist = self._behavior_distance(
                    program, elite, island.config.behavior_space.behavior_keys
                )
                distances.append(dist)

            avg_dist = sum(distances) / len(distances)

            # Route to island with most similar programs
            if avg_dist < min_distance:
                min_distance = avg_dist
                best_island = island

        return best_island

    @staticmethod
    def _behavior_distance(p1: Program, p2: Program, keys: list[str]) -> float:
        return sum(
            (p1.metrics[k] - p2.metrics[k]) ** 2 for k in keys
        ) ** 0.5
```

---

## Advanced Topics

### Dynamic Island Creation

Add islands at runtime based on discovered niches:

```python
class AdaptiveMultiIsland(MapElitesMultiIsland):
    async def add(self, program: Program) -> bool:
        # Check if program fits poorly in all islands
        best_fit_score = max(
            await self._compute_fit_score(program, island)
            for island in self.islands.values()
        )

        # If program doesn't fit well anywhere, create new island
        if best_fit_score < THRESHOLD:
            new_island = self._create_island_for_program(program)
            self.islands[new_island.id] = new_island
            logger.info(f"Created new island: {new_island.id}")

        # Route to best island
        return await super().add(program)
```

### Hierarchical Islands

Organize islands in a hierarchy:

```
Root Island (coarse behavior space)
    ↓
    ├── Sub-Island 1 (fine-grained: high fitness region)
    ├── Sub-Island 2 (fine-grained: low complexity region)
    └── Sub-Island 3 (fine-grained: fast runtime region)
```

### Co-Evolution

Evolve multiple species that interact:

```python
# Species 1: Solutions
solutions_island = Island(
    behavior_space=BehaviorSpace(keys=["fitness", "novelty"]),
    ...
)

# Species 2: Test cases
testcases_island = Island(
    behavior_space=BehaviorSpace(keys=["difficulty", "coverage"]),
    ...
)

# Fitness evaluation considers both species
```

### Meta-Evolution

Evolve the evolution strategy parameters:

```python
# Use a meta-island to optimize:
# - Island sizes
# - Migration rates
# - Selection pressures
# - Binning resolutions

meta_island = Island(
    behavior_space=BehaviorSpace(
        keys=["convergence_speed", "final_fitness"]
    ),
    # Each program is a set of evolution parameters
)
```

---

## Troubleshooting

### Common Issues

#### 1. Island Not Accepting Programs

```
[Island] fitness_island: added 0/100 programs
```

**Causes:**
- Missing behavior space dimensions in program metrics
- Incorrect bounds (all programs map to same cell)
- Archive selector rejecting all programs

**Solutions:**
```python
# Check program has required metrics
missing = set(island.behavior_space.behavior_keys) - program.metrics.keys()
if missing:
    print(f"Missing metrics: {missing}")

# Verify bounds are reasonable
print(island.behavior_space.feature_bounds)

# Test archive selector
print(island.archive_selector.is_better(new_program, current_elite))
```

#### 2. All Programs in Same Cell

```
[Island] 95% of programs in cell (10, 2)
```

**Causes:**
- Resolution too coarse
- Bounds too wide/narrow
- Programs not diverse enough

**Solutions:**
```yaml
# Increase resolution
resolution: [50, 20]  # instead of [10, 5]

# Adjust bounds to match actual value range
bounds: [[70.0, 90.0], ...]  # instead of [[0.0, 100.0], ...]

# Use different binning type
binning_types: [LOGARITHMIC, LINEAR]
```

#### 3. No Migration Happening

```
[Migration] 0 migrants transferred
```

**Causes:**
- `enable_migration=false`
- `migration_interval` not reached
- No programs selected by migrant_selector
- Migrants rejected by destination islands

**Solutions:**
```yaml
enable_migration: true
migration_interval: 50  # Check if N generations passed
max_migrants_per_island: 5  # Ensure > 0
```

```python
# Debug migrant selection
migrants = await island.select_migrants(5)
print(f"Selected {len(migrants)} migrants")

# Debug routing
destination = await router.route_mutant(migrant, dest_islands)
print(f"Routed to: {destination.id if destination else 'None'}")
```

#### 4. Island Size Explosion

```
[Island] fitness_island: 5000 programs (max_size=100)
```

**Causes:**
- `max_size=None` (unlimited)
- `archive_remover=None`
- Remover not being called

**Solutions:**
```yaml
islands:
  - max_size: 100  # Set limit
    archive_remover:  # Must specify remover
      _target_: ...FitnessArchiveRemover
```

#### 5. Poor Diversity

```
[Island] All programs have similarity > 0.95
```

**Causes:**
- Selection pressure too high
- Mutation too conservative
- Behavior space not capturing diversity

**Solutions:**
```python
# Use less greedy elite selector
elite_selector = RandomEliteSelector()  # Instead of FitnessProportional

# Expand behavior space
behavior_space = BehaviorSpace(
    keys=["fitness", "validity", "complexity", "novelty"],
    ...
)

# Increase mutation strength (in mutation operator)
```

### Debugging Tips

#### Visualize Behavior Space

```python
import matplotlib.pyplot as plt
import numpy as np

# Get all elites
elites = await island.get_elites()

# Extract behavior values
fitness_vals = [p.metrics["fitness"] for p in elites]
validity_vals = [p.metrics["validity"] for p in elites]

# Plot
plt.scatter(fitness_vals, validity_vals, alpha=0.5)
plt.xlabel("Fitness")
plt.ylabel("Validity")
plt.title(f"Island: {island.id} ({len(elites)} elites)")
plt.savefig("behavior_space.png")
```

#### Monitor Island Metrics

```python
metrics = await strategy.get_metrics()
print(metrics.to_dict())

# Output:
# {
#   'total_programs': 150,
#   'active_populations': 2,
#   'programs_per_population': 75.0,
#   'generation': 100,
#   'size/fitness_island': 80,
#   'size/simplicity_island': 70,
#   ...
# }
```

#### Track Migration History

```python
# In logs, search for:
# [Migration] Island X → Island Y: program ABC
# [Migration] Removed 5 programs, added 3 programs
```

---

## Best Practices

### Island Design

✅ **DO:**
- Use 2-4 islands maximum (complexity vs. benefit)
- Choose orthogonal behavior dimensions (minimize overlap)
- Set reasonable max_size (10-100 per island)
- Use different optimization criteria per island

❌ **DON'T:**
- Create too many islands (overhead increases)
- Use identical behavior spaces (redundant)
- Set max_size too large (memory issues)
- Forget to configure archive_remover when using max_size

### Behavior Space Design

✅ **DO:**
- Include 2-4 dimensions (sweet spot for diversity)
- Use metrics that actually characterize programs
- Set bounds based on observed value ranges
- Experiment with binning types

❌ **DON'T:**
- Use too many dimensions (curse of dimensionality)
- Include irrelevant metrics
- Use arbitrary bounds (e.g., [0, 1000] when values are [0.1, 0.9])
- Stick with LINEAR if data is non-uniform

### Selection Pressure

✅ **DO:**
- Balance exploitation (FitnessProportional) and exploration (Random)
- Use Tournament selection as a middle ground
- Adjust selection strength based on problem difficulty
- Monitor diversity metrics

❌ **DON'T:**
- Use purely random selection (too slow)
- Use purely greedy selection (premature convergence)
- Keep selection parameters fixed throughout evolution

### Migration Strategy

✅ **DO:**
- Start with infrequent migration (50-100 generations)
- Migrate small batches (3-5 programs)
- Monitor migration success rate
- Adjust based on island stagnation

❌ **DON'T:**
- Migrate every generation (too disruptive)
- Migrate large batches (homogenization)
- Enable migration with single island (no-op)

---

## Summary

GigaEvo's MAP-Elites Multi-Island system provides:

- ✅ **Quality-Diversity**: Maintains diverse, high-performing solutions
- ✅ **Multi-Objective**: Each island optimizes different criteria
- ✅ **Parallelism**: Islands evolve independently
- ✅ **Robustness**: Migration shares successful strategies
- ✅ **Scalability**: Easy to add new islands/objectives
- ✅ **Flexibility**: Customizable selectors, routers, removers

It powers GigaEvo's evolutionary search by exploring the full solution space while maintaining elite solutions across multiple dimensions.

---

**For more information:**
- Algorithm configs: `config/algorithm/`
- Strategy implementations: `gigaevo/evolution/strategies/`
- Evolution engine: `gigaevo/evolution/engine/core.py`
- Example experiments: `config/experiment/`
