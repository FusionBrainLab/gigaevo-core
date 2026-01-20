# GigaEvo DAG System

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Stage Lifecycle](#stage-lifecycle)
- [Building DAGs](#building-dags)
- [Creating Custom Stages](#creating-custom-stages)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **DAG (Directed Acyclic Graph) System** is GigaEvo's core execution engine for processing evolved programs. It orchestrates complex, multi-stage computations where each stage can depend on outputs from previous stages, ensuring:

- **Type Safety**: Compile-time validation of data flow between stages
- **Parallelism**: Concurrent execution of independent stages
- **Cacheability**: Automatic result reuse across runs
- **Fault Tolerance**: Graceful handling of stage failures with detailed diagnostics
- **Flexibility**: Declarative pipeline definition via Hydra configs

### Why DAGs?

Evolutionary computation requires sophisticated program evaluation:
1. **Code Execution** - Run the evolved program safely
2. **Validation** - Check outputs meet problem constraints
3. **Metrics Collection** - Gather multiple performance indicators
4. **Analysis** - Generate insights for LLM-based mutation
5. **Lineage Tracking** - Build evolutionary family trees

A DAG naturally expresses these dependencies while maximizing parallelism.

---

## Core Concepts

### 1. Stage

A **Stage** is an atomic unit of computation. Each stage:

```python
class MyStage(Stage):
    # Type-safe input specification
    InputsModel = MyInputs   # Pydantic model defining inputs
    OutputModel = MyOutput   # Pydantic model defining output

    # Cacheability: can results be reused across runs?
    cacheable = True

    async def compute(self, program: Program) -> MyOutput:
        # Your logic here
        result = do_something(self.params.input_field)
        return MyOutput(data=result)
```

**Key Properties:**
- **Type-Safe**: `InputsModel` and `OutputModel` define contracts
- **Async**: All stages run asynchronously for efficiency
- **Timeout**: Each stage has a configurable execution timeout
- **Cacheable**: Results can be reused if stage is deterministic

### 2. Data Flow Edges

**DataFlowEdge** connects stages by wiring outputs to inputs:

```python
DataFlowEdge(
    source_stage="StageA",
    destination_stage="StageB",
    input_name="my_input"  # Name in StageB.InputsModel
)
```

The DAG system validates:
- ✅ Type compatibility: `StageA.OutputModel` matches `StageB.InputsModel.my_input`
- ✅ Required inputs: All non-optional inputs have providers
- ✅ No duplicate inputs: Each input name receives data from exactly one source

### 3. Execution Order Dependencies

**ExecutionOrderDependency** enforces ordering without data transfer:

```python
# StageB runs only after StageA completes successfully
ExecutionOrderDependency.on_success("StageA")

# StageC runs after StageB fails
ExecutionOrderDependency.on_failure("StageB")

# StageD runs after StageC finishes (any outcome)
ExecutionOrderDependency.always_after("StageC")
```

Use cases:
- Ensure validation before execution
- Conditional branching based on success/failure
- Sequential ordering for side-effect stages

### 4. Cacheability

Stages can be **cacheable** or **non-cacheable**:

| Cacheable | Behavior |
|-----------|----------|
| `True` | Results persist across runs; stage skipped if valid cached result exists |
| `False` | Must re-execute every run; results only valid within current run |

**Rules:**
- ❌ Cacheable stage cannot depend on non-cacheable stage
- ✅ Non-cacheable stage can depend on cacheable stage
- 💡 Use `cacheable=False` for time-dependent or stateful stages

---

## Architecture

### Component Overview

```
┌─────────────┐
│ DagRunner   │  ← High-level orchestrator
└──────┬──────┘
       │ manages
       ↓
┌─────────────┐
│ DAG         │  ← Per-program execution instance
└──────┬──────┘
       │ uses
       ↓
┌─────────────┐
│ DAGAutomata │  ← Scheduling & validation logic
└──────┬──────┘
       │ operates on
       ↓
┌─────────────┐
│ Stage(s)    │  ← Individual computation units
└─────────────┘
```

### DAGAutomata

The **automaton** (finite state machine) manages stage execution logic:

**Responsibilities:**
1. **Validation** (at build time):
   - Type compatibility between connected stages
   - DAG acyclicity (no circular dependencies)
   - Input coverage (all required inputs have providers)
   - Cacheability constraints

2. **Scheduling** (at runtime):
   - Determine which stages are **ready** to run
   - Identify stages to **auto-skip** (impossible dependencies)
   - Detect **deadlocks** and **stalls**
   - Build input dictionaries for ready stages

3. **Gate States**:
   - `READY`: All dependencies satisfied, can execute
   - `WAIT`: Dependencies pending, cannot execute yet
   - `IMPOSSIBLE`: Dependencies failed/contradicted, auto-skip

### DAG (Execution Engine)

The **DAG** runs a single program through the pipeline:

**Execution Flow:**
```
1. Initialize all stages to PENDING
2. Loop until termination:
   a. Identify stages to auto-skip (impossible deps)
   b. Get ready stages from automata
   c. Launch ready stages (respecting max_parallel_stages)
   d. Collect completed stages
   e. Update program state in Redis
   f. Check for stalls/deadlocks
3. Persist final program state
```

**Termination Conditions:**
- ✅ All stages finalized (COMPLETED/FAILED/SKIPPED)
- ❌ Timeout exceeded (configurable per-DAG)
- ❌ Deadlock detected (no progress possible)

### DagRunner

The **DagRunner** manages multiple concurrent DAGs:

**Features:**
- **Polling Loop**: Continuously checks Redis for programs in "runnable" state
- **Concurrency Control**: Enforces `max_concurrent_dags` limit
- **Metrics Collection**: Tracks success rates, throughput, errors
- **Graceful Shutdown**: Awaits active DAGs before stopping

---

## Stage Lifecycle

### State Transitions

```
PENDING → RUNNING → COMPLETED ✓
                 ├→ FAILED     ✗
                 ├→ CANCELLED  ⊗
                 └→ SKIPPED    ⊘
```

**States:**
- `PENDING`: Waiting for dependencies
- `RUNNING`: Currently executing
- `COMPLETED`: Finished successfully, output available
- `FAILED`: Exception occurred, error details captured
- `CANCELLED`: Task cancelled (e.g., timeout, shutdown)
- `SKIPPED`: Auto-skipped due to impossible dependencies

### Execution Timeline

```python
# 1. Stage created (build time)
stage = MyStage(timeout=60.0)

# 2. Inputs attached (runtime, by DAG)
stage.attach_inputs({"input_field": upstream_output})

# 3. Validation (lazy, on first access)
inputs = stage.params  # Triggers Pydantic validation

# 4. Execution (async)
result = await stage.execute(program)

# 5. Result stored in program.stage_results[stage_name]
program.stage_results["MyStage"] = result
```

### Input/Output Flow

```
┌─────────────┐
│ StageA      │
│ Output: X   │
└──────┬──────┘
       │ DataFlowEdge(source="StageA", dest="StageB", input_name="x")
       ↓
┌─────────────┐
│ StageB      │
│ Input: x    │ ← stage.params.x receives X
│ Output: Y   │
└──────┬──────┘
       │ DataFlowEdge(source="StageB", dest="StageC", input_name="y")
       ↓
┌─────────────┐
│ StageC      │
│ Input: y    │ ← stage.params.y receives Y
└─────────────┘
```

---

## Building DAGs

### Method 1: Hydra Configuration (Recommended)

Define stages and edges declaratively:

```yaml
# config/pipeline/my_pipeline.yaml
dag_blueprint:
  _target_: gigaevo.runner.dag_blueprint.DAGBlueprint

  # Stage factories
  nodes:
    ValidateCode:
      _target_: gigaevo.programs.stages.validation.ValidateCodeStage
      _partial_: true
      timeout: 30.0

    Execute:
      _target_: gigaevo.programs.stages.python_executors.execution.CallProgramFunction
      _partial_: true
      function_name: entrypoint
      timeout: 60.0

    CollectMetrics:
      _target_: gigaevo.programs.stages.metrics.EnsureMetricsStage
      _partial_: true
      metrics_context: ${metrics_context}
      timeout: 10.0

  # Data flow: stage outputs → stage inputs
  data_flow_edges:
    - source_stage: Execute
      destination_stage: CollectMetrics
      input_name: candidate

  # Execution ordering: validate before execute
  exec_order_deps:
    Execute:
      - stage_name: ValidateCode
        condition: success

  max_parallel_stages: 4
  dag_timeout: 300.0
```

### Method 2: Programmatic (Advanced)

```python
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.dag.dag import DAG

# Define stages
stages = {
    "stage_a": StageA(timeout=30.0),
    "stage_b": StageB(timeout=60.0),
    "stage_c": StageC(timeout=45.0),
}

# Define data flow
edges = [
    DataFlowEdge(source_stage="stage_a", destination_stage="stage_b", input_name="input_from_a"),
    DataFlowEdge(source_stage="stage_b", destination_stage="stage_c", input_name="input_from_b"),
]

# Optional: execution order constraints
exec_deps = {
    "stage_b": [ExecutionOrderDependency.on_success("stage_a")],
}

# Build and run
dag = DAG(
    nodes=stages,
    data_flow_edges=edges,
    execution_order_deps=exec_deps,
    state_manager=state_manager,
    max_parallel_stages=4,
    dag_timeout=600.0,
    writer=log_writer,
)

await dag.run(program)
```

---

## Creating Custom Stages

### Basic Stage Template

```python
from gigaevo.programs.core_types import StageIO
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.program import Program

# 1. Define input model
class MyInputs(StageIO):
    """Inputs this stage requires."""
    required_field: str
    optional_field: int | None = None

# 2. Define output model
class MyOutput(StageIO):
    """Data this stage produces."""
    result: float
    metadata: dict[str, str]

# 3. Implement stage
class MyCustomStage(Stage):
    InputsModel = MyInputs
    OutputModel = MyOutput
    cacheable = True  # Results can be reused

    def __init__(self, *, my_config: str, **kwargs):
        super().__init__(**kwargs)
        self.my_config = my_config

    async def compute(self, program: Program) -> MyOutput:
        # Access validated inputs
        required = self.params.required_field
        optional = self.params.optional_field

        # Your logic
        result = await expensive_computation(required, optional)

        # Return typed output
        return MyOutput(
            result=result,
            metadata={"source": self.my_config}
        )
```

### Stage Patterns

#### Pattern 1: Void Input (No Dependencies)

```python
from gigaevo.programs.core_types import VoidInput

class IndependentStage(Stage):
    InputsModel = VoidInput  # No inputs needed
    OutputModel = MyOutput

    async def compute(self, program: Program) -> MyOutput:
        # Stage runs without dependencies
        data = analyze_program_code(program.code)
        return MyOutput(data=data)
```

#### Pattern 2: Void Output (Side-Effect Only)

```python
from gigaevo.programs.core_types import VoidOutput

class LoggingStage(Stage):
    InputsModel = MyInputs
    OutputModel = VoidOutput  # No output for downstream

    async def compute(self, program: Program) -> None:
        # Perform side effect
        log_to_database(self.params.data)
        return None  # VoidOutput allows None
```

#### Pattern 3: Early Failure Detection

```python
from gigaevo.programs.core_types import ProgramStageResult, StageError

class ValidationStage(Stage):
    InputsModel = MyInputs
    OutputModel = MyOutput

    async def compute(self, program: Program) -> ProgramStageResult | MyOutput:
        # Check preconditions
        if not self.is_valid(self.params.data):
            # Return explicit failure (won't propagate as exception)
            return ProgramStageResult.failure(
                error=StageError(
                    type="ValidationError",
                    message="Data failed validation checks",
                    stage=self.stage_name
                )
            )

        # Normal flow
        return MyOutput(result=process(self.params.data))
```

#### Pattern 4: Optional Inputs

```python
class FlexibleStage(Stage):
    class InputsModel(StageIO):
        required: str
        optional_a: int | None = None  # Optional inputs
        optional_b: list[str] | None = None

    OutputModel = MyOutput

    async def compute(self, program: Program) -> MyOutput:
        result = self.params.required

        # Check if optional input was provided
        if self.params.optional_a is not None:
            result = enhance_with_a(result, self.params.optional_a)

        if self.params.optional_b:
            result = enhance_with_b(result, self.params.optional_b)

        return MyOutput(data=result)
```

### Registering Stages

Use `@StageRegistry.register()` for discoverability:

```python
from gigaevo.programs.stages.stage_registry import StageRegistry

@StageRegistry.register(description="Analyzes code complexity")
class ComplexityStage(Stage):
    # ... implementation
```

---

## Configuration

### Pipeline Configuration Structure

```yaml
# Complete pipeline specification
dag_blueprint:
  _target_: gigaevo.runner.dag_blueprint.DAGBlueprint

  # 1. Stage Definitions
  nodes:
    StageName:
      _target_: module.path.StageClass
      _partial_: true        # Create factory, not instance
      timeout: 60.0          # Stage-specific timeout
      custom_param: value    # Stage constructor args

  # 2. Data Flow (Required)
  data_flow_edges:
    - source_stage: ProducerStage
      destination_stage: ConsumerStage
      input_name: field_name_in_consumer

  # 3. Execution Order (Optional)
  exec_order_deps:
    DependentStage:
      - stage_name: PrerequisiteStage
        condition: success  # or failure, always

  # 4. DAG-level Settings
  max_parallel_stages: 8   # Concurrent stage limit
  dag_timeout: 3600.0      # Total DAG execution timeout
```

### Common Stage Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `timeout` | float | Max execution time (seconds) |
| `_partial_` | bool | Create factory instead of instance (Hydra) |
| `cacheable` | bool | Enable result caching (class-level) |

### Example: Multi-Stage Pipeline

```yaml
dag_blueprint:
  _target_: gigaevo.runner.dag_blueprint.DAGBlueprint

  nodes:
    # 1. Validate code syntax
    ValidateCode:
      _target_: gigaevo.programs.stages.validation.ValidateCodeStage
      _partial_: true
      timeout: 10.0
      safe_mode: true

    # 2. Execute user function
    ExecuteProgram:
      _target_: gigaevo.programs.stages.python_executors.execution.CallProgramFunction
      _partial_: true
      function_name: entrypoint
      timeout: 120.0

    # 3. Validate output
    ValidateOutput:
      _target_: gigaevo.programs.stages.python_executors.execution.CallValidatorFunction
      _partial_: true
      path: ${problem.dir}/validate.py
      timeout: 30.0

    # 4. Compute complexity
    Complexity:
      _target_: gigaevo.programs.stages.complexity.ComputeComplexityStage
      _partial_: true
      timeout: 15.0

    # 5. Merge metrics
    MergeMetrics:
      _target_: gigaevo.programs.stages.json_processing.MergeDictStage
      _partial_: true
      timeout: 5.0

    # 6. Generate insights (LLM-powered)
    Insights:
      _target_: gigaevo.programs.stages.insights.InsightsStage
      _partial_: true
      llm: ${ref:llm}
      timeout: 60.0

  data_flow_edges:
    # ExecuteProgram → ValidateOutput
    - source_stage: ExecuteProgram
      destination_stage: ValidateOutput
      input_name: payload

    # ValidateOutput metrics → MergeMetrics
    - source_stage: ValidateOutput
      destination_stage: MergeMetrics
      input_name: first

    # Complexity metrics → MergeMetrics
    - source_stage: Complexity
      destination_stage: MergeMetrics
      input_name: second

    # Merged metrics → Insights
    - source_stage: MergeMetrics
      destination_stage: Insights
      input_name: metrics

  exec_order_deps:
    # Execute only after validation succeeds
    ExecuteProgram:
      - stage_name: ValidateCode
        condition: success

    # Generate insights after execution completes (even if failed)
    Insights:
      - stage_name: ExecuteProgram
        condition: always

  max_parallel_stages: 4
  dag_timeout: 600.0
```

**Execution Order** (with parallelism):
```
Time →
  0s: ValidateCode, Complexity  (parallel)
 10s: ExecuteProgram  (waits for ValidateCode)
130s: ValidateOutput  (uses ExecuteProgram output)
160s: MergeMetrics    (waits for both ValidateOutput + Complexity)
165s: Insights        (uses MergeMetrics output)
```

---

## Advanced Features

### 1. Automatic Skip Logic

When a stage's dependencies become **impossible** to satisfy, the DAG automatically skips it:

**Scenario:**
```
StageA (cacheable) → FAILED historically
StageB depends on StageA (data flow)
```

**Result:**
- StageB is **auto-skipped** (impossible to get input from StageA)
- Downstream stages depending on StageB are also skipped
- DAG continues executing independent branches

### 2. Stall Detection

The DAG monitors for **stalls** (no progress despite pending work):

```python
# If no progress for stall_grace_seconds (default: 30s)
logger.warning("[DAG] STALLED - Diagnostics:\n{blockers}")
```

**Blocker diagnostics include:**
- Which stages are blocked
- Why they're blocked (missing dependencies, waiting for completion)
- Status of upstream stages

### 3. Deadlock Detection

The DAG detects impossible situations:

```python
# Deadlock: stages to skip but cannot (not in PENDING state)
# OR: no ready stages, nothing running, but work remains
raise RuntimeError("DEADLOCK: {explanation}")
```

### 4. Type System Integration

The DAG validates types at **build time**:

```python
# GOOD: Compatible types
class StageA(Stage):
    OutputModel = FloatDictContainer  # Dict[str, float]

class StageB(Stage):
    class InputsModel(StageIO):
        data: FloatDictContainer  # ✓ Exact match
```

```python
# BAD: Incompatible types (caught at build)
class StageA(Stage):
    OutputModel = FloatDictContainer  # Dict[str, float]

class StageB(Stage):
    class InputsModel(StageIO):
        data: StringContainer  # ✗ Type mismatch!
# → ValueError: Type mismatch for edge ...
```

**Covariance Support:**
```python
# Generic type covariance works
class StageA(Stage):
    OutputModel = Box[MyData]  # Box[T]

class StageB(Stage):
    class InputsModel(StageIO):
        data: Box[MyData]  # ✓ Box[T] matches Box[T]
```

### 5. Cacheability Propagation

**Rule**: Cacheable stages cannot depend on non-cacheable stages.

```python
# VALID
CacheableStage (cacheable=True)
  → depends on →
CacheableStage (cacheable=True)

# VALID
NonCacheableStage (cacheable=False)
  → depends on →
CacheableStage (cacheable=True)

# INVALID (caught at build time)
CacheableStage (cacheable=True)
  → depends on →
NonCacheableStage (cacheable=False)
# → ValueError: Cacheability violation
```

### 6. Metrics & Monitoring

The DagRunner exposes real-time metrics:

```python
metrics = runner.metrics()
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Avg iterations/sec: {metrics.average_iterations_per_second:.1f}")
print(f"Active DAGs: {len(runner._active)}")
```

Metrics include:
- `dag_runs_started` / `dag_runs_completed` / `dag_errors`
- `dag_timeouts` / `orphaned_programs_discarded`
- `dag_build_failures` / `state_update_failures`

---

## Troubleshooting

### Common Issues

#### 1. Type Mismatch Error

```
ValueError: Type mismatch for edge SourceStage -> DestStage.input_name:
producer=OutputType not compatible with InputType
```

**Solution:** Ensure output type matches input annotation:
```python
# Producer
class SourceStage(Stage):
    OutputModel = MyOutput  # Must match

# Consumer
class DestStage(Stage):
    class InputsModel(StageIO):
        input_name: MyOutput  # Must match
```

#### 2. Missing Required Input

```
ValueError: Topology error: stage 'MyStage' is missing providers
for mandatory inputs: ['required_field']
```

**Solution:** Add a DataFlowEdge to provide the missing input:
```yaml
data_flow_edges:
  - source_stage: ProducerStage
    destination_stage: MyStage
    input_name: required_field
```

#### 3. Circular Dependency

```
ValueError: Cycle detected in DAG: StageA -> StageB -> StageA
```

**Solution:** Remove the cycle by:
- Changing data flow direction
- Removing unnecessary dependencies
- Splitting stages to break the cycle

#### 4. Cacheability Violation

```
ValueError: Cacheability violation: cacheable 'StageB' depends on
non-cacheable 'StageA' via data-flow
```

**Solution:** Either:
- Make StageA cacheable: `cacheable = True`
- Make StageB non-cacheable: `cacheable = False`

#### 5. Stage Timeout

```
StageState.FAILED - error: Stage timed out after 60.0s
```

**Solution:** Increase stage timeout:
```yaml
MyStage:
  timeout: 120.0  # Double the timeout
```

#### 6. Stalled DAG

```
[DAG] STALLED (no progress for 30s). Diagnostics:
[Blocker] 'StageX': data: 'input_field' <- ProducerStage needs COMPLETED
```

**Solution:**
- Check ProducerStage logs for failures
- Verify ProducerStage dependencies are satisfied
- Increase `stall_grace_seconds` if stage is legitimately slow

### Debugging Tips

#### Inspect Stage Results

```python
# After DAG run
for stage_name, result in program.stage_results.items():
    print(f"{stage_name}: {result.status.name}")
    if result.error:
        print(f"  Error: {result.error.pretty(include_traceback=True)}")
```

#### Visualize DAG Structure

```python
import networkx as nx
import matplotlib.pyplot as plt

# Build graph from DAGAutomata
G = nx.DiGraph()
for edge in dag.automata.topology.edges:
    G.add_edge(edge.source_stage, edge.destination_stage)

nx.draw(G, with_labels=True)
plt.savefig("dag_structure.png")
```

#### Check Execution Timeline

```python
for stage_name, result in program.stage_results.items():
    if result.started_at and result.finished_at:
        duration = result.duration_seconds()
        print(f"{stage_name}: {duration:.2f}s")
```

---

## Best Practices

### 1. Stage Design

✅ **DO:**
- Keep stages focused (single responsibility)
- Use descriptive stage and input names
- Include docstrings for InputsModel/OutputModel
- Handle expected errors gracefully (return ProgramStageResult.failure)
- Use type hints consistently

❌ **DON'T:**
- Mix multiple concerns in one stage
- Use mutable global state
- Ignore timeout settings (default may be too short/long)
- Return None unless OutputModel is VoidOutput

## Summary

The GigaEvo DAG System provides:

- ✅ **Type-Safe** data flow between stages
- ✅ **Parallel** execution for performance
- ✅ **Flexible** dependency management (data + execution order)
- ✅ **Robust** error handling with detailed diagnostics
- ✅ **Cacheable** stage results for efficiency
- ✅ **Declarative** configuration via Hydra
- ✅ **Extensible** stage system for custom logic

It powers GigaEvo's evolutionary computation by orchestrating complex, multi-stage program evaluations at scale.

---

**For more information:**
- Example pipelines: `config/pipeline/`
- Stage implementations: `gigaevo/programs/stages/`
- DAG internals: `gigaevo/programs/dag/`
- DagRunner: `gigaevo/runner/dag_runner.py`
