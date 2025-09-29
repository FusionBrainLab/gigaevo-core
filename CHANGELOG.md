## 0.9.0 (2025-01-XX)

### DAG Builder GUI Enhancements
- **Unique Name Management**: Implemented automatic counter appending for duplicate stage names (e.g., `TestStage`, `TestStage_1`, `TestStage_2`) to prevent export errors
- **Real-time Validation**: Added client-side validation in Stage Editor to prevent duplicate custom names with visual feedback
- **Connection Validation**: Fixed port type compatibility - execution edges only snap to execution ports, data edges only to data input ports
- **Color Consistency**: Fixed color calculation across all components (StageNode, MiniMap, NodeDetails) to use original stage type instead of unique names
- **Event Handling**: Added click debouncing and removed double-click handlers to prevent accidental duplicate stage creation
- **Backend Validation**: Enhanced DAG export validation to check for unique stage names and prevent faulty graph exports

### Development Infrastructure
- **Node.js Support**: Added comprehensive `.gitignore` patterns for Node.js dependencies, build outputs, and cache files
- **Installation Guide**: Updated README with detailed setup instructions for both Python backend and React frontend
- **Requirements Management**: Added `requirements.txt` for Python dependencies and improved startup scripts

### Technical Improvements
- **State Management**: Improved React state handling to prevent race conditions and duplicate node creation
- **Port System**: Enhanced connection validation with proper source/target handle checking
- **Export System**: Fixed edge mapping to use unique names for proper DAG structure validation

Notes:
- All duplicate stage instances now maintain consistent colors based on original stage type
- Connection system enforces proper port type compatibility with visual feedback
- Backend validation prevents export of DAGs with duplicate stage identifiers

## 0.7.0 (2025-09-20)

- Problem scaffolding and layout standardization:
  - Added `ProblemLayout` with centralized filenames/dirs and `scaffold()` utility
  - Introduced `ProblemContext` (moved to `src/problems/context.py`) and integrated into `run.py`
  - Added `tools/wizard.py` CLI to scaffold new problems with optional custom texts
  - Prompts scaffolding now includes all placeholders: `{task_definition}`, `{task_hints}`, `{metrics_description}`, `{count}`, `{parent_blocks}`
- Initial population loaders (OOP):
  - `DirectoryProgramLoader` and `RedisTopProgramsLoader` for clean, configurable population sourcing
  - Redis loader params (connections, timeouts) are user-configurable
- Cleanups:
  - Removed hardcoded paths/strings in favor of `ProblemLayout`
  - Deprecated inlined helpers in `run.py` to prevent drift

## 0.6.0 (2025-09-20)
- Centralized metrics into OOP classes:
  - Added `MetricSpec` and `MetricsContext` with strict validation (exactly one primary; bounds; orientation).
  - Introduced `MetricsFormatter` to render metrics, deltas, and a metrics description block for prompts.
  - Added range-normalized delta option and significance markers; included orientation (↑/↓ better).
- Pipeline and stages refactor:
  - Replaced `FactoryMetricsStage` with `EnsureMetricsStage` (strict selection, coercion, clamping).
  - Added `NormalizeMetricsStage` (not enabled by default) for 0–1 normalization and aggregate scoring.
  - Updated `GenerateLLMInsightsStage` and `GenerateLineageInsightsStage` to use `MetricsContext` and `MetricsFormatter`.
- Configuration & UX:
  - Externalized metrics to `problems/*/metrics.yaml`; removed CLI bounds; templates created per problem.
  - Injected an AVAILABLE METRICS section into mutation system prompts via `{metrics_description}`.
  - `LLMMutationOperator` now accepts a shared `MetricsFormatter`; removed ad-hoc helpers; using modern typing.
- Code quality & style:
  - Modern typing (list/dict), clearer helpers, compact logic, documented methods, and better logging.

Notes:
- Invalid programs’ fitness is clamped to primary lower_bound, keeping them in the lowest bucket by design.
- Normalization stage exists but is not wired into the default DAG.
