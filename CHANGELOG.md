## 0.6.0 (2025-09-20)

- Centralized metrics into OOP classes:
  - Added `MetricSpec` and `MetricsContext` with strict validation (exactly one primary; bounds; orientation).
  - Introduced `MetricsFormatter` to render metrics, deltas, and a metrics description block for prompts.
  - Added range-normalized delta option and significance markers; included orientation (↑/↓ better).
- Pipeline and stages refactor:
  - Replaced `FactoryMetricsStage` with `EnsureMetricsStage` (strict selection, coercion, clamping, Prometheus export).
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
