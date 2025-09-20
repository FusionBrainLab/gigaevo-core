# Metrics module overview

This module centralizes metric definitions and prompt formatting.

- MetricSpec: per-metric schema
  - key: unique name
  - description: short human description for prompts
  - decimals: numeric precision when rendering
  - is_primary: exactly one metric across specs must be primary
  - higher_is_better: orientation for selection/normalization
  - unit: optional unit string for display
  - lower_bound/upper_bound: used for behavior spaces and normalization
  - include_in_prompts: whether to show in prompts
  - significant_change: threshold to mark deltas with an asterisk

- MetricsContext: single source of truth
  - Built from problems/<name>/metrics.yaml via `from_dict`
  - `get_primary_spec()` → MetricSpec of the primary metric
  - `prompt_keys()` → ordered list of metric keys to render in prompts (respects `display_order` and `include_in_prompts`)
  - `get_bounds(key)` → (lower, upper) when available
  - `is_higher_better(key)` → bool orientation for a metric

- MetricsFormatter: rendering utilities for prompts
  - `format_metrics_block(metrics)` → bullet list with values, units, descriptions
  - `format_delta_block(parent, child, include_primary=False, use_range_normalization=False)`
    - prints deltas; can normalize by metric range when bounds are set

## Stage usage guidelines

- Stages must not hardcode metric descriptions or formatting; always use `MetricsContext` + `MetricsFormatter`.
- Validation/population belongs in `EnsureMetricsStage`, which enforces presence, numeric finiteness, and clamps to bounds.
- Behavior spaces, selection, and lineage should derive the primary metric and orientation from `MetricsContext`.

