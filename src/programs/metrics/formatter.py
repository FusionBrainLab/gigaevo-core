from __future__ import annotations

from typing import Any

from .context import MetricsContext


class MetricsFormatter:
    """Render metrics consistently for LLM prompts.

    Examples:
    - Metrics block (for keys in context.prompt_keys()):
      - fitness : 0.12345 (Main objective; ↑ better)
      - is_valid : 1 (Validity flag; ↑ better)

    - Delta block (child - parent):
      - fitness (Main objective; ↑ better) +0.01234 (9.8%)
      - is_valid (Validity flag; ↑ better) +0.00000
    """

    def __init__(self, context: MetricsContext, use_range_normalization: bool = False):
        self.context = context
        self.use_range_normalization = use_range_normalization

    def format_metrics_block(self, metrics: dict[str, float]) -> str:
        lines: list[str] = []
        for key in self.context.prompt_keys():
            spec = self.context.specs[key]
            decimals = spec.decimals
            desc = spec.description or ""
            unit = spec.unit or ""
            orient = "↑" if self.context.is_higher_better(key) else "↓"
            value = metrics[key]
            unit_str = f" {unit}" if unit else ""
            lines.append(f"- {key} : {value:.{decimals}f}{unit_str} ({desc}; {orient} better)")
        return "\n".join(lines)

    def format_delta_block(
        self,
        *,
        parent: dict[str, float],
        child: dict[str, float],
        include_primary: bool = False,
        use_range_normalization: bool | None = None,
    ) -> str:
        use_range_normalization = (
            self.use_range_normalization if use_range_normalization is None else use_range_normalization
        )
        lines: list[str] = []
        primary_key = self.context.get_primary_key()
        for key in self.context.prompt_keys():
            if not include_primary and key == primary_key:
                continue
            spec = self.context.specs[key]
            decimals = spec.decimals
            desc = spec.description or ""
            unit = spec.unit or ""
            signif = spec.significant_change or 0.0
            orient = "↑" if self.context.is_higher_better(key) else "↓"
            p = parent[key]
            c = child[key]
            delta = c - p
            unit_str = f" {unit}" if unit else ""
            percent = (100.0 * delta / abs(p)) if abs(p) > 1e-12 else None
            if use_range_normalization:
                bounds = self.context.get_bounds(key)
                if bounds is not None:
                    lo, hi = bounds
                    if lo is not None and hi is not None and hi > lo:
                        rng = hi - lo
                        delta = delta / rng
                        unit_str = ""  # range-normalized is unitless
                        percent = None
            mark = " *" if signif and abs(delta) >= signif else ""
            if percent is None:
                lines.append(f"- {key} ({desc}; {orient} better) {delta:+.{decimals}f}{unit_str}{mark}")
            else:
                lines.append(f"- {key} ({desc}; {orient} better) {delta:+.{decimals}f}{unit_str} ({percent:+.1f}%)" + mark)
        return "\n".join(lines) if lines else "N/A"

    def format_metrics_description(self) -> str:
        """Build a concise overview of available metrics from context.

        Example block:
        - fitness: Main objective (↑ better; [0.0, 1.0] range; unit="")
        - is_valid: Whether program is valid (↑ better; [0.0, 1.0] range; unit="")
        """
        ordered_keys = self.context.prompt_keys()
        primary_key = self.context.get_primary_key()
        keys: list[str] = [primary_key] + [k for k in ordered_keys if k != primary_key]
        lines: list[str] = []
        for key in keys:
            spec = self.context.specs[key]
            orient = "↑" if self.context.is_higher_better(key) else "↓"
            parts: list[str] = [f"{orient} better"]
            bounds = self.context.get_bounds(key)
            if bounds is not None and bounds[0] is not None and bounds[1] is not None:
                parts.append(f"[{bounds[0]}, {bounds[1]}] range")
            if spec.unit:
                parts.append(f"unit=\"{spec.unit}\"")
            lines.append(f"- {key}: {spec.description} (" + "; ".join(parts) + ")")
        return "\n".join(lines)


