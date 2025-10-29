from __future__ import annotations

from src.programs.metrics.context import VALIDITY_KEY, MetricsContext


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

    def __init__(self, context: MetricsContext):
        self.context = context

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
            is_sentinel_value = spec.is_sentinel(value)
            sentinel = (
                " [sentinel]" if is_sentinel_value and key != VALIDITY_KEY else ""
            )
            lines.append(
                f"- {key} : {value:.{decimals}f}{unit_str} ({desc}; {orient} better){sentinel}"
            )
        return "\n".join(lines)

    def format_delta_block(
        self,
        *,
        parent: dict[str, float],
        child: dict[str, float],
        include_primary: bool = False,
        style: str = "table",
    ) -> str:
        ctx = self.context
        primary = ctx.get_primary_key()
        keys = [k for k in ctx.prompt_keys() if include_primary or k != primary]

        def u(unit: str) -> str:
            return f" {unit}" if unit else ""

        rows = []
        for key in keys:
            spec = ctx.specs[key]
            decimals = spec.decimals
            unit = spec.unit or ""
            desc = spec.description or ""
            signif = float(spec.significant_change or 0.0)
            higher_better = ctx.is_higher_better(key)

            p = float(parent[key])
            c = float(child[key])
            d = c - p
            pct = (100.0 * d / abs(p)) if abs(p) > 1e-12 else None
            orient = "↑" if higher_better else "↓"
            impact = (
                "improved"
                if (d > 0) == higher_better
                else ("no change" if d == 0 else "worsened")
            )
            sig = "★" if signif and abs(d) >= signif else ""

            parent_s = f"{p:.{decimals}f}{u(unit)}"
            child_s = f"{c:.{decimals}f}{u(unit)}"
            delta_s = f"{d:+.{decimals}f}{u(unit)}"
            extra_pct_sign = ""
            if pct is not None and pct > 100:
                extra_pct_sign = ">"
            elif pct is not None and pct < -100:
                extra_pct_sign = "<"
            if pct is not None:
                pct = min(max(pct, -100.0), 100.0)
            pct_s = f"{extra_pct_sign}{pct:+.1f}%" if pct is not None else "—"

            rows.append(
                (
                    key,
                    desc,
                    f"{orient} better",
                    parent_s,
                    child_s,
                    delta_s,
                    pct_s,
                    f"{impact} {sig}".strip(),
                )
            )

        if not rows:
            return "N/A"

        if style == "bullets":
            return "\n".join(
                f"- {k} ({dirn}) | {p} → {c} | Δ {dl}"
                f"{f' ({pc})' if pc != '—' else ''} | {imp}"
                for k, _, dirn, p, c, dl, pc, imp in rows
            )

        header = (
            "| metric | direction | parent | child | Δ | %Δ | impact |\n"
            "|---|:--:|---:|---:|---:|---:|:--:|"
        )
        body = "\n".join(
            f"| {k} | {dirn} | {p} | {c} | {dl} | {pc} | {imp} |"
            for k, _, dirn, p, c, dl, pc, imp in rows
        )
        return f"{header}\n{body}"

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
                parts.append(f'unit="{spec.unit}"')
            lines.append(f"- {key}: {spec.description} (" + "; ".join(parts) + ")")
        return "\n".join(lines)
