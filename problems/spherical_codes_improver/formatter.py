"""Formatter for spherical_codes validation artifacts."""

from __future__ import annotations

from typing import Any

from gigaevo.programs.stages.formatter import FormatterStage


def format_spherical_feedback(data: dict[str, Any]) -> str:
    """Build compact spherical-codes feedback text from artifact data."""
    preview = data.get("feedback_preview")
    if isinstance(preview, str) and preview.strip():
        return preview

    if "error" in data:
        lines = [
            "## Spherical Codes Run Log",
            "",
            f"Crash: `{data.get('error', 'unknown error')}`",
        ]
        tb_tail = data.get("traceback_tail") or []
        if tb_tail:
            lines.append("")
            lines.append("Traceback tail:")
            lines.extend(f"- {line}" for line in tb_tail[-8:])
        return "\n".join(lines)

    summary = data.get("summary", {})
    final_best = data.get("final_best", {})
    baseline = data.get("baseline_max_cosine")
    success_steps = data.get("successful_step_indices", [])
    success_moments = data.get("success_moments", [])

    accepted = int(summary.get("stage_b_accepted", 0))
    rejected = int(summary.get("stage_b_rejected", 0))
    total_stage_b = accepted + rejected
    acceptance_rate = (100.0 * accepted / total_stage_b) if total_stage_b else 0.0

    final_best_cos = final_best.get("max_cosine")
    stage_b_delta: float | None = None
    if isinstance(baseline, (int, float)) and isinstance(final_best_cos, (int, float)):
        stage_b_delta = float(final_best_cos) - float(baseline)

    lines = [
        f"### Validation Results: N={data.get('n_points', '?')}, D={data.get('dimension', '?')}",
    ]
    lines.append("**Status:** ✅ Valid (No constraint violations)")
    if isinstance(final_best_cos, (int, float)):
        lines.append(f"**Final Max Cosine:** {float(final_best_cos):.5f}")
    lines.append("")
    lines.append("#### 📊 Execution Analysis")
    if isinstance(baseline, (int, float)):
        lines.append(
            f"*   **Stage A (improve only):** Achieved baseline max cosine of `{float(baseline):.5f}`."
        )
    if stage_b_delta is not None and stage_b_delta < 0:
        lines.append(
            f"*   **Stage B (perturb + improve):** Successfully reduced max cosine further by `{stage_b_delta:.5f}`."
        )
    elif stage_b_delta is not None:
        lines.append(
            "*   **Stage B (perturb + improve):** Did not improve beyond the Stage A baseline."
        )
    lines.append(
        f"*   **Perturbation Destructiveness:** `{acceptance_rate:.0f}%` Acceptance Rate ({accepted} accepted, {rejected} rejected)."
    )
    if success_steps:
        step_str = ", ".join(str(int(x)) for x in success_steps)
        lines.append(
            f"*   **Successful Intensity Scales:** Improvements were found at steps `{step_str}` of the refinement loop."
        )
    else:
        lines.append(
            "*   **Successful Intensity Scales:** No strict improvements in Stage B."
        )

    if success_moments:
        lines.append("")
        lines.append("#### ✅ Success Moments (latest)")
        for item in success_moments[-5:]:
            step = item.get("step", "?")
            cos = item.get("max_cosine")
            delta = item.get("delta_cosine")
            if isinstance(cos, (int, float)) and isinstance(delta, (int, float)):
                lines.append(
                    f"*   `{step}`: max cosine `{float(cos):.5f}` (delta `{float(delta):.5f}`)"
                )

    return "\n".join(lines)


class SphericalCodesArtifactFormatter(FormatterStage):
    """Format compact validator logs for mutation context."""

    def format_value(self, data: Any) -> str:
        if not data:
            return ""
        if not isinstance(data, dict):
            return f"## Spherical Codes Run Log\n\n{repr(data)}"
        return format_spherical_feedback(data)
