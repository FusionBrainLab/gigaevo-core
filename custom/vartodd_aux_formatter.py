"""
Interpretation layer for vartodd aux info.

Parses the raw concatenated string from validate.py (report + best_path)
into structured, LLM-friendly sections. The raw data from Daniil's domain
code (helper.py, mcts_dao.py) is left untouched — this module only
reformats for presentation to the mutation LLM.
"""

from __future__ import annotations

import re


def format_vartodd_aux(raw: str) -> str:
    """Reformat raw vartodd aux info into LLM-readable sections."""
    sections = _split_sections(raw)
    parts: list[str] = []

    parts.append("## Decomposition Search Report")
    parts.append("")

    # 1. Result summary (extracted from search_stat + path_stats)
    summary = _format_result_summary(sections)
    if summary:
        parts.append(summary)

    # 2. Rank progression
    progression = _format_rank_progression(sections.get("search_stat", ""))
    if progression:
        parts.append(progression)

    # 3. Decomposition path analysis
    path_analysis = _format_path_analysis(sections.get("path_stats", ""))
    if path_analysis:
        parts.append(path_analysis)

    # 4. Search parameters (rank-scheduled policy)
    policy = _format_policy(sections.get("best_policy", ""))
    if policy:
        parts.append(policy)

    # 5. Historical context (path backups)
    history = _format_history(sections.get("evo_path_stats", ""))
    if history:
        parts.append(history)

    return "\n".join(parts)


def _split_sections(raw: str) -> dict[str, str]:
    """Split the raw aux string into named sections."""
    sections: dict[str, str] = {}

    # The raw string is: <path_stats>\nbest_policy:\n<policy>\nsearch_stat:\n<stats>
    # ...total_evals: N\nbest_seen_times: N\nevo path statistics:\n<backups>\nthis path name: <name>

    # Split on known delimiters
    parts = re.split(
        r"\n(?=best_policy:|search_stat:|evo path statistics:|this path name:)",
        raw,
    )

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("best_policy:"):
            sections["best_policy"] = part[len("best_policy:") :].strip()
        elif part.startswith("search_stat:"):
            sections["search_stat"] = part[len("search_stat:") :].strip()
        elif part.startswith("evo path statistics:"):
            sections["evo_path_stats"] = part[len("evo path statistics:") :].strip()
        elif part.startswith("this path name:"):
            sections["path_name"] = part[len("this path name:") :].strip()
        else:
            # First section is path stats (format_path_stats_tiny output)
            sections["path_stats"] = part

    return sections


def _format_result_summary(sections: dict[str, str]) -> str | None:
    """Extract key results into a concise summary."""
    lines = ["### Result Summary", ""]

    path_stats = sections.get("path_stats", "")
    search_stat = sections.get("search_stat", "")

    # Extract initial and final rank from path stats
    initial_rank = None
    final_rank = None
    n_steps = 0
    for line in path_stats.splitlines():
        line = line.strip()
        if not line or line == "...":
            continue
        m = re.match(r"(\d+):", line)
        if m:
            rank = int(m.group(1))
            if initial_rank is None:
                initial_rank = rank
            if line.endswith(":final"):
                final_rank = rank
            n_steps += 1

    if initial_rank and final_rank:
        reduction = initial_rank - final_rank
        lines.append(
            f"- **Final rank: {final_rank}** (from {initial_rank}, "
            f"reduced by {reduction} over {n_steps - 1} steps)"
        )

    # Extract quantiles
    for line in search_stat.splitlines():
        m = re.match(r"rank 0\.9q=([\d.]+)", line)
        if m:
            lines.append(f"- Rank 90th percentile: {m.group(1)}")
        m = re.match(r"rank 0\.1q=([\d.]+)", line)
        if m:
            lines.append(f"- Rank 10th percentile: {m.group(1)}")

    # Extract total evals and best_seen_times
    m = re.search(r"total_evals:\s*(\d+)", search_stat)
    if m:
        lines.append(f"- Total evaluations: {m.group(1)}")

    m = re.search(r"best_seen_times:\s*(\d+)", search_stat)
    if m:
        times = int(m.group(1))
        stability = (
            "very stable"
            if times >= 5
            else "stable"
            if times >= 2
            else "fragile (seen only once)"
        )
        lines.append(f"- Best rank found {times} time(s) ({stability})")

    path_name = sections.get("path_name", "")
    if path_name:
        lines.append(f"- Path: {path_name}")

    if len(lines) <= 2:
        return None
    lines.append("")
    return "\n".join(lines)


def _format_rank_progression(search_stat: str) -> str | None:
    """Format the rank-over-time progression."""
    rank_lines = []
    for line in search_stat.splitlines():
        m = re.match(r"(?:Final )?Rank=(\d+) at eval=(\d+)", line.strip())
        if m:
            prefix = "**Final** " if line.strip().startswith("Final") else ""
            rank_lines.append(f"  {prefix}Rank {m.group(1)} at eval {m.group(2)}")

    if not rank_lines:
        return None

    lines = [
        "### Rank Progression (rank over evaluations)",
        "",
        *rank_lines,
        "",
    ]
    return "\n".join(lines)


def _format_path_analysis(path_stats: str) -> str | None:
    """Parse cryptic path stats into readable decomposition analysis."""
    if not path_stats or "path stats unavailable" in path_stats:
        return f"### Decomposition Path\n\n{path_stats}\n" if path_stats else None

    lines = ["### Decomposition Steps", ""]
    lines.append(
        "Each step reduces the matrix rank. Format: "
        "rank → reduction/quality/beam_width/TOHPE_status"
    )
    lines.append("")

    step_lines = path_stats.splitlines()
    parsed_steps = []
    tohpe_ended_at = None

    for line in step_lines:
        line = line.strip()
        if not line:
            continue
        if line == "...":
            parsed_steps.append("  ...")
            continue
        if line.endswith(":final"):
            m = re.match(r"(\d+):final", line)
            if m:
                parsed_steps.append(f"  → **Rank {m.group(1)}** (final)")
            continue

        # Parse: 411:r2/d+0/q50.00%;bd3/d0/m3;bs250/d+0tha6/tda6/b0
        m = re.match(
            r"(\d+):r(\d+)/d([+-]?\d+)/q([\d.]+)%;"
            r"bd(\d+)/d([+-]?\d+)/m([\d.]+);"
            r"bs(\d+)/d([+-]?\d+)"
            r"tha(\d+)/tda(\d+)/b(\d+)",
            line,
        )
        if m:
            rank = int(m.group(1))
            reduction = int(m.group(2))
            red_vs_best = int(m.group(3))
            quality_pct = float(m.group(4))
            basis_dim = int(m.group(5))
            beam_width = int(m.group(8))
            tohpe_accepted = int(m.group(10))
            total_accepted = int(m.group(11))
            beyond = int(m.group(12))

            # Detect TOHPE phase end
            if tohpe_accepted == 0 and tohpe_ended_at is None:
                tohpe_ended_at = rank

            quality_note = ""
            if red_vs_best < 0:
                quality_note = f" (best possible: {reduction - red_vs_best})"
            elif red_vs_best == 0:
                quality_note = " (optimal)"

            beyond_note = " [non-improving]" if beyond else ""

            parsed_steps.append(
                f"  Rank {rank}: reduced by {reduction}{quality_note}, "
                f"quality={quality_pct:.0f}%, basis_dim={basis_dim}, "
                f"beam={beam_width}, "
                f"TOHPE={tohpe_accepted}/{total_accepted}{beyond_note}"
            )
        else:
            # Fallback: include raw line
            parsed_steps.append(f"  {line}")

    lines.extend(parsed_steps)
    lines.append("")

    if tohpe_ended_at is not None:
        lines.append(
            f"**TOHPE phase ended at rank ~{tohpe_ended_at}** — "
            f"subsequent steps used direct candidate search only."
        )
        lines.append("")

    return "\n".join(lines)


def _format_policy(best_policy: str) -> str | None:
    """Format the rank-scheduled policy parameters."""
    if not best_policy:
        return None

    lines = [
        "### Search Policy (rank-scheduled parameters)",
        "",
        "These parameters change as the matrix rank decreases during decomposition:",
        "",
    ]

    # The policy is a dict repr — parse key-value pairs
    # Format: {'key': value, 'key': value, ...}
    param_descriptions = {
        "num_samples": "Number of candidate decompositions sampled per step",
        "top_pool": "Size of the top candidate pool kept between steps",
        "pool_scores": "Scoring weights for ranking candidates in the pool",
        "final_scores": "Scoring weights for the final candidate selection",
        "max_z_to_research": "Maximum matrix complexity (Z dimension) to explore",
        "gen_part": "Fraction of candidates generated vs reused from pool",
    }

    # Try to parse as Python dict-like structure
    try:
        # The output of dao_rank_to_str is a dict with tuple values
        # e.g. {'num_samples': [(ranks), (values)], ...}
        # It's printed via str(), so we get Python repr
        import ast

        policy_dict = ast.literal_eval(best_policy)
        if isinstance(policy_dict, dict):
            for key, val in policy_dict.items():
                desc = param_descriptions.get(key, key)
                lines.append(f"- **{key}**: {val}")
                lines.append(f"  _{desc}_")
            lines.append("")
            return "\n".join(lines)
    except (ValueError, SyntaxError):
        pass

    # Fallback: include raw policy with header
    lines.append(best_policy)
    lines.append("")
    return "\n".join(lines)


def _format_history(evo_stats: str) -> str | None:
    """Format evolutionary path backup statistics."""
    if not evo_stats:
        return None

    lines = [
        "### Historical Best Decompositions",
        "",
    ]

    for line in evo_stats.splitlines():
        line = line.strip()
        if line.startswith("best_paths="):
            val = line[len("best_paths=") :]
            lines.append(f"- Best saved paths: {val}")
        elif line.startswith("count_total="):
            val = line[len("count_total=") :]
            lines.append(f"- Total paths saved: {val}")

    lines.append("")
    return "\n".join(lines)
