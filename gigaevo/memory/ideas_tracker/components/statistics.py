import json
from typing import Any

import pandas as pd

from gigaevo.memory.ideas_tracker.utils.it_logger import IdeasTrackerLogger
from gigaevo.memory.ideas_tracker.utils.selected_ideas_6 import compute_origin_analysis


def compute_evolutionary_statistics(logger: IdeasTrackerLogger) -> None:
    """
    Run origin-based evolutionary statistics on saved banks/programs JSONs
    and inject per-idea metrics into banks.json under 'evolution_statistics'.

    Requires that dump_final_state and log_programs have already been called
    so that banks.json and programs.json exist in the session directory.
    """
    banks_path = logger.banks_file
    programs_path = logger.programs_file

    if banks_path is None or programs_path is None:
        return
    if not banks_path.exists() or not programs_path.exists():
        return

    try:
        df_summary, df_best_ideas = compute_origin_analysis(
            banks_path=str(banks_path),
            programs_path=str(programs_path),
        )
    except RuntimeError as exc:
        if str(exc) == "No valid programs with numeric generation and fitness found.":
            return
        raise

    if df_summary.empty:
        return

    best_ideas_records = df_best_ideas.to_dict(orient="records")
    sanitized: list[dict[str, Any]] = []
    for rec in best_ideas_records:
        sanitized.append({k: (v if pd.notna(v) else None) for k, v in rec.items()})
    logger.log_best_ideas({"best_ideas": sanitized})

    stats_by_idea: dict[str, dict[str, Any]] = {}
    for _, row in df_summary.iterrows():
        idea_id = row["idea_id"]
        quartile = row["quartile"]
        metrics = row.drop(["idea_id", "quartile", "description"]).to_dict()
        metrics = {k: (v if pd.notna(v) else None) for k, v in metrics.items()}
        if idea_id not in stats_by_idea:
            stats_by_idea[idea_id] = {}
        stats_by_idea[idea_id][quartile] = metrics

    with open(banks_path, "r", encoding="utf-8") as f:
        banks_data = json.load(f)

    for snapshot in banks_data:
        if not isinstance(snapshot, dict):
            continue
        for bank_key in ("active_bank", "inactive_bank"):
            bank = snapshot.get(bank_key, [])
            if not isinstance(bank, list):
                continue
            for idea in bank:
                if not isinstance(idea, dict):
                    continue
                idea_id = idea.get("id", "")
                if idea_id in stats_by_idea:
                    idea["evolution_statistics"] = stats_by_idea[idea_id]

    with open(banks_path, "w", encoding="utf-8") as f:
        json.dump(banks_data, f, indent=4)
