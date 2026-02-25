import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def _parse_json_like(value: Any) -> Any:
    """Parse JSON-ish strings from CSV back to Python objects when possible."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback for legacy CSVs with Python repr (' instead of ").
        try:
            return ast.literal_eval(text)
        except Exception:
            return value


def _coerce_bool_column(series: pd.Series) -> pd.Series:
    """Coerce a series to booleans for robust filtering logic."""
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe loaded from redis2pd CSV for IdeaTracker."""
    result = df.copy()

    if "is_root" in result.columns:
        result["is_root"] = _coerce_bool_column(result["is_root"])

    for col in ("parent_ids", "children_ids", "metadata_mutation_output"):
        if col in result.columns:
            result[col] = result[col].apply(_parse_json_like)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ideas tracker from a CSV exported by tools/redis2pd.py"
    )
    parser.add_argument("--csv-path", required=True, help="Path to redis2pd CSV file")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to unified YAML config (defaults to config/memory.yaml, ideas_tracker section)",
    )
    args = parser.parse_args()

    from gigaevo.llm.ideas_tracker.ideas_tracker import IdeaTracker
    from gigaevo.llm.ideas_tracker.utils.ideas_impact_ml import run_impact_pipeline

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _prepare_dataframe(df)

    tracker = IdeaTracker(config_path=args.config_path)
    last_gen = df["generation"].max()
    new_programs = tracker.get_new_programs(df)
    new_programs_processed = tracker.wrap_data(new_programs)

    pbar = tqdm.tqdm(total=len(new_programs), leave=False)
    for prog in new_programs_processed:
        tracker.process_program(prog)
        pbar.update(1)
    pbar.close()

    tracker.refresh_main_bank(last_gen)
    tracker.logger.dump_final_state(tracker.ideas_manager)
    tracker.logger.log_programs(tracker._programs_to_dicts())
    tracker.get_rankings()
    tracker.logger.log_best_ideas(tracker.top_ideas())

    if tracker.ml_impact_enabled:
        all_ideas = (
            tracker.ideas_manager.record_bank.all_ideas_cards()
            + tracker.ideas_manager.inactive_record_bank.all_ideas_cards()
        )
        if tracker.programs_card and all_ideas:
            impact_result = run_impact_pipeline(
                tracker.programs_card,
                all_ideas,
                n_iterations=tracker.ml_impact_n_iterations,
                include_interactions=tracker.ml_impact_include_interactions,
                max_interaction_pairs=tracker.ml_impact_max_interaction_pairs,
                min_idea_programs=tracker.ml_impact_min_idea_programs,
                confidence_level=tracker.ml_impact_confidence_level,
                random_state=tracker.ml_impact_random_state,
            )
            output_dir = Path(__file__).resolve().parent
            impact_result.summary.to_csv(output_dir / "impact_summary.csv")
            if impact_result.interactions is not None:
                impact_result.interactions.to_csv(
                    output_dir / "impact_interactions.csv"
                )


if __name__ == "__main__":
    main()
