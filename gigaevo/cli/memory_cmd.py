"""Memory-system audit and calibration commands."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from gigaevo.evolution.mutation.base import MutationOperator


def _float_list(
    _ctx: click.Context, _param: click.Parameter, value: str
) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise click.BadParameter("expected comma-separated numbers") from exc
    if not result:
        raise click.BadParameter("list cannot be empty")
    if not all(math.isfinite(item) for item in result):
        raise click.BadParameter("all values must be finite")
    return result


def _probability_list(
    ctx: click.Context, param: click.Parameter, value: str
) -> tuple[float, ...]:
    result = _float_list(ctx, param, value)
    if any(item <= 0.0 or item >= 1.0 for item in result):
        raise click.BadParameter("all probabilities must be between 0 and 1")
    return result


def _positive_list(
    ctx: click.Context, param: click.Parameter, value: str
) -> tuple[float, ...]:
    result = _float_list(ctx, param, value)
    if any(item <= 0.0 for item in result):
        raise click.BadParameter("all values must be positive")
    return result


def _mutation_operator(value: str | None) -> type[MutationOperator] | None:
    if value is None:
        return None
    from gigaevo.evolution.mutation.base import MutationOperator
    from gigaevo.memory_v2.models import import_qualified_class

    try:
        symbol = import_qualified_class(value)
    except (ImportError, AttributeError, TypeError) as exc:
        raise click.BadParameter(str(exc), param_hint="--mutation-operator") from exc
    if not issubclass(symbol, MutationOperator):
        raise click.BadParameter(
            "class must extend MutationOperator", param_hint="--mutation-operator"
        )
    return symbol


@click.group("memory")
def memory() -> None:
    """Inspect and calibrate the causal memory system."""


@memory.command("calibrate-safety")
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--prior-probabilities",
    default="0.025,0.05,0.10,0.15,0.20,0.30",
    callback=_probability_list,
    show_default=True,
    help="Candidate environment invalidity probabilities, comma-separated.",
)
@click.option(
    "--baseline-sds",
    default="0.15,0.30,0.75",
    callback=_positive_list,
    show_default=True,
    help="Candidate baseline/context log-odds coefficient prior SDs.",
)
@click.option(
    "--shared-effect-means",
    default="-0.693147,0.0,0.693147",
    callback=_float_list,
    show_default=True,
    help=(
        "Outcome-independent shared treatment log-odds prior means. The default "
        "tests half, unchanged, and double treatment odds."
    ),
)
@click.option(
    "--shared-effect-sd",
    type=click.FloatRange(min=0.0, min_open=True),
    default=0.20,
    show_default=True,
    help="Fixed shared treatment-effect prior SD during the grid replay.",
)
@click.option(
    "--card-effect-sd",
    type=click.FloatRange(min=0.0, min_open=True),
    default=0.60,
    show_default=True,
    help="Fixed card treatment-effect prior SD during the grid replay.",
)
@click.option(
    "--offer-rates",
    default="0.50,0.70,0.75,0.80",
    callback=_probability_list,
    show_default=True,
    help="Delivery rates for the randomized-overlap cost table.",
)
@click.option(
    "--min-observations",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Minimum closed proposals before emitting Hydra overrides.",
)
@click.option(
    "--min-gate-retention",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.25,
    show_default=True,
    help=(
        "Minimum replayed candidate fraction retained in both cold-start and "
        "later-new-card strata before emitting Hydra overrides."
    ),
)
@click.option(
    "--mutation-operator",
    default=None,
    help=(
        "Concrete MutationOperator class for a legacy ledger that lacks an adjacent "
        ".hydra/config.yaml. New ledgers do not need this option."
    ),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write the complete machine-readable calibration report to this JSON file.",
)
@click.option(
    "--top",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Number of ranked prior candidates shown per environment.",
)
@click.pass_context
def calibrate_safety(
    ctx: click.Context,
    inputs: tuple[Path, ...],
    prior_probabilities: tuple[float, ...],
    baseline_sds: tuple[float, ...],
    shared_effect_means: tuple[float, ...],
    shared_effect_sd: float,
    card_effect_sd: float,
    offer_rates: tuple[float, ...],
    min_observations: int,
    min_gate_retention: float,
    mutation_operator: str | None,
    output: Path | None,
    top: int,
) -> None:
    """Calibrate safety priors from one or more memory-v2 causal ledgers.

    INPUTS may be SQLite ledger files, checkpoint directories, or run
    directories. Predictions are replayed using only each decision's frozen
    fitted-observation set. Results from incompatible task/model/operator
    environments are never pooled.
    """

    from gigaevo.memory_v2.calibration import (
        calibrate_safety_priors,
        discover_ledger_paths,
        load_calibration_trajectory,
    )

    legacy_operator = _mutation_operator(mutation_operator)
    try:
        paths = discover_ledger_paths(inputs)
        trajectories = tuple(
            load_calibration_trajectory(path, legacy_mutation_operator=legacy_operator)
            for path in paths
        )
        report = calibrate_safety_priors(
            trajectories,
            prior_probabilities=prior_probabilities,
            baseline_sds=baseline_sds,
            shared_effect_means=shared_effect_means,
            shared_effect_sd=shared_effect_sd,
            card_effect_sd=card_effect_sd,
            offer_rates=offer_rates,
            min_observations=min_observations,
            min_gate_retention=min_gate_retention,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    formatter = ctx.find_root().obj["formatter"]
    if formatter.effective_format == "json":
        click.echo(json.dumps(report, indent=2))
        return

    rows: list[dict[str, Any]] = []
    for group in report["groups"]:
        environment = group["environment"]
        for rank, candidate in enumerate(group["grid_ranking"][:top], start=1):
            prior = candidate["prior"]
            gate = candidate["gate_replay"]
            rows.append(
                {
                    "task": environment["task_key"],
                    "model": environment["llm"]["model_name"],
                    "operator": environment["mutation_operator"].rsplit(".", 1)[-1],
                    "rank": rank,
                    "p_invalid": f"{prior['invalidity_prior_probability']:.3f}",
                    "baseline_sd": f"{prior['safety_baseline_prior_sd']:.3f}",
                    "shared_mean": (f"{prior['safety_shared_effect_prior_mean']:+.3f}"),
                    "log_loss": f"{candidate['log_loss']:.4f}",
                    "brier": f"{candidate['brier_score']:.4f}",
                    "bias": f"{candidate['calibration_bias']:+.4f}",
                    "cold_gate": (
                        f"{gate['cold_start']['certified']}/"
                        f"{gate['cold_start']['candidates']}"
                    ),
                    "new_card_gate": (
                        f"{gate['new_card_after_history']['certified_fraction']:.0%}"
                    ),
                    "status": group["status"],
                }
            )
    formatter.echo(rows, title="Memory-v2 safety-prior calibration")
    for group in report["groups"]:
        overrides = group["hydra_overrides"]
        if overrides:
            click.echo("Hydra overrides: " + " ".join(overrides))
    if output is not None:
        click.echo(f"Full report: {output}")
