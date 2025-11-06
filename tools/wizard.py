#!/usr/bin/env python3
"""
Problem scaffolding wizard.

Generates problem directory structure from YAML config files.
Supports different problem types with type-specific templates and utilities.

Usage:
    python tools/wizard.py config/wizard/simple_problem.yaml
    python tools/wizard.py my_config.yaml --overwrite
    python tools/wizard.py my_config.yaml --output-dir custom/path
    python tools/wizard.py my_config.yaml --problem-type prompt_evolution
"""

from __future__ import annotations

from pathlib import Path
import sys

import click
import yaml

from gigaevo.problems.layout import ProblemLayout as PL
from gigaevo.problems.wizard_config import WizardConfig


@click.command()
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing problem directory if it exists.",
)
@click.option(
    "--problem-type",
    default="base",
    type=str,
    help="Problem type determining templates and utilities (default: base).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory (default: problems/<problem.name>).",
)
def main(
    config: Path,
    overwrite: bool,
    problem_type: str,
    output_dir: Path | None,
) -> None:
    """
    Generate problem scaffolding from CONFIG file.

    CONFIG should be a YAML file with problem specification including
    metrics, function signatures, task description, and optional features.

    Examples:

        \b
        # Generate from config
        python tools/wizard.py config/wizard/simple_problem.yaml

        \b
        # Overwrite existing problem
        python tools/wizard.py my_config.yaml --overwrite

        \b
        # Custom output location
        python tools/wizard.py my_config.yaml --output-dir custom/path

        \b
        # Use different problem type
        python tools/wizard.py my_config.yaml --problem-type prompt_evolution
    """
    # Load config
    try:
        with open(config) as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        click.echo(click.style(f"❌ Failed to load config: {e}", fg="red"), err=True)
        sys.exit(1)

    # Validate config
    try:
        wizard_config = WizardConfig(**config_data)
    except Exception as e:
        click.echo(click.style("❌ Invalid config:", fg="red"), err=True)
        click.echo(f"   {e}", err=True)
        sys.exit(1)

    # Determine output directory
    if output_dir:
        target_dir = output_dir
    else:
        target_dir = Path("problems") / wizard_config.problem.name

    # Print summary
    click.echo(click.style("🔧 Generating problem scaffolding", fg="cyan", bold=True))
    click.echo(f"   Name: {wizard_config.problem.name}")
    click.echo(f"   Target: {target_dir}")
    click.echo(f"   Context: {wizard_config.problem.add_context}")
    click.echo(f"   Helper: {wizard_config.problem.add_helper}")
    click.echo()

    # Generate
    try:
        result = PL.scaffold(
            target_dir=target_dir,
            config=wizard_config,
            overwrite=overwrite,
            problem_type=problem_type,
        )

        # Success message
        click.echo()
        click.echo(
            click.style("✅ Problem scaffolded successfully", fg="green", bold=True)
        )
        click.echo(
            f"   📁 {result['files_generated']} files generated, {result['initial_programs']} initial programs"
        )

        # Next steps
        next_steps = ["Implement validate.py", "Implement initial_programs/*.py"]
        if result["add_context"]:
            next_steps.append("Implement context.py")
        click.echo(f"   📝 Next: {', '.join(next_steps)}")

    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {e}", fg="red"), err=True)
        if click.confirm("Show traceback?", default=False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
