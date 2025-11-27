from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys

import click
import yaml

from gigaevo.problems.config import ProblemConfig
from gigaevo.problems.layout import ProblemLayout as PL


def load_problem_config_class(problem_type: str) -> type:
    """
    Dynamically load config class for problem type.

    Looks for custom config in types/{problem_type}/config.py.
    Falls back to base ProblemConfig if not found.

    Args:
        problem_type: Problem type name (e.g., 'programs', 'prompt_evolution')

    Returns:
        Config class (ProblemConfig or subclass)
    """
    try:
        module = importlib.import_module(
            f"gigaevo.problems.types.{problem_type}.config"
        )

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, ProblemConfig) and obj is not ProblemConfig:
                return obj

    except ImportError:
        pass

    return ProblemConfig


@click.command()
@click.argument(
    "config_name",
    type=str,
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing problem directory if it exists.",
)
@click.option(
    "--problem-type",
    default="programs",
    type=str,
    help="Problem type determining templates and utilities (default: programs).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory (default: problems/<problem.name>).",
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="Validate config without generating files.",
)
def main(
    config_name: str,
    overwrite: bool,
    problem_type: str,
    output_dir: Path | None,
    validate_only: bool,
) -> None:
    """
    Generate problem scaffolding from CONFIG_NAME.

    CONFIG_NAME should be a YAML filename (e.g., 'heilbron.yaml') located in
    tools/wizard/config/ directory. The config should include metrics, function
    signatures, task description, and optional features.
    """
    # Construct path to config file in fixed directory
    wizard_dir = Path(__file__).parent
    config_path = wizard_dir / "config" / config_name

    # Check if config exists
    if not config_path.exists():
        click.echo(
            click.style(f"❌ Config not found: {config_path}", fg="red"), err=True
        )
        click.echo("   Available configs in tools/wizard/config/:", err=True)
        config_dir = wizard_dir / "config"
        if config_dir.exists():
            for cfg in sorted(config_dir.glob("*.yaml")):
                click.echo(f"   - {cfg.name}", err=True)
        sys.exit(1)

    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        click.echo(click.style(f"❌ Failed to load config: {e}", fg="red"), err=True)
        sys.exit(1)

    # Dynamically load config class for problem type
    ConfigClass = load_problem_config_class(problem_type)

    # Validate config
    try:
        problem_config = ConfigClass(**config_data)
    except Exception as e:
        click.echo(click.style("❌ Invalid config:", fg="red"), err=True)
        click.echo(f"   {e}", err=True)
        sys.exit(1)

    # Validate-only mode
    if validate_only:
        click.echo(click.style("✅ Configuration is valid!", fg="green", bold=True))
        click.echo(f"   Problem: {problem_config.name}")
        click.echo(f"   Description: {problem_config.description}")
        click.echo(f"   Metrics: {len(problem_config.metrics)} defined")
        click.echo(
            f"   Initial programs: {len(problem_config.initial_programs)} specified"
        )
        features = [
            f
            for f, enabled in [
                ("context", problem_config.add_context),
                ("helper", problem_config.add_helper),
            ]
            if enabled
        ]
        click.echo(f"   Features: {', '.join(features) if features else 'none'}")
        if problem_config.utils_imports:
            click.echo("   Utils imports:")
            for target, spec in [
                ("validate", problem_config.utils_imports.validator),
                ("helper", problem_config.utils_imports.helper),
                ("context", problem_config.utils_imports.context),
                ("initial_programs", problem_config.utils_imports.initial_programs),
            ]:
                if spec:
                    funcs = (
                        "*" if spec.functions == ["*"] else ", ".join(spec.functions)
                    )
                    click.echo(f"     - {target}: {funcs}")
        return

    # Determine output directory
    if output_dir:
        target_dir = output_dir
    else:
        target_dir = Path("problems") / problem_config.name

    # Print summary
    click.echo(click.style("🔧 Generating problem scaffolding", fg="cyan", bold=True))
    click.echo(f"   Name: {problem_config.name}")
    click.echo(f"   Target: {target_dir}")
    click.echo(f"   Context: {problem_config.add_context}")
    click.echo(f"   Helper: {problem_config.add_helper}")
    click.echo()

    # Generate
    try:
        result = PL.scaffold(
            target_dir=target_dir,
            config=problem_config,
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
        if problem_config.add_helper:
            next_steps.append("Implement helper.py")
        if problem_config.add_context:
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
