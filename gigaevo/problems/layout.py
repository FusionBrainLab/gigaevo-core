from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from gigaevo.problems.config import ProblemConfig
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricSpec


class ProblemLayout:
    """Standardized problem directory layout and scaffolding."""

    TASK_DESCRIPTION = "task_description.txt"
    VALIDATOR = "validate.py"
    CONTEXT_FILE = "context.py"
    METRICS_FILE = "metrics.yaml"

    INITIAL_PROGRAMS_DIR = "initial_programs"
    TEMPLATES_DIR = Path(__file__).parent / "types"

    @classmethod
    def required_files(cls, add_context: bool = False) -> list[str]:
        """Required files for a valid problem."""
        files = [cls.TASK_DESCRIPTION, cls.VALIDATOR, cls.METRICS_FILE]
        if add_context:
            files.append(cls.CONTEXT_FILE)
        return files

    @classmethod
    def required_directories(cls) -> list[str]:
        """Required directories for a valid problem."""
        return [cls.INITIAL_PROGRAMS_DIR]

    @classmethod
    def scaffold(
        cls,
        target_dir: Path,
        config: ProblemConfig,
        overwrite: bool = False,
        problem_type: str = "base",
    ) -> dict:
        """Generate problem directory from config.

        Args:
            target_dir: Where to create problem
            config: ProblemConfig with all parameters
            overwrite: Replace existing files
            problem_type: Problem type determining templates and utilities (default: base)

        Returns:
            dict with keys: target_dir, files_generated, initial_programs, add_context

        Raises:
            ValueError: If target exists and overwrite=False, or invalid problem_type
        """
        target_dir = Path(target_dir)

        if target_dir.exists() and not overwrite:
            raise ValueError(f"{target_dir} exists. Use --overwrite to replace.")

        template_dir = cls._get_template_dir(problem_type)
        env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        context = cls._build_template_context(config)

        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / cls.INITIAL_PROGRAMS_DIR).mkdir(exist_ok=True)

        file_map = cls._get_file_template_map(config)

        for output_file, template_name in file_map.items():
            template = env.get_template(template_name)
            content = template.render(**context)
            output_path = target_dir / output_file
            output_path.write_text(content)

        num_initial_programs = cls._generate_initial_programs(
            target_dir, config, env, context
        )

        return {
            "target_dir": target_dir,
            "files_generated": len(file_map),
            "initial_programs": num_initial_programs,
            "add_context": config.add_context,
        }

    @classmethod
    def _get_template_dir(cls, problem_type: str) -> Path:
        """Get template directory for problem type.

        Args:
            problem_type: Problem type name

        Returns:
            Path to template directory

        Raises:
            ValueError: If template directory doesn't exist
        """
        template_dir = cls.TEMPLATES_DIR / problem_type / "templates"
        if not template_dir.exists():
            raise ValueError(
                f"Unknown problem type: '{problem_type}'. "
                f"Template directory not found: {template_dir}\n"
                f"Available types: {', '.join(d.name for d in cls.TEMPLATES_DIR.iterdir() if d.is_dir() and not d.name.startswith('_'))}"
            )
        return template_dir

    @staticmethod
    def _to_dict(obj) -> dict:
        """Convert Pydantic model to dict (v1/v2 compatible)."""
        return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()

    @classmethod
    def _build_template_context(cls, config: ProblemConfig) -> dict:
        """Build Jinja template context with auto-generated is_valid metric."""
        primary_key = next(
            (key for key, spec in config.metrics.items() if spec.is_primary), None
        )

        is_valid_spec = MetricSpec(
            description="Whether the program is valid (1 valid, 0 invalid)",
            decimals=0,
            is_primary=False,
            higher_is_better=True,
            lower_bound=0.0,
            upper_bound=1.0,
            include_in_prompts=True,
            significant_change=1.0,
        )

        all_metrics = {**config.metrics, VALIDITY_KEY: is_valid_spec}

        display_order = (
            config.display_order.copy()
            if config.display_order
            else list(config.metrics.keys())
        )
        if VALIDITY_KEY not in display_order:
            display_order.append(VALIDITY_KEY)

        metrics_dict = {key: cls._to_dict(spec) for key, spec in all_metrics.items()}

        return {
            "problem": {"name": config.name, "description": config.description},
            "entrypoint": cls._to_dict(config.entrypoint),
            "validation": cls._to_dict(config.validation),
            "metrics": metrics_dict,
            "display_order": display_order,
            "primary_key": primary_key,
            "task_description": cls._to_dict(config.task_description),
            "add_context": config.add_context,
            "add_helper": config.add_helper,
        }

    @classmethod
    def _get_file_template_map(cls, config: ProblemConfig) -> dict[str, str]:
        """Map output filenames to template names."""
        file_map = {
            cls.TASK_DESCRIPTION: "task_description.jinja",
            cls.METRICS_FILE: "metrics.jinja",
            cls.VALIDATOR: "validate.jinja",
        }

        if config.add_context:
            file_map[cls.CONTEXT_FILE] = "context.jinja"

        if config.add_helper:
            file_map["helper.py"] = "helper.jinja"

        return file_map

    @classmethod
    def _generate_initial_programs(
        cls,
        target_dir: Path,
        config: ProblemConfig,
        env: Environment,
        context: dict,
    ) -> int:
        """Generate initial program stubs.

        Returns:
            Number of initial programs generated
        """
        if not config.initial_programs:
            return 0

        template = env.get_template("initial_program.jinja")

        for prog_spec in config.initial_programs:
            content = template.render(
                **context,
                program_name=prog_spec.name,
                program_description=prog_spec.description,
            )

            prog_path = target_dir / cls.INITIAL_PROGRAMS_DIR / f"{prog_spec.name}.py"
            prog_path.write_text(content)

        return len(config.initial_programs)
