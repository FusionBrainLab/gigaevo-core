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
        problem_type: str = "programs",
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
        cls._register_jinja_filters(env, problem_type)

        context = cls._build_template_context(config, problem_type)

        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / cls.INITIAL_PROGRAMS_DIR).mkdir(exist_ok=True)

        file_map = cls._get_file_template_map(config)

        for output_file, template_name in file_map.items():
            template = env.get_template(template_name)
            # Pass file-specific utils_import to template
            utils_import = cls._get_utils_import_for_file(config, output_file)
            content = template.render(**context, utils_import=utils_import)
            output_path = target_dir / output_file
            output_path.write_text(content)

        num_initial_programs = cls._generate_initial_programs(
            target_dir, config, env, context
        )

        return {
            "target_dir": target_dir,
            "files_generated": len(file_map),
            "initial_programs": num_initial_programs,
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
        """Convert Pydantic model to dict."""
        return obj.model_dump()

    @staticmethod
    def _register_jinja_filters(env: Environment, problem_type: str) -> None:
        """Register custom Jinja filters for template rendering."""
        from gigaevo.problems.config import FunctionSignature

        def param_string_filter(sig_dict: dict, with_types: bool = True) -> str:
            """Generate parameter string from signature dict."""
            sig = FunctionSignature(**sig_dict)
            return sig.get_param_string(with_types=with_types)

        def param_names_filter(sig_dict: dict) -> list[str]:
            """Extract parameter names from signature dict."""
            sig = FunctionSignature(**sig_dict)
            return sig.get_param_names()

        def utils_import_filter(utils_import: dict | None) -> str:
            """Generate import statement from utils_import spec.

            Args:
                utils_import: Dict with 'functions' key, or None

            Returns:
                Import statement string or empty string
            """
            if not utils_import:
                return ""
            functions = utils_import.get("functions", [])
            if not functions:
                return ""
            if functions == ["*"]:
                return f"from gigaevo.problems.types.{problem_type}.utils import *"
            return f"from gigaevo.problems.types.{problem_type}.utils import {', '.join(functions)}"

        env.filters["param_string"] = param_string_filter
        env.filters["param_names"] = param_names_filter
        env.filters["utils_import"] = utils_import_filter

    @classmethod
    def _build_template_context(cls, config: ProblemConfig, problem_type: str) -> dict:
        """Build Jinja template context with auto-generated is_valid metric.

        Args:
            config: Problem configuration
            problem_type: Problem type name (for utils import path)

        Returns:
            Template context dictionary
        """
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
            sentinel_value=0.0,
        )

        all_metrics = {**config.metrics, VALIDITY_KEY: is_valid_spec}

        metrics_dict = {key: cls._to_dict(spec) for key, spec in all_metrics.items()}

        helper_functions = None
        if config.helper_functions:
            helper_functions = [cls._to_dict(hf) for hf in config.helper_functions]

        context_spec = None
        if config.context_spec:
            context_spec = cls._to_dict(config.context_spec)

        # Collect all unique utils function names for task_description
        utils_functions: list[str] = []
        if config.utils_imports:
            all_funcs: set[str] = set()
            for spec in [
                config.utils_imports.validator,
                config.utils_imports.helper,
                config.utils_imports.context,
                config.utils_imports.initial_programs,
            ]:
                if spec and spec.functions != ["*"]:
                    all_funcs.update(spec.functions)
            utils_functions = sorted(all_funcs)

        return {
            "problem": {"name": config.name, "description": config.description},
            "entrypoint": cls._to_dict(config.entrypoint),
            "validation": cls._to_dict(config.validation),
            "metrics": metrics_dict,
            "primary_key": primary_key,
            "task_description": cls._to_dict(config.task_description),
            "add_context": config.add_context,
            "add_helper": config.add_helper,
            "helper_functions": helper_functions,
            "context_spec": context_spec,
            "problem_type": problem_type,
            "utils_functions": utils_functions,
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
    def _get_utils_import_for_file(
        cls, config: ProblemConfig, output_file: str
    ) -> dict | None:
        """Get utils import spec for a specific output file.

        Args:
            config: Problem configuration
            output_file: Output filename (e.g., 'validate.py', 'helper.py')

        Returns:
            Utils import spec as dict, or None if not configured
        """
        if not config.utils_imports:
            return None

        spec = None
        if output_file == cls.VALIDATOR:
            spec = config.utils_imports.validator
        elif output_file == "helper.py":
            spec = config.utils_imports.helper
        elif output_file == cls.CONTEXT_FILE:
            spec = config.utils_imports.context

        return cls._to_dict(spec) if spec else None

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

        # Get utils import for initial programs
        utils_import = None
        if config.utils_imports and config.utils_imports.initial_programs:
            utils_import = cls._to_dict(config.utils_imports.initial_programs)

        for prog_spec in config.initial_programs:
            content = template.render(
                **context,
                program_name=prog_spec.name,
                program_description=prog_spec.description,
                utils_import=utils_import,
            )

            prog_path = target_dir / cls.INITIAL_PROGRAMS_DIR / f"{prog_spec.name}.py"
            prog_path.write_text(content)

        return len(config.initial_programs)
