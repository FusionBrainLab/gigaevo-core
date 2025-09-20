from __future__ import annotations

from pathlib import Path
import yaml

from src.programs.metrics.context import MetricsContext
from src.problems.layout import ProblemLayout as PL


class ProblemContext:
    """Lightweight accessor for problem assets under a problem directory.

    Responsibilities:
    - Load and validate metrics.yaml into a MetricsContext
    - Provide convenient accessors for task description/hints and mutation prompts

    Example:
    pc = ProblemContext(Path("problems/heilbron_simplified"))
    pc.metrics_context  # -> MetricsContext
    pc.task_description # -> str
    pc.task_hints       # -> str
    pc.mutation_system_prompt # -> str
    pc.mutation_user_prompt   # -> str
    """

    def __init__(self, problem_dir: Path):
        self.problem_dir = problem_dir
        self._metrics_context: MetricsContext | None = None

    # ---------- Text helpers ----------
    def load_text(self, relative_path: str) -> str:
        path = self.problem_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Problem file not found: {path}")
        return path.read_text().strip()

    @property
    def task_description(self) -> str:
        return self.load_text(PL.TASK_DESCRIPTION)

    @property
    def task_hints(self) -> str:
        return self.load_text(PL.TASK_HINTS)

    @property
    def mutation_system_prompt(self) -> str:
        return self.load_text(PL.MUTATION_SYSTEM_PROMPT)

    @property
    def mutation_user_prompt(self) -> str:
        return self.load_text(PL.MUTATION_USER_PROMPT)

    @property
    def is_contextual(self) -> bool:
        """Whether this problem provides an auxiliary context.py.

        If true, callers may add an AddContext stage to supply runtime context.
        """
        return (self.problem_dir / PL.CONTEXT_FILE).exists()

    # ---------- Metrics ----------
    @property
    def metrics_context(self) -> MetricsContext:
        if self._metrics_context is None:
            self._metrics_context = self._load_metrics_context()
        return self._metrics_context

    def _load_metrics_context(self) -> MetricsContext:
        metrics_path = self.problem_dir / PL.METRICS_FILE
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.yaml in {self.problem_dir}")

        try:
            data = yaml.safe_load(metrics_path.read_text())
            if not isinstance(data, dict):
                raise ValueError("metrics.yaml must be a mapping")

            specs = data.get("specs") or {
                k: v for k, v in data.items() if k != "display_order"
            }
            display_order = data.get("display_order")
            ctx = MetricsContext.from_dict(
                specs=specs, display_order=display_order
            )

            # Strict validation: primary bounds and is_valid [0,1]
            primary = ctx.get_primary_spec()
            pb = ctx.get_bounds(primary.key)
            if pb is None:
                raise ValueError(
                    f"metrics.yaml error in {metrics_path}: Primary metric '{primary.key}' must define lower_bound and upper_bound"
                )
            if "is_valid" not in ctx.specs:
                raise ValueError(
                    f"metrics.yaml error in {metrics_path}: Missing required 'is_valid' metric spec"
                )
            vb = ctx.get_bounds("is_valid")
            if vb is None or vb != (0.0, 1.0):
                raise ValueError(
                    f"metrics.yaml error in {metrics_path}: 'is_valid' must have lower_bound=0.0 and upper_bound=1.0"
                )

            return ctx
        except Exception as e:
            raise ValueError(f"Failed to parse metrics.yaml: {e}") from e

    # ---------- Validation ----------
    def validate(self, add_context: bool = False) -> None:
        """Validate that required files/dirs exist and are minimally correct.

        Also verifies that `initial_programs/` contains at least one .py file.
        """
        required_files = PL.required_files(add_context)
        required_dirs = PL.required_directories()

        missing_files: list[str] = [
            f for f in required_files if not (self.problem_dir / f).exists()
        ]
        missing_dirs: list[str] = [
            d for d in required_dirs if not (self.problem_dir / d).exists()
        ]
        if missing_files or missing_dirs:
            items = missing_files + [f"{d}/" for d in missing_dirs]
            raise FileNotFoundError(
                f"Missing required files/directories in {self.problem_dir}: {', '.join(items)}"
            )

        # Ensure at least one initial program exists
        initial_dir = self.problem_dir / PL.INITIAL_PROGRAMS_DIR
        if not list(initial_dir.glob("*.py")):
            raise FileNotFoundError(
                f"No Python files found in {initial_dir}. At least one initial program is required."
            )


