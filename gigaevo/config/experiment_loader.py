from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gigaevo.config.schemas.experiment import ExperimentConfig


class ExperimentModuleError(Exception):
    """Raised when an experiment file cannot be loaded or validated."""


def load_experiment_module(path: Path | str) -> ModuleType:
    """Load an experiment ``.py`` file as an isolated Python module.

    Each experiment file is expected to export a top-level ``build``
    callable that returns an :class:`ExperimentConfig`. The function
    uses ``importlib.util.spec_from_file_location`` to load the module
    under a unique name (``exp_{stem}``) so that loading multiple
    experiments from the same Python process — for example a sweep
    that imports each variant — does not collide in ``sys.modules``.

    The function does not mutate ``sys.path``: experiment files are
    expected to import from ``gigaevo`` and its sub-packages using
    absolute imports, which work as long as ``gigaevo`` is installed.

    Args:
        path: Path to a Python file exporting ``build``.

    Returns:
        The loaded module object. The caller invokes ``module.build()``
        to obtain the :class:`ExperimentConfig` instance.

    Raises:
        FileNotFoundError: when ``path`` does not exist.
        ExperimentModuleError: when the spec / loader resolution
            fails, when the experiment module raises during top-level
            execution, or when ``build`` is missing, not callable, or
            has the wrong signature.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"experiment file not found: {resolved}")
    if not resolved.is_file():
        raise ExperimentModuleError(
            f"experiment path is not a file: {resolved}"
        )

    spec = importlib.util.spec_from_file_location(
        f"exp_{resolved.stem}", resolved
    )
    if spec is None or spec.loader is None:
        raise ExperimentModuleError(
            f"could not load experiment module spec from {resolved}"
        )

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ExperimentModuleError(
            f"experiment module {resolved} raised during import: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    build = getattr(module, "build", None)
    if build is None or not callable(build):
        raise ExperimentModuleError(
            f"experiment module {resolved} must export a callable named 'build'"
        )

    signature = inspect.signature(build)
    required_params = [
        p
        for p in signature.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if required_params:
        raise ExperimentModuleError(
            f"experiment module {resolved}: build() must accept zero "
            f"required arguments; got required parameter(s) "
            f"{[p.name for p in required_params]}"
        )

    return module


def build_experiment(path: Path | str) -> "ExperimentConfig":
    """Load an experiment file and call its ``build()`` to obtain the
    fully-resolved :class:`ExperimentConfig`. The returned instance is
    frozen and Pydantic-validated."""
    from gigaevo.config.schemas.experiment import ExperimentConfig

    module = load_experiment_module(path)
    cfg = module.build()
    if not isinstance(cfg, ExperimentConfig):
        raise ExperimentModuleError(
            f"experiment module {path}: build() returned "
            f"{type(cfg).__name__}, expected ExperimentConfig"
        )
    return cfg
