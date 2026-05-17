"""Convenience builders for the shipped problem directories.

The Phase 1 :class:`ProblemConfig` from
:mod:`gigaevo.config.schemas.problem` already takes an arbitrary
``problem_dir`` path and resolves the metrics + task description
lazily at build time. This module pins the canonical paths for the
shipped problem layouts under ``problems/`` so experiment files
compose with one line:

    from gigaevo.config.problem_presets import build_hotpotqa_static_f1
    cfg = ExperimentConfig(..., problem=build_hotpotqa_static_f1())

The presets correspond to the directories that actually ship with
runtime assets (metrics.yaml + task_description.txt). Experimental
variants under the same family — for example, the dozens of
algotune subproblems — get a parametrised helper rather than one
preset each; that keeps this module small while still covering every
shipped path.
"""

from __future__ import annotations

from pathlib import Path

from gigaevo.config.schemas.problem import ProblemConfig


_PROBLEMS_ROOT = Path("problems")


def _under(*parts: str) -> Path:
    return _PROBLEMS_ROOT.joinpath(*parts)


# ---------------------------------------------------------------------------
# Chain reasoning problems (HotpotQA, HoVer, etc.)
# ---------------------------------------------------------------------------


def build_hotpotqa_static_f1(
    *,
    name: str = "hotpotqa_static_f1",
    primary_metric: str | None = None,
    higher_is_better: bool | None = None,
) -> ProblemConfig:
    """HotpotQA chain with static-F1 reward; primary metric is
    token-level F1."""
    return ProblemConfig(
        name=name,
        problem_dir=_under("chains", "hotpotqa", "static_f1"),
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
    )


def build_hotpotqa_qa(
    *,
    name: str = "hotpotqa_qa",
) -> ProblemConfig:
    """HotpotQA QA variant (separate evaluator) under
    ``problems/chains/hotpotqa_qa/``."""
    return ProblemConfig(name=name, problem_dir=_under("chains", "hotpotqa_qa"))


def build_hover(*, name: str = "hover") -> ProblemConfig:
    """HoVer fact-verification chain under ``problems/chains/hover/``."""
    return ProblemConfig(name=name, problem_dir=_under("chains", "hover"))


def build_musique(*, name: str = "musique") -> ProblemConfig:
    """MuSiQue multi-hop reasoning chain."""
    return ProblemConfig(name=name, problem_dir=_under("chains", "musique"))


def build_aime(*, name: str = "aime") -> ProblemConfig:
    """AIME math-competition chain."""
    return ProblemConfig(name=name, problem_dir=_under("chains", "aime"))


def build_papillon(*, name: str = "papillon") -> ProblemConfig:
    """PaPILLon (privacy-aware chain) under ``problems/chains/papillon/``."""
    return ProblemConfig(name=name, problem_dir=_under("chains", "papillon"))


def build_ifbench(*, name: str = "ifbench") -> ProblemConfig:
    """IFBench instruction-following chain."""
    return ProblemConfig(name=name, problem_dir=_under("chains", "ifbench"))


# ---------------------------------------------------------------------------
# Geometric / optimization problems
# ---------------------------------------------------------------------------


def build_heilbron(*, name: str = "heilbron") -> ProblemConfig:
    """Heilbron triangle problem: maximise the minimum-area triangle
    formed by n points in the unit square."""
    return ProblemConfig(name=name, problem_dir=_under("heilbron"))


def build_hexagon_pack(*, name: str = "hexagon_pack") -> ProblemConfig:
    """Hexagon packing problem under ``problems/hexagon_pack/``."""
    return ProblemConfig(name=name, problem_dir=_under("hexagon_pack"))


def build_hexagon_improver(*, name: str = "hexagon_improver") -> ProblemConfig:
    """Hexagon improver variant."""
    return ProblemConfig(name=name, problem_dir=_under("hexagon_improver"))


def build_kissing_number(
    dim: int,
    *,
    name: str | None = None,
) -> ProblemConfig:
    """Kissing-number problem in ``dim`` dimensions. ``dim`` must
    correspond to a shipped directory (11 or 12 today)."""
    if dim not in (11, 12):
        raise ValueError(
            f"kissing-number problem ships for dim ∈ {{11, 12}}; got {dim}"
        )
    dir_name = f"kissing_number_{dim}d"
    return ProblemConfig(
        name=name or dir_name,
        problem_dir=_under(dir_name),
    )


def build_santa2025_n100(*, name: str = "santa2025_n100") -> ProblemConfig:
    return ProblemConfig(name=name, problem_dir=_under("santa2025_n100"))


def build_santa2025_tree_packing(
    *, name: str = "santa2025_tree_packing"
) -> ProblemConfig:
    return ProblemConfig(name=name, problem_dir=_under("santa2025_tree_packing"))


# ---------------------------------------------------------------------------
# AlgoTune family (50+ subproblems)
# ---------------------------------------------------------------------------


def build_algotune(
    subproblem: str,
    *,
    name: str | None = None,
) -> ProblemConfig:
    """Parametrised AlgoTune builder.

    ``subproblem`` is the directory name under ``problems/algotune/``,
    typically prefixed with ``algotune_`` (e.g. ``"battery_scheduling"``
    becomes ``problems/algotune/algotune_battery_scheduling/``). Pass
    the bare subproblem name; the prefix is added if missing."""
    if not subproblem.startswith("algotune_"):
        subproblem = f"algotune_{subproblem}"
    problem_dir = _under("algotune", subproblem)
    return ProblemConfig(
        name=name or subproblem,
        problem_dir=problem_dir,
    )


# ---------------------------------------------------------------------------
# Toy / debug problems
# ---------------------------------------------------------------------------


def build_toy_example(*, name: str = "toy_example") -> ProblemConfig:
    """Toy problem for smoke tests."""
    return ProblemConfig(name=name, problem_dir=_under("toy_example"))


def build_toy_kadane(*, name: str = "toy_kadane") -> ProblemConfig:
    """Toy Kadane's algorithm problem."""
    return ProblemConfig(name=name, problem_dir=_under("toy_kadane"))


# ---------------------------------------------------------------------------
# Prompt-evolution problems
# ---------------------------------------------------------------------------


def build_optimization(*, name: str = "optimization") -> ProblemConfig:
    """Generic numerical optimization problem under
    ``problems/optimization/``."""
    return ProblemConfig(name=name, problem_dir=_under("optimization"))


def build_dashboard(*, name: str = "dashboard") -> ProblemConfig:
    """Dashboard problem under ``problems/dashboard/`` — internal
    smoke / demo target."""
    return ProblemConfig(name=name, problem_dir=_under("dashboard"))


def build_alphaevolve(
    subproblem: str,
    *,
    name: str | None = None,
) -> ProblemConfig:
    """Parametrised builder for the AlphaEvolve subproblem family
    under ``problems/alphaevolve/``. Pass the bare subproblem name
    (e.g. ``"erdos_minimum_overlap"``); the prefix is added if missing
    so callers can also pass ``"alphaevolve_erdos_minimum_overlap"``
    for symmetry with the AlgoTune naming convention."""
    if subproblem.startswith("alphaevolve_"):
        subproblem = subproblem[len("alphaevolve_") :]
    problem_dir = _under("alphaevolve", subproblem)
    return ProblemConfig(
        name=name or f"alphaevolve_{subproblem}",
        problem_dir=problem_dir,
    )


def build_prompt_evolution(*, name: str = "prompt_evolution") -> ProblemConfig:
    """Prompt evolution problem — used by the coevolved-prompt
    fetcher topology where prompts themselves are evolved as
    programs."""
    return ProblemConfig(name=name, problem_dir=_under("prompt_evolution"))


def build_prompt_evolution_hover(
    *, name: str = "prompt_evolution_hover"
) -> ProblemConfig:
    """HoVer-specific prompt evolution variant."""
    return ProblemConfig(
        name=name, problem_dir=_under("prompt_evolution_hover")
    )
