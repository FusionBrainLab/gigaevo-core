"""Unit tests for the problem-directory preset builders."""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.config.problem_presets import (
    build_aime,
    build_algotune,
    build_heilbron,
    build_hexagon_improver,
    build_hexagon_pack,
    build_hotpotqa_qa,
    build_hotpotqa_static_f1,
    build_hover,
    build_ifbench,
    build_kissing_number,
    build_musique,
    build_papillon,
    build_prompt_evolution,
    build_prompt_evolution_hover,
    build_santa2025_n100,
    build_santa2025_tree_packing,
    build_toy_example,
    build_toy_kadane,
)
from gigaevo.config.schemas import ProblemConfig


class TestChainPresets:
    def test_hotpotqa_static_f1_canonical_path(self) -> None:
        cfg = build_hotpotqa_static_f1()
        assert isinstance(cfg, ProblemConfig)
        assert cfg.problem_dir == Path("problems/chains/hotpotqa/static_f1")
        assert cfg.name == "hotpotqa_static_f1"

    def test_hotpotqa_static_f1_accepts_overrides(self) -> None:
        cfg = build_hotpotqa_static_f1(
            name="custom",
            primary_metric="em",
            higher_is_better=False,
        )
        assert cfg.name == "custom"
        assert cfg.primary_metric == "em"
        assert cfg.higher_is_better is False

    @pytest.mark.parametrize(
        "builder,expected_subdir",
        [
            (build_hotpotqa_qa, ("chains", "hotpotqa_qa")),
            (build_hover, ("chains", "hover")),
            (build_musique, ("chains", "musique")),
            (build_aime, ("chains", "aime")),
            (build_papillon, ("chains", "papillon")),
            (build_ifbench, ("chains", "ifbench")),
        ],
    )
    def test_each_chain_preset_resolves_under_chains(
        self, builder, expected_subdir
    ) -> None:
        cfg = builder()
        assert cfg.problem_dir == Path("problems").joinpath(*expected_subdir)


class TestOptimizationPresets:
    @pytest.mark.parametrize(
        "builder,expected_subdir",
        [
            (build_heilbron, "heilbron"),
            (build_hexagon_pack, "hexagon_pack"),
            (build_hexagon_improver, "hexagon_improver"),
            (build_toy_example, "toy_example"),
            (build_toy_kadane, "toy_kadane"),
            (build_santa2025_n100, "santa2025_n100"),
            (build_santa2025_tree_packing, "santa2025_tree_packing"),
            (build_prompt_evolution, "prompt_evolution"),
            (build_prompt_evolution_hover, "prompt_evolution_hover"),
        ],
    )
    def test_optimization_preset_resolves_one_level_under_problems(
        self, builder, expected_subdir: str
    ) -> None:
        cfg = builder()
        assert cfg.problem_dir == Path("problems") / expected_subdir


class TestKissingNumberPreset:
    def test_dim_11(self) -> None:
        cfg = build_kissing_number(11)
        assert cfg.problem_dir == Path("problems/kissing_number_11d")
        assert cfg.name == "kissing_number_11d"

    def test_dim_12(self) -> None:
        cfg = build_kissing_number(12)
        assert cfg.problem_dir == Path("problems/kissing_number_12d")

    def test_unknown_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            build_kissing_number(7)


class TestAlgoTunePreset:
    def test_bare_name_gets_prefix(self) -> None:
        cfg = build_algotune("battery_scheduling")
        assert cfg.problem_dir == Path(
            "problems/algotune/algotune_battery_scheduling"
        )
        assert cfg.name == "algotune_battery_scheduling"

    def test_prefixed_name_passes_through(self) -> None:
        cfg = build_algotune("algotune_lp_box")
        assert cfg.problem_dir == Path("problems/algotune/algotune_lp_box")

    def test_explicit_name_override(self) -> None:
        cfg = build_algotune("chebyshev_center", name="custom_name")
        assert cfg.name == "custom_name"
        assert cfg.problem_dir == Path(
            "problems/algotune/algotune_chebyshev_center"
        )


class TestRealProblemDirsExist:
    """Defense-in-depth: every preset path that ships in the repo
    today must actually exist on disk. Catches a preset rename where
    the directory moves but the builder doesn't update."""

    @pytest.mark.parametrize(
        "builder",
        [
            build_hotpotqa_static_f1,
            build_hover,
            build_heilbron,
            build_toy_example,
            build_prompt_evolution,
        ],
    )
    def test_canonical_paths_exist(self, builder) -> None:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        cfg = builder()
        full = repo_root / cfg.problem_dir
        assert full.exists(), f"{full} missing — preset is stale"

    def test_algotune_battery_scheduling_exists(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        cfg = build_algotune("battery_scheduling")
        assert (repo_root / cfg.problem_dir).exists()
