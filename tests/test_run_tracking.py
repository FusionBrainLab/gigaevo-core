from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from run import _build_run_tracking_config


def test_build_run_tracking_config_uses_problem_name_and_start_time(
    tmp_path: Path,
):
    cfg = OmegaConf.create(
        {
            "problem": {"name": "demo/task name"},
            "redis": {"host": "localhost", "port": 6380, "db": 7},
            "tracking": {"enabled": True},
        }
    )

    tracking_config = _build_run_tracking_config(
        cfg,
        tmp_path,
        datetime(2026, 3, 20, 15, 4, 5),
    )

    assert tracking_config is not None
    assert tracking_config.csv_path == (
        tmp_path / "outputs" / "demo_task_name_20260320_150405.csv"
    )
    assert tracking_config.plot_output_dir == tmp_path / "outputs" / "run_plots"
    assert tracking_config.plot_output_stem == "demo_task_name_20260320_150405"
    assert tracking_config.redis_run_config.url() == "redis://localhost:6380/7"
    assert tracking_config.redis_run_config.display_label() == "demo/task name"
