"""Seed sweep: replay any experiment under four distinct seeds.

Run with::

    python -m gigaevo.sweep experiments/base.py sweeps/seeds.py

Each run lands in its own ``outputs/{experiment_id}/`` directory
because the resolved ``ExperimentConfig`` hash depends on ``seed``;
running with ``--parallel N`` fans the four runs across ``N``
processes.
"""

from __future__ import annotations


def define_sweep() -> list[list[str]]:
    return [["--seed", str(seed)] for seed in (1, 2, 3, 4)]
