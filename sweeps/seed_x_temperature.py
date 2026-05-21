"""Cartesian-product sweep: three seeds against two LLM temperatures.

Run with::

    python -m gigaevo.sweep experiments/base.py sweeps/seed_x_temperature.py --parallel 3

Override paths use the ``ExperimentConfig`` field tree: ``--seed``
lives on the root, ``--llm.models.0.temperature`` reaches the first
``ChatOpenAIConfig`` inside an ``EnsembleRouterConfig``. Adjust the
override paths if the target experiment uses a different LLM router
shape.
"""

from __future__ import annotations

from itertools import product


def define_sweep() -> list[list[str]]:
    seeds = (1, 2, 3)
    temperatures = (0.3, 0.8)
    return [
        [
            "--seed", str(seed),
            "--llm.models.0.temperature", str(temperature),
        ]
        for seed, temperature in product(seeds, temperatures)
    ]
