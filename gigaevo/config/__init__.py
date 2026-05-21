"""Typed configuration package.

The public surface lives in :mod:`gigaevo.config.schemas` and the
preset modules (``algorithm_presets``, ``engine_presets``,
``llm_presets``, ``pipeline_presets``, ``problem_presets``,
``runner_presets``, ``defaults``). Experiment files import from
those modules directly; this package's ``__init__`` intentionally
re-exports nothing.
"""
