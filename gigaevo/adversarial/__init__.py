"""Adversarial co-evolution pipeline components.

Provides:
- OpponentArchiveProvider: reads opponent programs from a MAP-Elites archive
- FetchOpponentResultsStage: executes opponents in parallel subprocesses
- AdversarialPipelineBuilder: standard pipeline + opponent fetching stage
"""
