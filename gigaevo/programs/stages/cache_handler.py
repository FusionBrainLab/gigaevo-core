"""Cache handlers for controlling stage re-execution behavior.

This module provides a CacheHandler abstraction that encapsulates caching policy,
separating it from stage execution logic.

Available handlers:
- AlwaysCached: Never rerun if FINAL result exists (default)
- NeverCached: Always rerun every DAG execution
- ProbabilisticCache: Rerun with configurable probability
- InputHashCache: Rerun only when inputs change
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gigaevo.programs.core_types import ProgramStageResult

from gigaevo.programs.core_types import FINAL_STATES


class CacheHandler(ABC):
    """Abstract base class for stage caching policies.

    CacheHandlers control when a stage should be re-executed vs using cached results.
    They can also augment results on completion (e.g., storing input hashes).
    """

    @abstractmethod
    def should_rerun(
        self,
        existing_result: ProgramStageResult | None,
        inputs_hash: str | None,
        finished_this_run: set[str],
    ) -> bool:
        """Determine if stage should be re-executed.

        Args:
            existing_result: Previous result for this stage (if any)
            inputs_hash: Hash of current inputs (computed by stage)
            finished_this_run: Set of stage names that finished in current DAG run

        Returns:
            True if stage should be re-executed, False to use cached result
        """
        pass

    def on_complete(
        self,
        result: ProgramStageResult,
        inputs_hash: str | None,
    ) -> ProgramStageResult:
        """Called when stage completes. Can augment result (e.g., store input hash).

        Args:
            result: The completed stage result
            inputs_hash: Hash of inputs used for this execution

        Returns:
            The (possibly modified) result
        """
        return result  # Default: no modification


class NeverCached(CacheHandler):
    """Always rerun on every DAG execution.

    Equivalent to cacheable=False. The stage will run every time
    the DAG is executed, regardless of previous results.
    """

    def should_rerun(
        self,
        existing_result: ProgramStageResult | None,
        inputs_hash: str | None,
        finished_this_run: set[str],
    ) -> bool:
        return True  # Always rerun


class ProbabilisticCache(CacheHandler):
    """Rerun with a configurable probability, even if cached.

    Useful for occasionally refreshing results or exploring
    whether re-execution produces different outcomes.

    Args:
        rerun_probability: Probability of re-running (0.0 to 1.0).
                          Default 0.1 means 10% chance of rerun.
    """

    def __init__(self, rerun_probability: float = 0.1):
        if not 0.0 <= rerun_probability <= 1.0:
            raise ValueError("rerun_probability must be between 0.0 and 1.0")
        self.rerun_probability = rerun_probability

    def should_rerun(
        self,
        existing_result: ProgramStageResult | None,
        inputs_hash: str | None,
        finished_this_run: set[str],
    ) -> bool:
        if not existing_result or existing_result.status not in FINAL_STATES:
            return True  # No cached result, must run

        # Cached result exists - rerun with probability
        return random.random() < self.rerun_probability


class InputHashCache(CacheHandler):
    """Rerun only when inputs have changed.

    Compares a hash of the current inputs with the hash stored in the
    previous result. If they differ, the stage is re-executed.
    """

    def should_rerun(
        self,
        existing_result: ProgramStageResult | None,
        inputs_hash: str | None,
        finished_this_run: set[str],
    ) -> bool:
        if not existing_result or existing_result.status not in FINAL_STATES:
            return True  # No cached result, must run
        # Compare stored hash with current inputs hash
        stored_hash = existing_result.input_hash
        if stored_hash is None:
            return True
        return inputs_hash != stored_hash  # Rerun if inputs changed

    def on_complete(
        self,
        result: ProgramStageResult,
        inputs_hash: str | None,
    ) -> ProgramStageResult:
        result.input_hash = inputs_hash
        return result


# Default instances for convenience
DEFAULT_CACHE = InputHashCache()
NO_CACHE = NeverCached()
