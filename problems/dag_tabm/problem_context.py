"""TabM-flavoured problem context using the shared FeatureGraph ABI."""

from problems.dag_tab.problem_context import DagTabProblemContext


class DagTabMProblemContext(DagTabProblemContext):
    """FeatureGraph context whose local task text describes the TabM estimator."""
