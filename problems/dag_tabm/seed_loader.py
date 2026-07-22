"""Dynamic seed loader for the TabM FeatureGraph problem."""

from problems.dag_tab.seed_loader import DagTabSeedLoader


class DagTabMSeedLoader(DagTabSeedLoader):
    """Create the same raw-feature neutral seed used by ``dag_tab``."""
