import importlib.util

# mmar-carl wheels need py>=3.12 (gated in pyproject); skip collection when absent
collect_ignore = (
    [] if importlib.util.find_spec("mmar_carl") else ["test_dag_changes.py"]
)
