from pathlib import Path
import sys

_DATASET_DIR = Path(sys.path[0])
_COMMON_DIR = _DATASET_DIR.parent / "_common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from tabular_problem import build  # noqa: E402

_PROBLEM = build(_DATASET_DIR.name)


def validate(model_factory):
    return _PROBLEM.validate(model_factory)


def score_on_test(model_factory):
    return _PROBLEM.score_on_test(model_factory)
