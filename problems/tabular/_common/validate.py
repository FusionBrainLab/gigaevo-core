from pathlib import Path
import sys


def _tabular_root(dataset_dir: Path) -> Path:
    for parent in dataset_dir.parents:
        if (parent / "_common").is_dir():
            return parent
    raise RuntimeError(f"cannot locate tabular/_common above {dataset_dir}")


def _dataset_id(dataset_dir: Path) -> str:
    marker = dataset_dir / "dataset_id.txt"
    return marker.read_text().strip() if marker.is_file() else dataset_dir.name


_DATASET_DIR = Path(sys.path[0]).resolve()
_COMMON_DIR = _tabular_root(_DATASET_DIR) / "_common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from tabular_problem import build  # noqa: E402

_PROBLEM = build(_dataset_id(_DATASET_DIR))


def validate(model_factory):
    return _PROBLEM.validate(model_factory)


def score_on_test(model_factory):
    return _PROBLEM.score_on_test(model_factory)
