"""Score a saved FeatureGraph JSON on the untouched test split with TabFM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .validate import score_on_test
except ImportError:
    from validate import score_on_test  # type: ignore[import-not-found, no-redef]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("program", type=Path)
    args = parser.parse_args()
    print(json.dumps(score_on_test(json.loads(args.program.read_text())), indent=2))


if __name__ == "__main__":
    main()
