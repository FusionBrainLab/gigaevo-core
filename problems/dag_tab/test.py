"""Score a saved FeatureGraph JSON on the existing tabular test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate import score_on_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("program", type=Path)
    args = parser.parse_args()
    print(json.dumps(score_on_test(json.loads(args.program.read_text())), indent=2))


if __name__ == "__main__":
    main()
