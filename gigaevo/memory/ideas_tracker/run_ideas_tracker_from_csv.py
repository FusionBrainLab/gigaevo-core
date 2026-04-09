"""
Entry point: run IdeaTracker on an evolution_data.csv archive.

Usage:
    python -m gigaevo.memory.ideas_tracker.run_ideas_tracker_from_csv \\
        --csv-path path/to/evolution_data.csv [cli.py options...]

The CSV must be produced by tools/redis2pd.py.  All other options
(--redis-prefix, --logs-dir, --no-memory-write, etc.) are forwarded
to the standard IdeaTracker CLI.
"""

from __future__ import annotations

from collections.abc import Sequence
import sys

from gigaevo.memory.ideas_tracker.cli import main as _cli_main


def main(argv: Sequence[str] | None = None) -> int:
    """Forward argv to cli.main; --csv-path is required but validated there."""
    args = list(argv) if argv is not None else sys.argv[1:]
    return _cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
