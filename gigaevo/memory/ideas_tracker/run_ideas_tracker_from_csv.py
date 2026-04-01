from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from gigaevo.memory.ideas_tracker.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--source", "csv", *sys.argv[1:]]))
