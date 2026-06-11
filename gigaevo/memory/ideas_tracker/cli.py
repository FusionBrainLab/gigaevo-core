from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from typing import Any

from loguru import logger
from pydantic import SecretStr

from gigaevo.llm.models import ChatOpenAI, MultiModelRouter
from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
from gigaevo.memory.ideas_tracker.csv_loader import load_programs_from_csv
from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker
from gigaevo.memory.ideas_tracker.redis_loader import load_programs_from_redis


def build_router(model: str, base_url: str, api_key: str) -> MultiModelRouter:
    """Memory LLM router for standalone CLI runs (run.py composes it via Hydra)."""
    return MultiModelRouter(
        models=[
            ChatOpenAI(
                model=model,
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=0.0,
            )
        ],
        probabilities=[1.0],
        name="memory",
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ideas tracker independently from run.py using an existing Redis run database."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Directory for the final memory write pipeline's card-bank artefacts. "
            "Required unless --no-memory-write is given."
        ),
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help=(
            "Write ideas_tracker logs into this existing directory. "
            "A timestamped subfolder will be created per run."
        ),
    )
    parser.add_argument(
        "--memory-write",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable the final memory write pipeline (default: enabled).",
    )
    parser.add_argument("--redis-host", default=None, help="Redis host override.")
    parser.add_argument(
        "--redis-port", type=int, default=None, help="Redis port override."
    )
    parser.add_argument("--redis-db", type=int, default=None, help="Redis DB override.")
    parser.add_argument(
        "--redis-prefix",
        default=None,
        help="Redis key prefix override. This usually matches the problem name.",
    )
    parser.add_argument(
        "--redis-label",
        default=None,
        help="Optional Redis label override for logging/debugging.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="Analyzer LLM API key. Defaults to $OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-3-flash-preview",
        help="Analyzer LLM model identifier.",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL for the analyzer LLM.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help=(
            "Path to evolution_data.csv exported by tools/redis2pd.py. "
            "When provided, programs are loaded from the CSV instead of Redis."
        ),
    )
    parser.add_argument(
        "--higher-is-better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fitness direction of the analyzed run (default: higher is better).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.api_key:
        parser.error("--api-key not given and OPENROUTER_API_KEY is not set")
    memory_write_enabled = args.memory_write is None or bool(args.memory_write)
    if memory_write_enabled and args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required unless --no-memory-write is given")

    backend = LocalMemoryBackendFactory()
    logger.info(
        "[Memory][IdeaTracker][CLI] using local memory backend (checkpoint_dir={})",
        args.checkpoint_dir,
    )

    tracker_kwargs: dict[str, Any] = {}
    if args.memory_write is not None:
        tracker_kwargs["memory_write_enabled"] = bool(args.memory_write)

    tracker = IdeaTracker(
        llm=build_router(args.model, args.base_url, args.api_key),
        logs_dir=args.logs_dir,
        redis_prefix=args.redis_prefix or "",
        checkpoint_dir=args.checkpoint_dir,
        backend=backend,
        fitness_higher_is_better=args.higher_is_better,
        **tracker_kwargs,
    )
    if args.csv_path is not None:
        programs = load_programs_from_csv(args.csv_path)
    else:
        programs = load_programs_from_redis(
            host=args.redis_host or "localhost",
            port=args.redis_port or 6379,
            db=args.redis_db or 0,
            prefix=args.redis_prefix or "",
        )
    tracker.run(programs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
