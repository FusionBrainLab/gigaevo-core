"""Durable audit trail of LLM I/O.

A ``BaseCallbackHandler`` that records, for every router call (plain and
structured), the exact messages sent and the text + token usage returned —
one complete JSON object per call into a per-router ``.jsonl`` file under the
run's output dir. Env-gated (mirrors the langfuse handler): inert unless
``GIGAEVO_LLM_IO_DUMP_DIR`` is set, which ``run.py`` does from the Hydra
output dir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from loguru import logger

DUMP_DIR_ENV = "GIGAEVO_LLM_IO_DUMP_DIR"

_WRITE_LOCK = threading.Lock()


def _message_to_record(message: BaseMessage) -> dict[str, Any]:
    return {
        "role": getattr(message, "type", message.__class__.__name__),
        "content": message.content,
    }


def _prompt_from_messages(messages: list[list[BaseMessage]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for batch in messages:
        for message in batch:
            flat.append(_message_to_record(message))
    return flat


def _usage_from_result(result: LLMResult) -> dict[str, Any] | None:
    llm_output = result.llm_output or {}
    usage = llm_output.get("token_usage")
    if usage:
        return dict(usage)
    for batch in result.generations:
        for gen in batch:
            message = getattr(gen, "message", None)
            meta = getattr(message, "usage_metadata", None)
            if meta:
                return dict(meta)
    return None


def _response_text(result: LLMResult) -> str:
    parts: list[str] = []
    for batch in result.generations:
        for gen in batch:
            parts.append(gen.text or "")
    return "\n".join(p for p in parts if p)


class PromptIODumpHandler(BaseCallbackHandler):
    """Capture prompts on start, write the full record on end/error.

    Records are keyed by ``run_id`` between the start and end callbacks so the
    prompt and its response land in the same JSON object even under concurrent
    calls. Append is mutex-guarded for thread-safe interleaving.
    """

    raise_error = False

    def __init__(self, *, dump_dir: Path, router_name: str) -> None:
        self._path = Path(dump_dir) / f"{router_name}.jsonl"
        self._router_name = router_name
        self._pending: dict[UUID, list[dict[str, Any]]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._pending[run_id] = _prompt_from_messages(messages)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._pending[run_id] = [{"role": "raw", "content": p} for p in prompts]

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        prompt = self._pending.pop(run_id, [])
        llm_output = response.llm_output or {}
        self._write(
            {
                "router": self._router_name,
                "run_id": str(run_id),
                "model": llm_output.get("model_name"),
                "prompt": prompt,
                "response": _response_text(response),
                "usage": _usage_from_result(response),
                "error": None,
            }
        )

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        prompt = self._pending.pop(run_id, [])
        self._write(
            {
                "router": self._router_name,
                "run_id": str(run_id),
                "model": None,
                "prompt": prompt,
                "response": None,
                "usage": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _WRITE_LOCK:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def create_io_dump_handler(router_name: str) -> PromptIODumpHandler | None:
    """Build a handler if ``GIGAEVO_LLM_IO_DUMP_DIR`` is set, else None."""
    dump_dir = os.environ.get(DUMP_DIR_ENV)
    if not dump_dir:
        return None
    path = Path(dump_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("[LLM][IODump] cannot create {} ({}); disabling", path, exc)
        return None
    return PromptIODumpHandler(dump_dir=path, router_name=router_name)
