"""Tests for the LLM prompt/response I/O dump handler.

The handler is a LangChain ``BaseCallbackHandler`` attached to every router
call (plain and structured). It records the exact messages sent and the text +
token usage returned, one complete JSON object per call, into a per-router
``.jsonl`` file under the run's output dir — a durable audit trail of "what was
the LLM input and what did it produce".
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from gigaevo.llm.io_dump import PromptIODumpHandler, create_io_dump_handler


def _result(text: str = "ANSWER") -> LLMResult:
    msg = AIMessage(
        content=text,
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    return LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model_name": "Qwen3-Test",
        },
    )


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestPromptIODumpHandler:
    def test_writes_complete_record_on_end(self, tmp_path: Path) -> None:
        handler = PromptIODumpHandler(dump_dir=tmp_path, router_name="mutation")
        rid = uuid4()
        handler.on_chat_model_start(
            {},
            [[SystemMessage(content="SYSTEM-RULES"), HumanMessage(content="SOLVE-X")]],
            run_id=rid,
        )
        handler.on_llm_end(_result("THE-OUTPUT"), run_id=rid)

        records = _read_records(tmp_path / "mutation.jsonl")
        assert len(records) == 1
        rec = records[0]
        assert rec["router"] == "mutation"
        assert rec["model"] == "Qwen3-Test"
        # Prompt captured verbatim, in order, with roles.
        joined = json.dumps(rec["prompt"])
        assert "SYSTEM-RULES" in joined
        assert "SOLVE-X" in joined
        # Output + token spend captured.
        assert "THE-OUTPUT" in rec["response"]
        assert rec["usage"]["total_tokens"] == 15
        assert rec["error"] is None

    def test_error_path_writes_record(self, tmp_path: Path) -> None:
        handler = PromptIODumpHandler(dump_dir=tmp_path, router_name="mem")
        rid = uuid4()
        handler.on_chat_model_start({}, [[HumanMessage(content="Q")]], run_id=rid)
        handler.on_llm_error(ValueError("boom"), run_id=rid)

        records = _read_records(tmp_path / "mem.jsonl")
        assert len(records) == 1
        assert "boom" in records[0]["error"]
        assert "Q" in json.dumps(records[0]["prompt"])

    def test_end_without_start_is_graceful(self, tmp_path: Path) -> None:
        handler = PromptIODumpHandler(dump_dir=tmp_path, router_name="r")
        handler.on_llm_end(_result(), run_id=uuid4())
        records = _read_records(tmp_path / "r.jsonl")
        assert len(records) == 1
        assert records[0]["prompt"] == []

    def test_multiple_calls_append(self, tmp_path: Path) -> None:
        handler = PromptIODumpHandler(dump_dir=tmp_path, router_name="r")
        for i in range(3):
            rid = uuid4()
            handler.on_chat_model_start(
                {}, [[HumanMessage(content=f"q{i}")]], run_id=rid
            )
            handler.on_llm_end(_result(f"a{i}"), run_id=rid)
        records = _read_records(tmp_path / "r.jsonl")
        assert len(records) == 3
        assert [r["response"] for r in records] == ["a0", "a1", "a2"]

    def test_non_chat_on_llm_start_captured(self, tmp_path: Path) -> None:
        handler = PromptIODumpHandler(dump_dir=tmp_path, router_name="r")
        rid = uuid4()
        handler.on_llm_start({}, ["RAW-STRING-PROMPT"], run_id=rid)
        handler.on_llm_end(_result(), run_id=rid)
        records = _read_records(tmp_path / "r.jsonl")
        assert "RAW-STRING-PROMPT" in json.dumps(records[0]["prompt"])


class TestFactory:
    def test_returns_none_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv("GIGAEVO_LLM_IO_DUMP_DIR", raising=False)
        assert create_io_dump_handler("mutation") is None

    def test_returns_handler_with_env(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("GIGAEVO_LLM_IO_DUMP_DIR", str(tmp_path))
        handler = create_io_dump_handler("mutation")
        assert isinstance(handler, PromptIODumpHandler)
