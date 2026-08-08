"""Tests for the agentic coding-harness backend.

Every test drives a fake harness — a small Python script honouring the same
workspace contract — so the suite needs no network and no harness installed.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tracemalloc
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import pytest

from gigaevo.llm import harness as harness_module
from gigaevo.llm.harness import (
    ANSWER_FILE,
    OUTPUT_FILE,
    SCHEMA_FILE,
    STDERR_FILE,
    STDOUT_FILE,
    SYSTEM_FILE,
    USER_FILE,
    HarnessChat,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.llm.token_tracking import TokenUsage
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Fake harness
# ---------------------------------------------------------------------------

_FAKE_HARNESS = """
import json, os, subprocess, sys, time
from pathlib import Path

# Always drain stdin first: a harness reads its prompt until EOF, so a backend
# that forgets to close stdin hangs here rather than silently "working".
Path("STDIN.txt").write_text(sys.stdin.read(), encoding="utf-8")

mode = os.environ.get("FAKE_MODE", "ok")

if mode == "hang":
    child = subprocess.Popen(["sleep", "300"])
    Path(os.environ["FAKE_PIDFILE"]).write_text(str(child.pid), encoding="utf-8")
    time.sleep(300)
    sys.exit(0)

if mode == "exit1":
    sys.stderr.write("fake harness failed\\n")
    sys.exit(1)

if mode == "exitenvelope":
    # A CLI that hits its own limit -- quota, auth, out of turns -- says so in
    # its stdout envelope, writes nothing to stderr, and exits nonzero.
    json.dump({"type": "result", "is_error": True, "subtype": "error_max_turns"}, sys.stdout)
    sys.stdout.flush()
    sys.exit(1)

if mode == "diesfast":
    # A rejected flag or expired auth: the leader dies at once, but the server
    # it already started is still in its process group.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(os.environ["FAKE_PIDFILE"]).write_text(str(child.pid), encoding="utf-8")
    sys.stderr.write("unknown option --nope\\n")
    sys.exit(3)

if mode == "deepjson":
    Path("OUTPUT.json").write_text("[" * 200000 + "]" * 200000, encoding="utf-8")
    sys.exit(0)

if mode == "bigstderr":
    # A wedged harness writes stderr for the whole timeout. Written sparsely so
    # the test costs no disk: the tail is what matters, not the bytes before it.
    os.lseek(2, int(os.environ["FAKE_STDERR_SIZE"]), os.SEEK_SET)
    os.write(2, b"the last thing it said\\n")
    sys.exit(7)

if mode == "fifostderr":
    # A harness with shell access swaps its own stderr log for a FIFO nothing
    # writes to. Opening that blocks until a writer appears -- which is never.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(os.environ["FAKE_PIDFILE"]).write_text(str(child.pid), encoding="utf-8")
    Path("STDERR.log").unlink()
    os.mkfifo("STDERR.log")
    time.sleep(300)
    sys.exit(0)

if mode == "nooutput":
    sys.exit(0)

if mode == "badjson":
    Path("OUTPUT.json").write_text("{not json", encoding="utf-8")
    sys.exit(0)

if mode == "notobject":
    Path("OUTPUT.json").write_text("[1, 2, 3]", encoding="utf-8")
    sys.exit(0)

if mode == "symlink":
    outside = Path(os.environ["FAKE_OUTSIDE"])
    outside.write_text(json.dumps({"text": "escaped"}), encoding="utf-8")
    Path("OUTPUT.json").symlink_to(outside)
    sys.exit(0)

# The next three modes fall through to the normal answer below: they exercise
# what the backend does around a SUCCESSFUL call.
if mode in ("escapee", "orphan"):
    # escapee leaves the process group, orphan stays in it.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=(mode == "escapee"),
    )
    Path(os.environ["FAKE_PIDFILE"]).write_text(str(child.pid), encoding="utf-8")

if mode == "bigstdout":
    chunk = "x" * 65536
    for _ in range(320):  # 20 MB
        sys.stdout.write(chunk)
    sys.stdout.flush()

if mode == "fifostdout":
    # The non-native twin of nativefifo: the answer still arrives in
    # OUTPUT.json, so only the silent usage read meets the FIFO.
    Path("STDOUT.log").unlink()
    os.mkfifo("STDOUT.log")

if mode == "usage":
    # The envelope `claude -p --output-format json` prints: the reply, the
    # token counts, and the cost the CLI would otherwise keep to itself.
    json.dump(
        {
            "type": "result",
            "result": "ok",
            "total_cost_usd": 0.0125,
            "num_turns": 6,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_creation_input_tokens": 9578,
                "cache_read_input_tokens": 21721,
                "service_tier": "standard",
            },
        },
        sys.stdout,
    )
    sys.stdout.flush()

if mode == "oddusage":
    # A harness reporting counts it should not: negative, boolean, absent.
    json.dump(
        {
            "usage": {"input_tokens": -5, "output_tokens": True},
            "total_cost_usd": "free",
            "num_turns": True,
        },
        sys.stdout,
    )
    sys.stdout.flush()

if mode == "hugecost":
    # JSON accepts integers float() cannot: a 501-digit cost is within the
    # json.loads digit limit but overflows the float conversion.
    json.dump(
        {"usage": {"input_tokens": 11, "output_tokens": 7}, "total_cost_usd": 10**500},
        sys.stdout,
    )
    sys.stdout.flush()

if mode == "prosestdout":
    sys.stdout.write("Thinking...\\nDone.\\n")
    sys.stdout.flush()

if mode == "noisy":
    sys.stderr.write("harness diagnostics: retrying tool call\\n")
    sys.stderr.flush()

if mode.startswith("pathschema"):
    # A codex-shaped CLI: the schema flag takes a file path, the CLI itself
    # writes the final message into the answer file, and stdout is a JSONL
    # event stream rather than a single envelope.
    flag = os.environ.get("FAKE_SCHEMA_FLAG", "--schema-path")
    schema_path = sys.argv[sys.argv.index(flag) + 1]
    argv_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Path("ARGV_SCHEMA.json").write_text(json.dumps(argv_schema), encoding="utf-8")
    Path("ARGV_SCHEMA_PATH.txt").write_text(schema_path, encoding="utf-8")
    oflag = os.environ.get("FAKE_ANSWER_FILE_FLAG", "-o")
    answer_path = Path(sys.argv[sys.argv.index(oflag) + 1])
    events = [
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.completed", "item": {"type": "reasoning"}},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1500,
                "cached_input_tokens": 1200,
                "output_tokens": 40,
            },
        },
    ]
    for event in events:
        sys.stdout.write(json.dumps(event) + "\\n")
    sys.stdout.flush()
    if mode == "pathschemamissing":
        sys.exit(0)
    wants_text = argv_schema.get("required") == ["text"]
    payload = (
        {"text": "ok"} if wants_text else json.loads(os.environ.get("FAKE_PAYLOAD", "{}"))
    )
    answer_path.write_text(json.dumps(payload), encoding="utf-8")
    sys.exit(0)

if mode.startswith("native"):
    # A harness given the schema on its command line answers with its own
    # envelope on stdout and writes no OUTPUT.json.
    if mode == "nativeprose":
        sys.stdout.write("Thinking...\\nDone.\\n")
        sys.exit(0)
    if mode == "nativebig":
        sys.stdout.write('{"answer": {"text": "' + "x" * (1 << 21) + '"}}')
        sys.exit(0)
    if mode == "nativefifo":
        # The stdout twin of fifostderr: the capture fd stays with the child,
        # so swapping the PATH only bites whoever opens it again by name.
        Path("STDOUT.log").unlink()
        os.mkfifo("STDOUT.log")
        sys.exit(0)
    if mode == "nativestdoutlink":
        outside = Path(os.environ["FAKE_OUTSIDE"])
        outside.write_text(
            json.dumps({"answer": {"text": "escaped"}}), encoding="utf-8"
        )
        Path("STDOUT.log").unlink()
        Path("STDOUT.log").symlink_to(outside)
        sys.exit(0)
    if mode == "nativedeepjson":
        sys.stdout.write("[" * 200000 + "]" * 200000)
        sys.exit(0)
    # The flag and key are env-configurable so a test can prove the backend
    # reads them from its instance rather than hardcoding the defaults.
    flag = os.environ.get("FAKE_SCHEMA_FLAG", "--schema")
    argv_schema = json.loads(sys.argv[sys.argv.index(flag) + 1])
    Path("ARGV_SCHEMA.json").write_text(json.dumps(argv_schema), encoding="utf-8")
    sflag = os.environ.get("FAKE_SYSTEM_FLAG", "--system")
    if sflag in sys.argv:
        Path("ARGV_SYSTEM.md").write_text(
            sys.argv[sys.argv.index(sflag) + 1], encoding="utf-8"
        )
    if mode == "nativemissing":
        json.dump({"usage": {"input_tokens": 3, "output_tokens": 4}}, sys.stdout)
        sys.exit(0)
    if mode == "nativenotobject":
        json.dump({"answer": [1, 2, 3]}, sys.stdout)
        sys.exit(0)
    wants_text = argv_schema.get("required") == ["text"]
    if mode == "nativedecoy":
        # A stray OUTPUT.json must lose to the envelope, not by absence but
        # by preference.
        Path("OUTPUT.json").write_text(
            json.dumps(
                {"text": "decoy"} if wants_text else {"archetype": "decoy", "score": 0}
            ),
            encoding="utf-8",
        )
    json.dump(
        {
            os.environ.get("FAKE_ANSWER_KEY", "answer"): {"text": "ok"}
            if wants_text
            else json.loads(os.environ.get("FAKE_PAYLOAD", "{}")),
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "total_cost_usd": 0.0125,
            "num_turns": 4,
        },
        sys.stdout,
    )
    sys.exit(0)

schema = json.loads(Path("SCHEMA.json").read_text(encoding="utf-8"))
is_text = schema.get("required") == ["text"] and "text" in schema.get("properties", {})
if is_text and mode != "ignoreschema":
    payload = {"text": "ok"}
else:
    payload = json.loads(os.environ.get("FAKE_PAYLOAD", "{}"))

tmp = Path("OUTPUT.json.tmp")
tmp.write_text(json.dumps(payload), encoding="utf-8")
tmp.rename("OUTPUT.json")
"""


class Detail(BaseModel):
    reason: str


class Answer(BaseModel):
    """Stand-in for the Pydantic schemas the real agents use."""

    archetype: str
    score: int = Field(ge=0)
    detail: Detail | None = None


@pytest.fixture
def fake_harness(tmp_path: Path) -> Path:
    script = tmp_path / "fake_harness.py"
    script.write_text(_FAKE_HARNESS, encoding="utf-8")
    return script


def _make_chat(fake_harness: Path, tmp_path: Path, **kwargs: Any) -> HarnessChat:
    params: dict[str, Any] = {
        "model_name": "fake-harness/v1",
        "command": [sys.executable, str(fake_harness)],
        "workspace_root": str(tmp_path / "ws"),
        "request_timeout": 30.0,
    }
    params.update(kwargs)
    return HarnessChat(**params)


def _workspaces(chat: HarnessChat) -> list[Path]:
    return sorted(p for p in chat._run_dir.iterdir() if p.is_dir())


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _reap(pidfile: Path) -> None:
    """Kill a deliberately-escaped test process, whatever the assertions did."""
    try:
        os.kill(int(pidfile.read_text()), 9)
    except (OSError, ValueError):
        pass


class _SpyHandler(BaseCallbackHandler):
    """Stands in for the Langfuse handler and the prompt I/O dump.

    Both reach the model only through the ``RunnableConfig``, so a path that
    drops the config loses tracing and the audit trail without failing.
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("start")

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("end")


# ---------------------------------------------------------------------------
# Workspace contract
# ---------------------------------------------------------------------------


class TestWorkspaceContract:
    def test_preflight_runs_the_harness_at_construction(self, fake_harness, tmp_path):
        """A misconfigured backend must fail at startup, not mid-run."""
        chat = _make_chat(fake_harness, tmp_path)
        assert len(_workspaces(chat)) == 1
        assert (_workspaces(chat)[0] / OUTPUT_FILE).is_file()

    def test_missing_command_raises_at_construction(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            HarnessChat(
                model_name="fake",
                command=["gigaevo-no-such-harness-binary"],
                workspace_root=str(tmp_path / "ws"),
            )

    def test_empty_command_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty command"):
            HarnessChat(
                model_name="fake", command=[], workspace_root=str(tmp_path / "ws")
            )

    def test_messages_split_into_system_and_user_files(self, fake_harness, tmp_path):
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = _make_chat(fake_harness, tmp_path)
        chat.invoke(
            [
                SystemMessage(content="you are a mutator"),
                HumanMessage(content="improve this program"),
            ]
        )
        workspace = _workspaces(chat)[-1]
        assert (workspace / SYSTEM_FILE).read_text() == "you are a mutator"
        user = (workspace / USER_FILE).read_text()
        assert "improve this program" in user
        assert "## human" in user

    def test_multiple_system_messages_are_joined(self, fake_harness, tmp_path):
        """Agents that append a second SystemMessage must not lose it."""
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = _make_chat(fake_harness, tmp_path)
        chat.invoke(
            [
                SystemMessage(content="rule one"),
                HumanMessage(content="body"),
                SystemMessage(content="rule two"),
            ]
        )
        workspace = _workspaces(chat)[-1]
        assert (workspace / SYSTEM_FILE).read_text() == "rule one\n\nrule two"
        assert (workspace / USER_FILE).read_text() == "## human\n\nbody"

    def test_block_list_content_is_flattened(self, fake_harness, tmp_path):
        """Reasoning and multimodal models return content as a list of blocks."""
        from langchain_core.messages import HumanMessage

        chat = _make_chat(fake_harness, tmp_path)
        chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "alpha "},
                        {"type": "text", "text": "beta"},
                    ]
                )
            ]
        )
        assert "alpha beta" in (_workspaces(chat)[-1] / USER_FILE).read_text()

    def test_system_only_conversation_leaves_an_empty_user_file(
        self, fake_harness, tmp_path
    ):
        """The files must always exist; the harness reads them unconditionally."""
        from langchain_core.messages import SystemMessage

        chat = _make_chat(fake_harness, tmp_path)
        chat.invoke([SystemMessage(content="only a system prompt")])
        workspace = _workspaces(chat)[-1]
        assert (workspace / USER_FILE).read_text() == ""
        assert (workspace / SYSTEM_FILE).read_text() == "only a system prompt"

    def test_a_lone_surrogate_in_a_prompt_does_not_strand_the_call(
        self, fake_harness, tmp_path
    ):
        """A truncated surrogate pair must not poison every call after it.

        ``json.loads`` accepts the escape, so such text rides out of a harness
        answer and into whatever archives it -- a card, an insight, a
        suggestion. UTF-8 cannot encode it, so every later prompt quoting that
        artifact would fail, permanently, and only under this backend: an HTTP
        one escapes it back to ASCII in the request body and never notices.
        The harness is handed that same escape.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.invoke("before \ud800 after")
        user = (_workspaces(chat)[-1] / USER_FILE).read_text(encoding="utf-8")
        assert "before \\ud800 after" in user
        assert json.dumps("before \ud800 after") == '"before \\ud800 after"'

    def test_identifying_params_name_the_backend(self, fake_harness, tmp_path):
        """LangChain keys caching and tracing on these; the command is identity."""
        chat = _make_chat(fake_harness, tmp_path)
        assert chat._identifying_params["model_name"] == "fake-harness/v1"
        assert chat._identifying_params["command"] == chat.command
        assert chat._llm_type == "gigaevo-harness"

    def test_preflight_rejects_a_harness_that_ignores_the_schema(
        self, fake_harness, tmp_path
    ):
        """Exiting 0 with a well-formed but wrong OUTPUT.json is the subtle failure.

        It is the one a real misconfiguration produces — a harness that answers
        in prose, or writes its own envelope — and it must be caught at startup
        rather than at the first mutation.
        """
        with pytest.raises(RuntimeError, match="preflight succeeded but"):
            _make_chat(
                fake_harness,
                tmp_path,
                env={"FAKE_MODE": "ignoreschema", "FAKE_PAYLOAD": '{"answer": "x"}'},
            )

    def test_instruction_reaches_the_harness_on_stdin(self, fake_harness, tmp_path):
        """stdin must be written and closed, or the harness waits forever."""
        chat = _make_chat(fake_harness, tmp_path)
        stdin_seen = (_workspaces(chat)[0] / "STDIN.txt").read_text()
        assert SCHEMA_FILE in stdin_seen
        assert OUTPUT_FILE in stdin_seen

    def test_prompts_dir_override_replaces_the_instruction(
        self, fake_harness, tmp_path
    ):
        """The instruction is a prompt, so `prompts.dir` must reach it."""
        custom = tmp_path / "prompts" / "harness"
        custom.mkdir(parents=True)
        (custom / "instruction.txt").write_text(
            "CUSTOM {system} {user} {schema} {output}", encoding="utf-8"
        )
        chat = _make_chat(fake_harness, tmp_path, prompts_dir=str(custom.parent))
        stdin_seen = (_workspaces(chat)[0] / "STDIN.txt").read_text()
        assert stdin_seen.startswith(f"CUSTOM {SYSTEM_FILE} {USER_FILE}")

    def test_unstructured_call_gets_a_text_schema(self, fake_harness, tmp_path):
        """Every call carries a schema, so the harness obeys exactly one rule."""
        chat = _make_chat(fake_harness, tmp_path)
        response = chat.invoke("hello")
        assert response.content == "ok"
        schema = json.loads((_workspaces(chat)[-1] / SCHEMA_FILE).read_text())
        assert schema["required"] == ["text"]

    def test_usage_metadata_is_zeroed_not_absent(self, fake_harness, tmp_path):
        """Zeros keep TokenTracker and the LLMCall event on their normal path."""
        chat = _make_chat(fake_harness, tmp_path)
        response = chat.invoke("hello")
        assert response.usage_metadata["total_tokens"] == 0
        assert response.response_metadata["token_usage"]["prompt_tokens"] == 0

    def test_workspace_is_the_harness_working_directory(self, fake_harness, tmp_path):
        """cwd is the workspace, which keeps a shell-capable agent out of the repo."""
        chat = _make_chat(fake_harness, tmp_path)
        workspace = _workspaces(chat)[0]
        assert (workspace / "STDIN.txt").is_file()

    def test_each_call_gets_its_own_workspace(self, fake_harness, tmp_path):
        chat = _make_chat(fake_harness, tmp_path)
        chat.invoke("one")
        chat.invoke("two")
        assert len(_workspaces(chat)) == 3  # preflight + 2

    def test_env_extends_the_parent_environment(self, tmp_path):
        """``env`` layers onto the parent environment, it does not replace it.

        ``config/llm/harness.yaml`` sets one variable, ``CLAUDE_CONFIG_DIR``.
        A harness spawned without the rest of the environment loses ``PATH``
        and ``HOME``, so it cannot find its own credentials or its MCP servers
        -- and it fails at run time, on the first call, not at construction.
        """
        chat = _make_chat(
            None,
            tmp_path,
            command=[
                "/bin/sh",
                "-c",
                'cat >/dev/null; printf \'{"text":"%s|%s"}\' '
                '"$GIGAEVO_MARKER" "${HOME:-<unset>}" > OUTPUT.json',
            ],
        )
        chat.env["GIGAEVO_MARKER"] = "injected"
        marker, home = chat.invoke("go").content.split("|")
        assert marker == "injected"
        assert home == os.environ["HOME"]


# ---------------------------------------------------------------------------
# Structured output — the three schema shapes callers actually pass
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_pydantic_schema_round_trips_to_a_model(self, fake_harness, tmp_path):
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "refactor", "score": 3})
        result = chat.with_structured_output(Answer).invoke("go")
        assert isinstance(result, Answer)
        assert result.archetype == "refactor"

    def test_pydantic_schema_is_written_portable(self, fake_harness, tmp_path):
        """Nested models arrive as $defs/$ref; the harness gets them inlined."""
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": 1})
        chat.with_structured_output(Answer).invoke("go")
        schema = json.loads((_workspaces(chat)[-1] / SCHEMA_FILE).read_text())
        assert "$defs" not in schema
        assert "$ref" not in json.dumps(schema)
        assert set(schema["required"]) == {"archetype", "score"}

    def test_raw_dict_schema_returns_a_dict(self, fake_harness, tmp_path):
        """program_author / equivalence / card_author pass portable dicts."""
        raw = {
            "type": "object",
            "properties": {"idea": {"type": "string"}},
            "required": ["idea"],
        }
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"idea": "swap the loss"})
        result = chat.with_structured_output(raw).invoke("go")
        assert result == {"idea": "swap the loss"}

    def test_named_envelope_schema_is_unwrapped(self, fake_harness, tmp_path):
        """structured_diff passes OpenAI's {"name", "schema"} envelope."""
        inner = {
            "type": "object",
            "properties": {"diff": {"type": "string"}},
            "required": ["diff"],
        }
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"diff": "-a\n+b"})
        result = chat.with_structured_output({"name": "Diff", "schema": inner}).invoke(
            "go"
        )
        assert result == {"diff": "-a\n+b"}
        written = json.loads((_workspaces(chat)[-1] / SCHEMA_FILE).read_text())
        assert written == inner

    def test_function_calling_envelope_is_unwrapped(self, fake_harness, tmp_path):
        """The router rewrites the envelope to {"name", "parameters"} for that method."""
        inner = {
            "type": "object",
            "properties": {"diff": {"type": "string"}},
            "required": ["diff"],
        }
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"diff": "-a\n+b"})
        result = chat.with_structured_output(
            {"name": "Diff", "parameters": inner}, method="function_calling"
        ).invoke("go")
        assert result == {"diff": "-a\n+b"}
        written = json.loads((_workspaces(chat)[-1] / SCHEMA_FILE).read_text())
        assert written == inner

    def test_schema_carrying_its_own_parameters_key_is_not_unwrapped(
        self, fake_harness, tmp_path
    ):
        """Envelope detection must not eat a schema that owns the same key.

        Unwrapping here would hand the harness the wrong schema and then
        validate the answer against it, so anything at all would be accepted.
        """
        raw = {
            "type": "object",
            "parameters": {"type": "object", "properties": {"wrong": {}}},
            "properties": {"idea": {"type": "string"}},
            "required": ["idea"],
        }
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"idea": "swap the loss"})
        result = chat.with_structured_output(raw).invoke("go")
        assert result == {"idea": "swap the loss"}
        assert json.loads((_workspaces(chat)[-1] / SCHEMA_FILE).read_text()) == raw

    def test_include_raw_shape_matches_the_router_contract(
        self, fake_harness, tmp_path
    ):
        """MultiModelRouter always asks for include_raw=True and reads both keys."""
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": 1})
        result = chat.with_structured_output(Answer, include_raw=True).invoke("go")
        assert isinstance(result["parsed"], Answer)
        assert result["raw"].usage_metadata["total_tokens"] == 0
        assert result["parsing_error"] is None

    def test_include_raw_hands_back_a_parse_failure_instead_of_raising(
        self, fake_harness, tmp_path
    ):
        """The router tracks tokens off `raw` before it raises; a backend that
        raises out of the wrapper loses the tokens of exactly the failed calls.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": -3})
        result = chat.with_structured_output(Answer, include_raw=True).invoke("go")

        assert result["parsed"] is None
        assert isinstance(result["parsing_error"], ValueError)
        assert '"score": -3' in result["raw"].content

    def test_negotiation_kwargs_are_accepted_and_ignored(self, fake_harness, tmp_path):
        """There is no wire format to negotiate — SCHEMA.json is the contract."""
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": 1})
        result = chat.with_structured_output(Answer, method="function_calling").invoke(
            "go"
        )
        assert isinstance(result, Answer)

    def test_unsupported_schema_type_rejected(self, fake_harness, tmp_path):
        chat = _make_chat(fake_harness, tmp_path)
        with pytest.raises(TypeError, match="unsupported structured-output schema"):
            chat.with_structured_output("not-a-schema")


# ---------------------------------------------------------------------------
# Failure contract — every failure is a ValueError, like the HTTP path
# ---------------------------------------------------------------------------


class TestFailureContract:
    @pytest.mark.parametrize(
        ("mode", "match"),
        [
            ("exit1", "exited 1"),
            ("nooutput", f"wrote no {OUTPUT_FILE}"),
            ("badjson", "not valid JSON"),
            ("notobject", "must hold a JSON object"),
            # Nesting deep enough to exhaust the C stack: json.loads raises
            # RecursionError here, which is not a ValueError.
            ("deepjson", "not valid JSON"),
        ],
    )
    def test_harness_failures_raise_value_error(
        self, fake_harness, tmp_path, mode, match
    ):
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = mode
        with pytest.raises(ValueError, match=match):
            chat.invoke("go")

    @pytest.mark.parametrize("call", ["invoke", "ainvoke"])
    async def test_unwritable_stderr_log_is_an_infrastructure_failure(
        self, fake_harness, tmp_path, monkeypatch, call
    ):
        """A full disk or an fd ceiling must not read as an invalid program."""
        real_open = Path.open

        def refuse_stderr(self, *args, **kwargs):
            if self.name == STDERR_FILE:
                raise PermissionError(13, "Permission denied")
            return real_open(self, *args, **kwargs)

        chat = _make_chat(fake_harness, tmp_path)
        monkeypatch.setattr(Path, "open", refuse_stderr)
        with pytest.raises(ValueError, match="infrastructure failure"):
            if call == "invoke":
                chat.invoke("go")
            else:
                await chat.ainvoke("go")

    def test_stderr_is_carried_into_the_error(self, fake_harness, tmp_path):
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "exit1"
        with pytest.raises(ValueError, match="fake harness failed"):
            chat.invoke("go")

    def test_a_nonzero_exit_quotes_stdout_where_a_cli_reports_errors(
        self, fake_harness, tmp_path
    ):
        """`claude -p` exits nonzero with stderr EMPTY: quota, auth and
        max-turns all land in the stdout envelope, so an error that quoted
        only stderr would say nothing.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "exitenvelope"
        with pytest.raises(ValueError, match="error_max_turns"):
            chat.invoke("go")

    def test_a_huge_stderr_log_costs_only_the_excerpt(self, fake_harness, tmp_path):
        """Keeping the last 2 KB must not spend the whole file.

        Nothing bounds STDERR.log: the child owns the fd, and the failure paths
        that read it are exactly the ones a runaway harness reaches -- it wrote
        for the full request_timeout, which is why it is being killed. Reading
        it whole spends its size in this process, freezes the loop for every
        other concurrent call while it does, and under a memory cap raises
        MemoryError, which is not a ValueError and so leaves the contract
        entirely -- booking a live harness as an invalid program.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "bigstderr"
        chat.env["FAKE_STDERR_SIZE"] = str(256 * 1024 * 1024)

        tracemalloc.start()
        try:
            with pytest.raises(ValueError, match="harness exited 7") as caught:
                chat.invoke("go")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert "the last thing it said" in str(caught.value)
        assert peak < 16 * 1024 * 1024, f"read {peak / 1e6:.0f} MB to keep 2 KB"

    def test_missing_required_field_raises_value_error(self, fake_harness, tmp_path):
        raw = {
            "type": "object",
            "properties": {"idea": {"type": "string"}},
            "required": ["idea"],
        }
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"unrelated": 1})
        with pytest.raises(ValueError, match="missing required fields"):
            chat.with_structured_output(raw).invoke("go")

    def test_schema_violation_raises_value_error(self, fake_harness, tmp_path):
        """Pydantic's ValidationError is a ValueError, so callers need no change."""
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": -5})
        with pytest.raises(ValueError):
            chat.with_structured_output(Answer).invoke("go")

    def test_unstructured_output_without_text_field_raises(
        self, fake_harness, tmp_path
    ):
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "ignoreschema"
        chat.env["FAKE_PAYLOAD"] = json.dumps({"answer": "ok"})
        with pytest.raises(ValueError, match="no string 'text' field"):
            chat.invoke("go")


# ---------------------------------------------------------------------------
# Process control
# ---------------------------------------------------------------------------


class TestProcessControl:
    def test_timeout_raises_and_kills_the_process_tree(self, fake_harness, tmp_path):
        """Harnesses spawn MCP children; killing only the leader leaks them."""
        pidfile = tmp_path / "grandchild.pid"
        chat = _make_chat(fake_harness, tmp_path, request_timeout=2.0)
        chat.env["FAKE_MODE"] = "hang"
        chat.env["FAKE_PIDFILE"] = str(pidfile)

        started = time.monotonic()
        with pytest.raises(ValueError, match="timed out after"):
            chat.invoke("go")
        elapsed = time.monotonic() - started
        assert elapsed < 20.0

        grandchild = int(pidfile.read_text())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"grandchild {grandchild} survived the process-tree kill")

    async def test_fast_failure_kills_the_group_it_left_behind(self, tmp_path):
        """The async path must survive the leader being reaped before the kill.

        asyncio reaps a child the moment it exits, freeing its pid while the
        process group -- the servers it started -- lives on. Deriving the group
        from the leader at that point raises ``ProcessLookupError``, which is an
        ``OSError``, not a ``ValueError``: the kill is skipped, the group leaks,
        and the caller gets a bare errno instead of the exit code and stderr.

        The harness here is ``/bin/sh``, not the Python fake: the reap has to
        win a race against ``create_subprocess_exec`` returning, and a Python
        interpreter takes far too long to start and exit to lose it. A real
        harness rejecting a flag or finding no credentials dies this fast.
        Concurrency is what makes the race deterministic -- one call at a time
        usually wins it, a loaded loop loses it every time.

        The failing branch must not read stdin, for the same reason: a harness
        that rejects a flag dies while parsing it, long before the prompt is
        written. Draining stdin first would hold the child alive until
        ``communicate`` closes the pipe, which is past the point of the race.
        """
        calls = 8
        pidfiles = [tmp_path / f"server-{i}.pid" for i in range(calls)]
        chats = []
        for pidfile in pidfiles:
            chat = _make_chat(
                None,
                tmp_path,
                command=[
                    "/bin/sh",
                    "-c",
                    '[ -n "$FAKE_PIDFILE" ] || { cat >/dev/null; '
                    'printf \'{"text":"ok"}\' > OUTPUT.json; exit 0; }; '
                    'sleep 300 & echo $! > "$FAKE_PIDFILE"; '
                    "echo 'unknown option --nope' >&2; exit 3",
                ],
            )
            chat.env["FAKE_PIDFILE"] = str(pidfile)
            chats.append(chat)

        try:
            outcomes = await asyncio.gather(
                *(c.ainvoke("go") for c in chats), return_exceptions=True
            )
            for outcome in outcomes:
                assert isinstance(outcome, ValueError), outcome
                assert "harness exited 3" in str(outcome)
            for pidfile in pidfiles:
                assert _wait_gone(int(pidfile.read_text()))
        finally:
            for pidfile in pidfiles:
                _reap(pidfile)

    async def test_kill_does_not_depend_on_looking_up_the_group(
        self, fake_harness, tmp_path, monkeypatch
    ):
        """Nothing may consult the leader's pid after it may already be reaped.

        Deterministic twin of the test above: ``getpgid`` raises exactly what
        the kernel raises once the leader is gone.
        """

        def already_reaped(pid: int) -> int:
            raise ProcessLookupError(3, "No such process")

        monkeypatch.setattr(harness_module.os, "getpgid", already_reaped)
        pidfile = tmp_path / "server.pid"
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "diesfast"
        chat.env["FAKE_PIDFILE"] = str(pidfile)
        try:
            with pytest.raises(ValueError, match="harness exited 3"):
                await chat.ainvoke("go")
            assert _wait_gone(int(pidfile.read_text()))
        finally:
            _reap(pidfile)

    async def test_async_path_keeps_file_io_off_the_event_loop(
        self, fake_harness, tmp_path, monkeypatch
    ):
        """Blocking reads and writes belong on a thread, not on the loop.

        The async path exists so that concurrent calls make progress. The
        prompt and the answer are ordinary blocking file I/O, and on the shared
        storage these workspaces are told to live on, a single one of them can
        take long enough to stall every other in-flight call -- which is the
        one thing the async path was added to prevent.

        Asserted by which thread the I/O runs on rather than by timing it: the
        stall is real but its size is the filesystem's to decide, so a
        threshold would only be measuring the machine's load.
        """
        loop_thread = threading.current_thread()
        seen: dict[str, threading.Thread] = {}

        def spy(name: str):
            original = getattr(HarnessChat, name)

            def wrapper(self, *args, **kwargs):
                seen[name] = threading.current_thread()
                return original(self, *args, **kwargs)

            monkeypatch.setattr(HarnessChat, name, wrapper)

        chat = _make_chat(fake_harness, tmp_path)
        spy("_write_workspace")
        spy("_read_output")
        await chat.ainvoke("go")

        assert set(seen) == {"_write_workspace", "_read_output"}
        for name, thread in seen.items():
            assert thread is not loop_thread, f"{name} ran on the event loop"

    async def test_an_unreadable_stderr_log_still_lets_the_kill_happen(
        self, fake_harness, tmp_path
    ):
        """Reading the excerpt must never be able to outlive the timeout.

        The excerpt is built inside the ``except``, before the ``finally`` that
        kills the group -- so anything that blocks the read holds off the kill
        the timeout exists to perform. A FIFO is the cheap way to block an
        ``open`` forever, and a harness with shell access can leave one here.
        Being a sync syscall on the loop thread, it takes every other in-flight
        call down with it and ``asyncio.wait_for`` cannot break it.
        """
        pidfile = tmp_path / "server.pid"
        chat = _make_chat(fake_harness, tmp_path, request_timeout=2.0)
        chat.env["FAKE_MODE"] = "fifostderr"
        chat.env["FAKE_PIDFILE"] = str(pidfile)
        try:
            with pytest.raises(ValueError, match="timed out after"):
                await asyncio.wait_for(chat.ainvoke("go"), timeout=30.0)
            assert _wait_gone(int(pidfile.read_text())), (
                "grandchild survived the FIFO-stderr kill"
            )
        finally:
            _reap(pidfile)

    def test_stderr_tail_returns_promptly_on_a_writerless_fifo(
        self, fake_harness, tmp_path
    ):
        """Fast companion to the kill-chain test above, pinning the read itself.

        The integration test is the one that proves the kill still lands on
        time, but it pays a spawn, a timeout and a reap grace to say so. This
        one names the function that regressed, in milliseconds. A FIFO with no
        writer is the shape that blocks: an ``open`` waits for the first writer,
        so a slow-but-present writer would not reproduce it.
        """
        chat = _make_chat(fake_harness, tmp_path)
        workspace = tmp_path / "fifo_probe"
        workspace.mkdir()
        os.mkfifo(workspace / STDERR_FILE)

        started = time.monotonic()
        assert chat._stderr_tail(workspace) == ""
        assert time.monotonic() - started < 1.0

    def test_an_unopenable_stderr_log_does_not_leak_a_descriptor(
        self, fake_harness, tmp_path
    ):
        """The excerpt runs on every failure, so its error path must not leak.

        ``os.open`` + ``os.fdopen`` is not ``open()``: a descriptor handed to
        ``fdopen`` is disowned before the close on the constructor's error
        path, so a failure there orphans it. The outer handler returns "" and
        says nothing, which makes the leak silent and unbounded in the number
        of failed calls. A directory at ``STDERR.log`` opens fine and then
        raises ``IsADirectoryError`` out of ``fdopen`` -- an ``OSError``, so
        every visible symptom is identical to a clean miss.
        """
        chat = _make_chat(fake_harness, tmp_path)
        workspace = tmp_path / "probe"
        workspace.mkdir()
        (workspace / STDERR_FILE).mkdir()

        before = len(os.listdir("/proc/self/fd"))
        for _ in range(50):
            assert chat._stderr_tail(workspace) == ""
        assert len(os.listdir("/proc/self/fd")) == before

    def test_parent_process_survives_the_kill(self, fake_harness, tmp_path):
        """start_new_session detaches the harness; without it killpg hits us."""
        chat = _make_chat(fake_harness, tmp_path, request_timeout=2.0)
        chat.env["FAKE_MODE"] = "hang"
        chat.env["FAKE_PIDFILE"] = str(tmp_path / "pid")
        with pytest.raises(ValueError, match="timed out after"):
            chat.invoke("go")
        # Still executing, and still able to run a subprocess.
        assert (
            subprocess.run([sys.executable, "-c", "pass"], check=False).returncode == 0
        )

    def test_call_does_not_hang_when_harness_reads_stdin(self, fake_harness, tmp_path):
        """Regression guard: an unclosed stdin turns every call into a timeout."""
        started = time.monotonic()
        chat = _make_chat(fake_harness, tmp_path, request_timeout=30.0)
        chat.invoke("go")
        assert time.monotonic() - started < 20.0

    def test_escaped_descendant_does_not_stall_the_call(self, fake_harness, tmp_path):
        """The call waits for the harness to exit, not for its descendants.

        A harness that leaves a daemon behind (its own session, inherited
        handles) is normal. Waiting on a stream instead of on the process
        turned every such call into a timeout, and the answer was already on
        disk.
        """
        pidfile = tmp_path / "escapee.pid"
        chat = _make_chat(fake_harness, tmp_path, request_timeout=20.0)
        chat.env["FAKE_MODE"] = "escapee"
        chat.env["FAKE_PIDFILE"] = str(pidfile)
        try:
            started = time.monotonic()
            assert chat.invoke("go").content == "ok"
            assert time.monotonic() - started < 15.0
        finally:
            _reap(pidfile)

    def test_in_group_children_are_reaped_after_a_successful_call(
        self, fake_harness, tmp_path
    ):
        """Success leaks as easily as failure: MCP servers outlive the leader."""
        pidfile = tmp_path / "orphan.pid"
        chat = _make_chat(fake_harness, tmp_path, request_timeout=20.0)
        chat.env["FAKE_MODE"] = "orphan"
        chat.env["FAKE_PIDFILE"] = str(pidfile)
        try:
            assert chat.invoke("go").content == "ok"
            assert _wait_gone(int(pidfile.read_text())), "child survived a clean call"
        finally:
            _reap(pidfile)

    def test_large_harness_stdout_is_not_accumulated(self, fake_harness, tmp_path):
        """Stdout goes to a file, so it must not be buffered into this process."""
        chat = _make_chat(fake_harness, tmp_path, request_timeout=60.0)
        chat.env["FAKE_MODE"] = "bigstdout"
        started = time.monotonic()
        assert chat.invoke("go").content == "ok"
        assert time.monotonic() - started < 30.0

    def test_token_counts_come_from_a_json_envelope_on_stdout(
        self, fake_harness, tmp_path
    ):
        """Cache traffic is billed input, so it belongs in the input total.

        Counting only ``input_tokens`` would report 11 for a call that was
        charged for 31,310 — the cached prefix is most of an agentic harness's
        spend, and it is the number a run's cost is read off.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "usage"
        usage = chat.invoke("go").usage_metadata

        assert usage["input_tokens"] == 11 + 9578 + 21721
        assert usage["output_tokens"] == 7
        assert usage["total_tokens"] == 11 + 9578 + 21721 + 7
        assert usage["input_token_details"] == {
            "cache_creation": 9578,
            "cache_read": 21721,
        }

    def test_token_counts_reach_the_tracker_through_response_metadata(
        self, fake_harness, tmp_path
    ):
        """``TokenTracker`` reads the OpenAI-shaped channel, not usage_metadata.

        Both are populated by the same call, and only this one buckets tokens
        per stage, so a backend that filled just ``usage_metadata`` would show
        real numbers in the dump and zeros in every run report.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "usage"
        tracked = TokenUsage.from_response(chat.invoke("go"))

        assert tracked is not None
        assert tracked.context == 11 + 9578 + 21721
        assert tracked.generated == 7
        assert tracked.total == 11 + 9578 + 21721 + 7

    def test_reported_cost_is_carried_on_the_message(self, fake_harness, tmp_path):
        """A CLI on a subscription is the only thing that knows what it spent."""
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "usage"
        assert chat.invoke("go").response_metadata["total_cost_usd"] == 0.0125

    def test_reported_turn_count_is_carried_on_the_message(
        self, fake_harness, tmp_path
    ):
        """Turn count is the dominant cost variable; losing it here would leave
        the fleet nothing to monitor it by.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "usage"
        assert chat.invoke("go").response_metadata["num_turns"] == 6

    @pytest.mark.parametrize("mode", ["prosestdout", "default", "bigstdout"])
    def test_a_harness_that_reports_no_usage_still_answers(
        self, fake_harness, tmp_path, mode
    ):
        """Most harnesses will never report counts, and that is not an error.

        Prose on stdout, silence, and an unbounded agentic log all have to land
        on the same zeros the backend reported before counts existed — the call
        itself must not fail, and the file must not be parsed to find out.
        """
        chat = _make_chat(fake_harness, tmp_path, request_timeout=60.0)
        chat.env["FAKE_MODE"] = mode
        started = time.monotonic()
        message = chat.invoke("go")

        assert message.content == "ok"
        assert message.usage_metadata == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        assert "total_cost_usd" not in message.response_metadata
        assert time.monotonic() - started < 30.0

    def test_a_fifo_swapped_into_stdout_yields_zeros_not_a_hang(
        self, fake_harness, tmp_path
    ):
        """Silent must also mean prompt: this read runs inside the router's
        semaphore slot, so a blocked open here stops the whole backend once
        every slot holds one.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "fifostdout"
        started = time.monotonic()
        message = chat.invoke("go")

        assert message.content == "ok"
        assert message.usage_metadata["input_tokens"] == 0
        assert time.monotonic() - started < 5.0

    def test_nonsense_counts_are_discarded_rather_than_believed(
        self, fake_harness, tmp_path
    ):
        """A negative or boolean count would corrupt every cumulative total.

        ``bool`` is an ``int`` subclass, so ``True`` would silently book as one
        output token; a negative would subtract from the run's spend.
        """
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "oddusage"
        message = chat.invoke("go")

        assert message.usage_metadata["input_tokens"] == 0
        assert message.usage_metadata["output_tokens"] == 0
        assert "total_cost_usd" not in message.response_metadata
        assert "num_turns" not in message.response_metadata

    def test_a_cost_too_big_for_a_float_is_dropped_not_raised(
        self, fake_harness, tmp_path
    ):
        """The counts beside a lying cost are still good — keep them."""
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "hugecost"
        message = chat.invoke("go")

        assert message.content == "ok"
        assert message.usage_metadata["input_tokens"] == 11
        assert "total_cost_usd" not in message.response_metadata

    def test_stdout_is_kept_beside_the_workspace(self, fake_harness, tmp_path):
        """The envelope stays on disk: it is the receipt for what the call cost."""
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "usage"
        chat.invoke("go")
        envelope = json.loads((_workspaces(chat)[-1] / STDOUT_FILE).read_text())
        assert envelope["total_cost_usd"] == 0.0125

    def test_stderr_is_kept_beside_the_workspace(self, fake_harness, tmp_path):
        """A successful-but-noisy call still leaves its diagnostics on disk."""
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "noisy"
        chat.invoke("go")
        stderr = (_workspaces(chat)[-1] / STDERR_FILE).read_text()
        assert "retrying tool call" in stderr

    async def test_cancelling_a_call_kills_the_harness(self, fake_harness, tmp_path):
        """A cancelled DAG must not leave a harness holding a semaphore slot.

        The slot is released the moment the task is cancelled, so a surviving
        process runs alongside its own replacement and breaks the concurrency
        cap exactly when the machine is loaded.
        """
        pidfile = tmp_path / "grandchild.pid"
        chat = _make_chat(fake_harness, tmp_path, request_timeout=300.0)
        chat.env["FAKE_MODE"] = "hang"
        chat.env["FAKE_PIDFILE"] = str(pidfile)

        task = asyncio.create_task(chat.ainvoke("go"))
        deadline = time.monotonic() + 20.0
        while not pidfile.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert pidfile.exists(), "harness never started"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            assert _wait_gone(int(pidfile.read_text())), "harness survived cancellation"
        finally:
            _reap(pidfile)


# ---------------------------------------------------------------------------
# Observability — the config carries Langfuse and the prompt I/O dump
# ---------------------------------------------------------------------------


class TestCallbackPropagation:
    def test_callbacks_fire_on_the_plain_path(self, fake_harness, tmp_path):
        spy = _SpyHandler()
        chat = _make_chat(fake_harness, tmp_path)
        chat.invoke("go", config={"callbacks": [spy]})
        assert spy.events == ["start", "end"]

    def test_callbacks_fire_on_the_structured_path(self, fake_harness, tmp_path):
        """Nearly all GigaEvo traffic is structured; it must not be untraced."""
        spy = _SpyHandler()
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": 1})
        chat.with_structured_output(Answer).invoke("go", config={"callbacks": [spy]})
        assert spy.events == ["start", "end"]

    async def test_callbacks_fire_on_the_async_structured_path(
        self, fake_harness, tmp_path
    ):
        spy = _SpyHandler()
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": 1})
        await chat.with_structured_output(Answer).ainvoke(
            "go", config={"callbacks": [spy]}
        )
        assert spy.events == ["start", "end"]


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


class TestContainment:
    def test_output_symlink_escaping_the_workspace_is_rejected(
        self, fake_harness, tmp_path
    ):
        """A symlink is how a harness reads back a file it was never given."""
        outside = tmp_path / "outside.json"
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "symlink"
        chat.env["FAKE_OUTSIDE"] = str(outside)
        with pytest.raises(ValueError, match="resolves outside its workspace"):
            chat.invoke("go")

    def test_default_workspace_root_is_private_and_unguessable(self, fake_harness):
        """A shared well-known path under /tmp is a pre-plantable symlink."""
        chat = HarnessChat(
            model_name="fake-harness/v1", command=[sys.executable, str(fake_harness)]
        )
        assert chat._run_dir.is_dir()
        assert not chat._run_dir.is_symlink()
        assert oct(chat._run_dir.stat().st_mode)[-3:] == "700"

    def test_configured_workspace_root_is_private_too(self, fake_harness, tmp_path):
        """Setting the root explicitly must not forfeit the private-workspace guarantee."""
        chat = HarnessChat(
            model_name="fake-harness/v1",
            command=[sys.executable, str(fake_harness)],
            workspace_root=str(tmp_path / "ws"),
        )
        assert oct(chat._run_dir.stat().st_mode)[-3:] == "700"

    def test_two_chats_sharing_a_root_do_not_share_a_run_directory(
        self, fake_harness, tmp_path
    ):
        """A pid-derived name collides, and both chats then hand out call 000000.

        The second chat's prompt lands on top of the first chat's live one, and
        the first chat reads back an answer to a question it never asked.
        """
        root = str(tmp_path / "ws")
        first = _make_chat(fake_harness, tmp_path, workspace_root=root)
        second = _make_chat(fake_harness, tmp_path, workspace_root=root)
        assert first._run_dir != second._run_dir

        first.invoke("first question")
        second.invoke("second question")
        asked = {
            (ws / USER_FILE).read_text(encoding="utf-8")
            for ws in _workspaces(first) + _workspaces(second)
        }
        assert any("first question" in text for text in asked)
        assert any("second question" in text for text in asked)


# ---------------------------------------------------------------------------
# Integration with MultiModelRouter
# ---------------------------------------------------------------------------


class TestRouterIntegration:
    def _router(self, chat: HarnessChat) -> MultiModelRouter:
        return MultiModelRouter(
            [chat],
            [1.0],
            writer=NullWriter(),
            name="harness-test",
            structured_output_method="json_schema",
        )

    def test_router_accepts_the_harness_as_a_model(self, fake_harness, tmp_path):
        """No HTTP probe fires: the harness exposes no base_url."""
        chat = _make_chat(fake_harness, tmp_path)
        router = self._router(chat)
        assert router.model_names == ["fake-harness/v1"]

    async def test_router_structured_output_round_trip(self, fake_harness, tmp_path):
        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "tune", "score": 7})
        router = self._router(chat)
        result = await router.with_structured_output(Answer).ainvoke("go")
        assert isinstance(result, Answer)
        assert result.score == 7

    async def test_router_records_the_model_for_attribution(
        self, fake_harness, tmp_path
    ):
        """structured_diff reports get_selected_model() with no fallback."""
        from gigaevo.llm.models import get_selected_model

        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": 1})
        router = self._router(chat)
        await router.with_structured_output(Answer).ainvoke("go")
        assert get_selected_model() == "fake-harness/v1"
        assert router.get_last_model() == "fake-harness/v1"

    async def test_router_unstructured_round_trip(self, fake_harness, tmp_path):
        chat = _make_chat(fake_harness, tmp_path)
        router = self._router(chat)
        response = await router.ainvoke("hello")
        assert response.content == "ok"

    async def test_a_parse_failure_is_reported_by_the_router_not_the_backend(
        self, fake_harness, tmp_path
    ):
        """`_process` tracks the raw message's tokens before it raises: the
        failed call's spend is real spend, and the usage assertion below fails
        if either the backend raises early or the router stops tracking.
        """
        from gigaevo.llm.models import get_last_token_usage

        chat = _make_chat(fake_harness, tmp_path)
        chat.env["FAKE_MODE"] = "usage"
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "a", "score": -3})
        router = self._router(chat)
        with pytest.raises(ValueError, match="Structured output parse failed"):
            await router.with_structured_output(Answer).ainvoke("go")
        tracked = get_last_token_usage()
        assert tracked is not None
        assert tracked.context == 11 + 9578 + 21721


# ---------------------------------------------------------------------------
# The real agent stack, driven by a fake harness
# ---------------------------------------------------------------------------


class TestMutationAgentIntegration:
    """Drives MutationAgent through the harness rather than through a mock.

    The router tests prove the two components fit; this proves the whole path a
    real run takes — prompt assembly, `with_structured_output` on a Pydantic
    schema, the LangGraph node, and the parse back into `MutationStructuredOutput`.
    """

    async def test_mutation_agent_round_trips_through_the_harness(
        self, fake_harness, tmp_path
    ):
        from gigaevo.llm.agents.mutation import MutationAgent
        from gigaevo.programs.program import Program

        chat = _make_chat(fake_harness, tmp_path)
        router = MultiModelRouter(
            [chat],
            [1.0],
            writer=NullWriter(),
            name="harness-mutation",
            structured_output_method="json_schema",
        )
        agent = MutationAgent(
            llm=router,
            system_prompt="You are a mutation agent.",
            user_prompt_template="Mutate {count} parents:\n{parent_blocks}",
            mutation_mode="rewrite",
        )
        # Set after construction: preflight already ran against the text schema.
        chat.env["FAKE_PAYLOAD"] = json.dumps(
            {
                "archetype": "Precision Optimization",
                "justification": "Tightened the inner loop bound.",
                "insights_used": [],
                "code": "def solve():\n    return 99\n",
            }
        )

        result = await agent.arun(
            input=[Program(code="def solve():\n    return 42\n")],
            mutation_mode="rewrite",
        )

        assert result["code"] == "def solve():\n    return 99"
        assert result["archetype"] == "Precision Optimization"
        assert result["structured_output"]["archetype"] == "Precision Optimization"
        # The agent's prompt really reached the harness, not a stub.
        assert (
            "You are a mutation agent."
            in (_workspaces(chat)[-1] / SYSTEM_FILE).read_text()
        )
        assert (
            "def solve():\n    return 42"
            in (_workspaces(chat)[-1] / USER_FILE).read_text()
        )


# ---------------------------------------------------------------------------
# Native structured output
# ---------------------------------------------------------------------------


def _native(fake_harness: Path, tmp_path: Path, **kwargs: Any) -> HarnessChat:
    return _make_chat(
        fake_harness,
        tmp_path,
        schema_flag="--schema",
        answer_key="answer",
        **kwargs,
    )


class TestNativeSchema:
    """A harness that takes the schema on argv and answers on stdout.

    The file handshake is the contract every harness can honour; this is the
    faster one for a harness that has its own structured output, and the two
    must not leak into each other.
    """

    def test_schema_goes_on_the_command_line(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        monkeypatch.setenv("FAKE_PAYLOAD", json.dumps({"archetype": "a", "score": 1}))
        chat = _native(fake_harness, tmp_path)

        chat.with_structured_output(Answer).invoke("go")

        workspace = _workspaces(chat)[-1]
        on_argv = json.loads((workspace / "ARGV_SCHEMA.json").read_text())
        assert on_argv["properties"].keys() >= {"archetype", "score"}
        # The same schema is still filed in the workspace as the audit record.
        assert json.loads((workspace / SCHEMA_FILE).read_text()) == on_argv

    def test_answer_comes_from_the_envelope(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        monkeypatch.setenv(
            "FAKE_PAYLOAD",
            json.dumps({"archetype": "Guided", "score": 3, "detail": {"reason": "r"}}),
        )
        chat = _native(fake_harness, tmp_path)

        answer = chat.with_structured_output(Answer).invoke("go")

        assert isinstance(answer, Answer)
        assert answer.archetype == "Guided"
        assert answer.detail is not None and answer.detail.reason == "r"
        assert not (_workspaces(chat)[-1] / OUTPUT_FILE).exists()

    def test_free_form_text_still_works(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Also the preflight path: construction alone would have failed it.
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _native(fake_harness, tmp_path)

        assert chat.invoke("go").content == "ok"

    def test_usage_still_read_from_the_same_envelope(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _native(fake_harness, tmp_path)

        message = chat.invoke("go")

        assert message.usage_metadata is not None
        assert message.usage_metadata["input_tokens"] == 11
        assert message.usage_metadata["output_tokens"] == 7
        assert message.response_metadata["total_cost_usd"] == 0.0125
        assert message.response_metadata["num_turns"] == 4

    def test_the_instruction_never_mentions_the_file_handshake(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Telling a harness to write OUTPUT.json costs the turns this mode buys.
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _native(fake_harness, tmp_path)

        stdin = (_workspaces(chat)[-1] / "STDIN.txt").read_text()

        assert OUTPUT_FILE not in stdin
        assert SYSTEM_FILE in stdin and USER_FILE in stdin

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("nativemissing", "answer.*input_tokens"),
            ("nativenotobject", "answer"),
            ("nativeprose", "not valid JSON"),
            ("nativebig", "over the"),
            # Nesting deep enough to exhaust the C stack: json.loads raises
            # RecursionError here, which is not a ValueError.
            ("nativedeepjson", "not valid JSON"),
        ],
    )
    def test_a_bad_envelope_fails_loudly(
        self,
        fake_harness: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
        expected: str,
    ) -> None:
        # _read_usage is silent by design; the answer path must not inherit that.
        monkeypatch.setenv("FAKE_MODE", mode)

        with pytest.raises(ValueError, match=expected):
            _native(fake_harness, tmp_path)

    async def test_the_async_path_reads_the_same_envelope(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        monkeypatch.setenv("FAKE_PAYLOAD", json.dumps({"archetype": "a", "score": 2}))
        chat = _native(fake_harness, tmp_path)

        answer = await chat.with_structured_output(Answer).ainvoke("go")

        assert isinstance(answer, Answer)
        assert answer.score == 2
        assert not (_workspaces(chat)[-1] / OUTPUT_FILE).exists()

    def test_a_fifo_swapped_into_stdout_fails_instead_of_hanging(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "nativefifo")

        started = time.monotonic()
        with pytest.raises(ValueError, match="not a regular file"):
            _native(fake_harness, tmp_path)
        # Promptly, by its own guard — not after 30s by the global timeout.
        assert time.monotonic() - started < 5.0

    def test_a_stdout_symlink_out_of_the_workspace_is_refused(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "outside.json"
        monkeypatch.setenv("FAKE_MODE", "nativestdoutlink")
        monkeypatch.setenv("FAKE_OUTSIDE", str(outside))

        with pytest.raises(ValueError, match="resolves outside"):
            _native(fake_harness, tmp_path)

    @pytest.mark.parametrize(
        "half", [{"schema_flag": "--schema"}, {"answer_key": "answer"}]
    )
    def test_half_configured_is_refused(
        self,
        fake_harness: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        half: dict[str, str],
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")

        with pytest.raises(ValueError, match="schema_flag and an answer channel"):
            _make_chat(fake_harness, tmp_path, **half)

    def test_flag_and_key_are_read_from_the_instance_not_hardcoded(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production wires --json-schema/structured_output while the fixture
        defaults to --schema/answer: a backend that hardcoded either literal
        would pass every test that reuses the defaults, then fail every real
        call.
        """
        monkeypatch.setenv("FAKE_MODE", "native")
        monkeypatch.setenv("FAKE_SCHEMA_FLAG", "--another-flag")
        monkeypatch.setenv("FAKE_ANSWER_KEY", "another_key")
        monkeypatch.setenv("FAKE_PAYLOAD", json.dumps({"archetype": "a", "score": 5}))
        chat = _make_chat(
            fake_harness,
            tmp_path,
            schema_flag="--another-flag",
            answer_key="another_key",
        )

        answer = chat.with_structured_output(Answer).invoke("go")

        assert answer.score == 5

    def test_a_stray_output_json_does_not_win_over_the_envelope(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proves the answer's SOURCE by preference, not by the decoy's absence."""
        monkeypatch.setenv("FAKE_MODE", "nativedecoy")
        monkeypatch.setenv(
            "FAKE_PAYLOAD", json.dumps({"archetype": "real", "score": 9})
        )
        chat = _native(fake_harness, tmp_path)

        answer = chat.with_structured_output(Answer).invoke("go")

        assert answer.archetype == "real"
        assert answer.score == 9

    def test_prompts_dir_override_selects_the_native_instruction(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`prompts.dir` and `schema_flag` ship together in production, so the
        override must reach instruction_native.txt, not instruction.txt.
        """
        monkeypatch.setenv("FAKE_MODE", "native")
        custom = tmp_path / "prompts" / "harness"
        custom.mkdir(parents=True)
        (custom / "instruction_native.txt").write_text(
            "CUSTOM-NATIVE {system} {user}", encoding="utf-8"
        )
        chat = _native(fake_harness, tmp_path, prompts_dir=str(custom.parent))

        stdin_seen = (_workspaces(chat)[0] / "STDIN.txt").read_text()
        assert stdin_seen.startswith(f"CUSTOM-NATIVE {SYSTEM_FILE} {USER_FILE}")

    def test_unset_keeps_the_file_handshake(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "ok")
        chat = _make_chat(fake_harness, tmp_path)

        chat.invoke("go")

        workspace = _workspaces(chat)[-1]
        assert (workspace / OUTPUT_FILE).exists()
        assert not (workspace / "ARGV_SCHEMA.json").exists()
        assert OUTPUT_FILE in (workspace / "STDIN.txt").read_text()


def _inline(fake_harness: Path, tmp_path: Path, **kwargs: Any) -> HarnessChat:
    return _native(fake_harness, tmp_path, system_flag="--system", **kwargs)


class TestInlinePrompts:
    """``system_flag``: the prompts travel in the call, not in files.

    System text on argv, user text on stdin, no instruction — the harness
    answers without spending turns reading SYSTEM.md and USER.md back.
    """

    def test_system_text_rides_the_command_line(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _inline(fake_harness, tmp_path)

        chat.invoke([SystemMessage(content="SYSTEM RULES"), HumanMessage(content="go")])

        workspace = _workspaces(chat)[-1]
        assert (workspace / "ARGV_SYSTEM.md").read_text() == "SYSTEM RULES"
        # The workspace files remain as the audit record of the same call.
        assert (workspace / SYSTEM_FILE).read_text() == "SYSTEM RULES"

    def test_stdin_is_the_user_text_not_an_instruction(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _inline(fake_harness, tmp_path)

        chat.invoke([SystemMessage(content="SYSTEM RULES"), HumanMessage(content="go")])

        workspace = _workspaces(chat)[-1]
        stdin_seen = (workspace / "STDIN.txt").read_text()
        assert stdin_seen == (workspace / USER_FILE).read_text()
        # No instruction: nothing points the harness at workspace files.
        assert SYSTEM_FILE not in stdin_seen
        assert USER_FILE not in stdin_seen

    def test_the_probe_exercises_the_system_flag(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Preflight exists to fail at startup: a probe that never sends the
        flag waves a misspelled system_flag through to the first mutation.
        """
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _inline(fake_harness, tmp_path)

        probe_workspace = _workspaces(chat)[0]
        assert (probe_workspace / "ARGV_SYSTEM.md").read_text() != ""
        stdin_seen = (probe_workspace / "STDIN.txt").read_text()
        assert stdin_seen == (probe_workspace / USER_FILE).read_text()

    def test_a_call_without_system_text_sends_no_empty_flag(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flag with an empty argument is a harness-dependent gamble that
        buys nothing.
        """
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _inline(fake_harness, tmp_path)

        chat.invoke("go")

        workspace = _workspaces(chat)[-1]
        assert not (workspace / "ARGV_SYSTEM.md").exists()
        stdin_seen = (workspace / "STDIN.txt").read_text()
        assert stdin_seen == (workspace / USER_FILE).read_text()

    def test_a_lone_surrogate_survives_inline_delivery(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The workspace files backslash-replace lone surrogates at write
        time; argv and stdin encode strictly, so the same text must be
        sanitized before it travels or the call dies in the encode.
        """
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _inline(fake_harness, tmp_path)

        chat.invoke(
            [
                SystemMessage(content="rules \ud800 end"),
                HumanMessage(content="go \ud800 now"),
            ]
        )

        workspace = _workspaces(chat)[-1]
        assert "rules \\ud800 end" in (workspace / "ARGV_SYSTEM.md").read_text()
        assert "go \\ud800 now" in (workspace / "STDIN.txt").read_text()

    async def test_async_system_rides_the_command_line(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _inline(fake_harness, tmp_path)

        await chat.ainvoke(
            [SystemMessage(content="ASYNC RULES"), HumanMessage(content="go")]
        )

        assert (_workspaces(chat)[-1] / "ARGV_SYSTEM.md").read_text() == "ASYNC RULES"

    def test_requires_the_native_envelope(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without schema_flag the answer channel is OUTPUT.json, which only
        the stdin instruction asks for — inline mode removes that instruction.
        """
        monkeypatch.setenv("FAKE_MODE", "ok")

        with pytest.raises(ValueError, match="system_flag"):
            _make_chat(fake_harness, tmp_path, system_flag="--system")

    def test_the_flag_is_read_from_the_instance_not_hardcoded(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        monkeypatch.setenv("FAKE_SYSTEM_FLAG", "--sys-b")
        chat = _native(fake_harness, tmp_path, system_flag="--sys-b")

        chat.invoke([SystemMessage(content="B RULES"), HumanMessage(content="go")])

        assert (_workspaces(chat)[-1] / "ARGV_SYSTEM.md").read_text() == "B RULES"

    async def test_structured_round_trip_through_the_router(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _inline(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "tune", "score": 7})
        router = MultiModelRouter(
            [chat],
            [1.0],
            writer=NullWriter(),
            name="harness-inline",
            structured_output_method="json_schema",
        )

        result = await router.with_structured_output(Answer).ainvoke("go")

        assert isinstance(result, Answer)
        assert result.score == 7
        assert not (_workspaces(chat)[-1] / OUTPUT_FILE).exists()


def _pathschema(fake_harness: Path, tmp_path: Path, **kwargs: Any) -> HarnessChat:
    return _make_chat(
        fake_harness,
        tmp_path,
        schema_flag="--schema-path",
        schema_as_path=True,
        answer_file_flag="-o",
        **kwargs,
    )


class TestSchemaPathAndAnswerFile:
    """The codex exec shape: the schema flag takes a file path, and the CLI
    itself writes the final message into the file named by
    ``answer_file_flag`` — the answer channel for a harness with no answer
    key in its stdout, which streams JSONL events instead of one envelope.
    """

    def test_the_schema_flag_carries_the_workspace_schema_path(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "pathschema")
        chat = _pathschema(fake_harness, tmp_path)

        chat.invoke("go")

        workspace = _workspaces(chat)[-1]
        assert (workspace / "ARGV_SCHEMA_PATH.txt").read_text() == str(
            workspace / SCHEMA_FILE
        )
        on_argv = json.loads((workspace / "ARGV_SCHEMA.json").read_text())
        assert on_argv.get("required") == ["text"]

    def test_the_answer_comes_from_the_answer_file(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "pathschema")
        monkeypatch.setenv(
            "FAKE_PAYLOAD", json.dumps({"archetype": "tune", "score": 7})
        )
        chat = _pathschema(fake_harness, tmp_path)

        answer = chat.with_structured_output(Answer).invoke("go")

        assert isinstance(answer, Answer)
        assert answer.score == 7
        workspace = _workspaces(chat)[-1]
        assert (workspace / ANSWER_FILE).is_file()
        assert not (workspace / OUTPUT_FILE).exists()

    def test_jsonl_events_report_usage_and_turns(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "pathschema")
        chat = _pathschema(fake_harness, tmp_path)

        message = chat.invoke("go")

        assert message.usage_metadata is not None
        assert message.usage_metadata["input_tokens"] == 1500
        assert message.usage_metadata["output_tokens"] == 40
        assert message.usage_metadata["input_token_details"] == {"cache_read": 1200}
        assert message.response_metadata["num_turns"] == 1
        usage = TokenUsage.from_response(message)
        assert usage is not None
        assert usage.turns == 1
        # The JSONL stream carries no cost; None keeps the cost series clean.
        assert usage.cost_usd is None

    def test_a_missing_answer_file_is_loud(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "pathschemamissing")

        with pytest.raises(ValueError, match=ANSWER_FILE):
            _pathschema(fake_harness, tmp_path)

    def test_the_two_answer_channels_are_exclusive(
        self, fake_harness: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="answer channel"):
            _make_chat(
                fake_harness,
                tmp_path,
                schema_flag="--schema",
                answer_key="answer",
                answer_file_flag="-o",
            )

    def test_schema_as_path_requires_schema_flag(
        self, fake_harness: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="schema_as_path"):
            _make_chat(fake_harness, tmp_path, schema_as_path=True)

    def test_an_answer_file_alone_is_rejected(
        self, fake_harness: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="schema_flag"):
            _make_chat(fake_harness, tmp_path, answer_file_flag="-o")

    async def test_the_async_path_reads_the_answer_file(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "pathschema")
        monkeypatch.setenv("FAKE_PAYLOAD", json.dumps({"archetype": "a", "score": 2}))
        chat = _pathschema(fake_harness, tmp_path)

        answer = await chat.with_structured_output(Answer).ainvoke("go")

        assert isinstance(answer, Answer)
        assert answer.score == 2

    def test_strict_schema_travels_and_its_nulls_strip_back(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI strict mode: every key required, optionals nullable-ized on
        the wire — and a returned null for one strips back out so the caller's
        pydantic default applies instead of failing validation.
        """

        class Padded(BaseModel):
            score: int
            notes: list[str] = []

        monkeypatch.setenv("FAKE_MODE", "pathschema")
        monkeypatch.setenv("FAKE_PAYLOAD", json.dumps({"score": 3, "notes": None}))
        chat = _pathschema(fake_harness, tmp_path, strict_schema=True)

        answer = chat.with_structured_output(Padded).invoke("go")

        assert answer.notes == []
        workspace = _workspaces(chat)[-1]
        on_argv = json.loads((workspace / "ARGV_SCHEMA.json").read_text())
        assert on_argv["additionalProperties"] is False
        assert sorted(on_argv["required"]) == ["notes", "score"]
        assert on_argv["properties"]["notes"]["anyOf"][-1] == {"type": "null"}


class TestStdinPrompts:
    """``stdin_prompts``: the prompt text itself travels on stdin — no
    instruction, no file-reading turns, no shell tool in the loop. The mode
    for a CLI whose sandbox cannot be trusted to read the workspace
    (``codex exec``, whose first sandboxed command flakes).
    """

    def test_system_and_user_go_to_stdin_verbatim(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "pathschema")
        chat = _pathschema(fake_harness, tmp_path, stdin_prompts=True)

        chat.invoke([SystemMessage(content="SYS RULES"), HumanMessage(content="go")])

        workspace = _workspaces(chat)[-1]
        assert (workspace / "STDIN.txt").read_text() == "SYS RULES\n\n## human\n\ngo"
        # SYSTEM.md and USER.md remain the audit record of what the call sent.
        assert (workspace / SYSTEM_FILE).read_text() == "SYS RULES"
        assert (workspace / USER_FILE).read_text() == "## human\n\ngo"

    def test_without_system_text_stdin_is_the_user_text_alone(
        self, fake_harness: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_MODE", "pathschema")
        chat = _pathschema(fake_harness, tmp_path, stdin_prompts=True)

        chat.invoke("go")

        workspace = _workspaces(chat)[-1]
        assert (workspace / "STDIN.txt").read_text() == "## human\n\ngo"

    def test_stdin_prompts_requires_schema_flag(
        self, fake_harness: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="stdin_prompts"):
            _make_chat(fake_harness, tmp_path, stdin_prompts=True)

    def test_stdin_prompts_and_system_flag_are_exclusive(
        self, fake_harness: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="one way to inline"):
            _make_chat(
                fake_harness,
                tmp_path,
                schema_flag="--schema",
                answer_key="answer",
                system_flag="--sys",
                stdin_prompts=True,
            )


class TestNativeRouterIntegration:
    """The combination production actually ships: native mode behind the router.

    TestRouterIntegration and TestMutationAgentIntegration exercise the file
    handshake; config/llm/harness.yaml sets schema_flag unconditionally, so the
    path a real run takes is this one.
    """

    def _router(self, chat: HarnessChat) -> MultiModelRouter:
        return MultiModelRouter(
            [chat],
            [1.0],
            writer=NullWriter(),
            name="harness-native",
            structured_output_method="json_schema",
        )

    async def test_router_structured_output_round_trip(
        self, fake_harness, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _native(fake_harness, tmp_path)
        chat.env["FAKE_PAYLOAD"] = json.dumps({"archetype": "tune", "score": 7})
        router = self._router(chat)

        result = await router.with_structured_output(Answer).ainvoke("go")

        assert isinstance(result, Answer)
        assert result.score == 7
        assert not (_workspaces(chat)[-1] / OUTPUT_FILE).exists()

    async def test_mutation_agent_round_trips_through_the_native_path(
        self, fake_harness, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        from gigaevo.llm.agents.mutation import MutationAgent
        from gigaevo.programs.program import Program

        monkeypatch.setenv("FAKE_MODE", "native")
        chat = _native(fake_harness, tmp_path)
        router = self._router(chat)
        agent = MutationAgent(
            llm=router,
            system_prompt="You are a mutation agent.",
            user_prompt_template="Mutate {count} parents:\n{parent_blocks}",
            mutation_mode="rewrite",
        )
        chat.env["FAKE_PAYLOAD"] = json.dumps(
            {
                "archetype": "Precision Optimization",
                "justification": "Tightened the inner loop bound.",
                "insights_used": [],
                "code": "def solve():\n    return 99\n",
            }
        )

        result = await agent.arun(
            input=[Program(code="def solve():\n    return 42\n")],
            mutation_mode="rewrite",
        )

        assert result["code"] == "def solve():\n    return 99"
        assert result["archetype"] == "Precision Optimization"
        workspace = _workspaces(chat)[-1]
        assert not (workspace / OUTPUT_FILE).exists()
        assert "You are a mutation agent." in (workspace / SYSTEM_FILE).read_text()
