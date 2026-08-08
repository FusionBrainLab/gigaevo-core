"""Agentic coding-harness backend for the GigaEvo LLM layer.

Drives an external coding CLI — ``claude -p``, ``codex exec``, or anything a
user can name — instead of an OpenAI-compatible chat endpoint. The harness is
a ``BaseChatModel`` held in ``MultiModelRouter.models``, the same way
``BalancedChatOpenAI`` is, so token tracking, the concurrency semaphore, model
attribution, Langfuse tracing and the prompt I/O dump all keep working.

Communication is a filesystem contract rather than an HTTP body. Each call gets
a workspace directory containing the prompt and the schema the answer must
satisfy; the harness writes its answer back into the same directory::

    <workspace_root>/<run_id>/<call_id>/
      SYSTEM.md      all SystemMessage content, joined
      USER.md        the remaining messages, role-tagged
      SCHEMA.json    the schema the answer must conform to
      OUTPUT.json    written by the harness, under the default file handshake
      ANSWER.json    written by the CLI itself, under ``answer_file_flag``
      STDOUT.log     the harness's stdout: token counts, and under
                     ``schema_flag`` with ``answer_key`` the answer
      STDERR.log     the harness's stderr, for debugging a failed call

Unstructured calls get a schema too — a single ``text`` field — so the harness
obeys one rule regardless of the caller. The workspace also becomes the
harness's working directory.

Token counts are optional and opt-in on the harness's side. If ``STDOUT.log``
holds a single JSON object with a ``usage`` mapping, its counts are reported on
the normal metadata channels; ``claude -p --output-format json`` prints exactly
that, and adds ``total_cost_usd``. Most harnesses print prose, an event stream,
or nothing at all — those report zeros, as this backend did before counts
existed.

The answer's channel depends on the mode. Under the default file handshake it
is always ``OUTPUT.json``, never stdout, so a harness that reports no usage is
in no way degraded. With ``schema_flag`` and ``answer_key`` set, the schema
goes on the command line, the answer comes back in the harness's own stdout
envelope under ``answer_key``, and ``OUTPUT.json`` is never written — the
turns a harness spends drafting and checking that file are never spent.
``answer_file_flag`` is the third channel, for a CLI that writes its final
message to a named file itself (``codex exec -o``): the answer lands in
``ANSWER.json`` mechanically, and stdout is free to be a JSONL event stream,
which ``_usage_from_events`` mines for per-turn usage. ``SCHEMA.json`` is
still written in every mode, as the workspace audit record — and under
``schema_as_path`` it is also what the schema flag points at.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
from typing import IO, Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from gigaevo.llm.models import _extract_content_text
from gigaevo.llm.schema_compat import (
    portable_json_schema,
    strict_json_schema,
    strip_strict_nulls,
)
from gigaevo.prompts import load_prompt

SYSTEM_FILE = "SYSTEM.md"
USER_FILE = "USER.md"
SCHEMA_FILE = "SCHEMA.json"
OUTPUT_FILE = "OUTPUT.json"
ANSWER_FILE = "ANSWER.json"
STDOUT_FILE = "STDOUT.log"
STDERR_FILE = "STDERR.log"

#: Schema handed to the harness when the caller wants free-form text.
TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

_STDERR_EXCERPT = 2000

#: Largest ``STDOUT.log`` this backend will parse looking for token counts. A
#: harness asked to print a JSON result prints one small object; a harness that
#: was not prints its whole agentic log, and nothing bounds that. Over the cap
#: the file is left unread rather than pulled into memory to find no counts.
_USAGE_LIMIT = 1 << 20

_NO_USAGE: dict[str, Any] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}

#: Grace given to a killed process group before we stop waiting on it. The
#: group is already under SIGKILL; anything still alive is unkillable (D-state)
#: and waiting longer would hang the caller rather than help it.
_REAP_GRACE = 10.0


def _looks_like_schema(candidate: dict[str, Any]) -> bool:
    return "type" in candidate or "properties" in candidate


def _resolve_schema(schema: Any) -> tuple[dict[str, Any], type[BaseModel] | None]:
    """Split a ``with_structured_output`` argument into wire schema and parser.

    Callers pass a Pydantic class, an already-portable raw dict, or the OpenAI
    ``{"name", "schema"}`` envelope that ``DiffSchema`` produces — which
    ``_schema_for_method`` rewrites to ``{"name", "parameters"}`` under
    ``function_calling``. Only the Pydantic form has a type to parse back into.

    A dict that is itself a schema is never unwrapped, so a schema whose own
    top-level keys happen to include ``parameters`` keeps its body.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return portable_json_schema(schema.model_json_schema()), schema
    if isinstance(schema, dict):
        if not _looks_like_schema(schema):
            for envelope_key in ("schema", "parameters"):
                inner = schema.get(envelope_key)
                if isinstance(inner, dict):
                    return portable_json_schema(inner), None
        return portable_json_schema(schema), None
    raise TypeError(f"unsupported structured-output schema: {type(schema).__name__}")


def _split_messages(messages: list[BaseMessage]) -> tuple[str, str]:
    """Render messages into the SYSTEM.md and USER.md bodies."""
    system: list[str] = []
    user: list[str] = []
    for message in messages:
        text = _extract_content_text(message.content)
        if isinstance(message, SystemMessage):
            system.append(text)
        else:
            user.append(f"## {message.type}\n\n{text}")
    return "\n\n".join(system), "\n\n".join(user)


def _sanitize(text: str) -> str:
    """Replace lone surrogates with the ``\\udXXX`` escape json.dumps uses.

    A model emits one by truncating a pair, and it survives json.loads on the
    way back in. UTF-8 refuses it everywhere a prompt travels — the workspace
    files, and under ``system_flag`` the argv and stdin encodes — and raising
    would strand every artifact that ever quoted one (a card, an insight, a
    suggestion): unusable from then on, and only under this backend. Sanitized
    once here, the harness sees what a model would: the same escape json.dumps
    puts in an HTTP request body.
    """
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _count(usage: dict[str, Any], key: str) -> int:
    """Read one non-negative token count, or nothing.

    ``bool`` is an ``int`` subclass, so an unchecked ``True`` books as a token
    and a negative subtracts from the run's spend — both survive into every
    cumulative total, where they can no longer be traced to the call.
    """
    value = usage.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _usage_from_envelope(envelope: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a harness result envelope into the two metadata channels.

    Cache reads and writes are billed input, so they belong in the input total:
    the cached prefix is most of an agentic harness's spend, and a backend
    reporting only ``input_tokens`` would show a handful of tokens for a call
    charged for tens of thousands. They are kept separately in
    ``input_token_details`` as well, because the two are priced differently.
    """
    if not isinstance(envelope, dict):
        return dict(_NO_USAGE), _openai_shaped(0, 0)
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return dict(_NO_USAGE), _openai_shaped(0, 0)

    cache_creation = _count(usage, "cache_creation_input_tokens")
    cache_read = _count(usage, "cache_read_input_tokens")
    prompt = _count(usage, "input_tokens") + cache_creation + cache_read
    completion = _count(usage, "output_tokens")

    usage_metadata: dict[str, Any] = {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cache_creation or cache_read:
        usage_metadata["input_token_details"] = {
            "cache_creation": cache_creation,
            "cache_read": cache_read,
        }

    response_metadata = _openai_shaped(prompt, completion)
    cost = envelope.get("total_cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        # A CLI on a subscription is the only thing that knows what it spent:
        # there is no per-token price to reconstruct it from downstream. JSON
        # accepts integers float() cannot -- a cost that overflows is a lie,
        # and the counts beside it are still good.
        with contextlib.suppress(OverflowError):
            response_metadata["total_cost_usd"] = float(cost)
    turns = envelope.get("num_turns")
    if isinstance(turns, int) and not isinstance(turns, bool) and turns >= 0:
        # Turn count is the dominant cost variable of an agentic backend --
        # each turn re-sends the conversation as billed input.
        response_metadata["num_turns"] = turns
    return usage_metadata, response_metadata


def _usage_from_events(raw: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover usage from a JSONL event stream (``codex exec --json``).

    OpenAI semantics, not Anthropic's: ``input_tokens`` already includes
    ``cached_input_tokens``, so nothing is summed — adding the cache detail
    back in would double-count most of the call. The last usage-bearing event
    wins, and the count of such events stands in for ``num_turns``: each one
    is a completed turn's bill.
    """
    usage: dict[str, Any] | None = None
    turns = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(event, dict):
            continue
        candidate = event.get("usage")
        if isinstance(candidate, dict):
            usage = candidate
            turns += 1
    if usage is None:
        return dict(_NO_USAGE), _openai_shaped(0, 0)

    prompt = _count(usage, "input_tokens")
    completion = _count(usage, "output_tokens")
    cached = _count(usage, "cached_input_tokens")

    usage_metadata: dict[str, Any] = {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached:
        usage_metadata["input_token_details"] = {"cache_read": cached}

    response_metadata = _openai_shaped(prompt, completion)
    response_metadata["num_turns"] = turns
    return usage_metadata, response_metadata


def _openai_shaped(prompt: int, completion: int) -> dict[str, Any]:
    """The channel ``TokenUsage.from_response`` reads, filled the same way."""
    return {
        "token_usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    }


def _harness_message(
    text: str,
    usage_metadata: dict[str, Any] | None = None,
    response_metadata: dict[str, Any] | None = None,
) -> AIMessage:
    """Wrap harness output as an AIMessage carrying whatever usage was reported.

    Both metadata channels are filled from the same numbers: ``usage_metadata``
    is what the I/O dump records, and ``response_metadata['token_usage']`` is
    what ``TokenTracker`` buckets per stage. Filling only one would show real
    counts in the audit trail and zeros in every run report.
    """
    return AIMessage(
        content=text,
        usage_metadata=usage_metadata or dict(_NO_USAGE),
        response_metadata=response_metadata or _openai_shaped(0, 0),
    )


def _kill_group(pgid: int) -> None:
    """Kill the harness process group.

    Harnesses start MCP servers and tool subprocesses, so killing the leader
    alone leaves orphans. Takes the leader's pid, which ``start_new_session``
    makes the group id too: asyncio reaps the leader the moment it exits, so
    ``getpgid`` would already raise ``ProcessLookupError`` here — while the
    group is still populated and still killable. The process got its own
    session at spawn, so this cannot reach the GigaEvo run.
    """
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


class HarnessChat(BaseChatModel):
    """Chat model backed by an agentic coding CLI.

    Usage in Hydra config::

        _target_: gigaevo.llm.harness.HarnessChat
        model_name: ${model_name}
        request_timeout: ${request_timeout}
        command: [claude, -p, --disallowedTools, Bash]
    """

    model_name: str
    command: list[str]
    request_timeout: float = 600.0
    #: Defaults to a fresh private temp directory, deliberately outside the
    #: repository: a harness with filesystem access takes its cwd from here.
    workspace_root: str = ""
    #: Environment overrides layered onto the parent environment. Point a
    #: harness's own state directory here to keep it off shared storage.
    env: dict[str, str] = Field(default_factory=dict)
    #: Flag that hands the JSON Schema to the harness on its command line, for
    #: a harness with structured output of its own. The answer then comes back
    #: in the harness's stdout envelope under ``answer_key``, and the
    #: ``OUTPUT.json`` handshake -- the contract every harness can honour -- is
    #: not used. Empty keeps that handshake. Set both fields or neither.
    schema_flag: str = ""
    #: Key in the stdout envelope holding the answer, under ``schema_flag``.
    answer_key: str = ""
    #: ``schema_flag`` takes the path of the workspace ``SCHEMA.json`` rather
    #: than the schema text itself, for a CLI whose flag wants a file
    #: (``codex exec --output-schema``).
    schema_as_path: bool = False
    #: Flag naming a file the CLI itself writes the final message into
    #: (``codex exec -o``) — the native answer channel for a harness with no
    #: answer key in its stdout. One channel per backend: set this or
    #: ``answer_key``, never both, and either only with ``schema_flag``.
    answer_file_flag: str = ""
    #: Rewrite the wire schema into the OpenAI strict-mode subset: every
    #: object closed (``additionalProperties: false``) and fully required,
    #: originally-optional properties made nullable. For a backend whose
    #: ``schema_flag`` lands in strict structured output (``codex exec``),
    #: which rejects anything else. The structured runnable strips the
    #: invited nulls back out against the original schema, so pydantic
    #: defaults apply as they would have without the rewrite.
    strict_schema: bool = False
    #: Flag that puts the system text on the harness's command line, with the
    #: user text going to stdin verbatim: no instruction, no file-reading
    #: turns, the harness answers the prompt directly. Requires ``schema_flag``
    #: and with it an answer channel -- once the stdin instruction is gone,
    #: nothing would tell the harness to write ``OUTPUT.json``, so the native
    #: channel is the only one left. ``SYSTEM.md`` and ``USER.md``
    #: are still written, as the audit record of what the call carried.
    system_flag: str = ""
    #: The prompt text itself travels on stdin — system then user, verbatim —
    #: instead of the instruction pointing the harness at the workspace files.
    #: No file-reading turns and no shell tool in the loop: the mode for a CLI
    #: with no system flag whose sandboxed reads cannot be trusted
    #: (``codex exec``, whose first sandboxed command flakes and turns missing
    #: prompts into schema-valid fallback answers). Requires ``schema_flag``
    #: for the same reason ``system_flag`` does, and excludes it: there is one
    #: way to inline per backend. ``SYSTEM.md`` and ``USER.md`` are still
    #: written, as the audit record of what the call carried.
    stdin_prompts: bool = False
    prompts_dir: str | None = None

    _run_dir: Path = PrivateAttr()
    _calls: itertools.count = PrivateAttr()
    _lock: threading.Lock = PrivateAttr()
    _instruction: str = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        if not self.command:
            raise ValueError("HarnessChat requires a non-empty command")
        if self.answer_key and self.answer_file_flag:
            raise ValueError(
                "HarnessChat takes one answer channel, not two: answer_key "
                "reads the stdout envelope, answer_file_flag reads the file "
                "the CLI writes — a backend cannot answer on both"
            )
        if bool(self.schema_flag) != bool(self.answer_key or self.answer_file_flag):
            raise ValueError(
                "HarnessChat needs schema_flag and an answer channel "
                "(answer_key or answer_file_flag) together: a schema on the "
                "command line is only usable if the answer can be found "
                "again, and either channel only exists under the schema"
            )
        if self.schema_as_path and not self.schema_flag:
            raise ValueError(
                "HarnessChat needs schema_flag to use schema_as_path: it "
                "only says how that flag's value travels"
            )
        if self.system_flag and not self.schema_flag:
            raise ValueError(
                "HarnessChat needs schema_flag and an answer channel to use "
                "system_flag: inlining the prompts drops the stdin "
                "instruction, and only the native channel can carry the "
                "answer without one"
            )
        if self.stdin_prompts and not self.schema_flag:
            raise ValueError(
                "HarnessChat needs schema_flag and an answer channel to use "
                "stdin_prompts: prompts on stdin drop the instruction, and "
                "without one the harness never writes OUTPUT.json"
            )
        if self.stdin_prompts and self.system_flag:
            raise ValueError(
                "HarnessChat takes one way to inline prompts, not two: "
                "system_flag puts the system text on argv, stdin_prompts "
                "puts everything on stdin"
            )
        root = Path(self.workspace_root) if self.workspace_root else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        # mkdtemp, never a derived name: a well-known path is a pre-plantable
        # symlink, a pid-derived one is shared by two chats in one process --
        # which then hand out the same call ids and overwrite each other's live
        # prompts -- and reused by the next process to inherit that pid, which
        # would read the previous run's answers back as its own. It also gets
        # 0700, which mkdir would not: mkdir's mode is masked by the umask and
        # ignored outright on an existing directory.
        self._run_dir = Path(tempfile.mkdtemp(prefix="gigaevo-harness-", dir=root))
        self._calls = itertools.count()
        self._lock = threading.Lock()
        # Inline mode sends the conversation itself, so there is no
        # instruction to load — and none to drift out of date.
        self._instruction = (
            ""
            if self.system_flag
            else load_prompt(
                "harness",
                "instruction_native" if self.schema_flag else "instruction",
                self.prompts_dir,
            )
        )
        logger.info(
            "[HarnessChat:{}] command={} workspaces={}",
            self.model_name,
            " ".join(self.command),
            self._run_dir,
        )
        self._preflight()

    @property
    def _llm_type(self) -> str:
        return "gigaevo-harness"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "command": self.command}

    # -- lifecycle ---------------------------------------------------------

    def _preflight(self) -> None:
        """Verify the harness runs and honours the workspace contract.

        Mirrors ``MultiModelRouter._verify_models``: a misconfigured backend
        must fail at startup rather than after the first mutation. It costs one
        real harness call per run.
        """
        if shutil.which(self.command[0]) is None:
            raise RuntimeError(
                f"[HarnessChat:{self.model_name}] command not found on PATH: "
                f"{self.command[0]}"
            )
        # A system message so the probe travels every configured channel:
        # without one, `_argv` omits `system_flag` and a misspelled flag
        # passes startup only to fail the first real call.
        probe = [
            SystemMessage(content="You are a preflight probe."),
            HumanMessage(content="Reply with the single word: ok"),
        ]
        schema = self._wire_schema(TEXT_SCHEMA)
        workspace, system, user = self._write_workspace(probe, schema)
        self._exec(
            workspace,
            self._argv(schema, system, workspace),
            self._stdin_text(system, user),
        )
        payload = self._read_answer(workspace)
        if not isinstance(payload.get("text"), str):
            raise RuntimeError(
                f"[HarnessChat:{self.model_name}] preflight succeeded but "
                f"{self._answer_source} had no string 'text' field: {payload!r}"
            )
        logger.info("[HarnessChat:{}] preflight ok", self.model_name)

    # -- the workspace contract -------------------------------------------

    def _write_workspace(
        self, messages: list[BaseMessage], json_schema: dict[str, Any]
    ) -> tuple[Path, str, str]:
        with self._lock:
            call_id = next(self._calls)
        workspace = self._run_dir / f"{call_id:06d}"
        system, user = map(_sanitize, _split_messages(messages))
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / SYSTEM_FILE).write_text(system, encoding="utf-8")
            (workspace / USER_FILE).write_text(user, encoding="utf-8")
            (workspace / SCHEMA_FILE).write_text(
                json.dumps(json_schema, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            # Surfaced as ValueError so it travels the same path as a bad
            # answer; the message says "infrastructure" so a full disk is not
            # silently counted as the LLM producing an invalid program.
            raise ValueError(
                f"[{self.model_name}] harness infrastructure failure: cannot "
                f"write workspace {workspace}: {exc}"
            ) from exc
        return workspace, system, user

    def _instruction_text(self) -> str:
        return self._instruction.format(
            system=SYSTEM_FILE,
            user=USER_FILE,
            schema=SCHEMA_FILE,
            output=OUTPUT_FILE,
        )

    def _stdin_text(self, system: str, user: str) -> str:
        """What the harness reads as its prompt.

        Under ``system_flag`` the user text itself; under ``stdin_prompts``
        the whole prompt, system then user; otherwise the instruction
        pointing the harness at the workspace files.
        """
        if self.system_flag:
            return user
        if self.stdin_prompts:
            return f"{system}\n\n{user}" if system else user
        return self._instruction_text()

    def _wire_schema(self, json_schema: dict[str, Any]) -> dict[str, Any]:
        """The schema as the backend sees it, in ``SCHEMA.json`` and on argv."""
        return strict_json_schema(json_schema) if self.strict_schema else json_schema

    def _argv(
        self, json_schema: dict[str, Any], system: str, workspace: Path
    ) -> list[str]:
        argv = list(self.command)
        if self.schema_flag:
            schema_arg = (
                str(workspace / SCHEMA_FILE)
                if self.schema_as_path
                else json.dumps(json_schema)
            )
            argv += [self.schema_flag, schema_arg]
        if self.answer_file_flag:
            argv += [self.answer_file_flag, str(workspace / ANSWER_FILE)]
        if self.system_flag and system:
            # No system text, no flag: an empty argument is a
            # harness-dependent gamble that buys nothing.
            argv += [self.system_flag, system]
        return argv

    @property
    def _answer_source(self) -> str:
        if not self.schema_flag:
            return OUTPUT_FILE
        if self.answer_file_flag:
            return ANSWER_FILE
        return f"{STDOUT_FILE} envelope key {self.answer_key!r}"

    def _spawn_kwargs(
        self, workspace: Path, stdout: Any, stderr: Any
    ) -> dict[str, Any]:
        """Arguments shared by the sync and async spawns.

        Both streams go to files rather than pipes. A pipe is accumulated whole
        in this process's memory, and an agentic harness's stdout has no bound;
        a pipe also makes the call wait for every descendant to close it, not
        for the harness to exit. A file costs a bounded read afterwards, and
        only when something asks for it.
        """
        return {
            "cwd": str(workspace),
            "env": {**os.environ, **self.env},
            "stdin": subprocess.PIPE,
            "stdout": stdout,
            "stderr": stderr,
            "start_new_session": True,
        }

    def _tail(self, workspace: Path, name: str) -> str:
        """Read the last few KB of a capture file, and only those.

        Seeks rather than slurping: nothing bounds these files. The child owns
        the fd, and the timeout path selects for exactly the harnesses that
        write without end -- a retrying MCP server, a spinner with no TTY to
        detect. Reading one whole to keep the tail costs its full size in
        memory, on the loop, on every failure.
        """
        path = workspace / name
        try:
            # O_NONBLOCK because this is the last thing standing between a
            # timeout and the kill it exists to perform: it runs on the loop,
            # inside the `except`, before the `finally`. A harness that leaves
            # a FIFO here blocks the open forever -- and a sync syscall on the
            # loop thread cannot be cancelled, so the process group survives
            # and every other call stops with it. For a FIFO or a device the
            # seek then fails, which is caught. POSIX gives the flag no effect
            # on a regular file, so a wedged mount under a user-set
            # `workspace_root` still blocks here; that one is not solvable from
            # this side.
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                handle = os.fdopen(fd, "rb")
            except BaseException:
                # fdopen takes ownership only once it succeeds; on failure --
                # the log replaced by a directory raises IsADirectoryError,
                # itself an OSError the outer handler would silently eat -- the
                # fd is disowned and never closed. `open()` by name does not
                # have this hole, but it cannot pass O_NONBLOCK.
                os.close(fd)
                raise
            with handle:
                handle.seek(max(0, handle.seek(0, os.SEEK_END) - _STDERR_EXCERPT))
                data = handle.read(_STDERR_EXCERPT)
        except OSError:
            return ""
        return data.decode("utf-8", "replace")

    def _stderr_tail(self, workspace: Path) -> str:
        return self._tail(workspace, STDERR_FILE)

    def _timeout_error(self, workspace: Path) -> ValueError:
        return ValueError(
            f"[{self.model_name}] harness timed out after {self.request_timeout}s, "
            f"workspace={workspace}, stderr={self._stderr_tail(workspace)!r}"
        )

    def _check_exit(self, returncode: int | None, workspace: Path) -> None:
        if returncode:
            # Both tails: a CLI that reports failures in its stdout envelope
            # -- quota, auth, out of turns -- exits nonzero with stderr empty,
            # and this message is the only place its reason reaches the log.
            raise ValueError(
                f"[{self.model_name}] harness exited {returncode}, "
                f"workspace={workspace}, stderr={self._stderr_tail(workspace)!r}, "
                f"stdout={self._tail(workspace, STDOUT_FILE)!r}"
            )

    def _open_capture(self, workspace: Path, name: str) -> IO[bytes]:
        try:
            return (workspace / name).open("wb")
        except OSError as exc:
            raise ValueError(
                f"[{self.model_name}] harness infrastructure failure: cannot "
                f"open {name}: {exc}, workspace={workspace}"
            ) from exc

    def _read_stdout(self, workspace: Path) -> bytes:
        """Bounded, swap-hardened read of ``STDOUT.log``.

        Shared by :meth:`_read_usage` and :meth:`_read_envelope`: every hazard
        raises ``ValueError`` and the caller decides what a hazard costs. The
        file sat in a directory the harness owned, so it gets the checks
        ``OUTPUT.json`` gets — and one more, because a FIFO left in its place
        would hang the open forever where a bad file merely fails it, and the
        harness that would have been that FIFO's writer is already dead. A
        hung read here never returns its router semaphore slot, so enough of
        them stops the whole backend, not one call.
        """
        path = workspace / STDOUT_FILE
        if path.resolve().parent != workspace.resolve():
            # Same non-boundary as OUTPUT.json's check, same rationale.
            raise ValueError(
                f"[{self.model_name}] {STDOUT_FILE} resolves outside its "
                f"workspace: {path.resolve()}, workspace={workspace}"
            )
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            raise ValueError(
                f"[{self.model_name}] harness infrastructure failure: cannot "
                f"read {STDOUT_FILE}: {exc}, workspace={workspace}"
            ) from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(
                    f"[{self.model_name}] {STDOUT_FILE} is not a regular file, "
                    f"workspace={workspace}"
                )
            if info.st_size > _USAGE_LIMIT:
                raise ValueError(
                    f"[{self.model_name}] harness stdout is over the "
                    f"{_USAGE_LIMIT}-byte cap for an answer envelope, "
                    f"workspace={workspace}"
                )
            # Bounded read from the fd already vetted, not a fresh open by
            # name: a harness that ignored the flag prints its whole agentic
            # log here instead of one small object, and nothing bounds that.
            chunks: list[bytes] = []
            size = 0
            while chunk := os.read(fd, 1 << 16):
                size += len(chunk)
                if size > _USAGE_LIMIT:
                    raise ValueError(
                        f"[{self.model_name}] harness stdout is over the "
                        f"{_USAGE_LIMIT}-byte cap for an answer envelope, "
                        f"workspace={workspace}"
                    )
                chunks.append(chunk)
        except OSError as exc:
            raise ValueError(
                f"[{self.model_name}] harness infrastructure failure: cannot "
                f"read {STDOUT_FILE}: {exc}, workspace={workspace}"
            ) from exc
        finally:
            os.close(fd)
        return b"".join(chunks)

    def _read_usage(self, workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Recover token counts from a harness that reports them on stdout.

        Every failure here yields zeros, and only an unreadable stdout logs a
        warning on the way. Reporting usage is the
        harness's option, not its obligation: prose, an event stream, an empty
        file and a deleted one are all ordinary, and none of them says anything
        about whether the call succeeded — the answer has already been read,
        from ``OUTPUT.json`` or, under ``schema_flag``, by the loud twin of
        this read in :meth:`_read_envelope`.
        """
        try:
            raw = self._read_stdout(workspace).decode("utf-8", "replace")
        except (OSError, ValueError) as exc:
            # ValueError covers every hazard _read_stdout raises, and none of
            # them is worth failing a served call over — but an over-cap or
            # unreadable stdout zeroes the longest, most expensive calls, so
            # say which ones vanished from the cost series.
            logger.warning(
                "[HarnessChat:{}] usage unreadable, reporting zeros: {}",
                self.model_name,
                exc,
            )
            return dict(_NO_USAGE), _openai_shaped(0, 0)
        try:
            envelope = json.loads(raw)
        except (ValueError, RecursionError):
            # Not one object -- perhaps a JSONL event stream (codex exec
            # --json), which reports usage per turn. Still silent on failure:
            # a stream with no usage events is a harness that reports nothing.
            return _usage_from_events(raw)
        return _usage_from_envelope(envelope)

    def _exec(self, workspace: Path, argv: list[str], stdin: str) -> None:
        with (
            self._open_capture(workspace, STDOUT_FILE) as stdout,
            self._open_capture(workspace, STDERR_FILE) as stderr,
        ):
            try:
                proc = subprocess.Popen(
                    argv,
                    text=True,
                    encoding="utf-8",
                    **self._spawn_kwargs(workspace, stdout, stderr),
                )
            except OSError as exc:
                raise ValueError(
                    f"[{self.model_name}] harness infrastructure failure: cannot "
                    f"start {self.command[0]!r}: {exc}"
                ) from exc
            try:
                proc.communicate(input=stdin, timeout=self.request_timeout)
            except subprocess.TimeoutExpired as exc:
                raise self._timeout_error(workspace) from exc
            finally:
                # Runs on success too: the leader exiting does not reap the MCP
                # servers it started, and they hold half a gigabyte each.
                _kill_group(proc.pid)
                try:
                    proc.wait(timeout=_REAP_GRACE)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "[HarnessChat:{}] harness pid {} survived SIGKILL "
                        "for {}s; leaking the process group",
                        self.model_name,
                        proc.pid,
                        _REAP_GRACE,
                    )
        self._check_exit(proc.returncode, workspace)

    async def _aexec(self, workspace: Path, argv: list[str], stdin: str) -> None:
        """Async twin of :meth:`_exec`.

        Native asyncio rather than a thread, so cancelling the DAG kills the
        harness instead of orphaning it: an executor thread cannot be
        interrupted, and an abandoned harness would keep its memory and run
        past the concurrency cap for the rest of ``request_timeout``.
        """
        payload = stdin.encode("utf-8")
        with (
            self._open_capture(workspace, STDOUT_FILE) as stdout,
            self._open_capture(workspace, STDERR_FILE) as stderr,
        ):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv, **self._spawn_kwargs(workspace, stdout, stderr)
                )
            except OSError as exc:
                raise ValueError(
                    f"[{self.model_name}] harness infrastructure failure: cannot "
                    f"start {self.command[0]!r}: {exc}"
                ) from exc
            try:
                await asyncio.wait_for(
                    proc.communicate(payload), timeout=self.request_timeout
                )
            except TimeoutError as exc:
                raise self._timeout_error(workspace) from exc
            finally:
                _kill_group(proc.pid)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_REAP_GRACE)
                except TimeoutError:
                    logger.warning(
                        "[HarnessChat:{}] harness pid {} survived SIGKILL "
                        "for {}s; leaking the process group",
                        self.model_name,
                        proc.pid,
                        _REAP_GRACE,
                    )
        self._check_exit(proc.returncode, workspace)

    def _read_output(self, workspace: Path, name: str = OUTPUT_FILE) -> dict[str, Any]:
        path = workspace / name
        if not path.is_file():
            raise ValueError(
                f"[{self.model_name}] harness wrote no {name}, workspace={workspace}"
            )
        if path.resolve().parent != workspace.resolve():
            # Catches the symlink an over-eager harness leaves when it answers
            # with a file from elsewhere. Not a boundary: a hardlink passes, and
            # so does a swap between this check and the read below. The harness
            # runs as this user and could copy the bytes in anyway -- containment
            # is the command's job, not this check's.
            raise ValueError(
                f"[{self.model_name}] {name} resolves outside its "
                f"workspace: {path.resolve()}, workspace={workspace}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, RecursionError) as exc:
            # ValueError covers JSONDecodeError, a bad encoding, and the integer
            # digit limit; RecursionError is what deep nesting raises, and it is
            # not a ValueError, so it would otherwise escape the contract.
            raise ValueError(
                f"[{self.model_name}] {name} is not valid JSON: {exc}, "
                f"workspace={workspace}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"[{self.model_name}] harness infrastructure failure: cannot "
                f"read {name}: {exc}, workspace={workspace}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"[{self.model_name}] {name} must hold a JSON object, got "
                f"{type(payload).__name__}, workspace={workspace}"
            )
        return payload

    def _read_answer(self, workspace: Path) -> dict[str, Any]:
        if not self.schema_flag:
            return self._read_output(workspace)
        if self.answer_file_flag:
            # The CLI wrote the answer file itself, into a directory it owned:
            # same hazards, same checks as the OUTPUT.json handshake.
            return self._read_output(workspace, ANSWER_FILE)
        return self._read_envelope(workspace)

    def _read_envelope(self, workspace: Path) -> dict[str, Any]:
        """Take the answer out of the harness's own stdout envelope.

        Loud everywhere :meth:`_read_usage` is silent: under ``schema_flag``
        the envelope carries the answer, so every hazard the shared
        :meth:`_read_stdout` raises is a failed call here, not a zero.
        """
        data = self._read_stdout(workspace)
        try:
            envelope = json.loads(data.decode("utf-8"))
        except (ValueError, RecursionError) as exc:
            excerpt = data.decode("utf-8", "replace")[:_STDERR_EXCERPT]
            raise ValueError(
                f"[{self.model_name}] harness stdout is not valid JSON: {exc}: "
                f"{excerpt!r}, workspace={workspace}"
            ) from exc
        answer = envelope.get(self.answer_key) if isinstance(envelope, dict) else None
        if not isinstance(answer, dict):
            # Quote the envelope itself: a harness that errored at its own
            # level -- quota, auth -- says so here and exits 0, and this
            # message is the only place that reason reaches the run log.
            excerpt = json.dumps(envelope)[:_STDERR_EXCERPT]
            raise ValueError(
                f"[{self.model_name}] harness stdout envelope holds no JSON "
                f"object at {self.answer_key!r}: {excerpt}, workspace={workspace}"
            )
        return answer

    # -- BaseChatModel -----------------------------------------------------

    def _result(
        self,
        payload: dict[str, Any],
        json_schema: dict[str, Any] | None,
        usage: tuple[dict[str, Any], dict[str, Any]],
    ) -> ChatResult:
        """Render a payload as a chat result.

        Structured calls carry the whole object as JSON text, the same shape an
        OpenAI-compatible model returns, so the structured runnable parses the
        content instead of reaching around the callback machinery for it.
        """
        if json_schema is None:
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"[{self.model_name}] {self._answer_source} has no string "
                    f"'text' field: {payload!r}"
                )
            content = text
        else:
            content = json.dumps(payload)
        return ChatResult(
            generations=[ChatGeneration(message=_harness_message(content, *usage))]
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        schema = self._wire_schema(TEXT_SCHEMA if json_schema is None else json_schema)
        workspace, system, user = self._write_workspace(messages, schema)
        self._exec(
            workspace,
            self._argv(schema, system, workspace),
            self._stdin_text(system, user),
        )
        payload = self._read_answer(workspace)
        return self._result(payload, json_schema, self._read_usage(workspace))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Off the loop: a harness is free to answer with megabytes, and reading
        # that inline stalls every other call sharing this loop.
        schema = self._wire_schema(TEXT_SCHEMA if json_schema is None else json_schema)
        workspace, system, user = await asyncio.to_thread(
            self._write_workspace, messages, schema
        )
        await self._aexec(
            workspace,
            self._argv(schema, system, workspace),
            self._stdin_text(system, user),
        )
        payload = await asyncio.to_thread(self._read_answer, workspace)
        usage = await asyncio.to_thread(self._read_usage, workspace)
        return self._result(payload, json_schema, usage)

    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        """Return a runnable that answers against ``schema``.

        ``method`` and other negotiation kwargs are accepted and ignored: the
        backend has exactly one wire format per mode — ``SCHEMA.json``, or the
        command line under ``schema_flag`` — so there is nothing to negotiate.
        """
        json_schema, model_cls = _resolve_schema(schema)
        return _HarnessStructuredOutput(self, json_schema, model_cls, include_raw)


class _HarnessStructuredOutput(Runnable):
    """Structured-output view over a :class:`HarnessChat`.

    Emits the ``{"raw", "parsed"}`` mapping ``_StructuredOutputRouter`` expects
    so the router's tracking and error contract apply unchanged. Both entry
    points go through ``HarnessChat`` itself rather than calling the workspace
    machinery directly, because that is what carries ``config`` — and with it
    the Langfuse handler and the prompt I/O dump the router attaches.
    """

    def __init__(
        self,
        chat: HarnessChat,
        json_schema: dict[str, Any],
        model_cls: type[BaseModel] | None,
        include_raw: bool,
    ) -> None:
        self._chat = chat
        self._json_schema = json_schema
        self._model_cls = model_cls
        self._include_raw = include_raw

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        message = self._chat.invoke(
            input, config, json_schema=self._json_schema, **kwargs
        )
        return self._finish(message)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        message = await self._chat.ainvoke(
            input, config, json_schema=self._json_schema, **kwargs
        )
        return self._finish(message)

    def _finish(self, message: BaseMessage) -> Any:
        try:
            payload = json.loads(_extract_content_text(message.content))
            if self._chat.strict_schema:
                # _json_schema is the ORIGINAL schema — the strip drops
                # exactly the nulls the wire rewrite invited, nothing else.
                payload = strip_strict_nulls(payload, self._json_schema)
            parsed = self._parse(payload)
        except ValueError as exc:
            if not self._include_raw:
                raise
            # Under include_raw the failure is the router's to raise: its
            # _process tracks the raw message's tokens first, and raising here
            # instead loses the tokens of exactly the calls that failed.
            return {"raw": message, "parsed": None, "parsing_error": exc}
        if not self._include_raw:
            return parsed
        return {"raw": message, "parsed": parsed, "parsing_error": None}

    def _parse(self, payload: dict[str, Any]) -> Any:
        if self._model_cls is not None:
            # ValidationError is a ValueError, matching the router's contract.
            return self._model_cls.model_validate(payload)
        missing = [
            key for key in self._json_schema.get("required", []) if key not in payload
        ]
        if missing:
            raise ValueError(
                f"[{self._chat.model_name}] {self._chat._answer_source} is "
                f"missing required fields {missing}, got keys {sorted(payload)}"
            )
        return payload
