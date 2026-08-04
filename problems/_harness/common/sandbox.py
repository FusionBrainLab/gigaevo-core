"""Supervised execution of LLM-generated candidate code.

Candidate code is untrusted in the specific sense that it was written by a model
and never read by a human: it may loop forever, fork, exhaust memory, print
megabytes, crash the interpreter, or return something that cannot be serialized.
Any of those must cost the run one proposal, not the run.

The candidate lives in a SPAWNED worker — a fresh interpreter that rebuilds the
program from its pickled class — and never in a forked one. That is not a
stylistic choice:

  * Importing a candidate that uses JAX starts ~440 threads in the importing
    process, and forking from a multithreaded process deadlocks the child (JAX
    prints the warning itself). GigaEvo hands `validate()` a class that has
    ALREADY been unpickled, so the grading process is already poisoned before the
    first proposal. A fork-based cage therefore hangs on proposal 0 for every
    JAX candidate — including the five programs that produced the published
    ImprovEvolve numbers.
  * The worker OUTLIVES the proposal, so the program keeps its JIT cache between
    calls, exactly as it did when the published pipeline ran it in-process.
    Re-spawning per proposal would charge a JAX program its ~3 s of compilation
    on all ~175 calls and hand the numpy-family arms an advantage the method
    never had.

Memory is capped by RESIDENT size, polled by the parent — not by RLIMIT_AS.
Address space is not memory: jaxlib maps ~50 GiB of it while its resident set
stays under a gigabyte, so an address-space cap of any defensible size aborts
every JAX candidate at import with SIGABRT.

The sandbox stays benchmark-free. It decides whether a proposal *ran*; whether
the config it produced is feasible is a validator's job, so values like NaN pass
through untouched rather than being silently rejected here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

import cloudpickle

from problems._harness.common.contracts import (
    Budget,
    Config,
    ProposalResult,
    ProposalStatus,
)
from problems._harness.common.errors import ProtocolError
from problems._harness.common.events import EventLogger

_HEADER = struct.Struct(">I")
_POLL_S = 0.05
_MAX_ERROR_CHARS = 2000
_REAP_S = 2.0
_REPO = Path(__file__).resolve().parents[2]


class CandidateTimeout(Exception):
    """The candidate did not answer inside the proposal's wall-clock deadline."""


class CandidateCrashed(Exception):
    """The candidate raised, died, or overran its memory cap."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message[:_MAX_ERROR_CHARS]


class CandidateReturnedGarbage(Exception):
    """The candidate returned something that cannot cross the process boundary."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message[:_MAX_ERROR_CHARS]


@dataclass(frozen=True)
class SandboxLimits:
    """Identical for every representation — an unequal sandbox is a confound.

    cpu_cores is what turns the wall-clock budget into an equal *compute* budget:
    a candidate whose BLAS calls fan out over every core would otherwise get many
    times the compute of a single-threaded rival in the same wall time. It is
    enforced twice — by CPU affinity, which the kernel inherits into anything the
    candidate forks and which the candidate cannot undo, and by the thread-count
    environment the maths libraries read, so a candidate does not oversubscribe
    its own core.

    gpu_visible is false for the same reason: a candidate reaching the GPU would
    get orders of magnitude more compute per wall second than a numpy rival, and
    its score would then depend on how much device memory somebody else's job
    happened to be holding.
    """

    wall_timeout_s: float
    resident_bytes: int | None = None
    cpu_cores: int | None = None
    capture_bytes: int = 8192
    gpu_visible: bool = False


def child_environment(limits: SandboxLimits) -> dict[str, str]:
    env = dict(os.environ)
    if not limits.gpu_visible:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if limits.cpu_cores is not None:
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[variable] = str(limits.cpu_cores)
    return env


def require_cpu_only_grading_process(limits: SandboxLimits) -> None:
    """Hiding the GPU from the candidate is not enough to hide it from the run.

    GigaEvo unpickles the candidate class into the GRADING process, so a JAX
    program initializes its CUDA backend there — before any code of ours runs, and
    therefore too late for us to set the variable ourselves. That process then
    reserves device memory belonging to whoever else is on the box, and its
    behaviour starts depending on how much of it they left free. A run launched
    without the variable is not the CPU-only protocol the manifest describes, so
    it fails at the first candidate rather than quietly producing numbers from a
    different experiment.

    Unset means "every GPU is visible" and is a failure; the variable must be
    present and empty.
    """
    if limits.gpu_visible or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return
    raise ProtocolError(
        "the manifest declares a CPU-only protocol (security.gpu_visible: false), "
        "but this process can see the GPU. Export CUDA_VISIBLE_DEVICES='' before "
        "launching the run or the smoke."
    )


def pin_cores(cpu_cores: int) -> None:
    """Offset the slice by pid so concurrent candidates do not all pile onto the
    same core, which would make wall time depend on how many runs are in flight."""
    available = sorted(os.sched_getaffinity(0))
    if cpu_cores >= len(available):
        return
    start = os.getpid() % len(available)
    chosen = [
        available[(start + offset) % len(available)] for offset in range(cpu_cores)
    ]
    os.sched_setaffinity(0, set(chosen))


def resident_bytes(pid: int) -> int:
    """Resident set of a process, in bytes. 0 once it is gone."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return 0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def terminate_group(process: subprocess.Popen) -> None:
    """Kill the group, so a candidate's own forked workers die with it, and then
    the worker itself in case it never reached its new session."""
    for kill in (lambda: os.killpg(process.pid, signal.SIGKILL), process.kill):
        try:
            kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.wait(timeout=_REAP_S)
    except subprocess.TimeoutExpired:
        pass


@dataclass
class CandidateWorker:
    """One spawned interpreter holding one live candidate program.

    The grading process never imports the candidate: it ships the class, and the
    worker rebuilds it. A worker killed for timing out or for eating its memory
    cap is replaced on the next proposal, so the cost of the restart lands on the
    candidate that earned it and cannot leak into the next one.
    """

    program_class: Any
    kwargs: dict[str, Any]
    limits: SandboxLimits
    deadline: float = field(default=0.0, init=False)
    _process: subprocess.Popen | None = field(default=None, init=False)
    _workdir: Path | None = field(default=None, init=False)
    _read_at: dict[str, int] = field(default_factory=dict, init=False)
    _unbuildable: CandidateCrashed | CandidateTimeout | None = field(
        default=None, init=False
    )

    def start(self, timeout_s: float) -> None:
        """Get the candidate ready to be called, on the harness's clock rather than
        the candidate's.

        Spawning an interpreter and importing what the candidate imports happens
        before the program computes anything, and on a busy box it is the larger cost:
        measured here on (d=12, N=343), a worker is ready in 2.2s alone and in 33-40s
        when 42 start at once, while the candidate's own call stays at 0.75s under
        both. Billed to the search budget, that made the score a property of the
        machine rather than of the program — under the seed gate's budget a candidate
        that passes cleanly alone was sentinelled under load, having never been called
        once — and it fell hardest on candidates that import heavy libraries, so an arm
        whose interface makes the model reach for JAX more often was penalised for its
        ARM. Start the worker here, then start the clock.

        A RESTART mid-run stays on the candidate's clock, deliberately (see the class
        docstring): the candidate that timed out or ate its memory cap earned that
        cost. This is the first start, which nobody earned.

        Failures are held, not raised: a candidate that cannot be built is a candidate
        whose proposals fail, not a harness that crashed mid-grade. The first call
        re-raises what happened here, and the supervisor classifies it exactly as it
        did when startup was lazy.
        """
        if self._process is not None or self._unbuildable is not None:
            return
        self.deadline = time.monotonic() + timeout_s
        try:
            self._spawn()
        except CandidateTimeout:
            # It never became ready. Leaving it to respawn on the first call would
            # only hang again, and bill the candidate for it.
            self._unbuildable = CandidateTimeout(
                f"candidate was not ready to be called within {timeout_s:.0f}s"
            )
        except CandidateCrashed:
            pass  # _spawn already held it in _unbuildable; call() re-raises it
        except Exception as err:
            # The BOX refusing to give us a worker — Popen raising OSError because 112
            # spawns went up at once against a memory cap — is not a candidate exception,
            # and starting eagerly moved it out of the supervised call that used to catch
            # it. Unheld, one unlucky fork would take down the whole grade, under exactly
            # the load that makes it hardest to read. It is a dead candidate slot, not a
            # broken run.
            self._unbuildable = CandidateCrashed(
                type(err).__name__, f"worker could not be started: {err}"
            )

    def call(self, method: str, args: tuple[Any, ...]) -> Config:
        """Run one method of the candidate, raising rather than returning a status
        so the adapters can compose calls exactly as they do in-process."""
        if self._unbuildable is not None:
            raise self._unbuildable
        if self._process is None:
            self._spawn()
        self._send({"method": method, "args": args})
        return self._receive()

    def drain_output(self) -> tuple[str, str]:
        """Whatever the candidate printed since the last proposal, tail-bounded so
        a candidate that prints in a loop cannot fill the event log."""
        return self._drain("stdout.txt"), self._drain("stderr.txt")

    def close(self) -> None:
        if self._process is not None:
            terminate_group(self._process)
            self._process = None
        self._release_workdir()

    def _release_workdir(self) -> None:
        if self._workdir is None:
            return
        for path in self._workdir.iterdir():
            path.unlink(missing_ok=True)
        self._workdir.rmdir()
        self._workdir = None

    def _drain(self, name: str) -> str:
        if self._workdir is None:
            return ""
        path = self._workdir / name
        if not path.exists():
            return ""
        with open(path, "rb") as handle:
            handle.seek(self._read_at.get(name, 0))
            blob = handle.read()
            self._read_at[name] = handle.tell()
        return blob[-self.limits.capture_bytes :].decode("utf-8", errors="replace")

    def _spawn(self) -> None:
        self._release_workdir()  # the previous worker's, if it was killed
        self._workdir = Path(tempfile.mkdtemp(prefix="gigaevo-harness-worker-"))
        self._read_at = {}
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-u",  # a killed worker must still have flushed what it printed
                "-m",
                "problems._harness.common.sandbox",
                str(self._workdir / "stdout.txt"),
                str(self._workdir / "stderr.txt"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=child_environment(self.limits),
            cwd=_REPO,
            start_new_session=True,
        )
        setup = {
            "program_class": self.program_class,
            "kwargs": self.kwargs,
            "cpu_cores": self.limits.cpu_cores,
        }
        try:
            self._send(setup)
            self._receive()
        except CandidateTimeout:
            raise
        except CandidateCrashed as crashed:
            # A class that cannot be rebuilt fails the same way every time;
            # respawning it 175 times would spend the run's budget re-deriving a
            # verdict the first attempt already settled.
            self._unbuildable = crashed
            raise
        except Exception as err:  # the class itself will not pickle into a worker
            self._unbuildable = CandidateCrashed(
                type(err).__name__, f"candidate class cannot be sent to a worker: {err}"
            )
            raise self._unbuildable from err

    def _send(self, payload: dict[str, Any]) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            blob = cloudpickle.dumps(payload)
            self._process.stdin.write(_HEADER.pack(len(blob)))
            self._process.stdin.write(blob)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as err:
            raise self._died(f"worker is gone: {err}") from err

    def _receive(self) -> Config:
        payload = cloudpickle.loads(self._read_frame())
        if payload["status"] is ProposalStatus.SUCCESS:
            return payload["value"]
        if payload["status"] is ProposalStatus.INVALID_RETURN:
            raise CandidateReturnedGarbage(payload["error_message"])
        raise CandidateCrashed(payload["error_type"], payload["error_message"])

    def _read_frame(self) -> bytes:
        (size,) = _HEADER.unpack(self._read_exactly(_HEADER.size))
        return self._read_exactly(size)

    def _read_exactly(self, size: int) -> bytes:
        assert self._process is not None and self._process.stdout is not None
        stream = self._process.stdout
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            self._guard()
            ready, _, _ = select.select([stream], [], [], _POLL_S)
            if not ready:
                continue
            chunk = stream.read1(remaining)
            if not chunk:
                raise self._died("worker exited without answering")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _guard(self) -> None:
        """The two ways a proposal is stopped from outside: it ran out of wall
        clock, or it ate memory its rivals are not allowed to eat."""
        assert self._process is not None
        if time.monotonic() >= self.deadline:
            terminate_group(self._process)
            self._process = None
            raise CandidateTimeout("candidate exceeded its wall timeout")
        cap = self.limits.resident_bytes
        if cap is not None and resident_bytes(self._process.pid) > cap:
            terminate_group(self._process)
            self._process = None
            raise CandidateCrashed(
                "MemoryCap",
                f"candidate exceeded its resident memory cap of {cap} bytes",
            )

    def _died(self, why: str) -> CandidateCrashed:
        assert self._process is not None
        process, self._process = self._process, None
        terminate_group(process)
        returncode = process.poll()
        if returncode is not None and returncode < 0:
            name = signal.Signals(-returncode).name
            return CandidateCrashed(name, f"candidate process was killed by {name}")
        return CandidateCrashed("NoResult", f"{why} (exited with {returncode})")


@dataclass
class SandboxSupervisor:
    """Satisfies SupervisedCall: (call, label) -> ProposalResult, never raises.

    `call` is a closure over the adapters, which drive a RemoteProgram — so what
    is timed and killed is the whole atomic proposal (for ImprovEvolve,
    improve(perturb(x)), not one method call), exactly as it is in-process.
    """

    worker: CandidateWorker
    budget: Budget | None = None
    logger: EventLogger | None = None

    def __call__(self, call: Callable[[], Config], label: str) -> ProposalResult:
        timeout_s = self._timeout_s()
        if timeout_s <= 0:
            result = ProposalResult(
                status=ProposalStatus.TIMEOUT,
                label=label,
                error_type="NoBudget",
                error_message="no wall time remained to launch the proposal",
            )
        else:
            self.worker.deadline = time.monotonic() + timeout_s
            started_at = time.monotonic()
            status, config, error_type, error_message = self._attempt(call)
            stdout, stderr = self.worker.drain_output()
            result = ProposalResult(
                status=status,
                label=label,
                config=config,
                elapsed_s=time.monotonic() - started_at,
                error_type=error_type,
                error_message=error_message,
                stdout=stdout,
                stderr=stderr,
            )
        if self.logger is not None:
            self.logger.emit_proposal(result)
        return result

    def _attempt(
        self, call: Callable[[], Config]
    ) -> tuple[ProposalStatus, Config | None, str | None, str | None]:
        try:
            config = call()
        except CandidateTimeout as err:
            return ProposalStatus.TIMEOUT, None, "WallTimeout", str(err)
        except CandidateReturnedGarbage as err:
            return ProposalStatus.INVALID_RETURN, None, "Unserializable", err.message
        except CandidateCrashed as err:
            return ProposalStatus.EXCEPTION, None, err.error_type, err.message
        except Exception as err:
            return ProposalStatus.EXCEPTION, None, type(err).__name__, str(err)
        if config is None:
            return (
                ProposalStatus.INVALID_RETURN,
                None,
                "NoneReturned",
                "candidate returned None",
            )
        return ProposalStatus.SUCCESS, config, None, None

    def _timeout_s(self) -> float:
        """A candidate may never run past the budget it shares with its rivals."""
        if self.budget is None:
            return self.worker.limits.wall_timeout_s
        return min(self.worker.limits.wall_timeout_s, self.budget.remaining_s)


@dataclass(frozen=True)
class RemoteProgram:
    """Stands in for the candidate wherever a program is expected.

    The adapters are written against a program exposing propose / generate_config
    / perturb / improve / solve. They are handed this instead and are none the
    wiser: every attribute is a method that runs in the worker. Keeping the
    adapters in the grading process is deliberate — the composition rules and the
    defensive copying stay in one tested place shared by all three arms, rather
    than being reimplemented on the far side of a pipe where an arm could quietly
    diverge from its rivals.
    """

    worker: CandidateWorker

    def __getattr__(self, method: str) -> Callable[..., Config]:
        if method.startswith("_"):
            raise AttributeError(method)
        return lambda *args: self.worker.call(method, args)


def _serve(stdout_path: str, stderr_path: str) -> None:
    """The worker. Runs in a fresh interpreter, talks length-prefixed cloudpickle
    frames on stdin/stdout, and never returns."""
    protocol_in = sys.stdin.buffer
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")

    # Anything the candidate prints — from Python or from C — must land in a
    # capture file and not in the protocol stream: otherwise one print() from a
    # candidate corrupts the frame the parent is mid-way through reading.
    for path, fd in ((stdout_path, 1), (stderr_path, 2)):
        capture = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(capture, fd)
        os.close(capture)

    def read_frame() -> dict[str, Any] | None:
        header = protocol_in.read(_HEADER.size)
        if len(header) < _HEADER.size:
            return None
        (size,) = _HEADER.unpack(header)
        return cloudpickle.loads(protocol_in.read(size))

    def write_blob(blob: bytes) -> None:
        protocol_out.write(_HEADER.pack(len(blob)))
        protocol_out.write(blob)
        protocol_out.flush()

    def write_frame(payload: dict[str, Any]) -> None:
        write_blob(cloudpickle.dumps(payload))

    def failure(err: BaseException) -> dict[str, Any]:
        return {
            "status": ProposalStatus.EXCEPTION,
            "value": None,
            "error_type": type(err).__name__,
            "error_message": str(err)[:_MAX_ERROR_CHARS],
        }

    setup = read_frame()
    if setup is None:
        return
    if setup["cpu_cores"] is not None:
        pin_cores(setup["cpu_cores"])
    try:
        program = setup["program_class"](**setup["kwargs"])
    except BaseException as err:
        write_frame(failure(err))
        return
    write_frame({"status": ProposalStatus.SUCCESS, "value": None})

    while True:
        request = read_frame()
        if request is None:
            return
        try:
            value = getattr(program, request["method"])(*request["args"])
        except BaseException as err:
            # BaseException: a candidate that calls sys.exit() has failed one
            # proposal, and must not take the run's warm worker down with it.
            write_frame(failure(err))
            continue
        try:
            blob = cloudpickle.dumps({"status": ProposalStatus.SUCCESS, "value": value})
        except BaseException as err:
            write_frame(
                {
                    "status": ProposalStatus.INVALID_RETURN,
                    "value": None,
                    "error_type": "Unserializable",
                    "error_message": f"candidate result is not serializable: {err}"[
                        :_MAX_ERROR_CHARS
                    ],
                }
            )
            continue
        write_blob(blob)


if __name__ == "__main__":
    _serve(sys.argv[1], sys.argv[2])
