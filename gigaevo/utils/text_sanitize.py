"""Pure text sanitization for LLM-derived strings.

Three increasingly strict modes that compose freely. Most callers want
``sanitize_for_log``; the other two are minimal helpers for cases where
the destination only rejects a narrower set of bytes.

All functions are pure ``str -> str``, idempotent, and preserve printable
Unicode (CJK, Greek letters, math symbols, emoji, directional arrows).
Lone UTF-16 surrogates always collapse to U+FFFD.

The threat surface this module guards covers compiler error text from
heterogeneous LLM targets (Python tracebacks, Triton MLIR diagnostics,
nvcc / ptxas / CUTLASS template explosions with embedded ANSI from gcc /
clang colorization, Mojo error formatter output with Unicode arrows,
Pallas / JAX jaxpr traces with ASCII art, CuTe layout errors with
``Layout<Shape<...>,Stride<...>>`` template syntax). Each toolchain emits
its own conventions; the sanitizers reject what would break log files,
JSON encoders, or Postgres TEXT columns, and preserve everything else.
"""

from __future__ import annotations

import re
from typing import Final

# ============================================================================
# Compiled patterns (private)
# ============================================================================


# ANSI escape sequences. Covers CSI (the common ``\x1b[...m`` colorization
# from gcc / clang / nvcc), OSC (xterm title-setting and similar), DCS / SOS
# / PM / APC string sequences, and single-character Fe escapes. Reference:
# ECMA-48 + xterm Ctrl Sequences.
_ANSI_RE: Final[re.Pattern[str]] = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI: ESC [ params intermediates final
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC: ESC ] ... BEL or ST
    r"|\x1b[PX^_][^\x1b]*\x1b\\"  # DCS / SOS / PM / APC: ESC X ... ST
    r"|\x1b[@-Z\\-_]"  # Fe single-char: ESC <C1 byte>
)

# C0 controls (0x00-0x1F + 0x7F DEL) except TAB (0x09) and LF (0x0A). CR
# (0x0D) is included so log-line forgery via "\r\nFAKE LINE" is defused
# while real multi-line tracebacks (which use LF only) pass through.
_C0_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# C1 controls (0x80-0x9F). Rare in modern output but legal in Python str and
# capable of confusing terminal parsers.
_C1_RE: Final[re.Pattern[str]] = re.compile(r"[\x80-\x9f]")

# Unicode BIDI overrides and isolates (U+202A..U+202E plus U+2066..U+2069).
# Invisible characters used to spoof text directionality; have no place in
# machine-readable log records.
_BIDI_RE: Final[re.Pattern[str]] = re.compile(r"[‪-‮⁦-⁩]")

# UTF-16 surrogate code points (U+D800-U+DFFF). Python ``str`` is sequence-of-
# code-points, not UTF-16, so a "high then low" pair is NOT decoded as one
# astral character; both halves remain independent code points that the
# UTF-8 encoder refuses (``surrogates not allowed``). Match every surrogate
# unconditionally and replace with U+FFFD. Real astral characters in Python
# ``str`` are single code points above U+FFFF, never surrogate pairs.
_LONE_SURROGATE_RE: Final[re.Pattern[str]] = re.compile(r"[\ud800-\udfff]")

# Identifier strip pattern. Anything outside the conservative charset is
# removed. Slash and colon are permitted because model names sometimes
# carry path-like or provider-prefix forms (``openai:gpt-4o-mini``,
# ``/local/path/to/model.gguf``); ``@`` is permitted because some configs
# route via ``model@host:port`` against local vllm / sglang / tgi servers.
_IDENTIFIER_STRIP_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._:/+@\-]")


def _escape_control(match: re.Match[str]) -> str:
    return f"\\x{ord(match.group()):02x}"


# ============================================================================
# Public API
# ============================================================================


def sanitize_for_log(text: str) -> str:
    """Make ``text`` safe for log sinks, JSON encoders, and Postgres TEXT.

    Strips ANSI escape sequences (no terminal control on log readers), strips
    BIDI overrides (no directional spoofing in log records), escapes C0 and
    C1 control characters except TAB and LF (visible but inert; for example
    ``\\x07`` BEL becomes the four-character literal ``\\x07``, and
    ``\\x0d`` CR is escaped so single-line log records cannot be forged via
    carriage-return overwriting), and replaces lone UTF-16 surrogates with
    U+FFFD. Result is valid UTF-8, safe for ``json.dumps`` /
    ``model_dump_json``, and safe for ``asyncpg`` TEXT columns.

    LF (``\\n``) is preserved so legitimate multi-line tracebacks survive.
    A consequence is that a hostile string containing ``"\\n[FORGED]"``
    will produce a second line in plain-text log sinks. Reasonable log
    parsers should recognize that authentic entries begin with a timestamp
    prefix and treat untimestamped continuations accordingly. Callers that
    cannot accept this residual risk (for example, plain-text sinks read
    by line-naive consumers) should pipeline through a stricter step that
    escapes ``\\n`` as well.
    """
    text = _ANSI_RE.sub("", text)
    text = _BIDI_RE.sub("", text)
    text = _C0_RE.sub(_escape_control, text)
    text = _C1_RE.sub(_escape_control, text)
    text = _LONE_SURROGATE_RE.sub("�", text)
    return text


def sanitize_for_json(text: str) -> str:
    """Minimum-viable fix for JSON encoders. Replaces lone UTF-16 surrogates
    with U+FFFD so ``json.dumps`` and ``pydantic.model_dump_json`` succeed.
    Preserves every other byte verbatim — control characters, ANSI escape
    sequences, BIDI overrides all pass through. Compose with
    ``sanitize_for_log`` when the destination also forbids those.
    """
    return _LONE_SURROGATE_RE.sub("�", text)


def sanitize_for_dbtext(text: str) -> str:
    """Make ``text`` safe for Postgres TEXT columns through asyncpg.

    Handles two failure modes asyncpg surfaces on LLM-derived text. First,
    the driver rejects literal NUL bytes outright (``A string literal cannot
    contain NUL (0x00) characters``) — NUL is replaced with U+FFFD. Second,
    asyncpg UTF-8 encodes string values before sending them on the wire,
    and Python's UTF-8 encoder refuses lone UTF-16 surrogates; those are
    also replaced with U+FFFD. ANSI escape sequences and other control
    bytes pass through verbatim because Postgres TEXT accepts them; compose
    with ``sanitize_for_log`` when the column is also displayed to humans
    or consumed by log readers.
    """
    text = text.replace("\x00", "�")
    text = _LONE_SURROGATE_RE.sub("�", text)
    return text


def deep_sanitize_for_json(value: object) -> object:
    """Walk a JSON-shaped structure and apply ``sanitize_for_json`` to every
    string leaf. Lists / tuples / dicts are rebuilt recursively; primitive
    non-string values (``int`` / ``float`` / ``bool`` / ``None``) pass
    through unchanged.

    Use at the boundary where an arbitrary container of LLM-derived text
    will be handed to ``json.dumps`` / ``orjson.dumps`` / pydantic
    ``model_dump_json``. A single lone surrogate buried anywhere in the
    structure would otherwise raise ``UnicodeEncodeError`` from inside the
    serializer and abort the write. This is the cheap belt that stops
    those aborts without rewriting every producer.

    Returns ``object`` rather than a tight type because the input shape is
    arbitrary JSON. Callers narrow at the call site via ``cast`` or by
    knowing the input shape; the function preserves the outermost
    container type (``dict`` stays ``dict``, ``list`` stays ``list``,
    ``tuple`` stays ``tuple``).
    """
    if isinstance(value, str):
        return sanitize_for_json(value)
    if isinstance(value, dict):
        return {
            deep_sanitize_for_json(k): deep_sanitize_for_json(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [deep_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return tuple(deep_sanitize_for_json(v) for v in value)
    return value


def clean_identifier(text: str, *, max_len: int | None = None) -> str:
    """Strip every character outside the conservative identifier charset
    ``[A-Za-z0-9._:/+@-]`` and optionally truncate to ``max_len`` characters.

    Returns an empty string if nothing in ``text`` survives. Callers decide
    whether to reject, fall back to a default, or warn. Intended for places
    where a string must be a stable, displayable, file-system-safe
    identifier (model names, cache keys, log tags).

    Raises ``ValueError`` if ``max_len`` is negative; the slice ``[:negative]``
    would silently drop trailing characters without warning, which is almost
    never what the caller meant.
    """
    if max_len is not None and max_len < 0:
        raise ValueError(f"max_len must be non-negative, got {max_len}")
    cleaned = _IDENTIFIER_STRIP_RE.sub("", text)
    if max_len is not None and len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


__all__ = (
    "clean_identifier",
    "deep_sanitize_for_json",
    "sanitize_for_dbtext",
    "sanitize_for_json",
    "sanitize_for_log",
)
