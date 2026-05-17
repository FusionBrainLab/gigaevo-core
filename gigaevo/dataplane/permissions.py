"""Move-only ``Token[Tag]`` linear permission tokens.

A token's presence in a function signature is the witness that the
caller is the sole writer of the subspace tagged by ``Tag``. Tokens are
*linear*: each one can be consumed exactly once, and they cannot be
duplicated.

Python's type system cannot enforce linearity at compile time, so this
module recovers most of the safety at runtime:

    - ``__copy__`` / ``__deepcopy__`` raise — tokens cannot be cloned
      via the standard ``copy`` machinery;
    - ``__reduce__`` / ``__reduce_ex__`` / ``__getstate__`` raise — no
      protocol-style serialiser can silently produce a duplicate;
    - a ``_consumed`` flag on each instance flips on ``consume()``; a
      second call raises :class:`TokenAlreadyConsumed`;
    - factories ``mint_root`` / ``mint_split`` / ``mint_split_n`` are
      the only legitimate paths to mint a token, and the split factories
      reject duplicate child tags so two orthogonal sub-tokens cannot
      accidentally claim the same subspace.

Concurrency model: the dataplane is single-threaded asyncio. ``consume``
is therefore not protected by a lock — the read-modify-write on
``_consumed`` is safe only because no two coroutines can preempt each
other between the read and the write. Tokens MUST NOT be shared across
threads or processes; the move-only contract implicitly forbids that.

A custom ruff rule (see ``lints.toml``) flags ``t.consume()`` followed
by any further use of ``t`` so most linear-flow violations are caught at
lint time too.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, SupportsIndex

from .errors import (
    TokenAlreadyConsumed,
    TokenNotPickleable,
    TokenTagCollisionError,
)


class Token[Tag]:
    """Linear, move-only permission witness for the subspace ``Tag``.

    Callers SHOULD mint tokens via :func:`mint_root` etc. rather than
    constructing directly — those factories exist as a single grep
    target.

    The class is effectively ``final``: :meth:`__init_subclass__` refuses
    any subclass. Subclassing would let a derived class override
    :meth:`__copy__` / :meth:`__deepcopy__` / :meth:`__reduce__` and slip
    a duplicated witness through Python's copy / pickle machinery,
    breaking the single-writer-per-subspace contract. The factories
    intentionally instantiate :class:`Token` directly so all valid mint
    paths produce the sealed class.
    """

    __slots__ = ("_tag", "_consumed")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # ``Token`` is the only valid concrete linear-permission class;
        # subclasses would defeat linearity by overriding the cloning
        # guards. The guard is documentary as much as runtime — the
        # error message points the reader at :func:`mint_split` which
        # is the legitimate way to derive a fresh witness.
        raise TypeError(
            "Token<Tag> is final: subclassing would let a derived class "
            "override __copy__ / __deepcopy__ / __reduce__ and silently "
            "duplicate the witness. Use mint_split to derive an "
            "orthogonal sub-token."
        )

    def __init__(self, tag: Tag) -> None:
        self._tag: Tag = tag
        self._consumed: bool = False

    # ── linearity enforcement ────────────────────────────────────────

    def __copy__(self) -> Token[Tag]:
        raise TypeError(
            "Token<Tag> is move-only: duplicating would create two "
            "simultaneous owners of the same subspace, breaking single-"
            "writer guarantees. Use mint_split to derive an orthogonal "
            "sub-token instead."
        )

    def __deepcopy__(self, _: dict[int, Any]) -> Token[Tag]:
        raise TypeError(
            "Token<Tag> is move-only: deepcopy would duplicate the "
            "witness. Use mint_split to derive an orthogonal sub-token "
            "instead."
        )

    def __reduce__(self) -> Any:
        raise TokenNotPickleable(tag_repr=repr(self._tag))

    def __reduce_ex__(self, _protocol: SupportsIndex) -> Any:
        raise TokenNotPickleable(tag_repr=repr(self._tag))

    def __getstate__(self) -> Any:
        # Some pickler / copier paths probe ``__getstate__`` before
        # falling through to ``__reduce__``; deny it for symmetry.
        raise TokenNotPickleable(tag_repr=repr(self._tag))

    def __setstate__(self, _state: Any) -> None:
        # Cannot legitimately be reached because ``__getstate__`` raises,
        # but pickle protocol 0 and ``copy.copy`` may probe attributes
        # directly. Closing this path defends against future protocols.
        raise TokenNotPickleable(tag_repr=repr(self._tag))

    # ── consumption ──────────────────────────────────────────────────

    @property
    def tag(self) -> Tag:
        """The phantom subspace tag carried by this token.

        Reading the tag is allowed even after :meth:`consume` — it's
        useful for debugging and post-consume cleanup logging.
        """
        return self._tag

    @property
    def consumed(self) -> bool:
        return self._consumed

    def consume(self) -> Tag:
        """Consume the token and return its tag.

        Each token can be consumed exactly once. A second call raises
        :class:`TokenAlreadyConsumed`.
        """
        if self._consumed:
            raise TokenAlreadyConsumed(tag_repr=repr(self._tag))
        self._consumed = True
        return self._tag

    def __repr__(self) -> str:
        state = "consumed" if self._consumed else "live"
        return f"Token({self._tag!r}, {state})"


# ── factories ─────────────────────────────────────────────────────────


def mint_root[Tag](tag: Tag) -> Token[Tag]:
    """Mint a fresh root token for ``tag``.

    There should be exactly one call site per logical subspace per
    process — typically at engine startup. The token then flows down
    the call graph via moves and ``mint_split``.
    """
    return Token(tag)


def _reject_duplicate_tags[T](tags: Iterable[T]) -> list[T]:
    """Return the materialised tag list after enforcing distinctness.

    Tags may be any value the caller cares to use — including
    non-hashable structures — so distinctness is checked by linear scan
    rather than by set membership. The cost is O(n^2); call sites mint
    a handful of children at a time, so this is fine.
    """
    materialised: list[T] = list(tags)
    for i, tag in enumerate(materialised):
        for earlier in materialised[:i]:
            if earlier == tag:
                raise TokenTagCollisionError(duplicate_tag_repr=repr(tag))
    return materialised


def mint_split[In, L, R](
    parent: Token[In],
    left_tag: L,
    right_tag: R,
) -> tuple[Token[L], Token[R]]:
    """Consume ``parent`` and mint two orthogonal sub-tokens.

    The parent is consumed so callers cannot accidentally use it after
    the split. ``left_tag`` and ``right_tag`` must be distinct values —
    a duplicate would silently mint two tokens claiming the same
    subspace, defeating the linearity guarantee. Distinctness is
    enforced at runtime via :class:`TokenTagCollisionError`.

    The parent is consumed *before* the duplicate check so an invalid
    split still surrenders the parent; the caller's flow is broken in
    a clearly visible way (token is gone *and* an exception fires)
    rather than silently leaving an apparently-valid parent in the
    caller's hand after a failed split.
    """
    parent.consume()
    if left_tag == right_tag:
        raise TokenTagCollisionError(duplicate_tag_repr=repr(left_tag))
    return Token(left_tag), Token(right_tag)


def mint_split_n[In, L](parent: Token[In], child_tags: Iterable[L]) -> list[Token[L]]:
    """Consume ``parent`` and mint N orthogonal sub-tokens.

    Useful when fanning a root permission across N workers or N actor
    instances. The child tags must be pairwise distinct; duplicates
    raise :class:`TokenTagCollisionError`.

    Parent consumption happens first (see :func:`mint_split` for the
    rationale).
    """
    parent.consume()
    tags = _reject_duplicate_tags(child_tags)
    return [Token(t) for t in tags]


__all__ = [
    "Token",
    "mint_root",
    "mint_split",
    "mint_split_n",
]
