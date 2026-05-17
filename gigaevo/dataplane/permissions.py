"""Move-only ``Token[Tag]`` linear permission tokens.

A token's presence in a function signature witnesses that the caller is
the sole writer of the subspace tagged by ``Tag``. Linear: consumed
exactly once, never duplicated. Python cannot enforce linearity at
compile time; runtime safety comes from:

    - ``__copy__`` / ``__deepcopy__`` raise.
    - ``__reduce__`` / ``__reduce_ex__`` / ``__getstate__`` raise.
    - ``consume()`` flips a ``_consumed`` flag; re-consumption raises
      :class:`TokenAlreadyConsumed`.
    - The :func:`mint_root` / :func:`mint_split` / :func:`mint_split_n`
      factories are the only mint paths; the split factories reject
      duplicate child tags.

Concurrency: the dataplane is single-threaded asyncio, so ``consume``
takes no lock. Tokens MUST NOT be shared across threads or processes;
the move-only contract forbids that implicitly. A ruff rule in
``lints.toml`` flags post-consume reuse statically.
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

    Mint via :func:`mint_root` / :func:`mint_split` / :func:`mint_split_n`
    (single grep target). The class is final via
    :meth:`__init_subclass__` — a subclass could override the cloning
    guards and slip a duplicate witness through copy/pickle.
    """

    __slots__ = ("_tag", "_consumed")

    def __init_subclass__(cls, **kwargs: Any) -> None:
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
        # Some pickler/copier paths probe __getstate__ before __reduce__.
        raise TokenNotPickleable(tag_repr=repr(self._tag))

    def __setstate__(self, _state: Any) -> None:
        # Unreachable in practice; closes the protocol surface defensively.
        raise TokenNotPickleable(tag_repr=repr(self._tag))

    # ── consumption ──────────────────────────────────────────────────

    @property
    def tag(self) -> Tag:
        """The phantom subspace tag. Readable even after :meth:`consume`."""
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
    """Mint a fresh root token for ``tag`` (one call site per subspace)."""
    return Token(tag)


def _reject_duplicate_tags[T](tags: Iterable[T]) -> list[T]:
    """Materialise ``tags`` and reject duplicates.

    Linear scan rather than set membership so non-hashable tags work;
    O(n^2) is fine since call sites mint a handful at a time.
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

    ``left_tag`` and ``right_tag`` MUST be distinct or
    :class:`TokenTagCollisionError` fires. The parent is consumed
    *before* the duplicate check so an invalid split still surrenders
    the parent — callers see a gone token plus an exception, not a
    deceptively-live parent.
    """
    parent.consume()
    if left_tag == right_tag:
        raise TokenTagCollisionError(duplicate_tag_repr=repr(left_tag))
    return Token(left_tag), Token(right_tag)


def mint_split_n[In, L](parent: Token[In], child_tags: Iterable[L]) -> list[Token[L]]:
    """Consume ``parent`` and mint N orthogonal sub-tokens.

    Child tags must be pairwise distinct. Parent consumption precedes the
    duplicate check (see :func:`mint_split`).
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
