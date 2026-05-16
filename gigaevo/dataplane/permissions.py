"""Move-only ``Token[Tag]`` linear permission tokens.

A token's presence in a function signature is the witness that the
caller is the sole writer of the subspace tagged by ``Tag``. Tokens are
*linear*: each one can be consumed exactly once, and they cannot be
duplicated.

Python's type system cannot enforce linearity at compile time, so this
module recovers most of the safety at runtime:

    - ``__copy__`` / ``__deepcopy__`` raise — tokens cannot be cloned
      via the standard ``copy`` machinery;
    - ``__reduce__`` and ``__reduce_ex__`` raise — pickle cannot
      silently produce a duplicate;
    - a ``_consumed`` flag on each instance flips on ``consume()``; a
      second call raises :class:`TokenAlreadyConsumed`;
    - factories ``mint_root`` / ``mint_split`` / ``mint_split_n`` /
      ``mint_combine`` are the only legitimate paths to mint a token.

A custom ruff rule (see ``lints.toml``) flags ``t.consume()`` followed
by any further use of ``t`` so most linear-flow violations are caught at
lint time too.
"""

from __future__ import annotations

from typing import Any, SupportsIndex

from .errors import TokenAlreadyConsumed, TokenNotPickleable


class Token[Tag]:
    """Linear, move-only permission witness for the subspace ``Tag``.

    Callers SHOULD mint tokens via :func:`mint_root` etc. rather than
    constructing directly — those factories exist as a single grep
    target.
    """

    __slots__ = ("_tag", "_consumed")

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


def mint_split[In, L, R](
    parent: Token[In],
    left_tag: L,
    right_tag: R,
) -> tuple[Token[L], Token[R]]:
    """Consume ``parent`` and mint two orthogonal sub-tokens.

    The parent is consumed so callers cannot accidentally use it after
    the split. Discipline on the caller: ``left_tag`` and ``right_tag``
    must denote disjoint subspaces of the parent's space — the type
    system has no way to verify the disjointness in Python.
    """
    parent.consume()
    return Token(left_tag), Token(right_tag)


def mint_split_n[In, L](parent: Token[In], child_tags: list[L]) -> list[Token[L]]:
    """Consume ``parent`` and mint N orthogonal sub-tokens.

    Useful when fanning a root permission across N workers or N actor
    instances. Same disjointness discipline as :func:`mint_split`.
    """
    parent.consume()
    return [Token(t) for t in child_tags]


def mint_combine[A, B, Tag](
    left: Token[A],
    right: Token[B],
    combined_tag: Tag,
) -> Token[Tag]:
    """Consume two orthogonal sub-tokens and mint one combined token.

    Inverse of :func:`mint_split`. The combined tag is caller-provided;
    it should denote the union of the two sub-spaces.
    """
    left.consume()
    right.consume()
    return Token(combined_tag)


__all__ = [
    "Token",
    "mint_combine",
    "mint_root",
    "mint_split",
    "mint_split_n",
]
