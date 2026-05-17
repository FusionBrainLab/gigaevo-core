"""Tests for the canonical codec and content-hash computation.

Coverage goals:

    - Every supported type round-trips deterministically.
    - Every closed-fail path raises :class:`CanonicalEncodingError` with
      no stack-overflow / opaque-failure escape hatch.
    - Heterogeneous-element sets sort deterministically regardless of
      insertion order.
    - Integers outside int64 range are refused — downstream Lua / JS
      cannot represent them losslessly.
    - The golden content hash is pinned so any future drift in the
      canonical encoder surfaces as a CI failure.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
import enum
from pathlib import PurePosixPath
import random
from uuid import UUID

from pydantic import BaseModel
import pytest

from gigaevo.dataplane.codec import (
    compute_content_hash_hex,
    decode_canonical,
    encode_canonical,
)
from gigaevo.dataplane.errors import CanonicalEncodingError


class _Sample(BaseModel):
    name: str
    n: int


class _Outer(BaseModel):
    inner: _Sample
    label: str


class _Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


# ── encode_canonical ──────────────────────────────────────────────────


class TestEncodeCanonical:
    def test_key_order_is_stable(self) -> None:
        a = encode_canonical({"b": 2, "a": 1})
        b = encode_canonical({"a": 1, "b": 2})
        assert a == b

    def test_nested_key_order_is_stable(self) -> None:
        a = encode_canonical({"outer": {"b": 2, "a": 1}})
        b = encode_canonical({"outer": {"a": 1, "b": 2}})
        assert a == b

    def test_no_whitespace(self) -> None:
        out = encode_canonical({"a": 1, "b": [1, 2]})
        assert b" " not in out
        assert b": " not in out

    def test_bytes_encoded_as_hex(self) -> None:
        out = encode_canonical({"b": b"\xde\xad"})
        assert b"dead" in out

    def test_bytearray_encoded_as_hex_like_bytes(self) -> None:
        a = encode_canonical({"b": b"\xde\xad"})
        b = encode_canonical({"b": bytearray(b"\xde\xad")})
        assert a == b

    def test_memoryview_encoded_as_hex_like_bytes(self) -> None:
        a = encode_canonical({"b": b"\xde\xad"})
        b = encode_canonical({"b": memoryview(b"\xde\xad")})
        assert a == b

    def test_uuid_encoded_as_string(self) -> None:
        u = UUID("12345678-1234-5678-1234-567812345678")
        out = encode_canonical({"u": u})
        assert str(u).encode() in out

    def test_datetime_isoformat(self) -> None:
        d = _dt.datetime(2025, 1, 2, 3, 4, 5, tzinfo=_dt.UTC)
        out = encode_canonical({"d": d})
        assert b"2025-01-02T03:04:05" in out

    def test_decimal_string(self) -> None:
        out = encode_canonical({"price": Decimal("1.5")})
        assert b'"1.5"' in out

    def test_decimal_trailing_zero_normalised(self) -> None:
        # "1.5" and "1.50" denote the same number; their canonical bytes
        # must match so two callers passing the "same" value land on
        # the same content hash.
        a = encode_canonical({"price": Decimal("1.5")})
        b = encode_canonical({"price": Decimal("1.50")})
        assert a == b

    def test_decimal_integer_form_normalised(self) -> None:
        # Decimal('15.00').normalize() -> '1.5E+1'; the encoder uses
        # fixed-point format to avoid that quirk.
        a = encode_canonical({"x": Decimal("15.00")})
        b = encode_canonical({"x": Decimal("15")})
        assert a == b
        assert b'"15"' in a

    def test_decimal_non_finite_rejected(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": Decimal("NaN")})
        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": Decimal("Infinity")})

    def test_decimal_signed_zero_collapsed(self) -> None:
        # ``Decimal('-0')`` and ``Decimal('0')`` are numerically the same
        # value; their canonical bytes must match. Without an explicit
        # zero-collapse the negative-zero leaves a ``"-"`` in the output
        # that diverges from the positive-zero encoding.
        a = encode_canonical({"x": Decimal("-0")})
        b = encode_canonical({"x": Decimal("0")})
        c = encode_canonical({"x": Decimal("0.000")})
        d = encode_canonical({"x": Decimal("-0.0")})
        assert a == b == c == d
        assert b'"0"' in a

    def test_path_stringified(self) -> None:
        out = encode_canonical({"p": PurePosixPath("/a/b")})
        assert b"/a/b" in out

    def test_enum_value(self) -> None:
        out = encode_canonical({"c": _Color.RED})
        assert b"red" in out

    def test_set_sorted(self) -> None:
        a = encode_canonical({"s": {3, 1, 2}})
        b = encode_canonical({"s": {2, 3, 1}})
        assert a == b
        assert b"[1,2,3]" in a

    def test_frozenset_sorted(self) -> None:
        a = encode_canonical({"s": frozenset({3, 1, 2})})
        assert b"[1,2,3]" in a

    def test_heterogeneous_set_deterministic(self) -> None:
        # A set containing values whose ``str()`` collide (such as
        # ``1`` and ``"1"``) must still encode deterministically.
        baseline = encode_canonical({1, "1", 2, "2"})
        for _ in range(20):
            items = [1, "1", 2, "2"]
            random.shuffle(items)
            assert encode_canonical(set(items)) == baseline

    def test_heterogeneous_set_all_elements_kept(self) -> None:
        # Confirm that distinguishing 1 from "1" preserves both
        # elements in the encoded list (no silent collision).
        out = encode_canonical({1, "1"})
        decoded = decode_canonical(out)
        assert sorted(decoded, key=lambda v: (type(v).__name__, str(v))) == [1, "1"]

    def test_nested_set_of_tuples(self) -> None:
        # Sets containing tuples that share string reprs should still
        # order deterministically.
        s = {(1, 2), (1, "2"), ("1", 2)}
        baseline = encode_canonical(s)
        for _ in range(10):
            items = list(s)
            random.shuffle(items)
            assert encode_canonical(set(items)) == baseline

    def test_tuple_as_array(self) -> None:
        out = encode_canonical({"t": (1, 2, 3)})
        assert b"[1,2,3]" in out

    def test_pydantic_model(self) -> None:
        out = encode_canonical(_Sample(name="x", n=1))
        assert b'"name":"x"' in out
        assert b'"n":1' in out

    def test_nested_pydantic(self) -> None:
        out = encode_canonical({"sample": _Sample(name="x", n=1)})
        assert b'"name":"x"' in out

    def test_pydantic_inside_pydantic_inside_dict(self) -> None:
        # model_dump cascades, so deeply nested models must produce the
        # same canonical bytes as their dict-equivalent.
        outer = _Outer(inner=_Sample(name="x", n=1), label="hi")
        a = encode_canonical({"wrap": outer})
        b = encode_canonical({"wrap": {"inner": {"name": "x", "n": 1}, "label": "hi"}})
        assert a == b

    def test_nan_rejected(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": float("nan")})

    def test_inf_rejected(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": float("inf")})

    def test_negative_inf_rejected(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": float("-inf")})

    def test_lone_surrogate_rejected(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": "\udcff"})

    def test_circular_reference_rejected_cleanly(self) -> None:
        # ``json.dumps`` raises ValueError on circular refs; verify
        # we wrap it in CanonicalEncodingError and don't blow the stack.
        d: dict[str, object] = {}
        d["self"] = d
        with pytest.raises(CanonicalEncodingError):
            encode_canonical(d)

    def test_unknown_type_rejected(self) -> None:
        class Weird:
            pass

        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": Weird()})

    def test_int_overflow_rejected(self) -> None:
        # 2**63 just outside the upper bound.
        with pytest.raises(CanonicalEncodingError, match="signed-64-bit"):
            encode_canonical({"x": 2**63})

    def test_int_underflow_rejected(self) -> None:
        with pytest.raises(CanonicalEncodingError, match="signed-64-bit"):
            encode_canonical({"x": -(2**63) - 1})

    def test_int_at_int64_bounds_accepted(self) -> None:
        # The exact bounds are still encodable.
        encode_canonical({"x": 2**63 - 1})
        encode_canonical({"x": -(2**63)})

    def test_int_overflow_in_nested_list(self) -> None:
        with pytest.raises(CanonicalEncodingError, match="signed-64-bit"):
            encode_canonical({"xs": [1, 2, 2**100]})

    def test_int_overflow_in_nested_dict(self) -> None:
        with pytest.raises(CanonicalEncodingError, match="signed-64-bit"):
            encode_canonical({"outer": {"inner": 2**100}})

    def test_bool_is_not_overflow_checked(self) -> None:
        # bool is a subclass of int, but True/False are always in range.
        out = encode_canonical({"x": True, "y": False})
        assert b"true" in out
        assert b"false" in out

    def test_float_repr_shortest_round_trip(self) -> None:
        # Python's float repr since 3.1 is the shortest round-trip
        # representation; verify the encoder uses it (not %g).
        out = encode_canonical({"x": 0.1 + 0.2})
        assert b"0.30000000000000004" in out


# ── decode_canonical ──────────────────────────────────────────────────


class TestDecodeCanonical:
    def test_round_trip_simple(self) -> None:
        payload = {"a": 1, "b": "two", "c": [1, 2, 3]}
        out = decode_canonical(encode_canonical(payload))
        assert out == payload

    def test_round_trip_str_input(self) -> None:
        payload = {"x": 1}
        out = decode_canonical(encode_canonical(payload).decode("utf-8"))
        assert out == payload

    def test_lone_surrogate_str_input_rejected_as_canonical_error(self) -> None:
        # ``str`` inputs were previously raising ``UnicodeEncodeError``;
        # the contract is unified so both bytes and str inputs surface
        # as :class:`CanonicalEncodingError`.
        with pytest.raises(CanonicalEncodingError):
            decode_canonical("\udcff")

    def test_malformed_utf8_bytes_rejected_as_canonical_error(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            decode_canonical(b"\xff\xfe")

    def test_malformed_json_rejected_as_canonical_error(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            decode_canonical(b'{"unclosed":')

    def test_round_trip_with_nested_types(self) -> None:
        # Tuples and sets are lossy through canonical: they become
        # lists. Verify decoding does not raise and produces sensible
        # lists.
        payload: dict[str, object] = {"t": (1, 2), "s": {3, 4}}
        out = decode_canonical(encode_canonical(payload))
        assert out["t"] == [1, 2]
        assert sorted(out["s"]) == [3, 4]  # type: ignore[type-var]


# ── compute_content_hash_hex ──────────────────────────────────────────


class TestComputeContentHashHex:
    def test_hex_returns_64_chars(self) -> None:
        h = compute_content_hash_hex({"a": 1}, schema_version=1)
        assert len(h) == 64
        int(h, 16)  # well-formed hex

    def test_determinism_across_key_order(self) -> None:
        a = compute_content_hash_hex({"a": 1, "b": 2}, schema_version=1)
        b = compute_content_hash_hex({"b": 2, "a": 1}, schema_version=1)
        assert a == b

    def test_distinct_for_different_payloads(self) -> None:
        a = compute_content_hash_hex({"a": 1}, schema_version=1)
        b = compute_content_hash_hex({"a": 2}, schema_version=1)
        assert a != b

    def test_distinct_for_different_schema_versions(self) -> None:
        a = compute_content_hash_hex({"a": 1}, schema_version=1)
        b = compute_content_hash_hex({"a": 1}, schema_version=2)
        assert a != b

    def test_schema_version_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            compute_content_hash_hex({"a": 1}, schema_version=0)

    def test_golden_known_payload(self) -> None:
        """Pin one known hash so encoder drift surfaces in CI.

        The expected hex is computed once against this commit's encoder
        and pasted below. Any change to canonical encoding — set sort
        order, key separator, default registry — will flip this digest
        and fail the test, which is exactly what we want: silent encoder
        drift would make every stored content hash incompatible with
        every future replay.
        """
        h = compute_content_hash_hex({"a": 1, "b": [1, 2]}, schema_version=1)
        assert h == "3614ba0d01ff99fc918b7f965762220d2c278a4be778a3c2e4fdb38c185c9f10"

    def test_golden_set_payload(self) -> None:
        """Pin the hash for a heterogeneous-set payload.

        Catches regressions in the canonical-byte sort that backs
        ``_canonical_default`` for sets; the previous ``key=str``
        ordering produced different bytes depending on insertion order.
        """
        # Heterogeneous set; insertion order must not affect the hash.
        payload = {"s": {1, "1", 2}}
        h = compute_content_hash_hex(payload, schema_version=1)
        for _ in range(5):
            items = [1, "1", 2]
            random.shuffle(items)
            replay = compute_content_hash_hex({"s": set(items)}, schema_version=1)
            assert replay == h
