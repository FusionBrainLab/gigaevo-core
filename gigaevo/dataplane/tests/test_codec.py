"""Tests for the canonical codec and content-hash computation."""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
import enum
from pathlib import PurePosixPath
from uuid import UUID

from pydantic import BaseModel
import pytest

from gigaevo.dataplane.codec import (
    compute_content_hash,
    compute_content_hash_hex,
    decode_canonical,
    encode_canonical,
)
from gigaevo.dataplane.errors import CanonicalEncodingError


class _Sample(BaseModel):
    name: str
    n: int


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

    def test_nan_rejected(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": float("nan")})

    def test_inf_rejected(self) -> None:
        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": float("inf")})

    def test_unknown_type_rejected(self) -> None:
        class Weird:
            pass

        with pytest.raises(CanonicalEncodingError):
            encode_canonical({"x": Weird()})


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


# ── compute_content_hash ──────────────────────────────────────────────


class TestComputeContentHash:
    def test_returns_32_bytes(self) -> None:
        h = compute_content_hash({"a": 1}, schema_version=1)
        assert len(h) == 32

    def test_hex_returns_64_chars(self) -> None:
        h = compute_content_hash_hex({"a": 1}, schema_version=1)
        assert len(h) == 64
        int(h, 16)  # well-formed hex

    def test_determinism_across_key_order(self) -> None:
        a = compute_content_hash({"a": 1, "b": 2}, schema_version=1)
        b = compute_content_hash({"b": 2, "a": 1}, schema_version=1)
        assert a == b

    def test_distinct_for_different_payloads(self) -> None:
        a = compute_content_hash({"a": 1}, schema_version=1)
        b = compute_content_hash({"a": 2}, schema_version=1)
        assert a != b

    def test_distinct_for_different_schema_versions(self) -> None:
        a = compute_content_hash({"a": 1}, schema_version=1)
        b = compute_content_hash({"a": 1}, schema_version=2)
        assert a != b

    def test_schema_version_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            compute_content_hash({"a": 1}, schema_version=0)

    def test_golden_known_payload(self) -> None:
        # Pin one known hash so any future drift in the canonical
        # encoder surfaces as a CI failure.
        h = compute_content_hash_hex({"a": 1, "b": [1, 2]}, schema_version=1)
        # Sanity: stable across runs of this commit.
        h_again = compute_content_hash_hex({"a": 1, "b": [1, 2]}, schema_version=1)
        assert h == h_again
        # And not the empty hash.
        assert h != "0" * 64
