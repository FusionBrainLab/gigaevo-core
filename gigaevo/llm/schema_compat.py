"""Portable JSON-schema emission for strict structured-output backends.

Pydantic-emitted schemas use constructs some providers reject: Gemini 400s on
$ref/$defs and const (probed 2026-07-02), and `discriminator`/`default` are
pydantic annotations without validation semantics under guided decoding. The
rewrites here produce an equivalent schema in the portable subset; vLLM accepts
both forms, so callers can emit the portable form unconditionally.

Keys inside a `properties` map are field names, never schema keywords — all
transforms leave them untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

UNSUPPORTED_KEYS = frozenset(
    {"$defs", "$ref", "const", "discriminator", "default", "prefixItems"}
)


def inline_refs(schema: dict) -> dict:
    """Resolve internal ``#/$defs/...`` references and drop the ``$defs`` block."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = resolve(defs[ref.rsplit("/", 1)[-1]])
                siblings = {k: resolve(v) for k, v in node.items() if k != "$ref"}
                return {**target, **siblings}
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def drop_annotations(
    schema: dict, keys: tuple[str, ...] = ("default", "discriminator")
) -> dict:
    """Remove annotation keys that carry no validation semantics."""
    return _map_schema_nodes(
        schema, lambda node: {k: v for k, v in node.items() if k not in keys}
    )


def const_to_enum(schema: dict) -> dict:
    """Rewrite ``const: x`` into the equivalent ``enum: [x]``."""

    def rewrite(node: dict) -> dict:
        if "const" in node:
            node = dict(node)
            node["enum"] = [node.pop("const")]
        return node

    return _map_schema_nodes(schema, rewrite)


def portable_json_schema(schema: dict) -> dict:
    """Compose all rewrites; the result validates the same documents."""
    return const_to_enum(drop_annotations(inline_refs(schema)))


def nonportable_keys(schema: dict) -> set[str]:
    """Schema keywords present that strict backends reject; empty = portable.

    ``prefixItems`` is flagged but has no rewrite — avoid it at model level.
    """
    found: set[str] = set()

    def collect(node: dict) -> dict:
        found.update(UNSUPPORTED_KEYS.intersection(node))
        return node

    _map_schema_nodes(schema, collect)
    return found


def _map_schema_nodes(
    node: Any, fn: Callable[[dict], dict], is_properties_map: bool = False
) -> Any:
    """Apply ``fn`` to every schema dict, skipping ``properties`` maps themselves."""
    if isinstance(node, dict):
        out = {
            k: _map_schema_nodes(
                v, fn, is_properties_map=(k == "properties" and not is_properties_map)
            )
            for k, v in node.items()
        }
        return out if is_properties_map else fn(out)
    if isinstance(node, list):
        return [_map_schema_nodes(v, fn) for v in node]
    return node
