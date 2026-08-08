"""Portable JSON-schema emission for strict structured-output backends.

Pydantic-emitted schemas use constructs some providers reject: Gemini 400s on
$ref/$defs and const (probed 2026-07-02), and `discriminator`/`default` are
pydantic annotations without validation semantics under guided decoding. The
rewrites here produce an equivalent schema in the portable subset; vLLM accepts
both forms, so callers can emit the portable form unconditionally.

Gemini also 400s on `maxItems` when the array's `items` is an `anyOf` union
(probed 2026-07-22; `minItems` is accepted). No rewrite preserves that bound, so
schemas state the cap in the field description instead of emitting it.

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
    """Resolve internal ``#/$defs/...`` references and drop the ``$defs`` block.

    Raises ValueError on recursive definitions: they cannot be inlined.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, stack: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                if name in stack:
                    raise ValueError(f"recursive $ref '{name}' cannot be inlined")
                target = resolve(defs[name], stack | {name})
                siblings = {
                    k: resolve(v, stack) for k, v in node.items() if k != "$ref"
                }
                return {**target, **siblings}
            return {k: resolve(v, stack) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, stack) for v in node]
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


def strict_json_schema(schema: dict) -> dict:
    """Rewrite into the OpenAI strict-mode subset (probed 2026-08-07).

    Strict structured output rejects any object node without
    ``additionalProperties: false`` and a ``required`` list naming every key in
    ``properties``. Optionality therefore cannot be expressed by omission; the
    convention is nullability, so originally-optional properties gain a
    ``null`` branch (appended to an existing ``anyOf``, else wrapped in one).
    ``strip_strict_nulls`` is the answer-side inverse.
    """

    def strictify(node: dict) -> dict:
        if "properties" not in node and node.get("type") != "object":
            return node
        node = dict(node)
        props = node.get("properties", {})
        required = set(node.get("required", []))
        node["properties"] = {
            key: prop if key in required else _nullable_form(prop)
            for key, prop in props.items()
        }
        node["required"] = list(props)
        node["additionalProperties"] = False
        return node

    return _map_schema_nodes(schema, strictify)


def strip_strict_nulls(payload: Any, schema: dict) -> Any:
    """Drop the nulls ``strict_json_schema`` invited, so defaults apply.

    ``schema`` is the ORIGINAL (pre-strict) schema: a null is dropped only for
    a key that was optional there and not already nullable — exactly the keys
    whose null branch the wire rewrite added. Nulls for required or genuinely
    nullable keys pass through (the latter mean ``None``, the former must fail
    validation loudly). Recurses via ``properties``/``items``; payloads under
    union (``anyOf``) or unknown keys pass through untouched.
    """
    if isinstance(payload, dict) and isinstance(schema, dict):
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        out = {}
        for key, value in payload.items():
            prop = props.get(key)
            if (
                value is None
                and key not in required
                and isinstance(prop, dict)
                and not _nullable(prop)
            ):
                continue
            out[key] = (
                strip_strict_nulls(value, prop) if isinstance(prop, dict) else value
            )
        return out
    if isinstance(payload, list) and isinstance(schema, dict):
        items = schema.get("items")
        if isinstance(items, dict):
            return [strip_strict_nulls(v, items) for v in payload]
    return payload


def _nullable_form(prop: Any) -> Any:
    if not isinstance(prop, dict) or _nullable(prop):
        return prop
    if isinstance(prop.get("anyOf"), list):
        return {**prop, "anyOf": [*prop["anyOf"], {"type": "null"}]}
    return {"anyOf": [prop, {"type": "null"}]}


def _nullable(prop: dict) -> bool:
    t = prop.get("type")
    if t == "null" or (isinstance(t, list) and "null" in t):
        return True
    branches = prop.get("anyOf")
    return isinstance(branches, list) and any(
        isinstance(b, dict) and b.get("type") == "null" for b in branches
    )


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
