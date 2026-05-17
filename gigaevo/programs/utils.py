from __future__ import annotations

import base64
import textwrap
from typing import Any

import cloudpickle


def pickle_b64_serialize(value: Any) -> str:
    return base64.b64encode(cloudpickle.dumps(value)).decode("utf-8")


def pickle_b64_deserialize(value: str) -> Any:
    # Deserializes a b64-encoded cloudpickle payload that originates from
    # the program-content round-trip in ``gigaevo.core_types`` (the same
    # process writes and reads it). The migration target is a typed wire
    # format so that this helper no longer needs the pickle escape hatch.
    return cloudpickle.loads(base64.b64decode(value.encode("utf-8")))  # noqa: TID251


def dedent_code(code: str) -> str:
    """Remove leading indentation and trailing whitespace from user code."""
    return textwrap.dedent(code).strip()
