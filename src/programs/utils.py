from __future__ import annotations

import base64
import pickle
import textwrap
from typing import Any


def pickle_b64_serialize(value: Any) -> str:
    return base64.b64encode(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("utf-8")


def pickle_b64_deserialize(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("utf-8")))


def dedent_code(code: str) -> str:
    """Remove leading indentation and trailing whitespace from user code."""
    return textwrap.dedent(code).strip()
