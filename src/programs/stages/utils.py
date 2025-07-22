"""Utility functions for MetaEvolve stages."""

import base64
import pickle
from typing import Any, Dict


def pickle_b64_serialize(value: Any) -> str:
    return base64.b64encode(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("utf-8")


def pickle_b64_deserialize(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("utf-8")))


def pretty_print_error(error: Dict[str, Any]) -> str:
    """Pretty print error information with comprehensive details."""
    if not isinstance(error, dict):
        return f"Error: {error}"

    msg = error.get("error_message", "Unknown error")
    stderr = error.get("stderr", "")
    stdout = error.get("stdout", "")
    exit_code = error.get("exit_code", "unknown")

    parts = [f"Error: {msg}"]

    if exit_code != "unknown":
        parts.append(f"Exit Code: {exit_code}")

    if stdout and stdout.strip():
        parts.append(f"Stdout:\n{stdout}")

    if stderr and stderr.strip():
        parts.append(f"Stderr:\n{stderr}")

    return "\n".join(parts)
