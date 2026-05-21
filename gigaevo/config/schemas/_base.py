from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class FrozenStrictModel(BaseModel):
    """Base class for every config schema.

    ``extra='forbid'`` turns a typo into a load-time ``ValidationError``;
    ``frozen=True`` makes the resolved config tree immutable after the
    CLI hands it to ``run_experiment``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def reject_empty_or_cwd_path(field_name: str, value: Path | None) -> Path | None:
    """Reject the two ``Path`` values that are almost certainly a bug:
    the empty string (Pydantic coerces ``""`` to ``Path(".")``) and the
    bare current-working-directory placeholder.

    Field-level validators across the schema package share this exact
    test; centralising it keeps the rejection message uniform across
    ``log_dir``, ``prompts_dir``, ``output_dir``, ``fallback_prompts_dir``
    and ``problem_dir``.
    """
    if value is None:
        return value
    if str(value) in ("", "."):
        raise ValueError(
            f"{field_name}: path must be real and non-empty; "
            f"got {value!r} which resolves to the current working directory"
        )
    return value
