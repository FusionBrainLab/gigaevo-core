from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenStrictModel(BaseModel):
    """Base class for every config schema.

    ``extra='forbid'`` turns a typo into a load-time ``ValidationError``;
    ``frozen=True`` makes the resolved config tree immutable after the
    CLI hands it to ``run_experiment``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
