from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from gigaevo.config.schemas._base import FrozenStrictModel, reject_empty_or_cwd_path

if TYPE_CHECKING:
    from gigaevo.problems.context import ProblemContext


class ProblemConfig(FrozenStrictModel):
    """Pointer to a problem directory plus optional metric overrides.

    The experiment-root layer builds the runtime ``ProblemContext`` via
    :meth:`build`, which materialises the ``MetricsContext`` from the
    directory's ``metrics.yaml``.

    ``primary_metric`` and ``higher_is_better`` are optional overrides
    for the values that ``MetricsContext`` derives from ``metrics.yaml``;
    leaving them as ``None`` defers to the on-disk declaration. Setting
    them explicitly is useful for experiments that want to optimise a
    secondary metric without editing the shared ``metrics.yaml``.
    """

    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    problem_dir: Path
    primary_metric: str | None = Field(default=None, min_length=1)
    higher_is_better: bool | None = None

    @field_validator("problem_dir")
    @classmethod
    def _problem_dir_not_empty(cls, value: Path) -> Path:
        # ``problem_dir`` is non-Optional so the helper's None branch is
        # never reached; the cast keeps the type system honest.
        return reject_empty_or_cwd_path("problem_dir", value)  # type: ignore[return-value]

    def build(self) -> ProblemContext:
        from gigaevo.problems.context import ProblemContext

        return ProblemContext(self.problem_dir)
