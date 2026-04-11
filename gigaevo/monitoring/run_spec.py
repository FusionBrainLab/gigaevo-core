from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunSpec:
    """Parsed run specification: prefix@db[:label].

    Immutable. Used as the canonical representation of a run reference
    throughout the monitoring package.
    """

    prefix: str
    db: int
    label: str

    @property
    def display_name(self) -> str:
        """Short display name for the run (the label)."""
        return self.label

    @classmethod
    def parse(cls, raw: str) -> RunSpec:
        """Parse 'prefix@db[:label]' into a RunSpec.

        Handles:
        - Quote stripping (single and double quotes)
        - Whitespace trimming
        - Prefixes containing '/' (normal for GigaEvo)
        - Optional label after the first ':' following the db number
        - Uses rfind("@") to handle any future '@' in prefixes

        Raises:
            ValueError: If the format is invalid (no '@', non-numeric db,
                        empty prefix, negative db).
        """
        s = raw.strip().strip('"').strip("'").strip()
        if not s:
            raise ValueError(f"Empty run spec: {raw!r}")

        at_idx = s.rfind("@")
        if at_idx == -1:
            raise ValueError(
                f"Run spec must contain '@': got {raw!r}. "
                f"Expected format: prefix@db[:label]"
            )

        prefix = s[:at_idx]
        rest = s[at_idx + 1 :]

        if not prefix:
            raise ValueError(
                f"Empty prefix in run spec: {raw!r}. Expected format: prefix@db[:label]"
            )

        # Split rest into db and optional label
        if ":" in rest:
            db_str, label = rest.split(":", 1)
        else:
            db_str = rest
            label = None

        # Validate db
        try:
            db = int(db_str)
        except ValueError:
            raise ValueError(
                f"Non-numeric db in run spec: {db_str!r} from {raw!r}. "
                f"Expected format: prefix@db[:label]"
            )

        if db < 0:
            raise ValueError(
                f"Negative db in run spec: {db} from {raw!r}. DB must be >= 0"
            )

        if label is None:
            label = f"{prefix}@{db}"

        return cls(prefix=prefix, db=db, label=label)
