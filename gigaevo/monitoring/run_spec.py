from __future__ import annotations

from dataclasses import dataclass

from gigaevo.experiment.manifest import RunRole


@dataclass(frozen=True)
class RunSpec:
    """Parsed run specification: prefix@db[:label].

    Immutable. Used as the canonical representation of a run reference
    throughout the monitoring package.

    ``role`` identifies population role. None for non-adversarial runs.
    """

    prefix: str
    db: int
    label: str
    role: RunRole | None = None

    @property
    def display_name(self) -> str:
        """Short display name for the run (the label)."""
        return self.label

    @property
    def needs_prefix(self) -> bool:
        """True when prefix was not provided and must be auto-discovered."""
        return self.prefix == ""

    @classmethod
    def parse(cls, raw: str) -> RunSpec:
        """Parse 'prefix@db[:label]' or just 'db' into a RunSpec.

        When only a bare db number is given (e.g. '2'), the prefix is left
        empty and must be resolved later via auto-discovery from Redis.

        Handles:
        - Quote stripping (single and double quotes)
        - Whitespace trimming
        - Prefixes containing '/' (normal for GigaEvo)
        - Optional label after the first ':' following the db number
        - Uses rfind("@") to handle any future '@' in prefixes

        Raises:
            ValueError: If the format is invalid (non-numeric db, negative db).
        """
        s = raw.strip().strip('"').strip("'").strip()
        if not s:
            raise ValueError(f"Empty run spec: {raw!r}")

        at_idx = s.rfind("@")
        if at_idx == -1:
            # Bare db number: "2" or "2:label"
            if ":" in s:
                db_str, label = s.split(":", 1)
            else:
                db_str = s
                label = None
            try:
                db = int(db_str)
            except ValueError:
                raise ValueError(
                    f"Run spec must contain '@' or be a bare db number: got {raw!r}. "
                    f"Expected format: prefix@db[:label] or just db"
                )
            if db < 0:
                raise ValueError(
                    f"Negative db in run spec: {db} from {raw!r}. DB must be >= 0"
                )
            return cls(prefix="", db=db, label=label or f"@{db}")

        prefix = s[:at_idx]
        rest = s[at_idx + 1 :]

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

        if not prefix:
            return cls(prefix="", db=db, label=label or f"@{db}")

        if label is None:
            label = f"{prefix}@{db}"

        return cls(prefix=prefix, db=db, label=label)
