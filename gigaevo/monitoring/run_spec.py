from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gigaevo.experiment.manifest import RunRole


@dataclass(frozen=True)
class RunSpec:
    """Parsed run specification: prefix@db[:label] or a disk storage path.

    Immutable. Used as the canonical representation of a run reference
    throughout the monitoring package.

    ``role`` identifies population role. None for non-adversarial runs.

    Disk-backed specs carry ``path`` (the storage root directory) and a
    sentinel ``db`` of -1 that must never reach a Redis connection —
    check ``is_disk`` first.
    """

    prefix: str
    db: int
    label: str
    role: RunRole | None = None
    path: str | None = None

    @property
    def display_name(self) -> str:
        """Short display name for the run (the label)."""
        return self.label

    @property
    def is_disk(self) -> bool:
        """True when this spec points at on-disk program storage."""
        return self.path is not None

    @property
    def needs_prefix(self) -> bool:
        """True when prefix must be auto-discovered from Redis."""
        return self.prefix == "" and self.path is None

    @classmethod
    def parse(cls, raw: str) -> RunSpec:
        """Parse 'prefix@db[:label]', just 'db', or a disk path into a RunSpec.

        When only a bare db number is given (e.g. '2'), the prefix is left
        empty and must be resolved later via auto-discovery from Redis.

        Specs starting with '/', './', '../', or '~' are disk storage
        paths ('/path/to/storage[:label]'). A slash-containing spec without
        an '@' is also a disk path, so common relative paths such as
        'outputs/run/storage' work without a leading './'. The prefix is
        resolved later by directory discovery.

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

        is_disk_path = s.startswith(("/", "./", "../", "~")) or (
            "/" in s and "@" not in s
        )
        if is_disk_path:
            path, label = s, None
            if ":" in s:
                cand_path, cand_label = s.rsplit(":", 1)
                if "/" not in cand_label:
                    path, label = cand_path, cand_label
            return cls(prefix="", db=-1, label=label or PurePath(path).name, path=path)

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
                    "Expected prefix@db[:label], a bare db number, or a disk path"
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

        if not label:
            label = f"{prefix}@{db}"

        return cls(prefix=prefix, db=db, label=label)
