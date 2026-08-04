"""Typed exception hierarchy for the AAAI-27 experiment protocol scaffold."""


class ProtocolError(Exception):
    """Base class for all protocol scaffold errors."""


class ManifestError(ProtocolError):
    """Manifest missing, malformed, unfrozen, or still containing placeholders."""


class GitError(ProtocolError):
    """Git state could not be read."""


class DirtyGitTreeError(GitError):
    """Git working tree has uncommitted changes."""


class MissingArtifactError(ProtocolError):
    """A file declared in the manifest is missing on disk."""


class LockError(ProtocolError):
    """protocol_lock.json missing or malformed."""


class LockMismatchError(LockError):
    """Recorded lock state disagrees with the current repository state."""


class HeldoutAccessError(ProtocolError):
    """Held-out data access attempted without ALLOW_HELDOUT=1."""
