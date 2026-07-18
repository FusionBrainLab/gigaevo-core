"""Write system: mutation diffs → librarian-authored cards in the bank.

Never imports ``read/``. The write-side eviction surface — the ``Evictor``
Protocol and ``NullEvictor`` in ``eviction.py`` — is self-contained; the
admission gate's periodic sweep is its only consumer. ``memory=v2`` wires its
causal retirement implementation through that seam.
"""
