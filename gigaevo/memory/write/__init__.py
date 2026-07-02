"""Write system: mutation diffs → librarian-authored cards in the bank.

Never imports ``read/`` — eviction consumes the :class:`CardScorer` Protocol
declared in ``eviction.py``; the read layer's reputation implements it and the
integration config wires them together.
"""
