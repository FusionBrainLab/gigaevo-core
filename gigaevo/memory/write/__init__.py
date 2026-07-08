"""Write system: mutation diffs → librarian-authored cards in the bank.

Never imports ``read/`` — eviction consumes the ``CardScorer`` /
``CardValueScorer`` Protocols declared in ``eviction.py``; the read layer's
reputation implements them and the integration config wires them together.
"""
