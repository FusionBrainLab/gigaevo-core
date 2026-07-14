"""Off-policy evaluation over the memory event ledger.

``reconcile`` is the pure, read-only DR-AIPW probe-ITT estimator (also a CLI:
``python -m gigaevo.memory.ope.reconcile <run>``). ``reporter`` wires that
estimator to run and persist automatically on every memory run.

The package ``__init__`` deliberately imports nothing: ``reconcile`` stays a
stdlib-only module, so the CLI and offline analysis do not pay the reporter's
event/monitoring import chain.
"""

from __future__ import annotations
