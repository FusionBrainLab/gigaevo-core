"""The one assembled memory subsystem.

Hydra recursively builds the component leaves (reputation, backend, llm,
retriever, selector, auction, budget) plus the ``_partial_`` evictor/
provider/tracker under a single ``_target_: gigaevo.memory.system.MemorySystem`` node
and passes them here as kwargs. This class is the *assembler*: it completes the
partials with the ONE shared reputation, binds the shared llm into the backend
partial, and exposes ``.provider`` (read side) and ``.tracker`` (write
side) — Null variants when a side is disabled. There is no ``${ref:memory.*}``
sibling web: sharing is a Python fact, not a YAML coincidence.

The two enable flags are the user's one knob: ``memory={none,reader,writer,full}``.
"""

from __future__ import annotations

import functools
from typing import Any

from gigaevo.evolution.engine.hooks import NullPostRunHook
from gigaevo.memory.provider import NullMemoryProvider


class MemorySystem:
    def __init__(
        self,
        *,
        reader_enabled: bool = False,
        writer_enabled: bool = False,
        reputation: Any = None,
        backend: Any = None,
        llm: Any = None,
        retriever: Any = None,
        selector: Any = None,
        auction: Any = None,
        budget: Any = None,
        evictor: Any = None,
        excluder: Any = None,
        provider: Any = None,
        tracker: Any = None,
    ) -> None:
        self.reputation = reputation

        # backend is the Hydra _partial_ over build_local_backend; bind the one
        # shared memory llm here so read and write build the same backed bank.
        if backend is not None:
            backend = functools.partial(backend, llm_service=llm)
        self.backend = backend

        if reader_enabled:
            self.provider = provider(
                backend=backend,
                retriever=retriever,
                selector=selector,
                auctioneer=auction,
                budgeter=budget,
                reputation=reputation,
                excluder=excluder,
            )
        else:
            self.provider = NullMemoryProvider()

        if writer_enabled:
            self.tracker = tracker(
                backend=backend,
                llm=llm,
                evictor=evictor(reputation=reputation),
                reputation=reputation,
            )
        else:
            self.tracker = NullPostRunHook()
