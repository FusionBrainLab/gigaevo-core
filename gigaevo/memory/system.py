"""The one assembled memory subsystem.

Hydra recursively builds the component leaves (reputation, dedup, backend, llm,
retriever, selector, auction, budget) plus the ``_partial_`` admitter/evictor/
provider/tracker under a single ``_target_: gigaevo.memory.MemorySystem`` node
and passes them here as kwargs. This class is the *assembler*: it completes the
partials with the ONE shared reputation, threads the ONE dedup config + the
shared llm into a single backend copy, and exposes ``.provider`` (read side) and
``.tracker`` (write side) — Null variants when a side is disabled. There is no
``${ref:memory.*}`` sibling web: sharing is a Python fact, not a YAML coincidence.

The two enable flags are the user's one knob: ``memory={none,reader,writer,full}``.
"""

from __future__ import annotations

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
        dedup: Any = None,
        backend: Any = None,
        llm: Any = None,
        retriever: Any = None,
        selector: Any = None,
        auction: Any = None,
        budget: Any = None,
        admitter: Any = None,
        evictor: Any = None,
        provider: Any = None,
        tracker: Any = None,
    ) -> None:
        self.reputation = reputation
        self.dedup = dedup

        if backend is not None:
            update: dict[str, Any] = {"llm": llm}
            # NullDeduplicator has no `.config`; only the LLM deduplicator does.
            dedup_config = getattr(dedup, "config", None)
            if dedup_config is not None:
                update["dedup"] = dedup_config
            backend = backend.model_copy(update=update)
        self.backend = backend

        if reader_enabled:
            self.provider = provider(
                backend=backend,
                retriever=retriever,
                selector=selector,
                auctioneer=auction,
                budgeter=budget,
                reputation=reputation,
            )
        else:
            self.provider = NullMemoryProvider()

        if writer_enabled:
            self.tracker = tracker(
                backend=backend,
                llm=llm,
                admitter=admitter(reputation=reputation),
                evictor=evictor(reputation=reputation),
                deduplicator=dedup,
                reputation=reputation,
            )
        else:
            self.tracker = NullPostRunHook()
