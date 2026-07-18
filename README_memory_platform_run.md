# Remote `memory_platform` backend — removed

> **This backend no longer exists.** The remote `gigaevo-memory`
> (Postgres + pgvector) backend and its `gigaevo.memory_platform` package were
> removed in the one-knob memory-config collapse
> (`memory={none,v2}`). The Hydra overrides this guide used to
> describe — `memory=api`, `ideas_tracker=default`, `namespace=…` — no longer
> exist; following the old instructions yields an immediate Hydra error.
>
> For the live, local memory flow (the only mode used by current experiments)
> see [`README_memory.md`](README_memory.md) and
> [`gigaevo/memory/README.md`](gigaevo/memory/README.md).
