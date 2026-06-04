"""Content-addressable stage-output store on RedisProgramStorage.

Stage outputs are written once under a content-derived cache id and read back by
that id, so a child program can reference (rather than copy) the parent stage
outputs that produced it. Writes are idempotent — the same id always maps to the
same bytes, so re-writing is a harmless no-op (dedup across sibling children).
"""

from __future__ import annotations


class TestStageOutputStore:
    async def test_put_then_get_roundtrips(self, fakeredis_storage):
        await fakeredis_storage.put_stage_output("cid-1", "blob-A")
        assert await fakeredis_storage.get_stage_output("cid-1") == "blob-A"

    async def test_get_unknown_id_returns_none(self, fakeredis_storage):
        assert await fakeredis_storage.get_stage_output("never-written") is None

    async def test_put_is_idempotent_first_write_wins(self, fakeredis_storage):
        """A content-derived id always maps to identical bytes; re-putting must
        neither raise nor clobber the stored value."""
        await fakeredis_storage.put_stage_output("cid-2", "blob-A")
        await fakeredis_storage.put_stage_output("cid-2", "blob-A")
        assert await fakeredis_storage.get_stage_output("cid-2") == "blob-A"

    async def test_distinct_ids_are_independent(self, fakeredis_storage):
        await fakeredis_storage.put_stage_output("cid-a", "A")
        await fakeredis_storage.put_stage_output("cid-b", "B")
        assert await fakeredis_storage.get_stage_output("cid-a") == "A"
        assert await fakeredis_storage.get_stage_output("cid-b") == "B"

    async def test_key_is_namespaced_under_prefix(self, fakeredis_storage):
        key = fakeredis_storage._keys.stage_output("cid-x")
        assert key == "test:stage_output:cid-x"
