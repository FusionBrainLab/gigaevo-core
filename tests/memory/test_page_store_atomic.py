import json

from gigaevo.memory._vendor.GAM_root.gam.schemas import InMemoryPageStore, Page


def make_page(header: str) -> Page:
    return Page(header=header, content=f"content-{header}", meta={"k": header})


def headers(pages: list[Page]) -> list[str]:
    return [p.header for p in pages]


def test_interrupted_save_keeps_previous_pages(tmp_path, monkeypatch):
    store = InMemoryPageStore(dir_path=str(tmp_path))
    store.save([make_page("a"), make_page("b")])

    def boom(*args, **kwargs):
        raise RuntimeError("interrupted mid-serialization")

    monkeypatch.setattr(json, "dump", boom)
    store.save([make_page("a"), make_page("b"), make_page("c")])

    reopened = InMemoryPageStore(dir_path=str(tmp_path))
    assert headers(reopened.load()) == ["a", "b"]
    assert list(tmp_path.glob("pages.*.tmp")) == []


def test_successful_save_is_clean(tmp_path):
    store = InMemoryPageStore(dir_path=str(tmp_path))
    store.save([make_page("a"), make_page("b")])
    store.save([make_page("a"), make_page("b"), make_page("c")])

    reopened = InMemoryPageStore(dir_path=str(tmp_path))
    assert headers(reopened.load()) == ["a", "b", "c"]
    assert list(tmp_path.glob("pages.*.tmp")) == []


def test_add_and_roundtrip_preserved(tmp_path):
    store = InMemoryPageStore(dir_path=str(tmp_path))
    store.add(make_page("a"))
    store.add(make_page("b"))

    reopened = InMemoryPageStore(dir_path=str(tmp_path))
    loaded = reopened.load()
    assert headers(loaded) == ["a", "b"]
    assert loaded[0].content == "content-a"
    assert loaded[0].meta == {"k": "a"}
