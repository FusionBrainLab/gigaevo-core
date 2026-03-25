import os, json, subprocess, shutil, time
from typing import Dict, Any, List

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    LuceneSearcher = None  # type: ignore

from GAM_root.gam.retriever.base import AbsRetriever
from GAM_root.gam.schemas import InMemoryPageStore, Hit, Page


def _safe_rmtree(path: str, max_retries: int = 3, delay: float = 0.5) -> None:
    """
    安全地删除目录树，带重试机制
    
    Args:
        path: 要删除的目录路径
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    """
    if not os.path.exists(path):
        return
    
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            # 确保目录真的被删除了
            if not os.path.exists(path):
                return
            time.sleep(delay)
        except OSError as e:
            if attempt == max_retries - 1:
                # 最后一次尝试仍然失败，强制删除
                try:
                    # 尝试更激进的删除方式
                    import subprocess
                    subprocess.run(['rm', '-rf', path], check=False, capture_output=True)
                    if not os.path.exists(path):
                        return
                except Exception:
                    pass
                raise OSError(f"无法删除目录 {path}: {e}")
            time.sleep(delay)


class BM25Retriever(AbsRetriever):
    """
        关键词检索器 (BM25 / Lucene)
        config 需要:
        {
            "index_dir": "xxx",   # 用来放 index/ 和 pages/
            "threads": 4
        }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if LuceneSearcher is None:
            raise ImportError("BM25Retriever requires pyserini to be installed")
        self.index_dir = self.config["index_dir"]
        self.searcher: LuceneSearcher | None = None
        self.pages: List[Page] = []

    def _pages_dir(self):
        return os.path.join(self.index_dir, "pages")

    def _lucene_dir(self):
        return os.path.join(self.index_dir, "index")

    def _docs_dir(self):
        return os.path.join(self.index_dir, "documents")

    def load(self) -> None:
        # 尝试从磁盘恢复
        if not os.path.exists(self._lucene_dir()):
            raise RuntimeError("BM25 index not found, need build() first.")
        self.pages = InMemoryPageStore.load(self._pages_dir()).load()
        self.searcher = LuceneSearcher(self._lucene_dir())  # type: ignore

    def build(self, page_store: InMemoryPageStore) -> None:
        # 0. 首先清理所有旧的目录和文件，确保干净的状态
        # 使用安全删除函数，带重试机制
        _safe_rmtree(self._lucene_dir())
        _safe_rmtree(self._docs_dir())
        
        # 1. 创建必要的目录
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self._docs_dir(), exist_ok=True)

        # 2. dump pages -> documents.jsonl (pyserini需要 id + contents)
        pages = page_store.load()
        docs_path = os.path.join(self._docs_dir(), "documents.jsonl")
        with open(docs_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(pages):
                text = (p.header + " " + p.content).strip()
                text = '\n'.join(p.content.split('\n')[1:])
                text = p.content
                json.dump({"id": str(i), "contents": text}, f, ensure_ascii=False)
                f.write("\n")

        # 3. 确保 lucene index 目录是干净的
        os.makedirs(self._lucene_dir(), exist_ok=True)

        # 4. 调 pyserini 构建 Lucene 索引
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", self._docs_dir(),
            "--index", self._lucene_dir(),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.config.get("threads", 1)),
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        
        # 添加重试机制，防止偶发的构建失败
        max_build_retries = 2
        for attempt in range(max_build_retries):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt == max_build_retries - 1:
                    print(f"[ERROR] Pyserini 索引构建失败:")
                    print(f"  stdout: {e.stdout}")
                    print(f"  stderr: {e.stderr}")
                    raise
                print(f"[WARN] Pyserini 索引构建失败，重试 {attempt + 1}/{max_build_retries}...")
                # 清理失败的索引
                _safe_rmtree(self._lucene_dir())
                os.makedirs(self._lucene_dir(), exist_ok=True)
                time.sleep(1)

        # 5. 把 pages 也固化到磁盘，供 load() / search() 反查
        # 创建临时 PageStore 实例来保存
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(pages)
        
        # 6. 更新内存镜像
        self.pages = pages
        self.searcher = LuceneSearcher(self._lucene_dir())  # type: ignore

    def update(self, page_store: InMemoryPageStore) -> None:
        # Lucene 没有好用的“增量追加+可删改文档”的轻量接口（有但复杂）；
        # 对现在这个原型我们可以直接全量重建，保持简单可靠。
        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if self.searcher is None:
            # 容错：如果忘了 load/build
            self.load()

        results_all: List[List[Hit]] = []
        for q in query_list:
            q = q.strip()
            if not q:
                results_all.append([])
                continue

            hits_for_q = []
            py_hits = self.searcher.search(q, k=top_k)
            for rank, h in enumerate(py_hits):
                # h.docid 是字符串 id
                idx = int(h.docid)
                if idx < 0 or idx >= len(self.pages):
                    continue
                page = self.pages[idx]
                snippet = page.content
                hits_for_q.append(
                    Hit(
                        page_id=str(idx),
                        snippet=snippet,
                        source="keyword",
                        meta={"rank": rank, "score": float(h.score)}
                    )
                )
            results_all.append(hits_for_q)
        return results_all
