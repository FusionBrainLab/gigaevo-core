import os
import json
import numpy as np
import requests
from typing import Dict, Any, List, Optional
from FlagEmbedding import FlagAutoModel
import faiss

from GAM_root.gam.retriever.base import AbsRetriever
from GAM_root.gam.schemas import InMemoryPageStore, Hit, Page


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    构建 FAISS 索引
    embeddings: (n, dim) 的 numpy 数组
    """
    dimension = embeddings.shape[1]
    # 使用内积索引（cosine similarity）
    index = faiss.IndexFlatIP(dimension)
    # L2 归一化以支持 cosine similarity（复制数组以避免修改原始数据）
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    index.add(embeddings_normalized)
    return index


def _search_faiss_index(index: faiss.Index, query_embeddings: np.ndarray, top_k: int):
    """
    在 FAISS 索引中搜索
    index: FAISS 索引
    query_embeddings: (n_queries, dim) 的查询向量
    top_k: 返回的 top-k 结果数
    返回: (scores_list, indices_list) 其中每个元素都是 (top_k,) 的数组
    """
    # L2 归一化查询向量（复制以避免修改原始数据）
    query_embeddings_normalized = query_embeddings.copy()
    faiss.normalize_L2(query_embeddings_normalized)
    
    # 搜索
    scores, indices = index.search(query_embeddings_normalized, top_k)
    
    scores_list = [scores[i] for i in range(len(query_embeddings))]
    indices_list = [indices[i] for i in range(len(query_embeddings))]
    
    return scores_list, indices_list


class DenseRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pages = None
        self.index = None
        self.doc_emb = None
        
        # 检查是否使用 API 模式
        self.api_url = config.get("api_url")  # 如 "http://localhost:8001"
        self.use_api = self.api_url is not None
        
        if self.use_api:
            # API 模式：不加载本地模型
            print(f"[DenseRetriever] 使用 API 模式: {self.api_url}")
            self.model = None
            # 测试连接
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"[DenseRetriever] API 服务连接成功: {response.json()}")
                else:
                    print(f"[DenseRetriever] 警告: API 服务响应异常: {response.status_code}")
            except Exception as e:
                print(f"[DenseRetriever] 警告: 无法连接到 API 服务: {e}")
        else:
            # 本地模式：加载模型
            model_name = config.get("model_name")
            print(f"[DenseRetriever] 使用本地模式，加载模型: {model_name}")
            try:
                # 检测是否有可用的 GPU
                import torch
                has_cuda = torch.cuda.is_available()
                default_device = "cuda:0" if has_cuda else "cpu"
                devices = config.get("devices", default_device)
                
                self.model = FlagAutoModel.from_finetuned(
                    model_name,
                    model_class=config.get("model_class", None),
                    normalize_embeddings=config.get("normalize_embeddings", True),
                    pooling_method=config.get("pooling_method", "cls"),
                    trust_remote_code=config.get("trust_remote_code", True),
                    query_instruction_for_retrieval=config.get("query_instruction_for_retrieval"),
                    use_fp16=config.get("use_fp16", False),
                    devices=devices
                )
                if self.model is None:
                    raise RuntimeError(f"模型加载失败：FlagAutoModel.from_finetuned() 返回了 None")
                print(f"[DenseRetriever] 模型加载成功，使用设备: {devices}")
            except Exception as e:
                error_msg = (
                    f"[DenseRetriever] 模型加载失败: {e}\n"
                    f"  模型名称: {model_name}\n"
                    f"  请检查: 1) 模型名称是否正确 2) 网络连接是否正常 3) 是否有足够的磁盘空间"
                )
                print(error_msg)
                raise RuntimeError(error_msg) from e


    # ---------- 内部小工具 ----------
    def _index_dir(self) -> str:
        return self.config["index_dir"]

    def _pages_dir(self) -> str:
        return os.path.join(self._index_dir(), "pages")

    def _emb_path(self) -> str:
        return os.path.join(self._index_dir(), "doc_emb.npy")

    def _encode_via_api(self, texts: List[str], encode_type: str = "corpus") -> np.ndarray:
        """
        通过 API 编码文本
        
        Args:
            texts: 文本列表
            encode_type: "corpus" 或 "query"
        
        Returns:
            embeddings as numpy array
        """
        # 验证输入
        if not texts:
            raise ValueError(f"[DenseRetriever] 文本列表为空，无法编码")
        
        # 过滤掉空字符串
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            raise ValueError(f"[DenseRetriever] 所有文本都为空，无法编码")
        
        if len(non_empty_texts) != len(texts):
            print(f"[DenseRetriever] 警告: 过滤掉了 {len(texts) - len(non_empty_texts)} 个空文本")
        
        try:
            request_data = {
                "texts": non_empty_texts,
                "type": encode_type,
                "batch_size": self.config.get("batch_size", 32),
                "max_length": self.config.get("max_length", 512),
            }
            
            response = requests.post(
                f"{self.api_url}/encode",
                json=request_data,
                timeout=300  # 5分钟超时，大批量编码可能需要较长时间
            )
            
            # 如果请求失败，打印详细的错误信息
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_response = response.json()
                    error_detail = f" 服务器错误信息: {error_response}"
                except:
                    error_detail = f" 响应内容: {response.text[:500]}"
                
                error_msg = (
                    f"[DenseRetriever] API 编码失败: {response.status_code} {response.reason}\n"
                    f"  请求URL: {self.api_url}/encode\n"
                    f"  请求参数: texts数量={len(non_empty_texts)}, type={encode_type}, "
                    f"batch_size={request_data['batch_size']}, max_length={request_data['max_length']}\n"
                    f"{error_detail}"
                )
                print(error_msg)
                response.raise_for_status()
            
            result = response.json()
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            
            # 如果过滤了空文本，需要补充空向量以保持索引对应
            if len(non_empty_texts) != len(texts):
                # 找到空文本的位置
                empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
                # 构建完整的 embeddings 数组
                full_embeddings = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
                non_empty_idx = 0
                for i in range(len(texts)):
                    if i not in empty_indices:
                        full_embeddings[i] = embeddings[non_empty_idx]
                        non_empty_idx += 1
                embeddings = full_embeddings
            
            return embeddings
        except requests.exceptions.RequestException as e:
            error_msg = (
                f"[DenseRetriever] API 编码失败（网络错误）: {e}\n"
                f"  请求URL: {self.api_url}/encode\n"
                f"  请检查: 1) API服务是否运行 2) URL是否正确 3) 网络连接是否正常"
            )
            print(error_msg)
            raise
        except Exception as e:
            print(f"[DenseRetriever] API 编码失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _encode_pages(self, pages: List[Page]) -> np.ndarray:
        # 和 build() / update() 保持一致的编码方式
        # 处理可能的 None 值
        texts = []
        for p in pages:
            header = p.header if p.header is not None else ""
            content = p.content if p.content is not None else ""
            text = (header + " " + content).strip()
            text = p.content
            texts.append(text)
        
        if self.use_api:
            # API 模式
            return self._encode_via_api(texts, encode_type="corpus")
        else:
            # 本地模式
            if self.model is None:
                raise RuntimeError("DenseRetriever 模型未初始化，无法编码。请检查模型加载是否成功。")
            return self.model.encode_corpus(
                texts,
                batch_size=self.config.get("batch_size", 32),
                max_length=self.config.get("max_length", 512),
            )

    # ---------- 对外接口 ----------
    def load(self) -> None:
        """
        从磁盘恢复：
        - pages 快照
        - doc_emb.npy
        - faiss 索引
        """
        # 如果load失败，不抛死，只打印，这样ResearchAgent可以再走build()
        try:
            # 读向量
            self.doc_emb = np.load(self._emb_path())
            # 重建 index
            self.index = _build_faiss_index(self.doc_emb)
            # 读 pages
            self.pages = InMemoryPageStore.load(self._pages_dir()).load()
        except Exception as e:
            print("DenseRetriever.load() failed, will need build():", e)

    def build(self, page_store: InMemoryPageStore) -> None:
        """
        全量重建向量索引。
        """
        os.makedirs(self._pages_dir(), exist_ok=True)

        # 1. 把当前 page_store 取出来
        self.pages = page_store.load()

        # 2. 全量编码
        self.doc_emb = self._encode_pages(self.pages)

        # 3. 建 faiss 索引
        self.index = _build_faiss_index(self.doc_emb)

        # 4. 持久化
        # 创建临时 PageStore 实例来保存
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(self.pages)
        np.save(self._emb_path(), self.doc_emb)

    def update(self, page_store: InMemoryPageStore) -> None:
        """
        增量更新：如果只是新增了一些 Page，或者后半段变了，
        我们就只重新编码“变化起点”之后的部分，而不是全量重算。
        """
        # 如果我们还没有 build 过，就直接走 build
        if not self.pages or self.doc_emb is None or self.index is None:
            self.build(page_store)
            return

        new_pages = page_store.load()
        old_pages = self.pages

        # 1. 找到第一个差异位置 diff_idx
        max_shared = min(len(new_pages), len(old_pages))
        diff_idx = max_shared  # 假设一开始完全一致
        for i in range(max_shared):
            if Page.equal(new_pages[i], old_pages[i]):
                continue
            diff_idx = i
            break

        # 2. 判断有没有实际变化
        changed = (diff_idx < max_shared) or (len(new_pages) != len(old_pages))
        if not changed:
            # 完全没变，直接返回
            return

        # 3. 我们保留前 diff_idx 段的老向量，后半段重新编码
        keep_emb = self.doc_emb[:diff_idx]

        tail_pages = new_pages[diff_idx:]
        tail_emb = self._encode_pages(tail_pages)

        new_doc_emb = np.concatenate([keep_emb, tail_emb], axis=0)

        # 4. 重新建 faiss 索引
        self.index = _build_faiss_index(new_doc_emb)

        # 5. 持久化 + 刷内存
        # 更新内存
        self.pages = new_pages
        self.doc_emb = new_doc_emb
        
        # 创建临时 PageStore 实例来保存
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(self.pages)
        np.save(self._emb_path(), self.doc_emb)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        """
        输入: 多个query
        输出: 对应多个query的检索结果 (Hit 列表)
        对于多个query，会对每个query进行独立搜索，然后按page_id聚合得分（累加），最后返回top_k结果
        """
        if self.index is None:
            # 如果还没 index（比如没调用 build/load），尝试load一下
            self.load()
            # 如果load也没成功，那 index 还是 None，就直接空返回
            if self.index is None:
                return [[] for _ in query_list]

        # 把所有 query 一起编码
        if self.use_api:
            # API 模式
            queries_emb = self._encode_via_api(query_list, encode_type="query")
        else:
            # 本地模式
            queries_emb = self.model.encode_queries(
                query_list,
                batch_size=self.config.get("batch_size", 32),
                max_length=self.config.get("max_length", 512),
            )

        # 使用自定义的 search 函数
        scores_list, indices_list = _search_faiss_index(self.index, queries_emb, top_k)

        # 按 page_id 聚合得分：如果同一个 page 被多个 query 搜索到，累加得分
        page_scores: Dict[str, float] = {}  # page_id -> 累计得分
        page_hits: Dict[str, Hit] = {}      # page_id -> Hit对象（保存第一个遇到的Hit作为代表）

        for scores, indices in zip(scores_list, indices_list):
            for rank, (idx, sc) in enumerate(zip(indices, scores)):
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= len(self.pages):
                    continue
                page = self.pages[idx_int]
                snippet = page.content
                page_id = str(idx_int)
                score = float(sc)

                if page_id in page_scores:
                    # 累加得分
                    page_scores[page_id] += score
                else:
                    # 第一次遇到这个page，保存得分和Hit对象
                    page_scores[page_id] = score
                    page_hits[page_id] = Hit(
                        page_id=page_id,
                        snippet=snippet,
                        source="vector",
                        meta={"rank": rank, "score": score},
                    )

        # 按总分排序，取 top k
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_pages = sorted_pages[:top_k]

        # 构建最终的hits列表（使用累加后的得分）
        final_hits: List[Hit] = []
        for rank, (page_id, total_score) in enumerate(top_k_pages):
            hit = page_hits[page_id]
            # 更新meta中的得分为累加后的总分
            updated_meta = hit.meta.copy() if hit.meta else {}
            updated_meta["rank"] = rank
            updated_meta["score"] = total_score
            final_hits.append(
                Hit(
                    page_id=hit.page_id,
                    snippet=hit.snippet,
                    source=hit.source,
                    meta=updated_meta
                )
            )

        # 返回 List[List[Hit]] 格式（只有一个列表，即聚合后的结果）
        return [final_hits]