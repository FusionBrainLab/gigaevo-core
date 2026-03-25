# General Agentic Memory (GAM)

<p align="center">
  <a href="https://arxiv.org/abs/2511.18423" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2511.18423-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://huggingface.co/papers/2511.18423" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Paper-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace">
  </a>
</p>

A general memory system for agents, powered by deep-research

[中文文档](README_CN.md) | English

<h5 align="center"> 🎉 If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

**General Agentic Memory (GAM)** provides a next-generation memory framework for AI agents, combining long-term retention with dynamic reasoning. Following the Just-in-Time (JIT) principle, it preserves full contextual fidelity offline while performing deep research online to build adaptive, high-utility context. With its dual-agent architecture—Memorizer and Researcher—GAM integrates structured memory with iterative retrieval and reflection, achieving state-of-the-art performance across LoCoMo, HotpotQA, RULER, and NarrativeQA benchmarks.

- **Paper**: <a href="[https://arxiv.org/abs/2511.18423](https://arxiv.org/abs/2511.18423)" target="_blank">[https://arxiv.org/abs/2511.18423](https://arxiv.org/abs/2511.18423)</a>
- **Huggingface**: <a href="[https://huggingface.co/papers/2511.18423](https://huggingface.co/papers/2511.18423)" target="_blank">[https://huggingface.co/papers/2511.18423](https://huggingface.co/papers/2511.18423)</a>
<!-- - **Website**: 
- **Documentation**: 
- **YouTube Video**:  -->

<span id='features'/>

## ✨ Key Features

* 🧠 **Just-in-Time (JIT) Memory Optimization**
</br> Unlike conventional Ahead-of-Time (AOT) systems, GAM performs intensive Memory Deep Research at runtime, dynamically retrieving and synthesizing high-utility context to meet real-time agent needs.

* 🔍 **Dual-Agent Architecture: Memorizer & Researcher**
</br> A cooperative framework where the Memorizer constructs structured memory from raw sessions, and the Researcher performs iterative retrieval, reflection, and summarization to deliver precise, adaptive context.

* 🚀 **Superior Performance Across Benchmarks**
</br> Achieves state-of-the-art results on LoCoMo, HotpotQA, RULER, and NarrativeQA, surpassing prior systems such as A-MEM、Mem0、 MemoryOS and LightMem in both F1 and BLEU-1 metrics.

* 🧩 **Modular & Extensible Design**
</br> Built to support flexible plug-ins for memory construction, retrieval strategies, and reasoning tools—facilitating easy integration into multi-agent frameworks or standalone LLM deployments.

* 🌐 **Cross-Model Compatibility**
</br> Compatible with leading LLMs such as GPT-4, GPT-4o-mini, and Qwen2.5, supporting both cloud-based and local deployments for research or production environments.

<span id='news'/>

## 📣 Latest News

- **2025-11**: Released GAM framework with modular evaluation suite
- **2025-11**: Support for HotpotQA, NarrativeQA, LoCoMo, and RULER benchmarks

## 📑 Table of Contents

* [✨ Features](#features)
* [🔥 News](#news)
* [🏗️ Project Structure](#structure)
* [🎯 Quick Start](#quick-start)
* [🕐 TTL for Production](#ttl-time-to-live-for-production)
* [🔬 Reproducing Paper Results](#reproduce)
* [📖 Documentation](#doc)
* [🌟 Citation](#cite)
* [🤝 Community](#community)

<span id='structure'/>

## 🏗️ System Architecture

![logo](./assets/GAM-memory.png)

## 🏗️ Project Structure

```
general-agentic-memory/
├── gam/                          # Core GAM package
│   ├── __init__.py
│   ├── agents/                   # Agent implementations
│   │   ├── memory_agent.py      # MemoryAgent - memory construction
│   │   └── research_agent.py    # ResearchAgent - deep research
│   ├── generator/                # LLM generators
│   │   ├── openai_generator.py  # OpenAI API generator
│   │   └── vllm_generator.py    # VLLM local generator
│   ├── retriever/                # Retrievers
│   │   ├── index_retriever.py   # Index retrieval
│   │   ├── bm25.py              # BM25 keyword retrieval
│   │   └── dense_retriever.py   # Dense semantic retrieval
│   ├── prompts/                  # Prompt templates
│   ├── schemas/                  # Data models
│   └── config/                   # Configuration management
├── eval/                         # Evaluation suite
│   ├── hotpotqa_test.py        # HotpotQA evaluation script
│   ├── narrativeqa_test.py     # NarrativeQA evaluation script
│   ├── locomo_test.py          # LoCoMo evaluation script
│   └── ruler_test.py           # RULER evaluation script
├── scripts/                      # Shell scripts
│   ├── eval_hotpotqa.sh
│   ├── eval_narrativeqa.sh
│   ├── eval_locomo.sh
│   ├── eval_ruler.sh
│   └── download_data.sh
├── download_data/                # Data download scripts
│   ├── download_narrativeqa.py  # NarrativeQA download script
│   └── download_ruler.py       # RULER download script
├── examples/                     # Usage examples
│   └── quickstart/              # Quick start examples
│       ├── README.md            # Examples documentation
│       ├── basic_usage.py       # Basic usage example
│       └── model_usage.py       # Model selection example
├── assets/                       # Resource files
├── docs/                         # Documentation
├── setup.py                      # Installation config
├── pyproject.toml               # Modern project config
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

<span id='quick-start'/>

## 🎯 Quick Start

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/VectorSpaceLab/general-agentic-memory.git
cd general-agentic-memory

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 💡 Basic Usage

```python
import os
from gam import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
    DenseRetrieverConfig,
    DenseRetriever,
    IndexRetrieverConfig,
    IndexRetriever,
    BM25RetrieverConfig,
    BM25Retriever
)

# 1. Configure and create generator
gen_config = OpenAIGeneratorConfig(
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
    temperature=0.3,
    max_tokens = 256
)
generator = OpenAIGenerator.from_config(gen_config)

# 2. Create memory and page stores
memory_store = InMemoryMemoryStore()
page_store = InMemoryPageStore()

# 3. Create MemoryAgent
memory_agent = MemoryAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store
)

# 4. Memorize documents
documents = [
    "Artificial Intelligence is a branch of computer science...",
    "Machine Learning is a subset of AI...",
    "Deep Learning uses neural networks..."
]

for doc in documents:
    memory_agent.memorize(doc)

# 5. Get memory state
memory_state = memory_store.load()
print(f"Built {len(memory_state.abstracts)} memory abstracts")

# 6. Create ResearchAgent for Q&A

retrievers={}
index_dir = './tmp'
try:
    page_index_dir = os.path.join(index_dir, "page_index")
    if os.path.exists(page_index_dir):
        import shutil
        shutil.rmtree(page_index_dir)
    
    index_config = IndexRetrieverConfig(
        index_dir=page_index_dir
    )
    index_retriever = IndexRetriever(index_config.__dict__)
    index_retriever.build(page_store)
    retrievers["page_index"] = index_retriever
except Exception as e:
    print(f"[WARN] page retriever error: {e}")

try:
    bm25_index_dir = os.path.join(index_dir, "bm25_index")
    if os.path.exists(bm25_index_dir):
        import shutil
        shutil.rmtree(bm25_index_dir)
    
    bm25_config = BM25RetrieverConfig(
        index_dir=bm25_index_dir,
        threads=1
    )
    bm25_retriever = BM25Retriever(bm25_config.__dict__)
    bm25_retriever.build(page_store)
    retrievers["keyword"] = bm25_retriever
except Exception as e:
    print(f"[WARN] BM25 retriever error: {e}")

try:
    dense_index_dir = os.path.join(index_dir, "dense_index")
    if os.path.exists(dense_index_dir):
        import shutil
        shutil.rmtree(dense_index_dir)
    
    dense_config = DenseRetrieverConfig(
        index_dir=dense_index_dir,
        model_name="BAAI/bge-m3"
    )
    dense_retriever = DenseRetriever(dense_config.__dict__)
    dense_retriever.build(page_store)
    retrievers["vector"] = dense_retriever
except Exception as e:
    print(f"[WARN] Dense retriever error: {e}")

research_agent_kwargs = {
    "page_store": page_store,
    "memory_store": memory_store,
    "retrievers": retrievers,
    "generator": generator,
    "max_iters": 5
}

research_agent = ResearchAgent(**research_agent_kwargs)

# 7. Perform research
research_result = research_agent.research(
    request="What is the difference between ML and DL?"
)

research_summary = research_result.integrated_memory


print(f"[OK] Research completed! Iteration count: {len(research_result.raw_memory.get('iterations', []))}")
print(f"Research Summary: {research_summary}```

### 🕐 TTL (Time-To-Live) for Production

For long-running applications, enable automatic cleanup of old memories and pages:

```python
from gam import TTLMemoryStore, TTLPageStore

# Create stores with 30-day TTL
memory_store = TTLMemoryStore(
    dir_path="./data",
    ttl_days=30,
    enable_auto_cleanup=True
)
page_store = TTLPageStore(
    dir_path="./data",
    ttl_days=30,
    enable_auto_cleanup=True
)

# Use with agents as normal
memory_agent = MemoryAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store
)

# Monitor cleanup statistics
stats = memory_store.get_stats()
print(f"Total: {stats['total']}, Valid: {stats['valid']}, Expired: {stats['expired']}")

# Manual cleanup (if auto-cleanup disabled)
removed = memory_store.cleanup_expired()
print(f"Removed {removed} expired entries")
```

**Key Features:**
- ✅ Prevents unbounded growth in long-running applications
- ✅ Auto-cleanup on load (configurable)
- ✅ Flexible TTL: days, hours, minutes, or seconds
- ✅ Statistics tracking: total, valid, expired counts
- ✅ Fully backward compatible with existing data
- ✅ TTL can be disabled (works like regular stores)

**See Also:** [`examples/quickstart/ttl_usage.py`](./examples/quickstart/ttl_usage.py) for complete examples.

```
```

### 📚 Complete Examples

For detailed examples and advanced usage:
- [`examples/quickstart/basic_usage.py`](./examples/quickstart/basic_usage.py) - Complete workflow with memory building and research
- [`examples/quickstart/model_usage.py`](./examples/quickstart/model_usage.py) - Model selection and configuration
- [`examples/quickstart/README.md`](./examples/quickstart/README.md) - Examples documentation

<span id='reproduce'/>

## 🔬 How to Reproduce the Results in the Paper

We provide a complete evaluation framework to reproduce the experimental results in the paper.

### Datasets

Because the datasets are large, they are **not** stored in this repository.  
Please download them from the original sources and place them under the `data/` directory as follows:

- **LoCoMo**

  - Download `locomo10.json` from  
    https://github.com/snap-research/locomo/blob/main/data/locomo10.json  
  - Save it as:
    - `data/locomo10.json`  

- **HotpotQA**

  - Download the following files from  
    https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main  
    - `eval_400.json`  
    - `eval_1600.json`  
    - `eval_3200.json`  
  - Place them under:
    - `data/hotpotqa/`  
      (or pass the exact file you want to evaluate via `--data-path`)

- **RULER**

  - Download the `data` folder from  
    https://huggingface.co/datasets/lighteval/RULER-131072-Qwen2.5-Instruct/tree/main  
  - Place it under:
    - `data/ruler/`  

- **NarrativeQA**

  - Download the `data` folder from  
    https://huggingface.co/datasets/deepmind/narrativeqa/tree/main  
  - Place it under:
    - `data/narrativeqa/`

### Quick Start

```bash
# 1. Prepare datasets
mkdir -p data
# Download the datasets from the links above and place them under data/
# following the suggested directory structure.
bash scripts/download_data.sh

# 2. Set environment variables
export OPENAI_API_KEY="your_api_key_here"

# 3. Run evaluations

# HotpotQA
bash scripts/eval_hotpotqa.sh

# NarrativeQA
bash scripts/eval_narrativeqa.sh

# LoCoMo
bash scripts/eval_locomo.sh

# RULER
bash scripts/eval_ruler.sh
```

### Using Python Directly

You can also run the evaluation scripts directly:

```bash
# HotpotQA
python eval/hotpotqa_test.py \
    --data data/hotpotqa/eval_400.json \
    --outdir ./results/hotpotqa \
    --memory-api-key $OPENAI_API_KEY \
    --memory-model gpt-4o-mini \
    --research-api-key $OPENAI_API_KEY \
    --research-model gpt-4o-mini \
    --working-api-key $OPENAI_API_KEY \
    --working-model gpt-4o-mini \
    --embedding-model-path BAAI/bge-m3

# NarrativeQA
python eval/narrativeqa_test.py \
    --data-dir data/narrativeqa \
    --split test \
    --outdir ./results/narrativeqa \
    --memory-api-key $OPENAI_API_KEY \
    --memory-model gpt-4o-mini \
    --research-api-key $OPENAI_API_KEY \
    --research-model gpt-4o-mini \
    --working-api-key $OPENAI_API_KEY \
    --working-model gpt-4o-mini \
    --embedding-model-path BAAI/bge-m3

# LoCoMo
python eval/locomo_test.py \
    --data data/locomo10.json \
    --outdir ./results/locomo \
    --memory-api-key $OPENAI_API_KEY \
    --memory-model gpt-4o-mini \
    --research-api-key $OPENAI_API_KEY \
    --research-model gpt-4o-mini \
    --working-api-key $OPENAI_API_KEY \
    --working-model gpt-4o-mini

# RULER
python eval/ruler_test.py \
    --data data/ruler/qa_1.jsonl \
    --outdir ./results/ruler/qa_1 \
    --memory-api-key $OPENAI_API_KEY \
    --memory-model gpt-4o-mini \
    --research-api-key $OPENAI_API_KEY \
    --research-model gpt-4o-mini \
    --working-api-key $OPENAI_API_KEY \
    --working-model gpt-4o-mini \
    --embedding-model-path BAAI/bge-m3
```

### Supported Datasets

| Dataset | Task Type | Metrics | Script |
|---------|-----------|---------|--------|
| **HotpotQA** | Multi-hop QA | F1 | [eval/hotpotqa_test.py](./eval/hotpotqa_test.py) |
| **NarrativeQA** | Narrative QA | F1 | [eval/narrativeqa_test.py](./eval/narrativeqa_test.py) |
| **LoCoMo** | Conversation Memory | F1, BLEU-1 | [eval/locomo_test.py](./eval/locomo_test.py) |
| **RULER** | Long Context | Accuracy | [eval/ruler_test.py](./eval/ruler_test.py) |

<span id='doc'/>

## 📖 Documentation

More detailed documentation is coming soon 🚀. Check these resources in the meantime:

- [Examples Documentation](./examples/quickstart/README.md) - Usage examples and tutorials
- [Evaluation Scripts](./eval/) - Direct evaluation scripts for each dataset

<span id='cite'/>

## 📣 Citation

**If you find this project useful, please consider citing our paper:**

```bibtex
@article{yan2025general,
  title={General Agentic Memory Via Deep Research},
  author={Yan, BY and Li, Chaofan and Qian, Hongjin and Lu, Shuqi and Liu, Zheng},
  journal={arXiv preprint arXiv:2511.18423},
  year={2025}
}
```

<span id='community'/>

## 🤝 Community

### 🎯 Contact Us

- GitHub Issues: [Report bugs or request features](https://github.com/VectorSpaceLab/general-agentic-memory/issues)
- Email: zhengliu1026@gmail.com

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VectorSpaceLab/general-agentic-memory&type=Date)](https://star-history.com/#VectorSpaceLab/general-agentic-memory&Date)

### 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the authors of the following datasets:
- HotpotQA
- NarrativeQA
- LoCoMo
- RULER

## Disclaimer

This is a research project. Please use it responsibly and ethically.

---

<p align="center">
Made with ❤️ by the GAM Team
</p>
