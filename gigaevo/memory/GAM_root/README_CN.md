# General Agentic Memory (GAM)

<p align="center">
  <a href="https://arxiv.org/abs/2511.18423" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2511.18423-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://huggingface.co/papers/2511.18423" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Paper-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace">
  </a>
</p>

智能体的通用记忆系统，基于深度研究驱动

[English](README.md) | 中文文档

<h5 align="center"> 🎉 如果你喜欢我们的项目，请在 GitHub 上给我们一个 star ⭐ 以获取最新更新。</h5>

**General Agentic Memory (GAM)** 为 AI 智能体提供下一代记忆框架，将长期记忆与动态推理相结合。遵循即时(JIT)原则，在离线时保持完整的上下文保真度，在线时执行深度研究以构建自适应、高效用的上下文。通过其双智能体架构——记忆构建器和研究者——GAM 集成了结构化记忆与迭代检索和反思，在 LoCoMo, HotpotQA, RULER, 和 NarrativeQA  等基准测试中达到了最先进的性能。


- **论文**: <a href="[https://arxiv.org/abs/2511.18423](https://arxiv.org/abs/2511.18423)" target="_blank">[https://arxiv.org/abs/2511.18423](https://arxiv.org/abs/2511.18423)</a>
- **Huggingface**: <a href="[https://huggingface.co/papers/2511.18423](https://huggingface.co/papers/2511.18423)" target="_blank">[https://huggingface.co/papers/2511.18423](https://huggingface.co/papers/2511.18423)</a>
<!-- - **网站**: 
- **文档**: 
- **YouTube 视频**:  -->

<span id='features'/>

## ✨ 核心特性

* 🧠 **即时 (JIT) 记忆优化**
</br> 与传统的预先 (AOT) 系统不同，GAM 在运行时执行密集的记忆深度研究，动态检索和合成高效用的上下文以满足实时智能体需求。

* 🔍 **双智能体架构：记忆构建器 & 研究者**
</br> 协作框架，记忆构建器从原始会话构建结构化记忆，研究者执行迭代检索、反思和总结以提供精确的自适应上下文。

* 🚀 **卓越的基准性能**
</br> 在 LoCoMo, HotpotQA, RULER, 和 NarrativeQA 上取得最先进结果，在 F1 和 BLEU-1 指标上超越 A-MEM、Mem0、 MemoryOS和 LightMem 等系统。

* 🧩 **模块化与可扩展设计**
</br> 支持灵活的插件化记忆构建、检索策略和推理工具，便于集成到多智能体框架或独立 LLM 部署中。

* 🌐 **跨模型兼容性**
</br> 兼容 GPT-4、GPT-4o-mini 和 Qwen2.5 等主流大语言模型，支持云端和本地部署，适用于研究或生产环境。

<span id='news'/>

## 📣 最新动态

- **2025-11**: 发布 GAM 框架及模块化评估套件
- **2025-11**: 支持 HotpotQA、NarrativeQA、LoCoMo 和 RULER 基准测试

## 📑 目录

* [✨ 核心特性](#features)
* [🔥 最新动态](#news)
* [🏗️ 项目结构](#structure)
* [🎯 快速开始](#quick-start)
* [🔬 复现论文结果](#reproduce)
* [📖 文档](#doc)
* [🌟 引用](#cite)
* [🤝 社区](#community)

<span id='structure'/>

## 🏗️ 系统架构

![logo](./assets/GAM-memory.png)

## 🏗️ 项目结构

```
general-agentic-memory/
├── gam/                          # 核心 GAM 包
│   ├── __init__.py
│   ├── agents/                   # 智能体实现
│   │   ├── memory_agent.py      # MemoryAgent - 记忆构建
│   │   └── research_agent.py    # ResearchAgent - 深度研究
│   ├── generator/                # LLM 生成器
│   │   ├── openai_generator.py  # OpenAI API 生成器
│   │   └── vllm_generator.py    # VLLM 本地生成器
│   ├── retriever/                # 检索器
│   │   ├── index_retriever.py   # 索引检索
│   │   ├── bm25.py              # BM25 关键词检索
│   │   └── dense_retriever.py   # Dense 语义检索
│   ├── prompts/                  # 提示词模板
│   ├── schemas/                  # 数据模型
│   └── config/                   # 配置管理
├── eval/                         # 评估基准套件
│   ├── hotpotqa_test.py        # HotpotQA 评估脚本
│   ├── narrativeqa_test.py     # NarrativeQA 评估脚本
│   ├── locomo_test.py          # LoCoMo 评估脚本
│   └── ruler_test.py           # RULER 评估脚本
├── scripts/                      # Shell 脚本
│   ├── eval_hotpotqa.sh
│   ├── eval_narrativeqa.sh
│   ├── eval_locomo.sh
│   ├── eval_ruler.sh
│   └── download_data.sh
├── download_data/                # 数据下载脚本
│   ├── download_narrativeqa.py  # NarrativeQA 下载脚本
│   └── download_ruler.py       # RULER 下载脚本
├── examples/                     # 使用示例
│   └── quickstart/              # 快速开始示例
│       ├── README.md            # 示例文档
│       ├── basic_usage.py       # 基础使用示例
│       └── model_usage.py       # 模型选择示例
├── assets/                       # 资源文件
├── docs/                         # 文档目录
├── setup.py                      # 安装配置
├── pyproject.toml               # 现代项目配置
├── requirements.txt             # 依赖列表
└── README.md                    # 项目说明
```

<span id='quick-start'/>

## 🎯 快速开始

### 🚀 安装

```bash
# 克隆仓库
git clone https://github.com/VectorSpaceLab/general-agentic-memory.git
cd general-agentic-memory

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 💡 基础用法

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


# 1. 配置并创建生成器
gen_config = OpenAIGeneratorConfig(
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
    temperature=0.3,
    max_tokens=256
)
generator = OpenAIGenerator.from_config(gen_config)

# 2. 创建记忆和页面存储
memory_store = InMemoryMemoryStore()
page_store = InMemoryPageStore()

# 3. 创建 MemoryAgent
memory_agent = MemoryAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store
)

# 4. 记忆文档
documents = [
    "人工智能是计算机科学的一个分支...",
    "机器学习是人工智能的一个子集...",
    "深度学习使用神经网络..."
]

for doc in documents:
    memory_agent.memorize(doc)

# 5. 获取记忆状态
memory_state = memory_store.load()
print(f"Built {len(memory_state.abstracts)} memory abstracts")

# 6. 创建 ResearchAgent 进行问答
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

# 7. 执行研究
research_result = research_agent.research(
    request="机器学习和深度学习有什么区别？"
)
research_summary = research_result.integrated_memory

print(f"[OK] 研究完成！迭代次数: {len(research_result.raw_memory.get('iterations', []))}")
print(f"研究摘要: {research_summary}")
```

### 📚 完整示例

详细示例和高级用法：
- [`examples/quickstart/basic_usage.py`](./examples/quickstart/basic_usage.py) - 完整工作流程，包含记忆构建和研究
- [`examples/quickstart/model_usage.py`](./examples/quickstart/model_usage.py) - 模型选择和配置
- [`examples/quickstart/README.md`](./examples/quickstart/README.md) - 示例文档

<span id='reproduce'/>

## 🔬 复现论文结果

我们提供了完整的评估框架来复现论文中的实验结果。

### 数据集准备

由于数据集文件较大，它们**不会**存储在此仓库中。  
请从原始来源下载它们，并按照以下方式放置在 `data/` 目录下：

- **LoCoMo**

  - 从以下地址下载 `locomo10.json`  
    https://github.com/snap-research/locomo/blob/main/data/locomo10.json  
  - 保存为：
    - `data/locomo10.json`  

- **HotpotQA**

  - 从以下地址下载以下文件  
    https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main  
    - `eval_400.json`  
    - `eval_1600.json`  
    - `eval_3200.json`  
  - 放置在：
    - `data/hotpotqa/`  
      (或通过 `--data-path` 传递要评估的确切文件)

- **RULER**

  - 从以下地址下载 `data` 文件夹  
    https://huggingface.co/datasets/lighteval/RULER-131072-Qwen2.5-Instruct/tree/main  
  - 放置在：
    - `data/ruler/`  

- **NarrativeQA**

  - 从以下地址下载 `data` 文件夹  
    https://huggingface.co/datasets/deepmind/narrativeqa/tree/main  
  - 放置在：
    - `data/narrativeqa/`

### 快速开始

```bash
# 1. 准备数据集
mkdir -p data
# 从上面的链接下载数据集，并按照建议的目录结构放置在 data/ 目录下
bash scripts/download_data.sh

# 2. 设置环境变量
export OPENAI_API_KEY="your_api_key_here"

# 3. 运行评估

# HotpotQA
bash scripts/eval_hotpotqa.sh

# NarrativeQA
bash scripts/eval_narrativeqa.sh

# LoCoMo
bash scripts/eval_locomo.sh

# RULER
bash scripts/eval_ruler.sh
```

### 直接使用 Python 运行

你也可以直接运行评估脚本：

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
    --data data/ruler/data/qa_1.jsonl \
    --outdir ./results/ruler/qa_1 \
    --memory-api-key $OPENAI_API_KEY \
    --memory-model gpt-4o-mini \
    --research-api-key $OPENAI_API_KEY \
    --research-model gpt-4o-mini \
    --working-api-key $OPENAI_API_KEY \
    --working-model gpt-4o-mini \
    --embedding-model-path BAAI/bge-m3
```

### 支持的数据集

| 数据集 | 任务类型 | 评估指标 | 脚本 |
|---------|----------|---------|------|
| **HotpotQA** | 多跳问答 | F1 | [eval/hotpotqa_test.py](./eval/hotpotqa_test.py) |
| **NarrativeQA** | 叙事问答 | F1 | [eval/narrativeqa_test.py](./eval/narrativeqa_test.py) |
| **LoCoMo** | 对话记忆 | F1, BLEU-1 | [eval/locomo_test.py](./eval/locomo_test.py) |
| **RULER** | 长上下文 | Accuracy | [eval/ruler_test.py](./eval/ruler_test.py) |

<span id='doc'/>

## 📖 文档

更详细的文档即将推出 🚀。同时可以查看这些资源：

- [示例文档](./examples/quickstart/README.md) - 使用示例和教程
- [评估脚本](./eval/) - 每个数据集的直接评估脚本

<span id='cite'/>

## 📣 引用

**如果你觉得这个项目有用，请考虑引用我们的论文：**

```bibtex
@article{yan2025general,
  title={General Agentic Memory Via Deep Research},
  author={Yan, BY and Li, Chaofan and Qian, Hongjin and Lu, Shuqi and Liu, Zheng},
  journal={arXiv preprint arXiv:2511.18423},
  year={2025}
}
```

<span id='community'/>

## 🤝 社区

### 🎯 联系我们

- GitHub Issues: [报告 bug 或请求功能](https://github.com/VectorSpaceLab/general-agentic-memory/issues)
- Email: zhengliu1026@gmail.com

### 🌟 Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=VectorSpaceLab/general-agentic-memory&type=Date)](https://star-history.com/#VectorSpaceLab/general-agentic-memory&Date)

### 🤝 贡献

欢迎贡献！请随时提交 issues 或 pull requests。

1. Fork 仓库
2. 创建你的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下数据集的作者：
- HotpotQA
- NarrativeQA
- LoCoMo
- RULER

## 免责声明

这是一个研究项目。请负责任和道德地使用它。

---

<p align="center">
Made with ❤️ by the GAM Team
</p>

