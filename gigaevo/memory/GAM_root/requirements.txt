# Core Dependencies (Required)
pydantic>=2.0.0
openai>=1.0.0
tiktoken>=0.7.0
tqdm>=4.65.0
numpy>=1.23.0
requests>=2.28.0

# Dense Retriever - Semantic Search (Recommended)
FlagEmbedding>=1.2.0
faiss-cpu>=1.7.4

# Evaluation Framework
datasets>=2.18.0

# Text Processing
nltk>=3.8.0

# Optional: BM25 Retriever - Keyword Search
# Uncomment if you need BM25Retriever
pyserini>=0.22.0

# Optional: VLLM - Local Model Inference
# Uncomment if you want to use local models
vllm>=0.6.0
torch>=2.0.0
transformers>=4.30.0

# Optional: Development Tools
# Uncomment for development
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
