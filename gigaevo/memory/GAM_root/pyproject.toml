[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "general-agentic-memory"
version = "0.1.0"
description = "A general memory system for agents, powered by deep-research"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "GAM Team", email = "zhengliu1026@gmail.com"},
]
keywords = ["ai", "memory", "agent", "llm", "nlp"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "tqdm",
    "tiktoken",
    "openai",
    "transformers",
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "black>=22.0",
    "flake8>=4.0",
    "mypy>=0.900",
]

[project.urls]
Homepage = "https://github.com/VectorSpaceLab/general-agentic-memory"
Repository = "https://github.com/VectorSpaceLab/general-agentic-memory"
Issues = "https://github.com/VectorSpaceLab/general-agentic-memory/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["gam*"]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
