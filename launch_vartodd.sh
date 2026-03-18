#!/usr/bin/env bash
# Launch script for vartodd circuit evolution experiment
# Uses Gemini 3 Flash Preview via OpenRouter

set -euo pipefail

DEPS_DIR="~/envs/evo_fast/lib/vartodd_deps"
CONDA_LIB="~/envs/evo_fast/lib"
PYTHON="~/envs/evo_fast/bin/python"

export LD_LIBRARY_PATH="${DEPS_DIR}:${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${DEPS_DIR}/libglibc_shim.so"
export PYTHONPATH="problems/vartodd:.:${PYTHONPATH:-}"
export GIGAEVO_PYTHON="${PYTHON}"

# Read OPENAI_API_KEY from .env (run.py's dotenv handles this,
# but we also export it so exec_runner subprocesses inherit it)
if [ -f .env ]; then
    export $(grep -E '^OPENAI_API_KEY=' .env | head -1)
fi

exec ${PYTHON} run.py \
    problem.name=vartodd \
    pipeline=mcts_evo \
    model_name=google/gemini-3-flash-preview \
    llm_base_url=https://openrouter.ai/api/v1 \
    redis.db=0 \
    max_mutations_per_generation=16
