#!/bin/bash
# Launch adversarial co-evolution: Optimizer (Pop A) vs Deceptive Landscapes (Pop B)
#
# Pop A (DB 1): Evolves optimizers, tested against Pop B's landscapes
# Pop B (DB 2): Evolves deceptive landscapes, tested against Pop A's optimizers
#
# Both populations use MainRunSyncHook to advance in lockstep.
# Each validator reads the opponent's MAP-Elites archive from Redis.
#
# Usage: bash problems/adversarial/launch_optimizer.sh

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ"

export GIGAEVO_PYTHON="${GIGAEVO_PYTHON:-python3}"
export PYTHONPATH="$PROJ"
export OPENAI_API_KEY="sk-gigaevo"
export NO_PROXY="INTERNAL_IP,INTERNAL_IP,INTERNAL_IP,INTERNAL_IP,INTERNAL_IP,INTERNAL_IP,localhost,127.0.0.1"

# --- Configuration ---
LLM_URL="http://localhost:8000/v1"
MODEL="Qwen3-235B-A22B-Thinking-2507"
MAX_GEN="${1:-10}"  # Default 10 generations, override via first arg

POP_A_DB=1
POP_A_PREFIX="adversarial/optimizer/pop_a"
POP_B_DB=2
POP_B_PREFIX="adversarial/optimizer/pop_b"

LOG_DIR="/tmp/adversarial_optimizer"
mkdir -p "$LOG_DIR"

echo "=== Adversarial Co-Evolution: Optimizer vs Landscapes ==="
echo "Pop A (optimizer):  DB=$POP_A_DB, prefix=$POP_A_PREFIX"
echo "Pop B (landscape):  DB=$POP_B_DB, prefix=$POP_B_PREFIX"
echo "Max generations:    $MAX_GEN"
echo "LLM:                $MODEL via $LLM_URL"
echo "Logs:               $LOG_DIR/"
echo ""

# --- Launch Pop A: Optimizer ---
# Reads opponent landscapes from Pop B's archive (DB 2)
# Syncs with Pop B via MainRunSyncHook
OPPONENT_REDIS_HOST=localhost \
OPPONENT_REDIS_PORT=6379 \
OPPONENT_REDIS_DB=$POP_B_DB \
OPPONENT_PREFIX=$POP_B_PREFIX \
nohup "$GIGAEVO_PYTHON" run.py \
  problem.name=adversarial/optimizer/pop_a \
  pipeline=adversarial \
  redis.db=$POP_A_DB \
  opponent_redis_db=$POP_B_DB \
  opponent_redis_prefix=$POP_B_PREFIX \
  max_generations=$MAX_GEN \
  llm_base_url=$LLM_URL \
  model_name=$MODEL \
  > "$LOG_DIR/pop_a.log" 2>&1 &

PID_A=$!
echo "Pop A launched: PID=$PID_A"

sleep 2

# --- Launch Pop B: Deceptive Landscape ---
# Reads opponent optimizers from Pop A's archive (DB 1)
# Syncs with Pop A via MainRunSyncHook
OPPONENT_REDIS_HOST=localhost \
OPPONENT_REDIS_PORT=6379 \
OPPONENT_REDIS_DB=$POP_A_DB \
OPPONENT_PREFIX=$POP_A_PREFIX \
nohup "$GIGAEVO_PYTHON" run.py \
  problem.name=adversarial/optimizer/pop_b \
  pipeline=adversarial \
  redis.db=$POP_B_DB \
  opponent_redis_db=$POP_A_DB \
  opponent_redis_prefix=$POP_A_PREFIX \
  max_generations=$MAX_GEN \
  llm_base_url=$LLM_URL \
  model_name=$MODEL \
  > "$LOG_DIR/pop_b.log" 2>&1 &

PID_B=$!
echo "Pop B launched: PID=$PID_B"

echo ""
echo "=== Both populations running ==="
echo "Monitor: PYTHONPATH=. python tools/status.py --run '$POP_A_PREFIX@$POP_A_DB:A' --run '$POP_B_PREFIX@$POP_B_DB:B'"
echo "Logs:    tail -f $LOG_DIR/pop_a.log  |  tail -f $LOG_DIR/pop_b.log"
echo "Kill:    kill $PID_A $PID_B"
