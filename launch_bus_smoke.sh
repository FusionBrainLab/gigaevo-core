#!/usr/bin/env bash
# Smoke test: 1 meta-prompt run (gemini 3.1 pro) + 3 bus-connected runs (gemini 3 flash)
# All runs: vartodd problem, mcts_evo pipeline, 3 generations
#
# Redis DB layout:
#   DB 7  — meta-prompt run (standalone, no bus)
#   DB 1  — bus run A
#   DB 2  — bus run B
#   DB 3  — bus run C
#   DB 15 — bus transport (shared stream + claims)

set -euo pipefail

PYTHON=~/envs/evo_fast/bin/python
DEPS_DIR="~/envs/evo_fast/lib/vartodd_deps"
CONDA_LIB="~/envs/evo_fast/lib"

export LD_LIBRARY_PATH="${DEPS_DIR}:${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${DEPS_DIR}/libglibc_shim.so"
export PYTHONPATH="problems/vartodd:.:${PYTHONPATH:-}"
export GIGAEVO_PYTHON="${PYTHON}"

# Load API key from .env
if [ -f .env ]; then
    export $(grep -E '^OPENAI_API_KEY=' .env | head -1)
fi

LOGDIR=/tmp/bus_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

echo "=== Bus Smoke Test ==="
echo "Logs: $LOGDIR"
echo ""

# --- Run P1: prompt evolution (gemini 3.1 pro, no bus) ---
# Reads fitness stats from the 3 bus runs, evolves mutation prompts.
MAIN_RUN_SOURCES="[{db:1,prefix:vartodd},{db:2,prefix:vartodd},{db:3,prefix:vartodd}]"
echo "[Run P1] prompt-evo  | DB=7  | gemini-3.1-pro | no bus | sources=db1+db2+db3"
nohup $PYTHON run.py \
    problem.name=prompt_evolution \
    pipeline=prompt_evolution_multi \
    llm=gemini31_pro \
    redis.db=7 \
    "+main_run_sources=$MAIN_RUN_SOURCES" \
    max_mutations_per_generation=3 \
    max_elites_per_generation=3 \
    max_generations=3 \
    num_parents=1 \
    > "$LOGDIR/run_P1_db7.log" 2>&1 &
PID0=$!

# --- Runs 1-3: bus-connected (gemini 3 flash) ---
for i in 1 2 3; do
    echo "[Run $i] bus worker   | DB=$i  | gemini-3-flash | migration_bus=bus"
    nohup $PYTHON run.py \
        problem.name=vartodd \
        pipeline=mcts_evo \
        llm=gemini3_flash \
        migration_bus=bus \
        redis.db=$i \
        max_generations=3 \
        optimization_time_budget=600 \
        dag_timeout=1200 \
        stage_timeout=600 \
        > "$LOGDIR/run${i}_bus_db${i}.log" 2>&1 &
    eval "PID${i}=$!"
done

echo ""
echo "PIDs: P1=$PID0  bus1=$PID1  bus2=$PID2  bus3=$PID3"
echo "$PID0 $PID1 $PID2 $PID3" > "$LOGDIR/pids.txt"
echo ""
echo "Monitor:"
echo "  tail -f $LOGDIR/run_P1_db7.log"
echo "  tail -f $LOGDIR/run1_bus_db1.log"
echo "  tail -f $LOGDIR/run2_bus_db2.log"
echo "  tail -f $LOGDIR/run3_bus_db3.log"
echo ""
echo "Bus stream:"
echo "  redis-cli -n 15 XLEN 'gigaevo:vartodd:migration_bus'"
echo "  redis-cli -n 15 KEYS 'gigaevo:vartodd:migration_bus:claim:*' | wc -l"
echo ""
echo "Check MigrationBus activity:"
echo "  grep MigrationBus $LOGDIR/run*_bus_*.log"
