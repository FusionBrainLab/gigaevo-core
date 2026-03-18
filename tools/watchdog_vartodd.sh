#!/usr/bin/env bash
# Watchdog for vartodd circuit evolution experiment
# Usage: bash tools/watchdog_vartodd.sh
# Runs once and prints status. Use with: watch -n 60 bash tools/watchdog_vartodd.sh

set -euo pipefail

REDIS_DB=0
PREFIX="vartodd"
PID=942353
LOG="/tmp/vartodd_evolution.log"
PYTHON="~/envs/evo_fast/bin/python"

echo "=== Vartodd Watchdog ($(date '+%Y-%m-%d %H:%M:%S')) ==="
echo ""

# 1. Process alive?
if kill -0 "$PID" 2>/dev/null; then
    echo "Process:  ALIVE (PID $PID)"
    UPTIME=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
    if [ -n "$UPTIME" ]; then
        HOURS=$((UPTIME / 3600))
        MINS=$(( (UPTIME % 3600) / 60 ))
        echo "Uptime:   ${HOURS}h ${MINS}m"
    fi
else
    echo "Process:  DEAD (PID $PID not found)"
    echo ""
    echo "Last 10 log lines:"
    tail -10 "$LOG" 2>/dev/null || echo "  (log not found)"
    exit 1
fi

# 2. Key count
KEY_COUNT=$(redis-cli -n "$REDIS_DB" dbsize 2>/dev/null | awk '{print $NF}')
echo "Redis keys: $KEY_COUNT"

# 3. Program count & fitness (programs stored as JSON strings)
PROG_COUNT=$(redis-cli -n "$REDIS_DB" keys "${PREFIX}:program:*" 2>/dev/null | wc -l)
echo "Programs:   $PROG_COUNT"
echo ""
echo "--- Program Status ---"
for key in $(redis-cli -n "$REDIS_DB" keys "${PREFIX}:program:*" 2>/dev/null | sort); do
    redis-cli -n "$REDIS_DB" get "$key" 2>/dev/null | $PYTHON -c "
import sys, json
d = json.load(sys.stdin)
pid = d.get('id','?')[:8]
state = d.get('state','?')
m = d.get('metrics', {})
fitness = m.get('fitness', '-')
valid = m.get('is_valid', '-')
print(f'  {pid}  state={state:<10}  fitness={fitness}  valid={valid}')
" 2>/dev/null || echo "  (parse error for $key)"
done

# 4. Optuna stage stats
echo ""
echo "--- Optuna Stats ---"
OPTUNA_OK=$(redis-cli -n "$REDIS_DB" get "${PREFIX}:metrics:history:dag_runner:dag:internals:OptunaOptStage:stage_success" 2>/dev/null | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d[-1][1] if d else 0)" 2>/dev/null || echo "?")
OPTUNA_FAIL=$(redis-cli -n "$REDIS_DB" get "${PREFIX}:metrics:history:dag_runner:dag:internals:OptunaOptStage:stage_failure" 2>/dev/null | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d[-1][1] if d else 0)" 2>/dev/null || echo "?")
echo "  Optuna success: $OPTUNA_OK  failure: $OPTUNA_FAIL"

# 5. Last log activity
if [ -f "$LOG" ]; then
    LAST_MOD=$(stat -c %Y "$LOG" 2>/dev/null || echo "0")
    NOW=$(date +%s)
    STALE=$((NOW - LAST_MOD))
    if [ "$STALE" -gt 300 ]; then
        echo ""
        echo "Log stale:  ${STALE}s since last write (WARNING)"
    else
        echo ""
        echo "Log fresh:  ${STALE}s since last write"
    fi
    echo ""
    echo "--- Last 5 log lines ---"
    tail -5 "$LOG"
fi
