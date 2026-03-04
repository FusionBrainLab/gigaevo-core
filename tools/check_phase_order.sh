#!/usr/bin/env bash
# Check that all required protocol phase documents exist and are in the correct state
# before proceeding to launch. Run from the repo root.
#
# Usage:
#   bash tools/check_phase_order.sh <experiment-name>
#
# Example:
#   bash tools/check_phase_order.sh hotpotqa_p3_crossover
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed

set -euo pipefail

EXPERIMENT=${1:?Usage: $0 <experiment-name>}
PROJ="$(git rev-parse --show-toplevel)"
EXP_DIR="$PROJ/experiments/$EXPERIMENT"
errors=0

ok()   { echo "[OK]   $1"; }
fail() { echo "[FAIL] $1"; errors=$((errors + 1)); }
warn() { echo "[WARN] $1"; }

echo "=== Phase gate check: $EXPERIMENT ==="
echo

# ── Phase 1: design doc ────────────────────────────────────────────────────
if [[ -f "$EXP_DIR/01_design.md" ]]; then
    ok "01_design.md exists"
else
    fail "01_design.md missing — run Phase 1 (ml-research-methodologist)"
fi

# ── Phase 2: review doc with APPROVED verdict ──────────────────────────────
if [[ -f "$EXP_DIR/02_review.md" ]]; then
    if grep -qi "## Overall.*APPROVED\|verdict.*APPROVED\|APPROVED" "$EXP_DIR/02_review.md"; then
        ok "02_review.md exists and contains APPROVED"
    else
        fail "02_review.md exists but does not contain APPROVED verdict"
    fi
else
    fail "02_review.md missing — run Phase 2 (reviewer-2-adversary)"
fi

# ── Phase 3: pre-registration committed to git ────────────────────────────
if [[ -f "$EXP_DIR/03_plan.md" ]]; then
    REL_PATH="experiments/$EXPERIMENT/03_plan.md"
    COMMIT=$(git -C "$PROJ" log --oneline -1 -- "$REL_PATH" 2>/dev/null || true)
    if [[ -n "$COMMIT" ]]; then
        ok "03_plan.md committed: $COMMIT"
    else
        fail "03_plan.md exists but is NOT committed to git (must commit before code changes)"
    fi

    # Check for placeholder hash not yet filled in
    if grep -q '<hash>' "$EXP_DIR/03_plan.md"; then
        warn "03_plan.md still contains placeholder <hash> — fill in commit hash"
    fi
else
    fail "03_plan.md missing — complete Phase 3 pre-registration before launch"
fi

# ── Evaluation script hash (if script exists) ─────────────────────────────
EVAL_SCRIPT="$EXP_DIR/run_test_eval.sh"
if [[ -f "$EVAL_SCRIPT" ]]; then
    ACTUAL_HASH=$(sha256sum "$EVAL_SCRIPT" | awk '{print $1}')
    if grep -q "$ACTUAL_HASH" "$EXP_DIR/03_plan.md" 2>/dev/null; then
        ok "run_test_eval.sh hash matches 03_plan.md ($ACTUAL_HASH)"
    elif grep -qi "N/A" "$EXP_DIR/03_plan.md" 2>/dev/null && \
         grep -qi "run_test_eval" "$EXP_DIR/03_plan.md" 2>/dev/null; then
        warn "run_test_eval.sh exists but 03_plan.md says N/A — verify this is intentional"
    else
        fail "run_test_eval.sh hash NOT pinned in 03_plan.md — add: sha256: $ACTUAL_HASH"
    fi
else
    if grep -qi "N/A" "$EXP_DIR/03_plan.md" 2>/dev/null; then
        ok "No run_test_eval.sh (N/A noted in 03_plan.md)"
    else
        warn "No run_test_eval.sh — if this experiment has a test split, implement it first"
    fi
fi

echo
if [[ $errors -eq 0 ]]; then
    echo "=== All phase gate checks passed. Proceed to launch. ==="
    exit 0
else
    echo "=== $errors check(s) failed — resolve before proceeding to launch. ==="
    exit 1
fi
