#!/usr/bin/env bash
# Verify that an experiment has completed all lifecycle phases before PR merge.
# Complements check_phase_order.sh (which gates Phase 4 entry).
# This script gates Phase 5 exit / PR merge.
#
# Usage:
#   bash tools/check_experiment_complete.sh <experiment-name>
#
# Example:
#   bash tools/check_experiment_complete.sh hotpotqa/nlp_prompts
#
# Exit codes:
#   0 — all checks passed; safe to merge
#   1 — one or more checks failed; do not merge

set -euo pipefail

EXPERIMENT=${1:?Usage: $0 <experiment-name>}
PROJ="$(git rev-parse --show-toplevel)"
EXP_DIR="$PROJ/experiments/$EXPERIMENT"
GH=$(command -v gh)
errors=0
warnings=0

ok()   { echo "[OK]   $1"; }
fail() { echo "[FAIL] $1"; errors=$((errors + 1)); }
warn() { echo "[WARN] $1"; warnings=$((warnings + 1)); }

echo "=== Experiment lifecycle check: $EXPERIMENT ==="
echo

# ── Phase 1–3: design, review, pre-registration ──────────────────────────────
# Experiments may store docs in experiments/<name>/ (template format) OR
# in docs/plans/ (older pre-template format). Accept either.
DOCS_IN_EXP=true
for doc in 01_design.md 02_review.md 03_plan.md; do
    if [[ ! -f "$EXP_DIR/$doc" ]]; then
        DOCS_IN_EXP=false
        break
    fi
done

if [[ "$DOCS_IN_EXP" == "true" ]]; then
    for doc in 01_design.md 02_review.md 03_plan.md; do
        REL="experiments/$EXPERIMENT/$doc"
        if git -C "$PROJ" log --oneline --max-count=1 -- "$REL" | grep -q .; then
            ok "$doc committed"
        else
            fail "$doc exists but not committed to git"
        fi
    done
else
    # Check for docs/plans/ alternative (pre-template experiments)
    # Try exact name, hyphenated variant, and keyword suffix (drop leading task prefix).
    # e.g. hotpotqa_nlp_prompts → hotpotqa-nlp-prompts → nlp-prompts → nlp_prompts
    EXP_HYPHEN="$(echo "$EXPERIMENT" | tr '_' '-')"
    EXP_SUFFIX="$(echo "$EXPERIMENT" | sed 's/^[^_]*_//')"   # drop first component
    EXP_SUFFIX_H="$(echo "$EXP_SUFFIX" | tr '_' '-')"
    PLANS_MATCH=$(git -C "$PROJ" ls-files "docs/plans/" \
        | grep -iE "$EXPERIMENT|$EXP_HYPHEN|$EXP_SUFFIX|$EXP_SUFFIX_H" || true)
    if [[ -n "$PLANS_MATCH" ]]; then
        ok "Phase 1–3 docs found in docs/plans/ (pre-template format): $PLANS_MATCH"
    else
        # Final fallback: any committed file in docs/plans/ referencing this experiment
        PLANS_ANY=$(git -C "$PROJ" ls-files "docs/plans/" | head -5 || true)
        warn "01/02/03_plan.md not in experiments/$EXPERIMENT/ — check docs/plans/ manually (found: $PLANS_ANY)"
    fi
fi

# Review verdict — check both locations
REVIEW_FILE=""
[[ -f "$EXP_DIR/02_review.md" ]] && REVIEW_FILE="$EXP_DIR/02_review.md"
if [[ -n "$REVIEW_FILE" ]]; then
    if grep -q '\[x\] APPROVED\|Verdict: APPROVED\|Overall Verdict: APPROVED\|## Overall.*: APPROVED' "$REVIEW_FILE"; then
        ok "02_review.md contains APPROVED verdict"
    else
        fail "02_review.md does not contain a clear APPROVED verdict"
    fi
fi

echo

# ── Phase 4: archives uploaded ────────────────────────────────────────────────
RELEASE_TAG="exp/${EXPERIMENT}"
if "$GH" release view "$RELEASE_TAG" &>/dev/null 2>&1; then
    ASSET_COUNT=$("$GH" release view "$RELEASE_TAG" --json assets -q '.assets | length')
    if [[ "$ASSET_COUNT" -gt 0 ]]; then
        ok "GitHub Release $RELEASE_TAG exists with $ASSET_COUNT asset(s)"
    else
        fail "GitHub Release $RELEASE_TAG exists but has NO assets — run archive_run.sh --upload"
    fi
else
    fail "GitHub Release $RELEASE_TAG not found — run archive_run.sh --upload for all runs"
fi

# environment_freeze.txt committed
ENV_FILE="experiments/$EXPERIMENT/environment_freeze.txt"
if git -C "$PROJ" log --oneline --max-count=1 -- "$ENV_FILE" | grep -q .; then
    ok "environment_freeze.txt committed"
else
    fail "environment_freeze.txt not committed (required for reproducibility)"
fi

echo

# ── Phase 5: results written ──────────────────────────────────────────────────
if [[ -f "$EXP_DIR/05_results.md" ]]; then
    REL="experiments/$EXPERIMENT/05_results.md"
    if git -C "$PROJ" log --oneline --max-count=1 -- "$REL" | grep -q .; then
        ok "05_results.md committed"
    else
        fail "05_results.md exists but not committed"
    fi

    # Check for unfilled placeholders
    if grep -q '_(pending)_\|TODO\|<fill' "$EXP_DIR/05_results.md"; then
        warn "05_results.md still contains placeholders — verify it is fully complete"
    fi

    # Check deviations section exists
    if grep -q "Deviations from Pre-Registration" "$EXP_DIR/05_results.md"; then
        ok "Deviations from Pre-Registration section present"
    else
        fail "Deviations from Pre-Registration section missing from 05_results.md"
    fi
else
    fail "05_results.md missing — Phase 5 not complete"
fi

# INDEX.md updated
if grep -q "$EXPERIMENT" "$PROJ/experiments/INDEX.md" 2>/dev/null; then
    if grep -q "✅" "$PROJ/experiments/INDEX.md"; then
        ok "experiments/INDEX.md contains entry for $EXPERIMENT"
    else
        warn "experiments/INDEX.md has entry for $EXPERIMENT but status may not be ✅ Complete"
    fi
else
    fail "experiments/INDEX.md has no entry for $EXPERIMENT — add before merging"
fi

echo

# ── PR description ────────────────────────────────────────────────────────────
if [[ -f "$EXP_DIR/PR_DESCRIPTION.md" ]]; then
    if grep -q '🟢\|Complete' "$EXP_DIR/PR_DESCRIPTION.md"; then
        ok "PR_DESCRIPTION.md shows Complete status"
    else
        warn "PR_DESCRIPTION.md does not show 🟢 Complete — update before merging"
    fi
    if grep -q 'Archives.*pending\|GitHub Release link' "$EXP_DIR/PR_DESCRIPTION.md"; then
        warn "PR_DESCRIPTION.md still has placeholder Archives link — add release URL"
    fi
fi

echo
if [[ $errors -eq 0 && $warnings -eq 0 ]]; then
    echo "=== All checks passed. Safe to merge. ==="
    exit 0
elif [[ $errors -eq 0 ]]; then
    echo "=== $warnings warning(s). Review before merging, but not blocking. ==="
    exit 0
else
    echo "=== $errors failure(s), $warnings warning(s). Resolve failures before merging. ==="
    exit 1
fi
