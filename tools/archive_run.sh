#!/usr/bin/env bash
# Archive all data from a completed evolution run.
#
# Exports Redis data to local files, then uploads them as a GitHub Release asset.
# Run BEFORE flushing Redis and BEFORE rebooting the machine.
#
# Usage:
#   bash tools/archive_run.sh --exp <experiment-name> --run <prefix@db:label> [--upload]
#
# Examples:
#   # Dry run: export locally, no upload
#   bash tools/archive_run.sh --exp hotpotqa/push --run "chains/hotpotqa/static_f1_600@10:C"
#
#   # Export and upload to GitHub Release
#   bash tools/archive_run.sh --exp hotpotqa/push --run "chains/hotpotqa/static_f1_600@10:C" --upload
#
#   # Archive all 4 runs and upload
#   for SPEC in "chains/hotpotqa/static_f1@8:A" "chains/hotpotqa/static@9:B" \
#               "chains/hotpotqa/static_f1_600@10:C" "chains/hotpotqa/static_f1_600@11:D"; do
#     bash tools/archive_run.sh --exp hotpotqa/push --run "$SPEC" --upload
#   done
#
# GitHub Release naming: exp/<experiment-name>
# Assets: <label>_archive.tar.gz, environment.txt (once per experiment)

set -euo pipefail

PYTHON=${GIGAEVO_PYTHON:-$(command -v python3)}
GH=$(command -v gh)
PROJ="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# ── Argument parsing ────────────────────────────────────────────────────────
EXPERIMENT=""
RUN_SPEC=""
UPLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)      EXPERIMENT="$2"; shift 2 ;;
        --run)      RUN_SPEC="$2";   shift 2 ;;
        --upload)   UPLOAD=true;     shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT" || -z "$RUN_SPEC" ]]; then
    echo "Usage: $0 --exp <experiment-name> --run <prefix@db:label> [--upload]"
    exit 1
fi

# ── Parse run spec ──────────────────────────────────────────────────────────
# Format: prefix@db:label  (e.g. chains/hotpotqa/static@0:K)
AT_IDX=$(echo "$RUN_SPEC" | grep -bo '@' | tail -1 | cut -d: -f1)
PREFIX="${RUN_SPEC:0:$AT_IDX}"
REST="${RUN_SPEC:$((AT_IDX+1))}"
DB="${REST%%:*}"
LABEL="${REST##*:}"

echo "=== Archiving run ==="
echo "  Experiment : $EXPERIMENT"
echo "  Run spec   : $RUN_SPEC"
echo "  Prefix     : $PREFIX"
echo "  Redis DB   : $DB"
echo "  Label      : $LABEL"
echo "  Upload     : $UPLOAD"
echo

# ── Output paths ─────────────────────────────────────────────────────────────
ARCHIVE_BASE="$PROJ/experiments/$EXPERIMENT/archives"
ARCHIVE_DIR="$ARCHIVE_BASE/$LABEL"
mkdir -p "$ARCHIVE_DIR/programs"

# ── Step 1: Export full evolution data ───────────────────────────────────────
echo "[1/5] Exporting full evolution history to CSV..."
PYTHONPATH="$PROJ" "$PYTHON" "$PROJ/tools/redis2pd.py" \
    --redis-db "$DB" \
    --redis-prefix "$PREFIX" \
    --output-file "$ARCHIVE_DIR/evolution_data.csv"
ROWS=$(tail -n +2 "$ARCHIVE_DIR/evolution_data.csv" | wc -l)
echo "      → $ROWS program records exported"

# ── Step 2: Save all programs with code ─────────────────────────────────────
echo "[2/5] Saving all programs with source code..."
PYTHONPATH="$PROJ" "$PYTHON" "$PROJ/tools/top_programs.py" \
    --run "$RUN_SPEC" \
    -n 9999 \
    --save-dir "$ARCHIVE_DIR/programs"
N_PROGRAMS=$(find "$ARCHIVE_DIR/programs" -name "*.py" | wc -l)
echo "      → $N_PROGRAMS program files saved"

# ── Step 3: Save top-50 as JSON ─────────────────────────────────────────────
echo "[3/5] Saving top-50 programs as JSON..."
PYTHONPATH="$PROJ" "$PYTHON" "$PROJ/tools/top_programs.py" \
    --run "$RUN_SPEC" \
    -n 50 \
    --json \
    > "$ARCHIVE_DIR/top50.json"
echo "      → top50.json written"

# ── Step 3b: Copy dry-run output (contains full resolved config) ─────────────
DRY_RUN_OUT="$ARCHIVE_BASE/$LABEL/dry_run_output.txt"
if [[ -f "$DRY_RUN_OUT" ]]; then
    echo "[3b/5] dry_run_output.txt already present — skipping"
else
    # Best-effort: find the most recent Hydra config for this experiment
    LATEST_HYDRA=$(find "$PROJ/outputs" -name "config.yaml" -path "*/.hydra/*" 2>/dev/null \
        | xargs ls -t 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_HYDRA" ]]; then
        cp "$LATEST_HYDRA" "$ARCHIVE_DIR/resolved_hydra_config.yaml"
        echo "[3b/5] Copied Hydra resolved config: $LATEST_HYDRA"
    else
        echo "[3b/5] WARN: No Hydra output found. Ensure dry_run_output.txt was saved at launch."
        echo "       Run: PYTHONPATH=. python run.py <params> dry_run=true > $DRY_RUN_OUT"
    fi
fi

# ── Step 4: Environment snapshot (once per experiment, not per run) ──────────
ENV_FILE="$ARCHIVE_BASE/environment.txt"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[4/5] Recording environment..."
    {
        echo "=== Python version ==="
        "$PYTHON" --version
        echo ""
        echo "=== pip freeze ==="
        "$PYTHON" -m pip freeze
        echo ""
        echo "=== System ==="
        uname -a
        echo ""
        echo "=== GPU ==="
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null \
            || echo "(no nvidia-smi or no GPU)"
        echo ""
        echo "=== Captured at ==="
        date -u
    } > "$ENV_FILE"
    echo "      → environment.txt written"
else
    echo "[4/5] Skipping environment.txt (already exists for this experiment)"
fi

# ── Step 5: Create tarball ───────────────────────────────────────────────────
echo "[5/5] Creating tarball..."
TARBALL="$ARCHIVE_BASE/${LABEL}_archive.tar.gz"
tar -czf "$TARBALL" -C "$ARCHIVE_BASE" "$LABEL"
SIZE=$(du -sh "$TARBALL" | cut -f1)
echo "      → $TARBALL ($SIZE)"

echo
echo "=== Local archive complete ==="
echo "  CSV rows   : $ROWS"
echo "  Programs   : $N_PROGRAMS"
echo "  Tarball    : $TARBALL ($SIZE)"
echo

# ── Upload to GitHub Release ─────────────────────────────────────────────────
if [[ "$UPLOAD" == "true" ]]; then
    RELEASE_TAG="exp/${EXPERIMENT}"
    RELEASE_TITLE="Archives: $EXPERIMENT"

    echo "=== Uploading to GitHub Release: $RELEASE_TAG ==="

    # Create release if it doesn't exist yet (idempotent — safe to run in parallel)
    if ! "$GH" release view "$RELEASE_TAG" &>/dev/null 2>&1; then
        echo "  Creating release $RELEASE_TAG..."
        "$GH" release create "$RELEASE_TAG" \
            --title "$RELEASE_TITLE" \
            --notes "Evolution run archives for experiment: $EXPERIMENT

Uploaded by tools/archive_run.sh. Each .tar.gz contains:
- evolution_data.csv (all programs, all generations, all metrics)
- programs/*.py (source code of all evaluated programs)
- top50.json (top 50 programs with full metadata)
" \
            --prerelease || echo "  Release already created by parallel run — continuing"
    else
        echo "  Release $RELEASE_TAG already exists — uploading asset to it"
    fi

    # Upload the run tarball
    echo "  Uploading ${LABEL}_archive.tar.gz..."
    "$GH" release upload "$RELEASE_TAG" "$TARBALL" --clobber

    # Upload environment.txt (once — clobber is idempotent)
    echo "  Uploading environment.txt..."
    "$GH" release upload "$RELEASE_TAG" "$ENV_FILE" --clobber

    RELEASE_URL=$("$GH" release view "$RELEASE_TAG" --json url -q .url)
    echo
    echo "=== Upload complete ==="
    echo "  Release URL: $RELEASE_URL"
    echo
    echo "  Add to PR description or comment:"
    echo "  > Archives: $RELEASE_URL"
fi

echo "Done. Verify before flushing Redis or rebooting."
