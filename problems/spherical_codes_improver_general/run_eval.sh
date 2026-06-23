#!/usr/bin/env bash
# Reproduce the full90 re-blast evaluation (champion vs E7 vs E8) at the
# calibrated protocol P* (R unbounded, M=10, sigma 1->1e-6, fresh=5, dry=0),
# best-of-3-seeds, 3540s wall per config, on the local 192-core box.
#
# Produces (in this dir):
#   reblast_raw/shard_0of1.json          raw per-(program,config,seed) results
#   squeeze_test/squeeze_{champion,E7,E8}.json + packings_champion.npz
#   configs_raw/{cohn,champion,E7,E8}.npz + index.json   (raw point sets)
#   report/table_full90.tex + figures + report.pdf
set -euo pipefail

PY=python3
TECTONIC=/home/user/conda/bin/tectonic
cd "$(dirname "$0")"

# spherical grader is single-threaded per worker; never let BLAS oversubscribe
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

HEADROOM=/tmp/headroom61.json   # {"headroom":[[d,n],...61], "dead":[[d,n],...29]}

# 1. grade every movable config (3 programs x 61 headroom configs x 3 seeds = 549 tasks)
nohup "$PY" reblast.py run \
    --headroom "$HEADROOM" \
    --wall 3540 --seeds 0,1,2 --workers 185 \
    --restarts 100000 --b-steps 10 --hi 1.0 --lo 1e-6 \
    --fresh-every 5 --dry-patience 0 \
    --shard 0/1 > reblast_run.log 2>&1 &
REBLAST_PID=$!
echo "reblast running PID $REBLAST_PID -> reblast_run.log"
wait "$REBLAST_PID"

# 2. best-of-3-seeds merge (+ carry the 29 proven-optimal configs at Cohn)
"$PY" reblast.py merge --headroom "$HEADROOM" --out-dir squeeze_test

# 3. raw per-(n,d) point configurations for downstream UMAP (done separately)
"$PY" dump_configs.py

# 4. partial-bold full90 table + stats/tail figures, then report PDF
"$PY" make_report.py \
    --results-dir squeeze_test \
    --packings squeeze_test/packings_champion.npz \
    --out-dir report
"$PY" build_report_pdf.py
"$TECTONIC" report/report.tex
echo "done -> report/report.pdf"
