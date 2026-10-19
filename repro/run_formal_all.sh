#!/usr/bin/env bash
# Formal (non-smoke) reproduction of M1-M4, ordered so the long
# Figure 2 DPMM run is last. All output is captured by the caller.
set -euo pipefail
cd "$(dirname "$0")/.."

export TQDM_DISABLE=1
PY=${PYTHON:-python}

run() {
    echo "============================================================"
    echo "[$(date '+%F %T')] START: $*"
    "$@"
    echo "[$(date '+%F %T')] DONE: $*"
}

run "$PY" repro/m1_theory/lemma_checks.py
run "$PY" repro/m2_main_complexity/compare_with_official.py --steps 20
run "$PY" repro/m2_main_complexity/complexity_sweep.py
run "$PY" repro/m3_figure1/reproduce_figure1.py
run "$PY" repro/m3_figure1/ablation_r.py
run "$PY" repro/m4_vfl_consensus/consensus_extra.py
run "$PY" repro/m4_vfl_consensus/labels_distributed.py
run "$PY" repro/m4_vfl_consensus/reproduce_figure2.py

echo "============================================================"
echo "[$(date '+%F %T')] ALL FORMAL RUNS FINISHED"
