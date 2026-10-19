#!/usr/bin/env bash
# Fast end-to-end check of all four reproduction modules (~1-2 minutes).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}

echo "== M1: assumptions and lemmas =="
"$PY" repro/m1_theory/lemma_checks.py --trials 10

echo "== M2: independent Main vs official =="
"$PY" repro/m2_main_complexity/compare_with_official.py --steps 10

echo "== M2: one small complexity sweep =="
"$PY" repro/m2_main_complexity/complexity_sweep.py --families kappa_f --max-steps 1200

echo "== M3: Figure 1 smoke =="
"$PY" repro/m3_figure1/reproduce_figure1.py --smoke

echo "== M3: r ablation smoke =="
"$PY" repro/m3_figure1/ablation_r.py --smoke --r-values 1e-3 1e-1

echo "== M4: Figure 2 smoke =="
"$PY" repro/m4_vfl_consensus/reproduce_figure2.py --smoke

echo "== M4: consensus smoke =="
"$PY" repro/m4_vfl_consensus/consensus_extra.py --smoke

echo "== M4: labels-distributed smoke =="
"$PY" repro/m4_vfl_consensus/labels_distributed.py --smoke

echo
echo "All smoke tests finished. Outputs are under repro/outputs/."
