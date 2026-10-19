#!/usr/bin/env bash
# Full reproduction of M1-M4. Expect several minutes (M3/M4 baselines dominate).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}

echo "== M1: assumptions and lemmas =="
"$PY" repro/m1_theory/lemma_checks.py

echo "== M2: independent Main vs official =="
"$PY" repro/m2_main_complexity/compare_with_official.py --steps 20

echo "== M2: complexity sweeps =="
"$PY" repro/m2_main_complexity/complexity_sweep.py

echo "== M3: Figure 1 =="
"$PY" repro/m3_figure1/reproduce_figure1.py

echo "== M3: r ablation =="
"$PY" repro/m3_figure1/ablation_r.py

echo "== M4: Figure 2 =="
"$PY" repro/m4_vfl_consensus/reproduce_figure2.py

echo "== M4: consensus =="
"$PY" repro/m4_vfl_consensus/consensus_extra.py

echo "== M4: labels-distributed =="
"$PY" repro/m4_vfl_consensus/labels_distributed.py

echo
echo "Full reproduction finished. Outputs are under repro/outputs/."
