#!/usr/bin/env bash
# Reproduces every number, table and figure of the paper (about 12-15 minutes on a 2-core CPU).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d mlran ]; then
  git clone https://github.com/faithfulco/mlran.git
  git -C mlran checkout 273fcac5af82ad60d60fa2bff427ebcb1db61273   # dataset version used in the paper
fi
"${PYTHON:-python3}" src/qsim.py                                                  # consistency checks against Qiskit / Qiskit Aer
"${PYTHON:-python3}" src/build_features.py mlran/6_experiments/FS_MLRan_Datasets --out data/mlran_study.csv
"${PYTHON:-python3}" src/run_experiments.py data/mlran_study.csv --out results
"${PYTHON:-python3}" src/circuit_analysis.py data/mlran_study.csv --out results   # Eq. 7 entropy, Eq. 8 QFI, error table
echo "Done. Tables: results/tables/  Figures: results/figs/  Per-seed values: results/per_seed.csv"
