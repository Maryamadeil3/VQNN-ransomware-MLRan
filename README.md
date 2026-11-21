# Code and results: variational quantum neural network framework for ransomware detection (MLRan)

## Data

MLRan (Onwuegbuche et al., *Journal of Network and Computer Applications*, 2026, doi:10.1016/j.jnca.2026.104475), behavioural release: `run_all.sh` clones https://github.com/faithfulco/mlran at commit `273fcac` (the version used in the paper). No malware binaries are downloaded or distributed.

`data/mlran_study.csv.gz` is the derived table used in the paper: one row per unique binary (4,876 after removing 4 duplicated SHA-256 entries) with SHA-256, split, label, family, first-seen year, the eight behavioural group counts and the 483 binary indicators. The scripts read the `.gz` file directly; `run_all.sh` rebuilds the uncompressed `data/mlran_study.csv` from MLRan.

## Environment

Python 3.13; numpy, scipy, pandas, scikit-learn, xgboost, matplotlib, qiskit, qiskit-aer (exact versions in `requirements.txt`). CPU only.

```
cd Code_and_Results_Package
pip install -r requirements.txt
bash run_all.sh
```

## Run order

`run_all.sh` runs the following steps (about 12-15 minutes on a 2-core CPU):

1. `python src/qsim.py` -> consistency checks of the state-vector and density-matrix simulators against Qiskit and Qiskit Aer.
2. `python src/build_features.py mlran/6_experiments/FS_MLRan_Datasets --out data/mlran_study.csv` -> derived feature table.
3. `python src/run_experiments.py data/mlran_study.csv --out results` -> VQNN and classical baselines over 10 seeds, statistics, ablations (optimiser budget, training-set size, circuit components, input encoding, sample age, hardware-like noise); writes `results/per_seed.csv`, `results/summary.json`, `results/tables/*.tex`, `results/figs/*.pdf`.
4. `python src/circuit_analysis.py data/mlran_study.csv --out results` -> entanglement entropy (Eq. 7), qubit contribution (Eq. 8) and error analysis; retrains the VQNN for every seed and checks that the reported MCC is reproduced exactly.

## Contents

```
src/        simulator (qsim.py), feature construction, experiments, circuit analysis
data/       derived MLRan feature table (compressed)
results/    outputs of the run reported in the paper (per_seed.csv holds every value)
figures/    figures of the manuscript
```

## Notes

- Seeds: 0, 42, 100, 123, 200, 256, 512, 1000, 2024, 9999.
- Fixed MLRan test set (975 samples first seen 2021-2024); per seed, the training pool is split 80/20 into training (3,120) and validation (781). Feature selection (mutual information, top 4) and thresholds (maximum validation MCC) use the training and validation parts only.
- VQNN: 4 qubits, angle encoding RY(pi x), two layers of RY/RZ rotations with a CNOT ring (16 parameters), readout of <Z> on qubit 0, binary cross-entropy, COBYLA (400 evaluations), exact simulation.
- The simulator is validated against Qiskit (exact) and Qiskit Aer (noisy, 1024 shots).
- Statistics: Wilcoxon signed-rank tests with Holm correction, effect sizes d_z, 95% t-intervals, exact McNemar tests.

## License

MIT (see `LICENSE`).
