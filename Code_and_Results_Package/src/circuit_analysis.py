#!/usr/bin/env python3
"""Step 3 - analyses of the trained circuits that the paper reports with Eqs. (7) and (8).

    python src/circuit_analysis.py data/mlran_study.csv --out results

For every seed the four-qubit VQNN is retrained with exactly the same splits, feature selection, scaling,
initialisation and budget as in run_experiments.py (COBYLA is deterministic), and the script checks that the
retrained circuit reproduces the reported test MCC. It then computes, on TRAINING data unless stated otherwise:

  * Quantum Feature Importance, Eq. (8):  QFI_i = E_x[<Z_i>^2] / sum_j E_x[<Z_j>^2]
    with <Z_i> the Pauli-Z expectation of qubit i (which encodes feature i) in the final state |psi(x)>.
    It is descriptive only: it can be computed only for the four encoded features and is NOT used to select them.
  * Random-forest impurity importance rank of all eight features (training part).
  * Entanglement entropy, Eq. (7):  S = - sum_k lambda_k^2 log2 lambda_k^2  (lambda_k: Schmidt coefficients)
    of the final state for the cuts {q0 | q1 q2 q3} and {q0 q1 | q2 q3}, on the TEST samples, for the trained
    circuit and for the same circuit at its random initial parameters.
  * Error summary (FP, FN, FPR, FNR) of every four-feature model, from results/per_seed.csv.
Writes results/tables/{qfi,entropy,errors,timeacc}.tex and results/circuit_analysis.json; appends macros to numbers.tex.
"""
import argparse, json, os, sys, re
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qsim
from run_experiments import SEEDS, FEATS, BUDGETS, train_vqnn, best_threshold, pm

def final_states(X, params, L=2):
    """(N, 2**n) final state vectors of the trained circuit."""
    n = X.shape[1]
    return qsim.encode(X) @ qsim.unitary(params, n, L).T

def z_expect(psi, n):
    pr = np.abs(psi) ** 2; out = []
    for q in range(n):
        bit = (np.arange(2 ** n) >> (n - 1 - q)) & 1             # qubit 0 = most significant index
        out.append(pr[:, bit == 0].sum(1) - pr[:, bit == 1].sum(1))
    return np.stack(out, 1)

def entropy(psi, n, k):
    """entanglement entropy (bits) between the first k qubits and the rest, from Schmidt coefficients."""
    M = psi.reshape(len(psi), 2 ** k, 2 ** (n - k))
    lam = np.linalg.svd(M, compute_uv=False)                     # Schmidt coefficients lambda_k
    p = lam ** 2; p = np.where(p > 1e-15, p, 1)                  # 0 log 0 = 0
    return -(p * np.log2(p)).sum(1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--out", default="results")
    ap.add_argument("--val-frac", type=float, default=0.2); a = ap.parse_args()
    df = pd.read_csv(a.data); R = pd.read_csv(f"{a.out}/per_seed.csv")
    pool = df[df.split == "train"].reset_index(drop=True); test = df[df.split == "test"].reset_index(drop=True)
    G, Gte, yte = pool[FEATS].values, test[FEATS].values, test.label.values
    qfi, rf_rank, ent, mi_all, sel_all = [], [], [], [], []
    for s in SEEDS:
        itr, iva = train_test_split(np.arange(len(pool)), test_size=a.val_frac, stratify=pool.label, random_state=s)
        ytr, yva = pool.label.values[itr], pool.label.values[iva]
        mi = mutual_info_classif(G[itr], ytr, random_state=s); top4 = np.sort(np.argsort(-mi)[:4])
        mm = MinMaxScaler().fit(G[itr][:, top4])
        Qtr, Qva, Qte = (np.clip(mm.transform(Z[:, top4]), 0, 1) for Z in (G[itr], G[iva], Gte))
        v = train_vqnn(Qtr, ytr, Qva, yva, s, max(BUDGETS)); th = v["params_at"](max(BUDGETS))
        t = best_threshold(yva, qsim.prob(Qva, th)); mcc = matthews_corrcoef(yte, (qsim.prob(Qte, th) >= t).astype(int))
        ref = R[(R.seed == s) & (R.model == "VQNN") & R.study.isna()].mcc.iloc[0]
        assert abs(mcc - ref) < 1e-9, (s, mcc, ref)              # identical to the reported run
        z2 = (z_expect(final_states(Qtr, th), 4) ** 2).mean(0); q = np.full(8, np.nan); q[top4] = z2 / z2.sum(); qfi.append(q)
        rf = RandomForestClassifier(200, random_state=s, n_jobs=-1).fit(G[itr], ytr)
        rf_rank.append(np.argsort(np.argsort(-rf.feature_importances_)) + 1); mi_all.append(mi); sel_all.append(np.isin(np.arange(8), top4))
        th0 = np.random.default_rng(s).uniform(0, 2 * np.pi, 16)  # the circuit's own random initialisation
        psi, psi0 = final_states(Qte, th), final_states(Qte, th0)
        row = {"seed": s}
        for k, cut in ((1, "q0|q1q2q3"), (2, "q0q1|q2q3")):
            e = entropy(psi, 4, k); row[f"{cut}_trained"] = e.mean(); row[f"{cut}_init"] = entropy(psi0, 4, k).mean()
            row[f"{cut}_rans"] = e[yte == 1].mean(); row[f"{cut}_ben"] = e[yte == 0].mean()
        row["encoded"] = entropy(qsim.encode(Qte), 4, 2).max()  # angle-encoded input is a product state
        ent.append(row); print(f"seed {s}: reproduced MCC {mcc:.4f}", flush=True)
    qfi, rf_rank, mi_all, sel_all, E = np.array(qfi), np.array(rf_rank), np.array(mi_all), np.array(sel_all), pd.DataFrame(ent)
    order = np.argsort(-mi_all.mean(0)); T = lambda f: "\\texttt{" + f.replace("_", "\\_") + "}"
    body = ""
    for i in order:
        qv = qfi[:, i][~np.isnan(qfi[:, i])]
        body += (f"{T(FEATS[i])} & {mi_all[:, i].mean():.3f} & " + (f"{qv.mean():.3f} $\\pm$ {qv.std(ddof=1):.3f}" if len(qv) else "--")
                 + f" & {rf_rank[:, i].mean():.1f} & {int(sel_all[:, i].sum())}/{len(SEEDS)} \\\\\n")
    open(f"{a.out}/tables/qfi.tex", "w").write(body)
    ent_tab = ""
    for cut, mx in (("q0|q1q2q3", 1), ("q0q1|q2q3", 2)):
        ent_tab += (f"$\\{{q_0\\}}\\,|\\,\\{{q_1,q_2,q_3\\}}$" if cut.startswith("q0|") else "$\\{q_0,q_1\\}\\,|\\,\\{q_2,q_3\\}$")
        ent_tab += f" & {mx} & {pm(E[f'{cut}_init'])} & {pm(E[f'{cut}_trained'])} & {pm(E[f'{cut}_rans'])} & {pm(E[f'{cut}_ben'])} \\\\\n"
    open(f"{a.out}/tables/entropy.tex", "w").write(ent_tab)
    M = R[R.study.isna()]; err = ""
    for m in ["VQNN", "1-feature", "LR", "MLP-S", "MLP-L", "RF", "XGB", "SVM", "LR-483"]:
        q = M[M.model == m]; fnr = q.fn / (q.fn + q.tp)
        err += f"{m} & {q.fp.mean():.1f} & {q.fn.mean():.1f} & {pm(q.fpr)} & {pm(fnr)} & {'false positives' if q.fp.mean() > q.fn.mean() else 'false negatives'} \\\\\n"
    open(f"{a.out}/tables/errors.tex", "w").write(err)
    ta = ""                                                    # training time next to accuracy (original Table 8 layout)
    for m, lab in (("VQNN", "VQNN (4 qubits)"), ("VQNN-8q", "VQNN (8 qubits)"), ("LR", "LR"), ("MLP-S", "MLP-S"), ("MLP-L", "MLP-L"),
                   ("RF", "RF"), ("XGB", "XGB"), ("SVM", "SVM"), ("RF-483", "RF-483"), ("XGB-483", "XGB-483")):
        q = M[M.model == m]; ta += f"{lab} & {pm(q.acc)} & {pm(q.mcc)} & {q.train_s.mean():.2f} \\\\\n"
    open(f"{a.out}/tables/timeacc.tex", "w").write(ta)
    res = dict(qfi={FEATS[i]: float(np.nanmean(qfi[:, i])) for i in range(8) if not np.isnan(qfi[:, i]).all()},
               rf_rank={FEATS[i]: float(rf_rank[:, i].mean()) for i in range(8)},
               entropy={c: float(E[c].mean()) for c in E.columns if c != "seed"})
    json.dump(res, open(f"{a.out}/circuit_analysis.json", "w"), indent=1)
    mac = dict(EntOneTrained=f"{E['q0|q1q2q3_trained'].mean():.2f}", EntOneInit=f"{E['q0|q1q2q3_init'].mean():.2f}",
               EntTwoTrained=f"{E['q0q1|q2q3_trained'].mean():.2f}", EntTwoInit=f"{E['q0q1|q2q3_init'].mean():.2f}")
    txt = open(f"{a.out}/numbers.tex").read()
    for k, v in mac.items():
        txt = re.sub(r"\\newcommand\{\\" + k + r"\}\{.*\}\n", "", txt) + f"\\newcommand{{\\{k}}}{{{v}}}\n"
    open(f"{a.out}/numbers.tex", "w").write(txt)
    print(json.dumps(res, indent=1)); print(open(f"{a.out}/tables/qfi.tex").read(), ent_tab, err)

if __name__ == "__main__":
    main()
