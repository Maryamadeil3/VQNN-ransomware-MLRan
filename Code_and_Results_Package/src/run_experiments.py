#!/usr/bin/env python3
"""Step 2 - complete evaluation protocol of the paper.

    python src/run_experiments.py data/mlran_study.csv --out results

Design (one fixed train/test setup, as requested by the reviewers):
  * TEST  = the fixed test split published with MLRan (975 samples, first submitted 2021-2024).
            It is used once per model and seed, for the reported metrics only.
  * POOL  = MLRan's training split (3,901 samples). For every seed it is divided, stratified, into
            TRAINING (80 %) and VALIDATION (20 %).
  * Feature ranking (mutual information) and all scalers / PCA are fitted on TRAINING only.
  * Every model's decision threshold is chosen on VALIDATION by one rule (max MCC); recall at a
    fixed false-positive rate uses the lowest threshold with validation FPR <= target.
  * The same 10 seeds and the same splits are used for every model, quantum and classical.
  * One factor at a time: optimiser budget, training-set size, circuit components, input encoding,
    hardware-like noise (evaluation only).
  * Statistics: mean, SD, 95 % t-interval; paired two-sided Wilcoxon signed-rank tests with Holm
    correction; effect size d_z; exact McNemar tests per seed; shared-error analysis.
Writes <out>/{per_seed.csv, errors_per_seed.csv, summary.json, numbers.tex, tables/*.tex, figs/*.pdf}.
"""
import argparse, json, os, sys, time, warnings, platform
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qsim

warnings.filterwarnings("ignore")
SEEDS = [0, 42, 100, 123, 200, 256, 512, 1000, 2024, 9999]
BUDGETS = [10, 25, 50, 100, 200, 400]
FPRS = [0.01, 0.05]
FEATS = ["api_calls", "registry_keys", "file_ops", "directory_ops", "dropped_files", "sandbox_signatures", "system_events", "strings"]
NOISE = [("exact", 0, 0, 0, None), ("shots 1024", 0, 0, 0, 1024),
         ("p2=0.5%", 0.0005, 0.005, 0.01, 1024), ("p2=1%", 0.001, 0.01, 0.01, 1024), ("p2=5%", 0.005, 0.05, 0.01, 1024), ("p2=10%", 0.01, 0.10, 0.02, 1024)]
YEARS = [("2021-2022", (2021, 2022)), ("2023-2024", (2023, 2024))]

# ------------------------------------------------------------------ metrics
def confusion(y, pred):
    return (int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum()),
            int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum()))

def best_threshold(y, s):
    cand = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 199)))
    return max(cand, key=lambda t: matthews_corrcoef(y, (s >= t).astype(int)))

def fpr_threshold(yva, sva, a):
    neg, ta = sva[yva == 0], np.inf                       # lowest threshold with validation FPR <= a
    for t in np.unique(sva)[::-1]:
        if (neg >= t).mean() <= a: ta = t
        else: break
    return ta

def metrics_at(yte, ste, t):
    pred = (ste >= t).astype(int); tp, tn, fp, fn = confusion(yte, pred)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / len(yte), prec=prec, rec=rec, f1=2 * prec * rec / max(prec + rec, 1e-12),
                mcc=matthews_corrcoef(yte, pred), fpr=fp / max(fp + tn, 1), tp=tp, tn=tn, fp=fp, fn=fn), pred

def evaluate(yva, sva, yte, ste):
    t = best_threshold(yva, sva); m, pred = metrics_at(yte, ste, t)
    m.update(auc=roc_auc_score(yte, ste), thr=float(t))
    for a in FPRS:
        ta = fpr_threshold(yva, sva, a)
        m[f"rec@{a:g}"] = float((ste[yte == 1] >= ta).mean()); m[f"fpr@{a:g}"] = float((ste[yte == 0] >= ta).mean())
    return m, pred

# ------------------------------------------------------------------ VQNN
def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

def train_vqnn(Xtr, ytr, Xva, yva, seed, maxiter, L=2, entangle=True, rz=True, track_val=False):
    kw = dict(L=L, entangle=entangle, rz=rz); rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2 * np.pi, qsim.n_params(Xtr.shape[1], L, rz)); hist, tt = [], [0.0]
    def loss(th):
        t0 = time.perf_counter(); l = bce(ytr, qsim.prob(Xtr, th, **kw)); tt[0] += time.perf_counter() - t0
        hist.append((th.copy(), l, bce(yva, qsim.prob(Xva, th, **kw)) if track_val else np.nan)); return l
    minimize(loss, th0, method="COBYLA", options=dict(maxiter=maxiter, rhobeg=0.5, tol=1e-4))
    def params_at(k):                                     # COBYLA is deterministic: budget k = first k evaluations
        h = hist[:k]; return h[int(np.argmin([x[1] for x in h]))][0]
    return dict(kw=kw, hist=hist, params_at=params_at, nfev=len(hist), time=tt[0])

# ------------------------------------------------------------------ classical models (fixed hyper-parameters, no tuning)
def classical(seed):
    return {
        "LR":    lambda: LogisticRegression(max_iter=2000),
        "MLP-S": lambda: MLPClassifier((3,), activation="tanh", max_iter=2000, random_state=seed),   # 4-3-1: 19 parameters
        "MLP-L": lambda: MLPClassifier((64, 32), max_iter=500, random_state=seed),
        "RF":    lambda: RandomForestClassifier(200, random_state=seed, n_jobs=-1),
        "XGB":   lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed, n_jobs=4, verbosity=0),
        "SVM":   lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
    }
def score(m, X): return m.decision_function(X) if isinstance(m, SVC) else m.predict_proba(X)[:, 1]
def n_par(m, nfeat, name):
    if hasattr(m, "coefs_"): return int(sum(w.size for w in m.coefs_ + m.intercepts_))
    return nfeat + 1 if name.startswith("LR") else np.nan

# ------------------------------------------------------------------ one seed
def run_seed(df, seed, a):
    pool = df[df.split == "train"].reset_index(drop=True); test = df[df.split == "test"].reset_index(drop=True)
    itr, iva = train_test_split(np.arange(len(pool)), test_size=a.val_frac, stratify=pool.label, random_state=seed)
    ytr, yva, yte = pool.label.values[itr], pool.label.values[iva], test.label.values
    G = pool[FEATS].values; Gte = test[FEATS].values
    bcols = [c for c in df.columns if c.startswith("b_")]
    B = pool[bcols].values.astype(float); Bte = test[bcols].values.astype(float)
    mi = mutual_info_classif(G[itr], ytr, random_state=seed); top4 = np.sort(np.argsort(-mi)[:4])
    rows, preds, rocs, aux = [], {}, {}, {}
    def add(name, nfeat, m, pred, t, npar, **extra):
        rows.append(dict(seed=seed, model=name, nfeat=nfeat, train_s=t, n_par=npar, **m, **extra)); preds[name] = pred
    def year_rows(name, sva_, ste_):
        t = best_threshold(yva, sva_)
        for yl, ys in YEARS:
            mk = test.year.isin(ys).values; m, _ = metrics_at(yte[mk], ste_[mk], t); m["auc"] = roc_auc_score(yte[mk], ste_[mk])
            rows.append(dict(seed=seed, model=name, study="year", x=yl, n=int(mk.sum()), **m))

    def scaled(cols):
        ss = StandardScaler().fit(G[itr][:, cols]); mm = MinMaxScaler().fit(G[itr][:, cols])
        S = [ss.transform(Z[:, cols]) for Z in (G[itr], G[iva], Gte)]
        Q = [np.clip(mm.transform(Z[:, cols]), 0, 1) for Z in (G[itr], G[iva], Gte)]
        return S, Q

    for nfeat, cols in ((4, top4), (8, np.arange(8))):
        (Str, Sva, Ste), (Qtr, Qva, Qte) = scaled(cols); tag = "" if nfeat == 4 else "-8f"
        if nfeat == 4:
            j = int(np.argmax(mi[cols])); sgn = np.sign(np.corrcoef(Str[:, j], ytr)[0, 1]) or 1
            m, p = evaluate(yva, sgn * Sva[:, j], yte, sgn * Ste[:, j]); add("1-feature", 1, m, p, 0.0, 1)
        for name, mk in classical(seed).items():
            if nfeat == 8 and name in ("MLP-S", "SVM"): continue
            t0 = time.perf_counter(); mdl = mk().fit(Str, ytr); t = time.perf_counter() - t0
            sva, ste = score(mdl, Sva), score(mdl, Ste); m, p = evaluate(yva, sva, yte, ste)
            add(name + tag, nfeat, m, p, t, n_par(mdl, nfeat, name))
            if nfeat == 4:
                rocs[name] = (yte, ste); year_rows(name, sva, ste)
        v = train_vqnn(Qtr, ytr, Qva, yva, seed, max(BUDGETS), track_val=(nfeat == 4))
        vname = "VQNN" if nfeat == 4 else "VQNN-8q"
        for k in (BUDGETS if nfeat == 4 else [max(BUDGETS)]):
            th = v["params_at"](k); sva, ste = qsim.prob(Qva, th), qsim.prob(Qte, th)
            m, p = evaluate(yva, sva, yte, ste)
            if k == max(BUDGETS):
                add(vname, nfeat, m, p, v["time"], len(th), nfev=v["nfev"])
                if nfeat == 4: rocs["VQNN"] = (yte, ste); year_rows("VQNN", sva, ste); th_best = th
            if nfeat == 4: rows.append(dict(seed=seed, model="VQNN", study="budget", x=k, **m))
        if nfeat == 4:
            aux["loss"] = np.array([[h[1], h[2]] for h in v["hist"]])
            # circuit ablations (same data, same budget, one component changed)
            for an, kw in (("no entanglement", dict(entangle=False)), ("1 layer", dict(L=1)), ("RY only (no RZ)", dict(rz=False))):
                va = train_vqnn(Qtr, ytr, Qva, yva, seed, max(BUDGETS), **kw); th = va["params_at"](va["nfev"])
                m, _ = evaluate(yva, qsim.prob(Qva, th, **va["kw"]), yte, qsim.prob(Qte, th, **va["kw"]))
                rows.append(dict(seed=seed, model="VQNN", study="ablation", x=an, n_par=len(th), **m))
            # training-set size (validation and test unchanged)
            rs = np.random.default_rng(seed)
            for n in sorted(set(a.sizes) | {len(itr)}):
                n = min(n, len(itr))
                sub = np.r_[rs.permutation(np.where(ytr == 0)[0])[: n // 2], rs.permutation(np.where(ytr == 1)[0])[: n - n // 2]]
                vs = train_vqnn(Qtr[sub], ytr[sub], Qva, yva, seed, max(BUDGETS)); th = vs["params_at"](vs["nfev"])
                m, _ = evaluate(yva, qsim.prob(Qva, th), yte, qsim.prob(Qte, th))
                rows.append(dict(seed=seed, model="VQNN", study="size", x=n, **m))
                for cn in ("LR", "MLP-S"):
                    mdl = classical(seed)[cn]().fit(Str[sub], ytr[sub]); m, _ = evaluate(yva, score(mdl, Sva), yte, score(mdl, Ste))
                    rows.append(dict(seed=seed, model=cn, study="size", x=n, **m))
            # hardware-like noise (evaluation of the trained circuit; threshold re-chosen on noisy validation scores)
            for nl, p1, p2, ro, shots in NOISE:
                kw = dict(p1=p1, p2=p2, readout=ro, shots=shots)
                sva = qsim.prob_noisy(Qva, th_best, rng=np.random.default_rng(seed), **kw)
                ste = qsim.prob_noisy(Qte, th_best, rng=np.random.default_rng(seed + 1), **kw)
                m, _ = evaluate(yva, sva, yte, ste); rows.append(dict(seed=seed, model="VQNN", study="noise", x=nl, **m))
    # input-encoding study: PCA of the 483 indicators (fitted on training) instead of the 4 group counts
    pca = PCA(4, random_state=seed).fit(B[itr]); P = [pca.transform(Z) for Z in (B[itr], B[iva], Bte)]
    ss = StandardScaler().fit(P[0]); mm = MinMaxScaler().fit(P[0])
    PS = [ss.transform(Z) for Z in P]; PQ = [np.clip(mm.transform(Z), 0, 1) for Z in P]
    vp = train_vqnn(PQ[0], ytr, PQ[1], yva, seed, max(BUDGETS)); th = vp["params_at"](vp["nfev"])
    m, _ = evaluate(yva, qsim.prob(PQ[1], th), yte, qsim.prob(PQ[2], th)); rows.append(dict(seed=seed, model="VQNN", study="encoding", x="PCA-4", **m))
    for cn in ("LR", "MLP-S"):
        mdl = classical(seed)[cn]().fit(PS[0], ytr); m, _ = evaluate(yva, score(mdl, PS[1]), yte, score(mdl, PS[2]))
        rows.append(dict(seed=seed, model=cn, study="encoding", x="PCA-4", **m))
    for cn in ("VQNN", "LR", "MLP-S"):
        r = next(r for r in rows if r.get("model") == cn and r.get("nfeat") == 4 and "study" not in r)
        rows.append(dict(seed=seed, model=cn, study="encoding", x="group counts (MI top-4)", **{k: r[k] for k in ("acc", "mcc", "auc", "f1")}))
    # full-information references: all 483 binary indicators
    for name, mk in (("LR-483", lambda: LogisticRegression(max_iter=3000)),
                     ("RF-483", lambda: RandomForestClassifier(200, random_state=seed, n_jobs=-1)),
                     ("XGB-483", lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed, n_jobs=4, verbosity=0))):
        t0 = time.perf_counter(); mdl = mk().fit(B[itr], ytr); t = time.perf_counter() - t0
        sva, ste = score(mdl, B[iva]), score(mdl, Bte); m, p = evaluate(yva, sva, yte, ste)
        add(name, 483, m, p, t, 484 if name == "LR-483" else np.nan); year_rows(name, sva, ste)
    # error-set analysis on the four-feature models
    cl = ["LR", "MLP-S", "MLP-L", "RF", "XGB", "SVM"]
    allwrong = np.all([preds[c] != yte for c in cl], axis=0); ok = preds["VQNN"] == yte
    err = dict(seed=seed, shared_wrong=int(allwrong.sum()), vqnn_right_of_shared=int((allwrong & ok).sum()),
               agree_lr=float((preds["VQNN"] == preds["LR"]).mean()))
    for c in cl + ["1-feature"]:
        b = int((ok & (preds[c] != yte)).sum()); c2 = int((~ok & (preds[c] == yte)).sum())
        err[f"b_{c}"], err[f"c_{c}"] = b, c2
        err[f"p_{c}"] = float(stats.binomtest(b, b + c2, 0.5).pvalue) if b + c2 else 1.0
    aux.update(mi=mi, top4=top4, rocs=rocs)
    return rows, err, aux

# ------------------------------------------------------------------ aggregation helpers
def ci95(v):
    v = np.asarray(v, float); return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
def pm(v, d=3): return f"{np.mean(v):.{d}f} $\\pm$ {np.std(v, ddof=1):.{d}f}"
def mci(v, d=3): return f"{np.mean(v):.{d}f} [{np.mean(v)-ci95(v):.{d}f}, {np.mean(v)+ci95(v):.{d}f}]"
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1)
    return adj
def fp(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
def key(m): return {"1-feature": "OneFeat", "MLP-S": "MlpS", "MLP-L": "MlpL", "VQNN-8q": "VQNNEight", "LR-483": "LRFull",
                    "RF-483": "RFFull", "XGB-483": "XGBFull", "LR-8f": "LREight", "MLP-L-8f": "MlpLEight", "RF-8f": "RFEight",
                    "XGB-8f": "XGBEight"}.get(m, m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--out", default="results")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--sizes", type=int, nargs="*", default=[100, 300, 1000])
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    os.makedirs(f"{a.out}/tables", exist_ok=True); os.makedirs(f"{a.out}/figs", exist_ok=True)
    rows, errs, auxs = [], [], []
    for s in a.seeds:
        t0 = time.time(); r, e, x = run_seed(df, s, a); rows += r; errs.append(e); auxs.append(x)
        print(f"seed {s} done in {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); E = pd.DataFrame(errs)
    R.to_csv(f"{a.out}/per_seed.csv", index=False); E.to_csv(f"{a.out}/errors_per_seed.csv", index=False)
    M = R[R.study.isna()]
    g = lambda model, col: M[M.model == model].sort_values("seed")[col].values
    W = lambda name, body: open(f"{a.out}/tables/{name}.tex", "w").write(body)
    order4 = ["VQNN", "1-feature", "LR", "MLP-S", "MLP-L", "RF", "XGB", "SVM"]
    order8 = ["VQNN-8q", "LR-8f", "MLP-L-8f", "RF-8f", "XGB-8f"]
    orderF = ["LR-483", "RF-483", "XGB-483"]
    lab = lambda m: m.replace("VQNN-8q", "VQNN (8 qubits)").replace("-8f", "")
    def perf_rows(order):
        out = ""
        for m in order:
            npar = g(m, "n_par")[0]; npar = "--" if np.isnan(npar) else f"{int(npar)}"
            out += f"{lab(m)} & {npar} & " + " & ".join(pm(g(m, c)) for c in ("acc", "f1", "mcc", "auc")) + " \\\\\n"
        return out
    W("main", perf_rows(order4)); W("eightf", perf_rows(order8)); W("full", perf_rows(orderF))
    W("classwise", "".join(f"{lab(m)} & " + " & ".join(pm(g(m, c)) for c in ("rec", "prec", "fpr", "rec@0.01", "rec@0.05")) + " \\\\\n" for m in order4 + orderF))
    comp = [m for m in order4 if m != "VQNN"]
    P = {c: [stats.wilcoxon(g("VQNN", c), g(m, c)).pvalue if np.any(g("VQNN", c) != g(m, c)) else 1.0 for m in comp] for c in ("mcc", "auc")}
    H = {c: holm(P[c]) for c in P}; summ = {}; body = ""
    for i, m in enumerate(comp):
        cells = []
        for c in ("mcc", "auc"):
            d = g("VQNN", c) - g(m, c); dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
            cells += [mci(d), f"{dz:.2f}", fp(H[c][i])]
            summ[f"{m}_{c}"] = dict(diff=float(d.mean()), lo=float(d.mean() - ci95(d)), hi=float(d.mean() + ci95(d)), dz=float(dz), p_holm=float(H[c][i]),
                                     n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))
        body += f"{m} & " + " & ".join(cells) + " \\\\\n"
    W("paired", body)
    W("mcnemar", "".join(f"{m} & {E[f'b_{m}'].sum()} & {E[f'c_{m}'].sum()} & {fp(E[f'p_{m}'].median())} & {(E[f'p_{m}'] < 0.05).sum()}/{len(E)} \\\\\n" for m in comp))
    A = R[R.study == "ablation"]
    ab = f"Full circuit & {int(g('VQNN','n_par')[0])} & " + " & ".join(pm(g("VQNN", c)) for c in ("acc", "mcc", "auc")) + " \\\\\n"
    for x in A.x.unique():
        q = A[A.x == x]; ab += f"{x} & {int(q.n_par.iloc[0])} & " + " & ".join(pm(q[c]) for c in ("acc", "mcc", "auc")) + " \\\\\n"
    ab += f"8 qubits, 8 features & {int(g('VQNN-8q','n_par')[0])} & " + " & ".join(pm(g("VQNN-8q", c)) for c in ("acc", "mcc", "auc")) + " \\\\\n"
    W("ablation", ab)
    Bd = R[R.study == "budget"]; W("budget", "".join(f"{k} & " + " & ".join(pm(Bd[Bd.x == k][c]) for c in ("acc", "mcc", "auc")) + " \\\\\n" for k in BUDGETS))
    S = R[R.study == "size"]; sizes = sorted(S.x.unique())
    W("size", "".join(f"{int(n)} & " + " & ".join(pm(S[(S.x == n) & (S.model == m)]["mcc"]) for m in ("VQNN", "LR", "MLP-S")) + " \\\\\n" for n in sizes))
    N = R[R.study == "noise"]
    W("noise", "".join(f"{nl.replace('%', chr(92) + '%')} & {p1:g} & {p2:g} & {ro:g} & {shots or '--'} & " + " & ".join(pm(N[N.x == nl][c]) for c in ("acc", "mcc", "auc")) + " \\\\\n"
                       for nl, p1, p2, ro, shots in NOISE))
    Y = R[R.study == "year"]
    W("year", "".join(f"{lab(m)} & " + " & ".join(f"{pm(Y[(Y.model == m) & (Y.x == yl)].mcc)} & {pm(Y[(Y.model == m) & (Y.x == yl)].auc)}" for yl, _ in YEARS) + " \\\\\n"
                      for m in ["VQNN", "LR", "MLP-S", "RF", "XGB", "RF-483"]))
    En = R[R.study == "encoding"]
    W("encoding", "".join(f"{x} & " + " & ".join(f"{pm(En[(En.x == x) & (En.model == m)].mcc)}" for m in ("VQNN", "LR", "MLP-S")) + " & "
                          + " & ".join(f"{pm(En[(En.x == x) & (En.model == m)].auc)}" for m in ("VQNN", "LR", "MLP-S")) + " \\\\\n"
                          for x in ["group counts (MI top-4)", "PCA-4"]))
    W("time", "".join(f"{lab(m)} & {np.mean(g(m,'train_s')):.2f} \\\\\n" for m in ["LR", "MLP-S", "MLP-L", "RF", "XGB", "SVM", "VQNN", "VQNN-8q", "RF-483", "XGB-483"]))
    mi = np.array([x["mi"] for x in auxs]); sel = np.zeros(8, int)
    for x in auxs: sel[x["top4"]] += 1
    W("features", "".join(f"\\texttt{{{f.replace('_', chr(92)+'_')}}} & {mi[:, i].mean():.3f} $\\pm$ {mi[:, i].std(ddof=1):.3f} & {sel[i]}/{len(auxs)} & {df[f].nunique()} \\\\\n"
                          for i, f in sorted(enumerate(FEATS), key=lambda t: -mi[:, t[0]].mean())))
    fam = df[df.label == 1].groupby(["family", "split"]).size().unstack(fill_value=0).sort_values("train", ascending=False)
    W("families", "".join(f"{f} & {r.get('train', 0)} & {r.get('test', 0)} \\\\\n" for f, r in fam.iterrows()))
    # macros
    nfev = np.mean(g("VQNN", "nfev")); tv = np.mean(g("VQNN", "train_s")); ntr = len(df[df.split == "train"]) - int(np.ceil(a.val_frac * len(df[df.split == "train"])))   # sklearn rounds the validation size up
    per_circ = tv / (nfev * ntr); best = max(comp, key=lambda m: np.mean(g(m, "mcc")))
    import scipy, sklearn, xgboost, qiskit
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "") if os.path.exists("/proc/cpuinfo") else platform.processor()
    te = df[df.split == "test"]; tr = df[df.split == "train"]
    mac = dict(NTotal=len(df), NRans=int(df.label.sum()), NBen=int((df.label == 0).sum()), NFam=df[df.label == 1].family.nunique(),
               NPool=len(tr), NTrain=ntr, NVal=len(tr) - ntr, NTest=len(te), NTestRans=int(te.label.sum()), NTestBen=int((te.label == 0).sum()),
               TestYears=f"{int(te.year.min())}--{int(te.year.max())}", PoolYears=f"{int(tr.year.min())}--{int(tr.year.max())}",
               NSeeds=len(a.seeds), MaxIter=max(BUDGETS), VQNNParams=int(g("VQNN", "n_par")[0]), VQNNTime=f"{tv:.1f}",
               NCircEvals=f"{int(nfev * ntr):,}".replace(",", "{,}"), MicrosecPerCirc=f"{per_circ*1e6:.2f}",
               FullCostHours=f"{nfev * 1_000_000 * per_circ / 3600:.2f}", BestClassical=best,
               SharedWrong=f"{E.shared_wrong.mean():.1f}", VQNNRightShared=f"{E.vqnn_right_of_shared.mean():.1f}", AgreeLR=f"{100*E.agree_lr.mean():.1f}",
               SoftwareVersions=(f"Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}, "
                                 f"XGBoost {xgboost.__version__}, Qiskit {qiskit.__version__}; CPU: {cpu} ({os.cpu_count()} cores)"))
    for m in order4 + order8 + orderF:
        for c, cn in (("acc", "Acc"), ("mcc", "MCC"), ("auc", "AUC"), ("f1", "F"), ("rec", "Rec"), ("fpr", "FPR"), ("rec@0.01", "RecOne"), ("rec@0.05", "RecFive")):
            mac[f"{key(m)}{cn}"] = pm(g(m, c))
    open(f"{a.out}/numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    agg = lambda D, by: {str(k): {c: float(v[c].mean()) for c in ("acc", "mcc", "auc")} for k, v in D.groupby(by)}
    json.dump(dict(args=vars(a), selection_counts=dict(zip(FEATS, sel.tolist())), mi_mean=dict(zip(FEATS, mi.mean(0).tolist())),
                   means={m: {c: float(np.mean(g(m, c))) for c in ("acc", "f1", "mcc", "auc", "rec", "prec", "fpr", "rec@0.01", "rec@0.05")} for m in order4 + order8 + orderF},
                   sds={m: {c: float(np.std(g(m, c), ddof=1)) for c in ("acc", "mcc", "auc")} for m in order4 + order8 + orderF},
                   paired_vs_vqnn=summ, ablation=agg(A, "x"), budget=agg(Bd, "x"), noise=agg(N, "x"),
                   size={f"{m}@{int(n)}": float(S[(S.x == n) & (S.model == m)].mcc.mean()) for m in ("VQNN", "LR", "MLP-S") for n in sizes},
                   year={f"{m}@{x}": {c: float(v[c].mean()) for c in ("mcc", "auc")} for (m, x), v in Y.groupby(["model", "x"])},
                   encoding={f"{m}@{x}": {c: float(v[c].mean()) for c in ("mcc", "auc")} for (m, x), v in En.groupby(["model", "x"])},
                   errors=dict(shared_wrong=float(E.shared_wrong.mean()), vqnn_right=float(E.vqnn_right_of_shared.mean()), agree_lr=float(E.agree_lr.mean()),
                               mcnemar_sig={m: int((E[f'p_{m}'] < 0.05).sum()) for m in comp}),
                   macros=mac), open(f"{a.out}/summary.json", "w"), indent=1, default=float)
    figures(a, df, R, M, auxs, order4)
    print("\n", M.groupby("model")[["acc", "f1", "mcc", "auc", "rec", "fpr"]].mean().round(3).loc[order4 + order8 + orderF])
    print("\nbest classical (4 features) by MCC:", best, "| results in", a.out)

def _style():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25, "savefig.bbox": "tight"})
    return plt

def fig_noise(a, R, M):
    plt = _style(); TEAL, PURP = "#1E8C93", "#6B4FA0"
    N = R[R.study == "noise"]; labs = [x[0] for x in NOISE]; lr = M[M.model == "LR"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for k, (c, lb) in enumerate((("auc", "ROC-AUC"), ("mcc", "MCC"))):
        ax[k].errorbar(range(len(labs)), [N[N.x == l][c].mean() for l in labs], yerr=[ci95(N[N.x == l][c]) for l in labs],
                       color=PURP, marker="o", capsize=3, label="VQNN (trained exactly, evaluated with noise)")
        ax[k].axhline(lr[c].mean(), color=TEAL, ls="--", label="LR, exact (reference)")
        ax[k].set_xticks(range(len(labs))); ax[k].set_xticklabels(labs, rotation=30, ha="right"); ax[k].set_ylabel(f"Test {lb}")
        lo = min(N[c].min(), lr[c].mean()) - 0.03; hi = max(N[c].max(), lr[c].mean()) + 0.03; ax[k].set_ylim(lo, hi)
    ax[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{a.out}/figs/fig_noise.pdf"); plt.close(fig)

def fig_year(a, R):
    plt = _style(); TEAL, ORNG = "#1E8C93", "#C9772B"
    Y = R[R.study == "year"]; models = ["VQNN", "LR", "MLP-S", "RF", "XGB", "RF-483"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9)); w = .38
    for k, (yl, _) in enumerate(YEARS):
        mu = [Y[(Y.model == m) & (Y.x == yl)].auc.mean() for m in models]; ci = [ci95(Y[(Y.model == m) & (Y.x == yl)].auc) for m in models]
        ax.bar(np.arange(len(models)) + (k - .5) * w, mu, w, yerr=ci, capsize=2, color=[TEAL, ORNG][k], label=f"test samples first seen {yl}")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=35, ha="right"); ax.set_ylim(.8, 1.02); ax.set_ylabel("Test ROC-AUC")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=1); ax.grid(axis="x", visible=False)
    fig.savefig(f"{a.out}/figs/fig_year.pdf"); plt.close(fig)

def redraw_from_csv(out):
    """Regenerate the noise and year figures from results/per_seed.csv without re-running the experiments."""
    import types; R = pd.read_csv(f"{out}/per_seed.csv"); M = R[R.study.isna()]; a = types.SimpleNamespace(out=out)
    fig_noise(a, R, M); fig_year(a, R)

# ------------------------------------------------------------------ figures
def figures(a, df, R, M, auxs, order4):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25, "savefig.bbox": "tight"})
    TEAL, PURP, GREY, ORNG = "#1E8C93", "#6B4FA0", "#8A8F98", "#C9772B"
    col = lambda m: PURP if m.startswith("VQNN") else (GREY if m == "1-feature" else TEAL)
    save = lambda fig, name: (fig.savefig(f"{a.out}/figs/{name}.pdf"), plt.close(fig))
    g = lambda m, c: M[M.model == m].sort_values("seed")[c].values
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, (c, t) in enumerate((("mcc", "MCC"), ("auc", "ROC-AUC"))):
        mu = [g(m, c).mean() for m in order4]; ci = [ci95(g(m, c)) for m in order4]
        ax[k].bar(range(len(order4)), mu, yerr=ci, color=[col(m) for m in order4], capsize=3, width=.7)
        ax[k].set_xticks(range(len(order4))); ax[k].set_xticklabels(order4, rotation=40, ha="right"); ax[k].set_ylabel(t)
        if c == "auc": ax[k].set_ylim(.5, 1)
        ax[k].grid(axis="x", visible=False)
    save(fig, "fig_main")
    fig, ax = plt.subplots(figsize=(3.5, 3.2)); grid = np.linspace(0, 1, 401)
    for m, c, ls in (("VQNN", PURP, "-"), ("LR", TEAL, "-"), ("MLP-S", TEAL, "--"), ("XGB", GREY, "-")):
        tpr = np.mean([np.interp(grid, *roc_curve(*x["rocs"][m])[:2]) for x in auxs], 0); ax.plot(grid, tpr, color=c, ls=ls, lw=1.6, label=m)
    ax.plot([0, 1], [0, 1], color="k", lw=.6, ls=":"); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate (recall)"); ax.legend(frameon=False)
    save(fig, "fig_roc")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8)); Bd = R[R.study == "budget"]
    ax[0].errorbar(BUDGETS, [Bd[Bd.x == k].mcc.mean() for k in BUDGETS], yerr=[ci95(Bd[Bd.x == k].mcc) for k in BUDGETS], color=PURP, marker="o", capsize=3, label="VQNN")
    ax[0].axhline(g("LR", "mcc").mean(), color=TEAL, ls="--", label="LR (reference)"); ax[0].set_xscale("log")
    ax[0].set_xlabel("COBYLA objective evaluations"); ax[0].set_ylabel("Test MCC"); ax[0].legend(frameon=False)
    n = min(len(x["loss"]) for x in auxs); L = np.array([x["loss"][:n] for x in auxs])
    for j, (lb, ls) in enumerate((("training", "-"), ("validation", "--"))): ax[1].plot(np.arange(1, n + 1), L[:, :, j].mean(0), color=PURP, ls=ls, label=lb)
    ax[1].set_xlabel("COBYLA objective evaluation"); ax[1].set_ylabel("Binary cross-entropy"); ax[1].legend(frameon=False)
    save(fig, "fig_budget")
    fig, ax = plt.subplots(figsize=(3.5, 2.8)); S = R[R.study == "size"]; xs = sorted(S.x.unique())
    for m, c, ls in (("VQNN", PURP, "-"), ("LR", TEAL, "-"), ("MLP-S", TEAL, "--")):
        q = [S[(S.x == x) & (S.model == m)].mcc for x in xs]
        ax.errorbar(xs, [v.mean() for v in q], yerr=[ci95(v) for v in q], color=c, ls=ls, marker="o", capsize=3, label=m)
    ax.set_xscale("log"); ax.set_xlabel("Training samples"); ax.set_ylabel("Test MCC"); ax.legend(frameon=False)
    save(fig, "fig_size")
    fig_noise(a, R, M)
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.4))
    for axx, f in zip(axs.ravel(), FEATS):
        bins = np.histogram_bin_edges(df[f], 25)
        axx.hist(df[f][df.label == 0], bins, color=TEAL, alpha=.6, density=True, label="goodware")
        axx.hist(df[f][df.label == 1], bins, color=PURP, alpha=.6, density=True, label="ransomware")
        axx.set_title(f, fontsize=8); axx.set_yticks([])
    axs[0, 0].legend(frameon=False, fontsize=7); fig.tight_layout(); save(fig, "fig_features")
    fig_year(a, R)

if __name__ == "__main__":
    main()
