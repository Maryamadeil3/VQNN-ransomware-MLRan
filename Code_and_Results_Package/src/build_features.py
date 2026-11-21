#!/usr/bin/env python3
"""Step 1 - build the study table from the public MLRan dataset (Onwuegbuche et al., JNCA 2026).

    git clone https://github.com/faithfulco/mlran
    python src/build_features.py mlran/6_experiments/FS_MLRan_Datasets --out data/mlran_study.csv

Input (all published by the MLRan authors in 6_experiments/FS_MLRan_Datasets/):
  MLRan_X_train_RFE.csv, MLRan_X_test_RFE.csv : 4,880 samples x 483 binary behavioural indicators
      (Cuckoo sandbox; the 483 indicators were chosen by the MLRan authors with mutual information and
      recursive feature elimination on THEIR training split only, so the test split is untouched)
  RFE_selected_feature_names_dic.json          : column id -> indicator name (e.g. "API:NtWriteFile")
  mlran_dataset_metadata.csv                   : SHA-256, family, ransomware type, first-submission year

Output: one row per unique binary (4 duplicated SHA-256 entries are dropped) with
  sample_id, sha256, split (MLRan's fixed split: train / test), label (1 ransomware, 0 goodware),
  family, rtype, year (first submission to VirusTotal),
  eight behavioural features g = log(1 + number of observed indicators in group g),
  and the 483 binary indicators (columns b_<id>) for the full-feature reference models.
"""
import argparse, json, os
import numpy as np, pandas as pd

# feature name -> (indicator-name prefix, description)
GROUPS = {
    "api_calls":          ("API:",       "Windows API functions invoked"),
    "registry_keys":      ("REG:",       "registry keys opened or read"),
    "file_ops":           ("FILE:",      "file-system objects opened, read, written or probed"),
    "directory_ops":      ("DIRECTORY:", "directories created or enumerated"),
    "dropped_files":      ("DROP:",      "types and extensions of dropped files"),
    "sandbox_signatures": ("SIGNATURE:", "Cuckoo behavioural signatures (injection, anti-VM, persistence, packing, ...)"),
    "system_events":      ("SYSTEM:",    "DLLs loaded and COM GUIDs used"),
    "strings":            ("STRING:",    "printable strings observed in memory and the binary"),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mlran_dir"); ap.add_argument("--out", default="data/mlran_study.csv")
    a = ap.parse_args(); d = a.mlran_dir
    names = json.load(open(os.path.join(d, "RFE_selected_feature_names_dic.json")))
    tr = pd.read_csv(os.path.join(d, "MLRan_X_train_RFE.csv")); tr["split"] = "train"
    te = pd.read_csv(os.path.join(d, "MLRan_X_test_RFE.csv")); te["split"] = "test"
    df = pd.concat([tr, te], ignore_index=True)
    meta = pd.read_csv(os.path.join(d, "mlran_dataset_metadata.csv")).set_index("sample_id")
    ids = [c for c in df.columns if c in names]
    assert len(ids) == len(names) == 483, (len(ids), len(names))
    B = df[ids].values.astype(np.int8)
    assert set(np.unique(B)) <= {0, 1}
    out = pd.DataFrame({
        "sample_id": df.sample_id, "sha256": meta.loc[df.sample_id, "sha256"].values, "split": df.split,
        "label": df.sample_type.astype(int), "family": meta.loc[df.sample_id, "ransomware_family"].fillna("goodware").values,
        "rtype": meta.loc[df.sample_id, "ransomware_type"].values,
        "year": meta.loc[df.sample_id, "first_submission_date_year"].values})
    for f, (pref, desc) in GROUPS.items():
        cols = [i for i, c in enumerate(ids) if names[c].startswith(pref)]
        out[f] = np.log1p(B[:, cols].sum(1))
        print(f"{f:20s} {len(cols):4d} indicators  ({desc})")
    full = pd.DataFrame(B, columns=[f"b_{c}" for c in ids])
    out = pd.concat([out, full], axis=1)
    dup = out.sha256.duplicated(keep="first")            # the same binary must not be counted twice
    print(f"removed {int(dup.sum())} duplicated binaries (same SHA-256 listed under two sample ids)")
    out = out[~dup].reset_index(drop=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); out.to_csv(a.out, index=False)
    print(out.groupby(["split", "label"]).size().unstack().rename(columns={0: "goodware", 1: "ransomware"}))
    print("years test:", out[out.split == "test"].year.min(), "-", out[out.split == "test"].year.max(),
          "| families:", out[out.label == 1].family.nunique(), "->", a.out)

if __name__ == "__main__":
    main()
