#!/usr/bin/env python3
"""
Step 2 - Build the genus x sample relative-abundance matrix.

* 'Unknown' (unclassified reads) is removed and each profile is re-scaled to 100%
  so samples with different unclassified fractions are comparable.
* Genera present (> 0) in fewer than 5% of samples are dropped.

Outputs: data/processed/genus_relab.tsv.gz (samples x genera, %), data/processed/metadata.tsv
"""
import json
import pandas as pd

MIN_PREVALENCE = 0.05

meta = pd.read_csv("data/raw/samples.tsv", sep="\t")
rows = {}
for run in meta.run_id:
    prof = json.load(open(f"data/raw/profiles/{run}.json"))["genus"]
    rows[run] = {g["scientific_name"]: g["relative_abundance"] for g in prof}

X = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
X = X.drop(columns=[c for c in ("Unknown", "Others") if c in X.columns])
X = X.div(X.sum(axis=1), axis=0) * 100
n_all = X.shape[1]
X = X.loc[:, (X > 0).mean() >= MIN_PREVALENCE]
X = X[sorted(X.columns)]

meta = meta.set_index("run_id").loc[X.index]
X.to_csv("data/processed/genus_relab.tsv.gz", sep="\t", float_format="%.6g")
meta.to_csv("data/processed/metadata.tsv", sep="\t")
print(f"{X.shape[0]} samples; {n_all} genera detected; {X.shape[1]} kept (prevalence >= {MIN_PREVALENCE:.0%})")
print(meta.groupby(["project_id", "label"]).size().unstack())
