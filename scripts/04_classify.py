#!/usr/bin/env python3
"""
Step 4 - CRC vs healthy classification.
  * Random Forest and L1-logistic regression on log10 genus abundances
  * Wilcoxon feature selection (top k genera) fitted INSIDE each training fold (no leakage)
  * 5x repeated 5-fold stratified CV on the pooled data
  * Leave-one-study-out (LOSO) validation across the 8 cohorts
  * SHAP values for the final Random Forest
"""
import sys; sys.path.insert(0, "src")
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from crcmb.data import load, log_transform
from crcmb.model import make_models, cross_validate, leave_one_study_out, mean_roc

K = 40
X, y, meta = load()
Xl = log_transform(X)
studies = meta.project_id.to_numpy()
models = make_models(k=K)
summary, loso_all = {}, []
COL = {"RandomForest": "#2f855a", "L1-LogisticRegression": "#6b46c1"}

fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=150)
for name, model in models.items():
    aucs, oof = cross_validate(model, Xl, y)
    grid, mt, sd = mean_roc(y, oof)
    ax.plot(grid, mt, c=COL[name], lw=2, label=f"{name}  AUC {aucs.mean():.3f} ± {aucs.std():.3f}")
    ax.fill_between(grid, mt - sd, mt + sd, color=COL[name], alpha=0.15, lw=0)
    lo = leave_one_study_out(model, Xl, y, studies); lo.insert(0, "model", name); loso_all.append(lo)
    summary[name] = dict(cv_auc_mean=round(aucs.mean(), 4), cv_auc_sd=round(aucs.std(), 4),
                         loso_auc_mean=round(lo.AUC.mean(), 4), loso_auc_min=round(lo.AUC.min(), 4),
                         loso_auc_max=round(lo.AUC.max(), 4))
    print(name, summary[name])
ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"5x5-fold CV, n={len(y)} ({y.sum()} CRC / {len(y)-y.sum()} healthy)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/figures/roc_cv.png")

loso = pd.concat(loso_all)
cohort_names = {"PRJEB6070": "France", "PRJEB7774": "Austria", "PRJEB10878": "China-HK", "PRJEB12449": "USA",
                "PRJEB27928": "Germany", "PRJNA447983": "Italy", "PRJNA731589": "China-1", "PRJNA763023": "China-2"}
loso["cohort"] = loso.held_out.map(cohort_names)
loso.to_csv("results/tables/loso_validation.tsv", sep="\t", index=False, float_format="%.4f")
print(loso.to_string(index=False))

fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=150)
piv = loso.pivot(index="cohort", columns="model", values="AUC").sort_values("RandomForest", ascending=False)
xs = np.arange(len(piv)); w = 0.38
for i, m in enumerate(["RandomForest", "L1-LogisticRegression"]):
    b = ax.bar(xs + (i - 0.5) * w, piv[m], w, color=COL[m], label=m)
    for r in b: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.005, f"{r.get_height():.2f}", ha="center", fontsize=6)
ax.axhline(0.5, ls="--", c="grey", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(piv.index, fontsize=8); ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC on held-out cohort")
ax.set_title("Leave-one-study-out validation (train on 7 cohorts, test on the 8th)", fontsize=9)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/figures/loso_auc.png")

# Final RF on all data -> feature importance + SHAP
rf = clone(models["RandomForest"]).fit(Xl, y)
feats = Xl.columns[rf.named_steps["select"].support_]
Xs = Xl[feats]
imp = pd.DataFrame({"genus": feats, "gini_importance": rf.named_steps["clf"].feature_importances_})
explainer = shap.TreeExplainer(rf.named_steps["clf"])
sv = explainer.shap_values(Xs.to_numpy())
sv = sv[:, :, 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
imp["mean_abs_shap"] = np.abs(sv).mean(axis=0)
imp = imp.sort_values("mean_abs_shap", ascending=False)
imp.to_csv("results/tables/rf_feature_importance.tsv", sep="\t", index=False, float_format="%.5f")
plt.figure(dpi=150)
shap.summary_plot(sv, Xs, max_display=20, show=False, plot_size=(6.5, 5.6))
plt.title("Random Forest SHAP values (log10 abundance)", fontsize=9)
plt.tight_layout(); plt.savefig("results/figures/shap_summary.png"); plt.close()

summary["n_samples"] = int(len(y)); summary["n_crc"] = int(y.sum()); summary["n_genera"] = int(X.shape[1])
summary["selected_features_k"] = K; summary["top_shap_genera"] = imp.genus.head(10).tolist()
json.dump(summary, open("results/tables/model_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
