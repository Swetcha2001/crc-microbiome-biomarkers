#!/usr/bin/env python3
"""Step 3 - Per-cohort Wilcoxon tests + Stouffer meta-analysis; volcano plot and cohort heatmap."""
import sys; sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from crcmb.data import load
from crcmb.stats import differential_abundance

X, y, meta = load()
res = differential_abundance(X, meta)
res.to_csv("results/tables/differential_abundance.tsv", sep="\t", index=False, float_format="%.4g")

sig = res[res.q_meta < 0.05]
up, down = sig[sig.z_meta > 0], sig[sig.z_meta < 0]
print(f"{len(res)} genera tested; FDR<0.05: {len(sig)} ({len(up)} enriched, {len(down)} depleted in CRC)")
print(up.head(12)[["genus", "z_meta", "q_meta", "gFC", "n_cohorts_same_direction"]].to_string(index=False))
print(down.head(8)[["genus", "z_meta", "q_meta", "gFC", "n_cohorts_same_direction"]].to_string(index=False))

# volcano: gFC vs -log10 q
fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
col = np.where(res.q_meta >= 0.05, "#bbbbbb", np.where(res.z_meta > 0, "#c53030", "#2b6cb0"))
ax.scatter(res.gFC, -np.log10(res.q_meta), c=col, s=14, alpha=0.8, linewidths=0)
for _, r in res.head(14).iterrows():
    ax.annotate(r.genus, (r.gFC, -np.log10(r.q_meta)), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
ax.axhline(-np.log10(0.05), ls="--", lw=0.8, c="k")
ax.set_xlabel("Generalized fold change (log10, CRC vs healthy)")
ax.set_ylabel("-log10 FDR (Stouffer meta-analysis)")
ax.set_title(f"Genus-level differential abundance, {len(meta)} samples / 8 cohorts", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/figures/volcano.png")

# heatmap of per-cohort z for top 25 genera
top = res.head(25)
zc = [c for c in res.columns if c.startswith("z_") and c != "z_meta"]
H = top.set_index("genus")[zc]
H.columns = [c[2:] for c in zc]
H["META"] = top.set_index("genus").z_meta
fig, ax = plt.subplots(figsize=(7, 7.2), dpi=150)
sns.heatmap(H, cmap="RdBu_r", center=0, vmin=-6, vmax=6, cbar_kws={"label": "Wilcoxon z (+ = enriched in CRC)"}, ax=ax,
            linewidths=0.3, linecolor="white")
ax.set_xlabel("Cohort"); ax.set_ylabel("")
ax.set_title("Top 25 genera: consistency across cohorts", fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7); plt.setp(ax.get_yticklabels(), fontsize=7, style="italic")
fig.tight_layout(); fig.savefig("results/figures/cohort_heatmap.png")
