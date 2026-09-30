"""Per-cohort differential abundance and meta-analysis."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def generalized_fold_change(case, ctrl, pseudocount=1e-4):
    """Mean difference of log10 quantiles (5%..95%) - robust to zero inflation (Wirbel et al. 2019)."""
    q = np.arange(0.05, 1.0, 0.05)
    return float(np.mean(np.log10(np.quantile(case, q) + pseudocount) - np.log10(np.quantile(ctrl, q) + pseudocount)))


def wilcoxon_z(case, ctrl):
    """Two-sided Mann-Whitney U test; returns (signed z, p). z > 0 => higher in cases."""
    if np.all(case == case[0]) and np.all(ctrl == ctrl[0]) and case[0] == ctrl[0]:
        return 0.0, 1.0
    u, p = stats.mannwhitneyu(case, ctrl, alternative="two-sided")
    n1, n2 = len(case), len(ctrl)
    z = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return float(z), float(p)


def differential_abundance(X, meta, label_col="label", case="CRC", study_col="project_id"):
    """
    For every genus: Wilcoxon test within each cohort, then combine cohorts with a
    sample-size weighted Stouffer's Z (signed). Also reports pooled gFC, prevalence and
    in how many cohorts the direction agrees with the pooled effect.
    """
    studies = sorted(meta[study_col].unique())
    out = []
    for g in X.columns:
        zs, ws, per = [], [], {}
        for s in studies:
            m = meta[study_col] == s
            a = X.loc[m & (meta[label_col] == case), g].to_numpy()
            b = X.loc[m & (meta[label_col] != case), g].to_numpy()
            z, _ = wilcoxon_z(a, b)
            zs.append(z); ws.append(np.sqrt(len(a) + len(b))); per[s] = z
        z_meta = float(np.dot(ws, zs) / np.sqrt(np.sum(np.square(ws))))
        p_meta = float(2 * stats.norm.sf(abs(z_meta)))
        case_all = X.loc[meta[label_col] == case, g].to_numpy()
        ctrl_all = X.loc[meta[label_col] != case, g].to_numpy()
        out.append(dict(
            genus=g, z_meta=z_meta, p_meta=p_meta,
            gFC=generalized_fold_change(case_all, ctrl_all),
            prev_case=float((case_all > 0).mean()), prev_ctrl=float((ctrl_all > 0).mean()),
            mean_case=float(case_all.mean()), mean_ctrl=float(ctrl_all.mean()),
            n_cohorts_same_direction=int(sum(np.sign(z) == np.sign(z_meta) for z in zs if z != 0)),
            **{f"z_{s}": per[s] for s in studies},
        ))
    res = pd.DataFrame(out)
    res["q_meta"] = multipletests(res.p_meta, method="fdr_bh")[1]
    res["direction"] = np.where(res.z_meta > 0, "enriched_in_CRC", "depleted_in_CRC")
    return res.sort_values("p_meta").reset_index(drop=True)
