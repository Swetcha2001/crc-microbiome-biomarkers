import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from crcmb.stats import wilcoxon_z, generalized_fold_change, differential_abundance
from crcmb.model import WilcoxonSelector, make_models, leave_one_study_out


def toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    X = pd.DataFrame(rng.lognormal(0, 1, (n, 6)), columns=[f"g{i}" for i in range(6)])
    X["g0"] += 5 * y                     # enriched in cases
    X["g1"] = X["g1"] * np.where(y, 0.1, 1)  # depleted in cases
    meta = pd.DataFrame({"label": np.where(y, "CRC", "Healthy"),
                         "project_id": np.tile(["A", "B", "C"], n // 3)}, index=X.index)
    return X, y, meta


def test_wilcoxon_sign():
    z, p = wilcoxon_z(np.array([5., 6, 7, 8]), np.array([1., 2, 3, 4]))
    assert z > 0 and p < 0.05


def test_wilcoxon_constant():
    assert wilcoxon_z(np.zeros(5), np.zeros(5)) == (0.0, 1.0)


def test_gfc_direction():
    assert generalized_fold_change(np.full(50, 10.0), np.full(50, 1.0)) > 0.9


def test_differential_abundance_recovers_signal():
    X, y, meta = toy()
    res = differential_abundance(X, meta).set_index("genus")
    assert res.loc["g0", "q_meta"] < 1e-3 and res.loc["g0", "direction"] == "enriched_in_CRC"
    assert res.loc["g1", "q_meta"] < 1e-3 and res.loc["g1", "direction"] == "depleted_in_CRC"
    assert res.loc["g0", "n_cohorts_same_direction"] == 3


def test_selector_picks_informative_features():
    X, y, _ = toy()
    sel = WilcoxonSelector(k=2).fit(X.to_numpy(), y)
    assert set(sel.support_) == {0, 1}


def test_loso_runs():
    X, y, meta = toy()
    res = leave_one_study_out(make_models(k=3)["RandomForest"], np.log10(X), y, meta.project_id.to_numpy())
    assert len(res) == 3 and (res.AUC > 0.8).all()
