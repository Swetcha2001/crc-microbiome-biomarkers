"""Classification models and leakage-free evaluation (feature selection inside CV folds)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class WilcoxonSelector(BaseEstimator, TransformerMixin):
    """Keep the k features with the smallest Mann-Whitney p-value (fit on training data only)."""

    def __init__(self, k=40):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)
        p = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            a, b = X[y == 1, j], X[y == 0, j]
            if np.ptp(np.r_[a, b]) > 0:
                p[j] = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        self.support_ = np.sort(np.argsort(p)[: min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def make_models(k=40, seed=42):
    return {
        "RandomForest": Pipeline([
            ("select", WilcoxonSelector(k=k)),
            ("clf", RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                                           class_weight="balanced", n_jobs=-1, random_state=seed)),
        ]),
        "L1-LogisticRegression": Pipeline([
            ("select", WilcoxonSelector(k=k)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                       class_weight="balanced", max_iter=5000, random_state=seed)),
        ]),
    }


def cross_validate(model, X, y, n_splits=5, n_repeats=5, seed=42):
    """Repeated stratified k-fold; returns per-fold AUCs and pooled out-of-fold predictions per repeat."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    X = np.asarray(X)
    aucs, oof = [], np.zeros((n_repeats, len(y)))
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[i // n_splits, te] = p
        aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs), oof


def leave_one_study_out(model, X, y, studies):
    """Train on all cohorts but one, test on the held-out cohort (cross-population generalisation)."""
    X = np.asarray(X)
    rows = []
    for s in sorted(set(studies)):
        te = np.asarray(studies) == s
        m = clone(model).fit(X[~te], y[~te])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(dict(held_out=s, n=int(te.sum()), n_case=int(y[te].sum()),
                         AUC=roc_auc_score(y[te], p), AUPRC=average_precision_score(y[te], p)))
    return pd.DataFrame(rows)


def mean_roc(y, oof):
    grid = np.linspace(0, 1, 101)
    tprs = []
    for p in oof:
        fpr, tpr, _ = roc_curve(y, p)
        tprs.append(np.interp(grid, fpr, tpr)); tprs[-1][0] = 0
    return grid, np.mean(tprs, axis=0), np.std(tprs, axis=0)
