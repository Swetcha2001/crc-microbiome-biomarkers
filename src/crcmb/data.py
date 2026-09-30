"""Loading and transforming the genus abundance matrix."""
import numpy as np
import pandas as pd

PSEUDOCOUNT = 1e-4  # in % units (1e-6 as a fraction)


def load(matrix="data/processed/genus_relab.tsv.gz", metadata="data/processed/metadata.tsv"):
    X = pd.read_csv(matrix, sep="\t", index_col=0)
    meta = pd.read_csv(metadata, sep="\t", index_col=0).loc[X.index]
    y = (meta["label"] == "CRC").astype(int).to_numpy()
    return X, y, meta


def log_transform(X, pseudocount=PSEUDOCOUNT):
    """log10 relative abundance with a small pseudocount (SIAMCAT-style)."""
    return np.log10(X + pseudocount)
