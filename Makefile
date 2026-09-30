.PHONY: all download matrix da classify test clean

all: matrix da classify

download:        ## re-download metadata + profiles from GMrepo (~10 min)
	python scripts/01_download_gmrepo.py

matrix:          ## build filtered genus abundance matrix
	python scripts/02_build_matrix.py

da:              ## per-cohort Wilcoxon + Stouffer meta-analysis
	python scripts/03_differential_abundance.py

classify:        ## RF / L1-LR with repeated CV, LOSO and SHAP
	python scripts/04_classify.py

test:
	pytest -q tests
