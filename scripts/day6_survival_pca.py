import pandas as pd, numpy as np, re
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
from pathlib import Path

EXPR="data/tcga_brca/rna_matrix.csv"
CLIN="data/tcga_brca/clinical/clinical_survival.csv"
OUT = Path("outputs/surv_tcga"); OUT.mkdir(parents=True, exist_ok=True)

# 1) RNA
expr = pd.read_csv(EXPR)
X = expr.drop(columns=["patient_id"]).set_index("sample_barcode")
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
# tamamen sıfır sütunları at
X = X.loc[:, (X.sum(axis=0) != 0)]

# 2) PCA
n_comp = min(64, X.shape[1])
pca = PCA(n_components=n_comp, random_state=0)
Z = pca.fit_transform(X.values)
Z = pd.DataFrame(Z, index=X.index, columns=[f"pc{i:03d}" for i in range(Z.shape[1])])
Z["patient_id"] = expr.set_index("sample_barcode")["patient_id"]

# 3) Klinik
clin = pd.read_csv(CLIN)
# Olası farklı sütun adları için sayısallaştırma
clin["OS_time"]  = pd.to_numeric(clin["OS_time"], errors="coerce")
clin["OS_event"] = pd.to_numeric(clin["OS_event"], errors="coerce").fillna(0).astype(int)

tbl = clin.merge(Z.groupby("patient_id").mean().reset_index(), on="patient_id").dropna(subset=["OS_time","OS_event"])
covars = [c for c in tbl.columns if c.startswith("pc")]

# z-score
mu = tbl[covars].mean(0); sd = tbl[covars].std(0).replace(0,1.0)
tbl[covars] = (tbl[covars]-mu)/sd

# düşük varyansı ele
var = tbl[covars].var(0)
covars = var[var>1e-8].index.tolist()

if len(covars)==0 or tbl.shape[0] < 20:
    raise SystemExit(f"[ERR] Yetersiz veri: n_pat={tbl.shape[0]}, n_covars={len(covars)}")

# 4) Cox
cph = CoxPHFitter(penalizer=0.1)
cph.fit(tbl[["OS_time","OS_event"]+covars], duration_col="OS_time", event_col="OS_event")

cph.summary.to_csv(OUT/"cox_summary_pca.csv")
open(OUT/"metrics_pca.txt","w").write(f"C-index: {cph.concordance_index_:.4f}\n")
print("[OK] PCA Cox C-index:", cph.concordance_index_)
print("[OUT]", OUT/"metrics_pca.txt", OUT/"cox_summary_pca.csv")
