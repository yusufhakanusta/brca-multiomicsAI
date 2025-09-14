import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sksurv.util import Surv

OUT = Path("outputs/surv_tcga")
risk_csv = OUT/"risk_scores_VAE.csv"
if not risk_csv.exists():
    risk_csv = OUT/"risk_scores_PCA.csv"

clin_csv = "data/tcga_brca/clinical/clinical_survival.csv"

# 1) verileri yükle
risk = pd.read_csv(risk_csv)
clin = pd.read_csv(clin_csv)
clin = clin.dropna(subset=["OS_time","OS_event"]).query("OS_time>0")

tbl = clin.merge(risk,on="patient_id",how="inner").reset_index(drop=True)
y = Surv.from_arrays(event=tbl["OS_event"].astype(bool).values,
                     time=tbl["OS_time"].values)
score = -tbl["risk"].values  # yüksek değer = yüksek risk için negatifleme

# 2) zaman grid (uçlardan biraz içeride)
tmin, tmax = float(tbl["OS_time"].min()), float(tbl["OS_time"].max())
eps = 1e-6
times_auc = np.percentile(tbl["OS_time"], [25,50,75])
times_auc = np.clip(times_auc, tmin+eps, tmax-eps)

times_brier = np.linspace(tmin+eps, tmax-eps, 30)

# 3) zaman-bağımlı AUC (sksurv)
auc, mean_auc = cumulative_dynamic_auc(y, y, score, times_auc)
plt.figure()
plt.plot(times_auc, auc, marker="o")
plt.xlabel("Time (months)"); plt.ylabel("AUC(t)")
plt.title("Time-dependent AUC")
plt.ylim(0,1)
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT/"time_auc.png", dpi=150)

# 4) Brier score için survival olasılık matrisi üret (Cox'la risk kovaryatı)
# Tek kovaryatlı Cox: risk skoru
df = tbl[["OS_time","OS_event","risk"]].copy()
cph = CoxPHFitter(penalizer=0.1)
cph.fit(df, duration_col="OS_time", event_col="OS_event")
# Her kişi için S_i(t) tahmini
# lifelines predict_survival_function kişi başı seri döndürür; bunları bir matrise birleştiriyoruz.
surv_mat = []
for i in range(df.shape[0]):
    sf = cph.predict_survival_function(df.iloc[[i]].drop(columns=["OS_time","OS_event"]), times=times_brier)
    # sf index=times, tek kolon -> numpy'a
    surv_mat.append(sf.values.ravel())
surv_mat = np.vstack(surv_mat)  # n_samples x n_times

# sksurv brier_score: (y_true_train, y_true_test, est_surv_probs, times)
bs = brier_score(y, y, surv_mat, times_brier)[0]

plt.figure()
plt.plot(times_brier, bs, marker=".")
plt.xlabel("Time (months)"); plt.ylabel("Brier score")
plt.title("Time-dependent Brier score (lower=better)")
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT/"time_brier.png", dpi=150)

print("[OK] time_auc.png ve time_brier.png yazıldı ->", OUT)
