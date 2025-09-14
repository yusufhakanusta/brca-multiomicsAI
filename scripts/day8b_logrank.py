import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

OUT = Path("outputs/surv_tcga"); OUT.mkdir(parents=True, exist_ok=True)
risk_csv = OUT/"risk_scores_VAE.csv"
if not risk_csv.exists(): risk_csv = OUT/"risk_scores_PCA.csv"
clin_csv = "data/tcga_brca/clinical/clinical_survival.csv"

risk = pd.read_csv(risk_csv)
clin = pd.read_csv(clin_csv).dropna(subset=["OS_time","OS_event"]).query("OS_time>0")
tbl  = clin.merge(risk, on="patient_id", how="inner").reset_index(drop=True)

# tertil grupları
q1,q2 = np.quantile(tbl["risk"], [1/3, 2/3])
def grp(x): return "Low" if x<=q1 else ("High" if x>q2 else "Mid")
tbl["group"] = tbl["risk"].apply(grp)

# KM eğrileri
plt.figure(figsize=(6,4))
km = KaplanMeierFitter()
for name in ["Low","Mid","High"]:
    m = tbl["group"]==name
    if m.sum()==0: continue
    km.fit(tbl.loc[m,"OS_time"], tbl.loc[m,"OS_event"], label=name)
    km.plot(ci_show=False)
plt.xlabel("Time (months)"); plt.ylabel("Survival probability")
plt.title("Kaplan–Meier by risk tertile")
plt.tight_layout()
plt.savefig(OUT/"km_risk_tertiles.png", dpi=150)

# Log-rank: Low vs High
mL, mH = tbl["group"]=="Low", tbl["group"]=="High"
res = logrank_test(tbl.loc[mL,"OS_time"], tbl.loc[mH,"OS_time"],
                   tbl.loc[mL,"OS_event"], tbl.loc[mH,"OS_event"])
p = res.p_value

# Grafiğe p-değeri anotasyonu
plt.figure(figsize=(6,4))
km = KaplanMeierFitter()
for name,mask,color in [("Low",mL,None),("High",mH,None)]:
    if mask.sum()==0: continue
    km.fit(tbl.loc[mask,"OS_time"], tbl.loc[mask,"OS_event"], label=name)
    km.plot(ci_show=False)
plt.xlabel("Time (months)"); plt.ylabel("Survival probability")
plt.title(f"Low vs High (log-rank p={p:.2e})")
plt.tight_layout()
plt.savefig(OUT/"km_low_vs_high_logrank.png", dpi=150)

open(OUT/"logrank_low_vs_high.txt","w").write(f"log-rank p-value (Low vs High): {p:.6e}\n")
print("[OK] km_risk_tertiles.png, km_low_vs_high_logrank.png ve logrank_low_vs_high.txt üretildi.")
