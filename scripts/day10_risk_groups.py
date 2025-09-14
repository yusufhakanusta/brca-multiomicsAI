import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Giriş dosyaları
OUT_TCGA = Path("outputs/surv_tcga")
OUT_MET  = Path("outputs/ext_valid")
clin_tcga = "data/tcga_brca/clinical/clinical_survival.csv"
clin_met  = "data/metabric/clinical_survival.csv"

risk_tcga = OUT_TCGA/"risk_scores_VAE.csv"
if not risk_tcga.exists():
    risk_tcga = OUT_TCGA/"risk_scores_PCA.csv"

risk_met = OUT_MET/"risk_scores_metabric.csv"

def risk_groups(df, time_col="OS_time", event_col="OS_event", risk_col="risk", title="TCGA"):
    # tertile bazlı risk grupları
    cuts = np.percentile(df[risk_col], [33, 66])
    df["risk_group"] = np.where(df[risk_col] <= cuts[0], "Low",
                          np.where(df[risk_col] <= cuts[1], "Medium", "High"))

    # KM eğrileri
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6,5))
    for g, dfg in df.groupby("risk_group"):
        kmf.fit(dfg[time_col], dfg[event_col], label=g)
        kmf.plot(ci_show=False)
    plt.xlabel("Time (months)"); plt.ylabel("Survival probability")
    plt.title(f"{title}: Kaplan–Meier by risk group")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    out_png = (OUT_TCGA if title=="TCGA" else OUT_MET)/f"km_{title}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Log-rank testi: Low vs High
    low  = df.query("risk_group=='Low'")
    high = df.query("risk_group=='High'")
    lr = logrank_test(low[time_col], high[time_col],
                      low[event_col], high[event_col])
    print(f"[{title}] Log-rank p=", lr.p_value)

# --- TCGA ---
dfc = pd.read_csv(clin_tcga)
dfr = pd.read_csv(risk_tcga)
tbl = dfc.merge(dfr,on="patient_id",how="inner").dropna(subset=["OS_time","OS_event","risk"])
tbl = tbl[tbl["OS_time"]>0]
risk_groups(tbl, title="TCGA")

# --- METABRIC ---
if Path(clin_met).exists() and risk_met.exists():
    dfc = pd.read_csv(clin_met)
    dfr = pd.read_csv(risk_met)
    tbl = dfc.merge(dfr,on="patient_id",how="inner").dropna(subset=["OS_time","OS_event","risk"])
    tbl = tbl[tbl["OS_time"]>0]
    risk_groups(tbl, title="METABRIC")

print("[OK] KM grafikleri ve log-rank p-değerleri kaydedildi.")
