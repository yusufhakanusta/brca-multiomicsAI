import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter

OUT = Path("outputs/surv_tcga"); OUT.mkdir(parents=True, exist_ok=True)
risk_csv = OUT/"risk_scores_VAE.csv"
if not risk_csv.exists(): risk_csv = OUT/"risk_scores_PCA.csv"
clin_csv = "data/tcga_brca/clinical/clinical_survival.csv"

risk = pd.read_csv(risk_csv)
clin = pd.read_csv(clin_csv).dropna(subset=["OS_time","OS_event"]).query("OS_time>0")
tbl  = clin.merge(risk, on="patient_id", how="inner").reset_index(drop=True)

# t* olarak medyan takip süresi (biraz içeride)
tmin, tmax = tbl["OS_time"].min(), tbl["OS_time"].max()
t_star = np.median(tbl["OS_time"])
t_star = float(np.clip(t_star, tmin+1e-6, tmax-1e-6))

# Tek kovaryatlı Cox (risk)
df = tbl[["OS_time","OS_event","risk"]].copy()
cph = CoxPHFitter(penalizer=0.1); cph.fit(df, duration_col="OS_time", event_col="OS_event")

# Beklenen S(t*)
pred_S = []
for i in range(df.shape[0]):
    sf = cph.predict_survival_function(df.iloc[[i]].drop(columns=["OS_time","OS_event"]), times=[t_star])
    pred_S.append(float(sf.values.ravel()[0]))
tbl["S_pred"] = pred_S

# Quintile gruplar
q = np.quantile(tbl["S_pred"], [0.2,0.4,0.6,0.8])
def bin5(s):
    if s<=q[0]: return "Q1 (low S)"
    if s<=q[1]: return "Q2"
    if s<=q[2]: return "Q3"
    if s<=q[3]: return "Q4"
    return "Q5 (high S)"
tbl["calib_group"] = tbl["S_pred"].apply(bin5)

# Gözlenen S(t*) (KM)
obs = []
km = KaplanMeierFitter()
for g in ["Q1 (low S)","Q2","Q3","Q4","Q5 (high S)"]:
    m = tbl["calib_group"]==g
    if m.sum()==0: continue
    km.fit(tbl.loc[m,"OS_time"], tbl.loc[m,"OS_event"])
    S_obs = float(km.survival_function_at_times([t_star]).values)
    S_hat = float(tbl.loc[m,"S_pred"].mean())
    obs.append((g, S_hat, S_obs, int(m.sum())))
cal = pd.DataFrame(obs, columns=["group","S_pred_mean","S_obs","n"])
cal.to_csv(OUT/"calibration_at_tstar.csv", index=False)

# Plot
plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1], "--", alpha=0.5)
plt.scatter(cal["S_pred_mean"], cal["S_obs"], s=40)
for _,r in cal.iterrows():
    plt.annotate(r["group"], (r["S_pred_mean"], r["S_obs"]), xytext=(5,5), textcoords="offset points", fontsize=8)
plt.xlabel(f"Predicted S(t*) @ t*={t_star:.1f} months")
plt.ylabel("Observed S(t*)")
plt.title("Calibration at t* (quintiles)")
plt.xlim(0,1); plt.ylim(0,1); plt.tight_layout()
plt.savefig(OUT/"calibration_tstar.png", dpi=150)
print("[OK] calibration_at_tstar.csv ve calibration_tstar.png yazıldı. t*=", t_star)
