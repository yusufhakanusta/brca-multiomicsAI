import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import cumulative_dynamic_auc, brier_score

def prepare_design(clin_csv, risk_csv):
    clin = pd.read_csv(clin_csv).dropna(subset=["OS_time","OS_event"])
    risk = pd.read_csv(risk_csv)
    df = clin.merge(risk, on="patient_id", how="inner").copy()
    df = df[df["OS_time"]>0]
    df["risk"] = pd.to_numeric(df["risk"], errors="coerce")

    num_cands = ["AGE","AGE_AT_DIAGNOSIS","AGE_AT_INDEX"]
    cat_cands = ["SEX","GENDER","AJCC_PATHOLOGIC_TUMOR_STAGE","AJCC_STAGE","TUMOR_STAGE","STAGE",
                 "PAM50","PAM50_PRED","INTRINSIC_SUBTYPE","INTRINSIC_SUBTYPE_PRED"]
    num_cols = [c for c in num_cands if c in df.columns]
    cat_cols = [c for c in cat_cands if c in df.columns]

    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    if num_cols:
        sc = StandardScaler(); df[num_cols] = sc.fit_transform(df[num_cols])

    obj_cols = [c for c in df.columns if df[c].dtype=='object' and c!="patient_id"]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    covars = [c for c in df.columns if c not in ["patient_id","OS_time","OS_event"]]
    for c in covars: df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = [c for c in covars if df[c].isna().all() or (df[c].std(skipna=True)==0) or np.isinf(df[c]).any()]
    if bad:
        df = df.drop(columns=bad)
        covars = [c for c in covars if c not in bad]

    df = df.dropna(subset=["OS_time","OS_event"]+covars)
    return df, covars

def to_structured_y(df):
    return np.array([(bool(e), float(t)) for e,t in zip(df["OS_event"].astype(int), df["OS_time"].astype(float))],
                    dtype=[('event', '?'), ('time', '<f8')])

def safe_times(df, proposal):
    tmax = df.loc[df["OS_event"]==1, "OS_time"].max() if (df["OS_event"]==1).any() else df["OS_time"].max()*0.9
    tmax = max(1e-3, tmax)
    return np.array([t for t in proposal if 0 < t < tmax])

def calib_plot(df, cph, covars, times, out_png, label):
    surv_funcs = cph.predict_survival_function(df[covars], times=times)
    S_hat = surv_funcs.T.values  # (n, len(times))
    q = pd.qcut(cph.predict_partial_hazard(df[covars]).rank(pct=True), 5, labels=False, duplicates='drop')
    km = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for g in sorted(pd.unique(q)):
        m = (q==g).values
        if m.sum() < 20: 
            continue
        km.fit(df.loc[m,"OS_time"], event_observed=df.loc[m,"OS_event"])
        S_obs = np.interp(times, km.survival_function_.index.values,
                          km.survival_function_["KM_estimate"].values,
                          left=1.0, right=km.survival_function_["KM_estimate"].values[-1])
        S_hat_g = S_hat[m].mean(0)
        plt.plot(1-S_hat_g, 1-S_obs, marker='o', label=f"Q{int(g)+1} (n={m.sum()})")
    plt.plot([0,1],[0,1],'--',linewidth=1)
    plt.xlabel("Predicted event prob up to t")
    plt.ylabel("Observed event prob up to t")
    plt.title(f"{label} calibration (t={int(times[-1])}m)")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def eval_one(label, clin_csv, risk_csv, out_dir, time_grid):
    out_dir.mkdir(parents=True, exist_ok=True)
    df, covars = prepare_design(clin_csv, risk_csv)
    if df.empty or df["OS_event"].sum()==0:
        raise SystemExit(f"[{label}] Yetersiz veri veya hiç event yok; AUC/kalibrasyon hesaplanamaz.")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df[["OS_time","OS_event"]+covars], duration_col="OS_time", event_col="OS_event")

    risk_scores = cph.predict_partial_hazard(df[covars]).values.ravel()
    times = safe_times(df, np.array(time_grid))
    if len(times)==0:
        T = df["OS_time"].values
        times = safe_times(df, np.quantile(T, [0.2,0.4,0.6]))

    y = to_structured_y(df)
    auc, mean_auc = cumulative_dynamic_auc(y, y, risk_scores, times)
    with open(out_dir/"time_auc.txt","w") as f:
        for t,a in zip(times, auc): f.write(f"t={t:.1f}\tAUC={a:.4f}\n")
        f.write(f"Mean AUC={mean_auc:.4f}\n")
    plt.figure(figsize=(6,4))
    plt.plot(times, auc, marker='o')
    plt.ylim(0.4,1.0); plt.xlabel("Time (months)"); plt.ylabel("AUC(t)")
    plt.title(f"{label} time-dependent AUC")
    plt.tight_layout(); plt.savefig(out_dir/"time_auc.png", dpi=150); plt.close()

    surv_funcs = cph.predict_survival_function(df[covars], times=times)
    S_hat = surv_funcs.T.values
    bs, _ = brier_score(y, y, S_hat, times)
    with open(out_dir/"brier_score.txt","w") as f:
        for t,b in zip(times, bs): f.write(f"t={t:.1f}\tBrier={b:.4f}\n")

    calib_plot(df, cph, covars, np.array([times[-1]]), out_dir/"calibration.png", label)
    print(f"[{label}] AUC(t) ve kalibrasyon dosyaları kaydedildi: {out_dir}")

if __name__=="__main__":
    # TCGA
    tcga_clin = "data/tcga_brca/clinical/clinical_survival.csv"
    from pathlib import Path as _P
    tcga_risk = _P("outputs/surv_tcga/risk_scores_VAE.csv")
    if not tcga_risk.exists(): tcga_risk = _P("outputs/surv_tcga/risk_scores_PCA.csv")
    if _P(tcga_clin).exists() and tcga_risk.exists():
        eval_one("TCGA", tcga_clin, str(tcga_risk), Path("outputs/surv_tcga/eval"), time_grid=[24, 60, 120])

    # METABRIC
    met_clin = "data/metabric/clinical/clinical_survival.csv"
    met_risk = "outputs/ext_valid/risk_scores_metabric.csv"
    if _P(met_clin).exists() and _P(met_risk).exists():
        eval_one("METABRIC", met_clin, met_risk, Path("outputs/ext_valid/eval"), time_grid=[60, 120, 180])
