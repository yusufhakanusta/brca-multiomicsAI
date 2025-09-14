import numpy as np, pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

EXPR = Path("data/tcga_brca/rna_matrix.csv")
CLIN = Path("data/tcga_brca/clinical/clinical_survival.csv")
OUT  = Path("outputs/surv_tcga"); OUT.mkdir(parents=True, exist_ok=True)

def load_features():
    # VAE latent (hasta-bazlı) varsa onu kullan; yoksa PCA fallback sonuçlarını üretelim
    lat_p = OUT/"latent_by_patient.tsv.gz"
    if lat_p.exists():
        Zp = pd.read_csv(lat_p, sep="\t")
        feat_cols = [c for c in Zp.columns if c.startswith("z")]
        source = "VAE"
    else:
        # PCA fallback: örnek-bazlı PCA yap ve hasta ortalaması al
        from sklearn.decomposition import PCA
        expr = pd.read_csv(EXPR)
        X = expr.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = X.loc[:, (X.sum(0)!=0)]
        n_comp = min(64, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=0)
        Z = pca.fit_transform(X.values)
        Z = pd.DataFrame(Z, index=X.index, columns=[f"pc{i:03d}" for i in range(Z.shape[1])])
        Z["patient_id"] = expr.set_index("sample_barcode")["patient_id"]
        Zp = Z.groupby("patient_id").mean().reset_index()
        feat_cols = [c for c in Zp.columns if c.startswith("pc")]
        source = "PCA"
    return Zp, feat_cols, source

def load_clin():
    clin = pd.read_csv(CLIN)
    clin["OS_time"]  = pd.to_numeric(clin["OS_time"], errors="coerce")
    clin["OS_event"] = pd.to_numeric(clin["OS_event"], errors="coerce").fillna(0).astype(int)
    clin = clin.dropna(subset=["OS_time"]).query("OS_time > 0")
    # Klinik kovaryatlar (varsa)
    extra = []
    for c in ["AGE","AGE_AT_DIAGNOSIS","AGE_AT_INDEX","SEX","GENDER"]:
        if c in clin.columns: extra.append(c)
    return clin, extra

def build_tables():
    Zp, feat_cols, source = load_features()
    clin, extra_cols = load_clin()
    tbl = clin.merge(Zp, on="patient_id", how="inner").dropna(subset=["OS_time","OS_event"])
    # SEX/GENDER string ise 0/1 yapalım
    if "SEX" in tbl.columns and tbl["SEX"].dtype == object:
        tbl["SEX"] = (tbl["SEX"].astype(str).str.upper().str.startswith("MALE")).astype(int)
    if "GENDER" in tbl.columns and tbl["GENDER"].dtype == object:
        tbl["GENDER"] = (tbl["GENDER"].astype(str).str.upper().str.startswith("MALE")).astype(int)
    # Klinik-only seti
    clin_cols = [c for c in ["AGE","AGE_AT_DIAGNOSIS","AGE_AT_INDEX","SEX","GENDER"] if c in tbl.columns]
    return tbl, feat_cols, clin_cols, source

def cv_cindex(tbl, xcols, n_splits=5, penalizer=0.1):
    if len(xcols) == 0 or tbl.shape[0] < 50:
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cidx = []
    X = tbl[xcols].copy()
    y_time, y_event = tbl["OS_time"].values, tbl["OS_event"].values
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        # z-score scaler train only on train
        sc = StandardScaler().fit(Xtr.values)
        Xtr_s = pd.DataFrame(sc.transform(Xtr.values), index=Xtr.index, columns=xcols)
        Xte_s = pd.DataFrame(sc.transform(Xte.values), index=Xte.index, columns=xcols)
        dftr = pd.concat([tbl[["OS_time","OS_event"]].iloc[tr].reset_index(drop=True),
                          Xtr_s.reset_index(drop=True)], axis=1)
        dfte = pd.concat([tbl[["OS_time","OS_event"]].iloc[te].reset_index(drop=True),
                          Xte_s.reset_index(drop=True)], axis=1)
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(dftr, duration_col="OS_time", event_col="OS_event")
        # risk skorları ile c-index
        risk = -cph.predict_partial_hazard(dfte)  # daha yüksek değer = daha iyi sağkalım için negatif yapalım
        c = concordance_index(dfte["OS_time"], risk.values, dfte["OS_event"])
        cidx.append(c)
    return float(np.mean(cidx))

def fit_final_and_km(tbl, xcols, label, out_dir=OUT):
    if len(xcols)==0: return None
    # z-score
    sc = StandardScaler().fit(tbl[xcols].values)
    Xs = pd.DataFrame(sc.transform(tbl[xcols].values), index=tbl.index, columns=xcols)
    df = pd.concat([tbl[["patient_id","OS_time","OS_event"]].reset_index(drop=True),
                    Xs.reset_index(drop=True)], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df.drop(columns=["patient_id"]), duration_col="OS_time", event_col="OS_event")
    # risk skoru (negatif partial hazard)
    risk = -cph.predict_partial_hazard(df.drop(columns=["patient_id"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_df = pd.DataFrame({"patient_id":df["patient_id"], "risk":risk.values})
    risk_df.to_csv(out_dir/f"risk_scores_{label}.csv", index=False)
    # KM için tertiller
    tert = np.quantile(risk_df["risk"], [1/3, 2/3])
    def grp(x):
        return "Low" if x<=tert[0] else ("High" if x>tert[1] else "Mid")
    g = risk_df["risk"].apply(grp)
    km_tbl = df[["OS_time","OS_event"]].copy()
    km_tbl["group"] = g.values

    # KM çiz
    plt.figure(figsize=(6,4))
    km = KaplanMeierFitter()
    for name in ["Low","Mid","High"]:
        m = km_tbl["group"]==name
        if m.sum()==0: continue
        km.fit(km_tbl.loc[m, "OS_time"], km_tbl.loc[m, "OS_event"], label=name)
        km.plot(ci_show=False)
    plt.xlabel("Time (months)"); plt.ylabel("Survival probability"); plt.title(f"KM by risk tertile ({label})")
    plt.tight_layout()
    plt.savefig(out_dir/f"km_{label}.png", dpi=150)
    return cph

def main():
    tbl, feat_cols, clin_cols, source = build_tables()
    print(f"[DATA] n_pat={tbl.shape[0]}, feats={len(feat_cols)} ({source}), clinical={len(clin_cols)}")

    # 3 model: clinical, latent, combined
    c_clin = cv_cindex(tbl, clin_cols) if len(clin_cols)>0 else np.nan
    c_lat  = cv_cindex(tbl, feat_cols)
    c_comb = cv_cindex(tbl, clin_cols+feat_cols) if len(clin_cols)>0 else np.nan
    pd.Series({
        "C-index_clinical": c_clin,
        f"C-index_{source}": c_lat,
        f"C-index_combined_{source}": c_comb
    }).to_csv(OUT/"cv_cindex_summary.csv")
    print("[CV] C-index:", {"clinical":c_clin, source:c_lat, f"combined_{source}":c_comb})

    # Final fit & KM (latent-only + combined)
    fit_final_and_km(tbl, feat_cols, label=source)
    if len(clin_cols)>0:
        fit_final_and_km(tbl, clin_cols+feat_cols, label=f"combined_{source}")

if __name__ == "__main__":
    main()
