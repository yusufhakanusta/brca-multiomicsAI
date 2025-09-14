import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

def pick_covariates(clin: pd.DataFrame):
    num_cands = ["AGE","AGE_AT_DIAGNOSIS","AGE_AT_INDEX"]
    cat_cands = ["SEX","GENDER","AJCC_PATHOLOGIC_TUMOR_STAGE","AJCC_STAGE","TUMOR_STAGE","STAGE",
                 "PAM50","PAM50_PRED","INTRINSIC_SUBTYPE","INTRINSIC_SUBTYPE_PRED"]
    num = [c for c in num_cands if c in clin.columns]
    cat = [c for c in cat_cands if c in clin.columns]
    return num, cat

def prepare_design(clin, risk_df):
    df = clin.merge(risk_df, on="patient_id", how="inner").copy()
    df = df.dropna(subset=["OS_time","OS_event","risk"])
    df = df[df["OS_time"]>0]

    # 1) Aday kovaryatlar (ama aşağıda genel bir güvenlik katmanı da var)
    num_cols, cat_cols = pick_covariates(df)

    # 2) risk sayısal olsun
    df["risk"] = pd.to_numeric(df["risk"], errors="coerce")

    # 3) Numerikleri temizle ve ölçekle
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if num_cols:
        sc = StandardScaler()
        df[num_cols] = sc.fit_transform(df[num_cols])

    # 4) Kategorikleri temizle (boş/sabit olanları at)
    for c in list(cat_cols):
        if c not in df.columns: 
            cat_cols.remove(c); continue
        df[c] = df[c].astype(str).str.strip()
        if df[c].nunique() <= 1:
            df.drop(columns=[c], inplace=True)
            cat_cols.remove(c)

    # 5) ***GÜVENLİK KATMANI***:
    #    Hangi kolon obj/string ise otomatik kategorik say ve one-hot yap (drop_first=True)
    obj_cols = [c for c in df.columns if df[c].dtype == "object" 
                and c not in ["patient_id"]]  # zaman/olay zaten numeric
    # 'patient_id' harici tüm stringler one-hot'a girecek
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # 6) Tüm kovaryatlar: OS_time, OS_event, patient_id dışındakiler
    covars = [c for c in df.columns if c not in ["patient_id","OS_time","OS_event"]]

    # 7) Numerik zorlama ve kötü sütunları ayıkla
    for c in covars:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop all-NaN or zero-variance columns
    bad = [c for c in covars 
           if df[c].isna().all() or (df[c].std(skipna=True) == 0) or np.isinf(df[c]).any()]
    if bad:
        df = df.drop(columns=bad)
        covars = [c for c in covars if c not in bad]

    # 8) Kalan NaN’leri satır bazında at (çok az olur)
    df = df.dropna(subset=["OS_time","OS_event"] + covars)
    return df, covars

def fit_and_save(df, covars, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not covars:
        print(f"[{label}] Uyarı: kovaryat bulunamadı, sadece risk ile deniyorum.")
        covars = ["risk"]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df[["OS_time","OS_event"]+covars], duration_col="OS_time", event_col="OS_event")
    # C-index
    cindex = cph.concordance_index_
    Path(out_dir/"combined_cindex.txt").write_text(f"C-index: {cindex:.4f}\n")
    print(f"[{label}] C-index: {cindex:.4f}")

    # HR tablosu
    summ = cph.summary.reset_index().rename(columns={"index":"covariate"})
    summ.to_csv(out_dir/"combined_hr_table.csv", index=False)
    print(f"[{label}] HR tablo -> combined_hr_table.csv")

    # Forest plot (HR with 95% CI) — en etkili 30’u göster
    hr = np.exp(summ["coef"].values)
    lo = np.exp(summ["coef lower 95%"].values)
    hi = np.exp(summ["coef upper 95%"].values)
    names = summ["covariate"].tolist()
    order = np.argsort(-np.abs(summ["z"].values))[:30]
    hr, lo, hi = hr[order], lo[order], hi[order]
    names = [names[i] for i in order]

    plt.figure(figsize=(7, 0.42*len(names)+1))
    y = np.arange(len(names))
    plt.hlines(y, lo, hi)
    plt.vlines(1, -1, len(names), linestyles="dashed", linewidth=1)
    plt.plot(hr, y, "o")
    plt.yticks(y, names)
    plt.xlabel("Hazard Ratio (95% CI)")
    plt.title(f"{label} Combined Cox")
    plt.tight_layout()
    plt.savefig(out_dir/"combined_forest.png", dpi=150)
    plt.close()
    print(f"[{label}] Forest plot -> combined_forest.png")

def run_one(label, clin_csv, risk_csv, out_dir):
    clin = pd.read_csv(clin_csv)
    risk = pd.read_csv(risk_csv)
    df, covars = prepare_design(clin, risk)
    if df.shape[0] < 50:
        print(f"[{label}] Uyarı: örnek sayısı az ({df.shape[0]}). Yine de deniyorum.")
    fit_and_save(df, covars, out_dir, label)

def main():
    # TCGA
    tcga_clin = "data/tcga_brca/clinical/clinical_survival.csv"
    tcga_risk = Path("outputs/surv_tcga/risk_scores_VAE.csv")
    if not tcga_risk.exists(): tcga_risk = Path("outputs/surv_tcga/risk_scores_PCA.csv")
    if Path(tcga_clin).exists() and tcga_risk.exists():
        run_one("TCGA", tcga_clin, str(tcga_risk), Path("outputs/surv_tcga/combined"))

    # METABRIC
    met_clin = "data/metabric/clinical/clinical_survival.csv"
    met_risk = "outputs/ext_valid/risk_scores_metabric.csv"
    if Path(met_clin).exists() and Path(met_risk).exists():
        run_one("METABRIC", met_clin, met_risk, Path("outputs/ext_valid/combined"))

if __name__ == "__main__":
    main()
