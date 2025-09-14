import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA

OUT = Path("outputs/surv_tcga"); OUT.mkdir(parents=True, exist_ok=True)

def log(*a): print(*a, flush=True)

def load_expr(expr_csv):
    df = pd.read_csv(expr_csv, low_memory=False)
    # Unnamed/boş kolonları tahriş etmeden raporla
    bad = [c for c in df.columns.astype(str) if c.lower().startswith("unnamed") or str(c).strip()==""]
    if bad: log(f"[WARN] Unnamed/boş kolonlar: {bad[:10]}{'...' if len(bad)>10 else ''}")
    if "sample_barcode" not in df.columns or "patient_id" not in df.columns:
        sys.exit("[ERR] rna_matrix.csv içinde 'sample_barcode' ve 'patient_id' yok.")
    X = df.drop(columns=["patient_id"]).set_index("sample_barcode")
    X = X.apply(pd.to_numeric, errors="coerce")
    n_nan = int(np.isnan(X.to_numpy()).sum()); n_inf = int(np.isinf(X.to_numpy()).sum())
    log(f"[RNA] samples={X.shape[0]} genes={X.shape[1]} | nonfinite (NaN={n_nan}, inf={n_inf})")
    # tamamen 0 sütunları rapor + drop (kopya üzerinde)
    zero_cols = list((X.fillna(0).sum(0)==0).index[(X.fillna(0).sum(0)==0).values])
    if zero_cols:
        log(f"[RNA] tamamen 0 gen: {len(zero_cols)} (drop for modeling copy)")
        X = X.loc[:, X.columns.difference(zero_cols)]
    return df, X

def load_clin(clin_csv, patient_ids):
    clin = pd.read_csv(clin_csv)
    for c in ["OS_time","OS_event"]:
        if c not in clin.columns:
            sys.exit(f"[ERR] clinical_survival.csv içinde {c} yok.")
    clin["OS_time"]  = pd.to_numeric(clin["OS_time"], errors="coerce")
    clin["OS_event"] = pd.to_numeric(clin["OS_event"], errors="coerce").fillna(0).astype(int)
    # 0 süreleri çıkar
    clin = clin[clin["OS_time"]>0]
    inter = clin[clin["patient_id"].isin(patient_ids)]
    ev1 = int((inter["OS_event"]==1).sum()); ev0 = int((inter["OS_event"]==0).sum())
    log(f"[CLIN] total={clin.shape[0]} | overlap={inter.shape[0]} | events={ev1} censored={ev0}")
    if ev1==0:
        log("[WARN] Overlap kümesinde hiç event yok → Cox kurulamaz.")
    return clin, inter

def try_vae(expr_df, X_for_checks, vae_ckpt):
    if not Path(vae_ckpt).exists():
        log("[INFO] VAE ckpt yok, PCA fallback’e geçilecek.")
        return None
    ck = torch.load(vae_ckpt, map_location="cpu")
    genes = list(ck["genes"]); mean = pd.Series(ck["mean"], index=genes)
    std   = pd.Series(ck["std"],  index=genes).replace(0,1.0)

    # Aynı gen sırala + eksik genleri 0 doldur
    X = expr_df.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce")
    for g in genes:
        if g not in X.columns: X[g]=0.0
    X = X[genes].fillna(0.0)
    # Z-score aynı scaler ile
    Xz = (X - mean) / std
    Xz = Xz.replace([np.inf,-np.inf], 0.0).fillna(0.0)
    Xz_np = Xz.to_numpy(np.float32)

    # nonfinite kontrol
    nnan = int(np.isnan(Xz_np).sum()); ninf = int(np.isinf(Xz_np).sum())
    if nnan or ninf:
        log(f"[ERR] Xz nonfinite (NaN={nnan}, inf={ninf}) → VAE iptal, PCA'ya geçilecek.")
        return None

    # Encoder tanımı
    class Enc(nn.Module):
        def __init__(self, d_in, d_lat):
            super().__init__()
            self.enc1 = nn.Linear(d_in,1024); self.enc2 = nn.Linear(1024,256)
            self.mu   = nn.Linear(256, ck["d_lat"])
            self.act  = nn.ELU()
        def forward(self, x):
            h = self.act(self.enc1(x)); h = self.act(self.enc2(h))
            return self.mu(h)

    enc = Enc(ck["d_in"], ck["d_lat"])
    state = {k:v for k,v in ck["state"].items() if k.startswith(("enc.","mu"))}
    enc.load_state_dict(state, strict=False); enc.eval()

    with torch.no_grad():
        Z = enc(torch.from_numpy(Xz_np)).numpy()

    if not np.all(np.isfinite(Z)):
        log("[ERR] latent Z içinde NaN/Inf var → PCA'ya geçilecek.")
        return None

    v = np.var(Z, axis=0)
    log(f"[VAE] latent shape={Z.shape}, var(min/mean/max)=({v.min():.4g}/{v.mean():.4g}/{v.max():.4g})")
    if float(v.mean()) == 0.0:
        log("[ERR] latent varyans 0 → PCA'ya geçilecek.")
        return None

    # Hasta bazına ortalama
    samp2pat = expr_df.set_index("sample_barcode")["patient_id"]
    Zdf = pd.DataFrame(Z, index=X.index, columns=[f"z{i:03d}" for i in range(Z.shape[1])])
    Zdf["patient_id"] = samp2pat
    Zp = Zdf.groupby("patient_id").mean().reset_index()
    # kaydet (opsiyonel)
    Zdf.to_csv(OUT/"latent_by_sample.tsv.gz", sep="\t", index=False)
    Zp.to_csv(OUT/"latent_by_patient.tsv.gz", sep="\t", index=False)
    return Zp

def run_cox(feature_df, clin_df, label="VAE"):
    tbl = clin_df.merge(feature_df, on="patient_id", how="inner").dropna(subset=["OS_time","OS_event"])
    covars = [c for c in tbl.columns if c.startswith("z") or c.startswith("pc")]
    if len(covars)==0:
        log(f"[{label}] covars=0 → Cox yok.")
        return False
    # z-score covars
    mu = tbl[covars].mean(0); sd = tbl[covars].std(0).replace(0,1.0)
    tbl[covars] = (tbl[covars]-mu)/sd
    # düşük varyans ele
    var = tbl[covars].var(0); covars = var[var>1e-8].index.tolist()
    if len(covars)==0 or tbl.shape[0]<20:
        log(f"[{label}] yetersiz: n_pat={tbl.shape[0]}, n_covars={len(covars)}")
        return False
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(tbl[["OS_time","OS_event"]+covars], duration_col="OS_time", event_col="OS_event")
    met = OUT/f"metrics_{label.lower()}.txt"
    open(met,"w").write(f"C-index: {cph.concordance_index_:.4f}\n")
    cph.summary.to_csv(OUT/f"cox_summary_{label.lower()}.csv")
    log(f"[{label}] OK C-index={cph.concordance_index_:.4f} → {met}")
    return True

def pca_fallback(expr_df, clin_inter):
    # X matrisi
    X = expr_df.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.loc[:, (X.sum(0)!=0)]
    n_comp = min(64, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    Z = pca.fit_transform(X.values)
    Z = pd.DataFrame(Z, index=X.index, columns=[f"pc{i:03d}" for i in range(Z.shape[1])])
    Z["patient_id"] = expr_df.set_index("sample_barcode")["patient_id"]
    Zp = Z.groupby("patient_id").mean().reset_index()
    return run_cox(Zp, clin_inter, label="PCA")

def main():
    expr_csv = "data/tcga_brca/rna_matrix.csv"
    clin_csv = "data/tcga_brca/clinical/clinical_survival.csv"
    vae_ckpt = "models/rna_vae.pt"

    df_expr, Xcheck = load_expr(expr_csv)
    clin, clin_inter = load_clin(clin_csv, set(df_expr["patient_id"].unique()))
    if int((clin_inter["OS_event"]==1).sum()) == 0:
        log("[FATAL] overlap’te event=0 → önce clinical_survival.csv üretimini tekrar gözden geçir.")
        sys.exit(1)

    Zp = try_vae(df_expr, Xcheck, vae_ckpt)
    ok = False
    if Zp is not None:
        ok = run_cox(Zp, clin_inter, label="VAE")
    if not ok:
        log("[INFO] PCA fallback deneniyor…")
        ok = pca_fallback(df_expr, clin_inter)
        if not ok:
            log("[FATAL] PCA ile de Cox kurulamadı. Klinik event veya veri temizliği sorunlu.")
            sys.exit(2)

if __name__ == "__main__":
    main()
