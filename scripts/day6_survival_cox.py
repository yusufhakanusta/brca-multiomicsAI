import argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from lifelines import CoxPHFitter

def load_encoder(ck):
    class Enc(nn.Module):
        def __init__(self, d_in, d_lat):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d_in,1024), nn.ReLU(),
                nn.Linear(1024,256), nn.ReLU()
            )
            self.mu  = nn.Linear(256, d_lat)
        def forward(self, x): return self.mu(self.enc(x))
    m = Enc(ck["d_in"], ck["d_lat"])
    # sadece encoder + mu katmanlarını al
    state = {k:v for k,v in ck["state"].items() if k.startswith(("enc.","mu"))}
    m.load_state_dict(state, strict=False); m.eval()
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expr_csv", default="data/tcga_brca/rna_matrix.csv")
    ap.add_argument("--clin_csv", default="data/tcga_brca/clinical/clinical_survival.csv")
    ap.add_argument("--vae_ckpt", default="models/rna_vae.pt")
    ap.add_argument("--out_dir", default="outputs/surv_tcga")
    ap.add_argument("--penalizer", type=float, default=0.1)
    args=ap.parse_args()

    # load checkpoint (scaler+genes içerir)
    ck = torch.load(args.vae_ckpt, map_location="cpu")
    genes = list(ck["genes"])
    mean  = pd.Series(ck["mean"], index=genes)
    std   = pd.Series(ck["std"],  index=genes).replace(0,1.0)
    enc = load_encoder(ck)

    # RNA
    df = pd.read_csv(args.expr_csv)
    if "patient_id" not in df.columns or "sample_barcode" not in df.columns:
        raise SystemExit("[ERR] rna_matrix.csv beklenen formatta değil.")
    Xdf = df.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # **Aynı gen sırası** + eksik genleri 0 doldur
    for g in genes:
        if g not in Xdf.columns: Xdf[g] = 0.0
    Xdf = Xdf[genes]

    # aynı standardizasyon
    Xz = (Xdf - mean) / std
    X  = torch.from_numpy(Xz.to_numpy(dtype=np.float32))

    with torch.no_grad():
        Z = enc(X).numpy()
    Z = pd.DataFrame(Z, index=Xdf.index, columns=[f"z{i:03d}" for i in range(Z.shape[1])])
    Z["patient_id"] = df.set_index("sample_barcode")["patient_id"]

    Zp = Z.groupby("patient_id").mean().reset_index()

    # Klinik (otomatik ürettiğimiz dosya)
    clin = pd.read_csv(args.clin_csv)
    for c in ["OS_time","OS_event"]:
        if c not in clin.columns: raise SystemExit(f"[ERR] klinikte {c} yok.")
    clin["OS_time"]  = pd.to_numeric(clin["OS_time"], errors="coerce")
    clin["OS_event"] = pd.to_numeric(clin["OS_event"], errors="coerce").fillna(0).astype(int)

    tbl = clin.merge(Zp, on="patient_id", how="inner").dropna(subset=["OS_time","OS_event"])
    covars = [c for c in tbl.columns if c.startswith("z")]

    # NaN/inf temizliği + z-score
    tbl[covars] = tbl[covars].apply(pd.to_numeric, errors="coerce")
    tbl.replace([np.inf,-np.inf], np.nan, inplace=True)
    before = tbl.shape[0]
    tbl = tbl.dropna(subset=covars)
    mu = tbl[covars].mean(0); sd = tbl[covars].std(0).replace(0,1.0)
    tbl[covars] = (tbl[covars]-mu)/sd

    # düşük varyanslıları ele
    var = tbl[covars].var(0)
    covars = var[var>1e-8].index.tolist()

    if len(covars)==0 or tbl.shape[0] < 20:
        raise SystemExit(f"[ERR] Cox için yetersiz veri: n_pat={tbl.shape[0]}, n_covars={len(covars)} (before={before}).")

    cph = CoxPHFitter(penalizer=args.penalizer)
    cph.fit(tbl[["OS_time","OS_event"]+covars], duration_col="OS_time", event_col="OS_event")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cph.summary.to_csv(out/"cox_summary.csv")
    open(out/"metrics.txt","w").write(f"C-index: {cph.concordance_index_:.4f}\n")
    pd.Series({
        "n_pat": tbl.shape[0],
        "n_covars": len(covars),
        "penalizer": args.penalizer,
        "c_index": float(cph.concordance_index_)
    }).to_csv(out/"qc_summary.csv")
    print(f"[OK] C-index: {cph.concordance_index_:.4f}")
    print("[OUT]", out/"metrics.txt", out/"cox_summary.csv", out/"qc_summary.csv")

if __name__=="__main__":
    main()
