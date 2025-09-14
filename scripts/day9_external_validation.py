import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

OUT = Path("outputs/ext_valid"); OUT.mkdir(parents=True, exist_ok=True)
tcga_risk = Path("outputs/surv_tcga/risk_scores_VAE.csv")
use_vae = tcga_risk.exists()

# --- TCGA model tarafı (risk-only Cox'tan beta almak için yeniden fit edeceğiz)
tcga_clin = pd.read_csv("data/tcga_brca/clinical/clinical_survival.csv").dropna(subset=["OS_time","OS_event"]).query("OS_time>0")
if use_vae:
    tcga_lat = pd.read_csv("outputs/surv_tcga/latent_by_patient.tsv.gz", sep="\t")
    feat_cols = [c for c in tcga_lat.columns if c.startswith("z")]
else:
    # PCA TCGA fit
    expr = pd.read_csv("data/tcga_brca/rna_matrix.csv")
    X = expr.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.loc[:, (X.sum(0)!=0)]
    n_comp = min(64, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    Z = pca.fit_transform(X.values)
    Z = pd.DataFrame(Z, index=X.index, columns=[f"pc{i:03d}" for i in range(Z.shape[1])])
    Z["patient_id"] = expr.set_index("sample_barcode")["patient_id"]
    tcga_lat = Z.groupby("patient_id").mean().reset_index()
    feat_cols = [c for c in tcga_lat.columns if c.startswith("pc")]

tcga_tbl = tcga_clin.merge(tcga_lat, on="patient_id", how="inner")
# Z-score
mu = tcga_tbl[feat_cols].mean(0); sd = tcga_tbl[feat_cols].std(0).replace(0,1.0)
tcga_tbl[feat_cols] = (tcga_tbl[feat_cols]-mu)/sd

cph_tcga = CoxPHFitter(penalizer=0.1)
cph_tcga.fit(tcga_tbl[["OS_time","OS_event"]+feat_cols], duration_col="OS_time", event_col="OS_event")

# --- METABRIC özelliğe projeksiyon
mb_expr = pd.read_csv("data/metabric/rna_matrix.csv")
if use_vae:
    ck = torch.load("models/rna_vae.pt", map_location="cpu")
    genes = list(ck["genes"]); mean = pd.Series(ck["mean"], index=genes); std = pd.Series(ck["std"], index=genes).replace(0,1.0)
    class Enc(nn.Module):
        def __init__(self,d_in,d_lat):
            super().__init__()
            self.enc1=nn.Linear(d_in,1024); self.enc2=nn.Linear(1024,256); self.mu=nn.Linear(256, ck["d_lat"]); self.act=nn.ELU()
        def forward(self,x): 
            h=self.act(self.enc1(x)); h=self.act(self.enc2(h)); return self.mu(h)
    enc = Enc(ck["d_in"], ck["d_lat"])
    enc.load_state_dict({k:v for k,v in ck["state"].items() if k.startswith(("enc.","mu"))}, strict=False); enc.eval()
    X = mb_expr.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce")
    for g in genes:
        if g not in X.columns: X[g]=0.0
    X = ((X[genes].fillna(0.0) - mean)/std).to_numpy(np.float32)
    with torch.no_grad(): Z = enc(torch.from_numpy(X)).numpy()
    mb_lat = pd.DataFrame(Z, index=mb_expr["sample_barcode"], columns=[f"z{i:03d}" for i in range(Z.shape[1])])
else:
    # PCA'yı TCGA üzerinde fit ettik; aynı dönüşümü METABRIC'e uygula
    # (pca objesini closure'da tutuyoruz)
    X = mb_expr.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.loc[:, (X.sum(0)!=0)]
    # TCGA'da fit edilen bileşen sayısına projekte et
    Z = pca.transform(X.reindex(columns=expr.drop(columns=["patient_id"]).columns, fill_value=0.0).values)
    mb_lat = pd.DataFrame(Z, index=X.index, columns=[f"pc{i:03d}" for i in range(Z.shape[1])])

mb_lat["patient_id"] = mb_expr["patient_id"].values
mb_lat = mb_lat.groupby("patient_id").mean().reset_index()

# METABRIC klinik
mb_clin = pd.read_csv("data/metabric/clinical_survival.csv").dropna(subset=["OS_time","OS_event"]).query("OS_time>0")

# TCGA scaler'ı METABRIC'e uygula
mb_tbl = mb_clin.merge(mb_lat, on="patient_id", how="inner")
mb_tbl[feat_cols] = (mb_tbl[feat_cols]-mu)/sd

# TCGA'da eğitilmiş Cox ile METABRIC risk skoru
risk = -cph_tcga.predict_partial_hazard(mb_tbl[feat_cols])
cidx = concordance_index(mb_tbl["OS_time"].values, risk.values, mb_tbl["OS_event"].values)

# KM tertilleri ve görselleştirme
import numpy as np
q1,q2 = np.quantile(risk, [1/3, 2/3])
def grp(x): return "Low" if x<=q1 else ("High" if x>q2 else "Mid")
g = risk.apply(grp)

plt.figure(figsize=(6,4))
km = KaplanMeierFitter()
for name in ["Low","Mid","High"]:
    m = (g==name).values
    km.fit(mb_tbl["OS_time"].values[m], mb_tbl["OS_event"].values[m], label=name)
    km.plot(ci_show=False)
plt.xlabel("Time (months)"); plt.ylabel("Survival probability")
plt.title(f"METABRIC KM tertiles | model from TCGA ({'VAE' if use_vae else 'PCA'})")
plt.tight_layout()
plt.savefig(OUT/("metabric_km.png"), dpi=150)

open(OUT/"metabric_metrics.txt","w").write(f"C-index (TCGA->METABRIC): {cidx:.4f}\n")
print(f"[OK] External validation: C-index={cidx:.4f} | outputs in {OUT}")
