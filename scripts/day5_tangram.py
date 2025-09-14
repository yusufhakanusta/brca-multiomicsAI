import re
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg
try:
    import scipy.sparse as sp
except Exception:
    sp = None

SPATIAL_H5 = Path("data/spatial_brca1/matrix/Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
SCREF_H5AD = Path("data/scref_brca/ref.h5ad")
OUT_DIR    = Path("outputs/tangram"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def _series_to_clean_symbols(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace(r"\.\d+$","",regex=True).str.upper()
    s = s.where(~s.isna() & (s.str.len()>0) & (s!="NA"))
    return s

def clean_varnames_to_symbols(adata, prefer_cols=("gene_symbol","gene_symbols","feature_name","gene_name","symbols","symbol","SYMBOL","GeneSymbol")):
    var = adata.var.copy()
    sym_col = next((c for c in prefer_cols if c in var.columns), None)
    sym = var[sym_col] if sym_col else pd.Series(adata.var_names)
    sym = _series_to_clean_symbols(sym)
    mask = sym.notna().to_numpy()
    adata = adata[:, mask].copy()
    adata.var["SYMBOL_CLEAN"] = sym[mask].to_numpy()
    adata.var_names = pd.Series(adata.var["SYMBOL_CLEAN"]).astype("string").to_numpy()
    adata.var_names_make_unique()
    return adata

# marker setleri
PAM50 = [
 "BAG1","BCL2","BIRC5","CCNB1","CDC20","CDH3","CENPF","CEP55","EGFR","ERBB2",
 "ESR1","EXO1","FGFR4","FOXA1","FOXC1","GPR160","GRB7","KIF2C","KRT14","KRT17",
 "KRT5","MELK","MIA","MKI67","MLPH","MYBL2","MYC","NAT1","ORC6","PGR",
 "PHGDH","PTTG1","RRM2","SLC39A6","TMEM45B","TYMS","UBE2C","UBE2T","XBP1",
 "ACTR3B","BLVRA","BCL2L2","CCNE1","KRT8","KRT18","KRT19","MMP11","GSTM1"
]
SUBTYPE_MARKERS = {
 "LumA": ["ESR1","PGR","BCL2","FOXA1","XBP1","MLPH","SLC39A6","BCL2L2","KRT8","KRT18","KRT19"],
 "LumB": ["MKI67","CCNB1","MYBL2","TYMS","UBE2C","PTTG1","EXO1","RRM2","CCNE1"],
 "HER2": ["ERBB2","GRB7","FGFR4","PHGDH"],
 "Basal": ["KRT5","KRT14","KRT17","EGFR","MIA","FOXC1","MMP11"]
}

def map_symbols_to_varnames(adata_raw, symbols_upper):
    sv = adata_raw.var
    cand_cols = [c for c in ["gene_name","feature_name","gene_symbol","symbols","symbol","SYMBOL","GeneSymbol"] if c in sv.columns]
    if cand_cols:
        sym = _series_to_clean_symbols(sv[cand_cols[0]])
        mask = sym.isin(symbols_upper)
        idx = np.where(mask.to_numpy())[0]
        if len(idx)>0:
            return [adata_raw.var_names[i] for i in idx]
    vn = pd.Series(adata_raw.var_names).astype("string").str.upper()
    m = vn.isin(symbols_upper)
    if m.any():
        idx = np.where(m.to_numpy())[0]
        return [adata_raw.var_names[i] for i in idx]
    return []

print("[1] load data")
adata_vis_raw = sc.read_10x_h5(SPATIAL_H5)
adata_sc_raw  = sc.read_h5ad(SCREF_H5AD)

print("[2] gene-name cleaning")
adata_vis_full = clean_varnames_to_symbols(adata_vis_raw.copy())
adata_sc_full  = clean_varnames_to_symbols(adata_sc_raw.copy())

print("[3] normalization")
for A in (adata_sc_full, adata_vis_full):
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)

shared = adata_sc_full.var_names.intersection(adata_vis_full.var_names)
print(f"[INFO] shared genes after cleaning: {len(shared)}")
adata_sc_map  = adata_sc_full[:, shared].copy()
adata_vis_map = adata_vis_full[:, shared].copy()

print("[4] tangram pp + mapping (auto, uniform)")
tg.pp_adatas(adata_sc_map, adata_vis_map, genes=list(shared))
candidate_cols = ["cell_type","CellType","celltype","leiden","louvain","cluster","clusters","annotation","anno","major_cell_type"]
cluster_col = next((c for c in candidate_cols if c in adata_sc_map.obs.columns), None)
if cluster_col:
    print(f"[INFO] clusters mode: {cluster_col}")
    ad_map = tg.map_cells_to_space(adata_sc_map, adata_vis_map, mode="clusters",
                                   cluster_label=cluster_col, density_prior="uniform",
                                   num_epochs=400, device="cpu")
else:
    print("[INFO] cells mode")
    ad_map = tg.map_cells_to_space(adata_sc_map, adata_vis_map, mode="cells",
                                   density_prior="uniform", num_epochs=400, device="cpu")

X = getattr(ad_map, "X", None)
if X is None: raise RuntimeError("Tangram mapping produced no matrix in ad_map.X")
if sp is not None and sp.issparse(X): X = X.tocsr()
W = X.T  # (spots x cols) ; cols = cells  veya clusters
print(f"[INFO] mapping matrix W shape: {W.shape}")

print("[5] cell-level scoring on FULL scRNA space")
PAM50_UP = [g.upper() for g in PAM50]
SUB_UP   = {k:[g.upper() for g in v] for k,v in SUBTYPE_MARKERS.items()}

score_cols = []
pam50_mapped = map_symbols_to_varnames(adata_sc_raw, PAM50_UP)
if len(pam50_mapped) >= 3:
    sc.tl.score_genes(adata_sc_full, gene_list=pam50_mapped, score_name="score_PAM50"); score_cols.append("score_PAM50")
for name, genes in SUB_UP.items():
    m = map_symbols_to_varnames(adata_sc_raw, genes)
    if len(m) >= 2:
        sc.tl.score_genes(adata_sc_full, gene_list=m, score_name=f"score_{name}"); score_cols.append(f"score_{name}")
if len(score_cols)==0: raise RuntimeError("Marker skorları üretilemedi (scRNA’da eşleşme yok).")

Xcell = adata_sc_full.obs[score_cols].to_numpy()
Xcell = (Xcell - Xcell.mean(0)) / (Xcell.std(0)+1e-6)
P_cell = np.exp(Xcell); P_cell = P_cell / P_cell.sum(1, keepdims=True)
cell_classes = [c.replace("score_","") for c in score_cols]

print("[6] match projection dimensionality")
# Eğer clusters modundaysak, W kolon sayısı = cluster sayısı olacaktır.
# Buna uygun olarak P_cell'i cluster'lara indirgememiz gerekiyor.
if cluster_col:
    # ad_map.var_names cluster isimlerinin sıra/etiketlerini taşır
    cluster_order = list(ad_map.obs_names)
    # cluster label'ı FULL scRNA'ya getir
    if cluster_col in adata_sc_raw.obs.columns:
        cl_series = adata_sc_raw.obs[cluster_col].astype(str)
        cl_series = cl_series.reindex(adata_sc_full.obs_names).astype(str)
    else:
        # mapping kopyasında varsa ordan çek
        cl_series = adata_sc_map.obs[cluster_col].astype(str)
        cl_series = cl_series.reindex(adata_sc_full.obs_names).astype(str)

    # her cluster için o clustera ait hücrelerin ortalama P_cell’i
    K = len(cluster_order); S = P_cell.shape[1]
    # güvenlik: W ile K eşleşsin
    import numpy as np
    # W shape: (spots x cols)
    # cols, cluster sayısı olmalı
    P_cluster = np.zeros((K, S), dtype=float)
    for k, cname in enumerate(cluster_order):
        mask = (cl_series.values == str(cname))
        if mask.any():
            P_cluster[k,:] = P_cell[mask,:].mean(axis=0)
        else:
            P_cluster[k,:] = 1.0 / S  # güvenli fallback
    P_target = P_cluster  # (clusters x classes)
else:
    # cells mode: W kolon sayısı hücre sayısı ile eşleşmeli
    # P_cell full scRNA’dan -> mapping kopyasındaki hücre sırasına çekelim
    P_df = pd.DataFrame(P_cell, index=adata_sc_full.obs_names, columns=range(P_cell.shape[1]))
    P_df = P_df.reindex(adata_sc_map.obs_names)  # mapping’teki hücre sırasına
    P_target = P_df.to_numpy()  # (cells x classes)

print("[6b] project to spots")
# W: (spots x cols), P_target: (cols x classes)
P_spot = W @ P_target
P_spot = P_spot / (P_spot.sum(1, keepdims=True) + 1e-8)

print("[7] save")
barcodes = adata_vis_map.obs_names.to_list()
df = pd.DataFrame(P_spot, columns=[f"p_{c}" for c in cell_classes])
df.insert(0, "barcode", barcodes)
pred_cols = [c for c in cell_classes if c != "PAM50"] or cell_classes
df["pred"] = df[[f"p_{c}" for c in pred_cols]].idxmax(1).str.replace("p_","")

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR/"spot_probs.csv", index=False)
labels = df[["barcode","pred"]].rename(columns={"pred":"subtype"})
Path("data/spatial_brca1").mkdir(parents=True, exist_ok=True)
labels.to_csv("data/spatial_brca1/labels.csv", index=False)
print("[OK] wrote outputs/tangram/spot_probs.csv and data/spatial_brca1/labels.csv")
