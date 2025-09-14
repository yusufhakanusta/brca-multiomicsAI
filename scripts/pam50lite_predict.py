import pandas as pd, numpy as np, re
from pathlib import Path

MARKERS = {
    # çekirdek marker’lar (PAM50 listesinden yaygın kullanılanlar)
    "LumA": [
        "ESR1","PGR","FOXA1","XBP1","BCL2","SLC39A6","BAG1","NAT1","KRT8","KRT18","KRT19"
    ],
    "LumB": [
        "MKI67","CCNB1","CDC20","CENPF","AURKA","UBE2C","PTTG1","MYBL2","TYMS","RRM2","EXO1"
    ],
    "HER2": [
        "ERBB2","GRB7","FGFR4","EGFR","CPB1"
    ],
    "Basal": [
        "KRT5","KRT14","KRT17","EGFR","FOXC1","MIA","KRT6A","KRT6B"
    ],
}

def _clean_cols(cols):
    out=[]
    for c in cols:
        s=str(c)
        s=s.split("|")[0]           # ensembl|symbol gibi ise
        s=re.sub(r"\.\d+$","",s)    # versiyon .1, .2 kaldır
        out.append(s.upper())
    return out

def score_subtypes(expr_df):
    # expr_df: örnek x gen (rna_matrix.csv’de böyle kaydetmiştik)
    # z-score gene-wise
    X = expr_df.copy()
    mu = X.mean(0); sd = X.std(0).replace(0,1.0)
    Z = (X - mu)/sd
    # skorlar
    S = {}
    for k,genes in MARKERS.items():
        g = [g for g in genes if g in Z.columns]
        if len(g)==0:
            S[k] = pd.Series(0.0, index=Z.index)
        else:
            S[k] = Z[g].mean(1)
    S = pd.DataFrame(S)
    pred = S.idxmax(axis=1)
    return S, pred

def run_one(expr_csv, clin_csv):
    df = pd.read_csv(expr_csv)
    assert "sample_barcode" in df.columns and "patient_id" in df.columns, "rna_matrix formatı beklenen değil"
    X = df.drop(columns=["patient_id"]).set_index("sample_barcode")
    # kolonları gene çevirmek: zaten gen isimleri kolonda; normalize et
    X.columns = _clean_cols(X.columns)
    # tamamen sıfır genleri at
    X = X.loc[:, (X.sum(0)!=0)]
    S, pred = score_subtypes(X)
    # örnek→hasta
    samp2pat = df.set_index("sample_barcode")["patient_id"]
    pred_pat = pred.to_frame("INTRINSIC_SUBTYPE_PRED")
    pred_pat["patient_id"] = samp2pat
    pred_pat = pred_pat.reset_index(drop=True).groupby("patient_id").agg(lambda s: s.value_counts().index[0]).reset_index()

    # klinik dosyaya ekle
    clin = pd.read_csv(clin_csv)
    if "INTRINSIC_SUBTYPE" in clin.columns and clin["INTRINSIC_SUBTYPE"].notna().any():
        # zaten gerçek alt tip varsa, ayrıca PRED sütunu olarak ekle
        out = clin.merge(pred_pat, on="patient_id", how="left")
    else:
        # gerçek yoksa PRED’i INTRINSIC_SUBTYPE olarak yazalım + ayrıca *_PRED de dursun
        out = clin.merge(pred_pat.rename(columns={"INTRINSIC_SUBTYPE_PRED":"INTRINSIC_SUBTYPE"}), on="patient_id", how="left")
        out = out.rename(columns={"INTRINSIC_SUBTYPE":"INTRINSIC_SUBTYPE"})
    out.to_csv(clin_csv, index=False)
    return S, pred, out

def main():
    Path("outputs/surv_tcga").mkdir(parents=True, exist_ok=True)
    # TCGA
    if Path("data/tcga_brca/rna_matrix.csv").exists() and Path("data/tcga_brca/clinical/clinical_survival.csv").exists():
        S, pred, out = run_one("data/tcga_brca/rna_matrix.csv", "data/tcga_brca/clinical/clinical_survival.csv")
        print("[OK] TCGA: INTRINSIC_SUBTYPE(_PRED) eklendi. n=", out.shape[0])
    # METABRIC
    if Path("data/metabric/rna_matrix.csv").exists() and Path("data/metabric/clinical/clinical_survival.csv").exists():
        S, pred, out = run_one("data/metabric/rna_matrix.csv", "data/metabric/clinical/clinical_survival.csv")
        print("[OK] METABRIC: INTRINSIC_SUBTYPE(_PRED) eklendi. n=", out.shape[0])

if __name__=="__main__":
    main()
