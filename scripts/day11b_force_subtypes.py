import pandas as pd, numpy as np, re, matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

MARKERS = {
  "LumA": ["ESR1","PGR","FOXA1","XBP1","BCL2","SLC39A6","BAG1","NAT1","KRT8","KRT18","KRT19"],
  "LumB": ["MKI67","CCNB1","CDC20","CENPF","AURKA","UBE2C","PTTG1","MYBL2","TYMS","RRM2","EXO1"],
  "HER2": ["ERBB2","GRB7","FGFR4","EGFR","CPB1"],
  "Basal":["KRT5","KRT14","KRT17","EGFR","FOXC1","MIA","KRT6A","KRT6B"],
}

def clean_cols(cols):
    out=[]
    for c in cols:
        s=str(c).split("|")[0]
        s=re.sub(r"\.\d+$","",s)
        out.append(s.upper())
    return out

def pam50lite_assign(expr_csv):
    df = pd.read_csv(expr_csv)
    X = df.drop(columns=["patient_id"]).set_index("sample_barcode")
    X.columns = clean_cols(X.columns)
    X = X.loc[:, (X.sum(0)!=0)]
    mu, sd = X.mean(0), X.std(0).replace(0,1.0)
    Z = (X-mu)/sd
    S={}
    for k,genes in MARKERS.items():
        g=[g for g in genes if g in Z.columns]
        S[k]=Z[g].mean(1) if g else pd.Series(0.0, index=Z.index)
    S=pd.DataFrame(S)
    pred = S.idxmax(axis=1).rename("Subtype")
    samp2pat = df.set_index("sample_barcode")["patient_id"]
    sub_pat = pred.to_frame().assign(patient_id=samp2pat).groupby("patient_id")["Subtype"]\
                     .agg(lambda s: s.value_counts().index[0]).reset_index()
    return sub_pat

def boxplot_and_test(tbl, out_png, label):
    # en sık 5 sınıf
    top = tbl["Subtype"].value_counts().index[:5]
    tbl = tbl[tbl["Subtype"].isin(top)]
    if tbl["Subtype"].nunique()<2:
        print(f"[{label}] Alt tip sayısı <2, grafik atlandı.")
        return
    order = tbl.groupby("Subtype").size().sort_values(ascending=False).index
    data = [tbl.loc[tbl["Subtype"]==g,"risk"].values for g in order]
    plt.figure(figsize=(6,4))
    plt.boxplot(data, labels=order, showfliers=False)
    plt.ylabel("Risk score"); plt.title(f"{label}: Risk by subtype")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    # istatistik
    if all(len(x)>=10 for x in data):
        _,p = stats.f_oneway(*data); test=f"ANOVA p={p:.2e}"
    else:
        _,p = stats.kruskal(*data);  test=f"Kruskal p={p:.2e}"
    Path(out_png.with_suffix(".txt")).write_text(test+"\n")
    print(f"[{label}] {out_png.name} -> {test}")

def main():
    OUT = Path("outputs/ext_valid"); OUT.mkdir(parents=True, exist_ok=True)
    clin = pd.read_csv("data/metabric/clinical/clinical_survival.csv")
    risk = pd.read_csv(OUT/"risk_scores_metabric.csv")
    sub  = pam50lite_assign("data/metabric/rna_matrix.csv")
    tbl  = clin.merge(risk,on="patient_id",how="inner").merge(sub,on="patient_id",how="left")
    tbl  = tbl.dropna(subset=["risk","OS_time","OS_event","Subtype"])
    boxplot_and_test(tbl, OUT/"risk_by_subtype_METABRIC.png", "METABRIC")

if __name__=="__main__":
    main()
