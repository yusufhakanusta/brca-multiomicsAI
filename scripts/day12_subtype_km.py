import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

OUT = Path("outputs/ext_valid"); OUT.mkdir(parents=True, exist_ok=True)
clin_csv = "data/metabric/clinical/clinical_survival.csv"
risk_csv = OUT/"risk_scores_metabric.csv"

def pick_subtype_column(df: pd.DataFrame):
    prefer = ["INTRINSIC_SUBTYPE", "INTRINSIC_SUBTYPE_PRED", "PAM50", "PAM50_PRED"]
    for p in prefer:
        if p in df.columns: return p
    for c in df.columns:
        cu = c.upper().replace("-"," ").strip()
        if any(k in cu for k in ["PAM50","INTRINSIC","SUBTYPE","CLAUDIN","MRNA SUBTYPE"]):
            return c
    return None

def km_by_subtype(tbl: pd.DataFrame, label: str):
    # en sık 5 sınıf
    top = tbl["Subtype"].value_counts().index[:5]
    tbl = tbl[tbl["Subtype"].isin(top)]
    if tbl["Subtype"].nunique() < 2:
        print(f"[{label}] Alt tip sayısı <2, KM atlandı.")
        return
    km = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for g, dfg in tbl.groupby("Subtype"):
        km.fit(dfg["OS_time"], dfg["OS_event"], label=g)
        km.plot(ci_show=False)
    plt.xlabel("Time (months)"); plt.ylabel("Survival probability")
    plt.title(f"{label}: KM by Subtype")
    plt.tight_layout()
    plt.savefig(OUT/f"km_by_subtype_{label}.png", dpi=150); plt.close()
    print(f"[OK] km_by_subtype_{label}.png")

def km_within_subtype_low_vs_high(tbl: pd.DataFrame, label: str):
    # risk tertilleri → Low vs High
    q1,q2 = np.quantile(tbl["risk"], [1/3, 2/3])
    tbl = tbl.assign(risk_group=np.where(tbl["risk"]<=q1,"Low", np.where(tbl["risk"]>q2,"High","Mid")))
    subtypes = tbl["Subtype"].value_counts().index[:5]
    lines=[]
    for s in subtypes:
        d = tbl[tbl["Subtype"]==s]
        mL, mH = d["risk_group"]=="Low", d["risk_group"]=="High"
        if mL.sum()<10 or mH.sum()<10:
            lines.append(f"{s}\tNA (grup küçük)")
            continue
        # çizim
        km = KaplanMeierFitter()
        plt.figure(figsize=(6,4))
        for name,mask in [("Low",mL),("High",mH)]:
            km.fit(d.loc[mask,"OS_time"], d.loc[mask,"OS_event"], label=name)
            km.plot(ci_show=False)
        res = logrank_test(d.loc[mL,"OS_time"], d.loc[mH,"OS_time"],
                           d.loc[mL,"OS_event"], d.loc[mH,"OS_event"])
        p = res.p_value
        plt.xlabel("Time (months)"); plt.ylabel("Survival probability")
        plt.title(f"{label}: {s} (Low vs High, log-rank p={p:.2e})")
        plt.tight_layout()
        fn = OUT/f"km_within_{label}_{s.replace('/','-')}_low_vs_high.png"
        plt.savefig(fn, dpi=150); plt.close()
        lines.append(f"{s}\t{p:.6e}")
        print(f"[OK] {fn.name}  (nL={int(mL.sum())}, nH={int(mH.sum())}, p={p:.2e})")
    Path(OUT/f"logrank_within_{label}.tsv").write_text("Subtype\tp_value\n"+"\n".join(lines)+"\n")
    print(f"[OK] logrank_within_{label}.tsv")

def main():
    clin = pd.read_csv(clin_csv).dropna(subset=["OS_time","OS_event"])
    risk = pd.read_csv(risk_csv)
    subcol = pick_subtype_column(clin)
    if subcol is None:
        # son çare: yoksa “Basal/LumA/…” tahmini için day11b/pam50lite çalıştırmanı önerir.
        raise SystemExit("[ERR] METABRIC’te subtype kolonu bulunamadı (INTRINSIC_SUBTYPE/_PRED/PAM50).")
    tbl = clin.merge(risk, on="patient_id", how="inner")
    tbl = tbl.rename(columns={subcol:"Subtype"}).dropna(subset=["Subtype"])
    tbl = tbl[tbl["OS_time"]>0]
    if tbl.empty:
        raise SystemExit("[ERR] METABRIC tablo boş (risk/clin kesişimi).")
    km_by_subtype(tbl, "METABRIC")
    km_within_subtype_low_vs_high(tbl, "METABRIC")

if __name__=="__main__":
    main()
