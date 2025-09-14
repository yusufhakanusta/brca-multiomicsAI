import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUT_TCGA = Path("outputs/surv_tcga"); OUT_TCGA.mkdir(parents=True, exist_ok=True)
OUT_EXT  = Path("outputs/ext_valid"); OUT_EXT.mkdir(parents=True, exist_ok=True)

# 1) Alt tip kolonunu bulucu
def find_subtype_column(df):
    cands = [
        "PAM50", "PAM50 mRNA", "CLAUDIN_SUBTYPE", "SUBTYPE",
        "INTRINSIC_SUBTYPE", "mRNA subtype", "Subtype"
    ]
    for c in df.columns:
        cl = c.strip().upper().replace(" ", "").replace("-", "")
        if any(k in cl for k in ["PAM50","CLAUDIN","INTRINSIC","SUBTYPE"]):
            return c
    for c in cands:
        if c in df.columns: return c
    return None

# 2) Risk vs alt tip (boxplot + ANOVA/Kruskal)
def risk_by_subtype(clin_csv, risk_csv, out_dir, label):
    clin = pd.read_csv(clin_csv)
    subcol = find_subtype_column(clin)
    if subcol is None:
        print(f"[{label}] Uyarı: klinikte alt tip kolonu bulunamadı.")
        return
    risk = pd.read_csv(risk_csv)
    df = clin.merge(risk, on="patient_id", how="inner")[["patient_id","risk",subcol]].dropna()
    if df.empty:
        print(f"[{label}] risk+subtype kesişim boş.")
        return
    # sadeleştir
    s = df[subcol].astype(str).str.strip()
    # çoklamaları normalize et (örn. "Luminal A" vs "LumA")
    norm = s.str.upper().str.replace(" ", "").str.replace("-", "")
    map_short = {"LUMINALA":"LumA","LUMINALB":"LumB","HER2ENRICHED":"HER2","BASAL":"Basal","NORMAL":"Normal"}
    cat = norm.map(lambda x: map_short.get(x, x))
    df["Subtype"] = cat
    # en sık 4–5 sınıfı al
    top = df["Subtype"].value_counts().index[:6]
    df = df[df["Subtype"].isin(top)]
    if df["Subtype"].nunique() < 2:
        print(f"[{label}] Alt tip sayısı <2, kutu grafiği atlandı.")
        return
    # Boxplot
    plt.figure(figsize=(6,4))
    order = df.groupby("Subtype").size().sort_values(ascending=False).index
    data = [df.loc[df["Subtype"]==g,"risk"].values for g in order]
    plt.boxplot(data, labels=order, showfliers=False)
    plt.ylabel("Risk score"); plt.title(f"{label}: Risk by subtype")
    plt.tight_layout(); plt.savefig(out_dir/f"risk_by_subtype_{label}.png", dpi=150); plt.close()
    # ANOVA/Kruskal
    if all(len(x)>=10 for x in data):
        F,p = stats.f_oneway(*data)
        test = f"ANOVA p={p:.2e}"
    else:
        H,p = stats.kruskal(*data)
        test = f"Kruskal p={p:.2e}"
    open(out_dir/f"risk_by_subtype_{label}.txt","w").write(test+"\n")
    print(f"[{label}] risk_by_subtype -> {test}")

# 3) Yorumlanabilirlik: Cox katsayıları -> en etkili latentler
def load_cox_summary(out_dir, tag):
    f_vae = out_dir/f"cox_summary_{tag}.csv"
    if f_vae.exists():
        return pd.read_csv(f_vae)
    # eski isimlendirme ihtimali
    alt = out_dir/f"cox_summary_{tag}.csv"
    return pd.read_csv(alt) if alt.exists() else None

def importance_and_top_genes_tcga():
    # VAE mi PCA mı?
    use_vae = (OUT_TCGA/"latent_by_patient.tsv.gz").exists()
    tag = "vae" if use_vae else "pca"
    summ = load_cox_summary(OUT_TCGA, tag)
    if summ is None:
        print("[TCGA] Cox özet bulunamadı, importance atlandı.")
        return
    # en etkili 10 latent (|coef| büyük)
    coef = summ.set_index("covariate")["coef"].abs().sort_values(ascending=False)
    top = coef.head(10).index.tolist()
    pd.Series(coef.head(50)).to_csv(OUT_TCGA/f"top_latent_importance_{tag}.csv")
    print(f"[TCGA] top latent ({tag}):", ", ".join(top[:5]), "...")

    # PCA ise: PC yüklerinden gen listesi çıkar
    if not use_vae:
        # PCA'yı TCGA üzerinde tekrar fit edip bileşen yüklerini alalım
        expr = pd.read_csv("data/tcga_brca/rna_matrix.csv")
        X = expr.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = X.loc[:, (X.sum(0)!=0)]
        from sklearn.decomposition import PCA
        n_comp = max(int(coef.head(1).index[0].replace("pc","")), 64)  # emniyetli sayıda bileşen
        pca = PCA(n_components=min(n_comp, X.shape[1]), random_state=0).fit(X.values)
        genes = X.columns
        want = [int(c.replace("pc","")) for c in top if c.startswith("pc")]
        rows=[]
        for k in want[:5]:  # en etkili 5 bileşen için gen listesi
            if k >= pca.components_.shape[0]: continue
            load = pca.components_[k]
            idx_pos = np.argsort(load)[-30:][::-1]
            idx_neg = np.argsort(load)[:30]
            rows.append(("pc%03d_toppos"%k, ";".join(genes[idx_pos][:30])))
            rows.append(("pc%03d_topneg"%k, ";".join(genes[idx_neg][:30])))
        pd.DataFrame(rows, columns=["component","genes"]).to_csv(OUT_TCGA/"pca_component_topgenes.csv", index=False)
        print("[TCGA] PCA gen yükleri yazıldı -> pca_component_topgenes.csv")
    else:
        # VAE ise: latent–gen korelasyonları (üst düzey, hızlı)
        expr = pd.read_csv("data/tcga_brca/rna_matrix.csv")
        X = expr.drop(columns=["patient_id"]).set_index("sample_barcode").apply(pd.to_numeric, errors="coerce").fillna(0.0)
        # latent by sample varsa kullan; yoksa by patient'tan expand etmeyelim:
        lat_samp = OUT_TCGA/"latent_by_sample.tsv.gz"
        if not lat_samp.exists():
            print("[TCGA] latent_by_sample.tsv.gz yok; gen korelasyonu atlandı.")
            return
        Z = pd.read_csv(lat_samp, sep="\t")
        zcols = [c for c in Z.columns if c.startswith("z")]
        Z = Z.set_index("patient_id") if "patient_id" in Z.columns else Z.set_index(Z.columns[0])
        # index eşleme: sample_barcode ile eşleşmeyebilir; hızlı yaklaşım — ortak indekslerin kesişimini al
        common = X.index.intersection(Z.index)
        if len(common) < 100:
            print("[TCGA] ortak index az; korelasyon atlandı.")
            return
        Xc = X.loc[common]; Zc = Z.loc[common, zcols]
        # her top latent için en korele 30 gen
        out=[]
        for c in top[:5]:
            if c not in Zc.columns: continue
            r = Xc.corrwith(Zc[c], axis=0)
            r = r.replace([np.inf,-np.inf], np.nan).dropna()
            pos = r.nlargest(30).index
            neg = r.nsmallest(30).index
            out.append((c+"_toppos", ";".join(pos)))
            out.append((c+"_topneg", ";".join(neg)))
        pd.DataFrame(out, columns=["latent","genes"]).to_csv(OUT_TCGA/"vae_latent_topgenes.csv", index=False)
        print("[TCGA] VAE latent ↔ gen korelasyon listesi yazıldı.")

def main():
    # --- Risk vs Subtype ---
    tcga_risk = OUT_TCGA/"risk_scores_VAE.csv"
    if not tcga_risk.exists(): tcga_risk = OUT_TCGA/"risk_scores_PCA.csv"
    risk_by_subtype("data/tcga_brca/clinical/clinical_survival.csv", str(tcga_risk), OUT_TCGA, "TCGA")

    met_risk = OUT_EXT/"risk_scores_metabric.csv"
    if met_risk.exists() and Path("data/metabric/clinical_survival.csv").exists():
        risk_by_subtype("data/metabric/clinical_survival.csv", str(met_risk), OUT_EXT, "METABRIC")

    # --- Importance + Top genler ---
    importance_and_top_genes_tcga()

if __name__ == "__main__":
    main()
