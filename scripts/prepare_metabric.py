import re, sys
from pathlib import Path
import pandas as pd
import numpy as np

HUB = Path.home() / "datahub/public/brca_metabric"
OUT_EXPR = Path("data/metabric/rna_matrix.csv")
OUT_CLIN = Path("data/metabric/clinical_survival.csv")

def pick_file(candidates, purpose):
    """
    candidates: list[str regex] — bunlardan ilk eşleşeni döndür
    """
    files = list(HUB.glob("*.txt")) + list(HUB.glob("*.tsv")) + list(HUB.glob("*.txt.gz")) + list(HUB.glob("*.tsv.gz"))
    files = [p for p in files if p.is_file()]
    # Önerilen kısa listeyi bilgi amaçlı yazdır:
    print(f"[SCAN] {HUB} içinde {len(files)} metin dosyası bulundu.")
    for pat in candidates:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for p in files:
            if rx.search(p.name):
                print(f"[SELECT] {purpose}: {p.name} (pattern: {pat})")
                return p
    # Eşleşme yoksa en yakınları göster
    print(f"[ERR] {purpose} dosyası bulunamadı. Mevcut ilk 10 dosya:")
    for p in files[:10]:
        print("   -", p.name)
    return None

def load_expression(fp):
    # METABRIC mrna dosyaları genelde genler satırda, örnekler sütunda olur.
    df = pd.read_csv(fp, sep="\t", comment="#", dtype=str, low_memory=False)
    # gen ismi kolonu
    gene_col = next((c for c in ["Hugo_Symbol","Gene_Symbol","gene","GENE_SYMBOL","ID"] if c in df.columns), df.columns[0])
    df = df.set_index(gene_col)
    # örnek x gen olacak şekilde transpoze et
    df = df.T
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.index.name = "sample_barcode"
    df["patient_id"] = df.index  # METABRIC'te genelde sample==patient
    return df

def parse_event(s):
    x = s.astype(str).str.upper().str.strip()
    ev = pd.Series(0, index=x.index)
    ev[x.str.contains(r"\bDECEASED\b|\bDEAD\b|(^|:)1($|:)", na=False)] = 1
    num = pd.to_numeric(x, errors="coerce")
    ev = np.where(num.isin([0,1]), num.fillna(ev), ev).astype(int)
    return pd.Series(ev, index=x.index)

def load_clinical(fp):
    clin = pd.read_csv(fp, sep="\t", comment="#", low_memory=False)
    pid = next((c for c in ["PATIENT_ID","patient_id","Case ID","case_id",
                            "submitter_id","SAMPLE_ID","sample_id"] if c in clin.columns), None)
    if pid is None:
        raise SystemExit(f"[ERR] Hasta ID kolonu bulunamadı. Kolonlar: {clin.columns[:12].tolist()}")
    clin = clin.rename(columns={pid:"patient_id"})

    # Aday uç noktalar
    cands = [("OS_MONTHS","OS_STATUS"),
             ("DFS_MONTHS","DFS_STATUS"), ("DFI_MONTHS","DFI_STATUS"),
             ("DSS_MONTHS","DSS_STATUS")]
    best = None
    for t,e in cands:
        if t in clin.columns and e in clin.columns:
            T = pd.to_numeric(clin[t], errors="coerce")
            E = parse_event(clin[e])
            df = pd.DataFrame({"patient_id":clin["patient_id"],"OS_time":T,"OS_event":E}).dropna()
            df = df[df["OS_time"]>0]
            if df.empty: continue
            ev1 = int((df["OS_event"]==1).sum())
            if ev1>0 and (best is None or ev1>best[2]):
                best = (t,e,ev1,df)
    if best is None:
        raise SystemExit("[ERR] Klinik uç noktası bulunamadı (OS/DFS/DFI/DSS). Dosyayı gözle kontrol edin.")
    print(f"[SELECT] clinical endpoint: {best[0]}/{best[1]} (events={best[2]})")
    return best[3].drop_duplicates("patient_id")

def main():
    HUB.mkdir(parents=True, exist_ok=True)
    Path("data/metabric").mkdir(parents=True, exist_ok=True)

    # 1) İfade dosyasını bul
    expr_candidates = [
        r"data_mrna.*microarray.*\.txt(\.gz)?$",
        r"data_mrna_illumina.*\.txt(\.gz)?$",
        r"mrna.*\.txt(\.gz)?$",
    ]
    expr_fp = pick_file(expr_candidates, "expression")
    if expr_fp is None:
        sys.exit(2)
    expr = load_expression(expr_fp)
    expr.to_csv(OUT_EXPR)
    print("[OK] wrote", OUT_EXPR, expr.shape)

    # 2) Klinik dosyasını bul
    clin_candidates = [
        r"data_clinical_patient\.txt(\.gz)?$",
        r"data_clinical\.txt(\.gz)?$",
        r"clinical.*\.txt(\.gz)?$",
    ]
    clin_fp = pick_file(clin_candidates, "clinical")
    if clin_fp is None:
        sys.exit(3)
    clin = load_clinical(clin_fp)
    clin.to_csv(OUT_CLIN, index=False)
    print("[OK] wrote", OUT_CLIN, clin.shape, "| events:", int((clin['OS_event']==1).sum()))

if __name__ == "__main__":
    main()
