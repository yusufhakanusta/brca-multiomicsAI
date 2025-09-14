import re, pandas as pd
from pathlib import Path

def find_subtype_col(df):
    pats = [
        r"^PAM50", r"INTRINSIC", r"SUBTYPE", r"CLAUDIN", r"mRNA subtype"
    ]
    cols = list(df.columns)
    for c in cols:
        s = c.upper().replace(" ", "").replace("-", "")
        if any(pat.replace("^","") in s for pat in ["PAM50","INTRINSIC","SUBTYPE","CLAUDIN","MRNASUBTYPE"]):
            return c
    return None

def normalize_subtype(s):
    s = str(s).strip()
    if not s or s.lower()=="nan": return None
    u = s.upper().replace(" ", "").replace("-", "")
    map_short = {"LUMINALA":"LumA","LUMINALB":"LumB","HER2ENRICHED":"HER2","BASAL":"Basal","NORMAL":"Normal"}
    return map_short.get(u, s)

def augment_one(project_dir, out_csv):
    proj = Path(project_dir)
    pat = None
    for fn in ["data_clinical_patient.txt","data_clinical.txt"]:
        p = proj / fn
        if p.exists():
            pat = pd.read_csv(p, sep="\t", comment="#", low_memory=False)
            pid = next((c for c in ["PATIENT_ID","patient_id","Case ID","case_id","submitter_id","SAMPLE_ID","sample_id"] if c in pat.columns), None)
            if pid:
                pat = pat.rename(columns={pid:"patient_id"})
            else:
                pat = None
            break
    if pat is None:
        print(f"[WARN] patient-level klinik yok: {proj}")
        return

    # sample-level clinical (alt tip genelde burada)
    samp = None
    for fn in ["data_clinical_sample.txt","data_clinical.txt"]:
        p = proj / fn
        if p.exists():
            samp = pd.read_csv(p, sep="\t", comment="#", low_memory=False)
            break

    subtype_map = None
    if samp is not None:
        # örnek→hasta id
        sid = next((c for c in ["SAMPLE_ID","SAMPLE_ID","sample_id","Sample ID","sample"] if c in samp.columns), None)
        pid = next((c for c in ["PATIENT_ID","patient_id","Case ID","case_id","submitter_id"] if c in samp.columns), None)
        subc = find_subtype_col(samp)
        if sid and (pid or "PATIENT_ID" in samp.columns) and subc:
            if pid is None: pid = "PATIENT_ID"
            m = samp[[sid, pid, subc]].dropna()
            m = m.rename(columns={pid:"patient_id", subc:"INTRINSIC_SUBTYPE"})
            m["INTRINSIC_SUBTYPE"] = m["INTRINSIC_SUBTYPE"].map(normalize_subtype)
            subtype_map = m.groupby("patient_id")["INTRINSIC_SUBTYPE"].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else None)
        else:
            # bazen alt tip doğrudan patient-level dosyada olur
            subc_pat = find_subtype_col(pat)
            if subc_pat:
                subtype_map = pat.set_index("patient_id")[subc_pat].map(normalize_subtype)

    # mevcut clinical_survival.csv’yi oku ve birleştir
    out_csv = Path(out_csv)
    if not out_csv.exists():
        print(f"[WARN] {out_csv} bulunamadı, atla.")
        return
    clin = pd.read_csv(out_csv)
    if subtype_map is not None:
        clin = clin.merge(subtype_map.rename("INTRINSIC_SUBTYPE"), on="patient_id", how="left")
        # eski alt tip kolonları varsa sadeleştir
        if "INTRINSIC_SUBTYPE_x" in clin.columns:
            clin["INTRINSIC_SUBTYPE"] = clin["INTRINSIC_SUBTYPE_x"].fillna(clin.get("INTRINSIC_SUBTYPE_y"))
            clin = clin.drop(columns=[c for c in clin.columns if c.startswith("INTRINSIC_SUBTYPE_")])
        clin.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv} güncellendi: INTRINSIC_SUBTYPE eklendi.")
    else:
        print(f"[WARN] {proj.name}: Alt tip sütunu bulunamadı.")

def main():
    tcga = str(Path.home()/ "datahub/public/brca_tcga_pan_can_atlas_2018")
    met  = str(Path.home()/ "datahub/public/brca_metabric")
    augment_one(tcga, "data/tcga_brca/clinical/clinical_survival.csv")
    augment_one(met,  "data/metabric/clinical/clinical_survival.csv")

if __name__ == "__main__":
    main()
