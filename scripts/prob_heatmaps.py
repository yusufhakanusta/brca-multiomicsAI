import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_positions(spatial_dir: Path):
    if (spatial_dir/"tissue_positions.csv").exists():
        pos = pd.read_csv(spatial_dir/"tissue_positions.csv")
        pos = pos.rename(columns={
            "barcode":"barcode",
            "pxl_row_in_fullres":"row",
            "pxl_col_in_fullres":"col",
            "in_tissue":"in_tissue",
        })
    else:
        pos = pd.read_csv(spatial_dir/"tissue_positions_list.csv", header=None)
        pos.columns = ["barcode","in_tissue","array_row","array_col","col","row"]
        pos = pos[["barcode","row","col","in_tissue"]]
    return pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial_dir", default="data/spatial_brca1/spatial")
    ap.add_argument("--pred_csv", default="outputs/predictions.csv")
    ap.add_argument("--out_dir", default="outputs/prob_maps")
    ap.add_argument("--dot", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.35)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    spatial = Path(args.spatial_dir)
    img = cv2.imread(str(spatial/"tissue_hires_image.png"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sf = json.load(open(spatial/"scalefactors_json.json"))
    scale = sf.get("tissue_hires_scalef") or sf.get("tissue_hires_scalefactor")

    pos = load_positions(spatial)
    pos["x_hires"] = (pos["col"]*scale).astype(int)
    pos["y_hires"] = (pos["row"]*scale).astype(int)
    pos = pos[pos["in_tissue"]==1]

    pred = pd.read_csv(args.pred_csv)
    df = pos.merge(pred, on="barcode", how="inner")

    # hangi sınıflar var ve olasılık kolonları
    classes = [c for c in ["LumA","LumB","HER2","Basal"] if f"p_{c}" in df.columns]
    if not classes:
        raise SystemExit("p_Class kolonları bulunamadı. batch_predict.py çıktısını kullanın.")

    for c in classes:
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        sc = df[[f"p_{c}","x_hires","y_hires"]].copy()
        # düşük olasılıkları hafifçe bastır (gürültü)
        sc["w"] = np.clip(sc[f"p_{c}"], 0, 1)
        plt.scatter(sc["x_hires"], sc["y_hires"], s=args.dot, c=sc["w"], cmap="magma", alpha=args.alpha, vmin=0, vmax=1)
        plt.colorbar(label=f"P({c})"); plt.title(f"Probability map: {c}")
        plt.axis("off")
        plt.savefig(out_dir/f"prob_{c}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_dir}/prob_{c}.png")

if __name__ == "__main__":
    main()
