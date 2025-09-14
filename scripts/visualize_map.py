import argparse, json
from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def load_positions(spatial_dir: Path):
    # başlıklı veya başlıksız pozisyon dosyası
    if (spatial_dir/"tissue_positions.csv").exists():
        pos = pd.read_csv(spatial_dir/"tissue_positions.csv")
        return pos.rename(columns={
            "barcode":"barcode",
            "pxl_row_in_fullres":"row",
            "pxl_col_in_fullres":"col",
            "in_tissue":"in_tissue",
        })
    else:
        pos = pd.read_csv(spatial_dir/"tissue_positions_list.csv", header=None)
        pos.columns = ["barcode","in_tissue","array_row","array_col","col","row"]
        return pos[["barcode","row","col","in_tissue"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial_dir", default="data/spatial_brca1/spatial")
    ap.add_argument("--pred_csv", default="outputs/predictions.csv")
    ap.add_argument("--out_png", default="outputs/prediction_map.png")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--dot", type=int, default=6)
    args = ap.parse_args()

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

    classes = sorted(df["pred"].unique().tolist())
    palette = {c: col for c,col in zip(
        classes, ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628","#f781bf","#999999"]
    )}

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    for c in classes:
        sub = df[df["pred"]==c]
        plt.scatter(sub["x_hires"], sub["y_hires"], s=args.dot, c=palette[c], label=c, alpha=args.alpha)
    plt.axis("off"); plt.legend(markerscale=2, frameon=True)
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=200, bbox_inches="tight")
    print(f"[OK] saved {args.out_png}")

if __name__ == "__main__":
    main()
