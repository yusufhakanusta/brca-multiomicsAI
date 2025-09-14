import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
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
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out_dir", default="outputs/spatial_stats")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # positions -> hires scale
    spatial = Path(args.spatial_dir)
    pos = load_positions(spatial)
    sf = json.load(open(spatial/"scalefactors_json.json"))
    scale = sf.get("tissue_hires_scalef") or sf.get("tissue_hires_scalefactor")
    pos["x_hires"] = (pos["col"]*scale).astype(int)
    pos["y_hires"] = (pos["row"]*scale).astype(int)
    pos = pos[pos["in_tissue"]==1].copy()

    pred = pd.read_csv(args.pred_csv)
    df = pos.merge(pred, on="barcode", how="inner").reset_index(drop=True)
    if len(df)==0:
        raise SystemExit("Pozisyonlarla tahminler eşleşmedi. pred_csv ve spatial_dir kontrol edin.")

    # k-NN grafı
    XY = df[["x_hires","y_hires"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(args.k+1, len(df)), metric="euclidean").fit(XY)
    idx = nn.kneighbors(XY, return_distance=False)[:,1:]  # ilk komşu kendisi, at
    G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(i, subtype=row["pred"])
    for i in range(len(df)):
        for j in idx[i]:
            if i<j:
                G.add_edge(i,j)

    # etkileşim matrisi
    classes = sorted(df["pred"].unique().tolist())
    pair_counts = {(a,b):0 for a in classes for b in classes}
    for u,v in G.edges():
        a = G.nodes[u]["subtype"]; b = G.nodes[v]["subtype"]
        pair_counts[(a,b)] += 1
        pair_counts[(b,a)] += 1
    mat = pd.DataFrame(index=classes, columns=classes, data=0, dtype=float)
    for a in classes:
        total_edges_from_a = sum(pair_counts[(a,bb)] for bb in classes)
        for b in classes:
            mat.loc[a,b] = pair_counts[(a,b)]/total_edges_from_a if total_edges_from_a>0 else 0.0
    mat.to_csv(out_dir/"interaction_matrix_rowNorm.csv")

    # clustering coefficient (alt tip bazında)
    # yöntem: her sınıf için alt ağ oluştur, o alt ağın ortalama yerel clustering'i
    cc_rows = []
    for c in classes:
        nodes_c = [n for n in G.nodes if G.nodes[n]["subtype"]==c]
        H = G.subgraph(nodes_c)
        if H.number_of_nodes()>=3:
            cc_val = nx.average_clustering(H)  # sadece sınıf-içi üçgenleri ölçer
        else:
            cc_val = np.nan
        cc_rows.append({"subtype":c, "clustering_coeff":cc_val, "n_nodes":len(nodes_c)})
    cc = pd.DataFrame(cc_rows)
    cc.to_csv(out_dir/"clustering_coeff.csv", index=False)

    # Isı haritası çiz
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(mat.values, cmap="viridis")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,f"{mat.values[i,j]:.2f}", ha="center", va="center", color="white" if mat.values[i,j]>0.5 else "black")
    ax.set_title("Interaction matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_dir/"interaction_heatmap.png", dpi=200); plt.close(fig)

    print(f"[OK] edges={G.number_of_edges()} nodes={G.number_of_nodes()}")
    print(f"[OK] wrote: {out_dir}/interaction_matrix_rowNorm.csv, clustering_coeff.csv, interaction_heatmap.png")

if __name__ == "__main__":
    main()
