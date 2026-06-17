"""Moran's I correlation between ground truth and reconstruction (paper Figs. 2c-e).

Just edit the paths and params below, then run:
    python morans_i.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


# ═══════════════════════════════════════════════════════════════════════════
# config — edit here
# ═══════════════════════════════════════════════════════════════════════════

RECON_PATH = "/root/autodl-tmp/wangjiaxiang/deepspatial_gt]/output/deepspatial_3d_starmap_brain_new.h5ad"
GT_PATH = "/root/autodl-tmp/wangjiaxiang/Datas/deepstarmap_mouse_brain.h5ad"
OUTPUT_DIR = "/root/autodl-tmp/wangjiaxiang/DeepSpatialniche/moran/morans_i_results_dsgt"

N_TOP_GENES = 100          # number of top HVGs
K_NEIGHBORS = 8             # k-NN for Moran's I spatial weights
MARKER_GENES = ["Gad1", "Gad2", "Slc17a7"]   # highlight these in the plot

# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_3d_coords(adata):
    xy = adata.obsm["spatial"]
    z = adata.obs["z_coord"].to_numpy().reshape(-1, 1)
    return np.column_stack([xy, z])


def _to_dense(adata, gene):
    x = adata[:, gene].X
    return x.toarray().flatten() if issparse(x) else np.asarray(x).flatten()


def _sanitize(adata):
    if issparse(adata.X):
        adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
        adata.X.data = np.clip(adata.X.data, -1e3, 1e3)
    else:
        adata.X = np.nan_to_num(np.asarray(adata.X), nan=0.0, posinf=0.0, neginf=0.0)
        adata.X = np.clip(adata.X, -1e3, 1e3)


# ═══════════════════════════════════════════════════════════════════════════
# HVG selection
# ═══════════════════════════════════════════════════════════════════════════

def select_top_hvgs(adata_gt, n_top_genes=N_TOP_GENES):
    adata = adata_gt.copy()
    _sanitize(adata)
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes, flavor="seurat", inplace=True)
        hvgs = adata.var_names[adata.var["highly_variable"]].tolist()
        if len(hvgs) >= n_top_genes:
            return hvgs[:n_top_genes]
    except Exception:
        pass

    if issparse(adata.X):
        mean = np.asarray(adata.X.mean(0)).ravel()
        mean_sq = np.asarray(adata.X.power(2).mean(0)).ravel()
        var = mean_sq - np.square(mean)
    else:
        var = np.asarray(adata.X).var(axis=0)
    var = np.nan_to_num(var, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    top_idx = np.argsort(var)[-n_top_genes:]
    return adata.var_names[top_idx].tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Moran's I
# ═══════════════════════════════════════════════════════════════════════════

def morans_i(values, coords, k=K_NEIGHBORS):
    n = len(values)
    if n < k + 1:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nn.kneighbors(coords)
    nbrs = indices[:, 1:]

    z = values - values.mean()
    denom = (z * z).sum()
    if denom == 0:
        return 0.0

    num = (z[:, None] * z[nbrs]).sum()
    W = n * k
    return (n / W) * (num / denom) if W > 0 else 0.0


def morans_i_per_gene(adata, genes, k=K_NEIGHBORS):
    coords = _get_3d_coords(adata)
    return np.array([morans_i(_to_dense(adata, g), coords, k=k) for g in genes])


# ═══════════════════════════════════════════════════════════════════════════
# plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_morans_i(mg, mr, r, gene_names, marker_indices, output_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mg, mr, s=20, alpha=0.5, edgecolors="none", c="#1f77b4", label="HVGs")
    if marker_indices:
        ax.scatter(mg[marker_indices], mr[marker_indices], s=60, edgecolors="black",
                   linewidth=0.8, c="#d62728", marker="D", zorder=5,
                   label="marker genes")
        for idx in marker_indices:
            ax.annotate(gene_names[idx], (mg[idx], mr[idx]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    lo = min(mg.min(), mr.min()) - 0.05
    hi = max(mg.max(), mr.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Ground Truth Moran's I")
    ax.set_ylabel("Reconstruction Moran's I")
    ax.set_title(f"Moran's I — top {len(mg)} HVGs  (Spearman R = {r:.4f})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(output_dir, "morans_i.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- load ---
    print(f"Loading reconstruction: {RECON_PATH}")
    adata_recon = sc.read_h5ad(RECON_PATH)
    print(f"  cells={adata_recon.n_obs}  genes={adata_recon.n_vars}")

    print(f"Loading GT: {GT_PATH}")
    adata_gt_all = sc.read_h5ad(GT_PATH)

    # subset GT to recon Z range
    z_gt = adata_gt_all.obsm["spatial"][:, 2].astype(float)
    z_min = adata_recon.obs["z_coord"].min() - 5
    z_max = adata_recon.obs["z_coord"].max() + 5
    adata_gt = adata_gt_all[(z_gt >= z_min) & (z_gt <= z_max)].copy()
    adata_gt.obs["z_coord"] = adata_gt.obsm["spatial"][:, 2].astype(float)
    adata_gt.obsm["spatial"] = adata_gt.obsm["spatial"][:, :2].copy()
    print(f"  GT subset: {adata_gt.n_obs} cells (Z=[{z_min:.0f}, {z_max:.0f}])")

    # --- common genes ---
    common_genes = sorted(set(adata_recon.var_names) & set(adata_gt.var_names))
    print(f"  Common genes: {len(common_genes)}")

    # --- select HVGs ---
    print(f"Selecting top {N_TOP_GENES} HVGs ...")
    top_genes = select_top_hvgs(adata_gt[:, common_genes])

    # --- Moran's I ---
    print(f"Computing Moran's I (k={K_NEIGHBORS}) ...")
    moran_gt = morans_i_per_gene(adata_gt, top_genes)
    moran_recon = morans_i_per_gene(adata_recon, top_genes)

    valid = ~(np.isnan(moran_gt) | np.isnan(moran_recon))
    mg, mr = moran_gt[valid], moran_recon[valid]
    r = spearmanr(mg, mr).correlation

    print(f"\n  Spearman R (Moran's I): {r:.4f}  (1 = perfect)")
    print(f"  Valid genes: {valid.sum()}/{len(top_genes)}")

    # --- marker genes ---
    print("\nMarker gene Moran's I:")
    marker_indices = []
    valid_names = np.array(top_genes)[valid]
    for gene in MARKER_GENES:
        if gene in adata_recon.var_names and gene in adata_gt.var_names:
            mi_gt = morans_i(_to_dense(adata_gt, gene), _get_3d_coords(adata_gt))
            mi_r = morans_i(_to_dense(adata_recon, gene), _get_3d_coords(adata_recon))
            print(f"  {gene:10s}  GT={mi_gt:.4f}  recon={mi_r:.4f}")
            hits = np.where(valid_names == gene)[0]
            if len(hits) > 0:
                marker_indices.append(hits[0])

    # --- save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pd.DataFrame({
        "gene": valid_names,
        "morans_i_gt": mg,
        "morans_i_recon": mr,
    }).to_csv(os.path.join(OUTPUT_DIR, "morans_i_per_gene.csv"), index=False)

    pd.DataFrame({
        "metric": ["morans_i_pearson_r", "n_hvgs", "n_valid"],
        "value": [r, N_TOP_GENES, int(valid.sum())],
    }).to_csv(os.path.join(OUTPUT_DIR, "morans_i_summary.csv"), index=False)

    plot_morans_i(mg, mr, r, valid_names, marker_indices, OUTPUT_DIR)

    print(f"\nDone. Results → {OUTPUT_DIR}/")
    print(f"  {OUTPUT_DIR}/morans_i.pdf")
    print(f"  {OUTPUT_DIR}/morans_i_per_gene.csv")
    print(f"  {OUTPUT_DIR}/morans_i_summary.csv")
