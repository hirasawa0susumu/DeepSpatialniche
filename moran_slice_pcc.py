"""Moran's I PCC between two spatial slices.

Usage:
    python moran_slice_pcc.py
"""

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

# ═══════════════════════════════════════════════════════════════════════════
# config
# ═══════════════════════════════════════════════════════════════════════════

PATH_A = "/root/autodl-tmp/wangjiaxiang/deepspatial_gt]/output/deepspatial_3d_imc_breastcancer_1.h5ad"
PATH_B = "/root/autodl-tmp/wangjiaxiang/Datas/imc_human_breastcancer/imc_human_breastcancer/imc_10.h5ad"



Z_TARGET = 200
Z_HALF_WIDTH = 2

# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_dense(adata, gene):
    x = adata[:, gene].X
    return x.toarray().flatten() if issparse(x) else np.asarray(x).flatten()


def morans_i(values, coords, k=8):
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


def morans_i_per_gene(adata, genes, k=8):
    coords = adata.obsm["spatial"]
    if coords.shape[1] == 3:
        coords = coords[:, :2]
    return np.array([morans_i(_to_dense(adata, g), coords, k=k) for g in genes])


def select_top_hvgs(adata, n_top=100):
    ad = adata.copy()
    if issparse(ad.X):
        ad.X.data = np.nan_to_num(ad.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        import scanpy as sc
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top, flavor="seurat", inplace=True)
        hvgs = ad.var_names[ad.var["highly_variable"]].tolist()
        if len(hvgs) >= n_top:
            return hvgs[:n_top]
    except Exception:
        pass
    if issparse(ad.X):
        var = np.asarray(ad.X.power(2).mean(0)).ravel() - np.square(np.asarray(ad.X.mean(0)).ravel())
    else:
        var = np.asarray(ad.X).var(axis=0)
    var = np.nan_to_num(var, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return ad.var_names[np.argsort(var)[-n_top:]].tolist()


def moran_pcc(adata_a, adata_b, top_genes=None, n_top=100, use_hvg=True,
              k=8, label_a="A", label_b="B", return_fig=True):
    """
    Moran's I PCC between two AnnData slices.

    Parameters
    ----------
    adata_a, adata_b : AnnData   two slices (must share var_names)
    top_genes : list or None     explicit gene list; overrides use_hvg/n_top
    n_top : int                  number of top HVGs to use (if use_hvg=True)
    use_hvg : bool               True → top HVGs; False → all common genes
    k : int                      k-NN for Moran's I spatial weights
    label_a, label_b : str       labels for scatter plot
    return_fig : bool            whether to return a matplotlib Figure

    Returns
    -------
    pearson_r : float
    fig : Figure or None
    """
    common_all = sorted(set(adata_a.var_names) & set(adata_b.var_names))
    if top_genes is not None:
        genes = [g for g in top_genes if g in common_all]
    elif use_hvg:
        genes = select_top_hvgs(adata_a[:, common_all], n_top=n_top)
    else:
        genes = common_all

    print(f"  common genes: {len(common_all)} | selected: {len(genes)}")
    if len(genes) < 5:
        return float("nan"), None

    mi_a = morans_i_per_gene(adata_a, genes, k=k)
    mi_b = morans_i_per_gene(adata_b, genes, k=k)

    valid = ~(np.isnan(mi_a) | np.isnan(mi_b))
    n_total = len(genes)
    n_nan_a = np.isnan(mi_a).sum()
    n_nan_b = np.isnan(mi_b).sum()
    n_both_nan = (np.isnan(mi_a) & np.isnan(mi_b)).sum()
    n_valid = valid.sum()
    if n_valid < 5:
        return float("nan"), None

    ma, mb = mi_a[valid], mi_b[valid]
    r, _ = pearsonr(ma, mb)
    print(f"  genes: {n_total} total | NaN in A: {n_nan_a} | NaN in B: {n_nan_b} "
          f"| both NaN: {n_both_nan} | valid: {n_valid}")

    fig = None
    if return_fig:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(ma, mb, s=15, alpha=0.5, edgecolors="none", c="#1f77b4")
        lo = min(ma.min(), mb.min()) - 0.05
        hi = max(ma.max(), mb.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Moran's I — {label_a}")
        ax.set_ylabel(f"Moran's I — {label_b}")
        ax.set_title(f"Moran's I PCC  (R = {r:.4f}  n = {valid.sum()}  k = {k})")
        fig.tight_layout()

    return r, fig


def moran_pcc_report(adata_a, adata_b, label_a="A", label_b="B"):
    """Diagnostic: print PCC under multiple parameter settings."""
    print(f"Moran's I PCC  |  {label_a} vs {label_b}")
    print("-" * 50)
    for use_hvg, n_top, k in [
        (True,  100,   8),
        (True,  100,  32),
        (True,  100, 128),
        (False, None,  8),
        (False, None, 32),
        (False, None, 128),
    ]:
        label = f"HVG={n_top}" if use_hvg else "all genes"
        r, _ = moran_pcc(adata_a, adata_b, use_hvg=use_hvg, n_top=n_top, k=k,
                         return_fig=False)
        print(f"  {label:12s}  k={k:3d}  →  R = {r:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    adata_a = ad.read_h5ad(PATH_A)
    adata_b = ad.read_h5ad(PATH_B)
    print(adata_b)

    mask_a = (adata_a.obs["z_coord"] >= Z_TARGET - Z_HALF_WIDTH) & \
             (adata_a.obs["z_coord"] <= Z_TARGET + Z_HALF_WIDTH)

    slice_a = adata_a[mask_a].copy()
    slice_a.obs["z_coord"][:] = Z_TARGET

    print(f"slice A: Z={Z_TARGET}±{Z_HALF_WIDTH} → {slice_a.n_obs} cells")


    # diagnostic: test multiple settings
    moran_pcc_report(slice_a, adata_b, label_a="Generated", label_b="GT")

    # single scatter with default settings
    r, fig = moran_pcc(slice_a, adata_b, k=8, label_a="Generated", label_b="GT")
    print(f"\nDefault (HVG=100, k=8): R = {r:.4f}")
    if fig is not None:
        fig.savefig("moran_pcc_scatter.pdf", dpi=150, bbox_inches="tight")
        print("Scatter → moran_pcc_scatter.pdf")
    plt.show()
